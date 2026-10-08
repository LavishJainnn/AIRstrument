import mido
from imp_mods import HandProcessor
import tensorflow.lite as tflite
import cv2
import numpy as np


#hashtables
chord_map = {
    0: [45, 49, 52],
    1: [47, 51, 54],
    2: [48, 52, 55],
    3: [50, 54, 57],
    4: [52, 56, 59],
    5: [53, 57, 60],
    6: [55, 59, 62],
    7: []
}

label_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'Idle'
}


#midi connection
midi_out = mido.open_output()

# for guitar
midi_out.send(mido.Message('program_change', program=25))

curr_chord = 7
pred_chord = 7
debounce = 1
stability_counter = 0

playing_chord = 7
prev_yi = None
prev_ym = None
is_strumming = False
strum_at = 0.08
reset_at = -0.04

lost_hand_frame = 0
velocity = 1
performance_mode = False


#setting tensors and using model
engine = HandProcessor()

interpreter = tflite.Interpreter('Chords_classifier.tflite')
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()


#opening cam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
print("Welcome to AIRstrument!")


while True:
    ret, frame = cap.read()

    if not ret:
        print("WebCam disconnected...\nEXITING...")
        break


    #frame set-up
    frame = cv2.flip(frame, 1)
    engine.frame_process(frame)

    if not performance_mode:
        engine.draw_hand_landmarkers(frame)


    #left hand part
    left_f = engine.norm_left(frame.shape)

    if np.any(left_f):
        input_data = np.array([left_f], dtype=np.float32)
        interpreter.set_tensor(in_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(out_details[0]['index'])
        raw_pred = np.argmax(output_data[0])
    else:
        raw_pred = 7

    #check for stability
    if raw_pred == pred_chord:
        stability_counter += 1
    else:
        stability_counter = 0
        pred_chord = raw_pred

    #change the chord
    if stability_counter > debounce:
        if curr_chord != pred_chord:
            if playing_chord == 7:
                for i in range(7):
                    for note in chord_map[i]:
                        midi_out.send(mido.Message("note_off", note=note))

                playing_chord = 7

            curr_chord = pred_chord


    #right hand part
    xi_coord, yi_coord, ym_coord = engine.kinematics_right()

    if yi_coord is not None and ym_coord is not None:
        lost_hand_frame = 0
        if prev_yi is not None and prev_ym is not None:

            delta_yi = yi_coord - prev_yi
            delta_ym = ym_coord - prev_ym

            if not is_strumming:

                if delta_yi > strum_at:
                    is_strumming = True
                    velocity = int(np.clip(89+((xi_coord - 0.5) * 152), 89, 127))

                    if curr_chord != 7:
                        for note in chord_map[curr_chord]:
                            midi_out.send(mido.Message("note_on", note=note, velocity=velocity))

                        playing_chord = curr_chord

                if delta_ym > strum_at:
                    is_strumming = True
                    velocity = int(np.clip(89+((xi_coord - 0.5) * 152), 72, 101))

                    if curr_chord != 7:
                        for note in chord_map[curr_chord]:
                            midi_out.send(mido.Message("note_on", note=note, velocity=velocity))

                        playing_chord = curr_chord

            elif (delta_yi < reset_at or delta_ym < reset_at) and is_strumming:
                    is_strumming = False

        prev_yi = yi_coord
        prev_ym = ym_coord

    else:
        lost_hand_frame += 1

        if lost_hand_frame > 5:
            prev_yi = None
            prev_ym = None
            is_strumming = False



    #interface
    if not performance_mode:
        status_color = (0, 255, 0) if is_strumming else (0, 0, 255)

        cv2.putText(frame, f"Holding: {label_map[curr_chord]}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(frame, f"Struming: {label_map[playing_chord]}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, status_color, 2)

        if xi_coord is not None:
            cv2.putText(frame, f"Volume: {int((velocity/127)*100)}%", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

    cv2.imshow("AIRstrument", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break


for i in range(7):
    for note in chord_map[playing_chord]:
        midi_out.send(mido.Message("note_off", note=note))


cap.release()
midi_out.close()
cv2.destroyAllWindows()