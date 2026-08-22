import cv2
import mido
import numpy as np
import tensorflow.lite as tflite
from imp_mods import HandProcessor


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

#default is a giant piano

# for continuous piano
# midi_out.send(mido.Message('program_change', program=48))


#var
current_play = 7
current_pred = 7
frames_held = 0
debounce = 3

engine = HandProcessor()

interpreter = tflite.Interpreter('Chords_classifier.tflite')
interpreter.allocate_tensors()
idetails = interpreter.get_input_details()
odetails = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
print("AIRstrument's AIR_Piano started successfully !\npress 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("WebCam disconnected. EXITING...")
        break

    frame = cv2.flip(frame, 1)


    #math
    engine.frame_process(frame)
    engine.draw_hand_landmarkers(frame)
    combined_f = engine.get_norm_f(frame.shape)
    input_data = np.array([combined_f], dtype=np.float32)


    #model
    interpreter.set_tensor(idetails[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(odetails[0]['index'])

    pred_idx = np.argmax(output_data[0])
    confidence = output_data[0][pred_idx]


# this was the state machine way
    #state machine
    if pred_idx == current_pred:
        frames_held += 1
    else:
        frames_held = 0
        current_pred = pred_idx


    #midi messages
    if frames_held >= debounce and current_pred != current_play:
        if current_play != 7:
            for note in chord_map[current_play]:
                midi_out.send(mido.Message("note_off", note = note))

        if current_pred != 7:
            for note in chord_map[current_pred]:
                midi_out.send(mido.Message("note_on", note = note, velocity=127))

        current_play = current_pred


    #output
    font_color = (0, 255, 0) if current_play != 7 else (0, 0, 255)
    cv2.putText(frame, f"{label_map[pred_idx]} Chord.", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, font_color, 2)

    cv2.imshow("AIR_Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#stopping all the chords
if current_play != 7:
    for note in chord_map[current_play]:
        midi_out.send(mido.Message("note_off", note = note))


cap.release()
midi_out.close()
cv2.destroyAllWindows()