import os
import cv2
import numpy as np
import csv
from imp_mods import HandProcessor


is_rec = False
frames_rec = 0
current_chord = ""


# collecting data
file = f'test.csv'
if not os.path.exists(file):
    with open(file, mode='w', newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ['Label']
        header += [f"L_{axis}{i}" for i in range(21) for axis in ('x', 'y')]
        header += [f"R_{axis}{i}" for i in range(21) for axis in ('x', 'y')]
        writer.writerow(header)


engine = HandProcessor()


# opencv setup
cap = cv2.VideoCapture(0)
print("Starting webcam")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Webcam issue")
        break

    frame = cv2.flip(frame, 1)
    engine.frame_process(frame)
    engine.draw_hand_landmarkers(frame)

    combined_f = engine.get_norm_f(frame.shape)
    detected_hand = np.any(combined_f != 0)

    if current_chord != "" and is_rec:
        if detected_hand:
            with open(file, mode='a', newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                raw_data = [current_chord] + combined_f.tolist()
                writer.writerow(raw_data)

            frames_rec += 1
            cv2.putText(frame, f"Recording {current_chord} chord ({frames_rec}/1024)", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            if frames_rec == 1024:
                print("Successfully recorded 1024 frames.")
                frames_rec = 0
                is_rec = False
                current_chord = ""

        else:
            cv2.putText(frame, "Waiting for hands...", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)

    else:
        cv2.putText(frame, f"Sitting idle...", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        if key == ord('s'):
            if is_rec:
                is_rec = False
            else:
                is_rec = True
            print(f"Recoding stopped\nRecorded frames: {frames_rec} of {current_chord} chord!")
        elif key == ord('q'):
            print(f"Recording terminated and saved successfully !!!")
            break
        elif key == ord('n'):
            current_chord = "NULL"
            is_rec = True
            print(f"Collecting data for {current_chord} chord")
        elif ord('a') <= key <= ord('z'):
            if not is_rec:
                is_rec = True
                current_chord = chr(key).upper()
                print(f"Collecting data for {current_chord} chord")

    cv2.imshow("Data collector0", frame)


cap.release()
cv2.destroyAllWindows()