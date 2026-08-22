import os
import urllib.request
from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe.tasks import python
import time
import cv2
import numpy as np


class HandProcessor:
    def __init__ (self, model_at = 'hand_landmarker.task'):


        # this code block makes sure of the hand_landmarker.task file
        if not os.path.exists(model_at):
            print("Downloading necessary tools from internet.")
            urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', model_at)
            print("Download complete !")


        # callback function for every frame results
        self.latest_results = None
        def callback_func(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp: int):
            self.latest_results = result


        # Mediapipe model setup
        base_option = python.BaseOptions(model_asset_path=model_at)
        options = vision.HandLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=callback_func
        )
        self.landmarker_model = vision.HandLandmarker.create_from_options(options)


        #log time
        self.start_time = time.time()


        # defining custom hand landmarkers function (dot stick)
    def draw_hand_landmarkers(self, d_frame):
        if self.latest_results is not None and self.latest_results.hand_landmarks:
            connection = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20), (13, 17)
            ]

            h, w, _ = d_frame.shape

            for hand_landmarker in self.latest_results.hand_landmarks:
                for bone in connection:
                    p1 = hand_landmarker[bone[0]]
                    p2 = hand_landmarker[bone[1]]

                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)

                    cv2.line(d_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for lm in hand_landmarker:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(d_frame, (x, y), 4, (0, 0, 255), -1)


    # normalization math
    def norm(self, ih, iw, target):
        norm_points = np.zeros(42)
        if self.latest_results is not None and self.latest_results.hand_landmarks:
            for idx, landmarker in enumerate(self.latest_results.hand_landmarks):
                label = self.latest_results.handedness[idx][0].category_name

                if label == target:
                    pixel_points = np.array([[lm.x * iw, lm.y * ih] for lm in landmarker])
                    wrist = pixel_points[0]
                    translated_loads = pixel_points - wrist
                    distances = np.linalg.norm(translated_loads, axis=1)
                    mx_dis = np.max(distances)

                    if mx_dis > 0:
                        scaled_dis = translated_loads / mx_dis
                    else:
                        scaled_dis = translated_loads

                    norm_points = scaled_dis.flatten()
                    break
        return norm_points


    #norm features
    def get_norm_f(self, frame_shape):
        ih, iw, _ = frame_shape
        left_f = self.norm(ih, iw, 'Left')
        right_f = self.norm(ih, iw, 'Right')
        return np.hstack((left_f, right_f))


    #convert to rgb from bgr
    def frame_process(self, frame, ):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int((time.time() - self.start_time) * 1000)
        self.landmarker_model.detect_async(mp_image, ts)