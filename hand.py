import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
pyautogui.FAILSAFE = False


class HandDetector:
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.canvas = np.zeros((int(self.vid.get(4)), int(self.vid.get(3))), np.uint8)

    def vid_capture(self):
        ret, frame = self.vid.read()
        frame = cv2.flip(frame, 1)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, imgRGB

    def detect_hands(self, frameRGB):
        hands_detected = self.hands.process(frameRGB)
        h, w, c = frameRGB.shape
        index_x = index_y = palm_x = thumb_x = 50  # Initially
        if hands_detected.multi_hand_landmarks:
            for hand in hands_detected.multi_hand_landmarks:
                for i, lm in enumerate(hand.landmark):
                    if i == 0:
                        palm_x = int(lm.x * w)+45
                    elif i == 4:
                        thumb_x = int(lm.x * w)+45
                    elif i == 8:
                        index_x = int(lm.x * w) + 45
                        index_y = int(lm.y * h)+25
            return index_x, index_y, palm_x, thumb_x

    def is_clicking(self, palm_x, thumb_x, clicked, init_time):
        contraction = (thumb_x - palm_x) if (thumb_x - palm_x) >= 0 else (thumb_x - palm_x) * -1
        if contraction < 100 and not clicked and time.time() - init_time > 3:
            return True
        elif contraction >= 100 and clicked:
            return False

    def main(self):
        init_time = time.time()
        while True:
            frame, frameRGB = self.vid_capture()
            if self.detect_hands(frameRGB):
                index_x, index_y, palm_x, thumb_x = self.detect_hands(frameRGB)
            else:
                index_x = index_y = palm_x = thumb_x = 50
            clicked = False
            if self.is_clicking(palm_x, thumb_x, clicked, init_time):#if clicked:
                clicked = True
                self.canvas = cv2.circle(self.canvas, (index_x+45, index_y-25), 3, (255, 0, 255), cv2.FILLED)
            cv2.imshow("frame", frame)
            cv2.imshow("canvas", self.canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

HandDetector().main()
