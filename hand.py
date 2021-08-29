import cv2
import mediapipe as mp


class HandDetector:
    vid = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils

    @classmethod
    def detect_hands(cls):
        while True:
            ret, frame = cls.vid.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_detected = cls.hands.process(imgRGB)
            index_cx = index_cy = None
            if hands_detected.multi_hand_landmarks:
                for hand in hands_detected.multi_hand_landmarks:
                    for id, lm in enumerate(hand.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 8:
                            index_cx = cx
                            index_cy = cy
                    cls.mpDraw.draw_landmarks(frame, hand, cls.mpHands.HAND_CONNECTIONS)
            map_frame = cv2.resize(frame, (1920, 1080))
            frame = cv2.circle(frame, (index_cx, index_cy), 3, (0, 255, 0), cv2.FILLED)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

HandDetector.detect_hands()
