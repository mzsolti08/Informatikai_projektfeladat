import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class ActivityDetector:
    def __init__(self):
        self.pose = mp_pose.Pose()

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            shoulder = landmarks[11].y
            hip = landmarks[23].y

            if abs(shoulder - hip) < 0.1:
                return "UL"
            else:
                return "ALL"

        return "Ismeretlen"