import cv2

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_id_counter = 0
        self.tracked_faces = {}

    def detect_and_track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []

        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            name = self._match_face(center)

            results.append((y, x + w, y + h, x, name))

        return results

    def _match_face(self, center):
        for face_id, prev_center in self.tracked_faces.items():
            dist = ((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2) ** 0.5

            if dist < 50:
                self.tracked_faces[face_id] = center
                return f"Face_{face_id}"

        self.face_id_counter += 1
        self.tracked_faces[self.face_id_counter] = center
        return f"Face_{self.face_id_counter}"