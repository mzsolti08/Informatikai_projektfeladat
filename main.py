import cv2
from modules.camera_module import init_camera
from modules.face_module import FaceDetector
from modules.activity_module import ActivityDetector
from utils.drawing_utils import draw_face, draw_activity
from utils.logger import log_event

def main():
    cap = init_camera()
    face_detector = FaceDetector()
    activity_detector = ActivityDetector()

    last_logged = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_and_track(frame)
        activity = activity_detector.detect(frame)

        for (top, right, bottom, left, name) in faces:
            draw_face(frame, top, right, bottom, left, name)

            # ne logoljon folyamatosan
            key = (name, activity)
            if key not in last_logged:
                log_event(name, activity)
                last_logged.add(key)

        draw_activity(frame, activity)

        cv2.imshow("Face & Activity System", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()