import cv2
from ultralytics import YOLO
from activity_logger import ActivityLogger

model = YOLO("yolov8n.pt")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

logger = ActivityLogger()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# ================= FACE TRACKING =================

tracked_faces = {}
next_face_id = 1

def get_face_id(x, y):

    global next_face_id

    for face_id, (px, py) in tracked_faces.items():

        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5

        if dist < 80:
            tracked_faces[face_id] = (x, y)
            return face_id

    new_id = f"Face_{next_face_id}"
    tracked_faces[new_id] = (x, y)

    next_face_id += 1

    return new_id

# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ================= YOLO =================

    small = cv2.resize(frame, (640, 360))
    results = model(small, stream=True, verbose=False)

    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 360

    labels = []

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            labels.append(label)

            if label == "person":
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # ================= FACE DETECTION =================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    detected_person = "UNKNOWN"

    for (x, y, w, h) in faces:

        center_x = x + w // 2
        center_y = y + h // 2

        face_id = get_face_id(center_x, center_y)

        detected_person = face_id

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

        cv2.putText(
            frame,
            face_id,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

    face_detected = len(faces) > 0

    # ================= ACTIVITY =================

    activity_rules = {
        "cell phone": "TELEFONOZIK",
        "tv": "TV-T NEZ",
        "book": "OLVAS",
        "bottle": "ISZIK",
        "cup": "KÁVÉZIK",
        "keyboard": "GEPPEL DOLGOZIK",
        "laptop": "DOLGOZIK",
        "couch": "PIHEN",
        "chair": "PIHEN",
        "bed": "ALSZIK"
    }

    activity_text = "NINCS AKTIVITAS"

    if face_detected:

        found_activity = False

        for obj, activity in activity_rules.items():

            if obj in labels:
                activity_text = activity
                found_activity = True
                break

        if not found_activity:
            activity_text = "EMBER DETEKTALVA"

        if len(faces) >= 2:
            activity_text = f"TOBB SZEMELY - {activity_text}"

    # ================= LOG =================

    logger.log(detected_person, activity_text)

    # ================= UI =================

    cv2.putText(
        frame,
        activity_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        3
    )

    cv2.imshow("Tevekenysegfigyelo rendszer", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()