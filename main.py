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

activity_text = "NINCS AKTIVITAS"

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
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ================= FACE DETECTION =================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    face_detected = len(faces) > 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(frame, "face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ================= ACTIVITY =================

    if face_detected and "cell phone" in labels:
        activity_text = "TELEFONOZIK"

    elif face_detected and "bed" in labels:
        activity_text = "ALSZIK"

    elif face_detected and "couch" in labels:
        activity_text = "PIHEN"

    elif face_detected and "laptop" in labels:
        activity_text = "DOLGOZIK"
        
    elif len(faces) >= 2:
        activity_text = "TOBB SZEMELY"

    elif face_detected:
        activity_text = "NINCS JELENLET"

    else:
        activity_text = "NINCS AKTIVITAS"

    # ================= LOG =================

    logger.log(activity_text)

    # ================= UI =================

    cv2.putText(frame, activity_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    cv2.imshow("Tevekenysegfigyelo rendszer", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()