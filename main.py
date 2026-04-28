import cv2
from ultralytics import YOLO

# YOLO
model = YOLO("yolov8n.pt")

# Arc detektor (OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

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

            # PERSON SZŰRÉS
            if label == "person":
                w = x2 - x1
                h = y2 - y1

                if w < 80 or h < 120:
                    continue

                ratio = w / h
                if ratio > 1.2 or ratio < 0.3:
                    continue

                color = (255, 0, 0)  # kék
            else:
                color = (0, 255, 0)  # zöld

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ================= FACE DETECTION =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # 🔴 arc külön szín
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("AI Detection (YOLO + Face)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()