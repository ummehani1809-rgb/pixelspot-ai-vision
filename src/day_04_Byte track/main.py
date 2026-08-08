from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Webcam
# cap = cv2.VideoCapture(0)

# Or video"
cap = cv2.VideoCapture(r"C:\Users\ummeh\Downloads\20260712-uhd_3840_2160_30fps.mp4")

prev_time = 0

while True:

    success, img = cap.read()

    if not success:
        break
    img = cv2.resize(img, (1280, 720))
    

    # -------- TRACK INSTEAD OF DETECT --------
    results = model.track(
        img,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    person_count = 0
    vehicle_count = 0

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # -----------------------------
            # Tracking ID
            # -----------------------------
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Count detections
            if class_name == "person":
                person_count += 1

            if class_name in ["car", "motorcycle", "bus", "truck"]:
                vehicle_count += 1

            # Draw only required classes
            if class_name in ["person", "car", "motorcycle", "bus", "truck"]:

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2
                )

                label = f"{class_name} ID:{track_id} {confidence:.2f}"

                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # ---------------- FPS ----------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        img,
        f"People : {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Vehicles : {vehicle_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"FPS : {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Day 4 - ByteTrack", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
