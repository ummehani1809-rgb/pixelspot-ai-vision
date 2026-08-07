from ultralytics import YOLO
import cv2
import time



model = YOLO("yolo11n.pt")


 #Webcam
#cap = cv2.VideoCapture(0)
#cap.set(480, 720)

# OR Video File
cap = cv2.VideoCapture(r"C:\Users\ummeh\Downloads\20260712-uhd_3840_2160_30fps.mp4")   # Replace with your video path
#cap.set(4, 480)
#cap.set(3,480)


prev_time = 0

while True:

    success, img = cap.read()

    if not success:
        print("Video Finished!")
        break
    img = cv2.resize(img, (1280, 720))

    # Run YOLO
    results = model(
    img,
    stream= True)

    person_count = 0
    vehicle_count = 0

    for r in results:

        for box in r.boxes:

            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Count Persons
            if class_name == "person":
                person_count += 1

            # Count Vehicles
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

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    #Display Counts
    cv2.putText(
        img,
        f"People : {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Vehicles : {vehicle_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"FPS : {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("YOLO Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()