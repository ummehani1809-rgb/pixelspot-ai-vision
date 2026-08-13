from ultralytics import YOLO
import cv2
import time


# =========================
# MODEL
# =========================

model = YOLO("yolo11n.pt")


# =========================
# VIDEO SOURCE
# =========================

# Webcam:
# cap = cv2.VideoCapture(0)

# Video:
cap = cv2.VideoCapture(
    r"C:\Users\ummeh\Downloads\4750042-hd_1920_1080_30fps.mp4"
)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")


# =========================
# FOOTFALL VARIABLES
# =========================

people_entered = 0
people_exited = 0

entered_ids = set()
exited_ids = set()

previous_positions = {}

peak_count = 0

# Counting line
line_y = 400

# FPS
prev_time = 0


# =========================
# MAIN LOOP
# =========================

while True:

    success, img = cap.read()

    if not success:
        print("Video ended or camera could not be read.")
        break


    # =========================
    # YOLO + BYTETRACK
    # =========================

    results = model.track(
        img,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )

    r = results[0]


    # =========================
    # PROCESS DETECTIONS
    # =========================

    if r.boxes is not None:

        for box in r.boxes:

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Confidence
            confidence = float(box.conf[0])

            # Class
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # We only want people
            if class_name != "person":
                continue

            # Tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])


            # =========================
            # CENTER POINT
            # =========================

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # =========================
            # DRAW PERSON
            # =========================

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2
            )

            cv2.circle(
                img,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )


            # =========================
            # LABEL
            # =========================

            label = (
                f"Person ID:{track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # =========================
            # FOOTFALL LOGIC
            # =========================

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]


                # =========================
                # ENTER
                # =========================

                if (
                    previous_y < line_y
                    and center_y >= line_y
                ):

                    if track_id not in entered_ids:

                        people_entered += 1

                        entered_ids.add(track_id)

                        print(
                            f"Person {track_id} ENTERED"
                        )


                # =========================
                # EXIT
                # =========================

                elif (
                    previous_y > line_y
                    and center_y <= line_y
                ):

                    if track_id not in exited_ids:

                        people_exited += 1

                        exited_ids.add(track_id)

                        print(
                            f"Person {track_id} EXITED"
                        )


            # Save current position
            previous_positions[track_id] = center_y


    # =========================
    # OCCUPANCY
    # =========================

    occupancy = max(
        0,
        people_entered - people_exited
    )


    # =========================
    # PEAK OCCUPANCY
    # =========================

    peak_count = max(
        peak_count,
        occupancy
    )


    # =========================
    # COUNTING LINE
    # =========================

    cv2.line(
        img,
        (0, line_y),
        (img.shape[1], line_y),
        (255, 0, 255),
        3
    )

    cv2.putText(
        img,
        "FOOTFALL COUNTING LINE",
        (20, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =========================
    # FPS
    # =========================

    current_time = time.time()

    if prev_time != 0:

        fps = 1 / (current_time - prev_time)

    else:

        fps = 0

    prev_time = current_time


    # =========================
    # DISPLAY ANALYTICS
    # =========================

    cv2.putText(
        img,
        f"People Entered : {people_entered}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"People Exited : {people_exited}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        img,
        f"Current Occupancy : {occupancy}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Peak Count : {peak_count}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"FPS : {int(fps)}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =========================
    # SHOW VIDEO
    # =========================

    cv2.imshow(
        "PixelSpot AI - Footfall Counter",
        img
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()