from ultralytics import YOLO
import cv2
import time

# =========================================================
# 1. LOAD YOLO MODEL
# =========================================================

model = YOLO("yolo11n.pt")


# =========================================================
# 2. VIDEO / CAMERA SOURCE
# =========================================================

# For webcam:
# cap = cv2.VideoCapture(0)

# For testing with a video:
cap = cv2.VideoCapture(r"C:\Users\ummeh\Downloads\4750042-hd_1920_1080_30fps.mp4")


# =========================================================
# 3. FOOTFALL VARIABLES
# =========================================================

people_entered = 0
people_exited = 0

# IDs that have already been counted
entered_ids = set()
exited_ids = set()

# Stores the previous Y position of every tracked person
previous_positions = {}

# Highest occupancy reached
peak_count = 0


# =========================================================
# 4. HORIZONTAL COUNTING LINE
# =========================================================

# Change this value depending on your video.
# Larger value = line lower in the image.

line_y = 400


# =========================================================
# 5. FPS
# =========================================================

prev_time = 0


# =========================================================
# 6. MAIN LOOP
# =========================================================

while True:

    success, img = cap.read()

    if not success:
        print("Video ended or camera could not be read.")
        break


    # =====================================================
    # 7. YOLO + BYTE TRACK
    # =====================================================

    results = model.track(
        img,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )


    # =====================================================
    # 8. PROCESS EACH DETECTED OBJECT
    # =====================================================

    for r in results:

        if r.boxes is None:
            continue


        for box in r.boxes:

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(int, box.xyxy[0])


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(box.conf[0])


            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            class_id = int(box.cls[0])
            class_name = model.names[class_id]


            # ------------------------------------------------
            # We only want PEOPLE
            # ------------------------------------------------

            if class_name != "person":
                continue


            # ------------------------------------------------
            # Tracking ID
            # ------------------------------------------------

            if box.id is None:
                continue

            track_id = int(box.id[0])


            # =================================================
            # 9. FIND CENTER OF PERSON
            # =================================================

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # =================================================
            # 10. DRAW PERSON
            # =================================================

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2
            )


            # Draw center point

            cv2.circle(
                img,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )


            # =================================================
            # 11. DISPLAY PERSON ID
            # =================================================

            label = f"Person ID:{track_id}  {confidence:.2f}"

            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # =================================================
            # 12. CHECK PREVIOUS POSITION
            # =================================================

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]


                # =================================================
                # ENTER
                # =================================================
                #
                # Person moves:
                #
                # ABOVE LINE
                #      ↓
                #      ↓
                # =====================  LINE
                #      ↓
                # BELOW LINE
                #
                # This assumes the entrance is BELOW the line.
                #

                if previous_y < line_y and center_y >= line_y:

                    if track_id not in entered_ids:

                        people_entered += 1

                        entered_ids.add(track_id)

                        print(
                            f"Person {track_id} ENTERED"
                        )


                # =================================================
                # EXIT
                # =================================================
                #
                # Person moves:
                #
                # BELOW LINE
                #      ↑
                #      ↑
                # =====================  LINE
                #      ↑
                # ABOVE LINE
                #

                elif previous_y > line_y and center_y <= line_y:

                    if track_id not in exited_ids:

                        people_exited += 1

                        exited_ids.add(track_id)

                        print(
                            f"Person {track_id} EXITED"
                        )


            # =================================================
            # 13. SAVE CURRENT POSITION
            # =================================================

            previous_positions[track_id] = center_y


    # =========================================================
    # 14. CURRENT OCCUPANCY
    # =========================================================

    occupancy = max(
        0,
        people_entered - people_exited
    )


    # =========================================================
    # 15. PEAK OCCUPANCY
    # =========================================================

    peak_count = max(
        peak_count,
        occupancy
    )


    # =========================================================
    # 16. DRAW COUNTING LINE
    # =========================================================

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


    # =========================================================
    # 17. CALCULATE FPS
    # =========================================================

    current_time = time.time()

    if prev_time != 0:

        fps = 1 / (current_time - prev_time)

    else:

        fps = 0

    prev_time = current_time


    # =========================================================
    # 18. DISPLAY PIXELSPOT ANALYTICS
    # =========================================================

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


    # =========================================================
    # 19. SHOW VIDEO
    # =========================================================

    cv2.imshow(
        "PixelSpot AI - Footfall Counter",
        img
    )


    # Press Q to quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# 20. CLEAN UP
# =========================================================

cap.release()
cv2.destroyAllWindows()