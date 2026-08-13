from pixelspot.config import load_config
from pixelspot.ccms.client import CCMSClient

config = load_config()

features = config["features"]

ccms = CCMSClient(
    enabled=config["ccms"]["enabled"]
)

print("\n===== PixelSpot Features =====")

print("Footfall :", features["footfall"])
print("Attention:", features["attention"])
print("Gender   :", features["gender"])
print("Age      :", features["age"])
print("Mood     :", features["mood"])
print("Vehicles :", features["vehicles"])

print("==============================\n")

import cv2

from pixelspot.tracking.tracker import Tracker

from pixelspot.analytics.footfall import (
    FootfallProcessor
)

from pixelspot.analytics.vehicle import (
    VehicleProcessor
)

from pixelspot.analytics.viewing_zone import (
    ViewingZoneProcessor
)

from pixelspot.aggregation.aggregator import (
    Aggregator
)


# ==================================================
# COMPONENTS
# ==================================================

tracker = Tracker()

footfall = FootfallProcessor(
    line_y=400
)

vehicle = VehicleProcessor()

viewing_zone = ViewingZoneProcessor()

aggregator = Aggregator()


# ==================================================
# VIDEO
# ==================================================

cap = cv2.VideoCapture(
    r"C:\Users\ummeh\Downloads\4750042-hd_1920_1080_30fps.mp4"
)

if not cap.isOpened():

    print("ERROR: Could not open video")

    exit()


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Video ended.")

        break


    # ==================================================
    # TRACKING
    # ==================================================

    tracks = tracker.track(frame)


    # ==================================================
    # PEOPLE
    # ==================================================

    people = [

        track

        for track in tracks

        if track["class"] == "person"

    ]


    # ==================================================
    # FOOTFALL
    # ==================================================

    footfall_result = footfall.process(
        people
    )


    # ==================================================
    # VEHICLES
    # ==================================================

    vehicle_result = vehicle.process(
        tracks
    )


    # ==================================================
    # VIEWING ZONE
    # ==================================================

    viewing_result = viewing_zone.process(
        people
    )


    # ==================================================
    # AGGREGATE
    # ==================================================

    metrics = aggregator.aggregate(

        footfall_result,

        vehicle_result,

        viewing_result
    )


    # ==================================================
    # DRAW TRACKS
    # ==================================================

    for track in tracks:

        x1, y1, x2, y2 = track["bbox"]

        track_id = track["id"]

        class_name = track["class"]

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (255, 0, 255),

            2
        )

        cv2.putText(

            frame,

            f"{class_name} ID:{track_id}",

            (x1, max(y1 - 10, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            2
        )


    # ==================================================
    # DRAW FOOTFALL LINE
    # ==================================================

    line_y = 400

    cv2.line(

        frame,

        (0, line_y),

        (frame.shape[1], line_y),

        (255, 0, 255),

        3
    )


    cv2.putText(

        frame,

        "FOOTFALL LINE",

        (20, line_y - 10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ==================================================
    # DRAW VIEWING ZONE
    # ==================================================

    cv2.rectangle(

        frame,

        (200, 150),

        (600, 500),

        (255, 255, 0),

        2
    )


    # ==================================================
    # DISPLAY FOOTFALL
    # ==================================================

    cv2.putText(

        frame,

        f"Entered: "
        f"{metrics['footfall']['people_entered']}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )


    cv2.putText(

        frame,

        f"Exited: "
        f"{metrics['footfall']['people_exited']}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 0, 255),

        2
    )


    cv2.putText(

        frame,

        f"Occupancy: "
        f"{metrics['footfall']['current_occupancy']}",

        (20, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 0),

        2
    )


    cv2.putText(

        frame,

        f"Peak: "
        f"{metrics['footfall']['peak_count']}",

        (20, 145),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ==================================================
    # VEHICLES
    # ==================================================

    cv2.putText(

        frame,

        f"Vehicles: "
        f"{metrics['vehicles']['total_vehicles']}",

        (20, 180),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ==================================================
    # VIEWING ZONE
    # ==================================================

    cv2.putText(

        frame,

        f"Viewing Zone: "
        f"{metrics['viewing_zone']['people_in_viewing_zone']}",

        (20, 215),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ==================================================
    # EVENT LOG
    # ==================================================

    for event in metrics["footfall"]["events"]:

        print(

            f"Person {event['track_id']} "
            f"{event['type']}"

        )


    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow(

        "PixelSpot AI Analytics",

        frame

    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()