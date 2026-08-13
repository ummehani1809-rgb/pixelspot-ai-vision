from ultralytics import YOLO


class Tracker:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.5
    ):

        self.model = YOLO(model_path)
        self.confidence = confidence

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            verbose=False
        )

        tracks = []

        result = results[0]

        if result.boxes is None:
            return tracks

        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = result.names[class_id]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            tracks.append({
                "id": track_id,
                "class": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "center_x": center_x,
                "center_y": center_y
            })

        return tracks
    