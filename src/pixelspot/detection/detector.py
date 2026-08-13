from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        return results
    