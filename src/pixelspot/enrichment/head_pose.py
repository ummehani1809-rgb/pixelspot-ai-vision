"""Head pose from pose-model keypoints.

Answers one question per person: which way is their head pointing,
horizontally? The answer lands on the track as ``head_yaw_deg`` -- 0 facing
the camera, +/-90 in profile (positive looking toward image right), 180
facing away, ``None`` unknown -- and the attention analytics read it there.

The estimate comes from the five COCO face keypoints (nose, eyes, ears) of a
YOLO pose model, not from a dedicated head-pose network. The geometry is
simple: facing the camera, the nose sits midway between the eyes; as the
head turns, the nose slides toward one eye, and in profile only one eye
survives; from behind there are ears but no face at all. That is a
heuristic, but a cheap one that runs on the same class of model already
deployed, and the analytics only need "at the screen or not", not degrees of
arc accuracy.

Pose inference runs every ``every_n_frames`` and the last yaw is carried
between runs -- a head does not turn far in two frames at 15 fps, and this
halves the second model's cost.
"""

from __future__ import annotations

from pixelspot.detection.detector import load_yolo, resolve_device
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

log = get_logger(__name__)

# COCO keypoint indices.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4

Keypoint = tuple[float, float, float]  # x, y, confidence


def yaw_from_keypoints(
    keypoints: list[Keypoint], min_confidence: float
) -> float | None:
    """Horizontal head yaw from the five COCO face keypoints.

    0 = facing the camera, positive = looking toward image right, +/-90 =
    profile, 180 = facing away, None = not enough visible to say.
    """
    def visible(index: int) -> Keypoint | None:
        if index >= len(keypoints):
            return None
        keypoint = keypoints[index]
        return keypoint if keypoint[2] >= min_confidence else None

    nose = visible(NOSE)
    left_eye = visible(LEFT_EYE)      # the person's left: image right, frontal
    right_eye = visible(RIGHT_EYE)
    ears = [visible(LEFT_EAR), visible(RIGHT_EAR)]

    if nose and left_eye and right_eye:
        span = abs(left_eye[0] - right_eye[0])
        if span < 1e-6:
            return 0.0
        middle = (left_eye[0] + right_eye[0]) / 2
        # Nose drift across the eye line, scaled so a nose over an eye
        # (offset half the span) reads as a 45 degree turn.
        offset = (nose[0] - middle) / span
        return max(-75.0, min(75.0, offset * 90.0))

    if nose and (left_eye or right_eye):
        eye = left_eye or right_eye
        # One eye left: profile, looking whichever way the nose leads.
        return 90.0 if nose[0] > eye[0] else -90.0

    if not nose and not left_eye and not right_eye:
        if any(ears):
            return 180.0  # a head with ears but no face is facing away
        return None

    return None


def _iou(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class HeadPoseEstimator:
    """Runs the pose model and writes ``head_yaw_deg`` onto person tracks."""

    def __init__(
        self,
        model,
        every_n_frames: int = 2,
        min_keypoint_confidence: float = 0.4,
        device: str | None = None,
        match_iou: float = 0.3,
    ):
        self.model = model
        self.every_n_frames = every_n_frames
        self.min_keypoint_confidence = min_keypoint_confidence
        self.device = device
        self.match_iou = match_iou
        self._cache: dict[int, float | None] = {}

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "HeadPoseEstimator":
        settings = config.perception.enrichment.head_pose
        if settings.backend != "pose_keypoints":
            log.warning(
                "head_pose backend %r is not implemented; using pose_keypoints",
                settings.backend,
            )
        return cls(
            model=load_yolo(settings.model),
            every_n_frames=settings.every_n_frames,
            min_keypoint_confidence=settings.min_keypoint_confidence,
            device=resolve_device(config.runtime),
        )

    def enrich(self, frame, tracks: list[Track], frame_index: int) -> None:
        """Set ``head_yaw_deg`` on every person track, in place."""
        people = [track for track in tracks if track.label.lower() == "person"]
        if not people:
            self._cache.clear()
            return

        if frame_index % self.every_n_frames == 0:
            self._infer(frame, people)

        for track in people:
            track.head_yaw_deg = self._cache.get(track.id)

        # Keep the cache to the ids still alive.
        alive = {track.id for track in people}
        for track_id in [cached for cached in self._cache if cached not in alive]:
            del self._cache[track_id]

    def _infer(self, frame, people: list[Track]) -> None:
        kwargs = {"verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(frame, **kwargs)[0]

        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or keypoints.data is None:
            return

        detections: list[tuple[tuple[float, ...], list[Keypoint]]] = []
        for box, kpts in zip(boxes.xyxy, keypoints.data):
            points = [
                (float(x), float(y), float(conf)) for x, y, conf in kpts.tolist()
            ]
            detections.append((tuple(float(v) for v in box.tolist()), points))

        for track in people:
            best, best_iou = None, self.match_iou
            for bbox, points in detections:
                overlap = _iou(track.bbox, bbox)
                if overlap >= best_iou:
                    best, best_iou = points, overlap
            if best is not None:
                self._cache[track.id] = yaw_from_keypoints(
                    best, self.min_keypoint_confidence
                )
