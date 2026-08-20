"""Face finding.

The YOLO detector finds *people*; the gender/age/mood classifiers need
*faces*. This is the step in between: OpenCV's YuNet face detector runs
every ``every_n_frames``, each face is matched to the person track it
belongs to, and the crop lands on the track as ``face_crop``.

The crop lives in memory for exactly as long as the classifiers need it and
is never written anywhere -- ``privacy.store_crops`` stays false and this
module has no code path that could violate it.

Faces smaller than ``min_size_px`` are skipped rather than classified: a
twelve-pixel face produces a coin-flip with a confidence score attached,
which is worse than admitting "unknown".
"""

from __future__ import annotations

import cv2

from pixelspot import paths
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

log = get_logger(__name__)

DEFAULT_MODEL = "models/face_detection_yunet_2023mar.onnx"

FaceBox = tuple[float, float, float, float]  # x, y, w, h


def match_faces_to_tracks(
    faces: list[FaceBox], tracks: list[Track], min_size_px: int
) -> dict[int, FaceBox]:
    """Assign each face to the person track whose box contains its centre.

    A face belongs to the *upper half* of a person box -- feet do not have
    faces, and in a crowd the head of a near person often overlaps the body
    of a far one. Ambiguity goes to the narrowest containing box, the person
    the face most plausibly belongs to. Too-small faces are dropped here.
    """
    assigned: dict[int, FaceBox] = {}

    for face in faces:
        x, y, w, h = face
        if min(w, h) < min_size_px:
            continue
        cx, cy = x + w / 2, y + h / 2

        best_track, best_width = None, None
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            if not (x1 <= cx <= x2 and y1 <= cy <= (y1 + y2) / 2):
                continue
            width = x2 - x1
            if best_width is None or width < best_width:
                best_track, best_width = track, width

        if best_track is not None:
            current = assigned.get(best_track.id)
            # One face per person: keep the larger, likelier one.
            if current is None or w * h > current[2] * current[3]:
                assigned[best_track.id] = face

    return assigned


class FaceFinder:
    """Runs YuNet and writes ``face_crop`` onto person tracks."""

    def __init__(
        self,
        model_path: str,
        min_size_px: int = 32,
        every_n_frames: int = 5,
        score_threshold: float = 0.6,
    ):
        resolved = paths.resolve(model_path)
        if not resolved.exists():
            raise ConfigError(f"face detection model not found: {resolved}")
        self.detector = cv2.FaceDetectorYN.create(
            str(resolved), "", (320, 320), score_threshold
        )
        self.min_size_px = min_size_px
        self.every_n_frames = every_n_frames
        self._input_size: tuple[int, int] | None = None
        log.info("loading model %s", resolved)

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "FaceFinder":
        settings = config.perception.enrichment.face
        return cls(
            model_path=settings.model or DEFAULT_MODEL,
            min_size_px=settings.min_size_px,
            every_n_frames=settings.every_n_frames,
        )

    def enrich(self, frame, tracks: list[Track], frame_index: int) -> None:
        """Set ``face_crop`` on person tracks, in place.

        Crops appear only on detection frames; in between they stay ``None``
        so a classifier never works from a stale face.
        """
        people = [track for track in tracks if track.label.lower() == "person"]
        if not people or frame_index % self.every_n_frames != 0:
            return

        height, width = frame.shape[:2]
        if self._input_size != (width, height):
            self.detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, detections = self.detector.detect(frame)
        if detections is None:
            return

        faces = [tuple(float(v) for v in row[:4]) for row in detections]
        for track_id, (x, y, w, h) in match_faces_to_tracks(
            faces, people, self.min_size_px
        ).items():
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(width, int(x + w)), min(height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            for track in people:
                if track.id == track_id:
                    track.face_crop = frame[y1:y2, x1:x2].copy()
                    break
