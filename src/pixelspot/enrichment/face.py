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

import math

import cv2
import numpy as np

from pixelspot import paths
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

log = get_logger(__name__)

DEFAULT_MODEL = "models/face_detection_yunet_2023mar.onnx"

FaceBox = tuple[float, float, float, float]  # x, y, w, h
Landmarks = tuple[tuple[float, float], ...]  # right eye, left eye, nose, mouth corners

# Context kept around the detected face box, as a fraction of its size. The
# age and gender models were trained on crops that include forehead, hairline
# and jaw -- the strongest age cues -- so a tight crop starves them.
CROP_MARGIN = 0.3

# InsightFace's genderage model was trained on one exact framing: a square
# window centred on the face box, 1.5x its larger side, warped to 96x96 with
# no rotation. A classifier fed a different framing than it was trained on
# loses accuracy before it even starts, so this transform is reproduced
# verbatim rather than approximated with the margin crop above.
INSIGHTFACE_INPUT = 96
INSIGHTFACE_BOX_SCALE = 1.5


def insightface_crop(frame, box: FaceBox, size: int = INSIGHTFACE_INPUT):
    """Warp the face into the square crop InsightFace models expect.

    Returns None for a degenerate box. Off-frame regions pad with black,
    same as InsightFace's own preprocessing.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    scale = size / (max(w, h) * INSIGHTFACE_BOX_SCALE)
    cx, cy = x + w / 2, y + h / 2
    matrix = np.array(
        [[scale, 0.0, size / 2 - scale * cx],
         [0.0, scale, size / 2 - scale * cy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR)


def align_face_crop(
    frame, box: FaceBox, landmarks: Landmarks | None, margin: float = CROP_MARGIN
):
    """Cut a face crop with context, rotated so the eyes are level.

    The classifiers were trained on aligned faces; a head tilted twenty
    degrees costs more accuracy than a worse model would. The rotation
    happens on the padded crop, not the frame, so it stays cheap, and the
    margin absorbs the corners rotation would otherwise clip.

    Returns None when the padded box has no area (a face at the frame edge).
    """
    height, width = frame.shape[:2]
    x, y, w, h = box

    pad_x, pad_y = w * margin, h * margin
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(width, int(x + w + pad_x))
    y2 = min(height, int(y + h + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]

    if landmarks is None:
        return crop.copy()

    (rx, ry), (lx, ly) = landmarks[0], landmarks[1]
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    if abs(angle) < 3.0:  # already level; skip the warp
        return crop.copy()

    crop_h, crop_w = crop.shape[:2]
    rotation = cv2.getRotationMatrix2D((crop_w / 2, crop_h / 2), angle, 1.0)
    return cv2.warpAffine(
        crop, rotation, (crop_w, crop_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


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

        # YuNet rows are x, y, w, h, then five landmark pairs (right eye,
        # left eye, nose, mouth corners), then the score.
        faces = []
        face_landmarks: dict[FaceBox, Landmarks] = {}
        for row in detections:
            box = tuple(float(v) for v in row[:4])
            faces.append(box)
            if len(row) >= 14:
                face_landmarks[box] = tuple(
                    (float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)
                )

        for track_id, box in match_faces_to_tracks(
            faces, people, self.min_size_px
        ).items():
            crop = align_face_crop(frame, box, face_landmarks.get(box))
            if crop is None:
                continue
            for track in people:
                if track.id == track_id:
                    track.face_crop = crop
                    track.face_crop_aligned = insightface_crop(frame, box)
                    break
