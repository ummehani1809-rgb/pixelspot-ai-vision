"""Multi-object tracking.

Detection answers "what is in this frame". Tracking answers "is that the same
person as last frame", which is the only reason a counter can say *how many*
people rather than how many person-shaped blobs it saw.

Everything comes from ``perception.tracker``:

``type`` / ``config``
    Which association algorithm, and the YAML that tunes it.
``max_age_s``
    How long a track survives an occlusion, in *seconds*. Trackers count in
    frames, so this is converted using ``runtime.target_fps`` and written into
    a derived tracker YAML.
``min_hits``
    How many frames a track must be seen before it is trusted. Ultralytics has
    no such knob, so it is applied here: young tracks are counted but withheld,
    which keeps a one-frame flicker of noise from ever reaching a counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pixelspot import paths
from pixelspot.detection.detector import InferenceParams, load_yolo
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import PixelSpotConfig

log = get_logger(__name__)

_BUILTIN_TRACKERS = {"bytetrack.yaml", "botsort.yaml"}


@dataclass
class Track:
    """One tracked object in one frame, in pixel coordinates."""

    id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    hits: int = 1
    # Filled by head-pose enrichment when enabled: 0 = facing the camera,
    # +/-90 = profile, 180 = facing away, None = not estimated.
    head_yaw_deg: float | None = None
    # Filled by face enrichment on detection frames: a BGR crop of this
    # person's face, held in memory for classification only -- never stored.
    face_crop: Any = None
    # The same face in InsightFace's fixed square framing, for classifiers
    # trained on that framing. Same lifetime and privacy rules as face_crop.
    face_crop_aligned: Any = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom centre of the box: where the object meets the floor."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


def _builtin_tracker_config(tracker_type: str) -> Path:
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{tracker_type}.yaml"


def resolve_tracker_config(config: PixelSpotConfig) -> str:
    """Return the tracker YAML path to hand to Ultralytics.

    Starts from the configured file (or the built-in for the configured type)
    and writes a derived copy with ``track_buffer`` set from ``max_age_s``, so
    a config expressed in seconds actually reaches the tracker. Any failure
    falls back to the stock file rather than taking the pipeline down over a
    tuning parameter.
    """
    tracker = config.perception.tracker
    candidate = paths.resolve(tracker.config)

    if candidate.exists():
        base = candidate
    elif tracker.config in _BUILTIN_TRACKERS:
        base = _builtin_tracker_config(tracker.type)
        if Path(tracker.config).stem != tracker.type:
            log.warning(
                "perception.tracker.config is %r but type is %r; using %s",
                tracker.config,
                tracker.type,
                base.name,
            )
    else:
        raise ConfigError(f"perception.tracker.config not found: {candidate}")

    track_buffer = max(1, round(tracker.max_age_s * config.runtime.target_fps))

    try:
        data = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
        if data.get("track_buffer") == track_buffer:
            return str(base)

        data["track_buffer"] = track_buffer
        derived = paths.OUTPUT_DIR / "tracker" / f"{tracker.type}.yaml"
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        log.debug(
            "tracker max_age_s=%.1f at %.1f fps -> track_buffer=%d (%s)",
            tracker.max_age_s,
            config.runtime.target_fps,
            track_buffer,
            derived,
        )
        return str(derived)
    except OSError as exc:
        log.warning("could not write derived tracker config (%s); using %s", exc, base)
        return str(base)


class Tracker:
    """Detect and associate in one pass, returning :class:`Track` objects."""

    def __init__(
        self,
        model,
        params: InferenceParams,
        tracker_config: str,
        min_hits: int = 1,
        max_age_s: float = 2.0,
    ):
        self.model = model
        self.params = params
        self.tracker_config = tracker_config
        self.min_hits = min_hits
        self.max_age_s = max_age_s
        self._hits: dict[int, int] = {}
        self._last_seen: dict[int, float] = {}

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "Tracker":
        model = load_yolo(config.perception.detector.model)
        params = InferenceParams.build(
            config.perception.detector, config.runtime, model.names
        )
        return cls(
            model=model,
            params=params,
            tracker_config=resolve_tracker_config(config),
            min_hits=config.perception.tracker.min_hits,
            max_age_s=config.perception.tracker.max_age_s,
        )

    def track(self, frame, timestamp: float) -> list[Track]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            **self.params.as_kwargs(),
        )

        tracks: list[Track] = []
        result = results[0]
        if result.boxes is None:
            self._expire(timestamp)
            return tracks

        for box in result.boxes:
            if box.id is None:
                # Detected but not yet associated with an identity.
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])

            hits = self._hits.get(track_id, 0) + 1
            self._hits[track_id] = hits
            self._last_seen[track_id] = timestamp

            if hits < self.min_hits:
                continue

            tracks.append(
                Track(
                    id=track_id,
                    label=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    hits=hits,
                )
            )

        self._expire(timestamp)
        return tracks

    def _expire(self, now: float) -> None:
        """Forget hit counts for ids the tracker itself has given up on."""
        cutoff = now - self.max_age_s
        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if last_seen < cutoff
        ]
        for track_id in stale:
            self._hits.pop(track_id, None)
            self._last_seen.pop(track_id, None)
