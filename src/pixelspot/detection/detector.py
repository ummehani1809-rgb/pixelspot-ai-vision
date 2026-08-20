"""Object detection.

Wraps Ultralytics YOLO so the rest of PixelSpot never passes a magic number to
it. Every inference argument -- weights, image size, confidence, IoU, device,
half precision, which classes to look for -- comes from
``perception.detector`` in the config.

Class *names* are resolved to the model's own class indices here, at
construction time. Asking for a class the weights were never trained on is a
configuration error worth failing on immediately, not something to discover
from an analytics counter that stays at zero all evening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pixelspot import paths
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import DetectorConfig, PixelSpotConfig, RuntimeConfig

log = get_logger(__name__)


def load_yolo(model_path: str):
    """Load YOLO weights, resolving *model_path* against the project root."""
    from ultralytics import YOLO  # imported lazily: it pulls in torch

    resolved = paths.resolve(model_path)
    if not resolved.exists():
        raise ConfigError(f"model weights not found: {resolved}")

    log.info("loading model %s", resolved)
    return YOLO(str(resolved))


def resolve_class_ids(names: dict[int, str], wanted: list[str]) -> list[int]:
    """Map configured class names onto the model's class indices."""
    by_name = {name.lower(): index for index, name in names.items()}
    ids: list[int] = []
    unknown: list[str] = []

    for label in wanted:
        index = by_name.get(label.lower())
        if index is None:
            unknown.append(label)
        else:
            ids.append(index)

    if unknown:
        raise ConfigError(
            f"perception.detector.classes contains {sorted(unknown)}, which this "
            f"model cannot detect. Available: {', '.join(sorted(by_name))}"
        )
    return sorted(set(ids))


def resolve_device(runtime: RuntimeConfig) -> str | None:
    """Translate ``runtime.device`` into what Ultralytics expects.

    ``auto`` becomes ``None`` so Ultralytics picks the best available device
    itself, which is the whole point of asking for auto.
    """
    return None if runtime.device == "auto" else runtime.device


@dataclass
class InferenceParams:
    """The per-call arguments shared by detection and tracking."""

    imgsz: int
    conf: float
    iou: float
    max_det: int
    half: bool
    classes: list[int]
    device: str | None

    @classmethod
    def build(
        cls,
        detector: DetectorConfig,
        runtime: RuntimeConfig,
        names: dict[int, str],
    ) -> "InferenceParams":
        device = resolve_device(runtime)
        half = detector.half

        # fp16 is a GPU feature. Silently running fp32 on CPU is fine; silently
        # crashing three hours into a deployment is not.
        if half and (device is None or device.startswith(("cpu", "mps"))):
            log.warning(
                "perception.detector.half is true but device is %s; "
                "running full precision",
                device or "auto",
            )
            half = False

        return cls(
            imgsz=detector.imgsz,
            conf=detector.conf,
            iou=detector.iou,
            max_det=detector.max_det,
            half=half,
            classes=resolve_class_ids(names, detector.classes),
            device=device,
        )

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "classes": self.classes,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        # Only passed when asked for: full precision is the default anyway, and
        # newer Ultralytics warns on every call that receives the flag at all.
        if self.half:
            kwargs["half"] = True
        return kwargs


class Detector:
    """Stateless detection. Use :class:`~pixelspot.tracking.tracker.Tracker`
    when identities across frames are needed."""

    def __init__(self, model, params: InferenceParams):
        self.model = model
        self.params = params

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "Detector":
        model = load_yolo(config.perception.detector.model)
        params = InferenceParams.build(
            config.perception.detector, config.runtime, model.names
        )
        return cls(model, params)

    def detect(self, frame):
        return self.model.predict(frame, **self.params.as_kwargs())
