"""Age bucket distribution.

The GoogleNet age model predicts one of eight research buckets (0-2, 4-6,
8-12, ... 60-100); the config declares its own buckets. The bridge is the
midpoint: a prediction of "25-32" is treated as "about 28" and lands in
whichever configured bucket holds 28. Votes are then cast in the config's
vocabulary, so changing buckets in the config never touches the model.

Biometric processing: aggregate counts only, no per-person output.
"""

from __future__ import annotations

from pixelspot.analytics.attributes import AttributeProcessor, DnnClassifier
from pixelspot.geometry import ResolvedGeometry
from pixelspot.settings.schema import PixelSpotConfig

DEFAULT_MODEL = "models/age_googlenet.onnx"

# GoogleNet (Adience) output buckets and the midpoint each one stands for.
MODEL_BUCKETS = [
    ("0-2", 1.0),
    ("4-6", 5.0),
    ("8-12", 10.0),
    ("15-20", 17.5),
    ("25-32", 28.5),
    ("38-43", 40.5),
    ("48-53", 50.5),
    ("60-100", 80.0),
]


def bucket_label(low: int, high: int) -> str:
    return f"{low}-{high}"


class AgeProcessor(AttributeProcessor):
    name = "age"

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "AgeProcessor":
        settings = config.analytics.age
        labels = [bucket_label(low, high) for low, high in settings.buckets]

        midpoint_of = dict(MODEL_BUCKETS)

        def into_configured_bucket(model_label: str) -> str | None:
            midpoint = midpoint_of.get(model_label)
            if midpoint is None:
                return None
            for (low, high) in settings.buckets:
                if low <= midpoint <= high:
                    return bucket_label(low, high)
            return None  # configured buckets do not cover this age

        classifier = None
        if settings.backend == "opencv_dnn":  # the schema rejects unknown names
            classifier = DnnClassifier(
                model_path=settings.model or DEFAULT_MODEL,
                input_size=(224, 224),
                labels=[label for label, _ in MODEL_BUCKETS],
                mean=(104.0, 117.0, 123.0),
            )
        return cls(
            classifier=classifier,
            labels=labels,
            every_n_frames=settings.every_n_frames,
            vote_window=settings.vote_window,
            min_confidence=settings.min_confidence,
            map_label=into_configured_bucket,
        )
