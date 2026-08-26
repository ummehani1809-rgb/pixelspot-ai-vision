"""Gender distribution.

Counts settled gender labels among the people currently tracked, using the
voting machinery from :mod:`pixelspot.analytics.attributes`. The default
``opencv_dnn`` backend is the GoogleNet gender model from the ONNX model
zoo, whose vocabulary happens to match the default config labels once
lowercased.

Biometric processing: aggregate counts only, no per-person output, and the
schema refuses a real backend unless face enrichment is explicitly on.
"""

from __future__ import annotations

from pixelspot.analytics.attributes import (
    AttributeProcessor,
    DnnClassifier,
    InsightFaceGenderAge,
)
from pixelspot.geometry import ResolvedGeometry
from pixelspot.settings.schema import PixelSpotConfig

DEFAULT_MODEL = "models/gender_googlenet.onnx"
INSIGHTFACE_MODEL = "models/genderage.onnx"

# GoogleNet output order.
MODEL_LABELS = ["male", "female"]


class GenderProcessor(AttributeProcessor):
    name = "gender"

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "GenderProcessor":
        settings = config.analytics.gender
        classifier = None
        if settings.backend == "opencv_dnn":  # the schema rejects unknown names
            classifier = DnnClassifier(
                model_path=settings.model or DEFAULT_MODEL,
                input_size=(224, 224),
                labels=MODEL_LABELS,
                mean=(104.0, 117.0, 123.0),
            )
        elif settings.backend == "insightface":
            classifier = InsightFaceGenderAge(
                model_path=settings.model or INSIGHTFACE_MODEL, head="gender"
            )
        return cls(
            classifier=classifier,
            labels=[label.lower() for label in settings.labels],
            every_n_frames=settings.every_n_frames,
            vote_window=settings.vote_window,
            min_confidence=settings.min_confidence,
        )
