"""Mood distribution.

The FER+ expression model (OpenCV model zoo) speaks seven emotions; the
config declares which ones it cares about (default: neutral, happy, sad,
angry, surprised). Model labels outside the configured set -- disgust,
fear by default -- simply do not vote, rather than being shoehorned into
the nearest configured mood.

Biometric processing: aggregate counts only, no per-person output.
"""

from __future__ import annotations

from pixelspot.analytics.attributes import AttributeProcessor, DnnClassifier
from pixelspot.geometry import ResolvedGeometry
from pixelspot.settings.schema import PixelSpotConfig

DEFAULT_MODEL = "models/facial_expression_recognition_mobilefacenet_2022july.onnx"

# Output order of the model, mapped to plain mood words.
MODEL_LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


class MoodProcessor(AttributeProcessor):
    name = "mood"

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "MoodProcessor":
        settings = config.analytics.mood
        classifier = None
        if settings.backend == "opencv_dnn":  # the schema rejects unknown names
            classifier = DnnClassifier(
                model_path=settings.model or DEFAULT_MODEL,
                input_size=(112, 112),
                labels=MODEL_LABELS,
                scale=1.0 / 255.0,
                swap_rb=True,  # the FER+ model was trained on RGB
            )
        return cls(
            classifier=classifier,
            labels=[label.lower() for label in settings.labels],
            every_n_frames=settings.every_n_frames,
            vote_window=settings.vote_window,
            min_confidence=settings.min_confidence,
        )
