"""Shared machinery for the face-attribute processors (gender, age, mood).

One frame's guess about a face is noise: lighting flickers, heads turn,
models are wrong ten percent of the time on their best day. So nothing here
reports a single classification. Each track accumulates votes -- one per
classified frame, kept in a window of ``vote_window`` -- and only the
majority of at least ``MIN_VOTES`` votes becomes that person's settled
label. Guesses under ``min_confidence`` never even vote.

Privacy is structural, not a flag check: these processors emit metrics only
(aggregate counts of settled labels), never per-person events, and they read
face crops from memory without ever holding onto one past the
classification call.

The concrete classifier is injected: an OpenCV DNN model in production
(:class:`DnnClassifier`), a stub in tests. Backend ``none`` means no
classifier, and everyone honestly reports as unknown.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Callable

import cv2
import numpy as np

from pixelspot import paths
from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import ConfigError

log = get_logger(__name__)

# A majority of fewer than this many votes is a coin toss, not a consensus.
MIN_VOTES = 3

# Profile faces are where every gender/age model fails hardest -- half the
# face is missing. When head-pose enrichment is on, a person looking more
# than this far off-camera does not get classified at all; their next frontal
# frame will vote instead. None (head-pose off) classifies as before.
FRONTAL_MAX_YAW_DEG = 30.0

# And a majority that barely wins the window is a coin toss too. A model that
# flip-flops between two labels on the same face is saying it cannot read
# this one; a 52/48 split reported as a label would be a guess dressed up as
# a fact. The winner has to dominate the window before it is reported.
SETTLE_FRACTION = 2 / 3

# label, confidence -- or None when the classifier abstains.
Classification = tuple[str, float] | None
Classifier = Callable[[np.ndarray], Classification]


class DnnClassifier:
    """A single-image classifier on OpenCV's DNN engine.

    Wraps ``cv2.dnn`` so each attribute model is one declaration: path,
    input size, preprocessing, output labels. Output scores are softmaxed
    when they do not already sum to one, so confidence always means a
    probability whichever family the model came from.
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int],
        labels: list[str],
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        swap_rb: bool = False,
    ):
        resolved = paths.resolve(model_path)
        if not resolved.exists():
            raise ConfigError(f"classifier model not found: {resolved}")
        log.info("loading model %s", resolved)
        self.net = cv2.dnn.readNet(str(resolved))
        self.input_size = input_size
        self.labels = labels
        self.mean = mean
        self.scale = scale
        self.swap_rb = swap_rb

    def __call__(self, face: np.ndarray) -> Classification:
        blob = cv2.dnn.blobFromImage(
            face, self.scale, self.input_size, self.mean, swapRB=self.swap_rb
        )
        self.net.setInput(blob)
        scores = self.net.forward().flatten().astype(np.float64)

        total = scores.sum()
        if not (0.99 <= total <= 1.01):  # logits, not probabilities
            exps = np.exp(scores - scores.max())
            scores = exps / exps.sum()

        index = int(scores.argmax())
        return self.labels[index], float(scores[index])


class InsightFaceGenderAge:
    """One head of InsightFace's joint gender/age model.

    The model predicts both attributes from one 96x96 face: two gender
    logits and an age regression scaled to [0, 1]. Each processor wants a
    plain single-attribute classifier, so this wraps the net once per head
    -- the model is around a megabyte, so loading it twice costs less than
    the plumbing to share it would.

    The age head reports the predicted year count as its label (``"27"``)
    with full confidence: a regression has no softmax to read confidence
    from, and the vote-over-time window in :class:`AttributeProcessor` is
    already the noise filter. The processor's ``map_label`` turns years
    into its configured bucket.

    ``wants_aligned`` asks the processor for the track's InsightFace-framed
    crop (see :func:`pixelspot.enrichment.face.insightface_crop`) -- the
    exact square framing this model was trained on -- instead of the padded
    margin crop the GoogleNet-era models expect.
    """

    wants_aligned = True

    def __init__(self, model_path: str, head: str):
        resolved = paths.resolve(model_path)
        if not resolved.exists():
            raise ConfigError(f"classifier model not found: {resolved}")
        log.info("loading model %s (%s head)", resolved, head)
        self.net = cv2.dnn.readNet(str(resolved))
        self.head = head

    def __call__(self, face: np.ndarray) -> Classification:
        blob = cv2.dnn.blobFromImage(face, 1.0, (96, 96), (0.0, 0.0, 0.0), swapRB=True)
        self.net.setInput(blob)
        prediction = self.net.forward().flatten().astype(np.float64)

        if self.head == "gender":
            logits = prediction[:2]  # female, male
            exps = np.exp(logits - logits.max())
            probabilities = exps / exps.sum()
            index = int(probabilities.argmax())
            return ("female", "male")[index], float(probabilities[index])

        age_years = float(prediction[2]) * 100.0
        if not 0.0 <= age_years <= 120.0:
            return None  # the regression only leaves [0, 120] on garbage input
        return str(int(round(age_years))), 1.0


class AttributeProcessor(BaseProcessor):
    """Vote-over-time classification of one attribute per person."""

    def __init__(
        self,
        classifier: Classifier | None,
        labels: list[str],
        every_n_frames: int = 5,
        vote_window: int = 15,
        min_confidence: float = 0.6,
        map_label: Callable[[str], str | None] | None = None,
    ):
        self.classifier = classifier
        self.labels = labels
        self.every_n_frames = every_n_frames
        self.vote_window = vote_window
        self.min_confidence = min_confidence
        # Model vocabulary -> configured vocabulary; None drops the vote.
        self.map_label = map_label or (
            lambda label: label if label in labels else None
        )
        self._votes: dict[int, deque[str]] = {}
        if classifier is None:
            log.warning(
                "analytics.%s backend is 'none'; everyone will report as unknown",
                self.name,
            )

    def process(self, context: FrameContext) -> ProcessorOutput:
        people = [
            track for track in context.tracks if track.label.lower() == "person"
        ]

        if self.classifier is not None and context.index % self.every_n_frames == 0:
            for track in people:
                yaw = track.head_yaw_deg
                if yaw is not None and abs(yaw) > FRONTAL_MAX_YAW_DEG:
                    continue
                crop = track.face_crop
                if getattr(self.classifier, "wants_aligned", False):
                    aligned = getattr(track, "face_crop_aligned", None)
                    crop = aligned if aligned is not None else crop
                if crop is None:
                    continue
                result = self.classifier(crop)
                if result is None:
                    continue
                label, confidence = result
                log.debug(
                    "%s: track %d raw %s %.2f (face %dx%d)",
                    self.name, track.id, label, confidence,
                    crop.shape[1], crop.shape[0],
                )
                if confidence < self.min_confidence:
                    continue
                mapped = self.map_label(label)
                if mapped is None:
                    continue
                votes = self._votes.get(track.id)
                if votes is None:
                    votes = self._votes[track.id] = deque(maxlen=self.vote_window)
                votes.append(mapped)

        # Forget people the tracker no longer reports.
        alive = {track.id for track in people}
        for track_id in [known for known in self._votes if known not in alive]:
            del self._votes[track_id]

        counts = {label: 0 for label in self.labels}
        unknown = 0
        for track in people:
            settled = self._settled(track.id)
            if settled is None:
                unknown += 1
            else:
                counts[settled] += 1

        return ProcessorOutput(metrics={**counts, "unknown": unknown})

    def _settled(self, track_id: int) -> str | None:
        votes = self._votes.get(track_id)
        if votes is None or len(votes) < MIN_VOTES:
            return None
        label, count = Counter(votes).most_common(1)[0]
        if count / len(votes) < SETTLE_FRACTION:
            return None
        return label

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        parts = [
            f"{label} {metrics[label]}"
            for label in self.labels
            if metrics.get(label)
        ]
        if not parts and not metrics.get("unknown"):
            return []
        parts.append(f"? {metrics.get('unknown', 0)}")
        return [f"{self.name.capitalize()}: " + "  ".join(parts)]
