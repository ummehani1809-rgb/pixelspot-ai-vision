"""Tests for the face-attribute stack (gender/age/mood).

The DNN classifiers are declarations over cv2.dnn and get one integration
test each way; everything decision-shaped -- voting, confidence gates,
label mapping, face-to-track matching -- is pinned with a stub classifier
so no test here needs a model or a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixelspot.analytics.age import AgeProcessor
from pixelspot.analytics.attributes import MIN_VOTES, AttributeProcessor
from pixelspot.analytics.base import FrameContext
from pixelspot.analytics.gender import GenderProcessor
from pixelspot.enrichment.face import insightface_crop, match_faces_to_tracks
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

CROP = np.zeros((64, 64, 3), dtype=np.uint8)


def face_track(track_id=1, crop=CROP, bbox=(100, 100, 200, 400), **fields):
    return Track(
        id=track_id,
        label="person",
        confidence=0.9,
        bbox=bbox,
        hits=10,
        face_crop=crop,
        **fields,
    )


def frame(tracks, index=0):
    return FrameContext(
        index=index, timestamp=100.0, width=1000, height=1000, tracks=tracks
    )


def build(classifier, labels=("male", "female"), **kwargs):
    processor = AttributeProcessor(
        classifier=classifier,
        labels=list(labels),
        every_n_frames=1,
        vote_window=15,
        min_confidence=0.6,
        **kwargs,
    )
    processor.name = "gender"
    return processor


def test_one_confident_guess_is_not_enough_to_settle():
    processor = build(lambda crop: ("male", 0.9))

    output = processor.process(frame([face_track()]))

    assert output.metrics == {"male": 0, "female": 0, "unknown": 1}


def test_a_majority_of_votes_settles_the_label():
    processor = build(lambda crop: ("male", 0.9))

    for index in range(MIN_VOTES):
        output = processor.process(frame([face_track()], index=index))

    assert output.metrics == {"male": 1, "female": 0, "unknown": 0}


def test_low_confidence_guesses_never_vote():
    processor = build(lambda crop: ("male", 0.4))

    for index in range(10):
        output = processor.process(frame([face_track()], index=index))

    assert output.metrics["unknown"] == 1


def test_labels_outside_the_config_do_not_vote():
    # The mood model says "disgust"; the config does not track disgust.
    processor = build(lambda crop: ("disgust", 0.99), labels=("happy", "sad"))

    for index in range(10):
        output = processor.process(frame([face_track()], index=index))

    assert output.metrics == {"happy": 0, "sad": 0, "unknown": 1}


def test_a_person_without_a_face_crop_is_unknown_not_classified():
    calls = []

    def classifier(crop):
        calls.append(crop)
        return ("male", 0.9)

    processor = build(classifier)
    output = processor.process(frame([face_track(crop=None)]))

    assert calls == []
    assert output.metrics["unknown"] == 1


def test_a_sideways_face_does_not_vote():
    calls = []

    def classifier(crop):
        calls.append(crop)
        return ("male", 0.9)

    processor = build(classifier)
    output = processor.process(frame([face_track(head_yaw_deg=80.0)]))

    assert calls == []
    assert output.metrics["unknown"] == 1


def test_a_frontal_face_still_votes_when_head_pose_is_on():
    processor = build(lambda crop: ("male", 0.9))

    for index in range(MIN_VOTES):
        output = processor.process(
            frame([face_track(head_yaw_deg=10.0)], index=index)
        )

    assert output.metrics["male"] == 1


def test_a_classifier_that_wants_aligned_gets_the_aligned_crop():
    aligned = np.ones((96, 96, 3), dtype=np.uint8)
    received = []

    def classifier(crop):
        received.append(crop)
        return ("male", 0.9)

    classifier.wants_aligned = True
    processor = build(classifier)
    processor.process(frame([face_track(face_crop_aligned=aligned)]))

    assert received and received[0] is aligned


def test_votes_die_with_their_track():
    processor = build(lambda crop: ("male", 0.9))

    for index in range(MIN_VOTES):
        processor.process(frame([face_track()], index=index))
    processor.process(frame([]))  # the person leaves

    assert processor._votes == {}


def test_the_vote_window_lets_a_settled_label_flip():
    answers = iter(["male"] * 3 + ["female"] * 12)
    processor = build(lambda crop: (next(answers), 0.9))

    output = None
    for index in range(15):
        output = processor.process(frame([face_track()], index=index))

    assert output.metrics == {"male": 0, "female": 1, "unknown": 0}


# ------------------------------------------------------------------
# Age bucket mapping
# ------------------------------------------------------------------


def age_config(backend="none"):
    return PixelSpotConfig.model_validate(
        {
            "source": {"type": "file", "uri": "datasets/crowd.mp4"},
            "perception": {"enrichment": {"face": {"enabled": True}}},
            "analytics": {"age": {"enabled": True, "backend": backend}},
        }
    )


def test_model_buckets_land_in_configured_buckets_by_midpoint():
    processor = AgeProcessor.from_config(age_config(), geometry=None)

    # "25-32" has midpoint 28.5, which belongs to the default 20-34 bucket.
    assert processor.map_label("25-32") == "20-34"
    assert processor.map_label("0-2") == "0-12"
    assert processor.map_label("60-100") == "55-120"
    assert processor.map_label("not-a-bucket") is None


# ------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------


def gender_config(backend):
    return PixelSpotConfig.model_validate(
        {
            "source": {"type": "file", "uri": "datasets/crowd.mp4"},
            "perception": {"enrichment": {"face": {"enabled": True}}},
            "analytics": {"gender": {"enabled": True, "backend": backend}},
        }
    )


def test_backend_none_reports_everyone_unknown(caplog):
    processor = GenderProcessor.from_config(gender_config("none"), geometry=None)

    output = processor.process(frame([face_track()]))

    assert output.metrics["unknown"] == 1
    assert "everyone will report as unknown" in caplog.text


def test_an_unknown_backend_is_rejected_by_the_schema():
    with pytest.raises(Exception, match="unknown backend"):
        gender_config("tensorflow")


@pytest.mark.skipif(
    not Path("models/gender_googlenet.onnx").exists(),
    reason="model weights not downloaded",
)
def test_the_real_gender_model_loads_and_classifies():
    processor = GenderProcessor.from_config(gender_config("opencv_dnn"), geometry=None)

    label, confidence = processor.classifier(CROP)

    assert label in ("male", "female")
    assert 0.0 <= confidence <= 1.0


# ------------------------------------------------------------------
# Face-to-track matching
# ------------------------------------------------------------------


def person(track_id, bbox):
    return Track(id=track_id, label="person", confidence=0.9, bbox=bbox, hits=10)


def test_a_face_belongs_to_the_person_whose_upper_half_holds_it():
    tracks = [person(1, (100, 100, 200, 400)), person(2, (300, 100, 400, 400))]
    faces = [(120.0, 120.0, 50.0, 50.0)]  # centre (145, 145): person 1's head

    assigned = match_faces_to_tracks(faces, tracks, min_size_px=32)

    assert list(assigned) == [1]


def test_a_face_at_the_feet_matches_nobody():
    tracks = [person(1, (100, 100, 200, 400))]
    faces = [(120.0, 330.0, 50.0, 50.0)]  # centre in the lower half

    assert match_faces_to_tracks(faces, tracks, min_size_px=32) == {}


def test_tiny_faces_are_dropped_not_guessed_at():
    tracks = [person(1, (100, 100, 200, 400))]
    faces = [(140.0, 140.0, 12.0, 12.0)]

    assert match_faces_to_tracks(faces, tracks, min_size_px=32) == {}


def test_an_overlapping_pair_gives_the_face_to_the_narrower_box():
    # A near person's wide box swallows a far person's narrow one.
    tracks = [person(1, (50, 50, 450, 600)), person(2, (150, 100, 250, 350))]
    faces = [(180.0, 120.0, 40.0, 40.0)]  # inside both upper halves

    assigned = match_faces_to_tracks(faces, tracks, min_size_px=32)

    assert list(assigned) == [2]


# ------------------------------------------------------------------
# InsightFace crop framing
# ------------------------------------------------------------------


def test_insightface_crop_centres_the_face_in_a_model_sized_square():
    # A white face box on a black frame: after the warp, the face centre
    # must land at the crop centre and the framing (1.5x the box) must put
    # black background in the corners.
    frame_image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_image[200:300, 300:400] = 255  # face box at (300, 200), 100x100

    crop = insightface_crop(frame_image, (300.0, 200.0, 100.0, 100.0))

    assert crop.shape == (96, 96, 3)
    assert crop[48, 48].tolist() == [255, 255, 255]
    assert crop[2, 2].tolist() == [0, 0, 0]


def test_insightface_crop_rejects_a_degenerate_box():
    frame_image = np.zeros((480, 640, 3), dtype=np.uint8)

    assert insightface_crop(frame_image, (10.0, 10.0, 0.0, 40.0)) is None
