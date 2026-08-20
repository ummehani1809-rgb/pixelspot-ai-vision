"""Tests for head-pose estimation.

The model wrapper is deliberately thin; everything that can be wrong lives
in ``yaw_from_keypoints``, so that is what gets pinned -- including the
mirror trap: the person's left eye is on the image's RIGHT when they face
the camera.
"""

from __future__ import annotations

from pixelspot.enrichment.head_pose import yaw_from_keypoints

MIN_CONF = 0.4


def keypoints(nose=None, left_eye=None, right_eye=None, left_ear=None, right_ear=None):
    """COCO order, confidence 0.9 for present points, 0.0 for absent."""
    return [
        (point[0], point[1], 0.9) if point else (0.0, 0.0, 0.0)
        for point in (nose, left_eye, right_eye, left_ear, right_ear)
    ]


def test_a_nose_centred_between_the_eyes_faces_the_camera():
    yaw = yaw_from_keypoints(
        keypoints(nose=(100, 105), left_eye=(110, 100), right_eye=(90, 100)),
        MIN_CONF,
    )
    assert yaw == 0.0


def test_a_nose_drifting_right_reads_as_looking_right():
    yaw = yaw_from_keypoints(
        keypoints(nose=(105, 105), left_eye=(110, 100), right_eye=(90, 100)),
        MIN_CONF,
    )
    assert 0 < yaw < 45


def test_a_nose_drifting_left_reads_as_looking_left():
    yaw = yaw_from_keypoints(
        keypoints(nose=(95, 105), left_eye=(110, 100), right_eye=(90, 100)),
        MIN_CONF,
    )
    assert -45 < yaw < 0


def test_one_eye_and_a_leading_nose_is_a_profile():
    looking_right = yaw_from_keypoints(
        keypoints(nose=(120, 105), left_eye=(110, 100)), MIN_CONF
    )
    looking_left = yaw_from_keypoints(
        keypoints(nose=(80, 105), right_eye=(90, 100)), MIN_CONF
    )
    assert looking_right == 90.0
    assert looking_left == -90.0


def test_ears_without_a_face_is_facing_away():
    yaw = yaw_from_keypoints(
        keypoints(left_ear=(85, 100), right_ear=(115, 100)), MIN_CONF
    )
    assert yaw == 180.0


def test_no_visible_head_at_all_is_unknown():
    assert yaw_from_keypoints(keypoints(), MIN_CONF) is None


def test_low_confidence_keypoints_are_treated_as_absent():
    # A confident nose but sub-threshold eyes: not enough to call a yaw.
    points = keypoints(nose=(100, 105))
    points[1] = (110.0, 100.0, 0.2)  # left eye below MIN_CONF
    points[2] = (90.0, 100.0, 0.2)
    assert yaw_from_keypoints(points, MIN_CONF) is None


def test_extreme_nose_drift_is_clamped_not_extrapolated():
    yaw = yaw_from_keypoints(
        keypoints(nose=(150, 105), left_eye=(110, 100), right_eye=(90, 100)),
        MIN_CONF,
    )
    assert yaw == 75.0
