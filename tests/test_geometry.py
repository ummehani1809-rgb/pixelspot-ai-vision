"""Tests for geometry resolution.

Normalized coordinates and oriented lines are the two places where a sign or a
scale factor can be wrong in a way nothing crashes on -- the counters just
count the wrong thing. Every rule is pinned here, with no video involved.
"""

from __future__ import annotations

import pytest

from pixelspot.geometry import ResolvedGeometry, ResolvedLine, ResolvedZone
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import GeometryConfig


def geometry(**overrides) -> GeometryConfig:
    data = {
        "zones": [
            {
                "id": "storefront",
                "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
            }
        ],
        "lines": [{"id": "entrance", "p1": [0.0, 0.4], "p2": [1.0, 0.4]}],
    }
    data.update(overrides)
    return GeometryConfig.model_validate(data)


# ==================================================================
# RESOLUTION
# ==================================================================


def test_normalized_points_scale_to_the_frame():
    resolved = ResolvedGeometry.resolve(geometry(), 1000, 500)

    assert resolved.zone("storefront").points[0] == (100.0, 50.0)
    assert resolved.line("entrance").p1 == (0.0, 200.0)
    assert resolved.line("entrance").p2 == (1000.0, 200.0)


def test_pixel_space_is_passed_through_untouched():
    config = GeometryConfig.model_validate(
        {
            "coordinate_space": "pixels",
            "lines": [{"id": "entrance", "p1": [0, 400], "p2": [1920, 400]}],
        }
    )
    resolved = ResolvedGeometry.resolve(config, 1920, 1080)

    assert resolved.line("entrance").p2 == (1920.0, 400.0)


def test_same_config_resolves_to_the_same_relative_place_at_any_resolution():
    small = ResolvedGeometry.resolve(geometry(), 640, 360)
    large = ResolvedGeometry.resolve(geometry(), 1920, 1080)

    assert small.line("entrance").p1[1] / 360 == large.line("entrance").p1[1] / 1080


def test_unknown_id_names_what_is_available():
    resolved = ResolvedGeometry.resolve(geometry(), 100, 100)

    with pytest.raises(ConfigError, match="storefront"):
        resolved.zone("nope")
    with pytest.raises(ConfigError, match="entrance"):
        resolved.line("nope")


# ==================================================================
# ZONES
# ==================================================================


def test_zone_contains_uses_the_polygon_not_its_bounding_box():
    # An L shape: the top right quadrant is inside the bounding box but
    # outside the polygon.
    zone = ResolvedZone(
        id="l_shape",
        points=((0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)),
    )

    assert zone.contains((25, 25)) is True
    assert zone.contains((25, 75)) is True
    assert zone.contains((75, 75)) is False
    assert zone.contains((150, 25)) is False


# ==================================================================
# LINES
# ==================================================================


def test_positive_direction_orients_the_distance():
    line = ResolvedLine(id="entrance", p1=(0, 400), p2=(1000, 400), positive="down")

    assert line.oriented_distance((500, 450)) == pytest.approx(50.0)
    assert line.oriented_distance((500, 350)) == pytest.approx(-50.0)


def test_reversing_the_positive_direction_flips_the_sign():
    down = ResolvedLine(id="a", p1=(0, 400), p2=(1000, 400), positive="down")
    up = ResolvedLine(id="b", p1=(0, 400), p2=(1000, 400), positive="up")

    assert down.oriented_distance((500, 450)) == -up.oriented_distance((500, 450))


def test_orientation_survives_the_endpoints_being_swapped():
    forwards = ResolvedLine(id="a", p1=(0, 400), p2=(1000, 400), positive="down")
    backwards = ResolvedLine(id="b", p1=(1000, 400), p2=(0, 400), positive="down")

    assert forwards.oriented_distance((500, 450)) > 0
    assert backwards.oriented_distance((500, 450)) > 0


def test_diagonal_line_is_oriented_by_the_dominant_axis():
    line = ResolvedLine(id="diag", p1=(0, 0), p2=(100, 100), positive="right")

    # Below-right of the diagonal is the side "right" points at.
    assert line.oriented_distance((80, 20)) > 0
    assert line.oriented_distance((20, 80)) < 0


def test_span_position_says_where_along_the_segment_a_point_sits():
    line = ResolvedLine(id="entrance", p1=(0, 400), p2=(1000, 400))

    assert line.span_position((0, 400)) == pytest.approx(0.0)
    assert line.span_position((1000, 400)) == pytest.approx(1.0)
    assert line.span_position((1500, 400)) == pytest.approx(1.5)


def test_positive_direction_parallel_to_the_line_is_rejected():
    with pytest.raises(ConfigError, match="parallel"):
        ResolvedLine(id="entrance", p1=(0, 400), p2=(1000, 400), positive="right")


def test_line_shorter_than_a_pixel_is_rejected():
    config = GeometryConfig.model_validate(
        {"lines": [{"id": "tiny", "p1": [0.5, 0.5], "p2": [0.5001, 0.5]}]}
    )

    with pytest.raises(ConfigError, match="not a line"):
        ResolvedGeometry.resolve(config, 100, 100)
