"""Tests for the analytics processors.

These are the numbers a customer sees, so the awkward cases are pinned rather
than the happy path: a track loitering on the line, a track that is one frame
old, a track that walks past the end of the line. Each of those has a config
setting behind it, and each of those settings is exercised here.

Frames are synthesised, so nothing here loads a model or opens a video.
"""

from __future__ import annotations

import pytest

from pixelspot.aggregation.aggregator import Aggregator
from pixelspot.analytics.base import FrameContext
from pixelspot.analytics.anomaly import AnomalyProcessor
from pixelspot.analytics.attention import AttentionProcessor
from pixelspot.analytics.audience_flow import AudienceFlowProcessor
from pixelspot.analytics.base import ProcessorOutput
from pixelspot.analytics.crossing import LineCrossingCounter
from pixelspot.analytics.crowd_density import CrowdDensityProcessor
from pixelspot.analytics.dwell import DwellProcessor
from pixelspot.analytics.heatmap import HeatmapProcessor
from pixelspot.analytics.parking import ParkingProcessor
from pixelspot.analytics.queue import QueueProcessor
from pixelspot.analytics import registry
from pixelspot.analytics.registry import build_processors
from pixelspot.analytics.screen_visibility import ScreenVisibilityProcessor
from pixelspot.analytics.traffic_direction import TrafficDirectionProcessor
from pixelspot.analytics.vehicle import VehicleProcessor
from pixelspot.analytics.viewing_zone import ViewingZoneProcessor
from pixelspot.geometry import ResolvedGeometry, ResolvedLine
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

WIDTH, HEIGHT = 1000, 1000

CONFIG = {
    "source": {"type": "file", "uri": "datasets/crowd.mp4"},
    "perception": {
        "detector": {"classes": ["person", "car", "bus"]},
        "tracker": {"max_age_s": 2.0},
    },
    "geometry": {
        "zones": [
            {
                "id": "storefront",
                "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
            }
        ],
        "lines": [{"id": "entrance", "p1": [0.0, 0.4], "p2": [1.0, 0.4]}],
    },
    "analytics": {
        "footfall": {"enabled": True, "lines": ["entrance"]},
        "viewing_zone": {"enabled": True, "zones": ["storefront"]},
        "vehicles": {"enabled": True, "classes": ["car", "bus"]},
    },
}


def build_config(**overrides) -> PixelSpotConfig:
    data = {key: dict(value) for key, value in CONFIG.items()}
    for section, value in overrides.items():
        data[section] = value
    return PixelSpotConfig.model_validate(data)


def build_geometry(config: PixelSpotConfig) -> ResolvedGeometry:
    return ResolvedGeometry.resolve(config.geometry, WIDTH, HEIGHT)


def track(track_id: int, x: float, y: float, label: str = "person", hits: int = 10):
    """A 40x40 box centred on (x, y)."""
    return Track(
        id=track_id,
        label=label,
        confidence=0.9,
        bbox=(int(x - 20), int(y - 20), int(x + 20), int(y + 20)),
        hits=hits,
    )


def context(tracks: list[Track], index: int = 0, timestamp: float = 100.0):
    return FrameContext(
        index=index, timestamp=timestamp, width=WIDTH, height=HEIGHT, tracks=tracks
    )


def processor_named(processors, name):
    return next(processor for processor in processors if processor.name == name)


# ==================================================================
# LINE CROSSING
# ==================================================================


def counter(**overrides) -> LineCrossingCounter:
    line = ResolvedLine(id="entrance", p1=(0, 400), p2=(1000, 400), positive="down")
    return LineCrossingCounter(line=line, **overrides)


def test_crossing_the_positive_way_counts_once():
    subject = counter()

    assert subject.update([track(1, 500, 300)], 100.0) == []
    crossings = subject.update([track(1, 500, 500)], 100.1)

    assert [crossing.direction for crossing in crossings] == ["positive"]
    assert (subject.positive, subject.negative) == (1, 0)


def test_crossing_back_counts_the_other_way():
    subject = counter()
    subject.update([track(1, 500, 300)], 100.0)
    subject.update([track(1, 500, 500)], 100.1)
    crossings = subject.update([track(1, 500, 300)], 100.2)

    assert [crossing.direction for crossing in crossings] == ["negative"]
    assert (subject.positive, subject.negative) == (1, 1)


def test_appearing_on_a_side_is_not_a_crossing():
    subject = counter()

    subject.update([track(1, 500, 500)], 100.0)
    subject.update([track(1, 500, 520)], 100.1)

    assert (subject.positive, subject.negative) == (0, 0)


def test_hysteresis_absorbs_jitter_on_the_line():
    subject = counter(hysteresis_px=20)
    subject.update([track(1, 500, 300)], 100.0)

    # Wobbling either side of the line but never past the threshold.
    for step, y in enumerate([395, 405, 398, 402, 396]):
        subject.update([track(1, 500, y)], 100.1 + step * 0.1)

    assert (subject.positive, subject.negative) == (0, 0)

    subject.update([track(1, 500, 500)], 101.0)
    assert subject.positive == 1


def test_young_tracks_do_not_count():
    subject = counter(min_track_age_frames=5)

    subject.update([track(1, 500, 300, hits=1)], 100.0)
    subject.update([track(1, 500, 500, hits=2)], 100.1)

    assert subject.positive == 0


def test_cooldown_bounds_a_recycled_track_id():
    subject = counter(cooldown_s=1.0)

    subject.update([track(1, 500, 300)], 100.0)
    subject.update([track(1, 500, 500)], 100.1)  # counted
    subject.update([track(1, 500, 300)], 100.2)  # inside the cooldown
    subject.update([track(1, 500, 500)], 100.3)

    assert (subject.positive, subject.negative) == (1, 0)


def test_crossing_beyond_the_end_of_the_segment_is_ignored():
    line = ResolvedLine(id="short", p1=(0, 400), p2=(200, 400), positive="down")
    subject = LineCrossingCounter(line=line)

    subject.update([track(1, 800, 300)], 100.0)
    subject.update([track(1, 800, 500)], 100.1)

    assert subject.positive == 0


def test_state_for_vanished_tracks_is_forgotten():
    subject = counter(retention_s=1.0)
    subject.update([track(1, 500, 300)], 100.0)
    subject.update([], 102.0)

    # The id is reused later on the far side; without expiry that would look
    # like a crossing nobody made.
    subject.update([track(1, 500, 500)], 102.1)
    assert subject.positive == 0


# ==================================================================
# FOOTFALL
# ==================================================================


def test_footfall_counts_entries_exits_and_occupancy():
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    footfall.process(context([track(1, 500, 300), track(2, 600, 300)], 0, 100.0))
    output = footfall.process(
        context([track(1, 500, 500), track(2, 600, 500)], 1, 101.0)
    )

    assert output.metrics["people_entered"] == 2
    assert output.metrics["current_occupancy"] == 2
    assert [event.type for event in output.events] == ["ENTER", "ENTER"]
    assert output.events[0].data == {"track_id": 1, "line_id": "entrance"}

    output = footfall.process(context([track(1, 500, 300)], 2, 103.0))
    assert output.metrics["people_exited"] == 1
    assert output.metrics["current_occupancy"] == 1
    assert output.metrics["peak_count"] == 2


def test_footfall_ignores_classes_it_was_not_asked_about():
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    footfall.process(context([track(1, 500, 300, label="car")], 0, 100.0))
    output = footfall.process(context([track(1, 500, 500, label="car")], 1, 101.0))

    assert output.metrics["people_entered"] == 0


def test_footfall_direction_follows_the_configured_positive_side():
    geometry = dict(CONFIG["geometry"])
    geometry["lines"] = [
        {"id": "entrance", "p1": [0.0, 0.4], "p2": [1.0, 0.4], "positive": "up"}
    ]
    config = build_config(geometry=geometry)
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    footfall.process(context([track(1, 500, 300)], 0, 100.0))
    output = footfall.process(context([track(1, 500, 500)], 1, 101.0))

    # Downwards is now an exit, purely because of the config.
    assert output.metrics["people_exited"] == 1
    assert output.metrics["people_entered"] == 0


# ==================================================================
# VIEWING ZONE
# ==================================================================


def test_viewing_zone_counts_only_people_inside_the_polygon():
    config = build_config()
    zone = ViewingZoneProcessor.from_config(config, build_geometry(config))

    output = zone.process(
        context(
            [
                track(1, 300, 300),  # inside
                track(2, 800, 800),  # outside
                track(3, 300, 300, label="car"),  # inside but not a person
            ]
        )
    )

    assert output.metrics["people_in_viewing_zone"] == 1
    assert output.metrics["per_zone"] == {"storefront": 1}


# ==================================================================
# VEHICLES
# ==================================================================


def test_vehicle_classes_come_from_the_config():
    config = build_config()
    vehicles = VehicleProcessor.from_config(config, build_geometry(config))

    output = vehicles.process(
        context(
            [
                track(1, 100, 100, label="car"),
                track(2, 200, 200, label="bus"),
                track(3, 300, 300, label="person"),
            ]
        )
    )

    assert output.metrics["total_vehicles"] == 2
    assert output.metrics["by_class"] == {"car": 1, "bus": 1}


def test_vehicles_can_be_restricted_to_zones():
    analytics = dict(CONFIG["analytics"])
    analytics["vehicles"] = {
        "enabled": True,
        "classes": ["car"],
        "zones": ["storefront"],
    }
    config = build_config(analytics=analytics)
    vehicles = VehicleProcessor.from_config(config, build_geometry(config))

    output = vehicles.process(
        context([track(1, 300, 300, label="car"), track(2, 900, 900, label="car")])
    )

    assert output.metrics["total_vehicles"] == 1


# ==================================================================
# DWELL
# ==================================================================


def build_dwell(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["dwell"] = {"enabled": True, "zones": ["storefront"], **settings}
    config = build_config(analytics=analytics)
    return DwellProcessor.from_config(config, build_geometry(config))


def test_a_walkthrough_shorter_than_min_dwell_emits_nothing():
    dwell = build_dwell(min_dwell_s=2.0)

    dwell.process(context([track(1, 300, 300)], timestamp=100.0))
    output = dwell.process(context([track(1, 800, 800)], timestamp=100.5))

    assert output.events == []
    assert output.metrics["completed_visits"] == 0


def test_a_stay_past_min_dwell_emits_one_event_on_exit():
    dwell = build_dwell(min_dwell_s=2.0)

    dwell.process(context([track(1, 300, 300)], timestamp=100.0))
    dwell.process(context([track(1, 310, 300)], timestamp=103.0))
    output = dwell.process(context([track(1, 800, 800)], timestamp=104.0))

    assert len(output.events) == 1
    event = output.events[0]
    assert event.type == "DWELL"
    assert event.data == {"track_id": 1, "zone_id": "storefront", "dwell_s": 3.0}
    assert output.metrics["completed_visits"] == 1
    assert output.metrics["avg_dwell_s"] == 3.0


def test_a_vanished_track_closes_only_after_tracker_retention():
    # tracker max_age_s is 2.0 in CONFIG
    dwell = build_dwell(min_dwell_s=2.0)

    dwell.process(context([track(1, 300, 300)], timestamp=100.0))
    dwell.process(context([track(1, 300, 300)], timestamp=103.0))

    # Gone for 1s: could be an occlusion, the visit stays open.
    output = dwell.process(context([], timestamp=104.0))
    assert output.events == []

    # Gone for 3s: the tracker itself would have dropped them. They left.
    output = dwell.process(context([], timestamp=106.0))
    assert len(output.events) == 1
    assert output.events[0].data["dwell_s"] == 3.0


def test_a_brief_occlusion_does_not_split_the_visit():
    dwell = build_dwell(min_dwell_s=2.0)

    dwell.process(context([track(1, 300, 300)], timestamp=100.0))
    dwell.process(context([], timestamp=101.0))  # hidden, within retention
    dwell.process(context([track(1, 300, 300)], timestamp=102.0))
    output = dwell.process(context([track(1, 800, 800)], timestamp=104.0))

    assert len(output.events) == 1
    assert output.events[0].data["dwell_s"] == 2.0


def test_dwell_duration_is_capped_at_max():
    dwell = build_dwell(min_dwell_s=2.0, max_dwell_s=5.0)

    dwell.process(context([track(1, 300, 300)], timestamp=100.0))
    dwell.process(context([track(1, 300, 300)], timestamp=200.0))
    output = dwell.process(context([track(1, 800, 800)], timestamp=201.0))

    assert output.events[0].data["dwell_s"] == 5.0


def test_currently_dwelling_needs_min_dwell_first():
    dwell = build_dwell(min_dwell_s=2.0)

    output = dwell.process(context([track(1, 300, 300)], timestamp=100.0))
    assert output.metrics["currently_dwelling"] == 0

    output = dwell.process(context([track(1, 300, 300)], timestamp=103.0))
    assert output.metrics["currently_dwelling"] == 1
    assert output.metrics["per_zone_dwelling"] == {"storefront": 1}


# ==================================================================
# CROWD DENSITY
# ==================================================================


def build_density(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["crowd_density"] = {
        "enabled": True,
        "zones": ["storefront"],
        "zone_areas_m2": {"storefront": 10.0},
        **settings,
    }
    config = build_config(analytics=analytics)
    return CrowdDensityProcessor.from_config(config, build_geometry(config))


def test_density_is_people_divided_by_configured_area():
    density = build_density()  # thresholds: low 0.3, medium 0.7, high 1.2

    output = density.process(
        context([track(i, 300, 300) for i in range(4)])  # 4 people / 10 m2
    )

    zone = output.metrics["per_zone"]["storefront"]
    assert zone == {"count": 4, "density": 0.4, "level": "low"}
    assert output.metrics["level"] == "low"


def test_each_level_change_is_one_event_named_after_the_level():
    density = build_density()

    first = density.process(context([track(i, 300, 300) for i in range(8)]))
    assert [event.type for event in first.events] == ["DENSITY_MEDIUM"]

    # Same level again: no new event.
    steady = density.process(context([track(i, 300, 300) for i in range(8)]))
    assert steady.events == []

    # Crowd grows past the high threshold: one event.
    surge = density.process(context([track(i, 300, 300) for i in range(13)]))
    assert [event.type for event in surge.events] == ["DENSITY_HIGH"]
    assert surge.events[0].data["zone_id"] == "storefront"


def test_a_zone_without_an_area_is_rejected_by_the_config():
    with pytest.raises(Exception, match="zone_areas_m2 has no entry"):
        build_density(zone_areas_m2={})


# ==================================================================
# HEATMAP
# ==================================================================


def build_heatmap(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["heatmap"] = {"enabled": True, "grid": [10, 10], **settings}
    config = build_config(analytics=analytics)
    return HeatmapProcessor.from_config(config, build_geometry(config))


def test_presence_accumulates_in_the_cell_under_the_person():
    heatmap = build_heatmap(decay=1.0)  # no fade, pure accumulation

    for index in range(3):
        output = heatmap.process(context([track(1, 250, 250)], index=index))

    # (250, 250) in a 1000x1000 frame on a 10x10 grid is cell (2, 2).
    assert output.metrics["hot_cells"][0] == {"x": 2, "y": 2, "weight": 3.0}
    assert output.metrics["max_weight"] == 3.0


def test_decay_cools_a_spot_nobody_stands_in():
    heatmap = build_heatmap(decay=0.5)

    heatmap.process(context([track(1, 250, 250)]))
    output = heatmap.process(context([]))  # everyone left

    assert output.metrics["max_weight"] == 0.5


def test_empty_scene_reports_no_hot_cells():
    heatmap = build_heatmap()

    output = heatmap.process(context([]))

    assert output.metrics["hot_cells"] == []
    assert heatmap.overlay_lines(output.metrics) == ["Heatmap: quiet"]


def test_unavailable_weight_mode_falls_back_to_presence(caplog):
    heatmap = build_heatmap(weight_by="attention")

    assert isinstance(heatmap, HeatmapProcessor)
    assert "using 'presence'" in caplog.text


# ==================================================================
# TRAFFIC DIRECTION
# ==================================================================


def build_traffic(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["traffic_direction"] = {"enabled": True, **settings}
    config = build_config(analytics=analytics)
    return TrafficDirectionProcessor.from_config(config, build_geometry(config))


def walk(processor, positions, start=100.0, step=0.1, track_id=1):
    """Run one track through a list of (x, y) positions, one frame per step."""
    output = None
    for index, (x, y) in enumerate(positions):
        output = processor.process(
            context([track(track_id, x, y)], index=index, timestamp=start + index * step)
        )
    return output


def test_a_track_moving_right_reads_as_east():
    traffic = build_traffic(min_speed_px_s=15.0)

    output = walk(traffic, [(100 + i * 30, 500) for i in range(5)])

    assert output.metrics["moving"] == 1
    assert output.metrics["dominant"] == "E"


def test_a_track_moving_down_reads_as_south():
    traffic = build_traffic()

    output = walk(traffic, [(500, 100 + i * 30) for i in range(5)])

    assert output.metrics["dominant"] == "S"


def test_jitter_below_min_speed_is_stationary_not_a_direction():
    traffic = build_traffic(min_speed_px_s=15.0)

    # 1px per 0.1s frame = 10 px/s, under the 15 px/s floor.
    output = walk(traffic, [(500 + i, 500) for i in range(5)])

    assert output.metrics["moving"] == 0
    assert output.metrics["stationary"] == 1
    assert output.metrics["dominant"] is None


# ==================================================================
# QUEUE
# ==================================================================


def build_queue(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["queue"] = {
        "enabled": True,
        "zones": ["storefront"],
        "min_people": 3,
        "min_dwell_s": 2.0,
        "cluster_radius_px": 100,
        **settings,
    }
    config = build_config(analytics=analytics)
    return QueueProcessor.from_config(config, build_geometry(config))


def cluster_tracks(n=3, x=300, y=300, spacing=50):
    """n people in a line, `spacing` px apart -- inside the storefront zone."""
    return [track(i + 1, x + i * spacing, y) for i in range(n)]


def test_enough_people_waiting_together_forms_a_queue():
    queue = build_queue()

    queue.process(context(cluster_tracks(), timestamp=100.0))
    output = queue.process(context(cluster_tracks(), timestamp=103.0))

    assert [event.type for event in output.events] == ["QUEUE_FORMED"]
    assert output.events[0].data == {"zone_id": "storefront", "length": 3}
    assert output.metrics["longest_queue"] == 3


def test_people_scattered_across_the_zone_are_not_a_queue():
    queue = build_queue(cluster_radius_px=100)

    # All waiting long enough, but 150px apart: no chain links them.
    queue.process(context(cluster_tracks(spacing=150), timestamp=100.0))
    output = queue.process(context(cluster_tracks(spacing=150), timestamp=103.0))

    assert output.events == []
    assert output.metrics["longest_queue"] == 0


def test_people_who_just_arrived_are_not_yet_a_queue():
    queue = build_queue(min_dwell_s=10.0)

    queue.process(context(cluster_tracks(), timestamp=100.0))
    output = queue.process(context(cluster_tracks(), timestamp=101.0))

    assert output.events == []


def test_a_queue_dissolving_emits_cleared():
    queue = build_queue()

    queue.process(context(cluster_tracks(), timestamp=100.0))
    queue.process(context(cluster_tracks(), timestamp=103.0))
    output = queue.process(context([], timestamp=110.0))

    assert [event.type for event in output.events] == ["QUEUE_CLEARED"]
    assert output.metrics["queues"] == 0


# ==================================================================
# AUDIENCE FLOW
# ==================================================================

TWO_ZONE_GEOMETRY = {
    "zones": [
        {"id": "a", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]},
        {"id": "b", "points": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]},
    ],
    "lines": [{"id": "entrance", "p1": [0.0, 0.4], "p2": [1.0, 0.4]}],
}


def build_flow(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["footfall"] = {"enabled": False}
    analytics["viewing_zone"] = {"enabled": False}
    analytics["audience_flow"] = {
        "enabled": True,
        "zones": ["a", "b"],
        "min_transition_s": 0.5,
        **settings,
    }
    config = build_config(analytics=analytics, geometry=TWO_ZONE_GEOMETRY)
    return AudienceFlowProcessor.from_config(config, build_geometry(config))


def test_settling_in_a_new_zone_is_one_transition():
    flow = build_flow()

    flow.process(context([track(1, 250, 250)], timestamp=100.0))  # in a
    flow.process(context([track(1, 750, 750)], timestamp=101.0))  # candidate b
    output = flow.process(context([track(1, 750, 750)], timestamp=102.0))  # settled

    assert [event.type for event in output.events] == ["TRANSITION"]
    assert output.events[0].data == {"track_id": 1, "from_zone": "a", "zone_id": "b"}
    assert output.metrics["matrix"] == {"a->b": 1}


def test_a_flicker_across_the_boundary_does_not_count():
    flow = build_flow(min_transition_s=0.5)

    flow.process(context([track(1, 250, 250)], timestamp=100.0))  # in a
    flow.process(context([track(1, 750, 750)], timestamp=100.1))  # blips into b
    output = flow.process(context([track(1, 250, 250)], timestamp=100.2))  # back in a

    assert output.events == []
    assert output.metrics["matrix"] == {}


def test_entering_from_outside_any_zone_is_not_a_transition():
    flow = build_flow(min_transition_s=0.0)

    flow.process(context([track(1, 500, 100)], timestamp=100.0))  # in neither zone
    output = flow.process(context([track(1, 250, 250)], timestamp=101.0))  # enters a

    assert output.events == []
    assert output.metrics["matrix"] == {}


# ==================================================================
# PARKING
# ==================================================================

BAY_GEOMETRY = {
    "zones": [
        {
            "id": "bay_1",
            "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
            "tags": ["parking"],
        }
    ],
    "lines": [{"id": "entrance", "p1": [0.0, 0.4], "p2": [1.0, 0.4]}],
}


def build_parking(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["footfall"] = {"enabled": False}
    analytics["viewing_zone"] = {"enabled": False}
    analytics["parking"] = {
        "enabled": True,
        "bays": ["bay_1"],
        "occupied_after_s": 2.0,
        "vacated_after_s": 2.0,
        **settings,
    }
    config = build_config(analytics=analytics, geometry=BAY_GEOMETRY)
    return ParkingProcessor.from_config(config, build_geometry(config))


def parked_car(present=True):
    return [track(1, 250, 250, label="car")] if present else []


def test_a_car_must_settle_before_the_bay_reads_occupied():
    parking = build_parking(occupied_after_s=2.0)

    output = parking.process(context(parked_car(), timestamp=100.0))
    assert output.events == []
    assert output.metrics["per_bay"] == {"bay_1": "filling"}

    output = parking.process(context(parked_car(), timestamp=103.0))
    assert [event.type for event in output.events] == ["BAY_OCCUPIED"]
    assert output.metrics["bays_occupied"] == 1


def test_a_drive_through_never_occupies_the_bay():
    parking = build_parking(occupied_after_s=2.0)

    parking.process(context(parked_car(), timestamp=100.0))
    output = parking.process(context(parked_car(False), timestamp=100.5))

    assert output.events == []
    assert output.metrics["per_bay"] == {"bay_1": "vacant"}


def test_a_brief_occlusion_does_not_vacate_the_bay():
    parking = build_parking(occupied_after_s=0.0, vacated_after_s=2.0)

    parking.process(context(parked_car(), timestamp=100.0))
    parking.process(context(parked_car(), timestamp=101.0))  # occupied
    parking.process(context(parked_car(False), timestamp=102.0))  # hidden
    output = parking.process(context(parked_car(), timestamp=103.0))  # visible again

    assert output.metrics["per_bay"] == {"bay_1": "occupied"}
    assert not any(event.type == "BAY_VACATED" for event in output.events)


def test_a_bay_empty_long_enough_emits_vacated():
    parking = build_parking(occupied_after_s=0.0, vacated_after_s=2.0)

    parking.process(context(parked_car(), timestamp=100.0))
    parking.process(context(parked_car(), timestamp=101.0))  # occupied
    parking.process(context(parked_car(False), timestamp=102.0))  # emptying
    output = parking.process(context(parked_car(False), timestamp=105.0))

    assert [event.type for event in output.events] == ["BAY_VACATED"]
    assert output.metrics["per_bay"] == {"bay_1": "vacant"}
    assert output.metrics["bays_free"] == 1


# ==================================================================
# ANOMALY
# ==================================================================


def build_anomaly(*rules):
    analytics = dict(CONFIG["analytics"])
    analytics["anomaly"] = {"enabled": True, "rules": list(rules)}
    config = build_config(analytics=analytics)
    return AnomalyProcessor.from_config(config, build_geometry(config))


def frame_with_metric(value, timestamp=100.0, index=0):
    ctx = context([], index=index, timestamp=timestamp)
    ctx.outputs["footfall"] = ProcessorOutput(metrics={"current_occupancy": value})
    return ctx


OCCUPANCY_RULE = {
    "id": "crowded",
    "metric": "footfall.current_occupancy",
    "condition": "above",
    "threshold": 10,
}


def test_crossing_a_threshold_fires_once_not_every_frame():
    anomaly = build_anomaly(OCCUPANCY_RULE)

    calm = anomaly.process(frame_with_metric(5, timestamp=100.0))
    assert calm.events == []

    breach = anomaly.process(frame_with_metric(15, timestamp=101.0))
    assert [event.type for event in breach.events] == ["ANOMALY"]
    assert breach.events[0].data["rule_id"] == "crowded"
    assert breach.metrics["active"] == ["crowded"]

    still_breached = anomaly.process(frame_with_metric(20, timestamp=102.0))
    assert still_breached.events == []

    recovered = anomaly.process(frame_with_metric(3, timestamp=103.0))
    assert [event.type for event in recovered.events] == ["ANOMALY_CLEARED"]
    assert recovered.metrics["active"] == []


def test_a_spike_is_judged_against_the_rules_own_history():
    anomaly = build_anomaly(
        {
            "id": "surge",
            "metric": "footfall.current_occupancy",
            "condition": "spike",
            "threshold": 3.0,
            "window": "10s",
        }
    )

    # A steady baseline of 2 for 8 seconds...
    for step in range(9):
        output = anomaly.process(frame_with_metric(2, timestamp=100.0 + step))
        assert output.events == []

    # ...then 10x that: a spike.
    output = anomaly.process(frame_with_metric(20, timestamp=109.0))
    assert [event.type for event in output.events] == ["ANOMALY"]


def test_a_rule_watching_a_missing_metric_warns_and_stays_silent(caplog):
    anomaly = build_anomaly(
        {
            "id": "ghost",
            "metric": "nope.nothing",
            "condition": "above",
            "threshold": 1,
        }
    )

    output = anomaly.process(frame_with_metric(5))

    assert output.events == []
    assert "not \nreported" not in caplog.text  # sanity: message is one line
    assert "ghost" in caplog.text and "nope.nothing" in caplog.text


# ==================================================================
# ATTENTION AND SCREEN VISIBILITY
# ==================================================================

# A screen at the top centre facing down into the scene, 120 degree cone.
SCREEN_GEOMETRY = {
    "zones": [
        {
            "id": "storefront",
            "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        }
    ],
    "lines": [{"id": "entrance", "p1": [0.0, 0.4], "p2": [1.0, 0.4]}],
    "screens": [
        {
            "id": "screen_a",
            "position": [0.5, 0.0],
            "facing": [0.0, 1.0],
            "fov_deg": 120,
        }
    ],
}

HEAD_POSE_PERCEPTION = {
    "detector": {"classes": ["person", "car", "bus"]},
    "tracker": {"max_age_s": 2.0},
    "enrichment": {"head_pose": {"enabled": True}},
}


def viewer(track_id, x, y, yaw, height=400):
    """A person `height` px tall centred on (x, y), head turned `yaw` degrees."""
    person = Track(
        id=track_id,
        label="person",
        confidence=0.9,
        bbox=(int(x - 50), int(y - height / 2), int(x + 50), int(y + height / 2)),
        hits=10,
        head_yaw_deg=yaw,
    )
    return person


def build_attention(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["attention"] = {
        "enabled": True,
        "zones": ["storefront"],
        "screen": "screen_a",
        "yaw_tolerance_deg": 35.0,
        "min_gaze_s": 1.0,
        **settings,
    }
    config = build_config(
        analytics=analytics,
        geometry=SCREEN_GEOMETRY,
        perception=HEAD_POSE_PERCEPTION,
    )
    return AttentionProcessor.from_config(config, build_geometry(config))


def test_a_sustained_frontal_gaze_counts_once():
    attention = build_attention(min_gaze_s=1.0)

    first = attention.process(context([viewer(1, 500, 500, yaw=5.0)], timestamp=100.0))
    assert first.metrics["attending"] == 0  # a glance is not attention yet

    held = attention.process(context([viewer(1, 500, 500, yaw=-10.0)], timestamp=101.5))
    assert held.metrics["attending"] == 1
    assert [event.type for event in held.events] == ["GAZE"]
    assert held.events[0].data == {"track_id": 1, "screen_id": "screen_a"}

    ongoing = attention.process(context([viewer(1, 500, 500, yaw=0.0)], timestamp=102.0))
    assert ongoing.metrics["attending"] == 1
    assert ongoing.events == []  # same look, no second event


def test_a_turned_head_is_not_attention():
    attention = build_attention()

    attention.process(context([viewer(1, 500, 500, yaw=90.0)], timestamp=100.0))
    output = attention.process(context([viewer(1, 500, 500, yaw=90.0)], timestamp=102.0))

    assert output.metrics["attending"] == 0
    assert output.events == []


def test_looking_away_and_back_is_a_new_look():
    attention = build_attention(min_gaze_s=1.0)

    attention.process(context([viewer(1, 500, 500, yaw=0.0)], timestamp=100.0))
    attention.process(context([viewer(1, 500, 500, yaw=0.0)], timestamp=101.5))  # GAZE 1
    attention.process(context([viewer(1, 500, 500, yaw=120.0)], timestamp=103.0))
    attention.process(context([viewer(1, 500, 500, yaw=0.0)], timestamp=104.0))
    output = attention.process(context([viewer(1, 500, 500, yaw=0.0)], timestamp=105.5))

    assert [event.type for event in output.events] == ["GAZE"]
    assert output.metrics["gazes_total"] == 2


def test_a_person_beside_the_screen_cone_cannot_attend():
    attention = build_attention()

    # Level with the screen edge: outside the 120 degree wedge.
    attention.process(context([viewer(1, 150, 120, yaw=0.0)], timestamp=100.0))
    output = attention.process(context([viewer(1, 150, 120, yaw=0.0)], timestamp=102.0))

    assert output.metrics["attending"] == 0


def test_an_unestimated_head_is_never_attending():
    attention = build_attention()

    attention.process(context([viewer(1, 500, 500, yaw=None)], timestamp=100.0))
    output = attention.process(context([viewer(1, 500, 500, yaw=None)], timestamp=102.0))

    assert output.metrics["attending"] == 0


def build_visibility(**settings):
    analytics = dict(CONFIG["analytics"])
    analytics["screen_visibility"] = {
        "enabled": True,
        "screen": "screen_a",
        **settings,
    }
    config = build_config(
        analytics=analytics,
        geometry=SCREEN_GEOMETRY,
        perception=HEAD_POSE_PERCEPTION,
    )
    return ScreenVisibilityProcessor.from_config(config, build_geometry(config))


def test_viewers_are_counted_wherever_they_look():
    visibility = build_visibility()

    output = visibility.process(
        context(
            [
                viewer(1, 500, 500, yaw=0.0),  # in the cone, facing the screen
                viewer(2, 600, 500, yaw=180.0),  # in the cone, facing away
                viewer(3, 150, 120, yaw=0.0),  # beside the cone
            ]
        )
    )

    assert output.metrics["viewers"] == 2


def test_a_person_beyond_max_distance_is_not_a_viewer():
    visibility = build_visibility(max_distance_m=8.0)

    # 400px tall in a 1000px frame is ~4.25m away; 100px is ~17m.
    output = visibility.process(
        context(
            [
                viewer(1, 500, 500, yaw=0.0, height=400),
                viewer(2, 600, 500, yaw=0.0, height=100),
            ]
        )
    )

    assert output.metrics["viewers"] == 1
    assert output.metrics["nearest_m"] == 4.2


# ==================================================================
# REGISTRY
# ==================================================================


def test_only_enabled_capabilities_are_built():
    config = build_config()
    processors = build_processors(config, build_geometry(config))

    assert [processor.name for processor in processors] == [
        "footfall",
        "viewing_zone",
        "vehicles",
    ]


def test_disabling_a_capability_removes_it_from_the_pipeline():
    analytics = dict(CONFIG["analytics"])
    analytics["viewing_zone"] = {"enabled": False}
    config = build_config(analytics=analytics)

    names = [
        processor.name for processor in build_processors(config, build_geometry(config))
    ]
    assert "viewing_zone" not in names


def test_enabled_but_unimplemented_capability_warns_instead_of_failing(
    caplog, monkeypatch
):
    # Every real capability is implemented now, so simulate a missing one.
    monkeypatch.delitem(registry.BUILDERS, "heatmap")
    analytics = dict(CONFIG["analytics"])
    analytics["heatmap"] = {"enabled": True}
    config = build_config(analytics=analytics)

    processors = build_processors(config, build_geometry(config))

    assert "heatmap" not in [processor.name for processor in processors]
    assert "no implementation yet" in caplog.text


# ==================================================================
# AGGREGATION
# ==================================================================


def build_aggregator(**overrides) -> Aggregator:
    settings = {"windows": ["1m", "1h"], "emit_interval_s": 60.0}
    settings.update(overrides)
    config = build_config(aggregation=settings)
    return Aggregator.from_config(config)


def run_frames(aggregator, footfall, frames):
    kept = []
    for index, (tracks, timestamp) in enumerate(frames):
        frame = context(tracks, index, timestamp)
        kept.extend(aggregator.update(frame, {"footfall": footfall.process(frame)}))
    return kept


def test_aggregator_keeps_the_latest_metrics_and_counts_events_per_window():
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")
    aggregator = build_aggregator()

    now = 1_000_000.0
    kept = run_frames(
        aggregator,
        footfall,
        [([track(1, 500, 300)], now), ([track(1, 500, 500)], now + 1)],
    )

    assert [event.type for event in kept] == ["ENTER"]
    assert aggregator.metrics["footfall"]["people_entered"] == 1

    counts = aggregator.window_counts(now + 2)
    assert counts["1m"] == {"footfall.ENTER": 1}
    # The same event has aged out of nothing yet, but will out of the minute.
    assert aggregator.window_counts(now + 120)["1m"] == {}
    assert aggregator.window_counts(now + 120)["1h"] == {"footfall.ENTER": 1}


def test_the_same_occurrence_reported_twice_is_suppressed():
    aggregator = build_aggregator(dedupe_events=True)
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    now = 1_000_000.0
    first = context([track(1, 500, 300)], 0, now)
    aggregator.update(first, {"footfall": footfall.process(first)})

    second = context([track(1, 500, 500)], 1, now + 1)
    crossing = footfall.process(second)
    assert len(aggregator.update(second, {"footfall": crossing})) == 1
    # Replaying the identical event, as a second processor watching the same
    # line would produce.
    assert aggregator.update(second, {"footfall": crossing}) == []
    assert aggregator.suppressed_events == 1


def test_a_genuine_repeat_outside_the_dedupe_window_still_counts():
    """Someone who walks in, out and back in did enter twice.

    Suppressing the second entry would leave the event stream disagreeing with
    the counter, which is worse than the duplicate dedupe exists to remove.
    """
    from pixelspot.aggregation.aggregator import DEDUPE_WINDOW_S

    aggregator = build_aggregator(dedupe_events=True)
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    now = 1_000_000.0
    first = context([track(1, 500, 300)], 0, now)
    aggregator.update(first, {"footfall": footfall.process(first)})

    second = context([track(1, 500, 500)], 1, now + 1)
    entered = footfall.process(second)
    assert len(aggregator.update(second, {"footfall": entered})) == 1

    later = context([track(1, 500, 500)], 2, now + 1 + DEDUPE_WINDOW_S + 0.1)
    repeat = footfall.process(context([track(1, 500, 300)], 2, now + 1.5))
    aggregator.update(later, {"footfall": repeat})
    again = footfall.process(later)

    assert [event.type for event in again.events] == ["ENTER"]
    assert len(aggregator.update(later, {"footfall": again})) == 1


def test_dedupe_can_be_turned_off():
    aggregator = build_aggregator(dedupe_events=False)
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    now = 1_000_000.0
    first = context([track(1, 500, 300)], 0, now)
    aggregator.update(first, {"footfall": footfall.process(first)})

    second = context([track(1, 500, 500)], 1, now + 1)
    crossing = footfall.process(second)
    assert len(aggregator.update(second, {"footfall": crossing})) == 1
    assert len(aggregator.update(second, {"footfall": crossing})) == 1


def test_snapshot_reports_metrics_windows_and_resets_the_interval():
    aggregator = build_aggregator(emit_interval_s=10.0)
    now = 1_000_000.0

    aggregator.update(context([], 0, now), {})
    assert aggregator.due() is False

    aggregator.update(context([], 1, now + 11), {})
    aggregator.metrics["footfall"] = {"people_entered": 3}
    assert aggregator.due() is True

    snapshot = aggregator.snapshot(fps=14.9)

    assert snapshot["metrics"]["footfall"] == {"people_entered": 3}
    assert set(snapshot["windows"]) == {"1m", "1h"}
    assert snapshot["fps"] == 14.9
    assert aggregator.due() is False


def test_history_older_than_the_longest_window_is_dropped():
    aggregator = build_aggregator(windows=["1m"])
    config = build_config()
    footfall = processor_named(build_processors(config, build_geometry(config)), "footfall")

    now = 1_000_000.0
    run_frames(
        aggregator,
        footfall,
        [([track(1, 500, 300)], now), ([track(1, 500, 500)], now + 1)],
    )
    assert len(aggregator._history) == 1

    aggregator._prune(now + 3600)
    assert len(aggregator._history) == 0
    # The running total is not history; it survives pruning.
    assert aggregator.total_events == 1


@pytest.mark.parametrize("emit", ["events", "metrics", "both"])
def test_console_sink_honours_what_it_was_asked_to_emit(emit, caplog):
    import logging

    from pixelspot.analytics.base import Event
    from pixelspot.settings.schema import ConsoleSinkConfig
    from pixelspot.sinks.console import ConsoleSink

    sink = ConsoleSink(ConsoleSinkConfig(enabled=True, emit=emit))
    caplog.set_level(logging.INFO)
    caplog.clear()

    sink.emit_events(
        [Event(type="ENTER", processor="footfall", timestamp=1.0, frame_index=7)]
    )
    sink.emit_metrics({"metrics": {"footfall": {"people_entered": 1}}, "windows": {}})

    assert ("ENTER" in caplog.text) is (emit in ("events", "both"))
    assert ("people_entered" in caplog.text) is (emit in ("metrics", "both"))
