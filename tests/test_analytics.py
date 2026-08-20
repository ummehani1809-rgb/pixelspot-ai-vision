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
from pixelspot.analytics.crossing import LineCrossingCounter
from pixelspot.analytics.registry import build_processors
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


def test_enabled_but_unimplemented_capability_warns_instead_of_failing(caplog):
    analytics = dict(CONFIG["analytics"])
    analytics["dwell"] = {"enabled": True, "zones": ["storefront"]}
    config = build_config(analytics=analytics)

    processors = build_processors(config, build_geometry(config))

    assert "dwell" not in [processor.name for processor in processors]
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
