"""Tests for the plumbing around the analytics: source, pacing and sinks.

A tiny video is written to a temp directory rather than reaching for the real
datasets, so these stay fast and pass on a machine that has no media checked
out. No model is loaded anywhere here.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from pixelspot.analytics.base import Event
from pixelspot.app import FramePacer
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import (
    CCMSSinkConfig,
    JsonlSinkConfig,
    SourceConfig,
)
from pixelspot.sinks import SinkGroup
from pixelspot.sinks.base import BaseSink
from pixelspot.sinks.ccms import CCMSSink
from pixelspot.sinks.jsonl import JsonlSink
from pixelspot.source import PlaybackClock, VideoSource

FRAME_COUNT = 5


@pytest.fixture
def video(tmp_path):
    """A five frame 64x48 video file."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for index in range(FRAME_COUNT):
        frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("no mp4 encoder available in this OpenCV build")
    return path


# ==================================================================
# VIDEO SOURCE
# ==================================================================


def test_file_source_reads_to_the_end_and_stops(video):
    source = VideoSource(SourceConfig(type="file", uri=str(video)))
    source.open()

    assert (source.width, source.height) == (64, 48)

    frames = 0
    while source.read() is not None:
        frames += 1
    source.release()

    assert frames == FRAME_COUNT


def test_looping_source_starts_over_instead_of_stopping(video):
    source = VideoSource(SourceConfig(type="file", uri=str(video), loop=True))
    source.open()

    for _ in range(FRAME_COUNT * 2 + 1):
        assert source.read() is not None
    source.release()


def test_missing_file_is_a_config_error_naming_the_path(tmp_path):
    with pytest.raises(ConfigError, match="missing.mp4"):
        VideoSource(SourceConfig(type="file", uri=str(tmp_path / "missing.mp4")))


def test_relative_paths_resolve_from_the_project_root_not_the_cwd(video, monkeypatch):
    # A source that resolves against the working directory would break the
    # moment the process is started by systemd or a container entrypoint.
    monkeypatch.chdir(video.parent)
    with pytest.raises(ConfigError):
        VideoSource(SourceConfig(type="file", uri="clip.mp4"))


# ==================================================================
# REAL-TIME FILE PLAYBACK
# ==================================================================


def test_playback_clock_reports_how_far_behind_real_time():
    times = iter([0.0, 0.5])
    clock = PlaybackClock(fps=10.0, now=lambda: next(times))

    assert clock.behind() == 0  # first call anchors the clock
    clock.consume()

    # Half a second in, frame 5 is due and only one frame was taken.
    assert clock.behind() == 4


def test_playback_clock_treats_a_long_stall_as_a_pause_not_debt():
    times = iter([0.0, 100.0, 100.05])
    clock = PlaybackClock(fps=10.0, now=lambda: next(times))

    assert clock.behind() == 0
    clock.consume()

    # A 100 second gap is model warm-up or a debugger, not playback debt.
    assert clock.behind() == 0
    clock.consume()
    assert clock.behind() == 0  # and afterwards the clock is back on pace


def test_realtime_file_source_skips_the_frames_it_fell_behind_by(video):
    source = VideoSource(SourceConfig(type="file", uri=str(video), realtime=True))
    source.open()
    times = iter([0.0, 0.35])
    source._playback = PlaybackClock(fps=10.0, now=lambda: next(times))

    first = source.read()   # anchors the clock
    second = source.read()  # 0.35s later: frame 3 is due; 1 and 2 are skipped
    source.release()

    # Frames were written as flat images of value index * 40.
    assert abs(int(first[0, 0, 0]) - 0) < 20
    assert abs(int(second[0, 0, 0]) - 120) < 20
    assert source.frames_skipped == 2


def test_file_source_without_realtime_still_delivers_every_frame(video):
    source = VideoSource(SourceConfig(type="file", uri=str(video)))
    source.open()

    values = []
    while (frame := source.read()) is not None:
        values.append(int(frame[0, 0, 0]))
    source.release()

    assert len(values) == FRAME_COUNT
    assert source.frames_skipped == 0


# ==================================================================
# PACING
# ==================================================================


def test_pacer_holds_the_loop_to_the_target_rate():
    pacer = FramePacer(target_fps=50.0)  # 20ms budget

    start = time.monotonic()
    for _ in range(5):
        pacer.tick()
        pacer.wait()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.08  # 4 full budgets, allowing for the first tick
    assert pacer.fps == pytest.approx(50.0, rel=0.5)


def test_pacer_does_not_bank_time_when_the_loop_runs_slow():
    pacer = FramePacer(target_fps=100.0)
    pacer.tick()
    time.sleep(0.05)  # five budgets late

    start = time.monotonic()
    pacer.wait()
    pacer.wait()

    # A pacer that repaid its debt would return instantly twice over.
    assert time.monotonic() - start < 0.05


# ==================================================================
# SINKS
# ==================================================================


def event(event_type: str = "ENTER", track_id: int = 1) -> Event:
    return Event(
        type=event_type,
        processor="footfall",
        timestamp=1_000_000.0,
        frame_index=3,
        data={"track_id": track_id, "line_id": "entrance"},
    )


def test_jsonl_sink_writes_one_labelled_record_per_line(tmp_path):
    sink = JsonlSink(JsonlSinkConfig(enabled=True, path=str(tmp_path / "out.jsonl")))
    sink.emit_events([event(), event(track_id=2)])
    sink.emit_metrics({"metrics": {"footfall": {"people_entered": 2}}})
    sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [record["record"] for record in records] == ["event", "event", "metrics"]
    assert records[0]["track_id"] == 1
    assert records[2]["metrics"]["footfall"]["people_entered"] == 2


def test_jsonl_sink_rotates_instead_of_filling_the_disk(tmp_path):
    path = tmp_path / "out.jsonl"
    sink = JsonlSink(JsonlSinkConfig(enabled=True, path=str(path), rotate_mb=1))
    sink.rotate_bytes = 500  # keep the test to a few writes

    for index in range(40):
        sink.emit_events([event(track_id=index)])
    sink.close()

    assert path.exists()
    assert path.with_suffix(".jsonl.1").exists()
    assert path.stat().st_size < 500


def test_ccms_sink_batches_and_spools(tmp_path):
    config = CCMSSinkConfig(
        enabled=True,
        base_url="https://example.invalid",
        api_key="secret",
        device_id="device-1",
        batch_size=3,
        spool_dir=str(tmp_path / "spool"),
    )
    sink = CCMSSink(config)

    sink.emit_events([event(track_id=1), event(track_id=2)])
    assert list((tmp_path / "spool").glob("*.json")) == []  # below batch_size

    sink.emit_events([event(track_id=3)])
    spooled = list((tmp_path / "spool").glob("*.json"))
    assert len(spooled) == 1

    payload = json.loads(spooled[0].read_text(encoding="utf-8"))
    assert payload["device_id"] == "device-1"
    assert len(payload["records"]) == 3
    assert "api_key" not in spooled[0].read_text(encoding="utf-8")


def test_ccms_sink_flushes_a_partial_batch_on_close(tmp_path):
    config = CCMSSinkConfig(
        enabled=True,
        base_url="https://example.invalid",
        api_key="secret",
        device_id="device-1",
        batch_size=50,
        spool_dir=str(tmp_path / "spool"),
    )
    sink = CCMSSink(config)
    sink.emit_events([event()])
    sink.close()

    assert len(list((tmp_path / "spool").glob("*.json"))) == 1


class _ExplodingSink(BaseSink):
    name = "exploding"

    def emit_events(self, events):
        raise RuntimeError("disk on fire")


class _RecordingSink(BaseSink):
    name = "recording"

    def __init__(self):
        self.seen: list[Event] = []

    def emit_events(self, events):
        self.seen.extend(events)


def test_a_failing_sink_is_dropped_and_the_others_keep_working(caplog):
    recording = _RecordingSink()
    group = SinkGroup([_ExplodingSink(), recording])

    group.emit_events([event()])
    group.emit_events([event(track_id=2)])

    assert len(recording.seen) == 2
    assert [sink.name for sink in group.sinks] == ["recording"]
    assert "disabled" in caplog.text
