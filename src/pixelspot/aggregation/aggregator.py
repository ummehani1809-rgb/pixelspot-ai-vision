"""Metric aggregation.

A sink does not want thirty frames a second of near-identical numbers, and a
dashboard cannot answer "how many people came in during the last hour" from a
single instantaneous count. This is the layer in between: it holds the latest
level from every processor, keeps a bounded history of events, and rolls that
history up over the windows named in ``aggregation.windows``.

Everything here is config driven:

``windows``
    Which time spans to roll events up over. The longest one also decides how
    much history is kept -- older events can no longer affect any answer, so
    they are dropped and memory stays bounded however long the process runs.
``emit_interval_s``
    How often a snapshot is due. The pipeline asks; this class answers.
    Elapsed time is measured with the frame timestamps the pipeline supplies
    rather than a clock of its own, so a recording replayed offline aggregates
    over the same windows it would have live.
``dedupe_events``
    Drop an event that repeats the same occurrence -- same processor, type,
    track and target -- within :data:`DEDUPE_WINDOW_S`, before it reaches a
    sink or a rollup. This is for the same crossing being *reported* twice,
    by two processors watching one line or by a processor that saw it on two
    consecutive frames. It is deliberately a short window: someone walking in,
    out and back in inside a minute did that twice, and suppressing the second
    entry would leave the event stream disagreeing with the counter.
    Suppressing repeats over longer spans is each processor's ``cooldown_s``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

from pixelspot.analytics.base import Event, FrameContext, ProcessorOutput
from pixelspot.settings.schema import PixelSpotConfig

# How long an occurrence stays "already reported".
DEDUPE_WINDOW_S = 1.0


class Aggregator:
    """Latest levels, plus event counts over each configured window."""

    def __init__(
        self,
        windows: Mapping[str, float],
        emit_interval_s: float = 60.0,
        dedupe_events: bool = True,
    ):
        self.windows = dict(windows)
        self.emit_interval_s = emit_interval_s
        self.dedupe_events = dedupe_events
        self.retention_s = max(self.windows.values(), default=emit_interval_s)

        self.metrics: dict[str, dict[str, Any]] = {}
        self.total_events = 0
        self.suppressed_events = 0

        self._history: deque[tuple[float, str, str]] = deque()
        self._recent: dict[tuple[Any, ...], float] = {}
        self._interval_start: float | None = None
        self._now = 0.0
        self._frame_index = 0

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "Aggregator":
        aggregation = config.aggregation
        return cls(
            windows=aggregation.window_seconds(),
            emit_interval_s=aggregation.emit_interval_s,
            dedupe_events=aggregation.dedupe_events,
        )

    def update(
        self, context: FrameContext, outputs: Mapping[str, ProcessorOutput]
    ) -> list[Event]:
        """Absorb one frame of processor output and return the events kept."""
        self._frame_index = context.index
        self._now = context.timestamp
        if self._interval_start is None or self._interval_start > self._now:
            # First frame, or the system clock jumped backwards.
            self._interval_start = self._now
        kept: list[Event] = []

        for name, output in outputs.items():
            self.metrics[name] = output.metrics
            for event in output.events:
                if self.dedupe_events and self._is_duplicate(event):
                    self.suppressed_events += 1
                    continue

                self._history.append((event.timestamp, event.processor, event.type))
                self.total_events += 1
                kept.append(event)

        self._prune(self._now)
        return kept

    def due(self, now: float | None = None) -> bool:
        """True when a metrics snapshot should be emitted."""
        if self._interval_start is None:
            return False
        now = self._now if now is None else now
        return now - self._interval_start >= self.emit_interval_s

    def snapshot(self, now: float | None = None, **extra: Any) -> dict[str, Any]:
        """Build a snapshot and start a new interval."""
        now = self._now if now is None else now
        self._prune(now)

        snapshot = {
            "timestamp": now,
            "frame_index": self._frame_index,
            "metrics": dict(self.metrics),
            "windows": self.window_counts(now),
            "events_total": self.total_events,
            **extra,
        }

        self._interval_start = now
        return snapshot

    def window_counts(self, now: float | None = None) -> dict[str, dict[str, int]]:
        """Event counts per window, keyed ``processor.type``."""
        now = self._now if now is None else now
        counts: dict[str, dict[str, int]] = {}

        for name, span in self.windows.items():
            cutoff = now - span
            window: dict[str, int] = {}
            for timestamp, processor, event_type in self._history:
                if timestamp >= cutoff:
                    label = f"{processor}.{event_type}"
                    window[label] = window.get(label, 0) + 1
            counts[name] = window

        return counts

    def _is_duplicate(self, event: Event) -> bool:
        key = event.key()
        last_seen = self._recent.get(key)
        if last_seen is not None and event.timestamp - last_seen < DEDUPE_WINDOW_S:
            return True
        self._recent[key] = event.timestamp
        return False

    def _prune(self, now: float) -> None:
        cutoff = now - self.retention_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        dedupe_cutoff = now - DEDUPE_WINDOW_S
        for key in [
            key for key, seen in self._recent.items() if seen < dedupe_cutoff
        ]:
            del self._recent[key]
