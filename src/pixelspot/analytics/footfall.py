"""Footfall counting.

Counts people across the lines named in ``analytics.footfall.lines``, in both
directions. Crossing a line in its configured ``positive`` direction is an
entry; the other way is an exit.

Nothing about the counting rule is written here. The line geometry comes from
``geometry.lines``, the anti-jitter rules from ``analytics.footfall``, and the
crossing arithmetic from :mod:`pixelspot.analytics.crossing`. This class only
decides what the two directions *mean* -- entries, exits, occupancy, peak --
which is the part that is specific to footfall.
"""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.analytics.crossing import LineCrossingCounter
from pixelspot.geometry import ResolvedGeometry
from pixelspot.settings.schema import PixelSpotConfig


class FootfallProcessor(BaseProcessor):
    name = "footfall"

    def __init__(
        self,
        counters: list[LineCrossingCounter],
        classes: tuple[str, ...] = ("person",),
    ):
        self.counters = counters
        self.classes = classes
        self.peak_count = 0

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "FootfallProcessor":
        settings = config.analytics.footfall
        counters = [
            LineCrossingCounter(
                line=geometry.line(line_id),
                hysteresis_px=settings.hysteresis_px,
                min_track_age_frames=settings.min_track_age_frames,
                cooldown_s=settings.cooldown_s,
                retention_s=config.perception.tracker.max_age_s,
            )
            for line_id in settings.lines
        ]
        return cls(counters=counters, classes=tuple(settings.classes))

    def process(self, context: FrameContext) -> ProcessorOutput:
        people = context.of_classes(self.classes)
        output = ProcessorOutput()

        per_line: dict[str, dict[str, int]] = {}
        for counter in self.counters:
            for crossing in counter.update(people, context.timestamp):
                event_type = "ENTER" if crossing.direction == "positive" else "EXIT"
                output.events.append(
                    self._event(
                        context,
                        event_type,
                        track_id=crossing.track_id,
                        line_id=crossing.line_id,
                    )
                )
            per_line[counter.line.id] = {
                "entered": counter.positive,
                "exited": counter.negative,
            }

        entered = sum(counts["entered"] for counts in per_line.values())
        exited = sum(counts["exited"] for counts in per_line.values())
        occupancy = max(0, entered - exited)
        self.peak_count = max(self.peak_count, occupancy)

        output.metrics = {
            "people_entered": entered,
            "people_exited": exited,
            "current_occupancy": occupancy,
            "peak_count": self.peak_count,
            "people_present": len(people),
            "per_line": per_line,
        }
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [
            f"Entered: {metrics.get('people_entered', 0)}",
            f"Exited: {metrics.get('people_exited', 0)}",
            f"Occupancy: {metrics.get('current_occupancy', 0)}",
            f"Peak: {metrics.get('peak_count', 0)}",
        ]
