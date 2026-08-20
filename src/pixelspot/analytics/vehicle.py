"""Vehicle counting.

Which classes count as a vehicle is a deployment decision, not a fact about the
world: a bus lane camera wants buses and nothing else, a car park wants cars and
motorcycles. So the set comes from ``analytics.vehicles.classes`` rather than a
constant in this file, and the schema already refuses a class the detector was
never told to look for.

Two optional narrowings, both off by default:

``zones``
    Only count vehicles whose centre is inside one of these zones. Without it,
    every vehicle anywhere in frame counts.
``lines``
    Also count vehicles crossing these lines, giving a flow rate through a
    gate rather than a snapshot of how many are visible.
"""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.analytics.crossing import LineCrossingCounter
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import PixelSpotConfig


class VehicleProcessor(BaseProcessor):
    name = "vehicles"

    def __init__(
        self,
        classes: tuple[str, ...],
        classify: bool = True,
        zones: list[ResolvedZone] | None = None,
        counters: list[LineCrossingCounter] | None = None,
    ):
        self.classes = classes
        self.classify = classify
        self.zones = zones or []
        self.counters = counters or []

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "VehicleProcessor":
        settings = config.analytics.vehicles
        counters = [
            LineCrossingCounter(
                line=geometry.line(line_id),
                retention_s=config.perception.tracker.max_age_s,
            )
            for line_id in settings.lines
        ]
        return cls(
            classes=tuple(settings.classes),
            classify=settings.classify,
            zones=geometry.select_zones(settings.zones),
            counters=counters,
        )

    def process(self, context: FrameContext) -> ProcessorOutput:
        vehicles = context.of_classes(self.classes)
        if self.zones:
            vehicles = [
                track
                for track in vehicles
                if any(zone.contains(track.center) for zone in self.zones)
            ]

        output = ProcessorOutput()
        metrics: dict[str, Any] = {"total_vehicles": len(vehicles)}

        if self.classify:
            counts = {label: 0 for label in self.classes}
            for track in vehicles:
                counts[track.label.lower()] = counts.get(track.label.lower(), 0) + 1
            metrics["by_class"] = counts

        if self.counters:
            per_line: dict[str, dict[str, int]] = {}
            for counter in self.counters:
                for crossing in counter.update(vehicles, context.timestamp):
                    output.events.append(
                        self._event(
                            context,
                            "VEHICLE_CROSS",
                            track_id=crossing.track_id,
                            line_id=crossing.line_id,
                            label=crossing.label,
                            direction=crossing.direction,
                        )
                    )
                per_line[counter.line.id] = {
                    "positive": counter.positive,
                    "negative": counter.negative,
                }
            metrics["per_line"] = per_line

        output.metrics = metrics
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [f"Vehicles: {metrics.get('total_vehicles', 0)}"]
