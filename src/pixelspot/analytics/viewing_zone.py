"""Viewing zone occupancy.

Reports how many tracked people stand inside each zone named in
``analytics.viewing_zone.zones`` right now. The zones themselves are arbitrary
polygons from ``geometry.zones``, not the axis-aligned rectangle this started
life as: a shop window catchment or a screen viewing cone is rarely a rectangle,
and forcing it to be one either counts the pavement or misses the corner.

Occupancy is a level, not an event, so this processor emits metrics only.
Entering and leaving over time is the ``dwell`` and ``audience_flow`` job.
"""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import PixelSpotConfig


class ViewingZoneProcessor(BaseProcessor):
    name = "viewing_zone"

    def __init__(
        self, zones: list[ResolvedZone], classes: tuple[str, ...] = ("person",)
    ):
        self.zones = zones
        self.classes = classes

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "ViewingZoneProcessor":
        settings = config.analytics.viewing_zone
        return cls(
            zones=geometry.select_zones(settings.zones),
            classes=tuple(settings.classes),
        )

    def process(self, context: FrameContext) -> ProcessorOutput:
        people = context.of_classes(self.classes)

        per_zone: dict[str, int] = {}
        inside_ids: set[int] = set()

        for zone in self.zones:
            count = 0
            for track in people:
                if zone.contains(track.center):
                    count += 1
                    inside_ids.add(track.id)
            per_zone[zone.id] = count

        return ProcessorOutput(
            metrics={
                # Sum over zones double counts anyone standing in two
                # overlapping zones; the unique count is the honest headline.
                "people_in_viewing_zone": len(inside_ids),
                "per_zone": per_zone,
            }
        )

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [f"Viewing zone: {metrics.get('people_in_viewing_zone', 0)}"]
