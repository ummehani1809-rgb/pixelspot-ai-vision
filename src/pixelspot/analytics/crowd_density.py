"""Crowd density.

Turns "N people in a zone" into people per square metre, which is the number
that means the same thing in a kiosk corner and a mall atrium. The zone's
real-world floor area comes from ``analytics.crowd_density.zone_areas_m2``;
the camera cannot know it, someone with a tape measure does.

Density is bucketed against the ``levels`` thresholds -- below ``low`` is
"empty", then "low", "medium", and "high" -- and a zone changing level emits
one event named after the level it entered (``DENSITY_HIGH`` and friends).
The level is in the event *type* on purpose: deduplication keys on type and
zone, so distinct level changes survive while a flapping zone repeating the
same change inside the dedupe window is absorbed.

The config schema guarantees every configured zone has an area, so there is
no missing-area case to handle here.
"""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import DensityLevels, PixelSpotConfig

LEVEL_ORDER = ["empty", "low", "medium", "high"]


class CrowdDensityProcessor(BaseProcessor):
    name = "crowd_density"

    def __init__(
        self,
        zones: list[ResolvedZone],
        zone_areas_m2: dict[str, float],
        levels: DensityLevels,
        classes: tuple[str, ...] = ("person",),
    ):
        self.zones = zones
        self.zone_areas_m2 = zone_areas_m2
        self.levels = levels
        self.classes = classes
        self._last_level: dict[str, str] = {}

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "CrowdDensityProcessor":
        settings = config.analytics.crowd_density
        return cls(
            zones=geometry.select_zones(settings.zones),
            zone_areas_m2=dict(settings.zone_areas_m2),
            levels=settings.levels,
        )

    def _level_for(self, density: float) -> str:
        if density >= self.levels.high:
            return "high"
        if density >= self.levels.medium:
            return "medium"
        if density >= self.levels.low:
            return "low"
        return "empty"

    def process(self, context: FrameContext) -> ProcessorOutput:
        people = context.of_classes(self.classes)
        output = ProcessorOutput()

        per_zone: dict[str, dict[str, Any]] = {}
        worst = "empty"

        for zone in self.zones:
            count = sum(1 for track in people if zone.contains(track.center))
            density = round(count / self.zone_areas_m2[zone.id], 3)
            level = self._level_for(density)
            per_zone[zone.id] = {"count": count, "density": density, "level": level}

            if LEVEL_ORDER.index(level) > LEVEL_ORDER.index(worst):
                worst = level

            if self._last_level.get(zone.id) != level:
                self._last_level[zone.id] = level
                output.events.append(
                    self._event(
                        context,
                        f"DENSITY_{level.upper()}",
                        zone_id=zone.id,
                        density=density,
                        count=count,
                    )
                )

        output.metrics = {"level": worst, "per_zone": per_zone}
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [f"Density: {metrics.get('level', 'empty')}"]
