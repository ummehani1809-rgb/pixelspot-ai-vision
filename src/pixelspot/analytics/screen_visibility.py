"""Screen visibility.

How many people are positioned to see the screen right now -- inside its
viewing cone and within its range -- regardless of where they are looking.
This is the denominator attention is judged against: five attending out of
six who could see the screen is a very different creative than five out of
five hundred.

Distance uses the monocular person-height estimate from
:func:`pixelspot.geometry.estimate_distance_m`; the range check prefers this
processor's own ``max_distance_m`` and falls back to the screen's.

A level, not an occurrence: metrics only.
"""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedScreen, estimate_distance_m
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import PixelSpotConfig

log = get_logger(__name__)


class ScreenVisibilityProcessor(BaseProcessor):
    name = "screen_visibility"

    def __init__(
        self,
        screen: ResolvedScreen | None,
        max_distance_m: float | None = None,
        classes: tuple[str, ...] = ("person",),
    ):
        self.screen = screen
        self.max_distance_m = max_distance_m or (
            screen.max_distance_m if screen else None
        )
        self.classes = classes
        if screen is None:
            log.warning(
                "screen_visibility is enabled but analytics.screen_visibility"
                ".screen names no screen; it will always report zero viewers"
            )

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "ScreenVisibilityProcessor":
        settings = config.analytics.screen_visibility
        return cls(
            screen=geometry.screen(settings.screen) if settings.screen else None,
            max_distance_m=settings.max_distance_m,
        )

    def process(self, context: FrameContext) -> ProcessorOutput:
        if self.screen is None:
            return ProcessorOutput(metrics={"viewers": 0, "nearest_m": None})

        viewers = 0
        nearest: float | None = None

        for track in context.of_classes(self.classes):
            if not self.screen.in_view_cone(track.center):
                continue
            distance = estimate_distance_m(
                track.bbox[3] - track.bbox[1], context.height
            )
            if distance is None:
                continue
            if self.max_distance_m is not None and distance > self.max_distance_m:
                continue
            viewers += 1
            nearest = distance if nearest is None else min(nearest, distance)

        return ProcessorOutput(
            metrics={
                "viewers": viewers,
                "nearest_m": round(nearest, 1) if nearest is not None else None,
            }
        )

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [f"Can see screen: {metrics.get('viewers', 0)}"]
