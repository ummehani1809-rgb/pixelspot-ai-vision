"""Attention measurement.

Screen visibility says who *could* see the screen; this says who is actually
looking. A person attends when three things hold at once: they are in an
attention zone (if any are configured), they are positioned to see the
screen (inside its viewing cone and range, when a screen is named), and
their head is turned toward the camera within ``yaw_tolerance_deg`` -- the
screen deployment puts the panel at the camera, so facing one is facing the
other.

A glance is not attention. The gaze has to hold for ``min_gaze_s`` before it
counts, at which point one ``GAZE`` event fires for that look; looking away
and back starts a new look. Blinks of estimation noise shorter than
``GAZE_GAP_S`` do not break a gaze -- the pose model wobbles frame to frame
in a way human necks do not.

Requires head-pose enrichment; the config schema enforces that pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import (
    ResolvedGeometry,
    ResolvedScreen,
    ResolvedZone,
    estimate_distance_m,
)
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

# A gaze survives estimation dropouts shorter than this.
GAZE_GAP_S = 0.5


@dataclass
class _Gaze:
    started_at: float
    last_seen_at: float
    counted: bool = False


class AttentionProcessor(BaseProcessor):
    name = "attention"

    def __init__(
        self,
        zones: list[ResolvedZone],
        screen: ResolvedScreen | None,
        yaw_tolerance_deg: float = 35.0,
        min_gaze_s: float = 1.0,
        classes: tuple[str, ...] = ("person",),
    ):
        self.zones = zones
        self.screen = screen
        self.yaw_tolerance_deg = yaw_tolerance_deg
        self.min_gaze_s = min_gaze_s
        self.classes = classes
        self._gazes: dict[int, _Gaze] = {}
        self._total = 0

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "AttentionProcessor":
        settings = config.analytics.attention
        return cls(
            zones=geometry.select_zones(settings.zones),
            screen=geometry.screen(settings.screen) if settings.screen else None,
            yaw_tolerance_deg=settings.yaw_tolerance_deg,
            min_gaze_s=settings.min_gaze_s,
        )

    def _eligible(self, track: Track, frame_height: int) -> bool:
        if self.zones and not any(
            zone.contains(track.center) for zone in self.zones
        ):
            return False
        if self.screen is None:
            return True
        if not self.screen.in_view_cone(track.center):
            return False
        if self.screen.max_distance_m is not None:
            distance = estimate_distance_m(track.bbox[3] - track.bbox[1], frame_height)
            if distance is None or distance > self.screen.max_distance_m:
                return False
        return True

    def process(self, context: FrameContext) -> ProcessorOutput:
        output = ProcessorOutput()
        now = context.timestamp
        attending = 0

        for track in context.of_classes(self.classes):
            gazing = (
                track.head_yaw_deg is not None
                and abs(track.head_yaw_deg) <= self.yaw_tolerance_deg
                and self._eligible(track, context.height)
            )
            gaze = self._gazes.get(track.id)

            if gazing:
                if gaze is None:
                    gaze = self._gazes[track.id] = _Gaze(now, now)
                gaze.last_seen_at = now
                if now - gaze.started_at >= self.min_gaze_s:
                    attending += 1
                    if not gaze.counted:
                        gaze.counted = True
                        self._total += 1
                        output.events.append(
                            self._event(
                                context,
                                "GAZE",
                                track_id=track.id,
                                screen_id=self.screen.id if self.screen else None,
                            )
                        )
            elif gaze is not None and now - gaze.last_seen_at > GAZE_GAP_S:
                del self._gazes[track.id]

        output.metrics = {
            "attending": attending,
            "gazes_total": self._total,
        }
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [
            f"Attending: {metrics.get('attending', 0)} "
            f"(total {metrics.get('gazes_total', 0)})"
        ]
