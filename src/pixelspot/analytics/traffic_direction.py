"""Traffic direction.

Which way is the crowd moving? Each track keeps a short trail of recent
positions; the displacement across that trail gives a speed and a heading.
The trail is the smoothing: bounding boxes wobble frame to frame, and a
heading read off two consecutive frames would mostly report that wobble.

Tracks slower than ``min_speed_px_s`` are standing, not moving, and are
excluded -- otherwise a still crowd would report whichever way its jitter
happened to lean. Headings are bucketed into ``bins`` equal slices of the
circle; with the default 8 they get compass names, where "north" is up on
screen, whatever the camera actually faces.

All classes are counted, not just people: on a road camera the traffic is
the cars. Direction is a level, so this emits metrics only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry
from pixelspot.settings.schema import PixelSpotConfig

# Screen coordinates: +x is east, +y is SOUTH (y grows downwards).
_COMPASS_8 = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]


class TrafficDirectionProcessor(BaseProcessor):
    name = "traffic_direction"

    def __init__(
        self,
        smoothing_frames: int = 8,
        bins: int = 8,
        min_speed_px_s: float = 15.0,
        retention_s: float = 2.0,
    ):
        self.smoothing_frames = smoothing_frames
        self.bins = bins
        self.min_speed_px_s = min_speed_px_s
        self.retention_s = retention_s
        # track_id -> trail of (timestamp, x, y)
        self._trails: dict[int, deque[tuple[float, float, float]]] = {}

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "TrafficDirectionProcessor":
        settings = config.analytics.traffic_direction
        return cls(
            smoothing_frames=settings.smoothing_frames,
            bins=settings.bins,
            min_speed_px_s=settings.min_speed_px_s,
            retention_s=config.perception.tracker.max_age_s,
        )

    def _label(self, bin_index: int) -> str:
        if self.bins == 8:
            return _COMPASS_8[bin_index]
        return f"{int(bin_index * 360 / self.bins)}deg"

    def process(self, context: FrameContext) -> ProcessorOutput:
        for track in context.tracks:
            trail = self._trails.get(track.id)
            if trail is None:
                trail = self._trails[track.id] = deque(maxlen=self.smoothing_frames)
            x, y = track.center
            trail.append((context.timestamp, x, y))

        # Forget tracks the tracker itself has given up on.
        for track_id in [
            track_id
            for track_id, trail in self._trails.items()
            if context.timestamp - trail[-1][0] > self.retention_s
        ]:
            del self._trails[track_id]

        per_direction: dict[str, int] = {}
        moving = 0
        stationary = 0
        bin_width = 360.0 / self.bins

        seen_ids = {track.id for track in context.tracks}
        for track_id in seen_ids:
            trail = self._trails[track_id]
            if len(trail) < 2:
                continue
            t0, x0, y0 = trail[0]
            t1, x1, y1 = trail[-1]
            elapsed = t1 - t0
            if elapsed <= 0:
                continue

            speed = math.hypot(x1 - x0, y1 - y0) / elapsed
            if speed < self.min_speed_px_s:
                stationary += 1
                continue

            moving += 1
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360.0
            bin_index = int(((angle + bin_width / 2) % 360.0) / bin_width)
            label = self._label(bin_index)
            per_direction[label] = per_direction.get(label, 0) + 1

        dominant = max(per_direction, key=per_direction.get) if per_direction else None
        return ProcessorOutput(
            metrics={
                "moving": moving,
                "stationary": stationary,
                "dominant": dominant,
                "per_direction": per_direction,
            }
        )

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        dominant = metrics.get("dominant")
        if dominant is None:
            return ["Flow: still"]
        return [f"Flow: {dominant} ({metrics.get('moving', 0)} moving)"]
