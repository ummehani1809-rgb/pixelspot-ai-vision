"""Presence heatmap.

Accumulates where people stand into a coarse grid over the frame. Each frame,
every tracked person adds one unit to the grid cell under their feet and the
whole grid is multiplied by ``decay``, so the map is a leaky integral: busy
spots glow, abandoned spots cool off on their own, and nothing needs a reset.

The grid is deliberately coarse (default 64x36). A per-pixel map would mostly
record bounding-box jitter; a cell a person wide is the resolution the
question "where do people linger" is actually asked at.

Only the hotspot summary goes into the metrics -- publishing thousands of
floats per frame through every sink would drown the numbers that matter. The
full grid stays on the processor (``grid`` attribute) for the renderer.

``weight_by`` modes other than ``presence`` need signals from later phases
(attention needs head pose), so they warn at startup and fall back to
presence until those land.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import PixelSpotConfig

log = get_logger(__name__)


class HeatmapProcessor(BaseProcessor):
    name = "heatmap"

    def __init__(
        self,
        grid_size: tuple[int, int] = (64, 36),
        decay: float = 0.98,
        classes: tuple[str, ...] = ("person",),
        top_cells: int = 5,
    ):
        self.columns, self.rows = grid_size
        self.decay = decay
        self.classes = classes
        self.top_cells = top_cells
        self.grid = np.zeros((self.rows, self.columns), dtype=np.float32)

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "HeatmapProcessor":
        settings = config.analytics.heatmap
        if settings.weight_by != "presence":
            log.warning(
                "heatmap.weight_by=%r is not available yet; using 'presence'",
                settings.weight_by,
            )
        return cls(grid_size=settings.grid, decay=settings.decay)

    def process(self, context: FrameContext) -> ProcessorOutput:
        self.grid *= self.decay

        for track in context.of_classes(self.classes):
            x, y = track.center
            column = min(int(x / context.width * self.columns), self.columns - 1)
            row = min(int(y / context.height * self.rows), self.rows - 1)
            self.grid[row, column] += 1.0

        flat = self.grid.ravel()
        order = np.argsort(flat)[::-1][: self.top_cells]
        hot_cells = [
            {
                "x": int(index % self.columns),
                "y": int(index // self.columns),
                "weight": round(float(flat[index]), 2),
            }
            for index in order
            if flat[index] > 0
        ]

        return ProcessorOutput(
            metrics={
                "max_weight": round(float(flat.max()), 2) if flat.size else 0.0,
                "hot_cells": hot_cells,
            }
        )

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        cells = metrics.get("hot_cells") or []
        if not cells:
            return ["Heatmap: quiet"]
        hottest = cells[0]
        return [f"Hotspot: cell ({hottest['x']},{hottest['y']}) w={hottest['weight']}"]
