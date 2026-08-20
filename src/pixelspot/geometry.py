"""Geometry resolution.

Config geometry is authored in normalized coordinates (0.0-1.0) so one config
file works against a 720p webcam and a 4K RTSP stream without edits. Nothing
downstream should ever deal with that: processors and rendering want pixels.
This module is the one place the conversion happens, right after the first
frame arrives and the capture resolution is finally known.

The other job here is turning ``positive: down`` into arithmetic. A line is
stored as a signed distance function whose *sign* is oriented so that positive
always means "the side the configured positive direction points at". A crossing
is then a sign change, identically for every line whatever its angle, and every
processor that counts crossings shares that one definition.

Screens are deliberately not resolved: no implemented processor consumes them
yet, and resolving a facing vector before anything reads it would just be
guesswork frozen into code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import GeometryConfig

Point = tuple[float, float]

# Below this a "line" is a point: nothing can be on one side of it.
_MIN_LENGTH_PX = 1.0

# Screen-space direction each `positive` value points at. y grows downwards.
_POSITIVE_AXES: dict[str, Point] = {
    "down": (0.0, 1.0),
    "up": (0.0, -1.0),
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
}


@dataclass
class ResolvedZone:
    """A polygon in pixel coordinates."""

    id: str
    points: tuple[Point, ...]
    tags: tuple[str, ...] = ()

    def contains(self, point: Point) -> bool:
        """Ray casting point-in-polygon test.

        Handles the concave and self-touching polygons an operator will
        inevitably draw, which a bounding-box test would not.
        """
        x, y = point
        inside = False
        points = self.points
        previous = len(points) - 1

        for current in range(len(points)):
            xi, yi = points[current]
            xj, yj = points[previous]
            if (yi > y) != (yj > y):
                crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < crossing_x:
                    inside = not inside
            previous = current

        return inside

    def polygon(self) -> list[tuple[int, int]]:
        """Integer vertices, for drawing."""
        return [(int(round(x)), int(round(y))) for x, y in self.points]


@dataclass
class ResolvedLine:
    """A counting line in pixel coordinates with an oriented normal."""

    id: str
    p1: Point
    p2: Point
    positive: str = "down"
    classes: tuple[str, ...] = ("person",)

    length: float = field(init=False, repr=False)
    normal: Point = field(init=False, repr=False)
    _sign: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        self.length = math.hypot(dx, dy)
        if self.length < _MIN_LENGTH_PX:
            raise ConfigError(
                f"line {self.id!r}: p1 and p2 are {self.length:.3f}px apart, "
                "which is not a line anything could cross"
            )

        # Unit normal. Moving along it increases the raw signed distance.
        self.normal = (dy / self.length, -dx / self.length)

        axis = _POSITIVE_AXES[self.positive]
        orientation = self.normal[0] * axis[0] + self.normal[1] * axis[1]
        if abs(orientation) < 1e-9:
            raise ConfigError(
                f"line {self.id!r}: positive direction {self.positive!r} runs "
                "parallel to the line, so no movement could ever cross it"
            )
        self._sign = 1.0 if orientation > 0 else -1.0

    def oriented_distance(self, point: Point) -> float:
        """Signed distance in pixels, positive on the ``positive`` side.

        Sign flips from negative to positive exactly when something moves
        across the line in the configured positive direction.
        """
        cross = (point[0] - self.p1[0]) * (self.p2[1] - self.p1[1]) - (
            point[1] - self.p1[1]
        ) * (self.p2[0] - self.p1[0])
        return self._sign * cross / self.length

    def span_position(self, point: Point) -> float:
        """Where *point* projects onto the segment: 0 at ``p1``, 1 at ``p2``.

        Values outside ``[0, 1]`` are beside the segment rather than across it,
        which is what stops a line drawn over a doorway from also counting
        people walking past the far end of the room.
        """
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return ((point[0] - self.p1[0]) * dx + (point[1] - self.p1[1]) * dy) / (
            self.length * self.length
        )

    def midpoint(self) -> Point:
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)

    def positive_arrow(self, length_px: float = 40.0) -> tuple[Point, Point]:
        """Start and end of an arrow showing which way counts as positive."""
        start = self.midpoint()
        end = (
            start[0] + self.normal[0] * self._sign * length_px,
            start[1] + self.normal[1] * self._sign * length_px,
        )
        return start, end


@dataclass
class ResolvedGeometry:
    """Every zone and line for this capture resolution, indexed by id."""

    width: int
    height: int
    zones: dict[str, ResolvedZone]
    lines: dict[str, ResolvedLine]

    @classmethod
    def resolve(
        cls, config: GeometryConfig, width: int, height: int
    ) -> "ResolvedGeometry":
        """Scale *config* to a ``width`` x ``height`` frame."""
        if width <= 0 or height <= 0:
            raise ConfigError(f"cannot resolve geometry for a {width}x{height} frame")

        normalized = config.coordinate_space == "normalized"

        def to_pixels(point: Point) -> Point:
            if not normalized:
                return (float(point[0]), float(point[1]))
            return (point[0] * width, point[1] * height)

        zones = {
            zone.id: ResolvedZone(
                id=zone.id,
                points=tuple(to_pixels(point) for point in zone.points),
                tags=tuple(zone.tags),
            )
            for zone in config.zones
        }
        lines = {
            line.id: ResolvedLine(
                id=line.id,
                p1=to_pixels(line.p1),
                p2=to_pixels(line.p2),
                positive=line.positive,
                classes=tuple(line.classes),
            )
            for line in config.lines
        }
        return cls(width=width, height=height, zones=zones, lines=lines)

    def zone(self, zone_id: str) -> ResolvedZone:
        try:
            return self.zones[zone_id]
        except KeyError:
            raise ConfigError(
                f"unknown zone {zone_id!r} (defined: "
                f"{', '.join(sorted(self.zones)) or 'none'})"
            ) from None

    def line(self, line_id: str) -> ResolvedLine:
        try:
            return self.lines[line_id]
        except KeyError:
            raise ConfigError(
                f"unknown line {line_id!r} (defined: "
                f"{', '.join(sorted(self.lines)) or 'none'})"
            ) from None

    def select_zones(self, zone_ids: list[str]) -> list[ResolvedZone]:
        return [self.zone(zone_id) for zone_id in zone_ids]

    def select_lines(self, line_ids: list[str]) -> list[ResolvedLine]:
        return [self.line(line_id) for line_id in line_ids]
