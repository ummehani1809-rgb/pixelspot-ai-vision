"""Audience flow.

Where do people go next? For each track this remembers the zone it is
settled in; when the track settles in a *different* configured zone, that is
one transition, and the from->to pair is counted into a flow matrix.

"Settled" is the important word. Two adjacent zones share an edge, and a
person standing on it flickers between them every frame; each candidate zone
therefore has to hold for ``min_transition_s`` before it becomes the settled
one. Entering from outside any zone starts a journey rather than counting as
a transition -- flow is between zones, and "came in from nowhere" is
footfall's story.

One committed zone-to-zone move is one ``TRANSITION`` event. The event's
``zone_id`` is the destination, so deduplication treats moves to different
places as different occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import PixelSpotConfig


@dataclass
class _Journey:
    """One track's position in the zone graph."""

    settled: str | None
    candidate: str | None
    candidate_since: float
    last_seen: float


class AudienceFlowProcessor(BaseProcessor):
    name = "audience_flow"

    def __init__(
        self,
        zones: list[ResolvedZone],
        min_transition_s: float = 0.5,
        retention_s: float = 2.0,
        classes: tuple[str, ...] = ("person",),
    ):
        self.zones = zones
        self.min_transition_s = min_transition_s
        self.retention_s = retention_s
        self.classes = classes
        self._journeys: dict[int, _Journey] = {}
        self._matrix: dict[str, int] = {}

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "AudienceFlowProcessor":
        settings = config.analytics.audience_flow
        return cls(
            zones=geometry.select_zones(settings.zones),
            min_transition_s=settings.min_transition_s,
            retention_s=config.perception.tracker.max_age_s,
        )

    def _zone_of(self, point: tuple[float, float]) -> str | None:
        for zone in self.zones:
            if zone.contains(point):
                return zone.id
        return None

    def process(self, context: FrameContext) -> ProcessorOutput:
        output = ProcessorOutput()
        now = context.timestamp

        for track in context.of_classes(self.classes):
            zone_now = self._zone_of(track.center)
            journey = self._journeys.get(track.id)
            if journey is None:
                journey = self._journeys[track.id] = _Journey(
                    settled=zone_now, candidate=None, candidate_since=now, last_seen=now
                )
                continue
            journey.last_seen = now

            if zone_now == journey.settled:
                journey.candidate = None
                continue

            if zone_now != journey.candidate:
                journey.candidate = zone_now
                journey.candidate_since = now
                if self.min_transition_s > 0:
                    continue

            if now - journey.candidate_since >= self.min_transition_s:
                if journey.settled is not None and zone_now is not None:
                    key = f"{journey.settled}->{zone_now}"
                    self._matrix[key] = self._matrix.get(key, 0) + 1
                    output.events.append(
                        self._event(
                            context,
                            "TRANSITION",
                            track_id=track.id,
                            from_zone=journey.settled,
                            zone_id=zone_now,
                        )
                    )
                journey.settled = zone_now
                journey.candidate = None

        for track_id in [
            track_id
            for track_id, journey in self._journeys.items()
            if now - journey.last_seen > self.retention_s
        ]:
            del self._journeys[track_id]

        output.metrics = {
            "transitions_total": sum(self._matrix.values()),
            "matrix": dict(self._matrix),
        }
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [f"Flows: {metrics.get('transitions_total', 0)}"]
