"""Dwell time.

Measures how long each tracked person stays inside each zone named in
``analytics.dwell.zones``. ``viewing_zone`` answers "how many are here right
now"; this answers "how long do they stay", which is the number a retailer
actually buys: a hundred people glancing for a second and five people reading
the window for a minute look identical in an occupancy count.

A visit opens the first frame a track is seen inside a zone and closes when
the track is seen outside it, or when it has been missing for longer than the
tracker keeps lost tracks alive (``perception.tracker.max_age_s``) -- gone
that long means the person left, not that they were briefly occluded. Visits
shorter than ``min_dwell_s`` are walk-throughs and emit nothing. Durations are
capped at ``max_dwell_s`` because hours-long visits are almost always a track
stuck on a mannequin or a poster, not a person.

One closed visit is one ``DWELL`` event. A person who leaves and comes back
is two visits on purpose: averaging them together would hide the difference
between one long look and two short ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import PixelSpotConfig


@dataclass
class _Visit:
    """One track's open stopwatch in one zone."""

    started_at: float
    last_seen_at: float


class DwellProcessor(BaseProcessor):
    name = "dwell"

    def __init__(
        self,
        zones: list[ResolvedZone],
        min_dwell_s: float = 2.0,
        max_dwell_s: float = 3600.0,
        retention_s: float = 2.0,
        classes: tuple[str, ...] = ("person",),
    ):
        self.zones = zones
        self.min_dwell_s = min_dwell_s
        self.max_dwell_s = max_dwell_s
        self.retention_s = retention_s
        self.classes = classes
        # (zone_id, track_id) -> open visit
        self._visits: dict[tuple[str, int], _Visit] = {}
        self._completed = 0
        self._total_dwell_s = 0.0

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "DwellProcessor":
        settings = config.analytics.dwell
        return cls(
            zones=geometry.select_zones(settings.zones),
            min_dwell_s=settings.min_dwell_s,
            max_dwell_s=settings.max_dwell_s,
            retention_s=config.perception.tracker.max_age_s,
        )

    def process(self, context: FrameContext) -> ProcessorOutput:
        people = context.of_classes(self.classes)
        seen_ids = {track.id for track in people}
        output = ProcessorOutput()

        per_zone_active: dict[str, int] = {}
        for zone in self.zones:
            inside = {t.id for t in people if zone.contains(t.center)}
            per_zone_active[zone.id] = 0

            for track_id in inside:
                visit = self._visits.get((zone.id, track_id))
                if visit is None:
                    self._visits[(zone.id, track_id)] = _Visit(
                        started_at=context.timestamp,
                        last_seen_at=context.timestamp,
                    )
                else:
                    visit.last_seen_at = context.timestamp
                    if context.timestamp - visit.started_at >= self.min_dwell_s:
                        per_zone_active[zone.id] += 1

            for (zone_id, track_id), visit in list(self._visits.items()):
                if zone_id != zone.id or track_id in inside:
                    continue
                # Seen outside the zone -> left. Not seen anywhere for longer
                # than the tracker would keep them -> also left; anything
                # shorter may just be an occlusion, so the visit stays open.
                walked_out = track_id in seen_ids
                expired = context.timestamp - visit.last_seen_at > self.retention_s
                if walked_out or expired:
                    del self._visits[(zone_id, track_id)]
                    event = self._close(context, zone_id, track_id, visit)
                    if event is not None:
                        output.events.append(event)

        output.metrics = {
            "currently_dwelling": sum(per_zone_active.values()),
            "completed_visits": self._completed,
            "avg_dwell_s": round(
                self._total_dwell_s / self._completed if self._completed else 0.0, 1
            ),
            "per_zone_dwelling": per_zone_active,
        }
        return output

    def _close(self, context, zone_id: str, track_id: int, visit: _Visit):
        duration = min(visit.last_seen_at - visit.started_at, self.max_dwell_s)
        if duration < self.min_dwell_s:
            return None
        self._completed += 1
        self._total_dwell_s += duration
        return self._event(
            context,
            "DWELL",
            track_id=track_id,
            zone_id=zone_id,
            dwell_s=round(duration, 1),
        )

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [
            f"Dwelling: {metrics.get('currently_dwelling', 0)}",
            f"Avg dwell: {metrics.get('avg_dwell_s', 0.0)}s",
        ]
