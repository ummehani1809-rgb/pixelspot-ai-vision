"""Queue detection.

A queue is people who have been *waiting* -- in the zone for at least
``min_dwell_s`` -- and are standing *together*: chained within
``cluster_radius_px`` of the next person. Both conditions matter. Dwell alone
flags browsers scattered around a shop; proximity alone flags a tour group
walking through. ``min_people`` of both at once is a line at a till.

Clustering is single-linkage: each person needs to be near *someone* in the
cluster, not near everyone, because a queue is long and thin -- its two ends
are nowhere near each other.

The zone transitioning to having a queue emits ``QUEUE_FORMED``; back below
``min_people`` emits ``QUEUE_CLEARED``. In between, the length is a metric.
Waiting time survives a brief occlusion the same way dwell does: only being
seen outside the zone, or vanishing longer than the tracker keeps lost
tracks, resets the clock.
"""

from __future__ import annotations

import math
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import PixelSpotConfig


def _largest_cluster(points: list[tuple[float, float]], radius: float) -> int:
    """Size of the biggest single-linkage cluster."""
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if math.dist(points[i], points[j]) <= radius:
                parent[find(i)] = find(j)

    sizes: dict[int, int] = {}
    for index in range(len(points)):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + 1
    return max(sizes.values(), default=0)


class QueueProcessor(BaseProcessor):
    name = "queue"

    def __init__(
        self,
        zones: list[ResolvedZone],
        min_people: int = 3,
        min_dwell_s: float = 10.0,
        cluster_radius_px: int = 120,
        retention_s: float = 2.0,
        classes: tuple[str, ...] = ("person",),
    ):
        self.zones = zones
        self.min_people = min_people
        self.min_dwell_s = min_dwell_s
        self.cluster_radius_px = cluster_radius_px
        self.retention_s = retention_s
        self.classes = classes
        # (zone_id, track_id) -> (waiting_since, last_seen)
        self._waiting: dict[tuple[str, int], tuple[float, float]] = {}
        self._queued: set[str] = set()

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "QueueProcessor":
        settings = config.analytics.queue
        return cls(
            zones=geometry.select_zones(settings.zones),
            min_people=settings.min_people,
            min_dwell_s=settings.min_dwell_s,
            cluster_radius_px=settings.cluster_radius_px,
            retention_s=config.perception.tracker.max_age_s,
        )

    def process(self, context: FrameContext) -> ProcessorOutput:
        people = context.of_classes(self.classes)
        seen_ids = {track.id for track in people}
        output = ProcessorOutput()
        now = context.timestamp

        per_zone: dict[str, int] = {}
        for zone in self.zones:
            inside = {t.id: t for t in people if zone.contains(t.center)}

            for track_id in inside:
                since, _ = self._waiting.get((zone.id, track_id), (now, now))
                self._waiting[(zone.id, track_id)] = (since, now)

            for (zone_id, track_id), (since, last_seen) in list(self._waiting.items()):
                if zone_id != zone.id or track_id in inside:
                    continue
                if track_id in seen_ids or now - last_seen > self.retention_s:
                    del self._waiting[(zone_id, track_id)]

            waiters = [
                inside[track_id].center
                for track_id in inside
                if now - self._waiting[(zone.id, track_id)][0] >= self.min_dwell_s
            ]
            length = _largest_cluster(waiters, float(self.cluster_radius_px))
            queued = length >= self.min_people
            per_zone[zone.id] = length if queued else 0

            if queued and zone.id not in self._queued:
                self._queued.add(zone.id)
                output.events.append(
                    self._event(context, "QUEUE_FORMED", zone_id=zone.id, length=length)
                )
            elif not queued and zone.id in self._queued:
                self._queued.discard(zone.id)
                output.events.append(
                    self._event(context, "QUEUE_CLEARED", zone_id=zone.id)
                )

        output.metrics = {
            "queues": len(self._queued),
            "longest_queue": max(per_zone.values(), default=0),
            "per_zone": per_zone,
        }
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        if not metrics.get("queues"):
            return ["Queue: none"]
        return [f"Queue: {metrics.get('longest_queue', 0)} waiting"]
