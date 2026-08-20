"""Line crossing detection.

Footfall and vehicle counting both need the same question answered -- did this
track cross that line, and which way -- so it is answered once, here.

Counting the raw sign change of the distance to a line is what produces the
classic doubled count: a track whose centre jitters on the line itself flips
sign repeatedly, and every flip is another person through the door. Three
config settings exist to stop that, and all three are enforced here:

``hysteresis_px``
    A track must be past the line by this much before that side is believed.
    Between the two thresholds the previous side stands.
``min_track_age_frames``
    Brand new tracks are often detector noise, or a fragment of a track that
    already exists. Their crossings update state but are not counted.
``cooldown_s``
    After counting, a track cannot count again for this long, which bounds the
    damage when a tracker recycles an id.
"""

from __future__ import annotations

from dataclasses import dataclass

from pixelspot.geometry import ResolvedLine
from pixelspot.tracking.tracker import Track

# A crossing only counts between the segment endpoints, with a small tolerance
# so a track that clips a corner still registers.
_SPAN_TOLERANCE = 0.02


@dataclass
class _TrackState:
    side: int
    last_seen: float
    last_counted: float = 0.0


@dataclass
class Crossing:
    """One counted traversal of one line."""

    line_id: str
    track_id: int
    label: str
    direction: str  # "positive" or "negative"


class LineCrossingCounter:
    """Counts traversals of a single line, one call per frame."""

    def __init__(
        self,
        line: ResolvedLine,
        hysteresis_px: float = 0.0,
        min_track_age_frames: int = 0,
        cooldown_s: float = 0.0,
        retention_s: float = 2.0,
    ):
        self.line = line
        self.hysteresis_px = hysteresis_px
        self.min_track_age_frames = min_track_age_frames
        self.cooldown_s = cooldown_s
        self.retention_s = retention_s

        self.positive = 0
        self.negative = 0
        self._states: dict[int, _TrackState] = {}

    def update(self, tracks: list[Track], now: float) -> list[Crossing]:
        """Advance every track state and return the crossings just counted."""
        crossings: list[Crossing] = []

        for track in tracks:
            point = track.center
            distance = self.line.oriented_distance(point)

            if distance > self.hysteresis_px:
                side = 1
            elif distance < -self.hysteresis_px:
                side = -1
            else:
                side = 0  # inside the dead band, undecided

            state = self._states.get(track.id)
            if state is None:
                self._states[track.id] = _TrackState(side=side, last_seen=now)
                continue

            previous = state.side
            state.last_seen = now
            if side == 0 or side == previous:
                continue

            state.side = side
            if previous == 0:
                # First time this track committed to a side. It appeared there,
                # it did not travel there.
                continue
            if not self._within_span(point):
                continue
            if track.hits < self.min_track_age_frames:
                continue
            if now - state.last_counted < self.cooldown_s:
                continue

            state.last_counted = now
            if side == 1:
                self.positive += 1
                direction = "positive"
            else:
                self.negative += 1
                direction = "negative"

            crossings.append(
                Crossing(
                    line_id=self.line.id,
                    track_id=track.id,
                    label=track.label,
                    direction=direction,
                )
            )

        self._expire(now)
        return crossings

    def _within_span(self, point: tuple[float, float]) -> bool:
        position = self.line.span_position(point)
        return -_SPAN_TOLERANCE <= position <= 1.0 + _SPAN_TOLERANCE

    def _expire(self, now: float) -> None:
        cutoff = now - self.retention_s
        stale = [
            track_id
            for track_id, state in self._states.items()
            if state.last_seen < cutoff
        ]
        for track_id in stale:
            del self._states[track_id]
