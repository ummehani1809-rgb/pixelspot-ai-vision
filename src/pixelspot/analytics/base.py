"""Shared vocabulary for analytics processors.

Every processor takes the same :class:`FrameContext` and returns the same
:class:`ProcessorOutput`, which is what lets the pipeline build its processor
list from the config and then stay out of the way. Adding a capability means
adding a class and one registry entry: no branch in the main loop, no new
argument threaded through the aggregator, no new line in the renderer.

The split inside the output matters. *Metrics* are a level -- the answer to
"what is true right now" -- and are safe to overwrite and to sample at whatever
rate a dashboard wants. *Events* happened once, at a known moment, and are lost
forever if dropped. They travel separately because they are aggregated, retried
and deduplicated differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pixelspot.tracking.tracker import Track


@dataclass(frozen=True)
class Event:
    """Something that happened at one instant, on one frame."""

    type: str
    processor: str
    timestamp: float
    frame_index: int
    data: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[Any, ...]:
        """Identity used for deduplication.

        Two events with the same key describe the same real-world occurrence,
        whichever frame happened to notice it.
        """
        return (
            self.processor,
            self.type,
            self.data.get("track_id"),
            self.data.get("line_id")
            or self.data.get("zone_id")
            or self.data.get("rule_id"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "processor": self.processor,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            **self.data,
        }

    def describe(self) -> str:
        details = " ".join(f"{name}={value}" for name, value in self.data.items())
        return f"{self.type} {details}".strip()


@dataclass
class ProcessorOutput:
    """One processor's answer for one frame."""

    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)


@dataclass
class FrameContext:
    """Everything a processor is allowed to know about the current frame.

    ``outputs`` holds what processors *earlier in this frame* concluded, in
    run order. Most processors never look at it; anomaly runs last on
    purpose and watches the others' metrics through it.
    """

    index: int
    timestamp: float
    width: int
    height: int
    tracks: list[Track]
    outputs: dict[str, ProcessorOutput] = field(default_factory=dict)

    def of_classes(self, classes: tuple[str, ...] | list[str]) -> list[Track]:
        wanted = {label.lower() for label in classes}
        return [track for track in self.tracks if track.label.lower() in wanted]


class Processor(Protocol):
    """What the pipeline requires of an analytics capability."""

    name: str

    def process(self, context: FrameContext) -> ProcessorOutput: ...

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]: ...


class BaseProcessor:
    """Default behaviour shared by the concrete processors."""

    name = "processor"

    def process(self, context: FrameContext) -> ProcessorOutput:  # pragma: no cover
        raise NotImplementedError

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        """Text for the on-screen overlay. Empty means draw nothing."""
        return []

    def _event(self, context: FrameContext, event_type: str, **data: Any) -> Event:
        return Event(
            type=event_type,
            processor=self.name,
            timestamp=context.timestamp,
            frame_index=context.index,
            data=data,
        )
