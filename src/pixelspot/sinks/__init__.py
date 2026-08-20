"""Output destinations, built from the ``sinks`` config block."""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import Event
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.sinks.base import BaseSink, Sink
from pixelspot.sinks.ccms import CCMSSink
from pixelspot.sinks.console import ConsoleSink
from pixelspot.sinks.jsonl import JsonlSink

log = get_logger(__name__)

__all__ = [
    "BaseSink",
    "CCMSSink",
    "ConsoleSink",
    "JsonlSink",
    "Sink",
    "SinkGroup",
    "build_sinks",
]


def build_sinks(config: PixelSpotConfig) -> list[Sink]:
    """Instantiate every enabled sink."""
    sinks: list[Sink] = []

    if config.sinks.console.enabled:
        sinks.append(ConsoleSink(config.sinks.console))
    if config.sinks.jsonl.enabled:
        sinks.append(JsonlSink(config.sinks.jsonl))
    if config.sinks.ccms.enabled:
        sinks.append(CCMSSink(config.sinks.ccms))

    if not sinks:
        log.warning("no sinks are enabled; analytics will not leave this process")
    return sinks


class SinkGroup:
    """Fans output out to every sink, and keeps one failure from taking the
    pipeline down.

    A sink is an output, not a dependency. A full disk or a broken network
    destination should cost its own records, not the counting -- so failures
    are logged once per sink and that sink is dropped.
    """

    def __init__(self, sinks: list[Sink]):
        self.sinks = list(sinks)

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "SinkGroup":
        return cls(build_sinks(config))

    def emit_events(self, events: list[Event]) -> None:
        if events:
            self._each(lambda sink: sink.emit_events(events))

    def emit_metrics(self, snapshot: dict[str, Any]) -> None:
        self._each(lambda sink: sink.emit_metrics(snapshot))

    def close(self) -> None:
        self._each(lambda sink: sink.close())
        self.sinks.clear()

    def _each(self, action) -> None:
        for sink in list(self.sinks):
            try:
                action(sink)
            except Exception:
                log.exception("sink %r failed and has been disabled", sink.name)
                self.sinks.remove(sink)
