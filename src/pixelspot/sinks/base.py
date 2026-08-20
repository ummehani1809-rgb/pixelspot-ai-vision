"""What every output destination has to provide.

Sinks are how analytics leave the process. Keeping them behind one small
interface means the pipeline emits once and does not know, or care, whether
that ends up on a terminal, in a file, or on a server -- and which of those
happen is entirely a matter of what is enabled under ``sinks``.

Events and metrics are delivered through separate calls because a destination
usually wants one and not the other: a console is useful for watching events
scroll past, a time-series store wants the periodic metric snapshot.
"""

from __future__ import annotations

from typing import Any, Protocol

from pixelspot.analytics.base import Event


class Sink(Protocol):
    name: str

    def emit_events(self, events: list[Event]) -> None: ...

    def emit_metrics(self, snapshot: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class BaseSink:
    """No-op defaults so a sink only implements what it actually supports."""

    name = "sink"

    def emit_events(self, events: list[Event]) -> None:
        return None

    def emit_metrics(self, snapshot: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None
