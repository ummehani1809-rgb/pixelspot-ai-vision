"""Console output.

Replaces the bare ``print`` the pipeline used to do on every crossing. What it
shows is ``sinks.console.emit``: events, metrics, or both. Output goes through
the logger, so it carries a timestamp, can be filtered by level and lands in
``runtime.log_file`` along with everything else -- none of which a print can do.
"""

from __future__ import annotations

import json
from typing import Any

from pixelspot.analytics.base import Event
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import ConsoleSinkConfig
from pixelspot.sinks.base import BaseSink

log = get_logger(__name__)


class ConsoleSink(BaseSink):
    name = "console"

    def __init__(self, config: ConsoleSinkConfig):
        self.show_events = config.emit in ("events", "both")
        self.show_metrics = config.emit in ("metrics", "both")

    def emit_events(self, events: list[Event]) -> None:
        if not self.show_events:
            return
        for event in events:
            log.info("%s", event.describe(), extra={"frame_id": event.frame_index})

    def emit_metrics(self, snapshot: dict[str, Any]) -> None:
        if not self.show_metrics:
            return
        for name, metrics in snapshot.get("metrics", {}).items():
            log.info("%-14s %s", name, json.dumps(metrics, default=str))
        for window, counts in snapshot.get("windows", {}).items():
            if counts:
                log.info("window %-5s %s", window, json.dumps(counts))
