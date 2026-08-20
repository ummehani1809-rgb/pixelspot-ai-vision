"""JSON Lines file output.

One JSON object per line, appended: the format that survives a process being
killed mid-write, because a truncated final line costs one record rather than
the whole file. Every record carries a ``record`` field so events and metric
snapshots can share a stream and be told apart downstream.

``sinks.jsonl.rotate_mb`` bounds the file. A device left running for a month
must not fill its disk with analytics, so at the limit the file is rolled aside
to ``<name>.1.jsonl`` and a fresh one started. Exactly one previous file is
kept: the point is a bounded footprint, not an archive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pixelspot import paths
from pixelspot.analytics.base import Event
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import JsonlSinkConfig
from pixelspot.sinks.base import BaseSink

log = get_logger(__name__)


class JsonlSink(BaseSink):
    name = "jsonl"

    def __init__(self, config: JsonlSinkConfig):
        self.path = paths.resolve(config.path)
        self.rotate_bytes = config.rotate_mb * 1024 * 1024
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        log.info("jsonl sink writing to %s", self.path)

    def emit_events(self, events: list[Event]) -> None:
        for event in events:
            self._write({"record": "event", **event.as_dict()})

    def emit_metrics(self, snapshot: dict[str, Any]) -> None:
        self._write({"record": "metrics", **snapshot})

    def _write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(payload, default=str) + "\n")
        self._handle.flush()
        self._rotate_if_needed()

    def _rotate_if_needed(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:  # pragma: no cover - file vanished under us
            return
        if size < self.rotate_bytes:
            return

        self._handle.close()
        previous = Path(f"{self.path}.1")
        previous.unlink(missing_ok=True)
        self.path.rename(previous)
        self._handle = self.path.open("a", encoding="utf-8")
        log.info("rotated %s at %.1f MB", self.path.name, size / 1024 / 1024)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
