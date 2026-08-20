"""CCMS sink.

Batches records up to ``sinks.ccms.batch_size`` before handing them to the
client, because one request per turnstile crossing is a poor trade on a link
that is metered, flaky, or both. Metric snapshots flush the pending batch with
them: a snapshot is already an interval boundary, so it is the natural moment
to stop holding events back.

Anything still pending is flushed on close, so shutting the pipeline down does
not quietly discard the last partial batch.
"""

from __future__ import annotations

from typing import Any

from pixelspot.analytics.base import Event
from pixelspot.ccms.client import CCMSClient
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import CCMSSinkConfig
from pixelspot.sinks.base import BaseSink

log = get_logger(__name__)

# How many batches worth of records to hold when spooling itself is failing.
_MAX_PENDING_BATCHES = 20


class CCMSSink(BaseSink):
    name = "ccms"

    def __init__(self, config: CCMSSinkConfig):
        self.client = CCMSClient(config)
        self.batch_size = config.batch_size
        self._max_pending = config.batch_size * _MAX_PENDING_BATCHES
        self._pending: list[dict[str, Any]] = []

    def emit_events(self, events: list[Event]) -> None:
        self._pending.extend(
            {"record": "event", **event.as_dict()} for event in events
        )
        if len(self._pending) >= self.batch_size:
            self.flush()

    def emit_metrics(self, snapshot: dict[str, Any]) -> None:
        self._pending.append({"record": "metrics", **snapshot})
        self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        if self.client.send(self._pending):
            self._pending.clear()
            return

        # Spooling failed, so the disk is full or unwritable. Holding every
        # record until it recovers would trade a logging problem for an
        # out-of-memory kill, so the backlog is bounded and the oldest go.
        overflow = len(self._pending) - self._max_pending
        if overflow > 0:
            del self._pending[:overflow]
            log.error("dropped %d undeliverable record(s) from the ccms backlog", overflow)

    def close(self) -> None:
        self.flush()
