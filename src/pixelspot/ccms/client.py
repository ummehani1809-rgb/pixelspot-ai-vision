"""CCMS delivery client.

The content management system is the eventual home for everything PixelSpot
counts. Its settings live under ``sinks.ccms``, and the schema already refuses
to enable it without a base URL, an API key and a device id -- all of which are
expected to arrive from the environment rather than a committed file.

HTTP delivery is not implemented yet. What is implemented is the part that
protects data in the meantime: every batch is written to ``spool_dir`` first.
A device on a shop floor loses its uplink regularly, and analytics that only
exist in memory are gone the moment that happens. Spooling first means delivery
can be added later and replay the backlog rather than starting from empty.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pixelspot import paths
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import CCMSSinkConfig

log = get_logger(__name__)


class CCMSClient:
    """Spools batches for the CCMS. Network delivery lands in a later phase."""

    def __init__(self, config: CCMSSinkConfig):
        self.config = config
        self.enabled = config.enabled
        self.device_id = config.device_id
        self.spool_dir = paths.resolve(config.spool_dir)
        self.spooled_batches = 0
        self._warned = False

        if self.enabled:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                "ccms client for device %s -> %s (spool %s)",
                self.device_id,
                config.base_url,
                self.spool_dir,
            )

    def send(self, records: list[dict[str, Any]]) -> bool:
        """Persist a batch for delivery. Returns whether it was accepted."""
        if not self.enabled or not records:
            return False

        if not self._warned:
            log.warning(
                "ccms upload is not implemented yet; batches are spooled to %s "
                "and will be replayed once delivery lands",
                self.spool_dir,
            )
            self._warned = True

        payload = {
            "device_id": self.device_id,
            "sent_at": time.time(),
            "records": records,
        }
        # Nanosecond-stamped name: unique, and sorts into send order.
        path = self.spool_dir / f"batch-{time.time_ns()}.json"

        try:
            path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except OSError as exc:
            log.error("could not spool %d record(s) to %s: %s", len(records), path, exc)
            return False

        self.spooled_batches += 1
        log.debug("spooled %d record(s) to %s", len(records), path.name)
        return True
