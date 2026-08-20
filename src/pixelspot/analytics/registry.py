"""Which analytics capabilities exist, and how to build them.

This is the join between the config and the pipeline. ``analytics.<name>
.enabled`` decides what runs; this table decides what that name builds into.
The pipeline itself never names a capability, so a new processor is one class
plus one entry here.

A capability that is enabled but not implemented yet is a warning rather than
an error. The schema deliberately accepts config for the later phases so a
deployment can be written once; refusing to start because a Phase 5 block was
switched on early would make that config unusable in the meantime.
"""

from __future__ import annotations

from typing import Callable

from pixelspot.analytics.base import Processor
from pixelspot.analytics.dwell import DwellProcessor
from pixelspot.analytics.footfall import FootfallProcessor
from pixelspot.analytics.vehicle import VehicleProcessor
from pixelspot.analytics.viewing_zone import ViewingZoneProcessor
from pixelspot.geometry import ResolvedGeometry
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import PixelSpotConfig

log = get_logger(__name__)

Builder = Callable[[PixelSpotConfig, ResolvedGeometry], Processor]

BUILDERS: dict[str, Builder] = {
    "footfall": FootfallProcessor.from_config,
    "viewing_zone": ViewingZoneProcessor.from_config,
    "dwell": DwellProcessor.from_config,
    "vehicles": VehicleProcessor.from_config,
}

IMPLEMENTED = frozenset(BUILDERS)


def build_processors(
    config: PixelSpotConfig, geometry: ResolvedGeometry
) -> list[Processor]:
    """Instantiate every enabled capability that has an implementation."""
    processors: list[Processor] = []

    for name in config.enabled_features():
        builder = BUILDERS.get(name)
        if builder is None:
            log.warning(
                "analytics.%s is enabled but has no implementation yet; skipping",
                name,
            )
            continue
        processors.append(builder(config, geometry))

    if not processors:
        log.warning(
            "no analytics capabilities are running; the pipeline will only "
            "detect and track"
        )
    return processors
