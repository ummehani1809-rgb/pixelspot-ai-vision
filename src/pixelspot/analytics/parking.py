"""Parking bay occupancy.

Each bay is a zone from ``analytics.parking.bays``, and each runs a small
state machine: vacant -> filling -> occupied -> emptying -> vacant. The two
timers are the whole point. A car must sit in the bay for
``occupied_after_s`` before it is parked -- driving through a bay is not
parking. And a bay must be empty for ``vacated_after_s`` before it is free --
a car briefly hidden by a passing truck has not left.

``BAY_OCCUPIED`` fires when the settle timer completes, ``BAY_VACATED`` when
the empty timer does. In between, per-bay state is a metric. Any vehicle
class holds a bay; which one is in the metrics, not the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry, ResolvedZone
from pixelspot.settings.schema import PixelSpotConfig

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")


@dataclass
class _Bay:
    state: str = "vacant"  # vacant | filling | occupied | emptying
    since: float = 0.0

    @property
    def held(self) -> bool:
        return self.state in ("occupied", "emptying")


class ParkingProcessor(BaseProcessor):
    name = "parking"

    def __init__(
        self,
        bays: list[ResolvedZone],
        occupied_after_s: float = 30.0,
        vacated_after_s: float = 15.0,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
    ):
        self.bays = bays
        self.occupied_after_s = occupied_after_s
        self.vacated_after_s = vacated_after_s
        self.classes = classes
        self._states: dict[str, _Bay] = {bay.id: _Bay() for bay in bays}

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "ParkingProcessor":
        settings = config.analytics.parking
        return cls(
            bays=geometry.select_zones(settings.bays),
            occupied_after_s=settings.occupied_after_s,
            vacated_after_s=settings.vacated_after_s,
        )

    def process(self, context: FrameContext) -> ProcessorOutput:
        vehicles = context.of_classes(self.classes)
        output = ProcessorOutput()
        now = context.timestamp

        for bay in self.bays:
            state = self._states[bay.id]
            present = any(bay.contains(vehicle.center) for vehicle in vehicles)

            if state.state == "vacant" and present:
                state.state, state.since = "filling", now
            elif state.state == "filling":
                if not present:
                    state.state = "vacant"
                elif now - state.since >= self.occupied_after_s:
                    state.state = "occupied"
                    output.events.append(
                        self._event(context, "BAY_OCCUPIED", zone_id=bay.id)
                    )
            elif state.state == "occupied" and not present:
                state.state, state.since = "emptying", now
            elif state.state == "emptying":
                if present:
                    state.state = "occupied"
                elif now - state.since >= self.vacated_after_s:
                    state.state = "vacant"
                    output.events.append(
                        self._event(context, "BAY_VACATED", zone_id=bay.id)
                    )

        held = sum(1 for state in self._states.values() if state.held)
        output.metrics = {
            "bays_total": len(self.bays),
            "bays_occupied": held,
            "bays_free": len(self.bays) - held,
            "per_bay": {bay_id: state.state for bay_id, state in self._states.items()},
        }
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        return [
            f"Parking: {metrics.get('bays_occupied', 0)}/{metrics.get('bays_total', 0)}"
        ]
