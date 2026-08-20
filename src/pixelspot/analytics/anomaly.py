"""Rule-based anomaly detection.

Watches the *other* processors' numbers and raises events when a configured
rule breaks. This is the one processor that reads ``context.outputs`` -- it
is declared last in the config schema, so it always runs after everything it
watches has reported.

A rule names a metric by dotted path from the processor down, e.g.
``footfall.current_occupancy`` or ``crowd_density.per_zone.storefront.count``,
and one of four conditions:

``above`` / ``below``
    The value crossed a fixed threshold. Fires ``ANOMALY`` once, on the way
    into breach, and ``ANOMALY_CLEARED`` once on the way out -- a breach that
    lasts an hour is one anomaly, not 54,000 of them.
``spike`` / ``drop``
    The value versus its own recent history: breached when it exceeds
    ``threshold`` times its mean over the rule's ``window`` (or falls below
    it, for drop). Nothing fires until a full window of history exists --
    startup is not a spike -- or while the baseline is zero, because "twice
    nothing" is not a signal.

A rule naming a metric that never appears (a typo, or its processor is
disabled) warns once and stays silent, matching how the registry treats
unimplemented capabilities.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pixelspot.analytics.base import BaseProcessor, FrameContext, ProcessorOutput
from pixelspot.geometry import ResolvedGeometry
from pixelspot.logging_setup import get_logger
from pixelspot.settings.schema import AnomalyRule, PixelSpotConfig, parse_duration

log = get_logger(__name__)


class AnomalyProcessor(BaseProcessor):
    name = "anomaly"

    def __init__(self, rules: list[AnomalyRule]):
        self.rules = rules
        self._windows_s = {rule.id: parse_duration(rule.window) for rule in rules}
        self._history: dict[str, deque[tuple[float, float]]] = {
            rule.id: deque() for rule in rules
        }
        self._breached: set[str] = set()
        self._warned: set[str] = set()

    @classmethod
    def from_config(
        cls, config: PixelSpotConfig, geometry: ResolvedGeometry
    ) -> "AnomalyProcessor":
        return cls(rules=list(config.analytics.anomaly.rules))

    def _resolve(self, outputs: dict[str, ProcessorOutput], path: str) -> float | None:
        head, _, rest = path.partition(".")
        output = outputs.get(head)
        if output is None:
            return None
        value: Any = output.metrics
        for part in rest.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return float(value) if isinstance(value, (int, float)) else None

    def _breaches(self, rule: AnomalyRule, value: float, now: float) -> bool | None:
        """True/False, or None while the rule cannot be evaluated yet."""
        if rule.condition == "above":
            return value > rule.threshold
        if rule.condition == "below":
            return value < rule.threshold

        history = self._history[rule.id]
        span = self._windows_s[rule.id]
        cutoff = now - span
        while history and history[0][0] < cutoff:
            history.popleft()
        baseline_samples = [sample for _, sample in history]
        history.append((now, value))

        if not baseline_samples or history[0][0] > cutoff + span * 0.5:
            return None  # less than half a window of history: too early to say
        baseline = sum(baseline_samples) / len(baseline_samples)
        if baseline <= 0:
            return None
        if rule.condition == "spike":
            return value > rule.threshold * baseline
        return value < rule.threshold * baseline  # drop

    def process(self, context: FrameContext) -> ProcessorOutput:
        output = ProcessorOutput()

        for rule in self.rules:
            value = self._resolve(context.outputs, rule.metric)
            if value is None:
                if rule.id not in self._warned:
                    self._warned.add(rule.id)
                    log.warning(
                        "anomaly rule %r watches metric %r, which is not "
                        "reported by any running processor",
                        rule.id, rule.metric,
                    )
                continue

            breached = self._breaches(rule, value, context.timestamp)
            if breached is None:
                continue

            if breached and rule.id not in self._breached:
                self._breached.add(rule.id)
                output.events.append(
                    self._event(
                        context,
                        "ANOMALY",
                        rule_id=rule.id,
                        metric=rule.metric,
                        condition=rule.condition,
                        value=round(value, 3),
                        threshold=rule.threshold,
                    )
                )
            elif not breached and rule.id in self._breached:
                self._breached.discard(rule.id)
                output.events.append(
                    self._event(context, "ANOMALY_CLEARED", rule_id=rule.id)
                )

        output.metrics = {
            "active": sorted(self._breached),
            "rules": len(self.rules),
        }
        return output

    def overlay_lines(self, metrics: dict[str, Any]) -> list[str]:
        active = metrics.get("active") or []
        if not active:
            return []
        return [f"ANOMALY: {', '.join(active)}"]
