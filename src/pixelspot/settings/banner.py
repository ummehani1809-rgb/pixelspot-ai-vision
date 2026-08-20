"""Startup banner.

Prints what the process is actually going to do before it does it: which files
the config came from, which source, which model and device, and which analytics
capabilities are on. On a device this log line is often the only evidence
available when a deployment misbehaves, so it states the resolved values rather
than the file contents.

Secrets are never rendered.
"""

from __future__ import annotations

from pixelspot import paths
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import LoadReport
from pixelspot.settings.schema import AnalyticsConfig, PixelSpotConfig

log = get_logger(__name__)

_WIDTH = 68
_RULE = "=" * _WIDTH


def _feature_rows(analytics: AnalyticsConfig) -> list[str]:
    """One aligned ``name  ON/off  detail`` row per processor."""
    rows: list[str] = []
    width = max(len(name) for name, _ in analytics.items())

    for name, block in analytics.items():
        state = "ON " if block.enabled else "off"
        detail = ""

        if block.enabled:
            targets: list[str] = []
            for attribute, label in (
                ("lines", "lines"),
                ("zones", "zones"),
                ("bays", "bays"),
            ):
                values = getattr(block, attribute, None)
                if values:
                    targets.append(f"{label}={','.join(values)}")

            screen = getattr(block, "screen", None)
            if screen:
                targets.append(f"screen={screen}")

            backend = getattr(block, "backend", None)
            if backend is not None:
                targets.append(f"backend={backend}")

            detail = "  " + "  ".join(targets) if targets else ""

        rows.append(f"  {name:<{width}}  {state}{detail}")

    return rows


def render_banner(config: PixelSpotConfig, report: LoadReport) -> str:
    """Build the multi-line startup banner."""
    lines: list[str] = [_RULE, "PixelSpot AI Analytics".center(_WIDTH), _RULE]

    lines.append(f"  config      {report.config_path}")
    if report.site_path:
        lines.append(f"  site        {report.site_path}")
    if report.env_keys:
        lines.append(f"  env         {len(report.env_keys)} override(s): "
                     + ", ".join(report.env_keys))
    if report.cli_keys:
        lines.append(f"  cli         {', '.join(report.cli_keys)}")

    lines.append("")
    source_uri = config.source.uri
    if config.source.type == "file":
        source_uri = str(paths.resolve(source_uri))
    lines.append(f"  source      {config.source.type}  {source_uri}")
    lines.append(
        f"  detector    {config.perception.detector.model}  "
        f"imgsz={config.perception.detector.imgsz}  "
        f"conf={config.perception.detector.conf}"
    )
    lines.append(
        f"  tracker     {config.perception.tracker.type}  "
        f"max_age={config.perception.tracker.max_age_s}s"
    )
    lines.append(
        f"  runtime     device={config.runtime.device}  "
        f"target_fps={config.runtime.target_fps}  "
        f"headless={config.runtime.headless}"
    )

    enrichment = config.perception.enrichment
    enabled_enrichment = [
        name
        for name, block in (("head_pose", enrichment.head_pose), ("face", enrichment.face))
        if block.enabled
    ]
    lines.append(
        "  enrichment  " + (", ".join(enabled_enrichment) if enabled_enrichment else "none")
    )

    lines.append(
        f"  geometry    {len(config.geometry.zones)} zone(s), "
        f"{len(config.geometry.lines)} line(s), "
        f"{len(config.geometry.screens)} screen(s)  "
        f"[{config.geometry.coordinate_space}]"
    )

    sinks = [
        name
        for name, block in (
            ("console", config.sinks.console),
            ("jsonl", config.sinks.jsonl),
            ("ccms", config.sinks.ccms),
        )
        if block.enabled
    ]
    lines.append("  sinks       " + (", ".join(sinks) if sinks else "none"))

    privacy = config.privacy
    lines.append(
        f"  privacy     aggregate_only={privacy.aggregate_only}  "
        f"store_crops={privacy.store_crops}  "
        f"blur_faces={privacy.blur_faces_in_output}"
    )

    lines.append("")
    lines.append("  CAPABILITIES")
    lines.extend(_feature_rows(config.analytics))

    enabled = config.enabled_features()
    lines.append("")
    lines.append(
        f"  {len(enabled)} of {len(config.analytics.items())} capabilities enabled"
    )
    lines.append(_RULE)

    return "\n".join(lines)


def log_startup_banner(config: PixelSpotConfig, report: LoadReport) -> None:
    """Emit the banner through the logger, one record per line."""
    for line in render_banner(config, report).splitlines():
        log.info(line)
