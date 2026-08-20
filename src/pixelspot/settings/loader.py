"""Configuration loading and layering.

Settings are assembled from four sources, each overriding the one before it::

    built-in defaults  <  config.yaml  <  sites/<name>.yaml  <  environment  <  CLI

The precedence order is deliberate. Files describe how a deployment is *meant*
to be configured and belong in version control; environment variables carry the
per-device facts that must not be committed (credentials, camera URLs); CLI
flags are for the operator standing in front of the machine right now, so they
win.

Environment overrides use the field path with ``__`` between levels::

    PIXELSPOT__ANALYTICS__FOOTFALL__ENABLED=false
    PIXELSPOT__RUNTIME__DEVICE=cuda:0
    PIXELSPOT__PERCEPTION__DETECTOR__CLASSES=["person","car"]

Values are parsed as JSON when possible so numbers, booleans and lists work,
falling back to the raw string otherwise.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from pixelspot import paths
from pixelspot.settings.schema import PixelSpotConfig

ENV_PREFIX = "PIXELSPOT__"

# ${VAR} or ${VAR:-fallback}
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised for any configuration problem that should stop startup."""


@dataclass
class LoadReport:
    """Where each layer of the resolved config came from.

    Recorded so the startup banner can state plainly which files were read and
    which overrides were applied, instead of leaving the operator guessing.
    """

    config_path: Path | None = None
    site_path: Path | None = None
    env_keys: list[str] = field(default_factory=list)
    cli_keys: list[str] = field(default_factory=list)


# ==================================================================
# MERGING
# ==================================================================


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* onto *base*.

    Nested dicts merge key by key. Lists replace wholesale rather than
    concatenating: a site that lists two zones means exactly those two zones,
    not the base zones plus two more.
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


# ==================================================================
# VARIABLE EXPANSION
# ==================================================================


def expand_variables(node: Any, environ: Mapping[str, str]) -> Any:
    """Replace ``${VAR}`` and ``${VAR:-default}`` inside string values.

    A value that is *exactly* one unset placeholder with no default becomes
    ``None`` rather than an empty string, so an unset optional setting stays
    absent instead of turning into a blank that passes validation.
    """
    if isinstance(node, dict):
        return {key: expand_variables(value, environ) for key, value in node.items()}
    if isinstance(node, list):
        return [expand_variables(item, environ) for item in node]
    if not isinstance(node, str):
        return node

    whole = _VAR_PATTERN.fullmatch(node)
    if whole is not None:
        name, default = whole.groups()
        if name in environ:
            return environ[name]
        return default if default is not None else None

    def substitute(match: re.Match[str]) -> str:
        name, default = match.groups()
        return environ.get(name, default if default is not None else "")

    return _VAR_PATTERN.sub(substitute, node)


# ==================================================================
# ENVIRONMENT OVERRIDES
# ==================================================================


def _coerce(raw: str) -> Any:
    """Interpret an environment value as JSON, else keep it as a string."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def env_overrides(environ: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Build a nested override dict from ``PIXELSPOT__*`` variables."""
    overrides: dict[str, Any] = {}
    applied: list[str] = []

    for raw_key in sorted(environ):
        if not raw_key.startswith(ENV_PREFIX):
            continue
        remainder = raw_key[len(ENV_PREFIX):]
        parts = [part.lower() for part in remainder.split("__")]
        if not remainder or any(not part for part in parts):
            raise ConfigError(
                f"malformed environment override {raw_key!r}: expected "
                f"{ENV_PREFIX}SECTION__FIELD"
            )

        cursor = overrides
        for part in parts[:-1]:
            nested = cursor.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ConfigError(
                    f"environment override {raw_key!r} conflicts with another "
                    "override that sets a value at the same path"
                )
            cursor = nested
        cursor[parts[-1]] = _coerce(environ[raw_key])
        applied.append(raw_key)

    return overrides, applied


# ==================================================================
# FILE READING
# ==================================================================


def read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping, with useful errors for the usual mistakes."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    if path.is_dir():
        raise ConfigError(f"config path is a directory, not a file: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML:\n{exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(raw).__name__}"
        )
    return raw


# ==================================================================
# ERROR FORMATTING
# ==================================================================


def format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error as a flat list of ``path: message`` lines."""
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        message = error["msg"]
        # Cross-reference checks raise one ValueError holding several problems.
        for piece in message.removeprefix("Value error, ").split("\n"):
            piece = piece.strip()
            if not piece:
                continue
            if location == "<root>":
                lines.append(f"  - {piece}")
            else:
                lines.append(f"  - {location}: {piece}")
    return "\n".join(dict.fromkeys(lines))


# ==================================================================
# ENTRY POINT
# ==================================================================


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    site: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[PixelSpotConfig, LoadReport]:
    """Assemble, validate and return the effective configuration.

    Args:
        config_path: Base YAML file. Defaults to ``<project>/config/config.yaml``.
        site: Name of a per-site override in ``config/sites/``.
        cli_overrides: Nested dict from command line flags, highest precedence.
        environ: Environment to read. Defaults to ``os.environ``.

    Raises:
        ConfigError: The config could not be read or failed validation.
    """
    environ = os.environ if environ is None else environ
    report = LoadReport()

    base_path = Path(config_path) if config_path else paths.DEFAULT_CONFIG
    report.config_path = base_path
    data = read_yaml(base_path)

    if site:
        site_path = paths.site_config(site)
        report.site_path = site_path
        data = deep_merge(data, read_yaml(site_path))

    data = expand_variables(data, environ)

    env_data, report.env_keys = env_overrides(environ)
    if env_data:
        data = deep_merge(data, env_data)

    if cli_overrides:
        data = deep_merge(data, cli_overrides)
        report.cli_keys = sorted(cli_overrides)

    try:
        config = PixelSpotConfig.model_validate(data)
    except ValidationError as exc:
        sources = [str(base_path)]
        if report.site_path:
            sources.append(str(report.site_path))
        raise ConfigError(
            "Invalid configuration ("
            + ", ".join(sources)
            + "):\n"
            + format_validation_error(exc)
        ) from exc

    return config, report


# ==================================================================
# RUNTIME RESOURCE CHECKS
# ==================================================================


def check_resources(config: PixelSpotConfig) -> list[str]:
    """Check that files the config points at actually exist.

    Kept separate from schema validation because these are facts about *this
    machine*, not about whether the config is well formed. A site config can be
    structurally perfect on a laptop that does not happen to hold the video.
    """
    problems: list[str] = []

    detector_model = paths.resolve(config.perception.detector.model)
    if not detector_model.exists():
        problems.append(
            f"perception.detector.model not found: {detector_model}"
        )

    head_pose = config.perception.enrichment.head_pose
    if head_pose.enabled:
        model = paths.resolve(head_pose.model)
        if not model.exists():
            problems.append(
                f"perception.enrichment.head_pose.model not found: {model}"
            )

    if config.source.type == "file":
        source = paths.resolve(config.source.uri)
        if not source.exists():
            problems.append(f"source.uri not found: {source}")

    if config.sinks.jsonl.enabled:
        parent = paths.resolve(config.sinks.jsonl.path).parent
        if not parent.exists() and not parent.parent.exists():
            problems.append(f"sinks.jsonl.path directory cannot be created: {parent}")

    return problems
