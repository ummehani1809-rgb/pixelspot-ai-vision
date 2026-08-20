"""PixelSpot configuration: schema, layered loading and the startup banner."""

from pixelspot.settings.banner import log_startup_banner, render_banner
from pixelspot.settings.loader import (
    ConfigError,
    LoadReport,
    check_resources,
    deep_merge,
    env_overrides,
    load_config,
)
from pixelspot.settings.schema import PixelSpotConfig

__all__ = [
    "ConfigError",
    "LoadReport",
    "PixelSpotConfig",
    "check_resources",
    "deep_merge",
    "env_overrides",
    "load_config",
    "log_startup_banner",
    "render_banner",
]
