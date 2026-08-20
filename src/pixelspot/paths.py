"""Project path resolution.

Everything that needs a file on disk goes through here. Nothing in PixelSpot
should build a path relative to the current working directory, because the
process may be started from anywhere: a shell, a systemd unit, a cron job or a
container entrypoint.

Resolution order for the project root:

1. ``PIXELSPOT_HOME`` environment variable, if set.
2. The nearest ancestor of the current working directory containing a marker.
3. The nearest ancestor of this file containing a marker.
4. The parent of the ``src`` directory this package lives in.
"""

from __future__ import annotations

import os
from pathlib import Path

# A directory is the project root if it contains any of these.
_ROOT_MARKERS = ("pyproject.toml", "config/config.yaml", ".git")

# .../src/pixelspot/paths.py -> .../src/pixelspot
PACKAGE_DIR = Path(__file__).resolve().parent


def _walk_up(start: Path) -> Path | None:
    """Return the nearest ancestor of *start* (inclusive) holding a marker."""
    for candidate in (start, *start.parents):
        for marker in _ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    return None


def find_project_root() -> Path:
    """Locate the project root directory."""
    override = os.environ.get("PIXELSPOT_HOME")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"PIXELSPOT_HOME points at {root}, which is not a directory"
            )
        return root

    for start in (Path.cwd(), PACKAGE_DIR):
        found = _walk_up(start)
        if found is not None:
            return found

    # Installed as a wheel with no marker anywhere: fall back to src/..
    return PACKAGE_DIR.parent.parent


PROJECT_ROOT = find_project_root()

CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"
SITES_DIR = CONFIG_DIR / "sites"
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "out"


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve *path* against the project root unless it is already absolute.

    Lets config files use short, portable paths such as ``models/yolo11n.pt``
    that mean the same thing on a laptop and on a device.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def site_config(name: str) -> Path:
    """Path to the named per-site override file, e.g. ``sites/store-42.yaml``."""
    return SITES_DIR / f"{name}.yaml"
