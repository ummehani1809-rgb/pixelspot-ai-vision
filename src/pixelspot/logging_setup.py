"""Logging configuration.

PixelSpot logs instead of printing. Print statements cannot be filtered by
severity, carry no timestamp, and cannot be redirected to a file or to the
system journal on a device. All three matter once this runs unattended.

Call :func:`configure_logging` exactly once, at process start, before any other
PixelSpot module is used.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Third-party loggers that are far too chatty at INFO level. Ultralytics in
# particular logs a line for every single inference call.
_NOISY_LOGGERS = (
    "ultralytics",
    "matplotlib",
    "PIL",
    "urllib3",
)

_configured = False


class _LevelAwareFormatter(logging.Formatter):
    """Appends the frame number to records that carry one.

    Any log call made as ``log.info("...", extra={"frame_id": 123})`` gets a
    ``[f123]`` suffix, which makes it possible to line up a log line with the
    exact frame that produced it when debugging a counting error.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        frame_id = getattr(record, "frame_id", None)
        if frame_id is not None:
            message = f"{message}  [f{frame_id}]"
        return message


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    quiet_third_party: bool = True,
) -> None:
    """Set up root logging for the process.

    Args:
        level: Minimum severity to emit. Name or numeric level.
        log_file: Optional path to also append log records to.
        quiet_third_party: Hold noisy libraries at WARNING regardless of level.
    """
    global _configured

    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"unknown log level: {level!r}")
        level = resolved

    root = logging.getLogger()
    root.setLevel(level)

    # Replace any handlers from a previous call so repeated setup in tests or
    # notebooks does not produce duplicated output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = _LevelAwareFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if quiet_third_party:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(max(level, logging.WARNING))

    _configured = True


def is_configured() -> bool:
    """True once :func:`configure_logging` has run."""
    return _configured


def get_logger(name: str) -> logging.Logger:
    """Return the PixelSpot logger for *name*.

    Modules should use ``get_logger(__name__)`` so the emitting module shows up
    in every line.
    """
    return logging.getLogger(name)
