"""Command line entry point.

Two commands:

``pixelspot run``
    Load config, log the startup banner and run the pipeline.

``pixelspot validate-config``
    Load and validate config, report every problem, and exit. Touches no
    camera, model or GPU, so it can be run on a device before restarting the
    service to confirm a new config will not take the deployment down.

Exit codes: ``0`` success, ``1`` configuration or runtime failure, ``2``
command line usage error, ``130`` interrupted.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from pixelspot import paths
from pixelspot.logging_setup import configure_logging, get_logger
from pixelspot.settings import ConfigError, check_resources, load_config
from pixelspot.settings.banner import log_startup_banner

log = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130


def _infer_source_type(uri: str) -> str:
    lowered = uri.lower()
    if lowered.startswith("rtsp://"):
        return "rtsp"
    if lowered.startswith(("http://", "https://")):
        return "http"
    if uri.isdigit():
        return "webcam"
    return "file"


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Turn command line flags into a nested config override dict."""
    overrides: dict[str, Any] = {}
    runtime: dict[str, Any] = {}

    if getattr(args, "source", None):
        overrides["source"] = {
            "uri": args.source,
            "type": _infer_source_type(args.source),
        }
    if getattr(args, "device", None):
        runtime["device"] = args.device
    if getattr(args, "headless", False):
        runtime["headless"] = True
    if getattr(args, "fps", None):
        runtime["target_fps"] = args.fps
    if getattr(args, "log_level", None):
        runtime["log_level"] = args.log_level

    if runtime:
        overrides["runtime"] = runtime
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixelspot",
        description="Edge computer-vision audience analytics.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        help=f"base config file (default: {paths.DEFAULT_CONFIG})",
    )
    common.add_argument(
        "--site",
        metavar="NAME",
        help="per-site override from config/sites/<NAME>.yaml",
    )
    common.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="override runtime.log_level",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", parents=[common], help="run the analytics pipeline"
    )
    run_parser.add_argument(
        "--source", metavar="URI", help="video file, camera index or stream URL"
    )
    run_parser.add_argument(
        "--device", metavar="DEV", help="auto, cpu, mps, cuda or cuda:<index>"
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="do not open a display window",
    )
    run_parser.add_argument(
        "--fps", type=float, metavar="N", help="override runtime.target_fps"
    )
    run_parser.set_defaults(handler=command_run)

    validate_parser = subparsers.add_parser(
        "validate-config",
        parents=[common],
        help="check the configuration and exit",
    )
    validate_parser.add_argument(
        "--skip-resource-check",
        action="store_true",
        help="validate structure only, do not check that files exist on this machine",
    )
    validate_parser.set_defaults(handler=command_validate)

    return parser


def command_run(args: argparse.Namespace) -> int:
    config, report = load_config(
        config_path=args.config,
        site=args.site,
        cli_overrides=_collect_overrides(args),
    )

    # Reconfigure now that the resolved level and log file are known.
    configure_logging(config.runtime.log_level, config.runtime.log_file)
    log_startup_banner(config, report)

    problems = check_resources(config)
    if problems:
        for problem in problems:
            log.error("%s", problem)
        return EXIT_FAILURE

    # Imported here, not at module scope: it pulls in OpenCV and torch, and
    # validate-config must stay runnable on a machine that has neither.
    try:
        from pixelspot.app import run_pipeline
    except ImportError as exc:
        log.error(
            "cannot start the pipeline: %s. Install the runtime dependencies "
            "with 'pip install -r requirements.txt'.",
            exc,
        )
        return EXIT_FAILURE

    return run_pipeline(config)


def command_validate(args: argparse.Namespace) -> int:
    config, report = load_config(config_path=args.config, site=args.site)

    sources = [str(report.config_path)]
    if report.site_path:
        sources.append(str(report.site_path))
    print(f"Configuration OK: {', '.join(sources)}")
    if report.env_keys:
        print(f"  environment overrides applied: {', '.join(report.env_keys)}")

    enabled = config.enabled_features()
    print(f"  capabilities enabled ({len(enabled)}): {', '.join(enabled) or 'none'}")

    if args.skip_resource_check:
        return EXIT_OK

    problems = check_resources(config)
    if problems:
        print("\nResource problems on this machine:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_FAILURE

    print("  all referenced files present")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bootstrap logging so failures during config loading are visible. The
    # resolved level from config is applied once loading succeeds.
    configure_logging(args.log_level or "INFO")

    try:
        return args.handler(args)
    except ConfigError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        log.warning("interrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
