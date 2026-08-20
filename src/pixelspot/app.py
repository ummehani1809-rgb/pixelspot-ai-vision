"""The pipeline.

Read a frame, track what is in it, ask each analytics processor what that
means, aggregate, emit, draw. That loop is the whole program, and it is
deliberately short: everything it does is assembled from the validated config
before the first frame, so the loop itself contains no policy and no thresholds
to drift out of sync with the config file.

Geometry is the one thing that cannot be built up front. Zones and lines are
authored in normalized coordinates and only become pixels once the source
reports its resolution, so the first frame is read before processors are built.
That ordering is why the loop below reads its next frame at the end rather than
the top.
"""

from __future__ import annotations

import time
from collections import deque

from pixelspot.aggregation.aggregator import Aggregator
from pixelspot.analytics.base import FrameContext
from pixelspot.analytics.registry import build_processors
from pixelspot.enrichment import FaceFinder, HeadPoseEstimator
from pixelspot.geometry import ResolvedGeometry
from pixelspot.logging_setup import get_logger
from pixelspot.render import Renderer
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.sinks import SinkGroup
from pixelspot.source import SourceError, VideoSource
from pixelspot.tracking.tracker import Tracker

log = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130


class FramePacer:
    """Holds the loop to ``runtime.target_fps``.

    Inference rate is a capacity decision, not something to be dictated by
    whatever rate the source happens to push. Capping it leaves CPU for the
    rest of the device and keeps a 30 fps camera from costing twice the power
    of a 15 fps one for no extra accuracy.
    """

    def __init__(self, target_fps: float, window: int = 30):
        self.budget = 1.0 / target_fps
        self._next_deadline = time.monotonic()
        self._durations: deque[float] = deque(maxlen=window)
        self._last_tick: float | None = None

    def tick(self) -> None:
        """Record the start of a frame."""
        now = time.monotonic()
        if self._last_tick is not None:
            self._durations.append(now - self._last_tick)
        self._last_tick = now

    def wait(self) -> None:
        """Sleep off whatever is left of this frame's time budget."""
        self._next_deadline += self.budget
        remaining = self._next_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -self.budget:
            # Running behind. Reset rather than accumulate a debt that would
            # make the loop sprint once it catches up.
            self._next_deadline = time.monotonic()

    @property
    def fps(self) -> float:
        if not self._durations:
            return 0.0
        average = sum(self._durations) / len(self._durations)
        return 1.0 / average if average > 0 else 0.0


def run_pipeline(config: PixelSpotConfig) -> int:
    """Run until the source ends, the operator quits, or a signal arrives."""
    source = VideoSource.from_config(config)

    try:
        source.open()
    except SourceError as exc:
        log.error("%s", exc)
        return EXIT_FAILURE

    frame = source.read()
    if frame is None:
        log.error("source %s delivered no frames", config.source.uri)
        source.release()
        return EXIT_FAILURE

    height, width = frame.shape[:2]
    geometry = ResolvedGeometry.resolve(config.geometry, width, height)
    log.info(
        "resolved geometry for %dx%d: %d zone(s), %d line(s)",
        width, height, len(geometry.zones), len(geometry.lines),
    )

    tracker = Tracker.from_config(config)
    enrichers = []
    if config.perception.enrichment.head_pose.enabled:
        enrichers.append(HeadPoseEstimator.from_config(config))
    if config.perception.enrichment.face.enabled:
        enrichers.append(FaceFinder.from_config(config))
    processors = build_processors(config, geometry)
    aggregator = Aggregator.from_config(config)
    sinks = SinkGroup.from_config(config)
    renderer = Renderer(config, geometry, processors)
    pacer = FramePacer(config.runtime.target_fps)

    log.info(
        "pipeline running: %s | quit with %s",
        ", ".join(processor.name for processor in processors) or "no analytics",
        "Ctrl-C" if config.runtime.headless else "q, Esc or Ctrl-C",
    )

    exit_code = EXIT_OK
    frame_index = 0

    try:
        while True:
            pacer.tick()
            tracks = tracker.track(frame, time.time())
            for enricher in enrichers:
                enricher.enrich(frame, tracks, frame_index)
            context = FrameContext(
                index=frame_index,
                timestamp=time.time(),
                width=width,
                height=height,
                tracks=tracks,
            )

            # Filled in run order so later processors (anomaly) can read
            # what earlier ones concluded about this same frame.
            for processor in processors:
                context.outputs[processor.name] = processor.process(context)
            sinks.emit_events(aggregator.update(context, context.outputs))

            if aggregator.due():
                sinks.emit_metrics(aggregator.snapshot(fps=round(pacer.fps, 2)))

            if not config.runtime.headless:
                renderer.draw(frame, context.tracks, aggregator.metrics, pacer.fps)
                if not renderer.show(frame):
                    log.info("stopped by operator")
                    break

            pacer.wait()

            frame = source.read()
            if frame is None:
                log.info("source finished after %d frames", frame_index + 1)
                break
            frame_index += 1

    except KeyboardInterrupt:
        log.warning("interrupted after %d frames", frame_index)
        exit_code = EXIT_INTERRUPTED
    except Exception:
        log.exception("pipeline failed on frame %d", frame_index)
        exit_code = EXIT_FAILURE
    finally:
        # One last snapshot so the final interval is not thrown away, then
        # close in the reverse order things were opened.
        if aggregator.metrics:
            sinks.emit_metrics(aggregator.snapshot(fps=round(pacer.fps, 2)))
        sinks.close()
        renderer.close()
        source.release()
        log.info(
            "processed %d frame(s), %d event(s), %.1f fps",
            frame_index + 1, aggregator.total_events, pacer.fps,
        )

    return exit_code
