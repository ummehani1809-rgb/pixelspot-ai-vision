"""Video input.

One class covers all four source types because the pipeline should not care
whether frames come from a file on disk or a camera on a pole. What differs is
what a failed read *means*, and that is the whole reason this file exists:

* For a file, a failed read is the end of the video. Either loop or stop.
* For a live source, a failed read is a network or USB problem. Retrying with
  exponential backoff is the correct response, and giving up is a decision the
  operator makes through ``source.reconnect``.

Live sources are also read on a background thread. A camera pushes frames at
its own rate; if the pipeline reads them one at a time and inference is slower
than the stream, OpenCV hands back frames from an ever growing internal buffer
and the analytics drift further behind real time with every frame. The reader
thread keeps draining the socket and applies ``source.buffer.drop_policy``, so
what the pipeline gets is the *current* view of the world rather than the
oldest frame nobody has looked at yet.

Files are read directly with no thread: there is no real-time to fall behind,
and dropping frames from a recording would silently change the counts. The one
exception is ``source.realtime``: when the operator asks for it, a file is
played back at its own recorded speed, and frames that inference was too slow
to reach are skipped just as a live camera would have dropped them. That trades
the every-frame guarantee for a preview that moves at the speed of the world it
recorded -- the right trade for demos, the wrong one for offline measurement,
which is why it is off by default.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import cv2

from pixelspot import paths
from pixelspot.logging_setup import get_logger
from pixelspot.settings.loader import ConfigError
from pixelspot.settings.schema import PixelSpotConfig, SourceConfig

log = get_logger(__name__)

LIVE_TYPES = frozenset({"webcam", "rtsp", "http"})


class SourceError(Exception):
    """Raised when the source cannot be opened or has permanently failed."""


class PlaybackClock:
    """Tracks how far behind real time a file playback has fallen.

    Started on the first frame; :meth:`behind` says how many frames the file
    should have advanced past what was consumed so far. A gap longer than
    ``max_stall_s`` is treated as a stall (model warm-up, a debugger, the
    machine asleep) rather than playback debt: the clock re-anchors and no
    frames are skipped, because a viewer who pauses a video does not expect
    it to jump when it resumes.
    """

    def __init__(self, fps: float, max_stall_s: float = 5.0, now=time.monotonic):
        self.fps = fps
        self.max_stall_s = max_stall_s
        self._now = now
        self._start: float | None = None
        self._consumed = 0

    def behind(self) -> int:
        """Frames to skip before the next read keeps pace with real time."""
        if self.fps <= 0:
            return 0
        now = self._now()
        if self._start is None:
            self._start = now
            return 0
        expected = int((now - self._start) * self.fps)
        lag = expected - self._consumed
        if lag > self.max_stall_s * self.fps:
            self._start = now - self._consumed / self.fps
            return 0
        return max(0, lag)

    def consume(self, count: int = 1) -> None:
        """Record frames taken from the file, read or skipped."""
        self._consumed += count

    def restart(self) -> None:
        """The file looped back to its beginning; start timing afresh."""
        self._start = None
        self._consumed = 0


class VideoSource:
    """A video input that knows how to reopen itself."""

    def __init__(self, config: SourceConfig):
        self.config = config
        self.is_live = config.type in LIVE_TYPES
        self.target: Any = self._resolve_target(config)

        self._capture: cv2.VideoCapture | None = None
        self._reader: _BufferedReader | None = None
        self._playback: PlaybackClock | None = None
        self._frames_read = 0
        self.frames_skipped = 0

    @classmethod
    def from_config(cls, config: PixelSpotConfig) -> "VideoSource":
        return cls(config.source)

    @staticmethod
    def _resolve_target(config: SourceConfig) -> Any:
        if config.type == "webcam":
            return int(config.uri)
        if config.type == "file":
            path = paths.resolve(config.uri)
            if not path.exists():
                raise ConfigError(f"source.uri not found: {path}")
            return str(path)
        return config.uri

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the source, retrying live sources per ``source.reconnect``."""
        reconnect = self.config.reconnect
        backoff = reconnect.initial_backoff_s
        attempt = 0

        while True:
            attempt += 1
            capture = self._open_capture()
            if capture.isOpened():
                self._capture = capture
                if self.is_live:
                    self._reader = _BufferedReader(capture, self.config.buffer)
                    self._reader.start()
                elif self.config.realtime:
                    self._playback = PlaybackClock(self.native_fps)
                log.info(
                    "source open: %s %s (%dx%d @ %.1f fps)",
                    self.config.type,
                    self.target,
                    self.width,
                    self.height,
                    self.native_fps,
                )
                return

            capture.release()

            if not self.is_live or not reconnect.enabled:
                raise SourceError(f"could not open source: {self.target}")
            if reconnect.max_attempts and attempt >= reconnect.max_attempts:
                raise SourceError(
                    f"could not open source after {attempt} attempts: {self.target}"
                )

            log.warning(
                "could not open %s (attempt %d); retrying in %.1fs",
                self.target,
                attempt,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, reconnect.max_backoff_s)

    def _open_capture(self) -> cv2.VideoCapture:
        """Open the capture, working around wedged webcam backends.

        On Windows the default backend (Media Foundation) can enter a state
        where the camera *opens* but never delivers a frame -- typically
        after a process holding it was killed. A capture that cannot produce
        one probe frame is useless however open it claims to be, so webcams
        are probed and the DirectShow backend is tried before giving up.
        """
        if self.config.type != "webcam":
            return cv2.VideoCapture(self.target)

        backends = [("default", cv2.CAP_ANY)]
        if sys.platform == "win32":
            backends.append(("DirectShow", cv2.CAP_DSHOW))

        capture = None
        for name, backend in backends:
            capture = cv2.VideoCapture(self.target, backend)
            if not capture.isOpened():
                capture.release()
                continue
            success, _ = capture.read()
            if success:
                return capture
            log.warning(
                "webcam %s opened via %s but delivered no frame; %s",
                self.target,
                name,
                "trying next backend" if backend != backends[-1][1] else "giving up",
            )
            capture.release()

        # Nothing produced a frame; hand back a closed capture so the
        # caller's retry/backoff path takes over.
        return capture if capture is not None else cv2.VideoCapture(self.target)

    def read(self):
        """Return the next frame, or ``None`` when the source is finished."""
        if self._capture is None:
            raise SourceError("read() called before open()")

        if self._reader is not None:
            frame = self._reader.read()
            if frame is not None:
                self._frames_read += 1
                return frame
            return self._recover()

        if self._playback is not None:
            for _ in range(self._playback.behind()):
                if not self._capture.grab():
                    break  # end of file; the read below reports it
                self._playback.consume()
                self.frames_skipped += 1

        success, frame = self._capture.read()
        if success:
            self._frames_read += 1
            if self._playback is not None:
                self._playback.consume()
            return frame

        if self.config.loop and self._frames_read:
            log.info("source ended after %d frames; looping", self._frames_read)
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self._playback is not None:
                self._playback.restart()
            success, frame = self._capture.read()
            if success:
                self._frames_read += 1
                if self._playback is not None:
                    self._playback.consume()
                return frame

        return None

    def _recover(self):
        """A live source stopped delivering. Reopen it if allowed."""
        reconnect = self.config.reconnect
        if not reconnect.enabled:
            log.error("source %s stopped and reconnect is disabled", self.target)
            return None

        log.warning("source %s stopped delivering frames; reconnecting", self.target)
        self.release()
        try:
            self.open()
        except SourceError as exc:
            log.error("%s", exc)
            return None
        return self.read()

    def release(self) -> None:
        if self.frames_skipped:
            log.info(
                "skipped %d frame(s) to keep playback at the recording's speed",
                self.frames_skipped,
            )
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def _property(self, flag: int, default: float = 0.0) -> float:
        if self._capture is None:
            return default
        value = self._capture.get(flag)
        return value if value and value > 0 else default

    @property
    def width(self) -> int:
        return int(self._property(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._property(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def native_fps(self) -> float:
        return float(self._property(cv2.CAP_PROP_FPS))

    @property
    def frames_read(self) -> int:
        return self._frames_read

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class _BufferedReader:
    """Drains a live capture on a thread so the pipeline never reads stale frames."""

    def __init__(self, capture: cv2.VideoCapture, buffer_config) -> None:
        self._capture = capture
        self._policy = buffer_config.drop_policy
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_config.max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pixelspot-reader", daemon=True)
        self._failed = threading.Event()
        self.dropped = 0

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            success, frame = self._capture.read()
            if not success:
                self._failed.set()
                return
            self._offer(frame)

    def _offer(self, frame) -> None:
        if self._policy == "block":
            self._queue.put(frame)
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1
            if self._policy == "drop_newest":
                return
            # drop_oldest: make room by discarding the frame nobody read.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except (queue.Empty, queue.Full):  # pragma: no cover - race with reader
                pass

    def read(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._failed.is_set() and self._queue.empty():
                    return None
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
