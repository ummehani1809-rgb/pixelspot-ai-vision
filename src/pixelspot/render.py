"""Preview rendering.

The overlay used to be a wall of ``cv2`` calls with the line position, the zone
rectangle and every text position written twice: once where it was drawn and
once where it was measured. Now the geometry is drawn from the same resolved
objects the processors count against, so what is on screen cannot disagree with
what is being counted -- which is the entire point of a preview window.

What the overlay says about analytics comes from the processors themselves,
through ``overlay_lines``. The renderer never names a capability, so a new
processor appears on screen without this file changing.

``privacy.blur_faces_in_output`` blurs the head region of every person before
anything else is drawn. It is an approximation -- the top fraction of the
person box, since no face detector runs in this phase -- and it is deliberately
crude in the safe direction: it blurs more than a face, never less.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2

from pixelspot.analytics.base import Processor
from pixelspot.geometry import ResolvedGeometry
from pixelspot.settings.schema import PixelSpotConfig
from pixelspot.tracking.tracker import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR. Anything not listed falls back to the default box colour.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "person": (255, 0, 255),
    "car": (0, 200, 255),
    "motorcycle": (0, 255, 255),
    "bus": (255, 160, 0),
    "truck": (200, 100, 255),
}
DEFAULT_COLOUR = (200, 200, 200)
ZONE_COLOUR = (255, 255, 0)
LINE_COLOUR = (255, 0, 255)
TEXT_COLOUR = (255, 255, 255)

# Fraction of a person box treated as the head when blurring.
_HEAD_FRACTION = 0.28

# Overlay sizes below are authored for a frame whose short side is this many
# pixels; larger frames scale everything up so the text stays readable once
# the window is shrunk to fit the screen.
_REFERENCE_SHORT_SIDE = 720.0


class Renderer:
    """Draws the preview window, or does nothing at all when headless."""

    def __init__(
        self,
        config: PixelSpotConfig,
        geometry: ResolvedGeometry,
        processors: Sequence[Processor] = (),
    ):
        self.geometry = geometry
        self.processors = list(processors)
        self.headless = config.runtime.headless
        self.window_name = config.runtime.window_name
        self.show_fps = config.runtime.show_fps
        self.blur_faces = config.privacy.blur_faces_in_output
        self._window_open = False
        self._scale = 1.0

    def _px(self, value: float) -> int:
        """A length authored for the reference frame, scaled to this one."""
        return max(1, int(round(value * self._scale)))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        frame,
        tracks: list[Track],
        metrics: dict[str, dict[str, Any]],
        fps: float | None = None,
    ):
        height, width = frame.shape[:2]
        self._scale = max(1.0, min(width, height) / _REFERENCE_SHORT_SIDE)

        if self.blur_faces:
            self._blur_heads(frame, tracks)

        self._draw_zones(frame)
        self._draw_lines(frame)
        self._draw_tracks(frame, tracks)
        self._draw_hud(frame, metrics, fps)
        return frame

    def _blur_heads(self, frame, tracks: list[Track]) -> None:
        height, width = frame.shape[:2]
        for track in tracks:
            if track.label.lower() != "person":
                continue
            x1, y1, x2, y2 = track.bbox
            head_height = max(1, int((y2 - y1) * _HEAD_FRACTION))
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y1 + head_height)
            if x2 <= x1 or y2 <= y1:
                continue
            region = frame[y1:y2, x1:x2]
            # Kernel scaled to the region so a distant face is blurred as
            # thoroughly as a close one.
            kernel = max(3, (min(region.shape[:2]) // 2) | 1)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)

    def _draw_zones(self, frame) -> None:
        for zone in self.geometry.zones.values():
            points = zone.polygon()
            for index in range(len(points)):
                cv2.line(
                    frame, points[index], points[(index + 1) % len(points)],
                    ZONE_COLOUR, self._px(2),
                )
            label_at = min(points, key=lambda point: (point[1], point[0]))
            cv2.putText(
                frame, zone.id,
                (label_at[0], max(label_at[1] - self._px(8), self._px(16))),
                FONT, 0.5 * self._scale, ZONE_COLOUR, self._px(1),
            )

    def _draw_lines(self, frame) -> None:
        for line in self.geometry.lines.values():
            p1 = (int(line.p1[0]), int(line.p1[1]))
            p2 = (int(line.p2[0]), int(line.p2[1]))
            cv2.line(frame, p1, p2, LINE_COLOUR, self._px(3))

            # Arrow showing which direction counts as positive.
            start, end = line.positive_arrow(length_px=self._px(40))
            cv2.arrowedLine(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                LINE_COLOUR, self._px(2), tipLength=0.35,
            )
            cv2.putText(
                frame, line.id,
                (int(start[0]) + self._px(8), int(start[1]) - self._px(8)),
                FONT, 0.6 * self._scale, TEXT_COLOUR, self._px(2),
            )

    def _draw_tracks(self, frame, tracks: list[Track]) -> None:
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            colour = CLASS_COLOURS.get(track.label.lower(), DEFAULT_COLOUR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self._px(2))
            cv2.putText(
                frame, f"{track.label} #{track.id}",
                (x1, max(y1 - self._px(10), self._px(20))),
                FONT, 0.5 * self._scale, TEXT_COLOUR, self._px(2),
            )

    def _draw_hud(
        self, frame, metrics: dict[str, dict[str, Any]], fps: float | None
    ) -> None:
        lines: list[str] = []
        if self.show_fps and fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        for processor in self.processors:
            lines.extend(processor.overlay_lines(metrics.get(processor.name, {})))

        if not lines:
            return

        # Dark panel behind the text so it stays readable over a bright scene.
        line_height = self._px(35)
        panel_height = self._px(20) + line_height * len(lines)
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (self._px(10), self._px(10)),
            (self._px(330), panel_height), (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for index, text in enumerate(lines):
            cv2.putText(
                frame, text, (self._px(20), self._px(45) + index * line_height),
                FONT, 0.7 * self._scale, TEXT_COLOUR, self._px(2),
            )

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def show(self, frame) -> bool:
        """Display a frame. Returns False when the operator asked to quit."""
        if self.headless:
            return True

        if not self._window_open:
            self._open_window(frame.shape[1], frame.shape[0])
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)  # q or Esc

    def _open_window(self, frame_width: int, frame_height: int) -> None:
        """Open a resizable window no larger than the screen.

        The OpenCV default sizes the window to the frame's exact pixel
        count, so anything larger than the monitor -- a portrait 4K phone
        clip, say -- shows only the corner that fits. Scale to fit instead;
        the operator can still resize or maximise by hand afterwards.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        screen_width, screen_height = self._screen_bounds()
        scale = min(screen_width / frame_width, screen_height / frame_height, 1.0)
        cv2.resizeWindow(
            self.window_name, int(frame_width * scale), int(frame_height * scale)
        )
        self._window_open = True

    @staticmethod
    def _screen_bounds() -> tuple[int, int]:
        """Usable screen size, with margin for the taskbar and title bar."""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0) - 80, user32.GetSystemMetrics(1) - 120
        except (ImportError, AttributeError, OSError):  # not Windows
            return 1520, 780

    def close(self) -> None:
        if self._window_open:
            cv2.destroyAllWindows()
            self._window_open = False
