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
                    ZONE_COLOUR, 2,
                )
            label_at = min(points, key=lambda point: (point[1], point[0]))
            cv2.putText(
                frame, zone.id, (label_at[0], max(label_at[1] - 8, 16)),
                FONT, 0.5, ZONE_COLOUR, 1,
            )

    def _draw_lines(self, frame) -> None:
        for line in self.geometry.lines.values():
            p1 = (int(line.p1[0]), int(line.p1[1]))
            p2 = (int(line.p2[0]), int(line.p2[1]))
            cv2.line(frame, p1, p2, LINE_COLOUR, 3)

            # Arrow showing which direction counts as positive.
            start, end = line.positive_arrow()
            cv2.arrowedLine(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                LINE_COLOUR, 2, tipLength=0.35,
            )
            cv2.putText(
                frame, line.id, (int(start[0]) + 8, int(start[1]) - 8),
                FONT, 0.6, TEXT_COLOUR, 2,
            )

    def _draw_tracks(self, frame, tracks: list[Track]) -> None:
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            colour = CLASS_COLOURS.get(track.label.lower(), DEFAULT_COLOUR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame, f"{track.label} #{track.id}", (x1, max(y1 - 10, 20)),
                FONT, 0.5, TEXT_COLOUR, 2,
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
        panel_height = 20 + 35 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for index, text in enumerate(lines):
            cv2.putText(
                frame, text, (20, 45 + index * 35), FONT, 0.7, TEXT_COLOUR, 2
            )

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def show(self, frame) -> bool:
        """Display a frame. Returns False when the operator asked to quit."""
        if self.headless:
            return True

        cv2.imshow(self.window_name, frame)
        self._window_open = True
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)  # q or Esc

    def close(self) -> None:
        if self._window_open:
            cv2.destroyAllWindows()
            self._window_open = False
