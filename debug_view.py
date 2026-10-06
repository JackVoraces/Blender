"""Tuning window: camera image, landmarks, FPS, tracking and calibration state,
and a bar for every robot control."""
import math

import cv2
import numpy as np

WINDOW = "Robot Narrator tracker"
_PANEL_W = 330
_ROW_H = 26
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (235, 235, 235)
_DIM = (150, 150, 150)
_GOLD = (40, 190, 245)        # BGR
_BLUE = (230, 160, 70)
_GREEN = (90, 210, 110)
_RED = (70, 70, 235)


class DebugView:
    def __init__(self, ranges, head_limits_deg, max_width=960, draw_landmarks=True, window=True):
        """ranges: {CONTROL: (lo, hi)}.  ``window=False`` only composes images (for tests)."""
        self.ranges = ranges
        self.head_limits = head_limits_deg
        self.max_width = int(max_width)
        self.draw_landmarks = draw_landmarks
        self._window = window
        if window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def _camera_image(self, sample, mirror):
        frame = sample.frame
        if frame is None:
            image = _blank(360, 640)
            cv2.putText(image, "no camera image (synthetic source)", (24, 190), _FONT, 0.7, _DIM, 1, cv2.LINE_AA)
            return image
        scale = min(1.0, self.max_width / frame.shape[1])
        if scale < 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            frame = frame.copy()
        if self.draw_landmarks and sample.landmarks is not None:
            h, w = frame.shape[:2]
            for point in sample.landmarks:
                cv2.circle(frame, (int(point.x * w), int(point.y * h)), 1, _GREEN, -1)
        return cv2.flip(frame, 1) if mirror else frame

    def _bar(self, canvas, x, y, label, value, lo, hi, colour, text):
        cv2.putText(canvas, label, (x, y + 15), _FONT, 0.42, _WHITE, 1, cv2.LINE_AA)
        bx, bw = x + 150, _PANEL_W - 150 - 62
        cv2.rectangle(canvas, (bx, y + 4), (bx + bw, y + 18), (70, 70, 70), 1)
        zero = bx + int(bw * (0.0 - lo) / (hi - lo))
        pos = bx + int(bw * (min(max(value, lo), hi) - lo) / (hi - lo))
        cv2.rectangle(canvas, (min(zero, pos), y + 5), (max(zero, pos), y + 17), colour, -1)
        cv2.line(canvas, (zero, y + 2), (zero, y + 20), _DIM, 1)
        cv2.putText(canvas, text, (bx + bw + 8, y + 15), _FONT, 0.42, _DIM, 1, cv2.LINE_AA)

    def show(self, sample, output, fps, mirror, status_lines=()):
        """Draw one frame and return the pressed key (255 when none)."""
        cv2.imshow(WINDOW, self.compose(sample, output, fps, mirror, status_lines))
        return cv2.waitKey(1) & 0xFF

    def compose(self, sample, output, fps, mirror, status_lines=()):
        """The window contents as an image."""
        image = self._camera_image(sample, mirror)
        rows = len(self.ranges) + 5
        height = max(image.shape[0], 70 + rows * _ROW_H + 20 * len(status_lines))
        canvas = _blank(height, image.shape[1] + _PANEL_W)
        canvas[:image.shape[0], :image.shape[1]] = image

        state, colour = ("TRACKING", _GREEN) if output.tracking else ("NO FACE", _RED)
        cv2.rectangle(canvas, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, f"{fps:4.1f} fps", (10, 21), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, state, (120, 21), _FONT, 0.6, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, "mirror on" if mirror else "mirror off", (250, 21), _FONT, 0.5, _DIM, 1, cv2.LINE_AA)

        if output.calibrating:
            cx, cy = image.shape[1] // 2, image.shape[0] // 2
            cv2.rectangle(canvas, (cx - 230, cy - 46), (cx + 230, cy + 40), (0, 0, 0), -1)
            cv2.putText(canvas, "Relax your face: calibrating neutral", (cx - 212, cy - 12), _FONT, 0.68, _GOLD, 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (cx - 200, cy + 8), (cx + 200, cy + 24), _DIM, 1)
            cv2.rectangle(canvas, (cx - 200, cy + 8),
                          (cx - 200 + int(400 * output.calibration_progress), cy + 24), _GOLD, -1)

        x = image.shape[1] + 12
        y = 10
        cv2.putText(canvas, "robot controls", (x, y + 14), _FONT, 0.5, _GOLD, 1, cv2.LINE_AA)
        y += _ROW_H
        for name, (lo, hi) in self.ranges.items():
            value = output.face.get(name, 0.0)
            self._bar(canvas, x, y, name, value, lo, hi, _GOLD, f"{value:+.2f}")
            y += _ROW_H
        y += 6
        cv2.putText(canvas, "head (degrees)", (x, y + 14), _FONT, 0.5, _BLUE, 1, cv2.LINE_AA)
        y += _ROW_H
        for axis, angle in zip(("yaw", "pitch", "roll"), output.head):
            limit = self.head_limits[axis]
            degrees = math.degrees(angle)
            self._bar(canvas, x, y, axis, degrees, -limit, limit, _BLUE, f"{degrees:+.0f}")
            y += _ROW_H
        for i, line in enumerate(reversed(status_lines)):
            cv2.putText(canvas, line, (x, canvas.shape[0] - 36 - i * 20), _FONT, 0.42, _DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, "C calibrate  N next camera  M mirror  L landmarks  Q quit",
                    (x, canvas.shape[0] - 12), _FONT, 0.38, _DIM, 1, cv2.LINE_AA)
        return canvas

    def close(self):
        if self._window:
            cv2.destroyWindow(WINDOW)


def _blank(height, width):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 24, 24)
    return canvas
