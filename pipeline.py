"""The signal chain between a source and the network:

    sample -> neutral calibration -> human-to-robot mapping (baseline, deadzone,
    gain, clamp) -> smoothing -> robot control values

plus head angles (baseline, mirror, gain, limits, smoothing) and graceful
behaviour when the face is lost.  No camera, MediaPipe or socket code here.
"""
import math
from dataclasses import dataclass, field

from . import head_pose
from .calibration import NeutralCalibration
from .face_mapping import FaceMapper
from .smoothing import Ema

_HEAD_AXES = ("yaw", "pitch", "roll")


@dataclass
class Output:
    time: float
    tracking: bool
    calibrating: bool
    calibration_progress: float
    face: dict                      # CONTROL -> smoothed value
    head: tuple                     # (yaw, pitch, roll) radians, avatar's own left/right
    targets: dict = field(default_factory=dict)   # CONTROL -> value before smoothing


class Pipeline:
    def __init__(self, settings, mapping):
        self.settings = settings
        self._mapping = mapping
        self.mirror = bool(settings["camera"]["mirror"])
        self.mapper = FaceMapper(mapping, mirror=self.mirror)
        smoothing = settings["smoothing"]
        self._fps = float(smoothing["reference_fps"])
        self.face_filter = Ema(smoothing["alpha"], self._fps, smoothing.get("per_control"))
        self.head_filter = Ema(settings["head"]["alpha"], self._fps)
        self.calibration = NeutralCalibration(settings["calibration"]["duration_s"])
        self._pending_calibration = bool(settings["calibration"]["auto_start"])
        self._hold = float(settings["tracking_loss"]["hold_s"])
        self._return_alpha = float(settings["tracking_loss"]["return_alpha"])
        self._last_time = None
        self._last_seen = None
        self._held = None

    # -- performer controls -------------------------------------------------

    def recalibrate(self):
        """Start a new neutral calibration on the next tracked frame."""
        self._pending_calibration = True

    def set_mirror(self, mirror):
        self.mirror = bool(mirror)
        self.mapper = FaceMapper(self._mapping, mirror=self.mirror)

    # -- per frame ----------------------------------------------------------

    def _head_targets(self, head):
        if head is None:
            return {axis: 0.0 for axis in _HEAD_AXES}
        angles = tuple(a - b for a, b in zip(head, self.calibration.head_baseline))
        if self.mirror:
            angles = head_pose.mirror_angles(angles)
        cfg = self.settings["head"]
        out = {}
        for axis, angle in zip(_HEAD_AXES, angles):
            limit = math.radians(cfg["limit_deg"][axis])
            out[axis] = min(max(angle * cfg["gain"][axis], -limit), limit)
        return out

    def process(self, sample):
        now = sample.time
        dt = 1.0 / self._fps if self._last_time is None else now - self._last_time
        self._last_time = now

        if sample.tracking:
            self._last_seen = now
            if self._pending_calibration:
                self._pending_calibration = False
                self.calibration.start(now)
            if self.calibration.active:
                self.calibration.add(now, sample.blendshapes, sample.head)
        calibrating = self.calibration.active or self._pending_calibration

        alpha = None
        if calibrating:
            # rest while the neutral face is being measured
            targets = self.mapper.neutral()
            head_targets = {axis: 0.0 for axis in _HEAD_AXES}
            self._held = None
        elif sample.tracking:
            targets = self.mapper.map(sample.blendshapes, self.calibration.baseline)
            head_targets = self._head_targets(sample.head)
            self._held = (targets, head_targets)
        elif self._held is not None and now - self._last_seen <= self._hold:
            targets, head_targets = self._held          # brief dropout: hold the pose
        else:
            targets = self.mapper.neutral()             # face gone: glide home, never snap
            head_targets = {axis: 0.0 for axis in _HEAD_AXES}
            alpha = self._return_alpha

        face = self.face_filter.update(targets, dt, alpha)
        head = self.head_filter.update(head_targets, dt, alpha)
        return Output(
            time=now,
            tracking=sample.tracking,
            calibrating=calibrating,
            calibration_progress=self.calibration.progress(now) if self.calibration.active else (0.0 if calibrating else 1.0),
            face=face,
            head=tuple(head[axis] for axis in _HEAD_AXES),
            targets=targets,
        )
