"""Neutral-face calibration.

For a couple of seconds the performer holds a relaxed face; the median of
every signal over that window becomes its neutral offset.  The median makes
a stray blink during calibration harmless.
"""
from statistics import median

MIN_SAMPLES = 8


class NeutralCalibration:
    def __init__(self, duration_s=2.0):
        self.duration = float(duration_s)
        self.baseline = {}                    # blendshape name -> neutral score
        self.head_baseline = (0.0, 0.0, 0.0)  # yaw, pitch, roll at rest
        self.completed = 0                    # how many calibrations have finished
        self._start = None
        self._shapes = {}
        self._head = ([], [], [])

    @property
    def active(self):
        return self._start is not None

    def start(self, now):
        self._start = now
        self._shapes = {}
        self._head = ([], [], [])

    def progress(self, now):
        if not self.active:
            return 1.0
        return min(max((now - self._start) / self.duration, 0.0), 1.0) if self.duration > 0 else 1.0

    def add(self, now, blendshapes, head):
        """Feed one tracked frame.  Returns True on the frame that completes the calibration."""
        if not self.active:
            return False
        for name, score in blendshapes.items():
            self._shapes.setdefault(name, []).append(score)
        if head is not None:
            for series, angle in zip(self._head, head):
                series.append(angle)
        count = max((len(v) for v in self._shapes.values()), default=0)
        if now - self._start < self.duration or count < MIN_SAMPLES:
            return False
        self.baseline = {name: median(scores) for name, scores in self._shapes.items()}
        if all(self._head):
            self.head_baseline = tuple(median(series) for series in self._head)
        self._start = None
        self.completed += 1
        return True
