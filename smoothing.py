"""Signal smoothing.  Raw tracking never drives geometry directly."""


class Ema:
    """Per-key exponential moving average.

    ``alpha`` is the blend per frame at ``reference_fps``; the actual frame
    time is compensated for, so the feel does not change with the camera's
    frame rate.  Keys start at 0 (neutral).
    """

    def __init__(self, alpha, reference_fps=30.0, per_key=None):
        self.alpha = float(alpha)
        self.reference_fps = float(reference_fps)
        self.per_key = dict(per_key or {})
        self.state = {}

    def reset(self):
        self.state.clear()

    def _blend(self, alpha, dt):
        alpha = min(max(alpha, 0.0), 1.0)
        frames = min(max(dt, 0.0), 0.25) * self.reference_fps
        return 1.0 - (1.0 - alpha) ** frames

    def update(self, targets, dt, alpha=None):
        """Move every key towards its target; returns the smoothed values.

        ``alpha`` overrides the configured value for every key (used for the
        slow glide back to neutral).
        """
        out = {}
        for key, target in targets.items():
            a = alpha if alpha is not None else self.per_key.get(key, self.alpha)
            previous = self.state.get(key, 0.0)
            value = previous + self._blend(a, dt) * (target - previous)
            self.state[key] = value
            out[key] = value
        return out
