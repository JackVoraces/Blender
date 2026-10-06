"""Human expression signals -> robot expression controls.

Not a one-to-one copy: a human smile becomes cheek motion, an eye squint
feeds the cheeks too, and everything is exaggerated by per-term gains.  The
table itself is data (config/face_mapping.json); this module only evaluates it.
"""
from dataclasses import dataclass

_HUMAN_SIDE = {"L": "Left", "R": "Right"}
_OTHER = {"L": "R", "R": "L"}


_MIN_HEADROOM = 0.25     # never divide by less than this when normalising


@dataclass(frozen=True)
class Term:
    src: str
    gain: float
    deadzone: float
    baseline: bool       # subtract the calibrated neutral score
    normalize: bool      # rescale so "from neutral to the maximum score" spans 0..1


def deadzone(value, width):
    """Remove a dead band around zero without leaving a step at its edge."""
    if value > width:
        return value - width
    if value < -width:
        return value + width
    return 0.0


class FaceMapper:
    def __init__(self, config, mirror=True):
        self.mirror = bool(mirror)
        defaults = config.get("defaults", {})
        self.controls = {}     # control name -> (lo, hi, [Term])
        for template, spec in config["controls"].items():
            lo, hi = spec.get("clamp", [0.0, 1.0])
            sides = ("L", "R") if "{S}" in template else (None,)
            for side in sides:
                human = None
                if side is not None:
                    human = _OTHER[side] if self.mirror else side
                terms = []
                for term in spec["terms"]:
                    src = term["src"]
                    if "{Side}" in src:
                        if human is None:
                            raise ValueError(f"{template}: '{src}' needs a sided control")
                        src = src.replace("{Side}", _HUMAN_SIDE[human])
                    terms.append(Term(
                        src=src,
                        gain=float(term.get("gain", defaults.get("gain", 1.0))),
                        deadzone=float(term.get("deadzone", defaults.get("deadzone", 0.0))),
                        baseline=bool(term.get("baseline", defaults.get("baseline", True))),
                        normalize=bool(term.get("normalize", defaults.get("normalize", False))),
                    ))
                name = template.replace("{S}", side) if side else template
                self.controls[name] = (float(lo), float(hi), terms)

    @property
    def names(self):
        return list(self.controls)

    @property
    def sources(self):
        return sorted({term.src for _lo, _hi, terms in self.controls.values() for term in terms})

    def neutral(self):
        return {name: 0.0 for name in self.controls}

    def map(self, blendshapes, baseline=None):
        """Unsmoothed robot control targets for one frame of blendshape scores."""
        baseline = baseline or {}
        out = {}
        for name, (lo, hi, terms) in self.controls.items():
            total = 0.0
            for term in terms:
                score = blendshapes.get(term.src, 0.0)
                if term.baseline:
                    rest = baseline.get(term.src, 0.0)
                    score -= rest
                    if term.normalize:
                        # A face that rests at 0.4 only has 0.6 left to give; make that the full range.
                        score /= max(1.0 - rest, _MIN_HEADROOM)
                total += term.gain * deadzone(score, term.deadzone)
            out[name] = min(max(total, lo), hi)
        return out
