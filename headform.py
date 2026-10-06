"""The head's base form: the one surface every plate is laid on.

Pure Python (no Blender), so the tracing tools and the Blender build share it.
It is never rendered itself.

Head space, metres: origin at the head pivot (base of the skull), +X the
character's left, -Y the face, +Z up.

The form is angular, like the drawings: seen from above, the front of the head
is a flat face, then two flat planes swept back each side (the planes the eyes,
brows and cheeks lie in), then the side of the head.  So every height has a
section made of straight runs:

    centre line --flat-- (c) --first plane-- (x2) --second plane-- (x3) --side-- (w)

and the back of the head, behind the ears, is a smooth dome.  Plates that lie in
the same run of that section are exactly coplanar, which is what lets flat plates
meet edge to edge.

A point of the surface is addressed either by where it appears in a view of the
drawings (front (x, z), side (y, z), back (x, z), top (x, y)), or on the *chart*:
(theta, z), theta the direction of the point from the head's axis, 0 the middle of
the face, +90 degrees the character's left side, 180 the back of the head.  The
chart is the head unwrapped, and it is where plates traced from different views
are fitted together.
"""
import math
from bisect import bisect_right
from functools import lru_cache

# z, then the section at that height (all metres):
#   F       depth of the front face (the centre line is at y = -F)
#   c       half-width of the flat front
#   x2, d2  where the first swept plane ends: x, and how far behind the front
#   x3, d3  where the second ends
#   w       half-width of the head (at y = 0)
#   b, nb   depth and squareness of the back (2 = ellipse)
#
# Measured, not designed, wherever the drawings allow.  Depths of the front are the side
# view's profile (less the relief of the plate on the centre line there); widths are the
# front view's silhouette; the swept planes come from plates seen in both views (the x of a
# plate's edge in the front view against the y of the same edge in the side view).
# From the chin to the eyes the flat front is the faceplate's own face: as wide as the plate less
# the strip down each side that the drawing shows bent back, and as far forward as the side
# view's profile.  The eye line is set back from it, between that and the forehead.
# The swept planes are kept shallower than the side view alone would have them (it shows the
# face as a sharper prow than the front and three-quarter views do): the head reads broad.
PROFILE = (
    # z        F       c       x2      d2      x3      d3      w       b       nb
    (-0.0850, 0.0600, 0.0100, 0.0200, 0.0180, 0.0245, 0.0400, 0.0260, 0.0280, 2.00),     # under the chin
    (-0.0712, 0.0927, 0.0180, 0.0340, 0.0330, 0.0365, 0.0640, 0.0380, 0.0330, 2.00),     # chin: the foot of the faceplate
    (-0.0560, 0.0942, 0.0257, 0.0410, 0.0350, 0.0425, 0.0670, 0.0440, 0.0380, 2.00),
    (-0.0440, 0.0960, 0.0287, 0.0510, 0.0390, 0.0530, 0.0710, 0.0545, 0.0420, 2.00),
    (-0.0330, 0.0976, 0.0305, 0.0615, 0.0430, 0.0645, 0.0730, 0.0660, 0.0460, 2.00),     # jaw angle
    (-0.0180, 0.0996, 0.0354, 0.0670, 0.0440, 0.0700, 0.0760, 0.0715, 0.0510, 2.00),
    (0.0000, 0.1017, 0.0389, 0.0700, 0.0440, 0.0725, 0.0770, 0.0735, 0.0590, 2.00),
    (0.0100, 0.1027, 0.0437, 0.0705, 0.0440, 0.0735, 0.0790, 0.0750, 0.0670, 2.00),
    (0.0230, 0.1034, 0.0455, 0.0710, 0.0440, 0.0750, 0.0800, 0.0770, 0.0740, 2.00),     # top corners of the faceplate
    (0.0320, 0.1030, 0.0300, 0.0660, 0.0400, 0.0748, 0.0740, 0.0775, 0.0780, 2.00),     # the sill under the eyes
    (0.0420, 0.0965, 0.0130, 0.0590, 0.0320, 0.0730, 0.0580, 0.0779, 0.0825, 2.00),
    (0.0500, 0.0930, 0.0080, 0.0570, 0.0280, 0.0720, 0.0520, 0.0780, 0.0850, 2.00),     # eye line: set back
    (0.0700, 0.0935, 0.0110, 0.0575, 0.0290, 0.0710, 0.0530, 0.0780, 0.0895, 2.05),
    (0.0850, 0.0945, 0.0140, 0.0580, 0.0300, 0.0700, 0.0540, 0.0765, 0.0905, 2.10),     # brow
    (0.0930, 0.0962, 0.0170, 0.0580, 0.0300, 0.0700, 0.0570, 0.0755, 0.0907, 2.10),
    (0.1000, 0.0990, 0.0200, 0.0580, 0.0310, 0.0700, 0.0620, 0.0745, 0.0907, 2.10),     # foot of the forehead
    (0.1120, 0.0964, 0.0200, 0.0575, 0.0310, 0.0700, 0.0630, 0.0738, 0.0897, 2.10),
    (0.1260, 0.0930, 0.0200, 0.0570, 0.0300, 0.0690, 0.0620, 0.0730, 0.0870, 2.10),
    (0.1400, 0.0886, 0.0180, 0.0540, 0.0280, 0.0650, 0.0590, 0.0715, 0.0815, 2.10),
    (0.1530, 0.0808, 0.0150, 0.0460, 0.0230, 0.0610, 0.0530, 0.0685, 0.0745, 2.10),
    (0.1650, 0.0712, 0.0140, 0.0420, 0.0220, 0.0575, 0.0480, 0.0630, 0.0660, 2.05),     # cranium rounding over
    (0.1750, 0.0590, 0.0130, 0.0350, 0.0180, 0.0470, 0.0370, 0.0525, 0.0540, 2.00),
    (0.1830, 0.0445, 0.0110, 0.0265, 0.0130, 0.0350, 0.0270, 0.0400, 0.0405, 2.00),
    (0.1890, 0.0270, 0.0075, 0.0160, 0.0075, 0.0210, 0.0160, 0.0240, 0.0240, 2.00),
    (0.1925, 0.0045, 0.0012, 0.0027, 0.0012, 0.0035, 0.0026, 0.0040, 0.0040, 2.00),     # crown
)
EDGE = 0.985        # view points are never taken closer to the silhouette than this fraction


class Pchip:
    """Monotone cubic interpolation (Fritsch-Carlson): smooth, and never overshoots the data."""

    def __init__(self, xs, ys):
        self.xs, self.ys = list(xs), list(ys)
        n = len(self.xs)
        h = [self.xs[i + 1] - self.xs[i] for i in range(n - 1)]
        d = [(self.ys[i + 1] - self.ys[i]) / h[i] for i in range(n - 1)]
        m = [0.0] * n
        m[0], m[-1] = d[0], d[-1]
        for i in range(1, n - 1):
            if d[i - 1] * d[i] > 0.0:
                w1, w2 = 2.0 * h[i] + h[i - 1], h[i] + 2.0 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        self.h, self.m = h, m

    def __call__(self, x):
        xs = self.xs
        x = min(max(x, xs[0]), xs[-1])
        i = min(max(bisect_right(xs, x) - 1, 0), len(xs) - 2)
        t = (x - xs[i]) / self.h[i]
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * self.ys[i] + (t3 - 2 * t2 + t) * self.h[i] * self.m[i]
                + (-2 * t3 + 3 * t2) * self.ys[i + 1] + (t3 - t2) * self.h[i] * self.m[i + 1])


class HeadForm:
    """Points are (x, y, z) tuples."""

    def __init__(self, profile=PROFILE):
        zs = [row[0] for row in profile]
        self._columns = [Pchip(zs, [row[i] for row in profile]) for i in range(1, len(profile[0]))]
        self.z_min, self.z_max = zs[0], zs[-1]
        self._section = lru_cache(maxsize=8192)(self._build_section)

    def _build_section(self, z):
        """The section at one height: (corners of the front [(x, y)], their directions, w, b, nb)."""
        front, c, x2, d2, x3, d3, w, b, nb = (column(z) for column in self._columns)
        corners = ((0.0, -front), (c, -front), (x2, d2 - front), (x3, d3 - front), (w, 0.0))
        return corners, tuple(math.atan2(x, -y) for x, y in corners), w, b, nb

    def section(self, z):
        return self._section(round(min(max(z, self.z_min), self.z_max), 6))

    def half_width(self, z):
        return self.section(z)[2]

    def depth(self, z, back=False):
        corners, _, _, b, _ = self.section(z)
        return b if back else -corners[0][1]

    # ---- the point under a position in one of the views ------------------------------------
    def front_point(self, x, z):
        corners, _, w, _, _ = self.section(z)
        a = min(abs(x), w * EDGE)
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            if a <= x1:
                y = y0 + (y1 - y0) * (a - x0) / max(x1 - x0, 1e-9)
                return (x, y, z)
        return (x, corners[-1][1], z)

    def back_point(self, x, z):
        _, _, w, b, nb = self.section(z)
        ratio = min(abs(x) / w, EDGE)
        return (x, b * (1.0 - ratio ** nb) ** (1.0 / nb), z)

    def side_point(self, y, z):
        """On the character's left side."""
        corners, _, w, b, nb = self.section(z)
        if y >= 0.0:
            ratio = min(y / b, EDGE)
            return (w * (1.0 - ratio ** nb) ** (1.0 / nb), y, z)
        for (x0, y0), (x1, y1) in zip(corners[1:], corners[2:]):
            if y <= y1:
                t = min(max((y - y0) / max(y1 - y0, 1e-9), 0.0), 1.0)
                return (x0 + (x1 - x0) * t, y, z)
        return (w, y, z)

    def top_point(self, x, y, z_from=0.14):
        """The surface seen from above: the height at which the section passes through (x, y)."""
        def outside(z):
            corners = self.section(z)[0]
            if y < corners[0][1]:
                return 1.0
            return abs(x) - self.side_point(y, z)[0]

        lo, hi = z_from, self.z_max
        if outside(lo) >= 0.0:
            return (x, y, lo)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if outside(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return (x, y, lo)

    def view_point(self, view, u, v):
        if view == "front":
            return self.front_point(u, v)
        if view == "side":
            return self.side_point(u, v)
        if view == "back":
            return self.back_point(u, v)
        return self.top_point(u, v)

    # ---- the chart: round the head ---------------------------------------------------------
    def point(self, theta, z):
        """Surface point in the direction ``theta`` (radians) from the axis, at height ``z``."""
        theta = math.atan2(math.sin(theta), math.cos(theta))
        a = abs(theta)
        corners, directions, w, b, nb = self.section(z)
        s, c = math.sin(a), math.cos(a)
        if a <= 0.5 * math.pi:
            r = 0.0
            for i in range(4):
                if a <= directions[i + 1] + 1e-12 or i == 3:
                    (x0, y0), (x1, y1) = corners[i], corners[i + 1]
                    ex, ey = x1 - x0, y1 - y0
                    r = (x0 * ey - y0 * ex) / (s * ey + c * ex)
                    break
        else:
            r = ((s / w) ** nb + (-c / b) ** nb) ** (-1.0 / nb)
        return (math.copysign(r * s, theta), -r * c, z)

    def theta(self, x, y, z=0.0):
        """Chart angle of a point, -pi..pi."""
        return math.atan2(x, -y)

    # ---- the runs of the section: where the form creases ------------------------------------
    # A second way round the head that follows its planes: ``t`` runs 0..1 across the flat
    # front, 1..2 over the first swept plane, 2..3 over the second, 3..4 along the side to the
    # ear, then 4..6 round the back to the middle of it (and on, to 8 at the other ear).
    # Negative on the character's right.  Whole numbers up to 3 are the creases of the form.
    def run_point(self, t, z):
        a = abs(t)
        corners, _, w, b, nb = self.section(z)
        if a <= 4.0:
            k = min(int(a), 3)
            (x0, y0), (x1, y1) = corners[k], corners[k + 1]
            f = a - k
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        else:
            angle = 0.5 * math.pi + (a - 4.0) * 0.25 * math.pi
            s, c = math.sin(angle), math.cos(angle)
            r = (abs(s / w) ** nb + abs(c / b) ** nb) ** (-1.0 / nb)
            x, y = r * s, -r * c           # (past 6 this is the character's right: x < 0)
        return (-x if t < 0.0 else x, y, z)

    def run_of(self, x, y, z, around=False):
        """The ``t`` of a surface point.  ``around``: a point of the back of the head on the
        character's right is given as 6..8 (carrying on round) rather than as -6..-4, so that a
        plate across the middle of the back is in one piece."""
        corners, directions, _, _, _ = self.section(z)
        angle = math.atan2(abs(x), -y)
        if angle <= 0.5 * math.pi:
            for k in range(4):
                if angle <= directions[k + 1] + 1e-12 or k == 3:
                    (x0, y0), (x1, y1) = corners[k], corners[k + 1]
                    ex, ey = x1 - x0, y1 - y0
                    f = ((abs(x) - x0) * ex + (y - y0) * ey) / max(ex * ex + ey * ey, 1e-18)
                    t = k + min(max(f, 0.0), 1.0)
                    break
        else:
            t = 4.0 + (angle - 0.5 * math.pi) / (0.25 * math.pi)
        if x < 0.0:
            t = 12.0 - t if around and t >= 4.0 else -t
        return t

    def run_normal(self, t, z, step=1e-3):
        """Unit outward normal at a point given by its run.  On a crease it is the mean of the
        two planes that meet there."""
        dt = 0.02
        a, b = self.run_point(t - dt, z), self.run_point(t + dt, z)
        lo, hi = max(z - step, self.z_min), min(z + step, self.z_max)
        c, d = self.run_point(t, lo), self.run_point(t, hi)
        across = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        up = (d[0] - c[0], d[1] - c[1], d[2] - c[2])
        n = (across[1] * up[2] - across[2] * up[1], across[2] * up[0] - across[0] * up[2],
             across[0] * up[1] - across[1] * up[0])
        length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        n = (n[0] / length, n[1] / length, n[2] / length)
        p = self.run_point(t, z)
        if n[0] * p[0] + n[1] * p[1] < 0.0:
            n = (-n[0], -n[1], -n[2])
        return n

    def normal(self, theta, z, step=1e-3):
        """Unit outward normal at a chart point."""
        a, b = self.point(theta - step, z), self.point(theta + step, z)
        lo, hi = max(z - step, self.z_min), min(z + step, self.z_max)
        c, d = self.point(theta, lo), self.point(theta, hi)
        t = (b[0] - a[0], b[1] - a[1], b[2] - a[2])         # round the head, towards +theta
        u = (d[0] - c[0], d[1] - c[1], d[2] - c[2])         # up
        n = (t[1] * u[2] - t[2] * u[1], t[2] * u[0] - t[0] * u[2], t[0] * u[1] - t[1] * u[0])
        length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        n = (n[0] / length, n[1] / length, n[2] / length)
        p = self.point(theta, z)
        if n[0] * p[0] + n[1] * p[1] < 0.0:                 # outwards
            n = (-n[0], -n[1], -n[2])
        return n

    def inset_point(self, theta, z, depth):
        """A point about ``depth`` inside the surface: the same section, shrunk towards the axis.

        Sections are never shrunk past half their size.
        """
        x, y, _ = self.point(theta, z)
        r = math.hypot(x, y) or 1.0
        scale = max(0.5, 1.0 - depth / r)
        return (x * scale, y * scale, z)
