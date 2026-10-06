"""Turn the traced panels of the reference drawings into the head's plate layout.

    python tools/trace_reference.py
    python tools/freeze_trace.py

Reads references/trace/<view>_regions.json and picks the panels listed in PLATES
(each by a point inside it; references/trace/<view>_regions.png shows them).
Every plate is taken from the view that shows it best:

    front   the anatomy sheet: everything that faces forwards
    side    the turnaround's side view: temples, jaw, the band behind the ear, ear, fin
    back    the turnaround's back view: the back of the skull

The plates are then fitted together on the *chart* of the head's base form
(rn_common/headform.py): the head unwrapped, angle round it against height.
There plates from different views that would collide are parted with a joint
(each view is believed where it looks at the surface most squarely), and each
plate is run on a little under its higher neighbours (its "under" lips), so
that nothing opens up when the upper plate moves.  A plate can also be told to
carry another whole: the brow bars lie on their housings, which run on behind
them.

Writes blender/scripts/rn_blender/head/traced.json, which is then the layout:
it can be edited by hand and does not depend on re-tracing.  Also writes
references/trace/chart_layout.png, the unwrapped head, to check the fit by eye.

Sided plates are authored as the character's LEFT and mirrored by the builder.
Centre-line plates are made exactly symmetric.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rn_common.headform import HeadForm          # noqa: E402
from trace_reference import straighten            # noqa: E402

TRACE = ROOT / "references" / "trace"
OUT = ROOT / "blender" / "scripts" / "rn_blender" / "head" / "traced.json"
FORM = HeadForm()

HEAD_TOP_Z = 0.188          # head-space height of the top of the crown plate
HEAD_HEIGHT = 0.2592        # crown plate top to the bottom edge of the mask, metres
MASK_FACE_Y = -0.1065       # depth of the top of the mask's face: places the side view front to back

# relief, thickness, chamfer (mm).  A plate occupies relief - thickness .. relief above the base
# form (or, for a part given a plane of its own, behind that plane).  L0-L2 and MASK are the
# shell: slabs lying on the planes of the form, a step apart so that neighbours read as separate
# plates.  The faceplate is one of them, no thicker than the rest.  A few parts stand proud of
# the shell: the keystone and the bridge of the nose (each on its own plane, their faces on the
# side view's profile), the block at the end of each brow, and the brow bars.
LAYERS = {
    "NOSE": (0.0, 5.0, 2.0),
    "KEY": (0.0, 12.0, 2.0),
    "BLOCK": (9.0, 10.5, 2.4),
    "BROW": (6.5, 2.7, 1.2),
    "MASK": (3.9, 2.4, 1.3),
    "L2": (2.6, 2.4, 1.3),
    "L1": (1.3, 2.4, 1.3),
    "L0": (0.0, 2.4, 1.3),
    "LIGHT": (-3.0, 1.2, 0.0),
    "FRAME": (-5.6, 1.5, 0.5),
}
SHELL = ("L0", "L1", "L2", "MASK")      # lie on the base form
BLOCKS = ("NOSE", "KEY", "BLOCK")       # solid down to the form: nothing passes under them
LIP_MM = (1.4, 1.0)         # an "under" lip: how far below its plate's face, and how thick
TUCK_MM = 6.5               # how far a plate reaches under a higher neighbour
LIP_INSET_MM = 2.2          # ...starting this far in from that neighbour's edge, so it never shows in the joint
JOINT_MM = 1.8              # the joint left between plates that had to be parted
EYE_FRAME_MM = 5.0          # the dark frame of each eye reaches this far beyond the bright core

# view -> plates: point inside the panel (image pixels), name, group, layer, kind, options
#   kind      "C" centre-line plate, "S" sided plate (the character's left)
#   options   under: plates this one carries whole          cut: [((x, y), (nx, ny))] keep that side
#             region: "eye" | "brow" trace kind      folds: ("u", [metres]) fold lines in its view
#             plane: (a, b, c) its face is y = a*x + b*z + c
#             taper: its face is this much narrower than its outline, the sides sloping back to it
#             hull: take the traced outline's convex hull (an annotation line cuts into the tracing)
#             fit_u: a flat part is fitted to the form only where its first view coordinate is below this
PLATES = {
    "front": [
        # centre line
        ((622, 150), "forehead_center_top", "FOREHEAD", "L0", "C", {"folds": ("u", [0.014, 0.038])}),
        ((620, 213), "forehead_center_upper", "FOREHEAD", "L1", "C", {"folds": ("u", [0.015])}),
        ((621, 313), "forehead_center_mid", "FOREHEAD", "L2", "C", {"folds": ("u", [0.020])}),
        ((621, 432), "forehead_center_lower", "FOREHEAD", "KEY", "C", {"plane": (0.0, 0.0, -0.1047)}),
        ((622, 542), "nose_bridge", "CENTRAL_FACE", "NOSE", "C",
         {"plane": (0.0, 0.0, -0.1048), "folds": ("u", [0.0075])}),
        # the faceplate: a flat shield lying on the front of the form, the strip down each side
        # bent back with it
        ((622, 738), "central_mask", "CENTRAL_FACE", "MASK", "C", {}),
        # forehead
        ((722, 218), "forehead_{S}_upper_inner", "FOREHEAD", "L0", "S", {}),
        ((773, 206), "forehead_{S}_upper", "FOREHEAD", "L1", "S", {}),
        ((759, 275), "forehead_{S}_outer", "FOREHEAD", "L0", "S", {}),
        ((744, 325), "forehead_{S}_lower", "FOREHEAD", "L1", "S", {}),
        ((702, 376), "forehead_{S}_wedge", "FOREHEAD", "L0", "S", {}),
        # brows: the gold housing runs on behind the brown bar, which lies on top of it
        ((744, 393), "brow_housing_{S}", "BROWS", "L2", "S", {"under": ["brow_{S}"]}),
        ((759, 438), "brow_{S}", "BROWS", "BROW", "S", {"region": "brow", "fit_u": 0.055}),
        # eyes
        ((780, 470), "upper_lid_{S}", "EYES", "L1", "S", {}),
        ((752, 537), "eye_light_{S}", "EYES", "LIGHT", "S", {"region": "eye"}),
        ((771, 583), "lower_lid_{S}", "EYES", "L1", "S", {}),
        # cheeks
        ((816, 636), "cheek_{S}", "CHEEKS", "L2", "S", {"hull": True}),
        ((764, 790), "cheek_aux_{S}", "CHEEKS", "L1", "S", {}),
    ],
    "side": [
        ((743, 294), "temple_{S}", "TEMPLES", "L1", "S", {}),
        ((735, 238), "crown_side_{S}", "FOREHEAD", "L0", "S", {}),
        ((730, 349), "brow_end_{S}", "BROWS", "BLOCK", "S", {}),
        ((727, 378), "eye_corner_{S}", "EYES", "L0", "S", {}),
        ((714, 461), "cheek_link_{S}", "CHEEKS", "L0", "S", {}),
        ((746, 446), "jaw_strip_{S}", "SIDE_FACE", "L0", "S", {}),
        ((741, 484), "jaw_{S}", "SIDE_FACE", "L1", "S", {}),
        ((817, 436), "ear_lower_{S}", "SIDE_FACE", "L1", "S", {}),
        # the long band behind the ear and the short plate above it are one traced region
        ((845, 345), "arc_{S}_upper", "SHELL", "L1", "S", {"cut": [((0, 295.5), (0, -1))]}),
        ((845, 345), "arc_{S}", "SHELL", "L0", "S", {"cut": [((0, 299.0), (0, 1))]}),
        ((863, 425), "arc_{S}_lower", "SHELL", "L1", "S", {}),
    ],
    "back": [
        ((1302, 199), "crown_rear", "SHELL", "L2", "C", {}),
        ((1301, 229), "occipital_upper", "SHELL", "L1", "C", {}),
        ((1301, 293), "occipital_mid", "SHELL", "L2", "C", {}),
        # one traced region holds this row's three plates; the joints run between these points
        ((1300, 366), "occipital_lower", "SHELL", "L1", "C",
         {"cut": [((1269.4, 336.0), (54.0, -5.75)), ((1331.1, 336.0), (-54.0, -6.5))]}),
        ((1300, 366), "occipital_{S}_lower", "SHELL", "L0", "S",
         {"cut": [((1266.6, 336.0), (-54.0, 5.75))]}),
        ((1300, 410), "occipital_base", "SHELL", "L2", "C", {}),
        ((1230, 245), "parietal_{S}_upper", "SHELL", "L0", "S", {}),
        ((1245, 311), "parietal_{S}", "SHELL", "L1", "S", {}),
    ],
}
# Landmarks that fix each view's scale and position.
CALIBRATION = {
    "front": {"crown": (622, 150), "mask": (622, 738)},
    "turn_front": {"crown": (152, 197), "mask": (150, 480)},
    "side": {"crown": (738, 208), "mask": (655, 459)},
    "back": {"crown": (1302, 199)},
}
FIN_SEED = (863, 321)                                    # front view
EAR_SEEDS = ((894, 488), (910, 504), (905, 565))         # front view
# The side view draws the fin and the ear ring as one inked shape and hides the back of the ring,
# so these are read off the drawing by hand (side-view pixels).
FIN_SIDE = [(835, 190), (813, 211), (797, 234), (780, 265), (768, 296), (759, 355), (806, 354), (836, 229)]
EAR_RING_OUTER = [(806, 345), (794, 336), (748.5, 368), (748.5, 411), (771, 444), (797, 438)]
EAR_RING_INNER = [(803, 357), (798, 351), (768, 376), (762, 407), (776, 435), (793, 433)]
EAR_HUB = [(800, 356), (770, 377), (764, 406), (778, 433), (797, 425), (803, 417), (808, 380), (804, 364)]
EAR_HUB_APEX = (790, 388)
EAR_COLLAR = {"centre": (784, 395), "outer": (44, 57), "inner": (21, 39)}       # centre, radii
EAR_KEEP_OUT = 0.90         # the shell stops this far out from the ear's centre, as a share of the collar

# The chart canvas: one pixel is RES metres of height, and RES of arc where the head is widest.
RES = 0.00025
ARC = 0.078                 # metres of arc a radian stands for
THETA = (-2.0, 5.3)         # the left of the head, its mirror image either side of it
Z_RANGE = (-0.090, 0.196)


def _brush(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _px(mm):
    return max(1, int(round(mm * 0.001 / RES)))


# ---- polygons ------------------------------------------------------------------------------------
def clip(polygon, keep):
    """Sutherland-Hodgman against one boundary; keep(p) -> signed distance, >= 0 is inside."""
    out = []
    for i, p in enumerate(polygon):
        q = polygon[(i + 1) % len(polygon)]
        dp, dq = keep(p), keep(q)
        if dp >= 0:
            out.append(list(p))
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t])
    return out


def cut(polygon, point, normal):
    """The part of a polygon on the side of the line through ``point`` that ``normal`` points to."""
    length = math.hypot(*normal)
    return clip(polygon, lambda p: ((p[0] - point[0]) * normal[0] + (p[1] - point[1]) * normal[1]) / length)


def symmetric(polygon, centre_x):
    """Right half of a centre-line outline plus its mirror image: an exactly symmetric polygon."""
    half = clip(polygon, lambda p: p[0] - centre_x)
    on_axis = [i for i, p in enumerate(half) if abs(p[0] - centre_x) < 1e-6]
    if len(on_axis) < 2:
        return polygon
    top = min(on_axis, key=lambda i: half[i][1])
    bottom = max(on_axis, key=lambda i: half[i][1])
    count = len(half)
    step = 1 if abs(half[(top + 1) % count][0] - centre_x) > 1e-6 else -1     # leave the axis
    chain, i = [], top
    while True:
        chain.append(half[i])
        if i == bottom:
            break
        i = (i + step) % count
    return chain + [[2.0 * centre_x - x, y] for x, y in reversed(chain[1:-1])]


def _contains(polygon, x, y):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def densify(polygon, step):
    """The same polygon with no edge longer than ``step``."""
    out = []
    for i, p in enumerate(polygon):
        q = polygon[(i + 1) % len(polygon)]
        pieces = max(1, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / step)))
        out += [(p[0] + (q[0] - p[0]) * k / pieces, p[1] + (q[1] - p[1]) * k / pieces) for k in range(pieces)]
    return out


# ---- the traced regions and the views ------------------------------------------------------------
def find_region(regions, seed, kind="panel"):
    """The traced panel a seed point belongs to: the smallest one containing it."""
    pool = [r for r in regions if r["kind"] == kind]
    holding = [r for r in pool if _contains(r["polygon"], *seed)]
    if holding:
        chosen = min(holding, key=lambda r: r["area"])
    else:
        chosen = min(pool, key=lambda r: math.hypot(r["centroid"][0] - seed[0], r["centroid"][1] - seed[1]))
        distance = math.hypot(chosen["centroid"][0] - seed[0], chosen["centroid"][1] - seed[1])
        if distance > 25:
            raise LookupError(f"no traced {kind} near {seed} (closest is {distance:.0f} px away)")
    if chosen["touches_crop_edge"] or not holding:
        print(f"  check seed {seed}: took region {chosen['id']} (area {chosen['area']}, "
              f"seed inside: {bool(holding)}, touches the crop edge: {chosen['touches_crop_edge']})")
    return chosen


def load(view):
    with open(TRACE / f"{view}_regions.json", encoding="utf-8") as fh:
        return json.load(fh)


def calibrate(traces):
    """Each view's pixel -> head-space mapping: u = sign * (px - ox) * scale, z = HEAD_TOP_Z - (py - oy) * scale."""
    views = {}
    for name in ("front", "turn_front", "side"):
        regions, marks = traces[name]["regions"], CALIBRATION[name]
        crown, mask = find_region(regions, marks["crown"]), find_region(regions, marks["mask"])
        top = min(p[1] for p in crown["polygon"])
        bottom = max(p[1] for p in mask["polygon"])
        views[name] = {"scale": HEAD_HEIGHT / (bottom - top), "oy": top, "sign": 1.0}
    for name in ("front", "turn_front"):
        centre = [spec for spec in PLATES["front"] if spec[4] == "C"] if name == "front" else None
        if centre:
            xs = [find_region(traces[name]["regions"], spec[0])["centroid"][0] for spec in centre]
        else:
            marks = CALIBRATION[name]
            xs = [find_region(traces[name]["regions"], marks[k])["centroid"][0] for k in ("crown", "mask")]
        views[name]["ox"] = sum(xs) / len(xs)
    # side: front to back from the face of the mask (the middle of its inked edge)
    mask = find_region(traces["side"]["regions"], CALIBRATION["side"]["mask"])
    scale = views["side"]["scale"]
    views["side"]["ox"] = min(p[0] for p in mask["polygon"]) - 1.3 - MASK_FACE_Y / scale
    # back: drawn at the size of the turnaround's front view; it has no chin to measure by
    regions = traces["back"]["regions"]
    xs = [find_region(regions, spec[0])["centroid"][0] for spec in PLATES["back"] if spec[4] == "C" and "cut" not in spec[5]]
    crown = find_region(regions, CALIBRATION["back"]["crown"])
    views["back"] = {"scale": views["turn_front"]["scale"], "oy": min(p[1] for p in crown["polygon"]),
                     "ox": sum(xs) / len(xs), "sign": -1.0}
    for name, view in views.items():
        view["image"], view["crop"] = traces[name]["image"], traces[name]["crop"]
    return views


def to_view(view, point):
    return [round(view["sign"] * (point[0] - view["ox"]) * view["scale"], 5),
            round(HEAD_TOP_Z - (point[1] - view["oy"]) * view["scale"], 5)]


# ---- the chart -----------------------------------------------------------------------------------
SHAPE = (int((Z_RANGE[1] - Z_RANGE[0]) / RES), int((THETA[1] - THETA[0]) * ARC / RES))


def chart_px(theta, z):
    return (theta - THETA[0]) * ARC / RES, (Z_RANGE[1] - z) / RES


def chart_at(x, y):
    return x * RES / ARC + THETA[0], Z_RANGE[1] - y * RES


def to_chart(view, outline):
    """A view outline carried onto the base form: [(theta, z)] with corners every millimetre or so."""
    out = []
    for u, v in densify(outline, 0.0012):
        x, y, z = FORM.view_point(view, u, v)
        theta = FORM.theta(x, y, z)
        if view == "back" and theta < 0.0:
            theta += 2.0 * math.pi          # the back of the head is one piece: 180 degrees, not +-180
        out.append((theta, z))
    return out


def mirrored(chart):
    """The same outline on the character's right."""
    middle = sum(t for t, _ in chart) / len(chart)
    return [((-t if middle < 0.5 * math.pi else 2.0 * math.pi - t), z) for t, z in reversed(chart)]


def mask_of(chart):
    out = np.zeros(SHAPE, np.uint8)
    cv2.fillPoly(out, [np.round(np.array([chart_px(t, z) for t, z in chart])).astype(np.int32)], 1)
    return out


def outline_of(mask, tolerance_mm=0.22):
    """Largest outline of a mask as a chart polygon [(theta, z)]."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, tolerance_mm * 0.001 / RES, True).reshape(-1, 2)
    return [[round(v, 6) for v in chart_at(float(x), float(y))] for x, y in polygon]


def in_view(view, chart, tidy=True):
    """A chart outline as its plate's view shows it: [(u, z)], its edges straightened."""
    out = []
    for theta, z in chart:
        x, y, _ = FORM.point(theta, z)
        out.append((y if view == "side" else x, z))
    if tidy:
        out = straighten(out, min_edge=0.002, max_shift=0.00045)
    return [[round(float(u), 5), round(float(v), 5)] for u, v in out]


def trust_maps():
    """How squarely each view looks at the base form, over the chart (0 = edge-on, 1 = face-on)."""
    step = 8
    rows, cols = SHAPE[0] // step + 2, SHAPE[1] // step + 2
    normals = np.zeros((rows, cols, 3), np.float32)
    for r in range(rows):
        _, z = chart_at(0, r * step)
        z = min(max(z, FORM.z_min + 0.002), FORM.z_max - 0.002)
        for c in range(cols):
            theta, _ = chart_at(c * step, 0)
            normals[r, c] = FORM.normal(theta, z)
    normals = cv2.resize(normals, (cols * step, rows * step), interpolation=cv2.INTER_LINEAR)[:SHAPE[0], :SHAPE[1]]
    return {"front": np.maximum(-normals[..., 1], 0.0), "side": np.abs(normals[..., 0]),
            "back": np.maximum(normals[..., 1], 0.0)}


def collect(traces, views):
    """Every plate with its outline(s) in view coordinates and on the chart."""
    entries = []
    for view_name, specs in PLATES.items():
        for seed, name, group, layer, kind, options in specs:
            sources = []
            for source_view, source_seed, cuts in [(view_name, seed, options.get("cut", ()))]:
                region = find_region(traces[source_view]["regions"], source_seed, options.get("region", "panel"))
                polygon = region["polygon"]
                for point, normal in cuts:
                    polygon = cut(polygon, point, normal)
                if options.get("hull"):
                    hull = cv2.convexHull(np.array(polygon, np.float32)).reshape(-1, 2)
                    polygon = [[float(x), float(y)] for x, y in straighten(hull)]
                if kind == "C":
                    polygon = symmetric(polygon, views[source_view]["ox"])
                outline = [to_view(views[source_view], p) for p in polygon]
                sources.append({"view": source_view, "outline": outline, "chart": to_chart(source_view, outline),
                                "rgb": region["rgb"]})
            entries.append({"name": name, "group": group, "layer": layer, "kind": kind, "view": view_name,
                            "options": options, "seed": seed, "rgb": sources[0]["rgb"], "sources": sources,
                            "outline": sources[0]["outline"]})
    # the ear: not a plate, but ground no shell plate may take
    cx, cy = EAR_COLLAR["centre"]
    rx, ry = (r * EAR_KEEP_OUT for r in EAR_COLLAR["outer"])
    ring = [to_view(views["side"], (cx + rx * math.cos(a), cy + ry * math.sin(a)))
            for a in (2.0 * math.pi * k / 48 for k in range(48))]
    entries.append({"name": "ear_{S}", "group": "EARS", "layer": "BLOCK", "kind": "S", "view": "side", "options": {},
                    "seed": (cx, cy), "rgb": [0, 0, 0], "outline": ring, "hidden": True,
                    "sources": [{"view": "side", "outline": ring, "chart": to_chart("side", ring), "rgb": [0, 0, 0]}]})
    return entries


def fit(entries):
    """Fit the plates together on the chart.  Sets entry["fitted"] (a shell plate's outline in its
    own view, after any parting from its neighbours) and entry["under"] (its lips, likewise);
    returns the masks, for the layout picture."""
    trust = trust_maps()
    everything = list(entries) + [{**e, "ghost": True} for e in entries if e["kind"] == "S"]
    masks, belief = [], []
    for entry in everything:
        best_mask, best_trust = None, None
        for source in entry["sources"]:
            chart = mirrored(source["chart"]) if entry.get("ghost") else source["chart"]
            mask, t = mask_of(chart), trust[source["view"]]
            if best_mask is None:
                best_mask, best_trust = mask, t
            else:                       # each view decides where it looks at the surface more squarely
                take = t > best_trust
                best_mask, best_trust = np.where(take, mask, best_mask), np.where(take, t, best_trust)
        masks.append(best_mask.astype(np.uint8))
        belief.append(best_trust)
    original = [m.copy() for m in masks]
    count = len(everything)
    shell = [e["layer"] in SHELL for e in everything]
    block = [e["layer"] in BLOCKS for e in everything]
    level = [LAYERS[e["layer"]][0] for e in everything]
    names = [(e["name"].replace("{S}", ""), bool(e.get("ghost"))) for e in everything]
    joint, near = _brush(_px(JOINT_MM)), _brush(_px(JOINT_MM + 0.8))

    # a plate that carries another runs on behind it, and across the joint between them
    for i, entry in enumerate(everything):
        for name in entry["options"].get("under", ()):
            j = names.index((name.replace("{S}", ""), bool(entry.get("ghost"))))
            masks[i] |= original[j] | (cv2.dilate(original[i], near) & cv2.dilate(original[j], near))

    # plates that cannot share ground: a block keeps its own; a plate of the front view (the
    # anatomy sheet, the first authority) keeps its own against the turnaround's; otherwise the
    # one whose view saw that ground more squarely keeps it.  Whoever gave way is then held a
    # joint's width off.
    parted = []
    for i in range(count):
        for j in range(i + 1, count):
            if not ((shell[i] and (shell[j] or block[j])) or (shell[j] and block[i])):
                continue
            both = masks[i] & masks[j]
            if int(both.sum()) < 4:
                continue
            if block[i] or block[j]:
                gives = (both, None) if block[j] else (None, both)
            else:
                sure_i, sure_j = (float(belief[k][both > 0].mean()) + (1.0 if everything[k]["view"] == "front" else 0.0)
                                  for k in (i, j))
                gives = (both, None) if sure_j > sure_i else (None, both)
            lost = [0, 0]
            for k, (index, taken) in enumerate(((i, gives[0]), (j, gives[1]))):
                if taken is not None:
                    masks[index] &= 1 - taken
                    lost[k] = int(taken.sum())
            parted.append((i, j) if lost[0] >= lost[1] else (j, i))
    for loser, winner in parted:
        masks[loser] &= 1 - cv2.dilate(masks[winner], joint)
    for i in range(count):
        if shell[i]:                    # tidy: no spikes, no crumbs
            masks[i] = cv2.morphologyEx(masks[i], cv2.MORPH_OPEN, _brush(_px(0.6)))
            parts, labels, stats, _ = cv2.connectedComponentsWithStats(masks[i], connectivity=4)
            if parts > 2:
                masks[i] = (labels == 1 + int(np.argmax(stats[1:, 4]))).astype(np.uint8)

    # lips: each shell plate reaches under its higher shell neighbours, but not into the joint
    reach = _brush(_px(TUCK_MM + 2.0 * JOINT_MM))
    solid = np.zeros(SHAPE, np.uint8)
    for i in range(count):
        if block[i]:
            solid |= cv2.dilate(masks[i], _brush(_px(0.8)))
    lips = []
    for i in range(count):
        over = [j for j in range(count) if shell[i] and shell[j] and level[j] > level[i] + 0.5]
        if not over:
            lips.append(None)
            continue
        higher = cv2.erode(np.maximum.reduce([masks[j] for j in over]), _brush(_px(LIP_INSET_MM)))
        lips.append(cv2.dilate(masks[i], reach) & higher & (1 - solid))
    distance = {}

    def far(i):
        if i not in distance:
            distance[i] = cv2.distanceTransform(1 - masks[i], cv2.DIST_L2, 5)
        return distance[i]

    for i in range(count):
        for j in range(i + 1, count):       # two lips on the same level split the ground between them
            if lips[i] is None or lips[j] is None or abs(level[i] - level[j]) > 0.5 or not (lips[i] & lips[j]).any():
                continue
            lead = far(i) - far(j)
            lips[i] = lips[i] & (lead < -0.5 * _px(JOINT_MM)).astype(np.uint8)
            lips[j] = lips[j] & (lead > 0.5 * _px(JOINT_MM)).astype(np.uint8)

    for i, entry in enumerate(entries):
        entry["under"] = []
        if not shell[i]:
            continue
        changed = int((masks[i] ^ original[i]).sum())
        if changed < _px(1.0) ** 2:
            entry["fitted"] = entry["outline"]                              # exactly as traced
        else:
            entry["fitted"] = in_view(entry["view"], outline_of(masks[i]))
        if lips[i] is not None:
            parts, labels, stats, _ = cv2.connectedComponentsWithStats(lips[i], connectivity=4)
            for label in range(1, parts):
                if stats[label][4] >= _px(1.4) ** 2:
                    entry["under"].append(in_view(entry["view"], outline_of((labels == label).astype(np.uint8))))
    return masks, lips, everything


def eye_frame(entries, masks):
    """The dark frame round an eye, as a front-view outline: the bright core let out all round."""
    by_name = {e["name"]: i for i, e in enumerate(entries)}
    core = masks[by_name["eye_light_{S}"]]
    frame = cv2.dilate(core, _brush(_px(EYE_FRAME_MM)))
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygon = cv2.approxPolyDP(max(contours, key=cv2.contourArea), _px(0.5), True).reshape(-1, 2)
    polygon = straighten(polygon, min_edge=_px(2.5), max_shift=_px(0.6))
    out = []
    for x, y in polygon:
        theta, z = chart_at(float(x), float(y))
        out.append([round(FORM.point(theta, z)[0], 5), round(z, 5)])
    return out


def draw_layout(masks, lips, everything):
    """references/trace/chart_layout.png: the head unwrapped, every plate where the build will put it."""
    tint = {"front": (60, 190, 235), "side": (90, 200, 120), "back": (230, 150, 80)}
    image = np.full((*SHAPE, 3), 250, np.uint8)
    for theta in range(-90, 271, 45):       # a line every 45 degrees round the head
        x = int(chart_px(math.radians(theta), 0.0)[0])
        cv2.line(image, (x, 0), (x, SHAPE[0] - 1), (215, 215, 215), 1)
        cv2.putText(image, f"{theta}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    for i, entry in enumerate(everything):
        colour = np.array(tint[entry["view"]], float)
        if entry["layer"] not in SHELL:
            colour = np.array((150, 150, 190), float)
        colour = colour * (0.55 if entry.get("ghost") else 1.0) + (110 if entry.get("ghost") else 0)
        shade = colour * (0.82 + 0.09 * (LAYERS[entry["layer"]][0] if entry["layer"] in SHELL else 0))
        image[masks[i] > 0] = np.clip(shade, 0, 255)
    for i, lip in enumerate(lips):
        if lip is not None:
            image[(lip > 0) & ((np.indices(SHAPE).sum(axis=0) % 6) < 2)] = (40, 40, 200)
    for i, entry in enumerate(everything):
        contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (40, 40, 40), 1, cv2.LINE_AA)
        if not entry.get("ghost") and contours:
            m = cv2.moments(max(contours, key=cv2.contourArea))
            if m["m00"]:
                label = entry["name"].replace("_{S}", "").replace("{S}_", "").replace("forehead", "fh")
                cv2.putText(image, label, (int(m["m10"] / m["m00"]) - 4 * len(label), int(m["m01"] / m["m00"]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(TRACE / "chart_layout.png"), image)


def main():
    traces = {name: load(name) for name in ("front", "turn_front", "side", "back")}
    views = calibrate(traces)
    entries = collect(traces, views)
    masks, lips, everything = fit(entries)
    draw_layout(masks, lips, everything)

    plates = []
    for entry in entries:
        if entry.get("hidden"):
            continue
        options = entry["options"]
        plate = {"name": entry["name"], "group": entry["group"], "layer": entry["layer"], "kind": entry["kind"],
                 "view": entry["view"], "rgb": entry["rgb"], "seed_px": list(entry["seed"]), "outline": entry["outline"]}
        if entry["layer"] in SHELL:
            plate["outline"] = entry["fitted"]
            if entry["under"]:
                plate["under"] = entry["under"]
        for key in ("plane", "folds", "taper", "fit_u"):
            if key in options:
                plate[key] = list(options[key]) if isinstance(options[key], (list, tuple)) else options[key]
        plates.append(plate)

    regions = traces["front"]["regions"]
    fin = find_region(regions, FIN_SEED)
    ears = [find_region(regions, s) for s in EAR_SEEDS]
    side = views["side"]
    collar = EAR_COLLAR
    extras = {
        "fin": {"rgb": fin["rgb"], "front": [to_view(views["front"], p) for p in fin["polygon"]],
                "side": [to_view(side, p) for p in FIN_SIDE]},
        "ear": {"rgb": ears[0]["rgb"],
                "front": [[to_view(views["front"], p) for p in r["polygon"]] for r in ears],
                "ring_outer": [to_view(side, p) for p in EAR_RING_OUTER],
                "ring_inner": [to_view(side, p) for p in EAR_RING_INNER],
                "hub": [to_view(side, p) for p in EAR_HUB], "hub_apex": to_view(side, EAR_HUB_APEX),
                "collar": {"centre": to_view(side, collar["centre"]),
                           "outer": [round(r * side["scale"], 5) for r in collar["outer"]],
                           "inner": [round(r * side["scale"], 5) for r in collar["inner"]]}},
        "eye_frame": eye_frame(entries, masks),
    }
    out = {
        "about": "Head plate layout traced from the reference drawings by tools/freeze_trace.py. Metres, head "
                 "space: x = the character's left, y = towards the back, z = up. 'outline' is the plate as "
                 "drawn, in the coordinates of its view: front and back (x, z), side (y, z); for the shell "
                 "plates (layers L0-L2) after being fitted to their neighbours, and with 'under', the lips on "
                 "which they run on beneath higher neighbours. Sided plates are the character's left; the builder mirrors them. "
                 "rgb is the plate's colour in the drawing (sRGB, 0-255).",
        "head_top_z": HEAD_TOP_Z,
        "views": {name: {k: (round(v, 9) if isinstance(v, float) else v) for k, v in view.items()}
                  for name, view in views.items()},
        "layers": {k: {"relief_mm": v[0], "thickness_mm": v[1], "chamfer_mm": v[2]} for k, v in LAYERS.items()},
        "shell_layers": list(SHELL), "lip_mm": {"drop": LIP_MM[0], "thickness": LIP_MM[1]},
        "plates": plates, "extras": extras,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    for name, view in views.items():
        print(f"{name}: scale {view['scale'] * 1000:.4f} mm/px, origin ({view['ox']:.1f}, {view['oy']:.1f}) px")
    shell = [p for p in plates if p["layer"] in SHELL]
    print(f"{len(plates)} plates: {len(shell)} on the shell ({sum('under' in p for p in shell)} with lips), "
          f"{len(plates) - len(shell)} standing proud of it -> {OUT.name}")


if __name__ == "__main__":
    main()
