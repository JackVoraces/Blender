"""Trace the reference drawings: find the outline of every inked panel.

    python tools/trace_reference.py            every view
    python tools/trace_reference.py side back  just these

Views are listed in VIEWS: the front anatomy sheet, and the five heads of the
turnaround.  For each one this writes into references/trace/

    <view>_regions.json    every panel: id, kind, colour, polygon (pixels of the full image),
                           plus the outline of the whole figure
    <view>_regions.png     the crop, enlarged, each panel tinted and numbered (for choosing seeds)

A panel is its bright patch, let out by a small margin to take in its shaded
edge, but never beyond its *cell* (the patch plus the ink around it, up to half
way to its neighbours, pulled in by a fixed inset).  That keeps a plate's own
straight edges and guarantees an even gap between neighbours.  The outline is
then fitted with a polygon and straightened.  The eyes are traced separately as
their hard white core, without the glow round them.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
OUT = REFERENCES / "trace"
ANATOMY = "head_anatomy_front.webp"
TURNAROUND = "Golden Android Helmet Turnaround Sheet.png"

# Lengths are in pixels of the drawing as it is segmented.  The turnaround heads are drawn at
# under half the size of the anatomy sheet with hairline joints, so each is enlarged first
# ("upscale") and its joints are found as thin lines darker than what lies either side of
# them ("thin": brush radius, darkness), not just as near-black ink.
_FINE = {"value": 150, "reach": 6, "inset": 2, "margin": 4, "epsilon": 3.0, "min_area": 120, "zoom": 2,
         "max_shift": 3.2, "min_edge": 8.0, "eye_area": 700, "brow_area": 400, "upscale": 1, "thin": None,
         "background": "sides"}
_SMALL = {"value": 150, "reach": 5, "inset": 2, "margin": 3, "epsilon": 2.6, "min_area": 260, "zoom": 4,
          "max_shift": 3.0, "min_edge": 8.0, "eye_area": 700, "brow_area": 500, "upscale": 3, "thin": (5, 46),
          "background": "top"}
VIEWS = {
    # name: image, crop (x0, y0, x1, y1), settings
    "front": (ANATOMY, (300, 100, 945, 1000), _FINE),
    "turn_front": (TURNAROUND, (5, 170, 305, 620), _SMALL),
    "turn_three_quarter": (TURNAROUND, (300, 170, 615, 620), _SMALL),
    "side": (TURNAROUND, (605, 170, 905, 620), _SMALL),
    "back_three_quarter": (TURNAROUND, (895, 170, 1180, 620), _SMALL),
    "back": (TURNAROUND, (1160, 170, 1445, 620), _SMALL),
}


def _brush(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def straighten(polygon, max_turn_deg=20.0, min_edge=8.0, max_shift=3.2):
    """Drop corners that are only a slight bend or the end of a very short edge.

    The drawn plates are polygons with a handful of straight edges; a traced outline has
    extra corners from the ink's texture.  A corner goes if removing it moves the outline
    by less than ``max_shift`` pixels and it either turns less than ``max_turn_deg`` or
    ends an edge shorter than ``min_edge``.  The least significant corner goes first.
    """
    points = [np.array(p, float) for p in polygon]
    while len(points) > 4:
        best = None
        for i, b in enumerate(points):
            a, c = points[i - 1], points[(i + 1) % len(points)]
            ab, bc, ac = b - a, c - b, c - a
            len_ab, len_bc, len_ac = np.linalg.norm(ab), np.linalg.norm(bc), np.linalg.norm(ac)
            if len_ab < 1e-9 or len_bc < 1e-9 or len_ac < 1e-9:
                best = (0.0, i)
                break
            turn = np.degrees(np.arccos(np.clip(np.dot(ab, bc) / (len_ab * len_bc), -1.0, 1.0)))
            shift = abs(ac[0] * (b - a)[1] - ac[1] * (b - a)[0]) / len_ac
            if shift < max_shift and (turn < max_turn_deg or min(len_ab, len_bc) < min_edge):
                if best is None or shift < best[0]:
                    best = (shift, i)
        if best is None:
            break
        points.pop(best[1])
    return np.array(points)


def segment(crop, origin, opts):
    """Panels and figure outline of one cropped drawing.  ``origin`` is the crop's corner in the full image."""
    x0, y0 = origin
    up = opts["upscale"]
    if up > 1:
        crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)

    def to_image(px, py):
        return round(float(px) / up + x0, 2), round(float(py) / up + y0, 2)

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)

    # 1. seeds: bright patches (panels) and the dark-brown brows
    dark = val < opts["value"]
    if opts["thin"]:
        radius, darkness = opts["thin"]
        luma = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark |= cv2.morphologyEx(luma, cv2.MORPH_BLACKHAT, _brush(radius)) > darkness
    ink = cv2.dilate(dark.astype(np.uint8), _brush(1))
    brow = ((hue <= 14) & (sat >= 110) & (val >= 55) & (val < opts["value"])).astype(np.uint8)
    brow = cv2.morphologyEx(brow, cv2.MORPH_OPEN, _brush(2))
    label_map = np.zeros(crop.shape[:2], np.int32)
    kinds, colours = {}, {}
    for kind, mask, min_area in (("panel", 1 - ink, opts["min_area"]), ("brow", brow, opts["brow_area"])):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for label in range(1, count):
            if stats[label][4] >= min_area:
                new = len(kinds) + 1
                label_map[labels == label] = new
                kinds[new] = kind
                colours[new] = [int(c) for c in np.median(crop[labels == label], axis=0)]

    # 2. cells: every ink pixel within reach joins the nearest seed
    unlabelled = (label_map == 0).astype(np.uint8)
    distance, nearest = cv2.distanceTransformWithLabels(unlabelled, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros(nearest.max() + 1, np.int32)
    lookup[nearest[label_map > 0]] = label_map[label_map > 0]
    cells = np.where((label_map == 0) & (distance <= opts["reach"]), lookup[nearest], label_map)

    # 2b. the eyes: only the hard white core, not the glow painted round it
    core = ((val >= 236) & (sat <= 70)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    eye_masks = [(labels == label).astype(np.uint8) for label in range(1, count) if stats[label][4] >= opts["eye_area"]]

    square = np.ones((3, 3), np.uint8)
    sources = [("eye", mask, [255, 250, 235]) for mask in eye_masks]
    for label, kind in kinds.items():
        # A plate's outline is its own bright patch, let out by a few pixels to take in its
        # shaded edge, so its straight edges stay straight.  It may never leave its cell (pulled
        # in by the inset), which is what keeps a fixed gap between close neighbours.
        own = cv2.dilate((label_map == label).astype(np.uint8), square, iterations=opts["margin"])
        cell = cv2.erode((cells == label).astype(np.uint8), square, iterations=opts["inset"])
        sources.append((kind, own & cell, colours[label]))
    tidy = {"max_shift": opts["max_shift"], "min_edge": opts["min_edge"]}
    regions = []
    for kind, mask, colour in sources:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if kind == "brow":
            # a brow is one convex slab; its housing overlaps it and nibbles the traced edge
            contour = cv2.convexHull(contour)
            polygon = straighten(cv2.approxPolyDP(contour, opts["epsilon"], True).reshape(-1, 2), max_turn_deg=28.0,
                                 min_edge=tidy["min_edge"] * 1.75, max_shift=tidy["max_shift"] * 1.6)
        else:
            polygon = straighten(cv2.approxPolyDP(contour, opts["epsilon"], True).reshape(-1, 2), **tidy)
        moments = cv2.moments(contour)
        if len(polygon) < 3 or moments["m00"] == 0:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        regions.append({
            "id": len(regions), "kind": kind, "area": int(mask.sum() / (up * up)), "rgb": colour,
            "centroid": list(to_image(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])),
            "touches_crop_edge": bool(bx == 0 or by == 0 or bx + bw >= mask.shape[1] or by + bh >= mask.shape[0]),
            "polygon": [list(to_image(px, py)) for px, py in polygon],
        })

    # 3. the figure: everything that is not parchment reachable from the edges of the crop
    parchment = ((sat < 75) & (val > 185)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(parchment, connectivity=4)
    background = np.zeros_like(parchment)
    for label in range(1, count):
        bx, by, bw, bh, area = stats[label]
        if opts["background"] == "top":
            # the turnaround: paper runs along the top of every crop; the white collar never reaches it
            outside = by == 0
        else:
            outside = (by == 0 or bx == 0 or bx + bw >= parchment.shape[1]) and by + bh < parchment.shape[0] - 2
        if area > 150 * up * up and outside:
            background[labels == label] = 1
    figure = cv2.morphologyEx(1 - background, cv2.MORPH_OPEN, _brush(3))
    contours, _ = cv2.findContours(figure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    return regions, [list(to_image(px, py)) for px, py in outline]


def draw(crop, origin, regions, outline, zoom):
    """The crop enlarged, each panel tinted and numbered, the figure outlined."""
    rng = np.random.default_rng(7)
    debug = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
    tint = debug.copy()
    for region in regions:
        points = ((np.array(region["polygon"]) - origin) * zoom).astype(np.int32)
        cv2.fillPoly(tint, [points], [int(c) for c in rng.integers(40, 255, 3)])
    debug = cv2.addWeighted(debug, 0.6, tint, 0.4, 0)
    cv2.polylines(debug, [((np.array(outline) - origin) * zoom).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    for region in regions:
        cx, cy = (np.array(region["centroid"]) - origin) * zoom
        text = str(region["id"])
        cv2.putText(debug, text, (int(cx) - 9, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(debug, text, (int(cx) - 9, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return debug


def main():
    names = sys.argv[1:] or list(VIEWS)
    unknown = [n for n in names if n not in VIEWS]
    if unknown:
        sys.exit(f"unknown view(s) {unknown}; choose from {list(VIEWS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    images = {}
    for name in names:
        file_name, (x0, y0, x1, y1), opts = VIEWS[name]
        if file_name not in images:
            # Annotation leader lines are left in: they are ink, and painting them out opened
            # gaps in the outlines they cross, which let panels leak into the background.
            images[file_name] = np.array(Image.open(REFERENCES / file_name).convert("RGB"))
        image = images[file_name]
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        regions, outline = segment(crop, (x0, y0), opts)
        with open(OUT / f"{name}_regions.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"image": file_name, "image_size": [image.shape[1], image.shape[0]],
                       "crop": [x0, y0, x1, y1], "regions": regions, "figure": outline}, fh)
        cv2.imwrite(str(OUT / f"{name}_regions.png"), draw(crop, (x0, y0), regions, outline, opts["zoom"]))
        counts = {k: sum(r["kind"] == k for r in regions) for k in ("panel", "brow", "eye")}
        corners = sum(len(r["polygon"]) for r in regions) / max(len(regions), 1)
        print(f"{name}: {counts}, {corners:.1f} corners per plate, figure outline {len(outline)} points")


if __name__ == "__main__":
    main()
