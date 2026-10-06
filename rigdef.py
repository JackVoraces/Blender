"""Rig definition: robot expression controls -> rigid plate motion.

The authored file (config/face_rig.json) is compact: names may contain the
side template ``{S}``, values are in mm / degrees / ms.  ``expand()`` turns
it into the explicit form (one entry per real node, SI units) that both
runtimes consume: Blender generates drivers from it, Godot evaluates it
directly.  ``evaluate()`` is the reference implementation of the maths; the
Blender drivers and the Godot script must agree with it.

Conventions
-----------
* Every plate has a *mechanical frame* (its rest frame, origin at the pivot).
  All motion below is expressed in that frame, Blender axes:
  +X = towards the character's left along the surface, +Y = into the head,
  +Z = up along the surface.
* Motion of a node is ``loc = clamp(sum(s_i * loc_i))`` and
  ``rot = clamp(sum(s_i * rot_i))`` (Euler XYZ, radians), where ``s_i`` is
  the control value (``when: both``), its positive part (``pos``) or the
  magnitude of its negative part (``neg``).
* The right side is the mirror image of the left: loc x is negated and
  rot y / rot z are negated.  Authoring is always done for the left side.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

AUTHORED_FORMAT = "rn.face_rig/1"
EXPANDED_FORMAT = "rn.face_rig.expanded/1"
SIDES = ("L", "R")
WHEN = ("both", "pos", "neg")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_MM = 0.001
_DEG = math.pi / 180.0
_MS = 0.001


class RigDefError(ValueError):
    """The rig definition is inconsistent."""


def load_authored(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") != AUTHORED_FORMAT:
        raise RigDefError(f"{path}: expected format {AUTHORED_FORMAT!r}, got {data.get('format')!r}")
    return data


def load_expanded(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") != EXPANDED_FORMAT:
        raise RigDefError(f"{path}: expected format {EXPANDED_FORMAT!r}, got {data.get('format')!r}")
    return data


# --------------------------------------------------------------------------
# side templates and mirroring


def _sides_of(name: str):
    """[(side, concrete_name)]; side is None for names without a template."""
    if "{S}" in name:
        return [(s, name.replace("{S}", s)) for s in SIDES]
    return [(None, name)]


def mirror_loc(v):
    return [-v[0], v[1], v[2]]


def mirror_rot(v):
    return [v[0], -v[1], -v[2]]


def _mirror_limits(limits, kind):
    """limits: [[lo, hi]] * 3.  Negating an axis swaps and negates its bounds."""
    flip = (0,) if kind == "loc" else (1, 2)
    out = [list(pair) for pair in limits]
    for i in flip:
        lo, hi = limits[i]
        out[i] = [-hi, -lo]
    return out


def _vec3(value, what):
    if value is None:
        return [0.0, 0.0, 0.0]
    if len(value) != 3:
        raise RigDefError(f"{what}: expected 3 numbers, got {value!r}")
    return [float(c) for c in value]


def _clean(x: float) -> float:
    """Round away float noise so generated files and drivers stay readable."""
    r = round(x, 9)
    return 0.0 if r == 0 else r


# --------------------------------------------------------------------------
# expansion


def expand(authored: dict) -> dict:
    """Authored (templated, mm/deg/ms) -> explicit (per node, m/rad/s)."""
    controls = {}
    for tmpl, spec in authored.get("controls", {}).items():
        lo, hi = spec.get("range", [0.0, 1.0])
        if not lo <= 0.0 <= hi:
            raise RigDefError(f"control {tmpl}: range {[lo, hi]} must contain 0 (neutral)")
        for _side, name in _sides_of(tmpl):
            if name in controls:
                raise RigDefError(f"control {name} defined twice")
            controls[name] = {"min": float(lo), "max": float(hi), "doc": spec.get("doc", "")}

    # explicit limits per concrete node, SI units; None = not specified
    limits = {}
    for tmpl, spec in authored.get("nodes", {}).items():
        lim = spec.get("limits", {})
        loc = [None, None, None]
        rot = [None, None, None]
        for axis, pair in lim.get("loc_mm", {}).items():
            loc[_AXIS_INDEX[axis]] = [pair[0] * _MM, pair[1] * _MM]
        for axis, pair in lim.get("rot_deg", {}).items():
            rot[_AXIS_INDEX[axis]] = [pair[0] * _DEG, pair[1] * _DEG]
        for side, name in _sides_of(tmpl):
            if name in limits:
                raise RigDefError(f"node {name} defined twice")
            if side == "R":
                limits[name] = {
                    "loc": _mirror_optional(loc, "loc"),
                    "rot": _mirror_optional(rot, "rot"),
                }
            else:
                limits[name] = {"loc": [p and list(p) for p in loc], "rot": [p and list(p) for p in rot]}

    nodes = {}
    for i, link in enumerate(authored.get("links", [])):
        what = f"link #{i} ({link.get('control')} -> {link.get('node')})"
        when = link.get("when", "both")
        if when not in WHEN:
            raise RigDefError(f"{what}: 'when' must be one of {WHEN}")
        loc = [c * _MM for c in _vec3(link.get("loc_mm"), what)]
        rot = [c * _DEG for c in _vec3(link.get("rot_deg"), what)]
        delay = float(link.get("delay_ms", 0.0)) * _MS
        c_sides = _sides_of(link["control"])
        n_sides = _sides_of(link["node"])
        if c_sides[0][0] is not None and n_sides[0][0] is not None:
            pairs = [(cs, cn, nn) for (cs, cn), (_ns, nn) in zip(c_sides, n_sides)]
        elif c_sides[0][0] is not None:
            pairs = [(cs, cn, n_sides[0][1]) for cs, cn in c_sides]
        elif n_sides[0][0] is not None:
            pairs = [(ns, c_sides[0][1], nn) for ns, nn in n_sides]
        else:
            pairs = [(None, c_sides[0][1], n_sides[0][1])]
        for side, control, node in pairs:
            if control not in controls:
                raise RigDefError(f"{what}: unknown control {control}")
            if when == "neg" and controls[control]["min"] >= 0.0:
                raise RigDefError(f"{what}: 'neg' needs a control whose range goes below 0")
            entry = nodes.setdefault(node, {"links": []})
            entry["links"].append({
                "control": control,
                "when": when,
                "loc": [_clean(c) for c in (mirror_loc(loc) if side == "R" else loc)],
                "rot": [_clean(c) for c in (mirror_rot(rot) if side == "R" else rot)],
                "delay": delay,
            })

    warnings = []
    for node, entry in nodes.items():
        reach = _reachable(entry["links"], controls)
        explicit = limits.get(node, {"loc": [None] * 3, "rot": [None] * 3})
        final = {}
        for kind in ("loc", "rot"):
            out = []
            for axis in range(3):
                pair = explicit[kind][axis]
                r_lo, r_hi = reach[kind][axis]
                if pair is None:
                    if r_lo != 0.0 or r_hi != 0.0:
                        warnings.append(f"{node}: no {kind} limit on axis {'xyz'[axis]}, using reachable range")
                    pair = [r_lo, r_hi]
                if not pair[0] <= 0.0 <= pair[1]:
                    raise RigDefError(f"{node}: {kind} limit {pair} on axis {'xyz'[axis]} must contain 0")
                out.append([_clean(pair[0]), _clean(pair[1])])
            final[kind] = out
        entry["limits"] = final
    for node in limits:
        if node not in nodes:
            warnings.append(f"{node}: has limits but no links")

    head = _expand_head(authored.get("head"))
    return {
        "format": EXPANDED_FORMAT,
        "units": "metres, radians, seconds; node-local Blender axes",
        "controls": controls,
        "nodes": nodes,
        "head": head,
        "warnings": warnings,
    }


def _mirror_optional(limits, kind):
    flip = (0,) if kind == "loc" else (1, 2)
    out = []
    for i, pair in enumerate(limits):
        if pair is None:
            out.append(None)
        elif i in flip:
            out.append([-pair[1], -pair[0]])
        else:
            out.append(list(pair))
    return out


def _factor_range(when, c_min, c_max):
    if when == "both":
        return c_min, c_max
    if when == "pos":
        return 0.0, max(c_max, 0.0)
    return 0.0, max(-c_min, 0.0)


def _reachable(links, controls):
    """Conservative per-channel range a node can reach before clamping."""
    out = {"loc": [[0.0, 0.0] for _ in range(3)], "rot": [[0.0, 0.0] for _ in range(3)]}
    for link in links:
        ctl = controls[link["control"]]
        s_lo, s_hi = _factor_range(link["when"], ctl["min"], ctl["max"])
        for kind in ("loc", "rot"):
            for axis in range(3):
                k = link[kind][axis]
                a, b = k * s_lo, k * s_hi
                out[kind][axis][0] += min(a, b)
                out[kind][axis][1] += max(a, b)
    return out


def _expand_head(head):
    if not head:
        return None
    bones = {}
    for name, spec in head.get("bones", {}).items():
        bones[name] = {"share": float(spec.get("share", 0.0)), "lag": float(spec.get("lag_ms", 0.0)) * _MS}
    total = sum(b["share"] for b in bones.values())
    if bones and abs(total - 1.0) > 1e-6:
        raise RigDefError(f"head bone shares must sum to 1, got {total}")
    limits = {}
    for axis in ("yaw", "pitch", "roll"):
        lo, hi = head.get("limits_deg", {}).get(axis, [-30.0, 30.0])
        limits[axis] = [lo * _DEG, hi * _DEG]
    return {"bones": bones, "limits": limits}


# --------------------------------------------------------------------------
# reference evaluation


def link_factor(when: str, value: float) -> float:
    if when == "pos":
        return max(value, 0.0)
    if when == "neg":
        return max(-value, 0.0)
    return value


def clamp_controls(expanded: dict, values: dict) -> dict:
    """Every control, clamped to its range; missing controls are neutral (0)."""
    out = {}
    for name, spec in expanded["controls"].items():
        out[name] = min(max(float(values.get(name, 0.0)), spec["min"]), spec["max"])
    return out


def evaluate(expanded: dict, values: dict) -> dict:
    """Instantaneous (no delays) plate motion: {node: {"loc": [m]*3, "rot": [rad]*3}}."""
    ctl = clamp_controls(expanded, values)
    out = {}
    for node, entry in expanded["nodes"].items():
        loc = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0]
        for link in entry["links"]:
            s = link_factor(link["when"], ctl[link["control"]])
            for axis in range(3):
                loc[axis] += s * link["loc"][axis]
                rot[axis] += s * link["rot"][axis]
        lim = entry["limits"]
        out[node] = {
            "loc": [min(max(loc[a], lim["loc"][a][0]), lim["loc"][a][1]) for a in range(3)],
            "rot": [min(max(rot[a], lim["rot"][a][0]), lim["rot"][a][1]) for a in range(3)],
        }
    return out


def write_json(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
