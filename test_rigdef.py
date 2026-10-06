"""Rig definition: expansion, mirroring, limits, and the reference evaluator."""
import math
import unittest
from pathlib import Path

from rn_common import rigdef

ROOT = Path(__file__).resolve().parents[1]
MM = 0.001
DEG = math.pi / 180.0


def _authored(**overrides):
    base = {
        "format": rigdef.AUTHORED_FORMAT,
        "controls": {"BROW_{S}_RAISE": {"range": [0, 1]}, "EYE_{S}_OPEN": {"range": [-1, 1]}, "JAW_OPEN": {"range": [0, 1]}},
        "nodes": {"brow_{S}": {"limits": {"loc_mm": {"x": [-1, 2], "z": [-2, 6]}, "rot_deg": {"y": [-10, 4]}}}},
        "links": [
            {"control": "BROW_{S}_RAISE", "node": "brow_{S}", "loc_mm": [1, 0, 5], "rot_deg": [2, -8, 3], "delay_ms": 40},
            {"control": "EYE_{S}_OPEN", "when": "neg", "node": "lid_{S}", "loc_mm": [0, 0, -10]},
            {"control": "EYE_{S}_OPEN", "when": "pos", "node": "lid_{S}", "loc_mm": [0, 0, 3]},
            {"control": "JAW_OPEN", "node": "brow_{S}", "loc_mm": [0.5, 0, 0]},
            {"control": "BROW_{S}_RAISE", "node": "nose", "loc_mm": [2, 0, 1], "rot_deg": [0, 0, 5]},
        ],
    }
    base.update(overrides)
    return base


class ExpandTests(unittest.TestCase):
    def setUp(self):
        self.rig = rigdef.expand(_authored())

    def test_sides_are_expanded(self):
        self.assertEqual(set(self.rig["controls"]),
                         {"BROW_L_RAISE", "BROW_R_RAISE", "EYE_L_OPEN", "EYE_R_OPEN", "JAW_OPEN"})
        self.assertEqual(set(self.rig["nodes"]), {"brow_L", "brow_R", "lid_L", "lid_R", "nose"})

    def test_units_are_si(self):
        link = self.rig["nodes"]["brow_L"]["links"][0]
        self.assertAlmostEqual(link["loc"][2], 5 * MM)
        self.assertAlmostEqual(link["rot"][1], -8 * DEG)
        self.assertAlmostEqual(link["delay"], 0.040)

    def test_right_side_is_mirrored(self):
        left = self.rig["nodes"]["brow_L"]["links"][0]
        right = self.rig["nodes"]["brow_R"]["links"][0]
        self.assertEqual(right["control"], "BROW_R_RAISE")
        self.assertAlmostEqual(right["loc"][0], -left["loc"][0])
        self.assertAlmostEqual(right["loc"][2], left["loc"][2])
        self.assertAlmostEqual(right["rot"][0], left["rot"][0])
        self.assertAlmostEqual(right["rot"][1], -left["rot"][1])
        self.assertAlmostEqual(right["rot"][2], -left["rot"][2])

    def test_limits_are_mirrored(self):
        left = self.rig["nodes"]["brow_L"]["limits"]
        right = self.rig["nodes"]["brow_R"]["limits"]
        self.assertAlmostEqual(left["loc"][0][0], -1 * MM)
        self.assertAlmostEqual(left["loc"][0][1], 2 * MM)
        self.assertAlmostEqual(right["loc"][0][0], -2 * MM)
        self.assertAlmostEqual(right["loc"][0][1], 1 * MM)
        self.assertAlmostEqual(right["rot"][1][0], -4 * DEG)
        self.assertAlmostEqual(right["rot"][1][1], 10 * DEG)

    def test_central_control_reaches_both_sides_mirrored(self):
        jaw_l = [l for l in self.rig["nodes"]["brow_L"]["links"] if l["control"] == "JAW_OPEN"][0]
        jaw_r = [l for l in self.rig["nodes"]["brow_R"]["links"] if l["control"] == "JAW_OPEN"][0]
        self.assertAlmostEqual(jaw_l["loc"][0], 0.5 * MM)
        self.assertAlmostEqual(jaw_r["loc"][0], -0.5 * MM)

    def test_sided_control_on_central_node_is_mirrored(self):
        links = {l["control"]: l for l in self.rig["nodes"]["nose"]["links"]}
        self.assertAlmostEqual(links["BROW_L_RAISE"]["loc"][0], 2 * MM)
        self.assertAlmostEqual(links["BROW_R_RAISE"]["loc"][0], -2 * MM)
        self.assertAlmostEqual(links["BROW_R_RAISE"]["rot"][2], -5 * DEG)

    def test_missing_limits_fall_back_to_reachable_range_with_a_warning(self):
        lid = self.rig["nodes"]["lid_L"]["limits"]
        self.assertAlmostEqual(lid["loc"][2][0], -10 * MM)
        self.assertAlmostEqual(lid["loc"][2][1], 3 * MM)
        self.assertTrue(any("lid_L" in w for w in self.rig["warnings"]))

    def test_unknown_control_is_rejected(self):
        bad = _authored(links=[{"control": "NOPE", "node": "brow_{S}", "loc_mm": [0, 0, 1]}])
        with self.assertRaises(rigdef.RigDefError):
            rigdef.expand(bad)

    def test_neg_needs_a_bipolar_control(self):
        bad = _authored(links=[{"control": "JAW_OPEN", "when": "neg", "node": "nose", "loc_mm": [0, 0, 1]}])
        with self.assertRaises(rigdef.RigDefError):
            rigdef.expand(bad)

    def test_limits_must_contain_neutral(self):
        bad = _authored(nodes={"brow_{S}": {"limits": {"loc_mm": {"z": [1, 6]}}}})
        with self.assertRaises(rigdef.RigDefError):
            rigdef.expand(bad)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.rig = rigdef.expand(_authored())

    def test_neutral_is_zero(self):
        for motion in rigdef.evaluate(self.rig, {}).values():
            self.assertEqual(motion["loc"], [0.0, 0.0, 0.0])
            self.assertEqual(motion["rot"], [0.0, 0.0, 0.0])

    def test_positive_and_negative_parts(self):
        closed = rigdef.evaluate(self.rig, {"EYE_L_OPEN": -1.0})["lid_L"]["loc"][2]
        wide = rigdef.evaluate(self.rig, {"EYE_L_OPEN": 1.0})["lid_L"]["loc"][2]
        self.assertAlmostEqual(closed, -10 * MM)
        self.assertAlmostEqual(wide, 3 * MM)

    def test_sides_are_independent(self):
        motion = rigdef.evaluate(self.rig, {"BROW_L_RAISE": 1.0})
        self.assertAlmostEqual(motion["brow_L"]["loc"][2], 5 * MM)
        self.assertEqual(motion["brow_R"]["loc"][2], 0.0)

    def test_motion_is_clamped_to_the_plate_limits(self):
        motion = rigdef.evaluate(self.rig, {"BROW_L_RAISE": 1.0, "JAW_OPEN": 1.0})["brow_L"]
        self.assertAlmostEqual(motion["loc"][0], 1.5 * MM)       # 1 + 0.5, inside [-1, 2]
        self.assertAlmostEqual(motion["rot"][1], -8 * DEG)        # inside [-10, 4]
        self.assertAlmostEqual(motion["rot"][0], 2 * DEG)         # x had no limit: reachable range
        over = rigdef.evaluate(self.rig, {"BROW_L_RAISE": 5.0})["brow_L"]
        self.assertAlmostEqual(over["loc"][2], 5 * MM)            # control itself is clamped to 1

    def test_a_tight_limit_stops_the_plate(self):
        tight = rigdef.expand(_authored(
            nodes={"brow_{S}": {"limits": {"loc_mm": {"z": [-2, 3]}, "rot_deg": {"y": [-4, 4]}}}}))
        motion = rigdef.evaluate(tight, {"BROW_L_RAISE": 1.0, "BROW_R_RAISE": 1.0})
        self.assertAlmostEqual(motion["brow_L"]["loc"][2], 3 * MM)     # wanted 5 mm
        self.assertAlmostEqual(motion["brow_L"]["rot"][1], -4 * DEG)   # wanted -8 degrees
        self.assertAlmostEqual(motion["brow_R"]["rot"][1], 4 * DEG)    # mirrored limit


class ProjectConfigTests(unittest.TestCase):
    """The real config files must stay consistent with each other."""

    def setUp(self):
        self.rig = rigdef.expand(rigdef.load_authored(ROOT / "config" / "face_rig.json"))

    def test_project_rig_expands_without_warnings(self):
        self.assertEqual(self.rig["warnings"], [])

    def test_every_tracker_control_exists_in_the_rig(self):
        from tracker.face_mapping import FaceMapper
        from tracker.settings import load_mapping
        mapper = FaceMapper(load_mapping(), mirror=True)
        self.assertEqual(sorted(set(mapper.names) - set(self.rig["controls"])), [])
        for name, (lo, hi, _terms) in mapper.controls.items():
            self.assertGreaterEqual(lo, self.rig["controls"][name]["min"], name)
            self.assertLessEqual(hi, self.rig["controls"][name]["max"], name)

    def test_exported_rig_matches_the_config(self):
        exported = ROOT / "godot" / "assets" / "face_rig.expanded.json"
        if not exported.exists():
            self.skipTest("nothing exported yet")
        self.assertEqual(rigdef.load_expanded(exported), self.rig,
                         "config/face_rig.json changed since the last export: run rn.rebuild_rig() in Blender")

    def test_exported_glb_contains_every_rig_node(self):
        from rn_common import glb
        path = ROOT / "godot" / "assets" / "robot_prototype.glb"
        if not path.exists():
            self.skipTest("nothing exported yet")
        info = glb.summary(path)
        self.assertEqual(sorted(set(self.rig["nodes"]) - set(info["nodes"])), [])
        for bone in self.rig["head"]["bones"]:
            self.assertTrue(info["nodes"][bone]["joint"], bone)
        self.assertEqual(info["animations"], 0)


if __name__ == "__main__":
    unittest.main()
