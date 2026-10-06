"""Tracker signal chain, tested without a camera, MediaPipe or a GUI."""
import json
import math
import socket
import unittest

from tracker import cameras, head_pose, sources
from tracker.calibration import NeutralCalibration
from tracker.face_mapping import FaceMapper, deadzone
from tracker.pipeline import Pipeline
from tracker.settings import load_mapping, load_settings
from tracker.smoothing import Ema
from tracker.sources import SCRIPT, Sample, SyntheticSource
from tracker.udp_sender import UdpSender, make_packet

MAPPING = {
    "format": "rn.face_mapping/1",
    "defaults": {"deadzone": 0.05, "gain": 1.0, "baseline": True},
    "controls": {
        "BROW_{S}_RAISE": {"clamp": [0, 1], "terms": [{"src": "browOuterUp{Side}", "gain": 2.0}]},
        "EYE_{S}_OPEN": {"clamp": [-1, 1], "terms": [{"src": "eyeBlink{Side}", "gain": -2.0},
                                                    {"src": "eyeWide{Side}", "gain": 2.0}]},
    },
}


def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
            "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
            "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


class DeadzoneTests(unittest.TestCase):
    def test_inside_is_zero_and_edges_are_continuous(self):
        self.assertEqual(deadzone(0.02, 0.05), 0.0)
        self.assertEqual(deadzone(-0.05, 0.05), 0.0)
        self.assertAlmostEqual(deadzone(0.30, 0.05), 0.25)
        self.assertAlmostEqual(deadzone(-0.30, 0.05), -0.25)


class MappingTests(unittest.TestCase):
    def test_gain_deadzone_baseline_clamp(self):
        mapper = FaceMapper(MAPPING, mirror=False)
        out = mapper.map({"browOuterUpLeft": 0.35}, {"browOuterUpLeft": 0.10})
        self.assertAlmostEqual(out["BROW_L_RAISE"], (0.35 - 0.10 - 0.05) * 2.0)
        self.assertEqual(out["BROW_R_RAISE"], 0.0)
        self.assertEqual(mapper.map({"browOuterUpLeft": 0.99})["BROW_L_RAISE"], 1.0)

    def test_normalize_gives_a_high_resting_score_its_full_range_back(self):
        config = dict(MAPPING, defaults={"deadzone": 0.0, "gain": 1.0, "baseline": True, "normalize": True})
        mapper = FaceMapper(config, mirror=False)
        rest = {"browOuterUpLeft": 0.4}
        self.assertAlmostEqual(mapper.map({"browOuterUpLeft": 0.7}, rest)["BROW_L_RAISE"], 2.0 * 0.3 / 0.6)
        self.assertEqual(mapper.map({"browOuterUpLeft": 0.4}, rest)["BROW_L_RAISE"], 0.0)
        # a rest score near 1 must not blow the value up
        self.assertLessEqual(mapper.map({"browOuterUpLeft": 1.0}, {"browOuterUpLeft": 0.99})["BROW_L_RAISE"], 0.1)

    def test_below_baseline_cannot_go_negative_on_a_unipolar_control(self):
        mapper = FaceMapper(MAPPING, mirror=False)
        self.assertEqual(mapper.map({"browOuterUpLeft": 0.0}, {"browOuterUpLeft": 0.4})["BROW_L_RAISE"], 0.0)

    def test_bipolar_control(self):
        mapper = FaceMapper(MAPPING, mirror=False)
        self.assertEqual(mapper.map({"eyeBlinkLeft": 0.9})["EYE_L_OPEN"], -1.0)
        self.assertAlmostEqual(mapper.map({"eyeWideLeft": 0.25})["EYE_L_OPEN"], 0.4)

    def test_sides_stay_independent(self):
        out = FaceMapper(MAPPING, mirror=False).map({"browOuterUpLeft": 0.5, "browOuterUpRight": 0.1})
        self.assertGreater(out["BROW_L_RAISE"], 0.8)
        self.assertLess(out["BROW_R_RAISE"], 0.15)

    def test_mirror_swaps_sides(self):
        shapes = {"browOuterUpLeft": 0.5}
        self.assertGreater(FaceMapper(MAPPING, mirror=False).map(shapes)["BROW_L_RAISE"], 0.5)
        mirrored = FaceMapper(MAPPING, mirror=True).map(shapes)
        self.assertEqual(mirrored["BROW_L_RAISE"], 0.0)
        self.assertGreater(mirrored["BROW_R_RAISE"], 0.5)

    def test_project_mapping_only_uses_real_mediapipe_names(self):
        with open(__file__.replace("test_tracker.py", "data/mediapipe_blendshape_names.json"), encoding="utf-8") as fh:
            known = set(json.load(fh)["names"])
        mapper = FaceMapper(load_mapping(), mirror=True)
        self.assertEqual(sorted(set(mapper.sources) - known), [])


class CalibrationTests(unittest.TestCase):
    def test_median_baseline_ignores_a_blink(self):
        cal = NeutralCalibration(duration_s=1.0)
        cal.start(0.0)
        done = False
        for i in range(31):
            blink = 0.9 if i in (10, 11) else 0.08
            done = cal.add(i / 30.0, {"eyeBlinkLeft": blink}, (0.05, -0.10, 0.02))
        self.assertTrue(done)
        self.assertFalse(cal.active)
        self.assertAlmostEqual(cal.baseline["eyeBlinkLeft"], 0.08)
        self.assertAlmostEqual(cal.head_baseline[1], -0.10)

    def test_progress(self):
        cal = NeutralCalibration(duration_s=2.0)
        self.assertEqual(cal.progress(5.0), 1.0)
        cal.start(10.0)
        self.assertAlmostEqual(cal.progress(11.0), 0.5)


class SmoothingTests(unittest.TestCase):
    def test_moves_by_alpha_per_reference_frame(self):
        ema = Ema(0.25, reference_fps=30)
        self.assertAlmostEqual(ema.update({"a": 1.0}, 1 / 30)["a"], 0.25)
        self.assertAlmostEqual(ema.update({"a": 1.0}, 1 / 30)["a"], 0.4375)

    def test_frame_rate_independent(self):
        slow, fast = Ema(0.25, 30), Ema(0.25, 30)
        for _ in range(30):
            a = slow.update({"a": 1.0}, 1 / 30)["a"]
        for _ in range(60):
            b = fast.update({"a": 1.0}, 1 / 60)["a"]
        self.assertAlmostEqual(a, b, places=6)

    def test_per_key_alpha(self):
        ema = Ema(0.1, 30, {"eye": 0.8})
        out = ema.update({"eye": 1.0, "brow": 1.0}, 1 / 30)
        self.assertAlmostEqual(out["eye"], 0.8)
        self.assertAlmostEqual(out["brow"], 0.1)


class HeadPoseTests(unittest.TestCase):
    def test_identity_is_zero(self):
        self.assertEqual(head_pose.matrix_to_angles(_rot("x", 0.0)), (0.0, 0.0, 0.0))

    def test_single_axes_and_signs(self):
        # camera space: +X image right, +Y up, +Z towards the camera
        yaw, pitch, roll = head_pose.matrix_to_angles(_rot("y", 0.3))     # face swings towards +X
        self.assertAlmostEqual(yaw, 0.3)
        self.assertAlmostEqual(pitch, 0.0)
        yaw, pitch, roll = head_pose.matrix_to_angles(_rot("x", -0.2))    # face tips towards +Y: looking up
        self.assertAlmostEqual(pitch, 0.2)
        yaw, pitch, roll = head_pose.matrix_to_angles(_rot("z", -0.25))   # crown leans towards +X
        self.assertAlmostEqual(roll, 0.25)

    def test_round_trip_with_scale(self):
        want = (0.31, -0.17, 0.12)
        m = _mul(_mul(_rot("y", want[0]), _rot("x", -want[1])), _rot("z", -want[2]))
        scaled = [[v * 7.5 for v in row] for row in m]
        for got, expected in zip(head_pose.matrix_to_angles(scaled), want):
            self.assertAlmostEqual(got, expected)

    def test_mirror(self):
        self.assertEqual(head_pose.mirror_angles((0.1, 0.2, -0.3)), (-0.1, 0.2, 0.3))


class PipelineTests(unittest.TestCase):
    def _run_script(self, mirror):
        settings = load_settings()
        settings["camera"]["mirror"] = mirror
        pipeline = Pipeline(settings, load_mapping())
        source = SyntheticSource(fps=30, realtime=False, jitter=0.006)
        peaks, lows, head_hi, head_lo = {}, {}, [0, 0, 0], [0, 0, 0]
        starts, start = {}, 0.0
        for name, seconds, _targets, _head in SCRIPT:
            starts[name] = (start, seconds)
            start += seconds
        for frame in range(int(start * 30)):
            sample = source.read()
            out = pipeline.process(sample)
            beat_start, seconds = starts[source.beat]
            phase = (frame / 30 - beat_start) / seconds
            if 0.4 <= phase <= 0.75:        # the settled middle of each beat, clear of the last one's tail
                for name, value in out.face.items():
                    key = (source.beat, name)
                    peaks[key] = max(peaks.get(key, -9), value)
                    lows[key] = min(lows.get(key, 9), value)
            for i in range(3):
                head_hi[i] = max(head_hi[i], out.head[i])
                head_lo[i] = min(head_lo[i], out.head[i])
        return pipeline, peaks, lows, head_hi, head_lo

    def test_scripted_performance_unmirrored(self):
        pipeline, peaks, lows, head_hi, head_lo = self._run_script(mirror=False)
        self.assertEqual(pipeline.calibration.completed, 1)
        # at rest, after calibration, nothing moves despite non-zero resting scores and jitter
        for name in pipeline.mapper.names:
            self.assertLess(abs(peaks[("rest", name)]), 0.02, name)
            self.assertLess(abs(lows[("rest", name)]), 0.02, name)
        # each beat moves the right control and leaves the other side alone
        self.assertGreater(peaks[("left_brow_up", "BROW_L_RAISE")], 0.9)
        self.assertLess(peaks[("left_brow_up", "BROW_R_RAISE")], 0.4)
        self.assertGreater(peaks[("right_brow_up", "BROW_R_RAISE")], 0.9)
        self.assertGreater(peaks[("frown", "BROW_L_LOWER")], 0.9)
        self.assertLess(lows[("blink", "EYE_L_OPEN")], -0.8)
        self.assertLess(lows[("blink", "EYE_R_OPEN")], -0.8)
        self.assertLess(lows[("left_wink", "EYE_L_OPEN")], -0.8)
        self.assertGreater(lows[("left_wink", "EYE_R_OPEN")], -0.1)
        self.assertGreater(peaks[("surprise", "EYE_L_OPEN")], 0.8)
        self.assertGreater(peaks[("smile", "CHEEK_L_RAISE")], 0.9)
        self.assertGreater(peaks[("right_smirk", "CHEEK_R_RAISE")], 0.8)
        self.assertLess(peaks[("right_smirk", "CHEEK_L_RAISE")], 0.1)
        # head: the resting camera offset is calibrated out; turns arrive with the right sign
        self.assertGreater(math.degrees(head_hi[0]), 18)
        self.assertLess(math.degrees(head_lo[0]), -18)
        self.assertGreater(math.degrees(head_hi[1]), 10)
        self.assertGreater(math.degrees(head_hi[2]), 10)

    def test_scripted_performance_mirrored(self):
        _pipeline, peaks, lows, head_hi, head_lo = self._run_script(mirror=True)
        self.assertGreater(peaks[("left_brow_up", "BROW_R_RAISE")], 0.9)
        self.assertLess(peaks[("left_brow_up", "BROW_L_RAISE")], 0.4)
        self.assertLess(lows[("left_wink", "EYE_R_OPEN")], -0.8)
        self.assertGreater(peaks[("right_smirk", "CHEEK_L_RAISE")], 0.8)
        # "tilt_left" leans to the performer's left, so the mirrored avatar leans to its right
        self.assertLess(math.degrees(head_lo[2]), -10)

    def test_face_lost_holds_then_glides_to_neutral(self):
        settings = load_settings()
        settings["calibration"]["auto_start"] = False
        pipeline = Pipeline(settings, load_mapping())
        shapes = {"browOuterUpLeft": 0.9, "browOuterUpRight": 0.9}
        t = 0.0
        for _ in range(30):
            out = pipeline.process(Sample(time=t, tracking=True, blendshapes=shapes, head=(0.3, 0.0, 0.0)))
            t += 1 / 30
        raised = out.face["BROW_L_RAISE"]
        self.assertGreater(raised, 0.9)
        self.assertGreater(abs(out.head[0]), 0.25)
        for _ in range(6):                      # 0.2 s dropout: pose is held
            out = pipeline.process(Sample(time=t, tracking=False))
            t += 1 / 30
        self.assertGreater(out.face["BROW_L_RAISE"], 0.9)
        previous = out.face["BROW_L_RAISE"]
        for _ in range(90):                     # 3 s gone: smooth return, no snapping
            out = pipeline.process(Sample(time=t, tracking=False))
            t += 1 / 30
            self.assertLessEqual(previous - out.face["BROW_L_RAISE"], 0.1)
            previous = out.face["BROW_L_RAISE"]
        self.assertLess(out.face["BROW_L_RAISE"], 0.01)
        self.assertLess(abs(out.head[0]), 0.01)

    def test_recalibration_rests_then_adopts_the_new_neutral(self):
        settings = load_settings()
        pipeline = Pipeline(settings, load_mapping())
        t = 0.0
        tense = {"browDownLeft": 0.5, "browDownRight": 0.5}
        for _ in range(90):
            out = pipeline.process(Sample(time=t, tracking=True, blendshapes=tense, head=(0, 0, 0)))
            t += 1 / 30
        self.assertEqual(pipeline.calibration.completed, 1)
        self.assertLess(out.face["BROW_L_LOWER"], 0.02)       # that tense face IS neutral now
        relaxed = {"browDownLeft": 0.1, "browDownRight": 0.1}
        for _ in range(30):
            out = pipeline.process(Sample(time=t, tracking=True, blendshapes=relaxed, head=(0, 0, 0)))
            t += 1 / 30
        self.assertLess(out.face["BROW_L_LOWER"], 0.02)       # below baseline: still nothing
        pipeline.recalibrate()
        for _ in range(90):
            out = pipeline.process(Sample(time=t, tracking=True, blendshapes=relaxed, head=(0, 0, 0)))
            t += 1 / 30
        self.assertEqual(pipeline.calibration.completed, 2)
        for _ in range(30):
            out = pipeline.process(Sample(time=t, tracking=True, blendshapes=tense, head=(0, 0, 0)))
            t += 1 / 30
        self.assertGreater(out.face["BROW_L_LOWER"], 0.5)


class CameraChoiceTests(unittest.TestCase):
    NAMES = ["USB Video Device", "Elgato Facecam", "OBS Virtual Camera"]

    def test_by_number(self):
        self.assertEqual(cameras.resolve(1, self.NAMES), (1, "Elgato Facecam"))
        self.assertEqual(cameras.resolve("2", self.NAMES), (2, "OBS Virtual Camera"))
        self.assertEqual(cameras.resolve(7, self.NAMES), (7, None))        # not listed: still usable by number

    def test_by_part_of_the_name(self):
        self.assertEqual(cameras.resolve("facecam", self.NAMES), (1, "Elgato Facecam"))
        self.assertEqual(cameras.resolve("  OBS ", self.NAMES), (2, "OBS Virtual Camera"))

    def test_exact_name_beats_partial(self):
        self.assertEqual(cameras.resolve("cam", ["Cam Pro", "Cam"]), (1, "Cam"))

    def test_unknown_name_says_what_exists(self):
        with self.assertRaises(LookupError) as caught:
            cameras.resolve("webcam 9000", self.NAMES)
        self.assertIn("Elgato Facecam", str(caught.exception))

    def test_a_configured_name_wins_over_the_index(self):
        original = cameras.list_names
        cameras.list_names = lambda: self.NAMES
        try:
            cam = {"name": "Elgato Facecam", "index": 0, "backend": "dshow"}
            self.assertEqual(sources.choose_camera(cam), (1, "Elgato Facecam"))
            cam["name"] = None
            self.assertEqual(sources.choose_camera(cam), (0, "USB Video Device"))
            cam["name"] = "missing"
            with self.assertRaises(sources.CameraError):
                sources.choose_camera(cam)
        finally:
            cameras.list_names = original


class UdpTests(unittest.TestCase):
    def test_packet_round_trip_on_localhost(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        port = receiver.getsockname()[1]
        sender = UdpSender([{"name": "test", "host": "127.0.0.1", "port": port}])
        try:
            sender.send(make_packet(7, True, False, {"BROW_L_RAISE": 0.123456, "EYE_R_OPEN": -1.0}, (0.1, -0.2, 0.3)))
            packet = json.loads(receiver.recvfrom(65535)[0].decode("utf-8"))
        finally:
            sender.close()
            receiver.close()
        self.assertEqual(packet["v"], 1)
        self.assertEqual(packet["seq"], 7)
        self.assertTrue(packet["tracking"])
        self.assertEqual(packet["face"], {"brow_l_raise": 0.1235, "eye_r_open": -1.0})
        self.assertEqual(packet["head"], {"yaw": 0.1, "pitch": -0.2, "roll": 0.3})

    def test_sending_to_nobody_is_harmless(self):
        sender = UdpSender([{"name": "nobody", "host": "127.0.0.1", "port": 9}])
        for i in range(5):
            sender.send(make_packet(i, True, False, {}, (0, 0, 0)))
        sender.close()
        self.assertEqual(sender.sent, 5)


if __name__ == "__main__":
    unittest.main()
