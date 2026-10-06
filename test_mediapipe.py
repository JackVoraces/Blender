"""Checks against the real, installed MediaPipe Face Landmarker (no camera needed).

These pin down the facts the tracker relies on, measured rather than assumed:
the blendshape names, which image side "Left" means, and the direction of
each head angle.  Uses tests/data/sample_portrait.png.
"""
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "face_landmarker.task"
PORTRAIT = ROOT / "tests" / "data" / "sample_portrait.png"

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    _MISSING = None if MODEL.exists() and PORTRAIT.exists() else "model or sample portrait not downloaded"
except ImportError as error:        # pragma: no cover
    _MISSING = f"mediapipe / opencv not installed ({error})"

from tracker import head_pose


def _squash_eye(image, landmarks, corner_a, corner_b, lid_top, lid_bottom, factor=0.25):
    """Crudely narrow one eye by squeezing its strip of pixels vertically."""
    h, w = image.shape[:2]
    xs = [landmarks[corner_a].x * w, landmarks[corner_b].x * w]
    cy = (landmarks[lid_top].y + landmarks[lid_bottom].y) / 2 * h
    half_w = abs(xs[1] - xs[0]) * 0.75
    cx = sum(xs) / 2
    reach = int(abs(landmarks[lid_bottom].y - landmarks[lid_top].y) * h * 1.6) + 6
    x0, x1, y0, y1 = int(cx - half_w), int(cx + half_w), int(cy - reach), int(cy + reach)
    out = image.copy()
    strip = image[y0:y1, x0:x1]
    new_h = max(2, int(strip.shape[0] * factor))
    top = y0 + (strip.shape[0] - new_h) // 2
    out[y0:top, x0:x1] = image[y0:y0 + 1, x0:x1]
    out[top + new_h:y1, x0:x1] = image[y1 - 1:y1, x0:x1]
    out[top:top + new_h, x0:x1] = cv2.resize(strip, (strip.shape[1], new_h), interpolation=cv2.INTER_AREA)
    return out


@unittest.skipIf(_MISSING, _MISSING)
class FaceLandmarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        cls.landmarker = vision.FaceLandmarker.create_from_options(options)
        cls.image = cv2.imread(str(PORTRAIT))
        cls.base = cls._detect(cls.image)

    @classmethod
    def tearDownClass(cls):
        cls.landmarker.close()

    @classmethod
    def _detect(cls, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = cls.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        assert result.face_blendshapes, "no face found"
        return {
            "shapes": {c.category_name: c.score for c in result.face_blendshapes[0]},
            "angles": [math.degrees(a) for a in head_pose.matrix_to_angles(result.facial_transformation_matrixes[0])],
            "matrix": np.array(result.facial_transformation_matrixes[0]),
            "landmarks": result.face_landmarks[0],
        }

    def test_blendshape_names_are_the_recorded_ones(self):
        with open(ROOT / "tests" / "data" / "mediapipe_blendshape_names.json", encoding="utf-8") as fh:
            recorded = json.load(fh)["names"]
        self.assertEqual(sorted(self.base["shapes"]), sorted(recorded))

    def test_the_portrait_reads_as_a_smile(self):
        shapes = self.base["shapes"]
        self.assertGreater(shapes["mouthSmileLeft"], 0.4)
        self.assertGreater(shapes["mouthSmileRight"], 0.4)
        self.assertLess(shapes["jawOpen"], 0.1)

    def test_left_means_the_eye_on_the_image_right(self):
        """i.e. the performer's own left, for an un-mirrored camera image."""
        lm = self.base["landmarks"]
        on_image_left = self._detect(_squash_eye(self.image, lm, 33, 133, 159, 145))["shapes"]
        on_image_right = self._detect(_squash_eye(self.image, lm, 362, 263, 386, 374))["shapes"]
        base = self.base["shapes"]
        # narrowing the eye on the image LEFT moves the "Right" signals, and vice versa
        self.assertGreater(on_image_left["eyeSquintRight"] - base["eyeSquintRight"], 0.3)
        self.assertLess(abs(on_image_left["eyeSquintLeft"] - base["eyeSquintLeft"]), 0.15)
        self.assertGreater(on_image_right["eyeSquintLeft"] - base["eyeSquintLeft"], 0.2)
        self.assertLess(abs(on_image_right["eyeSquintRight"] - base["eyeSquintRight"]), 0.15)

    def test_camera_space_axes(self):
        """+X is image right, +Y is up, the face sits at negative Z."""
        matrix = self.base["matrix"]
        nose = self.base["landmarks"][1]
        self.assertLess(nose.y, 0.5)                 # the face is in the upper half of the picture...
        self.assertGreater(matrix[1][3], 0.0)        # ...and its Y translation is positive
        self.assertLess(matrix[2][3], 0.0)
        h, w = self.image.shape[:2]
        wide = np.zeros((h, w * 2, 3), np.uint8)
        wide[:] = self.image[5, 5]
        wide[:, w:] = self.image                     # face moved to the right half of the frame
        self.assertGreater(self._detect(wide)["matrix"][0][3], 10.0)

    def test_mirroring_the_image_reverses_yaw_and_roll(self):
        yaw, pitch, roll = self.base["angles"]
        f_yaw, f_pitch, f_roll = self._detect(cv2.flip(self.image, 1))["angles"]
        self.assertAlmostEqual(f_yaw, -yaw, delta=2.0)
        self.assertAlmostEqual(f_roll, -roll, delta=2.0)
        self.assertAlmostEqual(f_pitch, pitch, delta=2.0)

    def test_roll_direction(self):
        """Turning the picture counter-clockwise leans the head to the image left: the
        performer's right, so roll (positive = towards their left shoulder) goes down."""
        h, w = self.image.shape[:2]
        for degrees in (12, -12):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
            turned = cv2.warpAffine(self.image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            change = self._detect(turned)["angles"][2] - self.base["angles"][2]
            self.assertAlmostEqual(change, -degrees, delta=3.0)


if __name__ == "__main__":
    unittest.main()
