"""Where human expression signals come from.

Every source yields ``Sample`` objects in the same shape, so the rest of the
tracker neither knows nor cares whether a webcam or a script is performing.
Blendshape names and left/right are MediaPipe's: the performer's own sides.
"""
import math
import time
from dataclasses import dataclass, field

from . import cameras, head_pose
from .settings import resolve


@dataclass
class Sample:
    time: float                          # seconds, monotonic
    tracking: bool                       # is a face visible?
    blendshapes: dict = field(default_factory=dict)   # MediaPipe category name -> score 0..1
    head: tuple = None                   # (yaw, pitch, roll) radians, performer's point of view
    frame: object = None                 # BGR image, for the debug view only
    landmarks: object = None             # normalised face landmarks, for the debug view only


# --------------------------------------------------------------------------
# scripted performance, for testing the pipeline without a camera

_REST = {      # a plausible resting face: scores are not zero at neutral
    "browDownLeft": 0.10, "browDownRight": 0.09, "browInnerUp": 0.02,
    "browOuterUpLeft": 0.03, "browOuterUpRight": 0.03,
    "eyeBlinkLeft": 0.08, "eyeBlinkRight": 0.07, "eyeWideLeft": 0.01, "eyeWideRight": 0.01,
    "eyeSquintLeft": 0.12, "eyeSquintRight": 0.11,
    "cheekSquintLeft": 0.0, "cheekSquintRight": 0.0,
    "mouthSmileLeft": 0.02, "mouthSmileRight": 0.02, "jawOpen": 0.01,
}
_REST_HEAD = (math.radians(3.0), math.radians(-6.0), math.radians(1.5))   # camera slightly off-axis

# (name, seconds, blendshape targets, head offset in degrees (yaw, pitch, roll))
SCRIPT = (
    ("neutral", 3.0, {}, (0, 0, 0)),
    ("left_brow_up", 1.4, {"browOuterUpLeft": 0.75, "browInnerUp": 0.30}, (0, 0, 0)),
    ("right_brow_up", 1.4, {"browOuterUpRight": 0.75, "browInnerUp": 0.30}, (0, 0, 0)),
    ("surprise", 1.4, {"browOuterUpLeft": 0.8, "browOuterUpRight": 0.8, "browInnerUp": 0.7,
                       "eyeWideLeft": 0.5, "eyeWideRight": 0.5}, (0, 4, 0)),
    ("frown", 1.4, {"browDownLeft": 0.75, "browDownRight": 0.75}, (0, -3, 0)),
    ("blink", 0.5, {"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.85}, (0, 0, 0)),
    ("left_wink", 1.0, {"eyeBlinkLeft": 0.85, "mouthSmileLeft": 0.4}, (0, 0, 0)),
    ("smile", 1.6, {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8,
                    "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5}, (0, 0, 0)),
    ("right_smirk", 1.2, {"mouthSmileRight": 0.8}, (0, 0, -4)),
    ("turn_left", 1.4, {}, (22, 0, 0)),
    ("turn_right", 1.4, {}, (-22, 0, 0)),
    ("look_up", 1.2, {}, (0, 14, 0)),
    ("tilt_left", 1.2, {}, (0, 0, 14)),
    ("rest", 1.0, {}, (0, 0, 0)),
)
SCRIPT_LENGTH = sum(beat[1] for beat in SCRIPT)
_EASE = 0.22     # seconds to move into / out of each beat


def _smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def script_state(t):
    """(beat name, blendshapes, head) of the scripted performance at time ``t`` (loops)."""
    t = t % SCRIPT_LENGTH
    start = 0.0
    for name, seconds, targets, head_deg in SCRIPT:
        if t < start + seconds:
            local = t - start
            ease = min(_EASE, seconds / 2.0)
            weight = _smoothstep(local / ease) * _smoothstep((seconds - local) / ease)
            shapes = dict(_REST)
            for key, target in targets.items():
                rest = _REST.get(key, 0.0)
                shapes[key] = rest + (target - rest) * weight
            head = tuple(rest + math.radians(offset) * weight for rest, offset in zip(_REST_HEAD, head_deg))
            return name, shapes, head
        start += seconds
    return "rest", dict(_REST), _REST_HEAD


def held_state(t, beat):
    """Neutral for the first beat's length (so calibration can run), then ``beat`` eased in and held."""
    lead_in = SCRIPT[0][1]
    for name, _seconds, targets, head_deg in SCRIPT:
        if name == beat:
            break
    else:
        raise ValueError(f"unknown beat {beat!r}; choose from {[b[0] for b in SCRIPT]}")
    weight = _smoothstep((t - lead_in) / _EASE)
    shapes = dict(_REST)
    for key, target in targets.items():
        rest = _REST.get(key, 0.0)
        shapes[key] = rest + (target - rest) * weight
    head = tuple(rest + math.radians(offset) * weight for rest, offset in zip(_REST_HEAD, head_deg))
    return (beat if weight > 0.0 else "neutral"), shapes, head


class SyntheticSource:
    """Plays SCRIPT in real time (or instantly, for tests) at a fixed frame rate.

    With ``hold`` set, it calibrates on a neutral face and then holds that one beat.
    """

    def __init__(self, fps=30.0, realtime=True, jitter=0.006, hold=None):
        self.fps = float(fps)
        self.realtime = realtime
        self.jitter = jitter
        self.hold = hold
        self.beat = "neutral"
        self.description = "synthetic scripted face" + (f" (holding {hold})" if hold else "")
        if hold:
            held_state(0.0, hold)       # fail early on an unknown beat name
        self._frame = 0
        self._t0 = time.monotonic()

    def read(self):
        t = self._frame / self.fps
        if self.realtime:
            delay = self._t0 + t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.beat, shapes, head = held_state(t, self.hold) if self.hold else script_state(t)
        if self.jitter:
            # small deterministic wobble standing in for tracker noise
            for i, key in enumerate(sorted(shapes)):
                noise = math.sin(self._frame * 1.7 + i * 2.3) * math.sin(self._frame * 0.31 + i)
                shapes[key] = min(max(shapes[key] + self.jitter * noise, 0.0), 1.0)
        self._frame += 1
        return Sample(time=self._t0 + t if self.realtime else t, tracking=True, blendshapes=shapes, head=head)

    def close(self):
        pass


# --------------------------------------------------------------------------
# webcam + MediaPipe Face Landmarker


class CameraError(RuntimeError):
    pass


def _backend(cv2, name):
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[str(name).lower()]


def choose_camera(cam):
    """Which camera the settings ask for: (index, name).  A name wins over an index.

    Names only exist for the DirectShow backend; other backends go by index.
    """
    names = cameras.list_names() if str(cam["backend"]).lower() == "dshow" else []
    choice = cam.get("name") or cam["index"]
    try:
        return cameras.resolve(choice, names)
    except LookupError as error:
        raise CameraError(str(error)) from None


class _FaceLandmarker:
    """MediaPipe Face Landmarker in video mode: BGR frame in, Sample out."""

    def __init__(self, settings):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._cv2 = cv2
        self._mp = mp
        model = resolve(settings["mediapipe"]["model_path"])
        if not model.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model}")
        mpc = settings["mediapipe"]
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=mpc["min_face_detection_confidence"],
            min_face_presence_confidence=mpc["min_face_presence_confidence"],
            min_tracking_confidence=mpc["min_tracking_confidence"],
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ms = -1

    def detect(self, frame, now):
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        stamp = max(self._last_ms + 1, int((now - self._t0) * 1000.0))    # must strictly increase
        self._last_ms = stamp
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, stamp)
        if not result.face_blendshapes:
            return Sample(time=now, tracking=False, frame=frame)
        shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        head = None
        if result.facial_transformation_matrixes:
            head = head_pose.matrix_to_angles(result.facial_transformation_matrixes[0])
        return Sample(time=now, tracking=True, blendshapes=shapes, head=head, frame=frame,
                      landmarks=result.face_landmarks[0])

    def close(self):
        self._landmarker.close()


class WebcamSource:
    """Local webcam -> MediaPipe Face Landmarker (blendshapes + head transform).

    Frames stay in memory: nothing is written to disk or sent anywhere.
    """

    def __init__(self, settings):
        import cv2
        cam = settings["camera"]
        self.index, self.name = choose_camera(cam)
        label = f"camera {self.index}" + (f" ({self.name})" if self.name else "")
        # Ask for the format while opening: some cameras (the Elgato Facecam among them) have
        # no 640x480 mode, which is what OpenCV would otherwise try first.
        wanted = [cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]), cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]),
                  cv2.CAP_PROP_FPS, int(cam["fps"])]
        if cam.get("fourcc"):
            wanted += [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"])]
        self._cap = cv2.VideoCapture(self.index, _backend(cv2, cam["backend"]), wanted)
        if not self._cap.isOpened():
            # Fall back to the camera's default mode, then ask for the size.  (Not done after a
            # successful open: re-negotiating the stream cut the Facecam to about 17 fps.)
            self._cap = cv2.VideoCapture(self.index, _backend(cv2, cam["backend"]))
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        if not self._cap.isOpened():
            raise CameraError(f"{label} did not open. Another program may be using it "
                              "(Camera Hub, OBS, a video call); close that, or pick another with --camera. "
                              "--list-cameras shows the choices.")
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.description = f"{label} {self.size[0]}x{self.size[1]}"
        try:
            self._tracker = _FaceLandmarker(settings)
        except Exception:
            self._cap.release()
            raise

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return self._tracker.detect(frame, time.monotonic())

    def close(self):
        self._tracker.close()
        self._cap.release()


class ImageSource:
    """A still photo fed through the real MediaPipe tracker, as if it were a camera.

    After a few still seconds (so calibration can run) the picture rocks gently
    from side to side, which the tracker sees as the head tilting.  For checking
    an installation, or the whole chain, without a webcam.
    """

    LEAD_IN = 3.0

    def __init__(self, settings, path, fps=30.0, sway_deg=11.0, period_s=4.0):
        import cv2
        self._cv2 = cv2
        self._image = cv2.imread(str(path))
        if self._image is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        self.fps = float(fps)
        self.sway = float(sway_deg)
        self.period = float(period_s)
        self.description = f"still image {path} (rocking +/-{sway_deg:g} deg)"
        self._tracker = _FaceLandmarker(settings)
        self._frame = 0
        self._t0 = time.monotonic()

    def read(self):
        t = self._frame / self.fps
        delay = self._t0 + t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._frame += 1
        frame = self._image
        swing = t - self.LEAD_IN
        if swing > 0 and self.sway:
            angle = self.sway * math.sin(2.0 * math.pi * swing / self.period)
            h, w = frame.shape[:2]
            matrix = self._cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            frame = self._cv2.warpAffine(frame, matrix, (w, h), borderMode=self._cv2.BORDER_REPLICATE)
        return self._tracker.detect(frame, time.monotonic())

    def close(self):
        self._tracker.close()
