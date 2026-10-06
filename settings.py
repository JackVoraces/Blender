"""Configuration loading.  Everything tunable lives in config/*.json."""
import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"

# Used for any key the settings file leaves out.
DEFAULTS = {
    "camera": {"name": None, "index": 0, "backend": "dshow", "width": 1280, "height": 720, "fps": 30,
               "fourcc": None, "mirror": True},
    "mediapipe": {"model_path": "models/face_landmarker.task",
                  "min_face_detection_confidence": 0.5,
                  "min_face_presence_confidence": 0.5,
                  "min_tracking_confidence": 0.5},
    "udp": {"send_rate_hz": 30, "targets": [{"name": "godot", "host": "127.0.0.1", "port": 8765}]},
    "calibration": {"duration_s": 2.0, "auto_start": True},
    "smoothing": {"alpha": 0.28, "reference_fps": 30, "per_control": {}},
    "head": {"gain": {"yaw": 1.0, "pitch": 1.0, "roll": 1.0},
             "limit_deg": {"yaw": 40, "pitch": 25, "roll": 25}, "alpha": 0.22},
    "tracking_loss": {"hold_s": 0.35, "return_alpha": 0.08},
    "debug_view": {"enabled": True, "draw_landmarks": True, "max_width": 960},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_settings(path=None) -> dict:
    return _merge(DEFAULTS, load_json(path or CONFIG_DIR / "tracker_settings.json"))


def load_mapping(path=None) -> dict:
    data = load_json(path or CONFIG_DIR / "face_mapping.json")
    if data.get("format") != "rn.face_mapping/1":
        raise ValueError(f"{path}: not a rn.face_mapping/1 file")
    return data


def resolve(path) -> Path:
    """Project-relative paths in the config become absolute."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
