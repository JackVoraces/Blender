"""Tracker application.

    python -m tracker                      webcam -> MediaPipe -> UDP, with the tuning window
    python -m tracker --source synthetic   scripted face instead of a camera
    python -m tracker --list-cameras       show which camera indices work
    python -m tracker --no-view --duration 6 --report
                                           headless timed run, prints a JSON summary

    python -m tracker --camera facecam     pick a camera by name (or by number)

Keys in the tuning window: C recalibrate neutral, N next camera, M mirror, L landmarks,
Q or Esc quit.
"""
import argparse
import json
import math
import sys
import time

from . import settings as settings_module
from .pipeline import Pipeline
from .sources import CameraError, SyntheticSource
from .udp_sender import UdpSender, make_packet


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tracker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("webcam", "synthetic", "image"), default="webcam")
    ap.add_argument("--image", default="tests/data/sample_portrait.png",
                    help="image source: the photo to track (default: the sample portrait)")
    ap.add_argument("--no-calibration", action="store_true",
                    help="skip the automatic neutral calibration (raw expression, no neutral offset)")
    ap.add_argument("--hold", default=None, metavar="BEAT",
                    help="synthetic source: hold one scripted expression (e.g. left_brow_up, smile, turn_left)")
    ap.add_argument("--camera", default=None, metavar="NAME_OR_INDEX",
                    help='camera to use, by (part of) its name or by number, e.g. --camera facecam '
                         '(overrides the settings file)')
    ap.add_argument("--settings", default=None, help="path to tracker_settings.json")
    ap.add_argument("--mapping", default=None, help="path to face_mapping.json")
    mirror = ap.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true", default=None)
    mirror.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--no-view", action="store_true", help="run without the tuning window")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--report", action="store_true", help="print a JSON summary when the run ends")
    ap.add_argument("--list-cameras", action="store_true",
                    help="show the cameras by name and exit (opens none of them)")
    return ap.parse_args(argv)


def _next_camera(source, settings):
    """Close the current camera and open the next one that works."""
    from . import cameras
    from .sources import WebcamSource
    count = max(len(cameras.list_names()), 1)
    current = source.index
    source.close()
    for step in range(1, count + 1):
        settings["camera"]["name"] = None
        settings["camera"]["index"] = (current + step) % count
        try:
            return WebcamSource(settings)
        except CameraError as error:
            print(f"tracker: {error}", file=sys.stderr, flush=True)
    raise CameraError("no camera could be opened")


class RunStats:
    """Numbers only: nothing here keeps or stores an image."""

    def __init__(self, names):
        self.frames = 0
        self.tracked = 0
        self.first = None
        self.last = None
        self.lo = {n: 0.0 for n in names}
        self.hi = {n: 0.0 for n in names}
        self.head_lo = [0.0, 0.0, 0.0]
        self.head_hi = [0.0, 0.0, 0.0]
        self.calibrated_at = None

    def add(self, sample, output):
        self.frames += 1
        self.tracked += int(sample.tracking)
        self.first = sample.time if self.first is None else self.first
        self.last = sample.time
        if output.calibrating:
            return
        if self.calibrated_at is None:
            self.calibrated_at = sample.time - self.first
        for name, value in output.face.items():
            self.lo[name] = min(self.lo[name], value)
            self.hi[name] = max(self.hi[name], value)
        for i, angle in enumerate(output.head):
            self.head_lo[i] = min(self.head_lo[i], angle)
            self.head_hi[i] = max(self.head_hi[i], angle)

    def summary(self):
        seconds = (self.last - self.first) if self.frames > 1 else 0.0
        return {
            "frames": self.frames,
            "seconds": round(seconds, 2),
            "fps": round((self.frames - 1) / seconds, 1) if seconds > 0 else 0.0,
            "tracked_fraction": round(self.tracked / self.frames, 3) if self.frames else 0.0,
            "calibration_finished_after_s": None if self.calibrated_at is None else round(self.calibrated_at, 2),
            "control_ranges": {n: [round(self.lo[n], 3), round(self.hi[n], 3)] for n in self.lo},
            "head_range_deg": {axis: [round(math.degrees(self.head_lo[i]), 1), round(math.degrees(self.head_hi[i]), 1)]
                               for i, axis in enumerate(("yaw", "pitch", "roll"))},
        }


def main(argv=None):
    args = parse_args(argv)
    settings = settings_module.load_settings(args.settings)
    mapping = settings_module.load_mapping(args.mapping)
    if args.camera is not None:
        numeric = args.camera.strip().lstrip("-").isdigit()
        settings["camera"]["index"] = int(args.camera) if numeric else settings["camera"]["index"]
        settings["camera"]["name"] = None if numeric else args.camera
    if args.mirror is not None:
        settings["camera"]["mirror"] = args.mirror
    if args.no_calibration:
        settings["calibration"]["auto_start"] = False

    if args.list_cameras:
        from . import cameras
        from .sources import choose_camera
        names = cameras.list_names()
        try:
            chosen = choose_camera(settings["camera"])[0]
        except CameraError as error:
            chosen = None
            print(f"tracker: {error}", file=sys.stderr)
        print("Cameras (use --camera NAME or --camera NUMBER, or set camera.name in config/tracker_settings.json):")
        for index, name in enumerate(names):
            print(f"  {index}: {name}" + ("   <- selected" if index == chosen else ""))
        return 0 if names else 1

    try:
        if args.source == "synthetic":
            source = SyntheticSource(fps=settings["camera"]["fps"], hold=args.hold)
        elif args.source == "image":
            from .sources import ImageSource
            source = ImageSource(settings, settings_module.resolve(args.image), fps=settings["camera"]["fps"])
        else:
            from .sources import WebcamSource
            source = WebcamSource(settings)
    except (CameraError, FileNotFoundError, ValueError) as error:
        print(f"tracker: {error}", file=sys.stderr)
        return 2

    pipeline = Pipeline(settings, mapping)
    sender = UdpSender(settings["udp"]["targets"])
    send_interval = 1.0 / float(settings["udp"]["send_rate_hz"])
    targets_text = "udp -> " + ", ".join(f"{n}:{p}" for n, (_h, p) in zip(sender.names, sender.targets))
    print(f"tracker: {source.description}; {targets_text}; mirror {'on' if pipeline.mirror else 'off'}", flush=True)

    view = None
    view_cfg = settings["debug_view"]
    if view_cfg["enabled"] and not args.no_view:
        from .debug_view import DebugView
        ranges = {name: (lo, hi) for name, (lo, hi, _terms) in pipeline.mapper.controls.items()}
        view = DebugView(ranges, settings["head"]["limit_deg"], view_cfg["max_width"], view_cfg["draw_landmarks"])

    stats = RunStats(pipeline.mapper.names)
    seq = 0
    fps = 0.0
    last_time = None
    next_send = float("-inf")
    started = time.monotonic()
    try:
        while True:
            sample = source.read()
            if sample is None:
                print("tracker: the source stopped delivering frames", file=sys.stderr)
                break
            output = pipeline.process(sample)
            # Send on a steady schedule: every frame when the camera runs at or below the send
            # rate (frame times jitter, so allow a quarter interval early), every other frame at 60 fps.
            if sample.time >= next_send - 0.25 * send_interval:
                note = "calibrating" if output.calibrating else getattr(source, "beat", None)
                sender.send(make_packet(seq, output.tracking, output.calibrating, output.face, output.head, note))
                next_send = max(next_send + send_interval, sample.time + 0.5 * send_interval)
                seq += 1
            stats.add(sample, output)
            if last_time is not None and sample.time > last_time:
                instant = 1.0 / (sample.time - last_time)
                fps = instant if fps == 0.0 else fps + 0.1 * (instant - fps)
            last_time = sample.time

            if view is not None:
                key = view.show(sample, output, fps, pipeline.mirror, (source.description, targets_text))
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("c"), ord("C")):
                    pipeline.recalibrate()
                elif key in (ord("m"), ord("M")):
                    pipeline.set_mirror(not pipeline.mirror)
                elif key in (ord("l"), ord("L")):
                    view.draw_landmarks = not view.draw_landmarks
                elif key in (ord("n"), ord("N")) and args.source == "webcam":
                    source = _next_camera(source, settings)
                    pipeline.recalibrate()          # a new viewpoint needs a new neutral
                    print(f"tracker: switched to {source.description}", flush=True)
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    except CameraError as error:
        print(f"tracker: {error}", file=sys.stderr)
    finally:
        source.close()
        sender.close()
        if view is not None:
            view.close()

    if args.report:
        report = stats.summary()
        report["source"] = source.description
        report["packets_sent"] = sender.sent
        report["mirror"] = pipeline.mirror
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
