# Robot Narrator

A live narrator avatar whose face is built from rigid mechanical plates. A webcam tracks the
performer's face; the plates slide, hinge and fan to act it out. Nothing deforms like skin.

**Status: phase 1 (technical prototype) is done and confirmed live on the webcam.
Phase 2 (the production head) is in progress**: the front of the head is built from outlines
traced off both reference sheets (front, side and back views: 48 moving plates on an angular
head form, with the dark cable harness underneath), but it is not yet a match for the drawings:
the neck, collar and jacket are stand-ins, the ears and the top of the head are simplified, and
it has no controls yet. The live demo (Godot, and `blender/robot_01_prototype.blend`) still
uses the phase 1 prototype head.

## Art direction

The head must look like `references/head_anatomy_front.webp`, as close to identical as a 3D
model can be. Specifically:

- plates are **flat and angular**, with straight edges and **real thickness** (visible
  chamfers); never soft or rounded;
- **deep yellow**, with strong contrast between the plates and the dark structure under them;
- that structure is **mechanical: cables, cords and conduits**, not blocks;
- plates **overlap**: some run on underneath others (the housings carry on behind the brows);
- the head is **broad and deep**, not narrow; the faceplate is a flat shield, not a deep block;
- the eyes are **angular and bright**: the hard white shape only, not the glow around it;
- each brown eyebrow is **one piece**;
- the face will be **extremely articulate**, more emotive than the tracked human face;
- sides, back, ears, fins and neck follow the turnaround sheet,
  `references/Golden Android Helmet Turnaround Sheet.png` (front, three-quarter, side, back).

## Try it

| Do this | What happens |
|---|---|
| Double-click `run_avatar.bat` | The Godot avatar window opens and waits for the tracker (UDP 8765). |
| Double-click `run_tracker.bat` | The tracker opens the webcam and its tuning window. Keep a relaxed face for 2 seconds while it calibrates, then perform. |
| `run_live.bat` | Both of the above. |
| In Blender, open `blender/robot_narrator.blend`, press **N** in the 3D view, tab **Robot** | Sliders for every control, a Neutral button, and **Start live link** (UDP 8766) so the same tracker drives Blender. |

**Choosing the camera.** Cameras are picked by name, so the choice survives devices being
plugged in. The default is set in `config/tracker_settings.json` (`camera.name`, currently
`"Elgato Facecam"`).

    run_tracker.bat --list-cameras          the cameras by name (opens none of them)
    run_tracker.bat --camera facecam        use a camera by part of its name, or by number
    press N in the tuning window            switch to the next camera while running

If a camera will not open, another program is usually holding it (Camera Hub, OBS, a call).

No camera needed for a dry run:

    run_tracker.bat --source synthetic      scripted face: brows, blink, wink, smile, head turns
    run_tracker.bat --source image          the sample portrait through the real MediaPipe tracker

Tuning-window keys: **C** recalibrate neutral, **N** next camera, **M** mirror on/off,
**L** landmarks, **Q** quit.

## How it fits together

    webcam -> MediaPipe Face Landmarker -> neutral calibration -> human-to-robot mapping
           -> smoothing -> UDP (localhost) -> avatar runtime -> rigid plate transforms

    tracker/   Python, MediaPipe, OpenCV.  Sends finished robot control values, 30 Hz.
    godot/     Live runtime.  Evaluates the rig definition, drives nodes and bones.
    blender/   Authoring.  Procedural geometry, the rig, drivers, export.
    config/    Everything tunable, as JSON.

There is one intermediate language between the human and the machine: the robot controls
(`BROW_L_RAISE`, `EYE_R_OPEN`, ...). The tracker only ever produces control values; the
runtimes only ever consume them. The manual sliders in Blender and the tracker write the
same `CTRL_FACE` properties.

### The mechanical rig

Every plate is its own rigid object with a **mechanical frame**: a pivot plus slide and
hinge axes, lying on the head surface (+X towards the character's left along the surface,
+Y into the head, +Z up along the surface). Plate motion is defined in that frame, in
`config/face_rig.json`:

    {"control": "BROW_{S}_RAISE", "node": "forehead_{S}", "loc_mm": [0, 0, 4.0], "rot_deg": [3.5, 0, 0], "delay_ms": 40}

- One control moves several plates; one plate sums several controls, then is clamped to its
  own legal range (`nodes` > `limits`). No physics, no collisions: ranges are designed.
- Write the left side only. `{S}` expands to L and R and the right side is mirrored.
- `delay_ms` makes a plate follow late, like a linkage taking up slack (Godot runtime).
- In Blender the frame is the object's parent-inverse matrix, so a plate's own location and
  rotation channels are pure animation, zero at rest. Drivers are generated from the JSON.
- GLB cannot carry drivers and does not need to: it carries geometry and the rest frames.
  Godot re-evaluates the same JSON (`godot/scripts/face_rig.gd`).

After editing `config/face_rig.json`, run `rn.rebuild_rig()` in Blender (see below). It
rebinds the drivers, re-exports, re-imports in Godot and runs the parity test.

### Conventions

- Blender: metres, Z up, the character faces -Y and its left is +X. Godot: Y up, faces +Z.
- Controls are 0 at neutral. Most run 0..1; `EYE_*_OPEN` runs -1 (closed) .. 0 .. +1 (wide).
- Head angles in packets are radians, from the avatar's own point of view: yaw > 0 turns to
  its left, pitch > 0 looks up, roll > 0 tips towards its left shoulder.
- `camera.mirror` (default on): the avatar acts as your reflection, so your left brow moves
  the brow on your left of the screen. Off: your left is the avatar's left.
- MediaPipe's "Left" is the performer's own left (measured, see Findings).

Packet (JSON, UDP, `127.0.0.1:8765` for Godot and `:8766` for Blender):

    {"v": 1, "seq": 412, "timestamp": 1791288000.12, "tracking": true, "calibrating": false,
     "head": {"yaw": 0.12, "pitch": -0.08, "roll": 0.02},
     "face": {"brow_l_raise": 0.42, "brow_r_raise": 0.37, "brow_l_lower": 0.0, "brow_r_lower": 0.03,
              "eye_l_open": -0.1, "eye_r_open": 0.0, "cheek_l_raise": 0.31, "cheek_r_raise": 0.28}}

## Tuning without touching code

| File | What it holds |
|---|---|
| `config/tracker_settings.json` | camera index and size, mirror, UDP targets, smoothing, head gains and limits, tracking-loss behaviour |
| `config/face_mapping.json` | human signal -> robot control: gains, deadzones, which signals feed which control |
| `config/face_rig.json` | robot control -> plate motion: strengths, delays, plate limits, head/neck/chest share |

## Working in Blender

Blender is driven through the Higgsfield connector or by hand from the Python console:

    import sys; sys.path.insert(0, r"C:\Projects\robot_narrator\blender\scripts")
    import rn_blender.pipeline as rn
    rn.build_head()                 # the production head, from head/traced.json (phase 2)
    rn.build_prototype()            # rebuild the phase 1 prototype from scratch (deterministic)
    rn.rebuild_rig()                # after editing config/face_rig.json
    rn.export_for_godot()           # GLB + rig, re-imported and verified in Godot
    rn.save("robot_01_prototype")   # working file + named milestone; previous file goes to backups/

The `.blend` contains a small boot script (`rn_boot.py`) that adds the **Robot** sidebar tab
when the file is opened. All geometry is procedural (`blender/scripts/rn_blender/`).

**How the head is built.**

1. `tools/trace_reference.py` finds every inked panel in each view of both sheets and fits it
   with a straight-edged polygon (`references/trace/<view>_regions.png` shows them numbered).
2. `tools/freeze_trace.py` picks the panels that are plates (by a seed point inside each),
   taking each from the view that shows it best: the anatomy sheet for everything that faces
   forwards, the turnaround's side view for temples, jaw, ears and fins, its back view for the
   back of the skull. It calibrates the views against each other, fits the plates together on
   the unwrapped head (`references/trace/chart_layout.png`), gives each a level, and works out
   where a plate runs on under a higher neighbour. It writes
   `blender/scripts/rn_blender/head/traced.json`, which is then the layout.
3. `rn_common/headform.py` is the head's base form: angular, a flat front and flat swept
   planes each side, measured from the front and side views.
4. `rn.build_head()` lays one rigid, thick, chamfered plate per outline on that form (flat where
   the form is flat, creased where it creases, so neighbours meet edge to edge), stands the
   flat parts proud of it (faceplate, keystone, nose, brows), and lays the cable harness
   underneath. Every plate has its own pivot frame.
   Each plate also carries the drawing's ink line: a dark shell a little larger than the plate,
   turned inside out, so only its far side shows as an outline.
5. `checks.intersections()` lists parts that cut through each other; `tools/compare_reference.py`
   puts the orthographic front render beside the drawing (`renders/head/compare_traced_*.png`).

To try a change without touching the head in an open Blender, build it in a background one:
`blender -b --factory-startup --python blender/scripts/rn_run.py -- trial` renders into
`renders/head/trial/`; `-- publish` renders into `renders/head/` and saves the working file.

## Tests and evidence

`run_tests.bat` runs everything:

- 53 Python tests: rig definition (expansion, mirroring, limits), the tracker chain
  (mapping, calibration, smoothing, head pose, face-lost behaviour, UDP round trip) and
  checks against the real MediaPipe model on a sample portrait.
- Blender/Godot parity: 31 test poses are posed in Blender, exported, re-posed by the Godot
  rig and compared part by part. Current worst error: 0.0002 mm, 0.00002 degrees.

Images in `renders/proto/`:

- `blender_pose_sheet.png`, `godot_pose_sheet.png`: the same ten poses in both.
- `godot_live_sheet.png`: Godot driven live by the scripted face over UDP.
- `godot_photo_sheet.png`: a smiling portrait through the real tracker: cheeks rise, head rocks.
- `tracker_view_sample_portrait.png`: the tuning window.
- `blender_viewport_panel.png`: the Blender slider panel.

## Findings so far (measured, not assumed)

- **Blender driver snapping.** `driver_add()` seeds each driver curve with two keys, and
  Blender snaps values within 1e-4 of a key, so plate motion under 0.1 mm stuck at zero.
  The generator clears the keys (`rig._add_driver`).
- **MediaPipe 1.0.1** returns 52 blendshape scores; the names are recorded in
  `tests/data/mediapipe_blendshape_names.json` and re-checked by the tests.
- **cheekSquintLeft/Right are always 0** with this model. Smile and eye squint lift the cheeks.
- **Resting scores are far from zero** (browDown rests near 0.4 on the sample face in video
  mode) and the model favours one side (eyeSquintLeft 0.50 vs Right 0.28 on a symmetric
  smile). Neutral calibration and the `normalize` option exist for this; per-side trims may
  still be needed once tuned on a real face.
- **Head tilt leaks a little into the brow scores** (about +-0.07 over +-20 degrees).
- **Camera numbers are not stable**: on this machine DirectShow lists a generic USB camera
  first, then the Elgato Facecam, then OBS Virtual Camera. Hence selection by name.
- **The Facecam must be given its format while opening.** Setting the size afterwards
  re-negotiated the stream and cut it to about 17 fps; opened correctly it delivers 1280x720
  at 60 fps (`sources.WebcamSource`).

## Implementation checklist

- [x] **1. Technical prototype**: skull + brow, eyelid, forehead and cheek plates (both sides),
      controls, GLB export, Godot import, MediaPipe tracker, UDP. Confirmed live on the webcam.
- [ ] **2. Head**: *in progress.* Plates traced from the front, side and back views on an
      angular head form, with overlaps and the cable harness. Still to do: neck, collar and
      jacket from the drawings; ears as drawn from the front; the plates over the top of the
      head; the plates of the upper sides; then a view-by-view pass against both sheets.
- [ ] **3. Mechanical face**: 25-35 separate plates with overlaps and clearances.
- [ ] **4. Control rig**: the full control list on `CTRL_FACE`.
- [ ] **5. Manual expressions**: ten test poses and a contact sheet. **Hard gate**: no
      tracking work until these read without labels.
- [ ] **6. Tracking**: full mapping, tuned on the performer's face.
- [ ] **7. Head motion**: neck and chest follow, limits, return to neutral.
- [ ] **8. Speech motion**: microphone envelope on the lower face.
- [ ] **9. Visual finish**: bevels, materials, recess geometry, lighting.
- [ ] **10. Body**: torso, arms, hands, clothing, humanoid rig.
- [ ] **11. Live runtime**: complete Godot application, transparent background, OBS-ready.
- [ ] **12. Polish**: secondary timing, gains, edge cases.

## Layout

    blender/    robot_narrator.blend (working file), robot_NN_*.blend (milestones), backups/
                scripts/rn_blender/   build, rig, export, live link, panel
    godot/      project, scripts, the exported GLB and rig, the parity test
    tracker/    the tracker application
    rn_common/  pure-Python code shared by Blender scripts and tests (rig definition, GLB reader)
    config/     all tunable values
    tests/      Python tests and test data
    references/ the design reference
    renders/    test renders and contact sheets
    tools/      contact-sheet builder, setup script, the Godot binary (not in Git)
    models/     the MediaPipe face model (not in Git)

## Setting up on another machine

Python 3.12, Blender 5.2 and a Godot 4.x Windows build are needed. `tools\setup.ps1` creates
the virtual environment, installs `requirements.txt` and downloads the MediaPipe model and the
sample portrait. Godot is not downloaded: copy `Godot_v4.x-stable_win64.exe` and its
`_console.exe` into `tools\godot`, with an empty file named `_sc_` beside them so Godot keeps
its settings inside this project.
