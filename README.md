# Digital Twin Generator (private)

Built robot -> verified, sim-ready model. Closed product: this repository has
no remote and is never pushed.

It sits on top of two standalone tools:

- **aruco-tag-maker** prints the tag tiles and writes `markers.json`
  (`dictionary`, `tag_size_mm`, `tags: {name: {id}}`). DTG reads that file
  directly; nothing is imported from the tag maker.
- **robo-prober** (v0.2) discovers the controller, arms it and runs the ROM
  sliders (Set ROM 1 / Set ROM 2, slew limits, watchdog). DTG never arms
  anything. It talks to an already-armed controller through the
  `set_pulse` / `armed` API in `dtg.camera`.

## Pipeline

| # | Step | Sensor | Module | Status |
|---|------|--------|--------|--------|
| 1 | Load scan (OBJ/PLY/STL/GLB, mesh or point cloud) | scanner file | `dtg.scan` | real |
| 2 | Scan health check + preview + fix toggles | none | `dtg.health`, `dtg.preview` | real, tuned on bot1 |
| 3 | Tag placement | none | `markers.json` from aruco-tag-maker | real (file input) |
| 4 | **Wiggle**: map each IO channel to a joint (child/parent link) and direction | camera | `dtg.linkage` | real on synthetic logs; live capture not run |
| 5 | **ROM sweep**: joint axis, origin, terminal ranges, µs <-> angle | camera | `dtg.kinematics` | real on synthetic logs; live capture not run |
| 6 | Register tags to the scan | none | `dtg.registration` | real for picked tag centres or 3+ manual points; finding tags in the scan automatically is **stubbed** |
| 7 | Cut the scan into links | none | `dtg.segment` | real (geodesic, tag-seeded); no Scan B motion vote yet |
| 8 | Missing parts | none | `dtg.parts`, `data/servos.json` | real: library servo bodies, stand-ins, CAD snap-in |
| 9 | Export URDF, comms map, report | none | `dtg.export` | real; inertials, MJCF and ros2_control are **stubs** |

The scanner is used only for geometry (Scan A, later Scan B). Wiggle and ROM
use the camera alone.

### Wiggle (before ROM)

The wiring is unknown, so the wiggle comes first. For each IO channel, one at
a time, the controller (already armed by the operator) moves the pulse a
little: base, +delta, base, -delta, base. The default delta is 80 µs and it is
clamped to 150 µs. The channel always goes back to its start pulse, even if
something fails. The camera tracks every tag. `infer_linkage` then works
through these steps:

1. Measure each tag's motion relative to the root (base) tag. Motion counts
   when it beats a noise-scaled threshold.
2. Build each tag's **signature**: the set of channels that move it.
3. Group tags with the same signature that also move rigidly together. Each
   group is one link.
4. Assign each channel's child link: the link with the smallest signature
   that contains the channel. Its parent is the link whose signature is the
   same set minus that channel.
5. Take the direction from the sign of the rotation about the canonical
   axis, which is the axis with its largest component made positive.

It never guesses. Channels where nothing moves come back as `no_response`.
These cases come back as `needs_confirmation` with a reason:

- two channels that move the same child
- a link with no tag between two joints
- a whole link that was hidden during a wiggle
- tags that share a signature but do not move rigidly together

### ROM (camera watching the sliders)

The ROM step runs for each discovered joint. While the operator slides the
servo and presses Set ROM 1 / Set ROM 2, these are recorded and fitted:

- **Q_t**, the child pose in the parent frame, recorded for every settled
  capture
- **axis**: SVD of the rotation vectors of the motion since the reference
  pose
- **origin**: least squares for the fixed point, placed where the axis
  passes closest to the child
- **µs <-> angle**: a straight-line fit, giving rad/µs, direction and the
  zero pulse
- **terminal range 1 / 2**: the joint position at each ROM mark, plus the
  child pose and every child tag pose in the parent frame. These ranges
  become the URDF `lower` and `upper`.

Quality gates mark a joint as `flagged` and never drop it silently:

- hinge cleanliness, s1/s0 < 0.1
- pivot residual
- linearity of angle against pulse
- a measured position at a ROM mark that lags the fitted line, which catches
  end stops and windup
- wiggle and ROM disagreeing on axis or direction
- too little motion

Joint zero is the **reference capture**: the pose the robot was scanned in.

### Health check (bot1 result)

`dtg health ~/Projects/robo-prober/assets/source/bot1.obj --out out/bot1-health`
takes about 18 s on 310k faces. It finds:

- **Turntable:** one support plane, the tilted disc, which is 40% of the
  surface. This includes the star feet and ridges outside the robot
  footprint and the skirt below the disc.
- **Wires:** three thin runs about 96, 59 and 47 mm long, each about 1.9 mm
  thick. They are the servo leads looping around the base.
- **Holes:** 61 boundary loops. 46 are small enough to fill and 15 are large
  openings (unscanned regions, gappy thin parts), which are reported only.
- **Fragments:** 7 small islands.
- **Low confidence:** 3 rough "mush" patches.

All fixes are opt-in and work on a copy:

`--fix planes wires holes fragments` writes `bot1.fixed.stl`. The fixes
remove 120k turntable faces and 23k wire faces, drop the shreds left behind,
and patch 46 holes with 932 faces, which are labelled `repaired`.

Heuristics you should know about:

- A wire is "thinner than 4.5 mm and runs longer than 30 mm". A long, thin,
  straight rail could match this, so check the preview.
- A support plane must be at least 1.5x wider than the robot's footprint, so
  a robot's own flat base plate is kept.

## Quick start

```sh
./run.sh                 # window (Linux/macOS); run.bat on Windows, run.command on macOS Finder
./run.sh health SCAN.obj --out out/health --fix planes wires holes fragments
./run.sh synth --out out/demo/observations.jsonl        # synthetic bot1-like camera log
./run.sh wiggle out/demo/observations.jsonl              # channel -> joint map
./run.sh build out/demo/observations.jsonl --out out/demo/twin --servo ch0=MG996R ch4=MG996R
./run.sh servos                                          # servo library and what is verified
```

The launchers find Python 3.11+, create `.venv`, install the dependencies
when `pyproject.toml` changes, and run. Tests:
`.venv/bin/pip install -e .[test] && .venv/bin/python -m pytest`.

## File formats

- **Camera log** (`*.jsonl`, `dtg.observations/1`). Line 1 is the header:
  source, markers and camera intrinsics. Every other line is one settled
  capture: `{t, phase: reference|wiggle|rom, event, channel, pulse_us,
  poses: {tag_id: T_camera_tag 4x4, metres}, reproj_px}`. The live
  recorder (`dtg.camera.ObservationRecorder`) and the synthetic generator
  write the same format, and `observations.replay()` replays it.
- **Scan tag centres** (`--scan-tags`): `{tag_id: [x, y, z]}` in scan
  coordinates. You need at least 3 tags that do not lie on one line.
- **Manual alignment** (`--manual-points`): `{"scan": [[x,y,z], ...],
  "model": [[x,y,z], ...]}` with at least 3 pairs. Scale is solved, so a
  scan in unknown units comes out in metres.
- **Outputs**:
  - `robot.urdf`: each link frame sits on its joint pivot, joint origins
    are pure translations, axes are in model coordinates, and limits come
    from the terminal ranges.
  - `meshes/<link>_<n>_<provenance>.stl`
  - `comms_map.json`: IO channel -> joint, status, direction, zero pulse,
    µs/rad, ROM 1/2 µs, safe range, servo model.
  - `report.json`: provenance per part (scanned / repaired / library-servo
    / stand-in / CAD), confidence and reasons per joint, flags, and what was
    not generated. It always says `"verified": false`.
  - `wiggle.json`, `kinematics.json`, `health.json` and preview PNGs.

## Servo library

`src/dtg/data/servos.json` holds 11 servos: SG90, MG90S, MG996R, DS3218,
DS3225, LD-20MG, S3003, HS-311, HS-645MG, SG92R and SC-1258TG. The values
were researched on 2026-10-22 and each entry lists its source URLs.

- **Dimensions verified from a manufacturer page:** Hitec HS-311 and
  HS-645MG only.
- **TowerPro SG90, MG90S, MG996R:** the masses come from TowerPro's pages.
  The body dimensions are DTG's reading of the letter-labelled drawings,
  and that mapping is not confirmed.
- **DS3218, DS3225, S3003, SC-1258TG:** taken from servodatabase.com, which
  is third-party. Retailer conflicts are listed in the notes.
- **LD-20MG:** from a distributor listing; the body length is unknown.
- **MG995** was dropped because its manufacturer URL served the MG996R page.

No source gave the shaft offset or the spline height, so those use
class defaults and the report lists each one as an assumption.
`servo_body_mesh` places the case on a fitted joint axis with the top of the
spline at the pivot.

## Test results (2026-10-22)

41 tests pass in about 26 s. The bot1 tests skip when the scan is absent; set
`DTG_BOT1` to point at it. Results on synthetic data (noise 0.4°, 0.5 mm
lateral, 1.2 mm depth, base wobble 0.15°, 1% tag dropout, seeds 0-3):

- **Axis:** error ≤ 1.2°.
- **Pivot:** off the true axis by ≤ 2 mm.
- **Terminal ranges:** within about 1°.
- **Slope:** within 1%.
- **Wiggle map:** exact for all three joints, both unused channels, and the
  scrambled channel numbers and directions.
- **Heavier dropout (5%):** produces flags, never a silent wrong mapping.
- **ArUco pose** on rendered oblique views: 0.8-1° and about 2 mm. A
  frontal view gives the known ~5° ambiguity, so mount tags at about 30°.
- **Segmentation** of a fused synthetic arm scan: 82% of vertices correct.
  The rest is near joint bridges, about 2% left unassigned on purpose.

## Not done yet (honest list)

- **Hardware path:**
  - The live camera capture, the wiggle driver and the ROM hooks in
    `dtg.camera` are written but have **not been run on hardware**.
  - Hooking `rom_sample` / `rom_mark` into robo-prober's ROM window is
    still to do; robo-prober is not modified here.
  - There is no camera calibration wizard. The intrinsics JSON comes from
    any OpenCV calibration.
- **Registration:** tags are not found automatically in the scan
  (`locate_tags_in_scan` raises NotImplementedError). You pick the tag
  centres (`refine_tag_centre` snaps them to the tile face) or use 3-point
  alignment.
- **Segmentation:**
  - Only geodesic splitting from tag seeds is implemented.
  - The Scan B motion vote from the plan is not.
  - The experimental joint-plane guide is off by default. It lowered
    accuracy because the pivot's position along the axis is not observable
    from tag motion.
- **Exports:**
  - Inertials (mass, CoM, inertia) are TODO comments in the URDF.
  - MJCF, ros2_control and the firmware header are stubs.
  - Effort and velocity are placeholders unless a library servo is
    assigned.
- **Health check:** the heuristics were tuned on one scan (bot1). On the
  voxel-staircase synthetic scan, one 49 mm "wire" is a false positive. Low
  confidence comes from roughness only; per-vertex scanner confidence from
  PLY files is not read yet.
- **Inputs not handled:** prismatic joints are fitted, but no synthetic test
  covers them. Continuous rotation, coupled joints and closed loops are
  flagged or unsupported.
- **Verification:** nothing is verified against the physical robot. That
  needs a sim-vs-real overlay check, which is not built.
