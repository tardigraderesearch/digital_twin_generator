**To use:** `cd digital_twin_generator`, then run `python digital_twin_generator.py`

# Digital Twin Generator

> **Work in progress.** Every step has been exercised end to end on a simulated bench; only the scan health check has run on real hardware so far. See the hardware test status table below before trusting it with a real robot.

Build an editable robot model from a scan and camera-observed joint motion.
The launcher checks Python and Tk, prepares a private `.venv`, asks before
installing dependencies, then opens the window. Nothing installs system-wide.

## The eight steps

| Step | What you do | What it produces |
|---|---|---|
| 1. Scan | Load OBJ/PLY/STL/GLB; review health checks and select fixes. | Preview, health report, optional cleaned copy; original unchanged. |
| 2. Tags | Load the tag maker's `markers.json`. | Named tags and camera preview. |
| 3. Wiggle | Choose controller, type the starting pulse, confirm cutoff access, then arm and wiggle. | Channel-to-link map; uncertain results need an answer. |
| 4. Range of motion | Move each joint to two safe ends and mark them. | Axis, pivot, pulse-to-angle fit and joint limits; hidden-tag warnings. |
| 5. Register | Supply picked scan tag centres or three matching point pairs. | Scan aligned to the camera model. |
| 6. Cut links | Split the scan using the registered tags and linkage. | Separate link meshes. |
| 7. Missing parts | Choose library servos, stand-ins or supplied CAD. | Geometry labelled with its source and assumptions. |
| 8. Export | Review flags and export. | URDF, STL meshes, communications map and report. |

The scanner supplies geometry. The camera observes motion for steps 3 and 4.
The controller API, safety controls, theme and servo database come from
`robo_prober`; the tag manifest comes from `aruco_tag_maker`.
DTG's bench requires explicit arming and a typed start pulse. E-STOP stays
visible; Escape stops motion. The device enforces limits and watchdog behavior.
Simulation exercises these paths but does not establish physical safety.

## Hardware test status

| Area | Passed on real hardware (board, date) | Physical test pending |
|---|---|---|
| Scan health check | Recorded bot1 robot scan, 2026-10-22; no board connection required | Generalization to other scans and scanners |
| Tags and camera preview | — | Live DTG workflow and calibration |
| Camera wiggle | — | Physical channel-to-link mapping |
| Camera range of motion | — | Physical axes, pivots, end marks and hidden-tag handling |
| Tag-to-scan registration | — | Alignment checked against a physical robot |
| Cut links | — | Physical geometry and joint boundaries |
| Missing parts | — | Fit of library bodies and supplied CAD |
| URDF/export | — | Model motion compared with the physical robot |

Only the scan health check has been exercised on real scan data. The remaining
workflow is **physical-test-pending**. A software pass or operator confirmation
is not physical verification; exported reports retain `verified: false`.

## Start with simulation

```sh
python digital_twin_generator.py doctor
make sim-env
sim/.venv/bin/python -m dtg gui --port sim:2
```

The virtual bench additionally needs the sibling `robo_prober` and
`aruco_tag_maker` repositories and a C++ compiler for the firmware simulator.
`make sim-env` prepares a separate `sim/.venv` for development; this is not
required for the normal app launcher. The simulated camera uses rendered tags,
not the connected OAK-D. Select the simulated controller before connecting.

Developer checks, from this repository:

```sh
.venv/bin/python -m pytest -q     # includes GUI tests when a display is available
make sim-test                    # harness tests; slow full session is opt-in
SIM_SLOW=1 make sim-test          # also run the full virtual bench session
sim/.venv/bin/python -m sim gui-walk --seed 2 --screenshots docs/screenshots --out runs/gui-walk
```

The GUI walkthrough drives all eight tabs on the simulated bench. Its scripted
operator uses the simulator's hidden truth to answer ambiguous results and
choose safe motion endpoints. Scan tag picks also come from that truth with
noise added. This checks the assisted workflow, not autonomous reconstruction.
Screenshots are saved in [docs/screenshots](docs/screenshots/).

## Saved work

Sessions save automatically under `~/digital_twin_generator_maps/`, or the
`DTG_MAPS_DIR` override. Reopen one with:

```sh
python digital_twin_generator.py gui --session /path/to/session
```

Each session keeps `session.json`, its tag manifest, camera observations,
health outputs and a `twin/` folder containing `robot.urdf`, meshes,
`comms_map.json` and `report.json`. Source scans remain unchanged.
The GUI and CLI share these session files. Use `--help` on `health`, `tags`,
`wiggle`, `rom`, `register`, `segment`, `parts` or `export` for individual steps.
`confirm` records answers to flagged channels and fitted joints.

## Limits before a physical test

- Automatic tag detection inside the scan is not implemented: provide tag
  centres or matching point pairs. At least three non-collinear pairs are needed.
- Segmentation uses geodesic tag seeds; the proposed second-scan motion vote
  is not implemented. Thin rails can be mistaken for wires by scan repair.
- Servo specifications come from `robo_prober.servo_db`. Their source and
  uncertainty travel into the output; missing dimensions use labelled defaults.
- Mass/inertia blocks, MJCF, ros2_control and firmware header export are unfinished.
  Effort and velocity may be placeholders without a library servo.
- There is no camera calibration wizard. Real camera use needs suitable intrinsics.
- Continuous rotation, coupled joints and closed loops are unsupported or flagged;
  prismatic fitting lacks a full synthetic workflow test.
- No Windows or macOS execution was performed during this Linux handoff.

## How the measurements work

### Wiggle (before ROM)

The wiring is unknown, so the wiggle comes first. For each IO channel, one at
a time, the controller (already armed by the operator) moves the pulse a
little: base, +delta, base, -delta, base. The default delta is 80 µs and it is
clamped to 150 µs. A channel that is not driven yet is first started at its
start pulse (robo-prober's explicit START). The wiggle ends at the start
pulse; if anything fails, the controller is stopped (all outputs released)
rather than moved back. Before the wiggle, `ArucoTracker.lock_focus()` sweeps
the lens and locks manual focus on the robot: on the real bench, autofocus
locked on the background and no tag was detected. The camera tracks every tag. `infer_linkage` then works
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

`python digital_twin_generator.py health /path/to/bot1.obj --out out/bot1-health`
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
