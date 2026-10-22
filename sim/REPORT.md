# Virtual bench report

The virtual bench is an end-to-end, simulated hardware test for robo-prober v0.2 and DTG. It runs the real
code paths with no robot and no person present. Regenerate this report with `make sim-mc`
(`python -m sim montecarlo --n 50 --report`).

## Bugs found and fixed (local commits)

**Safety**

1. **robo-prober rom-gui: first-move snap after E-STOP / Disarm with no map loaded** (`b15926e`). The
   "Start at" box kept the old value, so the next Enable jumped the servo 200 us (~20 deg). The box is now
   always refreshed; the rom-gui smoke (`make sim-gui`) now shows a 0.99 deg start (rest offset) and passes.
2. **DTG wiggle did a recovery move after any error**, transport faults included (`cd6f6b3`). It now stops
   the controller (all outputs released) and re-raises, as in robo-prober.

**Correctness: silent wrong channel->joint maps** (each was seen on the bench)

3. One noisy capture on an unused channel mapped it to a joint. Motion must now appear on both sides of the
   wiggle, go back and forth, and have a similar size on each side (`202d127`, `cf6d1ef`).
4. A hidden tag was placed on the static base by a 90th-percentile rigidity test that ignored the frames
   where it moved. The test now uses the worst frame (`202d127`).
5. The filter for "cannot be placed" placeholder links compared an int with frozenset keys and never
   filtered; a placeholder could become a silent parent (`202d127`).

**Robustness: real-camera conditions**

6. DTG's wiggle broke on a freshly armed v0.2 controller (no START) (`cd6f6b3`).
7. Square-tag PnP flipped between its two solutions (a static base tag "moved" 30-45 deg). Corners are now
   averaged, then one solve, and the solution is picked by continuity (`db235fd`, `be37c7f`).
8. Focus: plain Laplacian variance is pure noise at high gain. The sweep uses LoG on averaged frames
   (`658c813`). The focus sweep + lock is new, from the real OAK-D finding (`cd6f6b3`).
9. A hidden root tag blanked whole channels. Another proven-rigid base tag now stands in (`3dbe95d`).
10. Tags hidden at a Set ROM mark dropped the joint. The fit uses the same-pulse sample (`6069249`), or
    extrapolates from the pulse fit, flagged (`02e4c49`).
11. A joint whose parent joint was dropped made a two-root URDF, and `build_twin` crashed with no output.
    The joint is now left out and reported (`93ea96c`).

## What is simulated

| piece | how |
|---|---|
| robot | A 3-joint arm modelled on bot1: cylindrical base with turntable, yaw turret, shoulder pitch, hanging forearm. Link sizes are around the bot1 scan. Hidden truth is drawn per seed: which PCA9685 channel (0-7) drives which joint and which are unused, direction, pulse->angle gain (0.085-0.105 deg/us), centre trim, horn mis-mount (whole spline teeth plus a few degrees), speed, first-order lag, gear backlash, and hard stops 400-850 us either side of 1500 us. Commanding past a stop stalls the servo and logs a "damage risk" event. Released servos go limp where they are. Pure kinematics. |
| controller | robo-prober's **real C++ firmware core** (`rp_core.cpp`), built natively with `bench_main.cpp`, behind a pseudo terminal. The unmodified host API connects to it like a USB serial device. Its PWM outputs drive the virtual servos (`robo_prober.firmware_bench`, added on v0.2). |
| camera | A z-buffer software renderer, 1920x1080 with fx 1541 like the OAK-D at 1080p. Tags come from aruco-tag-maker (DICT_4X4_100, 30 mm black square, its markers.json and bitmaps), stuck on standoffs where the camera sees them at rest. The model adds dim-room shot and read noise at high gain, light varying per run, mains flicker, depth-dependent defocus driven by a lens position, an autofocus that locks on the far wall, and an occasional hand in front. It exposes a `cv2.VideoCapture`-like object, so frames go through DTG's **real** `ArucoTracker`. The noise is calibrated against the real OAK-D bench numbers. |
| operator | Locks focus with DTG's sweep, captures the reference, arms through robo-prober, and runs WIGGLE through DTG's `wiggle_channel` on channels 0-7. It re-runs flagged wiggles (as DTG asks) and confirms only what it can see. ROM runs through robo-prober's `RomSession` exactly as the sliders drive it: Enable at the shown start, slow targets, Widen when the band runs out, and Set ROM 1 / Set ROM 2 4-10 deg short of the stops it can see. DTG's `rom_sample` / `rom_mark` record along the way. Then it probes the firmware's safety behaviour. The rom-gui window gets a separate event-driven smoke test (`make sim-gui`). |
| scoring | DTG's full `build_twin` runs on the recorded log. Scoring compares it with the hidden truth in the true root-tag frame: map, direction, unused channels, axes, pivots, terminal ranges, us<->angle, URDF, and a twin check (random pulses inside the measured safe ranges; tag positions from URDF + comms map vs the robot). Safety is scored too. |

## Observations (not changed)

- Firmware LIMIT bands persist across sessions until reboot. A band widened in rom-gui and never confirmed
  by ROM is still active for the next client; DISARM does not restore 1400-1600 us.
- Esc E-STOP needs the ROM window to have keyboard focus. The red button and the watchdog (a frozen window
  releases in about 1 s) do not.
- Wiring DTG's ROM hooks into a `heartbeat="manual"` Tk thread puts settle + capture close to the 1 s
  watchdog. Run them off the UI thread.
- With one tag per link, the ends of a large ROM are often unseen (the tag turns away or goes behind
  another link). Use 2+ tags per link at different angles.

## What the bench cannot validate

- Real optics: distortion (zero here), rolling shutter, motion blur, glare on printed tiles, real ISP and
  auto-exposure. The noise matches the measured jitter only roughly.
- Real servos: backlash is a simple dead band. No sag, gravity, stall current, heating or horn slip, so "no
  hard-stop contact" only covers stops the operator can see.
- Scan registration on real tags, segmentation of a real scan, and the health check (no scan here).
- Humans: the scripted operator sees the stops perfectly and confirms flagged items from the truth.
- Replays rebuild the same robot, scene and noise draws, but thread timing differs.

## Monte-Carlo run

50 randomized robots, light level mixed, wall time 47 min. Seeds 9, 29, 33 and 42 were re-run after a harness fix: the scripted unplug probe had STARTed a servo at 1500 us where the watchdog probe had left it at 1530 us (a 3 deg jump made by the test, not by robo-prober). Real-time sessions in parallel on one desktop (load ~40 on 32 cores).

**Overall pass (every check below): 4% (2/50)**

### Safety under simulation (real firmware core, real host)

| behaviour | result |
|---|---|
| hard-stop contacts (damage risk) in all sessions | **0** |
| first-move jump at START (deg), per run max | median 2.6, worst 3.0 (threshold 3.0) |
| watchdog release on host silence (s) | median 1.001, max 1.002 |
| release after host unplug / crash (s) | median 1.001, max 1.003 |
| E-STOP release (s) | median 0.000, max 0.003 |
| E-STOP latches until cleared | 100% (50/50) |
| hardware e-stop input latches | 100% (50/50) |
| out-of-band / bad-token / bad-value moves rejected, output unchanged | 100% (50/50) |
| released servos stay put (no motion after release) | 100% (50/50) |
| ARM alone drives no output | 100% (50/50) |
| ARM refused without cutoff / with bad token; START, MOVE refused before ARM | 100% (50/50) |
| PWM edges audited (every output change the firmware made) | 422098 |
| PWM pulses commanding a joint past its hard stop | **0** |
| PWM pulses outside the marked ROM range after Set ROM 1+2 | **0** |
| pulses on unused channels beyond the +-80 us wiggle / default band | **0** |
| firmware slew (us/s, worst 50 ms window per run; cap 250) | median 235, worst 235 |
| MOVE 20 us past a marked ROM end rejected by the firmware, output unchanged | 100% (49/49) |

Hardware cases covered: every run has its own random wiring (3 of channels 0-7, 5 unused); 68 inverted joints and 121 horns mounted one or more spline teeth off across 50 runs.

### Checks

| check | pass rate |
|---|---|
| all_joints_fitted | 52% (26/50) |
| arm_outputs_nothing | 100% (50/50) |
| arming_guarded | 100% (50/50) |
| axis | 66% (33/50) |
| direction | 96% (48/50) |
| estop_latch | 100% (50/50) |
| map_exact | 50% (25/50) |
| no_errors | 100% (50/50) |
| no_first_move_snap | 98% (49/50) |
| no_hard_stop_contact | 100% (50/50) |
| no_silent_wrong_map | 100% (50/50) |
| out_of_band_rejected | 100% (50/50) |
| pivot | 6% (3/50) |
| pwm_never_past_stops | 100% (50/50) |
| released_servos_hold | 100% (50/50) |
| rom_limits_held | 100% (50/50) |
| slew_capped | 100% (50/50) |
| tags_on_right_links | 100% (50/50) |
| terminal_ranges | 72% (36/50) |
| twin_fk | 14% (7/50) |
| unused_channels_only_wiggled | 100% (50/50) |
| urdf_valid | 100% (50/50) |
| us_angle_gain | 78% (39/50) |
| watchdog | 100% (50/50) |
| zero_pulse | 96% (48/50) |

### Channel -> joint map (wiring, direction, unused channels)

| measure | result |
|---|---|
| exact on the first wiggle | 32% (16/50) |
| exact after re-running flagged channels (operator follows DTG's advice) | 50% (25/50) |
| runs still with flagged channels (operator must confirm) | 36% (18/50) |
| runs where the operator re-ran a wiggle | 54% (27/50) |
| **silent wrong mappings** (mapped, wrong child/parent) | **0** |

### Joint accuracy vs ground truth

116 of 150 joints fitted (77% (116/150)).

| metric | n | median | p95 | max | threshold |
|---|---|---|---|---|---|
| axis error (deg) | 116 | 0.79 | 38.57 | 67.00 | 2.0 |
| pivot error, perpendicular to axis (mm) | 116 | 4.22 | 14.66 | 157.39 | 3.0 |
| terminal range error (deg) | 116 | 0.55 | 5.07 | 162.32 | 2.0 |
| us -> angle gain error (%) | 116 | 0.34 | 6.34 | 30.52 | 3.0 |
| zero pulse error (us) | 116 | 1.91 | 10.07 | 73.14 | 15.0 |

Split by what DTG itself said about the joint (a large error on a joint DTG marked `ok` would be the dangerous case):

| DTG status | joints | axis deg (median / max) | pivot mm (median / max) | terminal deg (median / max) | gain % (median / max) | wrong direction |
|---|---|---|---|---|---|---|
| ok | 34 | 0.56 / 3.63 | 2.94 / 9.23 | 0.35 / 2.84 | 0.21 / 2.72 | 0 |
| flagged | 82 | 0.96 / 67.00 | 5.11 / 157.39 | 0.66 / 162.32 | 0.53 / 30.52 | 2 |

| direction correct | 116 | 98% (114/116) | | | all |

**Twin check** (command random pulses inside the measured safe ranges; where URDF + comms map put every tag vs where the robot really puts it):

per-run median 3.6 mm; per-run p95: median 7.8 mm, worst 168.7 mm (threshold p95 < 5.0 mm; 26 runs).

URDF valid (DTG validator + yourdfpy load): 100% (50/50).

### By light level

| light | runs | overall pass | map exact | all joints fitted |
|---|---|---|---|---|
| dim (full well < 35 e-, like the OAK-D bench) | 16 | 0% (0/16) | 56% (9/16) | 56% (9/16) |
| medium (35-80 e-) | 21 | 10% (2/21) | 52% (11/21) | 57% (12/21) |
| bright (> 80 e-) | 13 | 0% (0/13) | 38% (5/13) | 38% (5/13) |

Focus: with autofocus (locked on the wall) a median of 3 tags were detected; after DTG's sweep + lock the lens was within 3 (median) / 12 (worst) steps of the subject's sharp position.

### Failing seeds (replay: `python -m sim replay --seed N`)

| seed | light (e-) | failed checks |
|---|---|---|
| 0 | 34.716 | pivot, twin_fk |
| 1 | 39.637 | map_exact, all_joints_fitted, pivot, twin_fk |
| 2 | 39.281 | map_exact, pivot, twin_fk |
| 3 | 128.039 | axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 4 | 27.304 | axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 5 | 34.794 | axis, pivot, twin_fk |
| 6 | 19.074 | axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 7 | 72.74 | map_exact, all_joints_fitted, axis, pivot, twin_fk |
| 8 | 125.777 | map_exact, all_joints_fitted, pivot, twin_fk |
| 9 | 28.593 | map_exact, all_joints_fitted, pivot, twin_fk |
| 10 | 67.315 | map_exact, pivot, terminal_ranges, twin_fk |
| 11 | 33.294 | pivot, twin_fk |
| 12 | 73.465 | map_exact, all_joints_fitted, pivot, twin_fk untagged: ['turret'] |
| 13 | 20.274 | pivot |
| 14 | 136.554 | pivot, twin_fk |
| 15 | 61.778 | pivot, terminal_ranges, twin_fk |
| 16 | 95.989 | map_exact, all_joints_fitted, pivot, twin_fk untagged: ['forearm'] |
| 18 | 120.576 | map_exact, all_joints_fitted, axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 19 | 45.815 | map_exact, all_joints_fitted, pivot, twin_fk |
| 20 | 36.94 | pivot, twin_fk |
| 21 | 34.347 | axis, pivot, terminal_ranges, us_angle_gain, zero_pulse, twin_fk |
| 22 | 52.616 | map_exact, all_joints_fitted, axis, pivot, terminal_ranges, us_angle_gain, twin_fk untagged: ['forearm'] |
| 23 | 124.373 | pivot |
| 24 | 27.207 | map_exact, all_joints_fitted, axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 25 | 30.221 | map_exact, all_joints_fitted, pivot, twin_fk untagged: ['upper_arm'] |
| 26 | 22.919 | map_exact, all_joints_fitted, pivot, twin_fk |
| 27 | 27.8 | map_exact, all_joints_fitted, pivot, twin_fk untagged: ['forearm'] |
| 28 | 50.684 | pivot |
| 29 | 53.735 | axis, pivot |
| 30 | 25.434 | axis, pivot, us_angle_gain, twin_fk |
| 31 | 154.948 | axis, pivot, terminal_ranges, twin_fk |
| 32 | 23.464 | pivot, twin_fk |
| 33 | 43.93 | direction, axis, pivot, terminal_ranges, twin_fk |
| 34 | 51.423 | map_exact, all_joints_fitted, axis, pivot, terminal_ranges, us_angle_gain, zero_pulse, twin_fk |
| 35 | 41.751 | pivot, twin_fk |
| 36 | 47.183 | axis, pivot, twin_fk |
| 37 | 40.331 | map_exact, all_joints_fitted, axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 38 | 145.092 | map_exact, all_joints_fitted, pivot, twin_fk |
| 39 | 108.925 | map_exact, all_joints_fitted, twin_fk |
| 40 | 50.255 | all_joints_fitted, pivot, twin_fk |
| 42 | 47.166 | pivot |
| 43 | 27.575 | map_exact, all_joints_fitted, pivot, twin_fk |
| 44 | 126.572 | map_exact, all_joints_fitted, pivot, twin_fk, no_first_move_snap |
| 45 | 25.693 | map_exact, all_joints_fitted, pivot, twin_fk |
| 46 | 36.274 | map_exact, all_joints_fitted, pivot, twin_fk untagged: ['turret'] |
| 47 | 91.927 | direction, axis, pivot, terminal_ranges, us_angle_gain, twin_fk |
| 48 | 122.438 | map_exact, all_joints_fitted, pivot, twin_fk |
| 49 | 114.549 | map_exact, all_joints_fitted, pivot, twin_fk |

## Virtual camera vs the real OAK-D bench

seed 1  lens 143  autofocus False
| tag | detected | size px | centre jitter px | obliquity deg |
|---|---|---|---|---|
| base | 30/30 | 77 | 0.29 | 34 |
| base_2 | 30/30 | 75 | 0.71 | 40 |
| turret | 30/30 | 70 | 0.08 | 37 |
| upper_arm | 30/30 | 86 | 0.10 | 21 |
| forearm | 30/30 | 92 | 0.15 | 20 |
| forearm_2 | 30/30 | 92 | 0.12 | 26 |

seed 3  lens 136  autofocus False
| tag | detected | size px | centre jitter px | obliquity deg |
|---|---|---|---|---|
| base | 30/30 | 67 | 0.09 | 25 |
| base_2 | 30/30 | 69 | 0.10 | 27 |
| turret | 30/30 | 68 | 0.15 | 31 |
| upper_arm | 30/30 | 73 | 0.16 | 27 |
| forearm | 30/30 | 59 | 0.28 | 41 |

## Artifacts

- sim-run.gif: `runs/sim-video/seed-0002/sim-run.gif`
- sim-run.mp4: `runs/sim-video/seed-0002/sim-run.mp4`
- sim-still.png: `runs/sim-video/seed-0002/sim-still.png`
