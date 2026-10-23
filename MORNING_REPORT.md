# Digital Twin Generator — 2026-10-23

## Completed

- Finished the simulated scan generator and eight-tab GUI walkthrough that
  were left untracked. The scan now fails visibly if manifold fusion fails,
  rather than quietly substituting overlapping, unfused geometry.
- Connected the walkthrough's lighting option to the simulated bench.
- Corrected the controller-window smoke test for zero-relative ROM sliders;
  settling now times out explicitly instead of silently proceeding.
- Made the GUI walkthrough reject wrong channel maps and simulated safety
  failures even when a syntactically valid URDF was produced.
- Rewrote the README around the actual eight-step workflow, session files,
  launch commands and a hardware test status table.
- Added a screenshot index explaining simulated inputs and operator answers.

## Checks run

| Check | Result |
|---|---|
| Application suite, Linux real display | 78 passed, no skips |
| `make sim-test` | 10 passed; 1 opt-in long-session test skipped |
| Launcher `doctor`, twice | Both reported “All dependencies present. Good to go.” |
| Controller-window smoke, seed 2 | Passed after updating the slider test |
| Smoke E-STOP | Latched and released outputs; measured 0.000 s at reporting precision |
| Smoke frozen-window watchdog | Released outputs after 0.952 s |
| Smoke physical stop contacts | 0 in simulation |

The optional `SIM_SLOW=1` pytest case was not run. The eight-step GUI
walkthrough is a separate end-to-end run, described below. Windows and macOS
were not exercised. Existing private environments were reused; no system
packages were installed and no clean-clone installation was tested.

## Full GUI walkthrough result

Final run: `runs/handoff-final-2026-10-23/gui-walk.json` (seed 2, light 60).
A copy of its measurements is in [docs/simulation-result.json](docs/simulation-result.json).
The command exited 0 after all eight steps. All eight screenshots were
opened and visually reviewed.

- Three joints mapped correctly; five unused channels reported no response.
- Valid URDF with three joints and all eight session stages completed.
- Simulated audit: 0 hard-stop contacts, 0 pulses beyond physical stops,
  0 pulses outside the audited ROM, and 0 excessive unused-channel pulses.
  Firmware bands matched the marked ranges. Maximum slew: 188.7 µs/s.
- Registration: six tag-centre pairs, 0.7 mm RMS.
- Axis error: 0.142–0.455 degrees; pivot error: 0.268–2.956 mm.
- Twin position error across 100 tag samples: median 1.338 mm,
  95th percentile 2.853 mm, maximum 2.960 mm.
- Segmentation left 815 of 36,878 vertices unassigned; those boundaries
  still require review. No claim of perfect segmentation is made.
- The upper-arm second endpoint was hidden at 940 µs. Its fit retained an
  extrapolation warning and the scripted operator confirmed it using truth.
  This is not an independently measured endpoint.

The initial exploratory run exposed a missing ROM-band audit field in the
walkthrough; the final run explicitly reads each simulated controller band
and checks it against the recorded range. The smoke-test slider mismatch
was corrected in DTG's test harness only; robo_prober was unchanged.

## Physical test status and boundaries

Only the scan health check has prior real scan evidence (bot1, 2026-10-22).
Tags, camera wiggle, ROM, registration, link cutting, missing-part fit and
URDF motion remain physical-test-pending. Simulated success is not a physical
verification. Exported reports retain `verified: false`.

The scripted operator uses simulator truth to answer flagged fits, choose
safe endpoint margins and supply scan tag picks. This is an assisted workflow
test, not proof that DTG automatically reconstructs any unknown robot.
The synthetic scan omits realistic holes, cables and unscanned surfaces.

No OAK-D camera or `/dev/ttyACM0` was opened. No changes were made to
robo_prober, aruco_tag_maker, Unreal or the game repositories. Outputs and
source evidence were retained. Work stays local on `refine`; no remote was
added and nothing was pushed.

## Run it

```sh
cd ~/Projects/digital_twin_generator
python digital_twin_generator.py
```

For the simulated bench with its extra dependencies already prepared:

```sh
sim/.venv/bin/python -m dtg gui --port sim:2
```

Screenshots: [all eight tabs](docs/screenshots/README.md).
