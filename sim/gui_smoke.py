"""rom-gui smoke on the virtual bench: robo-prober's real ROM window, driven by
widget events (button invokes, slider variable, Esc), against the real
firmware core over the pty, with the virtual robot on the outputs.

Needs an X display (the window is withdrawn, so nothing appears on screen).
"""
from __future__ import annotations

import time


def run_gui_smoke(seed: int = 0) -> dict:
    import sys
    if sys.platform == "darwin":
        return {"seed": seed, "ok": True, "skipped": "GUI smoke is not run on macOS"}
    import tkinter as tk

    from robo_prober.api import connect
    from robo_prober.rom_gui import RomApp

    from .bench import build_world

    world = build_world(seed)
    j = world.truth.joints[1]                       # the shoulder: gravity-free in the sim, big ROM
    ch = j.channel
    out: dict = {"seed": seed, "channel": ch}
    with world.firmware() as bench:
        root = tk.Tk()
        root.withdraw()
        app = RomApp(root, port=bench.port, channels=[ch], operator="sim", run_timer=False,
                     connect_fn=lambda port: connect(port))

        def settle(target_us: int, timeout_s: float = 15.0) -> None:
            end = time.monotonic() + timeout_s
            while time.monotonic() < end:
                pump(0.1)
                if app.controller.status(ch)["current_us"] == target_us:
                    return

            raise TimeoutError(f"channel {ch} did not settle at {target_us} us")

        def pump(seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                app.tick(0.05)                      # what the window's 50 ms timer does
                root.update()
                time.sleep(0.05)

        app.arm_btn.invoke()                        # refused: cutoff not confirmed
        out["arm_refused_without_cutoff"] = not app.armed
        app.cutoff_var.set(True)
        app.arm_btn.invoke()
        out["armed"] = app.armed
        out["outputs_after_arm"] = dict(bench.active_channels())
        app.rows[ch]["start_var"].set("1500")
        app.rows[ch]["enable"].invoke()
        bench.wait_output(ch, 1500)
        target = 1500 + (1 if j.direction > 0 else -1) * 200   # 200 us toward the upper stop... slowly
        app.rows[ch]["var"].set(target - 1500)
        settle(target)
        out["reached_us"] = app.controller.status(ch)["current_us"]
        app.rows[ch]["rom1"].invoke()
        app.rows[ch]["var"].set(1500 - target)
        settle(1500 - (target - 1500))
        app.rows[ch]["rom2"].invoke()
        out["rom_range_us"] = list(app.roms[ch].range)
        fw = app.controller.status(ch)
        out["firmware_band_us"] = [fw["min_us"], fw["max_us"]]
        out["esc_bound_to_estop"] = bool(root.bind("<Escape>"))   # key events need a focused window
        app.estop_btn.invoke()                      # the big red button
        out["estop_release_s"] = round(bench.wait_released(1.0), 3)
        out["estop_latched"] = app.estop_latched
        app.cutoff_var.set(True)
        app.clear_estop_btn.invoke()
        app.arm_btn.invoke()
        app.rows[ch]["enable"].invoke()             # starts at the saved last position: no jump
        bench.wait_output(ch)
        pump(0.5)
        frozen = time.monotonic()                   # the window freezes: no more timer ticks, no HB
        bench.wait_released(2.0)
        out["frozen_window_release_s"] = round(time.monotonic() - frozen, 3)
        pump(0.2)                                   # the window wakes up and notices the fault
        out["fault_reported"] = bool(app.last_error) and not app.armed
        app.close()
    out["stalls"] = len(world.robot.stalls)
    out["max_start_jump_deg"] = max((s["jump_deg"] for s in world.robot.starts), default=0.0)
    out["ok"] = (out["arm_refused_without_cutoff"] and out["armed"] and not out["outputs_after_arm"]
                 and out["reached_us"] == target and out["rom_range_us"] == sorted((target, 3000 - target))
                 and out["firmware_band_us"] == out["rom_range_us"] and out["estop_latched"]
                 and out["estop_release_s"] < 0.15 and 0.9 <= out["frozen_window_release_s"] <= 1.3
                 and out["fault_reported"] and out["esc_bound_to_estop"] and out["stalls"] == 0
                 and out["max_start_jump_deg"] < 3.0)
    return out
