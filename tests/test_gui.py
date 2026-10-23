"""The window on a real display (skipped without one). Only this test's own windows are opened."""
import json
import os
import time

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture()
def root(tmp_path, monkeypatch):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no display")
    monkeypatch.setenv("DTG_MAPS_DIR", str(tmp_path / "maps"))
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def pump(root, app=None, seconds=0.3, until_idle=False, timeout=60):
    end = time.monotonic() + (timeout if until_idle else seconds)
    while time.monotonic() < end:
        root.update()
        if until_idle and app is not None and not app.busy:
            root.update()
            return
        time.sleep(0.02)


def test_tabs_in_pipeline_order_and_window_controls(root):
    from dtg.gui import TwinApp
    app = TwinApp(root)
    pump(root)
    titles = [app.tabs.tab(i, "text") for i in range(app.tabs.index("end"))]
    assert [t.split(" ", 1)[1].rstrip(" ✓!–✗") for t in titles] == [
        "Scan", "Tags", "Wiggle", "Range of motion", "Register", "Cut links", "Missing parts", "Export"]
    assert app.estop_btn.winfo_ismapped() and app.estop_btn.cget("text") == "E-STOP"
    assert app.arm_btn.instate(["disabled"])                 # nothing connected
    assert app.session.root.is_dir() and (app.session.root / "session.json").is_file()
    from robo_prober import theme
    assert root.cget("background") == theme.BG
    app.estop()                                              # nothing connected: says so, no crash
    assert "nothing connected" in app.status.get()


def test_offline_flow_on_a_recorded_log(root, tmp_path):
    from dtg import steps
    from dtg.gui import TwinApp
    from dtg.session import Session
    from dtg.synthetic import generate
    log = generate(seed=0)
    path = log.save(tmp_path / "observations.jsonl")
    (tmp_path / "markers.json").write_text(json.dumps(log.header["markers"]))
    s = Session.new("gui")
    steps.load_tags(s, tmp_path / "markers.json")
    s.import_log(path)
    app = TwinApp(root, session=str(s.root))
    pump(root, app, until_idle=True)
    rows = app.wiggle_table.get_children()
    assert len(rows) == 5 and sum(app.wiggle_table.set(r, "result") == "mapped" for r in rows) == 3
    assert len(app.tag_table.get_children()) == 6
    app.fit_rom()
    pump(root, app, until_idle=True)
    assert all(app.rom_table.set(r, "fit") == "ok" for r in app.rom_table.get_children())
    app.goto("parts")
    pump(root)
    assert len(app.servo_table.get_children()) == 3
    app.servo_table.selection_set(app.servo_table.get_children()[0])
    app.servo_var.set("towerpro-mg996r  (TowerPro MG996R)")
    app.assign_servo()
    app.run_parts()
    pump(root, app, until_idle=True)
    assert any(app.parts_table.set(r, "provenance") == "library-servo" for r in app.parts_table.get_children())
    app.run_export()
    pump(root, app, until_idle=True)
    assert "robot.urdf" in app.export_text.get("1.0", "end")
    assert (s.twin_dir / "robot.urdf").is_file()
    assert app.tabs.tab(7, "text").endswith("✓")


def test_hardware_bar_with_the_emulator(root):
    from dtg.gui import EMULATOR_LABEL, TwinApp
    app = TwinApp(root, port="emulator")
    assert app.source_var.get() == EMULATOR_LABEL
    app.connect()
    pump(root, app, until_idle=True)
    assert app.bench is not None and app.bench.connected and not app.bench.armed
    assert app.channels_var.get().startswith("0-15")
    app.arm()                                             # cutoff not ticked: refused
    assert not app.bench.armed and "cutoff_unconfirmed" in app.status.get()
    app.start_var.set("")
    app.wiggle_all()                                      # no start pulse typed: refused, nothing moves
    assert "bad_start" in app.status.get() and not app.bench.armed
    app.cutoff_var.set(True)
    app.arm()
    assert app.bench.armed
    root.event_generate("<Escape>")                       # Escape is the keyboard E-STOP
    pump(root)
    assert app.bench.estop_latched and not app.bench.armed
    assert app.clear_btn.instate(["!disabled"]) and app.arm_btn.instate(["disabled"])
    app.clear_estop()
    assert not app.bench.estop_latched
    app.close()
