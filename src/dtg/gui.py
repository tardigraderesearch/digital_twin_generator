"""The window: one tab per pipeline step, in order.

    1 Scan  2 Tags  3 Wiggle  4 Range of motion  5 Register  6 Cut links  7 Missing parts  8 Export

All logic lives in plain modules (``dtg.steps``, ``dtg.hardware``); this file
wires widgets to them and runs slow work in one worker thread so the window
stays responsive. Everything is saved to the session folder as it happens
(no save dialogs in the main flow). The theme is robo-prober's (shared with
aruco-tag-maker): dark, green accent.

Hardware safety: the controller bar and its E-STOP stay visible on every tab.
E-STOP (button or Escape) goes straight to the controller, even while a job
runs. Arming needs the "power cutoff within reach" tick; the first pulse of
every output is the start pulse the operator typed.
"""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, ttk

from . import __version__

ESTOP_BG = "#d50000"
ESTOP_ACTIVE = "#ff1744"
SIM_LABEL = "Simulated bench (seed {seed}, no hardware)"
EMULATOR_LABEL = "Built-in emulator (no camera, no robot)"
SCAN_TYPES = [("Scans", "*.obj *.ply *.stl *.glb *.gltf *.off"), ("All files", "*")]
FIX_LABELS = [("planes", "Remove floor / turntable"), ("wires", "Strip wires"),
              ("holes", "Fill small holes"), ("fragments", "Drop fragments")]
STATUS_MARK = {"done": "✓", "flagged": "!", "skipped": "–", "failed": "✗", "pending": ""}
HORN_WARNING = ("The first pulse moves each servo from wherever it is to the start pulse. Type the pulse it is "
                "most likely at now; a horn mounted off-centre still jumps.")


class TwinApp:
    def __init__(self, root: tk.Tk, *, session=None, port: str | None = None, run_timer: bool = True,
                 sim_seed: int = 0, sim_light: float = 60.0):
        from robo_prober import theme
        from .session import STEPS, Session
        self.root, self.theme = root, theme
        self.steps_meta = STEPS
        root.title(f"Digital Twin Generator {__version__}")
        root.minsize(1000, 720)
        root.geometry("1360x900")
        self._events: queue.Queue = queue.Queue()
        self.busy = False
        self.bench = None
        self.cache: dict = {}
        self.wiggle_res = None
        self.state = None
        self.scan_report = None
        self.health_image = None
        self._images: dict = {}
        self.queue_channels: list[int] = []
        self.stop_flag = threading.Event()
        self.rom_joint: int | None = None
        self.sim_seed = sim_seed
        self.sim_light = sim_light
        self.session = Session.open(session) if session else Session.new("robot")
        self.status = tk.StringVar(value="")
        self._vars()
        self._build()
        if port:
            self.source_var.set(SIM_LABEL.format(seed=port.split(":", 1)[1]) if port.startswith("sim:")
                                else SIM_LABEL.format(seed=sim_seed) if port == "sim"
                                else EMULATOR_LABEL if port == "emulator" else port)
        self.load_session_into_widgets()
        self.goto(self._first_open_step())
        if run_timer:
            root.after(60, self._poll)

    # ================================================================ variables
    def _vars(self):
        self.session_var = tk.StringVar()
        self.scan_var = tk.StringVar()
        self.fix_vars = {k: tk.BooleanVar(value=False) for k, _ in FIX_LABELS}
        self.markers_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.camera_var = tk.StringVar(value="0")
        self.calib_var = tk.StringVar()
        self.cutoff_var = tk.BooleanVar(value=False)
        self.hw_var = tk.StringVar(value="Not connected. Steps 3 and 4 need the controller and the camera.")
        self.start_var = tk.StringVar(value="")
        self.delta_var = tk.StringVar(value="80")
        self.channels_var = tk.StringVar(value="")
        self.gpio_var = tk.BooleanVar(value=False)
        self.auto_next_var = tk.BooleanVar(value=True)
        self.rom_pos_var = tk.StringVar(value="Start the ROM session, pick a joint, press Enable.")
        self.rom_step_var = tk.StringVar(value="40")
        self.rom_warn_var = tk.StringVar(value="")
        self.reg_var = tk.StringVar(value="Not registered yet.")
        self.xyz_var = tk.StringVar()
        self.servo_var = tk.StringVar()
        self.standins_var = tk.BooleanVar(value=True)
        self.cad_link_var = tk.StringVar()
        self.cad_path_var = tk.StringVar()
        self.cad_points_var = tk.StringVar()
        self.export_var = tk.StringVar(value="")

    # ================================================================ layout
    def _build(self):
        root, theme = self.root, self.theme
        style = theme.apply(root)
        style.configure("Hint.TLabel", foreground=theme.ACCENT, font=("TkDefaultFont", 11))
        style.configure("Warn.TLabel", foreground=theme.ERROR, font=("TkDefaultFont", 10, "bold"))
        style.configure("Pos.TLabel", foreground=theme.ACCENT, font=("TkDefaultFont", 20, "bold"))
        header = ttk.Frame(root, padding=(12, 10, 12, 0))
        header.pack(fill="x")
        ttk.Label(header, text="Digital Twin Generator", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.session_var, style="Muted.TLabel").pack(side="left", padx=12)
        ttk.Button(header, text="✕", style="Close.TButton", width=3, command=self.request_close).pack(side="right")
        ttk.Button(header, text="▢", style="Chrome.TButton", width=3,
                   command=self.toggle_fullscreen).pack(side="right", padx=(0, 6))
        ttk.Button(header, text="–", style="Chrome.TButton", width=3, command=root.iconify).pack(side="right", padx=(0, 6))
        ttk.Button(header, text="Open session…", command=self.open_session).pack(side="right", padx=(0, 12))
        ttk.Button(header, text="New session", command=self.new_session).pack(side="right", padx=(0, 6))
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: (self.toggle_fullscreen(False), self.estop()))
        root.bind("<Control-q>", lambda e: self.request_close())
        root.protocol("WM_DELETE_WINDOW", self.request_close)

        outer = ttk.Frame(root, padding=(12, 8, 12, 8))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        self._hardware_bar(outer)
        self.tabs = ttk.Notebook(outer)
        self.tabs.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.tab_frames = {}
        builders = {"scan": self._tab_scan, "tags": self._tab_tags, "wiggle": self._tab_wiggle, "rom": self._tab_rom,
                    "register": self._tab_register, "segment": self._tab_segment, "parts": self._tab_parts,
                    "export": self._tab_export}
        for key, title, hint in self.steps_meta:
            frame = ttk.Frame(self.tabs, padding=10)
            self.tabs.add(frame, text=title)
            frame.columnconfigure(0, weight=1)
            ttk.Label(frame, text=hint, style="Hint.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
            body = ttk.Frame(frame)
            body.grid(row=1, column=0, sticky="nsew")
            frame.rowconfigure(1, weight=1)
            self.tab_frames[key] = body
            builders[key](body)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())
        bar = ttk.Frame(outer)
        bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right")
        ttk.Label(bar, textvariable=self.status, style="Muted.TLabel").pack(side="left", fill="x", expand=True)
        root.bind("<space>", self._key_space)
        for key in ("y", "Y"):
            root.bind(f"<{key}>", lambda e: self._key_answer("confirmed"))
        for key in ("n", "N"):
            root.bind(f"<{key}>", lambda e: self._key_answer("rejected"))

    def _hardware_bar(self, outer):
        box = ttk.LabelFrame(outer, text="Controller + camera (steps 3 and 4)", padding=(10, 6))
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Controller").grid(row=0, column=0, sticky="w")
        self.source_box = ttk.Combobox(box, textvariable=self.source_var, width=40)
        self.source_box.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(box, text="Refresh", command=self.refresh_sources).grid(row=0, column=2)
        ttk.Label(box, text="Camera").grid(row=0, column=3, sticky="e", padx=(12, 4))
        ttk.Entry(box, textvariable=self.camera_var, width=6).grid(row=0, column=4)
        ttk.Label(box, text="Calibration").grid(row=0, column=5, sticky="e", padx=(12, 4))
        ttk.Entry(box, textvariable=self.calib_var, width=22).grid(row=0, column=6)
        ttk.Button(box, text="…", width=2, command=lambda: self._browse(self.calib_var, [("JSON", "*.json")])
                   ).grid(row=0, column=7, padx=(4, 0))
        row = ttk.Frame(box)
        row.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        self.connect_btn = ttk.Button(row, text="Connect (reads only)", style="Accent.TButton", command=self.connect)
        self.connect_btn.pack(side="left")
        ttk.Checkbutton(row, text="Physical power cutoff is within reach", variable=self.cutoff_var).pack(side="left", padx=12)
        self.arm_btn = ttk.Button(row, text="Arm", command=self.arm)
        self.arm_btn.pack(side="left")
        self.disarm_btn = ttk.Button(row, text="Disarm", command=self.disarm)
        self.disarm_btn.pack(side="left", padx=6)
        self.clear_btn = ttk.Button(row, text="Clear e-stop", command=self.clear_estop)
        self.clear_btn.pack(side="left")
        self.disconnect_btn = ttk.Button(row, text="Disconnect", command=self.disconnect)
        self.disconnect_btn.pack(side="left", padx=6)
        ttk.Label(row, textvariable=self.hw_var, style="Muted.TLabel", wraplength=520,
                  justify="left").pack(side="left", padx=10, fill="x", expand=True)
        self.estop_btn = tk.Button(box, text="E-STOP", command=self.estop, bg=ESTOP_BG, fg="white",
                                   activebackground=ESTOP_ACTIVE, activeforeground="white", highlightthickness=2,
                                   highlightbackground="white", font=("TkDefaultFont", 18, "bold"), width=8,
                                   height=2, relief="raised", bd=4)
        self.estop_btn.grid(row=0, column=8, rowspan=2, sticky="ne", padx=(12, 0))
        self.refresh_sources()
        self._hw_buttons()

    # ---------------------------------------------------------------- tab 1
    def _tab_scan(self, t):
        t.columnconfigure(1, weight=1)
        t.rowconfigure(2, weight=1)
        ttk.Label(t, text="Scan (OBJ/PLY/STL/GLB)", style="Big.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(t, textvariable=self.scan_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(t, text="Browse…", command=lambda: self._browse(self.scan_var, SCAN_TYPES, self.run_health)
                   ).grid(row=0, column=2)
        bar = ttk.Frame(t)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Button(bar, text="Run health check", style="Accent.TButton", command=self.run_health).pack(side="left")
        for key, text in FIX_LABELS:
            ttk.Checkbutton(bar, text=text, variable=self.fix_vars[key]).pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Write cleaned copy", command=lambda: self.run_health(fix=True)).pack(side="right")
        body = ttk.Frame(t)
        body.grid(row=2, column=0, columnspan=3, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.health_img = ttk.Label(body, text="The preview appears here, problem regions coloured.", anchor="center")
        self.health_img.grid(row=0, column=0, sticky="nsew")
        self.checklist = self._text(body, 0, 1)

    # ---------------------------------------------------------------- tab 2
    def _tab_tags(self, t):
        t.columnconfigure(1, weight=1)
        t.rowconfigure(2, weight=1)
        ttk.Label(t, text="markers.json", style="Big.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(t, textvariable=self.markers_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(t, text="Browse…", command=lambda: self._browse(
            self.markers_var, [("markers.json", "*.json"), ("All files", "*")], self.load_tags)).grid(row=0, column=2)
        bar = ttk.Frame(t)
        bar.grid(row=1, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Button(bar, text="Load", style="Accent.TButton", command=self.load_tags).pack(side="left")
        self.sim_tags_btn = ttk.Button(bar, text="Use the simulated bench's tags", command=self.load_sim_tags)
        self.sim_tags_btn.pack(side="left", padx=8)
        body = ttk.Frame(t)
        body.grid(row=2, column=0, columnspan=3, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self.tag_table = self._table(body, ("id", "name", "link (after the wiggle)", "seen by the camera"),
                                     (60, 160, 220, 140), 0, 0)
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.tags_text = self._text(right, 0, 0)
        self.tags_text.configure(height=9)
        self.tags_cam = ttk.Label(right, text="Connect to see which tags the camera finds.", anchor="center")
        self.tags_cam.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    # ---------------------------------------------------------------- tab 3
    def _tab_wiggle(self, t):
        t.columnconfigure(0, weight=1)
        t.rowconfigure(3, weight=1)
        r1 = ttk.Frame(t)
        r1.grid(row=0, column=0, sticky="ew")
        ttk.Label(r1, text="Start pulse (µs)", style="Big.TLabel").pack(side="left")
        ttk.Entry(r1, textvariable=self.start_var, width=7).pack(side="left", padx=6)
        ttk.Button(r1, text="Use 1500 (centre)", command=lambda: self.start_var.set("1500")).pack(side="left")
        ttk.Label(r1, text="Wiggle ± (µs)").pack(side="left", padx=(16, 4))
        ttk.Spinbox(r1, from_=20, to=150, increment=10, textvariable=self.delta_var, width=5).pack(side="left")
        ttk.Label(r1, text="Outputs").pack(side="left", padx=(16, 4))
        ttk.Entry(r1, textvariable=self.channels_var, width=16).pack(side="left")
        ttk.Checkbutton(r1, text="Include GPIO pins", variable=self.gpio_var,
                        command=self._fill_channels).pack(side="left", padx=8)
        ttk.Label(t, text=HORN_WARNING, style="Warn.TLabel", wraplength=1200, justify="left").grid(
            row=1, column=0, sticky="w", pady=(4, 4))
        r2 = ttk.Frame(t)
        r2.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(r2, text="Discover (reads only)", command=self.discover).pack(side="left")
        self.wiggle_btn = ttk.Button(r2, text="Arm + wiggle all", style="Accent.TButton", command=self.wiggle_all)
        self.wiggle_btn.pack(side="left", padx=6)
        ttk.Button(r2, text="Wiggle next (Space)", command=self.wiggle_next).pack(side="left")
        ttk.Button(r2, text="Re-wiggle flagged", command=self.rewiggle_flagged).pack(side="left", padx=6)
        ttk.Button(r2, text="Stop after this one", command=self.stop_flag.set).pack(side="left")
        ttk.Checkbutton(r2, text="Next starts by itself", variable=self.auto_next_var).pack(side="left", padx=8)
        ttk.Button(r2, text="Use a recorded log…", command=self.use_recorded_log).pack(side="right")
        body = ttk.Frame(t)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.wiggle_table = self._table(left, ("output", "what", "result", "joint", "parent -> child", "dir",
                                               "your answer", "why"),
                                        (60, 110, 130, 110, 190, 40, 100, 380), 0, 0)
        ans = ttk.Frame(left)
        ans.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(ans, text="Selected row:").pack(side="left")
        ttk.Button(ans, text="Yes, that is right (Y)", command=lambda: self.answer_channel("confirmed")).pack(side="left", padx=6)
        ttk.Button(ans, text="No (N)", command=lambda: self.answer_channel("rejected")).pack(side="left")
        ttk.Button(ans, text="Clear answer", command=lambda: self.answer_channel(None)).pack(side="left", padx=6)
        ttk.Button(ans, text="Wiggle this one again", command=self.rewiggle_selected).pack(side="left")
        self.cam_img = ttk.Label(body, text="Camera view appears here after Connect.", anchor="center")
        self.cam_img.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    # ---------------------------------------------------------------- tab 4
    def _tab_rom(self, t):
        t.columnconfigure(0, weight=1)
        t.rowconfigure(1, weight=1)
        top = ttk.Frame(t)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Start ROM session (arms)", style="Accent.TButton", command=self.start_rom).pack(side="left")
        ttk.Button(top, text="Enable selected", command=self.rom_enable).pack(side="left", padx=6)
        ttk.Button(top, text="Fit (analyse the captures)", command=self.fit_rom).pack(side="left")
        ttk.Button(top, text="Done: disarm", command=self.disarm).pack(side="left", padx=6)
        self.rom_table = self._table(t, ("output", "joint", "start", "ROM 1", "ROM 2", "tags at marks", "fit",
                                         "limits (deg)", "your answer", "why"),
                                     (60, 110, 60, 70, 70, 140, 70, 130, 100, 360), 1, 0)
        ctl = ttk.LabelFrame(t, text="Selected joint", padding=(10, 6))
        ctl.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(ctl, textvariable=self.rom_pos_var, style="Pos.TLabel").grid(row=0, column=0, columnspan=8, sticky="w")
        for i, (text, sign) in enumerate((("◀◀", -2), ("◀", -1), ("▶", 1), ("▶▶", 2))):
            ttk.Button(ctl, text=text, width=4, command=lambda s=sign: self.rom_step(s)).grid(row=1, column=i, padx=2)
        ttk.Label(ctl, text="step µs").grid(row=1, column=4, padx=(10, 2))
        ttk.Spinbox(ctl, from_=5, to=100, increment=5, textvariable=self.rom_step_var, width=5).grid(row=1, column=5)
        ttk.Button(ctl, text="Set ROM 1", style="Accent.TButton", command=lambda: self.rom_mark(1)).grid(row=1, column=6, padx=(16, 4))
        ttk.Button(ctl, text="Set ROM 2", style="Accent.TButton", command=lambda: self.rom_mark(2)).grid(row=1, column=7, padx=4)
        ttk.Button(ctl, text="Widen band", command=self.rom_widen).grid(row=1, column=8, padx=4)
        ttk.Button(ctl, text="Fit looks right (Y)", command=lambda: self.answer_joint("confirmed")).grid(row=1, column=9, padx=(16, 4))
        ttk.Button(ctl, text="Fit is wrong (N)", command=lambda: self.answer_joint("rejected")).grid(row=1, column=10)
        ttk.Label(ctl, textvariable=self.rom_warn_var, style="Warn.TLabel", wraplength=1200,
                  justify="left").grid(row=2, column=0, columnspan=11, sticky="w", pady=(4, 0))
        self.rom_table.bind("<<TreeviewSelect>>", lambda e: self._rom_selected())

    # ---------------------------------------------------------------- tab 5
    def _tab_register(self, t):
        t.columnconfigure(0, weight=1)
        t.rowconfigure(1, weight=1)
        bar = ttk.Frame(t)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Load tag centres (JSON)…", command=lambda: self.load_registration("scan_tags")).pack(side="left")
        ttk.Button(bar, text="Load 3-point alignment (JSON)…", command=lambda: self.load_registration("manual_points")
                   ).pack(side="left", padx=6)
        ttk.Button(bar, text="Register", style="Accent.TButton", command=self.run_register).pack(side="left", padx=6)
        ttk.Label(bar, textvariable=self.reg_var, style="Muted.TLabel").pack(side="left", padx=8)
        self.reg_table = self._table(t, ("tag", "link", "camera model x, y, z (mm)", "in the scan x, y, z"),
                                     (60, 140, 260, 260), 1, 0)
        row = ttk.Frame(t)
        row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(row, text="Selected tag's centre in the scan (x y z, scan units):").pack(side="left")
        ttk.Entry(row, textvariable=self.xyz_var, width=30).pack(side="left", padx=6)
        ttk.Button(row, text="Set", command=self.set_tag_centre).pack(side="left")

    # ---------------------------------------------------------------- tab 6
    def _tab_segment(self, t):
        t.columnconfigure(0, weight=3)
        t.columnconfigure(1, weight=2)
        t.rowconfigure(1, weight=1)
        ttk.Button(t, text="Cut the scan into links", style="Accent.TButton",
                   command=self.run_segment).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.seg_img = ttk.Label(t, text="Links appear here, one colour each.", anchor="center")
        self.seg_img.grid(row=1, column=0, sticky="nsew")
        self.seg_text = self._text(t, 1, 1)

    # ---------------------------------------------------------------- tab 7
    def _tab_parts(self, t):
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)
        t.rowconfigure(1, weight=1)
        bar = ttk.Frame(t)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(bar, text="Servo for the selected joint").pack(side="left")
        self.servo_box = ttk.Combobox(bar, textvariable=self.servo_var, width=34)
        self.servo_box.pack(side="left", padx=6)
        self.servo_box.bind("<KeyRelease>", self._servo_typeahead)
        ttk.Button(bar, text="Assign", command=self.assign_servo).pack(side="left")
        ttk.Checkbutton(bar, text="Stand-ins where nothing else", variable=self.standins_var).pack(side="left", padx=12)
        ttk.Button(bar, text="Build parts", style="Accent.TButton", command=self.run_parts).pack(side="left", padx=6)
        self.servo_table = self._table(t, ("output", "joint", "parent link", "servo"), (60, 120, 120, 260), 1, 0)
        self.parts_table = self._table(t, ("link", "provenance", "faces", "note"), (110, 110, 70, 420), 1, 1)
        cad = ttk.LabelFrame(t, text="CAD part (STL in mm, snapped by 3+ matching points)", padding=(8, 4))
        cad.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(cad, text="link").pack(side="left")
        self.cad_link_box = ttk.Combobox(cad, textvariable=self.cad_link_var, width=14)
        self.cad_link_box.pack(side="left", padx=4)
        ttk.Entry(cad, textvariable=self.cad_path_var, width=26).pack(side="left", padx=4)
        ttk.Button(cad, text="STL…", command=lambda: self._browse(self.cad_path_var, [("STL", "*.stl")])).pack(side="left")
        ttk.Entry(cad, textvariable=self.cad_points_var, width=26).pack(side="left", padx=4)
        ttk.Button(cad, text="Points JSON…", command=lambda: self._browse(self.cad_points_var, [("JSON", "*.json")])
                   ).pack(side="left")
        ttk.Button(cad, text="Add", command=self.add_cad).pack(side="left", padx=6)

    # ---------------------------------------------------------------- tab 8
    def _tab_export(self, t):
        t.columnconfigure(0, weight=1)
        t.rowconfigure(2, weight=1)
        bar = ttk.Frame(t)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Export URDF + comms map + report", style="Accent.TButton",
                   command=self.run_export).pack(side="left")
        ttk.Button(bar, text="Open the output folder", command=self.open_out).pack(side="left", padx=6)
        ttk.Label(t, textvariable=self.export_var, style="Warn.TLabel", wraplength=1200,
                  justify="left").grid(row=1, column=0, sticky="w", pady=6)
        self.export_text = self._text(t, 2, 0)

    # ================================================================ widgets helpers
    def _text(self, parent, row, col):
        th = self.theme
        w = tk.Text(parent, wrap="word", bg=th.SURFACE, fg=th.TEXT, insertbackground=th.ACCENT,
                    relief="flat", highlightthickness=1, highlightbackground=th.BORDER, padx=8, pady=6, height=10)
        w.grid(row=row, column=col, sticky="nsew", padx=(8 if col else 0, 0))
        return w

    def _table(self, parent, cols, widths, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky="nsew", padx=(8 if col else 0, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        tv = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        for c, w in zip(cols, widths):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="w", stretch=c == cols[-1])
        tv.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        sb.grid(row=0, column=1, sticky="ns")
        tv.configure(yscrollcommand=sb.set)
        tv.tag_configure("flag", foreground=self.theme.ERROR)
        tv.tag_configure("ok", foreground=self.theme.ACCENT)
        return tv

    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)

    def _show_image(self, label, path_or_image, key, max_w=None, max_h=None):
        from PIL import Image, ImageTk
        img = path_or_image if hasattr(path_or_image, "size") else Image.open(path_or_image)
        self.root.update_idletasks()
        w = max_w or max(320, label.winfo_width() or 700)
        h = max_h or max(240, label.winfo_height() or 500)
        scale = min(w / img.width, h / img.height)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        self._images[key] = ImageTk.PhotoImage(img, master=self.root)
        label.configure(image=self._images[key], text="")

    def _browse(self, var, types, then=None):
        path = filedialog.askopenfilename(filetypes=types)
        if path:
            var.set(path)
            if then:
                then()

    def _selected(self, table):
        sel = table.selection()
        return sel[0] if sel else None

    # ================================================================ worker
    def _work(self, label, fn, done=None):
        if self.busy:
            self.status.set("Busy: wait for the current job (E-STOP always works).")
            return False
        self.busy = True
        self.status.set(label + " …")
        self.progress.start(12)

        def target():
            try:
                self._events.put(("done", done, fn()))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self._events.put(("error", exc, traceback.format_exc()))
        threading.Thread(target=target, daemon=True).start()
        return True

    def _post(self, fn, *args):
        """Run fn(*args) on the Tk thread (from the worker)."""
        self._events.put(("call", fn, args))

    def _say(self, msg):
        self._events.put(("status", msg, None))

    def _poll(self):
        try:
            while True:
                kind, a, b = self._events.get_nowait()
                if kind == "status":
                    self.status.set(a)
                elif kind == "call":
                    a(*b)
                elif kind == "done":
                    self.busy = False
                    self.progress.stop()
                    if a is not None:
                        a(b)
                    self._hw_buttons()
                    self._refresh_tab_titles()
                else:
                    self.busy = False
                    self.progress.stop()
                    reason = getattr(a, "reason", None)
                    self.status.set(f"Problem{f' ({reason})' if reason else ''}: {a}")
                    self.root.after(0, self._hw_buttons)
                    print(b)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    # ================================================================ session
    def load_session_into_widgets(self):
        s = self.session
        self.session_var.set(f"session: {s.root}")
        if s.input("scan"):
            self.scan_var.set(s.input("scan"))
            for k in self.fix_vars:
                self.fix_vars[k].set(k in (s.input("fixes") or []))
            img = s.path("health", "health.png")
            if img.is_file():
                self.root.after(200, lambda: self._show_image(self.health_img, img, "health"))
            hj = s.path("health", "health.json")
            if hj.is_file():
                import json
                self._set_text(self.checklist, s.step("scan").get("summary", "") + "\n\n" + "\n".join(
                    f"- {f.get('kind')}: {f.get('message', '')}" for f in json.loads(hj.read_text()).get("findings", [])))
        if s.markers_path.is_file():
            self.markers_var.set(s.input("markers_source") or str(s.markers_path))
            self._fill_tags()
        if s.input("wiggle_start_us"):
            self.start_var.set(str(s.input("wiggle_start_us")))
        if s.has_log():
            self._work("Reading the camera log", lambda: __import__("dtg.steps").steps.analyse_wiggle(s),
                       self._wiggle_done)
        self.standins_var.set(bool(s.input("stand_ins", True)))
        self._refresh_tab_titles()

    def _first_open_step(self):
        for key, _, _ in self.steps_meta:
            if self.session.step(key)["status"] in ("pending", "flagged"):
                return key
        return "export"

    def goto(self, key):
        keys = [k for k, _, _ in self.steps_meta]
        self.tabs.select(keys.index(key))

    def _refresh_tab_titles(self):
        for i, (key, title, _) in enumerate(self.steps_meta):
            mark = STATUS_MARK.get(self.session.step(key)["status"], "")
            self.tabs.tab(i, text=f"{title} {mark}".strip())

    def _tab_changed(self):
        key = self.steps_meta[self.tabs.index("current")][0]
        if key == "register":
            self._fill_register()
        elif key == "parts":
            self._fill_servos()
        hint = self.steps_meta[self.tabs.index("current")][2]
        if not self.busy:
            self.status.set(self.session.step(key).get("summary") or hint)

    def new_session(self):
        from .session import Session
        if self.busy:
            return
        self.session = Session.new("robot")
        self.cache.clear()
        self.wiggle_res = self.state = None
        self._clear_all()
        self.load_session_into_widgets()
        self.goto("scan")

    def open_session(self):
        from .session import Session, maps_dir
        path = filedialog.askdirectory(initialdir=str(maps_dir()))
        if path:
            self.session = Session.open(path)
            self.cache.clear()
            self._clear_all()
            self.load_session_into_widgets()
            self.goto(self._first_open_step())

    def _clear_all(self):
        for tv in (self.tag_table, self.wiggle_table, self.rom_table, self.reg_table, self.servo_table, self.parts_table):
            tv.delete(*tv.get_children())
        for w in (self.checklist, self.tags_text, self.seg_text, self.export_text):
            self._set_text(w, "")

    # ================================================================ step 1
    def run_health(self, fix: bool = False):
        path = self.scan_var.get().strip()
        if not path:
            self.status.set("Pick a scan first (Browse…).")
            return
        fixes = [k for k, v in self.fix_vars.items() if v.get()] if fix else []
        from .steps import run_health
        self.cache.clear()
        self._work("Checking the scan (large scans take a minute)",
                   lambda: run_health(self.session, path, fixes, progress=self._say), self._health_done)

    def _health_done(self, result):
        scan, report, image, fixed = result
        self.scan_report = report
        self._set_text(self.checklist, "\n".join(report.checklist()) + (f"\n\nCleaned copy: {fixed}" if fixed else ""))
        if not fixed:
            for key, _ in FIX_LABELS:
                name = {"planes": "remove_planes", "wires": "strip_wires", "holes": "fill_holes",
                        "fragments": "drop_fragments"}[key]
                self.fix_vars[key].set(any(f.fix == name for f in report.findings))
        if image:
            self._show_image(self.health_img, image, "health")
        self.status.set(("Cleaned copy written; the original is untouched. " if fixed else
                         "Fixes preset from the findings; nothing changed. ") + "Next: 2. Tags.")

    # ================================================================ step 2
    def load_tags(self):
        path = self.markers_var.get().strip()
        if not path:
            self.status.set("Pick the markers.json from aruco-tag-maker.")
            return
        from .steps import load_tags
        try:
            load_tags(self.session, path)
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"Problem: {exc}")
            return
        self.cache.clear()
        self._fill_tags()
        self._refresh_tab_titles()
        self.status.set(self.session.step("tags")["summary"] + ". Next: 3. Wiggle.")

    def load_sim_tags(self):
        if self.bench is None or self.bench.markers_manifest() is None:
            self.status.set("Connect to the simulated bench first.")
            return
        from .steps import load_tags
        load_tags(self.session, manifest=self.bench.markers_manifest())
        self.markers_var.set(str(self.session.markers_path))
        self._fill_tags()
        self._refresh_tab_titles()
        self.status.set(self.session.step("tags")["summary"] + ". Next: 3. Wiggle.")

    def _fill_tags(self, seen=None):
        from .steps import PLACEMENT_TIPS, markers
        m = markers(self.session)
        self.tag_table.delete(*self.tag_table.get_children())
        if m is None:
            return
        link_of = {}
        if self.wiggle_res is not None:
            for link in self.wiggle_res.links:
                for k in link.tags:
                    link_of[int(k)] = link.name
        for name, tid in sorted(m.tags.items(), key=lambda kv: kv[1]):
            seen_txt = "" if seen is None else ("yes" if tid in seen else "NO")
            self.tag_table.insert("", "end", values=(tid, name + (" (root)" if tid == m.root_tag else ""),
                                                     link_of.get(tid, ""), seen_txt),
                                  tags=("flag",) if seen is not None and tid not in seen else ())
        self._set_text(self.tags_text, f"{m.dictionary}, black square {m.tag_size_mm} mm, {len(m.tags)} tags.\n\n"
                       "Placement:\n" + "\n".join(f"• {t}" for t in PLACEMENT_TIPS))

    # ================================================================ hardware
    def refresh_sources(self):
        from .hardware import sim_available
        values = []
        ok, _ = sim_available()
        if ok:
            values.append(SIM_LABEL.format(seed=self.sim_seed))
        values.append(EMULATOR_LABEL)
        try:
            import robo_prober as rp
            values += [p["device"] + "  " + (p.get("interface") or "") for p in rp.discover()]
        except Exception:  # noqa: BLE001 - listing only
            pass
        self.source_box.configure(values=values)
        if not self.source_var.get() and values:
            self.source_var.set(values[0])

    def _source(self):
        text = self.source_var.get().strip()
        if text.startswith("Simulated bench"):
            import re
            m = re.search(r"seed (\d+)", text)
            return f"sim:{m.group(1) if m else self.sim_seed}"
        if text == EMULATOR_LABEL:
            return "emulator"
        return text.split()[0] if text else ""

    def _hw_buttons(self):
        b = self.bench
        connected = b is not None and b.connected
        armed = connected and b.armed
        latched = connected and b.estop_latched
        self.connect_btn.state(["disabled"] if connected else ["!disabled"])
        self.disconnect_btn.state(["!disabled"] if connected else ["disabled"])
        self.arm_btn.state(["!disabled"] if connected and not armed and not latched else ["disabled"])
        self.disarm_btn.state(["!disabled"] if armed else ["disabled"])
        self.clear_btn.state(["!disabled"] if latched else ["disabled"])
        if not connected:
            self.hw_var.set("Not connected. Steps 3 and 4 need the controller and the camera.")
        else:
            mode = "E-STOP LATCHED: press Clear e-stop" if latched else ("ARMED" if armed else "connected, disarmed")
            self.hw_var.set(f"{b.source}: {mode}. " + (b.log[-1][9:] if b.log else ""))

    def connect(self):
        from .camera import CameraIntrinsics
        from .hardware import Bench
        source = self._source()
        if not source:
            self.status.set("Pick a controller.")
            return
        intr = None
        if self.calib_var.get().strip():
            intr = CameraIntrinsics.load(self.calib_var.get().strip())
        cam = self.camera_var.get().strip()
        camera = None if source.startswith("sim") or source == "emulator" else (int(cam) if cam.isdigit() else cam or None)
        bench = Bench(source, camera=camera, intrinsics=intr, sim_light=self.sim_light)

        def job():
            bench.connect()
            targets = bench.discover()
            if bench.world is not None or bench.camera is not None:
                from .steps import load_tags, markers
                if markers(self.session) is None and bench.markers_manifest():
                    load_tags(self.session, manifest=bench.markers_manifest())
                m = markers(self.session)
                if m is not None:
                    self._say("locking the camera focus on the robot")
                    bench.open_camera(m)
            return bench, targets
        self._work(f"Connecting to {source} (reads only, nothing moves)", job, self._connected)

    def _connected(self, result):
        self.bench, targets = result
        self._fill_channels()
        self._fill_tags()
        if self.bench.tracker is not None:
            self._work("Camera preview", self._preview_job, self._preview_done)
        self.status.set("Connected; nothing moved. " + (self.bench.survey.summary() if self.bench.survey else "")
                        + " Next: type the start pulse, tick the power cutoff, Arm + wiggle.")

    def _preview_job(self):
        img = self.bench.preview_image()
        seen = self.bench.visible_tags() if self.bench.tracker is not None else []
        return img, seen

    def _preview_done(self, result):
        img, seen = result
        if img is not None:
            self._show_image(self.cam_img, img, "cam", max_w=560, max_h=520)
            self._show_image(self.tags_cam, img, "tagcam", max_w=700, max_h=520)
        self._fill_tags(seen)

    def _fill_channels(self):
        from .hardware import format_channels
        if self.bench is None or not self.bench.targets:
            return
        chans = [t.channel for t in self.bench.targets if t.cls == "pca9685" or self.gpio_var.get()]
        if not chans:
            chans = [t.channel for t in self.bench.targets]
        self.channels_var.set(format_channels(chans))
        if self.wiggle_res is None:
            self.wiggle_table.delete(*self.wiggle_table.get_children())
            for t in self.bench.targets:
                if t.channel in chans:
                    self.wiggle_table.insert("", "end", iid=str(t.channel),
                                             values=(t.channel, t.key, "not wiggled yet", "", "", "", "", ""))

    def arm(self):
        if self.bench is None:
            self.status.set("Connect first.")
            return
        try:
            self.bench.arm(cutoff_confirmed=self.cutoff_var.get())
            self.status.set("Armed. Nothing moves until you wiggle or enable a joint.")
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"Not armed ({getattr(exc, 'reason', 'error')}): {exc}")
        self._hw_buttons()

    def disarm(self):
        if self.bench is not None and self.bench.connected:
            self.stop_flag.set()
            try:
                self.bench.disarm()
            except Exception as exc:  # noqa: BLE001
                self.status.set(f"Disarm problem: {exc}")
            self.rom_joint = None
        self._hw_buttons()

    def estop(self):
        if self.bench is None or not self.bench.connected:
            self.status.set("E-STOP: nothing connected.")
            return
        self.stop_flag.set()
        self.queue_channels.clear()
        try:
            self.bench.estop()
            self.status.set("E-STOP latched: every output released. Clear e-stop (with the cutoff tick) to go on.")
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"E-STOP sent; controller reply: {exc}. Use the physical power cutoff if anything moves.")
        self.rom_joint = None
        self._hw_buttons()

    def clear_estop(self):
        try:
            self.bench.clear_estop(cutoff_confirmed=self.cutoff_var.get())
            self.status.set("E-stop cleared. Nothing is armed.")
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"Not cleared ({getattr(exc, 'reason', 'error')}): {exc}")
        self._hw_buttons()

    def disconnect(self):
        if self.busy:
            self.status.set("Wait for the current job (or E-STOP).")
            return
        if self.bench is not None:
            self.bench.close()
        self.bench = None
        self._hw_buttons()

    # ================================================================ step 3
    def discover(self):
        if self.bench is None:
            self.connect()
            return
        self._work("Discovering (reads only)", self.bench.discover, lambda _: self._fill_channels())

    def _start_us(self):
        text = self.start_var.get().strip()
        if not text.isdigit():
            self.status.set("Type the start pulse first (bad_start). " + HORN_WARNING)
            return None
        return int(text)

    def _channels(self):
        from .hardware import parse_channels
        try:
            return parse_channels(self.channels_var.get())
        except ValueError:
            self.status.set("Outputs look like 0-7, 9, 104")
            return []

    def wiggle_all(self):
        self._begin_wiggle(self._channels(), reference=True)

    def wiggle_next(self):
        chans = self._channels()
        done = {int(i) for i in self.wiggle_table.get_children()
                if self.wiggle_table.set(i, "result") not in ("not wiggled yet", "")}
        todo = [c for c in chans if c not in done]
        if todo:
            self._begin_wiggle(todo[:1], reference=not self.session.has_log())

    def rewiggle_flagged(self):
        if self.wiggle_res is None:
            return
        self._begin_wiggle([c.channel for c in self.wiggle_res.channels if c.status == "needs_confirmation"],
                           reference=False)

    def rewiggle_selected(self):
        iid = self._selected(self.wiggle_table)
        if iid is not None:
            self._begin_wiggle([int(iid)], reference=False)

    def _begin_wiggle(self, channels, reference):
        if not channels:
            self.status.set("Nothing to wiggle.")
            return
        start = self._start_us()
        if start is None:
            return
        if self.bench is None or not self.bench.connected:
            self.status.set("Connect first (the controller bar at the top).")
            return
        if self.session.step("tags")["status"] != "done":
            self.status.set("Step 2 first: load markers.json (or the simulated bench's tags).")
            return
        if not self.bench.armed:          # arm in the same click, with the cutoff tick
            try:
                self.bench.arm(cutoff_confirmed=self.cutoff_var.get())
            except Exception as exc:  # noqa: BLE001
                self.status.set(f"Not armed ({getattr(exc, 'reason', 'error')}): {exc}")
                return
        self._hw_buttons()
        self.stop_flag.clear()
        delta = int(self.delta_var.get() or 80)
        auto_next = bool(self.auto_next_var.get())       # Tk variables are read here, never in the worker
        self.queue_channels = list(channels)
        from .steps import markers

        def job():
            b = self.bench
            if b.recorder is None:
                b.start_recording(self.session.log_path, markers(self.session))
                import json
                self.session.camera_path.write_text(json.dumps(b.camera_dict(), indent=2) + "\n")
            if reference and not self._has_reference():
                self._say("reference capture: the robot as it was scanned")
                b.reference()
            while self.queue_channels and not self.stop_flag.is_set():
                ch = self.queue_channels.pop(0)
                self._post(self._row_state, ch, "wiggling…")
                self._say(f"wiggling output {ch} (±{delta} µs around {start})")
                b.wiggle(ch, start_us=start, delta_us=delta)
                self._post(self._row_state, ch, "captured")
                if not auto_next:
                    break
            self.session.set_input("wiggle_start_us", start)
            from .steps import analyse_wiggle
            res = analyse_wiggle(self.session)
            img = b.preview_image()
            return res, img
        self._work("Wiggling", job, self._wiggle_live_done)

    def _has_reference(self):
        if not self.session.has_log():
            return False
        return '"phase": "reference"' in self.session.log_path.read_text()

    def _row_state(self, ch, text):
        if self.wiggle_table.exists(str(ch)):
            self.wiggle_table.set(str(ch), "result", text)
            self.wiggle_table.see(str(ch))

    def _wiggle_live_done(self, result):
        res, img = result
        if img is not None:
            self._show_image(self.cam_img, img, "cam", max_w=560, max_h=520)
        self._wiggle_done(res)

    def _wiggle_done(self, res):
        self.wiggle_res = res
        self.cache.clear()
        answers = self.session.confirmations.get("channels", {})
        keep = {int(i): self.wiggle_table.item(i, "values") for i in self.wiggle_table.get_children()}
        self.wiggle_table.delete(*self.wiggle_table.get_children())
        by_ch = {c.channel: c for c in res.channels}
        what = {t.channel: t.key for t in (self.bench.targets if self.bench else [])}
        for ch in sorted(set(keep) | set(by_ch)):
            c = by_ch.get(ch)
            if c is None:
                self.wiggle_table.insert("", "end", iid=str(ch), values=keep[ch])
                continue
            ans = (answers.get(str(ch)) or {}).get("verdict", "")
            tag = "flag" if c.status == "needs_confirmation" and not ans else ("ok" if c.status == "mapped" else ())
            self.wiggle_table.insert("", "end", iid=str(ch), tags=(tag,) if tag else (), values=(
                ch, what.get(ch, f"output {ch}"), c.status, c.joint or "", f"{c.parent} -> {c.child}" if c.child else "",
                "" if c.direction is None else f"{c.direction:+d}", ans, "; ".join(c.reasons)))
        self._fill_tags()
        self._fill_rom_rows()
        flagged = [c.channel for c in res.channels if c.status == "needs_confirmation"
                   and str(c.channel) not in answers]
        if flagged:
            first = str(flagged[0])
            self.wiggle_table.selection_set(first)
            self.wiggle_table.see(first)
            self.status.set(f"Output {flagged[0]} needs your eyes: did that joint move, and is the parent -> child "
                            "right? Y / N (or wiggle it again).")
        else:
            self.status.set(self.session.step("wiggle")["summary"] + ". Next: 4. Range of motion.")
        self._refresh_tab_titles()

    def answer_channel(self, verdict):
        iid = self._selected(self.wiggle_table)
        if iid is None or self.wiggle_res is None:
            self.status.set("Click a row first.")
            return
        self.session.confirm_channel(int(iid), verdict)
        from .steps import analyse_wiggle
        self._wiggle_done(analyse_wiggle(self.session))

    def _key_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text)):
            return None
        key = self.steps_meta[self.tabs.index("current")][0]
        if key == "wiggle":
            self.wiggle_next()
            return "break"
        return None

    def _key_answer(self, verdict):
        w = self.root.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text)):
            return None
        key = self.steps_meta[self.tabs.index("current")][0]
        if key == "wiggle":
            self.answer_channel(verdict)
        elif key == "rom":
            self.answer_joint(verdict)
        return "break"

    def use_recorded_log(self):
        path = filedialog.askopenfilename(filetypes=[("Camera log", "*.jsonl"), ("All files", "*")])
        if not path:
            return
        from .steps import analyse_wiggle
        self.session.import_log(path)
        self.cache.clear()
        self._work("Reading the camera log", lambda: analyse_wiggle(self.session), self._wiggle_done)

    # ================================================================ step 4
    def _rom_joints(self):
        if self.wiggle_res is None:
            return []
        answers = self.session.confirmations.get("channels", {})
        start = self.session.input("wiggle_start_us") or 1500
        return [(c.channel, c.joint, int(start)) for c in self.wiggle_res.channels if c.joint and (
            c.status == "mapped" or (answers.get(str(c.channel)) or {}).get("verdict") == "confirmed")]

    def _fill_rom_rows(self, fits=None):
        self.rom_table.delete(*self.rom_table.get_children())
        joints = self._rom_joints()
        if not joints:
            self.rom_table.insert("", "end", values=("", "Wiggle first (step 3); flagged outputs need your answer.",
                                                     "", "", "", "", "", "", "", ""))
            return
        marks = self._marks()
        answers = self.session.confirmations.get("joints", {})
        fits = fits or ({j.channel: j for j in self.state.kin.joints} if self.state is not None else {})
        for ch, name, start in joints:
            j = fits.get(ch)
            m = marks.get(ch, {})
            limits = f"{math.degrees(j.lower):+.1f} .. {math.degrees(j.upper):+.1f}" if j and j.type == "revolute" else ""
            ans = (answers.get(name) or {}).get("verdict", "")
            tag = "flag" if j is not None and j.status != "ok" and not ans else ("ok" if j is not None else ())
            self.rom_table.insert("", "end", iid=str(ch), tags=(tag,) if tag else (), values=(
                ch, name, start, m.get(1, {}).get("pulse", ""), m.get(2, {}).get("pulse", ""),
                " / ".join(m.get(w, {}).get("tags", "") for w in (1, 2) if w in m),
                j.status if j else "", limits, ans, "; ".join(j.reasons) if j else ""))

    def _marks(self):
        """ROM marks found in the camera log: {channel: {1|2: {pulse, tags}}}."""
        out: dict = {}
        if not self.session.has_log():
            return out
        from .observations import load_log
        for f in load_log(self.session.log_path).select(phase="rom"):
            if f.event in ("rom1", "rom2"):
                out.setdefault(f.channel, {})[int(f.event[-1])] = {
                    "pulse": int(f.pulse_us), "tags": ",".join(map(str, sorted(f.poses))) or "none seen"}
        return out

    def start_rom(self):
        joints = self._rom_joints()
        if not joints:
            self.status.set("Wiggle first (step 3): the ROM sweeps the joints the wiggle found.")
            return
        if self.bench is None or not self.bench.connected:
            self.status.set("Connect first.")
            return
        from .steps import markers
        cutoff = bool(self.cutoff_var.get())

        def job():
            if self.bench.recorder is None:
                self.bench.start_recording(self.session.log_path, markers(self.session))
            if self.wiggle_res is not None:     # marks retry, then warn, when the moving link's tags are hidden
                self.bench.recorder.expect_from_wiggle(self.wiggle_res)
            return self.bench.start_rom(joints, cutoff_confirmed=cutoff)
        self._work("Arming the ROM session", job, lambda _: (
            self._fill_rom_rows(), self.status.set("ROM armed. Pick a joint and press Enable: it starts at the "
                                                   "pulse the wiggle left it at.")))

    def _rom_selected(self):
        iid = self._selected(self.rom_table)
        if iid and iid.isdigit():
            self._show_rom_pos(int(iid))

    def _show_rom_pos(self, ch):
        rom = self.bench.rom if self.bench is not None else None
        if rom is None or ch not in rom.channels:
            return
        pos = rom.position(ch)
        start = rom.channels[ch].start_us
        low, high = rom.channels[ch].slider_range()
        if pos is None:
            self.rom_pos_var.set(f"output {ch}: not enabled (starts at {start} µs)")
        else:
            self.rom_pos_var.set(f"output {ch}: {pos - start:+d} µs   (abs {pos} µs; band {low}..{high})")

    def rom_enable(self):
        iid = self._selected(self.rom_table)
        if not iid or not iid.isdigit() or self.bench is None or self.bench.rom is None:
            self.status.set("Start the ROM session and click a joint first.")
            return
        ch = int(iid)

        def job():
            self.bench.rom_enable(ch)
            return ch
        self._work(f"Enabling output {ch} at its start pulse", job, self._rom_moved)

    def rom_step(self, sign):
        ch = self.rom_joint
        if ch is None or self.bench is None or self.bench.rom is None:
            self.status.set("Enable a joint first.")
            return
        step = int(self.rom_step_var.get() or 40) * (2 if abs(sign) == 2 else 1) * (1 if sign > 0 else -1)

        def job():
            pos = self.bench.rom.position(ch)
            self.bench.rom_move(ch, pos + step)
            return ch
        self._work(f"Moving output {ch} {step:+d} µs", job, self._rom_moved)

    def _rom_moved(self, ch):
        self.rom_joint = ch
        self._show_rom_pos(ch)
        self.rom_warn_var.set("")
        self.status.set("Move toward one safe end. Stop short of any hard stop, then Set ROM 1; then the other end.")

    def rom_mark(self, which):
        ch = self.rom_joint
        if ch is None:
            self.status.set("Enable a joint first.")
            return

        def job():
            return self.bench.rom_mark(ch, which)
        self._work(f"Set ROM {which}", job, lambda r: self._rom_marked(ch, which, r))

    def _rom_marked(self, ch, which, result):
        pulse, frame = result
        child_tags = []
        if self.wiggle_res is not None:
            c = next((c for c in self.wiggle_res.channels if c.channel == ch), None)
            if c is not None and c.child:
                child_tags = [int(k) for k in self.wiggle_res.link(c.child).tags]
        seen = sorted(frame.poses) if frame is not None else []
        vis = getattr(self.bench.recorder, "last_visibility", None) if self.bench is not None else None
        if vis is not None:
            hidden_msg = "" if vis.get("ok") else vis.get("message", "")
        else:
            hidden = child_tags and not set(child_tags) & set(seen)
            hidden_msg = (f"The moving link's tag(s) {child_tags} are hidden at ROM {which}: the fit will extrapolate "
                          "this end (flagged). Back off a little and set it again, or add a second tag at another "
                          "angle." if hidden else "")
        self.rom_warn_var.set(hidden_msg)
        self._fill_rom_rows()
        self._show_rom_pos(ch)
        self.status.set(f"ROM {which} = {pulse} µs. " + ("Now the other end." if which == 1 else
                                                         "Both ends set: the firmware now holds this joint inside them."))

    def rom_widen(self):
        ch = self.rom_joint
        if ch is None:
            return
        try:
            band = self.bench.rom_widen(ch)
            self.status.set(f"Band widened to {band[0]}..{band[1]} µs (logged).")
            self._show_rom_pos(ch)
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"Cannot widen: {exc}")

    def fit_rom(self):
        from .steps import analyse_rom
        self._work("Fitting axes, pivots and ranges", lambda: analyse_rom(self.session, self._say), self._rom_fitted)

    def _rom_fitted(self, st):
        self.state = st
        self.cache.clear()
        self.cache["key"] = None
        self._fill_rom_rows({j.channel: j for j in st.kin.joints})
        open_flags = self.session.step("rom").get("open_flags") or []
        self.status.set(self.session.step("rom")["summary"] + (
            f". Check {', '.join(open_flags)} on the robot: Y if the axis and the limits look right." if open_flags
            else ". Next: 5. Register."))
        self._refresh_tab_titles()

    def answer_joint(self, verdict):
        iid = self._selected(self.rom_table)
        if not iid or not iid.isdigit():
            self.status.set("Click a joint first.")
            return
        name = self.rom_table.set(iid, "joint")
        self.session.confirm_joint(name, verdict)
        self.rom_table.set(iid, "your answer", verdict)
        flagged = [self.rom_table.set(i, "joint") for i in self.rom_table.get_children()
                   if self.rom_table.set(i, "fit") == "flagged" and not self.rom_table.set(i, "your answer")]
        self.session.set_step("rom", "flagged" if flagged else "done",
                              self.session.step("rom").get("summary", ""), open_flags=flagged)
        self._refresh_tab_titles()

    # ================================================================ steps 5-8
    def _fill_register(self):
        if self.state is None or not self.session.has_log():
            return
        self.reg_table.delete(*self.reg_table.get_children())
        centres = self.session.input("scan_tags") or {}
        link_of = {int(k): link for link, tags in self.state.kin.link_tags.items() for k in tags}
        for tid, T in sorted(self.state.tag_poses.items()):
            p = [round(v * 1000, 1) for v in T[:3, 3]]
            c = centres.get(str(tid)) or centres.get(tid)
            self.reg_table.insert("", "end", iid=str(tid), values=(tid, link_of.get(tid, ""), f"{p[0]}, {p[1]}, {p[2]}",
                                                                   ", ".join(f"{v:g}" for v in c) if c else ""))

    def load_registration(self, key):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        import json
        self.session.set_input(key, json.loads(Path(path).read_text()))
        self.session.set_input("manual_points" if key == "scan_tags" else "scan_tags", None)
        self._fill_register()
        self.run_register()

    def set_tag_centre(self):
        iid = self._selected(self.reg_table)
        try:
            xyz = [float(v) for v in self.xyz_var.get().replace(",", " ").split()]
            assert len(xyz) == 3
        except (ValueError, AssertionError):
            self.status.set("Type three numbers: x y z")
            return
        if iid is None:
            self.status.set("Click a tag first.")
            return
        centres = dict(self.session.input("scan_tags") or {})
        centres[str(iid)] = xyz
        self.session.set_input("scan_tags", centres)
        self.session.set_input("manual_points", None)
        self._fill_register()

    def run_register(self):
        from .steps import run_register
        self._work("Registering the scan", lambda: run_register(self.session, self._say, self.cache), self._registered)

    def _registered(self, st):
        self.state = st
        self.reg_var.set(self.session.step("register")["summary"])
        self._fill_register()
        self.status.set(self.session.step("register")["summary"] + ". Next: 6. Cut links.")

    def run_segment(self):
        from .steps import run_segment
        self._work("Cutting the scan into links", lambda: run_segment(self.session, self._say, self.cache),
                   self._segmented)

    def _segmented(self, st):
        self.state = st
        img = st.out / "segmentation.png"
        if st.seg is not None and img.is_file():
            self._show_image(self.seg_img, img, "seg")
        seg = st.extra["pipeline"].get("segmentation")
        self._set_text(self.seg_text, seg if isinstance(seg, str) else "\n".join(
            f"{k}: {v}" for k, v in (seg or {}).get("stats", {}).items()))
        self.status.set(self.session.step("segment")["summary"] + ". Next: 7. Missing parts.")

    def _fill_servos(self):
        if self.state is None:
            return
        models = self.session.input("servo_models") or {}
        self.servo_table.delete(*self.servo_table.get_children())
        for j in self.state.kin.joints:
            self.servo_table.insert("", "end", iid=j.name, values=(
                j.channel, j.name, j.parent, models.get(j.name) or models.get(f"ch{j.channel}") or "none (stand-in)"))
        self.cad_link_box.configure(values=sorted(self.state.kin.link_tags))

    def _servo_typeahead(self, _event=None):
        text = self.servo_var.get().strip()
        if len(text) < 2:
            return
        from robo_prober import servo_db
        hits = servo_db.search(text, limit=12)
        self.servo_box.configure(values=[f"{e['id']}  ({e['brand']} {e['model']})" for e in hits])

    def assign_servo(self):
        iid = self._selected(self.servo_table)
        model = self.servo_var.get().split()[0] if self.servo_var.get().strip() else ""
        if iid is None or not model:
            self.status.set("Click a joint and type a servo (type-ahead over robo-prober's servo database).")
            return
        from .parts import find_servo
        try:
            find_servo(model)
        except KeyError as exc:
            self.status.set(str(exc))
            return
        models = dict(self.session.input("servo_models") or {})
        models[iid] = model
        self.session.set_input("servo_models", models)
        self._fill_servos()

    def add_cad(self):
        import json
        link, path, pts = self.cad_link_var.get().strip(), self.cad_path_var.get().strip(), self.cad_points_var.get().strip()
        if not (link and path and pts):
            self.status.set("CAD part: pick the link, the STL and a points JSON {points_part, points_model}.")
            return
        p = json.loads(Path(pts).read_text())
        parts = list(self.session.input("cad_parts") or [])
        parts.append({"link": link, "path": path, "points_part": p["points_part"], "points_model": p["points_model"]})
        self.session.set_input("cad_parts", parts)
        self.status.set(f"CAD part added to {link}. Press Build parts.")

    def run_parts(self):
        from .steps import parts_table, run_parts
        self.session.set_input("stand_ins", bool(self.standins_var.get()))
        self._work("Adding missing parts", lambda: (run_parts(self.session, self._say, self.cache),),
                   lambda r: self._parts_done(r[0], parts_table(r[0])))

    def _parts_done(self, st, rows):
        self.state = st
        self._fill_servos()
        self.parts_table.delete(*self.parts_table.get_children())
        for r in rows:
            self.parts_table.insert("", "end", values=(r["link"], r["provenance"], r["faces"], r["note"]))
        self.status.set(self.session.step("parts")["summary"] + ". Every part is labelled with where it came from. "
                        "Next: 8. Export.")

    def run_export(self):
        from .steps import run_export
        self._work("Exporting", lambda: run_export(self.session, self._say, self.cache), self._exported)

    def _exported(self, report):
        out = self.session.twin_dir
        lines = [f"Wrote {out}:", f"  robot.urdf  (checked: {report['urdf'].get('links')} links, {report['urdf'].get('joints')} joints"
                 f"{', loads in ' + report['urdf']['independent_loader'] if report['urdf'].get('independent_loader') else ''})",
                 "  comms_map.json  (output channel -> joint)", "  report.json", ""]
        for j in report["joints"]:
            lines.append(f"{j['name']:<14} output {j['channel']:>3}  {j['status']:<8} limits {j['limits'][0]:+.3f} .. "
                         f"{j['limits'][1]:+.3f} rad   operator: {j['operator']}")
        lines += ["", "Flags:"] + [f"  {f}" for f in report["flags"]] + ["", "Not generated: " + ", ".join(
            report["not_generated"]), "", report["verified_note"]]
        self._set_text(self.export_text, "\n".join(lines))
        unconfirmed = report.get("unconfirmed_flagged_joints") or []
        self.export_var.set(f"Flagged and not confirmed: {', '.join(unconfirmed)} (step 4: Y or N on each)."
                            if unconfirmed else "Simulation / software output: not verified on the physical robot.")
        self.status.set(self.session.step("export")["summary"])
        self._refresh_tab_titles()

    def open_out(self):
        from .platform_utils import open_folder
        if self.session.twin_dir.is_dir():
            open_folder(str(self.session.twin_dir))

    # ================================================================ window
    def toggle_fullscreen(self, state=None):
        if state is None:
            state = not bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", state)

    def request_close(self):
        if self.busy or (self.bench is not None and self.bench.armed):
            from tkinter import messagebox
            if not messagebox.askyesno("Digital Twin Generator",
                                       "A job is running or the controller is armed. Disarm and quit?"):
                return
        self.close()

    def close(self):
        self.stop_flag.set()
        try:
            if self.bench is not None:
                self.bench.close()
        finally:
            self.bench = None
            self.root.destroy()


def run(session=None, port=None, **kwargs) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Cannot open a window ({exc}). Use the command line: python digital_twin_generator.py --help")
        return 1
    from .platform_utils import enable_hidpi
    enable_hidpi()
    TwinApp(root, session=session, port=port, **kwargs)
    root.mainloop()
    return 0
