"""Desktop window (tkinter + ttk, same look as aruco-tag-maker).

All logic lives in plain modules; this file wires widgets to them and runs
slow steps in a worker thread so the window stays responsive.

Tabs follow the pipeline:
  1 Scan        Browse scan + markers.json, health check, preview, fix toggles
  2 Wiggle/ROM  Browse camera log (or make a synthetic demo), channel map
  3 Build       registration input, servo models, output folder, Build
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__

ERROR_FG = "#c62828"
SCAN_TYPES = [("Scans", "*.obj *.ply *.stl *.glb *.gltf *.off"), ("All files", "*")]
FIX_LABELS = [("remove_planes", "Remove planes (turntable / floor / feet)"), ("strip_wires", "Strip wires"),
              ("fill_holes", "Fill small holes"), ("drop_fragments", "Drop fragments")]


class TwinApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"Digital Twin Generator {__version__}")
        root.minsize(900, 680)
        root.geometry("1100x800")
        self._events: queue.Queue = queue.Queue()
        self.busy = False
        self.scan = None
        self.health = None
        self._preview_img = None
        self.scan_var = tk.StringVar()
        self.markers_var = tk.StringVar()
        self.log_var = tk.StringVar()
        self.tags_var = tk.StringVar()
        self.points_var = tk.StringVar()
        self.servos_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(Path.home() / "digital-twin-output"))
        self.status = tk.StringVar(value="1. Load a scan and run the health check. Nothing here changes your files.")
        self.fix_vars = {k: tk.BooleanVar(value=False) for k, _ in FIX_LABELS}
        self._build()
        root.after(50, self._poll)

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names() and self.root.tk.call("tk", "windowingsystem") == "x11":
            style.theme_use("clam")
        style.configure("Big.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure("Error.TLabel", foreground=ERROR_FG)
        style.configure("Accent.TButton", font=("TkDefaultFont", 11, "bold"), padding=(14, 6))
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self._scan_tab()
        self._log_tab()
        self._build_tab()
        ttk.Label(outer, textvariable=self.status, foreground="#444").pack(fill="x", pady=(8, 0))

    def _file_row(self, parent, label, var, command, row):
        ttk.Label(parent, text=label, style="Big.TLabel").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse…", command=command).grid(row=row, column=2)

    def _scan_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="1  Scan + health check")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)
        self._file_row(tab, "Scan (OBJ/PLY/STL/GLB)", self.scan_var, self.browse_scan, 0)
        self._file_row(tab, "markers.json", self.markers_var, self.browse_markers, 1)
        bar = ttk.Frame(tab)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        self.check_btn = ttk.Button(bar, text="Run health check", style="Accent.TButton", command=self.run_health)
        self.check_btn.pack(side="left")
        for key, text in FIX_LABELS:
            ttk.Checkbutton(bar, text=text, variable=self.fix_vars[key]).pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Save cleaned copy…", command=self.save_cleaned).pack(side="right")
        body = ttk.PanedWindow(tab, orient="vertical")
        body.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.preview = ttk.Label(body, text="Preview appears here (problem regions coloured).", anchor="center")
        body.add(self.preview, weight=3)
        self.checklist = tk.Text(body, height=9, wrap="word")
        body.add(self.checklist, weight=1)

    def _log_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="2  Wiggle + ROM (camera)")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)
        self._file_row(tab, "Camera log (.jsonl)", self.log_var, self.browse_log, 0)
        note = ("Wiggle (each channel moved a little, one at a time) maps channels to joints; then the ROM sweep "
                "(robo-prober sliders, Set ROM 1 / Set ROM 2) gives axes, origins and terminal ranges. "
                "The camera watches both; the scanner is not used. Live capture needs the camera + an armed "
                "controller (dtg.camera.ObservationRecorder); until then analyse a recorded or synthetic log.")
        ttk.Label(tab, text=note, wraplength=950, foreground="#555").grid(row=1, column=0, columnspan=3, sticky="w")
        bar = ttk.Frame(tab)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Button(bar, text="Analyse log", style="Accent.TButton", command=self.run_analyse).pack(side="left")
        ttk.Button(bar, text="Make synthetic demo log…", command=self.make_synthetic).pack(side="left", padx=8)
        cols = ("channel", "status", "joint", "link", "dir", "limits", "notes")
        self.table = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (60, 130, 120, 180, 40, 170, 400)):
            self.table.heading(c, text=c)
            self.table.column(c, width=w, anchor="w")
        self.table.grid(row=3, column=0, columnspan=3, sticky="nsew")

    def _build_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="3  Build twin")
        tab.columnconfigure(1, weight=1)
        self._file_row(tab, "Tag centres in scan (JSON)", self.tags_var,
                       lambda: self._browse_into(self.tags_var, [("JSON", "*.json")]), 0)
        self._file_row(tab, "…or 3-point alignment (JSON)", self.points_var,
                       lambda: self._browse_into(self.points_var, [("JSON", "*.json")]), 1)
        ttk.Label(tab, text="Servo models", style="Big.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(tab, textvariable=self.servos_var).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Label(tab, text="e.g. ch0=MG996R ch4=MG996R").grid(row=2, column=2, sticky="w")
        ttk.Label(tab, text="Output folder", style="Big.TLabel").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(tab, textvariable=self.out_var).grid(row=3, column=1, sticky="ew", padx=6)
        ttk.Button(tab, text="Browse…", command=self.browse_out).grid(row=3, column=2)
        bar = ttk.Frame(tab)
        bar.grid(row=4, column=0, columnspan=3, sticky="w", pady=10)
        self.build_btn = ttk.Button(bar, text="Build twin", style="Accent.TButton", command=self.run_build)
        self.build_btn.pack(side="left")
        ttk.Button(bar, text="Open output folder", command=self.open_out).pack(side="left", padx=8)
        self.build_log = tk.Text(tab, height=20, wrap="word")
        self.build_log.grid(row=5, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(5, weight=1)

    # ------------------------------------------------------------ browsing
    def _browse_into(self, var, types):
        path = filedialog.askopenfilename(filetypes=types)
        if path:
            var.set(path)

    def browse_scan(self):
        self._browse_into(self.scan_var, SCAN_TYPES)

    def browse_markers(self):
        self._browse_into(self.markers_var, [("markers.json", "*.json"), ("All files", "*")])

    def browse_log(self):
        self._browse_into(self.log_var, [("Observation log", "*.jsonl"), ("All files", "*")])

    def browse_out(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def open_out(self):
        from .platform_utils import open_folder
        if Path(self.out_var.get()).is_dir():
            open_folder(self.out_var.get())

    # ------------------------------------------------------------ worker plumbing
    def _work(self, label, fn, done):
        if self.busy:
            return
        self.busy = True
        self.status.set(label + " …")

        def target():
            try:
                self._events.put(("done", done, fn()))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self._events.put(("error", str(exc), traceback.format_exc()))
        threading.Thread(target=target, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, a, b = self._events.get_nowait()
                self.busy = False
                if kind == "done":
                    a(b)
                elif kind == "progress":
                    self.status.set(a)
                else:
                    self.status.set(f"Problem: {a}")
                    messagebox.showerror("Digital Twin Generator", f"{a}\n\n{b[-1500:]}")
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # ------------------------------------------------------------ actions
    def run_health(self):
        path = self.scan_var.get().strip()
        if not path:
            self.status.set("Pick a scan first.")
            return
        out = Path(self.out_var.get()) / "health"

        def job():
            from .health import check_scan
            from .preview import render_health
            from .scan import load_scan
            scan = load_scan(path)
            report = check_scan(scan)
            out.mkdir(parents=True, exist_ok=True)
            image = render_health(scan, report, out / "health.png")
            return scan, report, image
        self._work("Checking the scan (large scans take a minute)", job, self._health_done)

    def _health_done(self, result):
        self.scan, self.health, image = result
        self.checklist.delete("1.0", "end")
        self.checklist.insert("end", "\n".join(self.health.checklist()))
        for key, _ in FIX_LABELS:
            self.fix_vars[key].set(any(f.fix == key for f in self.health.findings))
        self._show_image(image)
        self.status.set(f"Health check done. Preview saved to {image}. Fix toggles preset; nothing changed yet.")

    def _show_image(self, path):
        from PIL import Image, ImageTk
        img = Image.open(path)
        width = max(400, self.preview.winfo_width() or 1000)
        img = img.resize((width, int(img.height * width / img.width)))
        self._preview_img = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._preview_img, text="")

    def save_cleaned(self):
        if self.health is None:
            self.status.set("Run the health check first.")
            return
        target = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL", "*.stl"), ("PLY", "*.ply")])
        if not target:
            return
        from .health import FixOptions, apply_fixes, save_fixed
        opts = FixOptions(**{k: v.get() for k, v in self.fix_vars.items()})
        self._work("Writing cleaned copy", lambda: (save_fixed(r := apply_fixes(self.scan, self.health, opts), target), r)[1],
                    lambda r: self.status.set(f"Saved {target}: {r.summary()} (original untouched)"))

    def make_synthetic(self):
        target = filedialog.asksaveasfilename(defaultextension=".jsonl", initialfile="synthetic-observations.jsonl")
        if not target:
            return
        from .synthetic import generate

        def job():
            log = generate()
            log.save(target)
            return target
        self._work("Generating synthetic log", job, lambda t: (self.log_var.set(t), self.status.set(
            f"Synthetic demo log written to {t} (bot1-like arm, labelled synthetic). Press Analyse.")))

    def run_analyse(self):
        path = self.log_var.get().strip()
        if not path:
            self.status.set("Pick a camera log first (or make a synthetic one).")
            return
        markers = self.markers_var.get().strip() or None

        def job():
            from .kinematics import infer_kinematics
            from .linkage import infer_linkage
            from .markers import load_markers, parse_markers
            from .observations import load_log
            log = load_log(path)
            m = load_markers(markers) if markers else (parse_markers(log.header["markers"]) if log.header.get("markers") else None)
            w = infer_linkage(log, m)
            return w, infer_kinematics(log, w)
        self._work("Analysing wiggle and ROM", job, self._analyse_done)

    def _analyse_done(self, result):
        import math
        wiggle, kin = result
        self.table.delete(*self.table.get_children())
        fits = {j.channel: j for j in kin.joints}
        for c in wiggle.channels:
            j = fits.get(c.channel)
            limits = f"{math.degrees(j.lower):+.1f}° .. {math.degrees(j.upper):+.1f}°" if j and j.type == "revolute" else ""
            notes = "; ".join((j.reasons if j else []) or c.reasons)
            self.table.insert("", "end", values=(c.channel, j.status if j else c.status, c.joint or "",
                                                 f"{c.parent} -> {c.child}" if c.child else "", c.direction or "",
                                                 limits, notes))
        self.status.set(f"{len(kin.joints)} joint(s) fitted; flags: {len(kin.flags)}. Next: tab 3.")

    def run_build(self):
        if not self.log_var.get().strip():
            self.status.set("Tab 2: pick a camera log first.")
            return
        from .health import FixOptions
        from .pipeline import BuildOptions, build_twin
        try:
            servos = dict(x.split("=", 1) for x in self.servos_var.get().split())
        except ValueError:
            self.status.set("Servo models must look like ch0=MG996R")
            return
        opt = BuildOptions(log=Path(self.log_var.get()), out=Path(self.out_var.get()),
                           markers=self.markers_var.get() or None, scan=self.scan_var.get() or None,
                           scan_tags=self.tags_var.get() or None, manual_points=self.points_var.get() or None,
                           fixes=FixOptions(**{k: v.get() for k, v in self.fix_vars.items()}), servo_models=servos)
        progress = lambda msg: self._events.put(("progress", msg, None))  # noqa: E731
        self._work("Building", lambda: build_twin(opt, progress), self._build_done)

    def _build_done(self, report):
        self.build_log.delete("1.0", "end")
        lines = [f"Wrote {self.out_var.get()}: robot.urdf, comms_map.json, report.json", ""]
        for j in report["joints"]:
            lines.append(f"{j['name']}  ch {j['channel']}  {j['status']}  limits {j['limits'][0]:+.3f} .. "
                         f"{j['limits'][1]:+.3f}  {'; '.join(j['reasons'])}")
        lines += ["", "Flags:"] + [f"  {f}" for f in report["flags"]] + ["", f"Not generated: {', '.join(report['not_generated'])}"]
        self.build_log.insert("end", "\n".join(lines))
        self.status.set("Build finished. Unverified until a sim-vs-real check passes.")


def run(**kwargs) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Cannot open a window ({exc}). Use the command line: dtg --help")
        return 1
    from .platform_utils import enable_hidpi
    enable_hidpi()
    app = TwinApp(root, **kwargs)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return 0 if app else 1
