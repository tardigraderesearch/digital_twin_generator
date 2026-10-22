"""Showcase video of one bench run: the virtual camera with DTG's detections
overlaid, next to a live plot of the true joint angles. No personal data:
the scene is synthetic.

FrameRecorder taps VirtualCamera.read(); save() writes sim-run.mp4 (ffmpeg)
and sim-run.gif (Pillow, sped up).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

PANEL_H = 540
COLORS = [(66, 135, 245), (245, 160, 66), (80, 200, 120)]  # BGR per joint


class FrameRecorder:
    def __init__(self, world, period_s: float = 0.4):
        from dtg.camera import make_detector
        self.world, self.period_s = world, period_s
        self.detector = make_detector(world.markers.dictionary)
        self.frames: list[tuple[float, bytes, dict]] = []
        self._next = 0.0

    def tap(self, t: float, frame: np.ndarray) -> None:
        if t < self._next:
            return
        self._next = t + self.period_s
        from dtg.camera import detect_corners
        corners = detect_corners(frame, self.world.markers, self.detector, clahe=True)
        view = frame.copy()
        for k, c in corners.items():
            pts = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], True, (60, 255, 60), 3, cv2.LINE_AA)
            cv2.circle(view, tuple(int(v) for v in c[0]), 6, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.putText(view, self.world.markers.name_for(k), tuple(int(v) for v in c.mean(axis=0) + (-30, -45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 255, 60), 2, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", cv2.resize(view, (960, 540), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        q = self.world.robot.joint_angles()
        self.frames.append((t, jpg.tobytes(), {"q": q, "tags": len(corners)}))

    # ------------------------------------------------------------------ output
    def _plot_panel(self, t_now: float, history: list, names: list[str], width: int = 640) -> np.ndarray:
        panel = np.full((PANEL_H, width, 3), 24, np.uint8)
        x0, x1, y0, y1 = 70, width - 20, 60, PANEL_H - 50
        cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
        t_end = max(t_now, 1.0)
        t_start = max(0.0, t_end - 120.0)
        lo, hi = -110.0, 110.0
        for v in (-90, -45, 0, 45, 90):
            y = int(y1 - (v - lo) / (hi - lo) * (y1 - y0))
            cv2.line(panel, (x0, y), (x1, y), (45, 45, 45), 1)
            cv2.putText(panel, f"{v:+d}", (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        for i, name in enumerate(names):
            pts = [(int(x0 + (t - t_start) / (t_end - t_start) * (x1 - x0)),
                    int(y1 - (h["q"][name] - lo) / (hi - lo) * (y1 - y0))) for t, h in history if t >= t_start]
            if len(pts) > 1:
                cv2.polylines(panel, [np.array(pts, np.int32)], False, COLORS[i % 3], 2, cv2.LINE_AA)
            cv2.putText(panel, name.replace("j_", ""), (x0 + 10 + i * 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS[i % 3], 2, cv2.LINE_AA)
        cv2.putText(panel, "true joint angle (deg) vs time", (x0, PANEL_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return panel

    def compose(self) -> list[np.ndarray]:
        names = [j.name for j in self.world.truth.joints]
        out, history = [], []
        for t, jpg, meta in self.frames:
            history.append((t, meta))
            cam = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            cv2.putText(cam, f"virtual camera  t={t:5.1f}s  tags detected: {meta['tags']}", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            out.append(np.hstack([cam, self._plot_panel(t, history, names)]))
        return out

    def save(self, out: Path) -> dict:
        frames = self.compose()
        if not frames:
            return {}
        out = Path(out)
        paths = {}
        mp4 = out / "sim-run.mp4"
        if shutil.which("ffmpeg"):
            h, w = frames[0].shape[:2]
            proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                     "-s", f"{w}x{h}", "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                                     "yuv420p", "-crf", "26", str(mp4)], stdin=subprocess.PIPE)
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
            proc.wait()
            paths["mp4"] = str(mp4)
        from PIL import Image
        step = max(1, len(frames) // 160)
        imgs = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2),
                                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                for f in frames[::step]]
        gif = out / "sim-run.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=100, loop=0, optimize=True)
        paths["gif"] = str(gif)
        cv2.imwrite(str(out / "sim-still.png"), frames[len(frames) // 2])
        paths["still"] = str(out / "sim-still.png")
        return paths
