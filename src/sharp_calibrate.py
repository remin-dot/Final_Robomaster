"""Sharp sensor calibration panel (pygame): record 5, 10, 15, 20 cm by hand.

Calibrates the two Sharp GP2Y0A41 distance sensors on the robot's SIDES:
    left  = sensor adaptor 1, port 2        right = sensor adaptor 2, port 1
For each: hold a flat white board (or the foam wall) at 5 cm from the sensor's
face, straight out from it, press Record 5 cm; then 10, 15, 20. The panel fits
a distance curve through the points (analog). If a port only switches on/off it
is not a Sharp (probably a corner IR module): check the port.

The front-corner IR obstacle modules (adaptor 1 port 1 = left, adaptor 2 port 2 =
right) need no calibration - their screw sets the switching distance (~5 cm).
The panel shows them live (clear / WALL) so the screws can be set.

Save writes the Sharp ports + calibration to config/settings.yaml (section
sharp_ir); the corner modules' section (ir_corner) is left as it is.

    python3 src/sharp_calibrate.py            # robot (Wi-Fi AP); press Connect
    python3 src/sharp_calibrate.py --demo     # no robot: sliders simulate the sensors

Close the mission panel first: the robot takes one SDK connection at a time.
Keys: Q quit.
"""

import argparse
import math
import os
import random
import re
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import yaml

from config_loader import load_config
from panel_pygame import UI
from sharp_ir import ADC_MAX, SYSTEM_VOLTAGE, Calibration, CornerSignal, fit_calibration, read_port

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(BASE_DIR, "config", "settings.yaml")
W, H = 1200, 780
DISTANCES = (5, 10, 15, 20)
SIDES = ("left", "right")
TITLE = {"left": "LEFT side Sharp", "right": "RIGHT side Sharp"}


# =============================================================================
# sources: the robot's sensor adaptor, or a simulation for trying the panel
# =============================================================================
class RobotSource:
    name = "robot"

    def __init__(self, connection="ap"):
        self.connection = connection
        self.ep = None

    def connect(self):
        try:
            from robomaster import robot
        except ImportError:
            raise RuntimeError("DJI robomaster SDK not installed (Mac: bash tools/macos/setup_robomaster_mac.sh)")
        ep = robot.Robot()
        try:
            if self.connection == "ap":
                ep.initialize(conn_type="ap", proto_type="udp")
            else:
                ep.initialize(conn_type=self.connection)
        except Exception as e:
            try:
                ep.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not reach the robot ({e}). Join its Wi-Fi (RMEP-xxxxxx) and close "
                               "the mission panel first.")
        self.ep = ep

    def read(self, adaptor, port):
        return self.read_both(adaptor, port)[0]

    def read_both(self, adaptor, port):
        """(analog, digital) of one port - one request (see sharp_ir.read_port)."""
        return read_port(self.ep.sensor_adaptor, (adaptor, port))

    def close(self):
        if self.ep is not None:
            try:
                self.ep.close()
            except Exception:
                pass
            self.ep = None


class DemoSource:
    """Simulated sensors: a distance slider per side; analog (Sharp curve) or digital (switch)."""
    name = "demo"

    def __init__(self):
        self.dist = {"left": 10.0, "right": 15.0}
        self.digital = {"left": False, "right": False}
        self.ports = {}
        self.corner_ports = {}                      # (adaptor, port) -> side of a corner module
        self.corner_wall = {"left": False, "right": False}

    def connect(self):
        time.sleep(0.3)

    def read_both(self, adaptor, port):
        if (adaptor, port) in self.corner_ports:
            # like the robot's modules: signal on the DIGITAL pin (LOW = wall), analog pin floats
            time.sleep(0.02)
            wall = self.corner_wall[self.corner_ports[(adaptor, port)]]
            return int(330 + random.random() * 60), 0 if wall else 1
        return self.read(adaptor, port), 0

    def read(self, adaptor, port):
        time.sleep(0.02)
        if (adaptor, port) in self.corner_ports:
            return self.read_both(adaptor, port)[0]
        side = self.ports.get((adaptor, port))
        if side is None:
            return None                               # nothing on that port
        d = self.dist[side]
        if self.digital[side]:                        # FC-51 style: LOW when an obstacle is inside ~5 cm
            return int(40 + random.random() * 10) if d <= 5.2 else int(1010 + random.random() * 8)
        v = 12.0 / (d + 0.42)                         # GP2Y0A41 curve
        return int(max(0, min(1023, v / SYSTEM_VOLTAGE * ADC_MAX + random.gauss(0, 4))))

    def close(self):
        pass


# =============================================================================
# model
# =============================================================================
class Calibrator:
    def __init__(self, config):
        cfg = (config.get("sharp_ir", {}) or {})
        first = lambda key, default: tuple((cfg.get(key) or [default])[0])
        self.port = {"left": list(first("left_ports", [1, 2])), "right": list(first("right_ports", [2, 1]))}
        self.trigger_cm = float(cfg.get("side_safe_cm", 9.0))     # "close to the side wall" badge
        cc = config.get("ir_corner", {}) or {}
        self.corner_port = {"left": tuple(cc.get("left_port", [1, 1])), "right": tuple(cc.get("right_port", [2, 2]))}
        self.corner_active_low = bool(cc.get("active_low", True))
        self.corner_threshold = float(cc.get("threshold_raw", 512))
        self.corner_enabled = bool(cc.get("enabled", True))
        self.corner_raw = {s: None for s in SIDES}
        self.corner_io = {s: None for s in SIDES}
        self.corner_sig = {s: CornerSignal(str(cc.get("signal", "auto"))) for s in SIDES}
        self.min_cm = float(cfg.get("min_cm", 4.0))
        self.max_cm = float(cfg.get("max_cm", 30.0))
        cal = cfg.get("calibration", {}) or {}
        # recorded points: side -> {cm: (median raw, spread, samples)}
        self.points = {s: {} for s in SIDES}
        for s in SIDES:
            for c, r in (cal.get(s, {}) or {}).get("points", []):
                self.points[s][int(round(c))] = (float(r), 0.0, 0)
        self.latest = {s: None for s in SIDES}
        self.history = {s: deque(maxlen=200) for s in SIDES}
        self.recording = None           # (side, cm, t_start, samples)
        self.lock = threading.Lock()
        self.source = None
        self.state = "disconnected"     # disconnected / connecting / connected
        self.error = ""
        self.running = True
        self.message = ""
        self.saved_at = None

    # ---------------------------------------------------------------- connection
    def connect(self, source):
        if self.state != "disconnected":
            return
        self.source, self.state, self.error = source, "connecting", ""

        def run():
            try:
                source.connect()
                self.state = "connected"
                threading.Thread(target=self._poll, daemon=True).start()
            except Exception as e:
                self.error, self.state = str(e), "disconnected"
                self.source = None
        threading.Thread(target=run, daemon=True).start()

    def disconnect(self):
        src, self.source = self.source, None
        self.state = "disconnected"
        if src is not None:
            src.close()

    def _poll(self):
        fails = {s: 0 for s in SIDES}
        while self.running and self.source is not None:
            for s in SIDES:
                src = self.source
                if src is None:
                    return
                a, p = self.port[s]
                raw = src.read(a, p)
                t = time.time()
                with self.lock:
                    self.latest[s] = raw
                    if raw is not None:
                        fails[s] = 0
                        self.history[s].append((t, raw))
                        rec = self.recording
                        if rec and rec[0] == s:
                            rec[3].append(raw)
                    else:
                        fails[s] += 1
            if self.corner_enabled:
                for s in SIDES:
                    src = self.source
                    if src is None:
                        return
                    adc, io = src.read_both(*self.corner_port[s]) if hasattr(src, "read_both") \
                        else (src.read(*self.corner_port[s]), None)
                    with self.lock:
                        self.corner_raw[s], self.corner_io[s] = adc, io
                        self.corner_sig[s].update(adc, io)
            time.sleep(0.01)

    def corner_state(self, side):
        with self.lock:
            return self.corner_sig[side].near(self.corner_active_low, self.corner_threshold)

    def sharp_status(self, side):
        """Nothing near reads low too - but a working Sharp goes up when a hand / board comes
        close. Never above raw 100 over the last 100+ readings = no signal."""
        with self.lock:
            vals = [r for _, r in list(self.history[side])]
        if len(vals) >= 100 and max(vals) < 100:
            return (f"NO SIGNAL (never above raw {max(vals)} - hold a hand at 5 cm: still nothing = "
                    "cable loose / wrong port / no power)")
        return None

    # ---------------------------------------------------------------- recording
    def record(self, side, cm, seconds=1.5):
        if self.state != "connected" or self.recording:
            return
        with self.lock:
            self.recording = (side, cm, time.time(), [])

        def finish():
            time.sleep(seconds)
            with self.lock:
                side_, cm_, _, samples = self.recording
                self.recording = None
            if len(samples) < 3:
                self.message = f"{side_} {cm_} cm: no readings - check the port"
                return
            s = sorted(samples)
            med = s[len(s) // 2]
            spread = s[int(len(s) * 0.9)] - s[int(len(s) * 0.1)]
            self.points[side_][cm_] = (float(med), float(spread), len(s))
            self.message = f"{side_} {cm_} cm recorded: raw {med:.0f} (spread {spread:.0f}, {len(s)} samples)"
        threading.Thread(target=finish, daemon=True).start()

    def record_progress(self):
        rec = self.recording
        if not rec:
            return None
        return rec[0], rec[1], min(1.0, (time.time() - rec[2]) / 1.5)

    def clear(self, side):
        self.points[side] = {}

    # ---------------------------------------------------------------- fit / result
    def fit(self, side):
        pts = [(cm, v[0]) for cm, v in sorted(self.points[side].items())]
        return fit_calibration(pts, self.min_cm, self.max_cm)

    def calibration(self, side):
        return Calibration(self.fit(side), self.min_cm, self.max_cm)

    def live_cm(self, side):
        raw = self.latest[side]
        cal = self.calibration(side)
        if raw is None or not cal.usable:
            return None, False
        cm = cal.to_cm(raw, self.trigger_cm)
        near = cal.is_near(raw) if cal.mode == "digital" else cm <= self.trigger_cm
        return cm, near

    # ---------------------------------------------------------------- save
    def save(self, path=SETTINGS):
        """Write the sharp_ir section of settings.yaml (other sections untouched)."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        old = (yaml.safe_load(text) or {}).get("sharp_ir", {}) or {}
        cal = {}
        for s in SIDES:
            c = self.fit(s)
            if c.get("mode") in ("analog", "digital", "flat"):
                cal[s] = c
        keep = {k: old.get(k, d) for k, d in (("M", 12.0), ("C", 0.0), ("min_cm", self.min_cm),
                                              ("max_cm", self.max_cm), ("wall_detect_cm", 16.9),
                                              ("poll_hz", 10), ("side_safe_cm", 9.0), ("side_danger_cm", 6.0))}
        flow = lambda v: yaml.safe_dump(v, default_flow_style=True, width=10000).strip()
        lines = [
            "sharp_ir:                   # 2 Sharp GP2Y0A41 distance sensors on the SIDES (analog, 4-30 cm)",
            "  mount: side               # calibrate with src/sharp_calibrate.py (5, 10, 15, 20 cm)",
            f"  left_ports: {flow([list(self.port['left'])])}      # [adaptor id, port]",
            f"  right_ports: {flow([list(self.port['right'])])}",
        ] + [f"  {k}: {v}" for k, v in keep.items()]
        if cal:
            lines.append(f"  calibration:              # saved {time.strftime('%Y-%m-%d %H:%M')}")
            for s, c in cal.items():
                lines.append(f"    {s}: {flow(c)}")
        block = "\n".join(lines) + "\n\n"
        m = re.search(r"^sharp_ir:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*", text, re.M)
        text = text[:m.start()] + block + text[m.end():] if m else text.rstrip() + "\n\n" + block
        # corner modules: only their polarity is set here (the rest of ir_corner is left alone)
        text = re.sub(r"(^ir_corner:[^\n]*\n(?:[ \t]+[^\n]*\n)*?  active_low:[ \t]*)(true|false)",
                      lambda m_: m_.group(1) + ("true" if self.corner_active_low else "false"), text, count=1, flags=re.M)
        yaml.safe_load(text)                     # never write a file that does not parse
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.saved_at = time.time()
        self.message = "saved to config/settings.yaml (sharp_ir)"


# =============================================================================
# window
# =============================================================================
class CalibratePanel:
    def __init__(self, cal, demo=None, headless=False):
        self.c = cal
        self.demo = demo
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), 0 if headless else (pygame.SCALED | pygame.RESIZABLE))
        pygame.display.set_caption("Sharp sensor calibration")
        self.ui = UI("light")
        self.clock = pygame.time.Clock()
        self.running = True
        self.drag = None

    def run(self):
        while self.running:
            self.step(pygame.event.get())
            self.clock.tick(30)
        self.c.running = False
        self.c.disconnect()
        pygame.quit()

    def step(self, events):
        for e in events:
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                self.running = False
        self.ui.begin(self.screen, events)
        self.draw(events)
        pygame.display.flip()
        return self.screen

    # ---------------------------------------------------------------- drawing
    def draw(self, events):
        ui, c = self.ui, self.c
        self.screen.fill(ui.pal["bg"])
        bar = pygame.Rect(0, 0, W, 52)
        ui.rrect(bar, "panel", 0)
        pygame.draw.line(self.screen, ui.pal["line"], (0, 51), (W, 51))
        x = 16 + ui.label("Sharp sensor calibration", 16, 15, "h") + 16
        col = {"connected": "ok", "connecting": "warn"}.get(c.state, "muted")
        src = c.source.name if c.source else ""
        pygame.draw.circle(self.screen, ui.col(col), (x + 5, 26), 5)
        ui.label(f"{c.state}{(' - ' + src) if src else ''}", x + 16, 18, "s", col)
        bx = W - 12
        for label, fn, kind, w, en in (
                ("Theme", lambda: ui.set_theme("dark" if ui.theme == "light" else "light"), "normal", 64, True),
                ("Disconnect", c.disconnect, "normal", 96, c.state == "connected"),
                ("Demo (no robot)", lambda: c.connect(self.demo or DemoSource()), "normal", 130,
                 c.state == "disconnected"),
                ("Connect robot", lambda: c.connect(RobotSource()), "primary", 124, c.state == "disconnected")):
            r = pygame.Rect(bx - w, 11, w, 30)
            bx -= w + 6
            if ui.button(r, label, kind, enabled=en):
                fn()
        if isinstance(c.source, DemoSource):
            c.source.ports = {tuple(c.port[s]): s for s in SIDES}
            c.source.corner_ports = {tuple(c.corner_port[s]): s for s in SIDES}

        for i, side in enumerate(SIDES):
            self.draw_side(side, pygame.Rect(12 + i * 594, 62, 582, 560))
        self.draw_bottom(pygame.Rect(12, 630, W - 24, 138))

    def draw_side(self, side, r):
        ui, c = self.ui, self.c
        ui.card(r)
        ui.label(TITLE[side], r.x + 14, r.y + 12, "h")
        # port selector
        a, p = c.port[side]
        x = r.x + 230
        for label, idx, lo, hi in (("adaptor", 0, 1, 6), ("port", 1, 1, 2)):
            ui.label(label, x, r.y + 16, "xs", "muted")
            x += ui.fonts["xs"].size(label)[0] + 6
            if ui.button(pygame.Rect(x, r.y + 10, 26, 26), "-", enabled=c.port[side][idx] > lo):
                c.port[side][idx] -= 1
            ui.label(str(c.port[side][idx]), x + 38, r.y + 13, "h", "text", "center")
            if ui.button(pygame.Rect(x + 50, r.y + 10, 26, 26), "+", enabled=c.port[side][idx] < hi):
                c.port[side][idx] += 1
            x += 90

        # live value
        raw = c.latest[side]
        y = r.y + 48
        ui.label("raw ADC", r.x + 14, y, "xs", "muted")
        ui.label("--" if raw is None else f"{raw:.0f}", r.x + 14, y + 12, "big",
                 "muted" if raw is None else "text")
        if raw is not None:
            ui.label(f"{raw / ADC_MAX * SYSTEM_VOLTAGE:.2f} V", r.x + 14, y + 48, "s", "muted")
        cm, near = c.live_cm(side)
        ui.label("calibrated", r.x + 150, y, "xs", "muted")
        ui.label("--" if cm is None else f"{cm:.1f} cm", r.x + 150, y + 12, "big",
                 "bad" if near else ("text" if cm is not None else "muted"))
        if near:
            badge = pygame.Rect(r.x + 150, y + 50, 150, 22)
            ui.rrect(badge, "bad", 11)
            ui.label(f"CLOSE  (< {c.trigger_cm:g} cm)", badge.centerx, badge.y + 3, "sb", "white", "center")
        fit = c.fit(side)
        mode = fit.get("mode", "none")
        mode_txt = {"analog": ("ANALOG (Sharp)", "ok"), "digital": ("ON / OFF - not a Sharp?", "bad"),
                    "flat": ("NO CHANGE", "bad"), "none": ("not calibrated", "muted")}[mode]
        ui.label("sensor type", r.x + 360, y, "xs", "muted")
        ui.label(mode_txt[0], r.x + 360, y + 14, "sb", mode_txt[1])
        tip = {"analog": "distance curve fitted through the points",
               "digital": "this port only switches on/off: it looks like a corner IR module, "
                          "not a Sharp - check the adaptor / port",
               "flat": "no change between distances: check the port, the wiring, "
                       "and that the sensor faces the board",
               "none": "record at least 2 distances"}[mode]
        for k, ln in enumerate(self.wrap(tip, "xs", 205)[:3]):
            ui.label(ln, r.x + 360, y + 34 + k * 14, "xs", "muted")

        # history graph (last 10 s)
        g = pygame.Rect(r.x + 14, r.y + 132, r.w - 28, 110)
        ui.rrect(g, "panel2", 6)
        ui.label("last 10 s", g.x + 6, g.y + 4, "xs", "muted")
        now = time.time()
        with c.lock:
            hist = [h for h in c.history[side] if h[0] > now - 10]
        ymax = 1023.0

        def gy(v):
            return int(g.bottom - 6 - (v / ymax) * (g.h - 12))
        for cm_, (med, _, _) in c.points[side].items():
            yy = gy(med)
            pygame.draw.line(self.screen, ui.pal["line"], (g.x, yy), (g.right, yy), 1)
            ui.label(f"{cm_} cm", g.right - 6, yy - 14, "xs", "muted", "right")
        if len(hist) > 1:
            pts = [(int(g.right - (now - t) / 10.0 * g.w), gy(v)) for t, v in hist]
            pygame.draw.lines(self.screen, ui.pal["accent"], False, pts, 2)

        # record table
        ty = r.y + 254
        ui.label("Hold a flat white board at the distance, straight out from the sensor's face",
                 r.x + 14, ty, "xs", "muted")
        prog = c.record_progress()
        for k, cm_ in enumerate(DISTANCES):
            yy = ty + 20 + k * 38
            rec_here = prog and prog[0] == side and prog[1] == cm_
            if ui.button(pygame.Rect(r.x + 14, yy, 130, 30), f"Record {cm_} cm", "primary" if rec_here else "normal",
                         enabled=c.state == "connected" and not prog):
                c.record(side, cm_)
            if rec_here:
                bar_ = pygame.Rect(r.x + 152, yy + 11, 140, 8)
                ui.rrect(bar_, "line", 4)
                ui.rrect(pygame.Rect(bar_.x, bar_.y, int(bar_.w * prog[2]), bar_.h), "accent", 4)
            elif cm_ in c.points[side]:
                med, spread, n = c.points[side][cm_]
                ui.label(f"raw {med:.0f}", r.x + 152, yy + 7, "s")
                if n:
                    ui.label(f"spread {spread:.0f}  ({n})", r.x + 222, yy + 8, "xs",
                             "warn" if spread > 40 else "muted")
            else:
                ui.label("not recorded", r.x + 152, yy + 8, "xs", "muted")
        if ui.button(pygame.Rect(r.x + 14, ty + 176, 130, 26), "Clear points", enabled=bool(c.points[side])):
            c.clear(side)

        # fit plot: distance (x) vs raw (y)
        fp = pygame.Rect(r.x + 330, r.y + 274, r.w - 344, 272)
        ui.rrect(fp, "panel2", 6)
        ui.label("raw vs distance", fp.x + 6, fp.y + 4, "xs", "muted")
        cmax = 30.0
        px = lambda cm_: int(fp.x + 26 + (cm_ / cmax) * (fp.w - 36))
        py = lambda v: int(fp.bottom - 20 - (v / 1023.0) * (fp.h - 40))
        pygame.draw.line(self.screen, ui.pal["line"], (fp.x + 26, fp.bottom - 20), (fp.right - 10, fp.bottom - 20))
        for cm_ in (5, 10, 15, 20, 25, 30):
            ui.label(str(cm_), px(cm_), fp.bottom - 17, "xs", "muted", "center")
        tx = px(c.trigger_cm)
        pygame.draw.line(self.screen, ui.pal["bad"], (tx, fp.y + 18), (tx, fp.bottom - 20), 1)
        calib = c.calibration(side)
        if calib.mode == "analog":
            curve = []
            for v in range(40, 1024, 12):
                cm_ = calib.A * v ** calib.B
                if 2 <= cm_ <= cmax:
                    curve.append((px(cm_), py(v)))
            if len(curve) > 1:
                pygame.draw.lines(self.screen, ui.pal["ok"], False, sorted(curve), 2)
        elif calib.mode == "digital":
            yy = py(calib.threshold)
            pygame.draw.line(self.screen, ui.pal["accent"], (fp.x + 26, yy), (fp.right - 10, yy), 1)
            ui.label("switch level", fp.right - 12, yy - 14, "xs", "accent", "right")
        for cm_, (med, _, _) in c.points[side].items():
            pygame.draw.circle(self.screen, ui.pal["text"], (px(cm_), py(med)), 5)
        if raw is not None and cm is not None:
            pygame.draw.circle(self.screen, ui.pal["bad"] if near else ui.pal["accent"], (px(min(cm, cmax)), py(raw)), 4)

    def draw_bottom(self, r):
        ui, c = self.ui, self.c
        ui.card(r)
        ui.label("Front-corner IR modules (no calibration: turn the screw until 'WALL' shows at ~5 cm)",
                 r.x + 14, r.y + 12, "s")
        x = r.x + 14
        for side, name in (("left", "front-left"), ("right", "front-right")):
            st = c.corner_state(side)
            a, p = c.corner_port[side]
            raw, io = c.corner_raw[side], c.corner_io[side]
            sig = c.corner_sig[side]
            col = "muted" if st is None else ("bad" if st else "ok")
            txt = "no reading" if st is None else ("WALL" if st else "clear")
            pill = pygame.Rect(x, r.y + 36, 250, 30)
            ui.rrect(pill, "panel2", 8)
            ui.rrect(pill, col, 8, 1)
            ui.label(f"{name}  {txt}", pill.x + 10, pill.y + 7, "sb", col)
            ui.label(f"{a}/{p}  {'dig' if sig.mode == 'io' else 'ana'}  io {'-' if io is None else io}  "
                     f"adc {'--' if raw is None else raw}", pill.right - 8, pill.y + 9, "xs", "muted", "right")
            stat = sig.status()
            ui.label(stat, pill.x, pill.bottom + 4, "xs", "ok" if stat.startswith("OK") else "warn", maxw=250)
            x += 262
        # a Sharp that gives no signal at all (cable / port / power)
        warn = [f"Sharp {s}: {m}" for s in SIDES for m in [c.sharp_status(s)] if m]
        if warn:
            ui.label("   ".join(warn), r.x + 14, r.y + 100, "s", "bad", maxw=r.w - 28)
        # polarity: which output level means "wall" (these modules: HIGH)
        pol = "HIGH = wall" if not c.corner_active_low else "LOW = wall"
        if ui.button(pygame.Rect(x, r.y + 36, 150, 30), f"invert ({pol})", font="xs"):
            c.corner_active_low = not c.corner_active_low
            c.message = "corner IR polarity flipped - check clear / WALL, then Save"
        if ui.button(pygame.Rect(r.right - 230, r.y + 34, 216, 38), "Save to settings.yaml", "primary"):
            try:
                c.save()
            except Exception as e:
                c.message = f"save failed: {e}"
        ui.label(c.message or "Record 5, 10, 15, 20 cm for each Sharp, then Save.", r.x + 700, r.y + 44, "s",
                 "ok" if c.saved_at and time.time() - c.saved_at < 5 else "muted", maxw=r.right - 240 - (r.x + 700))
        if c.error:
            ui.label(c.error, r.x + 14, r.y + 118, "s", "bad", maxw=r.w - 28)
        else:
            ui.label("Close the mission panel before connecting (one SDK connection at a time).   Q = quit",
                     r.x + 14, r.y + 84, "xs", "muted")
        if isinstance(c.source, DemoSource):
            self.demo_controls(pygame.Rect(r.x + 14, r.y + 104, r.w - 28, 26))

    def demo_controls(self, r):
        """Demo only: simulated distance for each sensor (+ analog / digital)."""
        ui, src = self.ui, self.c.source
        for i, side in enumerate(SIDES):
            x = r.x + i * (r.w // 2)
            ui.label(f"demo {side}: {src.dist[side]:.1f} cm", x, r.y + 4, "xs", "muted")
            track = pygame.Rect(x + 130, r.y + 10, 240, 6)
            ui.rrect(track, "line", 3)
            kx = int(track.x + (src.dist[side] - 2) / 28.0 * track.w)
            pygame.draw.circle(self.screen, ui.pal["accent"], (kx, track.centery), 7)
            if pygame.mouse.get_pressed()[0] and track.inflate(0, 16).collidepoint(ui.mouse):
                src.dist[side] = round(2 + max(0, min(1, (ui.mouse[0] - track.x) / track.w)) * 28, 1)
            if ui.button(pygame.Rect(x + 390, r.y, 150, 24),
                         f"corner {side}: {'WALL' if src.corner_wall[side] else 'clear'}", font="xs"):
                src.corner_wall[side] = not src.corner_wall[side]

    def wrap(self, s, font, width):
        f = self.ui.fonts[font]
        lines, cur = [], ""
        for word in str(s).split():
            nxt = (cur + " " + word).strip()
            if f.size(nxt)[0] <= width:
                cur = nxt
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="simulated sensors (no robot)")
    ap.add_argument("--connect", action="store_true", help="connect to the robot straight away")
    args = ap.parse_args()
    cal = Calibrator(load_config())
    panel = CalibratePanel(cal)
    if args.demo:
        cal.connect(DemoSource())
    elif args.connect:
        cal.connect(RobotSource())
    try:
        panel.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
