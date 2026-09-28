"""Mission panel: live camera (30 fps) with target segmentation + live maze map.

One OpenCV window, rendered from the main thread at 30 fps:

    +--------------------------------------+-------------------+
    |  camera view + segmentation overlay  |   6x6 maze map    |
    |  (FPS, crosshair, in-range flags)    |  path / walls /   |
    |                                      |  robot / targets  |
    +--------------------------------------+-------------------+
    |  telemetry  |  detections           |  target checklist |
    +--------------------------------------+-------------------+

The robot mission runs in a worker thread and only *updates* the panel state
(robot cell, heading, walls, target observations), so the SDK's blocking calls
never freeze the video.

Keys:  m = cycle view (overlay / segmentation / raw)   s = snapshot
       q / Esc = quit (asks the mission to stop)

Stand-alone usage:
    python3 src/mission_panel.py --robot                 # camera + detection only
    python3 src/mission_panel.py --demo img1.jpg ...     # offline demo, no robot
"""

import argparse
import csv
import json
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from target_vision import (COLORS, DEFAULT_CATALOGUE, DRAW_BGR, SHAPE_LABEL, SHAPES, TargetDetector,
                           draw_detections, draw_ignored, kind_label, kind_of, segmentation_mask, split_kind)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WIN_W, WIN_H = 1280, 720
WINDOW_NAME = "RoboMaster Rescue - Mission Panel"
HEADING_DEG = {0: 0, 1: 90, 2: 180, 3: 270}
MOVES = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# light theme palette (BGR)
BG = (241, 239, 236)
PANEL = (255, 255, 255)
LINE = (214, 210, 205)
TEXT = (40, 36, 32)
MUTED = (135, 128, 120)
ACCENT = (190, 110, 20)     # blue
OK = (60, 150, 40)          # green
WARN = (45, 45, 205)        # red
AMBER = (0, 140, 230)       # orange
PHASE_COLOR = {"COARSE": AMBER, "FINE": ACCENT, "LOCKED": OK, "FIRE": WARN}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def card(img, x1, y1, x2, y2, title=None):
    """White card with a thin border and an optional title."""
    cv2.rectangle(img, (x1, y1), (x2, y2), PANEL, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), LINE, 1)
    if title:
        put(img, title, (x1 + 10, y1 + 20), 0.45, ACCENT, 1)


# ======================================================================
# aim state (RoboFinal auto-aim HUD: phase, error vs tolerance, lock count)
# ======================================================================
class AimState:
    HOLD_S = 2.0  # keep showing the last result this long

    def __init__(self):
        self.lock = threading.Lock()
        self.tol = 1.5
        self.need = 2
        self.phase = "IDLE"
        self.color = None
        self.yaw = self.pitch = None
        self.count = 0
        self.end_ts = None
        self.hist = deque(maxlen=240)   # (t, yaw_err, pitch_err) for the sparkline
        self.samples = []               # full log -> CSV

    def configure(self, tol, need):
        self.tol, self.need = tol, need

    def begin(self, color):
        with self.lock:
            self.color, self.phase, self.count, self.end_ts = color, "COARSE", 0, None
            self.yaw = self.pitch = None

    def sample(self, yaw, pitch, count):
        with self.lock:
            t = time.time()
            self.yaw, self.pitch, self.count = yaw, pitch, count
            self.hist.append((t, yaw, pitch))
            self.samples.append((t, self.color, self.phase, round(yaw, 3), round(pitch, 3), count))

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def end(self):
        with self.lock:
            self.end_ts = time.time()

    def view(self):
        with self.lock:
            active = self.end_ts is None or time.time() - self.end_ts < self.HOLD_S
            phase = self.phase if (active and self.color) else "IDLE"
            return {"phase": phase, "color": self.color, "yaw": self.yaw, "pitch": self.pitch,
                    "count": self.count, "need": self.need, "tol": self.tol,
                    "active": active and self.color is not None, "hist": list(self.hist)}


# ======================================================================
# camera + detection worker
# ======================================================================
class CameraWorker:
    """Two threads: `capture` keeps the newest frame at the stream rate (the
    view never waits for detection) and `detect` segments the newest frame
    as fast as it can."""

    def __init__(self, read_frame, detector):
        self.read_frame = read_frame
        self.detector = detector
        self.lock = threading.Lock()
        self.new_frame = threading.Condition(self.lock)
        self.running = True
        self.frame = None
        self.frame_ts = 0.0
        self.frame_id = 0
        self.detections = []
        self.det_ts = 0.0        # capture time of the frame the detections belong to
        self.fps = 0.0
        self.det_fps = 0.0
        self.detect_ms = 0.0
        self._threads = [threading.Thread(target=self._capture_loop, daemon=True),
                         threading.Thread(target=self._detect_loop, daemon=True)]

    @staticmethod
    def _ema(old, new, a=0.1):
        return new if not old else (1 - a) * old + a * new

    def start(self):
        for t in self._threads:
            t.start()

    def set_source(self, read_frame):
        """Swap the camera source while running (Connect / Disconnect); None = no camera."""
        with self.lock:
            self.read_frame = read_frame
            self.frame = None
            self.detections = []
            self.fps = self.det_fps = 0.0

    def _capture_loop(self):
        last = None
        while self.running:
            read = self.read_frame
            if read is None:
                last = None
                time.sleep(0.05)
                continue
            try:
                frame = read()
            except Exception:
                frame = None
            if frame is None:
                time.sleep(0.005)
                continue
            now = time.time()
            with self.new_frame:
                self.frame = frame
                self.frame_ts = now
                self.frame_id += 1
                if last is not None and now > last:
                    self.fps = self._ema(self.fps, 1.0 / (now - last))
                self.new_frame.notify_all()
            last = now

    def _detect_loop(self):
        done_id, last = 0, None
        while self.running:
            with self.new_frame:
                while self.running and self.frame_id == done_id:
                    self.new_frame.wait(0.2)
                frame, ts, done_id = self.frame, self.frame_ts, self.frame_id
            if frame is None:
                continue
            t0 = time.time()
            dets = self.detector.detect(frame)
            t1 = time.time()
            with self.lock:
                self.detections = dets
                self.det_ts = ts
                self.detect_ms = self._ema(self.detect_ms, (t1 - t0) * 1000.0)
                if last is not None and t1 > last:
                    self.det_fps = self._ema(self.det_fps, 1.0 / (t1 - last))
            last = t1

    def latest(self):
        with self.lock:
            return self.frame, list(self.detections), self.frame_ts, self.frame_id

    def wait_fresh(self, after_ts, timeout=0.5):
        """Detections of a frame captured after `after_ts` (robot settled)."""
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self.det_ts > after_ts:
                    return self.frame, list(self.detections)
            time.sleep(0.01)
        with self.lock:
            return self.frame, list(self.detections)

    def stop(self):
        self.running = False


# ======================================================================
# map state
# ======================================================================
class MissionMap:
    def __init__(self, size_x=6, size_y=6, tile_m=0.6, start=(0, 0)):
        self.nx, self.ny = size_x, size_y
        self.tile = tile_m
        self.lock = threading.RLock()
        self.start = tuple(start)
        self.robot = tuple(start)
        self.heading = 0
        self.gimbal_abs = 0.0
        self.gimbal_rel = 0.0            # gimbal yaw relative to the chassis
        self.visited = {tuple(start)}
        self.path = [tuple(start)]
        self.walls = set()           # frozenset({cellA, cellB}) or (cell, dir) for outer
        self.open_edges = set()
        self.odom = []               # [(east_m, north_m)] relative to start cell centre
        self.targets = {}            # color -> dict
        self.known = {}              # targets loaded from a previous round
        self.shots = []              # [(color, cell, t)]
        self.traversed = set()       # edges the robot drove through: never a wall
        self.min_observations = 2    # sightings (scan stops) before a target counts
        self.same_card_m = 0.45      # same kind closer than this = the same card
        self.plan = []               # round 2: planned cell routes [[cell, ...], ...]
        self.plan_marks = []         # round 2: [(fire_cell, aim_deg, color)]

    # ---------------- robot / walls ----------------
    def set_robot(self, cell, heading=None, gimbal_abs=None, visited=None):
        with self.lock:
            cell = tuple(cell)
            if cell != self.robot:
                if abs(cell[0] - self.robot[0]) + abs(cell[1] - self.robot[1]) == 1:
                    edge = frozenset((tuple(self.robot), cell))
                    self.traversed.add(edge)
                    self.walls.discard(edge)
                    self.open_edges.add(edge)
                self.path.append(cell)
            self.robot = cell
            self.visited.add(cell)
            if visited:
                self.visited.update(tuple(v) for v in visited)
            if heading is not None:
                self.heading = heading
            if gimbal_abs is not None:
                self.gimbal_abs = gimbal_abs

    def add_odom(self, east_m, north_m):
        with self.lock:
            if not self.odom or math.hypot(east_m - self.odom[-1][0], north_m - self.odom[-1][1]) > 0.02:
                self.odom.append((east_m, north_m))

    def _edge(self, cell, d):
        nb = (cell[0] + MOVES[d][0], cell[1] + MOVES[d][1])
        if 0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny:
            return frozenset((tuple(cell), nb))
        return (tuple(cell), d)

    def mark_scan(self, cell, heading, dists, front_wall_mm=450, side_open_mm=450, min_valid_mm=60):
        """Add walls seen by the 4-way gimbal ToF scan (front/right/back/left).
        Edges the robot drove through stay open; readings below min_valid_mm
        (the robot's own body, sensor glitches) leave the edge unknown."""
        rel = {"front": 0, "right": 1, "back": 2, "left": 3}
        with self.lock:
            for key, off in rel.items():
                if key not in dists or dists[key] is None or dists[key] < min_valid_mm:
                    continue
                thr = front_wall_mm if key == "front" else side_open_mm
                edge = self._edge(cell, (heading + off) % 4)
                if edge in self.traversed:
                    continue
                if dists[key] <= thr:
                    self.walls.add(edge)
                    self.open_edges.discard(edge)
                else:
                    self.open_edges.add(edge)
                    self.walls.discard(edge)

    # ---------------- targets ----------------
    def cell_center_m(self, cell):
        return (cell[0] + 0.5) * self.tile, (cell[1] + 0.5) * self.tile

    def project(self, robot_cell, abs_deg, dist_m):
        """Map point (x_m, y_m) seen from the centre of robot_cell. abs_deg: 0=north, 90=east."""
        rx, ry = self.cell_center_m(robot_cell)
        th = math.radians(abs_deg)
        return rx + dist_m * math.sin(th), ry + dist_m * math.cos(th)

    def inside(self, x_m, y_m, margin_m=0.0):
        return (-margin_m <= x_m <= self.nx * self.tile + margin_m and
                -margin_m <= y_m <= self.ny * self.tile + margin_m)

    def visible_from(self, robot_cell, x_m, y_m):
        """False when a known wall lies between the robot cell and the point."""
        from route_planner import GridGraph
        with self.lock:
            walls = {e for e in self.walls if isinstance(e, frozenset)}
        return GridGraph(self.nx, self.ny, walls).line_of_sight(tuple(robot_cell), (x_m, y_m), self.tile)

    def _find_card(self, kind, x_m, y_m, radius_m=None):
        """Id of the mapped card of this kind nearest (x, y) within radius, or None."""
        radius_m = self.same_card_m if radius_m is None else radius_m
        best, best_d = None, radius_m
        for tid, t in self.targets.items():
            if t["kind"] != kind or not t["sw"]:
                continue
            d = math.hypot(t["sx"] / t["sw"] - x_m, t["sy"] / t["sw"] - y_m)
            if d <= best_d:
                best, best_d = tid, d
        return best

    def add_observation(self, color, shape, robot_cell, abs_deg, dist_m, weight=1.0):
        """Project a detection onto the map (abs_deg: 0=north, 90=east) and merge it
        with the same card seen before; a card of the same kind further away is a new one."""
        tx, ty = self.project(robot_cell, abs_deg, dist_m)
        w = weight / max(dist_m, 0.2)  # closer views are more reliable
        kind = kind_of(color, shape)
        with self.lock:
            tid = self._find_card(kind, tx, ty)
            if tid is None:
                n_same = sum(1 for t in self.targets.values() if t["kind"] == kind)
                tid = kind if n_same == 0 else f"{kind} #{n_same + 1}"
                self.targets[tid] = {
                    "id": tid, "kind": kind, "color": color, "shape": shape,
                    "sx": 0.0, "sy": 0.0, "sw": 0.0, "n": 0, "shot": False, "first_seen": time.time(),
                    "seen_from": tuple(robot_cell), "best_dist": dist_m, "views": {},
                }
            t = self.targets[tid]
            rc = tuple(robot_cell)
            t["views"][rc] = min(dist_m, t["views"].get(rc, dist_m))
            t["sx"] += tx * w
            t["sy"] += ty * w
            t["sw"] += w
            t["n"] += 1
            t["confirmed"] = t["n"] >= self.min_observations
            if dist_m < t["best_dist"]:
                t["best_dist"] = dist_m
                t["seen_from"] = tuple(robot_cell)
            return self._target_view(t)

    def _target_view(self, t):
        x, y = t["sx"] / t["sw"], t["sy"] / t["sw"]
        cell = (min(max(int(x / self.tile), 0), self.nx - 1), min(max(int(y / self.tile), 0), self.ny - 1))
        return {"id": t["id"], "kind": t["kind"], "color": t["color"], "shape": t["shape"],
                "x_m": round(x, 3), "y_m": round(y, 3),
                "cell": list(cell), "observations": t["n"], "shot": t["shot"],
                "seen_from": list(t["seen_from"]), "best_dist_m": round(t["best_dist"], 2),
                "views": [{"cell": list(c), "dist_m": round(d, 2)} for c, d in t.get("views", {}).items()],
                "confirmed": bool(t.get("confirmed") or t["shot"])}

    def mark_shot(self, tid):
        with self.lock:
            if tid in self.targets:
                self.targets[tid]["shot"] = True
            self.shots.append((tid, self.robot, time.time()))

    def to_verify(self, cell):
        """Cards not checked by a close look yet whose block is this cell or a neighbour."""
        out = []
        with self.lock:
            for t in self.targets.values():
                if t.get("verified") or t.get("swept"):
                    continue
                v = self._target_view(t)
                c = tuple(v["cell"])
                dist = abs(c[0] - cell[0]) + abs(c[1] - cell[1])
                # this block, or a neighbour with no wall in between (a card behind a wall
                # cannot be seen from here, so it must not be dropped for "not seen")
                if dist == 0 or (dist == 1 and frozenset((tuple(cell), c)) not in self.walls):
                    out.append(t["id"])
        return out

    def remove_target(self, tid):
        with self.lock:
            self.targets.pop(tid, None)

    def card_near(self, kind, x_m, y_m, radius_m=0.6, shot=None):
        """A mapped card of this kind near (x, y) (optionally only shot / not shot ones)."""
        with self.lock:
            for t in self.targets.values():
                if t["kind"] != kind or not t["sw"] or (shot is not None and t["shot"] != shot):
                    continue
                if math.hypot(t["sx"] / t["sw"] - x_m, t["sy"] / t["sw"] - y_m) <= radius_m:
                    return t["id"]
        return None

    def target_list(self, confirmed_only=True):
        with self.lock:
            views = [self._target_view(t) for t in self.targets.values()]
        return [v for v in views if v["confirmed"] or not confirmed_only]

    def to_json(self):
        with self.lock:
            return {
                "grid_size": [self.nx, self.ny],
                "tile_m": self.tile,
                "start": list(self.start),
                "end": list(self.robot),
                "path": [list(p) for p in self.path],
                "visited": sorted([list(v) for v in self.visited]),
                "walls": sorted([sorted(list(map(list, e))) if isinstance(e, frozenset) else [list(e[0]), e[1]]
                                 for e in self.walls], key=str),
                "open_edges": sorted([sorted(list(map(list, e))) for e in self.open_edges
                                      if isinstance(e, frozenset)], key=str),
                "targets": self.target_list(),
                "shots": [{"id": tid, "kind": self.targets[tid]["kind"] if tid in self.targets else tid,
                           "robot_cell": list(cell), "t": ts} for tid, cell, ts in self.shots],
            }

    def load_round(self, data):
        """Round 2: take grid size, walls and targets from the round-1 JSON."""
        from route_planner import parse_edges
        with self.lock:
            self.nx, self.ny = data["grid_size"]
            self.tile = data.get("tile_m", self.tile)
            self.walls = set(parse_edges(data.get("walls")))
            for e in data.get("walls", []):
                if len(e) == 2 and isinstance(e[1], int):
                    self.walls.add((tuple(e[0]), e[1]))
            self.open_edges = set(parse_edges(data.get("open_edges")))
            from route_planner import driven_edges
            driven = driven_edges(data)
            self.walls -= driven
            self.open_edges |= driven
            self.known = {}
            for t in data.get("targets", []):
                if t.get("confirmed", True):
                    t.setdefault("kind", kind_of(t["color"], t["shape"]))  # files from before kinds
                    t.setdefault("id", t["kind"])
                    self.known[t["id"]] = t

    def set_plan(self, legs):
        with self.lock:
            self.plan = [[tuple(c) for c in leg["path"]] for leg in legs]
            self.plan_marks = [(tuple(leg["fire_cell"]), leg["aim_deg"], leg["color"]) for leg in legs]

    # ---------------- drawing ----------------
    def geometry(self, size):
        pad = 26
        cell = (size - 2 * pad) / float(max(self.nx, self.ny))
        ox = pad + (size - 2 * pad - cell * self.nx) / 2.0
        oy = pad + (size - 2 * pad - cell * self.ny) / 2.0
        return ox, oy, cell

    def cell_at(self, x, y, size):
        """Map-image pixel -> cell (or None)."""
        ox, oy, cell = self.geometry(size)
        cx = int((x - ox) // cell)
        cy = self.ny - 1 - int((y - oy) // cell)
        if 0 <= cx < self.nx and 0 <= cy < self.ny and x >= ox and y >= oy:
            return (cx, cy)
        return None

    def render(self, size, fov_deg=None, title=None):
        img = np.full((size, size, 3), 250, np.uint8)
        ox, oy, cell = self.geometry(size)

        def px(cx, cy):  # cell coords (float, y up) -> pixels
            return int(round(ox + cx * cell)), int(round(oy + (self.ny - cy) * cell))

        def m2px(x_m, y_m):
            return px(x_m / self.tile, y_m / self.tile)

        with self.lock:
            # visited cells
            for (cx, cy) in self.visited:
                p1, p2 = px(cx, cy + 1), px(cx + 1, cy)
                cv2.rectangle(img, p1, p2, (238, 226, 205), -1)
            # start cell
            p1, p2 = px(self.start[0], self.start[1] + 1), px(self.start[0] + 1, self.start[1])
            cv2.rectangle(img, p1, p2, (200, 235, 200), -1)
            put(img, "S", (p1[0] + 4, p1[1] + 16), 0.5, (40, 120, 40), 2)

            # grid
            for i in range(self.nx + 1):
                cv2.line(img, px(i, 0), px(i, self.ny), (210, 210, 210), 1)
            for j in range(self.ny + 1):
                cv2.line(img, px(0, j), px(self.nx, j), (210, 210, 210), 1)
            for i in range(self.nx):
                put(img, str(i), (px(i + 0.5, 0)[0] - 4, px(0, 0)[1] + 16), 0.4, (120, 120, 120))
            for j in range(self.ny):
                put(img, str(j), (px(0, j + 0.5)[0] - 16, px(0, j + 0.5)[1] + 4), 0.4, (120, 120, 120))

            # walls
            for e in self.walls:
                if isinstance(e, frozenset):
                    a, b = sorted(e)
                    if a[0] != b[0]:   # vertical wall between x and x+1
                        x = max(a[0], b[0])
                        seg = (px(x, a[1]), px(x, a[1] + 1))
                    else:
                        y = max(a[1], b[1])
                        seg = (px(a[0], y), px(a[0] + 1, y))
                else:
                    (cx, cy), d = e
                    seg = {0: (px(cx, cy + 1), px(cx + 1, cy + 1)),
                           1: (px(cx + 1, cy), px(cx + 1, cy + 1)),
                           2: (px(cx, cy), px(cx + 1, cy)),
                           3: (px(cx, cy), px(cx, cy + 1))}[d]
                cv2.line(img, seg[0], seg[1], (50, 50, 50), 4, cv2.LINE_AA)
            cv2.rectangle(img, px(0, self.ny), px(self.nx, 0), (90, 90, 90), 2)

            # odometry trail (thin) + planned grid path (thick)
            if len(self.odom) > 1:
                sx, sy = self.cell_center_m(self.start)
                pts = np.array([m2px(sx + e, sy + n) for e, n in self.odom], np.int32)
                cv2.polylines(img, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)
            for route in self.plan:  # round-2 plan: dashed purple
                pts = [px(c[0] + 0.5, c[1] + 0.5) for c in route]
                for a, b in zip(pts, pts[1:]):
                    n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // 8))
                    for k in range(0, n, 2):
                        p0 = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
                        p1 = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
                        cv2.line(img, p0, p1, (200, 60, 160), 2, cv2.LINE_AA)
            for fc, adeg, color in self.plan_marks:  # firing spot + aim ray
                c0 = px(fc[0] + 0.5, fc[1] + 0.5)
                c1 = (int(c0[0] + math.sin(math.radians(adeg)) * cell * 0.9),
                      int(c0[1] - math.cos(math.radians(adeg)) * cell * 0.9))
                cv2.circle(img, c0, max(5, int(cell * 0.12)), DRAW_BGR.get(color, (0, 0, 0)), 2, cv2.LINE_AA)
                cv2.arrowedLine(img, c0, c1, DRAW_BGR.get(color, (0, 0, 0)), 1, cv2.LINE_AA, tipLength=0.2)
            if len(self.path) > 1:
                pts = np.array([px(c[0] + 0.5, c[1] + 0.5) for c in self.path], np.int32)
                cv2.polylines(img, [pts], False, (0, 140, 255), 3, cv2.LINE_AA)
                for p in pts[:-1]:
                    cv2.circle(img, tuple(int(v) for v in p), 3, (0, 140, 255), -1, cv2.LINE_AA)

            # known targets from the previous round (hollow)
            for t in self.known.values():
                if self._find_card(t["kind"], t["x_m"], t["y_m"], 0.6):
                    continue  # seen again this round
                c = m2px(t["x_m"], t["y_m"])
                self._draw_target_icon(img, c, t["color"], t.get("shape"), int(cell * 0.22), hollow=True)

            # targets found this round
            for t in self.targets.values():
                v = self._target_view(t)
                c = m2px(v["x_m"], v["y_m"])
                cc = v["cell"]
                col = DRAW_BGR.get(t["color"], (0, 0, 0))
                if not v["confirmed"]:  # one sighting so far: hollow + "?"
                    self._draw_target_icon(img, c, t["color"], t["shape"], int(cell * 0.18), hollow=True)
                    cv2.putText(img, "?", (c[0] + int(cell * 0.2), c[1] - 4), FONT, 0.5, col, 2, cv2.LINE_AA)
                    continue
                cv2.rectangle(img, px(cc[0], cc[1] + 1), px(cc[0] + 1, cc[1]), col, 2)
                self._draw_target_icon(img, c, t["color"], t["shape"], int(cell * 0.22))
                if t["shot"]:
                    cv2.putText(img, "HIT", (c[0] - 12, c[1] + int(cell * 0.22) + 14), FONT, 0.42,
                                (30, 150, 30), 2, cv2.LINE_AA)

            # robot + camera FOV cone
            rc = px(self.robot[0] + 0.5, self.robot[1] + 0.5)
            if fov_deg:
                r = cell * 2.0
                a0 = self.gimbal_abs - fov_deg / 2.0
                a1 = self.gimbal_abs + fov_deg / 2.0
                cone = [rc] + [(int(rc[0] + r * math.sin(math.radians(a))),
                                int(rc[1] - r * math.cos(math.radians(a))))
                               for a in np.linspace(a0, a1, 12)]
                ov = img.copy()
                cv2.fillPoly(ov, [np.array(cone, np.int32)], (120, 200, 255))
                cv2.addWeighted(ov, 0.25, img, 0.75, 0, dst=img)
            rad = max(8, int(cell * 0.25))
            cv2.circle(img, rc, rad, (40, 40, 220), -1, cv2.LINE_AA)
            hd = math.radians(HEADING_DEG.get(self.heading, 0))
            tip = (int(rc[0] + math.sin(hd) * rad * 1.6), int(rc[1] - math.cos(hd) * rad * 1.6))
            cv2.arrowedLine(img, rc, tip, (40, 40, 220), 2, cv2.LINE_AA, tipLength=0.4)
            gd = math.radians(self.gimbal_abs)
            gtip = (int(rc[0] + math.sin(gd) * cell * 0.7), int(rc[1] - math.cos(gd) * cell * 0.7))
            cv2.line(img, rc, gtip, (255, 140, 0), 2, cv2.LINE_AA)

        put(img, title or "N ^", (6, 16), 0.45, (90, 90, 90))
        return img

    @staticmethod
    def _draw_target_icon(img, c, color, shape, r, hollow=False):
        col = DRAW_BGR.get(color, (0, 0, 0))
        th = 2 if hollow else -1
        if shape == "circle":
            cv2.circle(img, c, r, col, th, cv2.LINE_AA)
        elif shape == "rect_wide":
            cv2.rectangle(img, (c[0] - r, c[1] - int(r * 0.6)), (c[0] + r, c[1] + int(r * 0.6)), col, th)
        elif shape == "rect_tall":
            cv2.rectangle(img, (c[0] - int(r * 0.6), c[1] - r), (c[0] + int(r * 0.6), c[1] + r), col, th)
        else:
            cv2.rectangle(img, (c[0] - int(r * 0.8), c[1] - int(r * 0.8)),
                          (c[0] + int(r * 0.8), c[1] + int(r * 0.8)), col, th)
        cv2.circle(img, c, 2, (0, 0, 0), -1)


# ======================================================================
# panel
# ======================================================================
class MissionPanel:
    VIEW_MODES = ("overlay", "segmentation", "raw")
    HEADER = 48

    def __init__(self, config, read_frame=None, telemetry=None, round_no=1):
        self.config = config
        vis = config.get("vision", {}) or {}
        tile = (config.get("movement", {}) or {}).get("distance", 0.6)

        self.detector = TargetDetector(config)
        self.worker = CameraWorker(read_frame, self.detector)
        self.tile = tile
        self.map = self._new_map()
        self.telemetry = telemetry or (lambda: {})
        self.target_fps = float(vis.get("panel_fps", 30))
        # which card kinds are shot ("blue circle", "red circle", ...) lives in detector.selected
        self.round_limits = {1: 600, 2: 300}
        self.round_limits.update({int(k): v for k, v in (vis.get("round_time_limit_s", {}) or {}).items()})

        self.data_dir = os.path.join(BASE_DIR, config.get("data_collection", {}).get("data_dir", "data/raw/run1"))
        os.makedirs(self.data_dir, exist_ok=True)

        self.controller = None                 # MissionController (control.py) drives the buttons
        self.camera_latency_s = float(vis.get("camera_latency_s", 0.2))
        self.wall_margin_m = float(vis.get("behind_wall_margin_m", 0.15))
        self.min_observations = int(vis.get("min_observations", 2))
        self._rejects = {}
        self.round_no = round_no
        self.round_t0 = None
        self.round_end = None
        self.status = "waiting"
        self.view_mode = 0
        self.ui_fps = 0.0
        self.abort = threading.Event()         # Finish / STOP / window closed
        self.paused = threading.Event()        # Pause
        self.mission_done = threading.Event()
        self._last_tick = None
        self._log = []
        self.aim = AimState()
        self.buttons = []                      # [(rect, callback, enabled)] rebuilt every frame
        self.tab = "targets"
        self._confirm_disconnect = 0.0
        self.editing = False                   # custom map editor open
        self.edit_text = ""
        self.edit_start = None
        self.edit_msg = ""
        self.MAP_RECT = (856, self.HEADER + 8, 416)   # x, y, size of the map on the canvas
        self.prev_round = None
        self.set_round(round_no)

    # ---------------- rounds ----------------
    def _new_map(self):
        grid = self.config.get("grid_map", {}) or {}
        start = grid.get("start", {"x": 0, "y": 0})
        m = MissionMap(grid.get("max_x", 5) + 1, grid.get("max_y", 5) + 1, self.tile,
                       (start.get("x", 0), start.get("y", 0)))
        m.min_observations = int((self.config.get("vision", {}) or {}).get("min_observations", 2))
        return m

    def round_file(self, n):
        return os.path.join(self.data_dir, f"round{n}_targets.json")

    def set_round(self, n):
        """Switch round (not while running). Round 2 shows the round-1 map."""
        if self.controller is not None and self.controller.running:
            return
        self.round_no = n
        self.round_t0 = self.round_end = None
        self.map = self._new_map()
        self.prev_round = None
        if n >= 2:
            prev = self.round_file(n - 1)
            if os.path.exists(prev):
                with open(prev, "r", encoding="utf-8") as f:
                    self.prev_round = json.load(f)
                self.map.load_round(self.prev_round)
                self.map.start = tuple(self.prev_round.get("start", self.map.start))
                self.map.robot = self.map.start
                self.map.path = [self.map.start]
                self.map.visited = {self.map.start}
                print(f"[panel] round {n} uses map {prev}")
            else:
                print(f"[panel] {prev} not found - round {n} will explore again")

    def prepare_round(self):
        """Fresh map / timer before Start."""
        self.set_round(self.round_no)
        self.aim.samples = []
        self.map.shots = []

    def checkpoint(self):
        """Mission threads call this between steps: waits while paused.
        Returns False when the round must end (Finish / STOP / closed)."""
        while self.paused.is_set() and not self.abort.is_set():
            time.sleep(0.05)
        return not self.abort.is_set()

    # ---------------- timer ----------------
    def elapsed(self):
        if not self.round_t0:
            return 0.0
        return (self.round_end or time.time()) - self.round_t0

    def time_limit(self):
        return self.round_limits.get(self.round_no, 600)

    def remaining(self):
        return self.time_limit() - self.elapsed()

    def splits(self):
        """[(color, seconds since round start)] for every hit."""
        t0 = self.round_t0 or 0
        return [(c, ts - t0) for c, _, ts in list(self.map.shots)]

    @staticmethod
    def fmt(sec):
        sec = max(0, int(sec))
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # ---------------- custom map editor ----------------
    def open_editor(self):
        ctl = self.controller
        if ctl is not None and not (ctl.idle or ctl.state == "disconnected"):
            self.log("cannot change the map while the robot is busy")
            return
        if self.round_no >= 2:
            self.log("round 2 uses the saved round-1 map size")
            return
        self.editing = True
        self.edit_text = f"{self.map.nx}x{self.map.ny}"
        self.edit_start = tuple(self.map.start)
        self.edit_msg = ""

    def _parse_size(self):
        m = re.fullmatch(r"\s*(\d{1,2})\s*[xX*]\s*(\d{1,2})\s*", self.edit_text)
        if not m:
            return None
        w, h = int(m.group(1)), int(m.group(2))
        return (w, h) if 1 <= w <= 12 and 1 <= h <= 12 else None

    def save_editor(self):
        size = self._parse_size()
        if not size:
            self.edit_msg = "size must look like 5x4 (1..12)"
            return
        w, h = size
        sx, sy = self.edit_start or (0, 0)
        sx, sy = min(sx, w - 1), min(sy, h - 1)
        grid = self.config.setdefault("grid_map", {})
        grid["max_x"], grid["max_y"] = w - 1, h - 1
        grid["start"] = {"x": sx, "y": sy}
        self.map = self._new_map()
        try:
            save_grid_to_settings(w, h, (sx, sy))
            self.log(f"map {w}x{h}, start ({sx},{sy}) saved to config/settings.yaml")
        except Exception as e:
            self.log(f"map set to {w}x{h} (not saved: {e})")
        self.editing = False

    # ---------------- input ----------------
    def on_key(self, key):
        """Returns False when the UI should close."""
        ctl = self.controller
        if self.editing:
            if key in (13, 10):
                self.save_editor()
            elif key == 27:
                self.editing = False
            elif key in (8, 127):
                self.edit_text = self.edit_text[:-1]
            elif chr(key) in "0123456789xX*" and len(self.edit_text) < 7:
                self.edit_text += chr(key).lower()
            return True
        if key in (ord("q"), 27):
            self.abort.set()
            return False
        if ctl is not None and not ctl.connected:
            if key in (13, 10):
                ctl.connect()
            return True
        if key == ord("m"):
            self.view_mode = (self.view_mode + 1) % len(self.VIEW_MODES)
        elif key == ord("g"):
            self.open_editor()
        elif key == ord("p") and ctl is not None:
            ctl.resume() if ctl.state == "paused" else ctl.pause()
        elif key == ord(" ") and ctl is not None:
            # RoboFinal: Space = STOP while a round runs
            ctl.stop() if ctl.running else ctl.start()
        return True

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2), cb, enabled in reversed(self.buttons):
            if x1 <= x <= x2 and y1 <= y <= y2:
                if enabled and cb is not None:
                    cb()
                return
        mx, my, ms = self.MAP_RECT
        if self.editing and mx <= x < mx + ms and my <= y < my + ms:
            size = self._parse_size()
            if size:
                preview = MissionMap(size[0], size[1], self.map.tile)
                c = preview.cell_at(x - mx, y - my, ms)
                if c:
                    self.edit_start = c

    # ---------------- mission-side API (called from mission thread) ----------------
    def start(self):
        self.worker.start()

    def start_round(self, round_no=None):
        if round_no is not None:
            self.round_no = round_no
        self.round_t0 = time.time()
        self.round_end = None
        self.status = "running"
        self.log(f"Round {self.round_no} started ({self.fmt(self.time_limit())} limit)")

    def log(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"{stamp} {msg}")
        self._log = self._log[-6:]
        print(f"[panel] {msg}")

    def update_robot(self, cell, heading, gimbal_abs=None, visited=None):
        self.map.set_robot(cell, heading, gimbal_abs, visited)

    def observe_targets(self, robot_cell, abs_deg, tof_mm=None, settle_ts=None, frames=3):
        """Look at a few fresh frames and put every real target on the map.

        A target counts only when it is seen in most of `frames` frames, is in
        front of the wall the ToF sees that way, inside the maze and not
        behind a known wall. Returns the valid-target detections (best frame).
        """
        latency = self.camera_latency_s
        after = (settle_ts or time.time()) + latency   # frames arrive ~0.2 s late over Wi-Fi
        seen = {}                                      # kind -> [detections]
        for _ in range(frames):
            _, dets = self.worker.wait_fresh(after, timeout=0.6)
            after = time.time()
            for d in dets:
                if d.is_card and d.distance_m is not None:  # every card goes on the map
                    seen.setdefault(d.kind, []).append(d)
        need = frames // 2 + 1
        found = []
        tof_m = tof_mm / 1000.0 if tof_mm and 60 < tof_mm < 8000 else None
        for kind, ds in seen.items():
            if len(ds) < need:
                continue
            ds.sort(key=lambda d: d.distance_m)
            d = ds[len(ds) // 2]                       # median by distance
            color = d.color
            dist = d.distance_m
            b = math.radians(d.bearing_deg)
            if tof_m is not None:
                if abs(d.bearing_deg) < 4.0 and 0.6 * dist <= tof_m <= 1.5 * dist:
                    dist = tof_m                       # plate is on the ToF beam: use it
                    d.distance_m = tof_m
                    d.in_range = d.is_target and tof_m <= self.detector.max_shoot_m
                elif abs(d.bearing_deg) < 25.0 and dist * math.cos(b) > tof_m + self.wall_margin_m:
                    self._reject(d, f"behind the wall ({dist:.2f} m > wall {tof_m:.2f} m)")
                    continue
            x_m, y_m = self.map.project(robot_cell, abs_deg + d.bearing_deg, dist)
            if not self.map.inside(x_m, y_m, margin_m=0.05):
                self._reject(d, "outside the maze")
                continue
            if not self.map.visible_from(robot_cell, x_m, y_m):
                self._reject(d, "behind a mapped wall")
                continue
            v = self.map.add_observation(color, d.shape, robot_cell, abs_deg + d.bearing_deg, dist)
            d.target_id = v["id"]
            tag = "" if d.is_target else " (not selected)"
            if v["observations"] == 1:
                self.log(f"seen {v['id']} near cell {tuple(v['cell'])} ({dist:.2f} m){tag}")
            elif v["confirmed"] and v["observations"] == self.map.min_observations:
                self.log(f"FOUND {v['id']} at cell {tuple(v['cell'])}{tag}")
            found.append(d)
        return found

    def _reject(self, d, why):
        key = (d.color, why)
        now = time.time()
        if now - self._rejects.get(key, 0) > 5.0:   # do not flood the log
            self._rejects[key] = now
            self.log(f"ignored {d.label}: {why}")

    @property
    def selected(self):
        """Card kinds to shoot, e.g. {"blue circle", "red circle"}."""
        return self.detector.selected

    def set_selected(self, kinds):
        self.detector.set_selected(kinds)
        names = ", ".join(kind_label(k) for k in sorted(kinds)) or "none"
        self.log(f"targets: {names}")

    def toggle_kind(self, kind):
        sel = set(self.selected)
        sel.symmetric_difference_update({kind})
        self.set_selected(sel)

    def save_selection(self):
        items = ", ".join(f'"{k}"' for k in sorted(self.selected))
        self.config.setdefault("vision", {})["shoot_targets"] = sorted(self.selected)
        save_setting("vision", "shoot_targets", f"[{items}]")
        self.log("target selection saved to config/settings.yaml")

    def should_shoot(self, tid):
        """A mapped card that is selected and not hit yet."""
        t = self.map.targets.get(tid)
        return bool(t) and t["kind"] in self.selected and not t["shot"]

    def all_designated_shot(self):
        """Every selected kind that was found is hit (round 2 early stop)."""
        mine = [t for t in self.map.targets.values() if t["kind"] in self.selected]
        return bool(mine) and all(t["shot"] for t in mine)

    def finish_round(self):
        self.round_end = time.time()
        self.status = "finished"
        data = self.save_round_files()
        self.log(f"Round {self.round_no} saved -> "
                 f"{os.path.relpath(os.path.join(self.data_dir, f'round{self.round_no}_map.png'), BASE_DIR)}")
        return data

    def save_round_files(self):
        """Write roundN_targets.json, roundN_map.png and the aim log (Save button / end of round)."""
        elapsed = self.elapsed()
        data = self.map.to_json()
        data.update({"round": self.round_no, "elapsed_s": round(elapsed, 1),
                     "time_limit_s": self.round_limits.get(self.round_no),
                     "designated_targets": sorted(self.selected),
                     "route_plan": getattr(self, "route_plan", None),   # round 2: algorithm comparison
                     "splits": [{"id": c, "t_s": round(t, 1)} for c, t in self.splits()]})
        for sh in data["shots"]:
            sh["t_s"] = round(sh["t"] - (self.round_t0 or sh["t"]), 1)
        json_path = self.round_file(self.round_no)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        img = self.map.render(900, title=f"Round {self.round_no} - path + targets  ({elapsed:.0f}s)")
        img = self._map_report(img, data)
        if self.aim.samples:
            aim_path = os.path.join(self.data_dir, f"round{self.round_no}_aim_log.csv")
            with open(aim_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "target", "phase", "yaw_err_deg", "pitch_err_deg", "lock_count"])
                t0 = self.round_t0 or self.aim.samples[0][0]
                for t, *rest in self.aim.samples:
                    w.writerow([round(t - t0, 3)] + rest)
        cv2.imwrite(os.path.join(self.data_dir, f"round{self.round_no}_map.png"), img)
        return data

    def _map_report(self, map_img, data):
        h = map_img.shape[0]
        side = np.full((h, 360, 3), 255, np.uint8)
        put(side, f"Round {data['round']}", (16, 40), 0.9, (30, 30, 30), 2)
        put(side, f"time {data['elapsed_s']:.0f}s / limit {data['time_limit_s']}s", (16, 72), 0.55, (60, 60, 60))
        put(side, f"start {tuple(data['start'])}  end {tuple(data['end'])}", (16, 98), 0.55, (60, 60, 60))
        put(side, f"cells visited {len(data['visited'])}", (16, 124), 0.55, (60, 60, 60))
        for i, sp in enumerate(data.get("splits", [])):
            put(side, f"hit {sp['id']} at {self.fmt(sp['t_s'])}", (16, 150 + i * 22), 0.5, (40, 120, 40))
        y = 170 + 22 * len(data.get("splits", []))
        put(side, "Targets", (16, y), 0.7, (30, 30, 30), 2)
        for t in data["targets"]:
            y += 34
            MissionMap._draw_target_icon(side, (30, y - 6), t["color"], t["shape"], 11)
            put(side, f"{t['id']}  cell {tuple(t['cell'])}  {'HIT' if t['shot'] else '-'}",
                (52, y), 0.55, (40, 40, 40))
        if not data["targets"]:
            put(side, "none found", (16, y + 34), 0.55, (120, 120, 120))
        return np.hstack([map_img, side])

    # ---------------- buttons ----------------
    def _btn(self, canvas, rect, label, cb=None, kind="normal", on=False, enabled=True, scale=0.48):
        """Draw a button and register it for clicks. kind: normal / primary / danger."""
        x1, y1, x2, y2 = rect
        if not enabled:
            fill, border, tc = (246, 245, 243), LINE, (175, 170, 165)
        elif kind == "primary":
            fill, border, tc = ACCENT, ACCENT, (255, 255, 255)
        elif kind == "danger":
            fill, border, tc = WARN, WARN, (255, 255, 255)
        elif on:
            fill, border, tc = (250, 236, 218), ACCENT, ACCENT
        else:
            fill, border, tc = PANEL, LINE, TEXT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        put(canvas, label, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2), scale, tc, 1)
        self.buttons.append((rect, cb, enabled))

    def _checkbox(self, canvas, x, y, checked, label, cb, enabled=True):
        cv2.rectangle(canvas, (x, y), (x + 16, y + 16), ACCENT if checked else PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + 16, y + 16), ACCENT if checked else LINE, 1)
        if checked:
            cv2.line(canvas, (x + 3, y + 8), (x + 7, y + 12), (255, 255, 255), 2)
            cv2.line(canvas, (x + 7, y + 12), (x + 13, y + 4), (255, 255, 255), 2)
        put(canvas, label, (x + 24, y + 13), 0.45, TEXT if enabled else MUTED)
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        self.buttons.append(((x, y, x + 24 + tw, y + 16), cb, enabled))

    def _section(self, canvas, title, x, y):
        put(canvas, title.upper(), (x, y + 12), 0.42, MUTED)
        return y + 22

    # ---------------- rendering (main thread) ----------------
    def render(self):
        self.buttons = []
        ctl = self.controller
        if ctl is not None and not ctl.connected:
            return self._render_connect()
        canvas = np.full((WIN_H, WIN_W, 3), BG, np.uint8)
        frame, dets, ts, _ = self.worker.latest()
        tel = self.telemetry() or {}
        H = self.HEADER

        self._draw_header(canvas)

        # camera
        cam_x, cam_y, cam_w, cam_h = 8, H + 8, 840, 472
        aim = self.aim.view()
        cv2.rectangle(canvas, (cam_x - 1, cam_y - 1), (cam_x + cam_w, cam_y + cam_h), LINE, 1)
        if frame is not None:
            mode = self.VIEW_MODES[self.view_mode]
            if mode == "segmentation":
                view = segmentation_mask(frame.shape, dets, targets_only=False)
            else:
                view = frame.copy()
            if mode != "raw":
                draw_ignored(view, self.detector.last_ignore_line)
            if mode == "overlay":
                draw_detections(view, dets)
            view = cv2.resize(view, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            self._draw_boresight(view, aim)
            canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = view
            stale = time.time() - ts > 1.0
            live = "LIVE" if not stale else "NO SIGNAL"
            badge = (f"{live}  CAM {self.worker.fps:4.1f} fps  DET {self.worker.det_fps:4.1f} fps "
                     f"({self.worker.detect_ms:3.0f} ms)  UI {self.ui_fps:4.1f} fps  [{mode}]")
            ov = canvas[cam_y:cam_y + 26, cam_x:cam_x + 600]
            ov[:] = (ov * 0.25 + np.array(PANEL) * 0.75).astype(np.uint8)
            if not stale and int(time.time() * 2) % 2:
                cv2.circle(canvas, (cam_x + 12, cam_y + 13), 5, WARN, -1, cv2.LINE_AA)
            put(canvas, badge, (cam_x + 22, cam_y + 18), 0.5, WARN if stale else OK)
            if stale:
                put(canvas, "NO SIGNAL", (cam_x + cam_w // 2 - 80, cam_y + cam_h // 2), 1.0, WARN, 2)
        else:
            cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w - 1, cam_y + cam_h - 1), (228, 225, 221), -1)
            put(canvas, "waiting for camera...", (cam_x + 300, cam_y + cam_h // 2), 0.7, MUTED)
        if ctl is not None and not ctl.armed:
            put(canvas, "DRY RUN - blaster off", (cam_x + cam_w - 210, cam_y + cam_h - 12), 0.5, AMBER, 2)

        by = cam_y + cam_h + 8
        self._draw_sensors(canvas, 8, by, 300, WIN_H - 8, tel)
        self._draw_detections_card(canvas, 308, by, 578, WIN_H - 8, dets)
        self._draw_aim(canvas, 586, by, 848, WIN_H - 8, aim)

        # right: map
        mx, my, ms = self.MAP_RECT
        if self.editing:
            mimg = self._editor_preview(ms)
        else:
            mimg = self.map.render(ms, fov_deg=self.detector.hfov_deg)
        canvas[my:my + ms, mx:mx + ms] = mimg
        cv2.rectangle(canvas, (mx - 1, my - 1), (mx + ms, my + ms), LINE, 1)
        if self.editing:
            self._draw_editor(canvas, mx, my, ms)

        # right-bottom: tabs
        ty0 = my + ms + 8
        card(canvas, mx, ty0, mx + ms, WIN_H - 8)
        for i, (key, label) in enumerate((("targets", "Targets"), ("select", "Select"), ("actions", "Actions"))):
            self._btn(canvas, (mx + 8 + i * 96, ty0 + 6, mx + 98 + i * 96, ty0 + 30), label,
                      cb=lambda k=key: setattr(self, "tab", k), on=self.tab == key)
        n_sel = len(self.selected)
        put(canvas, f"shooting {n_sel} kind{'s' if n_sel != 1 else ''}", (mx + 306, ty0 + 23), 0.4,
            ACCENT if n_sel else WARN)
        {"targets": self._draw_targets_tab, "select": self._draw_select_tab,
         "actions": self._draw_actions_tab}[self.tab](canvas, mx, ty0 + 36, ms)
        return canvas

    def _draw_header(self, canvas):
        ctl = self.controller
        H = self.HEADER
        cv2.rectangle(canvas, (0, 0), (WIN_W, H), PANEL, -1)
        put(canvas, "RoboMaster Rescue", (14, 30), 0.62, TEXT, 2)
        x = 212
        if ctl is not None:
            cv2.circle(canvas, (x + 5, 25), 5, OK, -1, cv2.LINE_AA)
            put(canvas, ctl.source_label(), (x + 16, 24), 0.42, OK, 1)
            sub = []
            if ctl.battery is not None:
                sub.append(f"battery {ctl.battery}%")
            if not ctl.armed:
                sub.append("DRY RUN")
            put(canvas, "  ".join(sub), (x + 16, 41), 0.38, AMBER if not ctl.armed else MUTED)
        x = 340
        busy = ctl is not None and not ctl.idle
        for n in (1, 2):
            self._btn(canvas, (x, 10, x + 78, 38), f"Round {n}", cb=lambda n=n: self.set_round(n),
                      on=self.round_no == n, enabled=not busy or self.round_no == n, scale=0.45)
            x += 84
        # state pill
        state = (ctl.state if ctl is not None else self.status).upper()
        col = {"RUNNING": OK, "PAUSED": AMBER, "DONE": ACCENT, "BUSY": ACCENT}.get(state, MUTED)
        (tw, _), _ = cv2.getTextSize(state, FONT, 0.45, 1)
        cv2.rectangle(canvas, (x + 4, 13), (x + 22 + tw, 35), col, 1)
        put(canvas, state, (x + 13, 29), 0.45, col, 1)
        self._draw_timer(canvas, x + 34 + tw)

        # right-hand buttons (like RoboFinal's header)
        if ctl is None:
            return
        bx = WIN_W - 10
        confirm = time.time() - self._confirm_disconnect < 3.0

        def disconnect():
            if time.time() - self._confirm_disconnect < 3.0:
                ctl.disconnect()
            else:
                self._confirm_disconnect = time.time()
        specs = [("Sure?" if confirm else "Disconnect", disconnect, "danger" if confirm else "normal", 96, True),
                 ("STOP", ctl.stop, "danger", 66, ctl.running),
                 ("Save", ctl.save, "normal", 54, True),
                 ("Finish", ctl.finish, "normal", 62, ctl.running),
                 ("Resume" if ctl.state == "paused" else "Pause",
                  ctl.resume if ctl.state == "paused" else ctl.pause, "normal", 70, ctl.running),
                 (f"Start round {self.round_no}", ctl.start, "primary", 112, ctl.can_start())]
        for label, cb, kind, w, enabled in specs:
            self._btn(canvas, (bx - w, 10, bx, 38), label, cb=cb, kind=kind, enabled=enabled, scale=0.45)
            bx -= w + 6

    def _draw_timer(self, canvas, x):
        """Countdown, elapsed / limit and the progress bar under the header."""
        limit = self.time_limit()
        el = self.elapsed()
        left = limit - el
        if not self.round_t0:
            col = MUTED
        elif left <= 0:
            col = WARN
        elif left <= 30:
            col = WARN if int(time.time() * 2) % 2 else AMBER    # blink in the last 30 s
        elif left <= 60:
            col = AMBER
        else:
            col = OK
        sign = "-" if left < 0 else ""
        put(canvas, f"LEFT {sign}{self.fmt(abs(left))}", (x, 27), 0.62, col, 2)
        put(canvas, f"{self.fmt(el)} / {self.fmt(limit)}", (x, 43), 0.38, TEXT if self.round_t0 else MUTED)
        H = self.HEADER
        frac = min(1.0, el / float(limit)) if limit else 0.0
        cv2.rectangle(canvas, (0, H), (WIN_W, H + 3), LINE, -1)
        cv2.rectangle(canvas, (0, H), (int(WIN_W * frac), H + 3), col, -1)

    def _draw_targets_tab(self, canvas, mx, y0, ms):
        """Every card found (this round + round 1), selected kinds first."""
        rows = []
        for t in list(self.map.targets.values()):
            rows.append(("now", self.map._target_view(t)))
        for k in self.map.known.values():
            if not self.map._find_card(k["kind"], k["x_m"], k["y_m"], 0.6):
                rows.append(("r1", k))
        rows.sort(key=lambda r: (r[1]["kind"] not in self.selected, r[1]["id"]))
        hits = dict(self.splits())
        if not rows:
            sel = ", ".join(kind_label(k) for k in sorted(self.selected)) or "none - pick in Select"
            put(canvas, "no cards found yet", (mx + 12, y0 + 20), 0.45, MUTED)
            put(canvas, f"shooting: {sel}"[:64], (mx + 12, y0 + 42), 0.4, TEXT)
            return
        for i, (src, v) in enumerate(rows[:5]):
            y = y0 + 20 + i * 34
            MissionMap._draw_target_icon(canvas, (mx + 24, y - 5), v["color"], v["shape"], 11,
                                         hollow=src == "r1")
            chosen = v["kind"] in self.selected
            put(canvas, kind_label(v["id"].split(" #")[0]) + (" #" + v["id"].split(" #")[1] if " #" in v["id"] else ""),
                (mx + 44, y), 0.45, TEXT if chosen else MUTED)
            put(canvas, f"cell {tuple(v['cell'])}", (mx + 200, y), 0.42, TEXT if chosen else MUTED)
            if src == "r1":
                status, col = f"round {self.round_no - 1}", MUTED
            elif v["shot"]:
                status, col = (f"HIT {self.fmt(hits[v['id']])}" if v["id"] in hits else "HIT"), OK
            elif not v["confirmed"]:
                status, col = "checking", AMBER
            elif chosen:
                status, col = "to shoot", ACCENT
            else:
                status, col = "not selected", MUTED
            put(canvas, status, (mx + 290, y), 0.45, col, 1)
        if len(rows) > 5:
            put(canvas, f"+{len(rows) - 5} more on the map", (mx + 12, y0 + 186), 0.38, MUTED)

    def _draw_select_tab(self, canvas, mx, y0, ms):
        """Which cards to shoot: any sheet colour x shape (blue circle, red circle, ...)."""
        ctl = self.controller
        editable = ctl is None or not ctl.running
        x = mx + 10
        col_w = (ms - 20 - 62) // 4
        shape_names = {"circle": "circle", "rect_wide": "wide rect", "rect_tall": "tall rect", "square": "square"}
        for j, sh in enumerate(SHAPES):   # header row: shape icons
            cx = x + 62 + j * col_w + col_w // 2
            MissionMap._draw_target_icon(canvas, (cx - 22, y0 + 6), "blue", sh, 7, hollow=True)
            put(canvas, shape_names[sh], (cx - 12, y0 + 11), 0.36, MUTED)
        for i, c in enumerate(COLORS):
            y = y0 + 22 + i * 30
            cv2.rectangle(canvas, (x, y + 5), (x + 12, y + 17), DRAW_BGR[c], -1)
            put(canvas, c, (x + 16, y + 16), 0.42, TEXT)
            for j, sh in enumerate(SHAPES):
                kind = kind_of(c, sh)
                on = kind in self.selected
                bx = x + 62 + j * col_w
                sheet = DEFAULT_CATALOGUE.get(c) == sh
                label = ("ON" if on else "-") + ("*" if sheet else "")
                self._btn(canvas, (bx + 2, y + 2, bx + col_w - 2, y + 24), label,
                          cb=lambda k=kind: self.toggle_kind(k), on=on, enabled=editable, scale=0.4)
        y = y0 + 22 + 4 * 30 + 4
        bw = (ms - 20 - 3 * 6) // 4
        for i, (label, cb) in enumerate((
                ("Sheet set*", lambda: self.set_selected({kind_of(c, s_) for c, s_ in DEFAULT_CATALOGUE.items()})),
                ("All", lambda: self.set_selected({kind_of(c, s_) for c in COLORS for s_ in SHAPES})),
                ("None", lambda: self.set_selected(set())),
                ("Save", self.save_selection))):
            self._btn(canvas, (x + i * (bw + 6), y, x + i * (bw + 6) + bw, y + 24), label, cb=cb,
                      kind="primary" if label == "Save" else "normal", enabled=editable, scale=0.42)
        put(canvas, "* = sheet's card.  Unselected cards: mapped, never shot.",
            (x, y + 40), 0.34, MUTED)

    def _draw_actions_tab(self, canvas, mx, y0, ms):
        """RoboFinal's Mission tab: one-off actions, blaster arm, gimbal buttons."""
        ctl = self.controller
        robot = ctl is not None and ctl.chassis is not None
        idle = ctl is not None and ctl.idle
        x, w = mx + 8, ms - 16
        half = (w - 6) // 2
        y = y0 + 4
        self._btn(canvas, (x, y, x + half, y + 28), "Look around now",
                  cb=ctl.look_around if ctl else None, enabled=robot and idle)
        self._btn(canvas, (x + half + 6, y, x + w, y + 28), "Aim & shoot here",
                  cb=ctl.aim_here if ctl else None, enabled=robot and idle)
        y += 34
        self._btn(canvas, (x, y, x + half, y + 28), "Fire once",
                  cb=ctl.fire_once if ctl else None, enabled=robot)
        self._btn(canvas, (x + half + 6, y, x + w, y + 28), f"Map {self.map.nx}x{self.map.ny}...",
                  cb=self.open_editor, enabled=(ctl is None or idle) and self.round_no == 1)
        y += 34
        armed = ctl.armed if ctl else False
        self._checkbox(canvas, x, y + 5, armed, "Blaster armed",
                       cb=(lambda: ctl.set_armed(not ctl.armed)) if ctl else None)
        self._btn(canvas, (x + half + 6, y, x + w, y + 26), "Calibrate: card at 1 m",
                  cb=ctl.calibrate_distance if ctl else None, enabled=idle and ctl is not None and ctl.connected,
                  scale=0.42)
        y += 32
        put(canvas, "GIMBAL", (x, y + 12), 0.42, MUTED)
        gx = x + 64
        bw = (w - 64 - 4 * 5) // 5
        for i, (label, cb) in enumerate((("Left 15", lambda: ctl.gimbal_move(yaw=-15)),
                                         ("Right 15", lambda: ctl.gimbal_move(yaw=15)),
                                         ("Up 5", lambda: ctl.gimbal_move(pitch=5)),
                                         ("Down 5", lambda: ctl.gimbal_move(pitch=-5)),
                                         ("Centre", lambda: ctl.gimbal_center()))):
            bx = gx + i * (bw + 5)
            self._btn(canvas, (bx, y, bx + bw, y + 26), label, cb=cb if ctl else None,
                      enabled=robot and idle, scale=0.4)
        y += 32
        put(canvas, "AIM TRIM", (x, y + 12), 0.42, MUTED)
        sh = (self.config.get("shooting", {}) or {})
        for i, (label, cb) in enumerate((("Up", lambda: ctl.nudge_aim(pitch=0.5)),
                                         ("Down", lambda: ctl.nudge_aim(pitch=-0.5)),
                                         ("Left", lambda: ctl.nudge_aim(yaw=-0.5)),
                                         ("Right", lambda: ctl.nudge_aim(yaw=0.5)),
                                         ("Save", lambda: ctl.save_trim()))):
            bx = gx + i * (bw + 5)
            self._btn(canvas, (bx, y, bx + bw, y + 24), label, cb=cb if ctl else None,
                      enabled=ctl is not None, scale=0.4)
        y += 30
        put(canvas, f"trim pitch {float(sh.get('aim_pitch_offset_deg', -3.5)):+.1f}  "
                    f"yaw {float(sh.get('aim_yaw_offset_deg', 0.0)):+.1f} deg   (shots high -> Down)",
            (x, y + 10), 0.36, MUTED)

    def _render_connect(self):
        """Connect screen (RoboFinal feature/final): source, Wi-Fi mode, round, Connect."""
        ctl = self.controller
        canvas = np.full((WIN_H, WIN_W, 3), BG, np.uint8)
        cv2.rectangle(canvas, (0, 0), (WIN_W, self.HEADER), PANEL, -1)
        cv2.line(canvas, (0, self.HEADER), (WIN_W, self.HEADER), LINE, 1)
        put(canvas, "RoboMaster Rescue", (14, 30), 0.62, TEXT, 2)
        put(canvas, "not connected", (210, 30), 0.45, MUTED)

        cw_, ch_ = 620, 560
        x0, y0 = (WIN_W - cw_) // 2, self.HEADER + (WIN_H - self.HEADER - ch_) // 2
        card(canvas, x0, y0, x0 + cw_, y0 + ch_)
        x, y, w = x0 + 28, y0 + 24, cw_ - 56
        put(canvas, "Connect", (x, y + 26), 1.0, TEXT, 2)
        y += 44
        put(canvas, "Round 1: explore the maze, find + shoot targets.  Round 2: go shoot them.",
            (x, y + 12), 0.42, MUTED)
        y += 30

        y = self._section(canvas, "Camera / robot", x, y)
        bw = (w - 12) // 3
        for i, (key, label) in enumerate((("robot", "RoboMaster robot"), ("webcam", "Webcam (this computer)"),
                                          ("demo", "Demo (no robot)"))):
            self._btn(canvas, (x + i * (bw + 6), y, x + i * (bw + 6) + bw, y + 34), label,
                      cb=lambda k=key: setattr(ctl, "source", k), on=ctl.source == key, scale=0.45)
        y += 46
        if ctl.source == "robot":
            y = self._section(canvas, "Connection", x, y)
            for i, (key, label) in enumerate((("ap", "Wi-Fi direct (AP)"), ("sta", "Router (STA)"), ("rndis", "USB"))):
                self._btn(canvas, (x + i * (bw + 6), y, x + i * (bw + 6) + bw, y + 30), label,
                          cb=lambda k=key: setattr(ctl, "connection", k), on=ctl.connection == key, scale=0.45)
            y += 38
            put(canvas, "AP: join the robot's Wi-Fi (RMEP-xxxxxx) on this computer first.", (x, y + 10), 0.4, MUTED)
            y += 26
        elif ctl.source == "webcam":
            put(canvas, "Live camera + detection only (no driving). macOS asks for camera access once.",
                (x, y + 10), 0.4, MUTED)
            y += 26
        else:
            put(canvas, "Sample pictures + a simulated robot walk, to try the panel.", (x, y + 10), 0.4, MUTED)
            y += 26

        y = self._section(canvas, "Round", x, y)
        half = (w - 6) // 2
        for i, label in enumerate(("Round 1 - explore + find", "Round 2 - shoot targets")):
            self._btn(canvas, (x + i * (half + 6), y, x + i * (half + 6) + half, y + 34), label,
                      cb=lambda n=i + 1: self.set_round(n), on=self.round_no == i + 1, scale=0.48)
        y += 42
        put(canvas, f"maze {self.map.nx}x{self.map.ny}, start {tuple(self.map.start)}  (change: Actions > Map)",
            (x, y + 10), 0.4, MUTED)
        f1 = self.round_file(1)
        if os.path.exists(f1):
            ts = time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(f1)))
            put(canvas, f"round 1 file saved {ts}", (x, y + 28), 0.4, OK)
        elif self.round_no == 2:
            put(canvas, "no round 1 file yet: round 2 would explore again", (x, y + 28), 0.4, AMBER)
        y += 44

        self._checkbox(canvas, x, y, ctl.armed, "Blaster armed (off = dry run: aims, never fires)",
                       cb=lambda: setattr(ctl, "armed", not ctl.armed))
        y += 26
        sel = ", ".join(kind_label(k) for k in sorted(self.selected)) or "NONE"
        put(canvas, f"shoot: {sel}"[:86], (x, y + 10), 0.4, ACCENT if self.selected else WARN)
        put(canvas, "(change it after Connect: Select tab)", (x, y + 26), 0.36, MUTED)
        y += 30

        busy = ctl.state == "connecting"
        r = (x, y0 + ch_ - 70, x + w, y0 + ch_ - 26)
        self._btn(canvas, r, "Connecting..." if busy else "Connect", cb=ctl.connect, kind="primary",
                  enabled=not busy, scale=0.7)
        if ctl.error:
            words, line, lines = ctl.error.split(), "", []
            for wd in words:
                if cv2.getTextSize(line + " " + wd, FONT, 0.45, 1)[0][0] > w:
                    lines.append(line)
                    line = wd
                else:
                    line = (line + " " + wd).strip()
            lines.append(line)
            for i, ln in enumerate(lines[:3]):
                put(canvas, ln, (x, r[1] - 50 + i * 16), 0.45, WARN)
        put(canvas, "Enter = Connect    q = quit", (x0 + cw_ - 220, y0 + ch_ - 8), 0.38, MUTED)
        return canvas

    def _draw_boresight(self, view, aim):
        """Crosshair; while aiming also the lock tolerance box and a big banner."""
        h, w = view.shape[:2]
        c = (w // 2, h // 2)
        col = PHASE_COLOR.get(aim["phase"], (255, 255, 255)) if aim["active"] else (255, 255, 255)
        cv2.line(view, (c[0] - 18, c[1]), (c[0] + 18, c[1]), col, 1)
        cv2.line(view, (c[0], c[1] - 18), (c[0], c[1] + 18), col, 1)
        if not aim["active"]:
            return
        f = self.detector.focal_px(w)
        r = max(4, int(f * math.tan(math.radians(aim["tol"]))))
        cv2.rectangle(view, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, 2)
        banner = {"LOCKED": "LOCKED", "FIRE": "FIRE!", "COARSE": "AIMING", "FINE": "AIMING"}.get(aim["phase"], aim["phase"])
        if aim["phase"] == "FIRE" and int(time.time() * 6) % 2:
            return  # flash, like the robot LED
        (tw, th), _ = cv2.getTextSize(banner, FONT, 1.1, 3)
        x, y = (w - tw) // 2, h - 40
        cv2.rectangle(view, (x - 12, y - th - 12), (x + tw + 12, y + 12), (255, 255, 255), -1)
        put(view, banner, (x, y), 1.1, col, 3)

    def _draw_sensors(self, canvas, x1, y1, x2, y2, tel):
        """Top view of the robot: front ToF + left/right Sharp IR bars (RoboFinal wall follower)."""
        card(canvas, x1, y1, x2, y2, "SENSORS")
        cx, cy = x1 + 82, y1 + 118
        bw, bh = 26, 34
        cv2.rectangle(canvas, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (90, 90, 90), -1)
        cv2.arrowedLine(canvas, (cx, cy + 8), (cx, cy - 10), (255, 255, 255), 2, tipLength=0.4)

        ir_max = float(tel.get("ir_max_cm", 30.0))
        wall = float(tel.get("ir_wall_cm", 16.9))
        side_len = 44
        for side, key in ((-1, "ir_left_cm"), (1, "ir_right_cm")):
            v = tel.get(key)
            x0 = cx + side * (bw // 2 + 2)
            xe = x0 + side * side_len
            cv2.line(canvas, (x0, cy), (xe, cy), LINE, 8)
            if v is not None:
                frac = min(max(v / ir_max, 0.0), 1.0)
                col = WARN if v <= wall else OK
                cv2.line(canvas, (x0, cy), (int(x0 + side * side_len * frac), cy), col, 8)
                wx = int(x0 + side * side_len * wall / ir_max)
                cv2.line(canvas, (wx, cy - 7), (wx, cy + 7), TEXT, 1)
                txt = f"{v:.1f}cm"
                (tw, _), _ = cv2.getTextSize(txt, FONT, 0.42, 1)
                put(canvas, txt, ((x0 + xe - tw) // 2, cy + 26), 0.42, col)
        put(canvas, "L", (cx - bw // 2 - side_len - 2, cy - 10), 0.4, MUTED)
        put(canvas, "R", (cx + bw // 2 + side_len - 8, cy - 10), 0.4, MUTED)

        tof = tel.get("tof_mm")
        top = y1 + 42
        cv2.line(canvas, (cx, cy - bh // 2 - 2), (cx, top), LINE, 8)
        if tof is not None:
            frac = min(max(tof / 1000.0, 0.0), 1.0)
            col = WARN if tof <= 300 else OK
            cv2.line(canvas, (cx, cy - bh // 2 - 2), (cx, int(cy - bh // 2 - 2 - (cy - bh // 2 - 2 - top) * frac)), col, 8)
            put(canvas, f"ToF {tof:.0f} mm", (cx + 10, top + 10), 0.42, col)

        rows = [("cell", str(tuple(self.map.robot))),
                ("head", f"{HEADING_DEG.get(self.map.heading, 0)} deg"),
                ("gimbal", f"{self.map.gimbal_abs:.0f} deg")]
        for k in ("yaw", "odom"):
            if k in tel:
                rows.append((k, str(tel[k]).replace(" m", "").replace(" deg", "").replace(", ", ",")))
        for i, (k, v) in enumerate(rows):
            y = y1 + 82 + i * 19
            put(canvas, k, (x1 + 166, y), 0.4, MUTED)
            put(canvas, v, (x1 + 212, y), 0.37, TEXT)

    def _draw_detections_card(self, canvas, x1, y1, x2, y2, dets):
        card(canvas, x1, y1, x2, y2, f"DETECTIONS  (<= {self.detector.max_shoot_m:.2f} m)")
        tgt = [d for d in dets if d.is_card]
        for i, d in enumerate(tgt[:4]):
            y = y1 + 44 + i * 20
            cv2.rectangle(canvas, (x1 + 10, y - 11), (x1 + 22, y + 1), DRAW_BGR.get(d.color, TEXT), -1)
            dist = f"{d.distance_m:.2f}m" if d.distance_m is not None else "--"
            put(canvas, f"{d.label}", (x1 + 28, y), 0.42, TEXT if d.is_target else MUTED)
            put(canvas, dist, (x1 + 150, y), 0.42, TEXT)
            if d.is_target:
                put(canvas, "IN RANGE" if d.in_range else "far", (x1 + 200, y), 0.42, OK if d.in_range else MUTED)
            else:
                put(canvas, "not sel.", (x1 + 200, y), 0.42, MUTED)
        if not tgt:
            put(canvas, "no card in view", (x1 + 10, y1 + 44), 0.42, MUTED)
        cv2.line(canvas, (x1 + 8, y1 + 124), (x2 - 8, y1 + 124), LINE, 1)
        for i, line in enumerate(self._log[-3:]):
            put(canvas, line[9:48], (x1 + 10, y1 + 142 + i * 15), 0.38, MUTED)

    def _draw_aim(self, canvas, x1, y1, x2, y2, aim):
        """RoboFinal auto-aim HUD: phase, error dot vs tolerance, lock counter, error trace."""
        card(canvas, x1, y1, x2, y2, f"AIM  (lock {aim['need']} frames)")
        # error plot: +-6 deg window, tolerance square in the middle
        span = 6.0
        px0, py0, sz = x1 + 10, y1 + 30, 100
        cv2.rectangle(canvas, (px0, py0), (px0 + sz, py0 + sz), (248, 247, 245), -1)
        cv2.rectangle(canvas, (px0, py0), (px0 + sz, py0 + sz), LINE, 1)
        mid = (px0 + sz // 2, py0 + sz // 2)
        cv2.line(canvas, (px0, mid[1]), (px0 + sz, mid[1]), LINE, 1)
        cv2.line(canvas, (mid[0], py0), (mid[0], py0 + sz), LINE, 1)
        t = int(sz / 2 * aim["tol"] / span)
        cv2.rectangle(canvas, (mid[0] - t, mid[1] - t), (mid[0] + t, mid[1] + t), OK, 1)

        def to_px(yaw, pitch):
            k = sz / 2 / span
            return (int(mid[0] + max(-span, min(span, yaw)) * k), int(mid[1] - max(-span, min(span, pitch)) * k))

        pcol = PHASE_COLOR.get(aim["phase"], MUTED)
        if aim["active"]:
            trail = [h for h in aim["hist"] if h[0] > time.time() - 4][-12:]
            for a, b in zip(trail, trail[1:]):
                cv2.line(canvas, to_px(a[1], a[2]), to_px(b[1], b[2]), (200, 190, 180), 1, cv2.LINE_AA)
            if aim["yaw"] is not None:
                cv2.circle(canvas, to_px(aim["yaw"], aim["pitch"]), 5, pcol, -1, cv2.LINE_AA)

        # phase + numbers
        tx = px0 + sz + 12
        put(canvas, aim["phase"], (tx, y1 + 46), 0.6, pcol, 2)
        if aim["active"] and aim["color"]:
            c_ = aim["color"].split(" ")[0]
            put(canvas, kind_label(aim["color"].split(" #")[0]) if " " in aim["color"] else c_,
                (tx, y1 + 66), 0.42, DRAW_BGR.get(c_, TEXT))
        if aim["yaw"] is not None and aim["active"]:
            put(canvas, f"yaw {aim['yaw']:+.2f}", (tx, y1 + 86), 0.42, TEXT)
            put(canvas, f"pit {aim['pitch']:+.2f}", (tx, y1 + 104), 0.42, TEXT)
        for i in range(aim["need"]):  # lock counter boxes
            bx = tx + i * 18
            filled = aim["active"] and i < aim["count"]
            cv2.rectangle(canvas, (bx, y1 + 114), (bx + 12, y1 + 126), OK if filled else LINE, -1 if filled else 1)

        # error trace (last 10 s): yaw blue, pitch green, tolerance band
        gx0, gy0, gx1, gy1 = x1 + 10, y1 + 138, x2 - 10, y2 - 8
        cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (248, 247, 245), -1)
        gm = (gy0 + gy1) // 2
        band = max(1, int((gy1 - gy0) / 2 * aim["tol"] / span))
        cv2.rectangle(canvas, (gx0, gm - band), (gx1, gm + band), (225, 240, 225), -1)
        cv2.line(canvas, (gx0, gm), (gx1, gm), LINE, 1)
        now = time.time()
        pts = [h for h in aim["hist"] if h[0] > now - 10]
        for idx, col in ((1, ACCENT), (2, OK)):
            poly = [(int(gx1 - (now - h[0]) / 10.0 * (gx1 - gx0)),
                     int(gm - max(-span, min(span, h[idx])) / span * (gy1 - gy0) / 2)) for h in pts]
            if len(poly) > 1:
                cv2.polylines(canvas, [np.array(poly, np.int32)], False, col, 1, cv2.LINE_AA)

    def _editor_preview(self, ms):
        size = self._parse_size() or (self.map.nx, self.map.ny)
        start = self.edit_start or (0, 0)
        start = (min(start[0], size[0] - 1), min(start[1], size[1] - 1))
        return MissionMap(size[0], size[1], self.map.tile, start).render(ms, title="click a cell = START")

    def _draw_editor(self, canvas, mx, my, ms):
        x1, y1 = mx + 20, my + ms - 150
        x2, y2 = mx + ms - 20, my + ms - 10
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PANEL, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), ACCENT, 1)
        put(canvas, "CUSTOM MAP  (width x height)", (x1 + 10, y1 + 22), 0.5, ACCENT)
        valid = self._parse_size() is not None
        cursor = "_" if int(time.time() * 2) % 2 else " "
        cv2.rectangle(canvas, (x1 + 10, y1 + 32), (x1 + 130, y1 + 62), (246, 244, 241), -1)
        cv2.rectangle(canvas, (x1 + 10, y1 + 32), (x1 + 130, y1 + 62), ACCENT, 1)
        put(canvas, self.edit_text + cursor, (x1 + 18, y1 + 55), 0.7, TEXT if valid else WARN, 2)
        st = self.edit_start or (0, 0)
        put(canvas, f"start ({st[0]},{st[1]})", (x1 + 145, y1 + 54), 0.5, TEXT)
        for i, preset in enumerate(("6x6", "5x4", "4x4", "5x5")):
            bx = x1 + 10 + i * 64
            self._btn(canvas, (bx, y1 + 72, bx + 56, y1 + 96), preset,
                      cb=lambda p=preset: setattr(self, "edit_text", p), on=self.edit_text == preset, scale=0.45)
        self._btn(canvas, (x2 - 150, y1 + 106, x2 - 80, y1 + 132), "SAVE", cb=self.save_editor, kind="primary")
        self._btn(canvas, (x2 - 72, y1 + 106, x2 - 8, y1 + 132), "CANCEL",
                  cb=lambda: setattr(self, "editing", False))
        put(canvas, self.edit_msg or "type e.g. 5x4, Enter = save", (x1 + 10, y1 + 124), 0.4,
            WARN if self.edit_msg else MUTED)

    def tick(self):
        now = time.time()
        if self._last_tick:
            dt = now - self._last_tick
            if dt > 0:
                self.ui_fps = 0.9 * self.ui_fps + 0.1 / dt if self.ui_fps else 1.0 / dt
        self._last_tick = now
        return self.render()

    def run_ui(self, stop_when=None):
        """Main-thread loop at `panel_fps`. Returns when q/Esc is pressed or
        stop_when() becomes true."""
        period = 1.0 / self.target_fps
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        deadline = time.time()
        while True:
            img = self.tick()
            cv2.imshow(WINDOW_NAME, img)
            # fixed-rate schedule (no drift): aim for exactly panel_fps
            deadline = max(deadline + period, time.time() - period)
            key = cv2.waitKey(max(1, int((deadline - time.time()) * 1000))) & 0xFF
            if key != 255 and not self.on_key(key):
                break
            if key == ord("s") and not self.editing:
                path = os.path.join(self.data_dir, f"panel_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(path, img)
                self.log(f"snapshot -> {os.path.relpath(path, BASE_DIR)}")
            if stop_when and stop_when():
                break

    def close(self):
        self.worker.stop()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass  # already closed (e.g. by destroyAllWindows)


def require_sdk():
    """Import the DJI SDK or explain why it is missing."""
    try:
        from robomaster import robot
        return robot
    except ImportError:
        import platform
        import sys
        raise SystemExit(
            "\n[!] The DJI 'robomaster' SDK is not installed.\n"
            f"    This machine: {platform.system()} {platform.machine()}, Python {sys.version.split()[0]}\n"
            "    On a Mac:  bash tools/macos/setup_robomaster_mac.sh   (from RoboFinal)\n"
            "    On Windows/Linux x86_64 with Python 3.6-3.8:  pip install -r requirements.txt\n"
            "    Without the robot you can still run:  python src/main_mission.py --source demo --connect\n")


def save_setting(section, key, value, path=None):
    """Set `key: value` inside a top-level section of config/settings.yaml, keeping comments."""
    path = path or os.path.join(BASE_DIR, "config", "settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(rf"^{section}:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*", text, re.M)
    if not m:
        text = text.rstrip() + f"\n\n{section}:\n  {key}: {value}\n"
    else:
        block = m.group(0)
        line = re.compile(rf"^(  {key}:[ \t]*)[^#\n]*?([ \t]*(#[^\n]*)?)$", re.M)
        if line.search(block):
            block = line.sub(lambda mm: f"{mm.group(1)}{value}{mm.group(2) or ''}", block, count=1)
        else:
            block = block.rstrip("\n") + f"\n  {key}: {value}\n\n"
        text = text[:m.start()] + block + text[m.end():]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_grid_to_settings(width, height, start=(0, 0), path=None):
    """Write the map size / start cell into config/settings.yaml, keeping comments."""
    path = path or os.path.join(BASE_DIR, "config", "settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^grid_map:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*", text, re.M)
    if not m:
        raise ValueError("grid_map section not found")
    block = m.group(0)
    new = re.sub(r"^  max_x:.*$", f"  max_x: {width - 1}          # {width}x{height} tiles -> indices 0..{width - 1}",
                 block, count=1, flags=re.M)
    new = re.sub(r"^  max_y:.*$", f"  max_y: {height - 1}", new, count=1, flags=re.M)
    new = re.sub(r"(^  start:[^\n]*\n\s+x:\s*)\d+(\s*\n\s+y:\s*)\d+",
                 rf"\g<1>{start[0]}\g<2>{start[1]}", new, count=1, flags=re.M)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:m.start()] + new + text[m.end():])


# ======================================================================
# stand-alone use
# ======================================================================
def run_app(config, source="robot", round_no=1, connection=None, resolution=None, demo_images=(),
            connect=False, autostart=False, armed=None, snapshot=None, ui="pygame"):
    """Open the panel with the Connect screen (RoboFinal-style control panel).
    ui = "pygame" (default window, panel_pygame.py) or "opencv" (the older cv2 window)."""
    from control import MissionController

    panel = MissionPanel(config, round_no=round_no)
    ctl = MissionController(config, panel, source=source, connection=connection, resolution=resolution,
                            demo_images=demo_images, armed=armed)
    panel.start()
    if connect or autostart:
        ctl.auto_start = autostart
        ctl.connect()
    if ui == "pygame":
        from panel_pygame import PygamePanel
        app = PygamePanel(panel, ctl, headless=bool(snapshot))
        if snapshot:  # headless check: wait for the round, save one frame
            t_end = time.time() + 60
            while time.time() < t_end and not (ctl.state == "done" or (not autostart and ctl.state == "idle")):
                app.step([])
                time.sleep(1 / 30.0)
            for _ in range(30):
                app.step([])
                time.sleep(1 / 30.0)
            pygame_save(app.screen, snapshot)
            print(f"snapshot -> {snapshot}  (camera {panel.worker.fps:.1f} fps)")
            ctl.shutdown()
            panel.worker.stop()
            return
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            ctl.shutdown()
            panel.worker.stop()
        return
    if snapshot:  # headless check: wait for the round, save one frame
        t_end = time.time() + 60
        while time.time() < t_end and not (ctl.state == "done" or (not autostart and ctl.state == "idle")):
            time.sleep(0.1)
        period = 1.0 / panel.target_fps
        deadline = time.time()
        for _ in range(60):
            img = panel.tick()
            deadline += period
            time.sleep(max(0.0, deadline - time.time()))
        cv2.imwrite(snapshot, img)
        print(f"snapshot -> {snapshot}  (camera {panel.worker.fps:.1f} fps, UI {panel.ui_fps:.1f} fps)")
        ctl.shutdown()
        panel.worker.stop()
        return
    try:
        panel.run_ui()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.shutdown()
        panel.close()


def pygame_save(surface, path):
    import pygame
    pygame.image.save(surface, path)


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config_loader import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", action="store_true", help="connect to the robot straight away")
    ap.add_argument("--webcam", action="store_true", help="connect to this computer's camera straight away")
    ap.add_argument("--demo", nargs="*", help="demo source with these images (starts a simulated round)")
    ap.add_argument("--resolution", default=None, choices=("360p", "540p", "720p"))
    ap.add_argument("--snapshot", help="render headless and save one panel image (testing)")
    ap.add_argument("--ui", choices=("pygame", "opencv"), default="pygame")
    args = ap.parse_args()

    config = load_config()
    if args.demo is not None:
        config.setdefault("data_collection", {})["data_dir"] = "data/demo"  # never overwrite real rounds
        run_app(config, source="demo", demo_images=args.demo, connect=True, autostart=True,
                snapshot=args.snapshot, ui=args.ui)
    elif args.webcam:
        run_app(config, source="webcam", connect=True, snapshot=args.snapshot, ui=args.ui)
    else:
        run_app(config, source="robot", resolution=args.resolution, connect=args.robot,
                snapshot=args.snapshot, ui=args.ui)


if __name__ == "__main__":
    main()
