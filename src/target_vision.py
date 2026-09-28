"""Target detection + segmentation for the hostage-rescue maze assignment.

The "villain" cards are coloured acrylic plates on a stick. The assignment
sheet uses these colours and shapes:

    colours  blue, red, yellow, green
    shapes   circle, wide rectangle (landscape), tall rectangle (portrait), square

(the sheet's own set is blue circle, red wide rect, yellow tall rect, green
square). Every colour x shape combination is recognised as a *card* and put on
the map; a card is a *target* (shot) only when its kind - e.g. "blue circle",
"red circle" - is in the selected set, which can be switched on the panel.
Shooting a card that was not selected costs -1, so the rest are only mapped.

Only the camera is used (no extra sensor), so the sensor budget of the
assignment is not affected.  Each frame is segmented per colour in HSV,
cleaned with morphology, then every blob is classified by shape.

Distance is estimated with a pinhole model from the card's known size, so
the robot can tell whether a target is within the 2-tile shooting range.
"""

import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLOR_CONFIG_PATH = os.path.join(BASE_DIR, "config", "color_config.json")

# colours and shapes on the assignment sheet
COLORS = ("blue", "red", "yellow", "green")
SHAPES = ("circle", "rect_wide", "rect_tall", "square")

# the sheet's own set (colour -> shape): the default selection
DEFAULT_CATALOGUE = {
    "blue": "circle",
    "red": "rect_wide",
    "yellow": "rect_tall",
    "green": "square",
}

# real card size (width, height) in metres per SHAPE, measured on the lab's cards
# by RoboFinal: circle 7 cm across, square 7 x 7 cm, rectangle 9 x 6 cm
# (settings.yaml -> vision.plate_size_m). Used for the distance estimate.
DEFAULT_PLATE_SIZE_M = {
    "circle": (0.07, 0.07),
    "square": (0.07, 0.07),
    "rect_wide": (0.09, 0.06),
    "rect_tall": (0.06, 0.09),
}


def kind_of(color, shape):
    """'blue', 'circle' -> 'blue circle' (the key of a card kind everywhere)."""
    return f"{color} {shape}"


def split_kind(kind):
    color, shape = kind.split(" ", 1)
    return color, shape


def kind_label(kind):
    color, shape = split_kind(kind)
    return f"{color} {SHAPE_LABEL.get(shape, shape)}"


def parse_selection(items, catalogue=None):
    """Config list -> set of kinds. Accepts 'blue circle', 'red rect_wide', 'red wide rect'
    and plain colours (old config: colour -> the sheet's shape for it)."""
    catalogue = catalogue or DEFAULT_CATALOGUE
    words = {"wide rect": "rect_wide", "tall rect": "rect_tall", "wide": "rect_wide", "tall": "rect_tall"}
    out = set()
    for item in items or []:
        item = str(item).strip().lower()
        if " " not in item:
            if item in catalogue:
                out.add(kind_of(item, catalogue[item]))
            continue
        color, shape = item.split(" ", 1)
        shape = words.get(shape, shape)
        if color in COLORS and shape in SHAPES:
            out.add(kind_of(color, shape))
    return out

DEFAULT_HSV = {
    "red": {"h_min": 165, "h_max": 10, "s_min": 100, "s_max": 255, "v_min": 60, "v_max": 255, "min_area": 250},
    "green": {"h_min": 45, "h_max": 85, "s_min": 80, "s_max": 255, "v_min": 40, "v_max": 255, "min_area": 250},
    "blue": {"h_min": 100, "h_max": 130, "s_min": 100, "s_max": 255, "v_min": 30, "v_max": 255, "min_area": 250},
    "yellow": {"h_min": 18, "h_max": 34, "s_min": 90, "s_max": 255, "v_min": 80, "v_max": 255, "min_area": 250},
}

# BGR drawing colours
DRAW_BGR = {
    "red": (40, 40, 230),
    "green": (60, 190, 40),
    "blue": (220, 110, 30),
    "yellow": (0, 220, 240),
}

SHAPE_LABEL = {
    "circle": "circle",
    "rect_wide": "wide rect",
    "rect_tall": "tall rect",
    "square": "square",
    "unknown": "?",
}

# detection runs on a frame resized to this width (speed; keeps 30 fps)
PROC_WIDTH = 640


@dataclass
class Detection:
    color: str
    shape: str
    is_target: bool          # a card whose kind is selected (to shoot)
    contour: np.ndarray      # in full-frame pixel coordinates
    bbox: tuple              # (x, y, w, h) full-frame pixels
    center: tuple            # (cx, cy) full-frame pixels
    area: float              # full-frame pixels^2
    bearing_deg: float = 0.0     # + = right of camera centre
    elevation_deg: float = 0.0   # + = above camera centre
    distance_m: float = None
    in_range: bool = False
    extra: dict = field(default_factory=dict)
    is_card: bool = False    # a real card of any sheet colour x shape (mapped, maybe not shot)

    @property
    def kind(self):
        return kind_of(self.color, self.shape)

    @property
    def label(self):
        return f"{self.color} {SHAPE_LABEL.get(self.shape, self.shape)}"


def _load_hsv():
    data = {k: dict(v) for k, v in DEFAULT_HSV.items()}
    if os.path.exists(COLOR_CONFIG_PATH):
        try:
            with open(COLOR_CONFIG_PATH, "r", encoding="utf-8") as f:
                for name, params in json.load(f).items():
                    data.setdefault(name, {}).update(params)
        except Exception as e:
            print(f"[vision] cannot read {COLOR_CONFIG_PATH}: {e}")
    return data


class TargetDetector:
    def __init__(self, config=None):
        config = config or {}
        vis = config.get("vision", {}) or {}
        move = config.get("movement", {}) or {}

        self.hsv = _load_hsv()
        self.catalogue = dict(DEFAULT_CATALOGUE)
        self.catalogue.update(vis.get("catalogue", {}) or {})
        self.plate = dict(DEFAULT_PLATE_SIZE_M)
        for name, wh in (vis.get("plate_size_m", {}) or {}).items():
            if name in SHAPES:
                self.plate[name] = (float(wh[0]), float(wh[1]))
        # colours that can be cards at all (drop one here if the room is full of it)
        self.card_colors = set(vis.get("card_colors", list(COLORS)))
        # kinds to shoot, e.g. {"blue circle", "red circle"} - switched on the panel
        sel = vis.get("shoot_targets")
        self.selected = parse_selection(sel, self.catalogue) if sel else {
            kind_of(c, s) for c, s in self.catalogue.items()}
        self.min_solidity = float(vis.get("min_solidity", 0.85))
        # a card is a small patch: this much of the same colour right around it = a wall/floor, not a card
        self.max_ring_fill = float(vis.get("max_ring_fill", 0.25))

        self.hfov_deg = float(vis.get("hfov_deg", 96.0))
        self.bottom_ignore = float(vis.get("bottom_ignore_ratio", 0.15))
        self.top_ignore = float(vis.get("top_ignore_ratio", 0.0))
        self.max_area_ratio = float(vis.get("max_area_ratio", 0.25))
        self.min_area_px = float(vis.get("min_area_px", 80))  # at the 640 px processing width
        # --- ignore the room: everything above the arena's white walls / above the horizon
        self.ignore_above_wall = bool(vis.get("ignore_above_wall", True))
        self.wall_s_max = int(vis.get("wall_s_max", 60))          # white / grey foam: low saturation
        self.wall_v_min = int(vis.get("wall_v_min", 120))         # ... and bright
        self.wall_margin = float(vis.get("wall_margin_frac", 0.03))  # keep this much (of the height) above the line
        self.wall_max_gap = float(vis.get("wall_max_gap_frac", 0.25))  # non-white gap still inside the wall band
        # cards hang lower than the camera, so their centre is below the horizon; the room is above it
        self.max_elevation_deg = vis.get("max_elevation_deg", 1.5)  # None / off = no horizon rule
        self.gimbal_pitch_deg = 0.0     # live gimbal pitch (set by the chassis) - moves the horizon
        # a card hangs at card height: tape on the floor (and things up in the room) are not at it
        self.camera_height_m = float(vis.get("camera_height_m", 0.25))
        self.card_min_height_m = float(vis.get("card_min_height_m", 0.06))
        self.card_max_height_m = float(vis.get("card_max_height_m", 0.30))
        self.last_ignore_line = None    # per full-res column: y above which everything was ignored
        self.last_wall_top = None       # per full-res column: top edge of the white foam walls (NaN = none)
        self.last_frame_size = None     # (W, H) of the frame those lines belong to
        self.last_pitch_deg = 0.0       # gimbal pitch when that frame was processed
        self.tile_m = float(move.get("distance", 0.6))
        self.max_shoot_tiles = float(vis.get("max_shoot_tiles", 2))
        self.kernel = np.ones((5, 5), np.uint8)

    # ------------------------------------------------------------------
    @property
    def max_shoot_m(self):
        return self.max_shoot_tiles * self.tile_m

    def focal_px(self, width):
        return (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def color_mask(self, hsv, name):
        p = self.hsv[name]
        lo_s, hi_s = p.get("s_min", 0), p.get("s_max", 255)
        lo_v, hi_v = p.get("v_min", 0), p.get("v_max", 255)
        if p["h_min"] <= p["h_max"]:
            return cv2.inRange(hsv, (p["h_min"], lo_s, lo_v), (p["h_max"], hi_s, hi_v))
        # hue wraps around 180 (red)
        m1 = cv2.inRange(hsv, (0, lo_s, lo_v), (p["h_max"], hi_s, hi_v))
        m2 = cv2.inRange(hsv, (p["h_min"], lo_s, lo_v), (179, hi_s, hi_v))
        return cv2.bitwise_or(m1, m2)

    @staticmethod
    def classify_shape(contour):
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        if area <= 0 or peri <= 0:
            return "unknown", {}

        circularity = 4.0 * math.pi * area / (peri * peri)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_fill = area / (math.pi * radius * radius) if radius > 0 else 0.0

        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        rect_fill = area / (rw * rh) if rw * rh > 0 else 0.0
        _, _, bw, bh = cv2.boundingRect(contour)
        aspect = bw / float(bh) if bh else 0.0  # plates stand upright -> use upright box

        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        info = {
            "circularity": round(circularity, 2),
            "circle_fill": round(circle_fill, 2),
            "rect_fill": round(rect_fill, 2),
            "aspect": round(aspect, 2),
            "vertices": len(approx),
        }

        # a disc fills ~78% of its bounding box; a rectangle fills ~100%
        if circularity >= 0.78 and circle_fill >= 0.72 and rect_fill < 0.88 and 0.75 <= aspect <= 1.33:
            return "circle", info
        if rect_fill >= 0.82 and 4 <= len(approx) <= 6:
            # a card is 9 x 6 cm (1.5 : 1); a long thin strip is tape on the floor or a wall edge
            if aspect > 2.4 or aspect < 0.42:
                info["rejected"] = "long thin strip (tape)"
                return "unknown", info
            if aspect > 1.28:
                return "rect_wide", info
            if aspect < 0.78:
                return "rect_tall", info
            return "square", info
        return "unknown", info

    # ------------------------------------------------------------------
    def detect(self, frame):
        """Return a list of Detection (full-frame coordinates), best first."""
        if frame is None:
            return []
        H, W = frame.shape[:2]
        scale = PROC_WIDTH / float(W) if W > PROC_WIDTH else 1.0
        small = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        h, w = small.shape[:2]

        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        y_top = int(h * self.top_ignore)
        y_bot = int(h * (1.0 - self.bottom_ignore))  # blaster barrel sits at the bottom

        f_px = self.focal_px(W)
        cx0, cy0 = W / 2.0, H / 2.0
        inv = 1.0 / scale
        results = []

        # the room around the arena (people, tables, yellow walls) never holds a card
        ignore_y = self.ignore_line(hsv, f_px * scale)          # per processing column, or None
        self.last_ignore_line = None if ignore_y is None else np.interp(
            np.arange(W) * scale, np.arange(w), ignore_y) * inv
        wt = self._wall_top_proc
        self.last_wall_top = None if wt is None else np.interp(np.arange(W) * scale, np.arange(w), wt) * inv
        self.last_frame_size = (W, H)
        self.last_pitch_deg = self.gimbal_pitch_deg
        ignore_mask = None
        if ignore_y is not None:
            ignore_mask = np.arange(h)[:, None] < ignore_y[None, :]

        for name in self.hsv:
            if name not in COLORS:
                continue
            mask = self.color_mask(hsv, name)
            mask[:y_top, :] = 0
            mask[y_bot:, :] = 0
            if ignore_mask is not None:
                mask[ignore_mask] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

            min_area_small = self.min_area_px
            max_area_small = h * w * self.max_area_ratio

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cands = []
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < min_area_small or a > max_area_small:
                    continue
                # a blob cut by the frame edge (walls, half-visible plates) has
                # an unreliable shape -> never treat it as a target
                sx, sy, sw, sh = cv2.boundingRect(cnt)
                if sx <= 2 or sy <= y_top + 2 or sx + sw >= w - 2 or sy + sh >= y_bot - 2:
                    continue
                # cut by the ignore line (something of the room reaching down): unreliable too
                if ignore_y is not None and sy <= ignore_y[sx:sx + sw].max() + 2:
                    continue
                shape, info = self.classify_shape(cnt)
                hull_a = cv2.contourArea(cv2.convexHull(cnt))
                solidity = a / hull_a if hull_a > 0 else 0.0
                info["solidity"] = round(solidity, 2)
                # same colour all around the blob = part of a wall / tape / floor
                mx_, my_ = max(3, int(sw * 0.35)), max(3, int(sh * 0.35))
                x0, y0 = max(0, sx - mx_), max(0, sy - my_)
                x1_, y1_ = min(w, sx + sw + mx_), min(h, sy + sh + my_)
                ring = mask[y0:y1_, x0:x1_].copy()
                ring[sy - y0:sy - y0 + sh, sx - x0:sx - x0 + sw] = 0
                ring_area = ring.size - sw * sh
                ring_fill = float(np.count_nonzero(ring)) / ring_area if ring_area > 0 else 0.0
                info["ring_fill"] = round(ring_fill, 2)
                full_cnt = (cnt.astype(np.float32) * inv).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(full_cnt)
                if shape in SHAPES:
                    # blur + downscale move the edge by a few px depending on the colour
                    # (yellow grows, blue shrinks): measure the size again at full resolution
                    x, y, bw, bh = self._refine_bbox(frame, (x, y, bw, bh))
                m = cv2.moments(full_cnt)
                if m["m00"] > 0:
                    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
                else:
                    cx, cy = x + bw / 2.0, y + bh / 2.0

                bearing = math.degrees(math.atan2(cx - cx0, f_px))
                elevation = math.degrees(math.atan2(cy0 - cy, f_px))
                dist = None
                size = self.plate.get(shape)
                if size and bw > 0 and bh > 0:
                    # both sides of the card -> average (a circle / square gives two estimates)
                    dist = f_px * 0.5 * (size[0] / float(bw) + size[1] / float(bh))

                # height above the floor from the distance and the (world) elevation
                height = None
                if dist is not None:
                    height = self.camera_height_m + dist * math.tan(math.radians(elevation + self.gimbal_pitch_deg))
                    info["height_m"] = round(height, 3)
                at_card_height = height is None or self.card_min_height_m <= height <= self.card_max_height_m
                is_card = (shape in SHAPES and name in self.card_colors and at_card_height
                           and solidity >= self.min_solidity and ring_fill <= self.max_ring_fill)
                if shape in SHAPES and not is_card:
                    info["rejected"] = ("not at card height (floor / room)" if not at_card_height else
                                        "not solid" if solidity < self.min_solidity else
                                        "part of a bigger patch" if ring_fill > self.max_ring_fill else "colour off")
                is_target = is_card and kind_of(name, shape) in self.selected
                cands.append(Detection(
                    color=name, shape=shape, is_target=is_target, is_card=is_card,
                    contour=full_cnt, bbox=(x, y, bw, bh), center=(int(cx), int(cy)),
                    area=a * inv * inv, bearing_deg=bearing, elevation_deg=elevation,
                    distance_m=dist,
                    in_range=bool(is_target and dist is not None and dist <= self.max_shoot_m),
                    extra=info,
                ))
            results.extend(self._drop_reflections(cands))

        # targets first, then other cards, then larger blobs
        results.sort(key=lambda d: (not d.is_target, not d.is_card, -d.area))
        return results

    def ignore_line(self, hsv, f_proc):
        """Per processing column, the row above which the picture is not the arena:
        the top of the white foam walls (+ floor) band, or the horizon (cards hang below
        the camera), whichever is lower. None = nothing to ignore."""
        h, w = hsv.shape[:2]
        line = np.full(w, -1.0)
        top = self.wall_top(hsv)          # also used for the camera wall distance (wall_ahead_m)
        self._wall_top_proc = top
        if self.ignore_above_wall:
            if top is not None:
                line = np.maximum(line, np.where(np.isnan(top), -1.0, top - self.wall_margin * h))
        if self.max_elevation_deg not in (None, False, "off"):
            # world elevation E appears at row cy - f * tan(E - gimbal_pitch)
            ang = math.radians(float(self.max_elevation_deg) - self.gimbal_pitch_deg)
            horizon_y = h / 2.0 - f_proc * math.tan(ang)
            line = np.maximum(line, horizon_y)
        return line if (line > 0).any() else None

    def wall_top(self, hsv):
        """Top edge of the arena's white band (foam walls + floor) per processing column.

        White = low saturation and bright. Each column is walked up from the arena floor
        (a white run in the lower part of the picture): white runs above it join while the
        gap between them is card-sized or smaller (a card in front of a wall, tape, seams),
        and the walk stops at a bigger gap - the room: people, chairs, white tables.
        The edge is then smoothed over neighbouring columns (RoboFinal's idea in
        final/looking.py). NaN = no arena in that column."""
        h, w = hsv.shape[:2]
        cols = 160
        rows = max(20, int(round(h * cols / float(w))))
        white = ((hsv[:, :, 1] <= self.wall_s_max) & (hsv[:, :, 2] >= self.wall_v_min)).astype(np.float32)
        white = (cv2.resize(white, (cols, rows), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
        max_gap = int(rows * self.wall_max_gap)      # bigger than any card seen from >= 0.3 m
        low = int(rows * 0.6)                        # the arena floor starts in the lower 40 %
        min_run = max(2, rows // 25)
        pad = np.zeros((1, cols), np.int8)
        edges = np.diff(np.vstack([pad, white.astype(np.int8), pad]), axis=0)
        top = np.full(cols, np.nan)
        for c in range(cols):
            starts = np.flatnonzero(edges[:, c] == 1)   # run = [start, end)
            ends = np.flatnonzero(edges[:, c] == -1)
            base = [k for k in range(len(starts)) if ends[k] > low and ends[k] - starts[k] >= min_run]
            if not base:
                continue
            k = max(base, key=lambda i: ends[i])      # the floor run nearest the robot
            t = starts[k]
            for j in range(k - 1, -1, -1):             # walk up while gaps are card-sized
                if t - ends[j] > max_gap:
                    break
                t = starts[j]
            top[c] = t
        half = max(3, cols // 24)
        smooth = np.full(cols, np.nan)
        for c in range(cols):
            win = top[max(0, c - half):c + half + 1]
            ok = ~np.isnan(win)
            if ok.sum() * 2 >= len(win):
                smooth[c] = np.median(win[ok])
        if np.isnan(smooth).all():
            return None
        xs = (np.arange(cols) + 0.5) * w / cols
        valid = ~np.isnan(smooth)
        out = np.interp(np.arange(w), xs[valid], smooth[valid] * h / rows)
        nearest = np.interp(np.arange(w), xs, valid.astype(float))
        out[nearest < 0.5] = np.nan
        return out

    def wall_ahead_m(self, wall_rise_m, half_width_m=0.17):
        """Distance to the nearest white wall in the robot's path, from the camera.

        The foam walls are taller than the camera, so a wall's top edge is above the
        horizon by an angle that grows as it gets closer:
            distance = f * wall_rise / (pixels above the horizon)
        wall_rise_m = wall height - camera height (calibrated against the ToF).
        Only the columns the robot body would sweep through (+-half_width_m) count.
        Returns (distance_m or inf, pixels_above_horizon_at_the_centre or None)."""
        top, size = self.last_wall_top, self.last_frame_size
        if top is None or size is None or wall_rise_m <= 0:
            return float("inf"), None
        if abs(self.last_pitch_deg) > 3.0:
            return float("inf"), None      # camera tilted: the wall-top geometry does not hold
        W, H = size
        f = self.focal_px(W)
        horizon = H / 2.0 + f * math.tan(math.radians(self.last_pitch_deg))
        cx = W / 2.0
        best = float("inf")
        centre = top[int(cx) - 4:int(cx) + 5]
        centre = centre[~np.isnan(centre) & (centre > 3)]
        centre_rise = float(horizon - np.median(centre)) if len(centre) else None
        for x in range(0, W, 4):
            row = top[x]
            if np.isnan(row) or row <= 3:
                # white right up to the picture's top edge: the wall's top is not in view
                # (a side wall close by, or the camera looking down) - no distance from it
                continue
            rise = horizon - row
            if rise <= 3:              # at / below the horizon: far away (or not a wall top)
                continue
            d = f * wall_rise_m / rise
            if abs(x - cx) <= f * half_width_m / max(d, 0.05):   # inside the robot's path at that range
                best = min(best, d)
        return best, centre_rise

    def set_selected(self, kinds):
        """Switch which card kinds are targets (takes effect from the next frame)."""
        self.selected = set(kinds)

    @staticmethod
    def _refine_bbox(frame, bbox):
        """Card box at full resolution: a pixel belongs to the card when its colour is
        nearer the card's centre colour than the surrounding colour (edge = halfway)."""
        x, y, bw, bh = bbox
        H, W = frame.shape[:2]
        pad = max(4, int(0.3 * max(bw, bh)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + bw + pad), min(H, y + bh + pad)
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        if roi.shape[0] < 6 or roi.shape[1] < 6:
            return bbox
        cx, cy = x + bw // 2 - x0, y + bh // 2 - y0
        ix, iy = max(1, bw // 5), max(1, bh // 5)
        inner = roi[cy - iy:cy + iy + 1, cx - ix:cx + ix + 1].reshape(-1, 3)
        border = np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]])
        if inner.size == 0:
            return bbox
        c_in, c_out = np.median(inner, axis=0), np.median(border, axis=0)
        if np.linalg.norm(c_in - c_out) < 25:  # no contrast: keep the first estimate
            return bbox
        fg = (np.linalg.norm(roi - c_in, axis=2) < np.linalg.norm(roi - c_out, axis=2)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=4)
        lab = labels[min(cy, fg.shape[0] - 1), min(cx, fg.shape[1] - 1)]
        if lab == 0:
            return bbox
        rx, ry, rw, rh = stats[lab, :4]
        # the edge bias is a few px: never move an edge further than that
        # (a floor reflection or the stick touching the card would pull it away)
        mx_, my_ = max(3, int(0.15 * bw)), max(3, int(0.15 * bh))
        nx0 = min(max(x0 + rx, x - mx_), x + mx_)
        ny0 = min(max(y0 + ry, y - my_), y + my_)
        nx1 = min(max(x0 + rx + rw, x + bw - mx_), x + bw + mx_)
        ny1 = min(max(y0 + ry + rh, y + bh - my_), y + bh + my_)
        return (int(nx0), int(ny0), int(nx1 - nx0), int(ny1 - ny0))

    @staticmethod
    def _drop_reflections(cands):
        """The shiny floor mirrors each plate just below it; keep the upper one."""
        keep = []
        for d in cands:
            x, y, w, h = d.bbox
            mirrored = False
            for o in cands:
                if o is d:
                    continue
                ox, oy, ow, oh = o.bbox
                overlap = min(x + w, ox + ow) - max(x, ox)
                if overlap > 0.5 * min(w, ow) and oy + oh <= y + 0.35 * h and o.area >= 0.5 * d.area:
                    mirrored = True
                    break
            if not mirrored:
                keep.append(d)
        return keep


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------
def draw_detections(frame, detections, show_all=True, alpha=0.45):
    """Overlay segmentation masks, outlines and labels onto frame (in place)."""
    if not detections:
        return frame
    overlay = frame.copy()
    for d in detections:
        if not d.is_target and not show_all:
            continue
        col = DRAW_BGR.get(d.color, (255, 255, 255))
        cv2.drawContours(overlay, [d.contour], -1, col, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    for d in detections:
        if not d.is_target and not show_all:
            continue
        col = DRAW_BGR.get(d.color, (255, 255, 255))
        x, y, w, h = d.bbox
        if d.is_target:
            cv2.drawContours(frame, [d.contour], -1, col, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 255, 255), 1, cv2.LINE_AA)
            cx, cy = d.center
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        elif d.is_card:
            # a real card, but its kind is not selected: mapped, never shot
            cv2.drawContours(frame, [d.contour], -1, col, 1, cv2.LINE_AA)
        else:
            # not a card (wall, tape, odd shape): outline only, no label clutter
            cv2.drawContours(frame, [d.contour], -1, (160, 160, 160), 1, cv2.LINE_AA)
            continue

        dist = f"{d.distance_m:.2f}m" if d.distance_m is not None else "--"
        tag = "TARGET" if d.is_target else "not selected"
        text = f"{d.label}  {dist}  {tag}"
        if d.in_range:
            text += "  IN RANGE"
        scale = 0.5 if d.is_target else 0.42
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y - 8 if y - 8 - th > 0 else y + h + th + 8
        cv2.rectangle(frame, (x - 2, ty - th - 4), (x + tw + 4, ty + 4), (20, 20, 20), -1)
        cv2.putText(frame, text, (x + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (80, 255, 120) if d.in_range else (col if d.is_target else (190, 190, 190)),
                    1, cv2.LINE_AA)
    return frame


def draw_ignored(frame, line, color=(255, 200, 0)):
    """Dim the part of the picture that detection ignores (above the arena walls /
    the horizon) and draw its edge. `line` = per-column y (TargetDetector.last_ignore_line)."""
    if line is None:
        return frame
    H, W = frame.shape[:2]
    if len(line) != W:
        line = np.interp(np.arange(W), np.linspace(0, W - 1, len(line)), line)
    ys = np.clip(np.nan_to_num(line, nan=-1).astype(int), -1, H - 1)
    above = np.arange(H)[:, None] <= ys[None, :]
    frame[above] = (frame[above] * 0.45 + np.array((60, 50, 40)) * 0.55).astype(np.uint8)
    pts = np.stack([np.arange(W), ys], axis=1)[ys >= 0]
    if len(pts) > 1:
        cv2.polylines(frame, [pts.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    return frame


def segmentation_mask(frame_shape, detections, targets_only=True):
    """Colour-coded segmentation image (black background)."""
    seg = np.zeros(frame_shape[:2] + (3,), dtype=np.uint8)
    for d in detections:
        if targets_only and not d.is_card:
            continue
        cv2.drawContours(seg, [d.contour], -1, DRAW_BGR.get(d.color, (255, 255, 255)), cv2.FILLED)
    return seg


if __name__ == "__main__":
    # quick offline test:  python3 src/target_vision.py image1.jpg [image2.jpg ...]
    import sys

    det = TargetDetector()
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"cannot read {path}")
            continue
        found = det.detect(img)
        print(f"\n{path}: {len(found)} blob(s)")
        for d in found:
            dist = f"{d.distance_m:.2f}m" if d.distance_m else "--"
            print(f"  {d.color:<6} {d.shape:<9} target={d.is_target!s:<5} bbox={d.bbox} "
                  f"bearing={d.bearing_deg:+.1f}deg dist={dist} {d.extra}")
        out = draw_detections(img.copy(), found)
        out_path = os.path.splitext(path)[0] + "_det.png"
        cv2.imwrite(out_path, out)
        print(f"  -> {out_path}")
