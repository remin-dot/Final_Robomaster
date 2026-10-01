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
COLOR_SAMPLES_PATH = os.path.join(BASE_DIR, "config", "color_samples.json")   # clicked pixels (color_calibrate)
COLOR_LUT_PATH = os.path.join(BASE_DIR, "config", "color_lut.npz")          # built from them (cache)
LUT_BITS = 5                                                               # 32 x 32 x 32 BGR cells


def build_color_lut(samples, max_de=None, per_class=900, k=7):
    """Sample pixels {class: [[b, g, r], ...]} (after the wall white balance) -> a 32^3 BGR
    lookup table of class ids (0 = none, 1.. = sorted class names). Each cell is labelled
    with the majority of its k nearest samples in Lab (perceptual colour distance), and only
    when the nearest one is within that class's own spread (so a colour nobody clicked stays
    "none"). A "background" class (walls, wood, beige, skin, clothes clicked on purpose)
    takes away the pixels that only look like a card colour. Returns (lut, names)."""
    names = sorted(samples)
    rng = np.random.default_rng(0)
    pts, lab_ids = [], []
    for i, n in enumerate(names):
        a = np.array(samples[n], dtype=np.uint8).reshape(-1, 3)
        if len(a) > per_class:
            a = a[rng.choice(len(a), per_class, replace=False)]
        pts.append(a)
        lab_ids.append(np.full(len(a), i + 1, np.int16))
    if not pts:
        return None, names
    bgr = np.concatenate(pts)
    ids = np.concatenate(lab_ids)
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # each class's own spread: nearest-other-sample distance inside the class, 95th percentile
    thr = np.zeros(len(names) + 1, np.float32)
    for i in range(1, len(names) + 1):
        c = lab[ids == i]
        if len(c) < 2:
            thr[i] = 12.0
            continue
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e9)
        thr[i] = max(8.0, float(np.percentile(d.min(1), 95)) * 2.5)
        if max_de:
            thr[i] = min(thr[i], max_de)
    step = 1 << (8 - LUT_BITS)
    ax = (np.arange(1 << LUT_BITS) * step + step // 2).astype(np.uint8)
    cells = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)   # b, g, r
    cl = cv2.cvtColor(cells.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    out = np.zeros(len(cells), np.uint8)
    for s0 in range(0, len(cells), 2048):
        q = cl[s0:s0 + 2048]
        d = ((q[:, None, :] - lab[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d, min(k, d.shape[1] - 1), axis=1)[:, :k]
        votes = ids[nn]
        dk = np.take_along_axis(d, nn, 1)
        best = np.array([np.bincount(v, minlength=len(names) + 1).argmax() for v in votes])
        # distance to the nearest sample OF THE WINNING CLASS
        near_d = np.sqrt(np.where(votes == best[:, None], dk, 1e12).min(1))
        out[s0:s0 + 2048] = np.where(near_d <= thr[best], best, 0)
    n = 1 << LUT_BITS
    return out.reshape(n, n, n), names


def load_color_lut():
    """The LUT from config/color_samples.json (rebuilt when the samples changed), or (None, [])."""
    if not os.path.exists(COLOR_SAMPLES_PATH):
        return None, []
    try:
        mt = os.path.getmtime(COLOR_SAMPLES_PATH)
        if os.path.exists(COLOR_LUT_PATH):
            z = np.load(COLOR_LUT_PATH, allow_pickle=False)
            if float(z["mtime"]) == mt:
                return z["lut"], [str(x) for x in z["names"]]
        with open(COLOR_SAMPLES_PATH, "r", encoding="utf-8") as f:
            samples = {k: v for k, v in json.load(f).items() if v}
        lut, names = build_color_lut(samples)
        if lut is not None:
            np.savez_compressed(COLOR_LUT_PATH, lut=lut, names=np.array(names), mtime=mt)
        return lut, names
    except Exception as e:
        print(f"[vision] colour samples not usable ({e}) - HSV ranges only")
        return None, []

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

# Default only; the live value comes from vision.processing_width.
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
    guess: str = None        # not a sure card (cut by the edge / odd outline): likely shape

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


ROI_CLASSES = ["circle", "square", "rect_wide", "rect_tall", "none"]
SHAPE_MODEL_PATH = os.path.join(BASE_DIR, "config", "shape_knn.npz")


class RoiClassifier:
    """Stage 3 (optional): the box around a blob -> circle / square / rect_wide / rect_tall /
    none. Features = HOG of the blob's silhouette (32 x 32); model = k-nearest neighbours
    over labelled examples (OpenCV only; confidence = share of the k nearest that agree).
    Examples come from recorded runs: roi_dataset.py -> roi_label.py -> roi_train.py.
    Without config/shape_knn.npz it is simply not used."""
    SIZE = 32
    K = 5
    min_conf = 0.8

    MIN_NONE = 30                   # "not a card" is only trusted with this many examples of it
    MIN_SIDE_PX = 16                # used only on blobs at least this big (processing resolution)

    def __init__(self, feats, labels):
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(feats.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32).reshape(-1, 1))
        self.n = len(labels)
        self.counts = np.bincount(labels.astype(int), minlength=len(ROI_CLASSES))

    @staticmethod
    def load(path=SHAPE_MODEL_PATH):
        if not os.path.exists(path):
            return None
        try:
            z = np.load(path)
            ver = int(z["version"]) if "version" in z.files else 1
            if ver != RoiClassifier.FEAT_VERSION:
                print("[vision] box classifier was trained with older features - retrain it (Vision tab: Train)")
                return None
            return RoiClassifier(z["feats"], z["labels"])
        except Exception as e:
            print(f"[vision] box classifier not usable ({e})")
            return None

    @staticmethod
    def crop(shape_hw, cnt, pad=0.2):
        """The blob's silhouette in a square box around it, 32 x 32."""
        H, W = shape_hw[:2]
        x, y, bw, bh = cv2.boundingRect(cnt)
        side = int(max(bw, bh) * (1 + 2 * pad)) + 2
        cx, cy = x + bw // 2, y + bh // 2
        m = np.zeros((side, side), np.uint8)
        cv2.drawContours(m, [cnt - np.array([[cx - side // 2, cy - side // 2]])], -1, 255, -1)
        return cv2.resize(m, (RoiClassifier.SIZE, RoiClassifier.SIZE), interpolation=cv2.INTER_AREA)

    FEAT_VERSION = 2
    GEO_WEIGHT = 1.5

    @staticmethod
    def silhouette_stats(m):
        """Rotation-safe area features of a silhouette: fill of its rotated rectangle, short /
        long side, fill + aspect of its upright box, overlap with fitted ellipse / rectangle,
        in-picture turn, polygon corners. None when there is no blob."""
        b = (m > 127).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if n <= 1:
            return None
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        area = int(st[big, cv2.CC_STAT_AREA])
        x, y, w, h = st[big, :4]
        blob = (lab == big).astype(np.uint8)
        cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        (rw, rh), ang = rect[1], rect[2]
        rm = np.zeros_like(blob)
        cv2.fillPoly(rm, [cv2.boxPoints(rect).astype(np.int32)], 1)
        em = np.zeros_like(blob)
        if len(c) >= 5:
            cv2.ellipse(em, cv2.fitEllipse(c), 1, -1)
        iou = lambda a_, b_: float(np.logical_and(a_, b_).sum()) / max(1, np.logical_or(a_, b_).sum())
        peri = cv2.arcLength(c, True)
        turn = ang % 90.0
        turn = turn - 90.0 if turn > 45 else turn
        return {"area": area, "fill": area / float(w * h), "aspect": w / float(h),
                "rect_fill": area / max(1.0, rw * rh), "ratio": min(rw, rh) / max(1.0, max(rw, rh)),
                "iou_e": iou(blob, em), "iou_r": iou(blob, rm), "turn": abs(turn),
                "corners": len(cv2.approxPolyDP(c, 0.04 * peri, True)),
                "pieces": (int(st[1:, cv2.CC_STAT_AREA].sum()) - area) / max(1, area),
                "edge": bool(x <= 0 or y <= 0 or x + w >= m.shape[1] or y + h >= m.shape[0])}

    @staticmethod
    def features(crop):
        """HOG of the silhouette + its area features (so a few examples already separate a
        circle from a rectangle and a wide from a tall one)."""
        s = RoiClassifier.SIZE
        hog = cv2.HOGDescriptor((s, s), (16, 16), (8, 8), (8, 8), 9)
        h = hog.compute(crop).reshape(-1).astype(np.float32)
        st = RoiClassifier.silhouette_stats(crop)
        if st is None:
            g = np.zeros(7, np.float32)
        else:
            g = np.array([st["rect_fill"], st["ratio"], math.log(max(st["aspect"], 1e-3)) / 2.0 + 0.5,
                          st["fill"], st["iou_e"], st["iou_r"], min(st["corners"], 10) / 10.0], np.float32)
        return np.concatenate([h, RoiClassifier.GEO_WEIGHT * g]).reshape(1, -1)

    def predict(self, bgr, cnt):
        f = self.features(self.crop(bgr.shape, cnt))
        k = min(self.K, self.n)
        _, res, neigh, _ = self.knn.findNearest(f, k)
        lab = int(res[0, 0])
        conf = float(np.mean(neigh[0] == lab))
        name = ROI_CLASSES[lab] if 0 <= lab < len(ROI_CLASSES) else None
        # augmented x8 in training: MIN_NONE real examples = 8 x MIN_NONE rows
        if name == "none" and self.counts[ROI_CLASSES.index("none")] < 8 * self.MIN_NONE:
            return None, 0.0             # too few "not a card" examples to overrule the detector
        return name, conf


class TargetDetector:
    def __init__(self, config=None):
        config = config or {}
        vis = config.get("vision", {}) or {}
        move = config.get("movement", {}) or {}
        # `final2` is the detector profile from
        # https://github.com/remin-dot/Final_2_robomaster (vision.py).  That repository
        # contains no neural-network weight file; its model is the calibrated HSV mask plus
        # the contour classifier below.  Keep this explicit so later optional LUT/ROI models
        # cannot silently change the requested detector.
        self.detector_profile = str(vis.get("detector_profile", "enhanced")).strip().lower()
        self.final2_profile = self.detector_profile in ("final2", "final_2")

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
        self.guess_min_solidity = float(vis.get("guess_min_solidity", 0.75))  # partial / odd card blobs
        self.guess_min_area_factor = float(vis.get("guess_min_area_factor", 2.0))
        self.static_spots = []          # (colour, bearing, elevation) of blobs fixed in the picture
        self.close_mode = False         # set during a close look (camera down inside the block)
        self.close_wall_top_deg = None  # ... elevation of the top of the wall it looks at (from the ToF)
        # colour model: "hsv" = calibrated ranges only; "veto" = ranges, minus pixels the sample
        # LUT (color_samples.json) calls background (beige, wood, walls) - safe with few samples;
        # "both" = a pixel must also be that colour by the samples; "lab" = the LUT alone.
        # Background samples always veto the blur-tolerant search.
        self.color_model = "hsv" if self.final2_profile else str(vis.get("color_model", "veto")).lower()
        self.lut, self.lut_names = (None, [])
        if self.color_model != "hsv":
            self.lut, self.lut_names = load_color_lut()
            if self.lut is None and self.color_model == "lab":
                self.color_model = "hsv"
        self._cls = None                # class-id map of the current (processing) frame
        # stage 2 - per blob: re-find its edge inside a box around it (colour distance card vs
        # wall + Otsu; GrabCut when that is poor), then the shape from AREA features (fill of
        # the box, fitted ellipse vs rectangle overlap) - blurred edges do not break those
        self.refine_roi = False if self.final2_profile else bool(vis.get("refine_roi", True))
        self.refine_grabcut = str(vis.get("refine_grabcut", "auto")).lower()   # auto | on | off
        self.shape_method = "contour" if self.final2_profile else str(vis.get("shape_method", "area")).lower()
        self.roi_clf = None             # optional HOG+SVM box classifier (config/shape_svm.xml)
        if not self.final2_profile and vis.get("roi_classifier", True):
            self.roi_clf = RoiClassifier.load()
        self.debug = None               # vision_debug.py: per-blob stage images
        self.guess_min_h_frac = float(vis.get("guess_min_height_frac", 0.08))  # visible part >= 8% of the picture
        self.neutral_sat = int(vis.get("neutral_sat", 70))          # saturation below this = wall / floor grey
        self.min_neutral_card = float(vis.get("min_neutral_around_card", 0.45))
        self.min_neutral_guess = float(vis.get("min_neutral_around_guess", 0.65))
        self.white_balance = bool(vis.get("white_balance", True))  # use the white walls as colour reference
        # a card is a small patch: this much of the same colour right around it = a wall/floor, not a card
        self.max_ring_fill = float(vis.get("max_ring_fill", 0.25))

        self.hfov_deg = float(vis.get("hfov_deg", 96.0))
        self.processing_width = 640 if self.final2_profile else max(
            320, int(vis.get("processing_width", PROC_WIDTH)))
        self.processing_fps = max(0.0, float(vis.get("processing_fps", 0.0)))
        self.bottom_ignore = float(vis.get("bottom_ignore_ratio", 0.15))
        self.top_ignore = float(vis.get("top_ignore_ratio", 0.0))
        self.max_area_ratio = float(vis.get("max_area_ratio", 0.25))
        # Absolute processing-frame area. At the configured 512 px width, the smallest
        # expected 1.2 m card is still about 160 px, leaving 2x margin above the default.
        # Scaling this down admitted many tiny room contours and halved detector throughput.
        self.min_area_px = float(vis.get("min_area_px", 80))
        # --- ignore the room: everything above the arena's white walls / above the horizon
        self.ignore_above_wall = bool(vis.get("ignore_above_wall", True))
        self.wall_s_max = int(vis.get("wall_s_max", 60))          # white / grey foam: low saturation
        self.wall_v_min = int(vis.get("wall_v_min", 120))         # ... and bright
        self.wall_margin = float(vis.get("wall_margin_frac", 0.03))  # keep this much (of the height) above the line
        self.wall_max_gap = float(vis.get("wall_max_gap_frac", 0.25))  # non-white gap still inside the wall band
        self.wall_edge_max_dip = float(vis.get("wall_edge_max_dip_frac", 0.035))
        self.wall_edge_max_below_horizon = float(
            vis.get("wall_edge_max_below_horizon_frac", 0.08))
        # cards hang lower than the camera, so their centre is below the horizon; the room is above it
        self.max_elevation_deg = 2.0 if self.final2_profile else vis.get("max_elevation_deg", 1.5)
        self.gimbal_pitch_deg = 0.0     # live gimbal pitch (set by the chassis) - moves the horizon
        # a card hangs at card height: tape on the floor (and things up in the room) are not at it
        self.camera_height_m = float(vis.get("camera_height_m", 0.25))
        self.card_min_height_m = float(vis.get("card_min_height_m", 0.06))
        self.card_max_height_m = float(vis.get("card_max_height_m", 0.45))
        self.last_ignore_line = None    # per full-res column: y above which everything was ignored
        self.last_wall_top = None       # per full-res column: top edge of the white foam walls (NaN = none)
        self.last_frame_size = None     # (W, H) of the frame those lines belong to
        self.last_pitch_deg = 0.0       # gimbal pitch when that frame was processed
        self.tile_m = float(move.get("distance", 0.6))
        self.max_shoot_tiles = float(vis.get("max_shoot_tiles", 2))
        kernel_px = 3 if self.final2_profile else max(1, int(vis.get("morph_kernel_px", 5)))
        if kernel_px % 2 == 0:
            kernel_px += 1
        self.kernel = np.ones((kernel_px, kernel_px), np.uint8)

    # ------------------------------------------------------------------
    @property
    def max_shoot_m(self):
        return self.max_shoot_tiles * self.tile_m

    def focal_px(self, width):
        return (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def class_map(self, bgr_small):
        """Per-pixel class id from the sample LUT (None without samples)."""
        if self.lut is None:
            return None
        q = (bgr_small >> (8 - LUT_BITS)).astype(np.intp)
        return self.lut[q[..., 0], q[..., 1], q[..., 2]]

    def _lut_id(self, name):
        return self.lut_names.index(name) + 1 if name in self.lut_names else None

    def color_mask(self, hsv, name):
        m = self._hsv_mask(hsv, name)
        cls = self._cls
        if cls is None or cls.shape != hsv.shape[:2] or self.color_model == "hsv":
            return m
        if self.color_model == "veto":
            bg = self._lut_id("background")
            # Green cards in the latest run were visible to HSV while turning, then
            # disappeared only in the settled/LUT pass.  Room-background veto is useful
            # for beige/yellow and red wood, but unsafe for green when lighting changes.
            if bg is not None and name != "green":
                m = m.copy()
                m[cls == bg] = 0          # looks like the sampled background (beige, wood, walls)
            return m
        cid = self._lut_id(name)
        if cid is None:
            return m                      # that colour was never sampled: ranges only
        lm = np.where(cls == cid, 255, 0).astype(np.uint8)
        return lm if self.color_model == "lab" else cv2.bitwise_and(m, lm)

    def _hsv_mask(self, hsv, name):
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

        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        info = {
            "circularity": round(circularity, 2),
            "circle_fill": round(circle_fill, 2),
            "rect_fill": round(rect_fill, 2),
            "aspect": round(aspect, 2),
            "vertices": len(approx),
        }

        # A distant circle can become a 6-8 sided contour after resize/blur. Keep the
        # roundness thresholds tolerant, but require at least six vertices so a true
        # four-corner square cannot pass this branch.
        if (circularity >= 0.70 and circle_fill >= 0.58 and rect_fill < 0.90
                and 0.65 <= aspect <= 1.40 and len(approx) >= 6):
            return "circle", info
        if rect_fill >= 0.82 and 4 <= len(approx) <= 6:
            # a card is 9 x 6 cm (1.5 : 1); a long thin strip is tape on the floor or a wall edge.
            # Upright it may be narrow: a tall rect at a 60 deg slant is ~0.33 wide per 1 tall
            # (floor tape is also caught by the card-height and white-wall checks)
            if aspect > 2.4 or aspect < 0.3:
                info["rejected"] = "long thin strip (tape)"
                return "unknown", info
            if aspect > 1.28:
                return "rect_wide", info
            if aspect < 0.78:
                return "rect_tall", info
            # A boxy circle was the latest false positive. Only call it square when
            # the contour has strong four-corner evidence; otherwise request a closer look.
            if rect_fill >= 0.90 and len(approx) <= 5 and circle_fill < 0.72:
                return "square", info
            info["ambiguous"] = "round/square"
            return "unknown", info
        return "unknown", info

    def _refine_blob(self, bgr, cnt, ignore_mask=None):
        """Stage 2 for one blob (processing resolution): in a box around it, how much more
        each pixel looks like the card's own colour than like what is around it (Lab
        distances) -> Otsu threshold in that box only (every card gets its own cut, lit or in
        shade) -> the part touching the blob's core. When that disagrees with the first mask,
        GrabCut (seeded: core = card, far ring = wall) cuts the edge. Returns (contour,
        info) or None to keep the first one."""
        H, W = bgr.shape[:2]
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 4 or bh < 4:
            return None
        pad = max(4, int(0.35 * max(bw, bh)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + bw + pad), min(H, y + bh + pad)
        roi = bgr[y0:y1, x0:x1]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        # colour (a, b) decides, brightness only a little: a shadow on the white wall is darker
        # but still colourless - it must stay "wall" (it made the cards grow ~40 %)
        lab[..., 0] *= 0.3
        coarse = np.zeros(roi.shape[:2], np.uint8)
        cv2.drawContours(coarse, [cnt - np.array([[x0, y0]])], -1, 1, -1)
        k = max(1, min(bw, bh) // 6)
        core = cv2.erode(coarse, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        if core.sum() < 4:
            core = coarse
        ring = cv2.dilate(coarse, np.ones((2 * k + 3, 2 * k + 3), np.uint8)) == 0
        if ignore_mask is not None:
            ring &= ~ignore_mask[y0:y1, x0:x1]          # the room above the wall is not "around the card"
        if core.sum() < 4 or ring.sum() < 8:
            return None
        c_card = np.median(lab[core > 0], axis=0)
        c_bg = np.median(lab[ring], axis=0)
        if np.linalg.norm(c_card - c_bg) < 12:
            return None                                  # no contrast to cut on
        d1 = np.linalg.norm(lab - c_card, axis=2)
        d2 = np.linalg.norm(lab - c_bg, axis=2)
        score = (255.0 * d2 / (d1 + d2 + 1e-3)).astype(np.uint8)
        score = cv2.GaussianBlur(score, (3, 3), 0)
        _, m = cv2.threshold(score, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method = "otsu"

        def pick(mask):
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
            if n <= 1:
                return None
            ov = [int(((labels == i) & (core > 0)).sum()) for i in range(1, n)]
            i = int(np.argmax(ov)) + 1
            return (labels == i).astype(np.uint8) if ov[i - 1] > 0 else None

        def iou(a_, b_):
            u = np.logical_or(a_, b_).sum()
            return float(np.logical_and(a_, b_).sum()) / u if u else 0.0
        m = pick(m)
        q = iou(m, coarse) if m is not None else 0.0
        use_gc = self.refine_grabcut == "on" or (self.refine_grabcut == "auto" and q < 0.6)
        if use_gc and roi.shape[0] * roi.shape[1] <= 160 * 160:
            gc = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
            gc[coarse > 0] = cv2.GC_PR_FGD
            gc[core > 0] = cv2.GC_FGD
            gc[ring] = cv2.GC_BGD
            try:
                bg_m, fg_m = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                cv2.grabCut(roi, gc, None, bg_m, fg_m, 2, cv2.GC_INIT_WITH_MASK)
                m2 = pick(((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8))
                if m2 is not None:
                    m, q, method = m2, iou(m2, coarse), "grabcut"
            except cv2.error:
                pass
        if m is None:
            return None
        area0, area1 = float(coarse.sum()), float(m.sum())
        if not (0.6 * area0 <= area1 <= 1.25 * area0) or q < 0.6:
            return None                                   # it ran into something else: keep the first
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea) + np.array([[x0, y0]])
        if self.debug is not None:
            self.debug.append({"box": (x0, y0, x1, y1), "score": score, "mask": m, "coarse": coarse,
                               "method": method, "iou": q})
        return c.astype(np.int32), {"refine": method, "refine_iou": round(q, 2)}

    @staticmethod
    def classify_by_area(contour):
        """Shape from AREA features, not from corners or edge length (a blurred / low-res
        card has soft edges, its corner count is noise):
          * fill of its upright box: circle pi/4 = 0.79, rectangle ~1.0
          * overlap (IoU) with the fitted ellipse vs with the fitted rectangle
          * upright aspect: square 1, wide 1.5, tall 0.67 (cards hang upright)"""
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        if area <= 0 or bw < 3 or bh < 3:
            return "unknown", {}
        fill = area / float(bw * bh)
        aspect = bw / float(bh)
        blob = np.zeros((bh + 4, bw + 4), np.uint8)
        cv2.drawContours(blob, [contour - np.array([[x - 2, y - 2]])], -1, 1, -1)
        rect = cv2.minAreaRect(contour - np.array([[x - 2, y - 2]]))
        rm = np.zeros_like(blob)
        cv2.fillPoly(rm, [cv2.boxPoints(rect).astype(np.int32)], 1)
        em = np.zeros_like(blob)
        if len(contour) >= 5:
            cv2.ellipse(em, cv2.fitEllipse(contour - np.array([[x - 2, y - 2]])), 1, -1)
        iou = lambda a_, b_: float(np.logical_and(a_, b_).sum()) / max(1, np.logical_or(a_, b_).sum())
        i_e, i_r = iou(blob, em), iou(blob, rm)
        info = {"fill": round(fill, 2), "aspect": round(aspect, 2), "iou_ellipse": round(i_e, 2),
                "iou_rect": round(i_r, 2), "rect_fill": round(fill, 2)}
        if aspect > 2.4 or aspect < 0.3:
            info["rejected"] = "long thin strip (tape)"
            return "unknown", info
        if i_e - i_r > 0.03 and fill < 0.88 and 0.65 <= aspect <= 1.45:
            return "circle", info
        if i_r - i_e > 0.02 or fill >= 0.88:
            if aspect > 1.28:
                return "rect_wide", info
            if aspect < 0.78:
                return "rect_tall", info
            return "square", info
        info["ambiguous"] = "round/square"
        return "unknown", info

    @staticmethod
    def _balance_to_walls(img):
        """Grey-world on the walls: the bright, unsaturated pixels are the white foam walls;
        scale B, G, R so they come out neutral. Card colours then look the same under warm
        or cold light and when the camera's own white balance drifts."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ref = (hsv[..., 1] < 45) & (hsv[..., 2] > 140)
        n = int(np.count_nonzero(ref))
        if n < 0.05 * ref.size:
            return img                      # not enough wall in view: leave it
        means = img[ref].reshape(-1, 3).mean(axis=0)
        gains = np.clip(means.mean() / np.maximum(means, 1.0), 0.75, 1.33)
        if np.all(np.abs(gains - 1.0) < 0.03):
            return img
        return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    @staticmethod
    def guess_shape(info):
        """Best guess of the shape from what is visible (round-ish -> circle, else by aspect)."""
        if info.get("circularity", 0) >= 0.7 and info.get("circle_fill", 0) >= 0.65:
            return "circle"
        a = info.get("aspect", 1.0) or 1.0
        return "rect_wide" if a > 1.28 else ("rect_tall" if a < 0.78 else "square")

    # ------------------------------------------------------------------
    def detect(self, frame):
        """Return a list of Detection (full-frame coordinates), best first."""
        if frame is None:
            return []
        H, W = frame.shape[:2]
        proc_width = self.processing_width
        scale = proc_width / float(W) if W > proc_width else 1.0
        small = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        h, w = small.shape[:2]

        if self.white_balance:
            small = self._balance_to_walls(small)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # HSV mode deliberately ignores the learned background LUT.  Besides avoiding a
        # sampled green card being vetoed as room background, this saves a full-frame LUT
        # lookup on every detector frame.
        self._cls = None if self.color_model == "hsv" else self.class_map(blurred)
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
            # close look (camera down inside the block): a card 0.2 m away is big
            max_area_small = h * w * (max(self.max_area_ratio, 0.45) if self.close_mode else self.max_area_ratio)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cands = []
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < min_area_small or a > max_area_small:
                    continue
                rinfo = {}
                if self.refine_roi and name in self.card_colors:
                    r = self._refine_blob(blurred, cnt, ignore_mask)
                    if r is not None:
                        cnt, rinfo = r
                        a = cv2.contourArea(cnt)
                # a blob cut by the frame edge (a card half in view, very close) has an
                # unreliable shape -> never a target, but kept as a GUESS (colour + likely
                # shape) so the robot can re-aim at it or look again from the next block
                sx, sy, sw, sh = cv2.boundingRect(cnt)
                clipped = sx <= 2 or sy <= y_top + 2 or sx + sw >= w - 2 or sy + sh >= y_bot - 2 or \
                    (self.close_mode and ignore_y is not None and sy <= ignore_y[sx:sx + sw].max() + 2)
                # cut by the ignore line (something of the room reaching down): not the arena -
                # except in a close look, where it is most likely a card partly above the line:
                # kept as cut off (a maybe-card the camera is turned at)
                cut_by_line = ignore_y is not None and sy <= ignore_y[sx:sx + sw].max() + 2
                if cut_by_line and not self.close_mode:
                    continue
                if self.shape_method == "area":
                    shape, info = self.classify_by_area(cnt)
                else:
                    shape, info = self.classify_shape(cnt)
                info.update(rinfo)
                _, _, cbw, cbh = cv2.boundingRect(cnt)
                # the box classifier learned from close-up captures: a far card (a few pixels)
                # is left to the area rules
                if self.roi_clf is not None and name in self.card_colors and not clipped and \
                        min(cbw, cbh) >= self.roi_clf.MIN_SIDE_PX:
                    # the trained box classifier has the last word on the shape (and on "not a card")
                    pred, conf = self.roi_clf.predict(blurred, cnt)
                    info["clf"] = f"{pred} {conf:.2f}"
                    if pred and conf >= self.roi_clf.min_conf:
                        if pred == "none":
                            info["rejected"] = "box classifier: not a card"
                            shape = "unknown"
                        else:
                            shape = pred
                if clipped:
                    info["clipped"] = True
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
                # a card hangs on the white foam wall: most of what is around it is neutral
                # (low saturation). Tape on a coloured mat, robot parts, clothes: not.
                rh = hsv[y0:y1_, x0:x1_]
                around = np.ones(rh.shape[:2], bool)
                around[sy - y0:sy - y0 + sh, sx - x0:sx - x0 + sw] = False
                sat = rh[..., 1][around]
                neutral = float(np.count_nonzero(sat < self.neutral_sat)) / sat.size if sat.size else 1.0
                info["neutral_around"] = round(neutral, 2)
                full_cnt = (cnt.astype(np.float32) * inv).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(full_cnt)
                if shape in SHAPES:
                    # blur + downscale move the edge by a few px depending on the colour
                    # (yellow grows, blue shrinks): measure the size again at full resolution
                    x, y, bw, bh = self._refine_bbox(frame, (x, y, bw, bh))
                m = cv2.moments(full_cnt)
                if shape in SHAPES:
                    # the aim point: the middle of the card measured at full resolution (the
                    # outline above is from the 640 px picture - off by up to a pixel or two)
                    cx, cy = x + bw / 2.0, y + bh / 2.0
                elif m["m00"] > 0:
                    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
                else:
                    cx, cy = x + bw / 2.0, y + bh / 2.0

                bearing = math.degrees(math.atan2(cx - cx0, f_px))
                elevation = math.degrees(math.atan2(cy0 - cy, f_px))
                dist = None
                size = self.plate.get(shape)
                if size and bw > 0 and bh > 0:
                    # from the height only: a card seen at a slant looks narrower (width x cos),
                    # but its height does not change (cards hang upright on the walls)
                    dist = f_px * size[1] / float(bh)
                    info["aspect_full"] = round(bw / float(bh), 3)

                # height above the floor from the distance and the (world) elevation
                height = None
                if dist is not None:
                    height = self.camera_height_m + dist * math.tan(math.radians(elevation + self.gimbal_pitch_deg))
                    info["height_m"] = round(height, 3)
                at_card_height = height is None or self.card_min_height_m <= height <= self.card_max_height_m
                # up close the white wall around a card is partly outside the picture: fewer
                # neutral pixels are there to count
                min_neutral = min(self.min_neutral_card, 0.3) if self.close_mode else self.min_neutral_card
                is_card = (shape in SHAPES and name in self.card_colors and at_card_height
                           and solidity >= self.min_solidity and ring_fill <= self.max_ring_fill
                           and neutral >= min_neutral)
                if shape in SHAPES and not is_card:
                    info["rejected"] = ("not at card height (floor / room)" if not at_card_height else
                                        "not solid" if solidity < self.min_solidity else
                                        "not on a white wall" if neutral < self.min_neutral_card else
                                        "part of a bigger patch" if ring_fill > self.max_ring_fill else "colour off")
                if clipped:
                    is_card = False                  # half a card: its shape and size are not known
                guess = None
                # a guess must look like a card that is only partly seen: card-sized, solid,
                # on white wall - not any sliver of colour at the picture edge
                vis_h = sh * inv
                maybe = (not is_card and name in self.card_colors and ring_fill <= self.max_ring_fill
                         and solidity >= self.guess_min_solidity and a >= self.guess_min_area_factor * min_area_small
                         and neutral >= self.min_neutral_guess and vis_h >= self.guess_min_h_frac * H)
                if clipped:
                    ok_guess = maybe
                else:   # whole blob but an odd outline (not tape, not floor/room height)
                    ok_guess = maybe and shape == "unknown" and "rejected" not in info
                if ok_guess:
                    guess = shape if shape in SHAPES else self.guess_shape(info)
                    info["guess"] = guess
                    info["guess_why"] = "cut by the picture edge" if clipped else "shape unclear"
                    if clipped:
                        dist = None                  # the visible part is not the whole card
                is_target = is_card and kind_of(name, shape) in self.selected
                cands.append(Detection(
                    color=name, shape=shape, is_target=is_target, is_card=is_card,
                    contour=full_cnt, bbox=(x, y, bw, bh), center=(int(cx), int(cy)),
                    area=a * inv * inv, bearing_deg=bearing, elevation_deg=elevation,
                    distance_m=dist,
                    in_range=bool(is_target and dist is not None and dist <= self.max_shoot_m),
                    extra=info, guess=guess,
                ))
            results.extend(self._drop_reflections(cands))

        # blobs fixed in the picture (they did not move when the gimbal turned: a robot LED,
        # a reflection on the lens) are never cards
        for d in results:
            for color, b, e in self.static_spots:
                if d.color == color and abs(d.bearing_deg - b) < 2.0 and abs(d.elevation_deg - e) < 2.0:
                    d.is_card = d.is_target = d.in_range = False
                    d.guess = None
                    d.extra["rejected"] = "moves with the camera (not in the arena)"
        # targets first, then other cards, then larger blobs
        results.sort(key=lambda d: (not d.is_target, not d.is_card, -d.area))
        return results

    def detect_moving(self, frame, blur_px, blur_dy_px=0.0):
        """Colour sightings in a frame taken while the gimbal turns (blur_px = smear length
        along x in full-frame pixels, from gimbal speed x exposure). A smeared card is wider,
        paler (mixed with the white wall) and its outline is gone, so the normal detect()
        loses it once the smear is longer than the card (300 deg/s: ~70 px). Here: the colour
        with a lower saturation floor, the smear taken off the width, a white wall above and
        below it (not beside: that is the smear), at card height. No shape - only "a <colour>
        card is about there"; the robot then looks at it standing still.
        Returns [{color, bearing, elevation, dist_m}] (the frame must be the one just given
        to detect(): its ignore line is reused)."""
        if frame is None:
            return []
        H, W = frame.shape[:2]
        proc_width = self.processing_width
        scale = proc_width / float(W) if W > proc_width else 1.0
        small = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        h, w = small.shape[:2]
        if self.white_balance:
            small = self._balance_to_walls(small)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cls = None if self.color_model == "hsv" else self.class_map(small)
        bg_id = self._lut_id("background")
        bx = abs(blur_px) * scale
        by = abs(blur_dy_px) * scale
        f_full = self.focal_px(W)
        y_top = int(h * self.top_ignore)
        y_bot = int(h * (1.0 - self.bottom_ignore))
        ignore = None
        if self.last_ignore_line is not None and len(self.last_ignore_line) == W:
            ignore = np.interp(np.arange(w), np.arange(W) * scale, self.last_ignore_line) * scale
        out = []
        for name in self.card_colors:
            p = self.hsv.get(name)
            if not p:
                continue
            relaxed = dict(p)
            # the longer the smear, the paler the card (mixed with the white wall)
            k = max(0.3, min(0.5, 30.0 / max(abs(blur_px), 1.0)))
            if name == "yellow":
                k = max(k, 0.75)     # pale yellow = beige / wood / warm-lit walls: keep yellow strict
            relaxed["s_min"] = max(32, int(p.get("s_min", 0) * k))
            relaxed["v_max"] = 255
            saved, self.hsv[name] = self.hsv[name], relaxed
            try:
                mask = self._hsv_mask(hsv, name)
            finally:
                self.hsv[name] = saved
            if cls is not None and bg_id is not None and name != "green":
                mask[cls == bg_id] = 0            # sampled as background (beige / wood / wall)
            mask[:y_top, :] = 0
            mask[y_bot:, :] = 0
            if ignore is not None:
                mask[np.arange(h)[:, None] < ignore[None, :]] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                a = cv2.contourArea(cnt)
                sx, sy, sw, sh = cv2.boundingRect(cnt)
                if a < 0.5 * self.min_area_px or sh < 6 or a > h * w * self.max_area_ratio:
                    continue
                # a smear longer than the card: the card's width is spread over ~the smear
                # length (paler). So only an upper bound: no longer than smear + a wide card
                true_h = sh - by
                if true_h < 5 or sw > bx + 2.0 * true_h + 4 or (bx < 0.5 * true_h and sw < 0.25 * true_h):
                    continue
                if sx <= 1 or sx + sw >= w - 1:
                    continue                 # cut by the picture edge: the next frame has it
                # white wall above and below (beside it is the smear itself)
                my = max(3, int(sh * 0.5))
                bands = [hsv[max(0, sy - my):sy, sx:sx + sw, 1], hsv[sy + sh:min(h, sy + sh + my), sx:sx + sw, 1]]
                sat = np.concatenate([b.ravel() for b in bands if b.size])
                if not sat.size or np.count_nonzero(sat < self.neutral_sat) / float(sat.size) < 0.5:
                    continue
                cx = (sx + sw / 2.0) / scale
                cy = (sy + sh / 2.0) / scale
                bearing = math.degrees(math.atan2(cx - W / 2.0, f_full))
                elevation = math.degrees(math.atan2(H / 2.0 - cy, f_full))
                dist = f_full * 0.075 / (true_h / scale)          # a card is 6-9 cm tall
                height = self.camera_height_m + dist * math.tan(math.radians(elevation + self.gimbal_pitch_deg))
                if not (self.card_min_height_m - 0.05 <= height <= self.card_max_height_m + 0.05):
                    continue
                if any(c == name and abs(bearing - b) < 2.0 and abs(elevation - e) < 2.0
                       for c, b, e in self.static_spots):
                    continue
                out.append({"color": name, "bearing": bearing, "elevation": elevation, "dist_m": dist})
        return out

    def add_static_spot(self, color, bearing_deg, elevation_deg):
        """A blob that stays put in the picture while the gimbal turns: ignore that spot."""
        self.static_spots.append((color, float(bearing_deg), float(elevation_deg)))

    def ignore_line(self, hsv, f_proc):
        """Per processing column, the row above which the picture is not the arena:
        the top of the white foam walls (+ floor) band, or the horizon (cards hang below
        the camera), whichever is lower. None = nothing to ignore."""
        h, w = hsv.shape[:2]
        line = np.full(w, -1.0)
        # the horizon row (camera height): below it is floor / low wall - never the room
        horizon_row = h / 2.0 - f_proc * math.tan(math.radians(-self.gimbal_pitch_deg))
        top = self.wall_top(hsv, horizon_row)    # also used for the camera wall distance (wall_ahead_m)
        self._wall_top_proc = top
        if self.close_mode and self.close_wall_top_deg is not None:
            # close look at a wall the ToF measured: its top is where the geometry says, not
            # where the white stops (a card near the top of the picture broke the white band and
            # was cut away as "room above the wall")
            ang = math.radians(self.close_wall_top_deg - self.gimbal_pitch_deg)
            row = h / 2.0 - f_proc * math.tan(ang) if ang < math.radians(89) else -1e9
            if row <= 0:
                return None                       # the wall fills the picture up to its top edge
            return np.full(w, row - self.wall_margin * h)
        has_wall = np.zeros(w, bool)
        if self.ignore_above_wall and top is not None:
            has_wall = ~np.isnan(top)
            # only what is ABOVE the wall's top edge goes - never anything inside the wall area
            line = np.where(has_wall, top - self.wall_margin * h, line)
        # inside the block (close look) the room is behind the walls: no horizon rule - a card
        # hung above the camera on a wall 0.2 m away is 20+ deg up (last runs missed those)
        if self.max_elevation_deg not in (None, False, "off") and not self.close_mode:
            # the horizon rule only where no wall top was found in that column (an open side /
            # looking past the maze): near a wall its top is above the horizon, and the horizon
            # would cut away the upper part of the wall - cards hang there too
            # world elevation E appears at row cy - f * tan(E - gimbal_pitch)
            ang = math.radians(float(self.max_elevation_deg) - self.gimbal_pitch_deg)
            horizon_y = h / 2.0 - f_proc * math.tan(ang)
            line = np.where(has_wall, line, np.maximum(line, horizon_y))
        return line if (line > 0).any() else None

    def wall_top(self, hsv, horizon_row=None):
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
        hz = None if horizon_row is None else horizon_row * rows / float(h)
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
                g0, g1 = ends[j], t                    # the gap rows [g0, g1)
                if g1 - g0 > max_gap:
                    break
                # a gap above the horizon may be the room seen past the wall (a person in front
                # of a white table): bridged only when it is a hole IN the wall (a card). Below
                # the horizon (floor tape, seams, cards - they hang lower than the camera) always.
                if g1 - g0 > 2 and hz is not None and g0 < hz and not self._hole_in_wall(white, c, g0, g1):
                    break                              # not a card in the wall: the room seen past it
                t = starts[j]
            # The foam wall is taller than the camera, so its top cannot be far below the
            # world horizon. A low "edge" is the white floor interrupted by a chair/table
            # leg; accepting it makes the mask trace the obstacle and cut through targets.
            if hz is not None and t > hz + rows * self.wall_edge_max_below_horizon:
                continue
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
        # A coloured card interrupts the white pixels.  It must be treated as a hole in one
        # continuous foam edge, never as permission for the boundary to dive down across the
        # card and mask it.  Clamp isolated downward notches to the neighbouring edge while
        # preserving the broad perspective/slope of the wall.
        radius = max(5, w // 24)
        local = np.full(w, np.nan)
        for c in range(w):
            win = out[max(0, c - radius):min(w, c + radius + 1)]
            ok = ~np.isnan(win)
            if ok.any():
                local[c] = np.median(win[ok])
        ok = ~np.isnan(out) & ~np.isnan(local)
        out[ok] = np.minimum(out[ok], local[ok] + self.wall_edge_max_dip * h)
        return out

    @staticmethod
    def _hole_in_wall(white, c, g0, g1):
        """The non-white rows [g0, g1) of column c are a hole IN the wall (a card hanging on it)
        when the wall is white on BOTH sides of it at those rows (a card is narrow). A gap that
        is not white on one side is the room seen past a wall's end / over a lower wall (a
        person, a table leg) - the white above it is the room (a white table), not the wall."""
        rows, cols = white.shape
        reach = max(2, int(round((g1 - g0) * 1.2)))    # a card is at most ~1.5x as wide as tall
        sides = []
        for lo, hi in ((c - reach - 2, c - 1), (c + 1, c + reach + 2)):
            lo, hi = max(0, lo), min(cols, hi)
            if hi <= lo:
                return False                           # at the picture edge: cannot tell
            block = white[g0:g1, lo:hi]
            # somewhere beside the gap the wall is white over most of those rows
            sides.append(bool((block.mean(axis=0) >= 0.6).any()))
        return all(sides)

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
        elif d.guess:
            # maybe a card (cut by the edge / odd outline): dashed-looking outline + "?"
            cv2.drawContours(frame, [d.contour], -1, col, 1, cv2.LINE_4)
            txt = f"{d.color} {SHAPE_LABEL.get(d.guess, d.guess)} ?"
            cv2.putText(frame, txt, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            continue
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
    """Draw only the foam/horizon edge used by detection; never cover the camera image."""
    if line is None:
        return frame
    H, W = frame.shape[:2]
    if len(line) != W:
        line = np.interp(np.arange(W), np.linspace(0, W - 1, len(line)), line)
    ys = np.clip(np.nan_to_num(line, nan=-1).astype(int), -1, H - 1)
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
