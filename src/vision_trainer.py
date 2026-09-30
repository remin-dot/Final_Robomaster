"""Auto label + auto train for the stage-3 box classifier (and colour samples), run in the
background from the Mission Panel's Vision tab or after each round.

Auto label - every card-coloured blob in the recorded frames (data/raw/.../frames_*/) gets a
label only when the evidence is strong, else it goes to "unsure" for a quick look:
  1. the round's final map (frames_*/targets.json, saved at the end of the round): a blob in
     the direction of a mapped card of the same colour takes that card's shape - the map
     voted it over many views (face-on views count most), the best evidence there is
  2. otherwise the detector's own answer, only when it is clear-cut: a circle whose ellipse
     fit beats the rectangle fit clearly, a rectangle that fills its box and whose aspect is
     well away from the square / wide / tall cut-offs
  3. "none" (not a card): blobs rejected for a hard reason (tape strip, not at card height,
     part of a bigger patch, fixed in the picture)
Near-identical crops (the same card over consecutive frames) are kept at most 3 times.

Auto train - kNN on HOG of the silhouettes; the model is saved only when the hold-out check
passes (overall >= 90 %, every shape with enough test examples >= 75 %), otherwise the old
model stays and the tab says what is missing.
"""
import glob
import json
import math
import os
import shutil
import threading
import time

import cv2
import numpy as np

import target_vision as tv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "data", "roi_dataset")
LABELS = tv.ROI_CLASSES + ["unsure"]
MODEL_INFO = os.path.join(BASE, "config", "shape_knn.json")
DONE_LIST = os.path.join(DATASET, "processed_frames.txt")
HARD_NONE = ("long thin strip", "not at card height", "part of a bigger patch", "moves with the camera")


def counts(root=DATASET):
    return {c: len([p for p in glob.glob(os.path.join(root, c, "*.png")) if not p.endswith("_m.png")])
            for c in LABELS}


def model_info():
    try:
        return json.load(open(MODEL_INFO))
    except Exception:
        return None


def _read_frames_csv(d):
    meta = {}
    p = os.path.join(d, "frames.csv")
    if os.path.exists(p):
        for line in open(p):
            f = line.rstrip("\n").split(",")
            if len(f) >= 6:
                try:
                    cell = tuple(int(v) for v in f[4].split()) if f[4] else None
                    meta[f[0]] = {"pitch": float(f[2]) if f[2] else 0.0, "yaw": float(f[3]) if f[3] else 0.0,
                                  "cell": cell, "heading": int(f[5]) if f[5] not in ("", "None") else 0}
                except ValueError:
                    pass
    return meta


def _map_cards(d):
    """The round's final cards (saved into the frames folder at the end of the round)."""
    p = os.path.join(d, "targets.json")
    if not os.path.exists(p):
        return [], 0.6
    try:
        data = json.load(open(p))
        return [t for t in data.get("targets", []) if t.get("confirmed")], float(data.get("tile_m", 0.6))
    except Exception:
        return [], 0.6


def clear_cut(d):
    """The detector's own answer, only when its area features are far from every cut-off."""
    ex = d.extra
    if not d.is_card or d.shape not in tv.SHAPES:
        return None
    fill, asp = ex.get("fill"), ex.get("aspect")
    ie, ir = ex.get("iou_ellipse"), ex.get("iou_rect")
    if None in (fill, asp, ie, ir):
        return None
    if d.shape == "circle":
        return "circle" if ie - ir >= 0.06 and fill <= 0.84 else None
    if fill < 0.9 or ir - ie < 0.05:
        return None
    if d.shape == "square" and 0.86 <= asp <= 1.18:
        return "square"
    if d.shape == "rect_wide" and asp >= 1.36:
        return "rect_wide"
    if d.shape == "rect_tall" and asp <= 0.71:
        return "rect_tall"
    return None


def auto_label(frame_dirs, root=DATASET, log=print, stop=None):
    """Label the blobs of the frames in these folders (each frame once - processed_frames.txt)."""
    for c in LABELS:
        os.makedirs(os.path.join(root, c), exist_ok=True)
    done = set(open(DONE_LIST).read().split()) if os.path.exists(DONE_LIST) else set()
    det = tv.TargetDetector({"vision": {"roi_classifier": False}})   # never label with its own model
    det.set_selected({tv.kind_of(c, s) for c in tv.COLORS for s in tv.SHAPES})
    added = {c: 0 for c in LABELS}
    seen_keys = {}
    new_done = []
    for d in frame_dirs:
        meta = _read_frames_csv(d)
        cards, tile = _map_cards(d)
        for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            if stop is not None and stop.is_set():
                break
            key_f = os.path.relpath(f, BASE)
            if key_f in done:
                continue
            new_done.append(key_f)
            img = cv2.imread(f)
            if img is None:
                continue
            H, W = img.shape[:2]
            m = meta.get(os.path.basename(f), {})
            for i, dd in enumerate(det.detect(img)):
                if dd.color not in det.card_colors or dd.extra.get("clipped"):
                    continue
                x, y, bw, bh = cv2.boundingRect(dd.contour)
                if min(bw, bh) < 12:
                    continue                                # too small to learn a shape from
                label, why = None, ""
                # 1. the final map
                if cards and m.get("cell") is not None:
                    a = (m["heading"] * 90 + m["yaw"] + dd.bearing_deg) % 360
                    cx, cy = (m["cell"][0] + 0.5) * tile, (m["cell"][1] + 0.5) * tile
                    best = None
                    for t in cards:
                        if t["color"] != dd.color:
                            continue
                        bt = math.degrees(math.atan2(t["x_m"] - cx, t["y_m"] - cy)) % 360
                        off = abs((bt - a + 180) % 360 - 180)
                        dist = math.hypot(t["x_m"] - cx, t["y_m"] - cy)
                        if off < 7 and (best is None or off < best[0]):
                            best = (off, t, dist)
                    if best is not None:
                        label, why = best[1]["shape"], "map"
                        cc = clear_cut(dd)
                        if cc and cc != label:
                            label, why = "unsure", "map vs detector"
                # 2. clear-cut detector answer
                if label is None:
                    cc = clear_cut(dd)
                    if cc:
                        label, why = cc, "clear-cut"
                # 3. hard rejects
                if label is None and any(r in str(dd.extra.get("rejected", "")) for r in HARD_NONE):
                    label, why = "none", "rejected"
                if label is None:
                    label, why = "unsure", "unclear"
                # near-identical: the same card in consecutive frames - at most 3
                sil = tv.RoiClassifier.crop(img.shape, dd.contour)
                k = (d, dd.color, label, round(dd.bearing_deg + m.get("yaw", 0.0)) // 5, bh // 6)
                if seen_keys.get(k, 0) >= 3:
                    continue
                seen_keys[k] = seen_keys.get(k, 0) + 1
                side = int(max(bw, bh) * 1.4) + 2
                cxp, cyp = x + bw // 2, y + bh // 2
                x0, y0 = max(0, cxp - side // 2), max(0, cyp - side // 2)
                col = img[y0:min(H, y0 + side), x0:min(W, x0 + side)]
                if col.size == 0:
                    continue
                name = f"{os.path.basename(d)}_{os.path.splitext(os.path.basename(f))[0]}_{i}_{dd.color}_{why.replace(' ', '')}"
                cv2.imwrite(os.path.join(root, label, name + ".png"), cv2.resize(col, (64, 64)))
                cv2.imwrite(os.path.join(root, label, name + "_m.png"), sil)
                added[label] += 1
    with open(DONE_LIST, "a") as fh:
        for k in new_done:
            fh.write(k + "\n")
    log("auto label: " + ", ".join(f"{c} +{n}" for c, n in added.items() if n) + f" ({len(new_done)} new frames)")
    return added


def auto_train(root=DATASET, log=print, min_per_shape=15, clean_first=True):
    """Denoise (roi_clean: suspects -> _quarantine), then train with the honest check
    (cross-validation, near-copies kept together, roi_train). The model is saved only when that
    check passes: overall >= 90 %, every shape with >= 10 test examples >= 80 %.
    Returns (saved, info)."""
    import roi_clean
    import roi_train
    if clean_first:
        roi_clean.clean(root, apply=True, log=lambda m: log("clean: " + m))
    items = [it for it in roi_clean.load(root) if it["stats"] is not None]
    per = {c: sum(1 for it in items if it["label"] == c) for c in tv.ROI_CLASSES}
    missing = [c for c in tv.SHAPES if per.get(c, 0) < min_per_shape]
    if missing:
        msg = "need more examples of " + ", ".join(f"{c} ({per.get(c, 0)}/{min_per_shape})" for c in missing)
        log("auto train: " + msg)
        return False, {"examples": per, "note": msg}
    acc, conf = roi_train.train(root, save=False, log=lambda m: log(m))
    weak = [tv.ROI_CLASSES[i] for i in range(len(tv.ROI_CLASSES))
            if tv.ROI_CLASSES[i] != "none" and conf[i].sum() >= 10 and conf[i, i] / conf[i].sum() < 0.8]
    info = {"examples": per, "accuracy": round(float(acc), 3), "time": time.strftime("%Y-%m-%d %H:%M"),
            "holdout": {tv.ROI_CLASSES[i]: f"{conf[i, i]}/{conf[i].sum()}" for i in range(len(tv.ROI_CLASSES))
                        if conf[i].sum()}}
    shape_acc = sum(conf[i, i] for i in range(4)) / max(1, conf[:4].sum())
    if shape_acc < 0.9 or weak:
        info["note"] = f"not saved: cross-check {shape_acc * 100:.0f} %" + (f", weak: {', '.join(weak)}" if weak else "")
        log("auto train: " + info["note"])
        return False, info
    roi_train.train(root, save=True, log=lambda m: None)
    info["note"] = f"saved ({shape_acc * 100:.0f} % cross-check, {len(items)} clean examples)"
    json.dump(info, open(MODEL_INFO, "w"), indent=2)
    log("auto train: " + info["note"])
    return True, info


def frame_dirs(data_dir):
    return sorted(d for d in glob.glob(os.path.join(data_dir, "frames_*")) if os.path.isdir(d))


class VisionTrainer:
    """Background jobs for the Vision tab: auto label, auto train, colour samples, review."""

    def __init__(self, panel):
        self.panel = panel
        self.busy = None                 # name of the running job
        self.status = ""
        self.stop = threading.Event()
        self.info = model_info()
        self.review = None               # {"items": [png paths], "sel": set(), "page": 0} while reviewing
        self.capture_label = "auto"      # label given to Capture examples
        self.last_capture = None         # 64 x 64 crop of the last capture (shown in the tab)

    def _log(self, msg):
        self.status = msg
        try:
            self.panel.log("vision: " + msg)
        except Exception:
            pass

    def _run(self, name, fn):
        if self.busy:
            return
        self.busy = name
        self.stop.clear()

        def go():
            try:
                fn()
            except Exception as e:
                self._log(f"{name} failed: {e!r}")
            finally:
                self.busy = None
        threading.Thread(target=go, daemon=True).start()

    def data_dir(self):
        return getattr(self.panel, "data_dir", os.path.join(BASE, "data", "raw", "run1"))

    def label_now(self, dirs=None):
        self._run("labelling", lambda: auto_label(dirs or frame_dirs(self.data_dir()), log=self._log, stop=self.stop))

    def train_now(self):
        def job():
            ok, info = auto_train(log=self._log)
            self.info = model_info() if ok else dict(info, saved=False)
            if ok:
                self.panel.detector.roi_clf = tv.RoiClassifier.load()      # used from the next frame
        self._run("training", job)

    def auto_now(self, dirs=None):
        def job():
            auto_label(dirs or frame_dirs(self.data_dir()), log=self._log, stop=self.stop)
            ok, info = auto_train(log=self._log)
            self.info = model_info() if ok else dict(info, saved=False)
            if ok:
                self.panel.detector.roi_clf = tv.RoiClassifier.load()
        self._run("label + train", job)

    def clean_now(self):
        def job():
            import roi_clean
            roi_clean.clean(DATASET, apply=True, log=self._log)
        self._run("cleaning", job)

    def colour_samples_now(self):
        def job():
            import color_samples_from_frames as csf
            fs = [f for d in frame_dirs(self.data_dir()) for f in glob.glob(os.path.join(d, "*.jpg"))]
            if not fs:
                self._log("no recorded frames yet")
                return
            s = csf.collect(fs)
            old = {}
            if os.path.exists(tv.COLOR_SAMPLES_PATH):
                old = json.load(open(tv.COLOR_SAMPLES_PATH))
            rng = np.random.default_rng()
            for c, v in s.items():
                allp = list(old.get(c, [])) + v
                if len(allp) > 6000:
                    allp = [allp[i] for i in rng.choice(len(allp), 6000, replace=False)]
                old[c] = allp
            json.dump(old, open(tv.COLOR_SAMPLES_PATH, "w"))
            det = self.panel.detector
            det.lut, det.lut_names = tv.load_color_lut()
            self._log("colour samples updated from " + f"{len(fs)} frames: " +
                      ", ".join(f"{c} {len(v)}" for c, v in old.items()))
        self._run("colour samples", job)

    # ---- capture from the live camera: one example of the card in the middle of the view
    def capture(self, frame, label="auto"):
        """Crop the card-coloured blob nearest the picture centre (the card you hold / placed in
        view) and save it as an example: the colour crop (to review) + its silhouette (what the
        classifier learns). label "auto" = the detector's clear-cut answer, else "unsure".
        Returns a short message for the tab."""
        if frame is None:
            return self._say("no camera picture")
        if getattr(self, "_cap_det", None) is None:
            self._cap_det = tv.TargetDetector({"vision": dict((self.panel.config.get("vision", {}) or {}),
                                                              roi_classifier=False)})
            self._cap_det.set_selected({tv.kind_of(c, s) for c in tv.COLORS for s in tv.SHAPES})
        det = self._cap_det
        H, W = frame.shape[:2]
        blobs = [d for d in det.detect(frame) if d.color in det.card_colors and not d.extra.get("clipped")
                 and min(d.bbox[2], d.bbox[3]) >= 10]
        if not blobs:
            return self._say("no card colour in view (or it touches the picture edge) - move it to the middle")
        d = min(blobs, key=lambda b: math.hypot(b.center[0] - W / 2, b.center[1] - H / 2))
        if label == "auto":
            lab = clear_cut(d) or "unsure"
        else:
            lab = label
        x, y, bw, bh = cv2.boundingRect(d.contour)
        side = int(max(bw, bh) * 1.4) + 2
        cx, cy = x + bw // 2, y + bh // 2
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        col = frame[y0:min(H, y0 + side), x0:min(W, x0 + side)]
        os.makedirs(os.path.join(DATASET, lab), exist_ok=True)
        name = f"cap_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{d.color}"
        cv2.imwrite(os.path.join(DATASET, lab, name + ".png"), cv2.resize(col, (64, 64)))
        cv2.imwrite(os.path.join(DATASET, lab, name + "_m.png"), tv.RoiClassifier.crop(frame.shape, d.contour))
        self.last_capture = cv2.resize(col, (64, 64))
        n = counts()[lab]
        seen = d.kind if d.is_card else f"{d.color} blob"
        return self._say(f"captured {seen} -> {lab} ({n} {lab} examples)"
                         + ("" if label == "auto" or not d.is_card or d.shape == label
                            else f"  (detector saw {d.shape})"))

    def _say(self, msg):
        self.status = msg
        return msg

    def delete_model(self):
        for p in (tv.SHAPE_MODEL_PATH, MODEL_INFO):
            if os.path.exists(p):
                os.remove(p)
        self.panel.detector.roi_clf = None
        self.info = None
        self._log("box classifier removed - shapes by area only")

    # ---- review of "unsure" crops (shown over the camera view)
    def start_review(self, label="unsure"):
        items = sorted(p for p in glob.glob(os.path.join(DATASET, label, "*.png")) if not p.endswith("_m.png"))
        self.review = {"label": label, "items": items, "sel": set(), "page": 0}

    def review_assign(self, new_label):
        r = self.review
        if not r:
            return
        for p in list(r["sel"]):
            for q in (p, p[:-4] + "_m.png"):
                if os.path.exists(q):
                    if new_label == "delete":
                        os.remove(q)
                    else:
                        os.makedirs(os.path.join(DATASET, new_label), exist_ok=True)
                        shutil.move(q, os.path.join(DATASET, new_label, os.path.basename(q)))
            r["items"].remove(p)
        r["sel"] = set()

    def after_round(self, frames_dir, targets_json=None, auto=True):
        """End of a round: keep the final map with its frames (labels by map), then label + train."""
        if not frames_dir or not os.path.isdir(frames_dir):
            return
        if targets_json and os.path.exists(targets_json):
            shutil.copy(targets_json, os.path.join(frames_dir, "targets.json"))
        if auto:
            self.auto_now([frames_dir])
