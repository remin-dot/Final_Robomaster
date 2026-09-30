"""Denoise the stage-3 examples (data/roi_dataset) before training.

An example is moved to data/roi_dataset/_quarantine/<label>/ (never deleted - put it back by
hand or with --restore) when:
  * broken silhouette: almost empty, cut by the crop edge, or in several pieces
  * its silhouette cannot be that shape from ANY viewing angle (cards hang upright, a slant
    only makes them narrower): a "tall" wider than high, a "square" clearly wide, a "circle"
    that fills its box like a rectangle (or far too little), a "wide" seen so slanted it looks
    square / tall (> ~50 deg: the silhouette no longer shows it is wide)
  * its nearest neighbours - not counting its own near-copies - almost all carry another label
    (a click on the wrong chip, a capture of the wrong card)
Near-copies (consecutive captures of the same card, same pose) are grouped: at most
MAX_PER_GROUP of each group are kept for training, and evaluation keeps a group on one side.

    .venv/bin/python src/roi_clean.py            # report only
    .venv/bin/python src/roi_clean.py --apply    # move the suspects to _quarantine
    .venv/bin/python src/roi_clean.py --restore  # put everything back
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_vision import ROI_CLASSES, RoiClassifier  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "data", "roi_dataset")
QUAR = "_quarantine"
MAX_PER_GROUP = 4


def shape_stats(m):
    return RoiClassifier.silhouette_stats(m)


def impossible(label, s):
    """Why this example teaches the classifier something wrong (or None). Cards on the arena
    walls hang upright; the robot sees them straight or turned away (narrower), never turned
    in the picture. Examples outside that - or so slanted the shape no longer shows - go."""
    if s is None or s["area"] < 30:
        return "empty silhouette"
    if s["edge"]:
        return "cut by the crop edge"
    if s["pieces"] > 0.15:
        return "several pieces"
    a, ie, ir = s["aspect"], s["iou_e"], s["iou_r"]
    if label == "circle":
        if ie < 0.85:
            return f"not an ellipse - part of a circle? (fit {ie:.2f})"
        if a > 1.25:
            return f"wider than tall - a slant only makes a circle narrower (aspect {a:.2f})"
        if a >= 0.7 and s["rect_fill"] > 0.93 and s["corners"] <= 5:
            return "has corners like a rectangle"          # (a thin slanted ellipse is left alone)
        return None
    if label in ("square", "rect_wide", "rect_tall"):
        if ie >= 0.93 and s["rect_fill"] < 0.84:
            return f"round, not a rectangle (ellipse fit {ie:.2f})"
        if s["turn"] > 25:
            return f"turned {s['turn']:.0f} deg in the picture (cards hang upright)"
        if label == "square" and a < 0.62:
            return f"square seen so slanted it looks tall (aspect {a:.2f})"
        if label == "square" and a > 1.35:
            return f"square that looks wide (aspect {a:.2f})"
        if label == "rect_tall" and a > 0.9:
            return f"tall that looks square (aspect {a:.2f})"
        if label == "rect_wide" and a < 1.0:
            return f"wide seen so slanted it looks square / tall (aspect {a:.2f})"
    return None


def load(root=DATASET):
    items = []
    for i, c in enumerate(ROI_CLASSES):
        for p in sorted(glob.glob(os.path.join(root, c, "*_m.png"))):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (RoiClassifier.SIZE, RoiClassifier.SIZE))
            items.append({"path": p, "label": c, "y": i, "mask": m, "stats": shape_stats(m)})
    return items


def groups(items, thr=6.0):
    """Near-copies: same label, silhouettes within thr mean grey levels -> one group id."""
    gid = [-1] * len(items)
    g = 0
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["label"], []).append(i)
    for idx in by.values():
        reps = []                                   # (group id, mask)
        for i in idx:
            m = items[i]["mask"].astype(np.float32)
            for gg, rm in reps:
                if np.abs(m - rm).mean() < thr:
                    gid[i] = gg
                    break
            else:
                gid[i] = g
                reps.append((g, m))
                g += 1
    return gid


def neighbour_vote(items, gid, feats, k=7):
    """Share of the k nearest examples (other groups only) that carry another label."""
    X = np.stack(feats).astype(np.float32)
    y = np.array([it["y"] for it in items])
    G = np.array(gid)
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    out = []
    for i in range(len(items)):
        di = d[i].copy()
        di[G == G[i]] = np.inf
        nn = np.argsort(di)[:k]
        nn = nn[np.isfinite(di[nn])]
        out.append(float(np.mean(y[nn] != y[i])) if len(nn) else 0.0)
    return out


def analyse(root=DATASET):
    items = load(root)
    if not items:
        return [], []
    gid = groups(items)
    feats = [RoiClassifier.features(it["mask"])[0] for it in items]
    wrong = neighbour_vote(items, gid, feats)
    suspects = []
    for it, w_ in zip(items, wrong):
        why = impossible(it["label"], it["stats"])
        if why is None and w_ >= 0.85 and it["label"] != "none":    # "none" is too small a class to vote
            why = f"its neighbours are another shape ({w_ * 100:.0f} %)"
        if why:
            suspects.append((it, why))
    # too many near-copies of one pose: keep MAX_PER_GROUP
    extra = []
    seen = {}
    bad = {id(it) for it, _ in suspects}
    for it, g in zip(items, gid):
        if id(it) in bad:
            continue
        seen[g] = seen.get(g, 0) + 1
        if seen[g] > MAX_PER_GROUP:
            extra.append((it, "near-copy of another example"))
    return items, suspects + extra


def move(path, root, to_q=True, why=""):
    lab = os.path.basename(os.path.dirname(path))
    dst = os.path.join(root, QUAR, lab) if to_q else None
    os.makedirs(dst, exist_ok=True)
    for p in (path, path.replace("_m.png", ".png")):
        if os.path.exists(p):
            shutil.move(p, os.path.join(dst, os.path.basename(p)))
    with open(os.path.join(root, QUAR, "reasons.csv"), "a", newline="") as f:
        csv.writer(f).writerow([lab, os.path.basename(path), why])


def restore(root=DATASET):
    n = 0
    for lab in ROI_CLASSES + ["unsure"]:
        for p in glob.glob(os.path.join(root, QUAR, lab, "*.png")):
            os.makedirs(os.path.join(root, lab), exist_ok=True)
            shutil.move(p, os.path.join(root, lab, os.path.basename(p)))
            n += 1
    return n


def clean(root=DATASET, apply=False, log=print):
    items, sus = analyse(root)
    by = {}
    for it, why in sus:
        key = why.split(" (")[0]
        by[key] = by.get(key, 0) + 1
    log(f"{len(items)} examples, {len(sus)} suspect: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    if apply:
        for it, why in sus:
            move(it["path"], root, why=why)
        log(f"moved {len(sus)} to {QUAR}/ (restore: roi_clean.py --restore)")
    return items, sus


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--root", default=DATASET)
    a = ap.parse_args()
    if a.restore:
        print("restored", restore(a.root))
    else:
        items, sus = clean(a.root, a.apply)
        per = {}
        for it, why in sus:
            per.setdefault(it["label"], []).append(why)
        for lab, whys in per.items():
            print(f"  {lab}: {len(whys)}")
