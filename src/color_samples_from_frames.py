"""Colour samples for the LUT, taken automatically from saved frames (recorded runs or
captures): pixels inside cards the current detector is SURE of -> that colour; pixels away
from every card-coloured blob -> background. Use it to start the sample file, then add and
correct by hand with color_calibrate.py (background clicks on beige / wood / clothes).

    .venv/bin/python src/color_samples_from_frames.py data/raw/run1/frames/*.jpg
    .venv/bin/python src/color_samples_from_frames.py capture_*.jpg --out config/color_samples.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_vision import COLOR_SAMPLES_PATH, COLORS, SHAPES, TargetDetector, kind_of  # noqa: E402


def collect(paths, per_frame_bg=300, max_per_class=6000, seed=0):
    det = TargetDetector({"vision": {"color_model": "hsv"}})
    det.set_selected({kind_of(c, s) for c in COLORS for s in SHAPES})
    rng = np.random.default_rng(seed)
    out = {c: [] for c in list(COLORS) + ["background"]}
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        dets = det.detect(img)
        bal = det._balance_to_walls(img) if det.white_balance else img
        colour_any = np.zeros(img.shape[:2], np.uint8)
        for d in dets:
            cv2.drawContours(colour_any, [d.contour], -1, 255, -1)
            if not d.is_card or d.color not in out:
                continue
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(m, [d.contour], -1, 255, -1)
            m = cv2.erode(m, np.ones((7, 7), np.uint8))          # the card's inside, not its edge
            px = bal[m > 0]
            if len(px):
                out[d.color].extend(px[rng.choice(len(px), min(len(px), 400), replace=False)].tolist())
        far = cv2.dilate(colour_any, np.ones((41, 41), np.uint8)) == 0
        ys, xs = np.nonzero(far)
        if len(ys):
            k = rng.choice(len(ys), min(len(ys), per_frame_bg), replace=False)
            out["background"].extend(bal[ys[k], xs[k]].tolist())
    for c in out:
        if len(out[c]) > max_per_class:
            out[c] = [out[c][i] for i in rng.choice(len(out[c]), max_per_class, replace=False)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", default=COLOR_SAMPLES_PATH)
    ap.add_argument("--add", action="store_true", help="add to the existing file instead of replacing it")
    a = ap.parse_args()
    s = collect(a.frames)
    if a.add and os.path.exists(a.out):
        old = json.load(open(a.out))
        for c, v in old.items():
            s[c] = list(v) + s.get(c, [])
    with open(a.out, "w") as f:
        json.dump(s, f)
    print(", ".join(f"{c}: {len(v)} px" for c, v in s.items()), "->", a.out)
    missing = [c for c in COLORS if not s.get(c)]
    if missing:
        print("no sure card of:", ", ".join(missing), "- click those with color_calibrate.py")


if __name__ == "__main__":
    main()
