"""Stage-3 examples: cut the box around every card-coloured blob in recorded frames.
Each example = <label>/<name>.png (the colour crop, to look at) + <name>_m.png (silhouette,
what the classifier learns from). The first label is the detector's own answer (a sure card
-> its shape, a rejected blob -> none); fix the wrong ones with roi_label.py.

    .venv/bin/python src/roi_dataset.py data/raw/run1/frames_*/*.jpg
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_vision import COLORS, ROI_CLASSES, SHAPES, PROC_WIDTH, RoiClassifier, TargetDetector, kind_of  # noqa

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "roi_dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    det = TargetDetector({"vision": {"roi_classifier": False}})
    det.set_selected({kind_of(c, s) for c in COLORS for s in SHAPES})
    for c in ROI_CLASSES:
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
    n = {c: 0 for c in ROI_CLASSES}
    for f in a.frames:
        img = cv2.imread(f)
        if img is None:
            continue
        H, W = img.shape[:2]
        base = os.path.splitext(os.path.basename(f))[0]
        for i, d in enumerate(det.detect(img)):
            if d.color not in det.card_colors or d.extra.get("clipped"):
                continue
            lab = d.shape if d.is_card and d.shape in SHAPES else "none"
            cnt = d.contour
            x, y, bw, bh = cv2.boundingRect(cnt)
            side = int(max(bw, bh) * 1.4) + 2
            cx, cy = x + bw // 2, y + bh // 2
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            col = img[y0:min(H, y0 + side), x0:min(W, x0 + side)]
            if col.size == 0 or min(bw, bh) < 6:
                continue
            name = f"{base}_{i}_{d.color}"
            cv2.imwrite(os.path.join(a.out, lab, name + ".png"), cv2.resize(col, (64, 64)))
            cv2.imwrite(os.path.join(a.out, lab, name + "_m.png"), RoiClassifier.crop(img.shape, cnt))
            n[lab] += 1
    print("examples:", n, "->", a.out)
    print("next: .venv/bin/python src/roi_label.py   (fix wrong labels), then src/roi_train.py")


if __name__ == "__main__":
    main()
