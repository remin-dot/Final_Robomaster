"""Train the stage-3 box classifier from data/roi_dataset (after roi_clean / roi_label).

  * features: HOG of the silhouette + its area features (RoiClassifier.features, version 2)
  * training examples are augmented: mirrored, turned +-4 deg, scaled +-8 %, edge grown /
    shrunk by a pixel (a real edge is never exact) - 8 versions of each
  * honest check: 5-fold cross-validation where near-copies (the same card in the same pose)
    stay on one side - a random split let copies of a test picture sit in the training set
    and showed 97 % for a model that was not that good
  * the final model is trained on everything and saved to config/shape_knn.npz

    .venv/bin/python src/roi_train.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_clean  # noqa: E402
from target_vision import ROI_CLASSES, SHAPE_MODEL_PATH, RoiClassifier  # noqa: E402

DATASET = roi_clean.DATASET


def augment(m):
    """8 versions of a silhouette (itself included)."""
    s = m.shape[0]
    out = [m, cv2.flip(m, 1)]
    for ang, sc in ((4, 1.0), (-4, 1.0), (0, 0.92), (0, 1.08)):
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
        out.append(cv2.warpAffine(m, M, (s, s), flags=cv2.INTER_LINEAR, borderValue=0))
    k = np.ones((3, 3), np.uint8)
    out.append(cv2.erode(m, k))
    out.append(cv2.dilate(m, k))
    return out


def feats_of(masks, hog_only=False):
    if hog_only:
        s = RoiClassifier.SIZE
        hog = cv2.HOGDescriptor((s, s), (16, 16), (8, 8), (8, 8), 9)
        return np.stack([hog.compute(m).reshape(-1) for m in masks]).astype(np.float32)
    return np.concatenate([RoiClassifier.features(m) for m in masks]).astype(np.float32)


def fit(items, aug=True, hog_only=False):
    masks, ys = [], []
    for it in items:
        vs = augment(it["mask"]) if aug else [it["mask"]]
        masks += vs
        ys += [it["y"]] * len(vs)
    X = feats_of(masks, hog_only)
    return X, np.array(ys, np.int32)


def cross_validate(items, gid, folds=5, aug=True, hog_only=False, seed=0):
    """Group k-fold: per-class right/total and the confusion matrix."""
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(set(gid)))
    rng.shuffle(ug)
    fold_of = {g: i % folds for i, g in enumerate(ug)}
    conf = np.zeros((len(ROI_CLASSES), len(ROI_CLASSES)), int)
    for f in range(folds):
        tr = [it for it, g in zip(items, gid) if fold_of[g] != f]
        te = [it for it, g in zip(items, gid) if fold_of[g] == f]
        if not tr or not te:
            continue
        X, y = fit(tr, aug, hog_only)
        clf = RoiClassifier(X, y)
        Xt = feats_of([it["mask"] for it in te], hog_only)
        _, res, _, _ = clf.knn.findNearest(Xt, clf.K)
        for it, r in zip(te, res[:, 0].astype(int)):
            conf[it["y"], r] += 1
    return conf


def report(conf):
    lines = []
    for i, c in enumerate(ROI_CLASSES):
        tot = conf[i].sum()
        if tot:
            wrong = {ROI_CLASSES[j]: int(conf[i, j]) for j in range(len(ROI_CLASSES)) if j != i and conf[i, j]}
            lines.append(f"{c} {conf[i, i]}/{tot}" + (f" (as {wrong})" if wrong else ""))
    acc = np.trace(conf) / max(1, conf.sum())
    return acc, lines


def train(root=DATASET, out=SHAPE_MODEL_PATH, log=print, save=True):
    """Cross-validate, then train on everything and save. Returns (accuracy, per-class, conf)."""
    items = roi_clean.load(root)
    items = [it for it in items if it["stats"] is not None]
    if len(items) < 20:
        raise SystemExit(f"only {len(items)} examples")
    gid = roi_clean.groups(items)
    conf = cross_validate(items, gid)
    acc, lines = report(conf)
    log(f"cross-validation (near-copies kept together): {acc * 100:.1f} %  -  " + ", ".join(lines))
    if save:
        X, y = fit(items)
        np.savez_compressed(out, feats=X, labels=y, version=RoiClassifier.FEAT_VERSION)
        log(f"saved {len(items)} examples x 8 versions -> {out}")
    return acc, conf


def load(root=DATASET):
    """(X, y) of the examples as they are (older callers)."""
    items = roi_clean.load(root)
    return fit(items, aug=False)


def main(root=DATASET, out=SHAPE_MODEL_PATH):
    return train(root, out)


if __name__ == "__main__":
    main()
