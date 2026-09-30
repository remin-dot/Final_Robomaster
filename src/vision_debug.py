"""See every stage of the card detector on saved frames (recorded runs / captures):

  1 picture + the final answer      2 colour stage (ranges + sample LUT, background grey)
  3 ignore line (room cut) + floor  4..6 per blob: card-vs-wall score in its box, first mask
                                        (blue), refined edge (green), fitted ellipse (magenta) /
                                        rectangle (yellow), area features and the verdict

    .venv/bin/python src/vision_debug.py data/raw/run1/frames_*/*.jpg
    .venv/bin/python src/vision_debug.py capture_*.jpg --out debug.png      (no window: save one)

Keys: N / P next / previous | R edge refine on/off | M colour model (hsv / veto / both) |
      A shape by area / by contour | S save this view as PNG | Q quit
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config  # noqa: E402
from target_vision import COLORS, PROC_WIDTH, SHAPES, TargetDetector, kind_of  # noqa: E402

PW, PH = 480, 270
PAINT = {"red": (40, 40, 230), "blue": (230, 110, 30), "green": (40, 190, 40), "yellow": (0, 220, 240)}


def put(img, text, org, scale=0.45, color=(255, 255, 255), bg=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 3), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(det, frame, name=""):
    H, W = frame.shape[:2]
    det.debug = []
    dets = det.detect(frame)
    dbg = det.debug
    det.debug = None
    sx, sy = PW / W, PH / H
    # 1 picture + answer
    p1 = cv2.resize(frame, (PW, PH))
    for d in dets:
        x, y, bw, bh = d.bbox
        col = (0, 220, 0) if d.is_card else ((0, 160, 255) if d.guess else (150, 150, 150))
        cv2.rectangle(p1, (int(x * sx), int(y * sy)), (int((x + bw) * sx), int((y + bh) * sy)), col, 2)
        lab = d.kind if d.is_card else (f"? {d.color} {d.guess}" if d.guess else
                                        f"x {d.color}: {str(d.extra.get('rejected', d.shape))[:22]}")
        put(p1, lab, (int(x * sx), max(12, int(y * sy) - 4)), 0.4, (255, 255, 255), col if d.is_card else (60, 60, 60))
    put(p1, f"1 answer  {name}", (6, 16))
    # 2 colour stage
    small = cv2.resize(frame, (int(W * PROC_WIDTH / W), int(H * PROC_WIDTH / W))) if W > PROC_WIDTH else frame
    bal = det._balance_to_walls(small) if det.white_balance else small
    blurred = cv2.GaussianBlur(bal, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    det._cls = det.class_map(blurred)
    p2 = (cv2.resize(small, (PW, PH)) * 0.25).astype(np.uint8)
    if det._cls is not None and det._lut_id("background"):
        bgm = cv2.resize((det._cls == det._lut_id("background")).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST)
        p2[bgm > 0] = (90, 90, 90)
    for c in det.card_colors:
        if c not in det.hsv:
            continue
        m = cv2.resize(det.color_mask(hsv, c), (PW, PH), interpolation=cv2.INTER_NEAREST)
        p2[m > 0] = PAINT.get(c, (255, 255, 255))
    put(p2, f"2 colour ({det.color_model}; grey = sampled background)", (6, 16))
    # 3 ignore line
    p3 = cv2.resize(frame, (PW, PH))
    line = det.last_ignore_line
    if line is not None:
        xs = np.arange(PW)
        ys = np.interp(xs / sx, np.arange(len(line)), line) * sy
        for x, y in zip(xs, ys):
            if y > 0:
                p3[:int(y), x] = (p3[:int(y), x] * 0.3).astype(np.uint8)
        pts = np.stack([xs, np.clip(ys, 0, PH - 1)], 1).astype(np.int32)
        cv2.polylines(p3, [pts], False, (0, 0, 255), 2)
    yb = int(PH * (1 - det.bottom_ignore))
    p3[yb:] = (p3[yb:] * 0.4).astype(np.uint8)
    put(p3, "3 ignored: above the red line (room), bottom band (barrel)", (6, 16))
    # 4..6 per blob
    tiles = []
    scale = small.shape[1] / float(W)
    for e in dbg[:6]:
        x0, y0, x1, y1 = e["box"]
        sc = cv2.applyColorMap(e["score"], cv2.COLORMAP_JET)
        t = cv2.resize(sc, (150, 150), interpolation=cv2.INTER_NEAREST)
        k = 150.0 / max(1, max(x1 - x0, y1 - y0))
        for m, col in ((e["coarse"], (255, 80, 0)), (e["mask"], (0, 255, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                c = (max(cs, key=cv2.contourArea) * k).astype(np.int32)
                cv2.drawContours(t, [c], -1, col, 2)
                if col == (0, 255, 0) and len(c) >= 5:
                    cv2.ellipse(t, cv2.fitEllipse(c), (255, 0, 255), 1)
                    cv2.polylines(t, [cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)], True, (0, 255, 255), 1)
        # the detection this box belongs to
        cxp, cyp = (x0 + x1) / 2 / scale, (y0 + y1) / 2 / scale
        d = min(dets, key=lambda d_: abs(d_.center[0] - cxp) + abs(d_.center[1] - cyp)) if dets else None
        if d is not None:
            ex = d.extra
            put(t, f"{e['method']} iou {e['iou']:.2f}", (3, 12), 0.35)
            put(t, f"fill {ex.get('fill', '-')} asp {ex.get('aspect', '-')}", (3, 26), 0.35)
            put(t, f"ell {ex.get('iou_ellipse', '-')} rect {ex.get('iou_rect', '-')}", (3, 40), 0.35)
            v = d.kind if d.is_card else str(ex.get("rejected", d.shape))[:20]
            put(t, v, (3, 146), 0.4, (255, 255, 255), (0, 150, 0) if d.is_card else (60, 60, 60))
        tiles.append(t)
    p4 = np.full((PH, PW * 2, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        x, y = (i % 6) * 160 + 5, 60 + (i // 6) * 160
        p4[y:y + 150, x:x + 150] = t
    put(p4, "4-6 each blob: score in its box (red = card, blue = wall) | first mask (blue line) | "
            "refined edge (green) | fitted ellipse / rectangle", (6, 16))
    put(p4, f"refine {'on' if det.refine_roi else 'off'} ({det.refine_grabcut} GrabCut) | shape by {det.shape_method} | "
            f"box classifier {'on' if det.roi_clf is not None else 'off'}", (6, 38))
    top = np.hstack([p1, p2, p3])
    bottom = np.hstack([p4, np.full((PH, PW, 3), 30, np.uint8)])
    return np.vstack([top, bottom])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config()
    det = TargetDetector(cfg)
    det.set_selected({kind_of(c, s) for c in COLORS for s in SHAPES})
    paths = [p for p in a.frames if os.path.exists(p)]
    if a.out:
        cv2.imwrite(a.out, render(det, cv2.imread(paths[0]), os.path.basename(paths[0])))
        print("saved", a.out)
        return
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((PW * 3, PH * 2))
    pygame.display.set_caption("Card detector - every stage")
    i, dirty, view = 0, True, None
    models = ["hsv", "veto", "both"]
    while True:
        if dirty:
            img = cv2.imread(paths[i])
            view = render(det, img, f"{i + 1}/{len(paths)} {os.path.basename(paths[i])}")
            surf = pygame.image.frombuffer(cv2.cvtColor(view, cv2.COLOR_BGR2RGB).tobytes(), (PW * 3, PH * 2), "RGB")
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            dirty = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN:
                k = ev.unicode.lower()
                if k == "q":
                    return
                if k == "n" or ev.key == pygame.K_RIGHT:
                    i = (i + 1) % len(paths)
                elif k == "p" or ev.key == pygame.K_LEFT:
                    i = (i - 1) % len(paths)
                elif k == "r":
                    det.refine_roi = not det.refine_roi
                elif k == "a":
                    det.shape_method = "contour" if det.shape_method == "area" else "area"
                elif k == "m":
                    det.color_model = models[(models.index(det.color_model) + 1) % 3] if det.color_model in models else "hsv"
                elif k == "s":
                    out = os.path.splitext(os.path.basename(paths[i]))[0] + "_debug.png"
                    cv2.imwrite(out, view)
                    print("saved", out)
                dirty = True
        pygame.time.wait(20)


if __name__ == "__main__":
    main()
