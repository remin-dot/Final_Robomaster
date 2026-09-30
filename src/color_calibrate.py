"""Card colour calibration: click on the real cards, under the real arena light.

    .venv/bin/python src/color_calibrate.py                 # robot camera (close the mission panel first)
    .venv/bin/python src/color_calibrate.py --webcam 0      # a webcam
    .venv/bin/python src/color_calibrate.py --image a.jpg b.jpg   # saved pictures (captures / screenshots)

1-4 pick the colour (blue, red, yellow, green), then click on a card of that colour
- several cards, near and far, lit and in shade. 5 = BACKGROUND: click on what is NOT a card
but looks a bit like one (white walls, wood, beige floor / tables, skin, clothes, the robot).
The clicked pixels are also saved (config/color_samples.json): the detector builds a colour
lookup table from them - a pixel must be a card colour by the samples, and background-looking
pixels are thrown out (last runs: beige / wood taken for yellow). Each click samples a small patch
(after the same white balance the detector uses). The range is worked out from all
samples (robust percentiles + a margin); the mask view shows what it would pick up.

Keys: 1-4 colour | 5 background | click sample | U undo | C clear colour | M mask view |
      SPACE freeze | N next picture | S save (color_config.json + color_samples.json) | Q quit
      --reset-samples starts color_samples.json from scratch
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_vision import COLOR_CONFIG_PATH, COLOR_SAMPLES_PATH, DEFAULT_HSV, TargetDetector  # noqa: E402

COLOURS = ["blue", "red", "yellow", "green"]
CLASSES = COLOURS + ["background"]
SWATCH = {"blue": (40, 110, 220), "red": (230, 40, 40), "yellow": (240, 220, 0), "green": (40, 190, 60),
          "background": (190, 180, 160)}
MAX_SAVED = 6000        # pixels kept per class in color_samples.json
PATCH = 4               # sample (2*PATCH+1)^2 pixels around a click
NOMINAL_HUE = {"blue": 115, "red": 0, "yellow": 26, "green": 65}   # OpenCV hue 0..179


def hue_gap(a, b):
    """Distance between two OpenCV hues on the 0..179 circle."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def hsv_range(samples, colour):
    """Samples [(h, s, v)] -> {h_min, h_max, s_min, s_max, v_min, v_max}. Red wraps past 180."""
    a = np.array(samples, dtype=np.float32)
    h, s, v = a[:, 0], a[:, 1], a[:, 2]
    wraps = colour == "red" or (np.mean(h < 20) > 0.2 and np.mean(h > 160) > 0.2)
    if wraps:
        h = np.where(h < 90, h + 180, h)
    return widen(np.percentile(h, 2), np.percentile(h, 98), np.median(h),
                 np.percentile(s, 5), np.percentile(v, 5), wraps)


def widen(h_lo, h_hi, h_med, s_low, v_low, wraps):
    """The range the detector uses, with room for what the clicks did not cover: a card
    further away, in shade or at a slant is less saturated and darker than the one clicked
    (the first version kept only 20 below the clicks - green needed saturation >= 235)."""
    lo, hi = min(h_lo - 5, h_med - 8), max(h_hi + 5, h_med + 8)     # at least +-8 around the colour
    if wraps:
        h_min, h_max = int(lo) % 180, int(round(hi)) % 180
    else:
        h_min, h_max = max(0, int(lo)), min(179, int(round(hi)))
    return {"h_min": h_min, "h_max": h_max,
            "s_min": int(min(120, max(50, s_low - 60))), "s_max": 255,
            "v_min": int(min(100, max(30, v_low - 70))), "v_max": 255}


class Source:
    def __init__(self, args):
        self.images, self.idx, self.ep, self.cap = [], 0, None, None
        if args.image:
            self.images = [cv2.imread(p) for p in args.image]
            self.images = [im for im in self.images if im is not None]
            if not self.images:
                raise SystemExit("none of the pictures could be read")
        elif args.webcam is not None:
            self.cap = cv2.VideoCapture(args.webcam)
        else:
            from mission_panel import require_sdk
            robot = require_sdk()
            self.ep = robot.Robot()
            try:   # same connection as the mission panel (AP mode needs UDP)
                if args.connection == "ap":
                    self.ep.initialize(conn_type="ap", proto_type="udp")
                else:
                    self.ep.initialize(conn_type=args.connection)
            except Exception as e:
                hint = {"ap": "Join this computer to the robot's Wi-Fi (RMEP-xxxxxx, robot switch on AP) - "
                              "it must get a 192.168.2.x address.",
                        "sta": "Are the robot and this computer on the same router network?",
                        "rndis": "Is the USB cable in the robot's intelligent controller micro-USB port?"}
                raise SystemExit(f"\n[!] Could not reach the robot ({e}).\n    {hint.get(args.connection, '')}\n"
                                 "    Or calibrate from saved pictures:  "
                                 ".venv/bin/python src/color_calibrate.py --image capture_*.jpg\n")
            self.ep.camera.start_video_stream(display=False, resolution=args.resolution)
            try:
                self.ep.gimbal.recenter().wait_for_completed(timeout=3)
            except Exception:
                pass

    def read(self):
        if self.images:
            return self.images[self.idx].copy()
        if self.cap is not None:
            ok, f = self.cap.read()
            return f if ok else None
        try:
            return self.ep.camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Exception:
            return None

    def next(self):
        if self.images:
            self.idx = (self.idx + 1) % len(self.images)

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.ep is not None:
            try:
                self.ep.camera.stop_video_stream()
                self.ep.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+")
    ap.add_argument("--webcam", type=int)
    ap.add_argument("--connection", default="ap")
    ap.add_argument("--resolution", default="540p")
    ap.add_argument("--reset-samples", action="store_true")
    args = ap.parse_args()

    det = TargetDetector({})
    ranges = {c: dict(det.hsv.get(c, DEFAULT_HSV[c])) for c in COLOURS}
    samples = {c: [] for c in COLOURS}
    bgr_samples = {c: [] for c in CLASSES}           # white-balanced BGR pixels, for the LUT
    history = []
    src = Source(args)

    pygame.init()
    W, H = 960, 540
    screen = pygame.display.set_mode((W + 300, H + 40))
    pygame.display.set_caption("Card colour calibration")
    font = pygame.font.SysFont("Helvetica", 16)
    small = pygame.font.SysFont("Helvetica", 13)
    colour, mask_view, frozen, frame = "blue", False, False, None
    msg, msg_t = "1-4 colour, click on cards, S save", time.time()

    def to_hsv(img):
        img = det._balance_to_walls(img) if det.white_balance else img
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    running = True
    clock = pygame.time.Clock()
    while running:
        if not frozen or frame is None:
            f = src.read()
            if f is not None:
                frame = cv2.resize(f, (W, H))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.unicode.lower()
                if k in "12345" and k:
                    colour = CLASSES[int(k) - 1]
                elif k == "m":
                    mask_view = not mask_view
                elif k == " ":
                    frozen = not frozen
                elif k == "n":
                    src.next()
                    frame = None
                elif k == "u" and history:
                    c, n = history.pop()
                    del bgr_samples[c][-n:]
                    if c in samples:
                        del samples[c][-n:]
                        if samples[c]:
                            ranges[c].update(hsv_range(samples[c], c))
                    msg, msg_t = f"undo ({c})", time.time()
                elif k == "c":
                    bgr_samples[colour].clear()
                    if colour in samples:
                        samples[colour].clear()
                        ranges[colour] = dict(det.hsv.get(colour, DEFAULT_HSV[colour]))
                    msg, msg_t = f"{colour} cleared", time.time()
                elif k == "s":
                    data = {}
                    if os.path.exists(COLOR_CONFIG_PATH):
                        try:
                            data = json.load(open(COLOR_CONFIG_PATH))
                        except Exception:
                            data = {}
                    for c in COLOURS:
                        if samples[c]:
                            data.setdefault(c, {}).update(ranges[c])
                    with open(COLOR_CONFIG_PATH, "w") as fh:
                        json.dump(data, fh, indent=4)
                    # the clicked pixels themselves (added to what was saved before)
                    old = {}
                    if os.path.exists(COLOR_SAMPLES_PATH) and not args.reset_samples:
                        try:
                            old = json.load(open(COLOR_SAMPLES_PATH))
                        except Exception:
                            old = {}
                    rng = np.random.default_rng()
                    for c in CLASSES:
                        allp = list(old.get(c, [])) + bgr_samples[c]
                        if len(allp) > MAX_SAVED:
                            allp = [allp[i] for i in rng.choice(len(allp), MAX_SAVED, replace=False)]
                        old[c] = allp
                    with open(COLOR_SAMPLES_PATH, "w") as fh:
                        json.dump(old, fh)
                    args.reset_samples = False
                    saved = [c for c in CLASSES if bgr_samples[c]]
                    few = [c for c in saved if sum(1 for cc, _ in history if cc == c) < 3]
                    msg, msg_t = (f"saved {', '.join(saved) or 'nothing (no samples)'}"
                                  + (f" - only 1-2 clicks for {', '.join(few)}: click more cards (far, shade)"
                                     if few else "")), time.time()
                elif k == "q" or ev.key == pygame.K_ESCAPE:
                    running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and frame is not None:
                x, y = ev.pos
                if x < W and y < H and colour == "background":
                    bal = det._balance_to_walls(frame) if det.white_balance else frame
                    patch = bal[max(0, y - PATCH):y + PATCH + 1, max(0, x - PATCH):x + PATCH + 1].reshape(-1, 3)
                    bgr_samples["background"].extend(patch.tolist())
                    history.append(("background", len(patch)))
                    msg, msg_t = f"background: sample {len(history)}", time.time()
                    continue
                if x < W and y < H:
                    hsv = to_hsv(frame)
                    patch = hsv[max(0, y - PATCH):y + PATCH + 1, max(0, x - PATCH):x + PATCH + 1].reshape(-1, 3)
                    patch = patch[patch[:, 1] >= 60]         # grey wall / floor pixels are not a card colour
                    if len(patch) < 10:
                        msg, msg_t = "that is grey (wall / floor) - click right on a card", time.time()
                        continue
                    # the click must be roughly that colour (a mis-click must not spoil the range)
                    hmed = float(np.median(patch[:, 0]))
                    near = min(COLOURS, key=lambda c: hue_gap(hmed, NOMINAL_HUE[c]))
                    if hue_gap(hmed, NOMINAL_HUE[colour]) > 25:
                        msg, msg_t = (f"hue {hmed:.0f} looks {near}, not {colour} - "
                                      f"press {COLOURS.index(near) + 1} for {near}"), time.time()
                        continue
                    samples[colour].extend(patch.tolist())
                    # the same pixels in white-balanced BGR (the LUT works on those)
                    bal = det._balance_to_walls(frame) if det.white_balance else frame
                    bp = bal[max(0, y - PATCH):y + PATCH + 1, max(0, x - PATCH):x + PATCH + 1].reshape(-1, 3)
                    hp = hsv[max(0, y - PATCH):y + PATCH + 1, max(0, x - PATCH):x + PATCH + 1].reshape(-1, 3)
                    bgr_samples[colour].extend(bp[hp[:, 1] >= 60].tolist())
                    history.append((colour, len(patch)))
                    ranges[colour].update(hsv_range(samples[colour], colour))
                    hm = np.median(patch[:, 0])
                    msg, msg_t = f"{colour}: sample {len(history)} (hue {hm:.0f})", time.time()

        screen.fill((245, 246, 248))
        if frame is not None:
            view = frame
            if mask_view and colour != "background":
                det.hsv[colour] = dict(ranges[colour], min_area=det.hsv.get(colour, {}).get("min_area", 250))
                m = det.color_mask(to_hsv(frame), colour)
                view = frame.copy()
                view[m == 0] = (view[m == 0] * 0.25).astype(np.uint8)
            surf = pygame.image.frombuffer(cv2.cvtColor(view, cv2.COLOR_BGR2RGB).tobytes(), (W, H), "RGB")
            screen.blit(surf, (0, 0))
        x0 = W + 16
        screen.blit(font.render("Card colours", True, (30, 30, 30)), (x0, 14))
        for i, c in enumerate(CLASSES):
            y = 48 + i * 80
            sel = c == colour
            pygame.draw.rect(screen, (220, 232, 255) if sel else (255, 255, 255), (x0 - 6, y - 6, 280, 84), border_radius=8)
            pygame.draw.rect(screen, SWATCH[c], (x0, y, 18, 18), border_radius=4)
            screen.blit(font.render(f"{i + 1}  {c}  ({len(bgr_samples[c])} px)", True, (30, 30, 30)), (x0 + 26, y))
            if c == "background":
                screen.blit(small.render("walls, wood, beige, skin, clothes", True, (70, 70, 70)), (x0 + 26, y + 26))
                continue
            r = ranges[c]
            screen.blit(small.render(f"H {r['h_min']}..{r['h_max']}   S >= {r['s_min']}   V >= {r['v_min']}",
                                     True, (70, 70, 70)), (x0 + 26, y + 26))
            screen.blit(small.render("sampled - will be saved" if samples[c] else "not sampled (kept as is)",
                                     True, (40, 140, 60) if samples[c] else (150, 150, 150)), (x0 + 26, y + 46))
        help_ = "click card | U undo | C clear | M mask | SPACE freeze | N next pic | S save | Q quit"
        screen.blit(small.render(help_, True, (90, 90, 90)), (12, H + 12))
        if time.time() - msg_t < 4:
            screen.blit(font.render(msg, True, (20, 90, 200)), (x0, H - 30))
        if mask_view:
            screen.blit(font.render(f"MASK: {colour}", True, (255, 255, 255)), (12, 10))
        pygame.display.flip()
        clock.tick(30)
    src.close()
    pygame.quit()


if __name__ == "__main__":
    main()
