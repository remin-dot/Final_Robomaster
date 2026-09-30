"""Fix stage-3 example labels: a grid of box crops per label. Click to select (several),
1 circle  2 square  3 rect_wide  4 rect_tall  5 none (not a card)  D delete - the selected
move to that label. TAB next label page, arrows / PgUp PgDn scroll, Q quit.

    .venv/bin/python src/roi_label.py
"""
import glob
import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_dataset import OUT  # noqa: E402
from target_vision import ROI_CLASSES  # noqa: E402

TILE, COLS, ROWS = 72, 14, 8


def items(label):
    return sorted(p for p in glob.glob(os.path.join(OUT, label, "*.png")) if not p.endswith("_m.png"))


def move(path, label):
    for p in (path, path[:-4] + "_m.png"):
        if os.path.exists(p):
            os.replace(p, os.path.join(OUT, label, os.path.basename(p)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((COLS * TILE, ROWS * TILE + 60))
    font = pygame.font.SysFont("Helvetica", 15)
    page, scroll, sel = 0, 0, set()
    cache = {}
    while True:
        label = ROI_CLASSES[page]
        its = items(label)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN:
                k = ev.unicode.lower()
                if k == "q":
                    return
                if ev.key == pygame.K_TAB:
                    page, scroll, sel = (page + 1) % len(ROI_CLASSES), 0, set()
                elif ev.key in (pygame.K_DOWN, pygame.K_PAGEDOWN):
                    scroll = min(max(0, len(its) - COLS), scroll + (COLS if ev.key == pygame.K_DOWN else COLS * ROWS))
                elif ev.key in (pygame.K_UP, pygame.K_PAGEUP):
                    scroll = max(0, scroll - (COLS if ev.key == pygame.K_UP else COLS * ROWS))
                elif k in "12345" and k:
                    for p in sel:
                        move(p, ROI_CLASSES[int(k) - 1])
                    sel = set()
                elif k == "d":
                    for p in sel:
                        for q in (p, p[:-4] + "_m.png"):
                            if os.path.exists(q):
                                os.remove(q)
                    sel = set()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                x, y = ev.pos
                i = scroll + (y // TILE) * COLS + x // TILE
                if y < ROWS * TILE and i < len(its):
                    sel ^= {its[i]}
        screen.fill((245, 246, 248))
        for j, p in enumerate(its[scroll:scroll + COLS * ROWS]):
            if p not in cache:
                cache[p] = pygame.transform.scale(pygame.image.load(p), (TILE - 6, TILE - 6))
            x, y = (j % COLS) * TILE, (j // COLS) * TILE
            screen.blit(cache[p], (x + 3, y + 3))
            if p in sel:
                pygame.draw.rect(screen, (230, 30, 30), (x + 1, y + 1, TILE - 2, TILE - 2), 3)
        txt = (f"[{label}] {len(its)} examples   selected {len(sel)}   "
               "1 circle 2 square 3 wide 4 tall 5 none  D delete  TAB next label  arrows scroll  Q quit")
        screen.blit(font.render(txt, True, (30, 30, 30)), (10, ROWS * TILE + 20))
        pygame.display.flip()
        pygame.time.wait(30)


if __name__ == "__main__":
    main()
