"""Mission panel in pygame (window, widgets, input).

Same panel as the OpenCV one, drawn with pygame like RoboFinal's control
panel (branch feature/final, src/slam/ui_pygame.py - the immediate-mode UI
kit below is adapted from it). MissionPanel (mission_panel.py) stays the
model: map, detections, timer, target selection; MissionController
(control.py) owns the robot. This file only draws and handles input, in the
main thread (pygame on macOS requires that).

    Connect screen   robot (AP / STA / USB) / webcam / demo, round, blaster, Connect
    Header           source, Round 1 / 2, state, countdown, Start round N,
                     Pause / Resume, Finish, Save, STOP, Disconnect, Theme
    Camera           live view + segmentation (Overlay / Mask / Raw)
    Bottom           Sensors (ToF + Sharp), Detections + log, Aim HUD
    Right            map, tabs Targets / Select / Actions / Vision

Keys: Space = start / STOP, P = pause, M = camera view, G = map editor,
S = screenshot, Q / Esc = quit (Esc first closes a dialog or the editor).
"""

import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pygame

from mission_panel import BASE_DIR, HEADING_DEG, MissionMap
from target_vision import (COLORS, DEFAULT_CATALOGUE, SHAPE_LABEL, SHAPES, draw_detections, draw_ignored,
                           kind_label, kind_of, segmentation_mask)

W, H = 1280, 760
HEADER = 50

PALETTES = {
    "light": dict(
        bg=(243, 244, 246), panel=(255, 255, 255), panel2=(248, 249, 251), line=(223, 226, 231),
        text=(22, 25, 29), muted=(100, 107, 117), accent=(37, 99, 235), ok=(22, 163, 74),
        warn=(217, 119, 6), bad=(220, 38, 38), btn=(238, 240, 243), hover=(226, 229, 234),
        onbg=(219, 234, 254), white=(255, 255, 255), camera=(226, 229, 234)),
    "dark": dict(
        bg=(14, 17, 22), panel=(22, 26, 33), panel2=(27, 32, 41), line=(42, 48, 59),
        text=(230, 233, 238), muted=(144, 153, 166), accent=(91, 140, 255), ok=(60, 207, 122),
        warn=(245, 165, 36), bad=(255, 93, 93), btn=(34, 40, 52), hover=(43, 51, 65),
        onbg=(37, 55, 95), white=(255, 255, 255), camera=(27, 32, 41)),
}
CARD_RGB = {"red": (235, 64, 64), "blue": (64, 128, 245), "green": (60, 200, 90), "yellow": (240, 210, 40)}
PHASE_COLOR = {"COARSE": "warn", "FINE": "accent", "LOCKED": "ok", "FIRE": "bad", "DRY RUN": "warn"}
VIEW_MODES = ("overlay", "segmentation", "raw")


# =============================================================================
# immediate-mode widgets (adapted from RoboFinal src/slam/ui_pygame.py)
# =============================================================================
class UI:
    def __init__(self, theme="light"):
        pygame.font.init()
        mono = pygame.font.match_font("menlo,consolas,dejavusansmono,couriernew,monospace")
        sans = pygame.font.match_font("helveticaneue,helvetica,segoeui,arial,dejavusans")
        self.fonts = {
            "xs": pygame.font.Font(sans, 11), "s": pygame.font.Font(sans, 13), "sb": pygame.font.Font(sans, 13),
            "h": pygame.font.Font(sans, 16), "t": pygame.font.Font(sans, 21), "big": pygame.font.Font(sans, 28),
            "m": pygame.font.Font(mono, 12),
        }
        for k in ("sb", "h", "t", "big"):
            self.fonts[k].set_bold(True)
        self.set_theme(theme)
        self.focus = None
        self.buf = ""
        self.screen = None

    def set_theme(self, name):
        self.theme = name
        self.pal = PALETTES[name]
        self._cache = {}

    def col(self, c):
        return self.pal[c] if isinstance(c, str) else c

    def begin(self, screen, events):
        self.screen = screen
        self.mouse = self._logical(pygame.mouse.get_pos())
        self.clicked = any(e.type == pygame.MOUSEBUTTONUP and getattr(e, "button", 1) == 1 for e in events)
        if self.clicked:  # use the click position (events carry it; tests post synthetic clicks)
            for e in events:
                if e.type == pygame.MOUSEBUTTONUP:
                    self.mouse = e.pos
        self.text_in = "".join(e.text for e in events if e.type == pygame.TEXTINPUT)
        self.keys = [e.key for e in events if e.type == pygame.KEYDOWN]
        self.cursor = pygame.SYSTEM_CURSOR_ARROW

    @staticmethod
    def _logical(pos):
        return pos  # pygame.SCALED already maps mouse positions to the logical size

    def text(self, s, font="s", color="text"):
        key = (str(s), font, color if isinstance(color, str) else tuple(color), self.theme)
        surf = self._cache.get(key)
        if surf is None:
            surf = self.fonts[font].render(str(s), True, self.col(color))
            if len(self._cache) > 3000:
                self._cache.clear()
            self._cache[key] = surf
        return surf

    def label(self, s, x, y, font="s", color="text", align="left", maxw=None):
        s = str(s)
        if maxw and self.fonts[font].size(s)[0] > maxw:
            while s and self.fonts[font].size(s + "…")[0] > maxw:
                s = s[:-1]
            s += "…"
        surf = self.text(s, font, color)
        if align == "right":
            x -= surf.get_width()
        elif align == "center":
            x -= surf.get_width() // 2
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def hit(self, rect):
        return rect.collidepoint(self.mouse)

    def rrect(self, rect, color, radius=6, width=0):
        pygame.draw.rect(self.screen, self.col(color), rect, width, border_radius=radius)

    def card(self, rect, title=None):
        self.rrect(rect, "panel", 10)
        self.rrect(rect, "line", 10, 1)
        if title:
            self.label(title, rect.x + 12, rect.y + 9, "sb", "muted")

    def button(self, rect, label, kind="normal", on=False, enabled=True, font="s"):
        hover = enabled and self.hit(rect)
        if kind == "primary":
            bg, fg, border = "accent", "white", "accent"
        elif kind == "danger":
            bg, fg, border = "bad", "white", "bad"
        elif on:
            bg, fg, border = "onbg", "accent", "accent"
        else:
            bg, fg, border = ("hover" if hover else "btn"), "text", "line"
        if not enabled:
            bg, fg, border = "btn", "muted", "line"
        self.rrect(rect, bg, 7)
        self.rrect(rect, border, 7, 1)
        if hover and kind != "normal":
            self.rrect(rect, "white", 7, 1)
        surf = self.text(label, "sb" if kind != "normal" or on else font, fg)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        if hover:
            self.cursor = pygame.SYSTEM_CURSOR_HAND
        return enabled and hover and self.clicked

    def checkbox(self, rect, value, label=None, enabled=True):
        box = pygame.Rect(rect.x, rect.y + (rect.h - 16) // 2, 16, 16)
        self.rrect(box, "accent" if value else "panel2", 4)
        self.rrect(box, "accent" if value else "line", 4, 1)
        if value:
            pygame.draw.lines(self.screen, self.pal["white"], False,
                              [(box.x + 3, box.y + 8), (box.x + 7, box.y + 12), (box.x + 13, box.y + 4)], 2)
        if label:
            self.label(label, box.right + 7, rect.y + (rect.h - 16) // 2, "s", "text" if enabled else "muted")
        if enabled and self.hit(rect) and self.clicked:
            return not value
        return value

    def field(self, key, rect, value):
        """Text box. Returns the text when editing ends (Enter / click away), else None."""
        active = self.focus == key
        hover = self.hit(rect)
        self.rrect(rect, "panel2", 6)
        self.rrect(rect, "accent" if active else ("muted" if hover else "line"), 6, 1)
        shown = self.buf if active else str(value)
        surf = self.text(shown, "h")
        self.screen.blit(surf, (rect.x + 8, rect.y + (rect.h - surf.get_height()) // 2))
        if active and (time.time() * 2) % 2 < 1.3:
            cx = rect.x + 9 + surf.get_width()
            pygame.draw.line(self.screen, self.pal["text"], (cx, rect.y + 6), (cx, rect.bottom - 6))
        if not active:
            if hover and self.clicked:
                self.focus, self.buf = key, shown
            return None
        commit = False
        for k in self.keys:
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                commit = True
            elif k == pygame.K_ESCAPE:
                self.focus = None
                return None
            elif k == pygame.K_BACKSPACE:
                self.buf = self.buf[:-1]
        self.buf += "".join(c for c in self.text_in if c in "0123456789xX*")[:7]
        if self.clicked and not hover:
            commit = True
        if commit:
            self.focus = None
            return self.buf
        return None


def bgr_to_surface(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")


# =============================================================================
# the panel
# =============================================================================
class PygamePanel:
    def __init__(self, panel, controller, theme="light", headless=False):
        self.p = panel
        self.ctl = controller
        self.headless = headless
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        flags = 0 if headless else (pygame.SCALED | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("RoboMaster Rescue - Mission Panel")
        pygame.key.set_repeat(0)
        self.ui = UI(theme)
        self.clock = pygame.time.Clock()
        self.tab = "targets"
        self.view_mode = 0
        self.confirm = None          # (message, callback) modal
        self.running = True
        self.ui_fps = 0.0
        self.cam_rect = pygame.Rect(10, HEADER + 10, 832, 468)
        self.map_rect = pygame.Rect(852, HEADER + 10, 418, 418)

    # ------------------------------------------------------------------ loop
    def run(self):
        pygame.key.start_text_input()
        while self.running:
            self.step(pygame.event.get())
            self.clock.tick(self.p.target_fps)
        pygame.quit()

    def step(self, events):
        """One frame: handle events, draw, flip. Returns the screen (tests)."""
        for e in events:
            if e.type == pygame.QUIT:
                self.p.abort.set()
                self.running = False
        self.ui.begin(self.screen, events)
        self._keys(events)
        self.draw()
        pygame.mouse.set_cursor(self.ui.cursor) if not self.headless else None
        pygame.display.flip()
        fps = self.clock.get_fps()
        self.ui_fps = fps if fps else self.ui_fps
        return self.screen

    def _keys(self, events):
        if self.ui.focus is not None:
            return  # typing into a field
        ctl, p = self.ctl, self.p
        rv = getattr(p, "_trainer", None)
        if rv is not None and rv.review is not None:
            keys = {pygame.K_1: "circle", pygame.K_2: "square", pygame.K_3: "rect_wide", pygame.K_4: "rect_tall",
                    pygame.K_5: "none", pygame.K_d: "delete"}
            for e in events:
                if e.type == pygame.KEYDOWN:
                    if e.key in keys and rv.review["sel"]:
                        rv.review_assign(keys[e.key])
                    elif e.key == pygame.K_ESCAPE:
                        rv.review = None
            return
        for e in events:
            if e.type != pygame.KEYDOWN:
                continue
            k = e.key
            if k == pygame.K_ESCAPE:
                if self.confirm:
                    self.confirm = None
                elif p.editing:
                    p.editing = False
                else:
                    self._quit()
            elif k == pygame.K_q:
                self._quit()
            elif not ctl.connected:
                if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    ctl.connect()
            elif k == pygame.K_SPACE:
                ctl.stop() if ctl.running else ctl.start()   # RoboFinal: Space = STOP
            elif k == pygame.K_p:
                ctl.resume() if ctl.state == "paused" else ctl.pause()
            elif k == pygame.K_m:
                self.view_mode = (self.view_mode + 1) % len(VIEW_MODES)
            elif k == pygame.K_g:
                p.open_editor()
            elif k == pygame.K_s:
                self.screenshot()
            elif k == pygame.K_c and self.tab == "vision":
                self.capture_now()

    def _quit(self):
        self.p.abort.set()
        self.running = False

    def screenshot(self):
        path = os.path.join(self.p.data_dir, f"panel_{datetime.now():%Y%m%d_%H%M%S}.png")
        pygame.image.save(self.screen, path)
        self.p.log(f"screenshot -> {os.path.relpath(path, BASE_DIR)}")

    def ask(self, message, callback):
        self.confirm = (message, callback)

    # ------------------------------------------------------------------ draw
    def draw(self):
        ui = self.ui
        self.screen.fill(ui.pal["bg"])
        modal = self.confirm is not None     # a dialog opened during this frame shows from the next one
        if modal:                            # a modal eats the clicks of the frame below it
            clicked, ui.clicked = ui.clicked, False
        if not self.ctl.connected:
            self.draw_connect()
        else:
            self.draw_header()
            if getattr(self.p, "_trainer", None) is not None and self.p._trainer.review is not None:
                self.draw_review()
            else:
                self.draw_camera()
            y = self.cam_rect.bottom + 8
            self.draw_sensors(pygame.Rect(10, y, 290, H - y - 10))
            self.draw_detections(pygame.Rect(308, y, 262, H - y - 10))
            self.draw_aim(pygame.Rect(578, y, 264, H - y - 10))
            self.draw_map()
            self.draw_tabs(pygame.Rect(852, self.map_rect.bottom + 8, 418, H - self.map_rect.bottom - 18))
        if modal:
            ui.clicked = clicked
        if self.confirm:
            self.draw_confirm()

    # ---------------------------------------------------------------- connect
    def draw_connect(self):
        ui, ctl, p = self.ui, self.ctl, self.p
        bar = pygame.Rect(0, 0, W, HEADER)
        ui.rrect(bar, "panel", 0)
        pygame.draw.line(self.screen, ui.pal["line"], (0, HEADER - 1), (W, HEADER - 1))
        x = 16 + ui.label("RoboMaster Rescue", 16, 15, "h") + 12
        ui.label("not connected", x, 18, "s", "muted")
        if ui.button(pygame.Rect(W - 76, 10, 64, 30), "Theme"):
            ui.set_theme("dark" if ui.theme == "light" else "light")

        card = pygame.Rect(0, 0, 600, 560)
        card.center = (W // 2, HEADER + (H - HEADER) // 2)
        ui.card(card)
        x, y, cw = card.x + 28, card.y + 22, card.w - 56
        ui.label("Connect", x, y, "big")
        y += 44
        ui.label("Round 1: explore the maze, find + shoot targets.   Round 2: go shoot them.", x, y, "s", "muted")
        y += 30

        y = self.section("Camera / robot", x, y)
        bw = (cw - 12) // 3
        for i, (key, lbl) in enumerate((("robot", "RoboMaster robot"), ("webcam", "Webcam (this computer)"),
                                        ("demo", "Demo (no robot)"))):
            if ui.button(pygame.Rect(x + i * (bw + 6), y, bw, 34), lbl, on=ctl.source == key):
                ctl.source = key
        y += 44
        if ctl.source == "robot":
            y = self.section("Connection", x, y)
            for i, (key, lbl) in enumerate((("ap", "Wi-Fi direct (AP)"), ("sta", "Router (STA)"), ("rndis", "USB"))):
                if ui.button(pygame.Rect(x + i * (bw + 6), y, bw, 30), lbl, on=ctl.connection == key):
                    ctl.connection = key
            y += 38
            ui.label("AP: join the robot's Wi-Fi (RMEP-xxxxxx) on this computer first.", x, y, "xs", "muted")
        elif ctl.source == "webcam":
            ui.label("Live camera + detection only (no driving). macOS asks for camera access once.",
                     x, y, "xs", "muted")
        else:
            ui.label("Sample pictures + a simulated robot walk, to try the panel.", x, y, "xs", "muted")
        y += 26

        y = self.section("Round", x, y)
        half = (cw - 6) // 2
        for i, lbl in enumerate(("Round 1 · explore + find", "Round 2 · shoot targets")):
            if ui.button(pygame.Rect(x + i * (half + 6), y, half, 34), lbl, on=p.round_no == i + 1):
                p.set_round(i + 1)
        y += 42
        ui.label(f"maze {p.map.nx} x {p.map.ny}, start {tuple(p.map.start)}   (change after Connect: Actions > Map)",
                 x, y, "xs", "muted")
        y += 16
        f1 = p.round_file(1)
        if os.path.exists(f1):
            ts = time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(f1)))
            ui.label(f"round 1 file saved {ts}", x, y, "xs", "ok")
        elif p.round_no == 2:
            ui.label("no round 1 file yet: round 2 would explore again", x, y, "xs", "warn")
        y += 24

        ctl.armed = ui.checkbox(pygame.Rect(x, y, cw, 22), ctl.armed, "Blaster armed (off = dry run: aims, never fires)")
        y += 28
        n = len(p.selected)
        sel = ", ".join(kind_label(k) for k in sorted(p.selected)) or "NONE"
        if n > 4:
            sel = f"{n} kinds selected (sheet set = 4) - check the Select tab!"
        ui.label(f"shoot: {sel}", x, y, "s", "bad" if n == 0 else ("warn" if n > 4 else "accent"), maxw=cw)
        y += 18
        ui.label("(change after Connect: Select tab)", x, y, "xs", "muted")

        busy = ctl.state == "connecting"
        r = pygame.Rect(x, card.bottom - 66, cw, 44)
        if ctl.error:
            for i, ln in enumerate(self.wrap(ctl.error, "s", cw)[:3]):
                ui.label(ln, x, r.y - 58 + i * 17, "s", "bad")
        if ui.button(r, "Connecting…" if busy else "Connect", "primary", enabled=not busy, font="h"):
            ctl.connect()
        ui.label("Enter = Connect    Q = quit", card.right - 20, card.bottom - 18, "xs", "muted", "right")

    # ---------------------------------------------------------------- header
    def draw_header(self):
        ui, ctl, p = self.ui, self.ctl, self.p
        bar = pygame.Rect(0, 0, W, HEADER)
        ui.rrect(bar, "panel", 0)
        x = 14 + ui.label("RoboMaster Rescue", 14, 8, "h")
        ui.label("assignment 3-1-69", 14, 29, "xs", "muted")
        x += 12
        pygame.draw.circle(self.screen, ui.pal["ok"], (x + 5, 17), 5)
        ui.label(ctl.source_label(), x + 14, 10, "sb", "ok")
        sub = []
        if ctl.battery is not None:
            sub.append(f"battery {ctl.battery}%")
        if not ctl.armed:
            sub.append("DRY RUN")
        if ctl.released:
            sub.append("RELEASED")
        ui.label("  ".join(sub), x + 14, 29, "xs", "warn" if (not ctl.armed or ctl.released) else "muted")
        x = 330
        busy = not ctl.idle
        for n in (1, 2):
            if ui.button(pygame.Rect(x, 10, 84, 30), f"Round {n}", on=p.round_no == n,
                         enabled=not busy or p.round_no == n):
                p.set_round(n)
            x += 90
        state = ctl.state.upper()
        col = {"RUNNING": "ok", "PAUSED": "warn", "DONE": "accent", "BUSY": "accent"}.get(state, "muted")
        tw = ui.fonts["sb"].size(state)[0] + 20
        pill = pygame.Rect(x + 2, 13, tw, 24)
        ui.rrect(pill, col, 12, 1)
        ui.label(state, pill.centerx, 17, "sb", col, "center")
        self.draw_timer(pill.right + 14)

        specs = [("Theme", "theme", "normal", 60, True),
                 ("Disconnect", "disconnect", "normal", 92, True),
                 ("STOP", "stop", "danger", 70, ctl.running),
                 ("Save", "save", "normal", 56, True),
                 ("Finish", "finish", "normal", 64, ctl.running),
                 ("Resume" if ctl.state == "paused" else "Pause", "pause", "normal", 72, ctl.running),
                 (f"Start round {p.round_no}", "start", "primary", 116, ctl.can_start())]
        bx = W - 10
        for label, key, kind, bw, enabled in specs:
            r = pygame.Rect(bx - bw, 10, bw, 30)
            bx -= bw + 6
            if ui.button(r, label, kind, enabled=enabled):
                if key == "theme":
                    ui.set_theme("dark" if ui.theme == "light" else "light")
                elif key == "disconnect":
                    self.ask("Disconnect from the " + ("robot?" if ctl.source == "robot" else ctl.source + "?"),
                             ctl.disconnect)
                elif key == "pause":
                    ctl.resume() if ctl.state == "paused" else ctl.pause()
                else:
                    getattr(ctl, key)()

    def draw_timer(self, x):
        ui, p = self.ui, self.p
        limit, el = p.time_limit(), p.elapsed()
        left = limit - el
        if not p.round_t0:
            col = "muted"
        elif left <= 0:
            col = "bad"
        elif left <= 30:
            col = "bad" if int(time.time() * 2) % 2 else "warn"   # blink in the last 30 s
        elif left <= 60:
            col = "warn"
        else:
            col = "ok"
        ui.label(("LEFT -" if left < 0 else "LEFT ") + p.fmt(abs(left)), x, 6, "t", col)
        ui.label(f"{p.fmt(el)} / {p.fmt(limit)}", x, 31, "xs", "text" if p.round_t0 else "muted")
        frac = min(1.0, el / float(limit)) if limit else 0.0
        pygame.draw.rect(self.screen, ui.pal["line"], (0, HEADER - 3, W, 3))
        pygame.draw.rect(self.screen, ui.col(col), (0, HEADER - 3, int(W * frac), 3))

    # ---------------------------------------------------------------- camera
    def draw_camera(self):
        ui, p = self.ui, self.p
        r = self.cam_rect
        frame, dets, ts, _ = p.worker.latest()
        aim = p.aim.view()
        if frame is None:
            ui.rrect(r, "camera", 8)
            ui.label("waiting for camera…", r.centerx, r.centery - 8, "h", "muted", "center")
        else:
            mode = VIEW_MODES[self.view_mode]
            if mode == "segmentation":
                view = segmentation_mask(frame.shape, dets, targets_only=False)
            else:
                view = frame.copy()
            if mode != "raw":   # dim what detection ignores: the room above the walls / horizon
                draw_ignored(view, p.detector.last_ignore_line)
            if mode == "overlay":
                draw_detections(view, dets)
            view = cv2.resize(view, (r.w, r.h), interpolation=cv2.INTER_LINEAR)
            p._draw_boresight(view, aim)
            self.screen.blit(bgr_to_surface(view), r.topleft)
            stale = time.time() - ts > 1.0
            badge = pygame.Rect(r.x + 8, r.y + 8, 440, 26)
            s = pygame.Surface(badge.size, pygame.SRCALPHA)
            s.fill((*ui.pal["panel"], 215))
            self.screen.blit(s, badge.topleft)
            if not stale and int(time.time() * 2) % 2:
                pygame.draw.circle(self.screen, ui.pal["bad"], (badge.x + 12, badge.centery), 5)
            ui.label(("LIVE" if not stale else "NO SIGNAL") +
                     f"   cam {p.worker.fps:4.1f} fps   det {p.worker.det_fps:4.1f} fps ({p.worker.detect_ms:3.0f} ms)"
                     f"   ui {self.ui_fps:4.1f} fps", badge.x + 24, badge.y + 5, "s", "bad" if stale else "ok")
            if stale:
                ui.label("NO SIGNAL", r.centerx, r.centery, "big", "bad", "center")
        # view mode switch (M)
        bx = r.right - 3 * 76 - 8
        for i, (m, lbl) in enumerate((("overlay", "Overlay"), ("segmentation", "Mask"), ("raw", "Raw"))):
            if ui.button(pygame.Rect(bx + i * 76, r.y + 8, 72, 26), lbl, on=self.view_mode == i):
                self.view_mode = i
        if not self.ctl.armed:
            ui.label("DRY RUN – blaster off", r.right - 12, r.bottom - 26, "sb", "warn", "right")
        pygame.draw.rect(self.screen, ui.pal["line"], r, 1, border_radius=2)

    # ---------------------------------------------------------------- sensors
    def draw_sensors(self, r):
        ui, p = self.ui, self.p
        ui.card(r, "SENSORS")
        tel = p.telemetry() or {}
        cx, cy = r.x + 84, r.y + 122
        body = pygame.Rect(0, 0, 26, 34)
        body.center = (cx, cy)
        ui.rrect(body, "text", 4)
        pygame.draw.polygon(self.screen, ui.pal["panel"], [(cx, cy - 10), (cx - 6, cy + 2), (cx + 6, cy + 2)])
        ir_max = float(tel.get("ir_max_cm", 30.0))
        wall = float(tel.get("ir_wall_cm", 16.9))
        side_len = 46
        # front-corner IR obstacle modules (on/off): a short diagonal at each front corner
        for side, key, name in ((-1, "corner_left_near", "FL"), (1, "corner_right_near", "FR")):
            state = tel.get(key)
            if state is None:
                continue                                   # not fitted / no reading
            x0, y0 = cx + side * 13, body.top + 2
            xe, ye = int(x0 + side * 22), int(y0 - 22)
            c = "bad" if state else "ok"
            pygame.draw.line(self.screen, ui.col(c), (x0, y0), (xe, ye), 6)
            ui.label(f"{name} {'WALL' if state else 'clear'}", xe + side * 3, ye - 13, "xs", c,
                     "left" if side > 0 else "right")
            if state:
                pygame.draw.circle(self.screen, ui.pal["bad"], (x0, y0), 5)
        # Sharp distance sensors on the sides: horizontal bars
        side_iter = ((-1, "ir_left_cm", "L"), (1, "ir_right_cm", "R"))
        for side, key, name in side_iter:
            v = tel.get(key)
            x0 = cx + side * 15
            xe = x0 + side * side_len
            pygame.draw.line(self.screen, ui.pal["line"], (x0, cy), (xe, cy), 8)
            if v is not None:
                frac = min(max(v / ir_max, 0.0), 1.0)
                c = "bad" if v <= wall else "ok"
                pygame.draw.line(self.screen, ui.col(c), (x0, cy), (int(x0 + side * side_len * frac), cy), 8)
                wx = int(x0 + side * side_len * wall / ir_max)
                pygame.draw.line(self.screen, ui.pal["text"], (wx, cy - 7), (wx, cy + 7), 1)
                ui.label(f"{v:.1f} cm", (x0 + xe) // 2, cy + 12, "xs", c, "center")
            ui.label(name, xe, cy - 22, "xs", "muted", "center")
        # a sensor that gives no signal (cable / port / power): say so right here
        warn = tel.get("sensor_warn")
        if warn:
            ui.label("! " + warn, r.x + 12, r.bottom - 20, "xs", "bad", maxw=r.w - 24)
        tof = tel.get("tof_mm")
        top = r.y + 38
        pygame.draw.line(self.screen, ui.pal["line"], (cx, body.top - 2), (cx, top), 8)
        if tof is not None:
            frac = min(max(tof / 1000.0, 0.0), 1.0)
            c = "bad" if tof <= 300 else "ok"
            pygame.draw.line(self.screen, ui.col(c), (cx, body.top - 2),
                             (cx, int(body.top - 2 - (body.top - 2 - top) * frac)), 8)
            ui.label(f"ToF {tof:.0f} mm", cx + 10, top - 2, "xs", c)
        rows = [("cell", str(tuple(p.map.robot))), ("heading", f"{HEADING_DEG.get(p.map.heading, 0)}°"),
                ("gimbal", f"{p.map.gimbal_abs:.0f}°")]
        cw = tel.get("cam_wall_m")
        if cw is not None:
            rows.insert(0, ("cam wall", "clear" if cw == float("inf") else f"{cw:.2f} m"))
        for k in ("yaw", "odom"):
            if k in tel:
                rows.append((k, str(tel[k]).replace(" m", "").replace(" deg", "°").replace(", ", ",")))
        for i, (k, v) in enumerate(rows):
            y = r.y + 70 + i * 20
            ui.label(k, r.x + 160, y, "xs", "muted")
            ui.label(v, r.x + 206, y, "xs", "text", maxw=r.right - r.x - 212)

    # ---------------------------------------------------------------- detections
    def draw_detections(self, r):
        ui, p = self.ui, self.p
        ui.card(r, f"DETECTIONS  ≤ {p.detector.max_shoot_m:.2f} m")
        _, dets, _, _ = p.worker.latest()
        cards = [d for d in dets if d.is_card]
        for i, d in enumerate(cards[:4]):
            y = r.y + 34 + i * 21
            pygame.draw.rect(self.screen, CARD_RGB.get(d.color, (0, 0, 0)), (r.x + 12, y + 3, 11, 11), border_radius=2)
            ui.label(d.label, r.x + 30, y, "s", "text" if d.is_target else "muted")
            ui.label(f"{d.distance_m:.2f} m" if d.distance_m else "--", r.x + 150, y, "s")
            if d.is_target:
                ui.label("IN RANGE" if d.in_range else "far", r.right - 10, y, "sb" if d.in_range else "s",
                         "ok" if d.in_range else "muted", "right")
            else:
                ui.label("not sel.", r.right - 10, y, "xs", "muted", "right")
        if not cards:
            ui.label("no card in view", r.x + 12, r.y + 34, "s", "muted")
        ly = r.y + 124
        pygame.draw.line(self.screen, ui.pal["line"], (r.x + 10, ly), (r.right - 10, ly))
        for i, line in enumerate(p._log[-4:]):
            ui.label(line[9:], r.x + 12, ly + 6 + i * 16, "xs", "muted", maxw=r.w - 22)

    # ---------------------------------------------------------------- aim
    def draw_aim(self, r):
        ui, p = self.ui, self.p
        aim = p.aim.view()
        ui.card(r, f"AIM   lock {aim['need']} frames")
        span, sz = 6.0, 96
        box = pygame.Rect(r.x + 12, r.y + 32, sz, sz)
        ui.rrect(box, "panel2", 4)
        ui.rrect(box, "line", 4, 1)
        pygame.draw.line(self.screen, ui.pal["line"], (box.x, box.centery), (box.right, box.centery))
        pygame.draw.line(self.screen, ui.pal["line"], (box.centerx, box.y), (box.centerx, box.bottom))
        t = int(sz / 2 * aim["tol"] / span)
        pygame.draw.rect(self.screen, ui.pal["ok"], (box.centerx - t, box.centery - t, 2 * t, 2 * t), 1)

        def to_px(yaw, pitch):
            k = sz / 2 / span
            return (int(box.centerx + max(-span, min(span, yaw)) * k),
                    int(box.centery - max(-span, min(span, pitch)) * k))

        pcol = PHASE_COLOR.get(aim["phase"], "muted")
        if aim["active"]:
            trail = [h for h in aim["hist"] if h[0] > time.time() - 4][-12:]
            if len(trail) > 1:
                pygame.draw.lines(self.screen, ui.pal["line"], False, [to_px(h[1], h[2]) for h in trail], 1)
            if aim["yaw"] is not None:
                pygame.draw.circle(self.screen, ui.col(pcol), to_px(aim["yaw"], aim["pitch"]), 5)
        tx = box.right + 12
        ui.label(aim["phase"], tx, r.y + 32, "t", pcol)
        if aim["active"] and aim["color"]:
            kind = aim["color"].split(" #")[0]
            c = kind.split(" ")[0]
            ui.label(kind_label(kind) if " " in kind else kind, tx, r.y + 58, "s", CARD_RGB.get(c, ui.pal["text"]))
            if aim["yaw"] is not None:
                ui.label(f"yaw   {aim['yaw']:+.2f}°", tx, r.y + 78, "xs")
                ui.label(f"pitch {aim['pitch']:+.2f}°", tx, r.y + 94, "xs")
        for i in range(aim["need"]):
            b = pygame.Rect(tx + i * 18, r.y + 112, 12, 12)
            filled = aim["active"] and i < aim["count"]
            ui.rrect(b, "ok" if filled else "line", 2, 0 if filled else 1)
        # error trace (last 10 s)
        g = pygame.Rect(r.x + 12, r.y + 138, r.w - 24, r.bottom - r.y - 146)
        ui.rrect(g, "panel2", 4)
        band = max(1, int(g.h / 2 * aim["tol"] / span))
        pygame.draw.rect(self.screen, ui.col("onbg"), (g.x, g.centery - band, g.w, 2 * band))
        now = time.time()
        pts = [h for h in aim["hist"] if h[0] > now - 10]
        for idx, c in ((1, "accent"), (2, "ok")):
            poly = [(int(g.right - (now - h[0]) / 10.0 * g.w), int(g.centery - max(-span, min(span, h[idx])) / span * g.h / 2))
                    for h in pts]
            if len(poly) > 1:
                pygame.draw.lines(self.screen, ui.col(c), False, poly, 1)

    # ---------------------------------------------------------------- map
    def draw_map(self):
        ui, p = self.ui, self.p
        r = self.map_rect
        editing_start = not p.editing and p.can_edit_start()
        if editing_start:   # handle the click first, so this frame already shows the new start
            strip = pygame.Rect(r.x + 6, r.bottom - 36, r.w - 12, 30)
            if ui.clicked and r.collidepoint(ui.mouse) and not strip.collidepoint(ui.mouse):
                c = p.map.cell_at(ui.mouse[0] - r.x, ui.mouse[1] - r.y, r.w)
                if c:
                    p.click_start(c)
        img = p._editor_preview(r.w) if p.editing else p.map.render(r.w, fov_deg=p.detector.hfov_deg)
        self.screen.blit(bgr_to_surface(img), r.topleft)
        pygame.draw.rect(self.screen, ui.pal["line"], r, 1)
        if p.editing:
            self.draw_editor(r)
        elif editing_start:
            self.draw_start_strip(r)

    def draw_start_strip(self, r):
        """The robot can be put down anywhere: click a map cell = start there,
        click the start cell again = turn which way it faces (N E S W)."""
        ui, p = self.ui, self.p
        strip = pygame.Rect(r.x + 6, r.bottom - 36, r.w - 12, 30)
        s = pygame.Surface(strip.size, pygame.SRCALPHA)
        s.fill((*ui.pal["panel"], 225))
        self.screen.blit(s, strip.topleft)
        m = p.map
        ui.label(f"Start {tuple(m.start)} facing {'NESW'[m.start_heading]}", strip.x + 8, strip.y + 7, "sb", "ok")
        ui.label("click cell = move, again = turn", strip.x + 160, strip.y + 8, "xs", "muted")
        dirty = getattr(p, "_start_dirty", False)
        if ui.button(pygame.Rect(strip.right - 62, strip.y + 3, 56, 24), "Save", "primary" if dirty else "normal",
                     font="xs"):
            p.save_start()

    def draw_editor(self, r):
        """Custom map size (e.g. 5x4) + start cell, saved to settings.yaml."""
        ui, p = self.ui, self.p
        # click a cell of the preview = start
        size = p._parse_size()
        box = pygame.Rect(r.x + 16, r.bottom - 150, r.w - 32, 138)
        if size and ui.clicked and r.collidepoint(ui.mouse) and not box.collidepoint(ui.mouse):
            c = MissionMap(size[0], size[1], p.map.tile).cell_at(ui.mouse[0] - r.x, ui.mouse[1] - r.y, r.w)
            if c:
                p.edit_start = c
        ui.card(box, "CUSTOM MAP  (width x height)")
        new = ui.field("mapsize", pygame.Rect(box.x + 12, box.y + 30, 110, 32), p.edit_text)
        if new is not None:
            p.edit_text = new
        st = p.edit_start or (0, 0)
        ui.label(f"start ({st[0]}, {st[1]})  – click a cell", box.x + 134, box.y + 38, "s")
        for i, preset in enumerate(("6x6", "5x4", "4x4", "5x5")):
            if ui.button(pygame.Rect(box.x + 12 + i * 62, box.y + 70, 56, 26), preset, on=p.edit_text == preset):
                p.edit_text = preset
                ui.focus = None
        ui.label(p.edit_msg or "Enter in the box, then Save", box.x + 12, box.y + 110, "xs",
                 "bad" if p.edit_msg else "muted")
        if ui.button(pygame.Rect(box.right - 150, box.y + 102, 70, 28), "Save", "primary"):
            p.save_editor()
        if ui.button(pygame.Rect(box.right - 74, box.y + 102, 62, 28), "Cancel"):
            p.editing = False

    # ---------------------------------------------------------------- tabs
    def draw_tabs(self, r):
        ui, p = self.ui, self.p
        ui.card(r)
        for i, (key, lbl) in enumerate((("targets", "Targets"), ("select", "Select"), ("actions", "Actions"),
                                        ("vision", "Vision"))):
            if ui.button(pygame.Rect(r.x + 8 + i * 80, r.y + 8, 76, 28), lbl, on=self.tab == key):
                self.tab = key
        n = len(p.selected)
        # more than the sheet's 4 kinds: every extra kind risks a -1 wrong-target shot
        ui.label(f"{n} kind{'s' if n != 1 else ''}" + (" !" if n > 4 else ""),
                 r.right - 12, r.y + 15, "sb", "bad" if n == 0 else ("warn" if n > 4 else "accent"), "right")
        inner = pygame.Rect(r.x + 10, r.y + 44, r.w - 20, r.h - 52)
        {"targets": self.tab_targets, "select": self.tab_select, "actions": self.tab_actions,
         "vision": self.tab_vision}[self.tab](inner)

    # ---------------------------------------------------------------- vision (box classifier)
    def tab_vision(self, r):
        """Stage-3 box classifier: examples, model, auto label / train, review the unsure."""
        from vision_trainer import counts
        ui, p, ctl = self.ui, self.p, self.ctl
        tr = p.trainer
        idle = not ctl.running and not tr.busy
        n = counts()
        short = (("circle", "circle"), ("square", "square"), ("rect_wide", "wide"), ("rect_tall", "tall"),
                 ("none", "none"), ("unsure", "unsure"))
        ui.label("examples: " + "  ".join(f"{s_} {n[c]}" for c, s_ in short), r.x, r.y, "xs", "text", maxw=r.w)
        info = tr.info or {}
        if p.detector.roi_clf is not None:
            ui.label("model in use  " + str(info.get("note", "(trained by hand)")), r.x, r.y + 18, "xs", "ok", maxw=r.w)
        else:
            ui.label("no model - shapes by area only" + (f"  ({info['note']})" if info.get("note") else ""),
                     r.x, r.y + 18, "xs", "muted", maxw=r.w)
        st = (f"[{tr.busy}…] " if tr.busy else "") + (tr.status or "record a round, then Label + train")
        ui.label(st, r.x, r.y + 36, "xs", "warn" if tr.busy else "muted", maxw=r.w)
        bw = (r.w - 12) // 3
        y = r.y + 56
        for i, (lbl, cb, kind) in enumerate((("Auto label", tr.label_now, "normal"),
                                             ("Train", tr.train_now, "normal"),
                                             ("Label + train", tr.auto_now, "primary"))):
            if ui.button(pygame.Rect(r.x + i * (bw + 6), y, bw, 28), lbl, kind, enabled=idle):
                cb()
        y += 34
        bw4 = (r.w - 18) // 4
        for i, (lbl, cb, kind, en) in enumerate(((f"Review ({n['unsure']})", tr.start_review, "normal", idle),
                                                 ("Clean data", tr.clean_now, "normal", idle),
                                                 ("Colour samples", tr.colour_samples_now, "normal", idle),
                                                 ("Remove model", tr.delete_model, "danger",
                                                  idle and p.detector.roi_clf is not None))):
            if ui.button(pygame.Rect(r.x + i * (bw4 + 6), y, bw4, 28), lbl, kind, enabled=en, font="xs"):
                cb()
        y += 34
        v = ui.checkbox(pygame.Rect(r.x, y, r.w - 60, 22), p._auto_after_round(),
                        "after each round: label + train automatically")
        if v != p._auto_after_round():
            p._toggle_auto_after_round()
        # capture from the camera: put ONE card in the middle of the view, pick its shape, Capture (C)
        y += 28
        chips = (("auto", "auto"), ("circle", "circle"), ("square", "square"), ("rect_wide", "wide"),
                 ("rect_tall", "tall"), ("none", "not card"))
        cw_ = 44
        for i, (key, lbl) in enumerate(chips):
            if ui.button(pygame.Rect(r.x + i * (cw_ + 3), y, cw_ + (10 if key == "none" else 0), 26), lbl,
                         on=tr.capture_label == key, font="xs"):
                tr.capture_label = key
        cx0 = r.x + 6 * (cw_ + 3) + 12
        if ui.button(pygame.Rect(cx0, y, r.right - cx0 - 40, 26), "Capture", "primary",
                     enabled=not tr.busy and self.ctl.connected):
            self.capture_now()
        if tr.last_capture is not None:
            self.screen.blit(pygame.transform.scale(bgr_to_surface(tr.last_capture), (32, 32)),
                             (r.right - 34, y - 3))

    def capture_now(self):
        frame, _, ts, _ = self.p.worker.latest()
        if frame is None or time.time() - ts > 1.0:
            self.p.trainer._say("no live camera picture - connect the camera first")
            return
        self.p.trainer.capture(frame.copy(), self.p.trainer.capture_label)

    def draw_review(self):
        """Review grid over the camera view: click crops, then a label (keys 1-5, D delete)."""
        ui, p = self.ui, self.p
        tr = p.trainer
        rv = tr.review
        r = self.cam_rect
        ui.rrect(r, "panel", 8)
        pygame.draw.rect(self.screen, ui.pal["line"], r, 1, border_radius=8)
        cols, rows, tile = 11, 5, 74
        per = cols * rows
        items = rv["items"]
        pages = max(1, (len(items) + per - 1) // per)
        rv["page"] = min(rv["page"], pages - 1)
        ui.label(f"Review '{rv['label']}': {len(items)} crops  (page {rv['page'] + 1}/{pages}) - click to select, "
                 "then a label (keys 1-5, D delete)", r.x + 12, r.y + 10, "s", "text", maxw=r.w - 24)
        cache = self.__dict__.setdefault("_thumbs", {})
        for j, path in enumerate(items[rv["page"] * per:(rv["page"] + 1) * per]):
            tx, ty = r.x + 12 + (j % cols) * tile, r.y + 36 + (j // cols) * tile
            surf = cache.get(path)
            if surf is None:
                img = cv2.imread(path)
                img = cv2.resize(img, (64, 64)) if img is not None else np.full((64, 64, 3), 200, np.uint8)
                surf = cache[path] = bgr_to_surface(img)
            self.screen.blit(surf, (tx, ty))
            rect = pygame.Rect(tx, ty, 64, 64)
            if path in rv["sel"]:
                pygame.draw.rect(self.screen, ui.pal["bad"], rect.inflate(6, 6), 3, border_radius=4)
            if ui.hit(rect) and ui.clicked:
                rv["sel"] ^= {path}
        by = r.bottom - 40
        bw = 84
        for i, (lbl, key) in enumerate((("1 circle", "circle"), ("2 square", "square"), ("3 wide", "rect_wide"),
                                        ("4 tall", "rect_tall"), ("5 not card", "none"), ("D delete", "delete"))):
            if ui.button(pygame.Rect(r.x + 12 + i * (bw + 4), by, bw, 30), lbl,
                         "danger" if key == "delete" else "normal", enabled=bool(rv["sel"]), font="xs"):
                tr.review_assign(key)
        bx = r.x + 12 + 6 * (bw + 4) + 8
        if ui.button(pygame.Rect(bx, by, 56, 30), "< page", font="xs"):
            rv["page"] = max(0, rv["page"] - 1)
        if ui.button(pygame.Rect(bx + 60, by, 56, 30), "page >", font="xs"):
            rv["page"] += 1
        if ui.button(pygame.Rect(bx + 120, by, 44, 30), "all", font="xs"):
            rv["sel"] = set(items[rv["page"] * per:(rv["page"] + 1) * per])
        if ui.button(pygame.Rect(r.right - 90, by, 78, 30), "Done", "primary"):
            tr.review = None

    def tab_targets(self, r):
        ui, p = self.ui, self.p
        rows = [("now", p.map._target_view(t)) for t in list(p.map.targets.values())]
        for k in p.map.known.values():
            if not p.map._find_card(k["kind"], k["x_m"], k["y_m"], 0.6):
                rows.append(("r1", k))
        rows.sort(key=lambda row: (row[1]["kind"] not in p.selected, row[1]["id"]))
        hits = dict(p.splits())
        if not rows:
            ui.label("no cards found yet", r.x + 4, r.y + 4, "s", "muted")
            sel = ", ".join(kind_label(k) for k in sorted(p.selected)) or "none – pick in Select"
            for i, ln in enumerate(self.wrap("shooting: " + sel, "s", r.w - 8)[:3]):
                ui.label(ln, r.x + 4, r.y + 26 + i * 17, "s")
            return
        for i, (src, v) in enumerate(rows[:6]):
            y = r.y + i * 30
            self.icon((r.x + 12, y + 9), v["color"], v["shape"], 9, hollow=src == "r1")
            chosen = v["kind"] in p.selected
            name = kind_label(v["id"].split(" #")[0]) + (" #" + v["id"].split(" #")[1] if " #" in v["id"] else "")
            ui.label(name, r.x + 30, y + 1, "s", "text" if chosen else "muted")
            ui.label(f"cell {tuple(v['cell'])}", r.x + 196, y + 1, "s", "text" if chosen else "muted")
            if src == "r1":
                st, c = f"round {p.round_no - 1}", "muted"
            elif v["shot"]:
                st, c = (f"HIT {p.fmt(hits[v['id']])}" if v["id"] in hits else "HIT"), "ok"
            elif not v["confirmed"]:
                st, c = "checking", "warn"
            elif chosen:
                st, c = "to shoot", "accent"
            else:
                st, c = "not selected", "muted"
            ui.label(st, r.right - 4, y + 1, "sb", c, "right")
        if len(rows) > 6:
            ui.label(f"+{len(rows) - 6} more on the map", r.x + 4, r.bottom - 16, "xs", "muted")

    def tab_select(self, r):
        """Any sheet colour x shape: e.g. blue circle + red circle."""
        ui, p = self.ui, self.p
        editable = not self.ctl.running
        names = {"circle": "circle", "rect_wide": "wide rect", "rect_tall": "tall rect", "square": "square"}
        cw = (r.w - 66) // 4
        for j, sh in enumerate(SHAPES):
            cx = r.x + 66 + j * cw + cw // 2
            self.icon((cx - 26, r.y + 7), None, sh, 6, hollow=True)
            ui.label(names[sh], cx - 16, r.y, "xs", "muted")
        for i, c in enumerate(COLORS):
            y = r.y + 18 + i * 31
            pygame.draw.rect(self.screen, CARD_RGB[c], (r.x + 2, y + 8, 12, 12), border_radius=3)
            ui.label(c, r.x + 20, y + 6, "s")
            for j, sh in enumerate(SHAPES):
                kind = kind_of(c, sh)
                on = kind in p.selected
                star = "*" if DEFAULT_CATALOGUE.get(c) == sh else ""
                if ui.button(pygame.Rect(r.x + 66 + j * cw + 2, y + 2, cw - 4, 26),
                             ("ON" if on else "–") + (" " + star if star else ""), on=on, enabled=editable):
                    p.toggle_kind(kind)
        y = r.y + 18 + 4 * 31 + 6
        bw = (r.w - 18) // 4
        presets = (("Sheet set *", lambda: p.set_selected({kind_of(c, s) for c, s in DEFAULT_CATALOGUE.items()})),
                   ("All", lambda: p.set_selected({kind_of(c, s) for c in COLORS for s in SHAPES})),
                   ("None", lambda: p.set_selected(set())),
                   ("Save", p.save_selection))
        for i, (lbl, cb) in enumerate(presets):
            if ui.button(pygame.Rect(r.x + i * (bw + 6), y, bw, 28), lbl, "primary" if lbl == "Save" else "normal",
                         enabled=editable):
                cb()
        ui.label("* = the sheet's card.  Unselected cards are mapped, never shot.", r.x, y + 34, "xs", "muted")

    def tab_actions(self, r):
        """RoboFinal's Mission tab: actions, blaster, calibration, gimbal, aim trim."""
        ui, p, ctl = self.ui, self.p, self.ctl
        robot = ctl.chassis is not None
        idle = ctl.idle
        half = (r.w - 6) // 2
        y = r.y
        if ui.button(pygame.Rect(r.x, y, half, 28), "Look around now", enabled=robot and idle):
            ctl.look_around()
        if ui.button(pygame.Rect(r.x + half + 6, y, half, 28), "Aim & shoot here", enabled=robot and idle):
            ctl.aim_here()
        y += 34
        if ui.button(pygame.Rect(r.x, y, half, 28), "Fire once", enabled=robot):
            ctl.fire_once()
        if ui.button(pygame.Rect(r.x + half + 6, y, half, 28), f"Map {p.map.nx} x {p.map.ny}…",
                     enabled=idle and p.round_no == 1):
            p.open_editor()
        y += 36
        # turning the blaster OFF during a round needs a second click within 3 s (last run it
        # was switched off by accident at 4:30 and nothing could be hit after that)
        confirm = time.time() - getattr(self, "_disarm_click", 0.0) < 3.0
        label = "Blaster armed" if not confirm else "Click again: blaster OFF?"
        armed = ui.checkbox(pygame.Rect(r.x, y, half, 26), ctl.armed, label)
        if armed != ctl.armed:
            if not armed and ctl.running and not confirm:
                self._disarm_click = time.time()
            else:
                self._disarm_click = 0.0
                ctl.set_armed(armed)
        if ui.button(pygame.Rect(r.x + half + 6, y, half, 28), "Calibrate: card at 1 m", enabled=idle):
            ctl.calibrate_distance()
        y += 36
        bw = (r.w - 62 - 4 * 5) // 5
        ui.label("Gimbal", r.x, y + 6, "xs", "muted")
        for i, (lbl, cb) in enumerate((("Left 15", lambda: ctl.gimbal_move(yaw=-15)),
                                      ("Right 15", lambda: ctl.gimbal_move(yaw=15)),
                                      ("Up 5", lambda: ctl.gimbal_move(pitch=5)),
                                      ("Down 5", lambda: ctl.gimbal_move(pitch=-5)),
                                      ("Centre", ctl.gimbal_center))):
            if ui.button(pygame.Rect(r.x + 62 + i * (bw + 5), y, bw, 26), lbl, enabled=robot and idle, font="xs"):
                cb()
        y += 32
        ui.label("Aim trim", r.x, y + 6, "xs", "muted")
        sh = p.config.get("shooting", {}) or {}
        for i, (lbl, cb) in enumerate((("Up", lambda: ctl.nudge_aim(pitch=0.5)),
                                      ("Down", lambda: ctl.nudge_aim(pitch=-0.5)),
                                      ("Left", lambda: ctl.nudge_aim(yaw=-0.5)),
                                      ("Right", lambda: ctl.nudge_aim(yaw=0.5)),
                                      ("Save", ctl.save_trim))):
            if ui.button(pygame.Rect(r.x + 62 + i * (bw + 5), y, bw, 26), lbl, font="xs"):
                cb()
        y += 32
        ui.label(f"trim pitch {float(sh.get('aim_pitch_offset_deg', 0.0)):+.1f}°  "
                 f"yaw {float(sh.get('aim_yaw_offset_deg', 0.0)):+.1f}°   (shots high -> Down)", r.x, y, "xs", "muted")
        y += 18
        if ctl.released:
            if ui.button(pygame.Rect(r.x, y, r.w, 28), "Wake robot (motors on, gimbal centred)", "primary", enabled=idle):
                ctl.wake()
        elif ui.button(pygame.Rect(r.x, y, r.w, 28), "Release robot (stop wheels + gimbal, free mode, gimbal limp)",
                       enabled=robot and idle):
            ctl.release()
        if not robot:
            ui.label("connect the robot for these buttons (demo / webcam = camera only)", r.x, y + 32, "xs", "muted")

    # ---------------------------------------------------------------- dialog
    def draw_confirm(self):
        ui = self.ui
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 110))
        self.screen.blit(shade, (0, 0))
        box = pygame.Rect(0, 0, 420, 140)
        box.center = (W // 2, H // 2)
        ui.card(box)
        msg, cb = self.confirm
        ui.label(msg, box.centerx, box.y + 30, "h", "text", "center")
        if ui.button(pygame.Rect(box.centerx - 130, box.bottom - 56, 120, 34), "Yes", "danger"):
            self.confirm = None
            cb()
        elif ui.button(pygame.Rect(box.centerx + 10, box.bottom - 56, 120, 34), "No"):
            self.confirm = None

    # ---------------------------------------------------------------- helpers
    def section(self, title, x, y):
        self.ui.label(title.upper(), x, y, "xs", "muted")
        return y + 18

    def wrap(self, s, font, width):
        f = self.ui.fonts[font]
        lines, cur = [], ""
        for word in str(s).split():
            nxt = (cur + " " + word).strip()
            if f.size(nxt)[0] <= width:
                cur = nxt
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        return lines

    def icon(self, c, color, shape, r, hollow=False):
        col = CARD_RGB.get(color, self.ui.pal["muted"]) if color else self.ui.pal["muted"]
        w = 2 if hollow else 0
        x, y = c
        if shape == "circle":
            pygame.draw.circle(self.screen, col, c, r, w)
        elif shape == "rect_wide":
            pygame.draw.rect(self.screen, col, (x - r, y - int(r * 0.6), 2 * r, int(r * 1.2)), w)
        elif shape == "rect_tall":
            pygame.draw.rect(self.screen, col, (x - int(r * 0.6), y - r, int(r * 1.2), 2 * r), w)
        else:
            pygame.draw.rect(self.screen, col, (x - int(r * 0.8), y - int(r * 0.8), int(r * 1.6), int(r * 1.6)), w)
