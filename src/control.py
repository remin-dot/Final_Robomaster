"""Connection + mission lifecycle behind the panel buttons.

Modelled on RoboFinal's final control panel (branch feature/final): a Connect
screen (robot / webcam / demo, Wi-Fi mode, round), then a header with
Start round N, Pause/Resume, Finish, Save, STOP and Disconnect, and an
Actions tab (Look around now, Aim & shoot here, Fire once, Blaster armed,
gimbal buttons).

Every slow action runs in a worker thread; the panel (main thread) only calls
these methods and reads `state`, so the video never freezes.

States: disconnected -> connecting -> idle -> running <-> paused -> done
        (idle/done -> busy while a one-off action such as "Look around" runs)
"""

import math
import os
import threading
import time
import traceback

import cv2
import numpy as np

from mission_panel import HEADING_DEG, BASE_DIR

SOURCES = ("robot", "webcam", "demo")
CONNECTIONS = (("ap", "Wi-Fi direct (AP)"), ("sta", "Router (STA)"), ("rndis", "USB"))


class MissionController:
    def __init__(self, config, panel, source="robot", connection=None, resolution=None,
                 demo_images=(), armed=None):
        self.config = config
        self.panel = panel
        self.source = source
        self.connection = connection or config.get("robot", {}).get("connection_type", "ap")
        self.resolution = resolution or config.get("vision", {}).get("resolution", "360p")
        # photos to loop in the demo; empty = a small simulated arena (cards at fixed places)
        self.demo_images = list(demo_images)
        shoot_cfg = config.get("shooting", {}) or {}
        self.armed = shoot_cfg.get("enabled", True) if armed is None else armed

        self.state = "disconnected"
        self.released = False      # gimbal asleep / no drive commands: robot can be moved by hand
        self.release_after_round = bool((config.get("movement", {}) or {}).get("release_after_round", True))
        self.error = ""
        self.ep_robot = None
        self.chassis = None
        self.shooter = None
        self.cap = None
        self.battery = None
        self._thread = None
        self._state_before_busy = "idle"
        self.auto_start = False
        panel.controller = self

    # ------------------------------------------------------------------ helpers
    @property
    def connected(self):
        return self.state not in ("disconnected", "connecting")

    @property
    def running(self):
        return self.state in ("running", "paused")

    @property
    def idle(self):
        return self.state in ("idle", "done")

    def can_start(self):
        return self.idle and self.source != "webcam"

    def _spawn(self, fn, *args):
        th = threading.Thread(target=fn, args=args, daemon=True)
        th.start()
        return th

    # ------------------------------------------------------------------ connect
    def connect(self):
        if self.state != "disconnected":
            return
        self.state = "connecting"
        self.error = ""
        self._spawn(self._connect)

    def _connect(self):
        try:
            {"robot": self._connect_robot, "webcam": self._connect_webcam, "demo": self._connect_demo}[self.source]()
            self.state = "idle"
            self.panel.log(f"connected: {self.source_label()}")
            if self.auto_start:
                self.start()
        except Exception as e:
            self.error = str(e) or e.__class__.__name__
            print(f"[connect] {self.error}")
            if not isinstance(e, RuntimeError):  # unexpected: keep the details for debugging
                traceback.print_exc()
            self._cleanup()
            self.state = "disconnected"

    def source_label(self):
        if self.source == "robot":
            return f"ROBOT ({self.connection.upper()})"
        return self.source.upper()

    def _connect_robot(self):
        try:
            from robomaster import robot
        except ImportError:
            raise RuntimeError("DJI robomaster SDK not installed. On a Mac run: "
                               "bash tools/macos/setup_robomaster_mac.sh")
        from chassis import ChassisController
        from target_shooter import TargetShooter

        ep = robot.Robot()
        try:
            if self.connection == "ap":
                ep.initialize(conn_type="ap", proto_type="udp")
            else:
                ep.initialize(conn_type=self.connection)
        except Exception as e:
            hint = {"ap": "Is this computer joined to the robot's Wi-Fi (RMEP-xxxxxx) and the robot switch on AP?",
                    "sta": "Are the robot and this computer on the same router network?",
                    "rndis": "Is the USB cable in the robot's intelligent controller micro-USB port?"}
            try:
                ep.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not reach the robot ({e}). {hint.get(self.connection, '')}")
        self.ep_robot = ep
        ep.camera.start_video_stream(display=False, resolution=self.resolution)
        try:
            ep.battery.sub_battery_info(freq=1, callback=lambda info: setattr(self, "battery", info))
        except Exception:
            pass

        ch = ChassisController(ep, self.config)
        ch.panel = self.panel
        self.shooter = TargetShooter(ep, self.panel, self.config, get_tof_mm=lambda: ch.current_tof_dist_mm)
        self.shooter.armed = self.armed
        ch.shooter = self.shooter
        ch.setup_csv_headers()
        ch.start_sensors()
        self.chassis = ch
        self.panel.telemetry = ch.telemetry
        self.panel.worker.set_source(lambda: ep.camera.read_cv2_image(strategy="newest", timeout=0.5))

    def _connect_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No webcam. Allow camera access for VS Code / Terminal in "
                               "System Settings > Privacy & Security > Camera, then try again.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap = cap

        def read():
            ok, frame = cap.read()
            return frame if ok else None

        self.panel.worker.set_source(read)

    # cards of the simulated demo arena: (kind, x_m, y_m) in map metres
    DEMO_CARDS = (("blue circle", 0.35, 2.15), ("red rect_wide", 2.15, 1.55), ("green square", 1.25, 0.35),
                  ("yellow rect_tall", 2.2, 2.9), ("red circle", 0.9, 1.2))

    def _connect_demo(self):
        clock = {"i": 0, "t": time.time()}
        photos = [f for f in (cv2.imread(p) for p in self.demo_images) if f is not None]

        def pace():  # like a 30 fps stream; never burst to catch up after a stall
            nxt = clock["t"] + 1 / 30.0
            now = time.time()
            if nxt < now - 0.1:
                nxt = now
            time.sleep(max(0.0, nxt - now))
            clock["t"] = nxt
            clock["i"] += 1

        def read():
            pace()
            if photos:
                return photos[(clock["i"] // 60) % len(photos)].copy()
            return self._demo_render()

        def telemetry():
            w = math.sin(time.time())
            return {"odom": "(+0.00, +0.00) m", "yaw": "+0.0 deg", "tof_mm": 650 + 250 * w,
                    "ir_left_cm": 14.0 + 4 * w, "ir_right_cm": 30.0, "ir_max_cm": 30.0, "ir_wall_cm": 16.9}

        self.panel.telemetry = telemetry
        self.panel.worker.set_source(read)

    def _demo_render(self, W=960, H=540):
        """What the gimbal camera would see in the demo arena from the robot's cell."""
        from target_vision import split_kind
        p = self.panel
        m = p.map
        img = np.empty((H, W, 3), np.uint8)
        img[: H // 2] = (214, 218, 222)                        # foam walls
        img[H // 2:] = (170, 176, 182)                         # floor
        cv2.line(img, (0, H // 2), (W, H // 2), (150, 155, 160), 2)
        f = p.detector.focal_px(W)
        rx, ry = m.cell_center_m(m.robot)
        cam = m.gimbal_abs
        colors = {"blue": (180, 80, 20), "red": (30, 30, 210), "green": (80, 150, 10), "yellow": (0, 210, 235)}
        for kind, tx, ty in sorted(self.DEMO_CARDS, key=lambda c: -math.hypot(c[1] - rx, c[2] - ry)):
            d = math.hypot(tx - rx, ty - ry)
            b = ((math.degrees(math.atan2(tx - rx, ty - ry)) - cam + 180) % 360) - 180
            if d < 0.2 or abs(b) > 44 or not m.visible_from(m.robot, tx, ty):
                continue
            color, shape = split_kind(kind)
            wm, hm = p.detector.plate[shape]
            w, h = f * wm / d, f * hm / d
            # card centre 12 cm below the camera; a camera pitched down (pitch < 0) sees it higher up
            elev = -math.degrees(math.atan2(0.12, d))
            pitch = p.detector.gimbal_pitch_deg
            cx, cy = W / 2 + f * math.tan(math.radians(b)), H / 2 - f * math.tan(math.radians(elev - pitch))
            cv2.line(img, (int(cx), int(cy)), (int(cx), int(cy + f * 0.10 / d)), (60, 60, 60), max(1, int(f * 0.008 / d)))
            if shape == "circle":
                cv2.circle(img, (int(cx), int(cy)), int(w / 2), colors[color], -1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                              colors[color], -1)
        return img

    def disconnect(self):
        if not self.connected:
            return
        self.state = "connecting"  # blocks the buttons while closing
        self._spawn(self._disconnect)

    def _disconnect(self):
        self.panel.abort.set()
        self.panel.paused.clear()
        if self._thread is not None:
            self._thread.join(timeout=20)
        self._cleanup()
        self.state = "disconnected"
        self.panel.log("disconnected")

    def _cleanup(self):
        self.panel.worker.set_source(None)
        self.panel.telemetry = lambda: {}
        ep, ch, cap = self.ep_robot, self.chassis, self.cap
        self.ep_robot = self.chassis = self.shooter = self.cap = None
        self.battery = None
        if ch is not None:
            try:
                ch.stop_sensors()
            except Exception:
                pass
        if ep is not None:
            for fn in (lambda: ep.camera.stop_video_stream(), lambda: ep.battery.unsub_battery_info(), ep.close):
                try:
                    fn()
                except Exception:
                    pass
        if cap is not None:
            cap.release()

    def shutdown(self):
        """Window closed: stop everything and release the robot."""
        self.panel.abort.set()
        self.panel.paused.clear()
        if self.running and self.ep_robot is not None:
            self._stop_motors()
        if self._thread is not None:
            self._thread.join(timeout=20)
        self._cleanup()

    # ------------------------------------------------------------------ mission
    def start(self):
        if not self.can_start():
            if self.source == "webcam":
                self.panel.log("webcam = camera only; connect the robot (or demo) to run a round")
            return
        p = self.panel
        p.abort.clear()
        p.paused.clear()
        p.prepare_round()
        if p.round_no >= 2 and p.prev_round is None and self.source == "robot":
            p.log("no round-1 file: round 2 explores again")
        self.state = "running"
        self._thread = self._spawn(self._run)

    def _run(self):
        p = self.panel
        try:
            self._wake()                     # after being carried back to the start
            p.start_round(p.round_no)
            if self.source == "demo":
                self._demo_mission()
            elif p.round_no >= 2 and p.prev_round is not None:
                self.chassis.navigate_to_targets(p.prev_round)
            else:
                self.chassis.explore_and_map_all()
        except Exception as e:
            traceback.print_exc()
            p.log(f"mission error: {e}")
        finally:
            if self.ep_robot is not None:
                self._stop_motors()
            p.finish_round()
            p.paused.clear()
            if self.state in ("running", "paused"):
                self.state = "done"
            if self.release_after_round and self.ep_robot is not None:
                self._release()               # carry it back to the start for the next round

    def pause(self):
        if self.state == "running":
            self.panel.paused.set()
            self.state = "paused"
            self.panel.log("paused")
            if self.ep_robot is not None:
                self._stop_motors()

    def resume(self):
        if self.state == "paused":
            self.panel.paused.clear()
            self.state = "running"
            self.panel.log("resumed")

    def finish(self):
        """End the round at the next safe point and save it."""
        if self.running:
            self.panel.abort.set()
            self.panel.paused.clear()
            self.panel.log("finishing round...")

    def stop(self):
        """Emergency stop: motors off now, mission ends and is saved."""
        self.panel.abort.set()
        self.panel.paused.clear()
        if self.ep_robot is not None:
            self._stop_motors()
        self.panel.log("STOP")

    # ------------------------------------------------------------------ release / wake
    def release(self):
        """Button: let go of the robot so it can be carried and the gimbal turned by hand."""
        if self.idle and self.ep_robot is not None:
            self._spawn(self._release)

    def wake(self):
        if self.idle and self.ep_robot is not None:
            self._spawn(self._wake)

    def _release(self):
        """Gimbal motors to sleep (turn it straight by hand) and no more drive commands.
        The SDK has no free-wheel command for the chassis: lift the robot to carry it."""
        ep = self.ep_robot
        if ep is None or self.released:
            return
        try:
            ep.chassis.drive_speed(x=0, y=0, z=0)     # last command: stand still, then nothing more
        except Exception:
            pass
        try:
            ep.gimbal.suspend()
        except Exception as e:
            self.panel.log(f"gimbal suspend failed: {e}")
        self.released = True
        self.panel.log("robot released: carry it to the start, straighten the gimbal by hand")

    def _wake(self):
        """Undo release: gimbal motors on again (the round then centres it on the chassis)."""
        ep = self.ep_robot
        if ep is None or not self.released:
            return
        try:
            ep.gimbal.resume()
            time.sleep(0.6)
            # a sleeping gimbal sags; level it before anything uses the camera
            ep.gimbal.recenter(pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=3)
            self.panel.detector.gimbal_pitch_deg = 0.0
        except Exception as e:
            self.panel.log(f"gimbal resume failed: {e}")
        self.released = False
        self.panel.log("robot awake")

    def _stop_motors(self):
        try:
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass
        try:
            self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass

    def save(self):
        self.panel.save_round_files()
        path = self.panel.round_file(self.panel.round_no)
        self.panel.log(f"saved -> {os.path.relpath(path, BASE_DIR)} (+ map png)")

    # ------------------------------------------------------------------ one-off actions
    def _action(self, name, fn):
        if not self.idle:
            return
        self._state_before_busy = self.state
        self.state = "busy"

        def run():
            try:
                self._wake()      # a one-off action needs the gimbal awake
                fn()
            except Exception as e:
                traceback.print_exc()
                self.panel.log(f"{name} failed: {e}")
            finally:
                self.state = self._state_before_busy

        self._spawn(run)

    def _needs_robot(self, what):
        if self.chassis is None:
            self.panel.log(f"{what}: needs the robot")
            return False
        return True

    def look_around(self):
        if not self._needs_robot("look around"):
            return

        def run():
            ch, m = self.chassis, self.panel.map
            ch._live_ctx = (m.robot, set(m.visited), m.nx - 1, m.ny - 1, m.heading)
            dists = ch.scan_surroundings_with_gimbal()
            m.mark_scan(m.robot, m.heading, dists)
            self.panel.log("look around done")

        self._action("look around", run)

    def aim_here(self):
        if not self._needs_robot("aim"):
            return

        def run():
            m = self.panel.map
            self.chassis._live_ctx = (m.robot, set(m.visited), m.nx - 1, m.ny - 1, m.heading)
            found = self.panel.observe_targets(m.robot, (HEADING_DEG.get(m.heading, 0) + m.gimbal_rel) % 360,
                                               tof_mm=self.chassis.current_tof_dist_mm)
            cands = [d for d in found if d.is_target and self.panel.should_shoot(d.target_id)]
            if not cands:
                self.panel.log("no selected target in view (Select tab)")
                return
            best = min(cands, key=lambda d: abs(d.bearing_deg))
            self.shooter.engage(best.kind, best.target_id)

        self._action("aim", run)

    def fire_once(self):
        if not self._needs_robot("fire"):
            return
        if not self.armed:
            self.panel.log("DRY RUN: blaster not armed")
            return
        try:
            self.ep_robot.blaster.fire(fire_type=self.shooter.fire_type, times=1)
            self.panel.log("fired once")
        except Exception as e:
            self.panel.log(f"fire failed: {e}")

    def set_armed(self, on):
        self.armed = bool(on)
        if self.shooter is not None:
            self.shooter.armed = self.armed
        self.panel.log("blaster ARMED" if on else "DRY RUN (blaster off)")

    def gimbal_move(self, yaw=0.0, pitch=0.0):
        if not self._needs_robot("gimbal"):
            return

        def run():
            self.ep_robot.gimbal.move(pitch=pitch, yaw=yaw, pitch_speed=90, yaw_speed=90).wait_for_completed(timeout=2)
            m = self.panel.map
            m.gimbal_rel = (m.gimbal_rel + yaw) % 360
            m.gimbal_abs = (HEADING_DEG.get(m.heading, 0) + m.gimbal_rel) % 360

        self._action("gimbal", run)

    def gimbal_center(self):
        if not self._needs_robot("gimbal"):
            return

        def run():
            self.ep_robot.gimbal.recenter(pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=3)
            m = self.panel.map
            m.gimbal_rel = 0.0
            m.gimbal_abs = HEADING_DEG.get(m.heading, 0)

        self._action("gimbal", run)

    def nudge_aim(self, pitch=0.0, yaw=0.0):
        """Aim trim (RoboFinal: shots too high -> Aim down)."""
        cfg = self.config.setdefault("shooting", {})
        cfg["aim_pitch_offset_deg"] = round(float(cfg.get("aim_pitch_offset_deg", -3.5)) + pitch, 2)
        cfg["aim_yaw_offset_deg"] = round(float(cfg.get("aim_yaw_offset_deg", 0.0)) + yaw, 2)
        if self.shooter is not None:
            self.shooter.pitch_offset = cfg["aim_pitch_offset_deg"]
            self.shooter.yaw_offset = cfg["aim_yaw_offset_deg"]

    def save_trim(self):
        from mission_panel import save_setting
        cfg = self.config.get("shooting", {})
        save_setting("shooting", "aim_pitch_offset_deg", cfg.get("aim_pitch_offset_deg", -3.5))
        save_setting("shooting", "aim_yaw_offset_deg", cfg.get("aim_yaw_offset_deg", 0.0))
        self.panel.log("aim trim saved to config/settings.yaml")

    def calibrate_distance(self, known_m=1.0):
        """Card straight ahead at `known_m`: fit the camera field of view so distances come out right."""
        def run():
            det = self.panel.detector
            dets, W = [], None
            for _ in range(5):
                frame, ds = self.panel.worker.wait_fresh(time.time(), timeout=0.6)
                if frame is not None:
                    W = frame.shape[1]
                dets += [d for d in ds if d.is_target]
            if not dets or W is None:
                self.panel.log("calibrate: put ONE card straight ahead at 1.0 m first")
                return
            best = min(dets, key=lambda d: abs(d.bearing_deg))
            size = det.plate.get(best.color)
            bw, bh = best.bbox[2], best.bbox[3]
            f_px = known_m / (0.5 * (size[0] / bw + size[1] / bh))
            hfov = math.degrees(2 * math.atan((W / 2.0) / f_px))
            det.hfov_deg = hfov
            self.config.setdefault("vision", {})["hfov_deg"] = round(hfov, 1)
            from mission_panel import save_setting
            save_setting("vision", "hfov_deg", round(hfov, 1))
            self.panel.log(f"calibrated with {best.label}: hfov {hfov:.1f} deg (saved)")

        self._action("calibrate", run)

    # ------------------------------------------------------------------ demo round
    def _demo_mission(self):
        """Simulated robot walk for the demo source (no hardware)."""
        p = self.panel
        route = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 2), (3, 2), (3, 1)]
        route = [c for c in route if c[0] < p.map.nx and c[1] < p.map.ny]
        for i, cell in enumerate(route):
            if not p.checkpoint():
                return
            nxt = route[min(i + 1, len(route) - 1)]
            d = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}.get((nxt[0] - cell[0], nxt[1] - cell[1]), 0)
            p.update_robot(cell, d, HEADING_DEG[d])
            p.map.mark_scan(cell, d, {"front": 900, "right": 250 if i % 2 else 900,
                                      "back": 900, "left": 250 if i % 3 == 0 else 900})
            for g in (0, 90, 180, 270):
                if not p.checkpoint():
                    return
                p.map.gimbal_rel = g
                p.map.gimbal_abs = (HEADING_DEG[d] + g) % 360
                time.sleep(0.1)  # let the camera show the new direction
                for f in p.observe_targets(cell, p.map.gimbal_abs, settle_ts=time.time()):
                    if f.in_range and p.should_shoot(f.target_id):
                        self._demo_aim(f)
                time.sleep(0.25)
            time.sleep(0.4)

    def _demo_aim(self, f):
        p = self.panel
        p.aim.configure(1.5, 2)
        p.aim.begin(f.kind)
        yaw, pitch = f.bearing_deg, f.elevation_deg
        for k in range(6):
            cnt = 0 if k < 4 else k - 3
            p.aim.set_phase("COARSE" if k == 0 else ("FINE" if cnt == 0 else "LOCKED"))
            p.aim.sample(yaw, pitch, cnt)
            yaw, pitch = yaw * 0.3, pitch * 0.3
            time.sleep(0.15)
        if self.armed:
            p.aim.set_phase("FIRE")
            p.map.mark_shot(f.target_id)
            p.log(f"FIRE at {f.label} (demo)")
        else:
            p.aim.set_phase("DRY RUN")
            p.log(f"DRY RUN: locked {f.label}, not firing")
        p.aim.end()
