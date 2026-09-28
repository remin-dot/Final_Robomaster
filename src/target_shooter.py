"""Aim the gimbal at a detected target with the camera and fire the blaster.

Closed loop on the camera only: the angular error of the target centre
(bearing / elevation from TargetDetector) is sent as a relative gimbal move
until the target sits on the boresight, then the range rule (<= 2 tiles) is
checked once more before firing.  Called from the mission thread.

Adapted from RoboFinal's auto-aim (src/robomaster_autoaim.py):
  * phases COARSE (first big move) -> FINE (corrections) -> LOCKED -> FIRE
  * fire only after `lock_frames` consecutive frames inside the tolerance
  * LED green while searching, red flash + attack sound when firing
  * every aim sample is logged (shown live on the panel, saved as CSV)
"""

import time

try:
    from robomaster import led as rm_led
    from robomaster import robot as rm_robot
except Exception:  # offline tests without the SDK
    rm_led = rm_robot = None


class TargetShooter:
    def __init__(self, ep_robot, panel, config, get_tof_mm=None):
        cfg = config.get("shooting", {}) or {}
        self.ep_robot = ep_robot
        self.panel = panel
        self.get_tof_mm = get_tof_mm or (lambda: None)
        self.fire_type = cfg.get("fire_type", "water")
        self.shots = int(cfg.get("shots_per_target", 1))
        self.tol_deg = float(cfg.get("aim_tolerance_deg", 1.5))
        self.pitch_offset = float(cfg.get("aim_pitch_offset_deg", -3.5))
        self.yaw_offset = float(cfg.get("aim_yaw_offset_deg", 0.0))
        self.latency = float((config.get("vision", {}) or {}).get("camera_latency_s", 0.2))
        self.yaw_sign = float(cfg.get("gimbal_yaw_sign", 1))
        self.pitch_sign = float(cfg.get("gimbal_pitch_sign", 1))
        self.max_iters = int(cfg.get("aim_max_iterations", 6))
        self.lock_frames = int(cfg.get("lock_frames", 2))
        self.armed = bool(cfg.get("enabled", True))  # False = dry run (RoboFinal "Blaster armed")
        self.dry_locked = set()                       # dry run: aim at each target only once
        panel.aim.configure(self.tol_deg, self.lock_frames)
        self._led(0, 255, 0, "on")

    # ------------------------------------------------------------------
    def _led(self, r, g, b, effect):
        try:
            eff = getattr(rm_led, "EFFECT_FLASH" if effect == "flash" else "EFFECT_ON", effect) if rm_led else effect
            self.ep_robot.led.set_led(comp="all", r=r, g=g, b=b, effect=eff)
        except Exception:
            pass

    def _fire_sound(self):
        try:
            sid = getattr(rm_robot, "SOUND_ID_ATTACK", None) or getattr(rm_robot, "SOUND_ID_SHOOT", 1)
            self.ep_robot.play_sound(sid)
        except Exception:
            pass

    def _find(self, kind):
        # frames arrive ~0.2 s late: one grabbed right after a move still shows the old view
        _, dets = self.panel.worker.wait_fresh(time.time() + self.latency, timeout=0.6 + self.latency)
        cands = [d for d in dets if d.is_target and d.kind == kind]
        # the one nearest the boresight (another card of the same kind may be in view)
        return min(cands, key=lambda d: abs(d.bearing_deg) + abs(d.elevation_deg)) if cands else None

    # ------------------------------------------------------------------
    def engage(self, kind, target_id=None):
        """Aim at the card of this kind ("blue circle") nearest the boresight and fire.
        target_id = the card on the map to mark as hit."""
        panel = self.panel
        tid = target_id or kind
        color = kind
        if kind not in panel.selected:       # never shoot a card that is not selected (-1)
            panel.log(f"{kind} is not selected - not shooting")
            return False
        if not self.armed and tid in self.dry_locked:
            return False
        aim = panel.aim
        gimbal = self.ep_robot.gimbal
        det = None
        locked_in_row = 0
        aim.begin(color)
        try:
            for i in range(self.max_iters + self.lock_frames):
                det = self._find(kind)
                if det is None:
                    aim.set_phase("LOST")
                    panel.log(f"lost {color} while aiming")
                    return False
                yaw_err = det.bearing_deg + self.yaw_offset
                pitch_err = det.elevation_deg + self.pitch_offset
                inside = abs(yaw_err) <= self.tol_deg and abs(pitch_err) <= self.tol_deg
                locked_in_row = locked_in_row + 1 if inside else 0
                aim.set_phase("LOCKED" if locked_in_row >= self.lock_frames else ("COARSE" if i == 0 else "FINE"))
                aim.sample(yaw_err, pitch_err, locked_in_row)
                if locked_in_row >= self.lock_frames:
                    break
                if inside:
                    continue  # confirm on the next frame, like RoboFinal's lock counter
                if i >= self.max_iters:
                    break
                try:
                    gimbal.move(pitch=self.pitch_sign * pitch_err, yaw=self.yaw_sign * yaw_err,
                                pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=1.5)
                except Exception as e:
                    panel.log(f"gimbal move failed: {e}")
                    return False
                time.sleep(0.08)

            if locked_in_row < self.lock_frames:
                aim.set_phase("TIMEOUT")
                panel.log(f"could not lock {color} (yaw err {det.bearing_deg:+.1f}, "
                          f"pitch err {det.elevation_deg + self.pitch_offset:+.1f} deg)")
                return False

            # range rule: pinhole estimate, confirmed by the gimbal ToF when it agrees
            dist = det.distance_m
            tof = self.get_tof_mm()
            if tof and dist and 0.6 * dist <= tof / 1000.0 <= 1.5 * dist:
                dist = tof / 1000.0
            max_m = panel.detector.max_shoot_m
            if dist is None or dist > max_m:
                aim.set_phase("OUT OF RANGE")
                panel.log(f"{color} is {dist or 0:.2f} m away (> {max_m:.2f} m) - not firing")
                return False

            if not self.armed:
                aim.set_phase("DRY RUN")
                self.dry_locked.add(tid)
                panel.log(f"DRY RUN: locked {det.label} at {dist:.2f} m, not firing")
                return False
            try:
                self.ep_robot.blaster.fire(fire_type=self.fire_type, times=self.shots)
            except Exception as e:
                panel.log(f"fire failed: {e}")
                return False
            aim.set_phase("FIRE")
            self._fire_sound()
            self._led(255, 0, 0, "flash")
            panel.map.mark_shot(tid)
            panel.log(f"FIRE -> {det.label} at {dist:.2f} m")
            time.sleep(0.5)
            return True
        finally:
            self._led(0, 255, 0, "on")
            aim.end()
