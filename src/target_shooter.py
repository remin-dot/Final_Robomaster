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

import math
import time

try:
    from robomaster import led as rm_led
    from robomaster import robot as rm_robot
except Exception:  # offline tests without the SDK
    rm_led = rm_robot = None


def damped_aim_speed(error, previous, dt, fine, coarse_kp, coarse_max,
                     fine_kp, fine_max, deadband, accel):
    """One axis of the aim controller: deadband, two gains, then a slew limit."""
    if abs(error) <= deadband:
        wanted = 0.0
    else:
        kp, vmax = (fine_kp, fine_max) if fine else (coarse_kp, coarse_max)
        wanted = max(-vmax, min(vmax, kp * error))
    step = accel * max(0.01, min(dt, 0.10))
    return max(previous - step, min(previous + step, wanted))


class TargetShooter:
    def __init__(self, ep_robot, panel, config, get_tof_mm=None, gimbal_hist=None):
        cfg = config.get("shooting", {}) or {}
        self.ep_robot = ep_robot
        self.panel = panel
        self.get_tof_mm = get_tof_mm or (lambda: None)
        self.fire_type = cfg.get("fire_type", "water")
        self.shots = int(cfg.get("shots_per_target", 1))
        self.tol_deg = float(cfg.get("aim_tolerance_deg", 1.5))
        self.pitch_offset = float(cfg.get("aim_pitch_offset_deg", 0.0))
        self.yaw_offset = float(cfg.get("aim_yaw_offset_deg", 0.0))
        self.latency = float((config.get("vision", {}) or {}).get("camera_latency_s", 0.2))
        self.yaw_sign = float(cfg.get("gimbal_yaw_sign", 1))
        self.pitch_sign = float(cfg.get("gimbal_pitch_sign", 1))
        self.max_iters = int(cfg.get("aim_max_iterations", 6))
        self.lock_frames = int(cfg.get("lock_frames", 2))
        self.lost_retries = int(cfg.get("lost_retries", 3))
        self.barrel_below_m = float(cfg.get("barrel_below_camera_m", 0.03))
        self.confirm_fall = bool(cfg.get("confirm_fall", True))      # fire again until the card falls
        self.max_shots = int(cfg.get("max_shots_per_target", 3))
        self.fall_wait_s = float(cfg.get("fall_wait_s", 0.6))
        self.tol_max_deg = float(cfg.get("aim_tolerance_max_deg", 3.5))
        self.armed = bool(cfg.get("enabled", True))  # False = dry run (RoboFinal "Blaster armed")
        # tracking aim: the gimbal angle feed (t, pitch, yaw) from the chassis, and its tuning
        self.gimbal_hist = gimbal_hist or (lambda: None)
        self.aim_kp = float(cfg.get("aim_kp_per_s", 7.0))           # deg/s per deg of error
        self.aim_vmax = float(cfg.get("aim_max_dps", 240.0))
        self.aim_fine_kp = float(cfg.get("aim_fine_kp_per_s", 2.0))
        self.aim_fine_vmax = float(cfg.get("aim_fine_max_dps", 35.0))
        self.aim_fine_zone = float(cfg.get("aim_fine_zone_deg", 3.0))
        self.aim_deadband = float(cfg.get("aim_deadband_deg", 0.20))
        self.aim_accel = float(cfg.get("aim_accel_dps2", 300.0))
        self.aim_target_jump = float(cfg.get("aim_target_jump_deg", 5.0))
        # "snap" = aimbot: time-optimal braking curve straight onto the card (fast, no swing);
        # "damped" = the gentle two-gain controller (slower, for a shaky gimbal)
        self.aim_mode = str(cfg.get("aim_mode", "snap")).lower()
        self.aim_brake = float(cfg.get("aim_brake_dps2", 900.0))
        self.aim_response_s = float(cfg.get("aim_response_s", 0.05))
        self.aim_feed_age_s = float(cfg.get("aim_feed_age_s", 0.03))
        self.aim_min_dps = float(cfg.get("aim_min_dps", 3.0))
        self.aim_snap_vmax = float(cfg.get("aim_snap_max_dps", 240.0))
        self.aim_near_kp = float(cfg.get("aim_near_kp_per_s", 10.0))
        self.aim_flick = bool(cfg.get("aim_flick", True))
        self.aim_settle_frames = int(cfg.get("aim_settle_frames", self.lock_frames))
        self.aim_timeout_s = float(cfg.get("aim_timeout_s", 2.5))
        self.last_lock_dist = None                                  # for the trim fit (control.nudge_aim)
        # lock window = this share of the card's size in the picture (aim at the MIDDLE of it),
        # never tighter than aim_tolerance_min_deg, never wider than aim_tolerance_deg
        self.center_frac = float(cfg.get("aim_center_frac", 0.2))
        self.tol_min_deg = float(cfg.get("aim_tolerance_min_deg", 0.6))
        self.lost_s = float(cfg.get("aim_lost_s", 0.6))             # card gone this long while aiming -> give up
        self.learn_latency = bool(cfg.get("learn_camera_latency", True))
        # after a miss with the aim on the middle: aim a bit higher, then a bit lower (share of
        # the card height) - a hit then tells which way the beads go, kept for the rest of the run
        self.bracket_frac = float(cfg.get("miss_bracket_frac", 0.35))
        self.learned_pitch = 0.0                                    # trim learned from bracketed hits this run
        self._bracket = 0.0
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

    def _focal(self):
        return self.panel.detector.focal_px(self.panel.detector.last_frame_size[0])

    def _measure_shape(self, det, dist_m):
        """Shape from the real size (ToF distance x pixel size). Height decides (a slant only
        narrows the width); width rules out a square that is too wide. None = not clear."""
        fam = ("square", "rect_wide", "rect_tall")
        if det.shape not in fam or not det.bbox[3]:
            return None
        try:
            f = self._focal()
        except Exception:
            return None
        h = det.bbox[3] * dist_m / f
        w = det.bbox[2] * dist_m / f
        best = None
        for sh in fam:
            wm, hm = self.panel.detector.plate.get(sh, (0.07, 0.07))
            if w > 1.15 * wm:              # wider than this card can ever look
                continue
            err = abs(math.log(h / hm))
            if best is None or err < best[0]:
                best = (err, sh)
        if best is None or best[0] > 0.12:
            return None
        return best[1]

    def pitch_offset_at(self, dist_m):
        """Pitch aim offset (deg, + = aim above the card centre). The barrel sits below the
        camera (shooting.barrel_below_camera_m): with the camera on the card the pellet would
        pass that far below it, which matters most up close - so aim up by atan(offset / d)."""
        d = dist_m if dist_m and dist_m > 0.1 else 0.6
        return self.pitch_offset + self.learned_pitch + self._bracket + \
            math.degrees(math.atan2(self.barrel_below_m, d))

    def _card_angles(self, det):
        """(width, height) of the card in the picture, in degrees - None without a size."""
        try:
            f = self.panel.detector.focal_px(self.panel.detector.last_frame_size[0])
            return math.degrees(math.atan2(det.bbox[2], f)), math.degrees(math.atan2(det.bbox[3], f))
        except Exception:
            return None

    def _tolerances(self, det):
        """Lock window: a small share of the card's size in the picture, so the shot goes to
        the MIDDLE of the card (a card is 6-9 cm: at 1 m, 1.5 deg off is already 2.6 cm - the
        last run locked 1.6 deg off and missed 3 times). Returns (yaw tol, pitch tol, half the
        card height) in degrees."""
        a = self._card_angles(det)
        if a is None:
            return self.tol_deg, self.tol_deg, self.tol_deg
        ang_w, ang_h = a
        clamp = lambda v: max(self.tol_min_deg, min(self.tol_deg, v))
        return clamp(self.center_frac * ang_w), clamp(self.center_frac * ang_h), max(self.tol_deg, 0.45 * ang_h)

    def _still_standing(self, kind, det, frames=3):
        """After a shot: is the card still up where it was? Looks at a few fresh frames
        (after it had time to fall). Returns the detection if it still stands, None if
        it is gone (fallen) in most of them."""
        time.sleep(self.fall_wait_s)
        color = kind.split(" ", 1)[0]
        seen = []
        after = time.time() + self.latency           # the first frame taken after the wait
        for _ in range(frames):
            after, dets = self.panel.worker.wait_det(after, timeout=0.3 + self.latency)
            best = None
            fam = ("square", "rect_wide", "rect_tall")
            shape = kind.split(" ", 1)[1] if " " in kind else ""
            for d in dets:
                if d.color != color or not d.is_card:
                    continue
                if d.shape != shape and not (d.shape in fam and shape in fam):
                    continue
                # the same card = about where it was (a fallen card lies lower, on the floor)
                if abs(d.bearing_deg - det.bearing_deg) <= 6.0 and abs(d.elevation_deg - det.elevation_deg) <= 6.0:
                    if best is None or abs(d.bearing_deg) < abs(best.bearing_deg):
                        best = d
            seen.append(best)
        standing = [d for d in seen if d is not None]
        if len(standing) >= frames // 2 + 1:
            return next((d for d in standing if d.is_card), standing[0])
        return None

    # ------------------------------------------------------------------ tracking aim
    def _tracking_ok(self):
        """The gimbal angle feed is live (the chassis subscribes it at 50 Hz)."""
        h = self.gimbal_hist()
        return bool(h) and time.time() - h[-1][0] < 0.3

    def _gimbal_at(self, t):
        """Gimbal (pitch, yaw) at time t, from the angle feed (interpolated)."""
        h = list(self.gimbal_hist() or [])
        if not h:
            return None
        if t <= h[0][0]:
            return h[0][1], h[0][2]
        for (t0, p0, y0), (t1, p1, y1) in zip(h, h[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return p0 + k * (p1 - p0), y0 + k * (y1 - y0)
        return h[-1][1], h[-1][2]

    def _gimbal_speed(self, t, win=0.06):
        """Gimbal turn rate (deg/s) around time t, from the angle feed."""
        a, b = self._gimbal_at(t - win), self._gimbal_at(t + win)
        if a is None or b is None:
            return 0.0
        return math.hypot(b[0] - a[0], b[1] - a[1]) / (2 * win)

    def _max_speed(self, t0, t1):
        """Fastest gimbal turn rate (deg/s) between t0 and t1 - the delay is never known to the
        millisecond, so "the gimbal was still when this frame was taken" is checked over a
        window around the moment, not at one instant."""
        h = [x for x in list(self.gimbal_hist() or []) if t0 - 0.03 <= x[0] <= t1 + 0.03]
        best = 0.0
        for (ta, pa, ya), (tb, pb, yb) in zip(h, h[1:]):
            if tb > ta:
                best = max(best, math.hypot(pb - pa, yb - ya) / (tb - ta))
        return best

    def _pick(self, dets, kind, near=None, gate=None):
        """The card of this kind (or its square / rect look-alike) nearest where the aim expects
        it (near = image bearing, elevation; default the boresight). gate: further than this
        (deg) from `near` is another card, not this one -> None (the last run swung between
        two yellow cards 41 deg apart)."""
        color, shape = kind.split(" ", 1) if " " in kind else (kind, "")
        fam = ("square", "rect_wide", "rect_tall")
        cands = [d for d in dets if d.is_card and d.kind == kind]
        if not cands and shape in fam:
            cands = [d for d in dets if d.is_card and d.color == color and d.shape in fam]
        if not cands:
            return None
        ref = near or (0.0, 0.0)
        dist = lambda d: math.hypot(d.bearing_deg - ref[0], d.elevation_deg - ref[1])
        best = min(cands, key=dist)
        if gate is not None and dist(best) > gate:
            return None
        return best

    def _learn_latency(self, samples):
        """The camera delay from this aim: the card does not move, so with the right delay
        (gimbal angle when the frame was taken) + (card in the frame) stays the same while the
        gimbal turns. A delay set too short makes the aim chase its own old frames and swing
        (the last run: +18 / -28 deg back and forth for 2.5 s)."""
        if not self.learn_latency or len(samples) < 5:
            return
        def spread(lat):
            vals = []
            for ts, b, e in samples:
                g = self._gimbal_at(ts - lat)
                if g is None:
                    return None
                vals.append((g[1] + self.yaw_sign * b, g[0] + self.pitch_sign * e))
            my = sum(v[0] for v in vals) / len(vals)
            mp = sum(v[1] for v in vals) / len(vals)
            return sum((v[0] - my) ** 2 + (v[1] - mp) ** 2 for v in vals) / len(vals)
        # only worth it when the gimbal really turned while those frames were taken
        g0, g1 = self._gimbal_at(samples[0][0] - self.latency), self._gimbal_at(samples[-1][0] - self.latency)
        if g0 is None or g1 is None or abs(g1[1] - g0[1]) + abs(g1[0] - g0[0]) < 4.0:
            return
        cands = [(spread(l / 100.0), l / 100.0) for l in range(5, 61, 1)]
        cands = [c for c in cands if c[0] is not None]
        if not cands:
            return
        best_v, best_l = min(cands)
        cur = spread(self.latency)
        if cur is None or best_v > 0.6 * cur:
            return                                     # not clearly better: keep what we have
        old = self.latency
        self.latency = round(0.3 * old + 0.7 * best_l, 3)
        try:
            self.panel.camera_latency_s = self.latency  # the looks wait for the right frame too
        except Exception:
            pass
        if abs(self.latency - old) >= 0.02:
            self.panel.log(f"camera delay measured {best_l:.2f} s (was {old:.2f}) - aim now uses {self.latency:.2f} s")

    def _aim_track(self, kind, tid, expect=None):
        """Smooth aim: every new frame gives the card's ABSOLUTE gimbal angle (the gimbal angle
        at the moment that frame was taken + where the card is in it), so the camera delay
        does not matter; the gimbal is driven continuously towards the median of the last 3
        of those (speed = kp x error, from its own 50 Hz angles) - no step / wait / look again.

          * sticks to ONE card: a detection far from where that card must be is another card
          * only frames taken while the gimbal turned slowly move the target (a frame from a
            fast turn is off by speed x delay error); the delay itself is measured (learned)
          * locks when frames taken with the gimbal still show the card's MIDDLE inside a small
            window, lock_frames times in a row; card gone for aim_lost_s -> give up at once
          * a blob that stays put in the picture while the gimbal turned (both frames taken with
            the gimbal still, so no delay trick) is part of the robot / lens: dropped

        expect = (image bearing, elevation) of the card when the aim starts (from the look that
        found it). Returns the locking detection, or None."""
        panel, aim, gimbal = self.panel, self.panel.aim, self.ep_robot.gimbal
        worker = panel.worker
        t_end = time.time() + self.aim_timeout_s
        targets = []                      # (abs pitch, abs yaw)
        samples = []                      # (frame time, bearing, elevation) for the delay fit
        still_ref, static_hits = None, 0  # (gimbal p, y, bearing, elevation) of a frame taken standing still
        last_ts, locked, i, misses = 0.0, 0, 0, 0
        last_seen = time.time()
        lock_abs = []                     # card's absolute angle from each frame counted for the lock
        still_targets = []                # card's absolute angle from frames taken with the gimbal still
        flick_ok_t = 0.0                  # last frame confirming the card near the flick spot
        self._lock_target = None
        prev_cmd_p, prev_cmd_y, cmd_ts, stall_since = 0.0, 0.0, time.time(), None
        det = None
        slow_dps = 40.0                   # frames taken turning faster than this do not move the target
        try:
            while time.time() < t_end:
                if not panel.checkpoint():
                    return None
                ts, dets = worker.wait_det(last_ts, timeout=0.06)
                now = time.time()
                g_now = self._gimbal_at(now)
                if ts > last_ts:
                    last_ts = ts
                    exp_t = ts - self.latency                  # when the picture was really taken
                    g = self._gimbal_at(exp_t) or g_now
                    spd = self._max_speed(exp_t - 0.15, exp_t + 0.05)     # delay may be off by that
                    near, gate = expect, None
                    if targets and g:
                        tp = sorted(t[0] for t in targets)[len(targets) // 2]
                        ty = sorted(t[1] for t in targets)[len(targets) // 2]
                        near = (self.yaw_sign * (ty - g[1]) - self.yaw_offset,
                                self.pitch_sign * (tp - g[0]) - self.pitch_offset_at(det.distance_m if det else None))
                        gate = 5.0 + 0.15 * spd                # a wrong delay shifts it by speed x error
                    elif expect is not None:
                        gate = 12.0
                    d = self._pick(dets, kind, near, gate)
                    # fixed in the picture? compare two frames both taken with the gimbal still
                    # (under 4 deg/s for the 0.3 s before): the camera delay cannot fake that.
                    # Checked on the blob at the SAME picture spot (the aim above drops it as
                    # "not where the card must be" once the gimbal has turned)
                    if still_ref is not None and d is None and g is not None and \
                            abs(g[1] - still_ref[1]) + abs(g[0] - still_ref[0]) > 6.0 and \
                            self._pick(dets, kind, (still_ref[2], still_ref[3]), 1.0) is not None:
                        last_seen = now          # the blob stayed put in the picture: check it, not "lost"
                    if g is not None and self._max_speed(exp_t - 0.3, exp_t + 0.05) < 4.0:
                        if still_ref is None:
                            if d is not None:
                                still_ref = (g[0], g[1], d.bearing_deg, d.elevation_deg)
                        elif abs(g[1] - still_ref[1]) + abs(g[0] - still_ref[0]) > 6.0:
                            same = self._pick(dets, kind, (still_ref[2], still_ref[3]), 1.0)
                            if same is not None:
                                static_hits += 1
                                if static_hits >= 2:
                                    self._drop_static(same, kind, tid)
                                    return None
                            elif d is not None:
                                still_ref, static_hits = (g[0], g[1], d.bearing_deg, d.elevation_deg), 0
                    if d is None:
                        misses += 1
                        if not targets and misses > 12:        # ~0.4 s of nothing at the start
                            aim.set_phase("LOST")
                            panel.log(f"lost {kind} while aiming")
                            return None
                        if targets and now - last_seen > self.lost_s:
                            aim.set_phase("LOST")
                            panel.log(f"lost {kind} while aiming (gone {now - last_seen:.1f} s)")
                            return None
                    elif g is not None:
                        last_seen, det = now, d
                        pe = d.elevation_deg + self.pitch_offset_at(d.distance_m)
                        ye = d.bearing_deg + self.yaw_offset
                        candidate = (g[0] + self.pitch_sign * pe, g[1] + self.yaw_sign * ye)
                        if not targets or spd < slow_dps:
                            if targets:
                                tp0 = sorted(t[0] for t in targets)[len(targets) // 2]
                                ty0 = sorted(t[1] for t in targets)[len(targets) // 2]
                                jumped = math.hypot(candidate[0] - tp0, candidate[1] - ty0) > self.aim_target_jump
                            else:
                                jumped = False
                            if not jumped:
                                targets.append(candidate)
                                targets = targets[-3:]
                        samples.append((ts, d.bearing_deg, d.elevation_deg))
                        if self._max_speed(exp_t - 0.15, exp_t + 0.05) < 4.0:
                            # a frame taken standing still: where the card is, exactly (no delay error)
                            still_targets = (still_targets + [candidate])[-3:]
                        if still_targets:
                            # a frame taken with the gimbal already near that spot, showing the card
                            # just there: confirms the flick (a blob fixed in the picture moves with
                            # the gimbal and fails this; a card that is gone gives no such frame)
                            sp0 = sorted(t[0] for t in still_targets)[len(still_targets) // 2]
                            sy0 = sorted(t[1] for t in still_targets)[len(still_targets) // 2]
                            ty_, tp_, _ = self._tolerances(d)
                            if abs(g[1] - sy0) + abs(g[0] - sp0) <= max(2.0, 2 * (ty_ + tp_)) and \
                                    math.hypot(candidate[0] - sp0, candidate[1] - sy0) <= max(1.0, ty_):
                                flick_ok_t = now
                        tol_y, tol_p, half_h = self._tolerances(d)
                        tp_med = sorted(t[0] for t in targets)[len(targets) // 2]
                        ty_med = sorted(t[1] for t in targets)[len(targets) // 2]
                        # the frame was taken with the gimbal (nearly) still, on the target
                        settled = spd < 8.0 and abs(g[1] - ty_med) <= tol_y and \
                            (abs(g[0] - tp_med) <= tol_p or (stall_since and time.time() - stall_since > 0.3))
                        inside = abs(ye) <= tol_y and (abs(pe) <= tol_p or
                                                       (stall_since is not None and abs(pe) <= half_h))
                        locked = locked + 1 if (settled and inside) else 0
                        cur_abs = (g[0] + self.pitch_sign * pe, g[1] + self.yaw_sign * ye)
                        lock_abs = (lock_abs + [cur_abs])[-locked:] if locked else []
                        aim.set_phase("LOCKED" if locked >= self.lock_frames else ("COARSE" if i == 0 else "FINE"))
                        aim.sample(ye, pe, locked)
                        i += 1
                        if locked >= self.lock_frames:
                            self.last_lock_dist = d.distance_m
                            # frames taken standing still: where the middle of the card is, exactly
                            self._lock_target = (sum(a[0] for a in lock_abs) / len(lock_abs),
                                                 sum(a[1] for a in lock_abs) / len(lock_abs))
                            self._learn_latency(samples)
                            return d
                # flick: the gimbal has arrived on where STILL frames put the card - no need to wait
                # for more frames (they would only say the same, 0.2-0.4 s later)
                if self.aim_mode == "snap" and self.aim_flick and still_targets and det is not None \
                        and g_now is not None and now - flick_ok_t < 0.25:
                    sp = sorted(t[0] for t in still_targets)[len(still_targets) // 2]
                    sy = sorted(t[1] for t in still_targets)[len(still_targets) // 2]
                    tol_y, tol_p, _ = self._tolerances(det)
                    if abs(g_now[1] - sy) <= tol_y and abs(g_now[0] - sp) <= tol_p and \
                            self._max_speed(now - 0.06, now) < 20.0 and \
                            math.hypot(sp - (targets[-1][0] if targets else sp), sy - (targets[-1][1] if targets else sy)) < 1.5:
                        self.last_lock_dist = det.distance_m
                        self._lock_target = (sp, sy)
                        aim.set_phase("LOCKED")
                        self._learn_latency(samples)
                        return det
                # drive towards the target angle (continuous, from the live gimbal angle)
                if targets and g_now is not None:
                    tp_med = sorted(t[0] for t in targets)[len(targets) // 2]
                    ty_med = sorted(t[1] for t in targets)[len(targets) // 2]
                    ey, ep = ty_med - g_now[1], tp_med - g_now[0]
                    fine = max(abs(ey), abs(ep)) <= self.aim_fine_zone
                    now_cmd = time.time()
                    dt_cmd = now_cmd - cmd_ts
                    if self.aim_mode == "snap":
                        vy, vp = self._snap_speeds(ey, ep)
                    else:
                        vy = damped_aim_speed(ey, prev_cmd_y, dt_cmd, fine, self.aim_kp, self.aim_vmax,
                                              self.aim_fine_kp, self.aim_fine_vmax,
                                              self.aim_deadband, self.aim_accel)
                        vp = damped_aim_speed(ep, prev_cmd_p, dt_cmd, fine, self.aim_kp, self.aim_vmax,
                                              self.aim_fine_kp, self.aim_fine_vmax,
                                              self.aim_deadband, self.aim_accel)
                    cmd_ts = now_cmd
                    # pitch at its mechanical limit: commanded but not moving -> stalled
                    if abs(vp) > 5 and abs(prev_cmd_p) > 5:
                        h = list(self.gimbal_hist() or [])[-8:]
                        if len(h) >= 2 and abs(h[-1][1] - h[0][1]) < 0.3:
                            stall_since = stall_since or time.time()
                        else:
                            stall_since = None
                    prev_cmd_p, prev_cmd_y = vp, vy
                    try:
                        gimbal.drive_speed(pitch_speed=vp, yaw_speed=vy)
                    except Exception as e:
                        panel.log(f"gimbal speed failed: {e}")
                        return None
            aim.set_phase("TIMEOUT")
            self._learn_latency(samples)
            if det is not None:
                panel.log(f"could not lock {kind} in {self.aim_timeout_s:.1f} s "
                          f"(yaw err {det.bearing_deg + self.yaw_offset:+.1f} deg)")
            else:
                panel.log(f"lost {kind} while aiming")
            return None
        finally:
            try:
                gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass

    def _snap_speeds(self, ey, ep):
        """Aimbot drive: time-optimal onto the target angle - full speed, then braking so it
        arrives with no swing past (v = sqrt(2 x brake x error)); once the gimbal will coast
        the rest of the way (its own speed x (feed age + response)), stop pushing.
        Returns (yaw speed, pitch speed) in deg/s."""
        h = list(self.gimbal_hist() or [])[-3:]
        if len(h) >= 2 and h[-1][0] > h[0][0]:
            dt = h[-1][0] - h[0][0]
            mp, my = (h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt
        else:
            mp = my = 0.0
        coast = self.aim_feed_age_s + self.aim_response_s

        def axis(e, meas):
            if abs(e) <= self.aim_deadband:
                return 0.0
            if meas * e > 0 and abs(e) < abs(meas) * coast:
                return 0.0
            # far: the braking curve; close: a proportional pull (the curve is too steep near 0 -
            # with the feed delay it would chatter back and forth over the last degree)
            v = min(self.aim_snap_vmax, math.sqrt(2.0 * self.aim_brake * abs(e)), self.aim_near_kp * abs(e))
            return math.copysign(max(v, self.aim_min_dps), e)
        return axis(ey, my), axis(ep, mp)

    def _onto_middle(self):
        """Right before the trigger: the lock frames were taken with the gimbal still, so they
        say exactly where the card's middle is - the last small step onto it (inside the lock
        window is up to ~1 deg off; this takes it to ~0)."""
        fin = getattr(self, "_lock_target", None)
        g = self._gimbal_at(time.time()) if self._tracking_ok() else None
        if not fin or not g:
            return
        dp, dy = fin[0] - g[0], fin[1] - g[1]
        if 0.1 < abs(dp) + abs(dy) < 3.0:
            try:
                self.ep_robot.gimbal.move(pitch=dp, yaw=dy, pitch_speed=90,
                                          yaw_speed=90).wait_for_completed(timeout=0.6)
            except Exception:
                pass
        # never pull the trigger on a moving / ringing gimbal: the bead leaves the way the
        # barrel points at that instant (last run fired 0.05 s after this step: 3 misses in a
        # row on cards the aim log shows centred)
        t_end = time.time() + 0.35
        while time.time() < t_end and self._max_speed(time.time() - 0.08, time.time()) > 5.0:   # 1 feed step = 3 deg/s
            time.sleep(0.02)

    def _drop_static(self, d, kind, tid):
        """A blob fixed in the picture: not a card. Only a spot near the picture's edge or
        bottom (robot parts, lens glare) is ignored from then on - never the middle, where
        real cards are aimed at."""
        panel = self.panel
        edge = abs(d.bearing_deg) > 0.25 * panel.detector.hfov_deg or d.elevation_deg < -12.0
        if edge:
            panel.detector.add_static_spot(d.color, d.bearing_deg, d.elevation_deg)
        if tid in panel.map.targets:
            panel.map.remove_target(tid)
        panel.aim.set_phase("NOT A CARD")
        panel.log(f"{kind} moves with the camera (part of the robot / a reflection) - not a card, dropped"
                  + ("" if edge else " (that spot is not blacklisted)"))

    def _find(self, kind):
        # frames arrive ~0.2 s late: one grabbed right after a move still shows the old view
        _, dets = self.panel.worker.wait_fresh(time.time() + self.latency, timeout=0.6 + self.latency)
        cands = [d for d in dets if d.is_card and d.kind == kind]
        if not cands:
            # the map knows the true shape; at a slant the raw frame may call it the look-alike
            # (wide rect <-> square <-> tall rect): same colour, rectangle family
            color, shape = kind.split(" ", 1) if " " in kind else (kind, "")
            fam = ("square", "rect_wide", "rect_tall")
            if shape in fam:
                cands = [d for d in dets if d.is_card and d.color == color and d.shape in fam]
        # the one nearest the boresight (another card of the same kind may be in view)
        return min(cands, key=lambda d: abs(d.bearing_deg) + abs(d.elevation_deg)) if cands else None

    # ------------------------------------------------------------------
    def engage(self, kind, target_id=None, expect=None):
        """Aim at the card of this kind ("blue circle") and fire. target_id = the card on the map
        to mark as hit; expect = (image bearing, elevation) where the look saw it (another card
        of the same colour may be in view), default the boresight."""
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
            if self._tracking_ok():
                # smooth closed loop on the gimbal's own angles (see _aim_track)
                self._bracket = 0.0
                det = self._aim_track(kind, tid, expect)
                if det is None:
                    return False
            else:
                # no gimbal angle feed (demo / old SDK): step, wait, look again
                last_move = (0.0, 0.0)
                prev_pitch_err = None
                stalled = 0
                for i in range(self.max_iters + self.lock_frames):
                    det = self._find(kind)
                    for _ in range(self.lost_retries):   # one missed frame (blur after a big move) is not "lost"
                        if det is not None:
                            break
                        time.sleep(0.1)
                        det = self._find(kind)
                    if det is None and any(last_move):
                        # the last move lost it (overshoot / blur): go back to where it was seen, look again
                        try:
                            gimbal.move(pitch=-last_move[0], yaw=-last_move[1], pitch_speed=120,
                                        yaw_speed=120).wait_for_completed(timeout=1.5)
                        except Exception:
                            pass
                        last_move = (0.0, 0.0)
                        time.sleep(0.1)
                        det = self._find(kind)
                    if det is None:
                        aim.set_phase("LOST")
                        panel.log(f"lost {color} while aiming")
                        return False
                    yaw_err = det.bearing_deg + self.yaw_offset
                    pitch_err = det.elevation_deg + self.pitch_offset_at(det.distance_m)
                    tol_y, tol_p, half_h = self._tolerances(det)
                    inside = abs(yaw_err) <= tol_y and abs(pitch_err) <= tol_p
                    # gimbal at its pitch limit (the error does not change after a pitch move) while the
                    # barrel line is still on the card: that is a hit, fire rather than time out
                    if prev_pitch_err is not None and abs(pitch_err - prev_pitch_err) < 0.4 and abs(last_move[0]) > 0.5:
                        stalled += 1
                    else:
                        stalled = 0
                    if not inside and stalled >= 1 and abs(yaw_err) <= tol_y and abs(pitch_err) <= half_h:
                        inside = True
                        panel.log(f"{color}: gimbal pitch at its limit, {pitch_err:+.1f} deg is still on the card")
                    prev_pitch_err = pitch_err
                    locked_in_row = locked_in_row + 1 if inside else 0
                    aim.set_phase("LOCKED" if locked_in_row >= self.lock_frames else ("COARSE" if i == 0 else "FINE"))
                    aim.sample(yaw_err, pitch_err, locked_in_row)
                    if locked_in_row >= self.lock_frames:
                        break
                    if inside:
                        last_move = (0.0, 0.0)
                        continue  # confirm on the next frame, like RoboFinal's lock counter
                    if i >= self.max_iters:
                        break
                    try:
                        gimbal.move(pitch=self.pitch_sign * pitch_err, yaw=self.yaw_sign * yaw_err,
                                    pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=1.5)
                        last_move = (self.pitch_sign * pitch_err, self.yaw_sign * yaw_err)
                    except Exception as e:
                        panel.log(f"gimbal move failed: {e}")
                        return False
                    time.sleep(0.08)

                if locked_in_row < self.lock_frames:
                    aim.set_phase("TIMEOUT")
                    panel.log(f"could not lock {color} (yaw err {yaw_err:+.1f}, pitch err {pitch_err:+.1f} deg)")
                    return False

            # range rule: pinhole estimate, confirmed by the gimbal ToF when it agrees
            dist = det.distance_m
            tof = self.get_tof_mm()
            tof_ok = bool(tof and dist and 0.6 * dist <= tof / 1000.0 <= 1.5 * dist)
            if tof_ok:
                dist = tof / 1000.0
                # the ToF is on the card now: its real size tells square from rectangle
                # (height does not change at a slant: 6 cm wide rect, 7 cm square, 9 cm tall rect)
                measured = self._measure_shape(det, dist)
                if measured:
                    old_tid = tid
                    tid = panel.map.set_measured_shape(tid, measured) or tid
                    new_kind = f"{det.color} {measured}"
                    if new_kind != kind:
                        panel.log(f"measured with the ToF: {old_tid} is really {new_kind} "
                                  f"({det.bbox[3] * dist / self._focal() * 100:.1f} cm tall)")
                        if new_kind not in panel.selected:
                            aim.set_phase("NOT SELECTED")
                            panel.log(f"{new_kind} is not selected - not shooting")
                            return False
                        kind = new_kind
            if not panel.fire_ok(tid):
                t = panel.map.targets.get(tid, {})
                aim.set_phase("SHAPE?")
                panel.log(f"{tid}: square or rectangle not sure (seen {t.get('best_view_deg', 90):.0f} deg "
                          f"off face-on, no ToF size) - not shooting yet")
                return False
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
            # fire, then watch the card: still standing -> correct the aim and fire again,
            # until it falls (or max_shots_per_target). Fallen -> stop, next target.
            fired = 0
            while True:
                self._onto_middle()
                try:
                    self.ep_robot.blaster.fire(fire_type=self.fire_type, times=self.shots)
                except Exception as e:
                    panel.log(f"fire failed: {e}")
                    if fired:
                        panel.map.mark_shot(tid)
                    return bool(fired)
                fired += 1
                aim.set_phase("FIRE")
                self._fire_sound()
                self._led(255, 0, 0, "flash")
                panel.log(f"FIRE {fired} -> {det.label} at {dist:.2f} m")
                if not self.confirm_fall:
                    panel.map.mark_shot(tid)
                    time.sleep(0.5)
                    return True
                still = self._still_standing(kind, det)   # the same whole card, never a "?" blob
                if still is None:
                    panel.map.mark_shot(tid)
                    aim.set_phase("DOWN")
                    panel.log(f"{det.label} is down after {fired} shot(s) - next target")
                    if self._bracket:
                        # the middle missed, this offset hit: the beads go that way - keep half of it
                        self.learned_pitch += 0.5 * self._bracket
                        panel.log(f"hit with the aim {self._bracket:+.1f} deg off the middle - aim trim "
                                  f"{self.learned_pitch:+.1f} deg for the rest of the run "
                                  f"(press Aim trim {'Up' if self._bracket > 0 else 'Down'} + Save to keep it)")
                    self._bracket = 0.0
                    return True
                if fired >= self.max_shots:
                    self._bracket = 0.0
                    panel.map.mark_shot(tid)
                    with panel.map.lock:
                        if tid in panel.map.targets:
                            panel.map.targets[tid]["missed"] = True     # mop-up may try it once more
                    aim.set_phase("STANDING")
                    panel.log(f"{det.label} still standing after {fired} shots - moving on")
                    return True
                # still there: the last bead missed. With the aim on the middle that is the trim
                # (the beads go a bit high / low): aim a little higher, then a little lower
                det = still
                yaw_err = det.bearing_deg + self.yaw_offset
                pitch_err = det.elevation_deg + self.pitch_offset_at(det.distance_m or dist)
                aim.sample(yaw_err, pitch_err, 0)
                tol_y, tol_p, _ = self._tolerances(det)
                centred = abs(yaw_err) <= tol_y and abs(pitch_err) <= tol_p
                a = self._card_angles(det)
                step = self.bracket_frac * a[1] if (a and centred and self.bracket_frac > 0) else 0.0
                self._bracket = {1: step, 2: -step}.get(fired, 0.0)
                panel.log(f"{det.label} still standing (miss) - "
                          + (f"aim was on the middle: next one {self._bracket:+.1f} deg "
                             f"{'higher' if self._bracket > 0 else 'lower'}" if self._bracket else
                             f"re-aim {yaw_err:+.1f} / {pitch_err:+.1f} deg"))
                if self._tracking_ok():
                    again = self._aim_track(kind, tid, (det.bearing_deg, det.elevation_deg))
                    if again is not None:
                        det = again
                else:
                    pitch_err = det.elevation_deg + self.pitch_offset_at(det.distance_m or dist)
                    try:
                        gimbal.move(pitch=self.pitch_sign * pitch_err, yaw=self.yaw_sign * yaw_err,
                                    pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=1.5)
                    except Exception:
                        pass
                    time.sleep(0.1)
        finally:
            self._bracket = 0.0
            self._led(0, 255, 0, "on")
            aim.end()
