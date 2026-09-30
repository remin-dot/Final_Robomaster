"""Wall centering + heading hold for one-cell moves, adapted from
nathon-aie/robomaster-assignment2-4x4_Dhai_8 (src/pid_controller.py: WallCenteringPID).

While the robot drives forward through a block, the two side Sharps keep it in the middle:
  both side walls       error = R - L            (equal gaps)
  left wall only        error = nominal - L      (nominal gap to that wall)
  right wall only       error = R - nominal
  no side wall          error = 0                (nothing to centre on - odometry line only)
(the reference repo splits each of these by "front wall / no front wall": the same error,
 8 cases in all). Error in mm -> sideways speed (m/s) through a PID with a 20 mm dead band;
heading error (deg) -> turn speed through a second PID (no dead band).
"""
import time


class PID:
    def __init__(self, kp, ki, kd, max_out, integral_limit, deadband=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_out, self.ilim, self.deadband = max_out, integral_limit, deadband
        self.reset()

    def reset(self):
        self.i, self.last, self.t = 0.0, None, None

    def __call__(self, err, dt=None):
        if abs(err) < self.deadband:
            return 0.0
        now = time.monotonic()
        if dt is None:
            dt = (now - self.t) if self.t is not None else 0.05
        self.t = now
        if dt > 0:
            self.i = max(-self.ilim, min(self.ilim, self.i + err * dt))
        d = (err - self.last) / dt if (self.last is not None and dt > 0) else 0.0
        self.last = err
        out = self.kp * err + self.ki * self.i + self.kd * d
        return max(-self.max_out, min(self.max_out, out))


class WallCentering:
    def __init__(self, cfg=None):
        c = cfg or {}
        self.nominal_mm = float(c.get("nominal_side_mm", 140))       # Sharp reading at a wall, robot centred
        self.wall_mm = float(c.get("side_wall_mm", 260))            # closer than this = a side wall
        self.tol_mm = float(c.get("tolerance_mm", 20))
        lat = c.get("lateral", {}) or {}
        yaw = c.get("yaw", {}) or {}
        self.lateral = PID(float(lat.get("kp", 0.0010)), float(lat.get("ki", 0.0001)), float(lat.get("kd", 0.0010)),
                           float(lat.get("max_speed", 0.08)), float(lat.get("integral_limit", 30)), self.tol_mm)
        self.yaw = PID(float(yaw.get("kp", 1.8)), float(yaw.get("ki", 0.05)), float(yaw.get("kd", 0.15)),
                       float(yaw.get("max_speed", 35)), float(yaw.get("integral_limit", 20)))
        # centring in place (no forward speed): firmer, little damping - the driving gains
        # above halve an offset only every ~2 s, too slow for a short stop at the centre
        cen = c.get("center", {}) or {}
        self.center = PID(float(cen.get("kp", 0.0025)), float(cen.get("ki", 0.0)), float(cen.get("kd", 0.0002)),
                          float(cen.get("max_speed", 0.08)), 30.0, self.tol_mm)

    def reset(self):
        self.lateral.reset()
        self.yaw.reset()
        self.center.reset()

    def lateral_error(self, left_mm, right_mm):
        """(error mm, case) - error > 0 = too close to the left: slide right."""
        has_l = left_mm is not None and left_mm < self.wall_mm
        has_r = right_mm is not None and right_mm < self.wall_mm
        if has_l and has_r:
            return right_mm - left_mm, "both walls"
        if has_l:
            return self.nominal_mm - left_mm, "left wall"
        if has_r:
            return right_mm - self.nominal_mm, "right wall"
        return 0.0, "no side wall"

    def speeds(self, left_mm, right_mm, heading_err_deg, dt=None):
        """(sideways m/s, turn deg/s in 'yaw increases' units, lateral error mm, case)."""
        err, case = self.lateral_error(left_mm, right_mm)
        return self.lateral(err, dt), self.yaw(heading_err_deg, dt), err, case
