"""IR sensors on the RoboMaster sensor adaptor - two groups:

  * 2 Sharp GP2Y0A41 distance sensors on the SIDES (analog, 4-30 cm)   [sharp_ir]
        left  = adaptor 1, port 2        right = adaptor 2, port 1
    calibrated with the pygame panel (`python3 src/sharp_calibrate.py`) at 5, 10,
    15, 20 cm; used to keep off the side walls.
  * 2 IR obstacle modules on the FRONT CORNERS, ~45 deg out (on/off)   [ir_corner]
        left  = adaptor 1, port 1        right = adaptor 2, port 2
    no calibration: their screw sets the switching distance (~10 cm); output LOW =
    obstacle. Used to stop / slide away before a corner touches a wall.

Sharp calibration modes (recorded points):
  * analog   (Sharp GP2Y0A41 and analog IR modules): the ADC value changes with the
             distance -> distance from the recorded points (log-log interpolation,
             a power-law fit outside them)
  * digital  (IR obstacle modules with a threshold screw, e.g. FC-51): the output
             only switches when something is closer than the screw setting -> near /
             not near from the learnt switching level
Without a calibration the RoboFinal Sharp formula is used:
    Vo = M / (L + 0.42) + C  ->  L = M / (Vo - C) - 0.42   (GP2Y0A41, 4-30 cm)

Ports that never answer are switched off instead of blocking every poll for 3 s
(the SDK "cmdid:0xf0 timeout" spam). Polled in a thread (~10 Hz), logged to CSV.
"""

import csv
import math
import threading
import time

ADC_MAX = 1023.0
SYSTEM_VOLTAGE = 3.3
SIDES = ("left", "right")


# =============================================================================
# calibration
# =============================================================================
def fit_calibration(points, min_cm=4.0, max_cm=30.0):
    """points = [(cm, raw), ...] recorded at known distances -> calibration dict.

    mode "analog":  raw changes smoothly with distance (a curve through the points)
    mode "digital": raw sits at two levels (switch module): near / far threshold
    mode "flat":    raw does not change - wrong port, sensor not facing the board,
                    or a switch module whose screw is not set to trigger at 5 cm
    """
    pts = sorted((float(c), float(r)) for c, r in points if r is not None)
    if len(pts) < 2:
        return {"mode": "none", "points": [list(p) for p in pts]}
    raws = [r for _, r in pts]
    lo, hi = min(raws), max(raws)
    out = {"points": [[round(c, 1), round(r, 1)] for c, r in pts]}
    if hi - lo < 40:
        out.update(mode="flat")
        return out
    span = hi - lo
    two_levels = all(min(abs(r - lo), abs(r - hi)) <= 0.12 * span for r in raws)
    rails = lo < 250 and hi > 750
    if two_levels and rails:
        near_raw = pts[0][1]                     # the closest recorded distance
        out.update(mode="digital", threshold_raw=round((lo + hi) / 2.0, 1),
                   near_is_low=bool(near_raw < (lo + hi) / 2.0))
        return out
    # analog: power law cm = A * raw^B through all points (least squares in log-log)
    xs = [math.log(max(r, 1.0)) for _, r in pts]
    ys = [math.log(c) for c, _ in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else -1.0
    a = math.exp(my - b * mx)
    out.update(mode="analog", A=round(a, 6), B=round(b, 4))
    return out


class Calibration:
    def __init__(self, cal=None, min_cm=4.0, max_cm=30.0):
        cal = cal or {}
        self.mode = cal.get("mode", "none")
        self.points = sorted((float(c), float(r)) for c, r in cal.get("points", []))
        self.A = float(cal.get("A", 1.0))
        self.B = float(cal.get("B", -1.0))
        self.threshold = float(cal.get("threshold_raw", 512))
        self.near_is_low = bool(cal.get("near_is_low", True))
        self.min_cm, self.max_cm = min_cm, max_cm

    @property
    def usable(self):
        return self.mode in ("analog", "digital")

    def is_near(self, raw):
        """Digital: the module's output says 'obstacle'."""
        return raw < self.threshold if self.near_is_low else raw > self.threshold

    def to_cm(self, raw, trigger_cm=5.0):
        if raw is None:
            return None
        if self.mode == "digital":
            return trigger_cm - 0.5 if self.is_near(raw) else self.max_cm
        # analog: interpolate between the recorded points in log-log, power law outside
        pts = sorted(self.points, key=lambda p: p[1])     # by raw
        if len(pts) >= 2 and pts[0][1] <= raw <= pts[-1][1]:
            for (c1, r1), (c2, r2) in zip(pts, pts[1:]):
                if r1 <= raw <= r2 and r2 > r1:
                    t = (math.log(max(raw, 1)) - math.log(max(r1, 1))) / (math.log(max(r2, 1)) - math.log(max(r1, 1)))
                    cm = math.exp(math.log(c1) + t * (math.log(c2) - math.log(c1)))
                    break
            else:
                cm = self.A * max(raw, 1.0) ** self.B
        else:
            cm = self.A * max(raw, 1.0) ** self.B
        return round(min(max(cm, self.min_cm), self.max_cm), 2)


# =============================================================================
# reading an adaptor port: analog AND digital pin in one request
# =============================================================================
def read_port(adaptor, port):
    """(adc, io) of one sensor-adaptor port. The adaptor answers every request with BOTH
    the analog pin (0..1023) and the digital IO pin (0 / 1); the SDK's get_adc / get_io
    each throw half away. One request instead of two, and a digital module (IR obstacle
    sensor: DO wired to the IO pin) is read where its signal really is - its analog pin
    only floats (~300-400, the "no signal" of the last run)."""
    if port is None:
        return None, None
    try:
        from robomaster import protocol
        client = adaptor._client
        proto = protocol.ProtoSensorGetData()
        proto._port = port[1]
        msg = protocol.Msg(client.hostbyte, protocol.host2byte(22, port[0]), proto)
        resp = client.send_sync_msg(msg)
        if resp:
            p = resp.get_proto()
            if p is not None:
                return int(p._adc), int(p._io)
        return None, None
    except Exception:
        try:                                   # no SDK internals (tests / other SDK version)
            return adaptor.get_adc(id=port[0], port=port[1]), None
        except Exception:
            return None, None


class CornerSignal:
    """Which pin of an IR obstacle module carries its signal, learnt while it runs.

    signal "io"  -> digital pin, "adc" -> analog pin, "auto" -> whichever has switched
    (digital pin until then: a module's analog pin only floats). A module whose pins
    never change is reported (unplugged, no power, or it simply never saw a wall)."""

    def __init__(self, signal="auto"):
        self.signal = signal
        self.io_seen = set()
        self.adc_lo, self.adc_hi = None, None
        self.adc, self.io = None, None

    def update(self, adc, io):
        self.adc, self.io = adc, io
        if io is not None:
            self.io_seen.add(int(io))
        if adc is not None:
            self.adc_lo = adc if self.adc_lo is None else min(self.adc_lo, adc)
            self.adc_hi = adc if self.adc_hi is None else max(self.adc_hi, adc)

    @property
    def mode(self):
        if self.signal in ("io", "adc"):
            return self.signal
        if len(self.io_seen) > 1:
            return "io"
        if self.adc_hi is not None and self.adc_hi - self.adc_lo > 400:
            return "adc"                        # a module wired to the analog pin (rails 0 / 1023)
        return "io" if self.io is not None else "adc"

    def near(self, active_low, threshold):
        if self.mode == "io":
            if self.io is None:
                return None
            return self.io == 0 if active_low else self.io == 1
        if self.adc is None:
            return None
        return self.adc < threshold if active_low else self.adc > threshold

    def status(self):
        if self.adc is None and self.io is None:
            return "no answer"
        if len(self.io_seen) > 1:
            return f"OK - switches (digital pin, now {self.io})"
        if self.adc_hi is not None and self.adc_hi - self.adc_lo > 400:
            return f"OK - switches (analog pin, now {self.adc})"
        floating = self.adc is not None and 150 < self.adc < 800
        return (f"no change yet: digital pin stays {self.io}, analog {'floats' if floating else 'stays'} "
                f"~{self.adc} - hold a hand ~5 cm in front of it")


# =============================================================================
# the two sensors
# =============================================================================
class SharpIR:
    def __init__(self, sensor_adaptor, config=None, csv_path=None, log=print):
        cfg = (config or {}).get("sharp_ir", {}) or {}
        self.adaptor = sensor_adaptor
        self.log = log
        # candidate (adaptor id, port) pairs, first one that answers wins
        self.candidates = {
            "left": [tuple(p) for p in cfg.get("left_ports", [[1, 2]])],    # side Sharps
            "right": [tuple(p) for p in cfg.get("right_ports", [[2, 1]])],
        }
        self.port = {"right": None, "left": None}     # chosen port, None = not found (yet)
        self.max_fail = int(cfg.get("max_failures", 3))
        self._fails = {"right": 0, "left": 0}
        self.M = float(cfg.get("M", 12.0))
        self.C = float(cfg.get("C", 0.0))
        self.min_cm = float(cfg.get("min_cm", 4.0))
        self.max_cm = float(cfg.get("max_cm", 30.0))
        self.wall_cm = float(cfg.get("wall_detect_cm", 16.9))
        self.trigger_cm = float(cfg.get("trigger_cm", 5.0))     # Sharp fallback for near() without corners
        self.mount = str(cfg.get("mount", "side"))              # where the Sharps sit: side | front_corner_45
        # front-corner IR obstacle modules (digital, no calibration)
        cc = (config or {}).get("ir_corner", {}) or {}
        self.corners_enabled = bool(cc.get("enabled", False))
        self.corner_trigger_cm = float(cc.get("trigger_cm", 10.0))
        self.corner_port = {"left": tuple(cc.get("left_port", [1, 1])), "right": tuple(cc.get("right_port", [2, 2]))}
        default_active_low = bool(cc.get("active_low", True))
        self.corner_active_low = {
            side: bool(cc.get(f"{side}_active_low", default_active_low)) for side in SIDES
        }
        default_threshold = float(cc.get("threshold_raw", 512))
        self.corner_threshold = {
            side: float(cc.get(f"{side}_threshold_raw", default_threshold)) for side in SIDES
        }
        self.corner_raw = {"left": None, "right": None}
        self.corner_io = {"left": None, "right": None}
        default_signal = str(cc.get("signal", "auto"))
        self.corner_sig = {
            side: CornerSignal(str(cc.get(f"{side}_signal", default_signal))) for side in SIDES
        }
        self._corner_fails = {"left": 0, "right": 0}
        self._sharp_count = {s: 0 for s in SIDES}        # readings so far / highest raw seen:
        self._sharp_max = {s: 0 for s in SIDES}          # the "no signal" check
        self._warned = {}
        self.period = 1.0 / float(cfg.get("poll_hz", 10))
        self.csv_path = csv_path
        cal = cfg.get("calibration", {}) or {}
        self.cal = {s: Calibration(cal.get(s), self.min_cm, self.max_cm) for s in SIDES}

        self.lock = threading.Lock()
        self.left_cm = self.max_cm
        self.right_cm = self.max_cm
        self.left_raw = None
        self.right_raw = None
        self.ts = 0.0
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------
    def adc_to_cm(self, adc_value, prev_cm, side=None):
        """Calibrated distance when that sensor has a calibration, else the RoboFinal
        Sharp formula. Returns -1.0 for an invalid reading."""
        if adc_value is None or adc_value <= 0:
            return -1.0
        if side is not None and self.cal[side].usable:
            return self.cal[side].to_cm(adc_value, self.trigger_cm)
        voltage = (adc_value / ADC_MAX) * SYSTEM_VOLTAGE
        if voltage < 0.4:                      # beyond the reliable range
            return self.max_cm
        if voltage - self.C <= 0:
            return self.max_cm
        distance_cm = self.M / (voltage - self.C) - 0.42
        if prev_cm < 6 and distance_cm > 8.0:  # fold-back when touching the wall
            return 3.0
        return round(min(max(distance_cm, self.min_cm), self.max_cm), 2)

    def _get(self, port):
        return read_port(self.adaptor, port)[0]

    def sharp_status(self, side):
        """'OK' or why that Sharp gives nothing useful. A Sharp always outputs ~0.3 V or
        more (raw ~90+) even with nothing in front; stuck near 0 V = no signal."""
        if self.port[side] is None:
            return "not found (port)"
        # low readings alone are normal (no wall within ~30 cm on that side); a sensor that
        # NEVER read more than ~0.3 V over 100+ readings (~10 s, walls came and went) is dead
        n, top = self._sharp_count[side], self._sharp_max[side]
        if n >= 100 and top < 100:
            return f"NO SIGNAL (never above raw {top} in {n} readings = ~0 V): cable loose / wrong port / no power"
        return "OK"

    def probe(self):
        """Find which adaptor port each sensor is on (a missing one costs one SDK timeout)."""
        for side in ("right", "left"):
            for port in self.candidates[side]:
                if self._get(port) is not None:
                    self.port[side] = port
                    self.log(f"IR {side}: adaptor {port[0]} port {port[1]} ({self.cal[side].mode})")
                    break
            else:
                self.log(f"Sharp {side} NOT FOUND on {self.candidates[side]} - check sharp_ir in settings.yaml")
        if self.corners_enabled:
            for side in SIDES:
                a, p = self.corner_port[side]
                if self._get((a, p)) is None:
                    self.log(f"corner IR {side} NOT FOUND on adaptor {a} port {p} - check ir_corner")
                    self.corner_port[side] = None
                else:
                    self.log(f"corner IR {side}: adaptor {a} port {p}")

    def _read_side(self, side):
        port = self.port[side]
        if port is None:
            return None
        raw = self._get(port)
        if raw is None:
            self._fails[side] += 1
            if self._fails[side] >= self.max_fail:
                self.port[side] = None
                self.log(f"IR {side} stopped answering (adaptor {port[0]} port {port[1]}) - switched off")
        else:
            self._fails[side] = 0
        return raw

    def read_once(self):
        """Poll both sensors now; returns (left_cm, right_cm)."""
        # Safety first: read the front-corner switches before the slower analogue
        # side sensors so a wall trigger reaches the drive loop with minimum delay.
        if self.corners_enabled:
            for side in SIDES:
                port = self.corner_port[side]
                if port is None:
                    continue
                adc, io = read_port(self.adaptor, port)
                if adc is None and io is None:
                    self._corner_fails[side] += 1
                    if self._corner_fails[side] >= self.max_fail:
                        self.corner_port[side] = None
                        self.log(f"corner IR {side} stopped answering - switched off")
                else:
                    self._corner_fails[side] = 0
                    with self.lock:
                        before = self.corner_sig[side].mode
                        self.corner_sig[side].update(adc, io)
                        self.corner_raw[side], self.corner_io[side] = adc, io
                    if self.corner_sig[side].mode != before:
                        self.log(f"corner IR {side}: its signal is on the "
                                 f"{'digital' if self.corner_sig[side].mode == 'io' else 'analog'} pin")
        r_raw = self._read_side("right")
        time.sleep(0.02)
        l_raw = self._read_side("left")
        with self.lock:
            r_cm = self.adc_to_cm(r_raw, self.right_cm, "right")
            l_cm = self.adc_to_cm(l_raw, self.left_cm, "left")
            # keep the last good value when a read fails
            if r_cm is not None and r_cm >= 0:
                self.right_cm = r_cm
            if l_cm is not None and l_cm >= 0:
                self.left_cm = l_cm
            self.right_raw, self.left_raw = r_raw, l_raw
            self.ts = time.time()
            for side, raw in (("right", r_raw), ("left", l_raw)):
                if raw is not None:
                    self._sharp_count[side] += 1
                    self._sharp_max[side] = max(self._sharp_max[side], raw)
        for side in SIDES:                      # say once (every 30 s) when a Sharp gives nothing
            st = self.sharp_status(side)
            if st.startswith("NO SIGNAL") and time.time() - self._warned.get(side, 0) > 30:
                self._warned[side] = time.time()
                p = self.port[side]
                self.log(f"Sharp {side} (adaptor {p[0]} port {p[1]}): {st}")
        with self.lock:
            return self.left_cm, self.right_cm

    def latest(self):
        with self.lock:
            return self.left_cm, self.right_cm

    def corner_near(self, side):
        """Front-corner IR obstacle module says 'wall' (None = no module / no reading)."""
        if not self.corners_enabled or self.corner_port[side] is None:
            return None
        with self.lock:
            return self.corner_sig[side].near(self.corner_active_low[side], self.corner_threshold[side])

    def corner_status(self, side):
        if not self.corners_enabled:
            return "off"
        if self.corner_port[side] is None:
            return "not found (port)"
        with self.lock:
            return self.corner_sig[side].status()

    def near(self, side):
        """That front corner is at a wall: the corner IR module when fitted,
        else the Sharp on that side closer than trigger_cm."""
        c = self.corner_near(side)
        if c is not None:
            return c
        with self.lock:
            cm = self.left_cm if side == "left" else self.right_cm
            raw = self.left_raw if side == "left" else self.right_raw
        cal = self.cal[side]
        if cal.mode == "digital" and raw is not None:
            return cal.is_near(raw)
        return cm is not None and cm <= self.trigger_cm

    def wall_left(self):
        return self.latest()[0] <= self.wall_cm

    def wall_right(self):
        return self.latest()[1] <= self.wall_cm

    # ------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        if self.csv_path:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["unix_timestamp", "ir1_raw", "irR_cm", "ir2_raw", "irL_cm",
                                        "cornerL_raw", "cornerR_raw", "cornerL_io", "cornerR_io"])
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _poll(self):
        self.probe()
        while self._running:
            t0 = time.time()
            try:
                self.read_once()
                if self.csv_path:
                    with self.lock:
                        row = [self.ts, self.right_raw, self.right_cm, self.left_raw, self.left_cm,
                               self.corner_raw["left"], self.corner_raw["right"],
                               self.corner_io["left"], self.corner_io["right"]]
                    with open(self.csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)
            except Exception as e:
                print(f"IR Logging Error: {e}")
            time.sleep(max(0.0, self.period - (time.time() - t0)))
