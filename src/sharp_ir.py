"""Two Sharp IR distance sensors (left / right) on the RoboMaster sensor adaptor.

Adapted from RoboFinal (github.com/blaxlit/RoboFinal, src/chassis.py):
  * ports      tried in order until one answers (robots are wired differently:
               RoboFinal's right sensor is adaptor 3/port 1, this robot's is 2/1);
               a port that never answers is switched off instead of blocking
               every poll for 3 s (the SDK "cmdid:0xf0 timeout" spam)
  * model      Vo = M / (L + 0.42) + C  ->  L = M / (Vo - C) - 0.42   (GP2Y0A41, 4-30 cm)
  * fold-back  closer than ~4 cm the output drops again and looks "far";
               if the previous reading was < 6 cm and the new one jumps above
               8 cm we keep reporting 3 cm
  * polling    background thread (~10 Hz), raw + cm logged to CSV

Only these 2 Sharp sensors are used (assignment limit: max 2).
"""

import csv
import threading
import time

ADC_MAX = 1023.0
SYSTEM_VOLTAGE = 3.3


class SharpIR:
    def __init__(self, sensor_adaptor, config=None, csv_path=None, log=print):
        cfg = (config or {}).get("sharp_ir", {}) or {}
        self.adaptor = sensor_adaptor
        self.log = log
        # candidate (adaptor id, port) pairs, first one that answers wins
        self.candidates = {
            "right": [tuple(p) for p in cfg.get("right_ports", [[2, 1], [3, 1]])],
            "left": [tuple(p) for p in cfg.get("left_ports", [[1, 2], [1, 1]])],
        }
        self.port = {"right": None, "left": None}     # chosen port, None = not found (yet)
        self.max_fail = int(cfg.get("max_failures", 3))
        self._fails = {"right": 0, "left": 0}
        self.M = float(cfg.get("M", 12.0))
        self.C = float(cfg.get("C", 0.0))
        self.min_cm = float(cfg.get("min_cm", 4.0))
        self.max_cm = float(cfg.get("max_cm", 30.0))
        self.wall_cm = float(cfg.get("wall_detect_cm", 16.9))
        self.period = 1.0 / float(cfg.get("poll_hz", 10))
        self.csv_path = csv_path

        self.lock = threading.Lock()
        self.left_cm = self.max_cm
        self.right_cm = self.max_cm
        self.left_raw = None
        self.right_raw = None
        self.ts = 0.0
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------
    def adc_to_cm(self, adc_value, prev_cm):
        """RoboFinal conversion. Returns -1.0 for an invalid reading."""
        if adc_value is None or adc_value <= 0:
            return -1.0
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
        try:
            return self.adaptor.get_adc(id=port[0], port=port[1])
        except Exception:
            return None

    def probe(self):
        """Find which adaptor port each sensor is on (a missing one costs one SDK timeout)."""
        for side in ("right", "left"):
            for port in self.candidates[side]:
                if self._get(port) is not None:
                    self.port[side] = port
                    self.log(f"Sharp {side}: adaptor {port[0]} port {port[1]}")
                    break
            else:
                self.log(f"Sharp {side} NOT FOUND on {self.candidates[side]} - check sharp_ir in settings.yaml")

    def _read_side(self, side):
        port = self.port[side]
        if port is None:
            return None
        raw = self._get(port)
        if raw is None:
            self._fails[side] += 1
            if self._fails[side] >= self.max_fail:
                self.port[side] = None
                self.log(f"Sharp {side} stopped answering (adaptor {port[0]} port {port[1]}) - switched off")
        else:
            self._fails[side] = 0
        return raw

    def read_once(self):
        """Poll both sensors now; returns (left_cm, right_cm)."""
        r_raw = self._read_side("right")
        time.sleep(0.02)
        l_raw = self._read_side("left")
        with self.lock:
            r_cm = self.adc_to_cm(r_raw, self.right_cm)
            l_cm = self.adc_to_cm(l_raw, self.left_cm)
            # keep the last good value when a read fails
            if r_cm >= 0:
                self.right_cm = r_cm
            if l_cm >= 0:
                self.left_cm = l_cm
            self.right_raw, self.left_raw = r_raw, l_raw
            self.ts = time.time()
            return self.left_cm, self.right_cm

    def latest(self):
        with self.lock:
            return self.left_cm, self.right_cm

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
                csv.writer(f).writerow(["unix_timestamp", "ir1_raw", "irR_cm", "ir2_raw", "irL_cm"])
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
                        row = [self.ts, self.right_raw, self.right_cm, self.left_raw, self.left_cm]
                    with open(self.csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)
            except Exception as e:
                print(f"IR Logging Error: {e}")
            time.sleep(max(0.0, self.period - (time.time() - t0)))
