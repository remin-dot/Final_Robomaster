import csv
import time
import os
import math
from datetime import datetime
import cv2
import numpy as np

from sharp_ir import SharpIR

try:
    from robomaster import robot
except Exception:
    robot = None


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class ChassisController:
    # ---- ความปลอดภัยข้าง (Sharp IR) ----
    SIDE_SAFE_CM = 9.0      # เริ่มเลื่อนหลบ (Sharp 4-30 cm; กลางช่องอ่านได้ ~13 cm) - sharp_ir.side_safe_cm
    SIDE_DANGER_CM = 6.0    # ชะลอเดินหน้า - sharp_ir.side_danger_cm
    STRAFE_V = 0.15
    KP_YAW_HOLD = 0.8
    FRONT_STOP_MM = 150     # เบรกฉุกเฉินด้านหน้า

    # ---- เดินด้วย odometry ----
    DECEL_ZONE_M = 0.18     # ระยะสุดท้ายที่เริ่มชะลอ (ลดระยะไหลหลังสั่งหยุด)
    MIN_V = 0.08            # ความเร็วต่ำสุดตอนชะลอ (m/s)
    COAST_INIT_M = 0.02     # ค่าเริ่มต้นระยะไหลหลังสั่งหยุด (เรียนรู้เองทุกครั้งที่เดิน)
    COAST_MAX_M = 0.06
    TIME_CAP_FACTOR = 1.7   # เพดานเวลา = เวลาปกติ x ค่านี้ + 0.3s (กัน odometry ค้าง)

    # ---- alignment ด้วยกำแพง ----
    ALIGN_MAX_MM = 450      # ใช้กำแพงที่ใกล้กว่านี้ในการจัดแนว
    ALIGN_PHI_DEG = 15.0    # มุมวัดซ้าย/ขวาของ ToF
    ALIGN_MAX_DEG = 20.0    # เบี้ยวเกินนี้ถือว่าวัดผิด ข้าม

    def __init__(self, ep_robot, config):
        self.ep_robot = ep_robot
        self.config = config
        self.ep_chassis = ep_robot.chassis
        self.ep_sensor = ep_robot.sensor
        self.ep_gimbal = ep_robot.gimbal
        self.ep_adaptor = ep_robot.sensor_adaptor

        self.current_tof_dist_mm = 9999
        self.current_yaw = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self._pos_count = 0

        move_cfg = config.get("movement", {})
        # z>0 ทำให้ yaw เพิ่มหรือลด (หาอัตโนมัติใน calibrate_yaw_sign)
        self.z_sign = 1
        # เลี้ยวขวา = yaw บวก (หมุนตามเข็ม)
        self.right_yaw_sign = move_cfg.get("right_yaw_sign", 1)
        # yaw ของทิศเหนือแผนที่ (อัปเดตอัตโนมัติจากกำแพง)
        self.yaw_origin = 0.0
        # ToF ด้านหน้าเมื่อหุ่นอยู่กลางช่องหันหน้าเข้ากำแพง (mm) ต้องวัดเองแล้วใส่ config
        self.front_center_mm = move_cfg.get("front_center_mm", None)
        self.align_enabled = move_cfg.get("align_enabled", False)

        self.coast_est = self.COAST_INIT_M
        # optional MissionPanel (live camera + map) and TargetShooter, set by main_mission
        self.panel = None
        self.shooter = None
        # 2 Sharp IR (left/right) - driver + calibration from RoboFinal, polled in a thread
        self.ir = SharpIR(self.ep_adaptor, config, log=self._log)
        self._left_cm = self.ir.max_cm
        self._right_cm = self.ir.max_cm
        ir_cfg = config.get("sharp_ir", {}) or {}
        # sensor faces are ~0.12 m from the centre: a centred robot in a 0.5 m way reads ~13 cm
        self.SIDE_SAFE_CM = float(ir_cfg.get("side_safe_cm", 9.0))
        self.SIDE_DANGER_CM = float(ir_cfg.get("side_danger_cm", 6.0))
        # one wall threshold for every gimbal direction (the ToF turns with the gimbal)
        self.wall_mm = float(move_cfg.get("wall_threshold_mm", 450))
        # order to try the open ways of a cell (relative to how the robot came in):
        # side cells first, then ahead - settings.yaml movement.explore_order
        order = [str(o).lower() for o in move_cfg.get("explore_order", ["left", "right", "front", "back"])]
        self.explore_order = [o for o in order if o in ("front", "right", "back", "left")] or \
            ["left", "right", "front", "back"]
        for o in ("left", "right", "front", "back"):
            if o not in self.explore_order:
                self.explore_order.append(o)
        self._tof_samples = []          # (t, mm) recent ToF readings
        vis_cfg = config.get("vision", {}) or {}
        # camera wall check: wall height - camera height (m), learnt from the ToF at every front scan
        self.wall_rise_m = float(vis_cfg.get("wall_rise_m", 0.12))
        self._rise_samples = []
        self.CAM_BLOCK_M = float(vis_cfg.get("camera_block_m", 0.40))   # before a move: wall this close ahead = blocked
        self.CAM_STOP_M = float(vis_cfg.get("camera_stop_m", 0.22))     # while moving: stop at this
        # ... and only when the ToF confirms something that close (normal cells: far wall ~850 mm)
        self.CAM_TOF_BLOCK_MM = float(vis_cfg.get("camera_tof_block_mm", 450))
        self.CAM_TOF_STOP_MM = float(vis_cfg.get("camera_tof_stop_mm", 300))
        self.last_move_note = ""
        self.HEADING_GUARD_DEG = float(move_cfg.get("heading_guard_deg", 12.0))
        self.verify_pitch_deg = float(vis_cfg.get("verify_pitch_deg", -12.0))   # close look: camera down
        self.verify_enabled = bool(vis_cfg.get("verify_sweep", True))
        self._odom0 = None              # (x, y, yaw_origin) when the map was fixed

        self.data_dir = config["data_collection"]["data_dir"]
        self.buffer_time = config["data_collection"]["buffer_time"]

        self.freq_pos = config["data_collection"]["frequencies"]["position"]
        self.freq_att = config["data_collection"]["frequencies"]["attitude"]
        self.freq_imu = config["data_collection"]["frequencies"]["imu"]
        self.freq_esc = config["data_collection"]["frequencies"]["esc"]
        self.freq_dist = config["data_collection"]["frequencies"]["distance"]

        self.default_speed = move_cfg["xy_speed"]
        self.default_distance = move_cfg["distance"]
        self.default_z_speed = move_cfg["z_speed"]
        self.default_angle = move_cfg["angle"]

        os.makedirs(self.data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")

        files = config["data_collection"]["files"]
        self.pos_file = os.path.join(self.data_dir, f"log_{date_str}_{files['position']}.csv")
        self.att_file = os.path.join(self.data_dir, f"log_{date_str}_{files['attitude']}.csv")
        self.imu_file = os.path.join(self.data_dir, f"log_{date_str}_{files['imu']}.csv")
        self.esc_file = os.path.join(self.data_dir, f"log_{date_str}_{files['esc']}.csv")
        self.dist_file = os.path.join(self.data_dir, f"log_{date_str}_{files['distance']}.csv")
        self.ir.csv_path = os.path.join(self.data_dir, f"log_{date_str}_{files.get('infrared', 'ir_data')}.csv")

    # ------------------------------------------------------------------
    # Sharp IR
    # ------------------------------------------------------------------
    def read_side_ir(self):
        """Latest (left_cm, right_cm) from the Sharp polling thread (4-30 cm)."""
        self._left_cm, self._right_cm = self.ir.latest()
        return self._left_cm, self._right_cm

    # ------------------------------------------------------------------
    # CSV / callbacks
    # ------------------------------------------------------------------
    def save_to_csv(self, filename, data):
        current_time = time.time()
        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [current_time]
            for item in data:
                if isinstance(item, (list, tuple)):
                    row.extend(item)
                else:
                    row.append(item)
            writer.writerow(row)

    def handle_position(self, data):
        self.save_to_csv(self.pos_file, data)
        self.pos_x, self.pos_y = data[0], data[1]
        if len(data) > 2:
            self.pos_z = data[2]
        self._pos_count += 1
        if self.panel is not None and self._odom0 is not None:
            # odometry frame (x forward, y right at power-on) -> map (east, north) from the start cell
            x0, y0, yo = self._odom0
            dx, dy = self.pos_x - x0, self.pos_y - y0
            n_ang = math.radians(yo)
            e_ang = math.radians(yo + 90.0 * self.right_yaw_sign)
            north = dx * math.cos(n_ang) + dy * math.sin(n_ang)
            east = dx * math.cos(e_ang) + dy * math.sin(e_ang)
            self.panel.map.add_odom(east, north)

    def handle_attitude(self, data):
        self.save_to_csv(self.att_file, data)
        self.current_yaw = data[0]

    def handle_imu(self, data):
        self.save_to_csv(self.imu_file, data)

    def handle_esc(self, data):
        self.save_to_csv(self.esc_file, data)

    def handle_distance(self, data):
        self.save_to_csv(self.dist_file, data)
        self.current_tof_dist_mm = data[0]
        self._tof_samples.append((time.time(), data[0]))
        del self._tof_samples[:-20]

    def _tof_fresh(self, n=3, timeout=0.6):
        """Median of n ToF readings taken after now (the gimbal has settled)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            fresh = [mm for t, mm in list(self._tof_samples) if t > t0]
            if len(fresh) >= n:
                return sorted(fresh)[len(fresh) // 2]
            time.sleep(0.02)
        fresh = [mm for t, mm in list(self._tof_samples) if t > t0]
        return sorted(fresh)[len(fresh) // 2] if fresh else self.current_tof_dist_mm

    def _on_gimbal_angle(self, angle_info):
        """(pitch, yaw, pitch_ground, yaw_ground) from the gimbal."""
        if self.panel is not None:
            self.panel.detector.gimbal_pitch_deg = float(angle_info[0])

    def _camera_wall_ahead(self, fresh=False):
        """Distance (m) to a white wall in the robot's path, seen by the camera (inf = none/unknown)."""
        p = self.panel
        if p is None:
            return float("inf")
        if fresh:   # a frame taken after the robot / gimbal stopped (Wi-Fi video is ~0.2 s late)
            p.worker.wait_fresh(time.time() + p.camera_latency_s, timeout=0.8)
        elif time.time() - p.worker.det_ts > 0.6:
            return float("inf")   # no recent frame: do not guess
        d, _ = p.detector.wall_ahead_m(self.wall_rise_m)
        return d

    def _learn_wall_rise(self, tof_mm):
        """Gimbal straight ahead at a wall: ToF distance + where the camera sees the wall top
        -> wall height above the camera (keeps the camera distance honest)."""
        p = self.panel
        if p is None or not (150 < tof_mm < 1000):
            return
        p.worker.wait_fresh(time.time() + p.camera_latency_s, timeout=0.8)
        _, rise = p.detector.wall_ahead_m(self.wall_rise_m)
        size = p.detector.last_frame_size
        if rise is None or rise < 6 or size is None:
            return
        f = p.detector.focal_px(size[0])
        k = (tof_mm / 1000.0) * rise / f
        if 0.03 < k < 0.45:
            self._rise_samples = (self._rise_samples + [k])[-15:]
            if len(self._rise_samples) >= 3:
                new = float(np.median(self._rise_samples))
                if abs(new - self.wall_rise_m) > 0.015:
                    self._log(f"camera wall height learnt: {new:.3f} m above the camera "
                              f"({len(self._rise_samples)} ToF samples, last {k:.3f})")
                self.wall_rise_m = new

    def _log(self, msg):
        if self.panel is not None:
            self.panel.log(msg)
        else:
            print(msg)

    def setup_csv_headers(self):
        with open(self.pos_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "x", "y", "z"])
        with open(self.att_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "yaw", "pitch", "roll"])
        with open(self.imu_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        with open(self.esc_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "esc_data"])
        with open(self.dist_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "tof1"])

    def start_sensors(self):
        print("Starting to collect sensor data...")

        if robot is not None:
            try:
                self.ep_robot.set_robot_mode(mode=robot.FREE)
            except Exception as e:
                print(f"[warn] set_robot_mode: {e}")

        # odometry ต้องถี่พอ (1-5Hz ทำให้หยุดช้าไปเป็น 10-30cm)
        self.ep_chassis.sub_position(freq=max(self.freq_pos, 20), callback=self.handle_position)
        # 20 Hz: at 5 Hz the yaw is 200 ms old, turns overshoot and stop at the wrong angle
        self.ep_chassis.sub_attitude(freq=max(self.freq_att, 20), callback=self.handle_attitude)
        self.ep_chassis.sub_imu(freq=self.freq_imu, callback=self.handle_imu)
        self.ep_chassis.sub_esc(freq=self.freq_esc, callback=self.handle_esc)
        # 20 Hz: at 5 Hz a scan could read the previous gimbal direction's wall
        self.ep_sensor.sub_distance(freq=max(self.freq_dist, 20), callback=self.handle_distance)
        self.ir.start()
        try:  # live gimbal pitch -> the camera's horizon rule (ignore the room above it)
            self.ep_gimbal.sub_angle(freq=10, callback=self._on_gimbal_angle)
        except Exception as e:
            print(f"[warn] gimbal sub_angle: {e}")

        try:
            self.ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
            time.sleep(0.5)
        except Exception:
            pass

    def stop_sensors(self):
        time.sleep(self.buffer_time)
        self.ep_chassis.unsub_position()
        self.ep_chassis.unsub_attitude()
        self.ep_chassis.unsub_imu()
        self.ep_chassis.unsub_esc()
        self.ep_sensor.unsub_distance()
        self.ir.stop()
        try:
            self.ep_gimbal.unsub_angle()
        except Exception:
            pass
        if self.panel is None:  # the Mission Panel owns its window
            cv2.destroyAllWindows()
        print("Data collection and saving to the file have been fully completed.")

    # ------------------------------------------------------------------
    # Gimbal / basic move
    # ------------------------------------------------------------------
    def reset_gimbal(self):
        try:
            self.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=200).wait_for_completed()
            time.sleep(0.1)
        except Exception:
            pass

    def move_forward(self, distance=None, speed=None):
        if distance is None:
            distance = self.default_distance
        if speed is None:
            speed = self.default_speed
        self.ep_chassis.move(x=distance, y=0, z=0, xy_speed=speed).wait_for_completed()

    def _stop(self, wait=0.3):
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(wait)

    # ------------------------------------------------------------------
    # ตรวจทิศ z กับ yaw
    # ------------------------------------------------------------------
    def calibrate_yaw_sign(self):
        print("--> [Calibrate] ตรวจทิศ z กับ yaw ...")
        self._stop(0.3)
        y0 = self.current_yaw
        self.ep_chassis.drive_speed(x=0, y=0, z=30)
        time.sleep(0.6)
        self._stop(0.3)
        d = wrap180(self.current_yaw - y0)
        if abs(d) < 3:
            print(f"[warn] yaw เปลี่ยนน้อยเกินไป ({d:.1f}°) ใช้ z_sign=+1 ตามเดิม")
            self.z_sign = 1
        else:
            self.z_sign = 1 if d > 0 else -1
        print(f"    z>0 ทำให้ yaw เปลี่ยน {d:+.1f}° -> z_sign={self.z_sign:+d}")
        self.turn_to_absolute_yaw(y0)
        self.yaw_origin = y0
        self._odom0 = (self.pos_x, self.pos_y, y0)   # map origin for the odometry trail
        if self.panel is not None:
            self.panel.map.odom = []
        print(f"    yaw_origin={y0:.1f}° (ทิศเหนือของแผนที่)")

    # ------------------------------------------------------------------
    # เดินตรง: distance วัดจาก odometry ของล้อ (ไม่ใช้เวลา) + IMU คุมทิศ + IR คุมข้าง
    # ------------------------------------------------------------------
    def safe_move_forward(self, distance=0.6, speed=0.3, stop_limit_mm=None, target_heading_deg=None,
                          camera_guard=True):
        """Drive one cell. camera_guard=False on a way already driven through (backtracking):
        it is known to be open, so only the ToF emergency stop applies.
        self.last_move_note says why a move gave up (logged in the exploration CSV)."""
        self.last_move_note = ""
        if stop_limit_mm is None:
            stop_limit_mm = self.FRONT_STOP_MM
        print(f"--> [Safe Move] เดินหน้า {distance}m ที่ {speed}m/s")
        self.reset_gimbal()

        # camera + ToF look before moving: a wrong turn would face a wall here
        if target_heading_deg is not None and abs(wrap180(target_heading_deg - self.current_yaw)) > self.TURN_OK_DEG:
            self.turn_to_absolute_yaw(target_heading_deg)
        cam_d = self._camera_wall_ahead(fresh=True) if camera_guard else float("inf")
        if cam_d < self.CAM_BLOCK_M:
            tof = self._tof_fresh(3)
            tof_ok = 60 < tof < 8000
            # the camera is a second opinion: it blocks only when the ToF also sees something CLOSE
            # (a normal next cell has its far wall ~850 mm away), or when the ToF has no reading
            if (tof_ok and tof < self.CAM_TOF_BLOCK_MM) or (not tof_ok and cam_d < self.CAM_STOP_M):
                self.last_move_note = f"camera {cam_d:.2f} m + ToF {tof:.0f} mm: wall ahead"
                self._log(f"camera: wall {cam_d:.2f} m ahead (ToF {tof:.0f} mm) - not moving")
                return False
            self._log(f"camera says wall {cam_d:.2f} m but ToF {tof:.0f} mm - trusting the ToF")

        sx, sy = self.pos_x, self.pos_y
        cnt0 = self._pos_count
        start_time = time.time()
        max_time = distance / speed * self.TIME_CAP_FACTOR + 0.3

        if target_heading_deg is None:
            target_heading_deg = self.current_yaw

        traveled = 0.0
        pos_ok = False
        stop_traveled = None     # ระยะตอนสั่งหยุด (ใช้เรียนรู้ระยะไหล)
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    print("[warn] odometry ช้า/ค้าง หยุดด้วยเพดานเวลา")
                    break

                pos_ok = self._pos_count != cnt0
                if pos_ok:
                    traveled = math.hypot(self.pos_x - sx, self.pos_y - sy)
                else:
                    traveled = elapsed * speed   # fallback ถ้า odometry ไม่มา

                # ระยะที่เหลือ หักระยะไหลที่เรียนรู้ไว้
                remaining = distance - (traveled + self.coast_est)
                if remaining <= 0:
                    stop_traveled = traveled
                    break

                progress = traveled / distance
                front_dist = self.current_tof_dist_mm

                if 0 < front_dist <= stop_limit_mm:
                    self._stop(0.3)
                    if progress >= 0.80:
                        print(f"-> [ถึงเป้าหมาย] พบกำแพงหน้าช่องที่ {front_dist}mm")
                        return True
                    self.last_move_note = f"ToF obstacle {front_dist} mm at {traveled:.2f} m"
                    print(f"!!! [สิ่งกีดขวาง] {front_dist}mm หยุดทันที")
                    self._retreat(traveled, sx, sy, pos_ok, speed)
                    return False

                # camera: a white wall coming into the robot's path
                cam_d = self._camera_wall_ahead() if camera_guard else float("inf")
                tof_now = self.current_tof_dist_mm
                # both must say CLOSE: driving into a cell the ToF passes 600 mm long before arriving
                if cam_d < self.CAM_STOP_M and (not 60 < tof_now < 8000 or tof_now < self.CAM_TOF_STOP_MM):
                    self.last_move_note = f"camera {cam_d:.2f} m + ToF {tof_now:.0f} mm while moving"
                    self._stop(0.3)
                    if progress >= 0.80:
                        print(f"-> [ถึงเป้าหมาย] กล้องเห็นกำแพงหน้าช่อง {cam_d:.2f} m")
                        return True
                    self._log(f"camera: wall {cam_d:.2f} m ahead while moving - stop")
                    self._retreat(traveled, sx, sy, pos_ok, speed)
                    return False

                # heading drifted too far (slip / bad turn): stop and turn back first
                if abs(wrap180(target_heading_deg - self.current_yaw)) > self.HEADING_GUARD_DEG:
                    self._stop(0.2)
                    self._log(f"heading off by {wrap180(target_heading_deg - self.current_yaw):+.0f} deg while moving - correcting")
                    t_fix = time.time()
                    if not self.turn_to_absolute_yaw(target_heading_deg):
                        self.last_move_note = "heading could not be corrected"
                        self._retreat(traveled, sx, sy, pos_ok, speed)
                        return False
                    start_time += time.time() - t_fix   # the correction does not count against the move's time cap
                    continue

                # ชะลอช่วงท้ายให้หยุดแม่น
                v_forward = speed
                if remaining < self.DECEL_ZONE_M:
                    v_forward = max(self.MIN_V, speed * remaining / self.DECEL_ZONE_M)

                yaw_error = wrap180(target_heading_deg - self.current_yaw)
                z_val = max(min(yaw_error * self.KP_YAW_HOLD, 30), -30)
                z_cmd = self.z_sign * z_val

                left_cm, right_cm = self.read_side_ir()
                y_speed = 0.0

                if left_cm < self.SIDE_SAFE_CM and left_cm <= right_cm:
                    y_speed = self.STRAFE_V
                elif right_cm < self.SIDE_SAFE_CM:
                    y_speed = -self.STRAFE_V
                elif left_cm < self.ir.max_cm - 2 and right_cm < self.ir.max_cm - 2:
                    # กำแพงสองข้าง ประคองกลางทาง (left มากกว่า = ชิดขวา -> เลื่อนซ้าย)
                    y_speed = max(min(-0.01 * (left_cm - right_cm), 0.15), -0.15)

                if min(left_cm, right_cm) < self.SIDE_DANGER_CM:
                    v_forward = min(v_forward, speed * 0.4)

                self.ep_chassis.drive_speed(x=v_forward, y=y_speed, z=z_cmd)
                time.sleep(0.05)

        except Exception as e:
            print(f"[-] เกิดข้อผิดพลาดในการเคลื่อนที่: {e}")
            self._stop(0.4)
            return False

        self._stop(0.5)
        if self._pos_count != cnt0:
            traveled = math.hypot(self.pos_x - sx, self.pos_y - sy)

        # เรียนรู้ระยะไหลหลังสั่งหยุด (ปรับทีละครึ่ง กันค่ากระโดด)
        if stop_traveled is not None and pos_ok:
            coast = max(0.0, traveled - stop_traveled)
            self.coast_est = min(self.COAST_MAX_M, 0.5 * self.coast_est + 0.5 * coast)

        print(f"   [Move] odom={traveled:.3f}m เวลา={time.time() - start_time:.2f}s "
              f"pos_updates={self._pos_count - cnt0} coast={self.coast_est:.3f}m")
        if traveled < 0.8 * distance and not self.last_move_note:
            self.last_move_note = f"only {traveled:.2f} m of {distance:.2f} m (time cap)"
        return traveled >= 0.8 * distance

    def _retreat(self, traveled, sx, sy, pos_ok, speed):
        """ถอยกลับเข้ากลางช่องเดิม"""
        if traveled <= 0.03:
            return
        print("-> [Retreat] ถอยกลับเข้ากลางช่องเดิม")
        v = min(speed, 0.2)
        if pos_ok:
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if math.hypot(self.pos_x - sx, self.pos_y - sy) <= 0.02:
                    break
                self.ep_chassis.drive_speed(x=-v, y=0, z=0)
                time.sleep(0.05)
        else:
            self.ep_chassis.drive_speed(x=-v, y=0, z=0)
            time.sleep(traveled / v)
        self._stop(0.4)

    # ------------------------------------------------------------------
    # หมุนด้วย absolute yaw
    # ------------------------------------------------------------------
    TURN_TOL_DEG = 2.0       # stop turning inside this error
    TURN_OK_DEG = 4.0        # after settling, an error above this = retry the turn

    def turn_to_absolute_yaw(self, target_yaw, attempts=3):
        """Turn in place to target_yaw and CHECK it got there.

        The old loop stopped after a fixed 4 s at <= 45 deg/s, so a 180 deg U-turn
        (every backtrack) ended ~30 deg short and the robot then drove into a wall.
        Now the time allowed grows with the angle, and the result is checked and
        retried. Returns True when the heading is within TURN_OK_DEG."""
        print(f"--> [Turn] หมุนไป {target_yaw:.1f}° (ปัจจุบัน {self.current_yaw:.1f}°)")
        self._stop(0.2)
        for attempt in range(attempts):
            err0 = abs(wrap180(target_yaw - self.current_yaw))
            limit = 1.5 + err0 / 30.0            # 180 deg -> 7.5 s at the slowest
            start_time = time.time()
            while (time.time() - start_time) < limit:
                error = wrap180(target_yaw - self.current_yaw)
                if abs(error) < self.TURN_TOL_DEG:
                    break
                z_speed = error * 1.5
                if z_speed > 0:
                    z_speed = max(min(z_speed, 60), 10)
                else:
                    z_speed = min(max(z_speed, -60), -10)
                self.ep_chassis.drive_speed(x=0, y=0, z=self.z_sign * z_speed)
                time.sleep(0.05)
            self._stop(0.35)
            final = wrap180(target_yaw - self.current_yaw)
            if abs(final) <= self.TURN_OK_DEG:
                return True
            self._log(f"turn off by {final:+.0f} deg - correcting ({attempt + 1}/{attempts})")
        self._log(f"turn failed: still {wrap180(target_yaw - self.current_yaw):+.0f} deg off")
        return False

    def heading_to_yaw(self, heading_index):
        """0=N, 1=E, 2=S, 3=W เทียบกับ yaw_origin"""
        s = self.right_yaw_sign
        mapping = {0: 0.0, 1: 90.0 * s, 2: 180.0, 3: -90.0 * s}
        return wrap180(self.yaw_origin + mapping[heading_index % 4])

    # ------------------------------------------------------------------
    # Alignment ด้วยกำแพง (ToF บน gimbal)
    # ------------------------------------------------------------------
    def _tof_avg(self, n=3):
        vals = []
        for _ in range(n):
            time.sleep(0.06)
            v = self.current_tof_dist_mm
            if 0 < v < 3000:
                vals.append(v)
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    def _tof_at(self, gimbal_yaw):
        try:
            self.ep_gimbal.moveto(pitch=0, yaw=gimbal_yaw, yaw_speed=180).wait_for_completed()
        except Exception:
            return None
        time.sleep(0.15)
        return self._tof_avg()

    def measure_wall_angle(self, base_deg):
        """วัดมุมเบี้ยวของกำแพงที่อยู่ทิศ base_deg (0=หน้า, 90=ขวา, -90=ซ้าย) เทียบกับตัวหุ่น
        ยิง ToF ที่ base±phi แล้วใช้ r(a)=D/cos(a-theta) หา theta
        theta>0 = กำแพงหมุนตามเข็มจากทิศ base (หุ่นต้องหมุนตามเข็ม theta เพื่อขนาน/ตั้งฉาก)"""
        phi = self.ALIGN_PHI_DEG
        r_p = self._tof_at(base_deg + phi)
        r_m = self._tof_at(base_deg - phi)
        if r_p is None or r_m is None:
            return None
        if max(r_p, r_m) > self.ALIGN_MAX_MM * 1.8:
            return None
        t = (r_m - r_p) / ((r_m + r_p) * math.tan(math.radians(phi)))
        return math.degrees(math.atan(t))

    def align_to_walls(self, dist, heading):
        """จัดหุ่นให้ตรงกับกำแพงที่ใกล้ที่สุด แล้วปรับ yaw_origin ตามกำแพง (แก้ gyro drift)"""
        cands = (("front", 0), ("right", 90), ("left", -90))
        valid = [(dist[k], base) for k, base in cands if 0 < dist[k] <= self.ALIGN_MAX_MM]
        if not valid:
            print("-> [Align] ไม่มีกำแพงใกล้ ข้าม")
            return False

        _, base = min(valid)
        theta = self.measure_wall_angle(base)
        self.reset_gimbal()

        if theta is None or abs(theta) > self.ALIGN_MAX_DEG:
            print(f"-> [Align] วัดมุมไม่ได้/ผิดปกติ ({theta}) ข้าม")
            return False

        print(f"-> [Align] กำแพง({base:+d}°) เบี้ยว {theta:+.1f}°")
        target = wrap180(self.current_yaw + theta)
        if abs(theta) >= 2.0:
            self.turn_to_absolute_yaw(target)

        expected = self.heading_to_yaw(heading)
        shift = wrap180(target - expected)
        if abs(shift) <= 25.0:
            self.yaw_origin = wrap180(self.yaw_origin + shift)
            print(f"   yaw_origin -> {self.yaw_origin:.1f}° (ปรับ {shift:+.1f}°)")
        return True

    def align_front_distance(self):
        """ถ้ามีกำแพงหน้า ปรับตำแหน่งหน้า-หลังให้ ToF = front_center_mm (ต้องตั้งใน config)"""
        if self.front_center_mm is None:
            return
        front = self._tof_avg()
        if front is None or front > self.ALIGN_MAX_MM:
            return
        t0 = time.time()
        while time.time() - t0 < 2.0:
            err = self.current_tof_dist_mm - self.front_center_mm
            if abs(err) <= 8:
                break
            v = max(min(err / 1000.0 * 1.2, 0.12), -0.12)
            if abs(v) < 0.04:
                v = 0.04 if v > 0 else -0.04
            self.ep_chassis.drive_speed(x=v, y=0, z=0)
            time.sleep(0.05)
        self._stop(0.3)

    # ------------------------------------------------------------------
    # Scan / draw
    # ------------------------------------------------------------------
    def scan_surroundings_with_gimbal(self):
        self.reset_gimbal()
        distances = {"front": 0, "right": 0, "back": 0, "left": 0}

        scan_sequence = (
            ("front", 0),
            ("right", 90),
            ("back", 180),
            ("left", -90),
        )

        for label, yaw in scan_sequence:
            self.ep_gimbal.moveto(pitch=0, yaw=yaw, yaw_speed=180).wait_for_completed()
            time.sleep(0.1)
            distances[label] = self._tof_fresh(3)     # readings taken after the gimbal settled
            if label == "front":
                self._learn_wall_rise(distances[label])
            self._draw_live_with_gimbal(yaw)
            self._look_for_targets(yaw)

        self.reset_gimbal()
        self._draw_live_with_gimbal(0)
        return distances

    def _look_for_targets(self, gimbal_relative_yaw, pitch=0.0):
        """Camera check while the gimbal is settled: put every valid target on
        the map, then aim + fire at designated ones that are within range.
        Returns the detections found."""
        ctx = getattr(self, "_live_ctx", None)
        if self.panel is None or ctx is None or not self.panel.checkpoint():
            return []
        pos, heading = ctx[0], ctx[4]
        abs_deg = ({0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0) + gimbal_relative_yaw) % 360
        # looking down the ToF hits the floor: no wall check then
        tof = self.current_tof_dist_mm if abs(pitch) < 1 else None
        self.panel.detector.gimbal_pitch_deg = pitch   # until the gimbal angle feed catches up
        found = self.panel.observe_targets(pos, abs_deg, tof_mm=tof, settle_ts=time.time())
        if self.shooter is None:
            return
        for det in found:
            tid = getattr(det, "target_id", None)
            if det.in_range and tid and self.panel.should_shoot(tid):   # selected kinds only
                self.shooter.engage(det.kind, tid)
        # back to the scan direction (engage may have moved the gimbal)
        if found:
            self.ep_gimbal.moveto(pitch=pitch, yaw=gimbal_relative_yaw, yaw_speed=180).wait_for_completed()
        return found

    def verify_sweep(self, pos):
        """Something was seen in (or next to) this block: aim the camera down and turn it all
        the way round to make sure it is a card. Cards hang lower than the camera, so
        looking down puts close ones in the middle of the picture. A sighting that this
        close look cannot find again is dropped from the map (a false detection)."""
        p = self.panel
        if p is None:
            return
        ids = p.map.to_verify(pos)
        if not ids:
            return
        before = {tid: p.map.targets[tid]["n"] for tid in ids if tid in p.map.targets}
        known_before = set(p.map.targets)
        p.log(f"close look at {pos}: " + ", ".join(ids))
        pitch = self.verify_pitch_deg
        for g in (0, 45, 90, 135, 180, -135, -90, -45):
            if not p.checkpoint():
                break
            self.ep_gimbal.moveto(pitch=pitch, yaw=g, pitch_speed=120, yaw_speed=180).wait_for_completed()
            time.sleep(0.15)
            self._draw_live_with_gimbal(g)
            self._look_for_targets(g, pitch=pitch)
        self.ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=120, yaw_speed=180).wait_for_completed()
        p.detector.gimbal_pitch_deg = 0.0
        # a far first sighting can be ~0.5 m off: a card of the same kind found by the close
        # look near it is the same card, seen properly now -> merge into it
        new_ids = [tid for tid in p.map.targets if tid not in before and tid not in known_before]
        for tid, n0 in before.items():
            t = p.map.targets.get(tid)
            if t is None or t["n"] > n0:
                continue
            tx, ty = t["sx"] / t["sw"], t["sy"] / t["sw"]
            for nid in list(new_ids):
                nt = p.map.targets.get(nid)
                if nt is None or nt["kind"] != t["kind"]:
                    continue
                if math.hypot(nt["sx"] / nt["sw"] - tx, nt["sy"] / nt["sw"] - ty) <= 1.0:
                    with p.map.lock:   # the close look is the better estimate: it replaces the far one
                        t["sx"], t["sy"], t["sw"] = nt["sx"], nt["sy"], nt["sw"]
                        t["n"] += nt["n"]
                        t["views"].update(nt["views"])
                        t["shot"] = t["shot"] or nt["shot"]
                    p.map.remove_target(nid)
                    new_ids.remove(nid)
                    break
        for tid, n0 in before.items():
            t = p.map.targets.get(tid)
            if t is None:
                continue
            t["swept"] = True
            if t["n"] > n0:
                t["verified"] = True
                t["confirmed"] = True
                p.log(f"confirmed {tid} (close look)")
            elif not t.get("confirmed") and not t["shot"]:
                p.map.remove_target(tid)
                p.log(f"dropped {tid}: not there on the close look")

    def telemetry(self):
        """Live values shown on the mission panel."""
        left, right = self.ir.latest()
        return {
            "odom": f"({self.pos_x:+.2f}, {self.pos_y:+.2f}) m",
            "yaw": f"{self.current_yaw:+.1f} deg",
            # numeric values for the sensor widget
            "tof_mm": self.current_tof_dist_mm,
            "ir_left_cm": left,
            "ir_right_cm": right,
            "ir_max_cm": self.ir.max_cm,
            "ir_wall_cm": self.ir.wall_cm,
            "cam_wall_m": self._camera_wall_ahead(),
            "wall_rise_m": self.wall_rise_m,
        }

    def _draw_live_with_gimbal(self, gimbal_relative_yaw):
        if self.panel is not None:
            self.panel.map.gimbal_rel = gimbal_relative_yaw
        ctx = getattr(self, "_live_ctx", None)
        if ctx is None:
            return
        pos, visited, max_x, max_y, heading = ctx
        deg_map = {0: 0, 1: 90, 2: 180, 3: 270}
        abs_deg = (deg_map.get(heading, 0) + gimbal_relative_yaw) % 360
        self.draw_live_grid(pos, visited, max_x, max_y, gimbal_abs_deg=abs_deg)

    def draw_live_grid(self, current_pos, visited_set, max_x=3, max_y=3, gimbal_abs_deg=None):
        if self.panel is not None:
            ctx = getattr(self, "_live_ctx", None)
            heading = ctx[4] if ctx else None
            self.panel.update_robot(current_pos, heading, gimbal_abs_deg, visited_set)
            return

        cell_px = 120
        width = (max_x + 1) * cell_px
        height = (max_y + 1) * cell_px

        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        for r in range(max_y + 1):
            for c in range(max_x + 1):
                plot_y = max_y - r
                x1 = c * cell_px
                y1 = plot_y * cell_px
                x2 = x1 + cell_px
                y2 = y1 + cell_px

                if (c, r) in visited_set:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (229, 239, 247), -1)

                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(img, f"({c},{r})", (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        cur_c, cur_r = current_pos
        plot_cur_y = max_y - cur_r
        center_x = cur_c * cell_px + cell_px // 2
        center_y = plot_cur_y * cell_px + cell_px // 2
        cv2.circle(img, (center_x, center_y), 30, (0, 0, 255), -1)
        cv2.putText(img, "ROBOT", (center_x - 26, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        if gimbal_abs_deg is not None:
            rad = math.radians(gimbal_abs_deg)
            arrow_len = cell_px * 0.42
            tip_x = int(center_x + math.sin(rad) * arrow_len)
            tip_y = int(center_y - math.cos(rad) * arrow_len)
            cv2.arrowedLine(img, (center_x, center_y), (tip_x, tip_y),
                            (255, 140, 0), 3, tipLength=0.35)

        cv2.imshow("SLAM Real-time Grid Monitor", img)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Round 2: drive straight to the targets saved in round 1
    # ------------------------------------------------------------------
    def navigate_to_targets(self, round_data):
        """Round 2: plan the quickest route to shoot every selected round-1 card, drive it,
        and re-plan whenever something changes.

        Five route algorithms (greedy nearest, greedy cover, DFS branch & bound, BFS state
        search, Dijkstra with turns - route_planner.RoutePlanner) run on the round-1 map
        and are scored with the same time model (moves, turns, stops, aiming); the cheapest
        wins. After every stop - or a blocked passage, or a missed target - the plan is
        made again from where the robot is, with what is left."""
        from route_planner import GridGraph, parse_edges, plan_best, driven_edges, MOVES as RMOVES
        from target_vision import kind_of

        panel = self.panel
        print("--- Round 2: นำทางไปยังเป้าที่บันทึกไว้ ---")
        self.calibrate_yaw_sign()

        nx, ny = round_data["grid_size"]
        tile = round_data.get("tile_m", self.default_distance)
        cell_m = self.config.get("movement", {}).get("distance", tile)
        driven = driven_edges(round_data)   # a wall on an edge the robot drove through is a misreading
        graph = GridGraph(nx, ny, parse_edges(round_data.get("walls")) - driven,
                          parse_edges(round_data.get("open_edges")) | driven)
        max_m = panel.detector.max_shoot_m
        pos = tuple(round_data.get("start", [0, 0]))
        start = pos
        heading = 0
        visited = {pos}
        deg = {0: 0, 1: 90, 2: 180, 3: 270}

        for t in round_data.get("targets", []):
            t.setdefault("kind", kind_of(t["color"], t["shape"]))   # files from before kinds
            t.setdefault("id", t["kind"])
        remaining = [t for t in round_data.get("targets", []) if t["kind"] in panel.selected]
        for k in sorted(panel.selected - {t["kind"] for t in remaining}):
            panel.log(f"{k} was not found in round 1 - skipped")
        exclude = {t["id"]: set() for t in remaining}   # firing cells that did not work, per target
        self.route_info = None

        def show():
            self._live_ctx = (pos, visited, nx - 1, ny - 1, heading)
            self.draw_live_grid(pos, visited, nx - 1, ny - 1, gimbal_abs_deg=deg[heading])

        def time_left():  # also waits here while the round is paused
            return panel.checkpoint() and panel.remaining() > 0

        def plan():
            res = plan_best(graph, remaining, pos, heading, tile, max_m, exclude)
            legs = []
            prev = pos
            for cell, idx in res["stops"]:
                lg_path = res["route"]  # full route for drawing
                for i in idx:
                    t = remaining[i]
                    a = math.degrees(math.atan2(t["x_m"] - (cell[0] + 0.5) * tile,
                                                t["y_m"] - (cell[1] + 0.5) * tile)) % 360
                    legs.append({"path": [prev, cell], "fire_cell": cell, "aim_deg": a, "color": t["color"]})
                prev = cell
            if res["stops"]:
                panel.map.set_plan([{"path": res["route"], "fire_cell": l["fire_cell"], "aim_deg": l["aim_deg"],
                                     "color": l["color"]} for l in legs] or [])
            return res

        show()
        first = True
        try:
            while remaining and time_left():
                res = plan()
                if not res["stops"]:
                    panel.log("no reachable firing spot for: " + ", ".join(t["id"] for t in remaining))
                    break
                if first or res["algorithm"] != (self.route_info or {}).get("algorithm"):
                    cmp_ = "  ".join(f"{r['algorithm']} {r['time_s']:.0f}s" for r in res["comparison"] if r.get("ok"))
                    panel.log(f"route: {res['algorithm']} - {res['moves']} moves, {res['turns']} turns, "
                              f"~{res['time_s']:.0f} s  |  {cmp_}")
                self.route_info = {k: res[k] for k in ("algorithm", "time_s", "moves", "turns", "comparison")}
                if first:
                    self.route_first, self.route_replans = self.route_info, 0   # the whole-round plan
                    panel.route_plan = self.route_info
                else:
                    self.route_replans += 1
                first = False
                cell, idx = res["stops"][0]
                stop_targets = [remaining[i] for i in idx]

                # drive to the first stop (quickest path, turns included)
                path = res["route"][:res["route"].index(tuple(cell)) + 1] if tuple(cell) in res["route"] else [pos]
                blocked = False
                for nxt in path[1:]:
                    if not time_left():
                        break
                    d = next(h for h, m in RMOVES.items() if m == (nxt[0] - pos[0], nxt[1] - pos[1]))
                    yaw = self.heading_to_yaw(d)
                    self.turn_to_absolute_yaw(yaw)
                    heading = d
                    show()
                    if self.safe_move_forward(distance=cell_m, target_heading_deg=yaw):
                        pos = nxt
                        visited.add(pos)
                        graph.open.add(frozenset((path[path.index(nxt) - 1], nxt)))
                        show()
                    else:
                        panel.log(f"blocked {pos}->{nxt}, re-planning")
                        edge = frozenset((pos, nxt))
                        graph.walls.add(edge)
                        graph.open.discard(edge)
                        panel.map.walls.add(edge)
                        blocked = True
                        break
                if blocked or pos != tuple(cell):
                    continue            # re-plan from here (or time is up)

                # at the stop: shoot every target planned from here
                for t in stop_targets:
                    if not time_left():
                        break
                    a = math.degrees(math.atan2(t["x_m"] - (pos[0] + 0.5) * tile, t["y_m"] - (pos[1] + 0.5) * tile))
                    rel = wrap180(a - deg[heading])
                    near = dict(kind=t["kind"], x_m=t["x_m"], y_m=t["y_m"], radius_m=0.6)
                    done = False
                    for off in (0, -15, 15, -30, 30):
                        if not time_left():
                            break
                        g = wrap180(rel + off)
                        self.ep_gimbal.moveto(pitch=0, yaw=g, yaw_speed=180).wait_for_completed()
                        time.sleep(0.15)
                        self._draw_live_with_gimbal(g)
                        self._look_for_targets(g)
                        if self.shooter is None or not self.shooter.armed:   # dry run: seeing it is enough
                            done = panel.map.card_near(**near) is not None
                        else:
                            done = panel.map.card_near(shot=True, **near) is not None
                        if done:
                            break
                    if done:
                        remaining.remove(t)
                    else:
                        exclude[t["id"]].add(tuple(pos))
                        panel.log(f"{t['id']} not hit from {pos}, trying another spot")
                        if len(exclude[t["id"]]) >= 3:
                            panel.log(f"giving up on {t['id']}")
                            remaining.remove(t)
                self.reset_gimbal()
        except KeyboardInterrupt:
            print("\n--> ยกเลิกโดยผู้ใช้")
        finally:
            try:
                self._stop(0.2)
                self.reset_gimbal()
            except KeyboardInterrupt:
                pass
        if not time_left() and remaining:
            panel.log("time is up" if panel.remaining() <= 0 else "stopped")
        panel.map.set_plan([])
        return {
            "start_grid": list(start),
            "end_grid": list(pos),
            "visited_cells": len(visited),
            "remaining_targets": [t["id"] for t in remaining],
            "route": getattr(self, "route_first", None) or self.route_info,
            "replans": getattr(self, "route_replans", 0),
        }

    # ------------------------------------------------------------------
    # Explore
    # ------------------------------------------------------------------
    def explore_and_map_all(self):
        print("--- เริ่มการสำรวจและสร้างแผนที่ (Robust Grid Exploration) ---")

        self.calibrate_yaw_sign()

        data_cfg = self.config.get("data_collection", {})
        files_cfg = data_cfg.get("files", {})
        data_dir = data_cfg.get("data_dir", "data/raw/run1")
        os.makedirs(data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")

        filename_key = files_cfg.get("exploration", "exploration_map_data")
        csv_path = os.path.join(data_dir, f"log_{date_str}_{filename_key}.csv")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "unix_timestamp", "grid_x", "grid_y",
                "real_x_m", "real_y_m", "heading", "heading_deg",
                "front_tof_mm", "right_ir_cm", "back_ir_cm", "left_ir_cm", "action",
                "odom_x_m", "odom_y_m", "yaw_deg"   # คอลัมน์ใหม่ท้ายสุด ของจริงจากล้อ/IMU
            ])

        def log_step(g_x, g_y, h, tof_val, r_val, b_val, l_val, act_label="VISIT", c_size=0.6):
            deg_map = {0: 0, 1: 90, 2: 180, 3: 270}
            with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
                c_writer = csv.writer(file)
                c_writer.writerow([
                    time.time(), g_x, g_y,
                    round(g_x * c_size, 3), round(g_y * c_size, 3),
                    h, deg_map.get(h, 0),
                    round(tof_val, 1),
                    round(r_val / 10.0, 2),
                    round(b_val / 10.0, 2),
                    round(l_val / 10.0, 2),
                    act_label,
                    round(self.pos_x, 3), round(self.pos_y, 3), round(self.current_yaw, 1)
                ])

        if not hasattr(self, "current_tof_dist_mm"):
            self.current_tof_dist_mm = 10000

        CELL_SIZE = self.config.get("movement", {}).get("distance", 0.6)
        FRONT_WALL_MM = self.wall_mm   # same for every direction: the ToF turns with the gimbal
        SIDE_OPEN_MM = self.wall_mm

        visited = set()
        stack = []
        blocked = set()
        scan_cache = {}       # (x, y) -> {map dir: ToF mm} (scanned once per cell)
        entry_heading = {}    # (x, y) -> heading when the robot first arrived there

        grid_cfg = self.config.get("grid_map", {})
        MAX_X = grid_cfg.get("max_x", 3)
        MAX_Y = grid_cfg.get("max_y", 3)
        total_cells = (MAX_X + 1) * (MAX_Y + 1)

        start_cfg = grid_cfg.get("start", {"x": 0, "y": 0})
        start_x = start_cfg.get("x", 0)
        start_y = start_cfg.get("y", 0)

        x, y = start_x, start_y
        heading = 0
        moves = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        def print_summary(end_x, end_y):
            coverage_pct = round(100.0 * len(visited) / total_cells, 2)
            print("\n" + "=" * 50)
            print("📊 สรุปผลภารกิจ SLAM Explore สำเร็จ:")
            print(f"   - จุดเริ่มต้น (Start Position): [{start_x}, {start_y}]")
            print(f"   - จุดสิ้นสุด (End Position):   [{end_x}, {end_y}]")
            print(f"   - พื้นที่สำรวจทั้งหมด (Coverage): {coverage_pct}% ({len(visited)}/{total_cells} cells)")
            print("=" * 50 + "\n")

        try:
            while True:
                if self.panel is not None and not self.panel.checkpoint():  # waits while paused
                    print("\n--> ยกเลิกการสำรวจจากหน้าต่าง Mission Panel")
                    print_summary(x, y)
                    break
                if self.panel is not None and self.panel.round_t0 and self.panel.remaining() <= 0:
                    print("\n--> หมดเวลารอบนี้แล้ว หยุดสำรวจ")
                    self.panel.log("time is up - stopping exploration")
                    print_summary(x, y)
                    break
                self._stop(0.4)

                visited.add((x, y))
                print(f"\n[Map] พิกัดปัจจุบัน: ({x}, {y}) | ทิศหันหน้า: {heading}")

                self._live_ctx = ((x, y), visited, MAX_X, MAX_Y, heading)
                self.draw_live_grid((x, y), visited, MAX_X, MAX_Y,
                                    gimbal_abs_deg={0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0))
                entry_heading.setdefault((x, y), heading)   # side order is relative to this

                if (x, y) in scan_cache:
                    # back in a cell already scanned (after a side branch): reuse it, no new scan
                    surrounding = {rel: scan_cache[(x, y)][(heading + off) % 4]
                                   for rel, off in (("front", 0), ("right", 1), ("back", 2), ("left", 3))}
                    rescanned = False
                else:
                    surrounding = self.scan_surroundings_with_gimbal()
                    rescanned = True

                    # จัดแนวกับกำแพง ก่อนตัดสินใจ/เดินต่อ
                    if self.align_enabled:
                        if self.align_to_walls(surrounding, heading):
                            self.reset_gimbal()
                            time.sleep(0.15)
                            self.align_front_distance()
                    scan_cache[(x, y)] = {(heading + off) % 4: surrounding[rel]
                                          for rel, off in (("front", 0), ("right", 1), ("back", 2), ("left", 3))}

                if self.panel is not None and rescanned:
                    self.panel.map.mark_scan((x, y), heading, surrounding, FRONT_WALL_MM, SIDE_OPEN_MM)
                if self.panel is not None and self.verify_enabled:
                    self.verify_sweep((x, y))   # something seen in / next to this block: look down, turn round
                    if self.panel.round_no >= 2 and self.panel.all_designated_shot():
                        print("\n--> [Round 2] ยิงเป้าที่กำหนดครบแล้ว หยุดภารกิจ")
                        print_summary(x, y)
                        break

                front_dist = surrounding["front"]
                right_dist = surrounding["right"]
                back_dist = surrounding["back"]
                left_dist = surrounding["left"]

                log_step(x, y, heading, front_dist, right_dist, back_dist, left_dist,
                         act_label="VISIT", c_size=CELL_SIZE)

                if len(visited) >= total_cells:
                    print_summary(x, y)
                    break

                # which ways are open (map directions), tried in explore_order relative to
                # the heading the robot first entered this cell with: e.g. a cell with open
                # cells left and right and a way ahead -> left side first, back, right side,
                # back, then the next cell ahead (no long backtrack for the side cells later)
                is_open = {d: scan_cache[(x, y)][d] > self.wall_mm for d in range(4)}
                eh = entry_heading[(x, y)]
                rel_dir = {"front": eh, "right": (eh + 1) % 4, "back": (eh + 2) % 4, "left": (eh + 3) % 4}
                open_dirs = [rel_dir[r] for r in self.explore_order if is_open[rel_dir[r]]]

                unvisited = []
                for d in open_dirs:
                    tx = x + moves[d][0]
                    ty = y + moves[d][1]
                    if (x, y, d) in blocked:
                        continue
                    if 0 <= tx <= MAX_X and 0 <= ty <= MAX_Y and (tx, ty) not in visited:
                        unvisited.append(d)

                if unvisited:
                    next_heading = unvisited[0]
                    stack.append((x, y, heading))

                    target_yaw = self.heading_to_yaw(next_heading)
                    self.turn_to_absolute_yaw(target_yaw)
                    heading = next_heading

                    success = self.safe_move_forward(distance=CELL_SIZE, target_heading_deg=target_yaw)
                    if success:
                        x += moves[next_heading][0]
                        y += moves[next_heading][1]
                    else:
                        print("-> [Obstacle] ชนสิ่งกีดขวางกลางทาง ยกเลิกเส้นทางนี้")
                        self._log(f"move {(x, y)} -> {(x + moves[next_heading][0], y + moves[next_heading][1])} "
                                  f"gave up: {self.last_move_note or 'blocked'}")
                        log_step(x, y, heading, self.current_tof_dist_mm, 0, 0, 0,
                                 act_label=f"BLOCKED ({self.last_move_note or 'blocked'})", c_size=CELL_SIZE)
                        blocked.add((x, y, next_heading))
                        stack.pop()
                else:
                    if not stack:
                        print_summary(x, y)
                        break

                    prev_x, prev_y, _prev_heading = stack.pop()
                    dx = prev_x - x
                    dy = prev_y - y
                    target_heading = 0
                    for h, m in moves.items():
                        if m == (dx, dy):
                            target_heading = h
                            break

                    target_yaw = self.heading_to_yaw(target_heading)
                    heading = target_heading
                    # the way back was driven before, so it is open: no camera guard, a few tries
                    moved = False
                    for attempt in range(3):
                        self.turn_to_absolute_yaw(target_yaw)
                        if self.safe_move_forward(distance=CELL_SIZE, target_heading_deg=target_yaw,
                                                  camera_guard=False):
                            moved = True
                            break
                        self._log(f"backtrack to {(prev_x, prev_y)} failed ({self.last_move_note}), retry {attempt + 1}/3")
                    if not moved:
                        # never pretend: the map would think the robot is somewhere it is not
                        self._log(f"cannot drive back to {(prev_x, prev_y)} - stopping here at {(x, y)}")
                        log_step(x, y, heading, self.current_tof_dist_mm, 0, 0, 0,
                                 act_label=f"STUCK ({self.last_move_note})", c_size=CELL_SIZE)
                        print_summary(x, y)
                        break
                    x, y = prev_x, prev_y

                    log_step(x, y, heading, self.current_tof_dist_mm, 300.0, 300.0, 300.0,
                             act_label="RETRACE", c_size=CELL_SIZE)

        except KeyboardInterrupt:
            print("\n--> ยกเลิกการสำรวจโดยผู้ใช้")
        finally:
            try:
                self._stop(0.2)
                self.reset_gimbal()
            except KeyboardInterrupt:
                pass

        report = {
            "start_grid": [start_x, start_y],
            "end_grid": [x, y],
            "visited_cells": len(visited),
            "total_grid_cells": total_cells,
            "coverage_percent": round(100.0 * len(visited) / total_cells, 2) if total_cells else 0.0,
            "exploration_log_csv": csv_path,
        }
        return report