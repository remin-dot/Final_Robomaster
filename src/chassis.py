import csv
import time
import os
import math
from datetime import datetime
import cv2
import numpy as np

try:
    from robomaster import robot
except Exception:
    robot = None


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class ChassisController:
    # ---- ความปลอดภัยข้าง (Sharp IR) ----
    SIDE_SAFE_CM = 15.0     # เริ่มเลื่อนหลบ (กันไว้ก่อนถึง 10cm)
    SIDE_DANGER_CM = 12.0   # ชะลอเดินหน้า
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
        self._ir_ts = 0.0
        self._left_cm = 80.0
        self._right_cm = 80.0

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

    # ------------------------------------------------------------------
    # Sharp IR
    # ------------------------------------------------------------------
    def adc_to_distance_cm(self, adc_val):
        """ADC -> cm (ช่วง 10-80cm) ต่ำกว่า 10cm ค่าจะเพี้ยน จึงกันไว้ที่ SIDE_SAFE_CM"""
        if adc_val is None or adc_val <= 0:
            return 80.0
        voltage = (adc_val / 1023.0) * 3.3
        if voltage <= 0.4:
            return 80.0
        distance_cm = 27.28 * (voltage ** -1.20)
        return min(round(distance_cm, 1), 80.0)

    def read_side_ir(self):
        now = time.time()
        if now - self._ir_ts >= 0.1:
            try:
                l_adc = self.ep_adaptor.get_adc(id=1, port=2)
                r_adc = self.ep_adaptor.get_adc(id=2, port=1)
                self._left_cm = self.adc_to_distance_cm(l_adc)
                self._right_cm = self.adc_to_distance_cm(r_adc)
            except Exception:
                pass
            self._ir_ts = now
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
        self.ep_chassis.sub_attitude(freq=self.freq_att, callback=self.handle_attitude)
        self.ep_chassis.sub_imu(freq=self.freq_imu, callback=self.handle_imu)
        self.ep_chassis.sub_esc(freq=self.freq_esc, callback=self.handle_esc)
        self.ep_sensor.sub_distance(freq=self.freq_dist, callback=self.handle_distance)

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
        print(f"    yaw_origin={y0:.1f}° (ทิศเหนือของแผนที่)")

    # ------------------------------------------------------------------
    # เดินตรง: distance วัดจาก odometry ของล้อ (ไม่ใช้เวลา) + IMU คุมทิศ + IR คุมข้าง
    # ------------------------------------------------------------------
    def safe_move_forward(self, distance=0.6, speed=0.3, stop_limit_mm=None, target_heading_deg=None):
        if stop_limit_mm is None:
            stop_limit_mm = self.FRONT_STOP_MM
        print(f"--> [Safe Move] เดินหน้า {distance}m ที่ {speed}m/s")
        self.reset_gimbal()

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
                    print(f"!!! [สิ่งกีดขวาง] {front_dist}mm หยุดทันที")
                    self._retreat(traveled, sx, sy, pos_ok, speed)
                    return False

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
                elif left_cm < 40 and right_cm < 40:
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
    def turn_to_absolute_yaw(self, target_yaw):
        print(f"--> [Turn] หมุนไป {target_yaw:.1f}° (ปัจจุบัน {self.current_yaw:.1f}°)")
        self._stop(0.2)

        start_time = time.time()
        while (time.time() - start_time) < 4.0:
            error = wrap180(target_yaw - self.current_yaw)
            if abs(error) < 2.0:
                break

            z_speed = error * 1.2
            if z_speed > 0:
                z_speed = max(min(z_speed, 45), 10)
            else:
                z_speed = min(max(z_speed, -45), -10)

            self.ep_chassis.drive_speed(x=0, y=0, z=self.z_sign * z_speed)
            time.sleep(0.05)

        self._stop(0.4)

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
            time.sleep(0.15)
            distances[label] = self.current_tof_dist_mm
            self._draw_live_with_gimbal(yaw)

        self.reset_gimbal()
        self._draw_live_with_gimbal(0)
        return distances

    def _draw_live_with_gimbal(self, gimbal_relative_yaw):
        ctx = getattr(self, "_live_ctx", None)
        if ctx is None:
            return
        pos, visited, max_x, max_y, heading = ctx
        deg_map = {0: 0, 1: 90, 2: 180, 3: 270}
        abs_deg = (deg_map.get(heading, 0) + gimbal_relative_yaw) % 360
        self.draw_live_grid(pos, visited, max_x, max_y, gimbal_abs_deg=abs_deg)

    def draw_live_grid(self, current_pos, visited_set, max_x=3, max_y=3, gimbal_abs_deg=None):
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
        FRONT_WALL_MM = 300
        SIDE_OPEN_MM = 450

        visited = set()
        stack = []
        blocked = set()

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
                self._stop(0.4)

                visited.add((x, y))
                print(f"\n[Map] พิกัดปัจจุบัน: ({x}, {y}) | ทิศหันหน้า: {heading}")

                self._live_ctx = ((x, y), visited, MAX_X, MAX_Y, heading)
                self.draw_live_grid((x, y), visited, MAX_X, MAX_Y,
                                    gimbal_abs_deg={0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0))

                surrounding = self.scan_surroundings_with_gimbal()

                # จัดแนวกับกำแพง ก่อนตัดสินใจ/เดินต่อ
                if self.align_enabled:
                    if self.align_to_walls(surrounding, heading):
                        self.reset_gimbal()
                        time.sleep(0.15)
                        self.align_front_distance()

                front_dist = surrounding["front"]
                right_dist = surrounding["right"]
                back_dist = surrounding["back"]
                left_dist = surrounding["left"]

                log_step(x, y, heading, front_dist, right_dist, back_dist, left_dist,
                         act_label="VISIT", c_size=CELL_SIZE)

                if len(visited) >= total_cells:
                    print_summary(x, y)
                    break

                open_dirs = []
                if front_dist > FRONT_WALL_MM:
                    open_dirs.append(heading)
                if right_dist > SIDE_OPEN_MM:
                    open_dirs.append((heading + 1) % 4)
                if left_dist > SIDE_OPEN_MM:
                    open_dirs.append((heading + 3) % 4)
                if back_dist > SIDE_OPEN_MM:
                    open_dirs.append((heading + 2) % 4)

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
                    self.turn_to_absolute_yaw(target_yaw)
                    heading = target_heading

                    self.safe_move_forward(distance=CELL_SIZE, target_heading_deg=target_yaw)
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