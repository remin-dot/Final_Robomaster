import csv
import os
import time
from datetime import datetime


class GimbalLidarController:
    def __init__(self, ep_robot, config):
        self.ep_robot = ep_robot
        self.ep_gimbal = ep_robot.gimbal
        self.ep_sensor = ep_robot.sensor

        self.config = config
        self.lidar_cfg = config.get("lidar_scan", {})
        self.data_dir = config["data_collection"]["data_dir"]

        os.makedirs(self.data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        lidar_file_name = config["data_collection"]["files"].get("lidar", "gimbal_lidar_data")
        self.csv_filepath = os.path.join(self.data_dir, f"log_{date_str}_{lidar_file_name}.csv")

        self.current_distance_mm = 0
        self.current_yaw_deg = 0.0

    def _sensor_data_handler(self, sub_info):
        """รับค่าระยะทางจาก ToF"""
        if isinstance(sub_info, (list, tuple)):
            self.current_distance_mm = sub_info[0]
        else:
            self.current_distance_mm = sub_info

    def _angle_data_handler(self, angle_info):
        """รับค่าองศาปัจจุบันของ Gimbal (pitch, yaw, pitch_ground, yaw_ground)"""
        # angle_info[1] คือ yaw_angle สัมพันธ์กับตัวหุ่น
        self.current_yaw_deg = angle_info[1]

    def save_to_csv(self, scan_data, filepath=None):
        target_path = filepath or self.csv_filepath
        try:
            with open(target_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["yaw_deg", "distance_mm"])
                writer.writerows(scan_data)
            print(f"[+] บันทึกข้อมูล LiDAR {len(scan_data)} จุดลงใน '{target_path}' สำเร็จ")
        except Exception as e:
            print(f"[-] เกิดข้อผิดพลาดในการบันทึก CSV: {e}")

    def scan(self):
        start_yaw = self.lidar_cfg.get("start_yaw", -75)
        end_yaw = self.lidar_cfg.get("end_yaw", 75)
        yaw_speed = self.lidar_cfg.get("yaw_speed", 25) # แนะนำความเร็ว 20-30 deg/s สำหรับ sweep
        pitch = self.lidar_cfg.get("pitch", 0)
        min_dist = self.lidar_cfg.get("min_valid_dist_mm", 100)
        max_dist = self.lidar_cfg.get("max_valid_dist_mm", 10000)

        collected_data = []

        # 1. สมัครรับข้อมูลทั้ง ToF และ มุม Gimbal Yaw
        self.ep_sensor.sub_distance(freq=20, callback=self._sensor_data_handler)
        self.ep_gimbal.sub_angle(freq=20, callback=self._angle_data_handler)
        time.sleep(0.5)

        print(f"[*] กำลังเคลื่อนที่ไปยังจุดเริ่มต้น {start_yaw}°...")
        action = self.ep_gimbal.moveto(pitch=pitch, yaw=start_yaw, pitch_speed=30, yaw_speed=50)
        action.wait_for_completed(timeout=3.0)
        time.sleep(0.5)

        print(f"[*] กำลังกวาดสแกนอย่างต่อเนื่องจาก {start_yaw}° ไปยัง {end_yaw}°...")

        # 2. สั่งกิมบอลหมุนด้วยความเร็วคงที่ drive_speed (ไม่ใช่ Action Command จึงไม่ชนกัน)
        sweep_speed = abs(yaw_speed) if end_yaw > start_yaw else -abs(yaw_speed)
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=sweep_speed)

        last_recorded_yaw = -999
        # 3. วนลูปอ่านค่าระหว่างที่กิมบอลกำลังหมุน
        while True:
            yaw = self.current_yaw_deg
            dist = self.current_distance_mm

            # เช็คเงื่อนไขเมื่อหมุนถึงปลายทาง
            if (sweep_speed > 0 and yaw >= end_yaw) or (sweep_speed < 0 and yaw <= end_yaw):
                break

            # บันทึกข้อมูลทุกๆ 1-2 องศาเพื่อไม่ให้จุดซ้ำซ้อน
            if abs(yaw - last_recorded_yaw) >= 1.5:
                if min_dist < dist < max_dist:
                    collected_data.append((round(yaw, 1), dist))
                    last_recorded_yaw = yaw

            time.sleep(0.02)

        # 4. หยุดการหมุนทันที
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        time.sleep(0.2)

        # 5. ยกเลิก sub และคืนตำแหน่งตรงกลาง
        self.ep_sensor.unsub_distance()
        self.ep_gimbal.unsub_angle()

        recenter_action = self.ep_gimbal.recenter(pitch_speed=50, yaw_speed=50)
        recenter_action.wait_for_completed(timeout=2.5)

        print(f"[*] สแกนเสร็จสิ้น ได้รับข้อมูล {len(collected_data)} จุด")
        return collected_data