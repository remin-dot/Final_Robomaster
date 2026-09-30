import os
import sys
import json
import time
import threading
import cv2
import numpy as np
from robomaster import robot

CONFIG_PATH = os.path.join("config", "color_config.json")

# ลำดับสีเป้าหมาย 4 สี (แดง -> เขียว -> ฟ้า -> เหลือง)
COLOR_SEQUENCE = ['red', 'green', 'blue', 'yellow']

DEFAULT_COLORS = {
    'red':    {'h_min': 165, 'h_max': 10,  's_min': 100, 's_max': 255, 'v_min': 60, 'v_max': 255, 'min_area': 1000},
    'green':  {'h_min': 35,  'h_max': 85,  's_min': 80,  's_max': 255, 'v_min': 50, 'v_max': 255, 'min_area': 1000},
    'blue':   {'h_min': 85,  'h_max': 132, 's_min': 80,  's_max': 255, 'v_min': 50, 'v_max': 255, 'min_area': 1000},
    'yellow': {'h_min': 18,  'h_max': 34,  's_min': 90,  's_max': 255, 'v_min': 80, 'v_max': 255, 'min_area': 1000}
}

def nothing(x):
    pass

# =========================================================================
# ⚙️ ตัวควบคุม PID ความแม่นยำสูงพร้อมระบบชดเชยแรงเสียดทาน (Anti-Friction)
# =========================================================================
class PrecisionPID:
    def __init__(self, kp=0.075, ki=0.012, kd=0.005, min_speed=3.0, max_speed=35.0, deadband=8):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.deadband = deadband

        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05
        self.last_time = now

        if abs(error) <= self.deadband:
            self.integral = 0.0
            self.prev_error = 0.0
            return 0.0

        self.integral += error * dt
        self.integral = np.clip(self.integral, -40, 40)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        if abs(output) > 0.3:
            sign = 1.0 if output > 0 else -1.0
            magnitude = max(abs(output), self.min_speed)
            output = sign * min(magnitude, self.max_speed)

        return float(output)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()


# =========================================================================
# 🎯 ระบบตรวจจับสีและรูปทรงพร้อมตัวกรองสภาพแวดล้อมโฟมขาว
# =========================================================================
class ShapeAndColorDetector:
    def __init__(self, bottom_ignore_ratio=0.20):
        self.bottom_ignore_ratio = bottom_ignore_ratio
        self.colors_data = self.load_config()
        self.active_color = COLOR_SEQUENCE[0]
        self.saved_feedback_timer = 0
        self.kernel = np.ones((5, 5), np.uint8)

        self.draw_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 120, 0),
            'yellow': (0, 255, 255)
        }

        self.setup_gui()

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_COLORS.copy()

    def save_config(self):
        os.makedirs("config", exist_ok=True)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.colors_data, f, indent=4)
        self.saved_feedback_timer = time.time()
        print(f"[+] บันทึกการตั้งค่าสี {self.active_color.upper()} เรียบร้อยแล้ว!")

    def setup_gui(self):
        cv2.namedWindow("Tuner & Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tuner & Calibration", 420, 480)
        c = self.colors_data[self.active_color]

        cv2.createTrackbar("H Min", "Tuner & Calibration", c['h_min'], 179, nothing)
        cv2.createTrackbar("H Max", "Tuner & Calibration", c['h_max'], 179, nothing)
        cv2.createTrackbar("S Min", "Tuner & Calibration", c['s_min'], 255, nothing)
        cv2.createTrackbar("S Max", "Tuner & Calibration", c['s_max'], 255, nothing)
        cv2.createTrackbar("V Min", "Tuner & Calibration", c['v_min'], 255, nothing)
        cv2.createTrackbar("V Max", "Tuner & Calibration", c['v_max'], 255, nothing)
        cv2.createTrackbar("Min Area", "Tuner & Calibration", c.get('min_area', 1000), 10000, nothing)

        cv2.createTrackbar("Offset X", "Tuner & Calibration", 100, 200, nothing)
        cv2.createTrackbar("Offset Y", "Tuner & Calibration", 100, 200, nothing)

    def update_trackbars_for_color(self, color_name):
        self.active_color = color_name
        c = self.colors_data[color_name]
        cv2.setTrackbarPos("H Min", "Tuner & Calibration", c['h_min'])
        cv2.setTrackbarPos("H Max", "Tuner & Calibration", c['h_max'])
        cv2.setTrackbarPos("S Min", "Tuner & Calibration", c['s_min'])
        cv2.setTrackbarPos("S Max", "Tuner & Calibration", c['s_max'])
        cv2.setTrackbarPos("V Min", "Tuner & Calibration", c['v_min'])
        cv2.setTrackbarPos("V Max", "Tuner & Calibration", c['v_max'])
        cv2.setTrackbarPos("Min Area", "Tuner & Calibration", c.get('min_area', 1000))

    def get_trackbar_values(self):
        self.colors_data[self.active_color] = {
            'h_min': cv2.getTrackbarPos("H Min", "Tuner & Calibration"),
            'h_max': cv2.getTrackbarPos("H Max", "Tuner & Calibration"),
            's_min': cv2.getTrackbarPos("S Min", "Tuner & Calibration"),
            's_max': cv2.getTrackbarPos("S Max", "Tuner & Calibration"),
            'v_min': cv2.getTrackbarPos("V Min", "Tuner & Calibration"),
            'v_max': cv2.getTrackbarPos("V Max", "Tuner & Calibration"),
            'min_area': cv2.getTrackbarPos("Min Area", "Tuner & Calibration")
        }
        offset_x = cv2.getTrackbarPos("Offset X", "Tuner & Calibration") - 100
        offset_y = cv2.getTrackbarPos("Offset Y", "Tuner & Calibration") - 100
        return self.colors_data[self.active_color], offset_x, offset_y

    def classify_shape(self, contour, area):
        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return "Unknown"

        approx = cv2.approxPolyDP(contour, 0.038 * peri, True)
        vertices = len(approx)
        circularity = (4 * np.pi * area) / (peri * peri)

        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            return "Square" if (0.75 <= aspect_ratio <= 1.30) else "Rectangle"
        elif circularity >= 0.70:
            return "Circle"
        else:
            return "Polygon"

    def is_on_white_background(self, hsv_img, bbox, margin=15):
        img_h, img_w = hsv_img.shape[:2]
        x, y, bw, bh = bbox

        outer_x1 = max(0, x - margin)
        outer_y1 = max(0, y - margin)
        outer_x2 = min(img_w, x + bw + margin)
        outer_y2 = min(img_h, y + bh + margin)

        outer_roi = hsv_img[outer_y1:outer_y2, outer_x1:outer_x2]
        if outer_roi.size == 0:
            return True

        s_channel = outer_roi[:, :, 1]
        v_channel = outer_roi[:, :, 2]
        white_pixels = np.sum((s_channel < 85) & (v_channel > 90))
        total_pixels = outer_roi.shape[0] * outer_roi.shape[1]

        return (white_pixels / total_pixels) > 0.25

    def process_frame(self, frame):
        params, offset_x, offset_y = self.get_trackbar_values()
        h, w = frame.shape[:2]

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if params['h_min'] <= params['h_max']:
            lower = np.array([params['h_min'], params['s_min'], params['v_min']])
            upper = np.array([params['h_max'], params['s_max'], params['v_max']])
            mask = cv2.inRange(hsv, lower, upper)
        else:
            mask1 = cv2.inRange(hsv, np.array([0, params['s_min'], params['v_min']]),
                                     np.array([params['h_max'], params['s_max'], params['v_max']]))
            mask2 = cv2.inRange(hsv, np.array([params['h_min'], params['s_min'], params['v_min']]),
                                     np.array([179, params['s_max'], params['v_max']]))
            mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        ignore_h = int(h * (1.0 - self.bottom_ignore_ratio))
        mask[ignore_h:h, 0:w] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area'] or area > (h * w * 0.45):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh
            if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            if solidity < 0.82:
                continue

            shape_name = self.classify_shape(cnt, area)
            if shape_name in ["Unknown", "Polygon"]:
                continue

            if not self.is_on_white_background(hsv, (x, y, bw, bh)):
                continue

            rect = cv2.minAreaRect(cnt)
            cx, cy = int(rect[0][0]), int(rect[0][1])

            detections.append({
                'bbox': (x, y, bw, bh),
                'center': (cx, cy),
                'area': area,
                'shape': shape_name,
                'contour': cnt
            })

        detections.sort(key=lambda item: item['area'], reverse=True)
        return detections, mask, params, offset_x, offset_y

    def draw_hud(self, frame, detections, params, target_aim_point, auto_aim=False, is_locked=False, firing=False, yaw_invert=False, current_idx=0, is_mission_done=False):
        h, w = frame.shape[:2]
        ignore_h = int(h * (1.0 - self.bottom_ignore_ratio))
        aim_x, aim_y = target_aim_point

        # 1. โซนตัดส่วนล่างของตัวหุ่น
        cv2.line(frame, (0, ignore_h), (w, ignore_h), (0, 0, 200), 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, ignore_h), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, "[ROBOT BLIND ZONE ~10CM - EXCLUDED]", (w // 2 - 160, ignore_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1, cv2.LINE_AA)

        # 2. วาด Bounding Box
        draw_color = self.draw_colors.get(self.active_color, (0, 255, 0))
        for idx, det in enumerate(detections):
            x, y, bw, bh = det['bbox']
            cx, cy = det['center']
            shape = det['shape']

            box_col = (0, 0, 255) if (idx == 0 and is_locked) else draw_color
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_col, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 1)
            cv2.line(frame, (aim_x, aim_y), (cx, cy), (150, 150, 150), 1)

            label = f"{self.active_color.upper()} {shape.upper()}"
            cv2.rectangle(frame, (x, y - 24), (x + len(label) * 10, y), box_col, -1)
            cv2.putText(frame, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. เป้าเล็ง Crosshair
        cross_col = (0, 0, 255) if firing else ((0, 255, 0) if is_locked else ((0, 255, 255) if auto_aim else (255, 255, 255)))
        cv2.drawMarker(frame, (aim_x, aim_y), cross_col, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(frame, (aim_x, aim_y), 12, cross_col, 1)

        # 4. ลำดับเป้าหมาย
        if is_mission_done:
            cv2.putText(frame, "MISSION COMPLETE: ALL 4 TARGETS HIT! [Press R to Restart]", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            seq_text = f"TARGET [{current_idx + 1}/4]: {self.active_color.upper()}"
            cv2.putText(frame, seq_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

        aim_state = "ON" if auto_aim else "OFF [T]"
        invert_state = "REVERSED (-)" if yaw_invert else "NORMAL (+)"
        queue_str = ' -> '.join([c.upper() for c in COLOR_SEQUENCE])
        cv2.putText(frame, f"AUTO-AIM: {aim_state} | YAW: {invert_state} [I] | QUEUE: {queue_str}", 
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if len(detections) > 0 and not is_mission_done:
            tcx, tcy = detections[0]['center']
            err_x = tcx - aim_x
            err_y = tcy - aim_y
            err_color = (0, 255, 0) if is_locked else (0, 200, 255)
            cv2.putText(frame, f"ERROR -> X: {err_x:+3d} px | Y: {err_y:+3d} px", 
                        (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, err_color, 2)

        if firing:
            cv2.rectangle(frame, (w // 2 - 160, 40), (w // 2 + 160, 85), (0, 0, 255), -1)
            cv2.putText(frame, f">> TARGET {current_idx} HIT! <<", (w // 2 - 140, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        elif is_locked:
            cv2.putText(frame, "[CENTER LOCKED]", (w // 2 - 90, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if time.time() - self.saved_feedback_timer < 2.0:
            cv2.rectangle(frame, (10, 100), (260, 130), (0, 180, 0), -1)
            cv2.putText(frame, "CONFIG SAVED!", (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "[R]: Reset Sequence | [N]: Next Color | [1-4]: Choose Color | [T]: Toggle Aim | [F]: Shoot",
                    (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        return frame


# =========================================================================
# 💥 ฟังก์ชันสั่งยิงจริงของฮาร์ดแวร์ พร้อมเสียงลำโพงยืนยัน
# =========================================================================
def execute_real_fire(ep_robot, ep_blaster):
    """
    สั่งยิงจริงทางฮาร์ดแวร์:
    1. สั่งยิงกลไกกระสุนจริง BEAD_FIRE (ถ้ามีลูกเจลจะพุ่งออก)
    2. สั่งยิงแสง IR สำหรับเป้าเซนเซอร์
    3. เล่นเสียงเอฟเฟกต์ยิงปืน (SOUND_ID_SHOOT) จากลำโพงหุ่นยนต์
    """
    def _fire():
        try:
            # 1. ยิงลูกกระสุนเจลจริง (ถ้าใส่ลูกเจล มอเตอร์จะหมุนยิงออก)
            ep_blaster.fire(fire_type=robot.BEAD_FIRE, count=1)
        except Exception as e:
            print(f"[!] Bead Fire Error (ตรวจสอบกลไกกระสุน): {e}")

        try:
            # 2. ยิงลำแสงอินฟราเรด (IR Fire)
            ep_blaster.fire(fire_type=robot.IR_FIRE, count=1)
        except Exception as e:
            print(f"[!] IR Fire Error: {e}")

        try:
            # 3. เล่นเสียงสังเคราะห์ยิงปืนจากลำโพงของหุ่นยนต์ยืนยัน
            ep_robot.play_sound(robot.SOUND_ID_SHOOT)
            print("[💥 BLASTER EXECUTED] ลั่นไกยิงจริงและส่งเสียงสำเร็จ!")
        except Exception as e:
            print(f"[!] Sound Playback Error: {e}")

    threading.Thread(target=_fire, daemon=True).start()


# =========================================================================
# 🚀 MAIN CONTROLLER LOOP
# =========================================================================
def main():
    color_keys = {
        ord('1'): 0,  # red
        ord('2'): 1,  # green
        ord('3'): 2,  # blue
        ord('4'): 3   # yellow
    }

    calibrator = ShapeAndColorDetector(bottom_ignore_ratio=0.20)
    ep_robot = robot.Robot()
    stream_started = False
    MANUAL_STEP_DEG = 5

    pid_yaw = PrecisionPID(kp=0.075, ki=0.015, kd=0.006, min_speed=3.0, max_speed=35.0, deadband=8)
    pid_pitch = PrecisionPID(kp=0.080, ki=0.015, kd=0.006, min_speed=3.0, max_speed=25.0, deadband=8)

    auto_aim_mode = False
    yaw_invert = False

    REQUIRED_LOCK_FRAMES = 5
    lock_counter = 0

    target_color_index = 0
    mission_completed = False
    POST_SHOT_STABILIZE_TIME = 1.2
    post_shot_timer = 0

    last_fire_time = 0
    is_firing_visual = False
    fire_visual_timer = 0

    last_cmd_time = 0
    CMD_INTERVAL = 0.05

    try:
        print("[*] กำลังเชื่อมต่อ RoboMaster ในโหมด AP...")
        ep_robot.initialize(conn_type="ap", proto_type="udp")
        print("[+] เชื่อมต่อสำเร็จ!")

        ep_camera = ep_robot.camera
        ep_gimbal = ep_robot.gimbal
        ep_blaster = ep_robot.blaster

        # ทดสอบไฟเล็ง / LED หน้าลำกล้อง
        try:
            ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
            ep_blaster.set_led(brightness=255, effect=robot.EFFECT_ON)
        except Exception:
            pass

        ep_gimbal.recenter(pitch_speed=50, yaw_speed=50).wait_for_completed()
        ep_camera.start_video_stream(display=False, resolution="720p")
        stream_started = True

        print("\n=========================================================")
        print(" [R]              : รีเซ็ตและเริ่มรัน Sequence ใหม่ทันที (Start)")
        print(" [N]              : ข้ามไปสีถัดไป (Skip to Next Color)")
        print(" [1-4]            : บังคับเลือกสีเป้าหมาย (1=Red, 2=Green, 3=Blue, 4=Yellow)")
        print(" [T]              : เปิด/ปิด Auto-Aim")
        print(" [I]              : สลับทิศทางหมุนซ้าย-ขวา (Invert Yaw)")
        print(" [F]              : กดยิงจริงแบบ Manual (ทดสอบเสียงและกระบอกปืน)")
        print(" [P]              : บันทึกค่าสีลง JSON")
        print(" [W/A/S/D]        : หมุน Gimbal Manual")
        print(" [Space]          : หันกิมบอลกลับตรงกลาง")
        print(" [Q] หรือ [Esc]   : ออกจากโปรแกรม")
        print(f" ลำดับการยิง: {' -> '.join([c.upper() for c in COLOR_SEQUENCE])}")
        print("=========================================================\n")

        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                continue

            h_img, w_img = frame.shape[:2]
            now = time.time()

            # 1. ประมวลผลภาพ
            detections, mask, current_params, offset_x, offset_y = calibrator.process_frame(frame)
            
            aim_point = (w_img // 2 + offset_x, h_img // 2 + offset_y)
            aim_x, aim_y = aim_point

            is_locked = False

            # 2. ระบบ Auto-Aim พร้อมยิงจริง
            if auto_aim_mode and not mission_completed:
                if (now - post_shot_timer) < POST_SHOT_STABILIZE_TIME:
                    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                else:
                    if len(detections) > 0:
                        target = detections[0]
                        cx, cy = target['center']

                        dx = cx - aim_x
                        dy = cy - aim_y

                        in_center_x = abs(dx) <= pid_yaw.deadband
                        in_center_y = abs(dy) <= pid_pitch.deadband

                        if in_center_x and in_center_y:
                            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                            lock_counter += 1
                            is_locked = True

                            if lock_counter >= REQUIRED_LOCK_FRAMES:
                                current_shot_color = COLOR_SEQUENCE[target_color_index]
                                print(f"\n[💥 DIRECT HIT!] ป้าย {current_shot_color.upper()} อยู่กึ่งกลาง -> ยิงจริงทันที!")
                                
                                # สั่งยิงฮาร์ดแวร์จริง
                                execute_real_fire(ep_robot, ep_blaster)
                                
                                last_fire_time = now
                                is_firing_visual = True
                                fire_visual_timer = now
                                post_shot_timer = now
                                lock_counter = 0
                                
                                pid_yaw.reset()
                                pid_pitch.reset()

                                # เลื่อนคิวไปยังสีถัดไป
                                target_color_index += 1
                                if target_color_index < len(COLOR_SEQUENCE):
                                    next_color = COLOR_SEQUENCE[target_color_index]
                                    calibrator.update_trackbars_for_color(next_color)
                                    print(f"[🎯 NEXT TARGET] สลับเป้าหมายเป็นสี: {next_color.upper()}")
                                else:
                                    mission_completed = True
                                    auto_aim_mode = False
                                    print("\n🎉 [MISSION COMPLETE] ยิงป้ายครบทั้ง 4 สีเรียบร้อยแล้ว!")
                        else:
                            lock_counter = 0
                            dir_mult = -1.0 if yaw_invert else 1.0
                            yaw_out = pid_yaw.compute(dx) * dir_mult
                            pitch_out = pid_pitch.compute(-dy)

                            if (now - last_cmd_time) >= CMD_INTERVAL:
                                ep_gimbal.drive_speed(pitch_speed=int(pitch_out), yaw_speed=int(yaw_out))
                                last_cmd_time = now
                    else:
                        lock_counter = 0
                        pid_yaw.reset()
                        pid_pitch.reset()
                        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

            if is_firing_visual and (time.time() - fire_visual_timer > 0.4):
                is_firing_visual = False

            # 3. แสดงผลหน้าต่าง HUD
            result_frame = calibrator.draw_hud(frame, detections, current_params,
                                               target_aim_point=aim_point,
                                               auto_aim=auto_aim_mode,
                                               is_locked=is_locked,
                                               firing=is_firing_visual,
                                               yaw_invert=yaw_invert,
                                               current_idx=target_color_index if not mission_completed else 3,
                                               is_mission_done=mission_completed)

            cv2.imshow("RoboMaster Live Calibration", result_frame)
            cv2.imshow("HSV Mask (Calibration)", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            elif key in (ord('r'), ord('R')):
                target_color_index = 0
                mission_completed = False
                auto_aim_mode = True
                calibrator.update_trackbars_for_color(COLOR_SEQUENCE[0])
                pid_yaw.reset()
                pid_pitch.reset()
                lock_counter = 0
                ep_gimbal.recenter(pitch_speed=60, yaw_speed=60)
                print("\n🔄 [RESET & START] เริ่มต้นรัน Sequence ใหม่: เป้าหมายที่ 1 -> RED (Auto-Aim: ON)")

            elif key in (ord('n'), ord('N')):
                if target_color_index < len(COLOR_SEQUENCE) - 1:
                    target_color_index += 1
                    next_color = COLOR_SEQUENCE[target_color_index]
                    calibrator.update_trackbars_for_color(next_color)
                    pid_yaw.reset()
                    pid_pitch.reset()
                    lock_counter = 0
                    print(f"⏩ [SKIP] ข้ามไปยังเป้าหมายที่ {target_color_index + 1}/4: {next_color.upper()}")
                else:
                    mission_completed = True
                    auto_aim_mode = False
                    print("🏁 [SKIP COMPLETE] สิ้นสุด Sequence แล้ว")

            elif key in color_keys:
                target_color_index = color_keys[key]
                mission_completed = False
                selected_color = COLOR_SEQUENCE[target_color_index]
                calibrator.update_trackbars_for_color(selected_color)
                pid_yaw.reset()
                pid_pitch.reset()
                lock_counter = 0
                print(f"🎯 [SELECT COLOR] บังคับเริ่มที่เป้าหมาย {target_color_index + 1}/4: {selected_color.upper()}")

            elif key in (ord('t'), ord('T')):
                auto_aim_mode = not auto_aim_mode
                lock_counter = 0
                pid_yaw.reset()
                pid_pitch.reset()
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                print(f"[*] Auto-Aim: {'ENABLED' if auto_aim_mode else 'DISABLED'}")

            elif key in (ord('i'), ord('I')):
                yaw_invert = not yaw_invert
                print(f"[*] Yaw Inverted: {yaw_invert}")

            elif key in (ord('f'), ord('F')):
                # ปุ่ม [F] สั่งยิง Manual ทดสอบเสียงและฮาร์ดแวร์
                if now - last_fire_time > 0.5:
                    print("\n[💥 MANUAL FIRE TEST] ทดสอบยิงฮาร์ดแวร์จริง!")
                    execute_real_fire(ep_robot, ep_blaster)
                    last_fire_time = now
                    is_firing_visual = True
                    fire_visual_timer = now

            elif key in (ord('p'), ord('P')):
                calibrator.save_config()

            elif key in (ord('w'), ord('W')):
                auto_aim_mode = False
                ep_gimbal.move(pitch=MANUAL_STEP_DEG, yaw=0, pitch_speed=60, yaw_speed=60)
            elif key in (ord('s'), ord('S')):
                auto_aim_mode = False
                ep_gimbal.move(pitch=-MANUAL_STEP_DEG, yaw=0, pitch_speed=60, yaw_speed=60)
            elif key in (ord('a'), ord('A')):
                auto_aim_mode = False
                ep_gimbal.move(pitch=0, yaw=MANUAL_STEP_DEG, pitch_speed=60, yaw_speed=60)
            elif key in (ord('d'), ord('D')):
                auto_aim_mode = False
                ep_gimbal.move(pitch=0, yaw=-MANUAL_STEP_DEG, pitch_speed=60, yaw_speed=60)
            elif key == 32:
                auto_aim_mode = False
                ep_gimbal.recenter(pitch_speed=60, yaw_speed=60)

    except Exception as e:
        print(f"[-] Error: {e}", file=sys.stderr)
    finally:
        if stream_started:
            ep_robot.camera.stop_video_stream()
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        ep_robot.close()
        cv2.destroyAllWindows()
        print("[*] ปิดระบบและตัดการเชื่อมต่อเรียบร้อย")

if __name__ == "__main__":
    main()