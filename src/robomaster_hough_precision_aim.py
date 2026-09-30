import os
import sys
import json
import time
import threading
import cv2
import numpy as np
from robomaster import robot

CONFIG_PATH = os.path.join("config", "color_config.json")

DEFAULT_COLORS = {
    'red':    {'h_min': 165, 'h_max': 10,  's_min': 80, 's_max': 255, 'v_min': 50, 'v_max': 255, 'min_area': 800},
    'green':  {'h_min': 35,  'h_max': 85,  's_min': 50, 's_max': 255, 'v_min': 40, 'v_max': 255, 'min_area': 800},
    'blue':   {'h_min': 85,  'h_max': 132, 's_min': 45, 's_max': 255, 'v_min': 40, 'v_max': 255, 'min_area': 800},
    'yellow': {'h_min': 18,  'h_max': 34,  's_min': 65, 's_max': 255, 'v_min': 70, 'v_max': 255, 'min_area': 800},
    'roi':    {'top': 0, 'bottom': 80, 'left': 0, 'right': 100}  # ROI เริ่มต้น: ตัดล่าง 20% กันปลายหุ่น
}

def nothing(x):
    pass

# =========================================================================
# ⚙️ ตัวควบคุม PID ความแม่นยำสูง
# =========================================================================
class PrecisionPID:
    def __init__(self, kp=0.080, ki=0.015, kd=0.006, min_speed=3.0, max_speed=35.0, deadband=7):
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
        self.integral = np.clip(self.integral, -35, 35)

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
# 🔍 คลาส Hough Shape & Color Detector พร้อมระบบ Dynamic ROI
# =========================================================================
class HoughShapeDetector:
    def __init__(self):
        self.colors_data = self.load_config()
        self.active_color = 'red'
        self.saved_feedback_timer = 0
        self.kernel = np.ones((5, 5), np.uint8)

        # โหมด: 'AUTO', 'HOUGH_CIRCLE', 'HOUGH_RECT'
        self.hough_mode = 'AUTO'

        self.draw_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 120, 0),
            'yellow': (0, 255, 255)
        }

        self.setup_gui()

    def load_config(self):
        loaded = DEFAULT_COLORS.copy()
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k in loaded and isinstance(v, dict):
                            loaded[k].update(v)
            except Exception:
                pass
        return loaded

    def save_config(self):
        os.makedirs("config", exist_ok=True)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.colors_data, f, indent=4)
        self.saved_feedback_timer = time.time()
        print(f"[+] บันทึกค่าสี {self.active_color.upper()} และ Region of Interest ลงไฟล์ JSON แล้ว!")

    def setup_gui(self):
        cv2.namedWindow("Tuner & Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tuner & Calibration", 440, 560)
        c = self.colors_data[self.active_color]
        r = self.colors_data.get('roi', {'top': 0, 'bottom': 80, 'left': 0, 'right': 100})

        # Trackbars สำหรับ HSV
        cv2.createTrackbar("H Min", "Tuner & Calibration", c['h_min'], 179, nothing)
        cv2.createTrackbar("H Max", "Tuner & Calibration", c['h_max'], 179, nothing)
        cv2.createTrackbar("S Min", "Tuner & Calibration", c['s_min'], 255, nothing)
        cv2.createTrackbar("S Max", "Tuner & Calibration", c['s_max'], 255, nothing)
        cv2.createTrackbar("V Min", "Tuner & Calibration", c['v_min'], 255, nothing)
        cv2.createTrackbar("V Max", "Tuner & Calibration", c['v_max'], 255, nothing)
        cv2.createTrackbar("Min Area", "Tuner & Calibration", c.get('min_area', 800), 10000, nothing)

        # Trackbars สำหรับ Region of Interest (ROI)
        cv2.createTrackbar("ROI Top %", "Tuner & Calibration", r['top'], 50, nothing)
        cv2.createTrackbar("ROI Btm %", "Tuner & Calibration", r['bottom'], 100, nothing)
        cv2.createTrackbar("ROI Lft %", "Tuner & Calibration", r['left'], 50, nothing)
        cv2.createTrackbar("ROI Rgt %", "Tuner & Calibration", r['right'], 100, nothing)

        # แถบชดเชยจุดกึ่งกลางเป้าเล็ง
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
        cv2.setTrackbarPos("Min Area", "Tuner & Calibration", c.get('min_area', 800))

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

        # อ่านค่า ROI
        roi_top = cv2.getTrackbarPos("ROI Top %", "Tuner & Calibration")
        roi_bottom = max(cv2.getTrackbarPos("ROI Btm %", "Tuner & Calibration"), roi_top + 10)
        roi_left = cv2.getTrackbarPos("ROI Lft %", "Tuner & Calibration")
        roi_right = max(cv2.getTrackbarPos("ROI Rgt %", "Tuner & Calibration"), roi_left + 10)

        self.colors_data['roi'] = {
            'top': roi_top, 'bottom': roi_bottom,
            'left': roi_left, 'right': roi_right
        }

        offset_x = cv2.getTrackbarPos("Offset X", "Tuner & Calibration") - 100
        offset_y = cv2.getTrackbarPos("Offset Y", "Tuner & Calibration") - 100
        return self.colors_data[self.active_color], self.colors_data['roi'], offset_x, offset_y

    def detect_hough_circles(self, edges_small, x_offset, y_offset, scale=2.0):
        try:
            circles = cv2.HoughCircles(
                edges_small,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=int(40 / scale),
                param1=60,
                param2=30,
                minRadius=int(12 / scale),
                maxRadius=int(220 / scale)
            )
            detected_circles = []
            if circles is not None:
                circles = np.reshape(circles, (-1, 3))
                for item in circles:
                    cx = int(item[0] * scale) + x_offset
                    cy = int(item[1] * scale) + y_offset
                    r = int(item[2] * scale)
                    if r <= 0:
                        continue
                    area = float(np.pi * (r ** 2))
                    detected_circles.append({
                        'shape': 'Hough Circle',
                        'center': (cx, cy),
                        'radius': r,
                        'bbox': (cx - r, cy - r, r * 2, r * 2),
                        'area': area
                    })
            return detected_circles
        except Exception:
            return []

    def detect_hough_rectangles(self, edges_small, x_offset, y_offset, min_area=800, scale=2.0):
        try:
            lines = cv2.HoughLinesP(
                edges_small,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=int(25 / scale),
                maxLineGap=int(15 / scale)
            )

            detected_rects = []
            if lines is not None and len(lines) >= 3:
                points = []
                lines_reshaped = np.reshape(lines, (-1, 4))
                for line in lines_reshaped:
                    x1 = int(line[0] * scale) + x_offset
                    y1 = int(line[1] * scale) + y_offset
                    x2 = int(line[2] * scale) + x_offset
                    y2 = int(line[3] * scale) + y_offset
                    points.append([x1, y1])
                    points.append([x2, y2])

                if len(points) >= 4:
                    points = np.array(points, dtype=np.int32)
                    rect = cv2.minAreaRect(points)
                    (cx, cy), (w_box, h_box), angle = rect
                    area = float(w_box * h_box)

                    if area >= min_area and min(w_box, h_box) > 20:
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        x, y, w, h = cv2.boundingRect(box)

                        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-5)
                        shape_label = "Hough Square" if aspect <= 1.25 else "Hough Rect"

                        detected_rects.append({
                            'shape': shape_label,
                            'center': (int(cx), int(cy)),
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'box_points': box,
                            'area': area
                        })

            return detected_rects
        except Exception:
            return []

    def process_frame(self, frame):
        params, roi_dict, offset_x, offset_y = self.get_trackbar_values()
        h, w = frame.shape[:2]

        # 1. คำนวณขอบเขต Region of Interest (พิกัดพิกเซล)
        rx1 = int(w * (roi_dict['left'] / 100.0))
        rx2 = int(w * (roi_dict['right'] / 100.0))
        ry1 = int(h * (roi_dict['top'] / 100.0))
        ry2 = int(h * (roi_dict['bottom'] / 100.0))

        rx2 = max(rx1 + 30, rx2)
        ry2 = max(ry1 + 30, ry2)
        roi_box = (rx1, ry1, rx2, ry2)

        # 2. ทำ Color Masking เต็มภาพ
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if params['h_min'] <= params['h_max']:
            lower = np.array([params['h_min'], params['s_min'], params['v_min']], dtype=np.uint8)
            upper = np.array([params['h_max'], params['s_max'], params['v_max']], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            mask1 = cv2.inRange(hsv, np.array([0, params['s_min'], params['v_min']], dtype=np.uint8),
                                     np.array([params['h_max'], params['s_max'], params['v_max']], dtype=np.uint8))
            mask2 = cv2.inRange(hsv, np.array([params['h_min'], params['s_min'], params['v_min']], dtype=np.uint8),
                                     np.array([179, params['s_max'], params['v_max']], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # 3. ตัดส่วนนอก ROI ทิ้ง (Zero out outside ROI)
        roi_mask = np.zeros_like(mask)
        roi_mask[ry1:ry2, rx1:rx2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        # สกัดเฉพาะชิ้นภาพใน ROI เพื่อส่งเข้า Canny & Hough (เร็วขึ้นมาก)
        cropped_mask = mask[ry1:ry2, rx1:rx2]
        edges_roi = cv2.Canny(cropped_mask, 50, 150)

        # ย่อสเกลสำหรับ Hough 2 เท่า
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1
        edges_small = cv2.resize(edges_roi, (max(1, roi_w // 2), max(1, roi_h // 2)), interpolation=cv2.INTER_NEAREST)

        # 4. ประมวลผล Hough พร้อมส่ง Offset ของ ROI เข้าไปบวกกลับ
        detections = []
        if self.hough_mode in ('AUTO', 'HOUGH_CIRCLE'):
            detections.extend(self.detect_hough_circles(edges_small, x_offset=rx1, y_offset=ry1, scale=2.0))

        if self.hough_mode in ('AUTO', 'HOUGH_RECT'):
            detections.extend(self.detect_hough_rectangles(edges_small, x_offset=rx1, y_offset=ry1, min_area=params['min_area'], scale=2.0))

        # Fallback เป็น Contours อัตโนมัติถ้า Hough ไม่พบ
        if len(detections) == 0:
            contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < params['min_area']:
                    continue
                # ย้ายพิกัด Contour กลับสู่พิกัดภาพรวม
                cnt_shifted = cnt + np.array([rx1, ry1])
                rect = cv2.minAreaRect(cnt_shifted)
                cx, cy = int(rect[0][0]), int(rect[0][1])
                x, y, bw, bh = cv2.boundingRect(cnt_shifted)
                detections.append({
                    'shape': 'Target',
                    'center': (cx, cy),
                    'bbox': (int(x), int(y), int(bw), int(bh)),
                    'area': float(area)
                })

        detections.sort(key=lambda item: item['area'], reverse=True)
        return detections, mask, edges_roi, params, roi_box, offset_x, offset_y

    def draw_hud(self, frame, detections, params, roi_box, target_aim_point, auto_aim=False, is_locked=False, firing=False, yaw_invert=False):
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = roi_box
        aim_x, aim_y = target_aim_point

        # 1. วาดแถบ Darkened Overlay นอกกรอบ ROI
        overlay = frame.copy()
        # ด้านบน
        if ry1 > 0:
            cv2.rectangle(overlay, (0, 0), (w, ry1), (20, 20, 20), -1)
        # ด้านล่าง (ตัดปลายหุ่น ~10cm)
        if ry2 < h:
            cv2.rectangle(overlay, (0, ry2), (w, h), (10, 10, 40), -1)
        # ด้านซ้าย
        if rx1 > 0:
            cv2.rectangle(overlay, (0, ry1), (rx1, ry2), (20, 20, 20), -1)
        # ด้านขวา
        if rx2 < w:
            cv2.rectangle(overlay, (rx2, ry1), (w, ry2), (20, 20, 20), -1)

        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # กรอบขอบเขต ROI
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(frame, "[ACTIVE SEARCH ROI]", (rx1 + 8, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        if ry2 < h:
            cv2.putText(frame, "[ROBOT BUMPER EXCLUDED]", (w // 2 - 110, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1, cv2.LINE_AA)

        # 2. วาดผลลัพธ์การตรวจจับ
        draw_color = self.draw_colors.get(self.active_color, (0, 255, 0))
        for idx, det in enumerate(detections):
            cx, cy = det['center']
            shape = det['shape']
            box_col = (0, 0, 255) if (idx == 0 and is_locked) else draw_color

            if 'radius' in det:
                cv2.circle(frame, (cx, cy), det['radius'], box_col, 2)
            elif 'box_points' in det:
                box_pts = np.int32(det['box_points']).reshape((-1, 1, 2))
                cv2.drawContours(frame, [box_pts], 0, box_col, 2)
            else:
                x, y, bw, bh = det['bbox']
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_col, 2)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 1)
            cv2.line(frame, (aim_x, aim_y), (cx, cy), (160, 160, 160), 1)

            bx, by, bw, bh = det['bbox']
            label = f"{self.active_color.upper()} {shape.upper()}"
            text_y = max(by, 25)
            cv2.rectangle(frame, (bx, text_y - 22), (bx + len(label) * 9, text_y), box_col, -1)
            cv2.putText(frame, label, (bx + 3, text_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. เป้าเล็ง Crosshair
        cross_col = (0, 0, 255) if firing else ((0, 255, 0) if is_locked else ((0, 255, 255) if auto_aim else (255, 255, 255)))
        cv2.drawMarker(frame, (aim_x, aim_y), cross_col, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(frame, (aim_x, aim_y), 10, cross_col, 1)

        # 4. ข้อมูลสถานะบนหน้าจอ
        cv2.putText(frame, f"TARGET: {self.active_color.upper()} (Press [V] to Change Color)", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, draw_color, 2)

        aim_state = "ON" if auto_aim else "OFF [T]"
        invert_state = "REVERSED (-)" if yaw_invert else "NORMAL (+)"
        cv2.putText(frame, f"AUTO-AIM: {aim_state} | YAW: {invert_state} [I] | HOUGH: [{self.hough_mode}] [H]", 
                    (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        if len(detections) > 0:
            tcx, tcy = detections[0]['center']
            err_x = tcx - aim_x
            err_y = tcy - aim_y
            err_color = (0, 255, 0) if is_locked else (0, 200, 255)
            cv2.putText(frame, f"HOUGH ERROR -> X: {err_x:+3d} px | Y: {err_y:+3d} px", 
                        (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, err_color, 2)

        if firing:
            cv2.rectangle(frame, (w // 2 - 130, 40), (w // 2 + 130, 85), (0, 0, 255), -1)
            cv2.putText(frame, ">> FIRING! <<", (w // 2 - 95, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        elif is_locked:
            cv2.putText(frame, "[HOUGH CENTER LOCKED]", (w // 2 - 110, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2, cv2.LINE_AA)

        if time.time() - self.saved_feedback_timer < 2.0:
            cv2.rectangle(frame, (10, 100), (280, 130), (0, 180, 0), -1)
            cv2.putText(frame, "SAVED CONFIG & ROI!", (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "[V]: Switch Color | [T]: Auto-Aim | [F]: Shoot | [I]: Invert Yaw | [P]: Save | [Q]: Quit",
                    (15, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        return frame


def shoot_air_async(ep_blaster):
    def _fire():
        try:
            ep_blaster.fire(fire_type=robot.BEAD_FIRE, count=1)
        except Exception as e:
            print(f"[-] ยิงไม่สำเร็จ: {e}")
    threading.Thread(target=_fire, daemon=True).start()


def main():
    color_list = ['red', 'green', 'blue', 'yellow']
    color_keys = {
        ord('1'): 'red',
        ord('2'): 'green',
        ord('3'): 'blue',
        ord('4'): 'yellow'
    }

    calibrator = HoughShapeDetector()
    ep_robot = robot.Robot()
    stream_started = False
    MANUAL_STEP_DEG = 5

    pid_yaw = PrecisionPID(kp=0.080, ki=0.015, kd=0.006, min_speed=3.0, max_speed=35.0, deadband=7)
    pid_pitch = PrecisionPID(kp=0.085, ki=0.015, kd=0.006, min_speed=3.0, max_speed=25.0, deadband=7)

    auto_aim_mode = False
    yaw_invert = False

    REQUIRED_LOCK_FRAMES = 4
    lock_counter = 0

    FIRE_COOLDOWN = 1.5
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

        ep_gimbal.recenter(pitch_speed=50, yaw_speed=50).wait_for_completed()
        ep_camera.start_video_stream(display=False, resolution="720p")
        stream_started = True

        print("\n=========================================================")
        print(" [V]           : สลับเปลี่ยนสี (แดง ➔ เขียว ➔ ฟ้า ➔ เหลือง)")
        print(" [ROI Sliders] : ปรับกรอบพื้นที่ตรวจจับ (Top, Bottom, Left, Right)")
        print(" [T]           : เปิด/ปิด Auto-Aim แบบ Hough Precision")
        print(" [H]           : สลับโหมด Hough (AUTO -> CIRCLE -> RECT)")
        print(" [I]           : สลับทิศทางหมุนซ้าย-ขวา (Invert Yaw)")
        print(" [F]           : กดยิงลมแบบ Manual")
        print(" [P]           : บันทึกค่าสีและ ROI ลง JSON")
        print(" [Q] หรือ [Esc] : ออกจากโปรแกรม")
        print("=========================================================\n")

        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Exception:
                frame = None

            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            h_img, w_img = frame.shape[:2]

            # 1. ประมวลผลภาพพร้อมกรอบ Region of Interest (ROI)
            try:
                detections, mask, edges_roi, current_params, roi_box, offset_x, offset_y = calibrator.process_frame(frame)
                aim_point = (w_img // 2 + offset_x, h_img // 2 + offset_y)
                aim_x, aim_y = aim_point
            except Exception as proc_err:
                print(f"[!] Processing warning: {proc_err}")
                detections, mask, edges_roi = [], np.zeros((h_img, w_img), dtype=np.uint8), np.zeros((h_img, w_img), dtype=np.uint8)
                roi_box = (0, 0, w_img, int(h_img * 0.8))
                aim_point = (w_img // 2, h_img // 2)
                aim_x, aim_y = aim_point

            is_locked = False
            now = time.time()

            # 2. ควบคุม Auto-Aim สู่จุดกึ่งกลางของ Hough
            if auto_aim_mode:
                try:
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

                            if lock_counter >= REQUIRED_LOCK_FRAMES and (now - last_fire_time) > FIRE_COOLDOWN:
                                print(f"[💥 HOUGH LOCK-ON HIT!] จุดศูนย์กลาง {target['shape']} อยู่กึ่งกลางเป๊ะ (dx:{dx}, dy:{dy}) -> สั่งยิงลม!")
                                shoot_air_async(ep_blaster)
                                last_fire_time = now
                                is_firing_visual = True
                                fire_visual_timer = now
                                lock_counter = 0
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
                except Exception as gimbal_err:
                    print(f"[!] Gimbal warning: {gimbal_err}")

            if is_firing_visual and (time.time() - fire_visual_timer > 0.4):
                is_firing_visual = False

            # 3. แสดงผลภาพพร้อมกรอบ ROI
            try:
                result_frame = calibrator.draw_hud(frame, detections, current_params,
                                                   roi_box=roi_box,
                                                   target_aim_point=aim_point,
                                                   auto_aim=auto_aim_mode,
                                                   is_locked=is_locked,
                                                   firing=is_firing_visual,
                                                   yaw_invert=yaw_invert)

                cv2.imshow("RoboMaster Hough Calibration", result_frame)
                if edges_roi.size > 0:
                    cv2.imshow("Hough Canny Edges (Inside ROI)", edges_roi)
            except Exception as draw_err:
                print(f"[!] Draw warning: {draw_err}")

            # 4. จัดการปุ่มกด (Window stay-alive)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            # ปุ่ม V: สลับเปลี่ยนสีถัดไป (Red -> Green -> Blue -> Yellow -> Red)
            elif key in (ord('v'), ord('V')):
                curr_idx = color_list.index(calibrator.active_color)
                next_color = color_list[(curr_idx + 1) % len(color_list)]
                calibrator.update_trackbars_for_color(next_color)
                pid_yaw.reset()
                pid_pitch.reset()
                lock_counter = 0
                print(f"[*] [V Pressed] สลับเป้าหมายเป็นสี: {next_color.upper()}")

            elif key in (ord('h'), ord('H')):
                modes = ['AUTO', 'HOUGH_CIRCLE', 'HOUGH_RECT']
                curr_idx = modes.index(calibrator.hough_mode)
                calibrator.hough_mode = modes[(curr_idx + 1) % len(modes)]
                print(f"[*] เปลี่ยนโหมด Hough เป็น: {calibrator.hough_mode}")

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
                now = time.time()
                if now - last_fire_time > 0.5:
                    print("\n[💥 BOOM!] Manual Fire (ยิงลม)!")
                    shoot_air_async(ep_blaster)
                    last_fire_time = now
                    is_firing_visual = True
                    fire_visual_timer = now

            elif key in (ord('p'), ord('P')):
                calibrator.save_config()

            elif key in color_keys:
                selected_color = color_keys[key]
                calibrator.update_trackbars_for_color(selected_color)
                pid_yaw.reset()
                pid_pitch.reset()
                lock_counter = 0
                print(f"[*] สลับเป้าหมายเป็นสี: {selected_color.upper()}")

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
        print(f"[-] Fatal Error: {e}", file=sys.stderr)
    finally:
        if stream_started:
            ep_robot.camera.stop_video_stream()
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        ep_robot.close()
        cv2.destroyAllWindows()
        print("[*] ปิดการเชื่อมต่อเรียบร้อย")

if __name__ == "__main__":
    main()