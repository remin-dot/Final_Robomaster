import os
import sys
import json
import time
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from robomaster import robot, blaster, led

CONFIG_PATH = os.path.join("config", "color_config.json")

# =========================================================================
# ⚙️ 1. การตั้งค่าโหมดและเป้าหมายภารกิจ
# =========================================================================
OPERATION_MODE = "TEST"

TARGETS_TO_SHOOT = [
    ("เขียว", "ผืนผ้า"),
    ("น้ำเงิน", "กลม")
]

GRID_SIZE_M = 0.6
SLIDE_DIRECTION = "right"
CHASSIS_SPEED = 0.45
SEARCH_TIMEOUT = 2.5
MAX_GRID_STEPS = 5

# กำหนดประเภทยิงให้รองรับทั้งเวอร์ชันเก่าและใหม่ของ RoboMaster SDK
FIRE_BEAD_TYPE = blaster.WATER_FIRE if hasattr(blaster, "WATER_FIRE") else getattr(blaster, "BEAD_FIRE", 1)
FIRE_IR_TYPE = blaster.INFRARED_FIRE if hasattr(blaster, "INFRARED_FIRE") else getattr(blaster, "IR_FIRE", 2)

DEFAULT_COLORS = {
    'red':    {'h_min': 165, 'h_max': 10,  's_min': 100, 's_max': 255, 'v_min': 60, 'v_max': 255, 'min_area': 700},
    'green':  {'h_min': 35,  'h_max': 85,  's_min': 80,  's_max': 255, 'v_min': 50, 'v_max': 255, 'min_area': 700},
    'blue':   {'h_min': 90,  'h_max': 130, 's_min': 90,  's_max': 255, 'v_min': 50, 'v_max': 255, 'min_area': 700},
    'yellow': {'h_min': 18,  'h_max': 34,  's_min': 90,  's_max': 255, 'v_min': 80, 'v_max': 255, 'min_area': 700}
}

THAI_TO_COLOR = {
    "แดง": "red", "red": "red",
    "เขียว": "green", "green": "green",
    "น้ำเงิน": "blue", "ฟ้า": "blue", "blue": "blue",
    "เหลือง": "yellow", "yellow": "yellow"
}

THAI_TO_SHAPE = {
    "กลม": "circle", "circle": "circle",
    "จัตุรัส": "square", "square": "square",
    "ผืนผ้า": "rectangle", "rectangle": "rectangle"
}

# =========================================================================
# 🎛️ 2. Precision PID Controller
# =========================================================================
class PrecisionPID:
    def __init__(self, kp=0.075, ki=0.015, kd=0.006, min_speed=3.0, max_speed=35.0, deadband=8):
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
# 🔍 3. Advanced Shape & Color Detector
# =========================================================================
class TargetDetector:
    def __init__(self, bottom_ignore_ratio=0.20):
        self.bottom_ignore_ratio = bottom_ignore_ratio
        self.colors_data = self.load_config()
        self.kernel = np.ones((5, 5), np.uint8)

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_COLORS.copy()

    def classify_shape(self, contour, area):
        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return "unknown"

        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        circle_ratio = area / circle_area if circle_area > 0 else 0
        circularity = (4 * np.pi * area) / (peri * peri)

        if circle_ratio >= 0.80 and circularity >= 0.74:
            return "circle"

        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        rect_area = w * h
        if rect_area == 0:
            return "unknown"

        extent = area / rect_area
        if extent < 0.80:
            return "unknown"

        short_side = min(w, h)
        long_side = max(w, h)
        aspect_ratio = short_side / long_side

        if aspect_ratio >= 0.82:
            return "square"
        elif 0.50 <= aspect_ratio <= 0.82:
            return "rectangle"

        return "unknown"

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
        white_pixels = np.sum((s_channel < 85) & (v_channel > 80))
        total_pixels = outer_roi.shape[0] * outer_roi.shape[1]
        return (white_pixels / total_pixels) > 0.15

    def scan_frame(self, frame, active_targets):
        h, w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ignore_h = int(h * (1.0 - self.bottom_ignore_ratio))

        detected_list = []
        debug_mask = np.zeros((h, w), dtype=np.uint8)

        needed_colors = set(t["color"] for t in active_targets)

        for col_name in needed_colors:
            if col_name not in self.colors_data:
                continue
            params = self.colors_data[col_name]

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
            mask[ignore_h:h, 0:w] = 0

            debug_mask = cv2.bitwise_or(debug_mask, mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < params.get('min_area', 700) or area > (h * w * 0.45):
                    continue

                shape_detected = self.classify_shape(cnt, area)

                match_found = False
                for target_item in active_targets:
                    if target_item["color"] == col_name and target_item["shape"] == shape_detected:
                        match_found = True
                        break

                if not match_found:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                if not self.is_on_white_background(hsv, (x, y, bw, bh)):
                    continue

                rect = cv2.minAreaRect(cnt)
                cx, cy = int(rect[0][0]), int(rect[0][1])

                detected_list.append({
                    'bbox': (x, y, bw, bh),
                    'center': (cx, cy),
                    'area': area,
                    'color': col_name,
                    'shape': shape_detected
                })

        detected_list.sort(key=lambda item: item['area'], reverse=True)
        return detected_list, debug_mask

# =========================================================================
# 💥 4. ฟังก์ชันสั่งยิง (แก้ไข Blaster Constants และ times=1)
# =========================================================================
def execute_fire(ep_robot, ep_blaster, robot_lock, is_real_mode=False):
    def _fire():
        with robot_lock:
            if is_real_mode:
                bead_ok = True
                try:
                    ep_blaster.fire(fire_type=FIRE_BEAD_TYPE, times=1)
                except Exception as e:
                    bead_ok = False
                    print(f"[!] Bead Fire Error: {e}")
                try:
                    ep_blaster.fire(fire_type=FIRE_IR_TYPE, times=1)
                except Exception:
                    pass
                if bead_ok:
                    try:
                        ep_robot.play_sound(robot.SOUND_ID_SHOOT)
                    except Exception:
                        pass
                else:
                    print("   [⚠️ FIRE FAILED] ยิงกระสุนจริงไม่สำเร็จ — ไม่เล่นเสียงยืนยัน")
            else:
                try:
                    ep_blaster.fire(fire_type=FIRE_IR_TYPE, times=1)
                    ep_robot.play_sound(robot.SOUND_ID_SHOOT)
                    print("   [SIMULATED SHOT] ล็อคเป้าสำเร็จ! จำลองการยิง (IR + เสียงยืนยัน)")
                except Exception as e:
                    print(f"[!] Sim Fire Error: {e}")

    threading.Thread(target=_fire, daemon=True).start()

# =========================================================================
# 🖱️ 4.5 Pre-flight Config GUI
# =========================================================================
def launch_config_gui():
    result = {"mode": None, "targets": []}

    root = tk.Tk()
    root.title("RoboMaster Mission Config")
    root.geometry("360x420")
    root.resizable(False, False)

    tk.Label(root, text="โหมดการทำงาน", font=("", 11, "bold")).pack(pady=(12, 2))
    mode_var = tk.StringVar(value="TEST")
    mode_frame = tk.Frame(root)
    mode_frame.pack()
    tk.Radiobutton(mode_frame, text="🟢 TEST (dry-run ไม่ยิงจริง)", variable=mode_var, value="TEST").pack(anchor="w")
    tk.Radiobutton(mode_frame, text="🔴 REAL (ยิงกระสุนเจลจริง)", variable=mode_var, value="REAL").pack(anchor="w")

    tk.Label(root, text="เพิ่มเป้าหมายเข้าคิว", font=("", 11, "bold")).pack(pady=(14, 2))
    pick_frame = tk.Frame(root)
    pick_frame.pack()

    color_var = tk.StringVar(value=list(DEFAULT_COLORS.keys())[0])
    shape_var = tk.StringVar(value="circle")

    tk.Label(pick_frame, text="สี:").grid(row=0, column=0, padx=4, sticky="e")
    color_menu = ttk.Combobox(pick_frame, textvariable=color_var,
                               values=list(DEFAULT_COLORS.keys()), state="readonly", width=12)
    color_menu.grid(row=0, column=1, padx=4)

    tk.Label(pick_frame, text="รูปร่าง:").grid(row=1, column=0, padx=4, sticky="e")
    shape_menu = ttk.Combobox(pick_frame, textvariable=shape_var,
                               values=["circle", "square", "rectangle"], state="readonly", width=12)
    shape_menu.grid(row=1, column=1, padx=4)

    queue_list = tk.Listbox(root, width=36, height=8)
    queue_list.pack(pady=10)

    def add_target():
        c, s = color_var.get(), shape_var.get()
        result["targets"].append({"color": c, "shape": s})
        queue_list.insert(tk.END, f"{c.upper()} - {s.upper()}")

    def remove_selected():
        sel = queue_list.curselection()
        if not sel:
            return
        idx = sel[0]
        queue_list.delete(idx)
        del result["targets"][idx]

    btn_frame = tk.Frame(root)
    btn_frame.pack()
    tk.Button(btn_frame, text="+ เพิ่ม", width=12, command=add_target).grid(row=0, column=0, padx=4)
    tk.Button(btn_frame, text="- ลบที่เลือก", width=12, command=remove_selected).grid(row=0, column=1, padx=4)

    def start_mission():
        if not result["targets"]:
            messagebox.showwarning("แจ้งเตือน", "ยังไม่ได้เพิ่มเป้าหมายในคิวเลย")
            return
        if mode_var.get() == "REAL":
            confirmed = messagebox.askyesno(
                "⚠️ ยืนยันโหมดยิงจริง",
                "โหมด REAL จะยิงกระสุนเจลจริงออกจากหุ่นยนต์เมื่อเจอเป้า\n"
                "ตรวจสอบพื้นที่ปลอดภัยแล้วหรือยัง?\n\nยืนยันเริ่มภารกิจ REAL?"
            )
            if not confirmed:
                return
        result["mode"] = mode_var.get()
        root.destroy()

    def cancel():
        result["mode"] = None
        root.destroy()

    tk.Button(root, text="🚀 เริ่มภารกิจ", bg="#4CAF50", fg="white",
              command=start_mission).pack(pady=(10, 2), ipadx=10)
    tk.Button(root, text="ยกเลิก", command=cancel).pack()

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    return result["mode"], result["targets"]

# =========================================================================
# 🚀 5. Main Control Loop
# =========================================================================
def main(operation_mode=None, target_list=None):
    global OPERATION_MODE

    if operation_mode is None:
        operation_mode = OPERATION_MODE
    if target_list is None:
        active_mission_targets = []
        for c_str, s_str in TARGETS_TO_SHOOT:
            col = THAI_TO_COLOR.get(c_str.strip().lower(), c_str)
            shp = THAI_TO_SHAPE.get(s_str.strip().lower(), s_str)
            active_mission_targets.append({"color": col, "shape": shp})
    else:
        active_mission_targets = [dict(t) for t in target_list]

    for t in active_mission_targets:
        if t["color"] not in DEFAULT_COLORS:
            raise ValueError(f"Unknown target color: '{t['color']}'")
        if t["shape"] not in ("circle", "square", "rectangle"):
            raise ValueError(f"Unknown target shape: '{t['shape']}'")

    detector = TargetDetector(bottom_ignore_ratio=0.20)
    ep_robot = robot.Robot()
    stream_started = False

    robot_lock = threading.Lock()

    pid_yaw = PrecisionPID(kp=0.075, ki=0.015, kd=0.006, min_speed=3.0, max_speed=35.0, deadband=8)
    pid_pitch = PrecisionPID(kp=0.080, ki=0.015, kd=0.006, min_speed=3.0, max_speed=25.0, deadband=8)

    remaining_targets = [dict(t) for t in active_mission_targets]
    is_real_mode = (operation_mode.upper() == "REAL")
    auto_mode = False
    is_sliding = False
    current_grid_step = 0
    grid_exhausted_warned = False
    prev_target_id = None

    REQUIRED_LOCK_FRAMES = 5
    lock_counter = 0

    no_target_timer = time.time()
    post_shot_timer = 0
    POST_SHOT_STABILIZE_TIME = 1.0

    last_cmd_time = 0
    CMD_INTERVAL = 0.05
    camera_none_count = 0

    try:
        print("[*] กำลังเชื่อมต่อ RoboMaster...")
        ep_robot.initialize(conn_type="ap", proto_type="udp")
        print("[+] เชื่อมต่อสำเร็จ!")

        ep_camera = ep_robot.camera
        ep_gimbal = ep_robot.gimbal
        ep_blaster = ep_robot.blaster
        ep_chassis = ep_robot.chassis
        ep_led = ep_robot.led

        try:
            ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
            ep_blaster.set_led(brightness=255, effect=led.EFFECT_ON)
        except Exception:
            pass

        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        ep_camera.start_video_stream(display=False, resolution="720p")
        stream_started = True

        def strafe_one_grid():
            nonlocal current_grid_step, is_sliding, no_target_timer
            with robot_lock:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                dy = -GRID_SIZE_M if SLIDE_DIRECTION == "right" else GRID_SIZE_M
                print(f"\n🚶 [STRAFE] ไม่พบเป้าหมาย -> สไลด์ {SLIDE_DIRECTION.upper()} {GRID_SIZE_M}m ไปยัง Grid #{current_grid_step + 1}...")
                ep_chassis.move(x=0, y=dy, z=0, xy_speed=CHASSIS_SPEED).wait_for_completed()
                ep_gimbal.recenter(pitch_speed=50, yaw_speed=50).wait_for_completed()

            current_grid_step += 1
            is_sliding = False
            no_target_timer = time.time()

        target_summary = [f"{t['color'].upper()} ({t['shape'].upper()})" for t in active_mission_targets]
        print("\n" + "=" * 65)
        print("🎯 ROBO-TARGET AUTO-AIM (รองรับ 4 สี 3 ทรงเรขาคณิต)")
        print(f"🚦 โหมดปัจจุบัน: {'[🔴 REAL - ยิงลูกจริง]' if is_real_mode else '[🟢 TEST - โหมดทดสอบ Dry-run]'}")
        print(f"📋 เป้าหมายภารกิจ: {target_summary}")
        print(" [M] : สลับโหมด TEST / REAL")
        print(" [T] : เปิด/ปิด Auto-Aim & Grid Scanning")
        print(" [R] : รีเซ็ตภารกิจ เริ่มต้นใหม่ทั้งหมด")
        print(" [Q] หรือ [Esc] : ออกจากโปรแกรม")
        print("=" * 65 + "\n")

        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                camera_none_count += 1
                if camera_none_count % 60 == 0:
                    print("⚠️ [CAMERA] ไม่ได้ภาพต่อเนื่อง — ตรวจสอบสาย/สตรีม")
                continue
            camera_none_count = 0

            h_img, w_img = frame.shape[:2]
            aim_x, aim_y = w_img // 2, h_img // 2
            now = time.time()

            detections, mask = detector.scan_frame(frame, remaining_targets)
            is_locked = False

            if auto_mode and len(remaining_targets) > 0 and not is_sliding:
                if (now - post_shot_timer) < POST_SHOT_STABILIZE_TIME:
                    with robot_lock:
                        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                else:
                    if len(detections) > 0:
                        no_target_timer = now
                        target = detections[0]
                        cx, cy = target['center']
                        dx = cx - aim_x
                        dy = cy - aim_y

                        target_id = (target['color'], target['shape'])
                        if target_id != prev_target_id:
                            pid_yaw.reset()
                            pid_pitch.reset()
                            lock_counter = 0
                        prev_target_id = target_id

                        in_center_x = abs(dx) <= pid_yaw.deadband
                        in_center_y = abs(dy) <= pid_pitch.deadband

                        if in_center_x and in_center_y:
                            with robot_lock:
                                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                            lock_counter += 1
                            is_locked = True

                            if lock_counter >= REQUIRED_LOCK_FRAMES:
                                hit_c = target['color']
                                hit_s = target['shape']
                                print(f"\n💥 [{'REAL FIRE' if is_real_mode else 'TEST HIT'}] ล็อคกึ่งกลางเป้า {hit_c.upper()} ({hit_s.upper()})!")

                                execute_fire(ep_robot, ep_blaster, robot_lock, is_real_mode=is_real_mode)

                                with robot_lock:
                                    ep_led.set_led(comp="all", r=255, g=0, b=0, effect=led.EFFECT_FLASH)
                                post_shot_timer = now
                                lock_counter = 0
                                prev_target_id = None

                                pid_yaw.reset()
                                pid_pitch.reset()

                                remaining_targets = [t for t in remaining_targets if not (
                                    t['color'] == hit_c and t['shape'] == hit_s
                                )]

                                if len(remaining_targets) == 0:
                                    print("\n🏆 [MISSION COMPLETE] ยิงเป้าหมายจริงครบทุกเป้าเรียบร้อยแล้ว!")
                                    auto_mode = False
                                    with robot_lock:
                                        ep_led.set_led(comp="all", r=0, g=255, b=0, effect=led.EFFECT_ON)
                        else:
                            lock_counter = 0
                            yaw_out = pid_yaw.compute(dx)
                            pitch_out = pid_pitch.compute(-dy)

                            if (now - last_cmd_time) >= CMD_INTERVAL:
                                with robot_lock:
                                    ep_gimbal.drive_speed(pitch_speed=int(pitch_out), yaw_speed=int(yaw_out))
                                last_cmd_time = now
                    else:
                        lock_counter = 0
                        prev_target_id = None
                        pid_yaw.reset()
                        pid_pitch.reset()
                        with robot_lock:
                            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

                        if (now - no_target_timer) > SEARCH_TIMEOUT:
                            if current_grid_step < MAX_GRID_STEPS:
                                is_sliding = True
                                threading.Thread(target=strafe_one_grid, daemon=True).start()
                            else:
                                if not grid_exhausted_warned:
                                    print("⚠️ [MISSION FAILED] สไลด์สุด Grid แล้วยังหาเป้าไม่ครบ — หยุด Auto-Aim")
                                    grid_exhausted_warned = True
                                auto_mode = False

            # =====================================================
            # 🖥️ HUD Graphics
            # =====================================================
            cross_col = (0, 0, 255) if is_locked else ((0, 255, 0) if auto_mode else (255, 255, 255))
            cv2.drawMarker(frame, (aim_x, aim_y), cross_col, cv2.MARKER_CROSS, 24, 2)
            cv2.circle(frame, (aim_x, aim_y), 12, cross_col, 1)

            for det in detections:
                x, y, bw, bh = det['bbox']
                box_col = (0, 0, 255) if is_locked else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_col, 2)
                lbl = f"{det['color'].upper()} {det['shape'].upper()}"
                cv2.putText(frame, lbl, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 2)

            mode_badge = "MODE: [REAL FIRE]" if is_real_mode else "MODE: [TEST / DRY-RUN]"
            mode_col = (0, 0, 255) if is_real_mode else (0, 255, 0)
            cv2.rectangle(frame, (w_img - 260, 15), (w_img - 15, 45), mode_col, -1)
            cv2.putText(frame, mode_badge, (w_img - 250, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            rem_str = "REMAINING: " + (", ".join([f"{t['color'].upper()}-{t['shape'].upper()}" for t in remaining_targets]) if remaining_targets else "MISSION COMPLETE")
            cv2.putText(frame, rem_str, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"AUTO-AIM: {'ON' if auto_mode else 'OFF [T]'} | GRID STEP: #{current_grid_step} | [M] Toggle Mode", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

            cv2.imshow("RoboMaster Live Aiming", frame)

            if not is_real_mode:
                cv2.imshow("TEST MODE: HSV Mask View", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('m'), ord('M')):
                is_real_mode = not is_real_mode
                if is_real_mode:
                    try:
                        cv2.destroyWindow("TEST MODE: HSV Mask View")
                    except Exception:
                        pass
                print(f"\n[*] สลับโหมดการทำงาน -> {'[🔴 REAL RUN - ยิงกระสุนจริง]' if is_real_mode else '[🟢 TEST MODE - ปิดยิงจริง]'}")
            elif key in (ord('t'), ord('T')):
                auto_mode = not auto_mode
                no_target_timer = time.time()
                pid_yaw.reset()
                pid_pitch.reset()
                lock_counter = 0
                prev_target_id = None
                with robot_lock:
                    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                print(f"[*] Auto-Aim & Scan: {'ENABLED' if auto_mode else 'DISABLED'}")
            elif key in (ord('r'), ord('R')):
                remaining_targets = [dict(t) for t in active_mission_targets]
                current_grid_step = 0
                grid_exhausted_warned = False
                auto_mode = True
                no_target_timer = time.time()
                pid_yaw.reset()
                pid_pitch.reset()
                lock_counter = 0
                prev_target_id = None
                with robot_lock:
                    ep_gimbal.recenter(pitch_speed=60, yaw_speed=60)
                print("\n🔄 [RESET] เริ่มต้นภารกิจใหม่ทั้งหมด")

    except Exception as e:
        print(f"[-] Error: {e}", file=sys.stderr)
    finally:
        if stream_started:
            ep_camera.stop_video_stream()
        try:
            with robot_lock:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass
        ep_robot.close()
        cv2.destroyAllWindows()
        print("[*] ปิดระบบและตัดการเชื่อมต่อเรียบร้อย")

if __name__ == "__main__":
    gui_mode, gui_targets = launch_config_gui()
    if gui_mode is None:
        print("[!] ยกเลิกการตั้งค่า — ไม่เริ่มภารกิจ")
        sys.exit(0)
    main(operation_mode=gui_mode, target_list=gui_targets)