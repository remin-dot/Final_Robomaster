import os
import time
import math
import csv
from robomaster import robot, blaster, led

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ==============================================================================
# 1. การตั้งค่าระบบและเป้าหมาย
# ==============================================================================
TOTAL_ROUNDS = 2  # รัน 2 รอบ (1-2-3 และ 1-2-3 รวม 6 นัด)

# ป้ายประจำแต่ละตำแหน่ง
MARKER_LEFT = "2"        # เป้า 1 (ซ้ายสุด)
MARKER_CENTER = "heart"  # เป้า 2 (ตรงกลาง)
MARKER_RIGHT = "1"       # เป้า 3 (ขวาสุด)

# คำนวณมุมหัน Yaw (เป้าห่าง 0.6m, ห่างจากหุ่น 1.0m) -> arctan(0.6/1.0) ≈ 30.96 องศา
APPROX_YAW = math.degrees(math.atan(0.6 / 1.0))
BASE_PITCH = -7.0  # มุมก้มเริ่มต้น

# รูปแบบกระสุน IR
FIRE_TYPE = blaster.IR_FIRE if hasattr(blaster, "IR_FIRE") else blaster.INFRARED_FIRE

# เกณฑ์ความคลาดเคลื่อนที่เหมาะสม (ไม่หลุดเฟรม และไม่ติดค้าง)
YAW_TOLERANCE = 0.045
PITCH_TOLERANCE = 0.065
CONSECUTIVE_LOCK_FRAMES = 2  # ยืนยัน 2 เฟรมต่อเนื่อง ป้องกันกระตุกและยิงได้เร็ว

# ความเร็วขั้นต่ำเอาชนะแรงเสียดทาน (Deadband Compensation)
MIN_YAW_SPEED = 7.0
MIN_PITCH_SPEED = 11.0

# ลำดับเป้าหมาย 1 -> 2 -> 3
TARGET_SEQUENCE = [
    {"name": "เป้า 1 (ซ้ายสุด)",  "yaw": -APPROX_YAW, "pitch": BASE_PITCH, "marker": MARKER_LEFT},
    {"name": "เป้า 2 (ตรงกลาง)", "yaw":  0.0,        "pitch": BASE_PITCH, "marker": MARKER_CENTER},
    {"name": "เป้า 3 (ขวาสุด)",  "yaw":  APPROX_YAW, "pitch": BASE_PITCH, "marker": MARKER_RIGHT}
]

# ==============================================================================
# 2. คลาสคำนวณ PID Controller
# ==============================================================================
class PIDController:
    def __init__(self, kp, ki, kd, limits=(-100, 100), integral_limit=15.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_limit, self.max_limit = limits
        self.integral_limit = integral_limit
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0 or dt > 0.5:
            dt = 0.02

        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral

        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        output = max(self.min_limit, min(self.max_limit, output))

        self.last_error = error
        self.last_time = now
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

# --------------------------------------------------
# พารามิเตอร์ระบบ
# --------------------------------------------------
pid_yaw = PIDController(kp=130.0, ki=1.8, kd=9.0, limits=(-160, 160))
pid_pitch = PIDController(kp=110.0, ki=1.8, kd=7.0, limits=(-90, 90))

active_marker_target = MARKER_CENTER
is_tracking = False
target_locked = False
lock_counter = 0
last_log_time = 0.0

ep_robot = None
ep_gimbal = None
ep_blaster = None
ep_led = None

time_response_data = []
start_run_time = 0.0
current_target_yaw_ref = 0.0

def play_robot_fire_sound(bot_instance):
    try:
        if hasattr(robot, "SOUND_ID_ATTACK"):
            bot_instance.play_sound(robot.SOUND_ID_ATTACK)
        elif hasattr(robot, "SOUND_ID_SHOOT"):
            bot_instance.play_sound(robot.SOUND_ID_SHOOT)
        else:
            bot_instance.play_sound(1)
    except Exception:
        pass

def on_sub_angle(angle_info):
    global time_response_data, start_run_time, current_target_yaw_ref
    pitch_angle, yaw_angle, _, _ = angle_info
    t = time.time() - start_run_time
    time_response_data.append((t, yaw_angle, pitch_angle, current_target_yaw_ref))

def on_detect_marker(marker_info):
    global target_locked, is_tracking, lock_counter, last_log_time
    global active_marker_target, ep_robot, ep_gimbal, ep_blaster, ep_led, pid_yaw, pid_pitch

    if not is_tracking or target_locked or not active_marker_target:
        return

    # กรองเฉพาะป้ายที่ตรงกับเป้าหมายรอบนี้
    matched_markers = [m for m in marker_info if m[4] == active_marker_target]
    if not matched_markers:
        return

    best_marker = min(matched_markers, key=lambda m: (m[0] - 0.5)**2 + (m[1] - 0.5)**2)
    x, y, _, _, _ = best_marker

    err_x = x - 0.5
    err_y = 0.5 - y

    yaw_speed = pid_yaw.compute(err_x)
    pitch_speed = pid_pitch.compute(err_y)

    # Deadband Boost ดันความเร็วขั้นต่ำไม่ให้มอเตอร์ค้าง
    if abs(err_x) > 0.015 and abs(yaw_speed) < MIN_YAW_SPEED:
        yaw_speed = MIN_YAW_SPEED if yaw_speed > 0 else -MIN_YAW_SPEED
    if abs(err_y) > 0.020 and abs(pitch_speed) < MIN_PITCH_SPEED:
        pitch_speed = MIN_PITCH_SPEED if pitch_speed > 0 else -MIN_PITCH_SPEED

    if time.time() - last_log_time > 0.25:
        print(f"   [เล็ง '{active_marker_target}'] ErrX: {err_x:+.3f}, ErrY: {err_y:+.3f} -> ความเร็ว Yaw: {yaw_speed:+.1f}, Pitch: {pitch_speed:+.1f}")
        last_log_time = time.time()

    # ตรวจสอบเงื่อนไขล็อกกึ่งกลางเป้าหมาย
    if abs(err_x) < YAW_TOLERANCE and abs(err_y) < PITCH_TOLERANCE:
        lock_counter += 1
        if lock_counter >= CONSECUTIVE_LOCK_FRAMES:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            is_tracking = False
            target_locked = True

            print("\n" + "="*55)
            print(f"🎯 [LOCKED] ล็อกกึ่งกลางเป้าหมาย '{active_marker_target}' สำเร็จ! (ErrX: {err_x:.3f}, ErrY: {err_y:.3f})")
            print("💥 [FIRE!] สั่งยิงกระสุน IR 1 นัด!")
            print("="*55 + "\n")

            ep_blaster.fire(fire_type=FIRE_TYPE, times=1)
            play_robot_fire_sound(ep_robot)
            ep_led.set_led(comp="all", r=255, g=0, b=0, effect=led.EFFECT_FLASH)

            if HAS_WINSOUND:
                winsound.Beep(2000, 150)

            pid_yaw.reset()
            pid_pitch.reset()
    else:
        lock_counter = 0
        ep_gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)

# ==============================================================================
# 3. ฟังก์ชันหลัก (Main Sequence)
# ==============================================================================
def main():
    global ep_robot, ep_gimbal, ep_blaster, ep_led, is_tracking, target_locked, lock_counter
    global start_run_time, current_target_yaw_ref, active_marker_target

    print("=======================================================")
    print(f"🚀 เริ่มภารกิจ RoboMaster Auto-Aim (ยิงแบบ 1-2-3 | {TOTAL_ROUNDS} รอบ)")
    print(f"🎯 รูปแบบป้าย: ซ้าย='{MARKER_LEFT}', กลาง='{MARKER_CENTER}', ขวา='{MARKER_RIGHT}'")
    print("=======================================================")

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")  # เปลี่ยนเป็น "sta" ได้หากต่อผ่าน Router

    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_led = ep_robot.led

    ep_led.set_led(comp="all", r=0, g=255, b=0, effect=led.EFFECT_ON)

    print("\n[1/3] เริ่มต้นระบบกล้องและเซ็ตตำแหน่ง Gimbal...")
    ep_camera.start_video_stream(display=False)
    ep_gimbal.recenter().wait_for_completed()
    time.sleep(1)

    start_run_time = time.time()
    current_target_yaw_ref = 0.0
    ep_gimbal.sub_angle(freq=20, callback=on_sub_angle)

    print("[2/3] เปิดระบบตรวจจับ Vision Marker...")
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    time.sleep(0.5)

    print(f"\n[3/3] เริ่มต้นการยิงเป้าหมาย...")

    for r_idx in range(1, TOTAL_ROUNDS + 1):
        print(f"\n#######################################################")
        print(f"🔄 เริ่มต้นการยิง [รอบที่ {r_idx}/{TOTAL_ROUNDS}]")
        print(f"#######################################################")

        for idx, target in enumerate(TARGET_SEQUENCE):
            active_marker_target = target["marker"]
            current_target_yaw_ref = target["yaw"]
            target_locked = False
            lock_counter = 0

            print(f"\n>>> กำลังมุ่งหน้าสู่ [รอบ {r_idx} | เป้า {idx + 1}/3] {target['name']} (ป้าย '{target['marker']}') <<<")

            # หมุนหยาบ (Coarse Move)
            ep_gimbal.moveto(pitch=target["pitch"], yaw=target["yaw"],
                             pitch_speed=160, yaw_speed=160).wait_for_completed()

            # เริ่มเล็งละเอียดด้วย PID
            pid_yaw.reset()
            pid_pitch.reset()
            is_tracking = True

            wait_start = time.time()
            while not target_locked and (time.time() - wait_start < 5.5):
                time.sleep(0.02)

            if target_locked:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                time.sleep(0.8)
                ep_led.set_led(comp="all", r=0, g=255, b=0, effect=led.EFFECT_ON)
                print(f"✅ ยิง {target['name']} สำเร็จเรียบร้อย!")
            else:
                print(f"⚠️ [TIMEOUT] ไม่สามารถล็อกกึ่งกลาง {target['name']} ทันเวลา -> ข้ามไปเป้าถัดไป")
                is_tracking = False
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

            time.sleep(0.3)

    print("\n=======================================================")
    print("✅ สิ้นสุดภารกิจ! กำลังบันทึกผลลัพธ์ลงไฟล์...")
    print("=======================================================")

    export_results()

    try:
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        ep_vision.unsub_detect_info(name="marker")
        ep_gimbal.unsub_angle()
        ep_camera.stop_video_stream()
        ep_robot.close()
        print(">> ปิดระบบหุ่นยนต์เรียบร้อย")
    except Exception as e:
        print(f">> แจ้งเตือนขณะปิดหุ่นยนต์: {e}")

# ==============================================================================
# 4. ฟังก์ชันส่งออกไฟล์ CSV และกราฟ Time Response
# ==============================================================================
def export_results():
    if not time_response_data:
        print("⚠️ ไม่มีข้อมูลมุม Gimbal สำหรับบันทึก")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
    csv_filename = os.path.join(script_dir, "gimbal_time_response.csv")
    plot_filename = os.path.join(script_dir, "time_response_plot.png")

    try:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_s", "Yaw_Angle_deg", "Pitch_Angle_deg", "Target_Yaw_deg"])
            writer.writerows(time_response_data)
        print(f"📄 บันทึกไฟล์ CSV สำเร็จ: {csv_filename}")
    except Exception as err:
        print(f"❌ บันทึก CSV ล้มเหลว: {err}")

    if HAS_MATPLOTLIB:
        try:
            times = [r[0] for r in time_response_data]
            yaws = [r[1] for r in time_response_data]
            pitches = [r[2] for r in time_response_data]
            targets = [r[3] for r in time_response_data]

            plt.figure(figsize=(12, 6))
            plt.plot(times, targets, 'r--', label='Target Coarse Yaw (deg)', linewidth=1.5)
            plt.plot(times, yaws, 'b-', label='Actual Gimbal Yaw (deg)', linewidth=1.5)
            plt.plot(times, pitches, 'g-', label='Actual Gimbal Pitch (deg)', linewidth=1.2)

            plt.title(f'Gimbal PID Response Tracking (1-2-3 | {TOTAL_ROUNDS} Rounds)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Angle (degrees)')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper right')
            plt.tight_layout()

            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print(f"📊 บันทึกรูปกราฟสำเร็จ: {plot_filename} (พร้อมนำไปใส่เล่มรายงาน)")
        except Exception as err:
            print(f"❌ บันทึกรูปภาพกราฟล้มเหลว: {err}")

if __name__ == '__main__':
    main()