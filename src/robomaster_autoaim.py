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
    # ตั้ง font รองรับภาษาไทย กัน glyph หายตอน savefig (Tahoma/Leelawadee มีมากับ Windows ทุกเครื่อง)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
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

# จุด engage deadband (error เกินนี้ถึงเริ่มดันความเร็วขั้นต่ำ)
YAW_DEADBAND_ENGAGE = 0.015
PITCH_DEADBAND_ENGAGE = 0.020

# error ขนาดนี้ถึงจะได้ความเร็วเต็ม MIN_SPEED, ต่ำกว่านี้ scale ลงตามสัดส่วน
# (แก้ limit cycle ที่เจอจากข้อมูลจริง: เดิมยัด MIN_SPEED คงที่ทันทีที่ error ผ่าน engage
#  ทำให้ error เล็กแค่ 0.02 ก็โดนความเร็วเท่า error 0.05 -> วิ่งเลย target ไปมา)
YAW_DEADBAND_FULL_ERROR = 0.05
PITCH_DEADBAND_FULL_ERROR = 0.07
DEADBAND_FLOOR_RATIO = 0.4  # อย่างน้อยยังขยับด้วยสัดส่วนนี้ของ MIN_SPEED กันมอเตอร์ค้างสนิท

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
    def __init__(self, kp, ki, kd, limits=(-100, 100), integral_limit=15.0, d_filter_alpha=0.35):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_limit, self.max_limit = limits
        self.integral_limit = integral_limit
        # กรอง derivative ด้วย low-pass (EMA) กัน noise จาก marker detection กระตุก
        self.d_filter_alpha = d_filter_alpha

        self.last_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
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

        raw_derivative = (error - self.last_error) / dt
        # low-pass filter บน derivative กัน spike จาก noise/เฟรมกระตุก
        self.filtered_derivative = (self.d_filter_alpha * raw_derivative
                                     + (1 - self.d_filter_alpha) * self.filtered_derivative)
        d_term = self.kd * self.filtered_derivative

        output = p_term + i_term + d_term
        saturated = output > self.max_limit or output < self.min_limit
        output = max(self.min_limit, min(self.max_limit, output))

        self.last_error = error
        self.last_time = now
        return output, p_term, i_term, d_term, saturated

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
        self.last_time = time.time()

# --------------------------------------------------
# พารามิเตอร์ระบบ
# --------------------------------------------------
pid_yaw = PIDController(kp=130.0, ki=1.8, kd=9.0, limits=(-160, 160))
pid_pitch = PIDController(kp=110.0, ki=1.8, kd=7.0, limits=(-90, 90))

active_marker_target = MARKER_CENTER
active_target_name = ""
is_tracking = False
target_locked = False
lock_counter = 0
last_log_time = 0.0

ep_robot = None
ep_gimbal = None
ep_blaster = None
ep_led = None

# --- Log 1: มุมกิมบอลจริงทั้งหมด (coarse move + fine aim) ใช้ดู overall tracking ---
time_response_data = []
start_run_time = 0.0
current_target_yaw_ref = 0.0
current_phase = "IDLE"  # "COARSE" ระหว่าง moveto, "FINE" ระหว่าง PID lock เอง

# --- Log 2: เฉพาะช่วง fine-aim ที่ PID เขียนเองควบคุมจริง (err, output, terms) ---
pid_response_data = []


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
    global time_response_data, start_run_time, current_target_yaw_ref, current_phase
    pitch_angle, yaw_angle, _, _ = angle_info
    t = time.time() - start_run_time
    time_response_data.append((t, yaw_angle, pitch_angle, current_target_yaw_ref, current_phase))


def on_detect_marker(marker_info):
    global target_locked, is_tracking, lock_counter, last_log_time
    global active_marker_target, active_target_name, ep_robot, ep_gimbal, ep_blaster, ep_led
    global pid_yaw, pid_pitch, pid_response_data, start_run_time

    if not is_tracking or target_locked or not active_marker_target:
        return

    # กรองเฉพาะป้ายที่ตรงกับเป้าหมายรอบนี้
    matched_markers = [m for m in marker_info if m[4] == active_marker_target]
    if not matched_markers:
        return

    best_marker = min(matched_markers, key=lambda m: (m[0] - 0.5) ** 2 + (m[1] - 0.5) ** 2)
    x, y, _, _, _ = best_marker

    err_x = x - 0.5
    err_y = 0.5 - y

    yaw_speed, yaw_p, yaw_i, yaw_d, yaw_sat = pid_yaw.compute(err_x)
    pitch_speed, pitch_p, pitch_i, pitch_d, pitch_sat = pid_pitch.compute(err_y)

    # Deadband Boost แบบ scale ตาม error ขนาด (ไม่ยัดความเร็วคงที่ทันทีเหมือนเดิม)
    if abs(err_x) > YAW_DEADBAND_ENGAGE and abs(yaw_speed) < MIN_YAW_SPEED:
        scale = max(DEADBAND_FLOOR_RATIO, min(1.0, abs(err_x) / YAW_DEADBAND_FULL_ERROR))
        boosted = MIN_YAW_SPEED * scale
        sign = 1.0 if (yaw_speed > 0 or (yaw_speed == 0 and err_x > 0)) else -1.0
        yaw_speed = boosted * sign

    if abs(err_y) > PITCH_DEADBAND_ENGAGE and abs(pitch_speed) < MIN_PITCH_SPEED:
        scale = max(DEADBAND_FLOOR_RATIO, min(1.0, abs(err_y) / PITCH_DEADBAND_FULL_ERROR))
        boosted = MIN_PITCH_SPEED * scale
        sign = 1.0 if (pitch_speed > 0 or (pitch_speed == 0 and err_y > 0)) else -1.0
        pitch_speed = boosted * sign

    # เก็บ log ของวง PID เอง ใช้ทำกราฟ error/output จริงตอน fine-aim
    pid_response_data.append((
        time.time() - start_run_time, active_target_name,
        err_x, err_y, yaw_speed, pitch_speed,
        yaw_p, yaw_i, yaw_d, yaw_sat,
        pitch_p, pitch_i, pitch_d, pitch_sat,
    ))

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

            print("\n" + "=" * 55)
            print(f"🎯 [LOCKED] ล็อกกึ่งกลางเป้าหมาย '{active_marker_target}' สำเร็จ! (ErrX: {err_x:.3f}, ErrY: {err_y:.3f})")
            print("💥 [FIRE!] สั่งยิงกระสุน IR 1 นัด!")
            print("=" * 55 + "\n")

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
    global start_run_time, current_target_yaw_ref, active_marker_target, active_target_name
    global current_phase

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
    current_phase = "IDLE"
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
            active_target_name = f"R{r_idx}-{target['name']}"
            current_target_yaw_ref = target["yaw"]
            target_locked = False
            lock_counter = 0

            print(f"\n>>> กำลังมุ่งหน้าสู่ [รอบ {r_idx} | เป้า {idx + 1}/3] {target['name']} (ป้าย '{target['marker']}') <<<")

            # หมุนหยาบ (Coarse Move) — ควบคุมโดย firmware ของหุ่นเอง ไม่ใช่ PID ที่เขียนเอง
            current_phase = "COARSE"
            ep_gimbal.moveto(pitch=target["pitch"], yaw=target["yaw"],
                              pitch_speed=160, yaw_speed=160).wait_for_completed()

            # เริ่มเล็งละเอียดด้วย PID ที่เขียนเอง (วงนี้คือของที่ต้อง tune จริง)
            pid_yaw.reset()
            pid_pitch.reset()
            current_phase = "FINE"
            is_tracking = True

            wait_start = time.time()
            while not target_locked and (time.time() - wait_start < 5.5):
                time.sleep(0.02)

            current_phase = "IDLE"

            if target_locked:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                time.sleep(0.8)
                ep_led.set_led(comp="all", r=0, g=255, b=0, effect=led.EFFECT_ON)
                print(f"✅ ยิง {target['name']} สำเร็จเรียบร้อย!")
            else:
                print(f"⚠️ [TIMEOUT] ไม่สามารถล็อกกึ่งกลาง {target['name']} ทันเวลา -> ข้ามไปเป้าถัดไป")
                is_tracking = False
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                # กัน integral ค้างข้ามเป้าถัดไปตอน timeout
                pid_yaw.reset()
                pid_pitch.reset()

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
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

    # ---------- (A) Log ภาพรวมมุมจริงทั้งหมด (coarse + fine) ----------
    if time_response_data:
        overall_csv = os.path.join(script_dir, "gimbal_time_response.csv")
        try:
            with open(overall_csv, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["Time_s", "Yaw_Angle_deg", "Pitch_Angle_deg", "Target_Yaw_deg", "Phase"])
                writer.writerows(time_response_data)
            print(f"📄 บันทึกไฟล์ CSV (overall) สำเร็จ: {overall_csv}")
        except Exception as err:
            print(f"❌ บันทึก overall CSV ล้มเหลว: {err}")
    else:
        print("⚠️ ไม่มีข้อมูลมุม Gimbal สำหรับบันทึก")

    # ---------- (B) Log เฉพาะวง PID เอง (fine-aim) ----------
    if pid_response_data:
        pid_csv = os.path.join(script_dir, "pid_fine_aim_response.csv")
        try:
            with open(pid_csv, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time_s", "Target", "ErrX", "ErrY", "YawSpeedCmd", "PitchSpeedCmd",
                    "YawP", "YawI", "YawD", "YawSaturated",
                    "PitchP", "PitchI", "PitchD", "PitchSaturated",
                ])
                writer.writerows(pid_response_data)
            print(f"📄 บันทึกไฟล์ CSV (fine-aim PID) สำเร็จ: {pid_csv}")
        except Exception as err:
            print(f"❌ บันทึก PID CSV ล้มเหลว: {err}")
    else:
        print("⚠️ ไม่มีข้อมูล fine-aim PID สำหรับบันทึก")

    if not HAS_MATPLOTLIB:
        return

    # ---------- กราฟ (A): ภาพรวมมุมจริง พร้อมแรเงาช่วง fine-aim ----------
    if time_response_data:
        try:
            times = [r[0] for r in time_response_data]
            yaws = [r[1] for r in time_response_data]
            pitches = [r[2] for r in time_response_data]
            targets = [r[3] for r in time_response_data]
            phases = [r[4] for r in time_response_data]

            plt.figure(figsize=(12, 6))
            plt.plot(times, targets, 'r--', label='Target Coarse Yaw (deg)', linewidth=1.5)
            plt.plot(times, yaws, 'b-', label='Actual Gimbal Yaw (deg)', linewidth=1.5)
            plt.plot(times, pitches, 'g-', label='Actual Gimbal Pitch (deg)', linewidth=1.2)

            # แรเงาช่วง FINE ให้เห็นชัดว่าช่วงไหนคือ PID เองทำงาน (ไม่ใช่ coarse moveto)
            in_fine = False
            fine_start = None
            for i, ph in enumerate(phases):
                if ph == "FINE" and not in_fine:
                    in_fine = True
                    fine_start = times[i]
                elif ph != "FINE" and in_fine:
                    in_fine = False
                    plt.axvspan(fine_start, times[i], color='orange', alpha=0.15)
            if in_fine:
                plt.axvspan(fine_start, times[-1], color='orange', alpha=0.15)

            plt.title(f'Gimbal Time Response (Overall) — orange = fine-aim PID window')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Angle (degrees)')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(script_dir, "time_response_plot.png"), dpi=300)
            plt.close()
            print("📊 บันทึกกราฟ overall สำเร็จ (time_response_plot.png)")
        except Exception as err:
            print(f"❌ บันทึกกราฟ overall ล้มเหลว: {err}")

    # ---------- กราฟ (B): เฉพาะวง PID fine-aim — err_x/err_y และ output ----------
    if pid_response_data:
        try:
            nan = float('nan')
            t, err_x, err_y, yaw_cmd, pitch_cmd = [], [], [], [], []
            boundary_times = []  # เวลาเริ่มแต่ละ episode ใช้ตีเส้นแบ่งบนกราฟ
            prev_target = None
            for row in pid_response_data:
                ts, tgt, ex, ey, yc, pc = row[0], row[1], row[2], row[3], row[4], row[5]
                if prev_target is not None and tgt != prev_target:
                    # แทรก NaN คั่นระหว่าง episode กันเส้นเชื่อมข้ามช่วง coarse move
                    # (ช่วงที่ไม่ได้ log เพราะยังไม่เข้า fine-aim ของเป้าถัดไป)
                    t.append(nan); err_x.append(nan); err_y.append(nan)
                    yaw_cmd.append(nan); pitch_cmd.append(nan)
                if tgt != prev_target:
                    boundary_times.append(ts)
                t.append(ts); err_x.append(ex); err_y.append(ey)
                yaw_cmd.append(yc); pitch_cmd.append(pc)
                prev_target = tgt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(t, err_x, 'b-o', markersize=3, label='Err_X (yaw, normalized)')
            axes[0].plot(t, err_y, 'g-o', markersize=3, label='Err_Y (pitch, normalized)')
            axes[0].axhline(YAW_TOLERANCE, color='b', linestyle=':', linewidth=1, label='Yaw tolerance')
            axes[0].axhline(-YAW_TOLERANCE, color='b', linestyle=':', linewidth=1)
            axes[0].axhline(PITCH_TOLERANCE, color='g', linestyle=':', linewidth=1, label='Pitch tolerance')
            axes[0].axhline(-PITCH_TOLERANCE, color='g', linestyle=':', linewidth=1)
            for bt in boundary_times:
                axes[0].axvline(bt, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
            axes[0].set_ylabel('Normalized error')
            axes[0].set_title('Fine-aim PID: error vs time (แต่ละ episode คั่นด้วยเส้นเทา, จุด = sample จริง)')
            axes[0].legend(loc='upper right', fontsize=8)
            axes[0].grid(True, linestyle=':', alpha=0.6)

            axes[1].plot(t, yaw_cmd, 'b-o', markersize=3, label='Yaw speed cmd')
            axes[1].plot(t, pitch_cmd, 'g-o', markersize=3, label='Pitch speed cmd')
            for bt in boundary_times:
                axes[1].axvline(bt, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Speed command')
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].grid(True, linestyle=':', alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(script_dir, "pid_fine_aim_plot.png"), dpi=300)
            plt.close()
            print("📊 บันทึกกราฟ fine-aim PID สำเร็จ (pid_fine_aim_plot.png)")
        except Exception as err:
            print(f"❌ บันทึกกราฟ fine-aim PID ล้มเหลว: {err}")


if __name__ == '__main__':
    main()
