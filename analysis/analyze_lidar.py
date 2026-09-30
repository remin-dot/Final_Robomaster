import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def get_latest_file(data_dir, pattern):
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def plot_lidar_map(data_dir, save_image=True):
    # ค้นหาไฟล์ log lidar ล่าสุด
    lidar_file = get_latest_file(data_dir, '*gimbal_lidar_data*.csv')
    if not lidar_file:
        print(f"[-] ไม่พบไฟล์ log gimbal lidar ใน: {data_dir}")
        return

    print(f"[+] กำลังประมวลผลไฟล์: {os.path.basename(lidar_file)}")
    df = pd.read_csv(lidar_file)

    if df.empty:
        print("[-] ไฟล์ไม่มีข้อมูลสแกน")
        return

    # แปลง Polar (yaw, distance) -> Cartesian (X, Y ในหน่วยเมตร)
    x_coords = []
    y_coords = []

    for _, row in df.iterrows():
        yaw_rad = math.radians(float(row['yaw_deg']))
        dist_m = float(row['distance_mm']) / 1000.0  # ปรับสเกลเป็นเมตรให้เข้ากับ analyze_logs.py

        # แกน +Y คือด้านหน้า, +X คือด้านขวา
        x = (dist_m + 0.70) * math.sin(yaw_rad)
        y = (dist_m + 0.70) * math.cos(yaw_rad)
        x_coords.append(x)
        y_coords.append(y)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x_coords, y_coords, color='crimson', s=25, label='Lidar Obstacles', zorder=2)
    ax.scatter([0], [0], color='dodgerblue', marker='^', s=160, label='Robot Center (0,0)', zorder=3)

    # วาดแกนกึ่งกลาง
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)


    ax.set_title('RoboMaster Gimbal LiDAR 2D Map', fontsize=14, pad=10)
    ax.set_xlabel('X (meters) [Left - / Right +]', fontsize=11)
    ax.set_ylabel('Y (meters) [Forward +]', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    if save_image:
        out_img = os.path.join(data_dir, 'gimbal_lidar_map.png')
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"[+] บันทึกรูปแผนที่สำเร็จ: {out_img}")

    plt.show()

if __name__ == '__main__':
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'run1'))
    plot_lidar_map(target_dir)