"""Class Work 8 - SLAM: Explore the Unknown World.

Builds the 3 deliverables:
  - Map             : Occupancy-grid PNG (visited cells + sensed walls)
  - Robot Trajectory: PNG of the real (x, y) path with start/end marked
  - Accuracy report : Map Accuracy % and Coverage %
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
try:
    from config_loader import load_config
except ImportError:
    def load_config():
        return {}


# ใส่พิกัดของช่องที่หุ่นสามารถเดินได้จริง (Free Cells) เมื่อทราบผังเขาวงกตจริง
# ตัวอย่าง: {(0, 0), (0, 1), (1, 1), ...}
GROUND_TRUTH_FREE_CELLS = None


def get_latest_file(data_dir, pattern):
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def classify_wall(distance_cm, wall_threshold_cm=45.0):
    """
    ตัดสินว่ามีกำแพงหรือไม่:
    - distance <= 0 หรือเป็น NaN: เซนเซอร์อ่านค่าหลุด (Invalid)
    - distance <= wall_threshold_cm: มีกำแพงขวาง (Wall Present -> True)
    - distance > wall_threshold_cm: ทางเปิดโล่ง (Open -> False)
    """
    if distance_cm is None or pd.isna(distance_cm):
        return None
    try:
        val = float(distance_cm)
    except (ValueError, TypeError):
        return None

    if val <= 0.0 or val > 600.0:  # กรองค่าหลุดสเกลและค่าสะท้อนพลาด
        return None

    return val <= wall_threshold_cm


def build_wall_votes(df, wall_threshold_cm=45.0):
    """
    นับคะแนนเสียงโหวตกำแพงในแต่ละทิศทางของแต่ละเซลล์
    Direction Index: 0=N(+y), 1=E(+x), 2=S(-y), 3=W(-x)
    """
    votes = {}  # (gx, gy) -> {dir: [wall_votes, open_votes]}

    def add_vote(cell, direction, is_wall):
        if is_wall is None:
            return
        d = votes.setdefault(cell, {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]})
        d[direction][0 if is_wall else 1] += 1

    for _, row in df.iterrows():
        # ข้ามขั้นตอน RETRACE เพื่อไม่ให้ค่าย้อนทางมาทำลายความถูกต้องของการสแกน 360 องศา
        if str(row.get("action", "")).strip().upper() == "RETRACE":
            continue

        try:
            cell = (int(row["grid_x"]), int(row["grid_y"]))
            h = int(row["heading"])
        except (ValueError, KeyError):
            continue

        # แปลงหน่วยระยะทางทุกทิศทางให้เป็นเซนติเมตร (cm) ก่อนส่งเข้าฟังก์ชัน
        # 1. Front (เดิมเก็บเป็น mm)
        f_val = row.get("front_tof_mm")
        f_cm = (f_val / 10.0) if pd.notna(f_val) else None
        front_wall = classify_wall(f_cm, wall_threshold_cm)

        # 2. Right (เดิมเก็บเป็น cm)
        r_cm = row.get("right_ir_cm")
        right_wall = classify_wall(r_cm, wall_threshold_cm)

        # 3. Back (เดิมเก็บเป็น cm)
        b_cm = row.get("back_ir_cm")
        if pd.isna(b_cm) and "back_tof_mm" in row:
            b_cm = row["back_tof_mm"] / 10.0
        back_wall = classify_wall(b_cm, wall_threshold_cm)

        # 4. Left (เดิมเก็บเป็น cm)
        l_cm = row.get("left_ir_cm")
        left_wall = classify_wall(l_cm, wall_threshold_cm)

        # ลงคะแนนตามระนาบสัมบูรณ์ (Global Orientation)
        add_vote(cell, h, front_wall)
        add_vote(cell, (h + 1) % 4, right_wall)
        add_vote(cell, (h + 2) % 4, back_wall)
        add_vote(cell, (h + 3) % 4, left_wall)

    walls = {}
    for cell, dirs in votes.items():
        walls[cell] = {}
        for d, (wall_v, open_v) in dirs.items():
            if wall_v == 0 and open_v == 0:
                continue
            # ใช้ Majority Voting: หากเสียงเท่ากันให้มองเป็นกำแพงเพื่อความปลอดภัย (Conservative)
            walls[cell][d] = wall_v >= open_v

    return walls


def plot_map(df, walls, max_x, max_y, start_cell, end_cell, out_path):
    fig, ax = plt.subplots(figsize=(max(7, max_x + 2.5), max(6, max_y + 2.5)))

    visited = {(int(r["grid_x"]), int(r["grid_y"])) for _, r in df.iterrows() if pd.notna(r.get("grid_x"))}

    # วาดพื้นตารางกริด
    for gx in range(max_x + 1):
        for gy in range(max_y + 1):
            color = "#E5EFF7" if (gx, gy) in visited else "#FAFAFA"
            ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, facecolor=color, edgecolor="#D3D3D3", linewidth=1, zorder=1))
            ax.text(gx, gy, f"({gx},{gy})", color="#888888", fontsize=9, ha="center", va="center", zorder=2)

    # พิกัดเส้นขอบกำแพงแต่ละทิศ (0=N, 1=E, 2=S, 3=W)
    seg = {
        0: lambda gx, gy: ((gx - 0.5, gy + 0.5), (gx + 0.5, gy + 0.5)),  # N edge
        1: lambda gx, gy: ((gx + 0.5, gy - 0.5), (gx + 0.5, gy + 0.5)),  # E edge
        2: lambda gx, gy: ((gx - 0.5, gy - 0.5), (gx + 0.5, gy - 0.5)),  # S edge
        3: lambda gx, gy: ((gx - 0.5, gy - 0.5), (gx - 0.5, gy + 0.5)),  # W edge
    }

    # วาดกำแพงรอบนอกสนามอัตโนมัติ (Boundary Walls)
    for gx in range(max_x + 1):
        ax.plot([gx - 0.5, gx + 0.5], [-0.5, -0.5], color="black", linewidth=4.5, zorder=4)
        ax.plot([gx - 0.5, gx + 0.5], [max_y + 0.5, max_y + 0.5], color="black", linewidth=4.5, zorder=4)
    for gy in range(max_y + 1):
        ax.plot([-0.5, -0.5], [gy - 0.5, gy + 0.5], color="black", linewidth=4.5, zorder=4)
        ax.plot([max_x + 0.5, max_x + 0.5], [gy - 0.5, gy + 0.5], color="black", linewidth=4.5, zorder=4)

    # วาดกำแพงภายในที่ตรวจพบจากการสแกน
    for (gx, gy), dirs in walls.items():
        for d, is_wall in dirs.items():
            if not is_wall:
                continue
            (x0, y0), (x1, y1) = seg[d](gx, gy)
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=4, zorder=5, solid_capstyle="round")

    # พล็อตจุดเริ่มต้นและจุดสิ้นสุด
    ax.plot(*start_cell, marker="s", color="#2CA02C", markersize=13, label=f"Start {start_cell}", zorder=6)
    ax.plot(*end_cell, marker="X", color="#D62728", markersize=14, label=f"End {end_cell}", zorder=6)

    ax.set_xticks(range(max_x + 1))
    ax.set_yticks(range(max_y + 1))
    ax.set_xlim(-0.6, max_x + 0.6)
    ax.set_ylim(-0.6, max_y + 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Grid X (East $\rightarrow$)")
    ax.set_ylabel(r"Grid Y (North $\uparrow$)")
    ax.set_title(f"SLAM Occupancy Grid Map\nVisited: {len(visited)}/{(max_x + 1) * (max_y + 1)} Cells", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(df, start_cell, end_cell, cell_size, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))

    # กรองเอาพิกัด real_x_m, real_y_m
    valid_pts = df.dropna(subset=["real_x_m", "real_y_m"])
    ax.plot(valid_pts["real_x_m"], valid_pts["real_y_m"], color="#1F77B4", linewidth=2.0,
            marker="o", markersize=4, zorder=2, label="Odometry Path")

    start_xy = (start_cell[0] * cell_size, start_cell[1] * cell_size)
    end_xy = (end_cell[0] * cell_size, end_cell[1] * cell_size)
    ax.plot(*start_xy, marker="s", color="#2CA02C", markersize=13, label="Start Pos", zorder=3)
    ax.plot(*end_xy, marker="X", color="#D62728", markersize=14, label="End Pos", zorder=3)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title("Robot Real Trajectory (Dead-Reckoning Odometry)", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_accuracy(visited, max_x, max_y, ground_truth_free_cells):
    total_cells = (max_x + 1) * (max_y + 1)
    if ground_truth_free_cells is None:
        ground_truth_free_cells = {(gx, gy) for gx in range(max_x + 1) for gy in range(max_y + 1)}

    correct = 0
    for gx in range(max_x + 1):
        for gy in range(max_y + 1):
            predicted_free = (gx, gy) in visited
            actual_free = (gx, gy) in ground_truth_free_cells
            if predicted_free == actual_free:
                correct += 1

    coverage_pct = round(100.0 * len(visited) / total_cells, 2) if total_cells else 0.0
    accuracy_pct = round(100.0 * correct / total_cells, 2) if total_cells else 0.0
    return coverage_pct, accuracy_pct, total_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Path to exploration_map_data csv")
    parser.add_argument("--ground-truth-json", default=None, help="JSON file with list of [x,y] free cells")
    parser.add_argument("--threshold", type=float, default=45.0, help="Wall distance threshold in cm (Default: 45.0)")
    args = parser.parse_args()

    config = load_config()
    grid_cfg = config.get("grid_map", {})
    max_x = grid_cfg.get("max_x", 4)
    max_y = grid_cfg.get("max_y", 3)
    cell_size = config.get("movement", {}).get("distance", 0.6)
    data_dir = config.get("data_collection", {}).get("data_dir", "data/raw/run1")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_abs = os.path.join(base_dir, data_dir)

    log_path = args.log or get_latest_file(data_dir_abs, "*exploration_map_data*.csv")
    if not log_path:
        print(f"[-] ไม่พบไฟล์ Exploration Log ในโฟลเดอร์ {data_dir_abs}")
        return 1

    print(f"[+] กำลังประมวลผล Log: {log_path}")
    df = pd.read_csv(log_path)
    if df.empty:
        print("[-] ไฟล์ Log ไม่มีข้อมูล")
        return 1

    visited = {(int(r["grid_x"]), int(r["grid_y"])) for _, r in df.iterrows() if pd.notna(r.get("grid_x"))}
    start_cell = (int(df.iloc[0]["grid_x"]), int(df.iloc[0]["grid_y"]))

    non_retrace = df[df["action"] != "RETRACE"]
    last_real_row = non_retrace.iloc[-1] if not non_retrace.empty else df.iloc[-1]
    end_cell = (int(last_real_row["grid_x"]), int(last_real_row["grid_y"]))

    walls = build_wall_votes(df, wall_threshold_cm=args.threshold)

    ground_truth_free_cells = GROUND_TRUTH_FREE_CELLS
    if args.ground_truth_json:
        with open(args.ground_truth_json, "r", encoding="utf-8") as f:
            ground_truth_free_cells = {tuple(c) for c in json.load(f)}

    coverage_pct, accuracy_pct, total_cells = compute_accuracy(
        visited, max_x, max_y, ground_truth_free_cells
    )

    map_png = os.path.join(data_dir_abs, "slam_map.png")
    traj_png = os.path.join(data_dir_abs, "slam_trajectory.png")
    plot_map(df, walls, max_x, max_y, start_cell, end_cell, map_png)
    plot_trajectory(df, start_cell, end_cell, cell_size, traj_png)

    report = {
        "log_file": log_path,
        "start_grid": list(start_cell),
        "start_real_m": [round(start_cell[0] * cell_size, 3), round(start_cell[1] * cell_size, 3)],
        "end_grid": list(end_cell),
        "end_real_m": [round(end_cell[0] * cell_size, 3), round(end_cell[1] * cell_size, 3)],
        "visited_cells": len(visited),
        "total_cells": total_cells,
        "coverage_percent": coverage_pct,
        "map_accuracy_percent": accuracy_pct,
        "ground_truth_used": ground_truth_free_cells is not None,
        "map_image": map_png,
        "trajectory_image": traj_png,
    }
    report_path = os.path.join(data_dir_abs, "slam_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SLAM Mission Report")
    print("=" * 60)
    print(f"Start Position : Grid={report['start_grid']} | Real={report['start_real_m']} m")
    print(f"End Position   : Grid={report['end_grid']} | Real={report['end_real_m']} m")
    print(f"Coverage       : {report['visited_cells']}/{total_cells} Cells ({coverage_pct}%)")
    print(f"Map Accuracy   : {accuracy_pct}%" + ("" if report["ground_truth_used"] else " (Placeholder)"))
    print(f"Map Image      : {map_png}")
    print(f"Trajectory Img : {traj_png}")
    print(f"Report JSON    : {report_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())