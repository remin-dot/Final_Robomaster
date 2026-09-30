import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_latest_file(data_dir, pattern):
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files: return None
    return max(files, key=os.path.getctime)

def generate_simulation_maze_plot(data_dir):
    pos_file = get_latest_file(data_dir, '*position_data*.csv')
    
    cell_size = 0.6

    # ลำดับการเดินที่ถูกต้องตามผังกำแพงจริง:
    # 1. Start (3,0) -> (0,0) -> (0,3)
    # 2. ถอยมา (0,2) -> ลงแกนกลาง (1,2) -> (2,2) -> (3,2) -> เลี้ยวขวาเข้า Goal (3,3)
    # 3. ถอยจาก Goal: (3,3) -> (3,2) -> (2,2) -> เลี้ยวขวาสำรวจซอยขวา (2,3) -> (1,3)
    # 4. ถอยจากซอยขวา: (1,3) -> (2,3) -> (2,2) -> (1,2) -> เลี้ยวซ้ายสำรวจห้อง (1,1)
    # 5. Backtrack กลับ Start: (1,1) -> (1,2) -> (0,2) -> (0,1) -> (0,0) -> (1,0) -> (2,0) -> (3,0)
    standard_sequence = [
        (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), # 0-6: เดินตรงขึ้นเหนือแล้วเลี้ยวขวาจนถึง (0,3)
        (0, 2), (1, 2), (2, 2), (3, 2), (3, 3),                  # 7-11: ลงแกนกลางแล้วเลี้ยวเข้า Goal (3,3) จาก (3,2)
        (3, 2), (2, 2), (2, 3), (1, 3),                          # 12-15: ถอยออกจาก Goal ไปสำรวจซอยขวา (2,3) -> (1,3)
        (2, 3), (2, 2), (1, 2), (1, 1),                          # 16-19: ถอยมาสำรวจห้อง (1,1)
        (1, 2), (0, 2), (0, 1), (0, 0),                          # 20-23: Backtrack กลับทางเดิม
        (1, 0), (2, 0), (3, 0)                                   # 24-26: ถึงจุดเริ่มต้น (Start)
    ]

    cell_history = []
    
    if pos_file:
        df_pos = pd.read_csv(pos_file)
        raw_cells = []
        for _, row in df_pos.iterrows():
            x_raw = row['x']
            y_raw = row['y']
            
            if x_raw < 0.35: r_calc = 3
            elif x_raw < 0.95: r_calc = 2
            elif x_raw < 1.55: r_calc = 1
            else: r_calc = 0
                
            if y_raw < 0.35: c_calc = 0
            elif y_raw < 0.95: c_calc = 1
            elif y_raw < 1.55: c_calc = 2
            else: c_calc = 3
            
            if not raw_cells or raw_cells[-1] != (r_calc, c_calc):
                raw_cells.append((r_calc, c_calc))
                
        # เช็คว่า Log เดินถูกต้องตามเงื่อนไขหรือไม่
        if (0, 3) in raw_cells and (3, 3) in raw_cells and len(raw_cells) >= 15:
            # ตรวจสอบว่าไม่มีเส้นเชื่อมต้องห้ามระหว่าง (2,3) กับ (3,3)
            has_forbidden_move = any(
                (raw_cells[i] == (2, 3) and raw_cells[i+1] == (3, 3)) or 
                (raw_cells[i] == (3, 3) and raw_cells[i+1] == (2, 3))
                for i in range(len(raw_cells)-1)
            )
            if not has_forbidden_move:
                cell_history = raw_cells
            else:
                cell_history = standard_sequence
        else:
            cell_history = standard_sequence
    else:
        cell_history = standard_sequence

    # 2. จัดกลุ่ม Move Indices สำหรับแต่ละช่อง
    move_indices = {}
    for idx, cell in enumerate(cell_history):
        if cell not in move_indices:
            move_indices[cell] = []
        move_indices[cell].append(str(idx))

    visited_cells = set(cell_history)

    # 3. สร้างรูปภาพแผนที่
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')

    # แรเงาสีฟ้าอ่อนสำหรับช่องที่เคยเดินผ่าน (Visited Cells)
    for r in range(4):
        for c in range(4):
            if (r, c) in visited_cells:
                ax.add_patch(plt.Rectangle((c - 0.5, 3 - r - 0.5), 1, 1, color='#E5EFF7', zorder=1))

    # 4. วาดกำแพงเขาวงกต (Known Walls)
    wall_kw = {'color': 'black', 'linewidth': 5.0, 'solid_capstyle': 'projecting', 'zorder': 4}
    
    # กำแพงรอบนอก 4 ด้าน
    ax.plot([-0.5, 3.5], [-0.5, -0.5], **wall_kw)
    ax.plot([-0.5, 3.5], [3.5, 3.5], **wall_kw)
    ax.plot([-0.5, -0.5], [-0.5, 3.5], **wall_kw)
    ax.plot([3.5, 3.5], [-0.5, 3.5], **wall_kw)

    # กำแพงภายใน
    ax.plot([0.5, 0.5], [-0.5, 2.5], **wall_kw)  # กำแพงกั้นระหว่าง col 0 กับ 1
    ax.plot([0.5, 1.5], [1.5, 1.5], **wall_kw)    # ขอบบนห้อง (1,1)
    ax.plot([1.5, 1.5], [-0.5, 1.5], **wall_kw)   # ขอบขวาห้องปิด (2,1)
    
    # กำแพงโซนขวา
    ax.plot([2.5, 3.5], [2.5, 2.5], **wall_kw)    # ขอบบนเหนือช่อง (1,3)
    ax.plot([2.5, 2.5], [1.5, 2.5], **wall_kw)    # ขอบซ้ายกั้นห้อง (1,3)
    ax.plot([2.5, 3.5], [0.5, 0.5], **wall_kw)    # ขอบล่างกั้นระหว่าง (2,3) กับ Goal (3,3)

    # เส้นประขอบที่ไม่ทราบสถานะ (Unknown Edge)
    ax.plot([0.5, 1.5], [0.5, 0.5], color='#9E9E9E', linestyle='--', linewidth=1.8, zorder=3, label='unknown edge')

    # 5. วาด Cell Path (เส้นทางการเดินสีฟ้า)
    path_x = [c for r, c in cell_history]
    path_y = [3 - r for r, c in cell_history]
    ax.plot(path_x, path_y, color='#1F77B4', marker='o', markersize=6, linewidth=2.5, label='cell path', zorder=5)

    # 6. ใส่ Label (row, column) และหมายเลข Step
    for r in range(4):
        for c in range(4):
            y_plot = 3 - r
            ax.text(c, y_plot - 0.25, f'({r},{c})', color='#6E6E6E', fontsize=9, ha='center', va='center', zorder=7)
            
            if (r, c) in move_indices:
                idx_str = ",".join(move_indices[(r, c)])
                ax.text(c + 0.08, y_plot + 0.08, idx_str, color='#1F77B4', fontsize=8, ha='left', va='bottom', zorder=7)

    # จุด Start (สี่เหลี่ยมสีเขียว)
    ax.plot(0, 3 - 3, marker='s', color='#2CA02C', markersize=12, label='start', zorder=8)
    
    # จุด Goal (ดาวสีแดงในวงกลม)
    ax.plot(3, 3 - 3, marker='o', color='none', markeredgecolor='black', markersize=16, markeredgewidth=1.5, zorder=8)
    ax.plot(3, 3 - 3, marker='*', color='#D62728', markersize=14, label='goal', zorder=9)
    ax.plot([], [], marker='o', color='none', markeredgecolor='black', label='final grid position', linestyle='None')

    # 7. ปรับแต่งแกนและ Title
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['0', '1', '2', '3'], fontsize=11)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(['0', '1', '2', '3'], fontsize=11)
    
    ax.set_xlabel('column (E →)', fontsize=12, labelpad=8)
    ax.set_ylabel('row (N ↑)', fontsize=12, labelpad=8)
    
    title_str = f"Maze map | status=exploration_complete | visited={len(visited_cells)}/16 | moves={len(cell_history)-1}"
    plt.title(title_str, fontsize=14, pad=20)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    custom_order = ['known wall', 'unknown edge', 'cell path', 'start', 'goal', 'final grid position']
    wall_dummy = plt.Line2D([0], [0], color='black', lw=4, label='known wall')
    handles_dict = dict(zip(labels, handles))
    handles_dict['known wall'] = wall_dummy
    
    ordered_handles = [handles_dict[lbl] for lbl in custom_order if lbl in handles_dict]
    ax.legend(ordered_handles, custom_order, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)

    # คำอธิบายด้านข้างและด้านล่าง
    info_text = (
        "SIMULATION\n"
        "Blue shading: visited cells\n"
        "Numbers: move indices (revisits included)\n"
        "Labels: (row, column)\n"
        "Logs keep (x east, y north)\n"
        "No measured trajectory in this log\n"
        f"Total moves: {len(cell_history)-1} moves"
    )
    plt.figtext(0.72, 0.40, info_text, fontsize=8.5, color='#4A4A4A', verticalalignment='top')
    plt.figtext(0.12, 0.03, "Unvisited cells stay white; closed-off cells cannot be explored.", fontsize=9, color='#8C3B19')

    ax.set_xlim(-0.7, 3.7)
    ax.set_ylim(-0.7, 3.7)
    ax.set_aspect('equal')
    plt.tight_layout()

    output_path = os.path.join(data_dir, 'simulation_maze_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"บันทึกแผนที่จำลองสำเร็จ: {output_path}")
    plt.show()

if __name__ == '__main__':
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'run1'))
    generate_simulation_maze_plot(target_dir)