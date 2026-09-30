import sys
import time
import heapq
from collections import deque
import pygame

# ==============================================================================
# 1. แผนที่และข้อมูลสนาม 4x4
# ==============================================================================
GRID_SIZE = 4
START = (1, 4)
GOAL = (4, 1)

COSTS = {
    (1, 4): 2, (2, 4): 3, (3, 4): 1,
    (1, 3): 2,           (3, 3): 4, (4, 3): 2,
    (1, 2): 3, (2, 2): 1,           (4, 2): 4,
    (1, 1): 2, (2, 1): 3, (3, 1): 2, (4, 1): 1,
}
OBSTACLES = {(4, 4), (2, 3), (3, 2)}

def get_neighbors(pos):
    """ลำดับทิศทาง: ขวา, ลง, ซ้าย, ขึ้น"""
    x, y = pos
    for dx, dy in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE and (nx, ny) not in OBSTACLES:
            yield (nx, ny)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path

# ==============================================================================
# 2. ฟังก์ชันคำนวณทั้ง 4 อัลกอริทึม
# ==============================================================================
def solve_bfs(start, goal):
    t0 = time.perf_counter()
    frontier = deque([start])
    came_from = {start: None}

    while frontier:
        curr = frontier.popleft()
        if curr == goal:
            break
        for nxt in get_neighbors(curr):
            if nxt not in came_from:
                came_from[nxt] = curr
                frontier.append(nxt)

    dt = time.perf_counter() - t0
    path = reconstruct_path(came_from, start, goal) if goal in came_from else []
    cost = sum(COSTS[p] for p in path[1:]) if path else 0
    return path, len(path) - 1, cost, dt

def solve_dfs(start, goal):
    t0 = time.perf_counter()
    # กำหนดลำดับทิศทางของ DFS ให้ลงก่อน
    stack = [(start, [start])]
    visited = set()
    found_path = []

    while stack:
        curr, path = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)

        if curr == goal:
            found_path = path
            break

        # ตรวจสอบทิศทางแบบ LIFO
        for dx, dy in [(0, 1), (-1, 0), (1, 0), (0, -1)]:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE and (nx, ny) not in OBSTACLES:
                if (nx, ny) not in visited:
                    stack.append(((nx, ny), path + [(nx, ny)]))

    dt = time.perf_counter() - t0
    cost = sum(COSTS[p] for p in found_path[1:]) if found_path else 0
    return found_path, len(found_path) - 1, cost, dt

def solve_dijkstra(start, goal):
    t0 = time.perf_counter()
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    closed_set = set()

    while frontier:
        cost, curr = heapq.heappop(frontier)
        if curr in closed_set:
            continue
        closed_set.add(curr)

        if curr == goal:
            break

        for nxt in get_neighbors(curr):
            new_cost = cost_so_far[curr] + COSTS[nxt]
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, nxt))
                came_from[nxt] = curr

    dt = time.perf_counter() - t0
    path = reconstruct_path(came_from, start, goal) if goal in came_from else []
    cost = sum(COSTS[p] for p in path[1:]) if path else 0
    return path, len(path) - 1, cost, dt

def solve_astar(start, goal):
    t0 = time.perf_counter()
    frontier = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    closed_set = set()

    while frontier:
        _, curr = heapq.heappop(frontier)
        if curr in closed_set:
            continue
        closed_set.add(curr)

        if curr == goal:
            break

        for nxt in get_neighbors(curr):
            new_g = g_score[curr] + COSTS[nxt]
            if nxt not in g_score or new_g < g_score[nxt]:
                g_score[nxt] = new_g
                f_score = new_g + heuristic(nxt, goal)
                heapq.heappush(frontier, (f_score, nxt))
                came_from[nxt] = curr

    dt = time.perf_counter() - t0
    path = reconstruct_path(came_from, start, goal) if goal in came_from else []
    cost = sum(COSTS[p] for p in path[1:]) if path else 0
    return path, len(path) - 1, cost, dt

# ==============================================================================
# 3. GUI Engine & Clickable Buttons (Pygame)
# ==============================================================================
WIDTH, HEIGHT = 620, 800
GRID_AREA_H = 600
CELL_PX = GRID_AREA_H // GRID_SIZE

COLOR_BG = (248, 250, 252)
COLOR_PANEL = (30, 41, 59)
COLOR_GRID_LINE = (226, 232, 240)
COLOR_OBSTACLE = (51, 65, 85)
COLOR_START = (220, 252, 231)
COLOR_GOAL = (254, 226, 226)
COLOR_PATH = (37, 99, 235)
COLOR_ROBOT = (234, 88, 12)

# ตำแหน่งปุ่มกด 5 ปุ่ม (BFS, DFS, Dijkstra, A*, Reset)
BUTTONS = {
    "BFS": pygame.Rect(15, 615, 105, 40),
    "DFS": pygame.Rect(132, 615, 105, 40),
    "Dijkstra": pygame.Rect(249, 615, 115, 40),
    "A*": pygame.Rect(376, 615, 105, 40),
    "Reset": pygame.Rect(493, 615, 110, 40),
}

def grid_to_screen(grid_pos):
    x, y = grid_pos
    col = x - 1
    row = GRID_SIZE - y
    return col * CELL_PX + 10, row * CELL_PX

def draw_interface(screen, fonts, current_algo, path, current_pos, stats, mouse_pos):
    screen.fill(COLOR_BG)

    # 1. วาดตาราง 4x4
    for x in range(1, GRID_SIZE + 1):
        for y in range(1, GRID_SIZE + 1):
            pos = (x, y)
            px, py = grid_to_screen(pos)
            rect = pygame.Rect(px, py, CELL_PX, CELL_PX)

            if pos in OBSTACLES:
                pygame.draw.rect(screen, COLOR_OBSTACLE, rect)
            elif pos == START:
                pygame.draw.rect(screen, COLOR_START, rect)
            elif pos == GOAL:
                pygame.draw.rect(screen, COLOR_GOAL, rect)
            else:
                pygame.draw.rect(screen, (255, 255, 255), rect)

            pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 2)

            if pos in OBSTACLES:
                t = fonts["sub"].render("OBSTACLE", True, (255, 255, 255))
                screen.blit(t, t.get_rect(center=rect.center))
            else:
                t_coord = fonts["sub"].render(f"({x},{y})", True, (148, 163, 184))
                screen.blit(t_coord, (px + 10, py + 8))

                cost_val = COSTS.get(pos, "")
                t_cost = fonts["bold"].render(f"Cost: {cost_val}", True, (30, 41, 59))
                screen.blit(t_cost, t_cost.get_rect(center=(rect.centerx, rect.centery + 10)))

    # 2. วาดเส้นทางที่คำนวณได้
    if path and len(path) > 1:
        points = [pygame.Rect(grid_to_screen(p)[0], grid_to_screen(p)[1], CELL_PX, CELL_PX).center for p in path]
        pygame.draw.lines(screen, COLOR_PATH, False, points, 8)
        for pt in points:
            pygame.draw.circle(screen, COLOR_PATH, pt, 6)

    # 3. วาดหุ่นยนต์
    if current_pos:
        r_px, r_py = grid_to_screen(current_pos)
        center = (r_px + CELL_PX // 2, r_py + CELL_PX // 2)
        pygame.draw.circle(screen, COLOR_ROBOT, center, 28)
        pygame.draw.circle(screen, (255, 255, 255), center, 30, 4)
        t_bot = fonts["sub"].render("BOT", True, (255, 255, 255))
        screen.blit(t_bot, t_bot.get_rect(center=center))

    # 4. แถบ Dashboard ด้านล่าง
    panel_rect = pygame.Rect(0, GRID_AREA_H, WIDTH, HEIGHT - GRID_AREA_H)
    pygame.draw.rect(screen, COLOR_PANEL, panel_rect)

    # 5. วาดปุ่มกดแบบ Interactive
    for name, rect in BUTTONS.items():
        is_hover = rect.collidepoint(mouse_pos)
        is_active = (current_algo == name)

        if is_active:
            btn_color = (16, 185, 129)  # เขียวเมื่อกำลังทำงาน
        elif is_hover:
            btn_color = (59, 130, 246)  # ฟ้าเมื่อชี้เมาส์
        else:
            btn_color = (71, 85, 105)   # เทาปกติ

        pygame.draw.rect(screen, btn_color, rect, border_radius=6)
        t_btn = fonts["bold"].render(name, True, (255, 255, 255))
        screen.blit(t_btn, t_btn.get_rect(center=rect.center))

    # 6. แสดงผลค่าสถิติ
    status_text = f"Selected: {current_algo or 'None (Click button to start)'}"
    screen.blit(fonts["bold"].render(status_text, True, (248, 250, 252)), (20, 675))

    stats_line = f"Steps: {stats.get('steps', 0)}  |  Total Cost: {stats.get('cost', 0)}  |  Calc Time: {stats.get('calc_time', 0.0):.6f}s"
    screen.blit(fonts["main"].render(stats_line, True, (226, 232, 240)), (20, 710))

    guide = "Use Mouse to click buttons or press keys: [1] BFS  [2] DFS  [3] Dijkstra  [4] A*  [R] Reset"
    screen.blit(fonts["sub"].render(guide, True, (148, 163, 184)), (20, 745))

    pygame.display.flip()

# ==============================================================================
# 4. Main Process Loop (Non-blocking Timer)
# ==============================================================================
def main():
    # คำนวณและแสดงตารางเปรียบเทียบ 4 ตัวบน Terminal ก่อน
    all_algos = [
        ("BFS", solve_bfs),
        ("DFS", solve_dfs),
        ("Dijkstra's Algorithm", solve_dijkstra),
        ("A*", solve_astar)
    ]
    print("\n" + "="*80)
    print("ตารางเปรียบเทียบ 4 อัลกอริทึม (สำหรับกรอกตารางบน)")
    print("="*80)
    print(f"{'Algorithm':<22}{'จำนวน step':<14}{'Total Cost':<14}{'เวลาในการคำนวณ (s)':<22}Path")
    print("-" * 80)
    for name, fn in all_algos:
        p, s, c, t = fn(START, GOAL)
        print(f"{name:<22}{s:<14}{c:<14}{t:<22.6f}{p}")
    print("="*80 + "\n")

    pygame.init()
    pygame.display.set_caption("RoboMaster Pathfinding 4 Algorithms")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    fonts = {
        "bold": pygame.font.SysFont("Arial", 18, bold=True),
        "main": pygame.font.SysFont("Arial", 16),
        "sub": pygame.font.SysFont("Arial", 13, bold=True),
    }

    current_algo = None
    active_path = []
    current_bot_pos = START
    stats = {"steps": 0, "cost": 0, "calc_time": 0.0}

    # ตัวแปรจัดการ Animation แบบ Smooth ไม่ค้างจอ
    is_animating = False
    anim_step_index = 0
    last_move_time = 0
    STEP_DELAY_MS = 300

    running = True
    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()
        current_time = pygame.time.get_ticks()

        # อัปเดตตำแหน่งหุ่นยนต์ทีละช่องตามเวลา
        if is_animating:
            if current_time - last_move_time >= STEP_DELAY_MS:
                last_move_time = current_time
                anim_step_index += 1
                if anim_step_index < len(active_path):
                    current_bot_pos = active_path[anim_step_index]
                else:
                    is_animating = False  # ถึง Goal แล้ว จอดนิ่ง ไม่เด้งกลับ

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ตรวจจับการคลิกเมาส์ที่ปุ่ม
            selected_algo = None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if BUTTONS["BFS"].collidepoint(event.pos):
                    selected_algo = "BFS"
                elif BUTTONS["DFS"].collidepoint(event.pos):
                    selected_algo = "DFS"
                elif BUTTONS["Dijkstra"].collidepoint(event.pos):
                    selected_algo = "Dijkstra"
                elif BUTTONS["A*"].collidepoint(event.pos):
                    selected_algo = "A*"
                elif BUTTONS["Reset"].collidepoint(event.pos):
                    current_algo = None
                    active_path = []
                    current_bot_pos = START
                    stats = {"steps": 0, "cost": 0, "calc_time": 0.0}
                    is_animating = False

            # ตรวจจับปุ่มคีย์บอร์ด (รองรับทั้งแถวบนและ Numpad)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_KP1):
                    selected_algo = "BFS"
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    selected_algo = "DFS"
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    selected_algo = "Dijkstra"
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    selected_algo = "A*"
                elif event.key in (pygame.K_r, pygame.K_ESCAPE):
                    current_algo = None
                    active_path = []
                    current_bot_pos = START
                    stats = {"steps": 0, "cost": 0, "calc_time": 0.0}
                    is_animating = False

            # ดำเนินการเมื่อเลือก Algorithm
            if selected_algo:
                current_algo = selected_algo
                if selected_algo == "BFS":
                    path, steps, cost, dt = solve_bfs(START, GOAL)
                elif selected_algo == "DFS":
                    path, steps, cost, dt = solve_dfs(START, GOAL)
                elif selected_algo == "Dijkstra":
                    path, steps, cost, dt = solve_dijkstra(START, GOAL)
                elif selected_algo == "A*":
                    path, steps, cost, dt = solve_astar(START, GOAL)

                active_path = path
                stats = {"steps": steps, "cost": cost, "calc_time": dt}
                current_bot_pos = START
                anim_step_index = 0
                last_move_time = pygame.time.get_ticks()
                is_animating = True

        draw_interface(screen, fonts, current_algo, active_path, current_bot_pos, stats, mouse_pos)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()