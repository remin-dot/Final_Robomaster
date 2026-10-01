import csv
import time
import os
import itertools
import math
from datetime import datetime
import cv2
import numpy as np

from sharp_ir import SharpIR
from navigation_safety import braking_speed, corridor_obstacle, information_rate

try:
    from robomaster import robot
except Exception:
    robot = None


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class ChassisController:
    # ---- ความปลอดภัยข้าง (Sharp IR) ----
    SIDE_SAFE_CM = 9.0      # เริ่มเลื่อนหลบ (Sharp 4-30 cm; กลางช่องอ่านได้ ~13 cm) - sharp_ir.side_safe_cm
    SIDE_DANGER_CM = 6.0    # ชะลอเดินหน้า - sharp_ir.side_danger_cm
    STRAFE_V = 0.15
    KP_YAW_HOLD = 0.8
    FRONT_STOP_MM = 150     # เบรกฉุกเฉินด้านหน้า

    # ---- เดินด้วย odometry ----
    DECEL_ZONE_M = 0.18     # ระยะสุดท้ายที่เริ่มชะลอ (ลดระยะไหลหลังสั่งหยุด)
    MIN_V = 0.08            # ความเร็วต่ำสุดตอนชะลอ (m/s)
    COAST_INIT_M = 0.02     # ค่าเริ่มต้นระยะไหลหลังสั่งหยุด (เรียนรู้เองทุกครั้งที่เดิน)
    COAST_MAX_M = 0.06
    TIME_CAP_FACTOR = 1.7   # เพดานเวลา = เวลาปกติ x ค่านี้ + 0.3s (กัน odometry ค้าง)

    # ---- alignment ด้วยกำแพง ----
    ALIGN_MAX_MM = 450      # ใช้กำแพงที่ใกล้กว่านี้ในการจัดแนว
    ALIGN_PHI_DEG = 15.0    # มุมวัดซ้าย/ขวาของ ToF
    ALIGN_MAX_DEG = 20.0    # เบี้ยวเกินนี้ถือว่าวัดผิด ข้าม

    def __init__(self, ep_robot, config):
        self.ep_robot = ep_robot
        self.config = config
        self.ep_chassis = ep_robot.chassis
        self.ep_sensor = ep_robot.sensor
        self.ep_gimbal = ep_robot.gimbal
        self.ep_adaptor = ep_robot.sensor_adaptor

        self.current_tof_dist_mm = 9999
        self._tof_last_ts = 0.0
        from collections import deque
        self.gimbal_hist = deque(maxlen=200)   # (t, pitch, yaw) from the gimbal angle feed
        self.current_yaw = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self._pos_count = 0

        move_cfg = config.get("movement", {})
        # z>0 ทำให้ yaw เพิ่มหรือลด (หาอัตโนมัติใน calibrate_yaw_sign)
        self.z_sign = 1
        # เลี้ยวขวา = yaw บวก (หมุนตามเข็ม)
        self.right_yaw_sign = move_cfg.get("right_yaw_sign", 1)
        # yaw ของทิศเหนือแผนที่ (อัปเดตอัตโนมัติจากกำแพง)
        self.yaw_origin = 0.0
        # ToF ด้านหน้าเมื่อหุ่นอยู่กลางช่องหันหน้าเข้ากำแพง (mm) ต้องวัดเองแล้วใส่ config
        self.front_center_mm = move_cfg.get("front_center_mm", None)
        self.align_enabled = move_cfg.get("align_enabled", False)

        self.coast_est = self.COAST_INIT_M
        # optional MissionPanel (live camera + map) and TargetShooter, set by main_mission
        self.panel = None
        self.shooter = None
        # 2 Sharp IR (left/right) - driver + calibration from RoboFinal, polled in a thread
        self.ir = SharpIR(self.ep_adaptor, config, log=self._log)
        self._left_cm = self.ir.max_cm
        self._right_cm = self.ir.max_cm
        ir_cfg = config.get("sharp_ir", {}) or {}
        # sensor faces are ~0.12 m from the centre: a centred robot in a 0.5 m way reads ~13 cm
        self.SIDE_SAFE_CM = float(ir_cfg.get("side_safe_cm", 9.0))
        self.SIDE_DANGER_CM = float(ir_cfg.get("side_danger_cm", 6.0))
        # one wall threshold for every gimbal direction (the ToF turns with the gimbal)
        self.wall_mm = float(move_cfg.get("wall_threshold_mm", 450))
        # order to try the open ways of a cell (relative to how the robot came in):
        # side cells first, then ahead - settings.yaml movement.explore_order
        order = [str(o).lower() for o in move_cfg.get("explore_order", ["left", "right", "front", "back"])]
        self.explore_order = [o for o in order if o in ("front", "right", "back", "left")] or \
            ["left", "right", "front", "back"]
        for o in ("left", "right", "front", "back"):
            if o not in self.explore_order:
                self.explore_order.append(o)
        self._tof_samples = []          # (t, mm) recent ToF readings
        vis_cfg = config.get("vision", {}) or {}
        # camera wall check: wall height - camera height (m), learnt from the ToF at every front scan
        self.wall_rise_m = float(vis_cfg.get("wall_rise_m", 0.12))
        self._rise_samples = []
        self.CAM_BLOCK_M = float(vis_cfg.get("camera_block_m", 0.40))   # before a move: wall this close ahead = blocked
        self.CAM_STOP_M = float(vis_cfg.get("camera_stop_m", 0.22))     # while moving: stop at this
        # ... and only when the ToF confirms something that close (normal cells: far wall ~850 mm)
        self.CAM_TOF_BLOCK_MM = float(vis_cfg.get("camera_tof_block_mm", 450))
        self.CAM_TOF_STOP_MM = float(vis_cfg.get("camera_tof_stop_mm", 300))
        self.last_move_note = ""
        corner_cfg = config.get("ir_corner", {}) or {}
        self.CORNER_TOF_MM = float(corner_cfg.get("confirm_tof_mm", 400))
        self.CORNER_SIDE_CM = float(corner_cfg.get("confirm_side_cm", 12))
        self.CORNER_HOLD_STOP_S = float(corner_cfg.get("single_hold_stop_s", 0.35))
        self.CORNER_SLOW_FRAC = float(corner_cfg.get("caution_speed_fraction", 0.25))
        self.CORNER_STEER_MPS = float(corner_cfg.get("avoid_steer_speed", 0.10))
        self._corner_warned = {}
        self.HEADING_GUARD_DEG = float(move_cfg.get("heading_guard_deg", 12.0))
        self.CELL_SPEED = float(move_cfg.get("cell_speed", 0.45))       # m/s for one-block moves
        self.FRONT_STOP_MM = float(move_cfg.get("front_stop_mm", 200))
        self.TURN_MAX_DPS = float(move_cfg.get("turn_max_dps", 90))     # chassis turn speed (deg/s)
        # braking at the end of a turn: speed = sqrt(2 * this * degrees left). The old linear
        # ramp over the last 40 deg spent ~1.7 s creeping in on every turn (87 s of turning last run)
        self.TURN_BRAKE_DPS2 = float(move_cfg.get("turn_brake_dps2", 200))
        # the chassis answers a speed command ~0.08 s late: brake that much earlier (without it
        # every turn ended 4-7 deg past the heading and needed a correction)
        self.TURN_LAG_S = float(move_cfg.get("turn_lag_s", 0.08))
        self.TURN_TIGHT_DPS = float(move_cfg.get("turn_tight_dps", 45))  # ... when still near a wall
        self.TURN_CLEAR_SIDE_CM = float(move_cfg.get("turn_clear_side_cm", 12))   # side Sharp: room to turn
        self.TURN_CLEAR_TOF_MM = float(move_cfg.get("turn_clear_tof_mm", 200))    # front ToF: room to turn
        self.SLIDE_STOP_MM = float(move_cfg.get("slide_stop_mm", 220))   # sideways: the chassis side is closer
        self.APPROACH_K = float(move_cfg.get("approach_slow_per_s", 1.2))  # m/s per m left to the stop distance
        self.DECEL_A = float(move_cfg.get("brake_mps2", 0.6))            # braking at the end of a move (m/s^2)
        self.UNKNOWN_SPEED = float(move_cfg.get("unknown_cell_speed", 0.30))
        self.TOF_STALE_S = float(move_cfg.get("tof_stale_s", 0.35))
        self.REACTION_S = float(move_cfg.get("reaction_time_s", 0.10))
        self.CLEARANCE_SCAN_DEG = float(move_cfg.get("clearance_scan_deg", 15.0))
        self.PRECHECK_BLOCK_M = float(move_cfg.get("precheck_block_m", 0.45))
        robot_half = float(move_cfg.get("robot_half_width_m", 0.16))
        safety_margin = float(move_cfg.get("obstacle_margin_m", 0.04))
        self.CORRIDOR_HALF_M = robot_half + safety_margin
        self.GIMBAL_TIMEOUT_S = float(move_cfg.get("gimbal_move_timeout_s", 2.5))
        # gimbal turn speed for scans / looks (was 300: blurred frames, ToF read while still turning)
        self.GIMBAL_DPS = float(move_cfg.get("gimbal_scan_dps", 200))
        self.GIMBAL_BRAKE_DPS2 = float(move_cfg.get("gimbal_brake_dps2", 600))   # speed-mode gimbal: braking
        self.GIMBAL_RESPONSE_S = float(move_cfg.get("gimbal_response_s", 0.05))  # its lag to a speed command
        self.GIMBAL_TOL_DEG = float(move_cfg.get("gimbal_tol_deg", 1.0))
        self.FEED_PITCH_SIGN = 1.0      # set by _check_pitch_sign at start-up
        self.PITCH_BY_SPEED = True      # False: pitch changes go through moveto (feed unusable)
        # scan = one continuous gimbal sweep (ToF sampled on the way), still looks only where a
        # card colour was seen; false = stop at each of the 4 directions (the old way)
        self.SWEEP_SCAN = bool(move_cfg.get("sweep_scan", True))
        self.SWEEP_DPS = float(move_cfg.get("sweep_dps", 150))
        self.SWEEP_WIN_DEG = float(move_cfg.get("sweep_window_deg", 10))
        self.SWEEP_CHECKS = int(move_cfg.get("sweep_still_looks", 4))
        self._sweep_checks_cfg = self.SWEEP_CHECKS
        # A continuous sweep can miss a target because the useful frame is blurred or
        # the colour mask briefly fails. Keep a bounded settled-look fallback, reduced
        # or skipped when the round-time budget becomes critical.
        self.SWEEP_FALLBACK_LOOKS = int(move_cfg.get("sweep_fallback_still_looks", 4))
        self.SWEEP_FALLBACK_MIN_REMAINING_S = float(
            move_cfg.get("sweep_fallback_min_remaining_s", 120.0))
        self.hurry = False
        self.END_RESERVE_S = float(move_cfg.get("end_reserve_s", 90))   # kept for second looks + shooting
        self.COVERAGE_TARGET_S = float(move_cfg.get(
            "coverage_target_s", max(60.0, 600.0 - self.END_RESERVE_S)))
        self.TOF_LATENCY_S = float(move_cfg.get("tof_latency_s", 0.05))
        # the chassis never turns: it keeps one heading and slides every way (mecanum)
        self.FIXED_HEADING = bool(move_cfg.get("fixed_heading", True))
        # move style: "face" = turn at the block centre to face the way, drive forward with
        # Sharp wall centering + heading PID, centre in place on arrival (the way the
        # robomaster-assignment2-4x4_Dhai_8 robot moves); "strafe" = never turn, slide every way
        self.MOVE_STYLE = str(move_cfg.get("move_style", "face")).lower()
        wp = move_cfg.get("wall_pid", {}) or {}
        from wall_pid import WallCentering
        self.wall_pid = WallCentering(wp) if wp.get("enabled", True) else None
        self.MAX_LATERAL_M = float(wp.get("max_lateral_deviation_m", 0.15))
        self.CENTER_TIME_S = float(wp.get("center_in_cell_s", 0.4))
        self.FINE_TURN_TOL = float(wp.get("fine_turn_tol_deg", 1.0))
        if self.MOVE_STYLE == "face":
            self.FIXED_HEADING = False
        self.SLIDE_SPEED = float(move_cfg.get("slide_speed", 0.28))
        # emergency stop "about to hit": time to contact under this (ToF ahead / side Sharps)
        self.TTC_STOP_S = float(move_cfg.get("emergency_ttc_s", 0.35))
        self.SIDE_HIT_CM = float(move_cfg.get("emergency_side_cm", 5.0))    # side Sharp this close = stop
        self.SIDE_MAX_CLOSING = float(move_cfg.get("emergency_side_max_closing_cm_s", 40.0))  # faster = sensor jump
        self.BUMP_G = float(move_cfg.get("emergency_bump_g", 0.55))         # IMU spike = it hit something
        # drive commands this long with no wheel turning, no odometry, no yaw change = the chassis
        # is not answering (0 = off)
        self.WHEEL_CHECK_S = float(move_cfg.get("wheel_check_s", 1.2))
        # cards seen in frames taken while the gimbal turns (blurred): exposure time for the smear
        # length, and how many of those sightings a block may check with a still look
        self.EXPOSURE_S = float(vis_cfg.get("exposure_s", 0.02))
        self.MOTION_MIN_DPS = float(vis_cfg.get("motion_min_dps", 25.0))
        self.MOTION_CHECKS = int(vis_cfg.get("motion_checks_per_block", 2))
        self.MOTION_MIN_FRAMES = int(vis_cfg.get("motion_min_frames", 3))       # colour-only sightings
        self._motion_checks_cfg = self.MOTION_CHECKS
        self.MOTION_MAX_ELEV_DEG = float(vis_cfg.get("motion_max_elev_deg", 4.0))  # colour-only: not above this
        self.SCAN_PITCH_DEG = float(vis_cfg.get("scan_pitch_deg", -5.0))
        self.CANDIDATE_CONFIRM_FRAMES = int(vis_cfg.get("candidate_confirm_frames", 5))
        self.CANDIDATE_CONFIRM_MIN_HITS = int(vis_cfg.get("candidate_confirm_min_hits", 2))
        self.CANDIDATE_RETRY_YAW_DEG = float(vis_cfg.get("candidate_retry_yaw_deg", 8.0))
        self.CANDIDATE_RETRY_FRAMES = int(vis_cfg.get("candidate_retry_frames", 3))
        self.CAMERA_STALE_S = float(vis_cfg.get("camera_stale_s", 1.0))
        self.CAMERA_RECOVER_WAIT_S = float(vis_cfg.get("camera_recover_wait_s", 3.0))
        self.SLIDE_HOLD_KP = float(move_cfg.get("slide_hold_kp", 2.5))   # sideways slide: pull back front-back drift
        self.SLIDE_HOLD_KI = float(move_cfg.get("slide_hold_ki", 6.0))
        # slide sideways / backwards instead of turning the chassis (mecanum wheels)
        self.strafe = bool(move_cfg.get("strafe_moves", True)) and \
            str(move_cfg.get("move_style", "face")).lower() != "face"
        self.recenter_enabled = bool(move_cfg.get("recenter", True))
        self.CENTER_BEFORE_TURN = bool(move_cfg.get("center_before_turn", True))
        self.CENTER_TOF_MM = float(move_cfg.get("center_tof_mm", 170))  # ToF to a wall 0.3 m away, robot centred
        self.RECENTER_DEADBAND_MM = float(move_cfg.get("recenter_deadband_mm", 40))
        self.RECENTER_MAX_M = float(move_cfg.get("recenter_max_m", 0.10))
        # a stop past this share of the move with the ToF under ARRIVE_TOF_MM = arrived (dead end)
        self.ARRIVE_MIN_PROGRESS = float(move_cfg.get("arrive_min_progress", 0.80))
        self.ARRIVE_TOF_MM = float(move_cfg.get("arrive_tof_mm", 280))
        self.verify_pitch_deg = float(vis_cfg.get("verify_pitch_deg", -10.0))   # close look: camera down
        self.CLOSE_PITCH_MIN_DEG = max(-20.0, float(vis_cfg.get("close_pitch_min_deg", -20.0)))
        self.verify_enabled = bool(vis_cfg.get("verify_sweep", True))
        self._swept_cells = set()       # blocks that had their full close look
        self._guesses = []              # maybe-cards seen (cut by the picture edge / odd outline)
        self._candidate_checked = set() # (cell, colour, direction bin), one focused check each
        self._not_cards = []            # (colour, x, y) checked by a look-back: not a card
        self._looked_back = set()       # blocks that already had their look-back
        self._odom0 = None              # (x, y, yaw_origin) when the map was fixed
        shoot_cfg = config.get("shooting", {}) or {}
        # shoot only a card in this block or the block right next to it (never across a block)
        self.reach_cells = int(shoot_cfg.get("reach_cells", 1))
        self.fire_range_margin = min(1.0, max(0.5, float(shoot_cfg.get("fire_range_margin", 0.90))))
        self.fire_max_cells = max(0, int(shoot_cfg.get("fire_max_cells", 1)))
        self.route_target_spots = max(1, int(shoot_cfg.get("route_target_spots", 2)))
        self.card_aim_height_m = float(shoot_cfg.get("card_aim_height_m", 0.14))
        import route_planner
        route_planner.REACH_CELLS = self.reach_cells
        route_planner.REACH_PATTERN = str(shoot_cfg.get("reach_pattern", "3x3")).lower()
        route_planner.STRAIGHT_BAND_M = float(shoot_cfg.get("straight_band_m", 0.2))
        route_planner.SHOT_MAX_VIEW_DEG = float(shoot_cfg.get("max_shot_view_deg", 55.0))
        self._tried_inside = set()      # (card id, block) - shot at from inside its own block already
        self._retried = set()           # card ids given a second go at the end of round 1
        self._tried_from = set()        # (card id, block) - aimed at from that block already
        self._aim_attempts = {}          # (card id, block) -> real shooter engagements
        self.AIM_ATTEMPTS_PER_CELL = max(1, int(shoot_cfg.get("aim_attempts_per_cell", 2)))
        self.MIN_SHOOT_M = float(shoot_cfg.get("min_shoot_m", 0.3))      # closer: back off first
        # best angle: a shot farther than good_shot_m or more slanted than good_shot_view_deg
        # waits when a block still to be explored gives a near, face-on shot (last run: 1.19 m
        # across a diagonal at 37 deg - 3 misses; face-on 0.6-0.9 m hit first time)
        self.GOOD_SHOT_M = float(shoot_cfg.get("good_shot_m", 0.95))
        self.GOOD_SHOT_VIEW = float(shoot_cfg.get("good_shot_view_deg", 30))
        self.BACK_OFF_M = float(shoot_cfg.get("back_off_max_m", 0.08))   # room inside a block
        self.SHOOT_SEARCH_OFFSETS = tuple(
            float(x) for x in shoot_cfg.get("search_offsets_deg", [0, -12, 12]))
        self.CLOSE_SHOOT_SEARCH_OFFSETS = tuple(
            float(x) for x in shoot_cfg.get("close_search_offsets_deg", [0, -8, 8]))
        # end of round 1: drive back to cards that were found but not hit yet
        self.mop_up = bool(shoot_cfg.get("mop_up_round1", True))
        self.MOP_UP_RESERVE_S = float(shoot_cfg.get("mop_up_reserve_s", 90))
        # shooting queue: the card found first is shot first - while exploring the robot takes a
        # detour of up to this many moves to a spot that can shoot the oldest card still up
        self.queue_detour = int(shoot_cfg.get("queue_detour_moves", 2))
        # end of round 1, time left: shoot again cards still standing after max shots (once)
        self.retry_missed = bool(shoot_cfg.get("retry_missed", True))
        # end of exploring: go back and look face-on at cards seen only nearly edge-on
        self.second_look = bool(shoot_cfg.get("second_look_edge_on", True))
        self.short_first = bool(move_cfg.get("explore_short_first", True))
        self.SHORT_BRANCH_MM = float(move_cfg.get("short_branch_mm", 1000))
        # a wall face counts as seen from a camera look this close / this much face-on
        self.SEE_RANGE_M = float(vis_cfg.get("see_range_m", 1.0))
        self.SEE_MAX_VIEW_DEG = float(vis_cfg.get("see_max_view_deg", 60.0))
        self.SEE_HALF_FOV_DEG = float(vis_cfg.get("hfov_deg", 96.0)) / 2.0 - 4.0
        self.SCAN_COST_S = float(move_cfg.get("scan_cost_s", 4.0))       # a stop to scan, in the plan
        # all = visit every reachable block (complete map); walls = skip blocks whose walls
        # the camera already saw well from next door (faster, map may have holes)
        self.explore_mode = str(move_cfg.get("explore_mode", "all")).lower()
        # visit-all: pick the next block from a planned tour over all blocks still to see
        self.TOUR_PLAN = bool(move_cfg.get("explore_tour", True))
        # look back the way it came in too (4 looks per block instead of 3)
        self.scan_came_from = bool(move_cfg.get("scan_came_from", False))
        # corridor block (side Sharps saw a wall on both sides while driving in, the way ahead is
        # known open from the last long ToF reading): camera looks left + right only
        self.fast_corridor = bool(move_cfg.get("fast_corridor", True))
        self.CORRIDOR_WALL_CM = float(move_cfg.get("corridor_wall_cm", 22.0))
        self._side_samples, self._last_body_dir = [], 0
        # which card first: found = in the order they were found; fastest = route planner's order
        self.shoot_order = str(shoot_cfg.get("order", "found")).lower()
        self.RESERVE_MARGIN_S = float(shoot_cfg.get("reserve_margin_s", 20.0))
        self._looks = []                # (cell, abs_deg) every camera look - for "seen" wall faces
        self._motion_checked = []       # (cell, colour, abs deg) sightings from turns already looked at
        self._motion_empty = []         # (colour, x, y) wall points a sighting check found empty
        self._reach_noted = set()

        self.data_dir = config["data_collection"]["data_dir"]
        self.buffer_time = config["data_collection"]["buffer_time"]

        self.freq_pos = config["data_collection"]["frequencies"]["position"]
        self.freq_att = config["data_collection"]["frequencies"]["attitude"]
        self.freq_imu = config["data_collection"]["frequencies"]["imu"]
        self.freq_esc = config["data_collection"]["frequencies"]["esc"]
        self.freq_dist = config["data_collection"]["frequencies"]["distance"]

        self.default_speed = move_cfg["xy_speed"]
        self.default_distance = move_cfg["distance"]
        self.default_z_speed = move_cfg["z_speed"]
        self.default_angle = move_cfg["angle"]

        os.makedirs(self.data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")

        files = config["data_collection"]["files"]
        self.pos_file = os.path.join(self.data_dir, f"log_{date_str}_{files['position']}.csv")
        self.att_file = os.path.join(self.data_dir, f"log_{date_str}_{files['attitude']}.csv")
        self.imu_file = os.path.join(self.data_dir, f"log_{date_str}_{files['imu']}.csv")
        self.esc_file = os.path.join(self.data_dir, f"log_{date_str}_{files['esc']}.csv")
        self.dist_file = os.path.join(self.data_dir, f"log_{date_str}_{files['distance']}.csv")
        self.ir.csv_path = os.path.join(self.data_dir, f"log_{date_str}_{files.get('infrared', 'ir_data')}.csv")

    # ------------------------------------------------------------------
    # Sharp IR
    # ------------------------------------------------------------------
    def read_side_ir(self):
        """Latest (left_cm, right_cm) from the Sharp polling thread (4-30 cm)."""
        self._left_cm, self._right_cm = self.ir.latest()
        return self._left_cm, self._right_cm

    def _control_side_ir(self):
        """Side distances for steering; ``None`` means missing/untrustworthy sensor."""
        left, right = self.read_side_ir()
        return (left if self.ir.usable("left") else None,
                right if self.ir.usable("right") else None)

    # ------------------------------------------------------------------
    # CSV / callbacks
    # ------------------------------------------------------------------
    def save_to_csv(self, filename, data):
        current_time = time.time()
        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [current_time]
            for item in data:
                if isinstance(item, (list, tuple)):
                    row.extend(item)
                else:
                    row.append(item)
            writer.writerow(row)

    def handle_position(self, data):
        self.save_to_csv(self.pos_file, data)
        self.pos_x, self.pos_y = data[0], data[1]
        if len(data) > 2:
            self.pos_z = data[2]
        self._pos_count += 1
        if self.panel is not None and self._odom0 is not None:
            # odometry frame (x forward, y right at power-on) -> map (east, north) from the start cell
            x0, y0, yo = self._odom0
            dx, dy = self.pos_x - x0, self.pos_y - y0
            n_ang = math.radians(yo)
            e_ang = math.radians(yo + 90.0 * self.right_yaw_sign)
            north = dx * math.cos(n_ang) + dy * math.sin(n_ang)
            east = dx * math.cos(e_ang) + dy * math.sin(e_ang)
            self.panel.map.add_odom(east, north)

    def handle_attitude(self, data):
        self.save_to_csv(self.att_file, data)
        self.current_yaw = data[0]

    def handle_imu(self, data):
        self.save_to_csv(self.imu_file, data)
        try:     # acc in g: a hit is a short sharp spike in the floor plane
            self._imu_acc = (time.time(), float(data[0]), float(data[1]))
        except Exception:
            pass

    def handle_esc(self, data):
        self.save_to_csv(self.esc_file, data)
        try:     # (speed[4] rpm, angle[4], timestamp[4], state[4])
            sp = data[0] if isinstance(data[0], (list, tuple)) else data[:4]
            self._esc_rpm = max(abs(float(v)) for v in sp)
            self._esc_ts = time.time()
            if self._esc_rpm > 20:
                self._esc_last_turn = self._esc_ts
        except Exception:
            pass

    # ------------------------------------------------------------------
    # wheel watchdog: the last run sent drive commands for 40 s and the wheels never turned
    # (ESC 0 rpm, odometry and yaw frozen) - it retried turns 12 times instead of saying so
    # ------------------------------------------------------------------
    def _drive_start(self):
        """Call when a drive / turn starts: remembers where the chassis was."""
        self._drv = (time.time(), getattr(self, "pos_x", 0.0),
                     getattr(self, "pos_y", 0.0), getattr(self, "current_yaw", 0.0))

    def _wheels_dead(self):
        """True when drive commands have gone out for WHEEL_CHECK_S and nothing answered:
        no wheel turned (ESC), no odometry, no yaw change. False while it cannot tell yet."""
        d = getattr(self, "_drv", None)
        if d is None or not getattr(self, "WHEEL_CHECK_S", 0):
            return False
        t0, x0, y0, yaw0 = d
        if time.time() - t0 < self.WHEEL_CHECK_S:
            return False
        if getattr(self, "_esc_last_turn", 0.0) > t0:
            return False                                   # a wheel turned (even if blocked by a wall)
        if math.hypot(self.pos_x - x0, self.pos_y - y0) > 0.01 or abs(wrap180(self.current_yaw - yaw0)) > 1.5:
            return False
        # no ESC feed at all -> cannot tell from the wheels: only odometry / yaw decide
        return True

    def _wheels_not_answering(self, what):
        """The chassis ignores drive commands: try to wake it once (mode + stop), then pause
        the round with a clear message. Returns True when it moves again."""
        self._log(f"WHEELS NOT ANSWERING while {what}: drive commands for {self.WHEEL_CHECK_S:.1f} s, "
                  f"wheels 0 rpm, odometry + yaw frozen - trying to wake the chassis")
        try:
            self.ep_chassis.drive_speed(x=0, y=0, z=0)
            if robot is not None:
                self.ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
                time.sleep(0.2)
                self.ep_robot.set_robot_mode(mode=robot.FREE)
            time.sleep(0.2)
        except Exception as e:
            self._log(f"chassis wake failed: {e}")
        self._drive_start()
        t0 = time.time()
        while time.time() - t0 < self.WHEEL_CHECK_S + 0.3:
            self.ep_chassis.drive_speed(x=0, y=0, z=self.z_sign * 30.0)      # a small turn test
            time.sleep(0.05)
            if not self._wheels_dead() and (getattr(self, "_esc_last_turn", 0.0) > t0 or
                                            abs(wrap180(self.current_yaw - self._drv[3])) > 1.5):
                break
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.2)
        if getattr(self, "_esc_last_turn", 0.0) > t0 or abs(wrap180(self.current_yaw - self._drv[3])) > 1.5:
            self._log("chassis answers again - carrying on")
            self._drv = None
            return True
        self._log("chassis still not answering: round PAUSED - check the chassis battery / power switch, "
                  "that the robot is not held in the DJI app or by its protection lock, lift and set it "
                  "down, then press Resume (or restart the robot and reconnect)")
        if getattr(self, "panel", None) is not None:
            self.panel.paused.set()
            self.panel.checkpoint()          # waits here until Resume / STOP
        self._drv = None
        return False

    def handle_distance(self, data):
        self.save_to_csv(self.dist_file, data)
        self.current_tof_dist_mm = data[0]
        self._tof_last_ts = time.time()
        self._tof_samples.append((self._tof_last_ts, data[0]))
        del self._tof_samples[:-20]

    def _tof_recent(self, max_age=0.25):
        """Median of recent valid ToF samples; None means navigation data is stale."""
        now = time.time()
        vals = [mm for t, mm in self._tof_samples if now - t <= max_age and 0 < mm < 8000]
        return sorted(vals)[len(vals) // 2] if vals else None

    def _tof_fresh(self, n=3, timeout=0.6):
        """Median of n ToF readings taken after now (the gimbal has settled)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            fresh = [mm for t, mm in list(self._tof_samples) if t > t0]
            if len(fresh) >= n:
                return sorted(fresh)[len(fresh) // 2]
            time.sleep(0.02)
        fresh = [mm for t, mm in list(self._tof_samples) if t > t0]
        return sorted(fresh)[len(fresh) // 2] if fresh else self.current_tof_dist_mm

    def _on_gimbal_angle(self, angle_info):
        """(pitch, yaw, pitch_ground, yaw_ground) from the gimbal."""
        self._gimbal_yaw_now = float(angle_info[1])
        pitch = self.FEED_PITCH_SIGN * float(angle_info[0])        # moveto's sign (down = negative)
        self.gimbal_hist.append((time.time(), pitch, float(angle_info[1])))
        if self.panel is not None:
            self.panel.detector.gimbal_pitch_deg = pitch

    def _camera_wall_ahead(self, fresh=False):
        """Distance (m) to a white wall in the robot's path, seen by the camera (inf = none/unknown)."""
        p = self.panel
        if p is None:
            return float("inf")
        if fresh:   # a frame taken after the robot / gimbal stopped (Wi-Fi video is ~0.2 s late)
            p.worker.wait_fresh(time.time() + p.camera_latency_s, timeout=0.8)
        elif time.time() - p.worker.det_ts > 0.6:
            return float("inf")   # no recent frame: do not guess
        d, _ = p.detector.wall_ahead_m(self.wall_rise_m)
        return d

    def _learn_wall_rise(self, tof_mm):
        """Gimbal straight ahead at a wall: ToF distance + where the camera sees the wall top
        -> wall height above the camera (keeps the camera distance honest)."""
        p = self.panel
        if p is None or not (150 < tof_mm < 1000):
            return
        p.worker.wait_fresh(time.time() + p.camera_latency_s, timeout=0.8)
        _, rise = p.detector.wall_ahead_m(self.wall_rise_m)
        size = p.detector.last_frame_size
        if rise is None or rise < 6 or size is None:
            return
        f = p.detector.focal_px(size[0])
        k = (tof_mm / 1000.0) * rise / f
        if 0.03 < k < 0.45:
            self._rise_samples = (self._rise_samples + [k])[-15:]
            if len(self._rise_samples) >= 3:
                new = float(np.median(self._rise_samples))
                if abs(new - self.wall_rise_m) > 0.015:
                    self._log(f"camera wall height learnt: {new:.3f} m above the camera "
                              f"({len(self._rise_samples)} ToF samples, last {k:.3f})")
                self.wall_rise_m = new

    def _corner_says_wall(self, side, tof_mm=None):
        """The corner IR module on that side says 'wall' AND another sensor agrees:
        the ToF sees something close ahead, or that side's Sharp is close. A module
        that says wall with nothing near (screw too sensitive - it sees the shiny
        floor - or the wrong polarity) is ignored, with a warning in the log."""
        if not self.ir.near(side):
            return False
        tof = self.current_tof_dist_mm if tof_mm is None else tof_mm
        tof_close = not (60 < tof < 8000) or tof < self.CORNER_TOF_MM
        l_cm, r_cm = self.ir.latest()
        side_close = (l_cm if side == "left" else r_cm) < self.CORNER_SIDE_CM
        if tof_close or side_close:
            return True
        now = time.time()
        if now - self._corner_warned.get(side, 0) > 10:
            self._corner_warned[side] = now
            raw = self.ir.corner_raw.get(side)
            self._log(f"front-{side} IR module says WALL but ToF {tof:.0f} mm / Sharp {min(l_cm, r_cm):.0f} cm "
                      f"see nothing near (raw {raw}) - ignored: adjust its screw or side-specific active_low")
        return False

    def _front_corner_hits(self):
        """Raw front-corner switches used as bumpers while the chassis is moving.

        Do not require ToF/Sharp agreement here: they look in different directions and
        that delay previously let a correctly triggered corner continue into a wall.
        """
        if not self.ir.corners_enabled:
            return ()
        return tuple(side for side in ("left", "right") if self.ir.corner_near(side) is True)

    def _make_corner_room(self, hits):
        """Move a few centimetres away from a corner that stopped the chassis."""
        hits = tuple(hits)
        left_cm, right_cm = self.ir.latest()
        if hits == ("left",) and self.ir.usable("right") and right_cm >= self.TURN_CLEAR_SIDE_CM:
            self._log("front-left IR: shift right before retry")
            self.nudge(90.0, 0.04)
        elif hits == ("right",) and self.ir.usable("left") and left_cm >= self.TURN_CLEAR_SIDE_CM:
            self._log("front-right IR: shift left before retry")
            self.nudge(-90.0, 0.04)
        else:
            self._log("front-corner IR: no verified clear side - hold position, do not reverse blind")
            return False
        return True

    def _retryable_move_failure(self):
        note = self.last_move_note or ""
        # A safety stop proves only that moving is unsafe *now*.  It is not evidence of a
        # permanent maze wall: people, a stale ToF frame, loss of centring and corner IR can
        # all trigger it.  Real walls come from the stationary four-way scan.  Keeping these
        # edges temporary prevents a false wall from cutting off unvisited map cells.
        return bool(note)

    def _corner_hard_stop(self, hits, since, now):
        return len(hits) >= 2 or any(
            since.get(side) is not None and now - since[side] >= self.CORNER_HOLD_STOP_S
            for side in hits
        )

    def _log(self, msg):
        if getattr(self, "panel", None) is not None:
            self.panel.log(msg)
        else:
            print(msg)

    def setup_csv_headers(self):
        with open(self.pos_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "x", "y", "z"])
        with open(self.att_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "yaw", "pitch", "roll"])
        with open(self.imu_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        with open(self.esc_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "esc_data"])
        with open(self.dist_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "tof1"])

    def start_sensors(self):
        print("Starting to collect sensor data...")

        if robot is not None:
            try:
                self.ep_robot.set_robot_mode(mode=robot.FREE)
            except Exception as e:
                print(f"[warn] set_robot_mode: {e}")

        # odometry ต้องถี่พอ (1-5Hz ทำให้หยุดช้าไปเป็น 10-30cm)
        self.ep_chassis.sub_position(freq=max(self.freq_pos, 20), callback=self.handle_position)
        # 20 Hz: at 5 Hz the yaw is 200 ms old, turns overshoot and stop at the wrong angle
        self.ep_chassis.sub_attitude(freq=max(self.freq_att, 20), callback=self.handle_attitude)
        self.ep_chassis.sub_imu(freq=max(self.freq_imu, 50), callback=self.handle_imu)   # 50 Hz: a bump is short
        self.ep_chassis.sub_esc(freq=self.freq_esc, callback=self.handle_esc)
        # 20 Hz: at 5 Hz a scan could read the previous gimbal direction's wall
        self.ep_sensor.sub_distance(freq=max(self.freq_dist, 20), callback=self.handle_distance)
        self.ir.start()
        try:  # live gimbal pitch -> the camera's horizon rule (ignore the room above it)
            # 50 Hz: the auto-aim closes its loop on these angles (and matches frames to them)
            self.ep_gimbal.sub_angle(freq=50, callback=self._on_gimbal_angle)
        except Exception as e:
            print(f"[warn] gimbal sub_angle: {e}")

        try:
            self._gimbal_recenter(pitch_speed=200, yaw_speed=200)
            time.sleep(0.5)
        except Exception:
            pass
        self._check_pitch_sign()

    def _check_pitch_sign(self):
        """The angle feed must use the same pitch sign as moveto (camera down = negative, e.g.
        verify_pitch_deg -10): the speed-mode gimbal loop steers on the feed, so an opposite
        sign would make "look down 10" settle 10 deg UP. Tilt up 10 with moveto and see which
        way the feed moves; flip the feed if needed. ~0.6 s at start-up."""
        try:
            self.FEED_PITCH_SIGN = 1.0
            self.ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=2)
            time.sleep(0.2)
            p0 = self.gimbal_hist[-1][1] if self.gimbal_hist else None
            self.ep_gimbal.moveto(pitch=10, yaw=0, pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=2)
            time.sleep(0.25)
            p1 = self.gimbal_hist[-1][1] if self.gimbal_hist else None
            self.ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=2)
            if p0 is None or p1 is None or abs(p1 - p0) < 3:
                self._log(f"gimbal pitch check: feed did not follow (feed {p0} -> {p1}) - speed-mode pitch off, "
                          "using moveto for pitch")
                self.PITCH_BY_SPEED = False
                return
            if p1 < p0:
                self.FEED_PITCH_SIGN = -1.0
                self.gimbal_hist.clear()
                self._log("gimbal pitch check: the angle feed reports pitch with the opposite sign to moveto - "
                          "flipped (camera-down looks would have gone UP)")
            else:
                self._log("gimbal pitch check: feed and moveto agree")
        except Exception as e:
            self._log(f"gimbal pitch check failed: {e} - using moveto for pitch")
            self.PITCH_BY_SPEED = False

    def stop_sensors(self):
        time.sleep(self.buffer_time)
        self.ep_chassis.unsub_position()
        self.ep_chassis.unsub_attitude()
        self.ep_chassis.unsub_imu()
        self.ep_chassis.unsub_esc()
        self.ep_sensor.unsub_distance()
        self.ir.stop()
        try:
            self.ep_gimbal.unsub_angle()
        except Exception:
            pass
        if self.panel is None:  # the Mission Panel owns its window
            cv2.destroyAllWindows()
        print("Data collection and saving to the file have been fully completed.")

    # ------------------------------------------------------------------
    # Gimbal / basic move
    # ------------------------------------------------------------------
    def reset_gimbal(self):
        try:
            self._gimbal_moveto(pitch=0, yaw=0, yaw_speed=200, what="reset")
            time.sleep(0.1)
        except Exception:
            pass

    def move_forward(self, distance=None, speed=None):
        if distance is None:
            distance = self.default_distance
        if speed is None:
            speed = self.default_speed
        self.ep_chassis.move(x=distance, y=0, z=0, xy_speed=speed).wait_for_completed()

    def nudge(self, rel_deg, dist_m, speed=0.15):
        """Small slide of dist_m in any direction relative to the chassis (0 = forward,
        90 = right), heading held, no turning (mecanum wheels). For short moves inside a
        block only: no wall guard."""
        if dist_m <= 0.005:
            return 0.0
        a = math.radians(rel_deg)
        vx, vy = speed * math.cos(a), speed * math.sin(a)
        sx, sy, cnt0 = self.pos_x, self.pos_y, self._pos_count
        hold = self.current_yaw
        t0 = time.time()
        traveled = 0.0
        while time.time() - t0 < dist_m / speed * 2.5 + 0.5:
            traveled = math.hypot(self.pos_x - sx, self.pos_y - sy) if self._pos_count != cnt0 \
                else (time.time() - t0) * speed
            if traveled >= dist_m - 0.01:
                break
            # This move is often an IR escape. Watch the side it moves toward so an
            # off-centre chassis cannot escape one wall by touching the opposite wall.
            left_cm, right_cm = self.ir.latest()
            side = None
            angle = rel_deg % 360.0
            if 45.0 <= angle <= 135.0:
                side, gap = "right", right_cm
            elif 225.0 <= angle <= 315.0:
                side, gap = "left", left_cm
            if side is not None and self.ir.usable(side) and gap <= self.SIDE_SAFE_CM:
                self._log(f"nudge stopped: {side} wall {gap:.0f} cm")
                break
            z = max(min(wrap180(hold - self.current_yaw) * self.KP_YAW_HOLD, 30), -30) * self.z_sign
            self.ep_chassis.drive_speed(x=vx, y=vy, z=z)
            time.sleep(0.05)
        self._stop(0.2)
        return traveled

    def straight_way(self, pos, d, walls, opened, max_x, max_y):
        """Moving pos -> next block in direction d is a plain straight stretch: on each side
        both blocks have the same, known kind of edge (wall + wall, or open + open). Anything
        else - a wall that ends (corner), a doorway, an edge not known yet - is not."""
        mv = self.MOVES4
        nb = (pos[0] + mv[d][0], pos[1] + mv[d][1])

        def state(c, s):
            o = (c[0] + mv[s][0], c[1] + mv[s][1])
            if not (0 <= o[0] <= max_x and 0 <= o[1] <= max_y):
                return "wall"
            e = frozenset((tuple(c), o))
            return "wall" if e in walls else ("open" if e in opened else None)

        for s in ((d + 1) % 4, (d + 3) % 4):
            a, b = state(pos, s), state(nb, s)
            if a is None or b is None or a != b:
                return False
        return True

    def move_cell(self, d, heading, distance, camera_guard=True, pos=None, walls=None, opened=None,
                  bounds=None, speed=None):
        """Drive one block in map direction d (0 N, 1 E, 2 S, 3 W). Returns (moved, heading the
        chassis faces now). Strafe mode: on a straight stretch the chassis keeps facing
        `heading` and slides (no turn). Where a wall ends or there is a doorway / an unknown
        edge (a corner the robot's corners could catch) it turns to face the way, so the two
        front-corner IR modules watch the corners. No map knowledge given: slide."""
        if self.strafe and d != heading:
            straight = self.FIXED_HEADING or pos is None or walls is None or \
                self.straight_way(tuple(pos), d, walls, opened or set(), *(bounds or (99, 99)))
            if straight:
                if (d - heading) % 2 == 1:          # sideways: a little slower (no front IR that way)
                    speed = min(speed or self.CELL_SPEED, self.SLIDE_SPEED)
                rel = (d - heading) % 4
                ok = self.safe_move_forward(distance=distance, target_heading_deg=self.heading_to_yaw(heading),
                                            camera_guard=camera_guard, body_dir=rel, speed=speed)
                return ok, heading
            print(f"-> corner / doorway ahead ({tuple(pos)} dir {d}): turn to face it - front IR on")
        if d == heading:                      # already facing the way: straight forward
            return self.safe_move_forward(distance=distance, target_heading_deg=self.heading_to_yaw(d),
                                          camera_guard=camera_guard, speed=speed), d
        yaw = self.heading_to_yaw(d)
        if self.CENTER_BEFORE_TURN:
            self._center_in_cell(self.heading_to_yaw(heading), duration=min(0.6, self.CENTER_TIME_S))
        if not self.turn_to_absolute_yaw(yaw):
            self.last_move_note = "turn stopped by safety guard"
            return False, heading
        return self.safe_move_forward(distance=distance, target_heading_deg=yaw,
                                      camera_guard=camera_guard, speed=speed), d

    def _path_clearance_scan(self, body_dir):
        """Check three ToF rays across the inflated chassis corridor."""
        base = self.GIMBAL_FOR_BODY[body_dir]
        rays = []
        for off in (-self.CLEARANCE_SCAN_DEG, 0.0, self.CLEARANCE_SCAN_DEG):
            if not self._gimbal_moveto(pitch=0, yaw=wrap180(base + off), pitch_speed=min(240, self.GIMBAL_DPS),
                                       yaw_speed=self.GIMBAL_DPS, what="clearance scan"):
                return 0.0
            time.sleep(0.08)                       # the ToF updates ~10 Hz and lags the gimbal
            rays.append((off, self._tof_fresh(3)))
        self._last_rays = rays
        if not self._gimbal_moveto(pitch=0, yaw=base, pitch_speed=min(240, self.GIMBAL_DPS),
                                   yaw_speed=self.GIMBAL_DPS, what="clearance centre"):
            return 0.0
        return corridor_obstacle(rays, self.CORRIDOR_HALF_M, self.PRECHECK_BLOCK_M)

    def recenter(self, dists):
        """Put the robot back in the middle of its block from the scan's ToF readings
        (front/right/back/left, relative to the chassis): walls on both sides of an axis ->
        halfway between them; one wall -> center_tof_mm from it. Small slides, no turning.
        Stops odometry drift (and sideways slip) from adding up block after block."""
        if not self.recenter_enabled:
            return
        near = lambda v: v is not None and 60 < v < self.wall_mm      # a wall of this block
        moves = []
        for a, b, rel in (("front", "back", 0.0), ("right", "left", 90.0)):
            ra, rb = dists.get(a), dists.get(b)
            if near(ra) and near(rb):
                off = (ra - rb) / 2.0                 # + = too close to b: slide towards a
            elif near(ra):
                off = ra - self.CENTER_TOF_MM
            elif near(rb):
                off = -(rb - self.CENTER_TOF_MM)
            else:
                continue
            if abs(off) < self.RECENTER_DEADBAND_MM:
                continue
            m = min(self.RECENTER_MAX_M, abs(off) / 1000.0)
            moves.append((rel if off > 0 else rel + 180.0, m))
        for rel, m in moves:
            self._log(f"recentre: slide {m * 100:.0f} cm {['forward', 'right', 'back', 'left'][int(rel % 360) // 90]}")
            self.nudge(wrap180(rel), m)

    def _stop(self, wait=0.3):
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(wait)

    def _wait_gimbal(self, action, what):
        """Wait for a gimbal action without ever freezing the mission thread."""
        try:
            done = action.wait_for_completed(timeout=self.GIMBAL_TIMEOUT_S)
            if done is False:
                raise TimeoutError(f"not completed in {self.GIMBAL_TIMEOUT_S:.1f}s")
            return True
        except Exception as e:
            try:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            self._log(f"gimbal {what} failed: {e} - continuing safely")
            return False

    def _gimbal_moveto(self, pitch=0, yaw=0, pitch_speed=30, yaw_speed=30, what="move"):
        """Point the gimbal (chassis-relative angles). With the 50 Hz angle feed live it is
        driven with speed commands (drive_speed) on a braking curve - it slows down as it
        arrives instead of a position action that runs past and swings back; without the
        feed the SDK's moveto action is used."""
        if self.gimbal_hist and time.time() - self.gimbal_hist[-1][0] < 0.3 and \
                (self.PITCH_BY_SPEED or abs(pitch - self.gimbal_hist[-1][1]) < self.GIMBAL_TOL_DEG):
            if self._gimbal_drive_to(pitch, yaw, pitch_speed, yaw_speed, what):
                return True
            # A live angle feed does not guarantee that speed-mode control is usable. In
            # particular, just after resume the feed can be fresh while drive_speed is ignored
            # or briefly inconsistent. Falling straight through used to make every direction
            # in the first map scan unknown, so the planner declared the start cell to be the
            # whole reachable map. Do not start a fallback command after an emergency STOP.
            if self.panel is not None and self.panel.abort.is_set():
                return False
            self._log(f"gimbal {what}: speed control did not settle - retrying with SDK moveto")
        try:
            action = self.ep_gimbal.moveto(pitch=pitch, yaw=yaw,
                                           pitch_speed=pitch_speed, yaw_speed=yaw_speed)
        except Exception as e:
            self._log(f"gimbal {what} command failed: {e}")
            return False
        return self._wait_gimbal(action, what)

    def _gimbal_drive_to(self, pitch, yaw, pitch_speed, yaw_speed, what):
        """drive_speed loop onto (pitch, yaw): speed = min(top speed, sqrt(2 x brake x error)),
        stop predicted one feed-delay early (the angle feed is ~40 ms old), then a slow last
        step. Returns True when within GIMBAL_TOL_DEG."""
        g = self.ep_gimbal
        brake = self.GIMBAL_BRAKE_DPS2
        lag = 0.04                                     # angle feed age
        t_end = time.time() + self.GIMBAL_TIMEOUT_S
        ok = False
        try:
            while time.time() < t_end:
                if self.panel is not None and self.panel.abort.is_set():
                    break
                h = self.gimbal_hist
                p_now, y_now = h[-1][1], h[-1][2]
                # chassis-relative yaw is NOT wrapped: the gimbal turns +-250 deg, never "the short way"
                ep, ey = pitch - p_now, yaw - y_now
                # measured speed (feed): stop commanding once it will coast the rest of the way
                if len(h) >= 3 and h[-1][0] > h[-3][0]:
                    dt = h[-1][0] - h[-3][0]
                    mp, my = (h[-1][1] - h[-3][1]) / dt, (h[-1][2] - h[-3][2]) / dt
                else:
                    mp = my = 0.0
                if abs(ep) <= self.GIMBAL_TOL_DEG and abs(ey) <= self.GIMBAL_TOL_DEG:
                    ok = True
                    break
                def axis(err, vmax, meas):
                    if abs(err) <= self.GIMBAL_TOL_DEG:
                        return 0.0
                    if meas * err > 0 and abs(err) < abs(meas) * (lag + self.GIMBAL_RESPONSE_S):
                        return 0.0                     # it coasts the rest: no swing past
                    v = min(vmax, math.sqrt(2.0 * brake * abs(err)))
                    return math.copysign(max(v, 10.0), err)
                vp, vy = axis(ep, pitch_speed, mp), axis(ey, yaw_speed, my)
                g.drive_speed(pitch_speed=vp, yaw_speed=vy)
                time.sleep(0.02)
        except Exception as e:
            self._log(f"gimbal {what} (speed) failed: {e} - continuing safely")
        finally:
            try:
                g.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
        if not ok:
            self._log(f"gimbal {what}: not there in {self.GIMBAL_TIMEOUT_S:.1f}s - continuing safely")
            return False
        time.sleep(0.03)                               # let it stand still (frames / ToF)
        return True

    def _gimbal_recenter(self, pitch_speed=100, yaw_speed=100):
        try:
            action = self.ep_gimbal.recenter(pitch_speed=pitch_speed, yaw_speed=yaw_speed)
        except Exception as e:
            self._log(f"gimbal recenter command failed: {e}")
            return False
        return self._wait_gimbal(action, "recenter")

    # ------------------------------------------------------------------
    # ตรวจทิศ z กับ yaw
    # ------------------------------------------------------------------
    def calibrate_yaw_sign(self, start_heading=0):
        print("--> [Calibrate] ตรวจทิศ z กับ yaw ...")
        self._stop(0.3)
        y0 = self.current_yaw
        self.ep_chassis.drive_speed(x=0, y=0, z=30)
        time.sleep(0.6)
        self._stop(0.3)
        d = wrap180(self.current_yaw - y0)
        if abs(d) < 3:
            print(f"[warn] yaw เปลี่ยนน้อยเกินไป ({d:.1f}°) ใช้ z_sign=+1 ตามเดิม")
            self.z_sign = 1
        else:
            self.z_sign = 1 if d > 0 else -1
        print(f"    z>0 ทำให้ yaw เปลี่ยน {d:+.1f}° -> z_sign={self.z_sign:+d}")
        self.turn_to_absolute_yaw(y0)
        # the robot faces map direction start_heading (0 N, 1 E, 2 S, 3 W): north is that much off
        s = self.right_yaw_sign
        off = {0: 0.0, 1: 90.0 * s, 2: 180.0, 3: -90.0 * s}[int(start_heading) % 4]
        self.yaw_origin = wrap180(y0 - off)
        self._odom0 = (self.pos_x, self.pos_y, self.yaw_origin)   # map origin for the odometry trail
        if self.panel is not None:
            self.panel.map.odom = []
        print(f"    yaw_origin={y0:.1f}° (ทิศเหนือของแผนที่)")

    # ------------------------------------------------------------------
    # เดินตรง: distance วัดจาก odometry ของล้อ (ไม่ใช้เวลา) + IMU คุมทิศ + IR คุมข้าง
    # ------------------------------------------------------------------
    GIMBAL_FOR_BODY = {0: 0, 1: 90, 2: 180, 3: -90}   # gimbal yaw that looks where the robot drives

    def safe_move_forward(self, distance=0.6, speed=None, stop_limit_mm=None, target_heading_deg=None,
                          camera_guard=True, body_dir=0):
        """Drive one cell (see _safe_move) - and remember the drive (time -> distance covered)
        so cards the camera saw on the way in can be placed in the new block."""
        self._move_rec = {"t0": time.time(), "dist": distance, "body_dir": body_dir, "track": [(time.time(), 0.0)],
                          "heading_deg": self.current_yaw}
        self._driving = True
        try:
            ok = self._safe_move(distance=distance, speed=speed, stop_limit_mm=stop_limit_mm,
                                 target_heading_deg=target_heading_deg, camera_guard=camera_guard,
                                 body_dir=body_dir)
        finally:
            self._driving = False
            self._move_rec["t1"] = time.time()
        if ok and body_dir in (0, 2) and self.wall_pid is not None and self.CENTER_TIME_S > 0:
            self._center_in_cell(target_heading_deg)
        return ok

    def _center_in_cell(self, target_yaw=None, duration=None):
        """Arrived: a short in-place correction on the side Sharps + heading (no forward speed)
        until the gaps are even (within the PID dead band) - the next scan and turn start from
        the middle of the block."""
        if self.wall_pid is None or self.ir.mount != "side":
            return True
        if not (self.ir.usable("left") or self.ir.usable("right")):
            self._log("centre check unavailable: both side IR sensors offline - not assuming centred")
            self._stop(0.1)
            return False
        target_yaw = self.current_yaw if target_yaw is None else target_yaw
        t_end = time.time() + (self.CENTER_TIME_S if duration is None else duration)
        self.wall_pid.reset()
        centred = False
        last_err = 0.0
        try:
            while time.time() < t_end:
                if self.panel is not None and self.panel.abort.is_set():
                    break
                l_cm, r_cm = self._control_side_ir()
                err, _ = self.wall_pid.lateral_error(
                    l_cm * 10.0 if l_cm is not None and l_cm < self.ir.max_cm - 1 else None,
                    r_cm * 10.0 if r_cm is not None and r_cm < self.ir.max_cm - 1 else None)
                last_err = err
                vy = self.wall_pid.center(err, dt=0.05)
                vz = self.wall_pid.yaw(wrap180(target_yaw - self.current_yaw), dt=0.05)
                if abs(err) < self.wall_pid.tol_mm and abs(vz) < 1.0:
                    centred = True
                    break
                self.ep_chassis.drive_speed(x=0, y=vy, z=self.z_sign * vz)
                time.sleep(0.05)
        finally:
            self._stop(0.1)
        if not centred and self.panel is not None and not self.panel.abort.is_set():
            self._log(f"centre check ended with lateral error {last_err / 10.0:+.1f} cm; "
                      "continuing slowly with collision guards")
        return centred

    def _fine_align(self, target_yaw, timeout=3.0):
        """After a turn: slow closed-loop correction until the heading is within FINE_TURN_TOL
        for 3 readings in a row (Dhai_8's align_turn_heading)."""
        t_end, good = time.time() + timeout, 0
        try:
            while time.time() < t_end:
                err = wrap180(target_yaw - self.current_yaw)
                if abs(err) <= self.FINE_TURN_TOL:
                    self.ep_chassis.drive_speed(x=0, y=0, z=0)
                    good += 1
                    if good >= 3:
                        return True
                else:
                    good = 0
                    v = max(4.0, min(15.0, abs(err) * 1.8))
                    self.ep_chassis.drive_speed(x=0, y=0, z=self.z_sign * math.copysign(v, err))
                time.sleep(0.03)
            return abs(wrap180(target_yaw - self.current_yaw)) <= self.TURN_OK_DEG
        finally:
            self._stop(0.1)

    def _safe_move(self, distance=0.6, speed=None, stop_limit_mm=None, target_heading_deg=None,
                          camera_guard=True, body_dir=0):
        """Drive one cell. body_dir = which way relative to the chassis: 0 forward, 1 right,
        2 back, 3 left - the mecanum wheels slide sideways / backwards without turning, and the
        gimbal (ToF + camera) points the way it drives to guard it. camera_guard=True performs
        the full left/centre/right corridor pre-scan; ``"direct"`` is used immediately after a
        stationary scan proved this edge open and keeps the straight ToF + camera guards
        without repeating three gimbal rays; False is for an edge already driven through.
        self.last_move_note says why a move gave up."""
        self.last_move_note = ""
        if speed is None:
            speed = self.CELL_SPEED
        if stop_limit_mm is None:
            stop_limit_mm = self.FRONT_STOP_MM
        print(f"--> [Safe Move] {['forward', 'right', 'back', 'left'][body_dir]} {distance}m ที่ {speed}m/s")
        if body_dir == 0:
            self.reset_gimbal()
        else:
            if not self._gimbal_moveto(pitch=0, yaw=self.GIMBAL_FOR_BODY[body_dir], pitch_speed=min(240, self.GIMBAL_DPS),
                                       yaw_speed=self.GIMBAL_DPS, what="point along travel"):
                self.last_move_note = "gimbal could not point along travel"
                return False
            time.sleep(0.05)

        # camera + ToF look before moving: a wrong turn would face a wall here
        if target_heading_deg is not None and abs(wrap180(target_heading_deg - self.current_yaw)) > self.TURN_OK_DEG:
            if not self.turn_to_absolute_yaw(target_heading_deg):
                self.last_move_note = "unsafe or failed heading correction before move"
                return False
        if camera_guard is True:
            self._last_rays = []
            obstacle_m = self._path_clearance_scan(body_dir)
            if obstacle_m is not None:
                rays = dict(getattr(self, "_last_rays", []) or [])
                centre = rays.get(0.0)
                side = [o for o, mm in rays.items() if o != 0.0 and 60 < mm < 8000
                        and mm / 1000.0 * math.cos(math.radians(o)) <= self.PRECHECK_BLOCK_M]
                if centre is not None and 60 < centre < 8000 and centre / 1000.0 > self.PRECHECK_BLOCK_M + 0.1 \
                        and len(side) == 1:
                    # something at one side of the way (a wall end / doorway post): the last run
                    # drove past such a post "slowly" and clipped it. Slide away from it, look again.
                    base = self.GIMBAL_FOR_BODY[body_dir]
                    away = wrap180(base - 90.0) if side[0] > 0 else wrap180(base + 90.0)
                    self._log(f"doorway post / wall end {obstacle_m:.2f} m at the "
                              f"{'right' if side[0] > 0 else 'left'} of the way - sliding 5 cm away, checking again")
                    self.nudge(away, 0.05)
                    obstacle_m = self._path_clearance_scan(body_dir)
                    if obstacle_m is None:
                        speed = min(speed, self.UNKNOWN_SPEED)
                if obstacle_m is None:
                    pass
                else:
                    self.last_move_note = f"clearance scan obstacle {obstacle_m:.2f} m ahead"
                    self._log(f"path corridor blocked at {obstacle_m:.2f} m - re-planning")
                    return False
        nav_sample_start = time.time()
        direct_tof = self._tof_fresh(3)
        if self._tof_last_ts <= nav_sample_start or not 0 < direct_tof < 8000:
            self.last_move_note = "no fresh ToF navigation data"
            self._log("ToF has no fresh valid range - not moving")
            return False
        # Discard the angled pre-scan rays. From here on every retained sample is
        # taken with the ToF pointing along the actual velocity vector.
        self._tof_samples[:] = [(t, mm) for t, mm in self._tof_samples if t >= nav_sample_start]
        # (fresh ToF readings are only waited for when both modules say "wall")
        if body_dir == 0 and (self.ir.corners_enabled or self.ir.mount != "side") and \
                self.ir.near("left") and self.ir.near("right") and \
                self._corner_says_wall("left", self._tof_fresh(3)) and self._corner_says_wall("right"):
            self.last_move_note = "both front-corner IR near a wall"
            self._log(f"front corners both < {self.ir.corner_trigger_cm:.0f} cm - not moving")
            return False
        cam_d = self._camera_wall_ahead(fresh=True) if camera_guard else float("inf")
        if cam_d < self.CAM_BLOCK_M:
            tof = self._tof_fresh(3)
            tof_ok = 60 < tof < 8000
            # the camera is a second opinion: it blocks only when the ToF also sees something CLOSE
            # (a normal next cell has its far wall ~850 mm away), or when the ToF has no reading
            if (tof_ok and tof < self.CAM_TOF_BLOCK_MM) or (not tof_ok and cam_d < self.CAM_STOP_M):
                self.last_move_note = f"camera {cam_d:.2f} m + ToF {tof:.0f} mm: wall ahead"
                self._log(f"camera: wall {cam_d:.2f} m ahead (ToF {tof:.0f} mm) - not moving")
                return False
            self._log(f"camera says wall {cam_d:.2f} m but ToF {tof:.0f} mm - trusting the ToF")

        sx, sy = self.pos_x, self.pos_y
        cnt0 = self._pos_count
        start_time = time.time()
        # Guard against stalled odometry, but allow the configured corner-caution speed.
        slow_fraction = min(0.4, max(0.15, self.CORNER_SLOW_FRAC))
        max_time = distance / (slow_fraction * speed) * 1.1 + 0.5

        if target_heading_deg is None:
            target_heading_deg = self.current_yaw

        traveled = 0.0
        pos_ok = False
        stop_traveled = None     # ระยะตอนสั่งหยุด (ใช้เรียนรู้ระยะไหล)
        drift_int = 0.0          # sideways slide: integral of the front-back drift
        corner_since = {"left": None, "right": None}
        corner_noted = set()
        # side Sharps once the robot is in the new block (forward / back moves only: then they
        # face that block's side walls) - a corridor block needs no full scan
        self._side_samples, self._last_body_dir = [], body_dir
        self._drive_start()
        self._hit_state = None
        if self.wall_pid is not None:
            self.wall_pid.reset()
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    print("[warn] odometry ช้า/ค้าง หยุดด้วยเพดานเวลา")
                    break
                if self._wheels_dead():
                    self._stop(0.1)
                    if not self._wheels_not_answering("driving"):
                        self.last_move_note = "wheels not answering"
                        return False
                    sx, sy = self.pos_x, self.pos_y
                    cnt0 = self._pos_count
                    start_time = time.time()
                    self._drive_start()
                    continue

                pos_ok = self._pos_count != cnt0
                if pos_ok:
                    traveled = math.hypot(self.pos_x - sx, self.pos_y - sy)
                    rec = getattr(self, "_move_rec", None)
                    if rec is not None:
                        rec["track"].append((time.time(), traveled))
                else:
                    traveled = elapsed * speed   # fallback ถ้า odometry ไม่มา

                # ระยะที่เหลือ หักระยะไหลที่เรียนรู้ไว้
                remaining = distance - (traveled + self.coast_est)
                if remaining <= 0:
                    stop_traveled = traveled
                    break

                progress = traveled / distance
                # One 10 cm corner warning slows and steers away. Both corners, or one that
                # remains active, stop the robot. This avoids treating a nearby side wall as
                # a blocked path while retaining a bounded emergency response.
                corner_hits = self._front_corner_hits() if body_dir == 0 else ()
                now = time.time()
                for side in ("left", "right"):
                    corner_since[side] = (corner_since[side] or now) if side in corner_hits else None
                if corner_hits and self._corner_hard_stop(corner_hits, corner_since, now):
                    self._stop(0.3)
                    sides = "+".join(corner_hits)
                    self.last_move_note = (f"front-{sides} IR emergency stop at "
                                           f"{traveled:.2f} m")
                    self._log(f"front-{sides} IR triggered at ~{self.ir.corner_trigger_cm:.0f} cm "
                              "while moving - emergency stop")
                    self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                    self._make_corner_room(corner_hits)
                    return False
                for side in corner_hits:
                    if side not in corner_noted:
                        self._log(f"front-{side} IR caution: slowing and steering away")
                        corner_noted.add(side)
                corner_noted.intersection_update(corner_hits)
                front_dist = self._tof_recent(self.TOF_STALE_S)
                if front_dist is None:
                    self.last_move_note = "ToF navigation data became stale"
                    self._log("ToF data stale while moving - emergency stop")
                    self._stop(0.3)
                    self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                    return False
                # sideways, the chassis side is closer to the wall than the ToF head: stop earlier
                stop_mm = max(stop_limit_mm, self.SLIDE_STOP_MM) if body_dir in (1, 3) else stop_limit_mm

                # ---- EMERGENCY STOP: about to hit (or just hit) something
                hit = self._about_to_hit(body_dir, front_dist, stop_mm, start_time)
                if hit:
                    kind_, why = hit
                    self._stop(0.25)
                    self._log(f"EMERGENCY STOP: {why}")
                    if kind_ == "side" and progress < 0.9:
                        # grazing a side wall: step off it and carry on with the move
                        self.nudge(90.0 if hit[1].startswith("left") else -90.0, 0.03)
                        self._hit_state = None
                        continue
                    if progress >= 0.80 or self._arrived_short(progress, front_dist):
                        return True
                    self.last_move_note = f"emergency stop: {why}"
                    self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                    return False

                min_speed_stop_mm = stop_mm + 1000.0 * (
                    self.MIN_V * self.REACTION_S + self.MIN_V * self.MIN_V / (2.0 * self.DECEL_A)
                )
                if 0 < front_dist <= min_speed_stop_mm:
                    self._stop(0.3)
                    if progress >= 0.80 or self._arrived_short(progress, front_dist):
                        print(f"-> [ถึงเป้าหมาย] พบกำแพงหน้าช่องที่ {front_dist}mm")
                        return True
                    self.last_move_note = f"ToF obstacle {front_dist} mm at {traveled:.2f} m"
                    print(f"!!! [สิ่งกีดขวาง] {front_dist}mm หยุดทันที")
                    self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                    return False

                # camera: a white wall coming into the robot's path
                cam_d = self._camera_wall_ahead() if camera_guard else float("inf")
                tof_now = self.current_tof_dist_mm
                # both must say CLOSE: driving into a cell the ToF passes 600 mm long before arriving
                if cam_d < self.CAM_STOP_M and (not 60 < tof_now < 8000 or tof_now < self.CAM_TOF_STOP_MM):
                    self.last_move_note = f"camera {cam_d:.2f} m + ToF {tof_now:.0f} mm while moving"
                    self._stop(0.3)
                    if progress >= 0.80 or self._arrived_short(progress, tof_now):
                        print(f"-> [ถึงเป้าหมาย] กล้องเห็นกำแพงหน้าช่อง {cam_d:.2f} m")
                        self.last_move_note = ""
                        return True
                    self._log(f"camera: wall {cam_d:.2f} m ahead while moving - stop")
                    self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                    return False

                # heading drifted too far (slip / bad turn): stop and turn back first
                if abs(wrap180(target_heading_deg - self.current_yaw)) > self.HEADING_GUARD_DEG:
                    self._stop(0.2)
                    self._log(f"heading off by {wrap180(target_heading_deg - self.current_yaw):+.0f} deg while moving - correcting")
                    t_fix = time.time()
                    if not self.turn_to_absolute_yaw(target_heading_deg):
                        self.last_move_note = "heading could not be corrected"
                        self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                        return False
                    start_time += time.time() - t_fix   # the correction does not count against the move's time cap
                    continue

                # ชะลอช่วงท้ายให้หยุดแม่น: constant braking (v = sqrt(2 a d)) - full speed until
                # it has to brake, then a smooth stop, instead of crawling the last centimetres
                v_forward = min(speed, max(self.MIN_V, math.sqrt(2.0 * self.DECEL_A * max(remaining, 0.0))))
                # something ahead (the gimbal ToF looks the way it drives): slow down as it gets
                # close, so a stop at the limit is gentle - not a 0.45 m/s arrival at a wall
                if 60 < front_dist < 8000:
                    sensor_limit = braking_speed(front_dist / 1000.0, stop_mm / 1000.0, speed,
                                                 self.REACTION_S, self.DECEL_A)
                    v_forward = min(v_forward, max(self.MIN_V, sensor_limit),
                                    max(self.MIN_V, self.APPROACH_K * (front_dist - stop_mm) / 1000.0))

                yaw_error = wrap180(target_heading_deg - self.current_yaw)
                if self.wall_pid is not None:
                    z_val = self.wall_pid.yaw(yaw_error, dt=0.05)      # heading PID (P + I + D)
                else:
                    z_val = max(min(yaw_error * self.KP_YAW_HOLD, 30), -30)
                z_cmd = self.z_sign * z_val

                left_cm, right_cm = self.read_side_ir()
                left_ctl = left_cm if self.ir.usable("left") else None
                right_ctl = right_cm if self.ir.usable("right") else None
                if body_dir in (0, 2) and progress >= 0.55:
                    self._side_samples.append((left_cm, right_cm))
                y_speed = 0.0

                corner_acted = False
                if body_dir == 0 and corner_hits:
                    y_speed = self.CORNER_STEER_MPS if "left" in corner_hits else -self.CORNER_STEER_MPS
                    v_forward = min(v_forward, max(self.MIN_V, speed * self.CORNER_SLOW_FRAC))
                    corner_acted = True
                x_corr = 0.0
                if body_dir in (1, 3):
                    # sliding sideways: nothing steers the robot's front-back position, and the
                    # mecanum wheels creep along it (last run: 20 cm over a 3-block slide, into the
                    # wall). Hold it with odometry, and back away from what the front corners see.
                    if pos_ok:
                        f = math.radians(self.current_yaw)
                        drift = (self.pos_x - sx) * math.cos(f) + (self.pos_y - sy) * math.sin(f)
                        # PI: the creep is steady, so a P term alone leaves a few cm per block
                        drift_int = max(min(drift_int + drift * 0.05, 0.05), -0.05)
                        x_corr = max(min(-(self.SLIDE_HOLD_KP * drift + self.SLIDE_HOLD_KI * drift_int),
                                         self.STRAFE_V), -self.STRAFE_V)
                    if self.ir.corners_enabled and (self.ir.near("left") or self.ir.near("right")):
                        x_corr = -self.STRAFE_V            # front close to a wall: ease back
                    # the Sharp on the side it slides towards is a bumper
                    ahead_cm = right_ctl if body_dir == 1 else left_ctl
                    if ahead_cm is not None and ahead_cm < self.SIDE_DANGER_CM:
                        self._stop(0.3)
                        if progress >= 0.80 or self._arrived_short(progress, self.current_tof_dist_mm):
                            return True
                        self.last_move_note = f"side Sharp {ahead_cm:.0f} cm in the way at {traveled:.2f} m"
                        self._log(f"sliding: wall {ahead_cm:.0f} cm at the {'right' if body_dir == 1 else 'left'} - stop")
                        self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                        return False
                if self.ir.mount == "side" and not corner_acted and body_dir in (0, 2):
                    # Sharp distance sensors on the sides: keep off the side walls
                    if left_ctl is not None and left_ctl < self.SIDE_SAFE_CM and \
                            (right_ctl is None or left_ctl <= right_ctl):
                        y_speed = self.STRAFE_V
                    elif right_ctl is not None and right_ctl < self.SIDE_SAFE_CM:
                        y_speed = -self.STRAFE_V
                    elif self.wall_pid is not None:
                        # wall centering PID: both walls -> equal gaps, one wall -> its nominal
                        # gap, none -> hold the line (Dhai_8's 8 wall cases)
                        err, _case = self.wall_pid.lateral_error(
                            left_ctl * 10.0 if left_ctl is not None and left_ctl < self.ir.max_cm - 1 else None,
                            right_ctl * 10.0 if right_ctl is not None and right_ctl < self.ir.max_cm - 1 else None)
                        y_speed = self.wall_pid.lateral(err, dt=0.05)
                    elif left_cm < self.ir.max_cm - 2 and right_cm < self.ir.max_cm - 2:
                        # กำแพงสองข้าง ประคองกลางทาง (left มากกว่า = ชิดขวา -> เลื่อนซ้าย)
                        y_speed = max(min(-0.01 * (left_cm - right_cm), 0.15), -0.15)
                    if any(v is not None and v < self.SIDE_DANGER_CM for v in (left_ctl, right_ctl)):
                        v_forward = min(v_forward, speed * 0.4)
                    # drifted off the block's centre line (slip, a bad reading): stop, not hit
                    if pos_ok and self.MAX_LATERAL_M > 0:
                        f = math.radians(self.current_yaw)
                        lateral = -(self.pos_x - sx) * math.sin(f) + (self.pos_y - sy) * math.cos(f)
                        if abs(lateral) > self.MAX_LATERAL_M:
                            self._stop(0.3)
                            self.last_move_note = f"drifted {lateral * 100:+.0f} cm off the line"
                            self._log(f"drifted {lateral * 100:+.0f} cm sideways off the line - stop, back to the block")
                            self._retreat(traveled, sx, sy, pos_ok, speed, body_dir)
                            return False

                # (forward speed, side correction) -> the chassis frame for this direction
                vx, vy = {0: (v_forward, y_speed), 2: (-v_forward, y_speed),
                          1: (x_corr, v_forward), 3: (x_corr, -v_forward)}[body_dir]
                self.ep_chassis.drive_speed(x=vx, y=vy, z=z_cmd)
                time.sleep(0.05)

        except Exception as e:
            print(f"[-] เกิดข้อผิดพลาดในการเคลื่อนที่: {e}")
            self._stop(0.4)
            return False

        self._stop(0.25)
        if self._pos_count != cnt0:
            traveled = math.hypot(self.pos_x - sx, self.pos_y - sy)

        # เรียนรู้ระยะไหลหลังสั่งหยุด (ปรับทีละครึ่ง กันค่ากระโดด)
        if stop_traveled is not None and pos_ok:
            coast = max(0.0, traveled - stop_traveled)
            self.coast_est = min(self.COAST_MAX_M, 0.5 * self.coast_est + 0.5 * coast)

        print(f"   [Move] odom={traveled:.3f}m เวลา={time.time() - start_time:.2f}s "
              f"pos_updates={self._pos_count - cnt0} coast={self.coast_est:.3f}m")
        if traveled < 0.8 * distance and not self.last_move_note:
            self.last_move_note = f"only {traveled:.2f} m of {distance:.2f} m (time cap)"
        return traveled >= 0.8 * distance

    def _about_to_hit(self, body_dir, front_mm, stop_mm, move_t0):
        """(kind, why) when the robot is about to hit something - or just did - else None.
          * ahead (the ToF looks the way it drives): time to contact at the measured speed
            under emergency_ttc_s, counting the stop distance as the contact point
          * a side Sharp (forward / back moves: they face the side walls) this close, or
            closing so fast it would touch within emergency_ttc_s
          * the IMU felt a bump: a spike over emergency_bump_g in the floor plane (not in the
            first 0.4 s of the move - that is the start)"""
        now = time.time()
        l_cm, r_cm = self.ir.latest()
        pos = (self.pos_x, self.pos_y)
        hist = getattr(self, "_hit_state", None) or []
        hist = [h for h in hist if now - h[0] <= 0.5] + [(now, pos, l_cm, r_cm)]
        self._hit_state = hist
        # rates over >= 0.2 s (Sharp noise is +-2 cm, odometry comes every ~75 ms)
        old_ = [h for h in hist if now - h[0] >= 0.2]
        if not old_:
            if body_dir in (0, 2) and self.ir.mount == "side" and len(hist) >= 3 and \
                    all(min(h[2], h[3]) <= self.SIDE_HIT_CM for h in hist[-3:]):
                return "side", f"{'left' if l_cm <= r_cm else 'right'} wall {min(l_cm, r_cm):.0f} cm"
            return None
        st = old_[-1]
        dt = now - st[0]
        v = math.hypot(pos[0] - st[1][0], pos[1] - st[1][1]) / dt           # m/s, odometry
        if 60 < (front_mm or 0) < 8000 and v > 0.05:
            room = (front_mm - max(60.0, 0.5 * stop_mm)) / 1000.0
            if room / v < self.TTC_STOP_S:
                return "ahead", f"{front_mm:.0f} mm ahead at {v:.2f} m/s - would hit in {max(room, 0) / v:.2f} s"
        if body_dir in (0, 2) and self.ir.mount == "side":
            for name, now_cm, was_cm in (("left", l_cm, st[2]), ("right", r_cm, st[3])):
                if now_cm >= self.ir.max_cm - 1:
                    continue
                closing = (was_cm - now_cm) / dt                                # cm/s towards it
                if closing > self.SIDE_MAX_CLOSING:
                    continue        # faster than the robot can drift: a wall end / sensor jump (last
                                    # run: 4 stops at "closing 105-116 cm/s"), not a wall coming in
                prev_close = [h for h in hist[-3:-1] if (h[2] if name == "left" else h[3]) <= self.SIDE_HIT_CM]
                if (now_cm <= self.SIDE_HIT_CM and prev_close) or (now_cm <= self.SIDE_HIT_CM + 3 and closing > 10
                                                                    and (now_cm - 2.0) / closing < self.TTC_STOP_S):
                    return "side", f"{name} wall {now_cm:.0f} cm, closing {max(closing, 0):.0f} cm/s"
        acc = getattr(self, "_imu_acc", None)
        if acc and now - acc[0] < 0.1 and now - move_t0 > 0.4 and math.hypot(acc[1], acc[2]) > self.BUMP_G:
            return "bump", f"IMU felt a bump ({math.hypot(acc[1], acc[2]):.2f} g)"
        return None

    def _arrived_short(self, progress, tof_mm):
        """Stopped more than half-way into the next block with a wall close ahead: that wall
        is the far wall of the new block (a dead end, or a card on it), so the robot IS in
        that block - just a little short of its centre. Not a blocked move."""
        if progress >= self.ARRIVE_MIN_PROGRESS and 60 < (tof_mm or 0) <= self.ARRIVE_TOF_MM:
            self._log(f"arrived {progress * 100:.0f}% in, wall {tof_mm:.0f} mm ahead = far wall of this block")
            return True
        return False

    def _retreat(self, traveled, sx, sy, pos_ok, speed, body_dir=0):
        """Return toward the previous cell with the ToF watching the retreat direction."""
        if traveled <= 0.03:
            return
        if not (self.ir.usable("left") or self.ir.usable("right")):
            self._log("retreat cancelled: both side IR sensors offline; refusing a blind reverse")
            self._stop(0.2)
            return
        print("-> [Retreat] ถอยกลับเข้ากลางช่องเดิม")
        v = min(speed, 0.12)
        retreat_dir = (body_dir + 2) % 4
        watched = self._gimbal_moveto(
            pitch=0, yaw=self.GIMBAL_FOR_BODY[retreat_dir],
            pitch_speed=min(240, self.GIMBAL_DPS), yaw_speed=self.GIMBAL_DPS,
            what="watch retreat path",
        )
        if watched:
            rear = self._tof_fresh(2, timeout=0.4)
            if 60 < rear <= self.FRONT_STOP_MM:
                self._log(f"retreat cancelled: wall {rear:.0f} mm behind")
                return

        def retreat_clear():
            if self.panel is not None and self.panel.abort.is_set():
                return False
            if not watched:
                return True
            mm = self._tof_recent(0.3)
            if mm is not None and 60 < mm <= self.FRONT_STOP_MM:
                self._log(f"retreat stopped: wall {mm:.0f} mm behind")
                return False
            return True

        if body_dir:
            ux, uy = {1: (0, -1), 2: (1, 0), 3: (0, 1)}[body_dir]
            t0 = time.time()
            while time.time() - t0 < (3.0 if pos_ok else traveled / v):
                if not retreat_clear():
                    break
                if pos_ok and math.hypot(self.pos_x - sx, self.pos_y - sy) <= 0.02:
                    break
                self.ep_chassis.drive_speed(x=ux * v, y=uy * v, z=0)
                time.sleep(0.05)
            self._stop(0.4)
            return
        if pos_ok:
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if not retreat_clear():
                    break
                if math.hypot(self.pos_x - sx, self.pos_y - sy) <= 0.02:
                    break
                self.ep_chassis.drive_speed(x=-v, y=0, z=0)
                time.sleep(0.05)
        else:
            t_end = time.time() + traveled / v
            while time.time() < t_end and retreat_clear():
                self.ep_chassis.drive_speed(x=-v, y=0, z=0)
                time.sleep(0.05)
        self._stop(0.4)

    # ------------------------------------------------------------------
    # หมุนด้วย absolute yaw
    # ------------------------------------------------------------------
    TURN_TOL_DEG = 2.0       # stop turning inside this error
    TURN_OK_DEG = 4.0        # after settling, an error above this = retry the turn

    def turn_to_absolute_yaw(self, target_yaw, attempts=3):
        """Turn in place to target_yaw and CHECK it got there.

        The old loop stopped after a fixed 4 s at <= 45 deg/s, so a 180 deg U-turn
        (every backtrack) ended ~30 deg short and the robot then drove into a wall.
        Now the time allowed grows with the angle, and the result is checked and
        retried. Returns True when the heading is within TURN_OK_DEG."""
        print(f"--> [Turn] หมุนไป {target_yaw:.1f}° (ปัจจุบัน {self.current_yaw:.1f}°)")
        if abs(wrap180(target_yaw - self.current_yaw)) <= self.TURN_OK_DEG:
            return True                          # already there: no stop, no settle
        self._stop(0.1)
        vmax = self.TURN_MAX_DPS
        if abs(wrap180(target_yaw - self.current_yaw)) > 20.0 and not self._room_to_turn():
            vmax = self.TURN_TIGHT_DPS           # still close to something after making room: slowly
        corner_seen = set()                      # corners that fired during this turn
        for attempt in range(attempts):
            corner_abort = False
            err0 = abs(wrap180(target_yaw - self.current_yaw))
            limit = 1.0 + err0 / (0.4 * vmax)    # generous for the angle at this speed
            start_time = time.time()
            self._drive_start()
            last_z = 0.0
            while (time.time() - start_time) < limit:
                error = wrap180(target_yaw - self.current_yaw)
                if abs(error) < self.TURN_TOL_DEG:
                    break
                if self._wheels_dead():
                    self._stop(0.1)
                    if not self._wheels_not_answering("turning"):
                        return False
                    start_time = time.time()
                    self._drive_start()
                    continue
                l_cm, r_cm = self.ir.latest()
                acc = getattr(self, "_imu_acc", None)
                bump = acc and time.time() - acc[0] < 0.1 and time.time() - start_time > 0.3 and \
                    math.hypot(acc[1], acc[2]) > self.BUMP_G
                if bump or (getattr(self.ir, "mount", "side") == "side" and
                             min(l_cm, r_cm) <= getattr(self, "SIDE_HIT_CM", 5.0) - 1):
                    self._stop(0.2)
                    self._log("EMERGENCY STOP while turning: " + (f"IMU bump {math.hypot(acc[1], acc[2]):.2f} g"
                              if bump else f"side wall {min(l_cm, r_cm):.0f} cm") + " - making room")
                    self._room_to_turn()
                    continue
                hits = self._front_corner_hits()
                if hits:
                    self._stop(0.2)
                    self._log(f"front-{'+'.join(hits)} IR triggered while turning - emergency stop")
                    corner_seen.update(hits)
                    # both corners in one turn: a sideways shift only swaps which corner is
                    # close (last run: left, right, left, "turn failed") - the tightness is
                    # front-back, so back away instead; and finish the turn slowly
                    self._make_corner_room(tuple(sorted(corner_seen)) if len(corner_seen) > 1 else hits)
                    vmax = self.TURN_TIGHT_DPS
                    corner_abort = True
                    break
                # fast in the middle, then a constant-deceleration stop (no overshoot, no crawl)
                left = abs(error) - 1.0 - abs(last_z) * self.TURN_LAG_S
                z_speed = max(12.0, min(vmax, math.sqrt(2.0 * self.TURN_BRAKE_DPS2 * max(0.0, left))))
                last_z = z_speed
                z_speed = z_speed if error > 0 else -z_speed
                self.ep_chassis.drive_speed(x=0, y=0, z=self.z_sign * z_speed)
                time.sleep(0.03)
            if corner_abort:
                continue
            self._stop(0.2)
            final = wrap180(target_yaw - self.current_yaw)
            if abs(final) <= self.TURN_OK_DEG:
                if self.wall_pid is not None and abs(final) > self.FINE_TURN_TOL:
                    self._fine_align(target_yaw)
                return True
            self._log(f"turn off by {final:+.0f} deg - correcting ({attempt + 1}/{attempts})")
        self._log(f"turn failed: still {wrap180(target_yaw - self.current_yaw):+.0f} deg off")
        return False

    def _room_to_turn(self):
        """The chassis corners sweep a circle (~0.2 m from the centre) when it turns in place.
        Last run it turned 180 deg at 150 deg/s pressed into a corner (left Sharp 13 cm, front
        ToF 130 mm) and hit the wall hard. Before a turn: side Sharps, front-corner IR and the
        front ToF (when the gimbal looks ahead) must show room - if not, slide away from what
        is close first. Returns True when there is room now."""
        left_cm, right_cm = self._control_side_ir()
        front_near = bool(self.ir.corners_enabled and (self.ir.near("left") or self.ir.near("right")))
        tof = self.current_tof_dist_mm
        if abs(getattr(self, "_gimbal_yaw_now", 0.0)) < 15 and 60 < tof < self.TURN_CLEAR_TOF_MM:
            front_near = True
        side_l = left_cm is not None and left_cm < self.TURN_CLEAR_SIDE_CM
        side_r = right_cm is not None and right_cm < self.TURN_CLEAR_SIDE_CM
        if not (front_near or side_l or side_r):
            return True
        fmt = lambda v: f"{v:.0f}" if v is not None else "n/a"
        self._log(f"too close to turn (Sharp L {fmt(left_cm)} / R {fmt(right_cm)} cm, front "
                  f"{'near' if front_near else 'clear'}) - making room first")
        if front_near:
            # Never reverse without a rear sensor.  Move sideways only when the Sharp on
            # that side is alive and explicitly reports enough room.
            choices = []
            if self.ir.usable("right") and right_cm is not None and right_cm >= self.TURN_CLEAR_SIDE_CM:
                choices.append((right_cm, 90.0))
            if self.ir.usable("left") and left_cm is not None and left_cm >= self.TURN_CLEAR_SIDE_CM:
                choices.append((left_cm, -90.0))
            if choices:
                _, angle = max(choices)
                self.nudge(angle, 0.04)
            else:
                self._log("front too close and no verified clear side - hold position; no blind reverse")
                return False
        if side_l and not side_r:
            self.nudge(90.0, min(0.08, (self.TURN_CLEAR_SIDE_CM - left_cm) / 100.0 + 0.02))
        elif side_r and not side_l:
            self.nudge(-90.0, min(0.08, (self.TURN_CLEAR_SIDE_CM - right_cm) / 100.0 + 0.02))
        left_cm, right_cm = self._control_side_ir()
        sides_clear = all(v is None or v >= self.TURN_CLEAR_SIDE_CM for v in (left_cm, right_cm))
        return sides_clear and not (
            self.ir.corners_enabled and (self.ir.near("left") or self.ir.near("right")))

    def heading_to_yaw(self, heading_index):
        """0=N, 1=E, 2=S, 3=W เทียบกับ yaw_origin"""
        s = self.right_yaw_sign
        mapping = {0: 0.0, 1: 90.0 * s, 2: 180.0, 3: -90.0 * s}
        return wrap180(self.yaw_origin + mapping[heading_index % 4])

    # ------------------------------------------------------------------
    # Alignment ด้วยกำแพง (ToF บน gimbal)
    # ------------------------------------------------------------------
    def _tof_avg(self, n=3):
        vals = []
        for _ in range(n):
            time.sleep(0.06)
            v = self.current_tof_dist_mm
            if 0 < v < 3000:
                vals.append(v)
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    def _tof_at(self, gimbal_yaw):
        if not self._gimbal_moveto(pitch=0, yaw=gimbal_yaw, yaw_speed=180, what="wall alignment"):
            return None
        time.sleep(0.15)
        return self._tof_avg()

    def measure_wall_angle(self, base_deg):
        """วัดมุมเบี้ยวของกำแพงที่อยู่ทิศ base_deg (0=หน้า, 90=ขวา, -90=ซ้าย) เทียบกับตัวหุ่น
        ยิง ToF ที่ base±phi แล้วใช้ r(a)=D/cos(a-theta) หา theta
        theta>0 = กำแพงหมุนตามเข็มจากทิศ base (หุ่นต้องหมุนตามเข็ม theta เพื่อขนาน/ตั้งฉาก)"""
        phi = self.ALIGN_PHI_DEG
        r_p = self._tof_at(base_deg + phi)
        r_m = self._tof_at(base_deg - phi)
        if r_p is None or r_m is None:
            return None
        if max(r_p, r_m) > self.ALIGN_MAX_MM * 1.8:
            return None
        t = (r_m - r_p) / ((r_m + r_p) * math.tan(math.radians(phi)))
        return math.degrees(math.atan(t))

    def align_to_walls(self, dist, heading):
        """จัดหุ่นให้ตรงกับกำแพงที่ใกล้ที่สุด แล้วปรับ yaw_origin ตามกำแพง (แก้ gyro drift)"""
        cands = (("front", 0), ("right", 90), ("left", -90))
        valid = [(dist[k], base) for k, base in cands if 0 < dist[k] <= self.ALIGN_MAX_MM]
        if not valid:
            print("-> [Align] ไม่มีกำแพงใกล้ ข้าม")
            return False

        _, base = min(valid)
        theta = self.measure_wall_angle(base)
        self.reset_gimbal()

        if theta is None or abs(theta) > self.ALIGN_MAX_DEG:
            print(f"-> [Align] วัดมุมไม่ได้/ผิดปกติ ({theta}) ข้าม")
            return False

        print(f"-> [Align] กำแพง({base:+d}°) เบี้ยว {theta:+.1f}°")
        target = wrap180(self.current_yaw + theta)
        if abs(theta) >= 2.0:
            self.turn_to_absolute_yaw(target)

        expected = self.heading_to_yaw(heading)
        shift = wrap180(target - expected)
        if abs(shift) <= 25.0:
            self.yaw_origin = wrap180(self.yaw_origin + shift)
            print(f"   yaw_origin -> {self.yaw_origin:.1f}° (ปรับ {shift:+.1f}°)")
        return True

    def align_front_distance(self):
        """ถ้ามีกำแพงหน้า ปรับตำแหน่งหน้า-หลังให้ ToF = front_center_mm (ต้องตั้งใน config)"""
        if self.front_center_mm is None:
            return
        front = self._tof_avg()
        if front is None or front > self.ALIGN_MAX_MM:
            return
        t0 = time.time()
        while time.time() - t0 < 2.0:
            err = self.current_tof_dist_mm - self.front_center_mm
            if abs(err) <= 8:
                break
            v = max(min(err / 1000.0 * 1.2, 0.12), -0.12)
            if abs(v) < 0.04:
                v = 0.04 if v > 0 else -0.04
            self.ep_chassis.drive_speed(x=v, y=0, z=0)
            time.sleep(0.05)
        self._stop(0.3)

    # ------------------------------------------------------------------
    # Scan / draw
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # sightings while the gimbal turns (the scan sweeps at 300 deg/s: blurred frames)
    # ------------------------------------------------------------------
    def _gimbal_at(self, t):
        """Gimbal (pitch, yaw) at time t from the 50 Hz angle feed (interpolated), or None."""
        h = list(self.gimbal_hist)
        if not h:
            return None
        if t <= h[0][0]:
            return h[0][1], h[0][2]
        for (t0, p0, y0), (t1, p1, y1) in zip(h, h[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return p0 + k * (p1 - p0), y0 + k * (y1 - y0)
        return h[-1][1], h[-1][2]

    def _motion_at(self, frame_ts):
        """For the camera worker: the frame taken at frame_ts - delay, was the gimbal turning?
        None when (nearly) still, else the smear in pixels and the gimbal angle then."""
        p = self.panel
        if p is None or not self.gimbal_hist or time.time() - self.gimbal_hist[-1][0] > 0.5:
            return None
        t = frame_ts - p.camera_latency_s
        a, b, g = self._gimbal_at(t - 0.03), self._gimbal_at(t + 0.03), self._gimbal_at(t)
        if a is None or b is None or g is None:
            return None
        vp, vy = (b[0] - a[0]) / 0.06, (b[1] - a[1]) / 0.06
        if math.hypot(vp, vy) < self.MOTION_MIN_DPS:
            if getattr(self, "_driving", False):
                return {"blur": (0.0, 0.0), "g": g}      # driving: log what the camera sees on the way
            return None
        size = p.detector.last_frame_size
        px_deg = p.detector.focal_px(size[0]) * math.pi / 180.0 if size else 7.5
        return {"blur": (vy * self.EXPOSURE_S * px_deg, vp * self.EXPOSURE_S * px_deg), "g": g}

    def _start_recording(self, data_dir):
        """Save a camera frame every vision.record_every_s (+ gimbal angle, cell) into
        <data_dir>/frames_<time>/ - the pictures for color_calibrate.py --image /
        color_samples_from_frames.py (tune the colours on real arena light, off the robot)."""
        vis = self.config.get("vision", {}) or {}
        p = self.panel
        if p is None or not vis.get("record_frames", True):
            return
        d = os.path.join(data_dir, "frames_" + datetime.now().strftime("%H%M%S"))
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return
        w = p.worker
        w.record_every_s = float(vis.get("record_every_s", 1.0))

        def meta(ts, _s=self):
            g = _s._gimbal_at(ts - p.camera_latency_s) or ("", "")
            ctx = getattr(_s, "_live_ctx", None)
            return {"pitch": round(g[0], 1) if g[0] != "" else "", "yaw": round(g[1], 1) if g[1] != "" else "",
                    "cell": f"{ctx[0][0]} {ctx[0][1]}" if ctx else "", "heading": ctx[4] if ctx else ""}
        w.meta_fn = meta
        w.record_dir = d
        p.last_frames_dir = d
        self._log(f"recording a frame every {w.record_every_s:.1f} s -> {d}")

    def _hook_motion(self):
        p = self.panel
        if p is not None and getattr(p, "worker", None) is not None and p.worker.motion_fn is None:
            p.worker.motion_fn = self._motion_at

    def _check_motion_sightings(self, pos, heading, t0, still_yaws, limit=None):
        """Cards seen in blurred frames while the gimbal turned since t0 (a colour + a
        direction): the ones no mapped card explains get a still look straight at them -
        mapped (and shot, when in reach) only if that look confirms it. Directions no still
        look covered come first; one seen only once inside a still look's view is skipped
        (that look saw it sharp and said no)."""
        p = self.panel
        limit = self.MOTION_CHECKS if limit is None else limit
        if p is None or limit <= 0 or not p.checkpoint():
            return
        base = {0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0)
        lat = p.camera_latency_s
        pts = []
        for ts, g, seen in list(p.worker.motion_log):
            if ts - lat < t0:
                continue
            for x in seen:
                e = g[0] + x["elevation"]
                # blur-only colour (no card outline) above the camera's level: the room over the
                # walls (last run: 65 yellow "nothing there" looks, the camera tilting UP at them)
                if x.get("kind") is None and e > self.MOTION_MAX_ELEV_DEG:
                    continue
                pts.append((x["color"], (base + g[1] + x["bearing"]) % 360, e, x.get("kind")))
        if not pts:
            return
        clusters = []                                  # [color, [abs yaws], [elevs], kinds]
        for c, a, e, k in pts:
            for cl in clusters:
                if cl[0] == c and abs(wrap180(a - cl[1][0])) < 7.0:
                    cl[1].append(a); cl[2].append(e); cl[3].add(k)
                    break
            else:
                clusters.append([c, [a], [e], {k}])
        tile = p.map.tile
        cx, cy = (pos[0] + 0.5) * tile, (pos[1] + 0.5) * tile
        # a card already on the map explains the sighting only when nothing is left to do for
        # it from here (hit / not selected / not in reach): a sweep has no still look of its own,
        # so a card still to confirm or to shoot gets its look
        def settled(t):
            if not t.get("confirmed"):
                return False
            if t["kind"] not in p.selected or self._card_done(t):
                return True
            return False                 # selected and still up: a fresh view must try to fire now
        mapped = [(t["color"], math.degrees(math.atan2(t["x_m"] - cx, t["y_m"] - cy)) % 360)
                  for t in p.map.target_list(False) if settled(t)]
        half = self.SEE_HALF_FOV_DEG - 4.0
        todo = []
        for c, yaws, elevs, kinds in clusters:
            a = sorted(yaws)[len(yaws) // 2]
            e = sorted(elevs)[len(elevs) // 2]
            if any(mc == c and abs(wrap180(ma - a)) < 10.0 for mc, ma in mapped):
                continue                               # a card already on the map
            if any(t["color"] == c and
                   abs(wrap180(math.degrees(math.atan2(t["x_m"] - cx, t["y_m"] - cy)) - a)) < 15.0 and
                   (t["kind"] not in p.selected or self._card_done(t))
                   for t in p.map.target_list(False)):
                continue                               # already hit or deliberately not selected
            if not any(kinds - {None}) and len(yaws) < self.MOTION_MIN_FRAMES:
                continue                               # colour only, in too few frames
            wx, wy = p.map.wall_point(pos, a)
            if any(ec == c and math.hypot(ex - wx, ey - wy) < 0.35 for ec, ex, ey in self._motion_empty):
                continue                               # looked there before (from any block): nothing
            if any(ch_ == tuple(pos) and cc == c and abs(wrap180(ca - a)) < 8.0
                   for ch_, cc, ca in self._motion_checked):
                continue
            covered = any(abs(wrap180(a - (base + y))) < half for y in still_yaws)
            if covered:
                # a still look saw that direction sharp and found nothing: trust it (last run:
                # 29 such checks, 1 useful - ~25 s lost)
                continue
            todo.append((covered, -len(yaws), c, a, e, kinds))
        todo.sort()
        for covered, n, c, a, e, kinds in todo[:limit]:
            if not p.checkpoint():
                return
            self._motion_checked.append((tuple(pos), c, a))
            rel = wrap180(a - base)
            # Elevation is relative to the image centre.  The map sweep itself looks
            # slightly down, so preserve that base pitch when stopping on the sighting.
            pit = max(-20.0, min(20.0, self.SCAN_PITCH_DEG + e))
            if not self._gimbal_moveto(pitch=pit, yaw=rel, pitch_speed=min(240, self.GIMBAL_DPS), yaw_speed=self.GIMBAL_DPS,
                                       what="check a sighting from a turn"):
                continue
            before = {t["id"] for t in p.map.target_list(False)}
            self._draw_live_with_gimbal(rel)
            self._look_for_targets(rel, pitch=pit, settle_ts=time.time())
            new = [t for t in p.map.target_list(False) if t["id"] not in before and t["color"] == c]
            if not new:
                self._motion_empty.append((c,) + tuple(p.map.wall_point(pos, a)))
            p.log(f"{c} card glimpsed while the gimbal was turning ({-n} blurred frame(s), "
                  f"{'inside' if covered else 'outside'} the still looks) - looked again: "
                  + (f"confirmed {new[0]['kind']}" if new else "nothing new there"))
        if todo:
            self._gimbal_moveto(pitch=self.SCAN_PITCH_DEG, yaw=getattr(self, "_gimbal_yaw_now", 0.0), pitch_speed=min(240, self.GIMBAL_DPS),
                                yaw_speed=self.GIMBAL_DPS, what="restore map-scan pitch")

    def _sweep_scan(self, known):
        """One continuous gimbal sweep (no stops) over the directions still unknown: the ToF is
        sampled all the way (each sample matched to the gimbal angle when it was measured) and
        the camera watches every frame (blur-tolerant). Wall distances = median of the samples
        within SWEEP_WIN_DEG of each direction. The camera then stops only where it saw a
        card colour (still look -> mapped / shot). ~1-1.5 s instead of 3-4 s per block."""
        distances = {"front": 0, "right": 0, "back": 0, "left": 0}
        distances.update(known)
        seq = [(lbl, yaw) for lbl, yaw in (("left", -90), ("front", 0), ("right", 90), ("back", 180))
               if lbl not in known]
        if not seq:
            return distances
        g_now = getattr(self, "_gimbal_yaw_now", 0.0)
        if abs(seq[-1][1] - g_now) < abs(seq[0][1] - g_now):
            seq.reverse()
        self._hook_motion()
        a0, a1 = seq[0][1], seq[-1][1]
        pad = self.SWEEP_WIN_DEG
        start = a0 - pad if a1 >= a0 else a0 + pad
        end = a1 + pad if a1 >= a0 else a1 - pad
        start, end = max(-240.0, min(240.0, start)), max(-240.0, min(240.0, end))
        if not self._gimbal_moveto(pitch=self.SCAN_PITCH_DEG, yaw=start, pitch_speed=min(240, self.GIMBAL_DPS),
                                   yaw_speed=self.GIMBAL_DPS, what="sweep start"):
            return self._stop_scan(known)
        t_scan = time.time()
        samples, last_ts = [], self._tof_last_ts
        direction = 1.0 if end >= start else -1.0
        t_end = time.time() + abs(end - start) / self.SWEEP_DPS + 1.5
        try:
            while time.time() < t_end:
                if self.panel is not None and self.panel.abort.is_set():
                    break
                y_now = self.gimbal_hist[-1][2] if self.gimbal_hist else start
                if (end - y_now) * direction <= 0.5:
                    break
                # slow down over the last few degrees: stop on the end, not past it
                v = min(self.SWEEP_DPS, max(20.0, math.sqrt(2.0 * self.GIMBAL_BRAKE_DPS2 * abs(end - y_now))))
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=direction * v)
                if self._tof_last_ts != last_ts:
                    last_ts = self._tof_last_ts
                    samples.append((last_ts, self.current_tof_dist_mm))
                time.sleep(0.01)
        finally:
            try:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
        # each ToF sample -> the gimbal yaw when it was measured
        pts = []
        for ts, mm in samples:
            g = self._gimbal_at(ts - self.TOF_LATENCY_S)
            if g is not None and 60 < mm < 8000:
                pts.append((g[1], mm))
        ctx = getattr(self, "_live_ctx", None)
        base = {0: 0, 1: 90, 2: 180, 3: 270}.get(ctx[4], 0) if ctx else 0
        for label, yaw in seq:
            # the ToF gives a reading every ~7-15 deg at sweep speed: the (up to 3) samples
            # nearest the direction (a flat wall 10 deg off reads only 1.5 % long)
            near = sorted((abs(y - yaw), mm) for y, mm in pts if abs(y - yaw) <= self.SWEEP_WIN_DEG)[:3]
            if near:
                vals = sorted(mm for _, mm in near)
                distances[label] = vals[len(vals) // 2]
            else:
                # the sweep missed it (ToF gap): one still reading there
                if self._gimbal_moveto(pitch=self.SCAN_PITCH_DEG, yaw=yaw, pitch_speed=min(240, self.GIMBAL_DPS),
                                       yaw_speed=self.GIMBAL_DPS, what=f"scan {label}"):
                    distances[label] = self._tof_fresh(3)
            if ctx:
                self._looks.append((tuple(ctx[0]), (base + yaw) % 360))   # that wall face was in view
        self._draw_live_with_gimbal(getattr(self, "_gimbal_yaw_now", 0.0))
        # Motion frames are cheap but can miss a card through blur or a temporary colour
        # threshold failure. Add settled fallback looks while enough time remains.
        if ctx is not None:
            # Map topology has priority. The mission loop records these sweep distances as
            # open/wall edges first, then performs the focused target check before moving on.
            self._deferred_motion_scan = (ctx[0], ctx[4], t_scan, [], self.SWEEP_CHECKS)
            remaining = self.panel.remaining() if self.panel is not None and self.panel.round_t0 else None
            fallback_n = min(self.SWEEP_FALLBACK_LOOKS, len(seq))
            if (not self.hurry and fallback_n and
                    (remaining is None or remaining >= self.SWEEP_FALLBACK_MIN_REMAINING_S)):
                for label, yaw in seq[:fallback_n]:
                    if self.panel is not None and (
                            not self.panel.checkpoint() or
                            (self.panel.round_t0 and self.panel.remaining() <= self.END_RESERVE_S)):
                        break
                    if not self._gimbal_moveto(pitch=self.SCAN_PITCH_DEG, yaw=yaw,
                                               pitch_speed=min(240, self.GIMBAL_DPS),
                                               yaw_speed=self.GIMBAL_DPS,
                                               what=f"fallback still look {label}"):
                        continue
                    settled_ts = time.time()
                    self._draw_live_with_gimbal(yaw)
                    self._look_for_targets(yaw, pitch=self.SCAN_PITCH_DEG, settle_ts=settled_ts)
        return distances

    def _stop_scan(self, known):
        return self.scan_surroundings_with_gimbal(known=known, sweep=False)

    def scan_surroundings_with_gimbal(self, known=None, sweep=None):
        """ToF + camera look in the 4 directions (relative to the chassis). known =
        {label: mm} for directions already known (the way the robot came in): not looked
        at again. The gimbal sweeps once from one side to the other (no back-and-forth)."""
        known = known or {}
        if (self.SWEEP_SCAN if sweep is None else sweep) and self.gimbal_hist and \
                time.time() - self.gimbal_hist[-1][0] < 0.3:
            return self._sweep_scan(known)
        distances = {"front": 0, "right": 0, "back": 0, "left": 0}
        distances.update(known)

        scan_sequence = [(lbl, yaw) for lbl, yaw in (("left", -90), ("front", 0), ("right", 90), ("back", 180))
                         if lbl not in known]
        # start from the end nearest to where the gimbal points now
        g_now = getattr(self, "_gimbal_yaw_now", 0.0)
        if scan_sequence and abs(scan_sequence[-1][1] - g_now) < abs(scan_sequence[0][1] - g_now):
            scan_sequence.reverse()

        self._hook_motion()
        ctx = getattr(self, "_live_ctx", None)
        t_scan = time.time()
        still_yaws = []
        for label, yaw in scan_sequence:
            if not self._gimbal_moveto(pitch=self.SCAN_PITCH_DEG, yaw=yaw, pitch_speed=min(240, self.GIMBAL_DPS),
                                       yaw_speed=self.GIMBAL_DPS, what=f"scan {label}"):
                distances[label] = 0
                continue
            still_yaws.append(yaw)
            settled_ts = time.time()
            look_started = time.time()
            distances[label] = self._tof_fresh(3)     # readings taken after the gimbal settled
            if label == "front":
                self._learn_wall_rise(distances[label])
            self._draw_live_with_gimbal(yaw)
            found = self._look_for_targets(yaw, pitch=self.SCAN_PITCH_DEG, settle_ts=settled_ts)
            if self.panel is not None:
                result = "TARGET" if found else ("WALL" if distances[label] <= self.wall_mm else "OPEN")
                self.panel.log(f"LOOK_RESULT cell={tuple(ctx[0]) if ctx else '?'} dir={label} "
                               f"tof={distances[label]:.0f}mm result={result} "
                               f"elapsed={time.time() - look_started:.2f}s")

        # cards seen only in blurred frames while the gimbal swung between the looks
        ctx = getattr(self, "_live_ctx", None)
        if ctx is not None:
            self._check_motion_sightings(ctx[0], ctx[4], t_scan, still_yaws)
        # no swing back to the centre: the next move points the gimbal where it drives anyway
        if not self.strafe:
            self.reset_gimbal()
            self._draw_live_with_gimbal(0)
        return distances

    def _look_for_targets(self, gimbal_relative_yaw, pitch=0.0, close=False, settle_ts=None):
        """Camera check while the gimbal is settled: put every valid target on
        the map, then aim + fire at designated ones that are within range.
        Returns the detections found."""
        ctx = getattr(self, "_live_ctx", None)
        if self.panel is None or ctx is None or not self.panel.checkpoint():
            return []
        pos, heading = ctx[0], ctx[4]
        abs_deg = ({0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0) + gimbal_relative_yaw) % 360
        if hasattr(self.panel, "log"):
            self.panel.log(f"LOOK cell={tuple(pos)} chassis={heading * 90}deg "
                           f"gimbal_rel={gimbal_relative_yaw:+.0f}deg "
                           f"gimbal_abs={abs_deg:.0f}deg pitch={pitch:+.0f}deg "
                           f"reason={'TARGET_VERIFY' if close else 'MAP_SCAN'}")
        self._looks.append((tuple(pos), abs_deg))          # wall faces in view count as seen
        # looking down the ToF hits the floor: no wall check then
        tof = self.current_tof_dist_mm if abs(pitch) < 1 else None
        self.panel.detector.gimbal_pitch_deg = pitch   # until the gimbal angle feed catches up
        found = self.panel.observe_targets(
            pos, abs_deg, tof_mm=tof,
            settle_ts=settle_ts if settle_ts is not None else time.time(),
        )
        found += self._check_glimpses(pos, heading, gimbal_relative_yaw, pitch)
        # maybe-cards: from the regular scan only a HINT (this block gets a close look); only
        # what the close look itself sees can send the robot to look back from the next block
        for gq in getattr(self.panel, "last_guesses", []):
            if not close:
                gq.update(gimbal_yaw=gimbal_relative_yaw, pitch=pitch, hint=True)
                self._guesses.append(gq)
                continue
            wx, wy = self.panel.map.wall_point(pos, gq["abs_deg"])
            if any(c == gq["color"] and math.hypot(fx - wx, fy - wy) < 0.4 for c, fx, fy in self._not_cards):
                continue                     # this colour here was already checked: not a card
            gq.update(gimbal_yaw=gimbal_relative_yaw, pitch=pitch)
            self._guesses.append(gq)
        if self.shooter is None:
            return
        if self.shoot_order == "found":       # several in view: the one found first goes first
            first = lambda d_: (self.panel.map.targets.get(getattr(d_, "target_id", None)) or {}).get("first_seen", 9e18)
            found = sorted(found, key=first)
        aimed = False
        for det in found:
            tid = getattr(det, "target_id", None)
            in_range = (det.distance_m is not None and
                        det.distance_m <= self.panel.detector.max_shoot_m * self.fire_range_margin)
            if not (in_range and tid and self.panel.should_shoot(tid)):   # selected kinds, shape sure
                continue
            target = self.panel.map._target_view(self.panel.map.targets[tid])
            tc = tuple(target["cell"])
            if max(abs(tc[0] - pos[0]), abs(tc[1] - pos[1])) > self.fire_max_cells:
                self.panel.log(f"{tid} visible but more than {self.fire_max_cells} cell away - move closer")
                continue
            twin = self.panel.map.hit_twin(tid)
            if twin:
                self.panel.log(f"{tid} is {twin} again (already hit) - merged, not shooting twice")
                continue
            attempt_key = (tid, tuple(pos))
            if attempt_key in self._tried_from or \
                    self._aim_attempts.get(attempt_key, 0) >= self.AIM_ATTEMPTS_PER_CELL:
                continue
            # This is a fresh, full-card camera detection with a measured distance inside the
            # assignment's two-tile limit.  Do not let the map's cell/wall classification veto
            # it: cards hang on walls, so the target point can legitimately be assigned to the
            # cell across that wall (the previous gate rejected a visible green card at 0.59 m).
            # Map reach remains useful for map-only route planning; live line-of-sight wins here.
            if self._wait_for_better_spot(pos, tid):
                continue
            kind = self.panel.map.targets[tid]["kind"]   # the map's (voted) kind
            if aimed:
                # the last aim turned the gimbal: back to where this look saw the cards
                self._gimbal_moveto(pitch=pitch, yaw=gimbal_relative_yaw, pitch_speed=min(240, self.GIMBAL_DPS),
                                    yaw_speed=self.GIMBAL_DPS, what="return to target scan")
            aimed = True
            self._aim_attempts[attempt_key] = self._aim_attempts.get(attempt_key, 0) + 1
            expect = (det.bearing_deg, det.elevation_deg)      # this card, not another of its colour
            if det.distance_m < self.MIN_SHOOT_M and self.BACK_OFF_M > 0:
                # too close to hit reliably (0.21 m: 5 misses): slide away from it a little
                # inside the block, shoot, slide back
                away = wrap180(gimbal_relative_yaw + det.bearing_deg + 180.0)
                back = min(self.BACK_OFF_M, self.MIN_SHOOT_M + 0.03 - det.distance_m)
                self.panel.log(f"{tid} only {det.distance_m:.2f} m away - backing off {back * 100:.0f} cm to shoot")
                moved = self.nudge(away, back)
                success = self.shooter.engage(kind, tid, expect)
                self.nudge(wrap180(away + 180.0), moved)
                target_now = self.panel.map.targets.get(tid)
                done = success or target_now is None or self._card_done(
                    self.panel.map._target_view(target_now))
                if done or self._aim_attempts[attempt_key] >= self.AIM_ATTEMPTS_PER_CELL:
                    self._tried_from.add(attempt_key)
                if tuple(target["cell"]) == tuple(pos):
                    self._tried_inside.add((tid, tuple(pos)))
                continue
            success = self.shooter.engage(kind, tid, expect)
            target_now = self.panel.map.targets.get(tid)
            done = success or target_now is None or self._card_done(
                self.panel.map._target_view(target_now))
            if done or self._aim_attempts[attempt_key] >= self.AIM_ATTEMPTS_PER_CELL:
                self._tried_from.add(attempt_key)
            if tuple(target["cell"]) == tuple(pos):
                self._tried_inside.add((tid, tuple(pos)))
        # back to the scan direction (engage may have moved the gimbal)
        if found:
            self._gimbal_moveto(pitch=pitch, yaw=gimbal_relative_yaw, yaw_speed=180,
                                what="restore scan direction")
        return found

    def _wait_for_better_spot(self, pos, tid):
        """True = do not shoot tid from here now: the shot is poor (far / slanted) and a block
        the exploration will still visit gives a good one. Near the end of the time nothing
        waits (the robot may never get there)."""
        p = self.panel
        # Assignment round 1 is a find-and-shoot pass. Once a selected target is
        # confirmed and within range, fire from the current safe position instead
        # of delaying for a theoretically better angle in a later cell. Round 2
        # remains route-optimised from the saved map.
        if p.round_no == 1:
            return False
        d, view = p.map.shot_quality(pos, tid)
        if d is None or (d <= self.GOOD_SHOT_M and (view is None or view <= self.GOOD_SHOT_VIEW)):
            return False
        if p.round_t0 and p.remaining() < 120:
            return False
        ctx = getattr(self, "_live_ctx", None)
        visited = ctx[1] if ctx else ()
        spot = p.map.better_spot(pos, tid, visited, self.GOOD_SHOT_M, self.GOOD_SHOT_VIEW, self.reach_cells)
        if spot is None:
            return False
        if (tid, pos, "better") not in self._reach_noted:
            self._reach_noted.add((tid, pos, "better"))
            p.log(f"{tid}: poor shot from {pos} ({d:.2f} m, {view if view is not None else 90:.0f} deg off "
                  f"face-on) - will shoot it from {spot} (not visited yet)")
        return True

    def _check_glimpses(self, pos, heading, g_yaw, pitch):
        """Point at brief or partial card candidates and confirm with fresh still frames."""
        p = self.panel
        out = []
        pending = list(getattr(p, "last_single", []))
        # A clipped/odd blob is just as useful for directing the camera as a one-frame full
        # card. Previously these guesses were deferred until a later cell sweep, which is why
        # visible targets stayed CHECKING and were revisited repeatedly.
        for g in list(getattr(p, "last_guesses", [])):
            pending.append({"kind": None, "color": g["color"], "bearing": g["bearing"],
                            "elevation": g["elevation"], "candidate": True})
        glimpses = []
        chassis_abs = {0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0)
        for s1 in pending:
            absolute = (chassis_abs + g_yaw + s1["bearing"]) % 360
            # A one-frame view of a card already mapped/verified (or dry-locked/shot) is not a
            # new candidate. Keep scanning; do not swing the gimbal back to the same target.
            dist = s1.get("distance")
            if s1.get("kind") and dist:
                tx, ty = p.map.project(pos, absolute, dist)
                known = p.map._find_card(s1["kind"], tx, ty, dist_m=dist)
                if known and known in p.map.targets:
                    kt = p.map.targets[known]
                    if kt.get("verified") or self._card_done(p.map._target_view(kt)):
                        continue
            key = (tuple(pos), s1["color"], int((absolute + 10) // 20) % 18)
            if key in self._candidate_checked:
                continue
            self._candidate_checked.add(key)
            glimpses.append(s1)
            # More than one card can be visible from a cell.  The old unconditional break
            # discarded every candidate after the first one (commonly a selected red square
            # beside another coloured card).  Keep this bounded so detection remains fast.
            if len(glimpses) >= max(1, self.MOTION_CHECKS):
                break
        for s1 in glimpses:
            if not p.checkpoint():
                break
            yaw = wrap180(g_yaw + s1["bearing"])
            # A very close card otherwise sits behind the barrel/bottom crop.
            # Point farther down so the whole card moves into the image before
            # shape confirmation; -25 deg is the RoboMaster gimbal's lower limit.
            pit = max(self.CLOSE_PITCH_MIN_DEG, min(20.0, pitch + s1["elevation"]))
            if not self._gimbal_moveto(pitch=pit, yaw=yaw, pitch_speed=min(240, self.GIMBAL_DPS),
                                       yaw_speed=self.GIMBAL_DPS, what="verify glimpse"):
                continue
            time.sleep(0.1)
            p.detector.gimbal_pitch_deg = pit
            abs_deg = ({0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0) + yaw) % 360
            # Spend extra frames only where the normal scan actually saw a candidate.  Two
            # agreeing still frames out of five recover short blur/exposure drop-outs without
            # slowing every map direction or weakening the final firing lock.
            frames = int(getattr(self, "CANDIDATE_CONFIRM_FRAMES", 5))
            hits = int(getattr(self, "CANDIDATE_CONFIRM_MIN_HITS", 2))
            got = p.observe_targets(pos, abs_deg, tof_mm=self.current_tof_dist_mm if abs(pit) < 1 else None,
                                    settle_ts=time.time(), frames=frames, min_hits=hits)

            def intended(ds):
                return [d for d in ds if (s1.get("kind") and d.kind == s1["kind"]) or
                        d.color == s1["color"]]

            hit = intended(got)
            # A sighting made while the gimbal/chassis was moving is delayed by the video
            # pipeline.  Its estimated yaw is often a few degrees stale; at close range that
            # is enough to put a 7 cm square outside the next frame.  Search only the two
            # neighbouring angles, only after a real colour/card candidate, and stop as soon
            # as the intended colour is confirmed.  This replaces expensive full re-sweeps.
            if not hit and self.CANDIDATE_RETRY_YAW_DEG > 0:
                side_frames = max(2, self.CANDIDATE_RETRY_FRAMES)
                side_hits = min(hits, max(1, side_frames - 1))
                for off in (-self.CANDIDATE_RETRY_YAW_DEG, self.CANDIDATE_RETRY_YAW_DEG):
                    if not p.checkpoint():
                        break
                    retry_yaw = wrap180(yaw + off)
                    if not self._gimbal_moveto(pitch=pit, yaw=retry_yaw,
                                               pitch_speed=min(240, self.GIMBAL_DPS),
                                               yaw_speed=self.GIMBAL_DPS,
                                               what="recover moving target direction"):
                        continue
                    retry_abs = (chassis_abs + retry_yaw) % 360
                    extra = p.observe_targets(
                        pos, retry_abs,
                        tof_mm=self.current_tof_dist_mm if abs(pit) < 1 else None,
                        settle_ts=time.time(), frames=side_frames, min_hits=side_hits,
                    )
                    got += extra
                    hit = intended(extra)
                    if hit:
                        break
            label = s1.get("kind") or f"{s1['color']} candidate"
            p.log(f"{label} seen briefly - focused check: {'confirmed' if hit else 'nothing there'}")
            out += got
        if glimpses:
            # back to where this look was pointing
            self._gimbal_moveto(pitch=pitch, yaw=g_yaw, pitch_speed=min(240, self.GIMBAL_DPS),
                                yaw_speed=self.GIMBAL_DPS, what="restore glimpse scan")
            p.detector.gimbal_pitch_deg = pitch
        p.last_single = []
        # Every candidate above already received a focused five-frame still check. Do not
        # enqueue the same partial blob for verify_sweep/_resolve_guesses again; that was the
        # source of repeated camera turns and long runs stuck on CHECKING.
        p.last_guesses = []
        return out

    def verify_sweep(self, pos):
        """Something was seen in (or next to) this block: aim the camera down and turn it all
        the way round to make sure it is a card. Cards hang lower than the camera, so
        looking down puts close ones in the middle of the picture. A sighting that this
        close look cannot find again is dropped from the map (a false detection)."""
        p = self.panel
        if p is None:
            return
        ids = [tid for tid in p.map.to_verify(pos)
               if tid in p.map.targets and
               not self._card_done(p.map._target_view(p.map.targets[tid]))]
        # a block with a card in it gets one full close look even when that card is sure:
        # another card may hang on the opposite wall, and cards on the side walls are only
        # seen face-on (true shape) from inside the block. A maybe-card (half in the picture)
        # seen from here also earns one.
        fresh = tuple(pos) not in self._swept_cells
        own = [tid for tid in p.map.cards_in(pos)
               if tid in p.map.targets and
               not p.map.targets[tid].get("verified") and
               not self._card_done(p.map._target_view(p.map.targets[tid]))] if fresh else []
        maybe = [g for g in self._guesses if g["cell"] == tuple(pos)] if fresh else []
        self._guesses = [g for g in self._guesses if g["cell"] != tuple(pos)]
        if not ids and not own and not maybe:
            return []
        if getattr(self, "hurry", False) and not ids and not maybe:
            return []                  # behind schedule: a sure card here needs no close look
        self._swept_cells.add(tuple(pos))
        if not ids and not own:
            own = [f"maybe {g['color']} {g['shape']}" for g in maybe[:3]]
        before = {tid: p.map.targets[tid]["n"] for tid in ids if tid in p.map.targets}
        known_before = set(p.map.targets)
        p.log(f"close look at {pos}: " + ", ".join(ids or own))
        pitch = self.verify_pitch_deg
        # the camera sees 96 deg: one look per WALL of this block (an open side holds no card)
        ctx = getattr(self, "_live_ctx", None)
        heading = ctx[4] if ctx else 0
        looks = []
        for g in (-90, 0, 90, 180):
            d = (heading + {-90: 3, 0: 0, 90: 1, 180: 2}[g]) % 4
            dx, dy = self.MOVES4[d]
            nb = (pos[0] + dx, pos[1] + dy)
            e = frozenset((tuple(pos), nb))
            if 0 <= nb[0] < p.map.nx and 0 <= nb[1] < p.map.ny and \
                    (e in p.map.open_edges or e in p.map.traversed):
                continue
            looks.append(g)
        lab = {-90: "left", 0: "front", 90: "right", 180: "back"}
        dists = self._last_scan[1] if getattr(self, "_last_scan", (None,))[0] == tuple(pos) else {}
        p.detector.close_mode = True
        try:
            for g in looks or (-90, 0, 90, 180):
                if not p.checkpoint() or (p.round_t0 and p.remaining() <= 0):
                    break
                # Aim at the expected card centre on THIS wall. The old scaled -10 deg look
                # was only -15 deg at 20 cm, leaving a low close card behind the barrel.
                mm = dists.get(lab[g])
                pit = pitch
                if mm and 60 < mm < 2000:
                    cam_h = float((self.config.get("vision", {}) or {}).get("camera_height_m", 0.25))
                    pit = math.degrees(math.atan2(self.card_aim_height_m - cam_h, mm / 1000.0))
                    pit = max(self.CLOSE_PITCH_MIN_DEG, min(-3.0, pit))
                if not self._gimbal_moveto(pitch=pit, yaw=g, pitch_speed=120,
                                           yaw_speed=180, what="close-look sweep"):
                    continue
                # where that wall's top is (learned wall height above the camera, ToF distance)
                p.detector.close_wall_top_deg = math.degrees(math.atan2(self.wall_rise_m, mm / 1000.0)) \
                    if mm and 60 < mm < 1500 else None
                settled_ts = time.time()
                self._draw_live_with_gimbal(g)
                self._look_for_targets(g, pitch=pit, close=True, settle_ts=settled_ts)
        finally:
            p.detector.close_mode = False
            p.detector.close_wall_top_deg = None
        # a card in the NEXT block (seen through an open side) is not on a wall of this block:
        # the wall looks above never face it. Aim straight at it; only a card the camera really
        # looked at may be dropped for "not there"
        looked = set()
        cx, cy = (pos[0] + 0.5) * p.map.tile, (pos[1] + 0.5) * p.map.tile
        for tid, n0 in before.items():
            t = p.map.targets.get(tid)
            if t is None or not t["sw"]:
                continue
            if tuple(p.map._target_view(t)["cell"]) == tuple(pos):
                looked.add(tid)                  # on a wall of this block: the wall looks covered it
                continue
            if t["n"] > n0 or not p.checkpoint() or (p.round_t0 and p.remaining() <= 0):
                continue
            tx, ty = t["sx"] / t["sw"], t["sy"] / t["sw"]
            a = math.degrees(math.atan2(tx - cx, ty - cy))
            rng_ = math.hypot(tx - cx, ty - cy)
            pit = max(-15.0, min(0.0, -math.degrees(math.atan2(0.12, max(rng_, 0.2)))))
            gy = wrap180(a - heading * 90)       # relative to where the chassis faces
            p.log(f"{tid} is in the next block - looking straight at it")
            if not self._gimbal_moveto(pitch=pit, yaw=gy, pitch_speed=120,
                                       yaw_speed=180, what="look at card next door"):
                continue
            settled_ts = time.time()
            self._draw_live_with_gimbal(gy)
            self._look_for_targets(gy, pitch=pit, settle_ts=settled_ts)
            looked.add(tid)
        # maybe-cards in this block (half in the picture, odd outline): point the camera right
        # at each one so the whole card is in view, and look again
        unresolved = self._resolve_guesses(pos)
        self._gimbal_moveto(pitch=0, yaw=0, pitch_speed=120, yaw_speed=180,
                            what="finish close-look sweep")
        p.detector.gimbal_pitch_deg = 0.0
        # a far first sighting can be ~0.5 m off: a card of the same kind found by the close
        # look near it is the same card, seen properly now -> merge into it
        new_ids = [tid for tid in p.map.targets if tid not in before and tid not in known_before]
        for tid, n0 in before.items():
            t = p.map.targets.get(tid)
            if t is None or t["n"] > n0:
                continue
            tx, ty = t["sx"] / t["sw"], t["sy"] / t["sw"]
            for nid in list(new_ids):
                nt = p.map.targets.get(nid)
                if nt is None or nt["kind"] != t["kind"]:
                    continue
                if math.hypot(nt["sx"] / nt["sw"] - tx, nt["sy"] / nt["sw"] - ty) <= 1.0:
                    with p.map.lock:   # the close look is the better estimate: it replaces the far one
                        t["sx"], t["sy"], t["sw"] = nt["sx"], nt["sy"], nt["sw"]
                        t["n"] += nt["n"]
                        t["views"].update(nt["views"])
                        t["shot"] = t["shot"] or nt["shot"]
                    p.map.remove_target(nid)
                    new_ids.remove(nid)
                    break
        for nid in new_ids:           # first seen during this close look: already looked at closely
            nt = p.map.targets.get(nid)
            if nt is not None:
                nt["swept"] = True
        for tid, n0 in before.items():
            t = p.map.targets.get(tid)
            if t is None:
                continue
            if tid not in looked and t["n"] <= n0:
                continue                         # never in view: a later close look checks it
            t["swept"] = True
            if t["n"] > n0:
                t["verified"] = True
                need = p.map.required_observations(t["shape"])
                # seen again by a separate, closer look = two independent sightings: enough even
                # for a circle / square (last run a red circle stayed "verified again" at 2 of 3
                # sightings and was never shot)
                t["confirmed"] = t.get("confirmed") or t["n"] >= min(need, 2)
                p.log(("confirmed " if t["confirmed"] else "verified again ") + tid + " (close look)")
            elif not t.get("confirmed") and not t["shot"] and tid in looked:
                p.map.remove_target(tid)
                p.log(f"dropped {tid}: not there on the close look")
        return unresolved

    def _resolve_guesses(self, pos):
        """Maybe-cards seen during this block's close look: re-aim straight at each (whole card
        in view) and look again. Returns the ones still unclear: [{color, shape, x_m, y_m,
        abs_deg}] - the robot then looks back at them from the next block."""
        p = self.panel
        heading_deg = {0: 0, 1: 90, 2: 180, 3: 270}.get(self._live_ctx[4], 0)
        todo, out = [], []
        for g in [g for g in self._guesses if g["cell"] == tuple(pos) and not g.get("hint")]:
            if any(o["color"] == g["color"] and abs(wrap180(o["abs_deg"] - g["abs_deg"])) < 20 for o in todo):
                continue                         # the same blob from two sweep directions
            wx, wy = p.map.wall_point(pos, g["abs_deg"])
            g.update(x_m=wx, y_m=wy)
            todo.append(g)
        self._guesses = [g for g in self._guesses if g["cell"] != tuple(pos)]
        for g in todo:
            if p.map.card_of_color_near(g["color"], g["x_m"], g["y_m"]):
                continue                         # a card found by the sweep is this blob
            if not p.checkpoint():
                break
            yaw = wrap180(g["gimbal_yaw"] + g["bearing"])
            pitch = max(-20.0, min(20.0, g["pitch"] + g["elevation"]))
            p.log(f"maybe a {g['color']} {g['shape']} ({g['why']}) - aiming right at it")
            if not self._gimbal_moveto(pitch=pitch, yaw=yaw, pitch_speed=120,
                                       yaw_speed=180, what="resolve possible card"):
                continue
            settled_ts = time.time()
            self._draw_live_with_gimbal(yaw)
            p.detector.close_mode = True
            dists = self._last_scan[1] if getattr(self, "_last_scan", (None,))[0] == tuple(pos) else {}
            lab = {-90: "left", 0: "front", 90: "right", 180: "back"}
            mm = dists.get(lab[min(lab, key=lambda a: abs(wrap180(yaw - a)))])
            p.detector.close_wall_top_deg = math.degrees(math.atan2(self.wall_rise_m, mm / 1000.0)) \
                if mm and 60 < mm < 1500 else None
            try:
                self._look_for_targets(yaw, pitch=pitch, close=True, settle_ts=settled_ts)
            finally:
                p.detector.close_mode = False
                p.detector.close_wall_top_deg = None
            self._guesses = [x for x in self._guesses if x["cell"] != tuple(pos)]
            if p.map.card_of_color_near(g["color"], g["x_m"], g["y_m"], 0.45):
                p.log(f"... it is a card: {p.map.card_of_color_near(g['color'], g['x_m'], g['y_m'], 0.45)}")
                continue
            out.append({k: g[k] for k in ("color", "shape", "x_m", "y_m", "abs_deg")})
        return out

    def _look_back(self, pos, heading, suspects, scan_cache, driven, blocked, visited, max_x, max_y, cell_m):
        """A maybe-card in this block stayed unclear: go to the next block from where its wall
        is seen face-on (~0.9 m, the whole card in view) and look back. A card found there is
        shot from there (the next block is within reach). Returns (cell, heading) where the
        robot is now."""
        p = self.panel
        moves = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        if tuple(pos) in self._looked_back:
            return pos, heading                  # one look-back per block is enough
        self._looked_back.add(tuple(pos))
        g = suspects[0]
        cx, cy = (pos[0] + 0.5) * p.map.tile, (pos[1] + 0.5) * p.map.tile
        dx, dy = g["x_m"] - cx, g["y_m"] - cy
        # card on the east wall -> seen face-on from the west neighbour, and so on
        facing = (3 if dx > 0 else 1) if abs(dx) >= abs(dy) else (2 if dy > 0 else 0)
        order = [facing] + [d for d in ((facing + 1) % 4, (facing + 3) % 4)]
        for d in order:
            nb = (pos[0] + moves[d][0], pos[1] + moves[d][1])
            if not (0 <= nb[0] <= max_x and 0 <= nb[1] <= max_y) or (pos[0], pos[1], d) in blocked:
                continue
            known = frozenset((tuple(pos), nb)) in driven
            if not known and scan_cache.get(tuple(pos), {}).get(d, 0) <= self.wall_mm:
                continue
            if self.explore_mode == "all" and nb in visited:
                continue          # a detour back: every block is visited anyway, the close looks decide
            p.log(f"still unsure about the {g['color']} {g['shape']} - looking back at it from {nb}")
            opened_k, walls_k = self._edge_knowledge(scan_cache, driven, blocked, max_x, max_y, cell_m)
            ok_move = False
            for _ in range(3):
                ok_move, heading = self.move_cell(d, heading, cell_m, camera_guard=not known, pos=pos,
                                                  walls=walls_k, opened=opened_k, bounds=(max_x, max_y))
                if ok_move or not self._retryable_move_failure():
                    break
            if not ok_move:
                if not self._retryable_move_failure():
                    blocked.add((pos[0], pos[1], d))
                continue
            driven.add(frozenset((tuple(pos), nb)))
            pos = nb
            self._live_ctx = (pos, visited, max_x, max_y, heading)
            self.draw_live_grid(pos, visited, max_x, max_y, gimbal_abs_deg=heading * 90)
            ncx, ncy = (pos[0] + 0.5) * p.map.tile, (pos[1] + 0.5) * p.map.tile
            for s_ in suspects:
                a = math.degrees(math.atan2(s_["x_m"] - ncx, s_["y_m"] - ncy))
                rng_ = math.hypot(s_["x_m"] - ncx, s_["y_m"] - ncy)
                pitch = max(-15.0, min(0.0, -math.degrees(math.atan2(0.12, max(rng_, 0.2)))))
                for off in (0, -12, 12):
                    if not p.checkpoint():
                        break
                    gy = wrap180(a - heading * 90 + off)      # relative to where the chassis faces
                    if not self._gimbal_moveto(pitch=pitch, yaw=gy, pitch_speed=120,
                                               yaw_speed=180, what="look back at card"):
                        continue
                    settled_ts = time.time()
                    self._draw_live_with_gimbal(gy)
                    self._look_for_targets(gy, pitch=pitch, settle_ts=settled_ts)
                    tid = p.map.card_of_color_near(s_["color"], s_["x_m"], s_["y_m"], 0.45)
                    if tid:
                        p.log(f"looked back: it is {tid}")
                        break
                else:
                    p.log(f"looked back: no {s_['color']} card there - it was not a card")
                    self._not_cards.append((s_["color"], s_["x_m"], s_["y_m"]))
            self._guesses = []
            self.reset_gimbal()
            return pos, heading
        p.log(f"no way to look back at the {g['color']} {g['shape']} - skipped")
        return pos, heading

    def _sensor_warn(self):
        """Short text for the panel when a Sharp gives no signal ("" when all is fine)."""
        bad = [f"Sharp {s[0].upper()}" for s in ("left", "right")
               if self.ir.sharp_status(s).startswith("NO SIGNAL")]
        return (", ".join(bad) + ": no signal - check cable / port (src/sensor_check.py)") if bad else ""

    def telemetry(self):
        """Live values shown on the mission panel."""
        left, right = self.ir.latest()
        return {
            "odom": f"({self.pos_x:+.2f}, {self.pos_y:+.2f}) m",
            "yaw": f"{self.current_yaw:+.1f} deg",
            # numeric values for the sensor widget
            "tof_mm": self.current_tof_dist_mm,
            "ir_left_cm": left,
            "ir_right_cm": right,
            "ir_max_cm": self.ir.max_cm,
            "ir_wall_cm": self.SIDE_SAFE_CM if self.ir.mount == "side" else self.ir.trigger_cm,
            "ir_mount": self.ir.mount,
            # front-corner IR obstacle modules (None = not fitted / no reading)
            "corner_left_near": self.ir.corner_near("left"),
            "corner_right_near": self.ir.corner_near("right"),
            "corner_left_raw": self.ir.corner_raw.get("left"),
            "corner_right_raw": self.ir.corner_raw.get("right"),
            "corner_left_io": self.ir.corner_io.get("left"),
            "corner_right_io": self.ir.corner_io.get("right"),
            "sensor_warn": self._sensor_warn(),
            "cam_wall_m": self._camera_wall_ahead(),
            "wall_rise_m": self.wall_rise_m,
        }

    def _draw_live_with_gimbal(self, gimbal_relative_yaw):
        if self.panel is not None:
            self.panel.map.gimbal_rel = gimbal_relative_yaw
        ctx = getattr(self, "_live_ctx", None)
        if ctx is None:
            return
        pos, visited, max_x, max_y, heading = ctx
        deg_map = {0: 0, 1: 90, 2: 180, 3: 270}
        abs_deg = (deg_map.get(heading, 0) + gimbal_relative_yaw) % 360
        self.draw_live_grid(pos, visited, max_x, max_y, gimbal_abs_deg=abs_deg)

    def draw_live_grid(self, current_pos, visited_set, max_x=3, max_y=3, gimbal_abs_deg=None):
        if self.panel is not None:
            ctx = getattr(self, "_live_ctx", None)
            heading = ctx[4] if ctx else None
            self.panel.update_robot(current_pos, heading, gimbal_abs_deg, visited_set)
            return

        cell_px = 120
        width = (max_x + 1) * cell_px
        height = (max_y + 1) * cell_px

        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        for r in range(max_y + 1):
            for c in range(max_x + 1):
                plot_y = max_y - r
                x1 = c * cell_px
                y1 = plot_y * cell_px
                x2 = x1 + cell_px
                y2 = y1 + cell_px

                if (c, r) in visited_set:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (229, 239, 247), -1)

                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(img, f"({c},{r})", (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        cur_c, cur_r = current_pos
        plot_cur_y = max_y - cur_r
        center_x = cur_c * cell_px + cell_px // 2
        center_y = plot_cur_y * cell_px + cell_px // 2
        cv2.circle(img, (center_x, center_y), 30, (0, 0, 255), -1)
        cv2.putText(img, "ROBOT", (center_x - 26, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        if gimbal_abs_deg is not None:
            rad = math.radians(gimbal_abs_deg)
            arrow_len = cell_px * 0.42
            tip_x = int(center_x + math.sin(rad) * arrow_len)
            tip_y = int(center_y - math.cos(rad) * arrow_len)
            cv2.arrowedLine(img, (center_x, center_y), (tip_x, tip_y),
                            (255, 140, 0), 3, tipLength=0.35)

        cv2.imshow("SLAM Real-time Grid Monitor", img)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Round 2: drive straight to the targets saved in round 1
    # ------------------------------------------------------------------
    def navigate_to_targets(self, round_data):
        """Round 2: plan the quickest route to shoot every selected round-1 card, drive it,
        and re-plan whenever something changes.

        Five route algorithms (greedy nearest, greedy cover, DFS branch & bound, BFS state
        search, Dijkstra with turns - route_planner.RoutePlanner) run on the round-1 map
        and are scored with the same time model (moves, turns, stops, aiming); the cheapest
        wins. After every stop - or a blocked passage, or a missed target - the plan is
        made again from where the robot is, with what is left."""
        from route_planner import GridGraph, parse_edges, plan_best, driven_edges, MOVES as RMOVES
        from target_vision import kind_of

        panel = self.panel
        print("--- Round 2: นำทางไปยังเป้าที่บันทึกไว้ ---")
        start_heading = int(round_data.get("start_heading", 0)) % 4   # put down facing the same way as round 1
        self.calibrate_yaw_sign(start_heading)

        pos = tuple(round_data.get("start", [0, 0]))
        for t in round_data.get("targets", []):
            t.setdefault("kind", kind_of(t["color"], t["shape"]))   # files from before kinds
            t.setdefault("id", t["kind"])
        remaining = [t for t in round_data.get("targets", []) if t["kind"] in panel.selected]
        for k in sorted(panel.selected - {t["kind"] for t in remaining}):
            panel.log(f"{k} was not found in round 1 - skipped")
        return self._shoot_route(round_data, remaining, pos, start_heading, {pos})

    def _shoot_route(self, round_data, remaining, pos, heading, visited):
        """Drive to a firing spot next to each card in `remaining` and shoot it (best route of
        route_planner, re-planned after every stop). Used by round 2 and, at the end of round 1,
        for cards that were found but not hit. Returns the round-2 style report dict."""
        from route_planner import GridGraph, parse_edges, plan_best, driven_edges, MOVES as RMOVES

        panel = self.panel
        nx, ny = round_data["grid_size"]
        tile = round_data.get("tile_m", self.default_distance)
        cell_m = self.config.get("movement", {}).get("distance", tile)
        driven = driven_edges(round_data)   # a wall on an edge the robot drove through is a misreading
        graph = GridGraph(nx, ny, parse_edges(round_data.get("walls")) - driven,
                          parse_edges(round_data.get("open_edges")) | driven)
        # firing_cells already applies the same 0.90 planning margin; pass the
        # assignment limit here so it is applied exactly once (1.20 -> 1.08 m).
        max_m = panel.detector.max_shoot_m
        pos = tuple(pos)
        start = pos
        remaining = list(remaining)
        deg = {0: 0, 1: 90, 2: 180, 3: 270}
        exclude = {t["id"]: set() for t in remaining}   # firing cells that did not work, per target
        self.route_info = None

        def show():
            self._live_ctx = (pos, visited, nx - 1, ny - 1, heading)
            self.draw_live_grid(pos, visited, nx - 1, ny - 1, gimbal_abs_deg=deg[heading])

        def time_left():  # also waits here while the round is paused
            return panel.checkpoint() and panel.remaining() > 0

        def plan():
            if self.shoot_order == "found":
                # found first, shot first: go for the earliest-found card still up (the route
                # planner only picks the way there - and anything else that stop also covers)
                remaining.sort(key=lambda t: t.get("first_seen") or 0.0)
                res = plan_best(graph, remaining[:1], pos, heading, tile, max_m, exclude, cost=self._route_cost())
            else:
                res = plan_best(graph, remaining, pos, heading, tile, max_m, exclude, cost=self._route_cost())
            legs = []
            prev = pos
            for cell, idx in res["stops"]:
                lg_path = res["route"]  # full route for drawing
                for i in idx:
                    t = remaining[i]
                    a = math.degrees(math.atan2(t["x_m"] - (cell[0] + 0.5) * tile,
                                                t["y_m"] - (cell[1] + 0.5) * tile)) % 360
                    legs.append({"path": [prev, cell], "fire_cell": cell, "aim_deg": a, "color": t["color"]})
                prev = cell
            if res["stops"]:
                panel.map.set_plan([{"path": res["route"], "fire_cell": l["fire_cell"], "aim_deg": l["aim_deg"],
                                     "color": l["color"]} for l in legs] or [])
            return res

        show()
        first = True
        try:
            while remaining and time_left():
                res = plan()
                if not res["stops"]:
                    panel.log("no reachable firing spot for: " + ", ".join(t["id"] for t in remaining))
                    break
                if first or res["algorithm"] != (self.route_info or {}).get("algorithm"):
                    cmp_ = "  ".join(f"{r['algorithm']} {r['time_s']:.0f}s" for r in res["comparison"] if r.get("ok"))
                    panel.log(f"route: {res['algorithm']} - {res['moves']} moves, {res['turns']} turns, "
                              f"~{res['time_s']:.0f} s  |  {cmp_}")
                self.route_info = {k: res[k] for k in ("algorithm", "time_s", "moves", "turns", "comparison")}
                if first:
                    self.route_first, self.route_replans = self.route_info, 0   # the whole-round plan
                    panel.route_plan = self.route_info
                else:
                    self.route_replans += 1
                first = False
                cell, idx = res["stops"][0]
                stop_targets = [remaining[i] for i in idx]

                # drive to the first stop (quickest path, turns included)
                path = res["route"][:res["route"].index(tuple(cell)) + 1] if tuple(cell) in res["route"] else [pos]
                blocked = False
                for nxt in path[1:]:
                    if not time_left():
                        break
                    d = next(h for h, m in RMOVES.items() if m == (nxt[0] - pos[0], nxt[1] - pos[1]))
                    known = frozenset((pos, nxt)) in driven   # driven before: open, a few tries
                    ok_move = False
                    retryable_failure = False
                    for attempt in range(3):
                        show()
                        ok_move, heading = self.move_cell(d, heading, cell_m, camera_guard=not known, pos=pos,
                                                          walls=graph.walls, opened=graph.open, bounds=(nx - 1, ny - 1))
                        if ok_move:
                            break
                        retryable_failure = self._retryable_move_failure()
                        if not retryable_failure:
                            break
                    if ok_move:
                        pos = nxt
                        visited.add(pos)
                        graph.open.add(frozenset((path[path.index(nxt) - 1], nxt)))
                        show()
                    else:
                        panel.log(f"blocked {pos}->{nxt}, re-planning")
                        edge = frozenset((pos, nxt))
                        graph.walls.add(edge)
                        graph.open.discard(edge)
                        if not retryable_failure:
                            panel.map.walls.add(edge)
                        blocked = True
                        break
                if blocked or pos != tuple(cell):
                    continue            # re-plan from here (or time is up)

                # at the stop: shoot every target planned from here
                for t in stop_targets:
                    if not time_left():
                        break
                    a = math.degrees(math.atan2(t["x_m"] - (pos[0] + 0.5) * tile, t["y_m"] - (pos[1] + 0.5) * tile))
                    rel = wrap180(a - deg[heading])
                    # a card in this block is close and below the camera: look down at it
                    near_m = math.hypot(t["x_m"] - (pos[0] + 0.5) * tile, t["y_m"] - (pos[1] + 0.5) * tile)
                    pitch = self._target_search_pitch(t, pos, near_m)
                    near = dict(kind=t["kind"], x_m=t["x_m"], y_m=t["y_m"], radius_m=0.6)

                    def hit():
                        if t.get("look"):
                            # a second look: done once a card of that colour is mapped there
                            # (in reach it was shot by the look itself)
                            return panel.map.card_near(**dict(near, radius_m=0.45)) is not None
                        if self.shooter is not None and self.shooter.armed:
                            return panel.map.card_near(shot=True, **near) is not None
                        tid = panel.map.card_near(**near)   # dry run / no blaster: aimed (or seen) is enough
                        return tid is not None and (self.shooter is None or tid in self.shooter.dry_locked)

                    done = hit()
                    # Round 1 already supplied the bearing. One direct look is enough;
                    # a miss may try one alternate firing cell, not five head angles.
                    for off in (0,):
                        if done or not time_left():
                            break
                        g = wrap180(rel + off)
                        if not self._gimbal_moveto(pitch=pitch, yaw=g, yaw_speed=180,
                                                   what="route target search"):
                            continue
                        settled_ts = time.time()
                        self._draw_live_with_gimbal(g)
                        self._look_for_targets(g, pitch=pitch, settle_ts=settled_ts)
                        done = hit()
                    if done:
                        remaining.remove(t)
                        if t.get("look"):
                            panel.log(f"second look: the {t['color']} card is there - mapped")
                    elif t.get("look"):
                        panel.log(f"second look from {pos}: no {t['color']} card there")
                        remaining.remove(t)
                    else:
                        exclude[t["id"]].add(tuple(pos))
                        panel.log(f"{t['id']} not hit from {pos}, trying another spot")
                        if len(exclude[t["id"]]) >= self.route_target_spots:
                            panel.log(f"giving up on {t['id']}")
                            remaining.remove(t)
                self.reset_gimbal()
        except KeyboardInterrupt:
            print("\n--> ยกเลิกโดยผู้ใช้")
        finally:
            try:
                self._stop(0.2)
                self.reset_gimbal()
            except KeyboardInterrupt:
                pass
        if not time_left() and remaining:
            panel.log("time is up" if panel.remaining() <= 0 else "stopped")
        panel.map.set_plan([])
        self._route_heading = heading
        return {
            "start_grid": list(start),
            "end_grid": list(pos),
            "visited_cells": len(visited),
            "remaining_targets": [t["id"] for t in remaining],
            "route": getattr(self, "route_first", None) or self.route_info,
            "replans": getattr(self, "route_replans", 0),
        }

    def _backtrack_route(self, pos, heading, visited, scan_cache, blocked, driven, max_x, max_y):
        """Cheapest route (Dijkstra over (cell, heading): a move 2.6 s, a 90 deg turn 1.2 s,
        a U-turn 2.2 s - route_planner.COST) to the nearest visited block that still has an
        open, unexplored way. Only ways driven before, or seen open from both sides, are used.
        Returns [pos, ..., goal] or None when nothing is left to explore."""
        import heapq
        from route_planner import COST, turn_cost
        moves = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        def inside(c):
            return 0 <= c[0] <= max_x and 0 <= c[1] <= max_y

        def frontier(c):
            sc = scan_cache.get(c)
            if sc is None:
                return False
            for d, (dx, dy) in moves.items():
                nb = (c[0] + dx, c[1] + dy)
                if inside(nb) and nb not in visited and (c[0], c[1], d) not in blocked and sc.get(d, 0) > self.wall_mm:
                    return True
            return False

        def usable(a, d, b):
            if frozenset((a, b)) in driven:
                return True
            sa, sb = scan_cache.get(a), scan_cache.get(b)
            return (sa is not None and sb is not None and (a[0], a[1], d) not in blocked
                    and sa.get(d, 0) > self.wall_mm and sb.get((d + 2) % 4, 0) > self.wall_mm)

        goals = {c for c in visited if frontier(c)}
        if not goals:
            return None
        start = (tuple(pos), heading)
        best = {start: 0.0}
        prev = {start: None}
        q = [(0.0, 0, start)]
        tie = itertools.count(1)
        while q:
            cost, _, (c, h) = heapq.heappop(q)
            if cost > best.get((c, h), float("inf")):
                continue
            if c in goals:
                path, node = [], (c, h)
                while node is not None:
                    path.append(node[0])
                    node = prev[node]
                path = path[::-1]
                return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]
            for d, (dx, dy) in moves.items():
                nb = (c[0] + dx, c[1] + dy)
                if nb not in visited or not usable(c, d, nb):
                    continue
                nc = cost + turn_cost(h, d) + COST["move_s"]
                if nc < best.get((nb, d), float("inf")):
                    best[(nb, d)] = nc
                    prev[(nb, d)] = (c, h)
                    heapq.heappush(q, (nc, next(tie), (nb, d)))
        return None

    # ------------------------------------------------------------------
    # coverage exploration: which blocks are still worth a visit
    # ------------------------------------------------------------------
    MOVES4 = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

    def _edge_knowledge(self, scan_cache, driven, blocked, max_x, max_y, tile):
        """(open edges, wall edges) known so far. A scan reading r one way from a block says
        the edge there is a wall (r short) or open; a long reading also says how many blocks
        further the way stays open and where its end wall is (ToF 1.4 m -> one more open
        block, then a wall) - walls are known without driving there."""
        inside = lambda c: 0 <= c[0] <= max_x and 0 <= c[1] <= max_y
        opened, walls = set(driven), set()
        far_open, far_walls = set(), set()                      # inferred from long readings
        off_m = tile / 2.0 - self.CENTER_TOF_MM / 1000.0        # ToF head -> block centre
        for (cx, cy), sc in scan_cache.items():
            for d, r in sc.items():
                if r is None or r < 60:
                    continue
                dx, dy = self.MOVES4[d]
                nb = (cx + dx, cy + dy)
                if not inside(nb):
                    continue
                e = frozenset(((cx, cy), nb))
                if r <= self.wall_mm:
                    if e not in driven:
                        walls.add(e)
                    continue
                opened.add(e)
                # how far the way goes: wall at centre + (k + 0.5) tiles
                k_f = ((r / 1000.0 + off_m) - tile / 2.0) / tile
                k = int(round(k_f))
                if k < 1 or abs(k_f - k) > 0.25:
                    continue                                   # between blocks: trust less
                # open edges up to 3 blocks out (a ~2 m reading); the end wall only up to 2
                cur = nb
                for _ in range(min(k, 3) - 1):
                    nxt = (cur[0] + dx, cur[1] + dy)
                    if not inside(nxt):
                        break
                    far_open.add(frozenset((cur, nxt)))
                    cur = nxt
                end = (cur[0] + dx, cur[1] + dy)
                if k <= 2 and inside(end):
                    far_walls.add(frozenset((cur, end)))
        for (cx, cy, d) in blocked:
            dx, dy = self.MOVES4[d]
            e = frozenset(((cx, cy), (cx + dx, cy + dy)))
            if e not in driven:
                walls.add(e)
                opened.discard(e)
        walls -= driven
        opened -= walls
        # what a scan saw right at the edge always wins over what a long reading suggests
        direct = opened | walls
        opened |= {e for e in far_open if e not in direct}
        walls |= {e for e in far_walls if e not in direct and e not in far_open}
        return opened, walls

    def _unseen_faces(self, cell, opened, walls, max_x, max_y, tile):
        """Wall faces of this block (the sides cards can hang on) that no camera look has
        seen well yet (close enough, in the picture, not too slanted, nothing in between),
        and how many of its edges are still unknown."""
        from route_planner import GridGraph
        inside = lambda c: 0 <= c[0] <= max_x and 0 <= c[1] <= max_y
        unknown, faces = 0, []
        for d, (dx, dy) in self.MOVES4.items():
            nb = (cell[0] + dx, cell[1] + dy)
            if not inside(nb):
                faces.append(d)                                # outer wall
                continue
            e = frozenset((tuple(cell), nb))
            if e in walls:
                faces.append(d)
            elif e not in opened:
                unknown += 1
        if not faces:
            return [], unknown
        # everything not known open blocks the view (an unknown edge may be a wall)
        block = set()
        for x in range(max_x + 1):
            for y in range(max_y + 1):
                for nb in ((x + 1, y), (x, y + 1)):
                    if inside(nb) and frozenset(((x, y), nb)) not in opened:
                        block.add(frozenset(((x, y), nb)))
        g = GridGraph(max_x + 1, max_y + 1, block)
        half_fov = self.SEE_HALF_FOV_DEG
        out = []
        for d in faces:
            dx, dy = self.MOVES4[d]
            px, py = (cell[0] + 0.5 + dx * 0.47) * tile, (cell[1] + 0.5 + dy * 0.47) * tile
            seen = False
            for lc, a in self._looks:
                lx, ly = (lc[0] + 0.5) * tile, (lc[1] + 0.5) * tile
                vx, vy = px - lx, py - ly
                dist = math.hypot(vx, vy)
                if dist > self.SEE_RANGE_M or dist < 1e-6:
                    continue
                bearing = math.degrees(math.atan2(vx, vy))
                if abs(wrap180(bearing - a)) > half_fov:
                    continue
                # face normal points into the block (-dx, -dy); angle to the line of sight
                cosv = (-vx * -dx + -vy * -dy) / dist
                if cosv < math.cos(math.radians(self.SEE_MAX_VIEW_DEG)):
                    continue
                if tuple(lc) != tuple(cell) and not g.line_of_sight(tuple(lc), (px, py), tile):
                    continue
                seen = True
                break
            if not seen:
                out.append(d)
        return out, unknown

    def _next_block(self, pos, heading, travel_dir, visited, opened, walls, blocked, dead, max_x, max_y, tile):
        """The block to go to next: the most still-unseen wall faces + unknown edges per second
        of driving (Dijkstra over known-open edges, turns counted unless it slides). Returns
        (goal, [pos, ..., goal]) or (None, None) when nothing is left worth a visit."""
        import heapq as hq
        from route_planner import COST, turn_cost
        inside = lambda c: 0 <= c[0] <= max_x and 0 <= c[1] <= max_y
        usable = opened - dead
        start = (tuple(pos), heading)          # turn costs start from where the chassis faces
        best = {start: 0.0}
        prev = {start: None}
        nmoves = {start: 0}
        q = [(0.0, 0, start)]
        n = 0
        reach = {}                                          # cell -> (cost, state)
        while q:
            t, _, (c, h) = hq.heappop(q)
            if t > best.get((c, h), 1e18):
                continue
            if c not in reach:
                reach[c] = (t, (c, h))
            for d, (dx, dy) in self.MOVES4.items():
                nb = (c[0] + dx, c[1] + dy)
                if not inside(nb) or frozenset((c, nb)) not in usable or (c[0], c[1], d) in blocked:
                    continue
                turn = 0.0 if self.strafe else turn_cost(h, d)
                nt = t + COST["move_s"] + turn
                if nt < best.get((nb, d), 1e18):
                    best[(nb, d)] = nt
                    prev[(nb, d)] = (c, h)
                    nmoves[(nb, d)] = nmoves[(c, h)] + 1
                    n += 1
                    hq.heappush(q, (nt, n, (nb, d)))
        def path_to(st):
            path, node = [], st
            while node is not None:
                path.append(node[0])
                node = prev[node]
            path = path[::-1]
            return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]

        visit_all = self.explore_mode == "all"
        # In complete-coverage mode, shoot opportunistically from the current cell and leave
        # target detours for the mop-up pass after every reachable cell has been visited.
        spot = None if visit_all else self._queue_spot(
            pos, reach, nmoves, opened, walls, max_x, max_y, tile
        )
        if spot is not None:
            c, st, tid = spot
            self._log(f"shooting queue: {tid} (found first) - to {c}, {nmoves[st]} move(s), to shoot it")
            return c, path_to(st)

        order_rel = {r: i for i, r in enumerate(self.explore_order)}
        rel_name = {0: "front", 1: "right", 2: "back", 3: "left"}
        choice = None
        hard = self._hard_card_blocks()          # cards at a hard angle: shot from inside their block
        # A selected target that could not be fired at from the discovery view takes
        # priority now. Enter its block and look straight/down before continuing the
        # coverage tour; delaying this until mop-up is why visible squares were passed.
        # During full-map Round 1, never drive back toward a wall-mounted target merely to
        # improve its view. Live detections shoot immediately; otherwise normal coverage will
        # reach the cell. This avoids target-induced wall approaches and ping-pong routes.
        hard_reachable = [] if visit_all else [(reach[c][0], c, reach[c][1]) for c in hard
                                               if c in reach and c != tuple(pos)]
        if hard_reachable:
            _, c, st = min(hard_reachable, key=lambda x: (x[0], x[1]))
            self._log(f"target priority: go to {c} now for a direct in-cell look and shot")
            return c, path_to(st)
        for c, (t, st) in reach.items():
            if c == tuple(pos):
                continue
            if c in visited:
                if visit_all:
                    continue                    # finish coverage before revisiting a difficult target
                faces, unknown = [], 0
                # a trip back into a visited block for a hard-angle card only when it is right
                # next door - further ones wait for the shooting pass at the end (no ping-pong)
                if c not in hard or nmoves[st] > 1:
                    continue
            else:
                faces, unknown = self._unseen_faces(c, opened, walls, max_x, max_y, tile)
            value = len(faces) + 1.5 * unknown + (3.0 if c in hard else 0.0)
            if visit_all and c not in visited:
                value = max(value, 1.0)              # the map must be complete: every block
            if value <= 0:
                continue
            # Preserve the configured relative direction only as a tie-break after utility/time.
            first = st
            while prev.get(first) is not None and prev[first] != start:
                first = prev[first]
            d0 = first[1]
            tie = order_rel.get(rel_name[(d0 - (travel_dir if travel_dir is not None else heading)) % 4], 9)
            # then dead ends first: a block with no other way on is cheap now, a trip back later
            came = prev[st][0] if prev.get(st) else None
            exits = 0
            for d, (dx, dy) in self.MOVES4.items():
                nb = (c[0] + dx, c[1] + dy)
                if inside(nb) and nb != came and nb not in visited and frozenset((c, nb)) not in walls:
                    exits += 1
            # Information per expected second. The previous ordering was nearest-first,
            # even though this selector's contract is next-best-view per unit time.
            rate = information_rate(value, t, 0.0 if c in visited else self.SCAN_COST_S)
            # Complete-coverage mode uses the nearest unvisited cell. Letting a distant cell
            # with more visible faces win caused long cross-map trips and repeated paths.
            key = (round(t, 1), tie, exits, -value) if visit_all else \
                  (-rate, round(t, 1), tie, exits, -value)
            if choice is None or key < choice[0]:
                choice = (key, c, st)
        if choice is None:
            return None, None
        if visit_all and getattr(self, "TOUR_PLAN", True):
            # look ahead: the order that visits EVERY block still to see in the fewest moves,
            # not just the nearest one (last run left (5,1) behind and drove 8 moves back to it)
            cands = [c for c in reach if c != tuple(pos) and c not in visited]
            first = self._tour_first(tuple(pos), cands, usable, blocked, inside)
            if first is not None and first != choice[1] and first in reach:
                self._log(f"route plan: {first} before {choice[1]} - {len(cands)} blocks to see, "
                          "fewer moves in total that way")
                return first, path_to(reach[first][1])
        return choice[1], path_to(choice[2])

    def _tour_first(self, start, cands, usable, blocked, inside):
        """First block of a short open tour from start through all cands (moves over known-open
        edges; nearest-neighbour, then 2-opt). None when there is nothing to plan."""
        if len(cands) < 2:
            return None
        from collections import deque as _dq

        def bfs(src):
            dist = {src: 0}
            q = _dq([src])
            while q:
                c = q.popleft()
                for d, (dx, dy) in self.MOVES4.items():
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in dist or not inside(nb) or frozenset((c, nb)) not in usable or (c[0], c[1], d) in blocked:
                        continue
                    dist[nb] = dist[c] + 1
                    q.append(nb)
            return dist
        nodes = [start] + list(cands)
        D = {a: bfs(a) for a in nodes}
        far = 99
        cost = lambda a, b: D[a].get(b, far)
        tour, left = [], set(cands)
        cur = start
        while left:
            nxt = min(left, key=lambda c: (cost(cur, c), c))
            tour.append(nxt)
            left.discard(nxt)
            cur = nxt
        def length(t):
            return cost(start, t[0]) + sum(cost(a, b) for a, b in zip(t, t[1:]))
        best = length(tour)
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    L = length(cand)
                    if L < best:
                        tour, best, improved = cand, L, True
        return tour[0]

    def shot_queue(self):
        """Selected cards on the map still up, the one found first at the front."""
        p = self.panel
        if p is None or self.shooter is None:
            return []
        q = [t for t in p.map.target_list() if not self._card_done(t) and t["kind"] in p.selected]
        q.sort(key=lambda t: t.get("first_seen") or 0.0)
        return q

    def _queue_spot(self, pos, reach, nmoves, opened, walls, max_x, max_y, tile):
        """(cell, search state, card id) of a spot within queue_detour moves that can shoot
        the card found first among those still up (then the next one, ...), or None."""
        if self.queue_detour <= 0 or self.panel is None or self.shooter is None:
            return None
        from route_planner import GridGraph, firing_cells
        g = GridGraph(max_x + 1, max_y + 1, {e for e in walls if isinstance(e, frozenset)},
                      {e for e in opened if isinstance(e, frozenset)})
        max_m = self.panel.detector.max_shoot_m
        for t in self.shot_queue():
            best = None
            for c, _d, _seen in firing_cells(g, t, tile, max_m):
                c = tuple(c)
                if c == tuple(pos) or (t["id"], c) in self._tried_from or c not in reach:
                    continue
                st = reach[c][1]
                if nmoves[st] > self.queue_detour:
                    continue
                key = (nmoves[st], reach[c][0])
                if best is None or key < best[0]:
                    best = (key, c, st)
            if best is not None:
                return best[1], best[2], t["id"]
        return None

    def _route_cost(self):
        """Route planner time model for this robot: sliding (mecanum) needs no turns."""
        return {"turn90_s": 0.0, "turn180_s": 0.0} if self.strafe else None

    def _shoot_reserve_s(self):
        """Seconds needed to go and shoot every found-but-not-hit selected card (route
        planner estimate) + a margin: exploring stops when less than that is left."""
        p = self.panel
        todo = self._unshot_ids()
        if not todo:
            return 0.0
        try:
            from route_planner import GridGraph, parse_edges, plan_best, driven_edges
            data = p.map.to_json()
            tg = [t for t in data["targets"] if t["id"] in todo]
            nx, ny = data["grid_size"]
            dr = driven_edges(data)
            g = GridGraph(nx, ny, parse_edges(data.get("walls")) - dr, parse_edges(data.get("open_edges")) | dr)
            res = plan_best(g, tg, tuple(p.map.robot), p.map.heading, data["tile_m"], p.detector.max_shoot_m,
                            cost=self._route_cost())
            return float(res.get("time_s") or 0.0) + self.RESERVE_MARGIN_S
        except Exception:
            return self.MOP_UP_RESERVE_S

    def _hard_card_blocks(self):
        """Blocks holding a selected card not hit yet that no straight shot from a neighbour
        can take (off to the side / too slanted) and that were not tried from inside yet:
        the robot goes into them, aims down and shoots."""
        p = self.panel
        if p is None or self.shooter is None:
            return set()
        out = set()
        for t in p.map.target_list():
            if self._card_done(t) or t["kind"] not in p.selected:
                continue
            c = tuple(t["cell"])
            if (t["id"], c) not in self._tried_inside:
                out.add(c)
        return out

    def shoot_here(self, pos, heading):
        """Aim at every selected, not-hit card this block may shoot (its own block, or straight
        ahead in the next one): turn the gimbal to where the map has it - camera down for a card
        in this block - and let the normal look/aim/fire do the rest."""
        p = self.panel
        if p is None or self.shooter is None or not p.checkpoint():
            return
        tile = p.map.tile
        cx, cy = (pos[0] + 0.5) * tile, (pos[1] + 0.5) * tile
        todo = [t for t in p.map.target_list()
                if not self._card_done(t) and t["kind"] in p.selected and (t["id"], tuple(pos)) not in self._tried_from
                and p.map.in_reach(pos, t["id"], self.reach_cells)]
        if self.shoot_order == "found":
            todo.sort(key=lambda t: t.get("first_seen") or 0.0)     # found first, shot first
        for t in todo:
            if not p.checkpoint() or (p.round_t0 and p.remaining() <= 0):
                break
            own_cell = tuple(t["cell"]) == tuple(pos)
            a = math.degrees(math.atan2(t["x_m"] - cx, t["y_m"] - cy))
            g = wrap180(a - heading * 90)
            near_m = math.hypot(t["x_m"] - cx, t["y_m"] - cy)
            pitch = self._target_search_pitch(t, pos, near_m)
            p.log(f"shooting {t['id']} from {tuple(pos)}" + (" (inside its block, camera down)" if pitch else ""))
            offsets = self.CLOSE_SHOOT_SEARCH_OFFSETS if own_cell or near_m < 0.55 \
                else self.SHOOT_SEARCH_OFFSETS
            for off in offsets:
                if not self._gimbal_moveto(pitch=pitch, yaw=wrap180(g + off), pitch_speed=min(240, self.GIMBAL_DPS),
                                           yaw_speed=self.GIMBAL_DPS, what="shoot-here search"):
                    continue
                settled_ts = time.time()
                self._draw_live_with_gimbal(wrap180(g + off))
                self._look_for_targets(wrap180(g + off), pitch=pitch, settle_ts=settled_ts)
                tt = p.map.targets.get(t["id"])
                if tt is None or tt["shot"]:
                    break
            # Mark only after the camera/shooter path above actually ran. The previous code
            # marked _tried_from before _look_for_targets, causing that function to skip the
            # shot entirely. A failed own-cell look is bounded here so routing cannot bounce
            # out of and back into the same IR-tight cell forever.
            if own_cell:
                self._tried_inside.add((t["id"], tuple(pos)))

    def _target_search_pitch(self, target, pos, distance_m):
        """Pitch toward card centre; fixed -10 deg hid 20-35 cm cards below the frame."""
        cam_h = float((self.config.get("vision", {}) or {}).get("camera_height_m", 0.25))
        pitch = math.degrees(math.atan2(self.card_aim_height_m - cam_h, max(distance_m, 0.2)))
        return max(self.CLOSE_PITCH_MIN_DEG, min(-2.0, pitch))

    def _corridor_known(self, pos, heading, d, prev, scan_cache, driven, blocked, max_x, max_y, tile):
        """Arrived in `pos` driving forward / back in map direction d. When the side Sharps saw
        a wall on BOTH sides while the robot was in the block, and the way ahead is already
        known open (a long ToF reading from the block before), this is a plain corridor block:
        returns {scan label: mm} for ahead and behind (not looked at again) and the Sharp
        readings - the camera then only looks left and right. None = do the normal scan."""
        if not self.fast_corridor or d is None or prev is None or self._last_body_dir not in (0, 2):
            return None
        s = self._side_samples
        if len(s) < 3:
            return None
        l_cm = sorted(v[0] for v in s)[len(s) // 2]
        r_cm = sorted(v[1] for v in s)[len(s) // 2]
        if l_cm > self.CORRIDOR_WALL_CM or r_cm > self.CORRIDOR_WALL_CM:
            return None                                      # a side opens: junction -> full scan
        opened, _ = self._edge_knowledge(scan_cache, driven, blocked, max_x, max_y, tile)
        dx, dy = self.MOVES4[d]
        ahead = (pos[0] + dx, pos[1] + dy)
        if not (0 <= ahead[0] <= max_x and 0 <= ahead[1] <= max_y) or frozenset((tuple(pos), ahead)) not in opened:
            return None                                      # the end of the corridor: look at that wall too
        prev_r = scan_cache.get(tuple(prev), {}).get(d) or 1500
        lab = {0: "front", 1: "right", 2: "back", 3: "left"}
        known = {lab[(d - heading) % 4]: max(self.wall_mm + 1, prev_r - tile * 1000),
                 lab[(d + 2 - heading) % 4]: 900}
        return known, (l_cm, r_cm)

    def _unshot_ids(self):
        p = self.panel
        return [t["id"] for t in p.map.target_list() if not self._card_done(t) and t["kind"] in p.selected]

    def _card_done(self, t):
        """Nothing more to do for this card: hit - or, with the blaster off (dry run), aimed
        at and locked once (it will never be 'hit', so it must not keep drawing the robot)."""
        if t["shot"]:
            return True
        if t.get("missed"):
            return True                    # strict per-target shot cap: never focus it again
        sh = self.shooter
        return sh is not None and not sh.armed and t["id"] in sh.dry_locked

    def _drive_in_hints(self, pos, heading):
        """Cards (or parts of cards) the camera saw while the robot drove into this block: where
        were they, given how far the robot had come at that moment? Those in this block (beside
        or ahead - last runs drove past them and the close look then missed them) become
        maybe-cards: the close look turns the camera straight at each."""
        p = self.panel
        rec = getattr(self, "_move_rec", None)
        if p is None or rec is None or not rec.get("t1") or rec.get("used"):
            return
        rec["used"] = True
        tr = rec["track"]
        if len(tr) < 2 or tr[-1][1] < 0.3:
            return                                       # no real move in (a failed / short move)
        tile = p.map.tile
        cx, cy = (pos[0] + 0.5) * tile, (pos[1] + 0.5) * tile
        base = {0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0)
        move_abs = math.radians(base + {0: 0, 1: 90, 2: 180, 3: 270}[rec["body_dir"] % 4])
        ux, uy = math.sin(move_abs), math.cos(move_abs)
        total = tr[-1][1]
        lat = p.camera_latency_s
        found = []
        for ts, g, seen in list(p.worker.motion_log):
            t = ts - lat
            if t < rec["t0"] or t > rec["t1"]:
                continue
            k = next((i for i, (tt, _) in enumerate(tr) if tt >= t), len(tr) - 1)
            done = tr[k][1]
            back = total - done                          # how far before the arrival point it was
            rx, ry = cx - ux * back, cy - uy * back
            for x in seen:
                d = x.get("dist_m")
                a = math.radians(base + g[1] + x["bearing"])
                if not d:
                    # no size (part of a card): where the ray meets this block's walls
                    if back > 0.35:
                        continue
                    wx, wy = p.map.wall_point(pos, math.degrees(a)) if back < 0.05 else (None, None)
                    if wx is None:
                        # from off-centre: step along the ray to the block's edge
                        d = 0.05
                        while d < 1.0:
                            px_, py_ = rx + math.sin(a) * d, ry + math.cos(a) * d
                            if not (pos[0] * tile - 0.02 <= px_ <= (pos[0] + 1) * tile + 0.02 and
                                    pos[1] * tile - 0.02 <= py_ <= (pos[1] + 1) * tile + 0.02):
                                break
                            d += 0.02
                    else:
                        d = math.hypot(wx - rx, wy - ry)
                px_, py_ = rx + math.sin(a) * d, ry + math.cos(a) * d
                inside = (pos[0] * tile - 0.08 <= px_ <= (pos[0] + 1) * tile + 0.08 and
                          pos[1] * tile - 0.08 <= py_ <= (pos[1] + 1) * tile + 0.08)
                if not inside:
                    continue
                h = self.config.get("vision", {}).get("camera_height_m", 0.25) + \
                    d * math.tan(math.radians(g[0] + x["elevation"]))
                found.append((x["color"], x.get("kind"), px_, py_, h))
        hints = []
        for c, kind, px_, py_, h in found:
            if any(hc == c and math.hypot(hx - px_, hy - py_) < 0.2 for hc, _, hx, hy, _ in hints):
                continue
            if p.map.card_of_color_near(c, px_, py_, 0.3):
                continue                                   # already on the map
            hints.append((c, kind, px_, py_, h))
        for c, kind, px_, py_, h in hints[:3]:
            dc = max(0.12, math.hypot(px_ - cx, py_ - cy))
            a_abs = math.degrees(math.atan2(px_ - cx, py_ - cy))
            rel = wrap180(a_abs - base)
            pit = max(-20.0, min(20.0, math.degrees(math.atan2(h - self.config.get("vision", {}).get(
                "camera_height_m", 0.25), dc))))
            shape = kind.split(" ", 1)[1] if kind and " " in kind else "unknown"
            self._guesses.append({"cell": tuple(pos), "color": c, "shape": shape, "bearing": 0.0, "elevation": 0.0,
                                  "abs_deg": a_abs % 360, "why": "seen while driving in", "gimbal_yaw": rel,
                                  "pitch": pit})
            p.log(f"{c} {'card' if kind else 'colour'} seen while driving in, {dc:.2f} m from the centre - "
                  "the close look will aim at it")

    def _edge_on_here(self, pos, heading):
        """A card seen only nearly edge-on earlier: if this block sees it face-on enough
        (<= 45 deg, <= 1.2 m, nothing in between), look at it now - not at the end of the round
        (last run: 3 of them, time ran out before the end-of-round looks)."""
        p = self.panel
        if not self.second_look or not p.checkpoint():
            return
        tile = p.map.tile
        cx, cy = (pos[0] + 0.5) * tile, (pos[1] + 0.5) * tile
        for e in p.map.edge_on_open():
            d = math.hypot(e["x_m"] - cx, e["y_m"] - cy)
            view = p.map.card_view_deg(pos, e["x_m"], e["y_m"])
            if d > 1.2 or d < 0.15 or view is None or view > 45 or not p.map.visible_from(pos, e["x_m"], e["y_m"]):
                continue
            with p.map.lock:
                for ee in p.map.edge_on:
                    if ee["color"] == e["color"] and math.hypot(ee["x_m"] - e["x_m"], ee["y_m"] - e["y_m"]) < 0.05:
                        ee["looked"] = True
            a = math.degrees(math.atan2(e["x_m"] - cx, e["y_m"] - cy))
            rel = wrap180(a - {0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0))
            pit = max(-20.0, min(0.0, math.degrees(math.atan2(-0.08, max(d, 0.2)))))
            p.log(f"the {e['color']} card seen edge-on earlier: {view:.0f} deg from here - looking now")
            if self._gimbal_moveto(pitch=pit, yaw=rel, pitch_speed=min(240, self.GIMBAL_DPS),
                                   yaw_speed=self.GIMBAL_DPS, what="edge-on card"):
                self._draw_live_with_gimbal(rel)
                self._look_for_targets(rel, pitch=pit, settle_ts=time.time())
            return                                     # one per block

    def _budget_check(self, n_visited, total):
        """Behind schedule (time per block so far x blocks left + the end-of-round reserve >
        time left): drop the extras - close looks only for cards still to confirm, one still
        look per sweep. Logged once."""
        p = self.panel
        if p is None or not p.round_t0 or n_visited < 4:
            return
        per = p.elapsed() / max(n_visited, 1)
        blocks_left = max(total - n_visited, 0)
        predicted_coverage_end = p.elapsed() + per * blocks_left
        # Round 1 has a hard 10-minute limit, but coverage targets 7:30. The remaining
        # 2:30 belongs to shot retries, temporarily unsafe edges and final map gaps.
        hurry = predicted_coverage_end > self.COVERAGE_TARGET_S
        if hurry and not self.hurry:
            self._log(f"behind 7:30 coverage target ({per:.0f} s per block, {blocks_left} blocks left, "
                      f"predicted finish {predicted_coverage_end:.0f} s): candidate-only checks from now on")
        self.hurry = hurry
        self.SWEEP_CHECKS = 1 if hurry else self._sweep_checks_cfg
        self.MOTION_CHECKS = 1 if hurry else self._motion_checks_cfg

    def _camera_ready_for_scan(self):
        """Do not map or leave a cell using a frozen frame from a dropped Wi-Fi stream."""
        p = self.panel
        if p is None:
            return True
        _, _, ts, _ = p.worker.latest()
        if ts and time.time() - ts <= self.CAMERA_STALE_S:
            return True
        p.log("camera frame is stale - waiting for live video before scanning or moving")
        end = time.time() + self.CAMERA_RECOVER_WAIT_S
        while time.time() < end and p.checkpoint():
            _, _, ts, _ = p.worker.latest()
            if ts and time.time() - ts <= self.CAMERA_STALE_S:
                p.log("camera live again - resuming")
                return True
            time.sleep(0.05)
        if p.checkpoint():
            p.log("camera still has NO SIGNAL - stopping instead of passing targets on a frozen image")
            p.abort.set()
        return False

    def _second_looks(self, pos, heading):
        """Cards seen only nearly edge-on (too slanted to map): drive to a block that sees that
        wall from the front and look again - mapped (and shot, when in reach) if it is a card.
        Returns (end cell, heading)."""
        p = self.panel
        if not self.second_look or p is None or not p.checkpoint():
            return pos, heading
        todo = p.map.edge_on_open()
        if not todo or p.remaining() <= 30:
            return pos, heading
        looks = [{"id": f"look {e['color']} #{i}", "kind": f"{e['color']} square", "color": e["color"],
                  "shape": "square", "x_m": e["x_m"], "y_m": e["y_m"], "look": True,
                  "first_seen": e["t"], "views": [], "cell": list(e["cell"])} for i, e in enumerate(todo)]
        p.log("seen only edge-on: " + ", ".join(f"{e['color']} near ({e['x_m']:.1f}, {e['y_m']:.1f}) m"
                                               for e in todo) + " - going to look from the front")
        with p.map.lock:
            for e in p.map.edge_on:
                e["looked"] = True                  # once each
        data = p.map.to_json()
        res = self._shoot_route(data, looks, pos, heading, set(map(tuple, data.get("visited", [pos]))))
        return tuple(res["end_grid"]), getattr(self, "_route_heading", heading)

    def _mop_up_round1(self, pos, heading):
        """End of round 1: every selected card on the map that is confirmed but not hit yet
        is shot from beside it (same route planner as round 2). Returns the end cell."""
        p = self.panel
        if not p.checkpoint() or p.remaining() <= 20:
            return pos
        data = p.map.to_json()
        todo = [t for t in data.get("targets", [])
                if t.get("confirmed") and not t.get("shot") and t["kind"] in p.selected
                and (self.retry_missed or not t.get("missed"))
                and t["id"] not in getattr(self.shooter, "dry_locked", set())]
        if self.retry_missed and getattr(self.shooter, "armed", False):
            # still standing after max shots: once more, from the best spot, with any trim the
            # misses taught (after the cards never shot at, found first first)
            again = [t for t in data.get("targets", []) if t.get("missed") and t["kind"] in p.selected
                     and t["id"] not in self._retried]
            for t in sorted(again, key=lambda t: t.get("first_seen") or 0.0):
                self._retried.add(t["id"])
                with p.map.lock:
                    if t["id"] in p.map.targets:
                        p.map.targets[t["id"]]["shot"] = False
                        p.map.targets[t["id"]]["missed"] = False
                t["shot"] = False
                t["first_seen"] = 1e10 + (t.get("first_seen") or 0.0)     # after the never-shot ones
                todo.append(t)
        if not todo:
            return pos
        todo.sort(key=lambda t: t.get("first_seen") or 0.0)
        p.log("not hit yet: " + ", ".join(t["id"] for t in todo) + " - going next to them")
        res = self._shoot_route(data, todo, pos, heading, set(map(tuple, data.get("visited", [pos]))))
        return tuple(res["end_grid"])

    # ------------------------------------------------------------------
    # Explore
    # ------------------------------------------------------------------
    def explore_and_map_all(self):
        print("--- เริ่มการสำรวจและสร้างแผนที่ (Robust Grid Exploration) ---")

        from mission_panel import parse_heading
        self.calibrate_yaw_sign(parse_heading(self.config.get("grid_map", {}).get("start", {}).get("heading", 0)))
        for side in ("left", "right"):          # a module that is "WALL" already: logs a warning now
            self._corner_says_wall(side, self._tof_fresh(3))

        data_cfg = self.config.get("data_collection", {})
        files_cfg = data_cfg.get("files", {})
        data_dir = data_cfg.get("data_dir", "data/raw/run1")
        os.makedirs(data_dir, exist_ok=True)
        self._start_recording(data_dir)
        date_str = datetime.now().strftime("%Y%m%d")

        filename_key = files_cfg.get("exploration", "exploration_map_data")
        csv_path = os.path.join(data_dir, f"log_{date_str}_{filename_key}.csv")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "unix_timestamp", "grid_x", "grid_y",
                "real_x_m", "real_y_m", "heading", "heading_deg",
                "front_tof_mm", "right_ir_cm", "back_ir_cm", "left_ir_cm", "action",
                "odom_x_m", "odom_y_m", "yaw_deg"   # คอลัมน์ใหม่ท้ายสุด ของจริงจากล้อ/IMU
            ])

        def log_step(g_x, g_y, h, tof_val, r_val, b_val, l_val, act_label="VISIT", c_size=0.6):
            deg_map = {0: 0, 1: 90, 2: 180, 3: 270}
            with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
                c_writer = csv.writer(file)
                c_writer.writerow([
                    time.time(), g_x, g_y,
                    round(g_x * c_size, 3), round(g_y * c_size, 3),
                    h, deg_map.get(h, 0),
                    round(tof_val, 1),
                    round(r_val / 10.0, 2),
                    round(b_val / 10.0, 2),
                    round(l_val / 10.0, 2),
                    act_label,
                    round(self.pos_x, 3), round(self.pos_y, 3), round(self.current_yaw, 1)
                ])

        if not hasattr(self, "current_tof_dist_mm"):
            self.current_tof_dist_mm = 10000

        CELL_SIZE = self.config.get("movement", {}).get("distance", 0.6)
        FRONT_WALL_MM = self.wall_mm   # same for every direction: the ToF turns with the gimbal
        SIDE_OPEN_MM = self.wall_mm

        visited = set()
        stack = []
        blocked = set()
        driven = set()        # edges the robot really drove through (always open)
        dead_edges = set()    # driven edges it later could not drive again (not used for routes)
        dead_edge_reopens = 0 # one recovery pass: drift/corner trouble is not a permanent wall
        self._swept_cells = set()
        self._not_cards, self._looked_back, self._guesses = [], set(), []
        self._candidate_checked = set()
        self._tried_inside, self._tried_from, self._retried = set(), set(), set()
        self._aim_attempts = {}
        scan_cache = {}       # (x, y) -> {map dir: ToF mm} (scanned once per cell)
        direct_scan_dirs = {} # directions physically measured here (not inferred/came-from)
        entry_heading = {}    # (x, y) -> heading when the robot first arrived there

        grid_cfg = self.config.get("grid_map", {})
        MAX_X = grid_cfg.get("max_x", 3)
        MAX_Y = grid_cfg.get("max_y", 3)
        total_cells = (MAX_X + 1) * (MAX_Y + 1)

        start_cfg = grid_cfg.get("start", {"x": 0, "y": 0})
        start_x = start_cfg.get("x", 0)
        start_y = start_cfg.get("y", 0)
        from mission_panel import parse_heading
        start_heading = parse_heading(start_cfg.get("heading", 0))   # set by clicking the map

        x, y = start_x, start_y
        heading = start_heading
        moves = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        def print_summary(end_x, end_y):
            coverage_pct = round(100.0 * len(visited) / total_cells, 2)
            print("\n" + "=" * 50)
            print("📊 สรุปผลภารกิจ SLAM Explore สำเร็จ:")
            print(f"   - จุดเริ่มต้น (Start Position): [{start_x}, {start_y}]")
            print(f"   - จุดสิ้นสุด (End Position):   [{end_x}, {end_y}]")
            print(f"   - พื้นที่สำรวจทั้งหมด (Coverage): {coverage_pct}% ({len(visited)}/{total_cells} cells)")
            print("=" * 50 + "\n")

        travel_dir = None      # map direction of the last move (explore order is relative to it)
        came_from = None       # map direction of the edge the robot came in through
        self._looks = []
        try:
            while True:
                if self.panel is not None and not self.panel.checkpoint():  # waits while paused
                    print("\n--> ยกเลิกการสำรวจจากหน้าต่าง Mission Panel")
                    print_summary(x, y)
                    break
                if self.panel is not None and self.panel.round_t0 and self.panel.remaining() <= 0:
                    print("\n--> หมดเวลารอบนี้แล้ว หยุดสำรวจ")
                    self.panel.log("time is up - stopping exploration")
                    print_summary(x, y)
                    break
                self._stop(0.15)

                if not self._camera_ready_for_scan():
                    print("\n--> กล้องไม่มีสัญญาณสด หยุดเพื่อไม่ให้ข้ามเป้า")
                    print_summary(x, y)
                    break

                visited.add((x, y))
                # side order (explore_order) is relative to how the robot first came into a block
                entry_heading.setdefault((x, y), travel_dir if travel_dir is not None else heading)
                self._live_ctx = ((x, y), visited, MAX_X, MAX_Y, heading)
                self.draw_live_grid((x, y), visited, MAX_X, MAX_Y,
                                    gimbal_abs_deg={0: 0, 1: 90, 2: 180, 3: 270}.get(heading, 0))

                rescanned = False
                if (x, y) not in scan_cache:
                    # Do not inspect a close target while the chassis is 5-8 cm from
                    # a side wall: the barrel/crop hides it and the next turn can clip
                    # the foam. Correct dangerous lateral offset before taking pictures.
                    if self.ir.mount == "side":
                        l0, r0 = self._control_side_ir()
                        if any(v is not None and v < self.SIDE_SAFE_CM + 2.0 for v in (l0, r0)):
                            self._log(f"off-centre before scan (Sharp L {l0 or 0:.0f} / R {r0 or 0:.0f} cm) "
                                      "- centring before target detection")
                            self._center_in_cell(self.heading_to_yaw(heading),
                                                 duration=max(2.0, self.CENTER_TIME_S))
                    scan_started = time.time()
                    # the way it came in is known open: not looked at again (the block behind
                    # was looked at from inside already)
                    known = {}
                    prev_cell = None if came_from is None else (x + moves[came_from][0], y + moves[came_from][1])
                    if came_from is not None and not self.scan_came_from and \
                            frozenset(((x, y), prev_cell)) in driven:
                        known[{0: "front", 1: "right", 2: "back", 3: "left"}[(came_from - heading) % 4]] = 900
                    corridor = self._corridor_known((x, y), heading, travel_dir, prev_cell, scan_cache, driven,
                                                    blocked, MAX_X, MAX_Y, CELL_SIZE)
                    if corridor:
                        known.update(corridor[0])
                        self._log(f"corridor {(x, y)}: walls both sides (Sharp L {corridor[1][0]:.0f} / "
                                  f"R {corridor[1][1]:.0f} cm), way ahead known - camera left + right only")
                    surrounding = self.scan_surroundings_with_gimbal(known=known)
                    # STOP/Finish can arrive while a multi-direction scan is in progress.
                    # Never cache or announce a partly aborted scan as SCAN_DONE: zero readings
                    # then leave no open edge and used to make a 6x6 map finish at the start.
                    if self.panel is not None and not self.panel.checkpoint():
                        self.panel.log(f"scan cancelled at cell={(x, y)} - map result discarded")
                        break
                    if self.panel is not None:
                        self.panel.log(f"SCAN_DONE cell={(x, y)} elapsed={time.time() - scan_started:.2f}s "
                                       f"mode={'sweep' if self.SWEEP_SCAN else 'stops'}")
                    self._last_scan = ((x, y), dict(surrounding))
                    rescanned = True
                    scan_cache[(x, y)] = {(heading + off) % 4: surrounding[rel]
                                          for rel, off in (("front", 0), ("right", 1), ("back", 2), ("left", 3))}
                    rel_off = {"front": 0, "right": 1, "back": 2, "left": 3}
                    direct_scan_dirs[(x, y)] = {
                        (heading + off) % 4 for rel, off in rel_off.items()
                        if rel not in known and surrounding.get(rel) is not None and
                        surrounding.get(rel) >= 60
                    }
                    self.recenter(surrounding)          # back to the middle of the block
                surrounding = {rel: scan_cache[(x, y)][(heading + off) % 4]
                               for rel, off in (("front", 0), ("right", 1), ("back", 2), ("left", 3))}

                if self.panel is not None and rescanned:
                    self.panel.map.mark_scan((x, y), heading, surrounding, FRONT_WALL_MM, SIDE_OPEN_MM)
                    deferred = getattr(self, "_deferred_motion_scan", None)
                    self._deferred_motion_scan = None
                    if deferred is not None:
                        self._check_motion_sightings(*deferred[:4], limit=deferred[4])
                self._budget_check(len(visited), total_cells)
                if self.panel is not None:
                    self._drive_in_hints((x, y), heading)   # cards seen on the way in: the close look aims at them
                    self._edge_on_here((x, y), heading)   # a card seen edge-on before, face-on from here
                if self.panel is not None and self.verify_enabled:
                    unclear = self.verify_sweep((x, y))   # something seen in / next to this block: look down, turn round
                    if unclear and self.panel.remaining() > 30:
                        nx_, ny_ = x, y
                        (nx_, ny_), heading = self._look_back((x, y), heading, unclear, scan_cache, driven, blocked,
                                                              visited, MAX_X, MAX_Y, CELL_SIZE)
                        if (nx_, ny_) != (x, y):
                            log_step(nx_, ny_, heading, self.current_tof_dist_mm, 0, 0, 0,
                                     act_label="LOOK BACK", c_size=CELL_SIZE)
                            came_from = next(h for h, m in moves.items() if m == (x - nx_, y - ny_))
                            travel_dir = (came_from + 2) % 4
                            x, y = nx_, ny_
                            continue          # carry on exploring from the block it looked back from
                    if self.panel.round_no >= 2 and self.panel.all_designated_shot() and \
                            (self.explore_mode != "all" or len(visited) >= total_cells):
                        print("\n--> [Round 2] ยิงเป้าที่กำหนดครบแล้ว หยุดภารกิจ")
                        print_summary(x, y)
                        break

                if rescanned:
                    log_step(x, y, heading, surrounding["front"], surrounding["right"], surrounding["back"],
                             surrounding["left"], act_label="VISIT", c_size=CELL_SIZE)
                # cards this block may shoot and are still up (in it, or straight ahead next door)
                if self.panel is not None:
                    self.shoot_here((x, y), heading)

                # In full-coverage mode never abandon unseen cells for a long mop-up route.
                # The live camera already shoots every reachable target as it is found; the
                # previous policy stopped with 126 s left and 10/36 cells unseen, then spent the
                # rest revisiting standing cards in a no-ammo practice.  Non-coverage modes may
                # still reserve time for a target-only finish.
                if (self.panel is not None and self.panel.round_t0 and self.mop_up and
                        self.explore_mode != "all"):
                    need = self._shoot_reserve_s()
                    if need and self.panel.remaining() < need:
                        self._log(f"{self.panel.remaining():.0f} s left, shooting what was found needs "
                                  f"~{need:.0f} s - stop exploring")
                        print_summary(x, y)
                        break

                # where next: the block with the most unseen wall faces / unknown edges per
                # second of driving. Nothing left worth a visit -> exploring is done (no drive
                # back to the start, no visits to blocks the camera already saw completely)
                opened, walls = self._edge_knowledge(scan_cache, driven, blocked, MAX_X, MAX_Y, CELL_SIZE)
                goal, route = self._next_block((x, y), heading, entry_heading[(x, y)], visited, opened, walls,
                                               blocked, dead_edges, MAX_X, MAX_Y, CELL_SIZE)
                if goal is None:
                    unvisited = sorted((cx, cy) for cx in range(MAX_X + 1) for cy in range(MAX_Y + 1)
                                       if (cx, cy) not in visited)
                    if (self.explore_mode == "all" and unvisited and dead_edges and
                            dead_edge_reopens < 1):
                        self._log(f"reopening {len(dead_edges)} temporarily unsafe open edge(s) "
                                  "for one centred retry before declaring cells unreachable")
                        dead_edges.clear()
                        dead_edge_reopens += 1
                        continue
                    print("-> [Explore] ไม่มีบล็อกที่ต้องดูเพิ่มแล้ว")
                    if self.explore_mode == "all":
                        self._log(f"all {total_cells} cells visited - exploring done" if not unvisited else
                                  f"every reachable cell visited; no safe route to {unvisited}")
                    else:
                        self._log("every wall face seen - exploring done")
                    print_summary(x, y)
                    break
                if len(route) > 2:
                    self._log(f"next: {goal} ({len(route) - 1} moves) - most still unseen per second")
                nxt = route[1]
                d = next(h for h, m in moves.items() if m == (nxt[0] - x, nxt[1] - y))
                edge = frozenset(((x, y), nxt))
                known_way = edge in driven
                moved = False
                # A corner stop means make room and retry; it does not prove this edge is a wall.
                retryable_failure = False
                for attempt in range(2):
                    move_speed = self.CELL_SPEED if known_way or nxt in visited else self.UNKNOWN_SPEED
                    direct_mm = scan_cache.get((x, y), {}).get(d)
                    # The stationary map sweep just measured this exact direction as open.
                    # Do not spend another 1-3 seconds sweeping three almost-identical rays;
                    # the fresh centre ToF, camera and corner IR remain active throughout.
                    measured_here = d in direct_scan_dirs.get((x, y), set())
                    guard = False if known_way else (
                        "direct" if measured_here and direct_mm and direct_mm > self.wall_mm else True)
                    moved, heading = self.move_cell(d, heading, CELL_SIZE, camera_guard=guard,
                                                    pos=(x, y), walls=walls, opened=opened, bounds=(MAX_X, MAX_Y),
                                                    speed=move_speed)
                    if moved:
                        break
                    retryable_failure = self._retryable_move_failure()
                    self._log(f"move {(x, y)} -> {nxt} failed ({self.last_move_note}), retry {attempt + 1}")
                    if not retryable_failure:
                        break
                    # A fresh ToF/camera contradiction needs a new stationary map scan, not
                    # repeated driving at the same obstacle with the same cached open edge.
                    if any(s in (self.last_move_note or "") for s in
                           ("clearance scan obstacle", "camera ", "ToF obstacle", "no fresh ToF")):
                        break
                if not moved:
                    # never pretend: the map would think the robot is somewhere it is not
                    self._log(f"move {(x, y)} -> {nxt} gave up: {self.last_move_note or 'blocked'}")
                    log_step(x, y, heading, self.current_tof_dist_mm, 0, 0, 0,
                             act_label=f"BLOCKED ({self.last_move_note or 'blocked'})", c_size=CELL_SIZE)
                    if retryable_failure:
                        dead_edges.add(edge)
                        self._log(f"edge {(x, y)}->{nxt} left open on the map; temporarily unsafe, not a wall")
                        if any(s in (self.last_move_note or "") for s in
                               ("clearance scan obstacle", "camera ", "ToF obstacle", "no fresh ToF")):
                            scan_cache.pop((x, y), None)
                            direct_scan_dirs.pop((x, y), None)
                            self._log(f"cached scan at {(x, y)} invalidated; re-scan before routing again")
                    else:
                        blocked.add((x, y, d))
                        if known_way:
                            dead_edges.add(edge)
                    continue
                driven.add(edge)
                dead_edges.discard(edge)
                travel_dir, came_from = d, (d + 2) % 4
                x, y = nxt
                if (x, y) in visited:
                    log_step(x, y, heading, self.current_tof_dist_mm, 300.0, 300.0, 300.0,
                             act_label="RETRACE", c_size=CELL_SIZE)

        except KeyboardInterrupt:
            print("\n--> ยกเลิกการสำรวจโดยผู้ใช้")
        finally:
            try:
                self._stop(0.2)
                self.reset_gimbal()
            except KeyboardInterrupt:
                pass

        # cards found while exploring but not hit (seen from too far, or the aim lost it):
        # drive next to each one and shoot it while round-1 time is left
        if self.panel is not None and self.panel.round_no < 2:
            try:
                (x, y), heading = self._second_looks((x, y), heading)
            except Exception as e:
                import traceback
                self._log(f"second look failed: {e!r} | " + traceback.format_exc().replace("\n", " | "))
        if self.mop_up and self.panel is not None and self.shooter is not None:
            try:
                x, y = self._mop_up_round1((x, y), heading)
            except Exception as e:
                import traceback
                self._log(f"round-1 mop-up failed: {e!r} | " + traceback.format_exc().replace("\n", " | "))

        if self.panel is not None:
            self.panel.worker.record_dir = None          # the round is over: stop recording frames
        report = {
            "start_grid": [start_x, start_y],
            "end_grid": [x, y],
            "visited_cells": len(visited),
            "total_grid_cells": total_cells,
            "coverage_percent": round(100.0 * len(visited) / total_cells, 2) if total_cells else 0.0,
            "exploration_log_csv": csv_path,
        }
        return report
