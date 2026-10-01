import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chassis import ChassisController


class FullCoverageTests(unittest.TestCase):
    def controller(self):
        controller = ChassisController.__new__(ChassisController)
        controller.explore_mode = "all"
        controller.explore_order = ["front", "right", "left", "back"]
        # Production setting: turn the chassis to face every move so front ToF and both
        # corner sensors protect it.  Coverage must still avoid needless revisits.
        controller.strafe = False
        controller.SCAN_COST_S = 4.0
        controller._queue_spot = lambda *args: self.fail("coverage must not detour for shooting")
        controller._hard_card_blocks = lambda: set()
        controller._unseen_faces = lambda *args: ([], 0)
        return controller

    def test_full_coverage_does_not_revisit_target_cell_just_to_get_closer(self):
        controller = self.controller()
        controller._hard_card_blocks = lambda: {(1, 0)}

        opened = {
            frozenset(((0, 0), (1, 0))),
            frozenset(((1, 0), (2, 0))),
        }
        goal, route = controller._next_block(
            (0, 0), 1, 1, {(0, 0), (1, 0)}, opened, set(), set(), set(), 2, 0, 0.6
        )

        self.assertEqual(goal, (2, 0))
        self.assertEqual(route, [(0, 0), (1, 0), (2, 0)])

    def test_face_forward_open_6x6_uses_minimum_35_moves_without_revisits(self):
        controller = self.controller()
        opened = set()
        for x in range(6):
            for y in range(6):
                if x < 5:
                    opened.add(frozenset(((x, y), (x + 1, y))))
                if y < 5:
                    opened.add(frozenset(((x, y), (x, y + 1))))

        directions = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
        position = (0, 0)
        travel_direction = heading = 0
        visited = {position}
        path = [position]
        while len(visited) < 36:
            _, route = controller._next_block(
                position, heading, travel_direction, visited, opened, set(), set(), set(), 5, 5, 0.6
            )
            next_position = route[1]
            delta = (next_position[0] - position[0], next_position[1] - position[1])
            travel_direction = directions[delta]
            position = next_position
            visited.add(position)
            path.append(position)

        self.assertEqual(len(path) - 1, 35)
        self.assertEqual(len(path), len(set(path)))

    def test_target_scan_reuses_gimbal_settle_time(self):
        seen = []

        class Panel:
            detector = SimpleNamespace(gimbal_pitch_deg=0.0)
            last_guesses = []

            @staticmethod
            def checkpoint():
                return True

            @staticmethod
            def observe_targets(_pos, _angle, **kwargs):
                seen.append(kwargs["settle_ts"])
                return []

        controller = ChassisController.__new__(ChassisController)
        controller.panel = Panel()
        controller.shooter = None
        controller.current_tof_dist_mm = 900
        controller._live_ctx = ((0, 0), set(), 5, 5, 0)
        controller._looks = []
        controller._guesses = []
        controller._not_cards = []
        controller._check_glimpses = lambda *_args: []
        controller._look_for_targets(0, settle_ts=123.0)

        self.assertEqual(seen, [123.0])

    def test_two_candidates_are_checked_and_stale_yaw_is_recovered(self):
        """Do not discard a red square because another candidate was queued first."""
        looks = []
        red_square = SimpleNamespace(kind="red square", color="red")

        class Map:
            targets = {}

            @staticmethod
            def project(_pos, _angle, _dist):
                return 0.3, 0.3

            @staticmethod
            def _find_card(*_args, **_kwargs):
                return None

        class Panel:
            detector = SimpleNamespace(gimbal_pitch_deg=0.0)
            map = Map()
            last_guesses = []
            last_single = [
                {"kind": "blue circle", "color": "blue", "bearing": -20.0,
                 "elevation": 0.0, "distance": 0.4},
                {"kind": "red square", "color": "red", "bearing": 20.0,
                 "elevation": 0.0, "distance": 0.4},
            ]

            @staticmethod
            def checkpoint():
                return True

            @staticmethod
            def observe_targets(_pos, angle, **_kwargs):
                looks.append(round(angle))
                # The moving frame was 8 degrees stale; the right-hand retry finds it.
                return [red_square] if round(angle) == 28 else []

            @staticmethod
            def log(_msg):
                pass

        controller = ChassisController.__new__(ChassisController)
        controller.panel = Panel()
        controller._candidate_checked = set()
        controller.MOTION_CHECKS = 2
        controller.CLOSE_PITCH_MIN_DEG = -25.0
        controller.GIMBAL_DPS = 200.0
        controller.CANDIDATE_CONFIRM_FRAMES = 5
        controller.CANDIDATE_CONFIRM_MIN_HITS = 2
        controller.CANDIDATE_RETRY_YAW_DEG = 8.0
        controller.CANDIDATE_RETRY_FRAMES = 3
        controller.current_tof_dist_mm = 900
        controller._gimbal_moveto = lambda **_kwargs: True

        with patch("chassis.time.sleep"):
            found = controller._check_glimpses((0, 0), 0, 0.0, -5.0)

        self.assertIn(340, looks)  # first candidate was checked
        self.assertIn(20, looks)   # second candidate was not discarded
        self.assertIn(28, looks)   # bounded yaw recovery found the delayed red square
        self.assertEqual(found, [red_square])

    def test_shoot_here_does_not_mark_target_tried_before_camera_can_fire(self):
        target = {
            "id": "red square", "kind": "red square", "cell": [1, 1],
            "x_m": 0.9, "y_m": 0.9, "shot": False,
        }

        class Map:
            tile = 0.6
            targets = {"red square": target}

            @staticmethod
            def target_list():
                return [target]

            @staticmethod
            def in_reach(*_args):
                return True

        panel = SimpleNamespace(
            map=Map(), selected={"red square"}, round_t0=None,
            checkpoint=lambda: True, remaining=lambda: 600,
            log=lambda _msg: None,
        )
        controller = ChassisController.__new__(ChassisController)
        controller.panel = panel
        controller.shooter = object()
        controller.reach_cells = 1
        controller.shoot_order = "fastest"
        controller._tried_from = set()
        controller._tried_inside = set()
        controller.CLOSE_SHOOT_SEARCH_OFFSETS = (0.0,)
        controller.SHOOT_SEARCH_OFFSETS = (0.0,)
        controller.CLOSE_PITCH_MIN_DEG = -25.0
        controller.card_aim_height_m = 0.14
        controller.config = {"vision": {"camera_height_m": 0.25}}
        controller.GIMBAL_DPS = 200.0
        controller._card_done = lambda _t: False
        controller._gimbal_moveto = lambda **_kwargs: True
        controller._draw_live_with_gimbal = lambda _yaw: None
        checked = []

        def look(*_args, **_kwargs):
            checked.append(("red square", (1, 1)) in controller._tried_from)
            return []

        controller._look_for_targets = look
        controller.shoot_here((1, 1), 0)

        self.assertEqual(checked, [False])
        self.assertIn(("red square", (1, 1)), controller._tried_inside)


if __name__ == "__main__":
    unittest.main()
