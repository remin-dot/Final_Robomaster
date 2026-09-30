import os
import sys
from types import SimpleNamespace
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chassis import ChassisController


class FullCoverageTests(unittest.TestCase):
    def controller(self):
        controller = ChassisController.__new__(ChassisController)
        controller.explore_mode = "all"
        controller.explore_order = ["front", "right", "left", "back"]
        controller.strafe = True
        controller.SCAN_COST_S = 4.0
        controller._queue_spot = lambda *args: self.fail("coverage must not detour for shooting")
        controller._hard_card_blocks = lambda: set()
        controller._unseen_faces = lambda *args: ([], 0)
        return controller

    def test_unvisited_cell_wins_over_hard_target_revisit(self):
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

    def test_open_6x6_uses_minimum_35_moves_without_revisits(self):
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


if __name__ == "__main__":
    unittest.main()
