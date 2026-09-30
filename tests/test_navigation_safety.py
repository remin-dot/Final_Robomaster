import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation_safety import braking_speed, corridor_obstacle, information_rate


class NavigationSafetyTests(unittest.TestCase):
    def test_corridor_blocks_centre_obstacle(self):
        self.assertAlmostEqual(
            corridor_obstacle([(-15, 900), (0, 320), (15, 850)], 0.20, 0.45),
            0.32,
        )

    def test_corridor_ignores_point_outside_inflated_width(self):
        self.assertIsNone(corridor_obstacle([(45, 400)], 0.20, 0.45))

    def test_braking_speed_respects_stopping_equation(self):
        speed = braking_speed(0.55, 0.20, 0.60, reaction_s=0.10, brake_mps2=0.60)
        stopping = speed * 0.10 + speed * speed / (2.0 * 0.60)
        self.assertLessEqual(stopping, 0.35 + 1e-9)
        self.assertLessEqual(speed, 0.60)

    def test_braking_speed_is_zero_inside_stop_distance(self):
        self.assertEqual(braking_speed(0.19, 0.20, 0.60), 0.0)

    def test_next_best_view_can_beat_nearest_cell(self):
        near = information_rate(value=1.0, travel_s=2.6, scan_s=4.0)
        farther_but_useful = information_rate(value=4.0, travel_s=5.2, scan_s=4.0)
        self.assertGreater(farther_but_useful, near)


if __name__ == "__main__":
    unittest.main()
