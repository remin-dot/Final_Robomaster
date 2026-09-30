import os
import sys
import unittest

import cv2
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mission_panel import MissionMap
from target_shooter import damped_aim_speed
from target_vision import TargetDetector


def contour(draw):
    image = np.zeros((100, 100), np.uint8)
    draw(image)
    found, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found[0]


class TargetShapeTests(unittest.TestCase):
    def test_circle_and_oblique_circle_never_become_square(self):
        circle = contour(lambda im: cv2.circle(im, (50, 50), 15, 255, -1))
        ellipse = contour(lambda im: cv2.ellipse(im, (50, 50), (15, 11), 0, 0, 360, 255, -1))
        self.assertEqual(TargetDetector.classify_shape(circle)[0], "circle")
        self.assertEqual(TargetDetector.classify_shape(ellipse)[0], "circle")

    def test_true_square_still_requires_four_corner_evidence(self):
        square = contour(lambda im: cv2.rectangle(im, (35, 35), (65, 65), 255, -1))
        self.assertEqual(TargetDetector.classify_shape(square)[0], "square")

    def test_round_shapes_need_three_observations(self):
        mission_map = MissionMap()
        mission_map.min_observations = 2
        mission_map.round_shape_min_observations = 3
        second = None
        for _ in range(2):
            second = mission_map.add_observation("green", "circle", (0, 0), 0, 0.4)
        self.assertFalse(second["confirmed"])
        third = mission_map.add_observation("green", "circle", (0, 0), 0, 0.4)
        self.assertTrue(third["confirmed"])


class AimControllerTests(unittest.TestCase):
    def test_deadband_stops_jitter(self):
        speed = damped_aim_speed(0.1, 0.0, 0.03, True, 3.5, 100, 2.0, 35, 0.2, 300)
        self.assertEqual(speed, 0.0)

    def test_fine_speed_is_capped(self):
        speed = damped_aim_speed(30, 0.0, 1.0, True, 3.5, 100, 2.0, 35, 0.2, 10000)
        self.assertEqual(speed, 35.0)

    def test_direction_reversal_is_slew_limited(self):
        speed = damped_aim_speed(-20, 20, 0.02, False, 3.5, 100, 2.0, 35, 0.2, 300)
        self.assertEqual(speed, 14.0)


if __name__ == "__main__":
    unittest.main()
