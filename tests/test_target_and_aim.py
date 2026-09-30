import os
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mission_panel import MissionMap
from target_shooter import TargetShooter, damped_aim_speed
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

    def test_agreeing_frames_at_one_still_look_confirm_immediately(self):
        mission_map = MissionMap()
        mission_map.min_observations = 2
        mission_map.round_shape_min_observations = 3
        target = mission_map.add_observation(
            "green", "circle", (0, 0), 0, 0.4, evidence=3,
        )
        self.assertTrue(target["confirmed"])
        self.assertEqual(target["observations"], 3)

    def test_known_open_edge_never_serializes_as_wall(self):
        mission_map = MissionMap()
        edge = frozenset(((0, 0), (1, 0)))
        mission_map.walls.add(edge)       # simulate a later noisy ToF contradiction
        mission_map.open_edges.add(edge)
        saved = mission_map.to_json()
        self.assertEqual(saved["walls"], [])
        self.assertEqual(len(saved["open_edges"]), 1)

    def test_traversed_edge_never_serializes_as_wall(self):
        mission_map = MissionMap()
        edge = frozenset(((0, 0), (0, 1)))
        mission_map.walls.add(edge)
        mission_map.traversed.add(edge)
        saved = mission_map.to_json()
        self.assertEqual(saved["walls"], [])


class AimControllerTests(unittest.TestCase):
    def test_no_ammo_practice_counts_only_validated_fire_as_hit(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.assume_hit_after_locked_fire = True
        marked, phases, logs = [], [], []
        shooter.panel = SimpleNamespace(
            map=SimpleNamespace(mark_shot=marked.append),
            aim=SimpleNamespace(set_phase=phases.append),
            log=logs.append,
        )
        self.assertTrue(shooter._accept_practice_fire("red square", SimpleNamespace(label="red square")))
        self.assertEqual(marked, ["red square"])
        self.assertEqual(phases, ["HIT"])
        self.assertIn("centred locked fire", logs[0])

        shooter.assume_hit_after_locked_fire = False
        self.assertFalse(shooter._accept_practice_fire("green circle", SimpleNamespace(label="green circle")))
        self.assertEqual(marked, ["red square"])

    def test_deadband_stops_jitter(self):
        speed = damped_aim_speed(0.1, 0.0, 0.03, True, 3.5, 100, 2.0, 35, 0.2, 300)
        self.assertEqual(speed, 0.0)

    def test_fine_speed_is_capped(self):
        speed = damped_aim_speed(30, 0.0, 1.0, True, 3.5, 100, 2.0, 35, 0.2, 10000)
        self.assertEqual(speed, 35.0)

    def test_direction_reversal_is_slew_limited(self):
        speed = damped_aim_speed(-20, 20, 0.02, False, 3.5, 100, 2.0, 35, 0.2, 300)
        self.assertEqual(speed, 14.0)

    def test_fresh_camera_detection_needs_no_transient_target_id(self):
        """The map owns identity; raw worker frames prove the selected card is still visible."""
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.fire_require_full_visibility = True
        shooter.fire_require_gimbal_still = False
        shooter.fire_max_detection_age_s = 1.0
        shooter.gimbal_hist = []
        mapped = {"red circle": {"kind": "red circle", "confirmed": True, "shot": False}}
        shooter.panel = SimpleNamespace(
            map=SimpleNamespace(targets=mapped),
            selected={"red circle"},
            detector=SimpleNamespace(last_frame_size=(640, 360)),
            worker=SimpleNamespace(det_ts=0.0),
        )
        det = SimpleNamespace(is_card=True, guess=None, kind="red circle", color="red",
                              shape="circle", bbox=(200, 100, 40, 40), extra={})
        ok, why = shooter._fire_candidate_ok(det, "red circle", "red circle")
        self.assertTrue(ok, why)

        det.kind, det.color, det.shape = "blue circle", "blue", "circle"
        ok, _ = shooter._fire_candidate_ok(det, "red circle", "red circle")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
