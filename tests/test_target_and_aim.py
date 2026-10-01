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
    def test_final2_profile_uses_repository_detector_settings(self):
        det = TargetDetector({"vision": {
            "detector_profile": "final2",
            # These conflicting values prove that the named source profile wins.
            "color_model": "veto",
            "processing_width": 512,
            "refine_roi": True,
            "shape_method": "area",
            "roi_classifier": True,
            "morph_kernel_px": 5,
        }})
        self.assertTrue(det.final2_profile)
        self.assertEqual(det.color_model, "hsv")
        self.assertEqual(det.processing_width, 640)
        self.assertFalse(det.refine_roi)
        self.assertEqual(det.shape_method, "contour")
        self.assertIsNone(det.roi_clf)
        self.assertEqual(det.kernel.shape, (3, 3))
        self.assertEqual(det.max_elevation_deg, 2.0)

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

    def test_close_same_cell_squares_of_every_colour_are_shape_sure(self):
        """No square colour may remain SHAPE? because a noisy map inferred a slanted wall."""
        for color in ("blue", "red", "yellow", "green"):
            with self.subTest(color=color):
                mission_map = MissionMap()
                target = mission_map.add_observation(
                    color, "square", (0, 0), 0, 0.35, view_deg=70.0, evidence=3,
                )
                self.assertTrue(target["confirmed"])
                self.assertTrue(mission_map.shape_sure(target["id"]))

    def test_slanted_square_outside_own_cell_still_needs_better_view(self):
        mission_map = MissionMap()
        target = mission_map.add_observation(
            "red", "square", (0, 0), 0, 0.75, view_deg=70.0, evidence=3,
        )
        self.assertTrue(target["confirmed"])
        self.assertFalse(mission_map.shape_sure(target["id"]))

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

    def test_round2_load_seeds_shooter_targets_and_resets_hits(self):
        mission_map = MissionMap()
        data = {
            "grid_size": [6, 6], "tile_m": 0.6, "start_heading": 0,
            "walls": [], "open_edges": [],
            "targets": [{
                "id": "blue rect_wide", "kind": "blue rect_wide",
                "color": "blue", "shape": "rect_wide",
                "x_m": 2.9, "y_m": 1.8, "cell": [4, 3],
                "observations": 4, "shot": True, "confirmed": True,
                "seen_from": [5, 3], "best_dist_m": 0.7,
                "views": [{"cell": [5, 3], "dist_m": 0.7}],
                "best_view_deg": 12.0, "shape_votes": {"rect_wide": 5.0},
            }],
        }
        mission_map.load_round(data)
        target = mission_map.target_list()[0]
        self.assertEqual(target["id"], "blue rect_wide")
        self.assertFalse(target["shot"])
        self.assertTrue(mission_map.targets[target["id"]]["loaded_from_round1"])

    def test_round2_saved_rectangle_kind_survives_oblique_live_view(self):
        mission_map = MissionMap()
        data = {
            "grid_size": [6, 6], "tile_m": 0.6, "start_heading": 0,
            "walls": [], "open_edges": [],
            "targets": [{
                "id": "blue rect_wide", "kind": "blue rect_wide",
                "color": "blue", "shape": "rect_wide",
                "x_m": 0.3, "y_m": 0.9, "cell": [0, 1],
                "observations": 3, "confirmed": True, "seen_from": [0, 0],
                "best_dist_m": 0.6, "views": [], "best_view_deg": 10.0,
                "shape_votes": {"rect_wide": 3.0},
            }],
        }
        mission_map.load_round(data)
        mission_map.add_observation("blue", "square", (0, 0), 0, 0.6,
                                    view_deg=55.0, evidence=5)
        self.assertIn("blue rect_wide", mission_map.targets)
        self.assertEqual(mission_map.targets["blue rect_wide"]["kind"], "blue rect_wide")


class AimControllerTests(unittest.TestCase):
    def test_all_shapes_selected_keeps_lock_when_close_shape_label_changes(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.panel = SimpleNamespace(selected={
            "green circle", "green rect_wide", "green rect_tall", "green square",
        })
        det = SimpleNamespace(is_card=True, kind="green rect_tall", color="green",
                              shape="rect_tall", bearing_deg=0.2, elevation_deg=-0.1)
        picked = shooter._pick([det], "green circle", (0.0, 0.0), 12.0)
        self.assertIs(picked, det)

    def test_confirmed_close_green_circle_tracks_through_tall_rect_shape_flip(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.close_lock_m = 0.55
        shooter._active_target_id = "green circle"
        target = {
            "kind": "green circle", "confirmed": True, "shot": False,
            "cell": [2, 1], "observations": 12, "best_dist_m": 0.17,
            "shape_votes": {"circle": 28.574},
        }
        shooter.panel = SimpleNamespace(
            selected={"green circle"},
            map=SimpleNamespace(targets={"green circle": target}, robot=(2, 1)),
            detector=SimpleNamespace(last_frame_size=(640, 360), plate={"circle": (0.07, 0.07)},
                                     focal_px=lambda width: 288.0),
        )
        det = SimpleNamespace(is_card=True, kind="green rect_tall", color="green",
                              shape="rect_tall", bearing_deg=0.4, elevation_deg=0.2,
                              distance_m=0.18, bbox=(280, 120, 60, 100))
        picked = shooter._pick([det], "green circle", (0.0, 0.0), 5.0)
        self.assertIs(picked, det)

    def test_confirmed_close_green_candidate_becomes_lock_point(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.close_lock_m = 0.55
        shooter._active_target_id = "green circle"
        target = {
            "kind": "green circle", "confirmed": True, "shot": False,
            "cell": [2, 1], "observations": 17, "best_dist_m": 0.18,
            "shape_votes": {"circle": 54.905},
        }
        shooter.panel = SimpleNamespace(
            selected={"green circle"},
            map=SimpleNamespace(targets={"green circle": target}, robot=(2, 1)),
            detector=SimpleNamespace(last_frame_size=(640, 360), plate={"circle": (0.07, 0.07)},
                                     focal_px=lambda width: 288.0),
        )
        det = SimpleNamespace(is_card=False, guess="square", kind="green unknown", color="green",
                              shape="unknown", bearing_deg=0.4, elevation_deg=0.2,
                              distance_m=None, bbox=(260, 80, 120, 110), extra={})
        picked = shooter._pick([det], "green circle", (0.0, 0.0), 5.0)
        self.assertIs(picked, det)
        self.assertAlmostEqual(det.distance_m, 288.0 * 0.07 / 110.0)

    def test_unconfirmed_or_far_green_does_not_override_selected_shape(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.close_lock_m = 0.55
        shooter._active_target_id = "green circle"
        target = {
            "kind": "green circle", "confirmed": True, "shot": False,
            "cell": [2, 1], "observations": 2, "best_dist_m": 0.17,
            "shape_votes": {"circle": 2.0},
        }
        shooter.panel = SimpleNamespace(
            selected={"green circle"},
            map=SimpleNamespace(targets={"green circle": target}, robot=(2, 1)),
        )
        det = SimpleNamespace(is_card=True, kind="green rect_tall", color="green",
                              shape="rect_tall", bearing_deg=0.4, elevation_deg=0.2)
        self.assertIsNone(shooter._pick([det], "green circle", (0.0, 0.0), 5.0))

    def test_confirmed_close_green_shape_flip_passes_strict_fire_gate(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.close_lock_m = 0.55
        shooter._active_target_id = "green circle"
        shooter.fire_require_full_visibility = True
        shooter.fire_require_gimbal_still = False
        shooter.fire_max_detection_age_s = 1.0
        shooter.gimbal_hist = []
        target = {
            "kind": "green circle", "confirmed": True, "shot": False,
            "cell": [2, 1], "observations": 12, "best_dist_m": 0.17,
            "shape_votes": {"circle": 28.574},
        }
        shooter.panel = SimpleNamespace(
            selected={"green circle"},
            map=SimpleNamespace(targets={"green circle": target}, robot=(2, 1)),
            detector=SimpleNamespace(last_frame_size=(640, 360)),
            worker=SimpleNamespace(det_ts=0.0),
        )
        det = SimpleNamespace(is_card=True, guess=None, kind="green rect_tall", color="green",
                              shape="rect_tall", bbox=(280, 120, 60, 100), extra={})
        ok, why = shooter._fire_candidate_ok(det, "green circle", "green circle")
        self.assertTrue(ok, why)

    def test_confirmed_close_full_frame_green_candidate_passes_fire_gate(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.close_lock_m = 0.55
        shooter._active_target_id = "green circle"
        shooter.fire_require_full_visibility = True
        shooter.fire_require_gimbal_still = False
        shooter.fire_max_detection_age_s = 1.0
        shooter.gimbal_hist = []
        target = {
            "kind": "green circle", "confirmed": True, "shot": False,
            "cell": [2, 1], "observations": 17, "best_dist_m": 0.18,
            "shape_votes": {"circle": 54.905},
        }
        shooter.panel = SimpleNamespace(
            selected={"green circle"},
            map=SimpleNamespace(targets={"green circle": target}, robot=(2, 1)),
            detector=SimpleNamespace(last_frame_size=(640, 360)),
            worker=SimpleNamespace(det_ts=0.0),
        )
        det = SimpleNamespace(is_card=False, guess="square", kind="green unknown", color="green",
                              shape="unknown", bbox=(250, 80, 140, 130), extra={})
        ok, why = shooter._fire_candidate_ok(det, "green circle", "green circle")
        self.assertTrue(ok, why)

        det.extra = {"clipped": True}
        ok, why = shooter._fire_candidate_ok(det, "green circle", "green circle")
        self.assertFalse(ok)
        self.assertIn("edge", why)

    def test_close_full_card_needs_three_locks_but_far_card_keeps_four(self):
        shooter = TargetShooter.__new__(TargetShooter)
        shooter.lock_frames = 4
        shooter.close_lock_frames = 3
        shooter.close_lock_m = 0.55
        self.assertEqual(shooter._lock_needed(SimpleNamespace(distance_m=0.35)), 3)
        self.assertEqual(shooter._lock_needed(SimpleNamespace(distance_m=0.80)), 4)

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
