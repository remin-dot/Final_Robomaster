import os
import sys
import threading
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chassis import ChassisController
from sharp_ir import CornerSignal, SharpIR


class FakeIR:
    corners_enabled = True

    def __init__(self, left=False, right=False, left_cm=30.0, right_cm=30.0):
        self.values = {"left": left, "right": right}
        self.distances = (left_cm, right_cm)

    def corner_near(self, side):
        return self.values[side]

    def latest(self):
        return self.distances


class CornerIRSafetyTests(unittest.TestCase):
    def test_active_low_io_switches_between_clear_and_wall(self):
        signal = CornerSignal("io")
        signal.update(500, 1)
        self.assertFalse(signal.near(active_low=True, threshold=512))
        signal.update(500, 0)
        self.assertTrue(signal.near(active_low=True, threshold=512))

    def test_repaired_left_and_right_io_are_active_low(self):
        sensors = SharpIR.__new__(SharpIR)
        sensors.corners_enabled = True
        sensors.corner_port = {"left": (1, 1), "right": (2, 2)}
        sensors.corner_threshold = {"left": 512, "right": 530}
        sensors.corner_active_low = {"left": True, "right": True}
        sensors.lock = threading.Lock()
        sensors.corner_sig = {"left": CornerSignal("io"), "right": CornerSignal("io")}
        sensors.corner_sig["left"].update(340, 1)
        sensors.corner_sig["right"].update(491, 1)
        self.assertFalse(sensors.corner_near("left"))
        self.assertFalse(sensors.corner_near("right"))
        sensors.corner_sig["right"].update(590, 0)
        self.assertTrue(sensors.corner_near("right"))

    def test_one_corner_is_an_emergency_without_other_sensor_confirmation(self):
        controller = ChassisController.__new__(ChassisController)
        controller.ir = FakeIR(right=True)
        self.assertEqual(controller._front_corner_hits(), ("right",))

    def test_both_clear_allows_motion(self):
        controller = ChassisController.__new__(ChassisController)
        controller.ir = FakeIR()
        self.assertEqual(controller._front_corner_hits(), ())

    def test_left_corner_shifts_right_before_retry(self):
        controller = ChassisController.__new__(ChassisController)
        controller.ir = FakeIR(left=True)
        controller.TURN_CLEAR_SIDE_CM = 12.0
        controller._log = lambda _message: None
        nudges = []
        controller.nudge = lambda angle, distance: nudges.append((angle, distance))
        controller._make_corner_room(("left",))
        self.assertEqual(nudges, [(90.0, 0.04)])

    def test_corner_stop_is_retryable_not_a_wall(self):
        controller = ChassisController.__new__(ChassisController)
        controller.last_move_note = "front-left IR emergency stop at 0.17 m"
        self.assertTrue(controller._retryable_move_failure())
        controller.last_move_note = "clearance scan obstacle 0.32 m ahead"
        self.assertFalse(controller._retryable_move_failure())

    def test_single_corner_warns_before_persistent_stop(self):
        controller = ChassisController.__new__(ChassisController)
        controller.CORNER_HOLD_STOP_S = 0.35
        since = {"left": 10.0, "right": None}
        self.assertFalse(controller._corner_hard_stop(("left",), since, 10.20))
        self.assertTrue(controller._corner_hard_stop(("left",), since, 10.36))
        self.assertTrue(controller._corner_hard_stop(("left", "right"), since, 10.01))

    def test_corner_trigger_aborts_a_turn_before_drive_command(self):
        controller = ChassisController.__new__(ChassisController)
        controller.ir = FakeIR(left=True)
        controller.current_yaw = 0.0
        controller.TURN_OK_DEG = 4.0
        controller.TURN_TOL_DEG = 1.5
        controller.TURN_MAX_DPS = 90.0
        controller.TURN_TIGHT_DPS = 45.0
        controller._room_to_turn = lambda: True
        controller._make_corner_room = lambda _hits: None
        stopped = []
        messages = []
        controller._stop = lambda delay=0: stopped.append(delay)
        controller._log = messages.append

        self.assertFalse(controller.turn_to_absolute_yaw(90.0, attempts=1))
        self.assertTrue(stopped)
        self.assertTrue(any("emergency stop" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
