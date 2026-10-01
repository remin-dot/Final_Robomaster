import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chassis import ChassisController


class FakeAction:
    def __init__(self):
        self.timeout = None

    def wait_for_completed(self, timeout=None):
        self.timeout = timeout
        return False


class FakeGimbal:
    def __init__(self):
        self.stopped = False
        self.moveto_calls = []

    def drive_speed(self, **_kwargs):
        self.stopped = True

    def moveto(self, **kwargs):
        self.moveto_calls.append(kwargs)
        return SuccessfulAction()


class SuccessfulAction:
    def wait_for_completed(self, timeout=None):
        return True


class GimbalTimeoutTests(unittest.TestCase):
    def test_timeout_stops_gimbal_and_returns(self):
        controller = ChassisController.__new__(ChassisController)
        controller.GIMBAL_TIMEOUT_S = 2.5
        controller.ep_gimbal = FakeGimbal()
        messages = []
        controller._log = messages.append
        action = FakeAction()
        self.assertFalse(controller._wait_gimbal(action, "test move"))
        self.assertEqual(action.timeout, 2.5)
        self.assertTrue(controller.ep_gimbal.stopped)
        self.assertIn("failed", messages[0])

    def test_speed_timeout_falls_back_to_sdk_position_move(self):
        controller = ChassisController.__new__(ChassisController)
        controller.GIMBAL_TIMEOUT_S = 2.5
        controller.GIMBAL_TOL_DEG = 1.0
        controller.PITCH_BY_SPEED = True
        controller.gimbal_hist = [(time.time(), 0.0, 0.0)]
        controller.ep_gimbal = FakeGimbal()
        controller.panel = SimpleNamespace(abort=threading.Event())
        controller._log = lambda _msg: None
        controller._gimbal_drive_to = lambda *_args: False

        self.assertTrue(controller._gimbal_moveto(pitch=-5, yaw=90, what="map scan"))
        self.assertEqual(len(controller.ep_gimbal.moveto_calls), 1)
        self.assertEqual(controller.ep_gimbal.moveto_calls[0]["yaw"], 90)

    def test_emergency_stop_does_not_start_gimbal_fallback(self):
        controller = ChassisController.__new__(ChassisController)
        controller.GIMBAL_TOL_DEG = 1.0
        controller.PITCH_BY_SPEED = True
        controller.gimbal_hist = [(time.time(), 0.0, 0.0)]
        controller.ep_gimbal = FakeGimbal()
        abort = threading.Event()
        abort.set()
        controller.panel = SimpleNamespace(abort=abort)
        controller._log = lambda _msg: None
        controller._gimbal_drive_to = lambda *_args: False

        self.assertFalse(controller._gimbal_moveto(pitch=-5, yaw=90, what="map scan"))
        self.assertEqual(controller.ep_gimbal.moveto_calls, [])


if __name__ == "__main__":
    unittest.main()
