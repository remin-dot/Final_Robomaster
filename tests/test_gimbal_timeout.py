import os
import sys
import unittest


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

    def drive_speed(self, **_kwargs):
        self.stopped = True


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


if __name__ == "__main__":
    unittest.main()
