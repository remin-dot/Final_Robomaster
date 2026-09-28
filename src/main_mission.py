"""Hostage-rescue maze mission (Assignment: 6x6 tiles, shoot the villain targets).

Opens the Mission Panel with a Connect screen, like RoboFinal's final control
panel (branch feature/final):

  Connect screen   RoboMaster robot (Wi-Fi AP / router STA / USB), webcam or
                   demo; Round 1 or 2; Blaster armed; Connect
  Header           Round 1 / Round 2, state, countdown, Start round N,
                   Pause / Resume, Finish, Save, STOP (or Space), Disconnect
  Camera           live robot camera (30 fps) with target segmentation
  Actions tab      Look around now, Aim & shoot here, Fire once, Map size,
                   Blaster armed, gimbal Left / Right / Up / Down / Centre

    python3 src/main_mission.py                       # panel, choose and Connect
    python3 src/main_mission.py --connect             # connect to the robot straight away
    python3 src/main_mission.py --round 2 --connect --autostart
    python3 src/main_mission.py --source webcam --connect   # this computer's camera, no robot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from mission_panel import run_app


def parse_args():
    ap = argparse.ArgumentParser(description="RoboMaster rescue mission")
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    ap.add_argument("--source", choices=("robot", "webcam", "demo"), default="robot")
    ap.add_argument("--connection", choices=("ap", "sta", "rndis"), default=None)
    ap.add_argument("--connect", action="store_true", help="connect without pressing Connect")
    ap.add_argument("--autostart", action="store_true", help="connect and start the round straight away")
    ap.add_argument("--no-shoot", action="store_true", help="dry run: detect + aim, never fire")
    ap.add_argument("--resolution", choices=("360p", "540p", "720p"), default=None)
    ap.add_argument("--ui", choices=("pygame", "opencv"), default="pygame",
                    help="panel window: pygame (default) or the older OpenCV one")
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config()
    if args.source == "demo":
        config.setdefault("data_collection", {})["data_dir"] = "data/demo"  # never overwrite real rounds
    run_app(config, source=args.source, round_no=args.round, connection=args.connection,
            resolution=args.resolution, connect=args.connect, autostart=args.autostart,
            armed=False if args.no_shoot else None, ui=args.ui)


if __name__ == "__main__":
    main()
