# RoboFinal

## RoboMaster camera viewer

Connect the computer to the RoboMaster robot (the default configuration uses
the robot's Wi-Fi access-point mode), then install the dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 src/camera_view.py
```

Press `q` or `Esc` in the camera window to stop. The connection type and video
resolution can also be selected explicitly:

```bash
python3 src/camera_view.py --connection ap --resolution 720p
```

Supported connection types are `ap`, `sta`, and `rndis`; supported resolutions
are `360p`, `540p`, and `720p`.

The viewer automatically looks for a solid red rectangular card. When it finds
one, it draws a green box around the card and shows `RED CARD DETECTED` in the
video window. If a small card is too far away to detect, lower the minimum pixel
area (lower values can also cause more false detections):

```bash
python3 src/camera_view.py --min-card-area 700
```

## Rescue mission panel (Assignment: 6x6 maze, shoot the villain targets)

### Install

```bash
bash tools/macos/setup_robomaster_mac.sh     # macOS (Apple Silicon / Intel), Python 3.8-3.12
python3 -m pip install -r requirements.txt   # Windows / Linux x86_64, Python 3.6-3.8
```

The Mac script (from RoboFinal `feature/final`) builds `.venv` with the official
SDK code and a PyAV-based video decoder, since DJI ships no macOS build.
Check the robot link with `.venv/bin/python tools/macos/check_robomaster.py`.

### Run

```bash
python3 src/main_mission.py                        # control panel: choose, then Connect
python3 src/main_mission.py --connect              # connect to the robot straight away
python3 src/main_mission.py --source webcam --connect   # this computer's camera, no robot
python3 src/main_mission.py --source demo --connect     # sample pictures + simulated walk
```

In VS Code: Run and Debug (Cmd+Shift+D) -> pick a configuration -> F5.

The window is drawn with **pygame** (`src/panel_pygame.py`, widget kit adapted
from RoboFinal's pygame console): resizable, light / dark **Theme**, a real
text box in the map editor and a confirmation dialog for Disconnect. The older
OpenCV window is still there: `--ui opencv`. The **Demo** source is a small
simulated arena (5 cards at fixed places, one of them a red circle that is not
in the sheet set) so the map, selection and round 2 can be tried without the
robot; `python3 src/mission_panel.py --demo a.jpg b.jpg` loops photos instead.

The panel works like RoboFinal's final control panel:

| Part | What it does |
| --- | --- |
| Connect screen | RoboMaster robot (Wi-Fi AP / router STA / USB), Webcam or Demo; Round 1 or 2; Blaster armed; **Connect** (Enter). Connection errors are shown here. |
| Header | Source, Round 1 / Round 2, state (IDLE, RUNNING, PAUSED, DONE), countdown, **Start round N**, Pause / Resume, Finish (end + save), Save, **STOP** (motors off now; also Space), Disconnect (click twice). |
| Camera | Live camera (robot stream or webcam, 30 fps) with target segmentation, LIVE / NO SIGNAL badge, aim box and AIMING / LOCKED / FIRE! banner. |
| Actions tab | Look around now, Aim & shoot here, Fire once, Map size editor, Blaster armed (off = dry run: aims, never fires), gimbal Left / Right / Up / Down / Centre. |
| Select tab | Which cards to shoot: a 4 x 4 grid of the sheet's colours (blue, red, yellow, green) x shapes (circle, wide rect, tall rect, square) - e.g. blue circle + red circle. Presets **Sheet set** (the assignment's blue circle, red wide rect, yellow tall rect, green square), All, None; **Save** writes `vision.shoot_targets`. Locked while a round runs. |
| Targets tab | Every card found: cell, HIT with split time, "to shoot", "not selected", or from round 1. |

Keys: `Space` start / STOP, `P` pause, `M` overlay / mask / raw, `G` map editor,
`S` screenshot, `Q` quit (`Esc` closes a dialog or the editor first).

**Target selection:** every colour x shape card is recognised and put on the
map; only the selected kinds are shot (a card that is not selected is outlined
"not selected" on the camera and never fired at - wrong target = -1). Two cards
of the same kind get their own ids (`blue circle`, `blue circle #2`). Round 2
drives to every round-1 card whose kind is selected at that moment.

**Ignoring the room:** detection skips everything above the top of the white foam
walls (traced column by column, walking up from the floor and bridging card-sized
gaps, so a card in front of a wall is kept) and everything more than
`max_elevation_deg` above the horizon - the cards hang lower than the camera, so
people, tables and chairs are never cards. Where there is no wall the line
follows the floor's far edge. The camera view dims that region and draws its edge
(Overlay / Mask). The horizon moves with the live gimbal pitch. Settings:
`ignore_above_wall`, `wall_s_max`, `wall_v_min`, `wall_margin_frac`,
`wall_max_gap_frac`, `max_elevation_deg` in `vision:`. Cards closer than ~0.4 m
reach the masked strip over the blaster barrel.

**Exploration order:** in each cell the open ways are tried in
`movement.explore_order` (default left, right, front, back), relative to the
heading the robot entered the cell with. A cell with open cells on both sides and
a way ahead is done left side -> back -> right side -> back -> ahead, so side
cells are not left for a long backtrack. Each cell is scanned once; coming back
reuses that scan. Put `right` first to start on the right.

**Close look:** when something was seen in the robot's block (or the next one
with no wall between), the camera looks down (`vision.verify_pitch_deg`, -12 deg:
cards hang below the camera) and turns all the way round in 8 steps. A card seen
again is confirmed (and its position corrected); a sighting that is not there is
dropped from the map.

**No more wrong turns into walls:** turns used to stop after a fixed 4 s at
<= 45 deg/s, so every U-turn ended ~40 deg short (reproduced in a test: 43 deg).
Now the time grows with the angle, the result is checked and re-tried, the yaw is
read at 20 Hz, and a heading more than 12 deg off while driving is corrected
first. The **camera** also measures the wall in the robot's path from the height
of the foam wall's top edge (`vision.wall_rise_m`, re-learnt from the ToF at every
front scan): it will not start a move toward a wall closer than 0.40 m and stops
when one gets within 0.22 m - but only when the ToF agrees (the camera alone
never blocks a move). Columns where the white runs off the top of the picture
(a side wall close by, or the camera looking down) give no distance, and none is
measured while the gimbal is tilted; waking the robot re-levels the gimbal.
The Sensors panel shows it as "cam wall".

**Floor lines:** tape on the floor is never a card - long thin strips and
L-shapes fail the shape checks, and anything whose height above the floor
(from its distance and elevation, `vision.camera_height_m`) is below
`card_min_height_m` (0.06 m) or above `card_max_height_m` is dropped.

**Round 2 route:** five algorithms plan the route on the round-1 map - greedy
nearest, greedy cover (most targets per second), DFS branch & bound, BFS over
(cell, targets shot) and Dijkstra over (cell, heading, targets shot) - and are
scored with one time model (moves, turns, stops, aiming, unproven firing spots,
passages round 1 never saw open). The cheapest wins; one stop can shoot several
targets. After every stop, blocked passage or miss it plans again from where the
robot is. The comparison is in the log and in `round2_targets.json`
(`route_plan`). On 183 random mazes the exact Dijkstra route was never beaten.

**Between rounds:** when a round ends the robot is released
(`movement.release_after_round`): the gimbal motors sleep so it can be turned
straight by hand, and no more drive commands are sent - lift the robot and carry
it to the start (the SDK has no free-wheel command, so the wheels may still
resist being pushed). **Start round** wakes it; the round then re-centres the
gimbal on the chassis and takes the direction it faces as north. Actions tab:
**Release robot** / **Wake robot** to do it by hand.

**Custom map:** Actions tab -> **Map WxH...** (or `g`): type e.g. `5x4`
(5 wide x 4 high) or pick a preset, click a cell to set the start, **SAVE** -
written to `grid_map` in `config/settings.yaml`.

**Timer:** countdown (orange at 1:00, blinking red in the last 30 s), elapsed /
limit, a progress bar, and the split time of every hit (also in
`roundN_map.png` and `roundN_targets.json`). Both rounds stop by themselves
when the time is up (10:00 / 5:00).

**Round 2** loads `round1_targets.json` (walls, open passages, target
positions); `src/route_planner.py` picks, for each designated target, the
nearest cell within range with a clear line of sight (cells the target was
seen from in round 1 first). The robot drives there, points the gimbal at the
saved position, sweeps +-30 deg if needed and shoots. A blocked move becomes a
wall and the route is replanned; up to 3 firing cells are tried per target.
Put the robot on the same start cell, facing the same way as in round 1.

* `src/target_vision.py` - HSV segmentation + shape classification. Only the
  assignment combinations count as targets: blue circle, red wide rectangle,
  yellow tall rectangle, green square. Other blobs (yellow wall, red marker
  card, floor reflections, blaster barrel) are ignored. Distance comes from the
  plate height (pinhole model) and is confirmed with the gimbal ToF when centred.
* `src/target_shooter.py` - centres the target with the gimbal, checks the
  <= 2 tile range, then fires.
* At the end of each round `data/raw/run1/roundN_map.png` (path + targets) and
  `roundN_targets.json` are saved.

**Panel widgets** (light theme): SENSORS shows a top view of the robot with the
front ToF and the left/right Sharp IR bars (red when closer than the 16.9 cm
wall threshold). AIM is the RoboFinal auto-aim HUD: phase (COARSE, FINE,
LOCKED, FIRE), the error dot inside the tolerance box, the lock counter (fires
after 2 frames in a row on target) and a 10 s error trace. The camera shows
the tolerance box and an AIMING / LOCKED / FIRE! banner. Every aim sample is
saved to `roundN_aim_log.csv`.

**Sharp IR** (`src/sharp_ir.py`, from RoboFinal): GP2Y0A41 4-30 cm,
`Vo = 12 / (L + 0.42)`, fold-back fix under 6 cm, 10 Hz polling thread logged
to `log_<date>_ir_data.csv`. The ports are found at start-up from
`sharp_ir.right_ports` / `left_ports` (this robot: right = adaptor 2 / port 1,
left = 1 / 2 - RoboFinal's calibration found the same); a port that never
answers is switched off instead of blocking every read with an SDK timeout.

### Before a real run (checklist)

1. **Cards:** `vision.plate_size_m` holds the lab cards (circle / square 7 cm,
   rectangles 9 x 6 cm). Put one card 1.0 m straight ahead and press
   Actions > **Calibrate: card at 1 m** once - it fits `vision.hfov_deg`.
2. **Aim:** start with the blaster unticked (dry run). Then fire a few shots
   and use Actions > **Aim trim** (shots high -> Down) and **Save**.
3. **Maze size / start:** Actions > **Map WxH...** (the assignment maze is 6x6).
4. **Round 2** uses `data/raw/run1/round1_targets.json` - run round 1 again
   after changing any of the above.

What the checks drop (seen in the first real run and in RoboFinal's round 1,
where the yellow room gave 12 "yellow objects"): blobs that are ragged or part
of a bigger patch of the same colour, anything further than the wall the ToF
sees that way, outside the maze, or behind a mapped wall. A target counts after
it is seen in 2 of 3 frames at 2 scan stops (`vision.min_observations`); a
first sighting shows as a hollow "?" on the map.

Sensors used: gimbal ToF + 2 Sharp IR + camera (within the assignment limits).
Set the `vision:` and `shooting:` sections in `config/settings.yaml` before a
run: measure `plate_height_m`, check `hfov_deg`, and list the targets you are
told to shoot in `shoot_targets` (a wrong target costs -1).
