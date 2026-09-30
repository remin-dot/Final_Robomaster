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

**IR sensors - two pairs:**

| | left | right | |
|---|---|---|---|
| Sharp GP2Y0A41 (sides, analog) | adaptor 1, port 2 | adaptor 2, port 1 | calibrate: `sharp_ir` |
| IR obstacle modules (front corners, ~45 deg, on/off) | adaptor 1, port 1 | adaptor 2, port 2 | no calibration: `ir_corner` |

While driving: a front corner module that sees a wall -> slide away and slow
down; both -> a wall right in front: stop (and do not start a move). The side
Sharps keep the robot off the side walls. Calibrate the Sharps with:

```bash
python3 src/sharp_calibrate.py          # robot (close the mission panel first)
python3 src/sharp_calibrate.py --demo   # no robot: sliders simulate the sensors
```

Each adaptor port has an analog pin and a digital pin; one request returns both.
The obstacle modules' signal is on the **digital** pin (their analog pin only floats
~300-400); `ir_corner.signal: auto` finds which pin switches, `active_low: true` =
LOW means wall (standard modules, LED on). A Sharp stuck near 0 V (raw < 70) is
reported as NO SIGNAL (panel, log). To see which port really has which sensor:

```bash
.venv/bin/python src/sensor_check.py      # live table: analog, digital, range seen, what it looks like
```

Hold a flat white board 5, 10, 15, 20 cm straight out from each Sharp's face and
press Record; the panel fits a distance curve (a port that only switches on/off
is not a Sharp - check the port). The corner modules are shown live at the
bottom (clear / WALL): turn their screws until WALL shows at ~5 cm. **Save**
writes `sharp_ir` in settings.yaml (the `ir_corner` section is left alone).

**Start anywhere:** before round 1, click a cell on the map to put the start
there, click it again to turn which way the robot faces (N = up the map, then E,
S, W); **Save** on the map strip stores it (`grid_map.start`). The robot's real
heading at the start becomes that map direction; round 2 reads it from the
round-1 file.

**Between rounds:** when a round ends the robot is released
(`movement.release_after_round`): wheels and gimbal get a last "stand still",
robot mode goes to FREE (chassis and gimbal no longer follow each other, so
turning one by hand does not move the other), the gimbal motors sleep so it can
be turned straight by hand, the LEDs go dim blue, and no more commands are sent.
The SDK has no wheel torque-off command, so the wheels may still resist being
pushed: lift the robot to carry it to the start. **Start round** wakes it (LEDs
green); the round then re-centres the gimbal on the chassis and takes the
direction it faces as north. Actions tab: **Release robot** / **Wake robot** to
do it by hand.

**Driving (slide on straights, turn at corners):** on a plain straight stretch
(both blocks have the same, known edges on each side: wall + wall or open + open)
the mecanum wheels slide sideways / backwards without turning (`movement.strafe_moves`);
the gimbal points the ToF and camera the way the robot drives to guard it. Where a
wall ends, at a doorway, or next to an edge not known yet, the robot turns to face
the way instead, so the two front-corner IR modules watch its corners. Moves run at
`cell_speed` 0.45 m/s with a smooth braking stop. After every scan the robot
slides back to the middle of its block from the ToF readings (`recenter`,
`center_tof_mm`: check that value on the robot - the ToF reading to a wall of the
block with the robot centred), so odometry drift and sideways slip do not add up.
Set `strafe_moves: false` to go back to turning (then at up to `turn_max_dps`).

**Exploring (round 1):** cards hang on walls, so the robot does not need to stand in
every block - it needs every *wall face* seen well (closer than `vision.see_range_m`,
in the picture, at most `see_max_view_deg` off face-on, nothing in between). A block
is visited only while one of its wall faces is unseen or one of its edges unknown.
Long ToF readings also tell where walls further on are (1.4 m down an open way =
one more open block, then a wall) without driving there; what a scan saw directly
always wins. Next block = the nearest (fewest moves) still worth a visit, ties by
`explore_order` relative to how the robot came into its block (side blocks first),
reached by the cheapest route. The way it came in is not scanned again, the gimbal
sweeps once from one side to the other, and it stops exploring when nothing is left
- or when the time left is only what the route planner needs to go and shoot the
cards found (+ `shooting.reserve_margin_s`). On the last run's 6x6 (simulated): 25
blocks visited instead of 36, 34 moves instead of 56, 0 chassis turns instead of 40.

A move that stops more than half-way into the next block with a wall close ahead
(`movement.arrive_tof_mm`) has arrived - that is the far wall of a dead-end block.

**Close look:** in a block that holds a card (or next to one that needs checking,
or where half a card was seen) the camera looks 10 deg down (`vision.verify_pitch_deg`)
at the 4 walls (4 looks cover the block with the 96 deg lens), so a second card on
the opposite wall is found too. A card closer than `shooting.min_shoot_m` is shot
after sliding a few cm away (`back_off_max_m`).

**Colour detection:** the white foam walls are the colour reference (the picture is
white-balanced on them, `vision.white_balance`), and a card must hang on white wall:
most of what is around it has to be neutral (`min_neutral_around_card`) - that
drops tape on mats, clothes, markers in the room, robot parts. Calibrate the card
colours on the real arena under the real light:

```bash
.venv/bin/python src/color_calibrate.py                  # robot camera (close the mission panel first)
.venv/bin/python src/color_calibrate.py --image capture_*.jpg   # or saved pictures
```

1-4 pick the colour, click on cards of that colour (near, far, lit, in shade), M
shows what the range picks up, S saves `config/color_config.json`.

**Square vs rectangle:** a card seen at a slant looks narrower (width x cos of
the angle), so a wide rect looks square and a square looks tall. The panel works
out the angle from the wall the card hangs on (map) and undoes it, trying each
square/rect guess because the distance comes from the card height (6 / 7 / 9 cm).
Only a real slant (over `slant_fix_min_deg`) is corrected, with the detector's own
cut-offs. Every sighting votes for a shape, face-on views count far more. The
decider is the ToF: when the gimbal is locked on a card the ToF hits it, and its
real height comes out (6 cm wide rect, 7 cm square, 9 cm tall rect - the height
does not change at a slant). A square or rectangle gets a bead only when its shape
is sure (ToF-measured, or seen within 35 deg of face-on) and that kind is selected
- a card the ToF shows is a kind not selected is never shot.

**Maybe-cards:** a card-sized, card-coloured blob on white wall that is cut by the
picture edge (too close) or has an odd outline is never shot as it is - it is shown
as "colour shape ?". Seen in the regular scan it only earns the block a close look;
in the close look the camera points straight at it so the whole card is in view; if
still unclear, the robot goes once per block to the next block from where that wall
is seen face-on (~0.9 m) and looks back. A place that turned out not to be a card is
remembered and not checked again.

**Hit until it falls:** after each bead the robot looks at the card again
(`shooting.fall_wait_s`). Still standing -> re-aim and fire again, up to
`shooting.max_shots_per_target` (3); gone -> next target. It only fires again at the
same whole card, never at a "?" blob. Turning **Blaster armed** off during a round
needs a second click within 3 s. Aiming: the barrel sits below
the camera, so it aims up by atan(`barrel_below_camera_m` / distance) plus the trim
`aim_pitch_offset_deg` (fine-tune with Aim trim in the panel and Save). The lock
window grows with the card's size in the picture, and a gimbal stuck at its pitch
limit still fires when the barrel line is on the card.

**One card, one entry:** sightings of one card merge with a radius that grows with
distance (far views are less exact), duplicates that drift together are joined, and
a card already hit is never shot again as a "new" one. Every round also saves its
full log as `roundN_log.txt`.

**Shooting rule:** a card is shot only when it is in the robot's block or the
block right next to it, straight ahead of the gimbal (not diagonal) with no wall
in between - never across a block (`shooting.reach_cells: 1`). From the next block
the card must also sit straight in line - within `straight_band_m` (0.2 m) of the
robot's row / column line, seen at most `max_shot_view_deg` (55 deg) slanted. A card
at a hard angle (near a corner of its block, or edge-on) is shot from inside its own
block: the robot goes there, looks down (`vision.verify_pitch_deg`), slides a few cm
away if it is closer than `min_shoot_m`, and fires. A card seen from
further away is put on the map and shot later from beside it. One block can hold
several cards (different kinds, or two of the same kind side by side): each one
is mapped and shot. If the aim loses a card for a moment (blur after a big gimbal
move) it waits up to `shooting.lost_retries` frames before giving up. At the end
of round 1, every selected card that was found but not hit is visited and shot
from the block next to it while time is left (`shooting.mop_up_round1`).

**Custom map:** Actions tab -> **Map WxH...** (or `g`): type e.g. `5x4`
(5 wide x 4 high) or pick a preset, click a cell to set the start, **SAVE** -
written to `grid_map` in `config/settings.yaml`.

**Timer:** countdown (orange at 1:00, blinking red in the last 30 s), elapsed /
limit, a progress bar, and the split time of every hit (also in
`roundN_map.png` and `roundN_targets.json`). Both rounds stop by themselves
when the time is up (10:00 / 5:00).

**Round 2** loads `round1_targets.json` (walls, open passages, target
positions); `src/route_planner.py` picks, for each designated target, a
firing cell in the target's block or right next to it with no wall between
(cells the target was seen from in round 1 first). The robot drives there, points the gimbal at the
saved position, sweeps +-30 deg if needed and shoots. A blocked move becomes a
wall and the route is replanned; up to 3 firing cells are tried per target.
Put the robot on the same start cell, facing the same way as in round 1.

* `src/target_vision.py` - HSV segmentation + shape classification. Only the
  assignment combinations count as targets: blue circle, red wide rectangle,
  yellow tall rectangle, green square. Other blobs (yellow wall, red marker
  card, floor reflections, blaster barrel) are ignored. Distance comes from the
  plate height (pinhole model) and is confirmed with the gimbal ToF when centred.
* `src/target_shooter.py` - centres the target with the gimbal, checks the
  range, then fires (the chassis only calls it for a card in this block or the
  next one).
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
