"""Live check of every sensor-adaptor port: which port has which sensor, and does it work.

    .venv/bin/python src/sensor_check.py            # robot on AP Wi-Fi (close the mission panel first)
    .venv/bin/python src/sensor_check.py --ids 1 2 3

For each adaptor id / port it shows the analog value (0..1023), the digital pin (0/1),
the range seen so far, and what that looks like:
  * Sharp (analog)       - the analog value follows your hand (big range, ~90+ with nothing in front)
  * obstacle module      - the digital pin flips 0 <-> 1 when your hand is ~5 cm in front
  * nothing / no signal  - analog stuck near 0 V, or floating ~300-400 with the digital pin not changing
Move a hand in front of each sensor. Ctrl+C to quit. It also says what settings.yaml expects.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config  # noqa: E402
from sharp_ir import read_port  # noqa: E402


def verdict(lo, hi, io_seen, adc):
    if adc is None:
        return "no answer"
    if len(io_seen) > 1:
        return "OBSTACLE MODULE (digital pin switches)"
    if hi - lo > 150 and lo < 700:
        return "SHARP / analog sensor (value follows the distance)"
    if hi - lo > 400:
        return "switch on the analog pin (rails)"
    if hi < 70:
        return "NO SIGNAL: ~0 V (nothing plugged / no power)"
    if 150 < adc < 800:
        return "floating (~unplugged) - or a module that has not switched yet"
    return "steady - wave a hand in front"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--connection", default="ap")
    args = ap.parse_args()

    from mission_panel import require_sdk
    robot = require_sdk()
    ep = robot.Robot()
    try:
        if args.connection == "ap":
            ep.initialize(conn_type="ap", proto_type="udp")
        else:
            ep.initialize(conn_type=args.connection)
    except Exception as e:
        raise SystemExit(f"\n[!] Could not reach the robot ({e}).\n"
                         "    Join this computer to the robot's Wi-Fi (RMEP-xxxxxx) - it must get a 192.168.2.x address.\n")

    cfg = load_config()
    sh, cc = cfg.get("sharp_ir", {}) or {}, cfg.get("ir_corner", {}) or {}
    expect = {}
    for side, key in (("left", "left_ports"), ("right", "right_ports")):
        for p in sh.get(key, [])[:1]:
            expect[tuple(p)] = f"Sharp {side}"
    for side in ("left", "right"):
        p = cc.get(f"{side}_port")
        if p:
            expect[tuple(p)] = f"IR front-{side}"

    ada = ep.sensor_adaptor
    ports = []
    print("looking for adaptors ...")
    for i in args.ids:
        adc, io = read_port(ada, (i, 1))
        if adc is None and io is None:
            print(f"  adaptor {i}: no answer (not connected / other id)")
            continue
        ports += [(i, 1), (i, 2)]
        print(f"  adaptor {i}: found")
    if not ports:
        ep.close()
        raise SystemExit("no sensor adaptor answered - check the adaptors' cables to the robot")

    stats = {p: {"lo": 9999, "hi": -1, "io": set(), "adc": None, "iov": None} for p in ports}
    print("\nwave a hand in front of each sensor ... (Ctrl+C to stop)\n")
    try:
        while True:
            t0 = time.time()
            for p in ports:
                adc, io = read_port(ada, p)
                s = stats[p]
                s["adc"], s["iov"] = adc, io
                if adc is not None:
                    s["lo"], s["hi"] = min(s["lo"], adc), max(s["hi"], adc)
                if io is not None:
                    s["io"].add(io)
            lines = [f"{'port':<8}{'settings.yaml':<16}{'analog':>7}{'dig':>5}{'seen':>12}   looks like",
                     "-" * 96]
            for p in ports:
                s = stats[p]
                seen = f"{s['lo']}-{s['hi']}" if s["hi"] >= 0 else "--"
                lines.append(f"{p[0]}/{p[1]:<6}{expect.get(p, '-'):<16}{str(s['adc']):>7}{str(s['iov']):>5}"
                             f"{seen:>12}   {verdict(s['lo'], s['hi'], s['io'], s['adc'])}")
            sys.stdout.write("\033[H\033[J" + "\n".join(lines) +
                             f"\n\n{1.0 / max(time.time() - t0, 1e-3):.0f} reads/s per port   Ctrl+C to stop\n")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ep.close()


if __name__ == "__main__":
    main()
