"""Display the live video feed from a connected DJI RoboMaster robot."""

import argparse
import sys

import cv2
import numpy as np
from robomaster import robot

from config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the camera feed from a connected RoboMaster robot."
    )
    parser.add_argument(
        "--connection",
        choices=("ap", "sta", "rndis"),
        help="RoboMaster connection type (defaults to config/settings.yaml).",
    )
    parser.add_argument(
        "--resolution",
        choices=("360p", "540p", "720p"),
        default="720p",
        help="Video resolution (default: 720p).",
    )
    parser.add_argument(
        "--min-card-area",
        type=int,
        default=1500,
        help="Minimum red-card area in pixels (default: 1500).",
    )
    return parser.parse_args()


def detect_red_card(frame, min_area=1500):
    """Return the best red rectangular region as (x, y, width, height), if any."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around the ends of OpenCV's 0-179 hue scale.
    low_red = cv2.inRange(hsv, np.array((0, 100, 70)), np.array((10, 255, 255)))
    high_red = cv2.inRange(
        hsv, np.array((170, 100, 70)), np.array((179, 255, 255))
    )
    mask = cv2.bitwise_or(low_red, high_red)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        bounding_area = width * height
        rectangularity = area / bounding_area if bounding_area else 0
        aspect_ratio = width / height if height else 0

        # Allow both portrait and landscape cards while rejecting thin red lines.
        if rectangularity >= 0.55 and 0.35 <= aspect_ratio <= 2.85:
            candidates.append((area, (x, y, width, height)))

    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def draw_detection(frame, card_box):
    """Draw the red-card detection result on a camera frame."""
    if card_box is None:
        cv2.putText(
            frame,
            "Searching for red card...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return

    x, y, width, height = card_box
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
    cv2.putText(
        frame,
        "RED CARD DETECTED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def show_camera(connection_type, resolution, min_card_area):
    ep_robot = robot.Robot()
    stream_started = False

    try:
        print(f"Connecting to RoboMaster using '{connection_type}' mode...")
        
        # ✅ แก้ไขบั๊ก proxy_addr โดยการดักด้วยการย่อหน้าที่ถูกต้อง
        if connection_type == "ap":
            ep_robot.initialize(conn_type="ap", proto_type="udp")
        else:
            ep_robot.initialize(conn_type=connection_type)

        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False, resolution=resolution)
        stream_started = True
        was_detected = False

        print("Camera connected; red-card detection is active.")
        print("Press q or Esc in the video window to quit.")
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                print("Waiting for a camera frame...", end="\r")
                continue

            card_box = detect_red_card(frame, min_card_area)
            is_detected = card_box is not None
            if is_detected and not was_detected:
                print("RED CARD DETECTED")
            elif was_detected and not is_detected:
                print("Red card no longer detected.")
            was_detected = is_detected

            draw_detection(frame, card_box)
            cv2.imshow("RoboMaster Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if stream_started:
            ep_robot.camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    config = load_config()
    connection_type = args.connection or config.get("robot", {}).get(
        "connection_type", "ap"
    )

    try:
        if args.min_card_area <= 0:
            raise ValueError("--min-card-area must be greater than zero")
        show_camera(connection_type, args.resolution, args.min_card_area)
    except KeyboardInterrupt:
        print("\nCamera viewer stopped.")
    except Exception as error:
        print(f"Camera viewer error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
