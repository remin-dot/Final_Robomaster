"""Small, hardware-independent helpers for safe omnidirectional driving."""

import math


def corridor_obstacle(readings, half_width_m, block_distance_m,
                      min_range_m=0.06, max_range_m=8.0):
    """Return the nearest obstacle distance along the travel corridor.

    ``readings`` contains ``(angle_deg, range_mm)`` rays relative to the
    direction of travel. A hit only blocks the move when the point lies inside
    the inflated robot width and before ``block_distance_m``.
    """
    nearest = None
    for angle_deg, range_mm in readings:
        distance = float(range_mm) / 1000.0
        if not min_range_m <= distance <= max_range_m:
            continue
        angle = math.radians(angle_deg)
        forward = distance * math.cos(angle)
        lateral = abs(distance * math.sin(angle))
        if 0.0 < forward <= block_distance_m and lateral <= half_width_m:
            nearest = forward if nearest is None else min(nearest, forward)
    return nearest


def braking_speed(clearance_m, stop_distance_m, cruise_mps,
                  reaction_s=0.10, brake_mps2=0.60):
    """Fastest speed that can still stop inside the measured clearance."""
    room = float(clearance_m) - float(stop_distance_m)
    if room <= 0.0 or brake_mps2 <= 0.0:
        return 0.0
    limit = -brake_mps2 * reaction_s + math.sqrt(
        (brake_mps2 * reaction_s) ** 2 + 2.0 * brake_mps2 * room
    )
    return min(float(cruise_mps), max(0.0, limit))


def information_rate(value, travel_s, scan_s=0.0):
    """Exploration value gained per expected second."""
    return float(value) / max(float(travel_s) + float(scan_s), 0.1)
