"""
Per-lane analysis: vehicle count, density level, speed and road clearance time.

Road clearance time (how long a green phase needs to clear the queue in a lane):

    load        = sum of vehicle weights in the lane   (car 1, motorcycle 0.5, bus/truck 2)
    queue (m)   = load x SPACING_M                     (7 m per car-equivalent)
    clearance   = STARTUP_LOST_TIME_S + queue / speed

speed is the measured average speed of the moving vehicles in the lane when
the camera is calibrated (METERS_PER_PIXEL > 0); otherwise a typical speed for
the density level is used. This is a simple queue-discharge model. It gives a
reasonable estimate, not an exact prediction.
"""

from collections import deque
from statistics import median

from traffic import config


def lane_of(x_centre: float, frame_width: int, n_lanes: int) -> int:
    """Lane index (0 = leftmost) for an x position; the frame is split into equal vertical lanes."""
    lane = int(x_centre / max(frame_width, 1) * n_lanes)
    return min(max(lane, 0), n_lanes - 1)


def classify_density(count: float, thresholds=config.DENSITY_THRESHOLDS) -> str:
    low, medium, high = thresholds
    if not low <= medium <= high:
        raise ValueError("thresholds must be in increasing order")
    if count < low:
        return "Low"
    if count < medium:
        return "Medium"
    if count < high:
        return "High"
    return "Very High"


def worst_density(levels) -> str:
    levels = list(levels)
    return max(levels, key=config.DENSITY_LEVELS.index) if levels else "Low"


class CountSmoother:
    """Median of the last `window` counts, so one bad frame does not change the density level."""

    def __init__(self, window: int = config.SMOOTHING_WINDOW) -> None:
        self.values = deque(maxlen=max(1, window))

    def update(self, value: float) -> float:
        self.values.append(value)
        return median(self.values)


def vehicle_weight(label: str, emergency: bool = False) -> float:
    if emergency:
        return config.VEHICLE_WEIGHTS["emergency"]
    return config.VEHICLE_WEIGHTS.get(label, 1.0)


def clearance_time(load: float, speed_mps: float) -> float:
    """Seconds of green needed to clear `load` car-equivalents moving at `speed_mps`."""
    if load <= 0:
        return 0.0
    speed = max(speed_mps, config.MIN_SPEED_MPS)
    return config.STARTUP_LOST_TIME_S + load * config.SPACING_M / speed


def lane_speed(speeds_px: list, meters_per_pixel: float, density: str) -> tuple[float, bool]:
    """(speed in m/s, measured?) Uses moving vehicles only; falls back to a typical speed."""
    if meters_per_pixel > 0:
        moving = [s * meters_per_pixel for s in speeds_px if s is not None and s * meters_per_pixel >= 0.5]
        if len(moving) >= 2:
            return median(moving), True
    return config.DEFAULT_SPEED_BY_DENSITY[density], False


def analyse_lanes(tracks, frame_width: int, n_lanes: int, smoothers: list,
                  thresholds=config.DENSITY_THRESHOLDS, meters_per_pixel: float = config.METERS_PER_PIXEL) -> dict:
    """Group tracked vehicles by lane and compute count, density, speed and clearance per lane."""
    lanes = [{"lane": i + 1, "count": 0, "by_type": {}, "load": 0.0, "speeds": []} for i in range(n_lanes)]
    for tr in tracks:
        lane = lanes[lane_of(tr.centre[0], frame_width, n_lanes)]
        lane["count"] += 1
        lane["by_type"][tr.label] = lane["by_type"].get(tr.label, 0) + 1
        lane["load"] += vehicle_weight(tr.label, tr.emergency)
        lane["speeds"].append(tr.speed_px_per_s())

    for lane, smoother in zip(lanes, smoothers):
        lane["smoothed"] = smoother.update(lane["count"])
        lane["density"] = classify_density(lane["smoothed"], thresholds)
        speed, measured = lane_speed(lane.pop("speeds"), meters_per_pixel, lane["density"])
        lane["speed_mps"] = round(speed, 2)
        lane["speed_measured"] = measured
        # scale the load to the smoothed count so the clearance time is stable too
        avg_weight = lane["load"] / lane["count"] if lane["count"] else 1.0
        load = avg_weight * lane["smoothed"]
        lane["load"] = round(load, 2)
        lane["clearance_s"] = round(clearance_time(load, speed), 1)

    return {
        "lanes": lanes,
        "total": sum(l["count"] for l in lanes),
        "density": worst_density(l["density"] for l in lanes),
        "clearance_s": max((l["clearance_s"] for l in lanes), default=0.0),  # lanes clear in parallel
    }
