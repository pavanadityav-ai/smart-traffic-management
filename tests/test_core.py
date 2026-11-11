"""Unit tests for lanes, density, clearance time, signal timing, the controller and the tracker.

Run with:  python -m pytest -q      (no GPU, video, YOLO or torch needed)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from traffic import config  # noqa: E402
from traffic.analysis import (CountSmoother, analyse_lanes, classify_density, clearance_time,  # noqa: E402
                              lane_of, lane_speed, worst_density)
from traffic.detection import Detection, is_emergency_class, keep_detection, list_models  # noqa: E402
from traffic.signal import SignalController, signal_timing  # noqa: E402
from traffic.tracking import Tracker, iou  # noqa: E402


# ---------- lanes and density ----------

def test_lane_of_splits_frame_equally():
    assert [lane_of(x, 300, 3) for x in (0, 99, 100, 250, 300)] == [0, 0, 1, 2, 2]


@pytest.mark.parametrize("count, level", [(0, "Low"), (4, "Low"), (5, "Medium"), (9, "Medium"),
                                          (10, "High"), (14, "High"), (15, "Very High"), (40, "Very High")])
def test_four_density_levels(count, level):
    assert classify_density(count) == level


def test_thresholds_must_increase():
    with pytest.raises(ValueError):
        classify_density(3, (10, 5, 15))


def test_worst_density():
    assert worst_density(["Low", "Very High", "Medium"]) == "Very High"
    assert worst_density([]) == "Low"


def test_smoother_ignores_one_bad_frame():
    s = CountSmoother(window=5)
    for _ in range(4):
        s.update(6)
    assert s.update(30) == 6


# ---------- clearance time ----------

def test_clearance_time_formula():
    # 10 car-equivalents x 7 m = 70 m at 5 m/s = 14 s, plus 2 s start-up
    assert clearance_time(10, 5.0) == pytest.approx(16.0)


def test_clearance_empty_lane_is_zero():
    assert clearance_time(0, 5.0) == 0.0


def test_clearance_uses_minimum_speed():
    assert clearance_time(1, 0.0) == pytest.approx(config.STARTUP_LOST_TIME_S + config.SPACING_M / config.MIN_SPEED_MPS)


def test_speed_falls_back_without_calibration():
    assert lane_speed([100.0, 120.0], 0.0, "High") == (config.DEFAULT_SPEED_BY_DENSITY["High"], False)


def test_speed_measured_with_calibration():
    speed, measured = lane_speed([100.0, 120.0, None], 0.05, "High")  # 5 and 6 m/s
    assert measured and speed == pytest.approx(5.5)


# ---------- signal timing ----------

def test_green_follows_clearance_time():
    t = signal_timing("High", 52.3)
    assert t["green_s"] == 53 and t["red_s"] == config.CYCLE_S - 53 - config.YELLOW_S


def test_green_never_below_density_minimum():
    assert signal_timing("Very High", 10)["green_s"] == config.MIN_GREEN_BY_DENSITY["Very High"]


def test_green_capped():
    assert signal_timing("Very High", 400)["green_s"] == config.MAX_GREEN_S


def test_red_never_below_minimum():
    assert signal_timing("Very High", 400)["red_s"] >= config.MIN_RED_S


def test_emergency_timing():
    t = signal_timing("Low", 0, emergency=True)
    assert (t["green_s"], t["red_s"]) == (config.EMERGENCY_GREEN_S, config.EMERGENCY_RED_S)


# ---------- controller (time is passed in, so the tests are exact) ----------

def run(ctrl, timing, start, end, emergency=False, step=0.5):
    t = start
    while t <= end:
        ctrl.update(timing, emergency, now=t)
        t += step


def test_full_cycle():
    ctrl = SignalController(start_time=0)
    timing = signal_timing("Medium", 30)  # green 30, red 85
    run(ctrl, timing, 0, config.MIN_RED_S)
    assert ctrl.phase == "GREEN" and ctrl.green_s == 30
    run(ctrl, timing, 10.5, 40)
    assert ctrl.phase == "YELLOW"
    run(ctrl, timing, 40.5, 45)
    assert ctrl.phase == "RED" and ctrl.red_s == 85


def test_phase_not_cut_short_by_density_change():
    ctrl = SignalController(start_time=0)
    run(ctrl, signal_timing("Low", 0), 0, 10)
    assert ctrl.phase == "GREEN"
    run(ctrl, signal_timing("Very High", 80), 10.5, 14)
    assert ctrl.green_s == config.MIN_GREEN_BY_DENSITY["Low"]


def test_emergency_switches_to_green_once():
    ctrl = SignalController(start_time=0)
    em = signal_timing("Low", 0, emergency=True)
    ctrl.update(em, True, now=3)
    assert ctrl.phase == "GREEN" and ctrl.green_s == 90
    ctrl.update(em, True, now=33)  # same emergency still visible: green must not restart
    assert ctrl.remaining(33) == pytest.approx(60)


# ---------- tracker ----------

def det(x, label="car", emergency=False):
    return Detection((x, 100, x + 40, 130), label, 0.9, emergency)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_tracker_counts_each_vehicle_once():
    tracker = Tracker()
    for frame in range(20):
        tracker.update([det(10 + 2 * frame), det(300 + 2 * frame, "bus")], t=frame / 25)
    assert tracker.unique_counts == {"car": 1, "bus": 1}


def test_tracker_keeps_occluded_vehicle():
    tracker = Tracker()
    for frame in range(5):
        tracker.update([det(10 + frame)], t=frame)
    visible = tracker.update([], t=5)  # hidden behind a bus for one frame
    assert len(visible) == 1


def test_tracker_drops_vehicle_after_max_misses():
    tracker = Tracker()
    for frame in range(5):
        tracker.update([det(10 + frame)], t=frame)
    for frame in range(config.TRACK_MAX_MISSES + 1):
        visible = tracker.update([], t=5 + frame)
    assert visible == []


def test_emergency_flag_follows_latest_detection():
    tracker = Tracker()
    for frame in range(3):
        tracker.update([det(10 + frame, "ambulance", emergency=True)], t=frame)
    tracks = tracker.update([det(13, "truck")], t=3)  # the model now calls it a truck
    assert not tracks[0].emergency


def test_analyse_lanes_groups_vehicles():
    tracker = Tracker()
    for frame in range(4):
        tracks = tracker.update([det(10), det(15 + 60), det(250, "truck")], t=frame)
    smoothers = [CountSmoother(1) for _ in range(3)]
    result = analyse_lanes(tracks, 300, 3, smoothers)
    assert [lane["count"] for lane in result["lanes"]] == [2, 0, 1]
    assert result["lanes"][2]["by_type"] == {"truck": 1}
    assert result["lanes"][2]["load"] == config.VEHICLE_WEIGHTS["truck"]
    assert result["clearance_s"] == max(lane["clearance_s"] for lane in result["lanes"])


# ---------- detection helpers ----------

def test_emergency_class_names():
    assert is_emergency_class("Ambulance") and is_emergency_class("fire_truck") and is_emergency_class("police car")
    assert not is_emergency_class("truck") and not is_emergency_class("fire hydrant")


def test_only_vehicles_kept():
    assert keep_detection("car") and keep_detection("ambulance")
    assert not keep_detection("person") and not keep_detection("traffic light")


def test_list_models_includes_custom_weights(tmp_path):
    (tmp_path / "emergency_v1.pt").write_bytes(b"")
    models = list_models(str(tmp_path))
    assert "faster_rcnn" in models and models[-1].endswith("emergency_v1.pt")
