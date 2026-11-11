"""Unit tests for density classification, smoothing and the signal controller.

Run with:  python -m pytest -q
These tests do not need a GPU, a video, Streamlit or YOLO.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.control import YELLOW_DURATION, TrafficSignalController, get_signal_recommendation  # noqa: E402
from utils.density import CountSmoother, classify_density  # noqa: E402
from utils.detection import is_emergency_class, list_models  # noqa: E402


# ---------- density ----------

@pytest.mark.parametrize("count, expected", [(0, "Low"), (9, "Low"), (10, "Medium"), (24, "Medium"), (25, "High"), (80, "High")])
def test_density_boundaries(count, expected):
    assert classify_density(count) == expected


def test_custom_thresholds():
    assert classify_density(5, low_threshold=3, high_threshold=6) == "Medium"


def test_bad_thresholds_raise():
    with pytest.raises(ValueError):
        classify_density(5, low_threshold=30, high_threshold=10)


def test_smoother_ignores_one_frame_spike():
    s = CountSmoother(window=5)
    for c in [12, 12, 12, 12]:
        s.update(c)
    assert s.update(40) == 12  # a single noisy frame does not change the level


def test_smoother_window_one_is_no_smoothing():
    s = CountSmoother(window=1)
    s.update(5)
    assert s.update(30) == 30


# ---------- signal recommendation ----------

def test_recommendation_by_density():
    assert get_signal_recommendation(5)["green_time"] == 20
    assert get_signal_recommendation(15)["green_time"] == 40
    assert get_signal_recommendation(30)["green_time"] == 60


def test_emergency_overrides_density():
    rec = get_signal_recommendation(3, emergency_detected=True)
    assert rec["green_time"] == 90 and rec["red_time"] == 10


# ---------- controller (time is passed in, so tests are exact) ----------

def run(ctrl, signal, start, end, step=0.5):
    t = start
    while t <= end:
        ctrl.update(signal, now=t)
        t += step


def test_full_cycle_red_green_yellow_red():
    ctrl = TrafficSignalController(start_time=0)
    low = get_signal_recommendation(5)  # green 20, red 95
    ctrl.update(low, now=0)
    assert ctrl.phase == "RED"
    run(ctrl, low, 0, 95)
    assert ctrl.phase == "GREEN" and ctrl.green_duration == 20
    run(ctrl, low, 95, 95 + 20)
    assert ctrl.phase == "YELLOW"
    run(ctrl, low, 115, 115 + YELLOW_DURATION)
    assert ctrl.phase == "RED"


def test_density_change_does_not_cut_a_phase_short():
    ctrl = TrafficSignalController(start_time=0)
    low, high = get_signal_recommendation(5), get_signal_recommendation(30)
    run(ctrl, low, 0, 95)                      # now GREEN for 20 s
    assert ctrl.phase == "GREEN"
    run(ctrl, high, 95.5, 110)                 # traffic gets heavy mid-phase
    assert ctrl.phase == "GREEN" and ctrl.green_duration == 20


def test_emergency_switches_to_green_immediately():
    ctrl = TrafficSignalController(start_time=0)
    ctrl.update(get_signal_recommendation(5), now=1)
    assert ctrl.phase == "RED"
    ctrl.update(get_signal_recommendation(5, emergency_detected=True), now=2)
    assert ctrl.phase == "GREEN" and ctrl.green_duration == 90


def test_emergency_triggers_only_once_per_event():
    ctrl = TrafficSignalController(start_time=0)
    em = get_signal_recommendation(5, emergency_detected=True)
    ctrl.update(em, now=0)
    ctrl.update(em, now=30)  # same emergency still visible: green must not restart
    assert ctrl.remaining(now=30) == pytest.approx(60)


def test_remaining_never_negative():
    ctrl = TrafficSignalController(start_time=0)
    assert ctrl.remaining(now=1000) == 0.0


# ---------- detection helpers (no YOLO needed) ----------

def test_emergency_class_names():
    assert is_emergency_class("Ambulance")
    assert is_emergency_class("fire_truck")
    assert not is_emergency_class("truck")
    assert not is_emergency_class("car")


def test_list_models_includes_custom_weights(tmp_path):
    (tmp_path / "ambulance_v1.pt").write_bytes(b"")
    models = list_models(str(tmp_path))
    assert models[:3] == ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    assert models[-1].endswith("ambulance_v1.pt")
