"""System tests: edge-case handling, database, alerts, the full pipeline and the REST API.

A FakeDetector and a generated video replace YOLO, so these run anywhere.
"""

import io
import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from traffic import config  # noqa: E402
from traffic.alerts import AlertManager, TwilioSMS  # noqa: E402
from traffic.detection import Detection  # noqa: E402
from traffic.pipeline import ProcessingJob, TrafficPipeline  # noqa: E402
from traffic.preprocess import FrameConditioner  # noqa: E402
from traffic.storage import Storage  # noqa: E402


class FakeDetector:
    """Cars moving right in two lanes; an ambulance appears from frame `ambulance_from`."""

    supports_emergency = True

    def __init__(self, ambulance_from=None):
        self.calls = 0
        self.ambulance_from = ambulance_from

    def detect(self, frame):
        self.calls += 1
        x = 2 * self.calls
        dets = [Detection((x, 50 + 30 * i, x + 30, 70 + 30 * i), "car", 0.9) for i in range(4)]
        dets.append(Detection((400 + x % 100, 200, 440 + x % 100, 230), "bus", 0.8))
        if self.ambulance_from is not None and self.calls >= self.ambulance_from:
            dets.append(Detection((500, 300, 560, 330), "ambulance", 0.9, emergency=True))
        return dets, 5.0


def textured(h=240, w=320, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def make_video(path, frames=60, fps=25, size=(640, 360)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    rng = np.random.default_rng(1)
    background = rng.integers(60, 200, (size[1], size[0], 3), dtype=np.uint8)
    for i in range(frames):
        frame = background.copy()
        cv2.rectangle(frame, (10 + 4 * i, 100), (60 + 4 * i, 130), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# ---------- edge cases ----------

def test_low_light_is_detected_and_brightened():
    dark = (textured() * 0.15).astype(np.uint8)
    out, flags = FrameConditioner(enhance=True).process(dark)
    assert flags["low_light"]
    assert out.mean() > dark.mean() * 1.5


def test_enhancement_is_off_by_default():
    dark = (textured() * 0.15).astype(np.uint8)
    out, flags = FrameConditioner().process(dark)
    assert flags["low_light"] and np.array_equal(out, dark)


def test_normal_frame_is_not_changed():
    frame = textured()
    out, flags = FrameConditioner().process(frame)
    assert not flags["low_light"] and not flags["low_contrast"]
    assert np.array_equal(out, frame)


def test_fog_low_contrast_detected():
    foggy = np.full((240, 320, 3), 170, np.uint8) + (textured() // 12)
    _, flags = FrameConditioner().process(foggy)
    assert flags["low_contrast"]


def test_blurry_frame_flagged():
    _, flags = FrameConditioner().process(cv2.GaussianBlur(textured(), (31, 31), 0))
    assert flags["blurry"]


def test_camera_shake_detected():
    cond = FrameConditioner()
    base = cv2.GaussianBlur(textured(480, 640), (5, 5), 0)
    cond.process(base)
    shifted = np.roll(base, 12, axis=1)
    _, flags = cond.process(shifted)
    assert flags["camera_shake"] and flags["shake_px"] == pytest.approx(12, abs=1.5)


def test_moving_vehicle_is_not_camera_shake():
    cond = FrameConditioner()
    base = cv2.GaussianBlur(textured(480, 640), (5, 5), 0)
    first, second = base.copy(), base.copy()
    cv2.rectangle(first, (100, 200), (200, 260), (0, 0, 255), -1)
    cv2.rectangle(second, (130, 200), (230, 260), (0, 0, 255), -1)  # the car moved 30 px, the camera did not
    cond.process(first)
    _, flags = cond.process(second)
    assert not flags["camera_shake"]


def test_frozen_camera_flagged():
    cond = FrameConditioner()
    frame = textured()
    for _ in range(config.FROZEN_FRAMES + 1):
        _, flags = cond.process(frame)
    assert flags["frozen"]


# ---------- alerts ----------

def test_alert_cooldown():
    alerts = AlertManager(cooldown_s=30)
    assert alerts.check(["ambulance"], now=0) is not None
    assert alerts.check(["ambulance"], now=10) is None   # same emergency, inside cooldown
    assert alerts.check(["ambulance"], now=31) is not None
    assert alerts.check([], now=100) is None


def test_sms_disabled_without_env(monkeypatch):
    for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "ALERT_TO_NUMBER"):
        monkeypatch.delenv(key, raising=False)
    assert TwilioSMS.from_env() is None
    assert AlertManager().check(["ambulance"], 0)["sms"] == "disabled"


def test_sms_sent_through_twilio(monkeypatch):
    sent = {}

    class Response:
        status_code = 201

    def fake_post(url, auth, timeout, data):
        sent.update(url=url, data=data)
        return Response()

    monkeypatch.setattr("traffic.alerts.requests.post", fake_post)
    done = []
    alerts = AlertManager(TwilioSMS("AC1", "tok", "+100", "+200"))
    alerts.check(["ambulance"], 0, on_sms_result=done.append)
    for _ in range(50):
        if done:
            break
        time.sleep(0.02)
    assert done[0]["sms"] == "sent"
    assert "AC1" in sent["url"] and sent["data"]["To"] == "+200" and "ambulance" in sent["data"]["Body"]


# ---------- storage ----------

def test_storage_round_trip():
    db = Storage(":memory:")
    run = db.start_run("test.mp4", "yolov8n.pt", 2)
    analysis = {"lanes": [{"lane": 1, "count": 3, "smoothed": 3, "density": "Low", "load": 3.0, "speed_mps": 8.0,
                           "clearance_s": 4.6},
                          {"lane": 2, "count": 12, "smoothed": 12, "density": "High", "load": 14.0, "speed_mps": 4.0,
                           "clearance_s": 26.5}]}
    db.log_traffic(run, 1.0, analysis, {"phase": "GREEN", "green_s": 40}, {"low_light": True, "blurry": False})
    row_id = db.log_emergency(run, {"time_s": 1.0, "types": ["ambulance"], "sms": "queued"})
    db.update_sms(row_id, "sent")
    db.finish_run(run, "finished", 25, 20.0, 12.0, {"car": 5})

    assert len(db.history(run)) == 2
    assert db.history(run)[0]["flags"] == '["low_light"]'
    assert db.emergencies(run)[0]["sms"] == "sent"
    assert db.summary(run)["density_seconds"] == {"High": 1, "Low": 1}
    assert db.runs()[0]["unique_counts"] == {"car": 5}
    assert db.export_csv(run).splitlines()[0].startswith("id,run_id,video_time")


# ---------- full pipeline ----------

def test_pipeline_emergency_turns_signal_green_and_logs_it():
    db = Storage(":memory:")
    run = db.start_run("synthetic", "fake", 3)
    pipe = TrafficPipeline(FakeDetector(ambulance_from=30), n_lanes=3, storage=db, run_id=run)
    frame = textured(360, 640)
    for i in range(1, 60):
        annotated, state = pipe.process(frame, t=i / 25)
    assert annotated.shape == frame.shape
    assert state["emergency"] and state["emergency_types"] == ["ambulance"]
    assert state["signal"]["phase"] == "GREEN" and state["signal"]["green_s"] == config.EMERGENCY_GREEN_S
    assert len(state["alerts"]) == 1
    assert len(db.emergencies(run)) == 1
    assert {row["video_time"] for row in db.history(run)} == {0.04, 1.04, 2.04}  # one row per lane per second
    assert state["unique_counts"]["car"] == 4


def test_job_runs_on_a_video_file(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    db = Storage(":memory:")
    run = db.start_run("clip.avi", "fake", 2)
    job = ProcessingJob(str(video), TrafficPipeline(FakeDetector(), n_lanes=2, storage=db, run_id=run),
                        db, run, pace=False)
    job.start()
    job.join(timeout=20)
    jpeg, state = job.snapshot()
    assert job.status == "finished" and job.progress == pytest.approx(1.0)
    assert jpeg[:2] == b"\xff\xd8"  # a JPEG image
    assert state["time_s"] == pytest.approx(60 / 25, abs=0.05)
    assert db.runs()[0]["status"] == "finished" and db.runs()[0]["frames"] == 60


def test_job_reports_bad_source(tmp_path):
    db = Storage(":memory:")
    run = db.start_run("missing", "fake", 1)
    job = ProcessingJob(str(tmp_path / "missing.mp4"), TrafficPipeline(FakeDetector(), 1, storage=db, run_id=run),
                        db, run)
    job.run()
    assert job.status == "error" and "Cannot open" in job.error


# ---------- combined detector ----------

def test_combined_detector_does_not_count_ambulance_twice():
    from traffic.detection import CombinedDetector

    class Main:
        name, supports_emergency = "main", False

        def detect(self, frame):
            return [Detection((0, 0, 50, 30), "car", 0.9), Detection((100, 0, 160, 30), "truck", 0.8)], 10.0

    class Emergency:
        name, supports_emergency = "em", True

        def detect(self, frame):
            return [Detection((102, 1, 160, 31), "ambulance", 0.9, True), Detection((0, 0, 50, 30), "car", 0.7)], 8.0

    dets, ms = CombinedDetector(Main(), Emergency()).detect(None)
    assert sorted(d.label for d in dets) == ["ambulance", "car"] and ms == 18.0


# ---------- REST API ----------

@pytest.fixture()
def client(tmp_path):
    from app import create_app
    app = create_app(detector_factory=lambda model, conf, emergency=None: FakeDetector(ambulance_from=5),
                     db_path=str(tmp_path / "t.db"), data_dir=str(tmp_path))
    app.testing = True
    return app.test_client()


def test_api_full_flow(client, tmp_path):
    assert client.get("/").status_code == 200
    assert "faster_rcnn" in client.get("/api/models").get_json()["models"]

    video = make_video(tmp_path / "clip.avi").read_bytes()
    res = client.post("/api/jobs", data={"video": (io.BytesIO(video), "clip.avi"), "lanes": "2", "pace": "0"},
                      content_type="multipart/form-data")
    assert res.status_code == 201
    job_id, run_id = res.get_json()["job_id"], res.get_json()["run_id"]

    for _ in range(200):
        status = client.get(f"/api/jobs/{job_id}").get_json()
        if status["status"] == "finished":
            break
        time.sleep(0.05)
    assert status["status"] == "finished"
    assert status["state"]["emergency"] and len(status["state"]["analysis"]["lanes"]) == 2

    detail = client.get(f"/api/runs/{run_id}").get_json()
    assert detail["run"]["status"] == "finished" and len(detail["emergencies"]) == 1
    assert len(client.get(f"/api/runs/{run_id}/history").get_json()) > 0
    assert client.get(f"/api/runs/{run_id}/export.csv").data.startswith(b"id,run_id")


@pytest.mark.parametrize("data, message", [
    ({}, "Upload a video"),
    ({"source": "0", "lanes": "9"}, "Lanes must be between 1 and 6"),
    ({"source": "0", "low": "10", "medium": "5"}, "increasing order"),
    ({"source": "0", "confidence": "abc"}, "must be numbers"),
])
def test_api_rejects_bad_input(client, data, message):
    res = client.post("/api/jobs", data=data)
    assert res.status_code == 400 and message in res.get_json()["error"]


def test_api_rejects_wrong_file_type(client):
    res = client.post("/api/jobs", data={"video": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_api_unknown_ids(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/runs/999").status_code == 404


# ---------- benchmark helpers (the full benchmark needs a GPU and COCO) ----------

def test_benchmark_helpers(tmp_path):
    from benchmark.compare_models import darken, display_name, run_model, select_images

    image = textured()
    dark = darken(image, seed=1)
    assert dark.mean() < image.mean() * 0.5

    class FakeCoco:
        def getImgIds(self, catIds):
            return {3: [5, 1], 8: [1, 9]}[catIds[0]]

        def loadImgs(self, img_id):
            return [{"file_name": f"{img_id}.jpg"}]

    assert select_images(FakeCoco(), [3, 8], 10) == [1, 5, 9]
    assert select_images(FakeCoco(), [3, 8], 2) == [1, 5]

    for img_id in (1, 5):
        cv2.imwrite(str(tmp_path / f"{img_id}.jpg"), image)
    results, times = run_model(FakeDetector(ambulance_from=1), FakeCoco(), [1, 5], tmp_path, {"car": 3, "bus": 6})
    assert {r["category_id"] for r in results} == {3, 6}  # the ambulance is not a COCO class, so it is skipped
    x, y, w, h = results[0]["bbox"]
    assert w > 0 and h > 0
    assert display_name("faster_rcnn").startswith("Faster R-CNN") and display_name("yolov8s.pt") == "YOLOv8s"
