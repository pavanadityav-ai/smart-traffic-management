"""
Flask web app and REST API.

Pages
  /          live dashboard (video stream, lanes, signal, clearance time, alerts, charts)
  /history   stored runs from the database, with charts and CSV export
  /compare   YOLOv8 vs Faster R-CNN results (from results/benchmark.json)

REST API
  GET  /api/models                 available detectors and whether SMS is set up
  POST /api/jobs                   start analysing an uploaded video, a webcam or a stream URL
  GET  /api/jobs/<id>              live status: counts per lane, density, clearance, signal, alerts
  GET  /api/jobs/<id>/stream       annotated video as an MJPEG stream
  POST /api/jobs/<id>/stop         stop a job
  GET  /api/runs                   all stored runs
  GET  /api/runs/<id>              summary and emergency log of one run
  GET  /api/runs/<id>/history      per-second, per-lane data of one run
  GET  /api/runs/<id>/export.csv   the same data as CSV
  GET  /api/benchmark              model comparison results

Run:  python app.py   then open http://127.0.0.1:5000
"""

import json
import os
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from traffic import config
from traffic.alerts import AlertManager, TwilioSMS
from traffic.detection import custom_models, list_models, make_detector
from traffic.pipeline import ProcessingJob, TrafficPipeline
from traffic.preprocess import FrameConditioner
from traffic.storage import Storage

BASE = Path(__file__).resolve().parent
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv"}


def parse_source(text: str):
    """'0' -> webcam 0; anything else (rtsp://, http://, a path) is passed to OpenCV as is."""
    text = text.strip()
    return int(text) if text.isdigit() else text


def create_app(detector_factory=make_detector, db_path: str | None = None, data_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB uploads
    data = Path(data_dir) if data_dir else BASE / "data"
    uploads = data / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    storage = Storage(db_path or str(data / "traffic.db"))
    jobs: dict[str, ProcessingJob] = {}
    app.config["JOBS"], app.config["STORAGE"] = jobs, storage

    def error(message: str, code: int = 400):
        return jsonify({"error": message}), code

    # ---- pages -----------------------------------------------------------------
    @app.get("/")
    def dashboard():
        return render_template("index.html")

    @app.get("/history")
    def history_page():
        return render_template("history.html")

    @app.get("/compare")
    def compare_page():
        return render_template("compare.html")

    # ---- API -------------------------------------------------------------------
    @app.get("/api/models")
    def models():
        return jsonify({"models": list_models(str(BASE / "models")),
                        "emergency_models": custom_models(str(BASE / "models")),
                        "sms_enabled": TwilioSMS.from_env() is not None,
                        "defaults": {"lanes": config.DEFAULT_LANES, "confidence": config.DEFAULT_CONFIDENCE,
                                     "thresholds": list(config.DENSITY_THRESHOLDS)}})

    @app.post("/api/jobs")
    def start_job():
        form = request.form
        try:
            lanes = int(form.get("lanes", config.DEFAULT_LANES))
            confidence = float(form.get("confidence", config.DEFAULT_CONFIDENCE))
            thresholds = tuple(int(form.get(k, d)) for k, d in zip(("low", "medium", "high"), config.DENSITY_THRESHOLDS))
            meters_per_pixel = float(form.get("meters_per_pixel", config.METERS_PER_PIXEL) or 0)
        except ValueError:
            return error("Lanes, confidence, thresholds and calibration must be numbers.")
        if not 1 <= lanes <= 6:
            return error("Lanes must be between 1 and 6.")
        if not 0.05 <= confidence <= 0.95:
            return error("Confidence must be between 0.05 and 0.95.")
        if not thresholds[0] <= thresholds[1] <= thresholds[2]:
            return error("Density thresholds must be in increasing order.")

        video = request.files.get("video")
        if video and video.filename:
            suffix = Path(video.filename).suffix.lower()
            if suffix not in VIDEO_TYPES:
                return error("Upload an .mp4, .avi, .mov or .mkv video.")
            path = uploads / f"{uuid.uuid4().hex[:8]}_{secure_filename(video.filename)}"
            video.save(path)
            source, label = str(path), video.filename
        elif form.get("source", "").strip():
            source = parse_source(form["source"])
            label = str(source)
        else:
            return error("Upload a video or enter a camera number / stream URL.")

        model = form.get("model", "yolov8n.pt")
        emergency_model = form.get("emergency_model", "").strip() or None
        try:
            detector = detector_factory(model, confidence, emergency_model)
        except Exception as exc:
            return error(f"Could not load the model: {exc}", 500)

        for job in jobs.values():  # one camera at a time
            job.stop()
        run_id = storage.start_run(label, getattr(detector, "name", model), lanes)
        pipeline = TrafficPipeline(
            detector, n_lanes=lanes, thresholds=thresholds, meters_per_pixel=meters_per_pixel,
            storage=storage, run_id=run_id, alerts=AlertManager(TwilioSMS.from_env()),
            conditioner=FrameConditioner(stabilise=form.get("stabilise", "1") == "1",
                                         enhance=form.get("enhance", "1") == "1"))
        job = ProcessingJob(source, pipeline, storage, run_id, pace=form.get("pace", "1") == "1")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = job
        job.start()
        return jsonify({"job_id": job_id, "run_id": run_id}), 201

    def get_job(job_id):
        job = jobs.get(job_id)
        if job is None:
            return None, error("Job not found.", 404)
        return job, None

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job, err = get_job(job_id)
        if err:
            return err
        _, state = job.snapshot()
        return jsonify({"status": job.status, "error": job.error, "progress": round(job.progress, 3),
                        "run_id": job.run_id, "sms_enabled": job.pipeline.alerts.sms_enabled, "state": state})

    @app.post("/api/jobs/<job_id>/stop")
    def stop_job(job_id):
        job, err = get_job(job_id)
        if err:
            return err
        job.stop()
        return jsonify({"status": "stopping"})

    @app.get("/api/jobs/<job_id>/stream")
    def stream(job_id):
        job, err = get_job(job_id)
        if err:
            return err

        def frames():
            last = None
            while True:
                jpeg, _ = job.snapshot()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                if job.status not in ("starting", "running"):
                    break
                time.sleep(0.04)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/runs")
    def runs():
        return jsonify(storage.runs())

    @app.get("/api/runs/<int:run_id>")
    def run_detail(run_id):
        run = next((r for r in storage.runs() if r["id"] == run_id), None)
        if run is None:
            return error("Run not found.", 404)
        return jsonify({"run": run, "summary": storage.summary(run_id), "emergencies": storage.emergencies(run_id)})

    @app.get("/api/runs/<int:run_id>/history")
    def run_history(run_id):
        return jsonify(storage.history(run_id))

    @app.get("/api/runs/<int:run_id>/export.csv")
    def run_csv(run_id):
        return Response(storage.export_csv(run_id), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=run_{run_id}.csv"})

    @app.get("/api/benchmark")
    def benchmark():
        path = BASE / "results" / "benchmark.json"
        if not path.exists():
            return error("No benchmark results yet. Run notebooks/benchmark_on_kaggle.ipynb.", 404)
        return jsonify(json.loads(path.read_text()))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
