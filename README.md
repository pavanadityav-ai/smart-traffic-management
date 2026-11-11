# 🚦 Advanced Intelligent System for Efficient and Smart Traffic Management

[![tests](https://github.com/pavanadityav-ai/smart-traffic-management/actions/workflows/tests.yml/badge.svg)](https://github.com/pavanadityav-ai/smart-traffic-management/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/models-YOLOv8%20%7C%20Faster%20R--CNN-orange)
![Backend](https://img.shields.io/badge/backend-Flask%20REST%20API-black)
![Database](https://img.shields.io/badge/database-SQLite-lightgrey)

A web system that watches traffic video, counts vehicles **per lane**, classifies traffic density, estimates **road clearance time**, and sets **adaptive signal timing**, with priority and alerts for emergency vehicles. All data is stored in a database for later analysis, and YOLOv8 is compared with Faster R-CNN on accuracy and speed.

**About this project.** It started as our B.Tech team project at Mohan Babu University (Computer Science and Engineering, AI & ML). I wrote the project specification: features, algorithms and tech stack. The team built a first Streamlit prototype that covered part of it. This repository is **my extended version**, which implements the full specification, with automated tests. The earlier prototype is in the commit history.

## How it works

```mermaid
flowchart LR
    V[Video file / webcam / CCTV stream] --> P[Edge-case handling<br>low light · fog · shake · blur · frozen]
    P --> D[Detection<br>YOLOv8 or Faster R-CNN]
    D --> T[Tracking<br>unique IDs · occlusion · speed]
    T --> L[Per-lane analysis<br>count · density · clearance time]
    L --> S[Signal controller<br>GREEN → YELLOW → RED]
    D -. emergency model .-> E[Emergency vehicle] --> S
    E --> A[Alerts<br>siren sound · SMS]
    L --> DB[(SQLite)]
    E --> DB
    S --> W[Flask REST API → HTML/CSS/JS dashboard]
```

## Specification and implementation

| Feature (from my specification) | Implementation | Code |
|---|---|---|
| Vehicle detection and counting | YOLOv8 (n/s/m) or Faster R-CNN; car, motorcycle, bus, truck; a tracker counts each vehicle once | `traffic/detection.py`, `traffic/tracking.py` |
| Density classification per lane | Low / Medium / High / Very High from the vehicle count in each lane (thresholds adjustable) | `traffic/analysis.py` |
| Road clearance time | Queue length (weighted by vehicle type) ÷ average speed, plus start-up time; speed measured from tracks when the camera is calibrated | `traffic/analysis.py` |
| Emergency vehicle detection | A second YOLOv8 model trained on ambulance / fire truck / police images runs alongside the main model | `traffic/detection.py`, `notebooks/train_emergency_detector.ipynb` |
| Emergency alert notification | Siren sound in the dashboard; SMS through Twilio (optional); 30 s cooldown | `traffic/alerts.py` |
| Traffic signal optimisation | Green time = clearance time, kept between a density minimum and 90 s; emergency → green at once | `traffic/signal.py` |
| Accuracy comparison (YOLOv8 vs R-CNN) | mAP@0.5, mAP@0.5:0.95, AP per vehicle type, FPS and inference time on COCO val2017 vehicles, with charts | `benchmark/compare_models.py`, `notebooks/benchmark_on_kaggle.ipynb` |
| Data storage and analysis | SQLite: runs, per-lane data every second, emergency log; History page with charts and CSV export | `traffic/storage.py` |
| Edge cases | Low light, fog / rain, camera shake, blur, frozen camera, occlusion (details below) | `traffic/preprocess.py`, `traffic/tracking.py` |
| Front end and back end | HTML, CSS, JavaScript (Chart.js) on a Flask REST API | `app.py`, `templates/`, `static/` |

## The rules in numbers

**Density per lane:** Low < 5 vehicles · Medium 5–9 · High 10–14 · Very High ≥ 15. The count is the median of the last 15 frames, so one bad frame does not change the level.

**Road clearance time** (per lane; lanes clear in parallel, so the slowest lane decides):

```
load       = sum of vehicle weights      car 1 · motorcycle 0.5 · bus 2 · truck 2
clearance  = 2 s start-up + load × 7 m ÷ speed
speed      = measured average speed of moving vehicles (if metres-per-pixel is set),
             otherwise a typical speed for the density level (8 / 6 / 4 / 3 m/s)
```

**Signal timing:** green = clearance time, at least 15 / 25 / 40 / 55 s for Low / Medium / High / Very High, at most 90 s. Red = 120 s cycle − green − 5 s yellow (at least 10 s). A running phase is never cut short; new timing applies from the next phase. **Emergency:** green immediately for 90 s, once per emergency.

These are simple, explainable models with adjustable values in `traffic/config.py`. They are estimates, not traffic-engineering predictions.

## Edge cases

| Problem | What the system does |
|---|---|
| Low light (night) | Detects low brightness; applies contrast enhancement (CLAHE) and gamma brightening before detection |
| Fog, rain, haze | Detects low contrast; applies CLAHE |
| Camera shake | Tracks image points with optical flow; if the whole image moves, the shift is smoothed out. If many points stay still, only vehicles moved, so traffic is not mistaken for shake |
| Occlusion | The tracker keeps a hidden vehicle for 15 frames, so it stays in the queue count and is not counted twice when it reappears |
| Blurry frame | Flagged (Laplacian variance), so you know the counts are less reliable |
| Frozen camera | Flagged when the identical frame repeats for 30 frames |

Every condition is shown on the dashboard and stored with the data. The benchmark also measures detection accuracy on darkened images, without and with the enhancement.

## Results

*Pending: run `notebooks/benchmark_on_kaggle.ipynb` (YOLOv8 vs Faster R-CNN) and `notebooks/train_emergency_detector.ipynb` (emergency model) on a free Kaggle GPU. The numbers and charts will be added here.*

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/models` | Available detectors, emergency models, SMS status |
| POST | `/api/jobs` | Start analysing an uploaded video, webcam number or stream URL (RTSP/HTTP) |
| GET | `/api/jobs/<id>` | Live state: per-lane counts, density, clearance, signal, alerts, image-quality flags |
| GET | `/api/jobs/<id>/stream` | Annotated video (MJPEG) |
| POST | `/api/jobs/<id>/stop` | Stop a job |
| GET | `/api/runs`, `/api/runs/<id>` | Stored runs, summary and emergency log |
| GET | `/api/runs/<id>/history`, `/api/runs/<id>/export.csv` | Per-second, per-lane data (JSON / CSV) |
| GET | `/api/benchmark` | Model comparison results |

## How to run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, upload a traffic video (or enter `0` for a webcam, or an RTSP address), and press **Start**. YOLOv8 weights download automatically on first use. `yolov8n` runs on a normal laptop CPU; Faster R-CNN is much slower without a GPU.

**SMS alerts (optional):** set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` and `ALERT_TO_NUMBER` before starting the app. Without them, alerts are sound-only.

**Tests** (57 tests; no GPU, video or YOLO needed, a fake detector is used):

```bash
pip install pytest
python -m pytest -q
```

## Deployment (planned)

The same Flask app can run on a local PC, on a cloud server reading the junction's RTSP camera stream, or on an edge device such as a Raspberry Pi next to the camera (with `yolov8n` and frame skipping). Only the local setup has been tested.

## Project structure

```
app.py                         Flask app: pages and REST API
traffic/config.py              all thresholds, timings and weights
traffic/detection.py           YOLOv8, Faster R-CNN and the combined emergency detector
traffic/tracking.py            IoU tracker: unique counts, occlusion, speed
traffic/analysis.py            lanes, density, clearance time
traffic/signal.py              signal timing and state machine
traffic/preprocess.py          edge-case handling
traffic/alerts.py              siren and SMS alerts
traffic/storage.py             SQLite database
traffic/pipeline.py            per-frame pipeline and background job
templates/, static/            dashboard, history and comparison pages (HTML, CSS, JS)
benchmark/compare_models.py    YOLOv8 vs Faster R-CNN benchmark
notebooks/                     Kaggle notebooks: benchmark, emergency-model training
tests/                         unit and system tests
```

## Limitations

- Lanes are equal vertical strips of the frame; a real junction needs lane shapes drawn for its camera.
- Speed is measured only after calibration (metres per pixel); otherwise typical speeds are used.
- The signal logic controls one approach; coordinating a full junction (or several) is a next step, for example with reinforcement learning.
- Emergency detection depends on the training dataset; sirens and flashing lights are not used.

## Credits

Original team project at Mohan Babu University; I wrote the specification and built and tested parts of the first version using AI tools. This extended version was written with AI assistance (Claude).

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [torchvision](https://pytorch.org/vision/) Faster R-CNN, OpenCV, Flask and Chart.js.
