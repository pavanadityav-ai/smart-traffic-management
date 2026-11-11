"""
The per-frame pipeline and the background job that runs it on a video or camera.

    frame -> edge-case handling -> detection -> tracking -> per-lane analysis
          -> signal timing -> emergency alert -> database (once per second)
"""

import threading
import time

import cv2

from traffic import config
from traffic.alerts import AlertManager
from traffic.analysis import CountSmoother, analyse_lanes
from traffic.preprocess import FrameConditioner
from traffic.signal import SignalController, signal_timing
from traffic.tracking import Tracker

DENSITY_BGR = {"Low": (80, 200, 80), "Medium": (0, 210, 255), "High": (0, 140, 255), "Very High": (60, 60, 230)}
PHASE_BGR = {"GREEN": (80, 200, 80), "YELLOW": (0, 210, 255), "RED": (60, 60, 230)}


class TrafficPipeline:
    def __init__(self, detector, n_lanes: int = config.DEFAULT_LANES, thresholds=config.DENSITY_THRESHOLDS,
                 meters_per_pixel: float = config.METERS_PER_PIXEL, storage=None, run_id=None,
                 alerts: AlertManager | None = None, conditioner: FrameConditioner | None = None,
                 start_time: float = 0.0) -> None:
        self.detector = detector
        self.n_lanes = n_lanes
        self.thresholds = thresholds
        self.meters_per_pixel = meters_per_pixel
        self.storage, self.run_id = storage, run_id
        self.alerts = alerts or AlertManager()
        self.conditioner = conditioner or FrameConditioner()
        self.tracker = Tracker()
        self.smoothers = [CountSmoother() for _ in range(n_lanes)]
        self.controller = SignalController(start_time)
        self._last_log = None
        self.frames = 0
        self.total_ms = 0.0

    def process(self, frame, t: float) -> tuple:
        """Process one BGR frame at time t (seconds). Returns (annotated frame, state dict)."""
        frame, flags = self.conditioner.process(frame)
        detections, inference_ms = self.detector.detect(frame)
        tracks = self.tracker.update(detections, t)
        analysis = analyse_lanes(tracks, frame.shape[1], self.n_lanes, self.smoothers,
                                 self.thresholds, self.meters_per_pixel)

        # an emergency vehicle counts once it is confirmed and still in view
        emergency_types = sorted({tr.label for tr in tracks if tr.emergency and tr.misses <= 5})
        emergency = bool(emergency_types)
        timing = signal_timing(analysis["density"], analysis["clearance_s"], emergency)
        self.controller.update(timing, emergency, t)
        signal = self.controller.state()

        alert = self.alerts.check(emergency_types, t, on_sms_result=self._sms_done)
        if alert and self.storage:
            alert["row_id"] = self.storage.log_emergency(self.run_id, alert)
        if self.storage and (self._last_log is None or t - self._last_log >= config.LOG_INTERVAL_S):
            self.storage.log_traffic(self.run_id, t, analysis, signal, flags)
            self._last_log = t

        self.frames += 1
        self.total_ms += inference_ms
        state = {
            "time_s": round(t, 1),
            "inference_ms": round(inference_ms, 1),
            "analysis": analysis,
            "signal": signal,
            "next_timing": timing,
            "emergency": emergency,
            "emergency_types": emergency_types,
            "alerts": self.alerts.alerts[-10:],
            "flags": flags,
            "unique_counts": dict(self.tracker.unique_counts),
            "supports_emergency": getattr(self.detector, "supports_emergency", False),
        }
        return draw(frame, tracks, self.n_lanes, analysis, signal), state

    def _sms_done(self, alert: dict) -> None:
        if self.storage and "row_id" in alert:
            self.storage.update_sms(alert["row_id"], alert["sms"])


def draw(frame, tracks, n_lanes: int, analysis: dict, signal: dict):
    out = frame.copy()
    h, w = out.shape[:2]
    for i in range(1, n_lanes):
        x = int(w * i / n_lanes)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        colour = (40, 40, 255) if tr.emergency else (255, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, f"#{tr.id} {tr.label}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                    cv2.LINE_AA)
    for lane in analysis["lanes"]:
        x = int(w * (lane["lane"] - 1) / n_lanes) + 6
        text = f"L{lane['lane']}: {lane['count']} {lane['density']}"
        cv2.rectangle(out, (x - 4, 4), (x + 8 * len(text), 24), (0, 0, 0), -1)
        cv2.putText(out, text, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DENSITY_BGR[lane["density"]], 1, cv2.LINE_AA)
    label = f"{signal['phase']} {signal['remaining_s']:.0f}s"
    cv2.rectangle(out, (w - 130, h - 34), (w - 6, h - 6), (0, 0, 0), -1)
    cv2.putText(out, label, (w - 124, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PHASE_BGR[signal["phase"]], 2, cv2.LINE_AA)
    return out


class ProcessingJob(threading.Thread):
    """Runs the pipeline on a video file, webcam index or stream URL (RTSP/HTTP) in the background."""

    def __init__(self, source, pipeline: TrafficPipeline, storage=None, run_id=None, pace: bool = True) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.pipeline = pipeline
        self.storage, self.run_id = storage, run_id
        self.pace = pace  # play files at their real speed so the dashboard looks natural
        self.status = "starting"
        self.error = None
        self.progress = 0.0
        self.fps = 0.0
        self._stop_event = threading.Event()
        self._busy_s = 0.0  # total processing time, for the average FPS
        self._lock = threading.Lock()
        self._jpeg = None
        self._state = None

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> tuple:
        with self._lock:
            return self._jpeg, self._state

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status, self.error = "error", f"Cannot open source: {self.source}"
            self._finish()
            return
        is_file = isinstance(self.source, str) and not self.source.lower().startswith(("rtsp://", "http://", "https://"))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
        start = time.monotonic()
        frame_idx = 0
        recent = []
        self.status = "running"
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                t = frame_idx / src_fps if is_file else time.monotonic() - start
                t0 = time.perf_counter()
                annotated, state = self.pipeline.process(frame, t)
                took = time.perf_counter() - t0
                self._busy_s += took
                recent = (recent + [took])[-30:]
                self.fps = len(recent) / sum(recent) if sum(recent) > 0 else 0.0
                state["fps"] = round(self.fps, 1)
                state["realtime_fps_needed"] = round(src_fps, 1)
                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self._jpeg = jpeg.tobytes() if ok else self._jpeg
                    self._state = state
                if total:
                    self.progress = min(frame_idx / total, 1.0)
                if is_file and self.pace:
                    ahead = t - (time.monotonic() - start)
                    if ahead > 0:
                        time.sleep(ahead)
            self.status = "stopped" if self._stop_event.is_set() else "finished"
        except Exception as exc:  # report the error on the dashboard instead of dying silently
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()
            self._finish()

    def _finish(self) -> None:
        if self.storage and self.run_id:
            p = self.pipeline
            avg_ms = p.total_ms / p.frames if p.frames else 0.0
            avg_fps = p.frames / self._busy_s if self._busy_s else 0.0
            self.storage.finish_run(self.run_id, self.status, p.frames, avg_fps, avg_ms, dict(p.tracker.unique_counts))
