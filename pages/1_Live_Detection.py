"""
pages/1_Live_Detection.py
Upload a traffic video: YOLOv8 counts vehicles frame by frame, the count is
smoothed and classified into a density level, and the signal controller runs
on video time.
"""

import os
import sys
import tempfile
import time

import cv2
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.control import TrafficSignalController, get_signal_recommendation  # noqa: E402
from utils.density import CountSmoother, get_density_emoji  # noqa: E402
from utils.detection import VehicleDetector, list_models  # noqa: E402


def render_traffic_light_html(phase, remaining, green_time, red_time, status):
    """HTML traffic light with the active lamp lit and a countdown."""
    dim = "#2a2a2a"

    def lamp(colour, active):
        background = colour if active else dim
        shadow = f"0 0 18px 6px {colour}" if active else "none"
        return f'<div style="width:64px;height:64px;border-radius:50%;background:{background};box-shadow:{shadow};"></div>'

    colours = {"GREEN": ("#00e676", "#00c853"), "YELLOW": ("#ffeb3b", "#fbc02d"), "RED": ("#ff1744", "#d50000")}
    timer_colour, border = colours[phase]
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:10px;padding:16px 8px;">
  <div style="background:#111;border-radius:14px;padding:14px 18px;display:flex;flex-direction:column;
              align-items:center;gap:12px;border:3px solid {border};box-shadow:0 4px 20px rgba(0,0,0,0.6);">
    {lamp('#ff1744', phase == 'RED')}
    {lamp('#ffeb3b', phase == 'YELLOW')}
    {lamp('#00e676', phase == 'GREEN')}
  </div>
  <div style="font-size:2.6rem;font-weight:900;font-family:monospace;color:{timer_colour};line-height:1;">{remaining:.0f}s</div>
  <div style="font-size:0.95rem;font-weight:700;letter-spacing:0.12em;color:{timer_colour};">{phase}</div>
  <div style="font-size:0.72rem;color:#aaa;text-align:center;max-width:200px;line-height:1.4;">
    {status}<br/>🟩 Green: {green_time}s &nbsp;|&nbsp; 🟥 Red: {red_time}s
  </div>
</div>
"""


st.set_page_config(page_title="Live Detection", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.selectbox(
        "YOLOv8 model",
        list_models(os.path.join(os.path.dirname(__file__), "..", "models")),
        help="n = fastest, s = small, m = most accurate. Custom .pt files in the models/ folder also appear here.",
    )
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.50, 0.05,
                           help="Detections below this confidence are ignored.")
    skip_frames = st.number_input("Process every Nth frame", min_value=1, max_value=10, value=1)
    smooth_window = st.slider("Smoothing window (frames)", 1, 30, 15,
                              help="Density uses the median count of the last N processed frames. 1 = no smoothing.")
    st.divider()
    st.header("📊 Density thresholds")
    low_thresh = st.number_input("Low → Medium boundary", value=10, min_value=1, max_value=50)
    high_thresh = st.number_input("Medium → High boundary", value=25, min_value=5, max_value=200)
    if low_thresh > high_thresh:
        st.error("The Low → Medium boundary must not be larger than the Medium → High boundary.")
        st.stop()

# ---- Page ----------------------------------------------------------------------
st.title("🎥 Live Traffic Detection")
st.markdown("*Upload a traffic video. YOLOv8 counts vehicles, the density level is classified, "
            "and the signal controller runs on the video's own clock.*")
st.divider()

video_file = st.file_uploader("📁 Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])

if video_file is None:
    st.info("👆 Upload a traffic video to begin.")
    st.markdown("""
| Feature | Detail |
|---|---|
| Vehicle types counted | Car · Motorcycle · Bus · Truck (COCO classes) |
| Density levels | Low (< 10) · Medium (10–24) · High (≥ 25), adjustable |
| Signal timing | Green 20 / 40 / 60 s by density, 90 s emergency override |
| Emergency vehicles | Needs custom-trained weights with ambulance / fire truck / police classes |
""")
    st.stop()

st.success(f"Uploaded: **{video_file.name}**")
if not st.button("▶️ Start analysis", type="primary"):
    st.stop()

suffix = os.path.splitext(video_file.name)[-1] or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(video_file.read())
    tmp_path = tmp.name

try:
    with st.spinner(f"Loading {model_name} …"):
        detector = VehicleDetector(model_name, confidence)
    if not detector.supports_emergency:
        st.info("ℹ️ This model has no emergency-vehicle classes, so emergency detection is off. "
                "Add custom-trained weights to the models/ folder to enable it.")

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("❌ Unable to open the video file.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.info(f"Video: **{total_frames}** frames · **{src_fps:.1f}** frames per second")

    vid_col, metrics_col = st.columns([3, 2])
    with vid_col:
        st.subheader("🎥 Live feed")
        frame_ph = st.empty()
        progress_ph = st.progress(0, text="Starting…")
    with metrics_col:
        st.subheader("📊 Live metrics")
        count_ph, types_ph, density_ph, signal_ph, alert_ph = (st.empty() for _ in range(5))
        st.divider()
        st.markdown("**🚦 Signal status**")
        light_ph = st.empty()
    st.subheader("📈 Performance")
    chart_ph = st.empty()

    fps_hist, inf_hist = [], []
    frame_idx = processed = 0
    controller = TrafficSignalController(start_time=0.0)
    smoother = CountSmoother(window=smooth_window)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % skip_frames:
            continue

        t0 = time.perf_counter()
        det = detector.detect(frame)
        elapsed = time.perf_counter() - t0
        fps_hist.append(1.0 / elapsed if elapsed > 0 else 0.0)
        inf_hist.append(det["inference_ms"])
        processed += 1

        video_time = frame_idx / src_fps  # seconds into the video
        smoothed = smoother.update(det["vehicle_count"])
        signal = get_signal_recommendation(smoothed, det["emergency"], low_thresh, high_thresh)
        controller.update(signal, now=video_time)

        frame_ph.image(cv2.cvtColor(det["annotated"], cv2.COLOR_BGR2RGB))
        if total_frames > 0:
            progress_ph.progress(min(frame_idx / total_frames, 1.0),
                                 text=f"Frame {frame_idx} / {total_frames} · video time {video_time:.1f} s")

        count_ph.metric("🚗 Vehicles (this frame / smoothed)", f"{det['vehicle_count']} / {smoothed}")
        types_ph.caption(" · ".join(f"{name}: {n}" for name, n in det["by_type"].items()))
        density_ph.metric("Traffic density", f"{get_density_emoji(signal['density'])} {signal['density']}")
        signal_ph.metric("🚦 Recommended green phase", f"{signal['green_time']} s", delta=f"Red: {signal['red_time']} s",
                         delta_color="off", help=signal["green_status"])
        if det["emergency"]:
            alert_ph.error("🚨 EMERGENCY VEHICLE DETECTED: " + " · ".join(det["emergency_types"]))
        elif detector.supports_emergency:
            alert_ph.success("No emergency vehicles detected")

        light_ph.markdown(render_traffic_light_html(controller.phase, controller.remaining(),
                                                    controller.green_duration, controller.red_duration,
                                                    controller.current_status), unsafe_allow_html=True)

        if processed % 20 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.5))
            ax1.plot(fps_hist[-60:], color="#1f77b4", linewidth=1.5)
            ax1.axhline(src_fps / skip_frames, color="red", linestyle="--", linewidth=1, label="needed for real time")
            ax1.set_title("Processing FPS", fontsize=10)
            ax1.set_xlabel("Processed frame")
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)
            ax2.plot(inf_hist[-60:], color="#ff7f0e", linewidth=1.5)
            ax2.set_title("Inference time (ms)", fontsize=10)
            ax2.set_xlabel("Processed frame")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            chart_ph.pyplot(fig)
            plt.close(fig)

    cap.release()
    progress_ph.progress(1.0, text="✅ Analysis complete")

    st.divider()
    st.subheader("📋 Analysis summary")
    avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
    avg_inf = sum(inf_hist) / len(inf_hist) if inf_hist else 0.0
    # With frame skipping we only need to process 1 of every N frames in time.
    real_time = avg_fps * skip_frames >= src_fps
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Frames processed", processed)
    s2.metric("Average processing FPS", f"{avg_fps:.1f}")
    s3.metric("Average inference time", f"{avg_inf:.1f} ms")
    s4.metric("Real-time capable?", "Yes ✅" if real_time else "No ❌",
              help=f"Needs {src_fps / skip_frames:.1f} processed frames per second with every {skip_frames} frame(s) processed.")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
