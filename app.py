"""
app.py: Smart Traffic Management System (home page)

Pages:
  pages/1_Live_Detection.py    YOLOv8 vehicle detection on an uploaded video
  pages/2_Signal_Simulator.py  interactive signal simulation (no video needed)

Run:  streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.control import YELLOW_DURATION, calculate_signal_timing  # noqa: E402

st.set_page_config(page_title="Smart Traffic Management System", page_icon="🚦", layout="wide",
                   initial_sidebar_state="expanded")

st.title("🚦 Smart Traffic Management System")
st.markdown("Real-time vehicle counting, traffic density classification and adaptive signal timing "
            "with **YOLOv8** and **Streamlit**.")
st.divider()
st.subheader("📌 Choose a page from the sidebar")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
### 🎥 Live Detection
Upload a traffic video. YOLOv8 counts vehicles in each frame, the count is smoothed
and classified as Low, Medium or High density, and the signal controller runs on the
video's own clock.

| Feature | Detail |
|---|---|
| Models | YOLOv8 n / s / m, or your own weights in `models/` |
| Vehicle types | Car · Motorcycle · Bus · Truck |
| Output | Annotated video · traffic light · FPS and inference-time charts |
""")
with c2:
    st.markdown("""
### 🎮 Signal Simulator
An interactive simulation: add or remove vehicles, spawn an emergency vehicle,
and watch the same signal controller respond. No video needed.

| Feature | Detail |
|---|---|
| Controls | Add · remove · emergency · clear · auto traffic |
| Animation | JavaScript canvas with moving vehicles |
| Logic | Same `TrafficSignalController` as Live Detection |
""")

st.divider()
st.subheader("🧠 Signal timing rules")
rows = ["| Condition | 🟢 Green | 🟡 Yellow | 🔴 Red |", "|---|---|---|---|"]
for label, density, emergency in [("Emergency vehicle", "Low", True), ("High density (≥ 25 vehicles)", "High", False),
                                  ("Medium density (10–24)", "Medium", False), ("Low density (< 10)", "Low", False)]:
    t = calculate_signal_timing(density, emergency)
    rows.append(f"| {label} | {t['green_time']} s | {YELLOW_DURATION} s | {t['red_time']} s |")
st.markdown("\n".join(rows))
st.caption("Emergency detection needs custom-trained weights with ambulance, fire truck or police classes; "
           "the standard YOLOv8 (COCO) models do not include them.")
