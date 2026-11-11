"""
pages/2_Signal_Simulator.py
Interactive traffic-signal simulation: add or remove vehicles, spawn an
emergency vehicle, and watch the same controller used on real video respond.
The road is drawn with a small JavaScript canvas; all timing logic is in Python.
"""

import json
import os
import random
import sys
import time

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.control import YELLOW_DURATION, TrafficSignalController, get_signal_recommendation  # noqa: E402

MAX_VEHICLES = 80

st.set_page_config(page_title="Signal Simulator", page_icon="🎮", layout="wide", initial_sidebar_state="expanded")
st.title("🎮 Smart Traffic Signal Simulator")
st.markdown("*Add vehicles, spawn an emergency, or clear the road, and watch the signal respond "
            "using the same controller as the live detection page.*")
st.divider()

# ---- Sidebar controls ----------------------------------------------------------
with st.sidebar:
    st.header("🎮 Simulation controls")
    add_n = st.number_input("Vehicles per click", 1, 20, 3)
    col_a, col_b = st.columns(2)
    add_clicked = col_a.button("➕ Add")
    remove_clicked = col_b.button("➖ Remove")
    emergency_clicked = st.button("🚨 Spawn emergency vehicle", type="primary")
    clear_clicked = st.button("🗑️ Clear all vehicles")
    st.divider()
    st.header("📊 Thresholds")
    low_thresh = st.number_input("Low → Medium", value=10, min_value=1, max_value=50)
    high_thresh = st.number_input("Medium → High", value=25, min_value=5, max_value=200)
    if low_thresh > high_thresh:
        st.error("The Low → Medium boundary must not be larger than the Medium → High boundary.")
        st.stop()
    st.divider()
    auto_traffic = st.toggle("🔄 Auto-generate traffic", value=True)
    auto_interval = st.slider("Auto-spawn interval (s)", 1, 10, 3)

# ---- Session state ---------------------------------------------------------------
ss = st.session_state
ss.setdefault("sim_count", 5)
ss.setdefault("sim_emergency", False)
ss.setdefault("sim_ctrl", TrafficSignalController(start_time=time.time()))
ss.setdefault("sim_log", [])
ss.setdefault("last_auto", time.time())

if add_clicked:
    ss.sim_count = min(ss.sim_count + add_n, MAX_VEHICLES)
    ss.sim_log.append(f"➕ Added {add_n} vehicles → {ss.sim_count} total")
if remove_clicked:
    removed = min(add_n, ss.sim_count)
    ss.sim_count -= removed
    ss.sim_log.append(f"➖ Removed {removed} vehicles → {ss.sim_count} total")
if emergency_clicked:
    ss.sim_emergency = True
    ss.sim_log.append("🚨 Emergency vehicle spawned")
if clear_clicked:
    ss.sim_count = 0
    ss.sim_emergency = False
    ss.sim_log.append("🗑️ Road cleared")

now = time.time()
if auto_traffic and now - ss.last_auto >= auto_interval:
    ss.sim_count = max(0, min(MAX_VEHICLES, ss.sim_count + random.randint(-2, 4)))
    ss.last_auto = now

# ---- Signal state from the Python controller ------------------------------------
count, emergency = ss.sim_count, ss.sim_emergency
signal = get_signal_recommendation(count, emergency, low_thresh, high_thresh)
ctrl = ss.sim_ctrl
ctrl.update(signal, now=now)
phase, remaining, density = ctrl.phase, ctrl.remaining(now), signal["density"]

# The emergency is over once its GREEN and YELLOW have finished
if emergency and phase == "RED":
    ss.sim_emergency = False
    ss.sim_log.append("✅ Emergency cleared, back to the normal cycle")

ss.sim_log = ss.sim_log[-40:]

js_data = json.dumps({
    "phase": phase, "remaining": round(remaining, 1),
    "greenTime": ctrl.green_duration, "yellowTime": YELLOW_DURATION, "redTime": ctrl.red_duration,
    "count": count, "density": density, "emergency": emergency, "status": ctrl.current_status,
    "lowThresh": low_thresh, "highThresh": high_thresh,
})

m1, m2, m3, m4 = st.columns(4)
m1.metric("🚗 Vehicles on road", count)
m2.metric("Traffic density", density)
m3.metric("Current phase", phase)
m4.metric("⏱ Time remaining", f"{remaining:.0f} s")
if emergency:
    st.error("🚨 EMERGENCY VEHICLE ACTIVE: signal overridden to GREEN for 90 s")
st.divider()

# ---- Canvas (JavaScript). Double braces {{ }} are literal braces in this f-string.
HTML = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ margin:0; background:#1a1a2e; font-family:'Segoe UI',sans-serif; }}
  canvas {{ display:block; margin:0 auto; max-width:100%; }}
  #hud {{ color:#eee; text-align:center; font-size:13px; padding:6px 0 2px; }}
</style>
</head>
<body>
<div id="hud">Loading…</div>
<canvas id="c"></canvas>
<script>
const DATA = {js_data};

const canvas = document.getElementById('c');
const W = 780, H = 520;
canvas.width = W;
canvas.height = H;
const ctx = canvas.getContext('2d');

const state = {{ ...DATA }};
let lastTick = performance.now();
let frameCount = 0;

const COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#ecf0f1','#c0392b','#2980b9'];
const VEHICLE_TYPES = [
  {{ label:'car',        w:36, h:20, emoji:'🚗' }},
  {{ label:'bus',        w:54, h:24, emoji:'🚌' }},
  {{ label:'truck',      w:50, h:22, emoji:'🚚' }},
  {{ label:'motorcycle', w:24, h:14, emoji:'🏍️' }},
];
const EM_VEHICLE = {{ label:'ambulance', w:44, h:22, emoji:'🚑' }};

const ROAD_Y = 220;   // centre line of the road
const LANE_H = 44;
const LIGHT_X = W/2;  // stop line
const MAX_ON_SCREEN = 24;

let vehicles = [];
let nextId = 0;
let emSpawned = false;  // one ambulance per emergency event

function spawnVehicle(emergency=false) {{
  const t = emergency ? EM_VEHICLE : VEHICLE_TYPES[Math.floor(Math.random()*VEHICLE_TYPES.length)];
  const lane = emergency ? 0 : Math.floor(Math.random()*2);
  vehicles.push({{
    id: nextId++,
    x: -t.w - Math.random()*220,
    y: ROAD_Y - LANE_H/2 + lane*LANE_H - t.h/2,
    speed: emergency ? 2.8 : 0.6 + Math.random()*0.8,
    typeLabel: t.label, type: t,
    color: emergency ? '#ff4444' : COLORS[Math.floor(Math.random()*COLORS.length)],
    waiting: false, passed: false, emergency: emergency,
  }});
}}

// Streamlit redraws this frame about once a second. window.name survives the
// redraw, so we keep the vehicles there and the animation does not restart.
function saveState() {{
  try {{
    const slim = vehicles.map(v => [v.id, Math.round(v.x*10)/10, Math.round(v.y*10)/10, v.speed,
                                    v.typeLabel, v.color, v.waiting?1:0, v.passed?1:0, v.emergency?1:0]);
    window.name = JSON.stringify({{ v: slim, n: nextId, e: emSpawned }});
  }} catch(e) {{}}
}}

function loadState() {{
  try {{
    const saved = JSON.parse(window.name || 'null');
    if (!saved || !saved.v) return false;
    nextId = saved.n || 0;
    emSpawned = !!saved.e;
    vehicles = saved.v.map(r => {{
      const t = r[8] ? EM_VEHICLE : (VEHICLE_TYPES.find(t => t.label === r[4]) || VEHICLE_TYPES[0]);
      return {{ id:r[0], x:r[1], y:r[2], speed:r[3], typeLabel:r[4], type:t, color:r[5],
               waiting:!!r[6], passed:!!r[7], emergency:!!r[8] }};
    }});
    return true;
  }} catch(e) {{ return false; }}
}}

if (!loadState()) {{
  for (let i = 0; i < Math.min(state.count, MAX_ON_SCREEN); i++) spawnVehicle(false);
}}
if (state.emergency && !emSpawned) {{ spawnVehicle(true); emSpawned = true; }}
if (!state.emergency) emSpawned = false;
saveState();

function drawRoad() {{
  const sky = ctx.createLinearGradient(0,0,0,H);
  sky.addColorStop(0,'#0d1117');
  sky.addColorStop(1,'#161b22');
  ctx.fillStyle = sky;
  ctx.fillRect(0,0,W,H);

  ctx.fillStyle = '#2d3436';
  ctx.fillRect(0, ROAD_Y - LANE_H, W, LANE_H*2);

  ctx.strokeStyle = '#fdcb6e';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(0, ROAD_Y - LANE_H); ctx.lineTo(W, ROAD_Y - LANE_H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, ROAD_Y + LANE_H); ctx.lineTo(W, ROAD_Y + LANE_H); ctx.stroke();

  ctx.strokeStyle = '#dfe6e9';
  ctx.lineWidth = 2;
  ctx.setLineDash([20,14]);
  ctx.beginPath(); ctx.moveTo(0, ROAD_Y); ctx.lineTo(W, ROAD_Y); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(LIGHT_X, ROAD_Y-LANE_H-4); ctx.lineTo(LIGHT_X, ROAD_Y+LANE_H+4); ctx.stroke();
}}

function drawTrafficLight() {{
  const lx = LIGHT_X + 20, ly = ROAD_Y - LANE_H - 130, poleH = 130;
  ctx.fillStyle = '#636e72';
  ctx.fillRect(lx+17, ly+poleH-10, 6, poleH);

  const dim = '#2d3436';
  const phases = {{ GREEN:'#00b894', YELLOW:'#fdcb6e', RED:'#d63031' }};
  const activeCol = phases[state.phase];

  ctx.shadowBlur = 28; ctx.shadowColor = activeCol;
  ctx.fillStyle = '#1e272e';
  ctx.beginPath(); ctx.roundRect(lx, ly, 40, 110, 8); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = activeCol; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(lx, ly, 40, 110, 8); ctx.stroke();

  [['#d63031', state.phase==='RED'], ['#fdcb6e', state.phase==='YELLOW'], ['#00b894', state.phase==='GREEN']]
    .forEach(([col, active], i) => {{
      ctx.shadowBlur = active ? 22 : 0;
      ctx.shadowColor = col;
      ctx.fillStyle = active ? col : dim;
      ctx.beginPath(); ctx.arc(lx+20, ly+20+i*34, 13, 0, Math.PI*2); ctx.fill();
      ctx.shadowBlur = 0;
    }});

  const ty = state.phase==='RED' ? ly+88 : state.phase==='YELLOW' ? ly+54 : ly+20;
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(Math.ceil(state.remaining)+'s', lx+20, ty+4);
  ctx.font = 'bold 15px monospace';
  ctx.fillStyle = activeCol;
  ctx.fillText(state.phase, lx+20, ly+130);
  ctx.textAlign = 'left';
}}

function drawVehicle(v) {{
  const {{ x, y, type, color }} = v;
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(x+3, y+type.h-3, type.w, 5);
  ctx.fillStyle = v.emergency ? '#ff4444' : color;
  ctx.beginPath(); ctx.roundRect(x, y, type.w, type.h, 4); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.18)';
  ctx.fillRect(x+4, y+3, type.w-8, 5);
  ctx.font = `${{Math.min(type.h-2,16)}}px serif`;
  ctx.textAlign = 'center';
  ctx.fillText(type.emoji, x+type.w/2, y+type.h-2);
  ctx.textAlign = 'left';
}}

function drawHUD() {{
  const densityColors = {{ Low:'#00b894', Medium:'#fdcb6e', High:'#d63031' }};
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath(); ctx.roundRect(12,12,260,90,10); ctx.fill();

  ctx.font = 'bold 13px Segoe UI';
  ctx.fillStyle = '#dfe6e9';
  ctx.fillText('🚗 Vehicles: ' + state.count, 24, 34);
  ctx.fillStyle = densityColors[state.density] || '#fff';
  ctx.fillText('● Density: ' + state.density, 24, 56);

  ctx.fillStyle = '#b2bec3';
  ctx.font = '11px Segoe UI';
  let line = '', sy = 76;
  (state.status || '').split(' ').forEach(w => {{
    if ((line + w).length > 34) {{ ctx.fillText(line, 24, sy); line = w + ' '; sy += 14; }}
    else line += w + ' ';
  }});
  if (line) ctx.fillText(line, 24, sy);

  if (state.emergency) {{
    ctx.fillStyle = '#d63031';
    ctx.font = 'bold 12px Segoe UI';
    ctx.fillText('🚨 EMERGENCY OVERRIDE', 24, sy+20);
  }}

  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.beginPath(); ctx.roundRect(W-170, H-54, 158, 42, 8); ctx.fill();
  ctx.font = '10px Segoe UI'; ctx.fillStyle = '#b2bec3';
  ctx.fillText(`Low<${{DATA.lowThresh}}  Mid<${{DATA.highThresh}}  High≥${{DATA.highThresh}}`, W-162, H-35);
  ctx.fillText(`🟢${{DATA.greenTime}}s 🟡${{DATA.yellowTime}}s 🔴${{DATA.redTime}}s`, W-162, H-18);
}}

function update(now) {{
  const dt = (now - lastTick) / 1000;
  lastTick = now;
  frameCount++;
  state.remaining = Math.max(0, state.remaining - dt);  // smooth local countdown

  const canGo = state.phase === 'GREEN';
  const slowDown = state.phase === 'YELLOW';
  const stopX = LIGHT_X - 8;

  vehicles.forEach(v => {{
    if (v.passed) {{ v.x += v.speed * 60 * dt; return; }}
    const front = v.x + v.type.w;

    let minGap = Infinity;  // distance to the nearest vehicle ahead in the same lane
    vehicles.forEach(o => {{
      if (o.id !== v.id && Math.abs(o.y - v.y) < 20 && o.x > v.x) minGap = Math.min(minGap, o.x - front);
    }});
    const blocked = minGap < 10;
    const atStopLine = front >= stopX - 4 && front <= stopX + 4;
    const mustStop = !canGo && !v.emergency && atStopLine;

    if (mustStop) {{
      v.waiting = true;
    }} else if (!blocked) {{
      v.waiting = false;
      v.x += (slowDown ? v.speed * 0.4 : v.speed) * 60 * dt;
    }}
    if (v.x > W + 20) v.passed = true;
  }});

  vehicles = vehicles.filter(v => !(v.passed && v.x > W + 80));
  const active = vehicles.filter(v => !v.passed).length;
  if (active < Math.min(state.count, MAX_ON_SCREEN)) spawnVehicle(false);  // at most one per tick
  if (frameCount % 10 === 0) saveState();
}}

function draw() {{
  ctx.clearRect(0,0,W,H);
  drawRoad();
  vehicles.forEach(drawVehicle);
  drawTrafficLight();
  drawHUD();
}}

function loop(now) {{
  update(now);
  draw();
  document.getElementById('hud').textContent =
    state.phase + ' · ' + Math.ceil(state.remaining) + 's remaining · ' + state.count + ' vehicles · ' + state.density + ' density';
  requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

sim_col, log_col = st.columns([3, 1])
with sim_col:
    components.html(HTML, height=560, scrolling=False)
with log_col:
    st.subheader("📋 Event log")
    for entry in reversed(ss.sim_log[-15:]):
        st.markdown(f"- {entry}")

st.divider()
st.subheader("🗂️ Current signal timing")
p1, p2, p3, p4 = st.columns(4)
p1.metric("🟢 Green phase", f"{ctrl.green_duration} s", help=signal["green_status"])
p2.metric("🟡 Yellow phase", f"{YELLOW_DURATION} s")
p3.metric("🔴 Red phase", f"{ctrl.red_duration} s", help=signal["red_status"])
p4.metric("🔁 Full cycle", f"{ctrl.green_duration + YELLOW_DURATION + ctrl.red_duration} s")
st.caption(f"**Next recommendation:** green {signal['green_time']} s, red {signal['red_time']} s "
           f"(used when the next phase starts)")

# Re-run about once a second so the Python signal state stays live
time.sleep(0.9)
st.rerun()
