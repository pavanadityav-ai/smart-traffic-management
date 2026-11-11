// Live dashboard: starts a job through the REST API, shows the MJPEG stream and polls the status.

const $ = (id) => document.getElementById(id);
const DENSITY_COLOURS = { "Low": "#3fb950", "Medium": "#d29922", "High": "#f0883e", "Very High": "#f85149" };
const LANE_COLOURS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#39c5cf"];
const FLAG_NAMES = { low_light: "Low light (enhanced)", low_contrast: "Fog / rain / haze (enhanced)",
                     blurry: "Blurry frame", camera_shake: "Camera shake (stabilised)", frozen: "Camera frozen" };

// Charts still work without the Chart.js CDN (for example offline): they are simply not drawn.
const makeChart = (el, cfg) => (window.Chart ? new Chart(el, cfg) : { data: cfg.data, update() {} });
if (window.Chart) { Chart.defaults.color = "#c9d1d9"; Chart.defaults.borderColor = "#30363d"; }

let jobId = null, pollTimer = null, lastChartTime = -1, lastAlertId = 0, audio = null;

// ---- charts -----------------------------------------------------------------
const chartOptions = (yTitle) => ({
  animation: false, responsive: true, maintainAspectRatio: false,
  scales: { x: { title: { display: true, text: "time (s)" }, ticks: { color: "#8b949e" } },
            y: { beginAtZero: true, title: { display: true, text: yTitle }, ticks: { color: "#8b949e" } } },
  plugins: { legend: { labels: { color: "#c9d1d9" } } },
});
const laneChart = makeChart($("lane-chart"), { type: "line", data: { labels: [], datasets: [] }, options: chartOptions("vehicles") });
const fpsChart = makeChart($("fps-chart"), {
  type: "line",
  data: { labels: [], datasets: [
    { label: "processing FPS", data: [], borderColor: "#58a6ff", pointRadius: 0 },
    { label: "needed for real time", data: [], borderColor: "#f85149", borderDash: [6, 4], pointRadius: 0 },
  ] },
  options: chartOptions("frames per second"),
});

function resetCharts(lanes) {
  laneChart.data.labels = [];
  laneChart.data.datasets = Array.from({ length: lanes }, (_, i) => ({
    label: `Lane ${i + 1}`, data: [], borderColor: LANE_COLOURS[i % LANE_COLOURS.length], pointRadius: 0 }));
  fpsChart.data.labels = [];
  fpsChart.data.datasets.forEach((d) => (d.data = []));
  laneChart.update(); fpsChart.update();
  lastChartTime = -1;
}

function addChartPoint(state) {
  if (state.time_s - lastChartTime < 1) return;  // one point per second
  lastChartTime = state.time_s;
  const t = Math.round(state.time_s);
  laneChart.data.labels.push(t);
  state.analysis.lanes.forEach((lane, i) => laneChart.data.datasets[i] && laneChart.data.datasets[i].data.push(lane.count));
  fpsChart.data.labels.push(t);
  fpsChart.data.datasets[0].data.push(state.fps);
  fpsChart.data.datasets[1].data.push(state.realtime_fps_needed);
  for (const chart of [laneChart, fpsChart]) {
    if (chart.data.labels.length > 120) { chart.data.labels.shift(); chart.data.datasets.forEach((d) => d.data.shift()); }
    chart.update();
  }
}

// ---- siren (Web Audio; needs one click first, which the Start button provides) ----
function siren() {
  if (!audio) return;
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.type = "square"; gain.gain.value = 0.05;
  osc.connect(gain).connect(audio.destination);
  const now = audio.currentTime;
  for (let i = 0; i < 4; i++) {
    osc.frequency.setValueAtTime(700, now + i * 0.5);
    osc.frequency.setValueAtTime(1000, now + i * 0.5 + 0.25);
  }
  osc.start(now); osc.stop(now + 2);
}

// ---- rendering ----------------------------------------------------------------
function cell(row, text) { const td = document.createElement("td"); td.textContent = text; row.appendChild(td); }

function render(data) {
  const state = data.state;
  $("progress-bar").style.width = `${Math.round((data.progress || 0) * 100)}%`;
  $("sms-status").textContent = data.sms_enabled ? "SMS alerts: on (Twilio)" : "SMS alerts: off (set the Twilio variables to enable)";
  if (!state) return;

  const s = state.signal;
  ["RED", "YELLOW", "GREEN"].forEach((p) => $(`lamp-${p}`).classList.toggle("on", s.phase === p));
  $("phase").textContent = s.phase;
  $("countdown").textContent = `${Math.ceil(s.remaining_s)} s`;
  $("reason").textContent = s.reason;

  const a = state.analysis;
  $("density").textContent = a.density;
  $("density").style.color = DENSITY_COLOURS[a.density];
  $("clearance").textContent = `${a.clearance_s} s`;
  $("next-timing").textContent = `${state.next_timing.green_s} s / ${state.next_timing.red_s} s`;
  $("inference").textContent = `${state.inference_ms} ms · ${state.fps} FPS`;

  const lanes = $("lanes");
  lanes.replaceChildren();
  a.lanes.forEach((lane) => {
    const card = document.createElement("div");
    card.className = "lane";
    card.style.borderLeftColor = DENSITY_COLOURS[lane.density];
    const title = document.createElement("strong");
    title.textContent = `Lane ${lane.lane}: ${lane.count} vehicles · ${lane.density}`;
    const detail = document.createElement("div");
    detail.className = "muted small";
    const types = Object.entries(lane.by_type).map(([k, v]) => `${k} ${v}`).join(", ") || "empty";
    const speed = `${lane.speed_mps} m/s${lane.speed_measured ? "" : " (typical)"}`;
    detail.textContent = `${types} · clearance ${lane.clearance_s} s · speed ${speed}`;
    card.append(title, detail);
    lanes.appendChild(card);
  });

  const unique = Object.entries(state.unique_counts).map(([k, v]) => `${k}: ${v}`).join(" · ");
  $("unique").textContent = unique || "—";

  const flags = $("flags");
  flags.replaceChildren();
  Object.entries(FLAG_NAMES).forEach(([key, name]) => {
    if (state.flags[key]) { const b = document.createElement("span"); b.className = "badge"; b.textContent = name; flags.appendChild(b); }
  });
  if (!state.supports_emergency) {
    const b = document.createElement("span"); b.className = "badge info";
    b.textContent = "Emergency detection off: this model has no ambulance / fire / police classes";
    flags.appendChild(b);
  }

  $("emergency-banner").classList.toggle("hidden", !state.emergency);
  $("emergency-types").textContent = state.emergency_types.join(", ");

  const body = $("alerts");
  body.replaceChildren();
  state.alerts.slice().reverse().forEach((al) => {
    const row = document.createElement("tr");
    cell(row, `${al.time_s} s`); cell(row, al.types.join(", ")); cell(row, al.sound ? "played" : "—"); cell(row, al.sms);
    body.appendChild(row);
  });
  const newest = state.alerts.length ? state.alerts[state.alerts.length - 1].id : 0;
  if (newest > lastAlertId) { lastAlertId = newest; siren(); }

  addChartPoint(state);
}

async function poll() {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) return;
  const data = await res.json();
  render(data);
  const labels = { running: "Running…", starting: "Starting…", finished: "Finished. Results are saved in History.",
                   stopped: "Stopped. Results are saved in History." };
  $("status").textContent = data.status === "error" ? `Error: ${data.error}` : labels[data.status];
  if (!["starting", "running"].includes(data.status)) {
    clearInterval(pollTimer);
    $("start").disabled = false; $("stop").disabled = true;
  }
}

// ---- form -----------------------------------------------------------------------
async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  const select = $("model");
  data.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m === "faster_rcnn" ? "Faster R-CNN (ResNet-50, slower)" : m;
    select.appendChild(opt);
  });
  const em = $("emergency-model");
  if (data.emergency_models.length) em.options[0].textContent = "None";
  data.emergency_models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m.split(/[\\/]/).pop();
    em.appendChild(opt);
  });
}

$("job-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!audio) { try { audio = new AudioContext(); } catch (e) { audio = null; } }
  const form = new FormData(event.target);
  form.set("stabilise", $("stabilise").checked ? "1" : "0");
  form.set("enhance", $("enhance").checked ? "1" : "0");
  $("start").disabled = true;
  $("status").textContent = "Loading the model… (the first run downloads the weights)";
  const res = await fetch("/api/jobs", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) { $("status").textContent = data.error; $("start").disabled = false; return; }
  jobId = data.job_id;
  lastAlertId = 0;
  resetCharts(Number(form.get("lanes")));
  $("stream").src = `/api/jobs/${jobId}/stream`;
  $("placeholder").classList.add("hidden");
  $("stop").disabled = false;
  clearInterval(pollTimer);
  pollTimer = setInterval(poll, 500);
});

$("stop").addEventListener("click", async () => {
  if (jobId) await fetch(`/api/jobs/${jobId}/stop`, { method: "POST" });
});

loadModels();
