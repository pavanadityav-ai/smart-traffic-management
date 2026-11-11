// History page: lists stored runs and shows the per-lane data of the selected run.

const $ = (id) => document.getElementById(id);
const COLOURS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#39c5cf"];
if (window.Chart) { Chart.defaults.color = "#c9d1d9"; Chart.defaults.borderColor = "#30363d"; }
let charts = [];

function cell(row, text) { const td = document.createElement("td"); td.textContent = text; row.appendChild(td); }

function card(label, value) {
  const div = document.createElement("div");
  div.className = "stat";
  const span = document.createElement("span"); span.textContent = label;
  const strong = document.createElement("strong"); strong.textContent = value;
  div.append(span, strong);
  return div;
}

function lineChart(canvas, rows, field) {
  if (!window.Chart) return null;
  const lanes = [...new Set(rows.map((r) => r.lane))];
  const times = [...new Set(rows.map((r) => Math.round(r.video_time)))];
  const datasets = lanes.map((lane, i) => ({
    label: `Lane ${lane}`, borderColor: COLOURS[i % COLOURS.length], pointRadius: 0,
    data: times.map((t) => { const r = rows.find((x) => x.lane === lane && Math.round(x.video_time) === t); return r ? r[field] : null; }),
  }));
  return new Chart(canvas, { type: "line", data: { labels: times, datasets },
    options: { animation: false, maintainAspectRatio: false, scales: { y: { beginAtZero: true }, x: { title: { display: true, text: "time (s)" } } } } });
}

async function showRun(run, row) {
  document.querySelectorAll("#runs tr").forEach((r) => r.classList.remove("selected"));
  row.classList.add("selected");
  const [detail, history] = await Promise.all([
    fetch(`/api/runs/${run.id}`).then((r) => r.json()),
    fetch(`/api/runs/${run.id}/history`).then((r) => r.json()),
  ]);
  $("detail").classList.remove("hidden");
  $("detail-title").textContent = `Run #${run.id}: ${run.source}`;

  const cards = $("cards");
  cards.replaceChildren();
  detail.summary.lanes.forEach((l) => cards.appendChild(card(`Lane ${l.lane} average / max`, `${l.avg_count} / ${l.max_count} vehicles`)));
  const seconds = detail.summary.density_seconds;
  const lanes = run.lanes || 1;
  ["Low", "Medium", "High", "Very High"].forEach((d) => cards.appendChild(card(`${d} density`, `${Math.round((seconds[d] || 0) / lanes)} s`)));
  cards.appendChild(card("Emergency alerts", detail.summary.emergencies));
  const unique = Object.entries(run.unique_counts || {}).map(([k, v]) => `${k} ${v}`).join(", ");
  cards.appendChild(card("Unique vehicles", unique || "—"));

  charts.forEach((c) => c && c.destroy());
  charts = [lineChart($("count-chart"), history, "count"), lineChart($("clearance-chart"), history, "clearance_s")];

  const body = $("emergencies");
  body.replaceChildren();
  if (!detail.emergencies.length) { const r = document.createElement("tr"); cell(r, "No emergencies"); body.appendChild(r); }
  detail.emergencies.forEach((e) => { const r = document.createElement("tr");
    cell(r, `${e.video_time} s`); cell(r, e.types); cell(r, e.sms); cell(r, e.created_at); body.appendChild(r); });
  $("csv").href = `/api/runs/${run.id}/export.csv`;
}

async function load() {
  const runs = await fetch("/api/runs").then((r) => r.json());
  const body = $("runs");
  body.replaceChildren();
  if (!runs.length) { const r = document.createElement("tr"); cell(r, "No runs yet. Analyse a video on the Live page."); body.appendChild(r); return; }
  runs.forEach((run) => {
    const row = document.createElement("tr");
    row.className = "clickable";
    [run.id, run.started_at.replace("T", " "), run.source, run.model, run.lanes, run.frames, run.avg_fps ?? "—", run.status]
      .forEach((v) => cell(row, v));
    row.addEventListener("click", () => showRun(run, row));
    body.appendChild(row);
  });
}

load();
