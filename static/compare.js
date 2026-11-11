// Model comparison page: reads results/benchmark.json through /api/benchmark.

const $ = (id) => document.getElementById(id);
const COLOURS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"];
if (window.Chart) { Chart.defaults.color = "#c9d1d9"; Chart.defaults.borderColor = "#30363d"; }

function bar(canvas, labels, datasets, yTitle) {
  if (!window.Chart) return;
  new Chart(canvas, { type: "bar", data: { labels, datasets },
    options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: yTitle } } } } });
}

async function load() {
  const res = await fetch("/api/benchmark");
  if (!res.ok) {
    $("dataset").textContent = "";
    $("missing").classList.remove("hidden");
    $("table").classList.add("hidden");
    $("charts").classList.add("hidden");
    return;
  }
  const data = await res.json();
  $("dataset").textContent = `${data.dataset} · ${data.images} images · ${data.device}`;

  const body = $("rows");
  data.models.forEach((m) => {
    const row = document.createElement("tr");
    [m.name, m.map50.toFixed(3), m.map50_95.toFixed(3), m.fps.toFixed(1), m.inference_ms.toFixed(1), m.params_m ?? "—"]
      .forEach((v) => { const td = document.createElement("td"); td.textContent = v; row.appendChild(td); });
    body.appendChild(row);
  });

  const names = data.models.map((m) => m.name);
  bar($("map-chart"), names, [{ label: "mAP@0.5", data: data.models.map((m) => m.map50), backgroundColor: COLOURS[0] }], "mAP");
  bar($("fps-chart"), names, [{ label: "FPS", data: data.models.map((m) => m.fps), backgroundColor: COLOURS[1] }], "FPS");
  const classes = Object.keys(data.models[0].ap_per_class);
  bar($("class-chart"), classes, data.models.map((m, i) => ({
    label: m.name, data: classes.map((c) => m.ap_per_class[c]), backgroundColor: COLOURS[i % COLOURS.length] })), "AP");
  if (data.low_light && data.low_light.length) {
    const ll = data.low_light;
    bar($("light-chart"), ll.map((r) => r.model), [
      { label: "normal", data: ll.map((r) => r.normal), backgroundColor: COLOURS[0] },
      { label: "dark", data: ll.map((r) => r.dark), backgroundColor: COLOURS[3] },
      { label: "dark + enhancement", data: ll.map((r) => r.dark_enhanced), backgroundColor: COLOURS[1] },
    ], "mAP@0.5");
  }
}

load();
