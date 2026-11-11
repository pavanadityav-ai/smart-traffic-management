"""
SQLite storage for runs, per-lane traffic data and emergency events.

  runs           one row per analysed video or camera session
  traffic_log    one row per lane per second: count, density, speed, clearance time, signal
  emergency_log  one row per emergency alert, with the SMS status

SQLite needs no server and ships with Python. The SQL is standard, so moving
to MySQL means changing the connection code only.
"""

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    source TEXT NOT NULL,
    model TEXT NOT NULL,
    lanes INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    frames INTEGER DEFAULT 0,
    avg_fps REAL,
    avg_inference_ms REAL,
    unique_counts TEXT
);
CREATE TABLE IF NOT EXISTS traffic_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    video_time REAL NOT NULL,
    lane INTEGER NOT NULL,
    count INTEGER NOT NULL,
    smoothed REAL NOT NULL,
    density TEXT NOT NULL,
    load REAL NOT NULL,
    speed_mps REAL NOT NULL,
    clearance_s REAL NOT NULL,
    phase TEXT NOT NULL,
    green_s INTEGER NOT NULL,
    flags TEXT
);
CREATE TABLE IF NOT EXISTS emergency_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    video_time REAL NOT NULL,
    types TEXT NOT NULL,
    sms TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traffic_run ON traffic_log(run_id, video_time);
"""


class Storage:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def _write(self, sql: str, params=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def _read(self, sql: str, params=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ---- writes ----------------------------------------------------------------
    def start_run(self, source: str, model: str, lanes: int) -> int:
        return self._write("INSERT INTO runs (started_at, source, model, lanes) VALUES (?, ?, ?, ?)",
                           (datetime.now().isoformat(timespec="seconds"), source, model, lanes))

    def log_traffic(self, run_id: int, video_time: float, analysis: dict, signal: dict, flags: dict) -> None:
        active = json.dumps([k for k, v in flags.items() if v is True]) if flags else "[]"
        rows = [(run_id, round(video_time, 2), lane["lane"], lane["count"], lane["smoothed"], lane["density"],
                 lane["load"], lane["speed_mps"], lane["clearance_s"], signal["phase"], signal["green_s"], active)
                for lane in analysis["lanes"]]
        with self.lock:
            self.conn.executemany(
                "INSERT INTO traffic_log (run_id, video_time, lane, count, smoothed, density, load, speed_mps, "
                "clearance_s, phase, green_s, flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self.conn.commit()

    def log_emergency(self, run_id: int, alert: dict) -> int:
        return self._write("INSERT INTO emergency_log (run_id, video_time, types, sms, created_at) VALUES (?, ?, ?, ?, ?)",
                           (run_id, alert["time_s"], ", ".join(alert["types"]), alert["sms"],
                            datetime.now().isoformat(timespec="seconds")))

    def update_sms(self, row_id: int, status: str) -> None:
        self._write("UPDATE emergency_log SET sms = ? WHERE id = ?", (status, row_id))

    def finish_run(self, run_id: int, status: str, frames: int, avg_fps: float, avg_ms: float, unique: dict) -> None:
        self._write("UPDATE runs SET status = ?, frames = ?, avg_fps = ?, avg_inference_ms = ?, unique_counts = ? "
                    "WHERE id = ?", (status, frames, round(avg_fps, 2), round(avg_ms, 1), json.dumps(unique), run_id))

    # ---- reads -----------------------------------------------------------------
    def runs(self) -> list[dict]:
        rows = self._read("SELECT * FROM runs ORDER BY id DESC")
        for r in rows:
            r["unique_counts"] = json.loads(r["unique_counts"]) if r["unique_counts"] else {}
        return rows

    def history(self, run_id: int) -> list[dict]:
        return self._read("SELECT * FROM traffic_log WHERE run_id = ? ORDER BY video_time, lane", (run_id,))

    def emergencies(self, run_id: int) -> list[dict]:
        return self._read("SELECT * FROM emergency_log WHERE run_id = ? ORDER BY id", (run_id,))

    def summary(self, run_id: int) -> dict:
        lanes = self._read("SELECT lane, ROUND(AVG(count), 1) AS avg_count, MAX(count) AS max_count, "
                           "ROUND(AVG(clearance_s), 1) AS avg_clearance_s FROM traffic_log WHERE run_id = ? "
                           "GROUP BY lane ORDER BY lane", (run_id,))
        density = self._read("SELECT density, COUNT(*) AS seconds FROM traffic_log WHERE run_id = ? "
                             "GROUP BY density", (run_id,))
        return {"lanes": lanes, "density_seconds": {d["density"]: d["seconds"] for d in density},
                "emergencies": len(self.emergencies(run_id))}

    def export_csv(self, run_id: int) -> str:
        rows = self.history(run_id)
        out = io.StringIO()
        if rows:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return out.getvalue()
