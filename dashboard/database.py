"""
SQLite database layer for Disease Drone dashboard.

Tables:
    - missions: scan/spray mission records
    - detections: ML detection results with GPS coords
    - spray_zones: clustered spray regions
    - treatments: spray event logs
    - field_health: periodic health snapshots for heatmap
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "disease_drone.db"


def get_db():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS missions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL CHECK(type IN ('scan', 'spray')),
            status TEXT NOT NULL DEFAULT 'planned'
                CHECK(status IN ('planned', 'in_progress', 'completed', 'aborted')),
            scan_area TEXT,
            waypoints TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            started_at TEXT,
            completed_at TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER REFERENCES missions(id),
            image_path TEXT,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT,
            lat REAL,
            lon REAL,
            detected_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS spray_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER REFERENCES missions(id),
            geometry TEXT NOT NULL,
            center_lat REAL NOT NULL,
            center_lon REAL NOT NULL,
            severity REAL DEFAULT 0.0,
            disease_type TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected', 'treated')),
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spray_zone_id INTEGER REFERENCES spray_zones(id),
            mission_id INTEGER REFERENCES missions(id),
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            spray_duration REAL,
            chemical TEXT,
            treated_at TEXT NOT NULL DEFAULT (datetime('now')),
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS field_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            health_score REAL NOT NULL CHECK(health_score BETWEEN 0 AND 1),
            disease_type TEXT,
            recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ── Mission CRUD ──

def create_mission(mission_type, scan_area=None, notes=None):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO missions (type, scan_area, notes) VALUES (?, ?, ?)",
        (mission_type, json.dumps(scan_area) if scan_area else None, notes),
    )
    conn.commit()
    mid = cur.lastrowid
    conn.close()
    return mid


def update_mission_status(mission_id, status):
    conn = get_db()
    ts_field = "started_at" if status == "in_progress" else "completed_at" if status in ("completed", "aborted") else None
    if ts_field:
        conn.execute(
            f"UPDATE missions SET status=?, {ts_field}=datetime('now') WHERE id=?",
            (status, mission_id),
        )
    else:
        conn.execute("UPDATE missions SET status=? WHERE id=?", (status, mission_id))
    conn.commit()
    conn.close()


def get_missions(limit=50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM missions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_mission(mission_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Detection CRUD ──

def add_detection(mission_id, class_name, confidence, lat, lon, bbox=None, image_path=None):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO detections (mission_id, class_name, confidence, lat, lon, bbox, image_path) VALUES (?,?,?,?,?,?,?)",
        (mission_id, class_name, confidence, lat, lon, json.dumps(bbox) if bbox else None, image_path),
    )
    conn.commit()
    did = cur.lastrowid
    conn.close()
    return did


def get_detections(mission_id=None, limit=500):
    conn = get_db()
    if mission_id:
        rows = conn.execute(
            "SELECT * FROM detections WHERE mission_id=? ORDER BY detected_at DESC LIMIT ?",
            (mission_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY detected_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Spray Zone CRUD ──

def add_spray_zone(mission_id, geometry, center_lat, center_lon, severity=0.0, disease_type=None):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO spray_zones (mission_id, geometry, center_lat, center_lon, severity, disease_type) VALUES (?,?,?,?,?,?)",
        (mission_id, json.dumps(geometry), center_lat, center_lon, severity, disease_type),
    )
    conn.commit()
    zid = cur.lastrowid
    conn.close()
    return zid


def update_spray_zone_status(zone_id, status):
    conn = get_db()
    conn.execute("UPDATE spray_zones SET status=? WHERE id=?", (status, zone_id))
    conn.commit()
    conn.close()


def get_spray_zones(mission_id=None, status=None):
    conn = get_db()
    query = "SELECT * FROM spray_zones WHERE 1=1"
    params = []
    if mission_id:
        query += " AND mission_id=?"
        params.append(mission_id)
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Treatment CRUD ──

def add_treatment(spray_zone_id, mission_id, lat, lon, spray_duration=None, chemical=None, notes=None):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO treatments (spray_zone_id, mission_id, lat, lon, spray_duration, chemical, notes) VALUES (?,?,?,?,?,?,?)",
        (spray_zone_id, mission_id, lat, lon, spray_duration, chemical, notes),
    )
    conn.commit()
    tid = cur.lastrowid
    conn.close()
    return tid


def get_treatments(mission_id=None, limit=200):
    conn = get_db()
    if mission_id:
        rows = conn.execute(
            "SELECT t.*, sz.disease_type FROM treatments t LEFT JOIN spray_zones sz ON t.spray_zone_id=sz.id WHERE t.mission_id=? ORDER BY t.treated_at DESC LIMIT ?",
            (mission_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT t.*, sz.disease_type FROM treatments t LEFT JOIN spray_zones sz ON t.spray_zone_id=sz.id ORDER BY t.treated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Field Health ──

def add_health_point(lat, lon, health_score, disease_type=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO field_health (lat, lon, health_score, disease_type) VALUES (?,?,?,?)",
        (lat, lon, health_score, disease_type),
    )
    conn.commit()
    conn.close()


def get_health_data(limit=2000):
    conn = get_db()
    rows = conn.execute(
        "SELECT lat, lon, health_score, disease_type, recorded_at FROM field_health ORDER BY recorded_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Stats ──

def get_dashboard_stats():
    conn = get_db()
    stats = {}
    stats["total_missions"] = conn.execute("SELECT COUNT(*) FROM missions").fetchone()[0]
    stats["active_missions"] = conn.execute("SELECT COUNT(*) FROM missions WHERE status='in_progress'").fetchone()[0]
    stats["total_detections"] = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    stats["total_treatments"] = conn.execute("SELECT COUNT(*) FROM treatments").fetchone()[0]
    stats["pending_zones"] = conn.execute("SELECT COUNT(*) FROM spray_zones WHERE status='pending'").fetchone()[0]

    # Disease distribution
    rows = conn.execute(
        "SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name ORDER BY count DESC"
    ).fetchall()
    stats["disease_distribution"] = {r["class_name"]: r["count"] for r in rows}

    # Average health
    row = conn.execute("SELECT AVG(health_score) as avg FROM field_health").fetchone()
    stats["avg_health"] = round(row["avg"], 2) if row["avg"] else None

    conn.close()
    return stats


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
