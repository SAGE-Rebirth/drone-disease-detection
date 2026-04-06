"""
Disease Drone Dashboard — FastAPI Backend

Endpoints:
    GET  /                              → Dashboard UI
    GET  /api/stats                     → Dashboard summary stats

    POST /api/missions                  → Create a mission (basic)
    GET  /api/missions                  → List missions
    GET  /api/missions/summary          → Mission history with aggregated counts
    GET  /api/missions/{id}             → Get mission detail (basic)
    GET  /api/missions/{id}/full        → Get mission with detections, zones, treatments
    PUT  /api/missions/{id}             → Update mission status

    POST /api/plan/scan                 → Plan scan waypoints from polygon
    POST /api/plan/spray                → Plan spray waypoints from approved zones
    POST /api/missions/{id}/simulate    → Simulate mission execution (no drone needed)

    GET  /api/detections                → List detections
    POST /api/detections                → Add detection(s)
    GET  /api/spray-zones               → List spray zones
    POST /api/spray-zones               → Add spray zone
    PUT  /api/spray-zones/{id}          → Approve/reject zone
    GET  /api/treatments                → List treatments
    POST /api/treatments                → Log treatment
    GET  /api/health                    → Health heatmap data
    POST /api/health                    → Add health point

    GET  /api/telemetry                 → Snapshot of current drone telemetry
    WS   /ws/telemetry                  → Live telemetry stream

    POST /api/demo/seed                 → Seed demo data
    POST /api/demo/clear                → Clear all data
    POST /api/demo/full-flow            → Run full end-to-end demo (scan→detect→spray)

Usage:
    uvicorn dashboard.app:app --reload --port 8000
"""

import asyncio
import json
import math
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Make engine importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from . import database as db
from engine.decision import (
    Detection,
    process_detections,
    DISEASE_SEVERITY_WEIGHTS,
)
from engine.planner import (
    generate_scan_waypoints,
    optimize_spray_path,
    to_mavlink_mission,
    mission_to_qgc_plan,
)


# ── Telemetry Simulator (no drone needed for demos) ──

class TelemetrySimulator:
    """Simulates drone flying along a waypoint path.

    Used when no real drone is connected — allows the dashboard's live
    tracking features to work for demos and development.
    """

    def __init__(self):
        self.active = False
        self.mission_id = None
        self.drone_type = "scout"        # 'scout' or 'treatment'
        self.waypoints = []
        self.current_idx = 0
        self.position = None              # (lat, lon)
        self.alt = 0.0
        self.heading = 0.0
        self.battery = 100
        self.speed = 0.0
        self.mode = "STANDBY"
        self.armed = False
        self.progress = 0.0               # 0-1
        self._segment_progress = 0.0      # 0-1 within current segment

    def start(self, mission_id: int, waypoints: list, drone_type: str = "scout"):
        """Begin simulating a mission."""
        self.active = True
        self.mission_id = mission_id
        self.drone_type = drone_type
        self.waypoints = [
            (wp["x"], wp["y"], wp["z"]) for wp in waypoints
            if wp.get("command") in (16, 22)  # waypoint or takeoff
        ]
        self.current_idx = 0
        self._segment_progress = 0.0
        self.battery = 100
        self.speed = 2.0
        self.mode = "AUTO"
        self.armed = True
        self.progress = 0.0
        if self.waypoints:
            self.position = (self.waypoints[0][0], self.waypoints[0][1])
            self.alt = self.waypoints[0][2]

    def stop(self):
        self.active = False
        self.mode = "STANDBY"
        self.armed = False
        self.speed = 0.0

    def step(self, dt: float = 0.5):
        """Advance simulation by `dt` seconds."""
        if not self.active or not self.waypoints:
            return
        if self.current_idx >= len(self.waypoints) - 1:
            self.stop()
            self.progress = 1.0
            return

        a = self.waypoints[self.current_idx]
        b = self.waypoints[self.current_idx + 1]

        # Approximate metres between waypoints
        mlat = 111_320
        mlon = 111_320 * math.cos(math.radians(a[0]))
        seg_dist = math.sqrt(
            ((b[0] - a[0]) * mlat) ** 2 + ((b[1] - a[1]) * mlon) ** 2
        )
        seg_dist = max(seg_dist, 0.1)

        step_dist = self.speed * dt
        self._segment_progress += step_dist / seg_dist

        if self._segment_progress >= 1.0:
            self.current_idx += 1
            self._segment_progress = 0.0
            self.position = (b[0], b[1])
            self.alt = b[2]
        else:
            t = self._segment_progress
            self.position = (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
            )
            self.alt = a[2] + (b[2] - a[2]) * t

        # Heading
        dx = (b[1] - a[1]) * mlon
        dy = (b[0] - a[0]) * mlat
        self.heading = (math.degrees(math.atan2(dx, dy)) + 360) % 360

        # Battery drain
        self.battery = max(15, self.battery - 0.05)

        # Overall progress
        if len(self.waypoints) > 1:
            self.progress = (self.current_idx + self._segment_progress) / (len(self.waypoints) - 1)

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "mission_id": self.mission_id,
            "drone_type": self.drone_type,
            "lat": round(self.position[0], 7) if self.position else None,
            "lon": round(self.position[1], 7) if self.position else None,
            "alt": round(self.alt, 2),
            "heading": round(self.heading, 1),
            "battery": round(self.battery, 1),
            "groundspeed": round(self.speed, 2),
            "mode": self.mode,
            "armed": self.armed,
            "progress": round(self.progress, 4),
            "waypoint_index": self.current_idx,
            "waypoint_count": len(self.waypoints),
        }


telemetry_sim = TelemetrySimulator()


# ── WebSocket Connection Manager ──

class WSConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSConnectionManager()


# ── Background Telemetry Loop ──

async def telemetry_loop():
    """Tick the telemetry simulator and broadcast updates over WebSocket."""
    while True:
        try:
            telemetry_sim.step(dt=0.5)
            if telemetry_sim.active:
                await ws_manager.broadcast({
                    "type": "telemetry",
                    "data": telemetry_sim.snapshot(),
                })
                if not telemetry_sim.active:
                    db.update_mission_status(telemetry_sim.mission_id, "completed")
                    await ws_manager.broadcast({
                        "type": "mission_complete",
                        "mission_id": telemetry_sim.mission_id,
                    })
        except Exception as e:
            print(f"telemetry_loop error: {e}")
        await asyncio.sleep(0.5)


# ── App Setup ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(telemetry_loop())
    yield
    task.cancel()


app = FastAPI(title="Disease Drone Dashboard", version="0.2.0", lifespan=lifespan)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Pages ──

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ── API: Stats ──

@app.get("/api/stats")
async def get_stats():
    return db.get_dashboard_stats()


# ── API: Missions ──

class MissionCreate(BaseModel):
    type: str = "scan"
    scan_area: list | None = None
    notes: str | None = None

class MissionUpdate(BaseModel):
    status: str

@app.post("/api/missions")
async def create_mission(body: MissionCreate):
    mid = db.create_mission(body.type, body.scan_area, body.notes)
    return {"id": mid, "status": "created"}

@app.get("/api/missions")
async def list_missions():
    return db.get_missions()

@app.get("/api/missions/summary")
async def mission_history(
    limit: int = 100,
    mission_type: str | None = None,
    status: str | None = None,
):
    """Mission history list with aggregated counts."""
    return db.get_missions_summary(limit=limit, mission_type=mission_type, status=status)

@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: int):
    m = db.get_mission(mission_id)
    if not m:
        raise HTTPException(404, "Mission not found")
    return m

@app.get("/api/missions/{mission_id}/full")
async def get_mission_full(mission_id: int):
    """Full mission detail: waypoints, detections, spray zones, treatments."""
    m = db.get_mission_full(mission_id)
    if not m:
        raise HTTPException(404, "Mission not found")
    return m

@app.put("/api/missions/{mission_id}")
async def update_mission(mission_id: int, body: MissionUpdate):
    db.update_mission_status(mission_id, body.status)
    return {"id": mission_id, "status": body.status}


# ── API: Mission Planning ──

class ScanPlanRequest(BaseModel):
    polygon: list                    # [[lat, lon], ...]
    altitude: float = 4.0
    overlap: float = 0.3
    camera_hfov_deg: float = 62.2
    camera_vfov_deg: float = 48.8
    flight_speed: float = 2.0
    save: bool = False               # if True, persist as a mission
    notes: str | None = None


@app.post("/api/plan/scan")
async def plan_scan(body: ScanPlanRequest):
    """Generate lawnmower waypoints for a scan polygon.

    Returns waypoints + estimated mission stats. If `save=True`, also
    creates a mission record in the database.
    """
    if not body.polygon or len(body.polygon) < 3:
        raise HTTPException(400, "Polygon must have at least 3 vertices")

    waypoints = generate_scan_waypoints(
        scan_polygon=body.polygon,
        altitude=body.altitude,
        overlap=body.overlap,
        camera_hfov_deg=body.camera_hfov_deg,
        camera_vfov_deg=body.camera_vfov_deg,
        flight_speed=body.flight_speed,
    )
    mavlink_items = to_mavlink_mission(waypoints)

    # Estimate mission stats
    total_dist_m = 0.0
    nav_wps = [wp for wp in waypoints if wp.command == 16]  # NAV_WAYPOINT
    for i in range(len(nav_wps) - 1):
        a, b = nav_wps[i], nav_wps[i + 1]
        mlat = 111_320
        mlon = 111_320 * math.cos(math.radians(a.lat))
        total_dist_m += math.sqrt(
            ((b.lat - a.lat) * mlat) ** 2 + ((b.lon - a.lon) * mlon) ** 2
        )

    duration_s = total_dist_m / max(body.flight_speed, 0.1)

    # Camera footprint area
    ground_w = 2 * body.altitude * math.tan(math.radians(body.camera_hfov_deg / 2))
    ground_h = 2 * body.altitude * math.tan(math.radians(body.camera_vfov_deg / 2))

    # Estimated images (each waypoint pair captures along the row)
    row_count = max(1, len(nav_wps) // 2)
    row_length = total_dist_m / max(row_count, 1)
    imgs_per_row = max(1, int(row_length / max(ground_w * (1 - body.overlap), 0.1)))
    estimated_images = row_count * imgs_per_row

    response = {
        "waypoints": mavlink_items,
        "waypoint_count": len(mavlink_items),
        "stats": {
            "total_distance_m": round(total_dist_m, 1),
            "estimated_duration_s": round(duration_s, 1),
            "estimated_duration_str": _fmt_duration(duration_s),
            "estimated_images": estimated_images,
            "row_count": row_count,
            "ground_footprint_m": [round(ground_w, 2), round(ground_h, 2)],
            "altitude": body.altitude,
            "overlap": body.overlap,
        },
    }

    if body.save:
        mission_id = db.create_mission(
            mission_type="scan",
            scan_area=body.polygon,
            notes=body.notes or f"Scan @ {body.altitude}m, {int(body.overlap*100)}% overlap",
        )
        db.update_mission_waypoints(mission_id, mavlink_items)
        response["mission_id"] = mission_id

    return response


class SprayPlanRequest(BaseModel):
    zone_ids: list[int] | None = None
    scan_mission_id: int | None = None
    home_lat: float | None = None
    home_lon: float | None = None
    altitude: float = 3.0
    hover_time: float = 5.0
    save: bool = False


@app.post("/api/plan/spray")
async def plan_spray(body: SprayPlanRequest):
    """Generate optimised spray path through approved zones."""
    if body.zone_ids:
        all_zones = db.get_spray_zones()
        zones = [z for z in all_zones if z["id"] in body.zone_ids]
    elif body.scan_mission_id:
        zones = db.get_spray_zones(mission_id=body.scan_mission_id, status="approved")
    else:
        zones = db.get_spray_zones(status="approved")

    if not zones:
        raise HTTPException(400, "No approved zones to spray")

    home = None
    if body.home_lat is not None and body.home_lon is not None:
        home = (body.home_lat, body.home_lon)

    waypoints = optimize_spray_path(
        spray_zones=zones,
        home=home,
        altitude=body.altitude,
        hover_time=body.hover_time,
    )
    mavlink_items = to_mavlink_mission(waypoints)

    # Estimate distance + duration
    total_dist_m = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        mlat = 111_320
        mlon = 111_320 * math.cos(math.radians(a.lat))
        total_dist_m += math.sqrt(
            ((b.lat - a.lat) * mlat) ** 2 + ((b.lon - a.lon) * mlon) ** 2
        )
    flight_time = total_dist_m / 3.0  # assume 3 m/s
    hover_total = len(zones) * body.hover_time
    duration_s = flight_time + hover_total

    response = {
        "waypoints": mavlink_items,
        "waypoint_count": len(mavlink_items),
        "stats": {
            "total_distance_m": round(total_dist_m, 1),
            "zone_count": len(zones),
            "estimated_duration_s": round(duration_s, 1),
            "estimated_duration_str": _fmt_duration(duration_s),
            "altitude": body.altitude,
        },
    }

    if body.save:
        mission_id = db.create_mission(
            mission_type="spray",
            notes=f"Spray mission: {len(zones)} zones",
        )
        db.update_mission_waypoints(mission_id, mavlink_items)
        response["mission_id"] = mission_id

    return response


# ── API: Mission Execution (simulated) ──

@app.post("/api/missions/{mission_id}/simulate")
async def simulate_mission(mission_id: int):
    """Simulate a mission's flight using the telemetry simulator."""
    mission = db.get_mission(mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")
    if not mission.get("waypoints"):
        raise HTTPException(400, "Mission has no waypoints — plan it first")

    waypoints = json.loads(mission["waypoints"])
    drone_type = "scout" if mission["type"] == "scan" else "treatment"

    telemetry_sim.start(mission_id, waypoints, drone_type=drone_type)
    db.update_mission_status(mission_id, "in_progress")

    return {
        "status": "simulating",
        "mission_id": mission_id,
        "waypoint_count": len(waypoints),
    }


@app.post("/api/simulator/stop")
async def stop_simulator():
    telemetry_sim.stop()
    return {"status": "stopped"}


# ── API: Detections ──

class DetectionCreate(BaseModel):
    mission_id: int
    class_name: str
    confidence: float
    lat: float
    lon: float
    bbox: list | None = None
    image_path: str | None = None


@app.get("/api/detections")
async def list_detections(mission_id: int | None = None):
    return db.get_detections(mission_id)


@app.post("/api/detections")
async def add_detection(body: DetectionCreate):
    did = db.add_detection(
        body.mission_id, body.class_name, body.confidence,
        body.lat, body.lon, body.bbox, body.image_path,
    )
    return {"id": did}


# ── API: Spray Zones ──

class SprayZoneCreate(BaseModel):
    mission_id: int
    geometry: list
    center_lat: float
    center_lon: float
    severity: float = 0.0
    disease_type: str | None = None


class SprayZoneUpdate(BaseModel):
    status: str


@app.get("/api/spray-zones")
async def list_spray_zones(mission_id: int | None = None, status: str | None = None):
    return db.get_spray_zones(mission_id, status)


@app.post("/api/spray-zones")
async def add_spray_zone(body: SprayZoneCreate):
    zid = db.add_spray_zone(
        body.mission_id, body.geometry, body.center_lat, body.center_lon,
        body.severity, body.disease_type,
    )
    return {"id": zid}


@app.put("/api/spray-zones/{zone_id}")
async def update_spray_zone(zone_id: int, body: SprayZoneUpdate):
    db.update_spray_zone_status(zone_id, body.status)
    return {"id": zone_id, "status": body.status}


# ── API: Treatments ──

class TreatmentCreate(BaseModel):
    spray_zone_id: int
    mission_id: int
    lat: float
    lon: float
    spray_duration: float | None = None
    chemical: str | None = None
    notes: str | None = None


@app.get("/api/treatments")
async def list_treatments(mission_id: int | None = None):
    return db.get_treatments(mission_id)


@app.post("/api/treatments")
async def add_treatment(body: TreatmentCreate):
    tid = db.add_treatment(
        body.spray_zone_id, body.mission_id,
        body.lat, body.lon, body.spray_duration, body.chemical, body.notes,
    )
    return {"id": tid}


# ── API: Health ──

class HealthPoint(BaseModel):
    lat: float
    lon: float
    health_score: float
    disease_type: str | None = None


@app.get("/api/health")
async def get_health():
    return db.get_health_data()


@app.post("/api/health")
async def add_health(body: HealthPoint):
    db.add_health_point(body.lat, body.lon, body.health_score, body.disease_type)
    return {"status": "ok"}


# ── API: Telemetry ──

@app.get("/api/telemetry")
async def get_telemetry():
    return telemetry_sim.snapshot()


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """Live telemetry & event stream."""
    await ws_manager.connect(websocket)
    # Send current snapshot on connect
    await websocket.send_text(json.dumps({
        "type": "telemetry",
        "data": telemetry_sim.snapshot(),
    }))
    try:
        while True:
            # We don't expect inbound messages, but keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ── Demo Data ──

@app.post("/api/demo/clear")
async def clear_demo():
    """Wipe all data from the database."""
    conn = db.get_db()
    for table in ["treatments", "spray_zones", "detections", "field_health", "missions"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    telemetry_sim.stop()
    return {"status": "cleared"}


@app.post("/api/demo/seed")
async def seed_demo():
    """Populate the database with realistic demo data for UI development."""
    CENTER_LAT, CENTER_LON = 12.9716, 77.5946
    SPREAD = 0.008

    diseases = ["leaf_blight", "leaf_spot", "rust", "powdery_mildew", "healthy"]
    chemicals = ["Copper fungicide", "Neem oil", "Mancozeb", "Sulfur dust"]

    scan_area = [
        [CENTER_LAT - SPREAD, CENTER_LON - SPREAD],
        [CENTER_LAT - SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON - SPREAD],
    ]
    scan_id = db.create_mission("scan", scan_area, "Demo scan mission")
    db.update_mission_status(scan_id, "completed")

    # Plan and store waypoints for the demo scan
    wps = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)
    db.update_mission_waypoints(scan_id, to_mavlink_mission(wps))

    # Detections across the field
    for _ in range(80):
        lat = CENTER_LAT + random.uniform(-SPREAD, SPREAD)
        lon = CENTER_LON + random.uniform(-SPREAD, SPREAD)
        disease = random.choices(diseases, weights=[25, 30, 15, 10, 20])[0]
        conf = random.uniform(0.55, 0.98) if disease != "healthy" else random.uniform(0.80, 0.99)
        db.add_detection(scan_id, disease, round(conf, 3), lat, lon)

    # Spray zones
    zone_centers = [
        (CENTER_LAT + 0.003, CENTER_LON - 0.002, "leaf_blight", 0.82),
        (CENTER_LAT - 0.004, CENTER_LON + 0.003, "leaf_spot", 0.65),
        (CENTER_LAT + 0.001, CENTER_LON + 0.005, "rust", 0.74),
        (CENTER_LAT - 0.002, CENTER_LON - 0.004, "powdery_mildew", 0.58),
    ]
    zone_ids = []
    for lat, lon, disease, severity in zone_centers:
        r = 0.001
        geometry = [
            [lat - r, lon - r], [lat - r, lon + r],
            [lat + r, lon + r], [lat + r, lon - r],
        ]
        zid = db.add_spray_zone(scan_id, geometry, lat, lon, severity, disease)
        zone_ids.append(zid)

    db.update_spray_zone_status(zone_ids[0], "approved")
    db.update_spray_zone_status(zone_ids[1], "approved")
    db.update_spray_zone_status(zone_ids[2], "treated")

    spray_id = db.create_mission("spray", notes="Demo spray mission")
    db.update_mission_status(spray_id, "completed")
    spray_wps = optimize_spray_path(
        [{"center_lat": z[0], "center_lon": z[1]} for z in zone_centers[:3]],
        home=(CENTER_LAT, CENTER_LON),
    )
    db.update_mission_waypoints(spray_id, to_mavlink_mission(spray_wps))

    for zid, (lat, lon, _, _) in zip(zone_ids[:3], zone_centers[:3]):
        db.add_treatment(
            zid, spray_id, lat, lon,
            spray_duration=random.uniform(3, 12),
            chemical=random.choice(chemicals),
        )
        db.update_spray_zone_status(zid, "treated")

    # Health grid
    steps = 15
    for i in range(steps):
        for j in range(steps):
            lat = CENTER_LAT - SPREAD + (2 * SPREAD * i / steps)
            lon = CENTER_LON - SPREAD + (2 * SPREAD * j / steps)
            base_health = 0.85
            for zlat, zlon, _, sev in zone_centers:
                dist = math.sqrt((lat - zlat) ** 2 + (lon - zlon) ** 2)
                if dist < 0.004:
                    base_health -= sev * (1 - dist / 0.004) * 0.4
            health = max(0.1, min(1.0, base_health + random.uniform(-0.08, 0.08)))
            db.add_health_point(
                lat, lon, round(health, 3),
                random.choice(diseases[:4]) if health < 0.6 else None,
            )

    return {"status": "seeded", "scan_mission": scan_id, "spray_mission": spray_id}


@app.post("/api/demo/full-flow")
async def demo_full_flow():
    """Create a planned scan mission ready to be simulated.

    Unlike `/api/demo/seed` (which creates a fully completed mission with
    detections), this creates a fresh scan mission with waypoints so the
    user can hit "Simulate" and watch the drone fly.
    """
    CENTER_LAT, CENTER_LON = 12.9716, 77.5946
    SPREAD = 0.005

    scan_area = [
        [CENTER_LAT - SPREAD, CENTER_LON - SPREAD],
        [CENTER_LAT - SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON - SPREAD],
    ]

    waypoints = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)
    mavlink_items = to_mavlink_mission(waypoints)

    mission_id = db.create_mission(
        mission_type="scan",
        scan_area=scan_area,
        notes="Demo scan — ready to simulate",
    )
    db.update_mission_waypoints(mission_id, mavlink_items)

    return {
        "mission_id": mission_id,
        "waypoint_count": len(mavlink_items),
        "status": "ready",
    }


# ── Helpers ──

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"
