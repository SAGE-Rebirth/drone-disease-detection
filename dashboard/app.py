"""
Disease Drone Dashboard — FastAPI Backend

Endpoints:
    GET  /                      → Dashboard UI
    GET  /api/stats             → Dashboard summary stats
    POST /api/missions          → Create a mission
    GET  /api/missions          → List missions
    GET  /api/missions/{id}     → Get mission detail
    PUT  /api/missions/{id}     → Update mission status
    GET  /api/detections        → List detections
    POST /api/detections        → Add detection(s)
    GET  /api/spray-zones       → List spray zones
    POST /api/spray-zones       → Add spray zone
    PUT  /api/spray-zones/{id}  → Approve/reject zone
    GET  /api/treatments        → List treatments
    POST /api/treatments        → Log treatment
    GET  /api/health            → Health heatmap data
    POST /api/health            → Add health point
    POST /api/demo/seed         → Seed demo data

Usage:
    uvicorn dashboard.app:app --reload --port 8000
"""

import random
import math
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

from . import database as db

app = FastAPI(title="Disease Drone Dashboard", version="0.1.0")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Initialize DB on startup
@app.on_event("startup")
def startup():
    db.init_db()


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

@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: int):
    m = db.get_mission(mission_id)
    if not m:
        return {"error": "not found"}, 404
    return m

@app.put("/api/missions/{mission_id}")
async def update_mission(mission_id: int, body: MissionUpdate):
    db.update_mission_status(mission_id, body.status)
    return {"id": mission_id, "status": body.status}


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


# ── Demo Data Seeder ──

@app.post("/api/demo/clear")
async def clear_demo():
    """Wipe all data from the database."""
    conn = db.get_db()
    for table in ["treatments", "spray_zones", "detections", "field_health", "missions"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    return {"status": "cleared"}

@app.post("/api/demo/seed")
async def seed_demo():
    """Populate database with realistic demo data for UI development."""
    # Center point (a farm field — using a generic agricultural area)
    CENTER_LAT, CENTER_LON = 12.9716, 77.5946  # Bangalore area
    SPREAD = 0.008  # ~800m spread

    diseases = ["leaf_blight", "leaf_spot", "rust", "powdery_mildew", "healthy"]
    chemicals = ["Copper fungicide", "Neem oil", "Mancozeb", "Sulfur dust"]

    # Create scan mission
    scan_area = [
        [CENTER_LAT - SPREAD, CENTER_LON - SPREAD],
        [CENTER_LAT - SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON - SPREAD],
    ]
    scan_id = db.create_mission("scan", scan_area, "Demo scan mission")
    db.update_mission_status(scan_id, "completed")

    # Add detections across the field
    for i in range(80):
        lat = CENTER_LAT + random.uniform(-SPREAD, SPREAD)
        lon = CENTER_LON + random.uniform(-SPREAD, SPREAD)
        disease = random.choices(diseases, weights=[25, 30, 15, 10, 20])[0]
        conf = random.uniform(0.55, 0.98) if disease != "healthy" else random.uniform(0.80, 0.99)
        db.add_detection(scan_id, disease, round(conf, 3), lat, lon)

    # Create spray zones (clusters of disease)
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

    # Approve some zones
    db.update_spray_zone_status(zone_ids[0], "approved")
    db.update_spray_zone_status(zone_ids[1], "approved")
    db.update_spray_zone_status(zone_ids[2], "treated")

    # Create spray mission
    spray_id = db.create_mission("spray", notes="Demo spray mission")
    db.update_mission_status(spray_id, "completed")

    # Add treatments
    for zid, (lat, lon, disease, _) in zip(zone_ids[:3], zone_centers[:3]):
        db.add_treatment(
            zid, spray_id, lat, lon,
            spray_duration=random.uniform(3, 12),
            chemical=random.choice(chemicals),
        )
        db.update_spray_zone_status(zid, "treated")

    # Add health grid data
    steps = 15
    for i in range(steps):
        for j in range(steps):
            lat = CENTER_LAT - SPREAD + (2 * SPREAD * i / steps)
            lon = CENTER_LON - SPREAD + (2 * SPREAD * j / steps)
            # Lower health near disease zones
            base_health = 0.85
            for zlat, zlon, _, sev in zone_centers:
                dist = math.sqrt((lat - zlat) ** 2 + (lon - zlon) ** 2)
                if dist < 0.004:
                    base_health -= sev * (1 - dist / 0.004) * 0.4
            health = max(0.1, min(1.0, base_health + random.uniform(-0.08, 0.08)))
            db.add_health_point(lat, lon, round(health, 3),
                              random.choice(diseases[:4]) if health < 0.6 else None)

    return {"status": "seeded", "scan_mission": scan_id, "spray_mission": spray_id}
