# Disease Drone — Resume Point

**Last session:** 2026-04-03

---

## Project Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| Architecture | Done | `architecture.md` — full system design |
| ML Pipeline | Scripts done, training pending | All code written, data preprocessed, needs GPU to train. Auto device detection added (MPS/CUDA/CPU). |
| Dashboard | Done (v1) | FastAPI + Leaflet.js, dark theme, working with demo data |
| Decision Engine | Done | `engine/decision.py` — pixel→GPS, DBSCAN clustering, convex hull spray zones, severity scoring |
| Mission Planner | Done | `engine/planner.py` — lawnmower waypoints, nearest-neighbour TSP, MAVLink + QGC export |
| Drone Comms | Done | `drone/comms.py` — pymavlink DroneLink class, mission upload, telemetry, camera/spray control |
| Image Ingestion | Done | `engine/ingest.py` — EXIF GPS extraction, ML inference hookup, folder watcher |
| Base Station Server | Done | `engine/base_station.py` — full orchestrator tying all components together |

---

## What's Been Built

### ML Pipeline (`ml/`)

```
ml/
├── SKILL.md                        ← ML specification doc
├── configs/
│   ├── dataset.yaml                ← 5 classes, absolute paths to processed data
│   └── train.yaml                  ← YOLOv8-nano hyperparameters
├── data/
│   ├── raw/plantvillage/           ← Downloaded (54K images, color/ folder used)
│   ├── raw/plantdoc/              ← Downloaded (2.5K images, classification mode)
│   ├── processed/                  ← YOLO format: train(39810)/val(11372)/test(5693)
│   └── scripts/
│       ├── download_data.py        ← Kaggle + GitHub downloader
│       └── preprocess.py           ← Filters to 5 classes, converts to YOLO, splits
├── training/
│   ├── train.py                    ← 2-phase training (frozen backbone → full)
│   └── evaluate.py                 ← Test eval with PoC target checks
├── inference/
│   └── detect.py                   ← detect_diseases(), detect_batch(), export_onnx()
└── models/                         ← Empty — no trained model yet
```

**Classes (5):** healthy, leaf_blight, leaf_spot, rust, powdery_mildew
- `fruit_rot` was dropped (0 samples in either dataset)
- `leaf_spot` is overrepresented (49%) — consider class weights during training

**Class distribution:**
| Class | Count |
|-------|-------|
| healthy | 15,931 |
| leaf_blight | 7,579 |
| leaf_spot | 27,863 |
| rust | 2,394 |
| powdery_mildew | 3,108 |

### Dashboard (`dashboard/`)

```
dashboard/
├── __init__.py
├── app.py                  ← FastAPI backend, REST API, demo data seeder + clear
├── database.py             ← SQLite: missions, detections, spray_zones, treatments, field_health
├── disease_drone.db        ← SQLite database (created on startup)
├── static/
│   ├── css/style.css       ← Dark theme, glow effects, animations
│   └── js/app.js           ← Leaflet map, draw tools, heatmap, live data
└── templates/
    └── index.html          ← Main dashboard page
```

**Run with:** `uvicorn dashboard.app:app --reload --port 8000`

**Features working:**
- Dark themed map (CARTO dark tiles) with Leaflet.js
- Draw scan area (polygon/rectangle) on map
- Detection markers color-coded by disease type
- Spray zone polygons with approve/reject actions
- Health heatmap toggle (green → red gradient)
- Stat cards with animated counters
- Tabbed right panel (Detections / Spray Zones / Treatment Log)
- Click-to-fly: click any card to pan map
- Toast notifications
- Load Demo Data / Clear All Data buttons
- Leaflet Draw controls styled to match dark theme

**Known issues fixed:**
- `TemplateResponse` API — uses `request=request, name=` kwargs (newer Starlette)
- `dataset.yaml` — changed from hardcoded absolute macOS path to relative path; resolved at runtime via `resolve_dataset_path()` in train.py and evaluate.py for cross-platform support (macOS/Windows/Linux)
- `dataset.yaml` comment said "6 classes" — corrected to 5 classes
- `train.yaml` device — changed from `cpu` to `auto`; auto-detects MPS (Apple Silicon), CUDA (NVIDIA), or CPU fallback
- `train.py` — added `detect_device()` using `platform.system()` + `torch.backends.mps` / `torch.cuda`
- Draw toolbar overlap — center button moved to top-right, hides in draw mode
- Leaflet Draw dark-themed via CSS overrides

### Engine & Drone Modules (built 2026-04-03)

```
engine/
├── __init__.py
├── decision.py             ← DBSCAN clustering, pixel→GPS, convex hull spray zones, severity scoring
├── planner.py              ← Lawnmower scan waypoints, nearest-neighbour TSP spray path, MAVLink + QGC export
├── ingest.py               ← EXIF GPS extraction, ML inference hookup, folder watcher
└── base_station.py         ← Full orchestrator: plan→fly→ingest→detect→cluster→spray

drone/
├── __init__.py
└── comms.py                ← DroneLink class: MAVLink connect, mission upload, telemetry, camera/spray control
```

**Decision Engine (`engine/decision.py`):**
- `pixel_to_gps()` — converts bounding box pixel centre to GPS using altitude + camera FOV trig
- `cluster_detections()` — DBSCAN on GPS coords, eps in metres, filters healthy + low-confidence
- `generate_spray_zones()` — convex hull + configurable buffer per cluster
- `score_severity()` — 0-1 score using disease type weight (0.4), confidence (0.4), count (0.2)
- `process_detections()` — end-to-end pipeline with optional DB persistence
- Smoke tested: 3 close detections + 1 outlier → 1 cluster → 1 spray zone

**Mission Planner (`engine/planner.py`):**
- `generate_scan_waypoints()` — lawnmower pattern over bounding box, configurable altitude/overlap/FOV
- `optimize_spray_path()` — nearest-neighbour TSP starting from home position
- `to_mavlink_mission()` — converts to pymavlink-compatible mission item dicts
- `mission_to_qgc_plan()` — exports QGroundControl-compatible .plan JSON
- Smoke tested: ~400m×330m polygon → 357 waypoints; 3 spray zones → 5 waypoints (takeoff + 3 + RTL)

**Image Ingestion (`engine/ingest.py`):**
- `extract_gps_from_exif()` — reads GPS lat/lon/altitude from EXIF DMS tags via exifread
- `process_image()` — EXIF → inference → pixel_to_gps → DB persist
- `process_folder()` — batch process all images in a directory
- `watch_folder()` — polling watcher for real-time image processing

**Drone Comms (`drone/comms.py`):**
- `DroneLink` class — MAVLink connection via pymavlink (UDP/TCP/serial)
- `connect()` — waits for heartbeat, auto-detects target system/component
- `upload_mission()` — full mission upload protocol with ACK handling
- `get_telemetry()` / `wait_for_telemetry()` — GPS, battery, mode, armed status
- `arm()`, `disarm()`, `set_mode()`, `arm_and_start_mission()`
- `trigger_camera()` — MAV_CMD_DO_DIGICAM_CONTROL
- `set_spray_pump()` — relay-based spray pump toggle

**Base Station (`engine/base_station.py`):**
- `BaseStation` class — central orchestrator
- `plan_scan_mission()` / `start_scan_mission()` — plan + upload to scout
- `process_scout_images()` — ingest → cluster → spray zones → health map update
- `watch_scout_images()` — real-time folder watcher mode
- `plan_spray_mission()` / `start_spray_mission()` — plan + upload to treatment drone
- `export_scan_plan()` / `export_spray_plan()` — QGC .plan file export
- Works without live drones (plans are saved to DB and can be exported)

---

## What Needs To Be Done Next

### 1. Train the ML Model (BLOCKED — needs GPU)

`train.yaml` now has `device: auto` which will auto-detect MPS on Apple Silicon:
```bash
python ml/training/train.py --epochs 5 --batch 16
```
Or override explicitly:
```bash
python ml/training/train.py --device mps --epochs 5 --batch 16
```

Once trained, evaluate:
```bash
python ml/training/evaluate.py
```

### 2. Dashboard Enhancements (later)

- Real-time WebSocket updates (drone telemetry, live detections)
- Wire new engine APIs into dashboard (scan area → mission planner, approval → spray mission)
- Mission progress tracking on map (drone position)
- Image viewer for individual detections
- Historical comparison (health over time)

---

## Environment

- **Python:** 3.11 (venv: `drn-env/`)
- **Key packages installed:** ultralytics, torch, opencv-python, albumentations, scikit-learn, fastapi, uvicorn, jinja2, shapely, pymavlink, exifread
- **All dependencies installed** — no missing packages
- **Run dashboard:** `source drn-env/bin/activate && uvicorn dashboard.app:app --reload --port 8000`

---

## File Tree (project files only)

```
disease-drone/
├── RESUME.md                   ← THIS FILE
├── architecture.md             ← System architecture doc
├── requirements.txt            ← Python dependencies
├── yolov8n.pt                  ← Pretrained COCO weights (downloaded)
├── ml/
│   ├── SKILL.md
│   ├── configs/dataset.yaml
│   ├── configs/train.yaml
│   ├── data/scripts/download_data.py
│   ├── data/scripts/preprocess.py
│   ├── data/raw/               ← PlantVillage + PlantDoc (downloaded)
│   ├── data/processed/         ← YOLO format (39810/11372/5693 split)
│   ├── training/train.py
│   ├── training/evaluate.py
│   ├── inference/detect.py
│   └── models/                 ← Empty (training not yet run)
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── database.py
│   ├── disease_drone.db
│   ├── static/css/style.css
│   ├── static/js/app.js
│   └── templates/index.html
├── engine/
│   ├── __init__.py
│   ├── decision.py             ← DBSCAN clustering, pixel→GPS, spray zones
│   ├── planner.py              ← Lawnmower waypoints, TSP spray path, MAVLink/QGC export
│   ├── ingest.py               ← EXIF GPS extraction, inference hookup, folder watcher
│   └── base_station.py         ← Full orchestrator
└── drone/
    ├── __init__.py
    └── comms.py                ← DroneLink: MAVLink connect, mission upload, telemetry
```
