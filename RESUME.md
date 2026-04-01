# Disease Drone — Resume Point

**Last session:** 2026-04-01

---

## Project Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| Architecture | Done | `architecture.md` — full system design |
| ML Pipeline | Scripts done, training pending | All code written, data preprocessed, needs GPU to train |
| Dashboard | Done (v1) | FastAPI + Leaflet.js, dark theme, working with demo data |
| Decision Engine | Not started | DBSCAN clustering, GPS offset calc, spray zone gen |
| Mission Planner | Not started | Lawnmower waypoints, TSP path optimizer, MAVLink upload |
| Drone Comms | Not started | pymavlink/dronekit integration |
| Image Ingestion | Not started | EXIF GPS extraction, feed to inference pipeline |
| Base Station Server | Not started | Orchestrator tying all components together |

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
- `dataset.yaml` — uses absolute path (YOLO resolves relative paths from wrong base)
- Draw toolbar overlap — center button moved to top-right, hides in draw mode
- Leaflet Draw dark-themed via CSS overrides

---

## What Needs To Be Done Next

### 1. Train the ML Model (BLOCKED — needs GPU)

CPU training on M4 is too slow for 40K images. Options:
- **`device=mps`** — Apple Silicon GPU, try this first:
  ```bash
  python ml/training/train.py --device mps --epochs 5 --batch 16
  ```
- **Google Colab** — upload `ml/` folder, use free T4 GPU
- **Subsample** — reduce dataset to 5K images for quick validation

Once trained, evaluate:
```bash
python ml/training/evaluate.py
```

### 2. Decision Engine (new: `engine/decision.py`)

Pure Python, no hardware needed. Build:
- `pixel_to_gps(bbox, image_gps, altitude, camera_fov)` — convert pixel detection to GPS coord
- `cluster_detections(detections, eps=2m, min_samples=2)` — DBSCAN clustering
- `generate_spray_zones(clusters)` — convex hull + 1m buffer per cluster
- `score_severity(cluster)` — severity based on detection count, confidence, disease type
- Wire output into dashboard API (create spray zones from clustered detections)

Dependencies: `scikit-learn` (DBSCAN), `shapely` (geometry), `geopy` (GPS math) — all in requirements.txt except shapely

### 3. Mission Planner (new: `engine/planner.py`)

- `generate_lawnmower_waypoints(scan_area_polygon, altitude, overlap, camera_fov)` — zigzag flight path
- `optimize_spray_path(spray_zones)` — nearest-neighbor or simple TSP
- `to_mavlink_mission(waypoints)` — convert to MAVLink mission items

### 4. Drone Communication (new: `drone/comms.py`)

- MAVLink connection handler (WiFi/serial)
- Upload mission waypoints to drone
- Monitor drone telemetry (GPS, battery, status)
- Trigger camera capture (scout) or spray pump (treatment)

Dependencies: `pymavlink` or `dronekit` — add to requirements.txt

### 5. Image Ingestion Pipeline (new: `engine/ingest.py`)

- Watch folder or receive images via API
- Extract GPS from EXIF metadata (`Pillow` or `exifread`)
- Run ML inference via `ml.inference.detect.detect_diseases()`
- Feed results to decision engine
- Push detections + spray zones to dashboard DB

### 6. Base Station Orchestrator (new: `engine/base_station.py`)

Ties everything together:
1. Dashboard sends scan area → mission planner generates waypoints
2. Upload waypoints to scout drone via comms
3. Scout flies, images come in → ingestion → inference → decision engine
4. Spray zones appear on dashboard for approval
5. Operator approves → mission planner generates spray path
6. Upload to treatment drone → fly and spray
7. Log everything

### 7. Dashboard Enhancements (later)

- Real-time WebSocket updates (drone telemetry, live detections)
- Mission progress tracking on map (drone position)
- Image viewer for individual detections
- Historical comparison (health over time)

---

## Environment

- **Python:** 3.11 (venv: `drn-env/`)
- **Key packages installed:** ultralytics, torch, opencv-python, albumentations, scikit-learn, fastapi, uvicorn, jinja2
- **Missing packages for next phase:** `shapely`, `pymavlink` or `dronekit`, `exifread`
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
├── drone/                      ← Empty (not yet built)
└── engine/                     ← Does not exist yet (decision engine, planner, ingest)
```
