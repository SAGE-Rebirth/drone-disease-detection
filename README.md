# Disease Drone

A two-drone precision agriculture system for automated crop disease detection and targeted treatment. A **Scout Drone** surveys fields and captures geotagged images; a **Treatment Drone** applies treatment only where disease is detected.

**Status:** Proof of Concept (ML training pending, all software modules complete)

## Project Structure

```
disease-drone/
├── ml/                         # Machine learning pipeline
│   ├── configs/                # Training & dataset YAML configs
│   ├── data/scripts/           # Data download & preprocessing
│   ├── training/               # Train & evaluate scripts
│   ├── inference/              # Detection & ONNX export
│   └── models/                 # Trained model weights (empty until training)
├── engine/                     # Core processing engine
│   ├── decision.py             # DBSCAN clustering, GPS conversion, spray zones
│   ├── planner.py              # Lawnmower scan & TSP spray path planning
│   ├── ingest.py               # EXIF GPS extraction, image processing pipeline
│   └── base_station.py         # Central orchestrator
├── drone/                      # Drone communication
│   └── comms.py                # MAVLink connection, telemetry, mission upload
├── dashboard/                  # Web dashboard (FastAPI + Leaflet.js)
│   ├── app.py                  # FastAPI backend & API routes
│   ├── database.py             # SQLite data layer
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS, assets
├── architecture.md             # Full system architecture document
├── RESUME.md                   # Session resume point & progress tracker
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv drn-env
source drn-env/bin/activate    # macOS/Linux
# drn-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## ML Pipeline

YOLOv8-nano (Ultralytics) trained on PlantVillage + PlantDoc datasets for 5-class crop disease detection.

**Classes:** `healthy`, `leaf_blight`, `leaf_spot`, `rust`, `powdery_mildew`

**Dataset:** ~57K images preprocessed into YOLO format (train: 39,810 / val: 11,372 / test: 5,693)

```bash
# Download training data (requires Kaggle API key)
python ml/data/scripts/download_data.py

# Preprocess into YOLO format
python ml/data/scripts/preprocess.py

# Train model (auto-detects MPS on Apple Silicon, CUDA on NVIDIA, or CPU fallback)
python ml/training/train.py

# Override device or hyperparameters
python ml/training/train.py --device mps --epochs 50 --batch 8

# Evaluate on test set
python ml/training/evaluate.py

# Run inference on a single image
python ml/inference/detect.py path/to/image.jpg
```

### Cross-Platform Device Support

Training automatically detects the best available hardware:

| Platform | Device | Config Value |
|----------|--------|-------------|
| macOS (Apple Silicon) | MPS GPU | `mps` |
| Windows/Linux (NVIDIA) | CUDA GPU | `0` |
| Any | CPU fallback | `cpu` |

Set `device: auto` in `ml/configs/train.yaml` (default) or override with `--device`.

## Decision Engine

Converts raw ML detections into actionable spray zones.

```python
from engine.decision import process_detections, Detection

detections = [
    Detection(lat=12.97, lon=77.59, class_name="leaf_blight", confidence=0.9),
    Detection(lat=12.97001, lon=77.59001, class_name="rust", confidence=0.85),
]

# Cluster detections and generate spray zones
zones = process_detections(detections, mission_id=1, db_module=db)
```

**Pipeline:** Filter low-confidence detections -> DBSCAN clustering (GPS coords) -> Convex hull + buffer -> Severity scoring -> Persist to DB

## Mission Planner

Generates flight paths for both scout and treatment drones.

```python
from engine.planner import generate_scan_waypoints, optimize_spray_path

# Lawnmower scan pattern
scan_area = [[12.970, 77.593], [12.970, 77.596], [12.974, 77.596], [12.974, 77.593]]
waypoints = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)

# Optimised spray path (nearest-neighbour TSP)
spray_wps = optimize_spray_path(spray_zones, home=(12.970, 77.593))

# Export for QGroundControl
from engine.planner import mission_to_qgc_plan
plan = mission_to_qgc_plan(waypoints)
```

## Image Ingestion

Processes drone images end-to-end: EXIF GPS extraction -> ML inference -> pixel-to-GPS conversion -> DB persistence.

```python
from engine.ingest import process_folder, watch_folder

# Batch process a folder of drone images
detections = process_folder("path/to/images/", mission_id=1, db_module=db)

# Real-time folder watcher (blocking)
watch_folder("path/to/incoming/", mission_id=1, db_module=db)
```

## Drone Communication

MAVLink interface via pymavlink for ArduPilot-based flight controllers.

```python
from drone.comms import DroneLink

drone = DroneLink("udp:127.0.0.1:14550")  # SITL simulator
# drone = DroneLink("/dev/ttyUSB0")        # Serial radio
# drone = DroneLink("tcp:192.168.1.10:5760")  # WiFi

drone.connect()
drone.upload_mission(mavlink_items)
drone.arm_and_start_mission()

telemetry = drone.get_telemetry()
print(f"Position: {telemetry.lat}, {telemetry.lon} | Battery: {telemetry.battery_remaining}%")
```

## Base Station Orchestrator

Ties all components together for the full end-to-end workflow.

```python
from engine.base_station import BaseStation

bs = BaseStation()

# Plan and export a scan mission (works without live drone)
mission_id, waypoints = bs.plan_scan_mission(scan_polygon)
bs.export_scan_plan(scan_polygon, "scan_mission.plan")

# Process scout images after flight
zones = bs.process_scout_images("path/to/images/", mission_id)

# Plan and export a spray mission from approved zones
spray_id, spray_wps = bs.plan_spray_mission(scan_mission_id=mission_id)
bs.export_spray_plan("spray_mission.plan", scan_mission_id=mission_id)
```

## Dashboard

Web-based control interface for mission management, detection visualization, and spray zone approval.

```bash
uvicorn dashboard.app:app --reload --port 8000
```

Then open `http://localhost:8000`. Use **Seed Demo Data** to populate the map with sample data.

**Features:**
- Dark themed map (CARTO dark tiles) with Leaflet.js
- Draw scan area polygons on map
- Detection markers colour-coded by disease type
- Spray zone polygons with approve/reject workflow
- Health heatmap toggle (green -> red gradient)
- Stat cards, tabbed panels, toast notifications

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/stats` | Summary statistics |
| POST | `/api/missions` | Create a mission |
| GET | `/api/missions` | List missions |
| GET | `/api/detections` | List detections |
| POST | `/api/detections` | Add detection(s) |
| GET | `/api/spray-zones` | List spray zones |
| POST | `/api/spray-zones` | Add spray zone |
| PUT | `/api/spray-zones/{id}` | Approve/reject a zone |
| GET | `/api/treatments` | List treatments |
| POST | `/api/treatments` | Log treatment |
| GET | `/api/health` | Health heatmap data |
| POST | `/api/demo/seed` | Seed demo data |
| POST | `/api/demo/clear` | Clear all data |

## Architecture

See [architecture.md](architecture.md) for the full system design, hardware recommendations, data flow, and budget breakdown.

### End-to-End Flow

```
1. Operator draws scan area on dashboard
2. Mission planner generates lawnmower waypoints
3. Scout drone flies mission, captures geotagged images
4. Images transferred to base station
5. ML model runs inference on each image
6. Detections clustered into spray zones (DBSCAN + convex hull)
7. Spray zones displayed on dashboard for operator approval
8. Operator approves -> treatment drone receives optimised spray path
9. Treatment drone flies to zones, applies treatment
10. All data logged to SQLite, health map updated
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Training & Inference | PyTorch, Ultralytics YOLOv8, ONNX Runtime |
| Decision Engine | scikit-learn (DBSCAN), Shapely (geometry) |
| Mission Planning | Custom lawnmower + nearest-neighbour TSP |
| Drone Communication | pymavlink (MAVLink protocol) |
| Image Processing | OpenCV, Pillow, ExifRead |
| Backend | FastAPI, SQLite |
| Frontend | Jinja2, Leaflet.js, OpenStreetMap |

## License

This project is for educational and research purposes.
