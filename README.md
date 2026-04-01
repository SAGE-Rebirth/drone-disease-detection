# Disease Drone

A two-drone precision agriculture system for automated crop disease detection and targeted treatment. A **Scout Drone** surveys fields and captures geotagged images; a **Treatment Drone** applies treatment only where disease is detected.

**Status:** Proof of Concept

## Project Structure

```
disease-drone/
├── ml/                     # Machine learning pipeline
│   ├── configs/            # Training & dataset YAML configs
│   ├── data/scripts/       # Data download & preprocessing
│   ├── training/           # Train & evaluate scripts
│   └── inference/          # Detection/inference script
├── dashboard/              # Web dashboard (FastAPI + Leaflet.js)
│   ├── app.py              # FastAPI backend & API routes
│   ├── database.py         # SQLite data layer
│   ├── templates/          # Jinja2 HTML templates
│   └── static/             # CSS, JS, assets
├── drone/                  # Drone integration (planned)
├── architecture.md         # Full system architecture document
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv drn-env
source drn-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ML Pipeline

The ML pipeline uses YOLOv8-nano (Ultralytics) to classify crop diseases into 5 classes: `healthy`, `leaf_blight`, `leaf_spot`, `rust`, `powdery_mildew`.

```bash
# Download training data
python ml/data/scripts/download_data.py

# Preprocess dataset
python ml/data/scripts/preprocess.py

# Train model
python ml/training/train.py

# Evaluate model
python ml/training/evaluate.py

# Run inference on images
python ml/inference/detect.py --source path/to/images
```

## Dashboard

A web-based control interface for mission management, detection visualization, and spray zone approval.

```bash
# Start the dashboard
uvicorn dashboard.app:app --reload --port 8000
```

Then open `http://localhost:8000`. Use the **Seed Demo Data** button to populate the map with sample detections and spray zones.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/stats` | Summary statistics |
| POST | `/api/missions` | Create a mission |
| GET | `/api/missions` | List missions |
| GET | `/api/detections` | List detections |
| GET | `/api/spray-zones` | List spray zones |
| PUT | `/api/spray-zones/{id}` | Approve/reject a zone |
| GET | `/api/treatments` | List treatments |
| GET | `/api/health` | Health heatmap data |
| POST | `/api/demo/seed` | Seed demo data |

## Architecture

See [architecture.md](architecture.md) for the full system design, hardware recommendations, data flow, and budget breakdown.

### High-Level Flow

```
Scout Drone → captures images → Base Station (ML inference)
  → disease detections clustered into spray zones
  → operator approves zones via dashboard
  → Treatment Drone flies to zones and applies treatment
```

## Tech Stack

- **ML:** PyTorch, Ultralytics YOLOv8, ONNX Runtime
- **Backend:** FastAPI, SQLite
- **Frontend:** Jinja2, Leaflet.js, OpenStreetMap
- **Drone Comms:** MAVLink / ArduPilot (planned)

## License

This project is for educational and research purposes.
