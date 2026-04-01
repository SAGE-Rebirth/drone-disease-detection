# Disease Drone — System Architecture

## Overview

A two-drone precision agriculture system for crop disease detection and targeted treatment. A **Scout Drone** surveys fields and identifies diseased regions; a **Treatment Drone** applies agricultural treatment only where needed.

**Scope:** Proof of Concept (PoC) — small to medium budget.

---

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER DASHBOARD                           │
│  (Web app — define scan area, monitor status, view health map)  │
└──────────────┬──────────────────────────────────┬───────────────┘
               │ scan commands                    │ health reports
               ▼                                  ▲
┌──────────────────────────┐       ┌──────────────────────────────┐
│      BASE STATION        │       │     LOGGING & ANALYTICS      │
│  (Laptop/Mini PC)        │       │  (SQLite + health map viz)   │
│                          │       │                              │
│  • ML inference server   │──────▶│  • Detection history         │
│  • Decision engine       │       │  • Treatment logs            │
│  • Mission planner       │       │  • Field health over time    │
│  • Drone comms (MAVLink) │       └──────────────────────────────┘
└─────┬──────────────┬─────┘
      │              │
      │ WiFi/Radio   │ WiFi/Radio
      ▼              ▼
┌────────────┐  ┌────────────┐
│SCOUT DRONE │  │ TREATMENT  │
│            │  │   DRONE    │
│• Camera    │  │            │
│• Edge ML   │  │• Spray     │
│  (optional)│  │  system    │
│• GPS       │  │• GPS       │
│• Flight    │  │• Flight    │
│  controller│  │  controller│
└────────────┘  └────────────┘
```

---

## 1. Scout Drone

**Purpose:** Fly over the defined scan area, capture crop images, and (optionally) run lightweight edge inference.

### Hardware (PoC Budget)

| Component | Recommendation | Est. Cost |
|-----------|---------------|-----------|
| Frame + Flight Controller | DJI Tello EDU or custom F450 with Pixhawk | $100–$300 |
| Camera | Raspberry Pi Camera Module v3 (12MP) or onboard drone camera | $25–$35 |
| Edge Compute (optional) | Raspberry Pi 4 / Raspberry Pi Zero 2W | $35–$75 |
| GPS Module | BN-880 or similar u-blox based | $15–$25 |
| Battery | 3S/4S LiPo | $25–$50 |
| **Subtotal** | | **$200–$485** |

> **PoC Simplification:** For initial PoC, skip edge inference entirely. Scout drone just captures geotagged images and uploads them to the base station. Add edge inference later as an optimization.

### Flight Plan
- Lawnmower/zigzag pattern over the defined scan area
- Altitude: 3–5 meters for leaf-level detail (adjustable)
- Overlap: 30% between image captures for stitching
- Image capture interval: GPS-triggered or time-based

### Data Output
- JPEG images with EXIF GPS metadata
- Or: video stream with GPS timestamps

---

## 2. Base Station (ML + Decision Engine)

**Purpose:** Run detailed disease detection inference, cluster results, and generate spray coordinates.

### Hardware
- Any laptop or mini PC with decent CPU (GPU optional for PoC)
- For faster inference: NVIDIA Jetson Nano ($150) or laptop with GPU

### Software Stack

```
Images from Scout Drone
        │
        ▼
┌──────────────────┐
│  IMAGE INGESTION  │  ← receives images via WiFi/USB
│  & Preprocessing  │  ← resize, normalize, tile large images
└────────┬─────────┘
         ▼
┌──────────────────┐
│   ML INFERENCE    │  ← YOLOv8-nano or EfficientNet-Lite
│                   │  ← classifies: healthy / disease_type
│                   │  ← outputs: bounding boxes + confidence + GPS
└────────┬─────────┘
         ▼
┌──────────────────┐
│  DECISION ENGINE  │  ← filters low-confidence detections
│                   │  ← clusters nearby detections
│                   │  ← assigns severity scores
│                   │  ← generates spray coordinates
└────────┬─────────┘
         ▼
┌──────────────────┐
│  MISSION PLANNER  │  ← converts spray coords to waypoints
│                   │  ← optimizes treatment drone flight path
│                   │  ← sends mission via MAVLink
└──────────────────┘
```

### ML Model Strategy (PoC)

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Architecture | YOLOv8-nano | Fast, small, good accuracy, easy to train |
| Task | Object Detection | Localize disease on leaves/fruits with bounding boxes |
| Training Data | PlantVillage + PlantDoc + custom drone images | Free public datasets + augmentation |
| Framework | Ultralytics (PyTorch) | Simple API, export to ONNX/TFLite |
| Inference | ONNX Runtime or PyTorch | CPU-friendly, no GPU required |
| Classes | healthy, leaf_blight, leaf_spot, rust, powdery_mildew, fruit_rot (expandable) | Common diseases, start small |

### Decision Engine Logic

```python
# Pseudocode
for detection in detections:
    if detection.confidence < THRESHOLD:
        continue
    gps_coord = image_gps + pixel_to_offset(detection.bbox, altitude, fov)
    filtered.append(detection)

clusters = DBSCAN(filtered, eps=2m, min_samples=2)

spray_zones = []
for cluster in clusters:
    zone = convex_hull(cluster.points)
    zone.buffer(1m)  # safety margin
    spray_zones.append(zone)
```

---

## 3. Treatment Drone

**Purpose:** Fly to identified spray zones and apply targeted treatment.

### Hardware (PoC Budget)

| Component | Recommendation | Est. Cost |
|-----------|---------------|-----------|
| Frame | Custom hex/quad with payload capacity (1–2 kg) | $150–$300 |
| Flight Controller | Pixhawk or similar ArduPilot compatible | $50–$100 |
| Spray System | Small peristaltic pump + nozzle + reservoir (500ml) | $30–$60 |
| GPS Module | BN-880 | $15–$25 |
| Battery | 4S/6S LiPo (higher capacity for payload) | $40–$80 |
| **Subtotal** | | **$285–$565** |

> **PoC Simplification:** For initial demo, the treatment drone can just fly the path and log "would spray here" without an actual spray system. Prove the navigation and targeting accuracy first.

### Flight Behavior
- Receives waypoint list from base station
- Flies optimized path (nearest-neighbor or simple TSP)
- At each spray zone: hover, activate pump for calculated duration
- Return to base when done or battery low

---

## 4. User Dashboard

**Purpose:** Control interface for the farmer/operator.

### PoC Implementation
- Simple web app (Flask or Streamlit)
- Features:
  - Draw scan area on a map (Leaflet.js)
  - Start/stop survey mission
  - View detection results overlaid on map
  - View health heatmap
  - Approve/reject spray mission before treatment drone launches
  - View treatment logs

### Tech Stack
- **Backend:** Python (Flask/FastAPI)
- **Frontend:** Streamlit (fastest for PoC) or simple HTML + Leaflet.js
- **Database:** SQLite (sufficient for PoC)
- **Map Tiles:** OpenStreetMap (free)

---

## 5. Communication

| Link | Protocol | Notes |
|------|----------|-------|
| Dashboard ↔ Base Station | HTTP/WebSocket | Same machine or local network |
| Base Station ↔ Scout Drone | MAVLink over WiFi/telemetry radio | ArduPilot/PX4 standard |
| Base Station ↔ Treatment Drone | MAVLink over WiFi/telemetry radio | Same as above |
| Image Transfer | WiFi (in-flight) or USB (post-landing) | WiFi for real-time, USB for PoC simplicity |

---

## 6. Data Flow (End-to-End)

```
1. Operator draws scan area on dashboard
2. Base station generates lawnmower waypoints
3. Scout drone flies mission, captures geotagged images
4. Images transferred to base station (WiFi or post-flight USB)
5. ML model runs inference on each image
6. Detections filtered, clustered into spray zones
7. Spray zones displayed on dashboard for operator approval
8. Operator approves → treatment drone receives waypoints
9. Treatment drone flies to zones, applies treatment
10. All data logged: detections, spray events, timestamps, GPS
11. Health map updated and displayed on dashboard
```

---

## 7. Budget Summary (PoC)

| Component | Estimated Cost |
|-----------|---------------|
| Scout Drone (hardware) | $200–$485 |
| Treatment Drone (hardware) | $285–$565 |
| Base Station (existing laptop) | $0 (use existing) |
| ML Training (cloud GPU if needed) | $0–$50 (Google Colab free tier or Kaggle) |
| Misc (wires, connectors, SD cards) | $30–$50 |
| **Total** | **$515–$1,150** |

---

## 8. PoC Phases

### Phase 1: ML Model (Current Focus)
- Collect/prepare training data (PlantVillage, PlantDoc datasets)
- Train YOLOv8-nano for disease detection
- Evaluate accuracy on test set
- Export model for inference

### Phase 2: Base Station Software
- Image ingestion pipeline
- Inference server
- Decision engine (clustering, spray zone generation)
- Basic dashboard (Streamlit)

### Phase 3: Scout Drone Integration
- Flight controller setup (ArduPilot)
- Autonomous waypoint mission
- Image capture with GPS tagging
- Image transfer to base station

### Phase 4: Treatment Drone Integration
- Flight controller setup
- Waypoint navigation to spray zones
- Spray system integration (or simulated)

### Phase 5: End-to-End Demo
- Full pipeline: scan → detect → approve → treat
- Health map visualization
- Performance metrics

---

## 9. Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| ML Training | Python, PyTorch, Ultralytics YOLOv8 |
| ML Inference | ONNX Runtime / PyTorch |
| Backend | Python, FastAPI |
| Dashboard | Streamlit (PoC) |
| Database | SQLite |
| Maps | Leaflet.js + OpenStreetMap |
| Drone Firmware | ArduPilot (open source) |
| Drone Comms | MAVLink, pymavlink |
| Flight Planning | dronekit or pymavlink |
| Image Processing | OpenCV, Pillow |
| Clustering | scikit-learn (DBSCAN) |
| GPS Utils | geopy |

---

## 10. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| ML model accuracy too low | Start with well-known datasets, fine-tune iteratively, augment data |
| Drone flight regulations | Check local regulations, fly in permitted areas, maintain visual line of sight |
| GPS accuracy insufficient for spray targeting | Use RTK GPS for treatment drone ($50–$100 module), or accept 2–3m buffer zones |
| Weather/wind affects spray | PoC constraint: operate in calm conditions only |
| Budget overrun | Phase the build — ML first (zero hardware cost), add drones incrementally |
