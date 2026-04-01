# ML Skill — Crop Disease Detection

## Objective

Train and deploy a YOLOv8-nano object detection model to identify crop diseases from aerial drone imagery. The model localizes diseased regions on leaves/fruits with bounding boxes, confidence scores, and class labels.

---

## Model Specification

| Attribute | Value |
|-----------|-------|
| Architecture | YOLOv8-nano |
| Task | Object Detection |
| Framework | Ultralytics (PyTorch) |
| Input | RGB images (640×640 default) |
| Output | Bounding boxes + class label + confidence score |
| Inference Runtime | ONNX Runtime (CPU) or PyTorch |
| Export Formats | ONNX, TFLite (for optional edge deployment) |

## Classes

Start with 6 classes — expand as needed:

1. `healthy`
2. `leaf_blight`
3. `leaf_spot`
4. `rust`
5. `powdery_mildew`
6. `fruit_rot`

---

## Training Data

### Public Datasets

| Dataset | Type | Notes |
|---------|------|-------|
| PlantVillage | Classification (leaf images) | ~54K images, 38 classes — needs conversion to detection format |
| PlantDoc | Detection (bounding boxes) | ~2,500 images, 27 classes — already has bbox annotations |

### Data Preparation

1. **Download** datasets into `ml/data/raw/`
2. **Filter** to relevant disease classes and map to our 6-class taxonomy
3. **Convert** annotations to YOLO format (`class x_center y_center width height`, normalized)
4. **Split** into train/val/test (70/20/10)
5. **Augment** training set:
   - Random horizontal/vertical flip
   - Rotation (±15°)
   - Brightness/contrast jitter
   - Mosaic augmentation (built into Ultralytics)
   - Simulated aerial perspective (zoom/crop to mimic drone altitude variation)
6. **Store** processed data in `ml/data/processed/` following YOLO directory structure:
   ```
   ml/data/processed/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### Dataset Config

Create `ml/configs/dataset.yaml`:
```yaml
path: ../data/processed
train: train/images
val: val/images
test: test/images

names:
  0: healthy
  1: leaf_blight
  2: leaf_spot
  3: rust
  4: powdery_mildew
  5: fruit_rot
```

---

## Training

### Environment

- Python 3.11 (project venv: `drn-env`)
- Key packages: `ultralytics`, `torch`, `opencv-python`, `onnxruntime`

### Training Config

Create `ml/configs/train.yaml`:
```yaml
model: yolov8n.pt          # nano pretrained weights
data: configs/dataset.yaml
epochs: 100
imgsz: 640
batch: 16
patience: 15                # early stopping
device: cpu                 # or 0 for GPU
project: models
name: disease_det_v1
save: true
plots: true
```

### Training Script

Location: `ml/training/train.py`

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="configs/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=15,
    project="models",
    name="disease_det_v1",
)
```

### Training Workflow

1. Start with pretrained YOLOv8-nano (COCO weights) — transfer learning
2. Train on our dataset with frozen backbone for 10 epochs (warm-up)
3. Unfreeze and train full model for remaining epochs
4. Use early stopping (patience=15) to prevent overfitting
5. Best weights saved to `ml/models/disease_det_v1/weights/best.pt`

---

## Evaluation

### Metrics

| Metric | Target (PoC) |
|--------|-------------|
| mAP@0.5 | ≥ 0.70 |
| mAP@0.5:0.95 | ≥ 0.45 |
| Precision | ≥ 0.75 |
| Recall | ≥ 0.70 |
| Inference Speed (CPU) | < 100ms per image |

### Evaluation Script

Location: `ml/training/evaluate.py`

```python
from ultralytics import YOLO

model = YOLO("models/disease_det_v1/weights/best.pt")
metrics = model.val(data="configs/dataset.yaml", split="test")
```

### Evaluation Outputs

- Confusion matrix → `ml/models/disease_det_v1/confusion_matrix.png`
- PR curve → `ml/models/disease_det_v1/PR_curve.png`
- Per-class metrics table
- Sample predictions on test images

---

## Inference

### Inference Script

Location: `ml/inference/detect.py`

Core function:
```python
from ultralytics import YOLO

def detect_diseases(image_path: str, model_path: str, conf_threshold: float = 0.4):
    """
    Run disease detection on a single image.

    Returns list of detections:
        [{"bbox": [x1,y1,x2,y2], "class": str, "confidence": float}, ...]
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf_threshold)
    # parse results into structured output
    ...
```

### ONNX Export

For production/base station deployment:
```python
model = YOLO("models/disease_det_v1/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

Exported model stored at: `ml/models/disease_det_v1/weights/best.onnx`

---

## Integration Points

| Consumer | Interface | Notes |
|----------|-----------|-------|
| Base Station Inference Server | `detect_diseases()` function or REST endpoint | FastAPI wraps the detection function |
| Decision Engine | Detection list with GPS-mapped coordinates | Inference output + GPS offset calculation |
| Dashboard | Detection results + annotated images | For operator review before spray approval |
| Edge (future) | TFLite model on Raspberry Pi | Optional on-drone inference |

---

## Directory Structure

```
ml/
├── SKILL.md              ← this file
├── configs/
│   ├── dataset.yaml      ← dataset paths and class names
│   └── train.yaml        ← training hyperparameters
├── data/
│   ├── raw/              ← original downloaded datasets
│   ├── processed/        ← YOLO-formatted train/val/test splits
│   └── scripts/          ← data download & preprocessing scripts
├── training/
│   ├── train.py          ← training entrypoint
│   └── evaluate.py       ← evaluation entrypoint
├── inference/
│   └── detect.py         ← inference API
├── models/               ← trained model weights and artifacts
└── notebooks/            ← exploratory analysis and experiments
```

---

## Next Steps (Phase 1 Checklist)

- [ ] Download PlantVillage and PlantDoc datasets
- [ ] Write data preprocessing script (filter classes, convert annotations, split)
- [ ] Create `dataset.yaml` config
- [ ] Run training with pretrained YOLOv8-nano
- [ ] Evaluate on test set — hit mAP@0.5 ≥ 0.70
- [ ] Export best model to ONNX
- [ ] Build `detect.py` inference function
- [ ] Test inference on sample drone-like images
