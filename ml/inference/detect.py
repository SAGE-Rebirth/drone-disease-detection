"""
Inference module for crop disease detection.

Provides:
    - detect_diseases(): Run detection on a single image
    - detect_batch(): Run detection on multiple images
    - export_onnx(): Export model to ONNX format
    - DetectionResult: Structured output dataclass

Usage:
    from ml.inference.detect import detect_diseases, export_onnx

    # Single image
    results = detect_diseases("path/to/image.jpg")
    for det in results:
        print(det.class_name, det.confidence, det.bbox)

    # Export to ONNX
    export_onnx("models/disease_det_v1/weights/best.pt")
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ML_DIR / "models" / "disease_det_v1" / "weights" / "best.pt"

CLASS_NAMES = ["healthy", "leaf_blight", "leaf_spot", "rust", "powdery_mildew"]


@dataclass
class DetectionResult:
    """Single detection result."""
    bbox: list  # [x1, y1, x2, y2] in pixels
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self):
        return asdict(self)


def detect_diseases(
    image_path: str,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
) -> list[DetectionResult]:
    """
    Run disease detection on a single image.

    Args:
        image_path: Path to input image.
        model_path: Path to model weights. Defaults to best.pt.
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        imgsz: Input image size.

    Returns:
        List of DetectionResult objects.
    """
    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

            detections.append(DetectionResult(
                bbox=[round(v, 2) for v in bbox],
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf, 4),
            ))

    return detections


def detect_batch(
    image_paths: list[str],
    model_path: Optional[str] = None,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
) -> dict[str, list[DetectionResult]]:
    """
    Run disease detection on multiple images.

    Returns:
        Dict mapping image path → list of DetectionResult.
    """
    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    model = YOLO(model_path)
    results = model.predict(
        source=image_paths,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    output = {}
    for img_path, result in zip(image_paths, results):
        detections = []
        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

            detections.append(DetectionResult(
                bbox=[round(v, 2) for v in bbox],
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf, 4),
            ))
        output[img_path] = detections

    return output


def export_onnx(
    model_path: Optional[str] = None,
    imgsz: int = 640,
    simplify: bool = True,
) -> str:
    """
    Export YOLO model to ONNX format.

    Returns:
        Path to exported ONNX file.
    """
    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    print(f"Exporting to ONNX: {model_path}")
    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=simplify)
    print(f"Exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disease detection inference")
    parser.add_argument("image", nargs="?", help="Image path for detection")
    parser.add_argument("--model", type=str, default=None, help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX")
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx(args.model)
    elif args.image:
        detections = detect_diseases(args.image, model_path=args.model, conf_threshold=args.conf)
        if not detections:
            print("No detections.")
        for det in detections:
            print(f"  {det.class_name} ({det.confidence:.2%}) @ {det.bbox}")
    else:
        print("Provide an image path or use --export-onnx")
