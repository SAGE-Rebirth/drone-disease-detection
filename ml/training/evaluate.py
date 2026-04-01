"""
Evaluate trained YOLOv8 disease detection model on the test set.

Outputs:
    - mAP, precision, recall metrics
    - Confusion matrix
    - PR curve
    - Sample predictions

Usage:
    python ml/training/evaluate.py
    python ml/training/evaluate.py --model models/disease_det_v1/weights/best.pt
    python ml/training/evaluate.py --model models/disease_det_v1/weights/best.pt --split test
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ML_DIR / "models" / "disease_det_v1" / "weights" / "best.pt"
DEFAULT_DATA = ML_DIR / "configs" / "dataset.yaml"


def evaluate(model_path, data_path, split="test", conf=0.25, iou=0.6):
    """Run evaluation and print metrics."""
    print(f"Model:   {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Split:   {split}")
    print()

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=split,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision:      {metrics.box.mp:.4f}")
    print(f"  Recall:         {metrics.box.mr:.4f}")

    # Per-class metrics
    class_names = model.names
    print(f"\n{'Class':<20} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 58)
    for i, (p, r, ap50, ap) in enumerate(
        zip(
            metrics.box.p,
            metrics.box.r,
            metrics.box.ap50,
            metrics.box.ap,
        )
    ):
        name = class_names.get(i, f"class_{i}")
        print(f"  {name:<18} {p:>8.4f} {r:>8.4f} {ap50:>8.4f} {ap:>10.4f}")

    # Check against PoC targets
    print("\n--- PoC Target Check ---")
    targets = {
        "mAP@0.5": (metrics.box.map50, 0.70),
        "mAP@0.5:0.95": (metrics.box.map, 0.45),
        "Precision": (metrics.box.mp, 0.75),
        "Recall": (metrics.box.mr, 0.70),
    }
    for name, (actual, target) in targets.items():
        status = "PASS" if actual >= target else "MISS"
        print(f"  [{status}] {name}: {actual:.4f} (target: {target:.2f})")

    print(f"\nPlots saved to: {metrics.save_dir}")
    return metrics


def predict_samples(model_path, data_dir, n=10, conf=0.4):
    """Run inference on sample images and save annotated results."""
    model = YOLO(str(model_path))
    test_images = list((Path(data_dir) / "test" / "images").glob("*.*"))

    if not test_images:
        print("No test images found.")
        return

    samples = test_images[:n]
    print(f"\nRunning predictions on {len(samples)} sample images...")

    results = model.predict(
        source=[str(p) for p in samples],
        conf=conf,
        save=True,
        project=str(ML_DIR / "models" / "disease_det_v1"),
        name="sample_predictions",
    )

    print(f"Sample predictions saved to: {ML_DIR / 'models' / 'disease_det_v1' / 'sample_predictions'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate disease detection model")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model weights path")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Dataset config path")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--samples", type=int, default=10, help="Number of sample predictions")
    args = parser.parse_args()

    metrics = evaluate(args.model, args.data, split=args.split, conf=args.conf)

    # Also run sample predictions
    processed_dir = ML_DIR / "data" / "processed"
    if processed_dir.exists():
        predict_samples(args.model, processed_dir, n=args.samples)


if __name__ == "__main__":
    main()
