"""
Train YOLOv8-nano for crop disease detection.

Uses transfer learning from COCO-pretrained weights.
Config-driven via ml/configs/train.yaml.

Usage:
    python ml/training/train.py
    python ml/training/train.py --config configs/train.yaml
    python ml/training/train.py --epochs 50 --batch 8  # override config
"""

import argparse
import platform
import tempfile
import yaml
from pathlib import Path

import torch
from ultralytics import YOLO

# Project root = ml/
ML_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ML_DIR / "configs" / "train.yaml"


def detect_device():
    """Auto-detect the best available device for training.

    Returns:
        str: 'mps' on Apple Silicon, 'cuda'/'0' if NVIDIA/AMD GPU available, 'cpu' otherwise.
    """
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        print("Detected Apple Silicon — using MPS backend")
        return "mps"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Detected GPU — using CUDA device 0 ({gpu_name})")
        return "0"
    print("No GPU detected — falling back to CPU")
    return "cpu"


def load_config(config_path):
    """Load training config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(data_config_path):
    """Ensure the 'path' field in dataset.yaml is an absolute, platform-correct path.

    If the dataset config uses a relative path, resolve it relative to
    the config file's own directory so the project is portable across OS.
    """
    data_config_path = Path(data_config_path)
    with open(data_config_path, "r") as f:
        ds_cfg = yaml.safe_load(f)

    raw_path = ds_cfg.get("path", "")
    if raw_path and not Path(raw_path).is_absolute():
        resolved = (data_config_path.parent / raw_path).resolve()
        ds_cfg["path"] = str(resolved)
        # Write back a temp copy so YOLO sees the absolute path
        tmp = Path(tempfile.mkdtemp()) / "dataset.yaml"
        with open(tmp, "w") as f:
            yaml.dump(ds_cfg, f, default_flow_style=False)
        return str(tmp)
    return str(data_config_path)


def train(config):
    """Run YOLO training with the given config."""
    model_name = config.pop("model", "yolov8n.pt")
    data_path = config.get("data", "configs/dataset.yaml")

    # Auto-detect device if set to "auto" or not specified
    if config.get("device", "auto") == "auto":
        config["device"] = detect_device()

    # Resolve data path relative to ml/ directory
    if not Path(data_path).is_absolute():
        data_path = str(ML_DIR / data_path)

    # Resolve dataset.yaml 'path' field to an absolute, cross-platform path
    config["data"] = resolve_dataset_path(data_path)

    # Resolve project path relative to ml/ directory
    project = config.get("project", "models")
    if not Path(project).is_absolute():
        config["project"] = str(ML_DIR / project)

    output_dir = Path(config["project"]) / config.get("name", "train")
    print(f"Loading model: {model_name}")
    print(f"Dataset config: {config['data']}")
    print(f"Output: {output_dir}")
    print()

    model = YOLO(model_name)

    # Phase 1: Freeze backbone for warm-up (first 10 epochs)
    freeze_epochs = min(10, config.get("epochs", 100))
    full_epochs = config.get("epochs", 100)

    print(f"--- Phase 1: Frozen backbone ({freeze_epochs} epochs) ---")
    config_phase1 = {**config, "epochs": freeze_epochs, "freeze": 10}
    # Use a temp name for phase 1
    config_phase1["name"] = config.get("name", "disease_det_v1") + "_warmup"
    results_warmup = model.train(**config_phase1)

    # Phase 2: Full training (unfreeze all layers)
    remaining = full_epochs - freeze_epochs
    if remaining > 0:
        print(f"\n--- Phase 2: Full model ({remaining} epochs) ---")
        # Load best weights from warmup
        warmup_best = Path(config["project"]) / config_phase1["name"] / "weights" / "best.pt"
        if warmup_best.exists():
            model = YOLO(str(warmup_best))
        config_phase2 = {**config, "epochs": remaining}
        results = model.train(**config_phase2)
    else:
        results = results_warmup

    # Print final results
    best_weights = Path(config["project"]) / config.get("name", "disease_det_v1") / "weights" / "best.pt"
    print(f"\n--- Training Complete ---")
    print(f"Best weights: {best_weights}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano disease detector")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to train.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, 0, etc.)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.device is not None:
        config["device"] = args.device

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        train(config)


if __name__ == "__main__":
    main()
