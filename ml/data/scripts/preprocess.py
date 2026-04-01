"""
Preprocess PlantVillage and PlantDoc datasets into YOLO detection format.

PlantVillage is a classification dataset (folder-per-class, no bounding boxes).
We convert it to detection format by treating each image as a full-image detection
(bbox = entire image), which works for single-leaf images.

PlantDoc has bounding box annotations and maps directly to detection format.

Output structure:
    ml/data/processed/
    ├── train/images/  train/labels/
    ├── val/images/    val/labels/
    └── test/images/   test/labels/

Usage:
    python ml/data/scripts/preprocess.py
"""

import os
import csv
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ----- Paths -----
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PLANTVILLAGE_DIR = RAW_DIR / "plantvillage"
PLANTDOC_DIR = RAW_DIR / "plantdoc"

# ----- Class mapping -----
# Our 6 target classes
CLASS_NAMES = ["healthy", "leaf_blight", "leaf_spot", "rust", "powdery_mildew"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Map PlantVillage folder names → our classes
# PlantVillage folders are like "Tomato___Late_blight", "Apple___healthy", etc.
PLANTVILLAGE_MAP = {
    # Healthy
    "Apple___healthy": "healthy",
    "Blueberry___healthy": "healthy",
    "Cherry_(including_sour)___healthy": "healthy",
    "Corn_(maize)___healthy": "healthy",
    "Grape___healthy": "healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "leaf_spot",
    "Peach___healthy": "healthy",
    "Pepper,_bell___healthy": "healthy",
    "Potato___healthy": "healthy",
    "Raspberry___healthy": "healthy",
    "Soybean___healthy": "healthy",
    "Squash___Powdery_mildew": "powdery_mildew",
    "Strawberry___healthy": "healthy",
    "Tomato___healthy": "healthy",

    # Leaf blight
    "Corn_(maize)___Northern_Leaf_Blight": "leaf_blight",
    "Potato___Late_blight": "leaf_blight",
    "Potato___Early_blight": "leaf_blight",
    "Tomato___Late_blight": "leaf_blight",
    "Tomato___Early_blight": "leaf_blight",

    # Leaf spot
    "Apple___Black_rot": "leaf_spot",
    "Apple___Cedar_apple_rust": "rust",
    "Cherry_(including_sour)___Powdery_mildew": "powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "leaf_spot",
    "Corn_(maize)___Common_rust_": "rust",
    "Grape___Black_rot": "leaf_spot",
    "Grape___Esca_(Black_Measles)": "leaf_spot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "leaf_blight",
    "Pepper,_bell___Bacterial_spot": "leaf_spot",
    "Strawberry___Leaf_scorch": "leaf_spot",
    "Tomato___Bacterial_spot": "leaf_spot",
    "Tomato___Leaf_Mold": "leaf_spot",
    "Tomato___Septoria_leaf_spot": "leaf_spot",
    "Tomato___Target_Spot": "leaf_spot",

    # Rust
    "Apple___Apple_scab": "rust",

    # Powdery mildew
    "Peach___Bacterial_spot": "leaf_spot",

    # Fruit rot
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "leaf_spot",
    "Tomato___Tomato_mosaic_virus": "leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "leaf_spot",
}

# Map PlantDoc class names → our classes
PLANTDOC_MAP = {
    # Healthy
    "Apple leaf": "healthy",
    "Blueberry leaf": "healthy",
    "Cherry leaf": "healthy",
    "Corn leaf": "healthy",
    "grape leaf": "healthy",
    "Grape leaf": "healthy",
    "Peach leaf": "healthy",
    "Bell_pepper leaf": "healthy",
    "Pepper bell leaf": "healthy",
    "Potato leaf": "healthy",
    "Raspberry leaf": "healthy",
    "Soyabean leaf": "healthy",
    "Soybean leaf": "healthy",
    "Squash Powdery mildew leaf": "powdery_mildew",
    "Strawberry leaf": "healthy",
    "Tomato leaf": "healthy",

    # Blight
    "Potato leaf early blight": "leaf_blight",
    "Potato leaf late blight": "leaf_blight",
    "Tomato Early blight leaf": "leaf_blight",
    "Tomato late blight leaf": "leaf_blight",
    "Tomato leaf late blight": "leaf_blight",
    "Corn leaf blight": "leaf_blight",
    "grape leaf black rot": "leaf_spot",

    # Leaf spot
    "Apple Scab Leaf": "rust",
    "Apple rust leaf": "rust",
    "Bell_pepper leaf spot": "leaf_spot",
    "Tomato leaf bacterial spot": "leaf_spot",
    "Tomato leaf mosaic virus": "leaf_spot",
    "Tomato leaf yellow virus": "leaf_spot",
    "Tomato Septoria leaf spot": "leaf_spot",
    "Corn Gray leaf spot": "leaf_spot",
    "Corn rust leaf": "rust",

    # Mold / mildew
    "Tomato mold leaf": "powdery_mildew",
    "Cherry Powdery mildew leaf": "powdery_mildew",

    # Two-spotted spider mite
    "Tomato two spotted spider mites leaf": "leaf_spot",

    # Black rot
    "Apple Black rot Leaf": "leaf_spot",
}


def process_plantvillage():
    """
    Convert PlantVillage classification images to YOLO detection format.
    Each image gets a full-image bounding box label.
    """
    samples = []

    if not PLANTVILLAGE_DIR.exists():
        print("[PlantVillage] Not found, skipping. Run download_data.py first.")
        return samples

    # PlantVillage has structure: plantvillage/color/<class>/, plantvillage/segmented/, etc.
    # Only use color/ images to avoid triplicates from grayscale/segmented.
    color_dir = PLANTVILLAGE_DIR / "color"
    search_root = color_dir if color_dir.exists() else PLANTVILLAGE_DIR

    search_dirs = []
    for item in search_root.iterdir():
        if item.is_dir() and item.name in PLANTVILLAGE_MAP:
            search_dirs.append(item)

    if not search_dirs:
        # Fallback: search recursively
        for item in PLANTVILLAGE_DIR.rglob("*"):
            if item.is_dir() and item.name in PLANTVILLAGE_MAP:
                search_dirs.append(item)

    print(f"[PlantVillage] Found {len(search_dirs)} class directories")

    for class_dir in search_dirs:
        class_name = PLANTVILLAGE_MAP.get(class_dir.name)
        if class_name is None:
            continue

        class_id = CLASS_TO_ID[class_name]
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))

        for img_path in images:
            # Full-image bounding box in YOLO format: class x_center y_center width height
            # Normalized: center=(0.5, 0.5), size=(1.0, 1.0)
            label_line = f"{class_id} 0.5 0.5 1.0 1.0"
            samples.append({
                "image_path": img_path,
                "label_line": label_line,
                "source": "plantvillage",
                "class_name": class_name,
            })

    print(f"[PlantVillage] Collected {len(samples)} samples")
    return samples


def parse_plantdoc_xml(xml_path):
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objects.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})

    return img_w, img_h, objects


def process_plantdoc():
    """
    Convert PlantDoc dataset (Pascal VOC format) to YOLO detection format.
    """
    samples = []

    if not PLANTDOC_DIR.exists():
        print("[PlantDoc] Not found, skipping. Run download_data.py first.")
        return samples

    # PlantDoc structure varies — look for XML annotations
    xml_files = list(PLANTDOC_DIR.rglob("*.xml"))
    print(f"[PlantDoc] Found {len(xml_files)} annotation files")

    if not xml_files:
        # Fallback: treat as classification dataset (folder-per-class)
        print("[PlantDoc] No XML annotations found, treating as classification dataset")
        for class_dir in PLANTDOC_DIR.rglob("*"):
            if not class_dir.is_dir():
                continue
            class_name = PLANTDOC_MAP.get(class_dir.name)
            if class_name is None:
                continue

            class_id = CLASS_TO_ID[class_name]
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))

            for img_path in images:
                label_line = f"{class_id} 0.5 0.5 1.0 1.0"
                samples.append({
                    "image_path": img_path,
                    "label_line": label_line,
                    "source": "plantdoc",
                    "class_name": class_name,
                })

        print(f"[PlantDoc] Collected {len(samples)} samples (classification mode)")
        return samples

    # Process XML annotations
    for xml_path in xml_files:
        try:
            img_w, img_h, objects = parse_plantdoc_xml(xml_path)
        except Exception as e:
            print(f"  Warning: failed to parse {xml_path}: {e}")
            continue

        if img_w == 0 or img_h == 0:
            continue

        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".JPG", ".png", ".jpeg"]:
            candidate = xml_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        label_lines = []
        class_name_first = None
        for obj in objects:
            mapped = PLANTDOC_MAP.get(obj["name"])
            if mapped is None:
                continue
            class_id = CLASS_TO_ID[mapped]
            if class_name_first is None:
                class_name_first = mapped

            xmin, ymin, xmax, ymax = obj["bbox"]
            # Convert to YOLO format (normalized x_center, y_center, width, height)
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if label_lines:
            samples.append({
                "image_path": img_path,
                "label_line": "\n".join(label_lines),
                "source": "plantdoc",
                "class_name": class_name_first,
            })

    print(f"[PlantDoc] Collected {len(samples)} samples (detection mode)")
    return samples


def split_and_save(samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split samples into train/val/test and save to YOLO directory structure.
    Stratified by class_name to maintain class balance across splits.
    """
    random.seed(seed)

    # Group by class
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["class_name"]].append(s)

    splits = {"train": [], "val": [], "test": []}

    for class_name, class_samples in by_class.items():
        random.shuffle(class_samples)
        n = len(class_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend(class_samples[:n_train])
        splits["val"].extend(class_samples[n_train:n_train + n_val])
        splits["test"].extend(class_samples[n_train + n_val:])

    # Shuffle each split
    for split_name in splits:
        random.shuffle(splits[split_name])

    # Save
    for split_name, split_samples in splits.items():
        img_dir = PROCESSED_DIR / split_name / "images"
        lbl_dir = PROCESSED_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(split_samples):
            src_img = sample["image_path"]
            ext = src_img.suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                ext = ".jpg"

            # Unique filename: source_index_originalname
            dst_name = f"{sample['source']}_{i:05d}{ext}"
            dst_img = img_dir / dst_name
            dst_lbl = lbl_dir / (dst_name.rsplit(".", 1)[0] + ".txt")

            # Copy image
            shutil.copy2(str(src_img), str(dst_img))

            # Write label
            with open(dst_lbl, "w") as f:
                f.write(sample["label_line"] + "\n")

        print(f"  {split_name}: {len(split_samples)} samples → {img_dir}")

    return splits


def main():
    print("=" * 60)
    print("Disease Drone — Data Preprocessing")
    print("=" * 60)

    # Clean processed dir
    if PROCESSED_DIR.exists():
        print(f"\nClearing existing processed data at {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    all_samples = []

    print("\n--- Processing PlantVillage ---")
    all_samples.extend(process_plantvillage())

    print("\n--- Processing PlantDoc ---")
    all_samples.extend(process_plantdoc())

    if not all_samples:
        print("\nNo samples found! Make sure datasets are downloaded.")
        print("Run: python ml/data/scripts/download_data.py")
        return

    # Print class distribution
    print(f"\n--- Class Distribution ({len(all_samples)} total) ---")
    dist = defaultdict(int)
    for s in all_samples:
        dist[s["class_name"]] += 1
    for cls in CLASS_NAMES:
        print(f"  {cls}: {dist[cls]}")

    # Split and save
    print("\n--- Splitting (70/20/10) ---")
    splits = split_and_save(all_samples)

    print(f"\n--- Done ---")
    print(f"Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
