"""
Download PlantVillage and PlantDoc datasets for disease detection training.

PlantVillage: Classification dataset (~54K images, 38 classes)
    - Source: Kaggle (abdallahalidev/plantvillage-dataset)
PlantDoc: Detection dataset (~2,500 images, 27 classes, bbox annotations)
    - Source: GitHub (pratikkayal/PlantDoc-Dataset)

Usage:
    python ml/data/scripts/download_data.py
"""

import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path

# Resolve paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "raw"

PLANTVILLAGE_DIR = RAW_DIR / "plantvillage"
PLANTDOC_DIR = RAW_DIR / "plantdoc"


def download_plantvillage():
    """
    Download PlantVillage dataset from Kaggle.
    Requires: kaggle CLI configured with API key (~/.kaggle/kaggle.json)
    """
    if PLANTVILLAGE_DIR.exists() and any(PLANTVILLAGE_DIR.iterdir()):
        print(f"[PlantVillage] Already exists at {PLANTVILLAGE_DIR}, skipping.")
        return

    PLANTVILLAGE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "plantvillage.zip"

    print("[PlantVillage] Downloading from Kaggle...")
    print("  NOTE: Requires 'kaggle' CLI with API key configured.")
    print("  Run: pip install kaggle")
    print("  Place kaggle.json in ~/.kaggle/kaggle.json")
    print()

    ret = os.system(
        f"kaggle datasets download -d abdallahalidev/plantvillage-dataset "
        f"-p {RAW_DIR} --unzip"
    )
    if ret != 0:
        print("[PlantVillage] Kaggle download failed.")
        print("  Alternative: manually download from:")
        print("  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print(f"  Extract to: {PLANTVILLAGE_DIR}")
        return

    # Move extracted contents into plantvillage dir
    extracted = RAW_DIR / "plantvillage dataset"
    if extracted.exists():
        for item in extracted.iterdir():
            shutil.move(str(item), str(PLANTVILLAGE_DIR / item.name))
        extracted.rmdir()

    print(f"[PlantVillage] Downloaded to {PLANTVILLAGE_DIR}")


def download_plantdoc():
    """
    Download PlantDoc dataset from GitHub.
    """
    if PLANTDOC_DIR.exists() and any(PLANTDOC_DIR.iterdir()):
        print(f"[PlantDoc] Already exists at {PLANTDOC_DIR}, skipping.")
        return

    PLANTDOC_DIR.mkdir(parents=True, exist_ok=True)

    print("[PlantDoc] Downloading from GitHub...")

    zip_url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = RAW_DIR / "plantdoc.zip"

    try:
        import urllib.request
        urllib.request.urlretrieve(zip_url, str(zip_path))
    except Exception as e:
        print(f"[PlantDoc] Download failed: {e}")
        print("  Alternative: manually download from:")
        print("  https://github.com/pratikkayal/PlantDoc-Dataset")
        print(f"  Extract to: {PLANTDOC_DIR}")
        return

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(RAW_DIR))

    # Move contents from extracted folder
    extracted = RAW_DIR / "PlantDoc-Dataset-master"
    if extracted.exists():
        for item in extracted.iterdir():
            shutil.move(str(item), str(PLANTDOC_DIR / item.name))
        extracted.rmdir()

    zip_path.unlink(missing_ok=True)
    print(f"[PlantDoc] Downloaded to {PLANTDOC_DIR}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Download directory: {RAW_DIR}\n")

    download_plantvillage()
    print()
    download_plantdoc()

    print("\n--- Done ---")
    print(f"PlantVillage: {PLANTVILLAGE_DIR}")
    print(f"PlantDoc:     {PLANTDOC_DIR}")


if __name__ == "__main__":
    main()
