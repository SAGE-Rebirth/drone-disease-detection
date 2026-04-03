"""
Image Ingestion Pipeline — Process drone images into detections.

Responsibilities:
    - Extract GPS coordinates from EXIF metadata
    - Run ML inference on each image
    - Convert pixel detections to GPS coordinates
    - Feed geo-located detections to the decision engine
    - Push detections to the dashboard database

Usage:
    from engine.ingest import process_image, process_folder

    detections = process_image("path/to/image.jpg", mission_id=1)
    all_dets   = process_folder("path/to/images/", mission_id=1)
"""

import logging
import time
from pathlib import Path

import exifread

from engine.decision import Detection, pixel_to_gps

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ── EXIF GPS Extraction ──

def _convert_dms_to_decimal(dms_tag, ref_tag) -> float | None:
    """Convert EXIF DMS (degrees/minutes/seconds) to decimal degrees."""
    if dms_tag is None:
        return None

    values = dms_tag.values
    d = float(values[0].num) / float(values[0].den)
    m = float(values[1].num) / float(values[1].den)
    s = float(values[2].num) / float(values[2].den)

    decimal = d + m / 60 + s / 3600

    if ref_tag and str(ref_tag) in ("S", "W"):
        decimal = -decimal

    return decimal


def extract_gps_from_exif(image_path: str | Path) -> dict | None:
    """Extract GPS coordinates and altitude from image EXIF data.

    Args:
        image_path: Path to a JPEG/TIFF image with EXIF GPS tags.

    Returns:
        Dict with 'lat', 'lon', and optionally 'altitude' keys.
        None if no GPS data is found.
    """
    image_path = Path(image_path)
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception as e:
        logger.warning("Failed to read EXIF from %s: %s", image_path, e)
        return None

    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")

    if lat_tag is None or lon_tag is None:
        return None

    lat = _convert_dms_to_decimal(lat_tag, lat_ref)
    lon = _convert_dms_to_decimal(lon_tag, lon_ref)

    if lat is None or lon is None:
        return None

    result = {"lat": lat, "lon": lon}

    alt_tag = tags.get("GPS GPSAltitude")
    if alt_tag:
        alt_val = alt_tag.values[0]
        result["altitude"] = float(alt_val.num) / float(alt_val.den)

    return result


# ── Single Image Processing ──

def process_image(
    image_path: str | Path,
    mission_id: int,
    altitude_m: float = 4.0,
    camera_hfov_deg: float = 62.2,
    camera_vfov_deg: float = 48.8,
    conf_threshold: float = 0.4,
    inference_fn=None,
    db_module=None,
) -> list[Detection]:
    """Process a single drone image through the ingestion pipeline.

    Steps:
        1. Extract GPS from EXIF
        2. Run ML inference to get bounding-box detections
        3. Convert each detection's pixel centre to GPS
        4. Persist detections to the dashboard DB

    Args:
        image_path: Path to the image file.
        mission_id: Mission ID for DB association.
        altitude_m: Drone altitude (used if EXIF altitude is missing).
        camera_hfov_deg: Horizontal field of view.
        camera_vfov_deg: Vertical field of view.
        conf_threshold: Minimum confidence threshold.
        inference_fn: Callable that takes an image path and returns a list
                      of detection dicts with 'bbox', 'class_name',
                      'confidence' keys. If None, imports from ml.inference.
        db_module: Optional dashboard.database module for persistence.

    Returns:
        List of geo-located Detection objects.
    """
    image_path = Path(image_path)

    # 1. Extract GPS
    gps = extract_gps_from_exif(image_path)
    if gps is None:
        logger.warning("No GPS data in %s — skipping", image_path.name)
        return []

    image_lat = gps["lat"]
    image_lon = gps["lon"]
    alt = gps.get("altitude", altitude_m)

    # 2. Run ML inference
    if inference_fn is None:
        from ml.inference.detect import detect_diseases
        raw_detections = detect_diseases(
            str(image_path), conf_threshold=conf_threshold,
        )
    else:
        raw_detections = inference_fn(str(image_path))

    if not raw_detections:
        return []

    # We need image dimensions for pixel → GPS conversion.
    # Read from the first detection's context or load the image header.
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception:
        img_w, img_h = 640, 640  # fallback to model input size

    # 3. Convert pixel detections to GPS
    geo_detections: list[Detection] = []
    for det in raw_detections:
        # det can be a DetectionResult dataclass or a dict
        if hasattr(det, "bbox"):
            bbox = det.bbox
            class_name = det.class_name
            confidence = det.confidence
        else:
            bbox = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]

        # Bounding box centre in pixels
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        det_lat, det_lon = pixel_to_gps(
            bbox_center_px=(cx, cy),
            image_width=img_w,
            image_height=img_h,
            image_lat=image_lat,
            image_lon=image_lon,
            altitude_m=alt,
            camera_hfov_deg=camera_hfov_deg,
            camera_vfov_deg=camera_vfov_deg,
        )

        geo_det = Detection(
            lat=det_lat,
            lon=det_lon,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            image_path=str(image_path),
        )
        geo_detections.append(geo_det)

        # 4. Persist to DB
        if db_module is not None:
            db_module.add_detection(
                mission_id=mission_id,
                class_name=class_name,
                confidence=confidence,
                lat=det_lat,
                lon=det_lon,
                bbox=bbox,
                image_path=str(image_path),
            )

    logger.info(
        "Processed %s: %d detections", image_path.name, len(geo_detections),
    )
    return geo_detections


# ── Batch / Folder Processing ──

def process_folder(
    folder_path: str | Path,
    mission_id: int,
    altitude_m: float = 4.0,
    conf_threshold: float = 0.4,
    inference_fn=None,
    db_module=None,
) -> list[Detection]:
    """Process all images in a folder.

    Args:
        folder_path: Directory containing drone images.
        mission_id: Mission ID for DB association.
        altitude_m: Default altitude if EXIF is missing.
        conf_threshold: Minimum detection confidence.
        inference_fn: Custom inference function (optional).
        db_module: Dashboard database module (optional).

    Returns:
        Combined list of all geo-located detections.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("Folder not found: %s", folder)
        return []

    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info("Found %d images in %s", len(images), folder)

    all_detections: list[Detection] = []
    for img_path in images:
        dets = process_image(
            img_path,
            mission_id=mission_id,
            altitude_m=altitude_m,
            conf_threshold=conf_threshold,
            inference_fn=inference_fn,
            db_module=db_module,
        )
        all_detections.extend(dets)

    logger.info(
        "Folder complete: %d images, %d total detections",
        len(images), len(all_detections),
    )
    return all_detections


# ── Folder Watcher (simple polling) ──

def watch_folder(
    folder_path: str | Path,
    mission_id: int,
    poll_interval: float = 2.0,
    altitude_m: float = 4.0,
    conf_threshold: float = 0.4,
    inference_fn=None,
    db_module=None,
    on_detection=None,
):
    """Watch a folder for new images and process them as they arrive.

    This is a simple polling-based watcher suitable for the PoC.
    For production, consider watchdog or inotify.

    Args:
        folder_path: Directory to watch.
        mission_id: Mission ID.
        poll_interval: Seconds between polls.
        altitude_m: Default altitude.
        conf_threshold: Min confidence.
        inference_fn: Custom inference function.
        db_module: Dashboard database module.
        on_detection: Optional callback(detections) called after each image.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    processed: set[str] = set()
    logger.info("Watching %s for new images (poll every %.1fs)...", folder, poll_interval)

    try:
        while True:
            current_files = {
                str(p) for p in folder.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            }
            new_files = current_files - processed

            for fpath in sorted(new_files):
                logger.info("New image detected: %s", Path(fpath).name)
                dets = process_image(
                    fpath,
                    mission_id=mission_id,
                    altitude_m=altitude_m,
                    conf_threshold=conf_threshold,
                    inference_fn=inference_fn,
                    db_module=db_module,
                )
                processed.add(fpath)

                if on_detection and dets:
                    on_detection(dets)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Watcher stopped. Processed %d images.", len(processed))
