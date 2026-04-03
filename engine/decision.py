"""
Decision Engine — Cluster detections into spray zones.

Responsibilities:
    - Convert pixel bounding-box centres to GPS coordinates
    - Cluster nearby detections with DBSCAN
    - Generate convex-hull spray zones with a safety buffer
    - Score severity per cluster
    - Push results to the dashboard database

Usage:
    from engine.decision import process_detections

    spray_zones = process_detections(detections, mission_id=1)
"""

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint

# Earth radius in metres (WGS-84 mean)
EARTH_RADIUS = 6_371_000

# Severity weights per disease type (higher → more urgent)
DISEASE_SEVERITY_WEIGHTS = {
    "leaf_blight": 0.9,
    "rust": 0.85,
    "powdery_mildew": 0.75,
    "leaf_spot": 0.6,
    "healthy": 0.0,
}


@dataclass
class Detection:
    """A single geo-located detection."""
    lat: float
    lon: float
    class_name: str
    confidence: float
    bbox: list | None = None
    image_path: str | None = None


@dataclass
class SprayZone:
    """A spray zone derived from a cluster of detections."""
    geometry: list            # [[lat, lon], ...] polygon vertices
    center_lat: float
    center_lon: float
    severity: float
    disease_type: str
    detection_count: int
    detections: list = field(default_factory=list)


# ── GPS Math ──

def _metres_per_degree_lat(lat: float) -> float:
    """Metres per degree of latitude at the given latitude."""
    return (math.pi / 180) * EARTH_RADIUS


def _metres_per_degree_lon(lat: float) -> float:
    """Metres per degree of longitude at the given latitude."""
    return (math.pi / 180) * EARTH_RADIUS * math.cos(math.radians(lat))


def pixel_to_gps(
    bbox_center_px: tuple[float, float],
    image_width: int,
    image_height: int,
    image_lat: float,
    image_lon: float,
    altitude_m: float,
    camera_hfov_deg: float = 62.2,
    camera_vfov_deg: float = 48.8,
) -> tuple[float, float]:
    """Convert a pixel coordinate to a GPS coordinate.

    Assumes the camera is nadir (pointing straight down) and the image
    centre corresponds to (image_lat, image_lon).

    Args:
        bbox_center_px: (x, y) pixel coordinates of detection centre.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        image_lat: GPS latitude of image centre.
        image_lon: GPS longitude of image centre.
        altitude_m: Drone altitude in metres AGL.
        camera_hfov_deg: Horizontal field of view in degrees.
        camera_vfov_deg: Vertical field of view in degrees.

    Returns:
        (lat, lon) of the detection.
    """
    # Ground footprint covered by the image (metres)
    ground_w = 2 * altitude_m * math.tan(math.radians(camera_hfov_deg / 2))
    ground_h = 2 * altitude_m * math.tan(math.radians(camera_vfov_deg / 2))

    # Pixel offset from image centre (in metres)
    cx, cy = bbox_center_px
    dx_m = (cx - image_width / 2) / image_width * ground_w
    dy_m = (image_height / 2 - cy) / image_height * ground_h  # y-axis inverted

    # Convert metre offset to lat/lon offset
    dlat = dy_m / _metres_per_degree_lat(image_lat)
    dlon = dx_m / _metres_per_degree_lon(image_lat)

    return (image_lat + dlat, image_lon + dlon)


# ── Clustering ──

def cluster_detections(
    detections: list[Detection],
    eps_metres: float = 2.0,
    min_samples: int = 2,
    confidence_threshold: float = 0.4,
) -> list[list[Detection]]:
    """Cluster detections using DBSCAN on GPS coordinates.

    Args:
        detections: List of Detection objects.
        eps_metres: Maximum distance (metres) between two detections
                    in the same cluster.
        min_samples: Minimum detections to form a cluster.
        confidence_threshold: Drop detections below this confidence.

    Returns:
        List of clusters, where each cluster is a list of Detection objects.
        Noise points (label == -1) are discarded.
    """
    # Filter out healthy and low-confidence
    filtered = [
        d for d in detections
        if d.class_name != "healthy" and d.confidence >= confidence_threshold
    ]
    if len(filtered) < min_samples:
        return []

    # Build coordinate matrix
    coords = np.array([[d.lat, d.lon] for d in filtered])

    # Convert eps from metres to approximate degrees (use mean lat)
    mean_lat = coords[:, 0].mean()
    eps_deg = eps_metres / _metres_per_degree_lat(mean_lat)

    # DBSCAN in lat/lon space (haversine would be more precise for large
    # areas, but at field scale the flat-earth approximation is fine)
    db = DBSCAN(eps=eps_deg, min_samples=min_samples)
    labels = db.fit_predict(coords)

    clusters: dict[int, list[Detection]] = {}
    for label, det in zip(labels, filtered):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(det)

    return list(clusters.values())


# ── Spray Zone Generation ──

def generate_spray_zones(
    clusters: list[list[Detection]],
    buffer_metres: float = 1.0,
) -> list[SprayZone]:
    """Convert detection clusters into spray zones.

    Each zone is the convex hull of the cluster's GPS points
    expanded by `buffer_metres`.

    Args:
        clusters: Output of cluster_detections().
        buffer_metres: Safety margin around the convex hull.

    Returns:
        List of SprayZone objects.
    """
    zones = []
    for cluster in clusters:
        points = [(d.lon, d.lat) for d in cluster]  # Shapely uses (x, y) = (lon, lat)
        mp = MultiPoint(points)

        if len(points) < 3:
            hull = mp.convex_hull.buffer(
                buffer_metres / _metres_per_degree_lon(cluster[0].lat)
            )
        else:
            hull = mp.convex_hull.buffer(
                buffer_metres / _metres_per_degree_lon(cluster[0].lat)
            )

        # Extract polygon vertices as [[lat, lon], ...]
        coords = list(hull.exterior.coords)
        geometry = [[round(lat, 7), round(lon, 7)] for lon, lat in coords]

        center_lon, center_lat = hull.centroid.x, hull.centroid.y
        severity = score_severity(cluster)
        disease_type = _dominant_disease(cluster)

        zones.append(SprayZone(
            geometry=geometry,
            center_lat=round(center_lat, 7),
            center_lon=round(center_lon, 7),
            severity=round(severity, 3),
            disease_type=disease_type,
            detection_count=len(cluster),
            detections=cluster,
        ))

    return zones


def score_severity(cluster: list[Detection]) -> float:
    """Score a cluster's severity (0–1).

    Factors: disease type weight, mean confidence, detection count.
    """
    if not cluster:
        return 0.0

    weights = [DISEASE_SEVERITY_WEIGHTS.get(d.class_name, 0.5) for d in cluster]
    confs = [d.confidence for d in cluster]

    # Weighted combination
    type_score = sum(weights) / len(weights)
    conf_score = sum(confs) / len(confs)
    count_score = min(len(cluster) / 10.0, 1.0)  # caps at 10 detections

    severity = 0.4 * type_score + 0.4 * conf_score + 0.2 * count_score
    return min(severity, 1.0)


def _dominant_disease(cluster: list[Detection]) -> str:
    """Return the most frequent disease type in the cluster."""
    counts: dict[str, int] = {}
    for d in cluster:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    return max(counts, key=counts.get)


# ── End-to-End ──

def process_detections(
    detections: list[Detection],
    mission_id: int,
    eps_metres: float = 2.0,
    min_samples: int = 2,
    buffer_metres: float = 1.0,
    confidence_threshold: float = 0.4,
    db_module=None,
) -> list[SprayZone]:
    """Full pipeline: filter → cluster → spray zones → persist to DB.

    Args:
        detections: Raw detections (with GPS already resolved).
        mission_id: Mission ID to associate zones with.
        eps_metres: DBSCAN neighbourhood radius.
        min_samples: DBSCAN min cluster size.
        buffer_metres: Convex-hull buffer.
        confidence_threshold: Min confidence to keep.
        db_module: Optional dashboard.database module for persistence.
                   If None, zones are returned without saving.

    Returns:
        List of generated SprayZone objects.
    """
    clusters = cluster_detections(
        detections,
        eps_metres=eps_metres,
        min_samples=min_samples,
        confidence_threshold=confidence_threshold,
    )
    zones = generate_spray_zones(clusters, buffer_metres=buffer_metres)

    if db_module is not None:
        for zone in zones:
            db_module.add_spray_zone(
                mission_id=mission_id,
                geometry=zone.geometry,
                center_lat=zone.center_lat,
                center_lon=zone.center_lon,
                severity=zone.severity,
                disease_type=zone.disease_type,
            )

    return zones
