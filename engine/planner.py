"""
Mission Planner — Generate flight waypoints for scout and treatment drones.

Responsibilities:
    - Generate lawnmower/zigzag waypoints for scan missions
    - Optimise spray path using nearest-neighbour TSP
    - Convert waypoints to MAVLink mission items

Usage:
    from engine.planner import generate_scan_waypoints, optimize_spray_path

    waypoints = generate_scan_waypoints(scan_polygon, altitude=4, overlap=0.3)
    spray_wp  = optimize_spray_path(spray_zones, home=(lat, lon))
"""

import math
from dataclasses import dataclass

# ── MAVLink command constants (from MAVLink spec) ──
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3

EARTH_RADIUS = 6_371_000


@dataclass
class Waypoint:
    """A single waypoint in a mission."""
    lat: float
    lon: float
    alt: float           # metres AGL
    command: int = MAV_CMD_NAV_WAYPOINT
    param1: float = 0    # hold time (seconds) for waypoint
    param2: float = 0    # acceptance radius (metres)
    param3: float = 0    # pass-through (0 = stop, >0 = pass radius)
    param4: float = 0    # yaw angle
    seq: int = 0         # sequence number in mission
    frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT


# ── GPS Utilities ──

def _metres_per_deg_lat() -> float:
    return (math.pi / 180) * EARTH_RADIUS


def _metres_per_deg_lon(lat: float) -> float:
    return (math.pi / 180) * EARTH_RADIUS * math.cos(math.radians(lat))


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two GPS points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Scan Mission: Lawnmower Pattern ──

def generate_scan_waypoints(
    scan_polygon: list[list[float]],
    altitude: float = 4.0,
    overlap: float = 0.3,
    camera_hfov_deg: float = 62.2,
    camera_vfov_deg: float = 48.8,
    flight_speed: float = 2.0,
    camera_trigger_dist: float | None = None,
) -> list[Waypoint]:
    """Generate lawnmower/zigzag waypoints over a scan area.

    The polygon is treated as an axis-aligned bounding box for simplicity
    in the PoC. The drone sweeps east-west rows, stepping north after each.

    Args:
        scan_polygon: [[lat, lon], ...] polygon vertices.
        altitude: Flight altitude in metres AGL.
        overlap: Fraction of overlap between adjacent rows (0–1).
        camera_hfov_deg: Horizontal field of view.
        camera_vfov_deg: Vertical field of view.
        flight_speed: Cruise speed in m/s.
        camera_trigger_dist: Distance-based camera trigger (metres).
                             If None, auto-calculated from altitude and FOV.

    Returns:
        List of Waypoint objects (includes takeoff and RTL).
    """
    # Compute ground footprint from altitude and FOV
    ground_w = 2 * altitude * math.tan(math.radians(camera_hfov_deg / 2))
    ground_h = 2 * altitude * math.tan(math.radians(camera_vfov_deg / 2))

    # Row spacing (north-south step) with overlap
    row_spacing_m = ground_h * (1 - overlap)

    # Auto camera trigger distance (along-track, with overlap)
    if camera_trigger_dist is None:
        camera_trigger_dist = ground_w * (1 - overlap)

    # Bounding box of the scan polygon
    lats = [p[0] for p in scan_polygon]
    lons = [p[1] for p in scan_polygon]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mean_lat = (min_lat + max_lat) / 2

    # Convert row spacing to degrees
    row_spacing_deg = row_spacing_m / _metres_per_deg_lat()

    # Generate row latitudes
    row_lats = []
    lat = min_lat
    while lat <= max_lat:
        row_lats.append(lat)
        lat += row_spacing_deg

    # Build waypoints
    waypoints: list[Waypoint] = []
    seq = 0

    # Takeoff from first corner
    wp_takeoff = Waypoint(
        lat=min_lat, lon=min_lon, alt=altitude,
        command=MAV_CMD_NAV_TAKEOFF, seq=seq,
    )
    waypoints.append(wp_takeoff)
    seq += 1

    # Camera trigger
    wp_cam = Waypoint(
        lat=min_lat, lon=min_lon, alt=altitude,
        command=MAV_CMD_DO_SET_CAM_TRIGG_DIST, seq=seq,
        param1=round(camera_trigger_dist, 1),
    )
    waypoints.append(wp_cam)
    seq += 1

    # Speed command
    wp_speed = Waypoint(
        lat=min_lat, lon=min_lon, alt=altitude,
        command=MAV_CMD_DO_CHANGE_SPEED, seq=seq,
        param1=1,  # airspeed
        param2=flight_speed,
    )
    waypoints.append(wp_speed)
    seq += 1

    # Lawnmower rows
    for i, row_lat in enumerate(row_lats):
        if i % 2 == 0:
            # Left to right
            wp_start = Waypoint(lat=row_lat, lon=min_lon, alt=altitude, seq=seq)
            seq += 1
            wp_end = Waypoint(lat=row_lat, lon=max_lon, alt=altitude, seq=seq)
            seq += 1
        else:
            # Right to left
            wp_start = Waypoint(lat=row_lat, lon=max_lon, alt=altitude, seq=seq)
            seq += 1
            wp_end = Waypoint(lat=row_lat, lon=min_lon, alt=altitude, seq=seq)
            seq += 1
        waypoints.extend([wp_start, wp_end])

    # Disable camera trigger
    wp_cam_off = Waypoint(
        lat=row_lats[-1] if row_lats else min_lat,
        lon=min_lon, alt=altitude,
        command=MAV_CMD_DO_SET_CAM_TRIGG_DIST, seq=seq,
        param1=0,
    )
    waypoints.append(wp_cam_off)
    seq += 1

    # Return to launch
    wp_rtl = Waypoint(
        lat=min_lat, lon=min_lon, alt=altitude,
        command=MAV_CMD_NAV_RETURN_TO_LAUNCH, seq=seq,
    )
    waypoints.append(wp_rtl)

    return waypoints


# ── Spray Mission: TSP Path Optimiser ──

def optimize_spray_path(
    spray_zones: list[dict],
    home: tuple[float, float] | None = None,
    altitude: float = 3.0,
    hover_time: float = 5.0,
) -> list[Waypoint]:
    """Generate an optimised flight path through spray zone centres.

    Uses nearest-neighbour heuristic for a simple TSP solution.

    Args:
        spray_zones: List of spray zone dicts with 'center_lat', 'center_lon',
                     'severity', 'disease_type' keys.
        home: (lat, lon) of home/launch position. If None, uses first zone.
        altitude: Spray altitude in metres AGL.
        hover_time: Seconds to hover at each spray zone.

    Returns:
        Ordered list of Waypoint objects (takeoff → zones → RTL).
    """
    if not spray_zones:
        return []

    # Extract centres
    centres = [
        (z.get("center_lat", z.get("lat")), z.get("center_lon", z.get("lon")))
        for z in spray_zones
    ]

    if home is None:
        home = centres[0]

    # Nearest-neighbour TSP starting from home
    visited_order = _nearest_neighbour_tsp(home, centres)

    # Build mission
    waypoints: list[Waypoint] = []
    seq = 0

    # Takeoff
    waypoints.append(Waypoint(
        lat=home[0], lon=home[1], alt=altitude,
        command=MAV_CMD_NAV_TAKEOFF, seq=seq,
    ))
    seq += 1

    # Visit spray zones in optimised order
    for idx in visited_order:
        lat, lon = centres[idx]
        waypoints.append(Waypoint(
            lat=lat, lon=lon, alt=altitude,
            command=MAV_CMD_NAV_WAYPOINT, seq=seq,
            param1=hover_time,  # hold time for spraying
        ))
        seq += 1

    # RTL
    waypoints.append(Waypoint(
        lat=home[0], lon=home[1], alt=altitude,
        command=MAV_CMD_NAV_RETURN_TO_LAUNCH, seq=seq,
    ))

    return waypoints


def _nearest_neighbour_tsp(
    home: tuple[float, float],
    points: list[tuple[float, float]],
) -> list[int]:
    """Nearest-neighbour TSP heuristic.

    Returns indices into `points` in visit order.
    """
    if not points:
        return []

    n = len(points)
    visited = [False] * n
    order = []
    current = home

    for _ in range(n):
        best_dist = float("inf")
        best_idx = -1
        for i in range(n):
            if visited[i]:
                continue
            d = _haversine(current[0], current[1], points[i][0], points[i][1])
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx == -1:
            break
        visited[best_idx] = True
        order.append(best_idx)
        current = points[best_idx]

    return order


# ── Mission Statistics ──

def mission_stats(
    waypoints: list[Waypoint],
    flight_speed: float = 2.0,
    altitude: float | None = None,
    camera_hfov_deg: float = 62.2,
    camera_vfov_deg: float = 48.8,
    overlap: float = 0.3,
    hover_time: float = 0.0,
) -> dict:
    """Compute mission statistics from a waypoint list.

    Used by the dashboard to display mission previews.

    Args:
        waypoints: List of Waypoint objects (NAV_WAYPOINT items used for distance).
        flight_speed: Cruise speed in m/s.
        altitude: Override altitude (else taken from waypoints).
        camera_hfov_deg: Horizontal FOV for footprint and image-count math.
        camera_vfov_deg: Vertical FOV.
        overlap: Image overlap fraction (0-1) for image-count estimate.
        hover_time: Per-waypoint hover seconds (added to duration for spray missions).

    Returns:
        Dict with total_distance_m, estimated_duration_s, estimated_duration_str,
        estimated_images, row_count, ground_footprint_m, altitude.
    """
    nav_wps = [w for w in waypoints if w.command == MAV_CMD_NAV_WAYPOINT]

    if altitude is None and nav_wps:
        altitude = nav_wps[0].alt
    altitude = altitude or 0.0

    # Total flight distance through nav waypoints
    total_dist = 0.0
    for i in range(len(nav_wps) - 1):
        a, b = nav_wps[i], nav_wps[i + 1]
        total_dist += _haversine(a.lat, a.lon, b.lat, b.lon)

    flight_time = total_dist / max(flight_speed, 0.1)
    hover_total = hover_time * len(nav_wps)
    duration_s = flight_time + hover_total

    # Camera ground footprint at this altitude
    ground_w = 2 * altitude * math.tan(math.radians(camera_hfov_deg / 2))
    ground_h = 2 * altitude * math.tan(math.radians(camera_vfov_deg / 2))

    # Image count estimate: footprint along-track with overlap
    row_count = max(1, len(nav_wps) // 2)
    if row_count > 0 and ground_w > 0:
        row_length = total_dist / max(row_count, 1)
        imgs_per_row = max(1, int(row_length / max(ground_w * (1 - overlap), 0.1)))
        estimated_images = row_count * imgs_per_row
    else:
        estimated_images = 0

    return {
        "total_distance_m": round(total_dist, 1),
        "estimated_duration_s": round(duration_s, 1),
        "estimated_duration_str": _fmt_duration(duration_s),
        "estimated_images": estimated_images,
        "row_count": row_count,
        "ground_footprint_m": [round(ground_w, 2), round(ground_h, 2)],
        "altitude": round(altitude, 2),
        "waypoint_count": len(waypoints),
        "nav_waypoint_count": len(nav_wps),
    }


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


# ── MAVLink Export ──

def to_mavlink_mission(waypoints: list[Waypoint]) -> list[dict]:
    """Convert waypoints to MAVLink mission item dicts.

    Compatible with pymavlink's MAVLink_mission_item_message format.

    Returns:
        List of dicts, each representing a MAVLink mission item.
    """
    items = []
    for i, wp in enumerate(waypoints):
        items.append({
            "seq": i,
            "frame": wp.frame,
            "command": wp.command,
            "current": 1 if i == 0 else 0,
            "autocontinue": 1,
            "param1": wp.param1,
            "param2": wp.param2,
            "param3": wp.param3,
            "param4": wp.param4,
            "x": wp.lat,
            "y": wp.lon,
            "z": wp.alt,
        })
    return items


def mission_to_qgc_plan(waypoints: list[Waypoint]) -> dict:
    """Export mission as a QGroundControl-compatible JSON plan.

    This can be loaded directly in QGroundControl for simulation or review.
    """
    items = []
    for i, wp in enumerate(waypoints):
        items.append({
            "autoContinue": True,
            "command": wp.command,
            "doJumpId": i + 1,
            "frame": wp.frame,
            "params": [wp.param1, wp.param2, wp.param3, wp.param4,
                       wp.lat, wp.lon, wp.alt],
            "type": "SimpleItem",
        })

    return {
        "fileType": "Plan",
        "version": 1,
        "groundStation": "DiseaseDrone",
        "mission": {
            "cruiseSpeed": 2,
            "hoverSpeed": 1,
            "items": items,
            "plannedHomePosition": {
                "lat": waypoints[0].lat if waypoints else 0,
                "lon": waypoints[0].lon if waypoints else 0,
                "alt": 0,
            },
        },
    }
