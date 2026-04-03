"""
Base Station Orchestrator — Ties all system components together.

End-to-end workflow:
    1. Dashboard sends scan area → planner generates waypoints
    2. Upload waypoints to scout drone
    3. Scout flies, images arrive → ingest → inference → decision engine
    4. Spray zones appear on dashboard for operator approval
    5. Operator approves → planner generates spray path
    6. Upload to treatment drone → fly and spray
    7. Log everything

Usage:
    from engine.base_station import BaseStation

    bs = BaseStation()
    bs.start_scan_mission(scan_polygon)
    bs.process_scout_images("path/to/images/")
    bs.start_spray_mission(approved_zone_ids)
"""

import json
import logging
from pathlib import Path

from dashboard import database as db
from engine.decision import process_detections
from engine.ingest import process_folder, watch_folder
from engine.planner import (
    generate_scan_waypoints,
    optimize_spray_path,
    to_mavlink_mission,
    mission_to_qgc_plan,
)

logger = logging.getLogger(__name__)


class BaseStation:
    """Central orchestrator for the Disease Drone system.

    Manages the lifecycle of scan and spray missions, coordinating
    between the ML pipeline, decision engine, mission planner,
    dashboard database, and drone communication.
    """

    def __init__(
        self,
        scout_connection: str | None = None,
        treatment_connection: str | None = None,
        default_altitude: float = 4.0,
        default_overlap: float = 0.3,
        conf_threshold: float = 0.4,
        cluster_eps_m: float = 2.0,
        cluster_min_samples: int = 2,
        spray_buffer_m: float = 1.0,
    ):
        """
        Args:
            scout_connection: MAVLink URI for the scout drone (optional).
            treatment_connection: MAVLink URI for the treatment drone (optional).
            default_altitude: Default flight altitude in metres.
            default_overlap: Image overlap fraction for scan patterns.
            conf_threshold: Minimum detection confidence.
            cluster_eps_m: DBSCAN neighbourhood radius in metres.
            cluster_min_samples: DBSCAN minimum cluster size.
            spray_buffer_m: Buffer around spray zone convex hulls (metres).
        """
        self.scout_conn_str = scout_connection
        self.treatment_conn_str = treatment_connection
        self.default_altitude = default_altitude
        self.default_overlap = default_overlap
        self.conf_threshold = conf_threshold
        self.cluster_eps_m = cluster_eps_m
        self.cluster_min_samples = cluster_min_samples
        self.spray_buffer_m = spray_buffer_m

        self._scout = None
        self._treatment = None

    # ── Drone Connections ──

    def connect_scout(self, connection_string: str | None = None):
        """Connect to the scout drone."""
        from drone.comms import DroneLink

        conn_str = connection_string or self.scout_conn_str
        if not conn_str:
            raise ValueError("No scout drone connection string provided")

        self._scout = DroneLink(conn_str)
        self._scout.connect()
        self._scout.request_data_stream(rate_hz=4)
        logger.info("Scout drone connected")

    def connect_treatment(self, connection_string: str | None = None):
        """Connect to the treatment drone."""
        from drone.comms import DroneLink

        conn_str = connection_string or self.treatment_conn_str
        if not conn_str:
            raise ValueError("No treatment drone connection string provided")

        self._treatment = DroneLink(conn_str)
        self._treatment.connect()
        self._treatment.request_data_stream(rate_hz=4)
        logger.info("Treatment drone connected")

    # ── Phase 1: Scan Mission ──

    def plan_scan_mission(
        self,
        scan_polygon: list[list[float]],
        altitude: float | None = None,
        overlap: float | None = None,
    ) -> tuple[int, list]:
        """Plan a scan mission and save to database.

        Args:
            scan_polygon: [[lat, lon], ...] defining the scan area.
            altitude: Flight altitude (uses default if None).
            overlap: Image overlap fraction (uses default if None).

        Returns:
            (mission_id, waypoints) tuple.
        """
        alt = altitude or self.default_altitude
        ovl = overlap or self.default_overlap

        # Generate waypoints
        waypoints = generate_scan_waypoints(
            scan_polygon, altitude=alt, overlap=ovl,
        )

        # Create mission in DB
        mission_id = db.create_mission(
            mission_type="scan",
            scan_area=scan_polygon,
            notes=f"Scan at {alt}m, {ovl*100:.0f}% overlap, {len(waypoints)} waypoints",
        )

        # Store waypoints in mission record
        mavlink_items = to_mavlink_mission(waypoints)
        conn = db.get_db()
        conn.execute(
            "UPDATE missions SET waypoints=? WHERE id=?",
            (json.dumps(mavlink_items), mission_id),
        )
        conn.commit()
        conn.close()

        logger.info(
            "Scan mission %d planned: %d waypoints over %d-vertex area",
            mission_id, len(waypoints), len(scan_polygon),
        )
        return mission_id, waypoints

    def start_scan_mission(
        self,
        scan_polygon: list[list[float]],
        altitude: float | None = None,
        overlap: float | None = None,
    ) -> int:
        """Plan and upload a scan mission to the scout drone.

        Args:
            scan_polygon: Scan area polygon.
            altitude: Flight altitude.
            overlap: Image overlap.

        Returns:
            mission_id.
        """
        mission_id, waypoints = self.plan_scan_mission(
            scan_polygon, altitude, overlap,
        )

        if self._scout is None:
            logger.warning(
                "Scout drone not connected — mission %d planned but not uploaded",
                mission_id,
            )
            return mission_id

        mavlink_items = to_mavlink_mission(waypoints)
        self._scout.upload_mission(mavlink_items)
        db.update_mission_status(mission_id, "in_progress")
        self._scout.arm_and_start_mission()

        logger.info("Scan mission %d started", mission_id)
        return mission_id

    def export_scan_plan(
        self,
        scan_polygon: list[list[float]],
        output_path: str | Path,
        altitude: float | None = None,
        overlap: float | None = None,
    ) -> str:
        """Export a scan mission as a QGroundControl plan file.

        Useful for loading in QGC for simulation without a live drone.

        Returns:
            Path to the saved .plan file.
        """
        alt = altitude or self.default_altitude
        ovl = overlap or self.default_overlap

        waypoints = generate_scan_waypoints(
            scan_polygon, altitude=alt, overlap=ovl,
        )
        plan = mission_to_qgc_plan(waypoints)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2)

        logger.info("QGC plan exported to %s", output_path)
        return str(output_path)

    # ── Phase 2: Image Processing ──

    def process_scout_images(
        self,
        image_folder: str | Path,
        mission_id: int,
    ) -> list:
        """Process all scout drone images and generate spray zones.

        Steps:
            1. Run ingestion pipeline on all images in the folder
            2. Run decision engine to cluster detections into spray zones
            3. Update mission status

        Args:
            image_folder: Path to folder containing scout images.
            mission_id: The scan mission ID.

        Returns:
            List of SprayZone objects.
        """
        # Ingest all images → geo-located detections
        detections = process_folder(
            folder_path=image_folder,
            mission_id=mission_id,
            altitude_m=self.default_altitude,
            conf_threshold=self.conf_threshold,
            db_module=db,
        )

        if not detections:
            logger.warning("No detections from scout images")
            db.update_mission_status(mission_id, "completed")
            return []

        # Cluster into spray zones
        zones = process_detections(
            detections=detections,
            mission_id=mission_id,
            eps_metres=self.cluster_eps_m,
            min_samples=self.cluster_min_samples,
            buffer_metres=self.spray_buffer_m,
            confidence_threshold=self.conf_threshold,
            db_module=db,
        )

        # Update health map
        for det in detections:
            health_score = 1.0 if det.class_name == "healthy" else max(0.1, 1.0 - det.confidence)
            db.add_health_point(
                det.lat, det.lon, round(health_score, 3),
                det.class_name if det.class_name != "healthy" else None,
            )

        db.update_mission_status(mission_id, "completed")
        logger.info(
            "Scout processing complete: %d detections → %d spray zones",
            len(detections), len(zones),
        )
        return zones

    def watch_scout_images(
        self,
        image_folder: str | Path,
        mission_id: int,
    ):
        """Watch for incoming scout images in real-time (blocking).

        Images are processed as they land in the folder.
        Press Ctrl+C to stop.
        """
        logger.info("Starting real-time image watcher for mission %d", mission_id)
        db.update_mission_status(mission_id, "in_progress")

        watch_folder(
            folder_path=image_folder,
            mission_id=mission_id,
            altitude_m=self.default_altitude,
            conf_threshold=self.conf_threshold,
            db_module=db,
        )

    # ── Phase 3: Spray Mission ──

    def plan_spray_mission(
        self,
        zone_ids: list[int] | None = None,
        scan_mission_id: int | None = None,
        home: tuple[float, float] | None = None,
        altitude: float = 3.0,
    ) -> tuple[int, list]:
        """Plan a spray mission from approved zones.

        Args:
            zone_ids: Specific spray zone IDs to target. If None,
                      uses all 'approved' zones from scan_mission_id.
            scan_mission_id: Scan mission to pull approved zones from.
            home: (lat, lon) home position. If None, uses first zone centre.
            altitude: Spray altitude.

        Returns:
            (mission_id, waypoints) tuple.
        """
        # Fetch approved zones
        if zone_ids:
            all_zones = db.get_spray_zones()
            zones = [z for z in all_zones if z["id"] in zone_ids]
        elif scan_mission_id:
            zones = db.get_spray_zones(
                mission_id=scan_mission_id, status="approved",
            )
        else:
            zones = db.get_spray_zones(status="approved")

        if not zones:
            raise ValueError("No approved spray zones found")

        # Generate optimised path
        waypoints = optimize_spray_path(
            spray_zones=zones, home=home, altitude=altitude,
        )

        # Create spray mission in DB
        mission_id = db.create_mission(
            mission_type="spray",
            notes=f"Spray mission: {len(zones)} zones, {len(waypoints)} waypoints",
        )

        mavlink_items = to_mavlink_mission(waypoints)
        conn = db.get_db()
        conn.execute(
            "UPDATE missions SET waypoints=? WHERE id=?",
            (json.dumps(mavlink_items), mission_id),
        )
        conn.commit()
        conn.close()

        logger.info(
            "Spray mission %d planned: %d zones, %d waypoints",
            mission_id, len(zones), len(waypoints),
        )
        return mission_id, waypoints

    def start_spray_mission(
        self,
        zone_ids: list[int] | None = None,
        scan_mission_id: int | None = None,
        home: tuple[float, float] | None = None,
        altitude: float = 3.0,
    ) -> int:
        """Plan and upload a spray mission to the treatment drone.

        Returns:
            mission_id.
        """
        mission_id, waypoints = self.plan_spray_mission(
            zone_ids=zone_ids,
            scan_mission_id=scan_mission_id,
            home=home,
            altitude=altitude,
        )

        if self._treatment is None:
            logger.warning(
                "Treatment drone not connected — mission %d planned but not uploaded",
                mission_id,
            )
            return mission_id

        mavlink_items = to_mavlink_mission(waypoints)
        self._treatment.upload_mission(mavlink_items)
        db.update_mission_status(mission_id, "in_progress")
        self._treatment.arm_and_start_mission()

        logger.info("Spray mission %d started", mission_id)
        return mission_id

    def export_spray_plan(
        self,
        output_path: str | Path,
        zone_ids: list[int] | None = None,
        scan_mission_id: int | None = None,
        home: tuple[float, float] | None = None,
        altitude: float = 3.0,
    ) -> str:
        """Export a spray mission as a QGroundControl plan file."""
        _, waypoints = self.plan_spray_mission(
            zone_ids=zone_ids,
            scan_mission_id=scan_mission_id,
            home=home,
            altitude=altitude,
        )
        plan = mission_to_qgc_plan(waypoints)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2)

        logger.info("QGC spray plan exported to %s", output_path)
        return str(output_path)

    # ── Status ──

    def get_scout_telemetry(self) -> dict | None:
        """Get latest scout drone telemetry."""
        if self._scout is None:
            return None
        from dataclasses import asdict
        return asdict(self._scout.get_telemetry())

    def get_treatment_telemetry(self) -> dict | None:
        """Get latest treatment drone telemetry."""
        if self._treatment is None:
            return None
        from dataclasses import asdict
        return asdict(self._treatment.get_telemetry())

    def disconnect_all(self):
        """Disconnect from all drones."""
        if self._scout:
            self._scout.close()
            self._scout = None
        if self._treatment:
            self._treatment.close()
            self._treatment = None
        logger.info("All drones disconnected")
