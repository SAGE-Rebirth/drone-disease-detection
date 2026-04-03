"""
Drone Communication — MAVLink interface for scout and treatment drones.

Responsibilities:
    - Connect to a drone (or SITL simulator) via MAVLink
    - Upload mission waypoints
    - Monitor telemetry (GPS, battery, status)
    - Trigger camera capture or spray pump via servo/relay commands
    - Arm/disarm and mode changes

Usage:
    from drone.comms import DroneLink

    drone = DroneLink("udp:127.0.0.1:14550")
    drone.connect()
    drone.upload_mission(waypoints)
    drone.arm_and_start_mission()
    telemetry = drone.get_telemetry()
    drone.close()
"""

import logging
import time
from dataclasses import dataclass

from pymavlink import mavutil

logger = logging.getLogger(__name__)

# ArduPilot flight modes (copter)
MODE_STABILIZE = 0
MODE_GUIDED = 4
MODE_AUTO = 3
MODE_RTL = 6
MODE_LAND = 9


@dataclass
class Telemetry:
    """Snapshot of drone telemetry."""
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0          # metres AGL (relative)
    heading: float = 0.0       # degrees
    groundspeed: float = 0.0   # m/s
    battery_voltage: float = 0.0
    battery_remaining: int = -1  # percent, -1 = unknown
    armed: bool = False
    mode: str = "UNKNOWN"
    gps_fix: int = 0           # 0=no fix, 2=2D, 3=3D


class DroneLink:
    """MAVLink connection to a drone or SITL simulator.

    Supports common operations: mission upload, telemetry, arm/disarm,
    camera trigger, and spray pump relay control.
    """

    def __init__(self, connection_string: str, baud: int = 57600):
        """
        Args:
            connection_string: MAVLink connection URI.
                Examples:
                    "udp:127.0.0.1:14550"      (SITL / simulator)
                    "/dev/ttyUSB0"              (serial telemetry radio)
                    "tcp:192.168.1.10:5760"     (WiFi)
            baud: Serial baud rate (ignored for UDP/TCP).
        """
        self.connection_string = connection_string
        self.baud = baud
        self.conn = None
        self._target_system = 1
        self._target_component = 1

    def connect(self, timeout: float = 30.0):
        """Establish MAVLink connection and wait for heartbeat.

        Args:
            timeout: Max seconds to wait for first heartbeat.

        Raises:
            TimeoutError: If no heartbeat received within timeout.
        """
        logger.info("Connecting to %s ...", self.connection_string)
        self.conn = mavutil.mavlink_connection(
            self.connection_string, baud=self.baud,
        )

        hb = self.conn.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise TimeoutError(
                f"No heartbeat from {self.connection_string} "
                f"within {timeout}s"
            )

        self._target_system = self.conn.target_system
        self._target_component = self.conn.target_component
        logger.info(
            "Connected — system %d, component %d",
            self._target_system, self._target_component,
        )

    def close(self):
        """Close the MAVLink connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Connection closed")

    # ── Telemetry ──

    def get_telemetry(self) -> Telemetry:
        """Read latest telemetry from cached MAVLink messages.

        This reads from the message buffer — it does not block.
        Call in a loop or after recv_match() for fresh data.
        """
        t = Telemetry()

        # Drain message buffer to get latest
        while self.conn.recv_match(blocking=False):
            pass

        gps = self.conn.messages.get("GLOBAL_POSITION_INT")
        if gps:
            t.lat = gps.lat / 1e7
            t.lon = gps.lon / 1e7
            t.alt = gps.relative_alt / 1000.0
            t.heading = gps.hdg / 100.0 if gps.hdg != 65535 else 0

        vfr = self.conn.messages.get("VFR_HUD")
        if vfr:
            t.groundspeed = vfr.groundspeed

        bat = self.conn.messages.get("SYS_STATUS")
        if bat:
            t.battery_voltage = bat.voltage_battery / 1000.0
            t.battery_remaining = bat.battery_remaining

        hb = self.conn.messages.get("HEARTBEAT")
        if hb:
            t.armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            mode_map = self.conn.mode_mapping()
            if mode_map:
                rev_map = {v: k for k, v in mode_map.items()}
                t.mode = rev_map.get(hb.custom_mode, f"MODE_{hb.custom_mode}")

        gps2 = self.conn.messages.get("GPS_RAW_INT")
        if gps2:
            t.gps_fix = gps2.fix_type

        return t

    def wait_for_telemetry(self, timeout: float = 5.0) -> Telemetry:
        """Block until fresh telemetry is available."""
        self.conn.recv_match(
            type=["GLOBAL_POSITION_INT", "HEARTBEAT"],
            blocking=True, timeout=timeout,
        )
        return self.get_telemetry()

    # ── Mission Upload ──

    def upload_mission(self, mission_items: list[dict]):
        """Upload a mission to the drone's flight controller.

        Args:
            mission_items: List of MAVLink mission item dicts
                           (from planner.to_mavlink_mission()).
        """
        n = len(mission_items)
        logger.info("Uploading %d mission items...", n)

        # Send mission count
        self.conn.mav.mission_count_send(
            self._target_system, self._target_component, n,
        )

        # Wait for each MISSION_REQUEST and send the corresponding item
        for i in range(n):
            msg = self.conn.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                blocking=True, timeout=10,
            )
            if msg is None:
                raise TimeoutError(f"No MISSION_REQUEST for item {i}")

            item = mission_items[msg.seq]
            self.conn.mav.mission_item_send(
                self._target_system,
                self._target_component,
                item["seq"],
                item["frame"],
                item["command"],
                item["current"],
                item["autocontinue"],
                item["param1"],
                item["param2"],
                item["param3"],
                item["param4"],
                item["x"],   # lat
                item["y"],   # lon
                item["z"],   # alt
            )

        # Wait for ACK
        ack = self.conn.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
        if ack and ack.type == 0:
            logger.info("Mission upload successful (%d items)", n)
        else:
            ack_type = ack.type if ack else "timeout"
            logger.error("Mission upload failed — ACK type: %s", ack_type)
            raise RuntimeError(f"Mission upload failed: ACK={ack_type}")

    # ── Arm / Mode / Start ──

    def set_mode(self, mode: str):
        """Set the flight mode by name (e.g. 'AUTO', 'GUIDED', 'RTL').

        Args:
            mode: Flight mode name (case-insensitive).
        """
        mode_map = self.conn.mode_mapping()
        if mode_map is None:
            raise RuntimeError("Could not retrieve mode mapping")

        mode_upper = mode.upper()
        if mode_upper not in mode_map:
            raise ValueError(
                f"Unknown mode '{mode}'. Available: {list(mode_map.keys())}"
            )

        mode_id = mode_map[mode_upper]
        self.conn.set_mode(mode_id)
        logger.info("Set mode → %s", mode_upper)

    def arm(self):
        """Arm the drone motors."""
        self.conn.arducopter_arm()
        self.conn.motors_armed_wait()
        logger.info("Motors armed")

    def disarm(self):
        """Disarm the drone motors."""
        self.conn.arducopter_disarm()
        self.conn.motors_disarmed_wait()
        logger.info("Motors disarmed")

    def arm_and_start_mission(self):
        """Arm the drone, switch to AUTO mode, and begin the mission."""
        self.set_mode("GUIDED")
        time.sleep(1)
        self.arm()
        time.sleep(1)
        self.set_mode("AUTO")
        logger.info("Mission started in AUTO mode")

    # ── Camera & Spray Control ──

    def trigger_camera(self):
        """Trigger a single camera capture via MAVLink command."""
        self.conn.mav.command_long_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL,
            0,  # confirmation
            0, 0, 0, 0,
            1,  # shot (1 = single capture)
            0, 0,
        )
        logger.debug("Camera trigger sent")

    def set_spray_pump(self, on: bool, relay_channel: int = 0):
        """Toggle the spray pump via a relay channel.

        Args:
            on: True to activate, False to deactivate.
            relay_channel: Relay/servo channel number (default 0).
        """
        self.conn.mav.command_long_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_RELAY,
            0,
            relay_channel,
            1 if on else 0,
            0, 0, 0, 0, 0,
        )
        state = "ON" if on else "OFF"
        logger.info("Spray pump %s (relay %d)", state, relay_channel)

    # ── Utility ──

    def request_data_stream(self, rate_hz: int = 4):
        """Request all data streams at the given rate.

        Useful after connecting to ensure telemetry flows.
        """
        self.conn.mav.request_data_stream_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate_hz,
            1,  # start
        )
        logger.info("Requested data streams at %d Hz", rate_hz)

    @property
    def is_connected(self) -> bool:
        return self.conn is not None
