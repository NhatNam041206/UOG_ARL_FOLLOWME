"""
Internal config loading for the mqtt_bridge module. Not part of the public contract — external
callers use interface.py only.

servo_center_degrees is READ from the EXISTING steering: section, not duplicated into a new
mqtt_bridge: key — same convention as followme_orchestrator.config's own reuse of that same key
(see modules/followme_orchestrator/config.py).
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

_DEFAULT_BROKER_PORT = 1883
_DEFAULT_TOPIC = "autobot/control/followme"
_DEFAULT_SERVO_CENTER_DEGREES = 90.0  # fallback if steering.servo_center_degrees is unset


@dataclass
class MqttBridgeConfig:
    broker_host: Optional[str] = None   # REQUIRED (fail-closed) — Pi 4's IP on the shared network
    broker_port: int = _DEFAULT_BROKER_PORT
    topic: str = _DEFAULT_TOPIC
    publish_hz: Optional[float] = None  # REQUIRED (fail-closed) — uncalibrated, see docs/parameters.md

    servo_center_degrees: float = _DEFAULT_SERVO_CENTER_DEGREES

    def is_calibrated(self) -> bool:
        return self.broker_host is not None and self.publish_hz is not None


def load_config(thresholds_path: str = "config/thresholds.yaml") -> MqttBridgeConfig:
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds config not found at: '{os.path.abspath(thresholds_path)}'")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}
    section: Dict[str, Any] = thresholds.get("mqtt_bridge", {}) or {}
    steering_section: Dict[str, Any] = thresholds.get("steering", {}) or {}

    return MqttBridgeConfig(
        broker_host=section.get("broker_host"),
        broker_port=section.get("broker_port", _DEFAULT_BROKER_PORT),
        topic=section.get("topic", _DEFAULT_TOPIC),
        publish_hz=section.get("publish_hz"),
        servo_center_degrees=steering_section.get("servo_center_degrees", _DEFAULT_SERVO_CENTER_DEGREES),
    )
