"""
MQTT Bridge module — public contract.

THIS IS THE ONLY FILE OTHER MODULES MAY IMPORT FROM modules.mqtt_bridge. Everything else in this
package (config.py, codec.py, client.py) is an internal implementation detail and may change
without notice.

Purpose: publishes the current per-frame FollowMeCommand (see
modules/followme_orchestrator/interface.py) as a drive command over MQTT, from this pipeline's
host (Rasp 5) to the motor controller (Pi 4) via a shared broker. This module never reaches into
followme_orchestrator's internals — it only consumes the already-typed FollowMeCommand object
main.py's own per-frame loop already receives from followme_orchestrator.step(), per
docs/architecture.md design rule #2 (own-instance isolation, only main.py composes across module
boundaries). No code on the Pi 4 / subscriber side is implemented or assumed here beyond the wire
contract below — see docs/mqtt_handoff_pi4.md, the handoff note for that side's owner.

Wire contract:
    topic:   config's mqtt_bridge.topic (default "autobot/control/followme")
    QoS 0, not retained
    payload: "<servo_angle_pwm 0-180>,<is_moving 0|1>" — plain string, e.g. "95,1" (see codec.py)

Fail-closed, same convention as every other module in this project (docs/architecture.md rule #1
/ docs/parameters.md's status legend): while mqtt_bridge.broker_host or mqtt_bridge.publish_hz is
`null` in thresholds.yaml, publish() always returns False without attempting to connect or send
anything, and configure() logs this once rather than raising — a followme run must never crash
because MQTT config alone is missing or a broker is unreachable.
"""
import logging
from typing import Optional

from .client import MqttClient
from .codec import encode
from .config import MqttBridgeConfig, load_config

logger = logging.getLogger(__name__)

__all__ = ["configure", "publish", "close"]

_config: Optional[MqttBridgeConfig] = None
_client: Optional[MqttClient] = None
_last_publish_time: Optional[float] = None


def configure(config_path: str = "config/thresholds.yaml") -> None:
    """
    Must be called once before the first publish() call — same convention as
    followme_orchestrator.configure(). Reads the mqtt_bridge: section of thresholds.yaml for
    broker host/port/topic/publish_hz, and reuses the existing steering: section's
    servo_center_degrees for the "stopped" angle encoded whenever the robot isn't moving.

    Fail-closed: if mqtt_bridge.broker_host or mqtt_bridge.publish_hz is null, this logs a
    warning and leaves the module in a state where publish() always returns False — it does not
    raise, and does not attempt a broker connection. Otherwise starts the MQTT client
    asynchronously (see client.py) — a slow or unreachable broker at this point still returns
    from this call promptly; connection failures surface later as publish() returning False.
    """
    global _config, _client, _last_publish_time
    _config = load_config(config_path)
    _last_publish_time = None

    if _client is not None:
        _client.close(encode(False, None, _config.servo_center_degrees))
        _client = None

    if not _config.is_calibrated():
        logger.warning(
            "mqtt_bridge: broker_host and/or publish_hz not set in thresholds.yaml's mqtt_bridge "
            "section — publish() will no-op (always return False) until both are configured."
        )
        return

    _client = MqttClient(
        _config.broker_host, _config.broker_port, _config.topic,
        stop_payload=encode(False, None, _config.servo_center_degrees),
    )


def publish(command, timestamp: float) -> bool:
    """
    command: the FollowMeCommand object returned by followme_orchestrator.step().
    Rate-limits internally to the configured publish_hz — safe to call every frame.

    Returns True if a publish actually occurred this call, False if skipped (rate limit not yet
    elapsed, uncalibrated config, or not connected). Never raises on a network error or an
    encoding failure — logs and returns False instead.
    """
    global _last_publish_time

    if _config is None:
        logger.error("mqtt_bridge: publish() called before configure().")
        return False
    if _client is None or not _config.is_calibrated():
        return False

    if _last_publish_time is not None and (timestamp - _last_publish_time) < (1.0 / _config.publish_hz):
        return False

    try:
        payload = encode(command.should_move, command.steering_angle_degrees, _config.servo_center_degrees)
    except ValueError:
        logger.exception("mqtt_bridge: failed to encode command, skipping this publish")
        return False

    sent = _client.publish(payload)
    if sent:
        _last_publish_time = timestamp
    return sent


def close() -> None:
    """
    Publishes an explicit stop command, then disconnects cleanly. Call once when the followme
    loop exits (normal exit or Ctrl+C, via main.py's existing cleanup path) — safe to call even
    if configure() left the module uncalibrated/unconnected (no-op in that case).
    """
    global _client
    if _config is not None and _client is not None:
        _client.close(encode(False, None, _config.servo_center_degrees))
    _client = None
