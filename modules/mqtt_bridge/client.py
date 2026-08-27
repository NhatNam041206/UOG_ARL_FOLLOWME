"""
paho-mqtt connection wrapper for mqtt_bridge. Not part of the public contract — external callers
use interface.py only.

QoS 0, retain off on every publish, per the agreed wire contract. Registers a Last Will and
Testament at connect time — the SAME "stopped" encoding this module's own close() sends
explicitly — so an unclean Rasp 5 disconnect (process killed, network drop) still leaves the Pi 4
subscriber with a stop command on the broker. This does NOT cover Rasp 5 hanging while the TCP
socket stays open (no clean or unclean disconnect ever fires in that case) — that gap is the Pi 4
subscriber's own staleness-timeout responsibility; see docs/mqtt_handoff_pi4.md.

Auto-reconnects via paho-mqtt's own loop_start() background thread. Every connect/publish failure
is caught and logged here — never raised up into main.py's per-frame loop, so a network problem
alone can never crash a followme run.

Pinned to paho-mqtt's 1.x line (see requirements.txt) specifically to use its stable
on_connect/on_disconnect callback signature (client, userdata, flags, rc) — paho-mqtt 2.x changed
that signature and requires an explicit callback_api_version choice; 1.x avoids that complexity
for this module's needs.
"""
import logging

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MqttClient:
    def __init__(self, host: str, port: int, topic: str, stop_payload: str):
        self.topic = topic
        self._connected = False

        self._client = mqtt.Client()
        self._client.will_set(topic, payload=stop_payload, qos=0, retain=False)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        try:
            self._client.connect_async(host, port)
            self._client.loop_start()
        except Exception:
            logger.exception(f"mqtt_bridge: failed to start MQTT connection to {host}:{port}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info(f"mqtt_bridge: connected to broker, publishing to '{self.topic}'")
        else:
            self._connected = False
            logger.warning(f"mqtt_bridge: broker connect failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning(f"mqtt_bridge: unexpected disconnect (rc={rc}), auto-reconnecting")

    def publish(self, payload: str) -> bool:
        """Never raises — returns False on any failure (not connected, send error)."""
        if not self._connected:
            return False
        try:
            result = self._client.publish(self.topic, payload=payload, qos=0, retain=False)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception:
            logger.exception("mqtt_bridge: publish failed")
            return False

    def close(self, stop_payload: str) -> None:
        """Publishes an explicit stop payload (belt-and-suspenders alongside the LWT, which only
        fires on an UNCLEAN disconnect — this covers the clean-exit path), then disconnects."""
        if self._connected:
            try:
                self._client.publish(self.topic, payload=stop_payload, qos=0, retain=False)
            except Exception:
                logger.exception("mqtt_bridge: failed to publish explicit stop on close")
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            logger.exception("mqtt_bridge: error during MQTT disconnect")
