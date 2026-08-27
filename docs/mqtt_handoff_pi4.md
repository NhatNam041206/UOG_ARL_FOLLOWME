# MQTT Handoff — Pi 4 Motor Controller

For whoever owns the Pi 4 side of this robot. This describes the wire contract Rasp 5's
`modules/mqtt_bridge` publishes — nothing about the Pi 4 subscriber itself is implemented or
assumed here; that side is entirely out of scope for this module (see
[`docs/architecture.md`](architecture.md)).

## Wire contract

| | |
|---|---|
| **Topic** | `autobot/control/followme` (configurable — see `config/thresholds.yaml`'s `mqtt_bridge.topic`; this is the default) |
| **QoS** | 0 |
| **Retained** | No |
| **Payload** | Plain string (not JSON): `"<servo_angle_pwm 0-180>,<is_moving 0\|1>"` — e.g. `"95,1"` |
| **Publish rate** | Whatever `config/thresholds.yaml`'s `mqtt_bridge.publish_hz` is configured to on the Rasp 5 side — see [`docs/parameters.md`](parameters.md#mqtt_bridge) for its current (uncalibrated at time of writing) value. |

`servo_angle_pwm` is the absolute PWM angle to write directly to the steering servo (90 = straight
ahead by default — see `config/thresholds.yaml`'s `steering.servo_center_degrees`). `is_moving`
is `1` when the robot should drive forward, `0` when it should be stopped; there is no
speed/reverse field — this project has no such concept yet. When `is_moving` is `0`, the angle is
always the configured center angle, never a stale or leftover steering value.

## Stop signaling — what's covered and what isn't

Two mechanisms deliver a stop command to this topic on the Rasp 5 side going away:

1. **Explicit stop on clean exit** — when the Rasp 5 process exits normally (including Ctrl+C),
   `mqtt_bridge.close()` publishes one final `"<servo_center_degrees>,0"` message before
   disconnecting.
2. **MQTT Last Will and Testament** — registered at connect time with the same payload shape, QoS
   0, not retained. The broker delivers it if the Rasp 5 process's connection drops uncleanly
   (crash, killed process, network cable pulled, etc.).

**Explicit gap, stated plainly: neither mechanism fires if the Rasp 5 process hangs while its TCP
connection to the broker stays open** — no clean disconnect and no LWT trigger happens in that
case, because the connection never actually closes. This cannot be detected or fixed from the
Rasp 5 side (a hung process can't publish anything, including its own "I'm hung" message).

**This is the Pi 4 subscriber's own responsibility to guard against**, via a staleness timeout:
if no message arrives on this topic for some multiple of the expected publish interval
(recommended: 3–5× `1 / publish_hz`), the Pi 4 side should force a local stop regardless of the
last command it received. This module will not implement that logic — it lives entirely on the
subscriber side, which is out of scope here.

## Status at time of writing

`mqtt_bridge.broker_host` and `mqtt_bridge.publish_hz` are both `null` in
`config/thresholds.yaml` (fail-closed, uncalibrated) — nothing is published yet until both are
filled in with real values (the Pi 4's actual IP address, and an empirically-tuned publish rate).
See [`docs/parameters.md`](parameters.md#mqtt_bridge).
