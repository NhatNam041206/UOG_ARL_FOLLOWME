"""
FollowMeCommand -> wire payload encoding for mqtt_bridge. Not part of the public contract —
external callers use interface.py only.

Wire payload: "<servo_angle_pwm 0-180>,<is_moving 0|1>" — a plain string, not JSON, per the agreed
wire contract (see docs/mqtt_handoff_pi4.md). should_move and steering_angle_degrees are handled
as two independent fields right up to this encode boundary, even though they are always sent
together atomically in one publish() call — is_moving is never inferred from the angle.

Angle: modules.followme_orchestrator.interface.FollowMeCommand.steering_angle_degrees is already
the ABSOLUTE servo angle, ready to write directly to the servo — confirmed against
modules/followme_orchestrator/steering_controller.py's SteeringController.update() docstring
("Returns the ABSOLUTE servo angle in degrees... no further conversion needed downstream") and
FollowMeCommand's own field comment in modules/followme_orchestrator/interface.py. So no
degrees-to-PWM scaling happens here — only rounding to the nearest integer PWM value and a hard
validity check against the servo's physical 0-180 range, which raises (rather than silently
clamping) so a miscalibrated steering: section or an out-of-range upstream value fails loudly
during development instead of driving the servo to a wrong angle silently.

When should_move is False or steering_angle_degrees is None, the angle is always encoded as
servo_center_degrees (rounded) instead — never a stale or garbage angle while the robot isn't
being told to move.
"""
from typing import Optional

_PWM_MIN = 0
_PWM_MAX = 180


def encode(should_move: bool, steering_angle_degrees: Optional[float], servo_center_degrees: float) -> str:
    angle = steering_angle_degrees if (should_move and steering_angle_degrees is not None) else servo_center_degrees

    servo_angle_pwm = round(angle)
    if not (_PWM_MIN <= servo_angle_pwm <= _PWM_MAX):
        raise ValueError(
            f"mqtt_bridge.codec: servo_angle_pwm={servo_angle_pwm} out of valid range "
            f"[{_PWM_MIN}, {_PWM_MAX}] (input angle={angle}) — check config/thresholds.yaml's "
            f"steering section, or a miscalibrated FollowMeCommand upstream."
        )

    is_moving = 1 if should_move else 0
    return f"{servo_angle_pwm},{is_moving}"
