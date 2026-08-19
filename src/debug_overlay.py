"""
Debug-only visual overlay for Stage 2 (FollowPipeline). Deliberately kept out of main.py so
the production/background run path (no --ui flag) never imports cv2's GUI-drawing calls or
pays their cost — this module is only imported when the user explicitly asks to see it.

Draws: per-track bboxes (target vs non-target vs aspect-ratio-vetoed), the dynamic ROI region
currently in use (if any), a directional arrow toward the target, and a status/telemetry panel
(mode, ROI failure streak, warm-up state, FPS) so a human can see what the pipeline is doing
frame-by-frame without reading the CSV log.
"""
import math
import cv2


def render(frame, pipeline, angle_result, current_fps: float):
    """
    Build and return a BGR debug frame (does not mutate `frame`). `pipeline` is the live
    FollowPipeline instance — reads its `last_detections`, `last_used_roi`, `last_roi_bounds`,
    `_roi_failure_count`, `smoothing_mode`, `_startup_warmup_done` for visualization only.
    """
    display = frame.copy()
    h, w = display.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 1. Dotted yellow vertical line marking frame center (steering reference)
    for dash_y in range(0, h, 16):
        cv2.line(display, (center_x, dash_y), (center_x, min(dash_y + 10, h)), (0, 255, 255), 1)

    # 2. Dynamic ROI region currently in use (cyan) — the whole point of this overlay request:
    # visualize the moving crop that constrains YOLO detection once a target is sticky-tracked.
    if pipeline.last_used_roi and pipeline.last_roi_bounds is not None:
        rx1, ry1, rx2, ry2 = pipeline.last_roi_bounds
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
        cv2.putText(display, "ROI", (rx1 + 4, max(14, ry1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # 3. Per-track bboxes: green=target, red=other, orange tag when aspect ratio gate vetoed it.
    detections = pipeline.last_detections
    target_det = None

    for det in detections:
        tid = det["track_id"]
        x1, y1, x2, y2 = det["bbox"]
        is_target = angle_result.target_found and angle_result.track_id == tid
        if is_target:
            target_det = det

        color = (0, 255, 0) if is_target else (0, 0, 255)
        thickness = 2 if is_target else 1
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

        if is_target:
            label = f"TARGET Score:{angle_result.similarity_score:.2f} Ang:{angle_result.angle_offset_deg:+.1f}deg"
            cv2.putText(display, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif det.get("aspect_ratio_pass") is False:
            # Explicitly flag tracks the aspect ratio hard gate vetoed, even if similarity alone
            # would have passed — the exact failure mode this gate exists to make visible.
            cv2.putText(display, "AR-MISMATCH", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # 4. Directional arrow toward the locked target
    if angle_result.target_found and target_det is not None:
        tx1, ty1, tx2, ty2 = target_det["bbox"]
        target_cx = int((tx1 + tx2) / 2.0)
        target_cy = int((ty1 + ty2) / 2.0)
        dx, dy = target_cx - center_x, target_cy - center_y
        dist = math.hypot(dx, dy)
        if dist > 1.0:
            max_len = min(dist, 65.0)
            end_x = int(center_x + (dx / dist) * max_len)
            end_y = int(center_y + (dy / dist) * max_len)
            cv2.arrowedLine(display, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2, tipLength=0.35)

    # 5. Header status line
    status_text = (
        f"TARGET LOCKED (Ang:{angle_result.angle_offset_deg:+.1f}deg Ratio:{angle_result.size_ratio:.2f})"
        if angle_result.target_found else "SEARCHING TARGET..."
    )
    status_color = (0, 255, 0) if angle_result.target_found else (0, 0, 255)
    cv2.putText(display, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # 6. Telemetry panel: FPS, smoothing mode, ROI state, warm-up state
    roi_state = "ROI" if pipeline.last_used_roi else "FULL-FRAME"
    telemetry = (
        f"FPS:{current_fps:.1f}  Mode:{pipeline.smoothing_mode}  Detect:{roi_state}  "
        f"ROIFail:{pipeline._roi_failure_count}/{pipeline.roi_failure_max_frames}"
    )
    cv2.putText(display, telemetry, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 7. Per-stage timing (ms) — pinpoints whether detect or verify dominates a slow frame,
    # instead of only seeing an aggregate FPS number.
    timing = pipeline.last_timing_ms
    timing_text = (
        f"Timing(ms) detect:{timing['detect_ms']:.1f}  verify:{timing['verify_ms']:.1f}  "
        f"total:{timing['total_ms']:.1f}"
    )
    cv2.putText(display, timing_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if pipeline.smoothing_mode == "voting" and not pipeline._startup_warmup_done:
        cv2.putText(display, "VOTING WARM-UP IN PROGRESS", (20, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    return display
