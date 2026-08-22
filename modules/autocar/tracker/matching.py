import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two (N,4)/(M,4) xyxy arrays -> (N,M)."""
    a = np.expand_dims(boxes_a, 1)
    b = np.expand_dims(boxes_b, 0)

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def iou_distance(tracks, detections) -> np.ndarray:
    """1 - IoU cost matrix between objects exposing an `.xyxy` property."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)

    track_boxes = np.array([t.xyxy for t in tracks])
    det_boxes = np.array([d.xyxy for d in detections])
    return 1.0 - iou_batch(track_boxes, det_boxes)


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Hungarian assignment, rejecting matches costlier than `thresh`.

    Returns (matches[N,2] track_idx/det_idx, unmatched_track_idxs, unmatched_det_idxs).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_dets = [i for i in range(cost_matrix.shape[1]) if i not in matched_cols]

    matches = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_tracks, unmatched_dets
