"""
decode.py - Row-anchor decoding for UFLD output, ported 1:1 from the official
Ultra-Fast-Lane-Detection demo.py post-processing (softmax-weighted expectation
over the griding classes, "no-lane" background class = last griding index).
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LaneDetection:
    lane_index: int
    points_px: np.ndarray      # (N, 2) [x, y] in original image pixel coords, ordered near->far
    valid_mask: np.ndarray     # (num_row_anchors,) bool, True where this lane has a detection at that row anchor


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_lanes(
    out: np.ndarray,
    img_w: int,
    img_h: int,
    row_anchor: List[int],
    griding_num: int,
    min_points: int = 3,
) -> List[LaneDetection]:
    """
    Args:
        out: raw network output, shape (griding_num + 1, num_row_anchors, num_lanes).
        img_w, img_h: original camera image size to project pixel coordinates onto.
        row_anchor: canonical row-anchor list (in 288-px input-space coordinates).
        griding_num: number of griding (column) classes, excluding the "no-lane" class.
        min_points: a lane slot is discarded if it has fewer valid row detections than this.

    Returns:
        One LaneDetection per lane slot that has >= min_points valid detections.
    """
    num_row_anchors = len(row_anchor)
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_rev = out[:, ::-1, :]  # reverse row-anchor axis, matches official decode
    prob = _softmax(out_rev[:-1, :, :], axis=0)
    idx = (np.arange(griding_num) + 1).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)  # (num_row_anchors, num_lanes)

    argmax_cls = np.argmax(out_rev, axis=0)  # (num_row_anchors, num_lanes)
    loc[argmax_cls == griding_num] = 0.0  # last class == "no lane at this row"

    num_lanes = out.shape[2]
    detections: List[LaneDetection] = []

    for lane_i in range(num_lanes):
        valid_mask = np.zeros(num_row_anchors, dtype=bool)
        pts = []
        for k in range(num_row_anchors):
            loc_val = loc[k, lane_i]
            if loc_val <= 0:
                continue
            row_idx = num_row_anchors - 1 - k  # undo the row-anchor reversal
            x_px = loc_val * col_sample_w * img_w / 800.0 - 1.0
            y_px = img_h * (row_anchor[row_idx] / 288.0) - 1.0
            pts.append((x_px, y_px))
            valid_mask[row_idx] = True

        if len(pts) < min_points:
            continue

        pts_arr = np.array(pts, dtype=np.float64)
        order = np.argsort(pts_arr[:, 1])  # near (large y) -> far (small y) becomes far->near; sort ascending y
        pts_arr = pts_arr[order]
        detections.append(LaneDetection(lane_index=lane_i, points_px=pts_arr, valid_mask=valid_mask))

    return detections


def encode_lane_label(x_px: float, img_w: int, griding_num: int) -> int:
    """
    Inverse of decode_lanes' per-point pixel formula: given a ground-truth lane pixel
    x coordinate, returns the 0-indexed griding class a training label should use so
    that decode_lanes reconstructs (approximately) the same x_px at inference time.
    Used by training-data preparation, kept next to decode_lanes so both formulas
    are changed together if either ever needs to.
    """
    col_sample_w = (800.0 - 1.0) / (griding_num - 1)
    loc = (x_px + 1.0) * 800.0 / img_w / col_sample_w
    bin_idx = int(round(loc)) - 1
    return int(np.clip(bin_idx, 0, griding_num - 1))
