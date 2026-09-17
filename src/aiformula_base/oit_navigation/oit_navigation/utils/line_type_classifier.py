"""
line_type_classifier.py - Heuristic classification of a detected lane boundary into
solid (実線) / dashed (破線) / double (二重線), from UFLD row-anchor detections.

UFLD itself has no line-style head (CULane/TuSimple don't label it), so this works
purely from the geometry of what was already detected:
  - solid vs dashed: how continuously the lane is detected across its own row-anchor span.
    A real gap in a dashed line shows up as a run of missing row-anchor detections;
    a solid line has almost no gaps.
  - double: another detected lane slot runs abnormally close (in BEV lateral distance)
    to this one over a shared longitudinal range, which is what a physically separate
    parallel line looks like once projected to metric space.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

LINE_TYPE_SOLID = "実線"
LINE_TYPE_DASHED = "破線"
LINE_TYPE_DOUBLE = "二重線"
LINE_TYPE_UNKNOWN = "不明"

# OpenCV's Hershey fonts (cv2.putText) can't render Japanese glyphs (they draw as "?").
# Use these ASCII labels for on-image overlays; keep the Japanese labels for terminal logs.
LINE_TYPE_ASCII = {
    LINE_TYPE_SOLID: "SOLID",
    LINE_TYPE_DASHED: "DASHED",
    LINE_TYPE_DOUBLE: "DOUBLE",
    LINE_TYPE_UNKNOWN: "UNKNOWN",
}


@dataclass
class LineTypeResult:
    line_type: str
    continuity_ratio: float
    gap_runs: int


def _continuity_stats(valid_mask: np.ndarray) -> Optional[LineTypeResult]:
    valid_rows = np.nonzero(valid_mask)[0]
    if len(valid_rows) < 3:
        return None

    first, last = valid_rows[0], valid_rows[-1]
    span_mask = valid_mask[first:last + 1]
    span = len(span_mask)
    continuity_ratio = float(np.count_nonzero(span_mask)) / span

    # Count contiguous False-runs (gaps) inside the lane's own detected span.
    gap_runs = 0
    in_gap = False
    for is_valid in span_mask:
        if not is_valid and not in_gap:
            gap_runs += 1
            in_gap = True
        elif is_valid:
            in_gap = False

    return LineTypeResult(line_type=LINE_TYPE_UNKNOWN, continuity_ratio=continuity_ratio, gap_runs=gap_runs)


def classify_line_type(
    valid_mask: np.ndarray,
    bev_metric_pts: np.ndarray,
    other_lanes_bev_metric_pts: List[np.ndarray],
    solid_ratio_threshold: float = 0.85,
    dashed_min_ratio: float = 0.15,
    double_line_max_dist_m: float = 0.55,
    double_line_min_dist_m: float = 0.08,
) -> LineTypeResult:
    """
    Args:
        valid_mask: this lane's per-row-anchor detection mask (from ufld.decode.LaneDetection).
        bev_metric_pts: this lane's own points already converted to BEV metric (x, y).
        other_lanes_bev_metric_pts: BEV metric points of every other detected lane slot
            (not just the chosen ego pair), used only for the double-line check.
        solid_ratio_threshold: >= this fraction of its own span detected -> solid.
        dashed_min_ratio: below this, the detection is too sparse to trust -> unknown.
        double_line_max_dist_m / double_line_min_dist_m: lateral distance band (in the
            paint-to-paint sense) that indicates two separate physical lines rather than
            one lane boundary or two unrelated lane boundaries.
    """
    stats = _continuity_stats(valid_mask)
    if stats is None or bev_metric_pts is None or len(bev_metric_pts) == 0:
        return LineTypeResult(LINE_TYPE_UNKNOWN, 0.0, 0)

    # Double-line check takes priority: does another lane slot hug this one closely
    # over a shared forward-distance range?
    if _has_parallel_double_line(bev_metric_pts, other_lanes_bev_metric_pts,
                                  double_line_min_dist_m, double_line_max_dist_m):
        return LineTypeResult(LINE_TYPE_DOUBLE, stats.continuity_ratio, stats.gap_runs)

    if stats.continuity_ratio >= solid_ratio_threshold:
        line_type = LINE_TYPE_SOLID
    elif stats.continuity_ratio >= dashed_min_ratio and stats.gap_runs >= 1:
        line_type = LINE_TYPE_DASHED
    else:
        line_type = LINE_TYPE_UNKNOWN

    return LineTypeResult(line_type, stats.continuity_ratio, stats.gap_runs)


def _has_parallel_double_line(
    this_lane_pts: np.ndarray,
    other_lanes_pts: List[np.ndarray],
    min_dist_m: float,
    max_dist_m: float,
) -> bool:
    for other_pts in other_lanes_pts:
        if other_pts is None or len(other_pts) == 0:
            continue
        if other_pts is this_lane_pts:
            continue

        # Compare only where the two lanes overlap in forward distance (x).
        x_lo = max(this_lane_pts[:, 0].min(), other_pts[:, 0].min())
        x_hi = min(this_lane_pts[:, 0].max(), other_pts[:, 0].max())
        if x_hi - x_lo < 0.5:
            continue

        this_in_range = this_lane_pts[(this_lane_pts[:, 0] >= x_lo) & (this_lane_pts[:, 0] <= x_hi)]
        if len(this_in_range) == 0:
            continue

        other_y_interp = np.interp(this_in_range[:, 0], other_pts[::-1, 0], other_pts[::-1, 1])
        lateral_dist = np.abs(this_in_range[:, 1] - other_y_interp)
        median_dist = float(np.median(lateral_dist))

        if min_dist_m <= median_dist <= max_dist_m:
            return True

    return False
