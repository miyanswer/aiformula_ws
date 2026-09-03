"""Unit tests for BEV Transformer, Lane Curve Fitter, Pure Pursuit, and Velocity Profiler."""

import math
import numpy as np
import pytest

from ai_formula_oit_2027.utils.bev_transformer import BEVTransformer
from ai_formula_oit_2027.utils.lane_curve_fitter import LaneCurveFitter
from ai_formula_oit_2027.utils.bev_lane_extractor import BEVLaneExtractor
from ai_formula_oit_2027.utils.pure_pursuit import AdaptivePurePursuit
from ai_formula_oit_2027.utils.velocity_profiler import HorizonVelocityProfiler


def test_bev_transformer_conversions():
    transformer = BEVTransformer(
        src_w=1920,
        src_h=1080,
        cam_height=0.56,
        dist_min=1.15,
        dist_max=10.0,
        lateral_max=5.0,
        bev_w=640,
        bev_h=640,
    )

    # Test center point (x=5.0m, y=0.0m)
    u, v = transformer.robot_xy_to_bev_px(5.0, 0.0)
    assert 0 <= u < 640
    assert 0 <= v < 640

    x_back, y_back = transformer.bev_px_to_robot_xy(u, v)
    assert pytest.approx(x_back, abs=0.1) == 5.0
    assert pytest.approx(y_back, abs=0.1) == 0.0


def test_bev_lane_extractor_straight_and_curved():
    extractor = BEVLaneExtractor(
        bev_w=640,
        bev_h=640,
        lane_width_m=3.5,
        dist_min=1.15,
        dist_max=10.0,
        lateral_max=5.0,
        n_windows=10,
        margin_px=50,
    )

    # 1. Straight road with 3.5m width (left: y=+1.75m, u=208; right: y=-1.75m, u=432)
    mask_straight = np.zeros((640, 640), dtype=np.uint8)
    mask_straight[100:550, 205:211] = 255
    mask_straight[100:550, 429:435] = 255

    center_pts, left_pts, right_pts, _ = extractor.extract_lane_trajectories(mask_straight)
    assert left_pts is not None
    assert right_pts is not None
    assert center_pts is not None
    assert pytest.approx(float(np.mean(center_pts[:, 1])), abs=0.3) == 0.0

    # 2. Sharp curved road (left lane bends from u=208 to u=380, crossing vehicle center line u=320)
    mask_curve = np.zeros((640, 640), dtype=np.uint8)
    for v in range(100, 550):
        t = (550 - v) / 450.0
        u_left = int(208 + 160 * (t**1.5))
        u_right = int(432 + 160 * (t**1.5))
        if 0 <= u_left < 636:
            mask_curve[v, u_left:u_left+4] = 255
        if 0 <= u_right < 636:
            mask_curve[v, u_right:u_right+4] = 255

    c_pts_c, l_pts_c, r_pts_c, _ = extractor.extract_lane_trajectories(mask_curve)
    assert l_pts_c is not None
    assert r_pts_c is not None
    assert c_pts_c is not None

    # 3. Clean Continuous Right Lane + Noisy/Broken Left Lane -> Prioritize Right Offset (-1.75m)
    mask_noisy_left = np.zeros((640, 640), dtype=np.uint8)
    mask_noisy_left[450:470, 205:211] = 255
    mask_noisy_left[100:550, 429:435] = 255

    c_pts_n, l_pts_n, r_pts_n, _ = extractor.extract_lane_trajectories(mask_noisy_left)
    assert r_pts_n is not None
    assert c_pts_n is not None
    assert pytest.approx(float(np.mean(c_pts_n[:, 1])), abs=0.3) == 0.0


def test_pure_pursuit_steering():
    pp = AdaptivePurePursuit(
        min_lookahead=1.2,
        max_lookahead=6.0,
        lookahead_ratio=0.4,
    )
    # Target straight ahead (x=3.0, y=0.0) -> Steering omega is 0
    omega_straight, k_straight = pp.compute_steering((3.0, 0.0), current_speed=1.5)
    assert pytest.approx(omega_straight, abs=1e-3) == 0.0
    assert pytest.approx(k_straight, abs=1e-3) == 0.0

    # Target to the left (x=3.0, y=+1.0) -> Positive counter-clockwise steering omega
    omega_left, k_left = pp.compute_steering((3.0, 1.0), current_speed=1.5)
    assert omega_left > 0.0
    assert k_left > 0.0


def test_horizon_velocity_profiler_anticipation():
    profiler = HorizonVelocityProfiler(
        max_speed=2.0,
        min_speed=0.4,
        far_curve_slowdown_gain=2.0,
    )

    # Straight road (near = 0, far = 0) -> Max speed
    v_straight = profiler.compute_target_speed(near_curvature=0.0, far_curvature=0.0, current_speed=1.5)
    assert v_straight == 2.0

    # Approaching sharp corner (near = 0, far = 0.5) -> Pre-braking deceleration
    v_approaching_corner = profiler.compute_target_speed(near_curvature=0.0, far_curvature=0.5, current_speed=1.5)
    assert v_approaching_corner < 2.0
    assert v_approaching_corner >= 0.4
