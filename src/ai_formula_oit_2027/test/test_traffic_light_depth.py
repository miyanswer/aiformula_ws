#!/usr/bin/env python3
"""
test_traffic_light_depth.py - 垂直奥行き計算ロジックの単体テスト
"""

import sys
import os

# パス追加
sys.path.insert(0, os.path.abspath("src/ai_formula_oit_2027"))

from ai_formula_oit_2027.utils.traffic_light_depth import TrafficLightDepthEstimator


def test_depth_calculation():
    real_h = 0.32       # 32cm
    fy = 189.5          # focal length px
    estimator = TrafficLightDepthEstimator(real_height=real_h, focal_length_y=fy)

    print("=== Traffic Light Depth Estimator Unit Test ===")
    print(f"Condition: Real Height = {real_h} m, Focal Length Y = {fy} px")
    print("-" * 55)

    test_cases = [
        # (bbox_height_px, expected_depth_m, description)
        (60.64, 1.0, "近距離: 高さ 60.64 px -> 約 1.0 m"),
        (30.32, 2.0, "中距離: 高さ 30.32 px -> 約 2.0 m"),
        (12.13, 5.0, "遠距離: 高さ 12.13 px -> 約 5.0 m"),
        (6.06, 10.0, "最遠距離: 高さ 6.06 px -> 約 10.0 m"),
    ]

    for h_px, expected_z, desc in test_cases:
        calculated_z = estimator.calculate_depth(h_px)
        assert calculated_z is not None, f"Calculation failed for h={h_px}"
        err = abs(calculated_z - expected_z)
        print(f"[{desc}]")
        print(f"  Input Height : {h_px:6.2f} px")
        print(f"  Calculated Z : {calculated_z:6.3f} m (Expected: ~{expected_z:.2f} m, Error: {err:.4f} m)")
        assert err < 0.02, f"Error too large: {err}"

    # BBox からの推定テスト (x1, y1, x2, y2)
    bbox = (500.0, 400.0, 530.0, 430.32)  # 高さ 30.32 px
    z_from_bbox = estimator.estimate_from_bbox(bbox)
    print(f"\n[BBox Estimation Test: (500, 400, 530, 430.32)]")
    print(f"  Calculated Z: {z_from_bbox:.3f} m")
    assert abs(z_from_bbox - 2.0) < 0.02

    # 異常値・極小BBoxのガードテスト
    tiny_bbox = (500.0, 400.0, 502.0, 401.0)  # 高さ 1 px
    z_tiny = estimator.estimate_from_bbox(tiny_bbox)
    print(f"\n[Tiny BBox Guard Test: height = 1.0 px]")
    print(f"  Result: {z_tiny} (Expected: None)")
    assert z_tiny is None

    print("\n✅ All unit tests PASSED successfully!")


if __name__ == '__main__':
    test_depth_calculation()
