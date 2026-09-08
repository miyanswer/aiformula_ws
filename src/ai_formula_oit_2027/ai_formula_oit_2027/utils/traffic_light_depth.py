#!/usr/bin/env python3
"""
traffic_light_depth.py - 信号機の高さから垂直奥行き(Z)を算出するモジュール

幾何モデル:
    ピンホールカメラモデルの相似則により、画面の水平/垂直位置や角度にかかわらず、
    バウンディングボックスの高さ h_px からカメラ正面方向の垂直奥行き Z は以下の式で直接求まります。
    
    Z = (f_y * H_real) / h_px

    - H_real: 信号機の実際の物理的な高さ (デフォルト: 0.32m = 32cm)
    - f_y: カメラの垂直焦点距離 [pixel] (デフォルト: 189.5)
    - h_px: YOLO等の検出枠の高さ [pixel] (y2 - y1)
"""

from typing import Optional, Tuple, Dict


class TrafficLightDepthEstimator:
    """
    信号機のバウンディングボックス高さから垂直奥行き(Z[m])を逆算する推定器
    """

    def __init__(
        self,
        real_height: float = 0.32,
        focal_length_y: float = 189.5,
        min_bbox_height: float = 3.0,
        max_valid_distance: float = 50.0,
        smoothing_alpha: float = 0.7,
    ):
        """
        Args:
            real_height (float): 信号機の実際の高さ [m] (デフォルト: 0.32m)
            focal_length_y (float): カメラの垂直焦点距離 [pixel] (デフォルト: 189.5)
            min_bbox_height (float): ゼロ除算・ノイズ防止用の最小BBox高さ [pixel]
            max_valid_distance (float): 異常値クリップ用の最大許容距離 [m]
            smoothing_alpha (float): 指数平滑化の重み (1.0でフィルタ無効、0.0で更新なし)
        """
        self.real_height = float(real_height)
        self.focal_length_y = float(focal_length_y)
        self.min_bbox_height = float(min_bbox_height)
        self.max_valid_distance = float(max_valid_distance)
        self.smoothing_alpha = float(smoothing_alpha)

        # トラッキング用の距離キャッシュ {track_id: smoothed_depth}
        self._depth_cache: Dict[int, float] = {}

    def calculate_depth(self, bbox_height_px: float) -> Optional[float]:
        """
        バウンディングボックスのピクセル高さから垂直奥行き Z [m] を直接計算します。

        Args:
            bbox_height_px (float): BBoxの高さ [pixel] (y2 - y1)

        Returns:
            Optional[float]: 垂直奥行き Z [m] (計算不能な場合は None)
        """
        if bbox_height_px < self.min_bbox_height:
            return None

        # 幾何学的逆算: Z = (f * H) / h
        depth = (self.focal_length_y * self.real_height) / bbox_height_px

        # 範囲チェック
        if depth > self.max_valid_distance or depth <= 0.0:
            return None

        return float(depth)

    def estimate_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        track_id: Optional[int] = None,
    ) -> Optional[float]:
        """
        (x1, y1, x2, y2) 形式のバウンディングボックスから垂直奥行き Z [m] を推定します。

        Args:
            bbox (Tuple[float, float, float, float]): (x1, y1, x2, y2)
            track_id (Optional[int]): トラッキングID（平滑化フィルタ用、省略可能）

        Returns:
            Optional[float]: 推定された垂直奥行き Z [m]
        """
        x1, y1, x2, y2 = bbox
        bbox_height_px = abs(y2 - y1)

        raw_depth = self.calculate_depth(bbox_height_px)
        if raw_depth is None:
            return None

        # トラッキングIDが指定されている場合は平滑化フィルタ（EMA）を適用
        if track_id is not None and 0.0 < self.smoothing_alpha < 1.0:
            if track_id in self._depth_cache:
                prev_depth = self._depth_cache[track_id]
                smoothed_depth = (
                    self.smoothing_alpha * raw_depth
                    + (1.0 - self.smoothing_alpha) * prev_depth
                )
            else:
                smoothed_depth = raw_depth
            self._depth_cache[track_id] = smoothed_depth
            return smoothed_depth

        return raw_depth

    def clear_cache(self):
        """平滑化キャッシュをリセット"""
        self._depth_cache.clear()
