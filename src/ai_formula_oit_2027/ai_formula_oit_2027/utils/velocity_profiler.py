"""Horizon-based Velocity Profiler with Anticipative Corner Braking and Slew Limits."""

import math
from typing import Tuple
import numpy as np


class HorizonVelocityProfiler:
    """
    Computes optimal target velocity using both near and far-horizon road curvatures,
    enabling anticipative braking before sharp corners and smooth corner-exit acceleration.
    """

    def __init__(
        self,
        max_speed: float = 1.8,             # Maximum straight-line speed [m/s]
        min_speed: float = 0.35,            # Minimum corner crawl speed [m/s]
        max_lat_accel: float = 2.0,         # Max lateral acceleration limit (G-force constraint) [m/s^2]
        far_curve_slowdown_gain: float = 1.8, # Braking urgency factor for distant corners
        max_linear_accel: float = 1.6,      # Max longitudinal acceleration [m/s^2]
        max_linear_decel: float = 3.5,      # Max longitudinal braking deceleration [m/s^2]
        max_angular_accel: float = 4.0,     # Max steering rate [rad/s^2]
    ):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        self.k_far = far_curve_slowdown_gain
        self.max_accel = max_linear_accel
        self.max_decel = max_linear_decel
        self.max_ang_accel = max_angular_accel

        self.last_v = 0.0
        self.last_w = 0.0

    def compute_target_speed(
        self,
        near_curvature: float,
        far_curvature: float,
        current_speed: float,
    ) -> float:
        """
        Calculates safe forward speed considering both current turn radius and future corner shape.
        """
        abs_k_near = abs(near_curvature)
        abs_k_far = abs(far_curvature)

        # 1. Immediate cornering constraint (Lateral G Limit): v <= sqrt(a_lat / |k_near|)
        if abs_k_near > 1e-3:
            v_near_limit = math.sqrt(max(0.01, self.max_lat_accel / abs_k_near))
        else:
            v_near_limit = self.max_speed

        # 2. Lookahead Anticipative Braking (Distant corner entry):
        # If a sharp corner is seen 5-7m ahead, begin decelerating in advance
        if abs_k_far > 1e-3:
            v_far_limit = self.max_speed / (1.0 + self.k_far * abs_k_far)
        else:
            v_far_limit = self.max_speed

        v_cand = min(v_near_limit, v_far_limit, self.max_speed)
        return float(np.clip(v_cand, self.min_speed, self.max_speed))

    def apply_slew_rate(
        self,
        target_v: float,
        target_w: float,
        dt: float,
    ) -> Tuple[float, float]:
        """Smooths control commands respecting physical acceleration limits."""
        if dt <= 0.0:
            dt = 0.05

        # Linear speed slew
        dv = target_v - self.last_v
        max_up = self.max_accel * dt
        max_down = -self.max_decel * dt
        clamped_dv = float(np.clip(dv, max_down, max_up))
        v_out = self.last_v + clamped_dv

        # Angular rate slew
        dw = target_w - self.last_w
        max_dw = self.max_ang_accel * dt
        clamped_dw = float(np.clip(dw, -max_dw, max_dw))
        w_out = self.last_w + clamped_dw

        self.last_v = v_out
        self.last_w = w_out
        return v_out, w_out

    def reset(self) -> None:
        self.last_v = 0.0
        self.last_w = 0.0
