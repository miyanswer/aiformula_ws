"""Ultra-lightweight 2D BEV Visual Odometry and Distance Tracking."""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class VehiclePose:
    x: float = 0.0          # Global X [m]
    y: float = 0.0          # Global Y [m]
    yaw: float = 0.0        # Global Heading [rad]
    distance_s: float = 0.0 # Total accumulated distance along course [m]
    lap_count: int = 1      # Current lap
    linear_velocity: float = 0.0 # Instantaneous speed [m/s]
    angular_velocity: float = 0.0 # Instantaneous yaw rate [rad/s]


class BEVVisualOdometry:
    """
    Computes frame-to-frame 2D rigid transformation from BEV lane point clouds,
    tracking accumulated distance s, vehicle pose (x, y, yaw), and speed without external sensors.
    """

    def __init__(
        self,
        max_valid_step_dist: float = 1.0,    # Max reasonable displacement between consecutive frames [m]
        max_valid_step_yaw: float = 0.5,     # Max reasonable delta yaw [rad]
        lap_length_m: Optional[float] = None, # If set, automatically loops distance s per lap
    ):
        self.max_valid_step_dist = max_valid_step_dist
        self.max_valid_step_yaw = max_valid_step_yaw
        self.lap_length_m = lap_length_m

        self.pose = VehiclePose()
        self.prev_points: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Resets pose and distance accumulator."""
        self.pose = VehiclePose(x=x, y=y, yaw=yaw, distance_s=0.0, lap_count=1)
        self.prev_points = None
        self.prev_timestamp = None

    def trigger_lap_completion(self) -> None:
        """Manual or landmark-triggered lap reset."""
        self.pose.lap_count += 1
        self.pose.distance_s = 0.0

    def update(
        self,
        current_bev_points: np.ndarray,
        timestamp: Optional[float] = None,
        fallback_cmd_vel: Optional[Tuple[float, float]] = None,
    ) -> VehiclePose:
        """
        Estimates delta motion and updates vehicle odometry pose.
        Args:
            current_bev_points: (N, 2) metric points in base_link.
            timestamp: Optional POSIX timestamp.
            fallback_cmd_vel: (v, omega) used as fallback if points are insufficient.
        """
        now = timestamp if timestamp is not None else time.time()
        dt = (now - self.prev_timestamp) if self.prev_timestamp is not None else 0.05
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05

        dx, dy, dyaw = 0.0, 0.0, 0.0
        success = False

        if (
            self.prev_points is not None
            and current_bev_points is not None
            and len(self.prev_points) >= 12
            and len(current_bev_points) >= 12
        ):
            # Compute 2D rigid transform between point sets
            dx_pts, dy_pts, dyaw_pts, success = self._match_2d_points(self.prev_points, current_bev_points)
            if success:
                # Robot displacement is the negative of point cloud shift
                dx_est = -dx_pts
                dy_est = -dy_pts
                dyaw_est = -dyaw_pts
                step_dist = math.sqrt(dx_est**2 + dy_est**2)
                if step_dist < self.max_valid_step_dist and abs(dyaw_est) < self.max_valid_step_yaw:
                    dx, dy, dyaw = dx_est, dy_est, dyaw_est
                else:
                    success = False

        # Fallback to internal dead reckoning with command velocity if scan match fails
        if not success and fallback_cmd_vel is not None:
            v_cmd, w_cmd = fallback_cmd_vel
            dx = v_cmd * dt
            dy = 0.0
            dyaw = w_cmd * dt

        # Update integrated pose
        step_forward = dx
        self.pose.distance_s += max(0.0, step_forward)

        # Handle lap wrap-around if total length is known
        if self.lap_length_m is not None and self.pose.distance_s >= self.lap_length_m:
            self.pose.lap_count += 1
            self.pose.distance_s -= self.lap_length_m

        # Global position update
        self.pose.x += dx * math.cos(self.pose.yaw) - dy * math.sin(self.pose.yaw)
        self.pose.y += dx * math.sin(self.pose.yaw) + dy * math.cos(self.pose.yaw)
        self.pose.yaw = (self.pose.yaw + dyaw + math.pi) % (2.0 * math.pi) - math.pi

        self.pose.linear_velocity = float(dx / dt) if dt > 0 else 0.0
        self.pose.angular_velocity = float(dyaw / dt) if dt > 0 else 0.0

        # Save state
        self.prev_points = current_bev_points.copy() if current_bev_points is not None else None
        self.prev_timestamp = now

        return self.pose

    def _match_2d_points(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
    ) -> Tuple[float, float, float, bool]:
        """
        Fast 2D centroid and covariance matching (Kabsch / Procrustes 2D).
        """
        try:
            # Downsample to common size
            n = min(len(pts_prev), len(pts_curr), 40)
            p_prev = pts_prev[:n]
            p_curr = pts_curr[:n]

            c_prev = np.mean(p_prev, axis=0)
            c_curr = np.mean(p_curr, axis=0)

            p_prev_centered = p_prev - c_prev
            p_curr_centered = p_curr - c_curr

            # 2x2 Covariance matrix H = P_prev^T * P_curr
            H = np.dot(p_prev_centered.T, p_curr_centered)
            U, _, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # Ensure right-handed rotation matrix (det(R) == 1)
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = np.dot(Vt.T, U.T)

            dyaw = math.atan2(R[1, 0], R[0, 0])
            t = c_curr - np.dot(c_prev, R.T)
            dx = float(t[0])
            dy = float(t[1])

            return dx, dy, dyaw, True
        except Exception:
            return 0.0, 0.0, 0.0, False
