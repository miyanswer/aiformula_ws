"""Robust 2nd/3rd degree polynomial curve fitting for dotted and continuous lane lines."""

import math
from typing import Optional, Tuple
import numpy as np


class LaneCurveFitter:
    """
    Fits continuous polynomial curves to fragmented/dotted BEV lane points,
    smoothly interpolating gaps and computing precise local curvatures.
    """

    def __init__(
        self,
        poly_order: int = 2,
        lane_width: float = 3.0,
        min_fit_points: int = 8,
        ransac_iterations: int = 20,
        ransac_inlier_thresh: float = 0.25,  # [m]
    ):
        self.poly_order = poly_order
        self.lane_width = lane_width
        self.half_width = lane_width / 2.0
        self.min_fit_points = min_fit_points
        self.ransac_iterations = ransac_iterations
        self.ransac_inlier_thresh = ransac_inlier_thresh

        self.last_coeffs: Optional[np.ndarray] = None
        self.last_confidence: float = 0.0

    def fit_curve(
        self,
        points: np.ndarray,
        lateral_offset: float = 0.0,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Fits y = f(x) = c0*x^2 + c1*x + c2 (order 2) with RANSAC outlier rejection.
        Args:
            points: (N, 2) array of (x, y) coordinates in robot frame.
            lateral_offset: Optional y-shift (e.g. -1.5m for left lane to center).
        Returns:
            (poly_coeffs, confidence) where poly_coeffs is [a, b, c] for a*x^2 + b*x + c.
        """
        if points is None or len(points) < self.min_fit_points:
            return None, 0.0

        xs = points[:, 0]
        ys = points[:, 1] + lateral_offset

        # RANSAC for robust polynomial fitting
        best_inliers = None
        max_inliers = 0
        best_coeffs = None
        n_pts = len(xs)

        for _ in range(self.ransac_iterations):
            sample_idx = np.random.choice(n_pts, size=min(n_pts, 6), replace=False)
            try:
                cand_coeffs = np.polyfit(xs[sample_idx], ys[sample_idx], deg=self.poly_order)
            except Exception:
                continue

            y_pred = np.polyval(cand_coeffs, xs)
            residuals = np.abs(ys - y_pred)
            inliers = residuals < self.ransac_inlier_thresh
            inlier_count = int(np.sum(inliers))

            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_inliers = inliers

        if best_inliers is not None and max_inliers >= self.min_fit_points:
            try:
                best_coeffs = np.polyfit(xs[best_inliers], ys[best_inliers], deg=self.poly_order)
            except Exception:
                best_coeffs = None

        if best_coeffs is None:
            # Fallback standard polyfit
            try:
                best_coeffs = np.polyfit(xs, ys, deg=self.poly_order)
                max_inliers = n_pts
            except Exception:
                return None, 0.0

        confidence = float(np.clip(max_inliers / max(1, n_pts), 0.0, 1.0))
        self.last_coeffs = best_coeffs
        self.last_confidence = confidence

        return best_coeffs, confidence

    def sample_trajectory(
        self,
        coeffs: np.ndarray,
        x_start: float = 1.0,
        x_end: float = 8.0,
        num_points: int = 35,
    ) -> np.ndarray:
        """
        Generates a uniformly sampled, continuous (x, y) trajectory along the fitted polynomial.
        Returns:
            np.ndarray of shape (num_points, 2)
        """
        if coeffs is None:
            return np.empty((0, 2))

        sample_x = np.linspace(x_start, x_end, num=num_points)
        sample_y = np.polyval(coeffs, sample_x)
        return np.column_stack((sample_x, sample_y))

    def evaluate_curvature(self, coeffs: np.ndarray, x: float) -> float:
        """
        Computes the signed geometric curvature kappa = y'' / (1 + (y')^2)^(3/2) at forward distance x.
        Positive kappa indicates left turn, negative indicates right turn.
        """
        if coeffs is None:
            return 0.0

        if len(coeffs) == 3:
            # y = a*x^2 + b*x + c -> y' = 2*a*x + b, y'' = 2*a
            a, b, _ = coeffs
            dy = 2.0 * a * x + b
            ddy = 2.0 * a
        elif len(coeffs) == 4:
            # y = a*x^3 + b*x^2 + c*x + d
            a, b, c, _ = coeffs
            dy = 3.0 * a * (x**2) + 2.0 * b * x + c
            ddy = 6.0 * a * x + 2.0 * b
        else:
            return 0.0

        denom = (1.0 + dy**2) ** 1.5
        if denom < 1e-6:
            return 0.0

        kappa = float(ddy / denom)
        return kappa
