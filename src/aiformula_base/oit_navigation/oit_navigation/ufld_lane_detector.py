#!/usr/bin/env python3
"""
ufld_lane_detector.py - UFLD (Ultra-Fast-Lane-Detection) based ego-lane detector.

Pipeline per frame:
  camera image -> UFLD row-anchor inference -> per-lane pixel points
  -> IPM (BEVTransformer) to metric BEV points
  -> line-type heuristic (solid / dashed / double) for the two ego lane boundaries
  -> terminal log + annotated image for RViz confirmation.

Mirrors the threading / decimation pattern of yolop_lane_detector.py so it can be
swapped in for A/B comparison without changing the surrounding launch/verification
workflow.
"""

import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header

from oit_navigation.utils.image_util import cv2_to_imgmsg, imgmsg_to_cv2
from oit_navigation.utils.bev_transformer import BEVTransformer
from oit_navigation.utils.line_type_classifier import (
    classify_line_type,
    LINE_TYPE_UNKNOWN,
    LINE_TYPE_ASCII,
)
from oit_navigation.ufld.model import ParsingNet
from oit_navigation.ufld.decode import decode_lanes, LaneDetection
from oit_navigation.ufld.constant import (
    TUSIMPLE_ROW_ANCHOR,
    TUSIMPLE_GRIDING_NUM,
    TUSIMPLE_NUM_LANES,
    TUSIMPLE_INPUT_SIZE,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LANE_COLORS = {
    "left": (255, 180, 0),   # cyan (BGR)
    "right": (0, 220, 255),  # yellow (BGR)
    "other": (140, 140, 140),
}


class UFLDLaneDetectorNode(Node):
    """UFLD ego-lane detector: image -> lane points -> BEV -> line-type classification."""

    def __init__(self):
        super().__init__('ufld_lane_detector')

        if torch.get_num_threads() > 2:
            torch.set_num_threads(2)

        self._declare_parameters()
        self._load_parameters()

        self.bev_transformer = BEVTransformer(
            cam_height=self.camera_height,
            dist_min=self.dist_min,
            dist_max=self.dist_max,
            lateral_max=self.lateral_max,
            bev_w=640,
            bev_h=640,
        )

        self._init_detector()

        self._infer_msg: Optional[Tuple[np.ndarray, Header]] = None
        self._msg_lock = threading.Lock()
        self._is_inferring = False
        self._latest_dets: List[LaneDetection] = []
        self._latest_img_shape: Optional[Tuple[int, int]] = None
        self._frame_count = 0
        self._last_log_time = 0.0
        self._running = True

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.is_compressed = self.input_image_topic.endswith('/compressed') or 'compressed' in self.input_image_topic
        sub_type = CompressedImage if self.is_compressed else Image

        self.image_sub = self.create_subscription(
            sub_type, self.input_image_topic, self._image_callback, qos_sensor
        )
        self.annotated_image_pub = self.create_publisher(Image, self.annotated_image_topic, 1)
        self.bev_image_pub = self.create_publisher(Image, self.bev_image_topic, 1)

        self._worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(
            f"UFLD Lane Detector Initialized: topic={self.input_image_topic} (compressed={self.is_compressed}), "
            f"device={self.device_str}, frame_skip={self.frame_skip}, weights={self.weight_path}"
        )

    def _declare_parameters(self):
        self.declare_parameter('input_image_topic', '/aiformula_sensing/zed_node/left_image/undistorted')
        self.declare_parameter('annotated_image_topic', '/aiformula_visualization/ufld_lane_detector/annotated_image')
        self.declare_parameter('bev_image_topic', '/aiformula_visualization/ufld_lane_detector/bev_image')

        self.declare_parameter('weight_path', '/aiformula_ws/models/pretrained/ufld_tusimple_r18.pth')
        self.declare_parameter('use_device', 'cpu')
        self.declare_parameter('frame_skip', 2)

        self.declare_parameter('camera_height', 0.56)
        self.declare_parameter('dist_min', 1.15)
        self.declare_parameter('dist_max', 10.0)
        self.declare_parameter('lateral_max', 5.0)

        self.declare_parameter('solid_ratio_threshold', 0.85)
        self.declare_parameter('dashed_min_ratio', 0.15)
        self.declare_parameter('double_line_min_dist_m', 0.08)
        self.declare_parameter('double_line_max_dist_m', 0.55)

        self.declare_parameter('log_interval_sec', 1.0)

    def _load_parameters(self):
        p = self.get_parameter
        self.input_image_topic = p('input_image_topic').value
        self.annotated_image_topic = p('annotated_image_topic').value
        self.bev_image_topic = p('bev_image_topic').value

        self.weight_path = p('weight_path').value
        self.device_str = str(p('use_device').value)
        self.frame_skip = max(1, int(p('frame_skip').value))

        self.camera_height = float(p('camera_height').value)
        self.dist_min = float(p('dist_min').value)
        self.dist_max = float(p('dist_max').value)
        self.lateral_max = float(p('lateral_max').value)

        self.solid_ratio_threshold = float(p('solid_ratio_threshold').value)
        self.dashed_min_ratio = float(p('dashed_min_ratio').value)
        self.double_line_min_dist_m = float(p('double_line_min_dist_m').value)
        self.double_line_max_dist_m = float(p('double_line_max_dist_m').value)

        self.log_interval_sec = float(p('log_interval_sec').value)

    def _init_detector(self):
        if self.device_str == 'cpu':
            self.use_device = torch.device('cpu')
        elif torch.cuda.is_available() and self.device_str != 'cpu':
            self.use_device = torch.device(f'cuda:{self.device_str}' if self.device_str.isdigit() else 'cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.use_device = torch.device('mps')
        else:
            self.use_device = torch.device('cpu')

        weights = self.weight_path
        if not os.path.exists(weights):
            alt_paths = [
                f"/aiformula_ws/{weights}",
                f"/aiformula_ws/models/pretrained/{os.path.basename(weights)}",
                "/aiformula_ws/models/pretrained/ufld_tusimple_r18.pth",
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    weights = alt
                    break

        if not os.path.exists(weights):
            self.get_logger().error(f"UFLD model weights not found at: {self.weight_path}")
            self.detector = None
            return

        self.get_logger().info(f"Loading UFLD weights from: {weights}")
        self.detector = ParsingNet(cls_dim=(TUSIMPLE_GRIDING_NUM + 1, len(TUSIMPLE_ROW_ANCHOR), TUSIMPLE_NUM_LANES))
        checkpoint = torch.load(weights, map_location=self.use_device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.detector.load_state_dict(state_dict, strict=True)
        self.detector = self.detector.to(self.use_device).float()
        self.detector.eval()
        self.get_logger().info("UFLD Model successfully loaded.")

    def _image_callback(self, msg):
        self._frame_count += 1

        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            image = imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if image is None:
            return

        if self.frame_skip <= 1 or (self._frame_count % self.frame_skip == 0):
            with self._msg_lock:
                if not self._is_inferring:
                    self._infer_msg = (image.copy(), msg.header)

        with self._msg_lock:
            cur_dets = self._latest_dets
            cur_shape = self._latest_img_shape

        annotated = image.copy()
        left_result = right_result = None
        bev_canvas = self._blank_bev_canvas()
        if cur_dets and cur_shape is not None and cur_shape == image.shape[:2]:
            left_result, right_result, bev_canvas = self._render_and_classify(annotated, cur_dets, image.shape[1])

        self._maybe_log(left_result, right_result)

        annotated_msg = cv2_to_imgmsg(annotated, encoding='bgr8', frame_id=msg.header.frame_id, stamp=msg.header.stamp)
        self.annotated_image_pub.publish(annotated_msg)

        bev_msg = cv2_to_imgmsg(bev_canvas, encoding='bgr8', frame_id=msg.header.frame_id, stamp=msg.header.stamp)
        self.bev_image_pub.publish(bev_msg)

    def _inference_worker(self):
        while rclpy.ok() and self._running:
            item = None
            with self._msg_lock:
                if self._infer_msg is not None:
                    item = self._infer_msg
                    self._infer_msg = None
                    self._is_inferring = True

            if item is None:
                time.sleep(0.005)
                continue

            if self.detector is None:
                with self._msg_lock:
                    self._is_inferring = False
                time.sleep(0.05)
                continue

            frame, header = item
            try:
                self._run_inference(frame)
            except Exception as e:
                self.get_logger().warning(f"Error during UFLD inference: {str(e)}")
            finally:
                with self._msg_lock:
                    self._is_inferring = False

    def _run_inference(self, image: np.ndarray):
        t_start = time.perf_counter()
        img_h, img_w = image.shape[:2]

        net_w, net_h = TUSIMPLE_INPUT_SIZE
        resized = cv2.resize(image, (net_w, net_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.use_device)

        with torch.no_grad():
            out = self.detector(tensor)
        out_np = out[0].cpu().numpy()

        dets = decode_lanes(out_np, img_w, img_h, TUSIMPLE_ROW_ANCHOR, TUSIMPLE_GRIDING_NUM)

        with self._msg_lock:
            self._latest_dets = dets
            self._latest_img_shape = (img_h, img_w)

        if self._frame_count % 30 == 0:
            infer_ms = (time.perf_counter() - t_start) * 1000.0
            self.get_logger().info(f"[UFLD Inference] Latency: {infer_ms:.1f} ms | lanes_detected={len(dets)}")

    def _select_ego_pair(self, dets: List[LaneDetection], img_w: int):
        mid_x = img_w / 2.0
        left_det, right_det = None, None
        left_best, right_best = -1.0, float('inf')

        for d in dets:
            bottom_x = float(d.points_px[-1, 0])
            if bottom_x < mid_x and bottom_x > left_best:
                left_best = bottom_x
                left_det = d
            elif bottom_x >= mid_x and bottom_x < right_best:
                right_best = bottom_x
                right_det = d

        return left_det, right_det

    def _render_and_classify(self, annotated: np.ndarray, dets: List[LaneDetection], img_w: int):
        self.bev_transformer.ensure_source_size(img_w, annotated.shape[0])
        bev_pts_by_lane = {id(d): self.bev_transformer.image_pts_to_bev_metric(d.points_px) for d in dets}

        left_det, right_det = self._select_ego_pair(dets, img_w)
        bev_canvas = self._blank_bev_canvas()

        for d in dets:
            role = "left" if d is left_det else ("right" if d is right_det else "other")
            color = _LANE_COLORS[role]
            for (x, y) in d.points_px:
                cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
            self._draw_bev_points(bev_canvas, bev_pts_by_lane[id(d)], color)

        left_result = self._classify_side(left_det, dets, bev_pts_by_lane)
        right_result = self._classify_side(right_det, dets, bev_pts_by_lane)

        for det, result, label in ((left_det, left_result, "L"), (right_det, right_result, "R")):
            if det is None or result is None:
                continue
            px, py = det.points_px[-1]
            color = _LANE_COLORS["left" if label == "L" else "right"]
            ascii_text = f"{label}:{LINE_TYPE_ASCII.get(result.line_type, '?')}"
            cv2.putText(
                annotated, ascii_text, (int(px) - 20, int(py) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            self._label_bev_lane(bev_canvas, bev_pts_by_lane[id(det)], ascii_text, color)

        return left_result, right_result, bev_canvas

    def _blank_bev_canvas(self) -> np.ndarray:
        bev = self.bev_transformer
        canvas = np.full((bev.bev_h, bev.bev_w, 3), 30, dtype=np.uint8)

        for x in np.arange(np.ceil(bev.dist_min), bev.dist_max + 0.01, 1.0):
            u0, v = bev.robot_xy_to_bev_px(float(x), bev.lateral_max)
            u1, _ = bev.robot_xy_to_bev_px(float(x), -bev.lateral_max)
            cv2.line(canvas, (u0, v), (u1, v), (60, 60, 60), 1)
        for y in np.arange(-bev.lateral_max, bev.lateral_max + 0.01, 1.0):
            u, v0 = bev.robot_xy_to_bev_px(bev.dist_min, float(y))
            _, v1 = bev.robot_xy_to_bev_px(bev.dist_max, float(y))
            cv2.line(canvas, (u, v0), (u, v1), (60, 60, 60), 1)

        ego_u, ego_v = bev.bev_w // 2, int(bev.v_near)
        cv2.circle(canvas, (ego_u, ego_v), 8, (255, 100, 0), -1)
        cv2.putText(canvas, "EGO", (ego_u - 18, ego_v + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return canvas

    def _draw_bev_points(self, canvas: np.ndarray, bev_metric_pts: np.ndarray, color: Tuple[int, int, int]) -> None:
        for (x, y) in bev_metric_pts:
            u, v = self.bev_transformer.robot_xy_to_bev_px(float(x), float(y))
            if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
                cv2.circle(canvas, (u, v), 5, color, -1)

    def _label_bev_lane(self, canvas: np.ndarray, bev_metric_pts: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        if bev_metric_pts is None or len(bev_metric_pts) == 0:
            return
        u, v = self.bev_transformer.robot_xy_to_bev_px(float(bev_metric_pts[0, 0]), float(bev_metric_pts[0, 1]))
        cv2.putText(canvas, text, (int(u) - 30, max(15, int(v) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _classify_side(self, this_det, all_dets, bev_pts_by_lane):
        if this_det is None:
            return None
        this_pts = bev_pts_by_lane[id(this_det)]
        other_pts = [bev_pts_by_lane[id(d)] for d in all_dets if d is not this_det]
        return classify_line_type(
            this_det.valid_mask,
            this_pts,
            other_pts,
            solid_ratio_threshold=self.solid_ratio_threshold,
            dashed_min_ratio=self.dashed_min_ratio,
            double_line_min_dist_m=self.double_line_min_dist_m,
            double_line_max_dist_m=self.double_line_max_dist_m,
        )

    def _maybe_log(self, left_result, right_result):
        now = time.time()
        if now - self._last_log_time < self.log_interval_sec:
            return
        self._last_log_time = now

        def fmt(result):
            if result is None:
                return "検出なし"
            if result.line_type == LINE_TYPE_UNKNOWN:
                return f"不明(連続率{result.continuity_ratio:.2f})"
            return f"{result.line_type}(連続率{result.continuity_ratio:.2f}, 途切れ{result.gap_runs}回)"

        self.get_logger().info(f"[UFLD] 左={fmt(left_result)} | 右={fmt(right_result)}")

    def destroy_node(self):
        self._running = False
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UFLDLaneDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
