#!/usr/bin/env python3
"""
traffic_light_detector_node.py - 信号機検出 & 垂直奥行き推定 ROS 2 ノード

概要:
    YOLOモデル (traffic_light.pt) を用いてカメラ画像から信号機を検出し、
    バウンディングボックスの高さから垂直奥行き Z [m] を幾何学的に逆算して配信・可視化します。
    
計算式:
    Z = (f_y * H_real) / h_px
    (角度の計算は行わず、ピンホールの垂直相似から直接光軸方向の奥行きを算出)
"""

import os
import json
import time
from typing import Optional, List, Dict, Any

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32, String

from ai_formula_oit_2027.utils.traffic_light_depth import TrafficLightDepthEstimator


class TrafficLightDetectorNode(Node):
    """YOLO 信号機検出 & 垂直奥行き逆算ノード"""

    def __init__(self):
        super().__init__('traffic_light_detector')

        self._declare_parameters()
        self._load_parameters()

        self.cv_bridge = CvBridge()

        # 奥行き計算インスタンスの初期化
        self.depth_estimator = TrafficLightDepthEstimator(
            real_height=self.real_height,
            focal_length_y=self.focal_length_y,
            min_bbox_height=self.min_bbox_height,
            max_valid_distance=self.max_valid_distance,
            smoothing_alpha=self.smoothing_alpha,
        )

        # YOLOモデルのロード
        self.model = None
        self._init_yolo_model()

        # パブリッシャ
        self.pub_debug_image = self.create_publisher(Image, self.debug_image_topic, 10)
        self.pub_nearest_distance = self.create_publisher(Float32, f"{self.base_topic}/nearest_distance", 10)
        self.pub_red_distance = self.create_publisher(Float32, f"{self.base_topic}/red_distance", 10)
        self.pub_green_distance = self.create_publisher(Float32, f"{self.base_topic}/green_distance", 10)
        self.pub_status = self.create_publisher(String, f"{self.base_topic}/status", 10)

        # サブスクライバ
        self.sub_image = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10,
        )

        self.get_logger().info(
            f"🚦 TrafficLightDetectorNode 初期化完了:\n"
            f"  - Model Path    : {self.model_path}\n"
            f"  - Image Topic   : {self.image_topic}\n"
            f"  - Real Height   : {self.real_height} m (32cm正方形)\n"
            f"  - Focal Length Y: {self.focal_length_y} px\n"
            f"  - Device        : {self.device}\n"
            f"  - Conf Threshold: {self.conf_threshold}"
        )

    def _declare_parameters(self):
        """ROS 2 パラメータ宣言"""
        self.declare_parameter('model_path', 'models/traffic_light.pt')
        self.declare_parameter('image_topic', '/aiformula_sensing/zed_node/left_image/undistorted')
        self.declare_parameter('debug_image_topic', '/aiformula_perception/traffic_light/debug_image')

        self.declare_parameter('base_topic', '/aiformula_perception/traffic_light')
        self.declare_parameter('real_height', 0.32)
        self.declare_parameter('focal_length_y', 2256.0)

        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('min_bbox_height', 4.0)
        self.declare_parameter('max_valid_distance', 50.0)
        self.declare_parameter('smoothing_alpha', 0.7)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_debug_image', True)

    def _load_parameters(self):
        """パラメータの取得"""
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        self.base_topic = self.get_parameter('base_topic').get_parameter_value().string_value
        self.real_height = self.get_parameter('real_height').get_parameter_value().double_value
        self.focal_length_y = self.get_parameter('focal_length_y').get_parameter_value().double_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.min_bbox_height = self.get_parameter('min_bbox_height').get_parameter_value().double_value
        self.max_valid_distance = self.get_parameter('max_valid_distance').get_parameter_value().double_value
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value

    def _init_yolo_model(self):
        """YOLOモデルのロード"""
        # パスの存在確認（相対パス、ワークスペースルート基準などを探索）
        resolved_path = self.model_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(os.getcwd(), self.model_path),
                f"/aiformula_ws/{self.model_path}",
                os.path.expanduser(f"~/{self.model_path}"),
                "models/traffic_light.pt",
                "/aiformula_ws/models/traffic_light.pt",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        self.get_logger().info(f"🧠 YOLOモデルをロード中: {resolved_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(resolved_path)
            self.get_logger().info(f"✅ モデルロード成功! クラス: {self.model.names}")
        except Exception as e:
            self.get_logger().error(f"❌ YOLOモデルのロードに失敗しました: {e}")
            self.model = None

    def _image_callback(self, msg: Image):
        """カメラ画像コールバック"""
        if self.model is None:
            return

        start_time = time.time()

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge変換エラー: {e}")
            return

        # YOLO 推論実行
        results = self.model.predict(
            source=cv_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        nearest_depth: Optional[float] = None
        nearest_red_depth: Optional[float] = None
        nearest_green_depth: Optional[float] = None

        annotated_image = cv_image.copy() if self.publish_debug_image else None

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                x1, y1, x2, y2 = xyxy
                bbox_h = float(y2 - y1)
                bbox_w = float(x2 - x1)

                # 垂直奥行き Z の計算 (Z = f_y * H_real / h_px)
                # track_id はバウンディングボックスの並び順 (i) を簡易指定
                depth_z = self.depth_estimator.estimate_from_bbox((x1, y1, x2, y2), track_id=i)

                det_info = {
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "bbox": [round(float(v), 1) for v in [x1, y1, x2, y2]],
                    "bbox_height_px": round(bbox_h, 1),
                    "depth_z_m": round(depth_z, 3) if depth_z is not None else None,
                }
                detections.append(det_info)

                # 最近傍距離の更新
                if depth_z is not None:
                    if nearest_depth is None or depth_z < nearest_depth:
                        nearest_depth = depth_z

                    if "red" in cls_name.lower():
                        if nearest_red_depth is None or depth_z < nearest_red_depth:
                            nearest_red_depth = depth_z
                    elif "green" in cls_name.lower():
                        if nearest_green_depth is None or depth_z < nearest_green_depth:
                            nearest_green_depth = depth_z

                # デバッグ描画
                if annotated_image is not None:
                    self._draw_detection(annotated_image, x1, y1, x2, y2, cls_name, conf, bbox_h, depth_z)

        infer_time_ms = (time.time() - start_time) * 1000.0

        # ROS 2 トピック配信
        self._publish_results(msg.header, detections, nearest_depth, nearest_red_depth, nearest_green_depth)

        # デバッグ画像の配信
        if self.publish_debug_image and annotated_image is not None:
            self._draw_hud(annotated_image, infer_time_ms, detections, nearest_depth)
            try:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                debug_msg.header = msg.header
                self.pub_debug_image.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"デバッグ画像配信エラー: {e}")

    def _draw_detection(
        self,
        img: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cls_name: str,
        conf: float,
        bbox_h: float,
        depth_z: Optional[float],
    ):
        """検出枠と推定距離の描画"""
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        # クラス別カラー (BGR)
        if "red" in cls_name.lower():
            color = (0, 0, 255)       # 赤
        elif "green" in cls_name.lower():
            color = (0, 255, 0)       # 緑
        else:
            color = (0, 255, 255)     # 黄色

        # バウンディングボックス
        cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color, 2)

        # ラベルテキスト: "red: 0.89 | Z: 4.25m"
        if depth_z is not None:
            label = f"{cls_name} {conf:.2f} | Z={depth_z:.2f}m (h={bbox_h:.0f}px)"
        else:
            label = f"{cls_name} {conf:.2f} | h={bbox_h:.0f}px"

        # ラベル背景
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (ix1, max(0, iy1 - lh - 8)), (ix1 + lw + 6, iy1), color, -1)
        cv2.putText(img, label, (ix1 + 3, iy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, infer_ms: float, detections: List[Dict[str, Any]], nearest_z: Optional[float]):
        """画面上部のHUD情報描画"""
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 40), (25, 25, 25), -1)

        z_str = f"{nearest_z:.2f} m" if nearest_z is not None else "---"
        hud_text = f"Traffic Light: {len(detections)} det | Nearest Z: {z_str} | Infer: {infer_ms:.1f}ms | (fy={self.focal_length_y}, H={self.real_height}m)"
        cv2.putText(img, hud_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    def _publish_results(
        self,
        header: Header,
        detections: List[Dict[str, Any]],
        nearest_depth: Optional[float],
        nearest_red: Optional[float],
        nearest_green: Optional[float],
    ):
        """推定距離とステータスの配信"""
        # 1. 最も近い信号機の距離
        if nearest_depth is not None:
            msg_nearest = Float32(data=float(nearest_depth))
            self.pub_nearest_distance.publish(msg_nearest)

        # 2. 赤信号の距離
        if nearest_red is not None:
            msg_red = Float32(data=float(nearest_red))
            self.pub_red_distance.publish(msg_red)

        # 3. 緑信号の距離
        if nearest_green is not None:
            msg_green = Float32(data=float(nearest_green))
            self.pub_green_distance.publish(msg_green)

        # 4. JSONステータス配信
        status_payload = {
            "timestamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
            "num_detections": len(detections),
            "nearest_depth_m": nearest_depth,
            "nearest_red_depth_m": nearest_red,
            "nearest_green_depth_m": nearest_green,
            "detections": detections,
        }
        status_msg = String(data=json.dumps(status_payload))
        self.pub_status.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
