#!/usr/bin/env python3
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class StandaloneVideoPublisher(Node):
    def __init__(self):
        super().__init__('standalone_video_publisher')
        
        default_topic = "/aiformula_sensing/zed_node/left_image/undistorted"
        default_frame_id = "zed_left_camera_optical_frame"
        
        self.declare_parameter('video_path', '/aiformula_ws/mp4/shihou_video_2026_08_24_13_51_47.mp4')
        self.declare_parameter('topic_name', default_topic)
        self.declare_parameter('frame_id', default_frame_id)
        self.declare_parameter('loop', True)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('resize_width', 0)
        self.declare_parameter('resize_height', 0)
        
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.target_fps = self.get_parameter('fps').get_parameter_value().double_value
        self.resize_width = self.get_parameter('resize_width').get_parameter_value().integer_value
        self.resize_height = self.get_parameter('resize_height').get_parameter_value().integer_value
        
        self.publisher_ = self.create_publisher(Image, self.topic_name, 10)
        self.bridge = CvBridge()
        
        if not os.path.exists(self.video_path):
            alt_path = os.path.join(os.getcwd(), self.video_path.lstrip('/'))
            if os.path.exists(alt_path):
                self.video_path = alt_path
            else:
                self.get_logger().error(f"Video file not found: {self.video_path}")
                return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video: {self.video_path}")
            return
            
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.get_logger().info(
            f"Opened video: {self.video_path} ({width}x{height}, {video_fps:.1f} fps, {total_frames} frames)"
        )
        self.get_logger().info(f"Publishing to topic: {self.topic_name} at {self.target_fps:.1f} fps")
        
        timer_period = 1.0 / self.target_fps if self.target_fps > 0 else 1.0 / 15.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.frame_count = 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.get_logger().info("Video reached end. Looping to start...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                self.get_logger().info("Video playback finished.")
                self.timer.cancel()
                return

        if not ret or frame is None:
            return

        if self.resize_width > 0 and self.resize_height > 0:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        
        self.publisher_.publish(msg)
        self.frame_count += 1
        if self.frame_count % 150 == 0:
            self.get_logger().info(f"Published {self.frame_count} frames to {self.topic_name}")

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StandaloneVideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
