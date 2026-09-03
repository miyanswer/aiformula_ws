#!/usr/bin/env python3
"""
Pure Pursuit Controller Node Entrypoint for AI Formula OIT 2027.
"""

import rclpy
from ai_formula_oit_2027.bev_lane_tracker_node import BEVLaneTrackerNode


def main(args=None):
    rclpy.init(args=args)
    node = BEVLaneTrackerNode()
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
