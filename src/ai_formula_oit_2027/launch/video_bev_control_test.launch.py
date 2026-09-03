import os.path as osp
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from common_python.launch_util import get_frame_ids_and_topic_names


def generate_launch_description():
    VEHICLE_NAME = "ai_car1"
    CAMERA_NAME = "zedx"
    CAMERA_SN = "SN48311510"
    CAMERA_RESOLUTION = "nHD"

    FRAME_IDS, TOPIC_NAMES = get_frame_ids_and_topic_names()

    pkg_2027_dir = get_package_share_directory("ai_formula_oit_2027")
    sample_vehicle_dir = get_package_share_directory("sample_vehicle")
    object_road_detector_dir = get_package_share_directory("object_road_detector")

    default_params_file = osp.join(pkg_2027_dir, "config", "controller_params.yaml")

    launch_args = [
        DeclareLaunchArgument(
            "video_path",
            default_value="/aiformula_ws/mp4/shihou_video_2026_08_24_13_51_47.mp4",
            description="Absolute path to the MP4 video file",
        ),
        DeclareLaunchArgument(
            "use_device",
            default_value="cpu",
            description="Inference device: 'cpu' or '0' (CUDA GPU)",
        ),
        DeclareLaunchArgument(
            "fps",
            default_value="15.0",
            description="Playback frame rate in FPS",
        ),
        DeclareLaunchArgument(
            "loop",
            default_value="true",
            description="Loop video playback when finished",
        ),
        DeclareLaunchArgument(
            "weight_path",
            default_value="/aiformula_ws/models/shiho_lane_mask_v2_best.pth",
            description="Path to the YOLOP weight pth file",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path to controller params YAML",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Launch RViz2 for visualization",
        ),
    ]

    # 1. TF Broadcaster
    vehicle_tf_broadcaster = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            osp.join(sample_vehicle_dir, "launch", "vehicle_tf_broadcaster.launch.py"),
        ),
        launch_arguments={
            "vehicle_name": VEHICLE_NAME,
            "use_sim_time": "false",
        }.items(),
    )

    # 2. Video Publisher Node (Stream video to camera topic)
    video_publisher_node = Node(
        package="oit_navigation",
        executable="video_publisher",
        name="video_publisher",
        output="screen",
        parameters=[{
            "video_path": LaunchConfiguration("video_path"),
            "topic_name": TOPIC_NAMES["sensing"]["zedx"]["left_image"]["undistorted"],
            "frame_id": FRAME_IDS["zedx"]["left"],
            "fps": LaunchConfiguration("fps"),
            "loop": LaunchConfiguration("loop"),
        }],
    )

    # 3. YOLOP Road / Lane Segmentation Node
    object_road_detector_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            osp.join(object_road_detector_dir, "launch", "object_road_detector.launch.py"),
        ),
        launch_arguments={
            "camera_name": CAMERA_NAME,
            "camera_sn": CAMERA_SN,
            "camera_resolution": CAMERA_RESOLUTION,
            "use_device": LaunchConfiguration("use_device"),
            "weight_path": LaunchConfiguration("weight_path"),
        }.items(),
    )

    # 4. Direct 2D BEV Lane Tracker & Localization Node (NO PointCloud needed!)
    bev_lane_tracker_node = Node(
        package="ai_formula_oit_2027",
        executable="bev_lane_tracker",
        name="bev_lane_tracker",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    # 5. RViz2 Visualization Node
    rviz_config_file = osp.join(pkg_2027_dir, "config", "bev_control.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        launch_args
        + [
            vehicle_tf_broadcaster,
            video_publisher_node,
            object_road_detector_node,
            bev_lane_tracker_node,
            rviz_node,
        ]
    )
