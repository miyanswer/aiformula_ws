import os.path as osp
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("ai_formula_oit_2027")
    default_params_file = osp.join(pkg_dir, "config", "traffic_light_params.yaml")

    launch_args = [
        DeclareLaunchArgument(
            "launch_zed",
            default_value="false",
            description="Launch ZED camera alongside detector (true/false)",
        ),
        DeclareLaunchArgument(
            "camera_model",
            default_value="zedx",
            description="ZED Camera model: 'zedx', 'zed2i', 'zed2', 'zedm'",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path to traffic light detector params YAML",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/aiformula_sensing/zed_node/left_image/undistorted",
            description="Input camera image topic",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="models/traffic_light.pt",
            description="Path to YOLO traffic light model (.pt)",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cpu",
            description="Inference device: 'cpu', 'mps', or '0'",
        ),
    ]

    # ZED Camera Launch (Conditional)
    zed_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            osp.join(pkg_dir, "launch", "zed_camera.launch.py")
        ),
        launch_arguments={
            "camera_model": LaunchConfiguration("camera_model"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_zed")),
    )

    traffic_light_node = Node(
        package="ai_formula_oit_2027",
        executable="traffic_light_detector",
        name="traffic_light_detector",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
            }
        ],
    )

    return LaunchDescription(launch_args + [zed_camera_launch, traffic_light_node])

