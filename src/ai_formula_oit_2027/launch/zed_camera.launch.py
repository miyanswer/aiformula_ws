import os.path as osp
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_zed_node(context):
    camera_model = LaunchConfiguration("camera_model").perform(context)
    grab_resolution = LaunchConfiguration("grab_resolution").perform(context)
    grab_frame_rate = int(LaunchConfiguration("grab_frame_rate").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)

    # zed_wrapper パッケージ内の設定ファイル探索 (公式ラッパーのパス)
    try:
        zed_wrapper_dir = get_package_share_directory("zed_wrapper")
        default_common_yaml = osp.join(zed_wrapper_dir, "config", "common.yaml")
        default_camera_yaml = osp.join(zed_wrapper_dir, "config", f"{camera_model}.yaml")
    except Exception:
        default_common_yaml = ""
        default_camera_yaml = ""

    config_common_path = LaunchConfiguration("config_common_path").perform(context)
    if not config_common_path and default_common_yaml and osp.isfile(default_common_yaml):
        config_common_path = default_common_yaml

    config_camera_path = LaunchConfiguration("config_camera_path").perform(context)
    if not config_camera_path and default_camera_yaml and osp.isfile(default_camera_yaml):
        config_camera_path = default_camera_yaml

    params_list = []
    if config_common_path and osp.isfile(config_common_path):
        params_list.append(config_common_path)
    if config_camera_path and osp.isfile(config_camera_path):
        params_list.append(config_camera_path)

    params_list.append({
        "use_sim_time": use_sim_time.lower() == "true",
        "general.camera_model": camera_model,
        "general.grab_resolution": grab_resolution,
        "general.grab_frame_rate": grab_frame_rate,
    })

    return [
        Node(
            package="zed_wrapper",
            namespace="/aiformula_sensing",
            executable="zed_wrapper",
            name="zed_node",
            output="screen",
            parameters=params_list,
            remappings=[
                ("~/left/image_rect_color", "/aiformula_sensing/zed_node/left_image/undistorted"),
                ("~/right/image_rect_color", "/aiformula_sensing/zed_node/right_image/undistorted"),
                ("~/imu/data", "/aiformula_sensing/zed_node/imu"),
                ("~/depth/depth_registered", "/aiformula_sensing/zed_node/depth/depth_registered"),
                ("~/point_cloud/cloud_registered", "/aiformula_sensing/zed_node/point_cloud/cloud_registered"),
            ],
        )
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "camera_model",
            default_value="zedx",
            description="ZED Camera model: 'zedx', 'zed2i', 'zed2', 'zedm', 'zed'",
        ),
        DeclareLaunchArgument(
            "grab_resolution",
            default_value="HD1080",
            description="Native grab resolution: 'HD1200', 'HD1080', 'SVGA'",
        ),
        DeclareLaunchArgument(
            "grab_frame_rate",
            default_value="30",
            description="Grab frame rate in FPS: '60', '30', '15'",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Enable simulation time mode.",
        ),
        DeclareLaunchArgument(
            "config_common_path",
            default_value="",
            description="Optional path to custom common.yaml",
        ),
        DeclareLaunchArgument(
            "config_camera_path",
            default_value="",
            description="Optional path to custom camera config YAML",
        ),
    ]

    return LaunchDescription(
        [
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="1"),
            *launch_args,
            OpaqueFunction(function=get_zed_node),
        ]
    )
