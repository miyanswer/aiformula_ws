import os.path as osp
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('ai_formula_oit_2027')
    default_params_file = osp.join(pkg_dir, 'config', 'controller_params.yaml')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS 2 parameters file for BEV controller',
    )

    bev_lane_tracker_node = Node(
        package='ai_formula_oit_2027',
        executable='bev_lane_tracker',
        name='bev_lane_tracker',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        declare_params_file_cmd,
        bev_lane_tracker_node,
    ])
