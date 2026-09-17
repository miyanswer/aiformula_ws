import os.path as osp
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# MP4 動画で UFLD (Ultra-Fast-Lane-Detection) レーン検出ノード単体をデバッグするための起動ファイル。
# YOLOP パイプラインには触れず、UFLD の検出結果(逆透視変換後のBEV座標・実線/破線/二重線判定)を
# annotated_image とターミナルログの両方で確認する。
#   MP4 -> video_publisher -> .../left_image/undistorted/compressed
#        -> ufld_lane_detector -> annotated_image + ターミナルへ 左右レーンの線種を出力
#        -> rviz2

_CAMERA_TOPIC = "/aiformula_sensing/zed_node/left_image/undistorted"


def _cleanup_old_processes():
    """前回起動のゾンビプロセスを掃除する。"""
    try:
        subprocess.run(
            ["pkill", "-9", "-f",
             "video_publisher|ufld_lane_detector|rviz2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def generate_launch_description():
    _cleanup_old_processes()

    pkg_oit_navigation = get_package_share_directory("oit_navigation")
    default_params_file = osp.join(pkg_oit_navigation, "config", "ufld_params.yaml")
    default_rviz_file = osp.join(pkg_oit_navigation, "config", "ufld_rviz.rviz")

    launch_args = [
        DeclareLaunchArgument(
            "video_path",
            default_value="/aiformula_ws/mp4/shihou_video_2026_08_24_13_51_47.mp4",
            description="デバッグに使う MP4 動画の絶対パス",
        ),
        DeclareLaunchArgument(
            "fps",
            default_value="15.0",
            description="動画の再生フレームレート [FPS]",
        ),
        DeclareLaunchArgument(
            "loop",
            default_value="true",
            description="動画を最後まで再生したら先頭へループするか",
        ),
        DeclareLaunchArgument(
            "use_device",
            default_value="cpu",
            description="推論デバイス: 'cpu' / 'mps' / '0' (CUDA)",
        ),
        DeclareLaunchArgument(
            "weight_path",
            default_value="/aiformula_ws/models/pretrained/ufld_tusimple_r18.pth",
            description="UFLD モデル (.pth) のパス。公式 TuSimple 学習済み重み。",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="ufld_lane_detector のパラメータ YAML",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="RViz2 で annotated_image (レーン点+線種ラベル) を表示するか",
        ),
    ]

    video_publisher_node = Node(
        package="oit_navigation",
        executable="video_publisher",
        name="video_publisher",
        output="screen",
        parameters=[{
            "video_path": LaunchConfiguration("video_path"),
            "topic_name": _CAMERA_TOPIC,
            "frame_id": "zed_left_camera_optical_frame",
            "fps": LaunchConfiguration("fps"),
            "loop": LaunchConfiguration("loop"),
        }],
    )

    ufld_node = Node(
        package="oit_navigation",
        executable="ufld_lane_detector",
        name="ufld_lane_detector",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "input_image_topic": _CAMERA_TOPIC + "/compressed",
                "use_device": LaunchConfiguration("use_device"),
                "weight_path": LaunchConfiguration("weight_path"),
            },
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="ufld_rviz",
        output="screen",
        arguments=["-d", default_rviz_file],
        condition=IfCondition(LaunchConfiguration("rviz")),
        on_exit=Shutdown(),
    )

    return LaunchDescription(
        launch_args
        + [
            video_publisher_node,
            ufld_node,
            rviz_node,
        ]
    )
