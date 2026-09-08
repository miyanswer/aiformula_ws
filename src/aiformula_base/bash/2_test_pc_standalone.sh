#!/bin/bash
# ==============================================================================
# 2_test_pc_standalone.sh
# 
# [PC単体検証用] 実機が手元になくてもPCだけで機能追加やアルゴリズムの動作確認を行うスクリプト。
# MP4動画の再生、YOLOPレーン検出、BEVレーン追従、信号機検出、RViz2による可視化を一括起動します。
#
# 使用例:
#   bash 2_test_pc_standalone.sh
#   bash 2_test_pc_standalone.sh /path/to/custom_video.mp4 cpu
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ROS 2 環境セットアップ
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f "${WS_DIR}/install/setup.bash" ]; then
    source "${WS_DIR}/install/setup.bash"
fi

# デフォルト動画パス
DEFAULT_VIDEO="${WS_DIR}/src/ai_formula_oit_2027/data/front_camera_sample.mp4"
VIDEO_PATH="${1:-$DEFAULT_VIDEO}"
DEVICE="${2:-cuda}"

echo "========================================="
echo "  [AI Formula] PC Standalone Test Mode"
echo "  Video Path : ${VIDEO_PATH}"
echo "  Device     : ${DEVICE}"
echo "  RViz2      : Enabled"
echo "========================================="

# Launchの実行
ros2 launch sample_launchers pc_standalone_test.launch.py \
    video_path:="${VIDEO_PATH}" \
    use_device:="${DEVICE}" \
    rviz:=true
