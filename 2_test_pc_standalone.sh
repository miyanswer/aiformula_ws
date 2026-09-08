#!/bin/bash
# ==============================================================================
# 2_test_pc_standalone.sh (ルート直下実行ラッパー)
#
# 実機なしでPC単体で動画入力・認識・追従・信号機・RViz2の動作確認を行うスクリプト。
# - macOS / Docker環境: 自動的に Docker コンテナ内で実行します。
# - Ubuntu / 実機環境: ネイティブの ROS 2 環境で直接実行します。
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROS 2 コマンドがネイティブに存在するかチェック (Ubuntu/実機環境)
if command -v ros2 &> /dev/null; then
    echo "========================================="
    echo "  [AI Formula] Running in Native ROS 2"
    echo "========================================="
    bash "${SCRIPT_DIR}/src/aiformula_base/bash/2_test_pc_standalone.sh" "$@"
else
    # macOS 等で Docker 環境を使用する場合
    echo "========================================="
    echo "  [AI Formula] Running inside Docker Container"
    echo "========================================="
    
    # コンテナが起動しているか確認
    if ! docker compose ps --services --filter "status=running" | grep -q "aiformula_ws"; then
        echo "[INFO] Starting Docker container..."
        docker compose up -d
    fi
    
    VIDEO_PATH="${1:-/aiformula_ws/src/ai_formula_oit_2027/data/front_camera_sample.mp4}"
    DEVICE="${2:-cpu}"
    
    echo "  Video : ${VIDEO_PATH}"
    echo "  Device: ${DEVICE}"
    echo "========================================="
    echo "  Web GUI: http://localhost:8080 (ブラウザでRViz2等の画面が確認できます)"
    echo "========================================="

    docker compose exec aiformula_ws bash -c \
        "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch sample_launchers pc_standalone_test.launch.py video_path:=${VIDEO_PATH} use_device:=${DEVICE} rviz:=true"
fi
