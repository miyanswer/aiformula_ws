#!/usr/bin/env python3
"""
Webカメラから FHD (1920x1080), 15fps で動画を撮影・保存するスクリプト。

操作方法:
  [r] キー: 録画の開始 / 停止 (トグル)
  [s] キー: 現在のフレームを静止画 (PNG) として保存
  [q] キー: 終了
"""

import argparse
import os
import sys
import time
from datetime import datetime
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="WebカメラからのFHD 15fps動画録画スクリプト")
    parser.add_argument("--task", type=str, default="default", help="タスク名 (例: traffic_light, t_junction, crosswalk, jinmen_dog)")
    parser.add_argument("--camera-id", type=int, default=0, help="カメラデバイスID (デフォルト: 0)")
    parser.add_argument("--width", type=int, default=1920, help="解像度 幅 (デフォルト: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="解像度 高さ (デフォルト: 1080)")
    parser.add_argument("--fps", type=float, default=15.0, help="目標FPS (デフォルト: 15.0)")
    parser.add_argument("--output-dir", type=str, default="", help="動画保存先ディレクトリ (省略時は data/tasks/<task>/raw_videos)")
    parser.add_argument("--auto-start", action="store_true", help="起動と同時に自動で録画を開始する")
    parser.add_argument("--duration", type=float, default=0, help="自動停止までの秒数 (0の場合は手動停止)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 保存先ディレクトリの決定
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("data", "tasks", args.task, "raw_videos")

    os.makedirs(output_dir, exist_ok=True)
    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    print(f"=== Webカメラ録画ツール ===")
    print(f"タスク名: {args.task}")
    print(f"カメラID: {args.camera_id}")
    print(f"目標解像度: {args.width}x{args.height}")
    print(f"目標FPS: {args.fps}")
    print(f"保存先: {output_dir}")
    print("==========================")

    # カメラの初期化 (macOSではAVFoundationバックエンドを推奨)
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(args.camera_id, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"[エラー] カメラ ID {args.camera_id} を開くことができませんでした。")
        print("・カメラが接続されているか確認してください。")
        print("・Macの「システム設定」->「プライバシーとセキュリティ」->「カメラ」で権限が許可されているか確認してください。")
        return 1

    # 解像度・FPSを設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 実際のカメラ取得プロパティを確認
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[情報] カメラ設定完了: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
    if actual_width != args.width or actual_height != args.height:
        print(f"[警告] カメラが要求解像度 ({args.width}x{args.height}) に非対応のため、{actual_width}x{actual_height} で撮影します。")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    is_recording = args.auto_start
    record_start_time = None
    recorded_frames = 0
    current_video_path = None

    frame_interval = 1.0 / args.fps
    last_frame_time = time.time()
    fps_display = 0.0
    fps_calc_count = 0
    fps_calc_start = time.time()

    def start_recording():
        nonlocal writer, is_recording, record_start_time, recorded_frames, current_video_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_video_path = os.path.join(output_dir, f"video_{timestamp}_{actual_width}x{actual_height}_{int(args.fps)}fps.mp4")
        writer = cv2.VideoWriter(current_video_path, fourcc, args.fps, (actual_width, actual_height))
        is_recording = True
        record_start_time = time.time()
        recorded_frames = 0
        print(f"[録画開始] -> {current_video_path}")

    def stop_recording():
        nonlocal writer, is_recording, record_start_time, recorded_frames, current_video_path
        if writer is not None:
            writer.release()
            writer = None
        duration = time.time() - record_start_time if record_start_time else 0
        print(f"[録画停止] 保存完了: {current_video_path} ({duration:.1f}秒, {recorded_frames}フレーム)")
        is_recording = False
        record_start_time = None

    if is_recording:
        start_recording()

    print("\n操作方法:")
    print("  [r] キー: 録画 開始 / 停止")
    print("  [s] キー: 静止画 (PNG) 保存")
    print("  [q] キー: 終了\n")

    window_name = "Camera Stream (15fps FHD) - Press [r] to Record, [q] to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            current_time = time.time()
            elapsed_since_last = current_time - last_frame_time

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[エラー] フレーム取得に失敗しました。")
                time.sleep(0.05)
                continue

            last_frame_time = current_time

            # FPS計算
            fps_calc_count += 1
            if current_time - fps_calc_start >= 1.0:
                fps_display = fps_calc_count / (current_time - fps_calc_start)
                fps_calc_count = 0
                fps_calc_start = current_time

            # 録画中ならフレームを保存
            if is_recording and writer is not None:
                writer.write(frame)
                recorded_frames += 1

                # 自動停止条件
                if args.duration > 0 and (current_time - record_start_time) >= args.duration:
                    print(f"[情報] 指定時間 ({args.duration}秒) に達したため録画を停止します。")
                    stop_recording()

            # プレビュー表示用オーバーレイ作成
            display_frame = frame.copy()
            overlay_h, overlay_w = display_frame.shape[:2]

            # ステータス情報テキスト
            if is_recording:
                rec_duration = current_time - record_start_time if record_start_time else 0
                mins, secs = divmod(int(rec_duration), 60)
                status_text = f"REC [{mins:02d}:{secs:02d}] - {recorded_frames} frames"
                # 赤丸インジケータ
                cv2.circle(display_frame, (30, 35), 12, (0, 0, 255), -1)
                cv2.putText(display_frame, status_text, (50, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                status_text = "STANDBY (Press 'r' to Record)"
                cv2.circle(display_frame, (30, 35), 12, (100, 100, 100), -1)
                cv2.putText(display_frame, status_text, (50, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

            info_text = f"{actual_width}x{actual_height} | Live: {fps_display:.1f} FPS (Target: {args.fps:.0f} FPS)"
            cv2.putText(display_frame, info_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            # キー入力処理 (FPSレートを意識したウェイト)
            wait_ms = max(1, int((frame_interval - (time.time() - current_time)) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q") or key == 27:  # 'q' or ESC
                break
            elif key == ord("r"):  # 録画トグル
                if is_recording:
                    stop_recording()
                else:
                    start_recording()
            elif key == ord("s"):  # スクリーンショット
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                ss_path = os.path.join(screenshot_dir, f"capture_{ts}.png")
                cv2.imwrite(ss_path, frame)
                print(f"[静止画保存] -> {ss_path}")

    finally:
        if is_recording:
            stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        print("[完了] プログラムを終了しました。")


if __name__ == "__main__":
    sys.exit(main())
