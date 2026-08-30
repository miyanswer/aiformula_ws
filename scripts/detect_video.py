#!/usr/bin/env python3
"""
detect_video.py - 学習済みYOLOモデルを用いた動画推論・再生検証ツール

概要:
  models/ 配下の学習済みモデル（models/<task>.pt 等）を指定した動画ファイル（.mp4等）に適用し、
  リアルタイムで物体検出の認識状況を再生・確認・検証できるツールです。

特徴:
  - モデルと動画の自動検出＆対話的選択メニュー対応
  - Apple Silicon GPU (MPS) / NVIDIA GPU (CUDA) / CPU 自動検出
  - 一時停止、コマ送り/コマ戻し、シーク、信頼度閾値のリアルタイム調整に対応
  - 検出枠付き動画のMP4保存（--save）対応
  - 画面上に見やすいHUD（推論時間、FPS、検出物体数、再生シークバー）を表示

操作キー一覧:
  [SPACE]     : 一時停止 / 再生
  [d] / [→]   : 1フレーム進む（一時停止中）
  [a] / [←]   : 1フレーム戻る（一時停止中）
  [f]         : 5秒早送り
  [r]         : 5秒巻き戻し
  [+] / [-]   : 信頼度閾値(conf)の調整 (±0.05)
  [s]         : 現在の検出フレームを画像として保存
  [q] / [ESC] : 終了

使い方:
  # 対話的にモデルと動画を選択して再生
  make detect-video
  # または python3 scripts/detect_video.py

  # タスクを指定して実行
  make detect-video TASK=traffic_light_red

  # モデルと動画を直接指定して実行
  python3 scripts/detect_video.py --model models/traffic_light_red.pt --video mp4/20260824_124734.mp4

  # 推論結果をMP4動画として保存
  python3 scripts/detect_video.py --task traffic_light_red --save
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


def detect_best_device():
    """実行デバイスの自動判定"""
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def find_available_models() -> List[Tuple[str, str]]:
    """models/ や runs/ 配下の利用可能な学習済みモデルをリストアップ"""
    model_list = []
    seen = set()

    # 1. models/ ディレクトリ
    if os.path.exists("models"):
        for f in sorted(os.listdir("models")):
            if f.endswith(".pt") and not f.startswith("."):
                name = f.replace(".pt", "")
                p = os.path.join("models", f)
                model_list.append((name, p))
                seen.add(name)

    # 2. runs/ 配下の best.pt
    for p in glob.glob("runs/**/weights/best.pt", recursive=True):
        parts = Path(p).parts
        t_name = parts[-3] if len(parts) >= 3 else "custom"
        if t_name not in seen:
            model_list.append((t_name, p))
            seen.add(t_name)

    return model_list


def find_available_videos(task: str = "") -> List[str]:
    """mp4/ や data/tasks/<task>/ 配下の動画ファイルをリストアップ"""
    video_exts = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV")
    search_dirs = []

    if task:
        search_dirs.extend([
            f"data/tasks/{task}/raw_videos",
            f"data/tasks/{task}",
        ])

    search_dirs.extend(["mp4", "data/raw_videos", "data"])

    found_videos = []
    seen_paths = set()

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for ext in video_exts:
            for v_path in sorted(glob.glob(os.path.join(s_dir, "**", ext), recursive=True)):
                abs_p = os.path.abspath(v_path)
                if abs_p not in seen_paths and not os.path.basename(v_path).startswith("."):
                    found_videos.append(v_path)
                    seen_paths.add(abs_p)

    return found_videos


def select_model(task: str, explicit_model: str) -> str:
    """モデルパスの決定または対話選択"""
    if explicit_model and os.path.isfile(explicit_model):
        return explicit_model

    if task:
        task_cands = [
            f"models/{task}.pt",
            f"runs/detect/runs/train/{task}/weights/best.pt",
            f"runs/train/{task}/weights/best.pt",
        ]
        for c in task_cands:
            if os.path.isfile(c):
                return c

    available = find_available_models()
    if not available:
        if os.path.isfile("yolo11n.pt"):
            return "yolo11n.pt"
        raise FileNotFoundError("学習済みモデル (.pt) が見つかりませんでした。先に make train を実行してください。")

    if len(available) == 1:
        return available[0][1]

    # 対話メニュー
    print("\n" + "=" * 65)
    print("🤖 利用可能な学習済みモデル一覧:")
    print("=" * 65)
    for idx, (name, path) in enumerate(available):
        print(f"  [{idx + 1}] {name:<25} ({path})")
    print(f"  [0] 一般事前学習モデル (yolo11n.pt)")
    print("=" * 65)

    sys.stdout.write(f"👉 使用するモデル番号を選択してください [1-{len(available)}, デフォルト=1]: ")
    sys.stdout.flush()
    choice = sys.stdin.readline().strip()

    if choice.isdigit():
        val = int(choice)
        if 1 <= val <= len(available):
            return available[val - 1][1]
        elif val == 0:
            return "yolo11n.pt"

    return available[0][1]


def select_video(task: str, explicit_video: str) -> str:
    """動画パスの決定または対話選択"""
    if explicit_video and os.path.isfile(explicit_video):
        return explicit_video

    available = find_available_videos(task)
    if not available:
        raise FileNotFoundError("動画ファイル (.mp4, .mov 等) が mp4/ や data/tasks/ に見つかりませんでした。")

    if len(available) == 1:
        return available[0]

    # 対話メニュー
    print("\n" + "=" * 65)
    print("🎥 利用可能な動画ファイル一覧:")
    print("=" * 65)
    for idx, path in enumerate(available[:15]):  # 最大15件表示
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{idx + 1:2d}] {os.path.basename(path):<40} ({size_mb:.1f} MB) -> {path}")
    if len(available) > 15:
        print(f"  ... 他 {len(available) - 15} 件の動画があります")
    print("=" * 65)

    sys.stdout.write(f"👉 再生する動画番号を選択してください [1-{min(len(available), 15)}, デフォルト=1]: ")
    sys.stdout.flush()
    choice = sys.stdin.readline().strip()

    if choice.isdigit():
        val = int(choice)
        if 1 <= val <= len(available):
            return available[val - 1]

    return available[0]


def parse_args():
    parser = argparse.ArgumentParser(description="学習済みYOLOモデルによる動画再生 & 物体認識テスト")
    parser.add_argument("--task", type=str, default="", help="タスク名 (例: traffic_light_red, jinmen_dog)")
    parser.add_argument("--model", type=str, default="", help="モデルファイルパス (.pt)")
    parser.add_argument("--video", "--source", type=str, default="", help="動画ファイルパス (.mp4, .mov等)")
    parser.add_argument("--conf", type=float, default=0.35, help="初期信頼度しきい値 (デフォルト: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoUしきい値 (デフォルト: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="推論画像サイズ (デフォルト: 640)")
    parser.add_argument("--device", type=str, default="", help="実行デバイス (空欄で自動検出: mps / cuda / cpu)")
    parser.add_argument("--save", action="store_true", help="認識結果を動画ファイルとして保存する")
    parser.add_argument("--output-dir", type=str, default="data/detection_outputs", help="保存先ディレクトリ")
    return parser.parse_args()


def format_time(seconds: float) -> str:
    """秒数を mm:ss 形式にフォーマット"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def main():
    args = parse_args()

    # Ultralytics のインポート
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[エラー] 'ultralytics' ライブラリが見つかりません。")
        print("  pip3 install -r requirements-yolo.txt を実行してください。")
        return 1

    device = args.device if args.device else detect_best_device()

    # モデルと動画の選択
    try:
        model_path = select_model(args.task, args.model)
        video_path = select_video(args.task, args.video)
    except FileNotFoundError as e:
        print(f"\n[エラー] {e}")
        return 1

    print("\n" + "=" * 65)
    print("=== YOLO 動画再生 & 認識検証ツール ===")
    print("=" * 65)
    print(f"🎯 使用モデル   : {model_path}")
    print(f"🎥 入力動画     : {video_path}")
    print(f"🎚️ 信頼度閾値   : {args.conf:.2f} (キーボード [+] [-] でリアルタイム変更可)")
    print(f"⚡ 実行デバイス : {device}")
    print(f"💾 動画保存     : {'有効 (' + args.output_dir + ')' if args.save else '無効'}")
    print("=" * 65)

    # モデルのロード
    print(f"\n🧠 モデル '{model_path}' をロード中...")
    model = YOLO(model_path)
    model_classes = model.names
    print(f"🏷️ 認識対象クラス: {model_classes}")

    # 動画のオープン
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[エラー] 動画を開けませんでした: {video_path}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps if total_frames > 0 else 0

    print(f"📹 動画情報: {frame_width}x{frame_height} @ {video_fps:.1f}fps, 全 {total_frames} フレーム ({format_time(duration_sec)})")

    # 保存用VideoWriterの設定
    video_writer = None
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        v_basename = os.path.splitext(os.path.basename(video_path))[0]
        m_basename = os.path.splitext(os.path.basename(model_path))[0]
        save_path = os.path.join(args.output_dir, f"{v_basename}_{m_basename}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, (frame_width, frame_height))
        print(f"💾 出力動画パス: {save_path}")

    window_name = f"YOLO Video Detection - {os.path.basename(video_path)} ({os.path.basename(model_path)})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n🎮 操作ガイド:")
    print("  [SPACE]     : 一時停止 / 再生")
    print("  [d] / [→]   : 1フレーム進む（一時停止中）")
    print("  [a] / [←]   : 1フレーム戻る（一時停止中）")
    print("  [f] / [r]   : 5秒早送り / 5秒巻き戻し")
    print("  [+] / [-]   : 信頼度閾値の調整 (±0.05)")
    print("  [s]         : 現在のフレームを画像として保存")
    print("  [q] / [ESC] : 終了\n")

    current_conf = args.conf
    is_paused = False
    step_forward = False
    step_backward = False

    fps_history = []
    fps_time = time.time()

    while True:
        if not is_paused or step_forward or step_backward:
            if step_backward:
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_pos - 2))
                step_backward = False

            ret, frame = cap.read()
            if not ret:
                print("\n🏁 動画の末尾に到達しました。（ループ再生します）")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            step_forward = False

            # 推論実行
            infer_start = time.time()
            results = model.predict(
                source=frame,
                conf=current_conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            infer_time_ms = (time.time() - infer_start) * 1000

            # FPS計算
            now = time.time()
            fps_val = 1.0 / max(1e-5, now - fps_time)
            fps_time = now
            fps_history.append(fps_val)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # バウンディングボックスの描画
            annotated_frame = np.ascontiguousarray(results[0].plot().copy())
            boxes = results[0].boxes
            n_detections = len(boxes) if boxes is not None else 0

            # 検出クラスごとの集計
            detected_counts = {}
            if boxes is not None:
                for cls_idx in boxes.cls.cpu().numpy():
                    c_name = model_classes.get(int(cls_idx), f"class_{int(cls_idx)}")
                    detected_counts[c_name] = detected_counts.get(c_name, 0) + 1

            # 画面上部: HUD情報のオーバーレイ描画
            h, w = annotated_frame.shape[:2]
            
            # 半透明背景バー
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # 1行目: モデル・動画・推論速度・FPS
            cur_time_sec = current_frame_idx / video_fps
            line1 = f"Model: {os.path.basename(model_path)} | Infer: {infer_time_ms:.1f}ms | {avg_fps:.1f} FPS | Conf: {current_conf:.2f}"
            cv2.putText(annotated_frame, line1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # 2行目: 検出サマリー
            det_str = ", ".join([f"{k}: {v}" for k, v in detected_counts.items()]) if detected_counts else "None"
            status_tag = " [PAUSED]" if is_paused else " [PLAYING]"
            line2 = f"Detections ({n_detections}): {det_str}{status_tag}"
            cv2.putText(annotated_frame, line2, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if n_detections > 0 else (180, 180, 180), 2, cv2.LINE_AA)

            # 画面下部: プログレスバー & タイムスタンプ
            progress_ratio = current_frame_idx / max(1, total_frames)
            bar_w = int(w * progress_ratio)
            cv2.rectangle(annotated_frame, (0, h - 35), (bar_w, h), (0, 165, 255), -1)
            
            time_str = f"Frame: {current_frame_idx}/{total_frames} ({format_time(cur_time_sec)} / {format_time(duration_sec)}) | [SPACE]: Pause, [Q]: Quit"
            cv2.putText(annotated_frame, time_str, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if video_writer:
                video_writer.write(annotated_frame)

        cv2.imshow(window_name, annotated_frame)

        # キー入力待機
        wait_ms = 0 if is_paused else max(1, int(1000.0 / video_fps) - 5)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), 27):  # q または ESC
            print("\n👋 再生を終了しました。")
            break
        elif key == ord(" "):      # SPACE
            is_paused = not is_paused
            print(f"⏸️ 一時停止: {'ON' if is_paused else 'OFF'}")
        elif key in (ord("d"), 83):  # d または 右矢印
            is_paused = True
            step_forward = True
        elif key in (ord("a"), 81):  # a または 左矢印
            is_paused = True
            step_backward = True
        elif key == ord("f"):      # 5秒早送り
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur_pos + int(video_fps * 5)))
            print("⏩ 5秒早送り")
        elif key == ord("r"):      # 5秒巻き戻し
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur_pos - int(video_fps * 5)))
            print("⏪ 5秒巻き戻し")
        elif key in (ord("+"), ord("=")):  # 信頼度UP
            current_conf = min(0.95, current_conf + 0.05)
            print(f"🎚️ 信頼度閾値 UP: {current_conf:.2f}")
        elif key in (ord("-"), ord("_")):  # 信頼度DOWN
            current_conf = max(0.05, current_conf - 0.05)
            print(f"🎚️ 信頼度閾値 DOWN: {current_conf:.2f}")
        elif key == ord("s"):      # スクリーンショット保存
            ss_dir = os.path.join(args.output_dir, "screenshots")
            os.makedirs(ss_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ss_path = os.path.join(ss_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(ss_path, annotated_frame)
            print(f"📸 スクリーンショットを保存しました: {ss_path}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
