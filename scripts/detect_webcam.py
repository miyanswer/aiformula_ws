#!/usr/bin/env python3
"""
学習済みYOLOモデル (models/<task>.pt や best.pt 等) を用いて、Webカメラや動画に対してリアルタイム物体認識を行うスクリプト。

操作方法:
  [q] または [ESC]: 終了
  [s]: 検出結果画像を保存
  [SPACE]: 一時停止 / 再開
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch


def detect_best_device():
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def find_model_path(task: str, explicit_model: str) -> str:
    # 1. 明示的に指定されたモデルパスが存在する場合
    if explicit_model and explicit_model != "yolo11n.pt" and os.path.exists(explicit_model):
        return explicit_model

    # 2. タスク名が指定されている場合
    if task:
        candidates = [
            f"models/{task}.pt",
            f"runs/train/{task}/weights/best.pt",
            f"runs/detect/runs/train/{task}/weights/best.pt",
            f"data/tasks/{task}/best.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    # 3. models/ ディレクトリ内のモデル一覧を探索
    models_dir = "models"
    available_models = []
    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt") and not f.startswith("."):
                available_models.append((f.replace(".pt", ""), os.path.join(models_dir, f)))

    # 4. runs/train 配下のモデルを探索
    runs_dir = "runs/train"
    if os.path.exists(runs_dir):
        for t_name in sorted(os.listdir(runs_dir)):
            bp = os.path.join(runs_dir, t_name, "weights", "best.pt")
            if os.path.isfile(bp) and (t_name, bp) not in available_models:
                available_models.append((t_name, bp))

    # モデルが1つだけならそれを自動選択
    if len(available_models) == 1:
        return available_models[0][1]

    # 複数ある場合は選択メニューを表示 (ターミナル対話)
    if len(available_models) > 1 and sys.stdin.isatty():
        print("\n" + "=" * 60)
        print("🤖 利用可能な学習済みモデル一覧:")
        for idx, (t_name, m_path) in enumerate(available_models):
            print(f"  [{idx + 1}] {t_name}  ({m_path})")
        print(f"  [0] 一般事前学習モデル (yolo11n.pt)")
        print("=" * 60)

        sys.stdout.write(f"👉 実行するモデル番号を選択してください [1-{len(available_models)}, デフォルト=1]: ")
        sys.stdout.flush()
        user_choice = sys.stdin.readline().strip()

        if user_choice.isdigit():
            val = int(user_choice)
            if 1 <= val <= len(available_models):
                return available_models[val - 1][1]
            elif val == 0:
                return "yolo11n.pt"
        elif user_choice == "":
            return available_models[0][1]

    # フォールバック候補
    fallback_candidates = [
        "models/jinmen_dog.pt",
        "runs/train/yolo_custom/weights/best.pt",
        "runs/detect/runs/train/yolo_custom/weights/best.pt",
    ]
    for c in fallback_candidates:
        if os.path.exists(c):
            return c

    return explicit_model if explicit_model else "yolo11n.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Webカメラ/動画でのリアルタイムYOLO物体認識")
    parser.add_argument("--task", type=str, default="", help="タスク名 (例: traffic_light, t_junction, crosswalk, jinmen_dog)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="モデル重みファイルパス (省略時はタスク名から自動解決)")
    parser.add_argument("--source", type=str, default="0", help="カメラID (例: 0) または 動画ファイルパス (例: data/raw_videos/sample.mp4)")
    parser.add_argument("--conf", type=float, default=0.40, help="信頼度しきい値 (0.0〜1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoUしきい値")
    parser.add_argument("--imgsz", type=int, default=640, help="推論画像サイズ (640推奨)")
    parser.add_argument("--device", type=str, default="", help="実行デバイス (mps / cuda / cpu)")
    parser.add_argument("--save", action="store_true", help="推論結果の動画を保存する")
    parser.add_argument("--output-dir", type=str, default="data/detection_outputs", help="保存先ディレクトリ")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[エラー] 'ultralytics' ライブラリが見つかりません。")
        print("以下を実行してインストールしてください:")
        print("  pip3 install -r requirements-yolo.txt")
        return 1

    device = args.device if args.device else detect_best_device()
    model_path = find_model_path(args.task, args.model)

    print("=" * 60)
    print("=== YOLO リアルタイム物体認識 ===")
    print("=" * 60)
    print(f"🎯 対象モデル   : {model_path}")
    print(f"📹 入力ソース   : {args.source}")
    print(f"🎚️ 信頼度しきい値: {args.conf}")
    print(f"⚡ 使用デバイス : {device}")
    print("=" * 60)

    # モデルのロード
    model = YOLO(model_path)

    # 入力ソースの判定
    is_cam = args.source.isdigit()
    src = int(args.source) if is_cam else args.source

    if is_cam and sys.platform == "darwin":
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[エラー] 入力ソースを開けませんでした: {args.source}")
        return 1

    # カメラの場合はFHD/高解像度を要求
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 15.0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    writer = None
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"detect_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (actual_w, actual_h))
        print(f"[情報] 検出結果を保存します: {out_path}")

    window_name = f"YOLO Detection - {os.path.basename(model_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fps_display = 0.0
    fps_count = 0
    fps_start = time.time()
    paused = False

    print("\n操作方法:")
    print("  [q] または [ESC]: 終了")
    print("  [s]: 検出フレームを保存")
    print("  [SPACE]: 一時停止 / 再開\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if not is_cam:
                        print("[情報] 動画の終端に達しました。")
                        break
                    continue

                # YOLO 推論
                infer_start = time.time()
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )
                infer_ms = (time.time() - infer_start) * 1000

                # 描画済みフレームの取得 (描画可能な書き込み可能コピーを作成)
                annotated_frame = np.ascontiguousarray(results[0].plot().copy())

                # FPS計算
                fps_count += 1
                curr_time = time.time()
                if curr_time - fps_start >= 0.5:
                    fps_display = fps_count / (curr_time - fps_start)
                    fps_count = 0
                    fps_start = curr_time

                # 検出オブジェクト数集計
                boxes = results[0].boxes
                n_detections = len(boxes) if boxes is not None else 0

                # UI オーバーレイ
                info_line1 = f"Model: {os.path.basename(model_path)} | Infer: {infer_ms:.1f}ms ({fps_display:.1f} FPS) | Detections: {n_detections}"
                cv2.putText(annotated_frame, info_line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                if writer is not None:
                    writer.write(annotated_frame)

                display_img = annotated_frame
            else:
                # 一時停止中
                display_img = annotated_frame.copy()
                cv2.putText(display_img, "PAUSED (Press SPACE to Resume)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                ss_dir = "data/detection_outputs/screenshots"
                os.makedirs(ss_dir, exist_ok=True)
                ss_path = os.path.join(ss_dir, f"detect_frame_{int(time.time()*1000)}.jpg")
                cv2.imwrite(ss_path, display_img)
                print(f"📸 スクリーンショットを保存しました: {ss_path}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[完了] 終了しました。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
