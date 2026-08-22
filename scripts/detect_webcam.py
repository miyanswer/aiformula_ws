#!/usr/bin/env python3
"""
学習済みYOLOモデル (best.pt 等) を用いて、Webカメラや動画に対してリアルタイム物体認識を行うスクリプト。

操作方法:
  [q] または [ESC]: 終了
  [s]: 検出結果画像を保存
  [SPACE]: 一時停止 / 再開
"""

import argparse
import os
import sys
import time
import cv2
import torch


def detect_best_device():
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Webカメラ/動画でのリアルタイムYOLO物体認識")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="モデル重みファイルパス (例: runs/train/yolo_custom/weights/best.pt)")
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

    print("=== YOLO リアルタイム物体認識 ===")
    print(f"モデル: {args.model}")
    print(f"入力ソース: {args.source}")
    print(f"信頼度しきい値: {args.conf}")
    print(f"デバイス: {device}")
    print("================================")

    # モデルのロード
    model = YOLO(args.model)

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

    window_name = f"YOLO Detection - {os.path.basename(args.model)}"
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

                # 描画済みフレームの取得
                annotated_frame = results[0].plot()

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
                info_line1 = f"Inference: {infer_ms:.1f}ms ({fps_display:.1f} FPS) | Detections: {n_detections}"
                cv2.putText(annotated_frame, info_line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

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
                ss_path = os.path.join(ss_dir, f"detect_shot_{int(time.time()*1000)}.jpg")
                cv2.imwrite(ss_path, display_img)
                print(f"[静止画保存] -> {ss_path}")

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[完了] 終了しました。")


if __name__ == "__main__":
    sys.exit(main())
