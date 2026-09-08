#!/usr/bin/env python3
"""
detect_traffic_light_depth.py - YOLO 信号機認識 & 垂直奥行き推定 スタンドアロン検証ツール

概要:
    動画ファイル (.mp4等) または Webカメラ から信号機を検出し、
    バウンディングボックスの高さから垂直奥行き Z [m] をリアルタイム計算・表示します。

使い方:
    # 既定の動画でテスト
    python3 scripts/detect_traffic_light_depth.py --video mp4/shihou_video_2026_08_24_13_51_47.mp4

    # 焦点距離や実寸法を指定
    python3 scripts/detect_traffic_light_depth.py --model models/traffic_light.pt --fy 189.5 --height 0.32
"""

import argparse
import os
import sys
import time
from typing import Optional, List

import cv2
import numpy as np

# utilsモジュールのインポート
sys.path.insert(0, os.path.abspath("src/ai_formula_oit_2027"))
from ai_formula_oit_2027.utils.traffic_light_depth import TrafficLightDepthEstimator


def load_yaml_params():
    """config/traffic_light_params.yaml からパラメータを自動取得"""
    yaml_path = "src/ai_formula_oit_2027/config/traffic_light_params.yaml"
    params = {"fy": 2256.0, "height": 0.32, "conf": 0.35, "model": "models/traffic_light.pt"}

    if os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                p = data.get("/**", {}).get("traffic_light_detector", {}).get("ros__parameters", {})
                if "focal_length_y" in p:
                    params["fy"] = float(p["focal_length_y"])
                if "real_height" in p:
                    params["height"] = float(p["real_height"])
                if "conf_threshold" in p:
                    params["conf"] = float(p["conf_threshold"])
                if "model_path" in p:
                    params["model"] = str(p["model_path"])
        except Exception:
            pass
    return params


def parse_args():
    yaml_p = load_yaml_params()
    parser = argparse.ArgumentParser(description="YOLO 信号機 垂直奥行き推定 検証ツール")
    parser.add_argument("--model", type=str, default=yaml_p["model"], help="モデルパス (.pt)")
    parser.add_argument("--video", type=str, default="", help="動画ファイルパス (.mp4等、省略時は自動検索)")
    parser.add_argument("--fy", type=float, default=yaml_p["fy"], help="カメラ垂直焦点距離 (pixel)")
    parser.add_argument("--height", type=float, default=yaml_p["height"], help="信号機の物理的な実高さ (m)")
    parser.add_argument("--conf", type=float, default=yaml_p["conf"], help="信頼度閾値")
    parser.add_argument("--device", type=str, default="cpu", help="推論デバイス ('cpu', 'mps', '0')")
    parser.add_argument("--save", action="store_true", help="結果動画を保存")
    return parser.parse_args()



def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[エラー] 'ultralytics' がインストールされていません。")
        return 1

    if not os.path.isfile(args.model):
        print(f"[エラー] モデルが見つかりません: {args.model}")
        return 1

    # 動画ファイルの決定
    video_path = args.video
    if not video_path:
        mp4_candidates = [
            "mp4/shihou_video_2026_08_24_13_51_47.mp4",
            "mp4/20260824_124734.mp4",
        ]
        for c in mp4_candidates:
            if os.path.isfile(c):
                video_path = c
                break
        if not video_path and os.path.isdir("mp4"):
            for f in os.listdir("mp4"):
                if f.endswith((".mp4", ".mov", ".avi")):
                    video_path = os.path.join("mp4", f)
                    break

    if not video_path or not os.path.isfile(video_path):
        print(f"[エラー] 動画ファイルが見つかりません。--video で指定してください。")
        return 1

    print("=" * 65)
    print("🚦 信号機 垂直奥行き逆算 検証ツール")
    print("=" * 65)
    print(f"🎯 モデル   : {args.model}")
    print(f"🎥 入力動画 : {video_path}")
    print(f"📐 物理高さ : {args.height:.2f} m (32cm正方形規格)")
    print(f"🔍 焦点距離 : fy = {args.fy:.1f} px")
    print(f"⚡ 計算式   : Z = ({args.fy:.1f} * {args.height:.2f}) / h_px")
    print("=" * 65)

    estimator = TrafficLightDepthEstimator(
        real_height=args.height,
        focal_length_y=args.fy,
        smoothing_alpha=0.7,
    )

    model = YOLO(args.model)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[エラー] 動画を開けませんでした: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = "Traffic Light Depth Estimation (Direct Z from Height)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n🎮 操作: [SPACE] 一時停止, [Q/ESC] 終了\n")

    is_paused = False
    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )

            annotated = frame.copy()
            nearest_z = None
            n_det = 0

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                n_det = len(boxes)
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")

                    x1, y1, x2, y2 = xyxy
                    bbox_h = float(y2 - y1)
                    depth_z = estimator.estimate_from_bbox((x1, y1, x2, y2), track_id=i)

                    if depth_z is not None:
                        if nearest_z is None or depth_z < nearest_z:
                            nearest_z = depth_z

                    # 描画
                    color = (0, 0, 255) if "red" in cls_name.lower() else (0, 255, 0)
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    if depth_z is not None:
                        label = f"{cls_name} {conf:.2f} | Z={depth_z:.2f}m (h={bbox_h:.0f}px)"
                    else:
                        label = f"{cls_name} {conf:.2f} (h={bbox_h:.0f}px)"

                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(annotated, (int(x1), max(0, int(y1) - lh - 8)), (int(x1) + lw + 6, int(y1)), color, -1)
                    cv2.putText(annotated, label, (int(x1) + 3, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # HUD
            cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 20), -1)
            z_txt = f"{nearest_z:.2f} m" if nearest_z is not None else "---"
            hud1 = f"Traffic Light: {n_det} detected | Nearest Depth Z: {z_txt} | fy = {estimator.focal_length_y:.1f} px | H = {args.height:.2f}m"
            hud2 = f"[SPACE]: Pause | [d]/[a]: Step | [+][-]: Adjust fy (+/-50px) | [Q]: Quit"
            cv2.putText(annotated, hud1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, hud2, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated)

        wait_ms = 0 if is_paused else max(1, int(1000.0 / fps))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            is_paused = not is_paused
        elif key in (ord('d'), 83):  # d または 右矢印
            is_paused = True
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key in (ord('a'), 81):  # a または 左矢印
            is_paused = True
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur_pos - 2))
        elif key in (ord('+'), ord('=')):  # fy 増加 (+50px)
            estimator.focal_length_y += 50.0
            print(f"🔍 焦点距離 fy UP: {estimator.focal_length_y:.1f} px")
        elif key in (ord('-'), ord('_')):  # fy 減少 (-50px)
            estimator.focal_length_y = max(10.0, estimator.focal_length_y - 50.0)
            print(f"🔍 焦点距離 fy DOWN: {estimator.focal_length_y:.1f} px")
        elif key == ord(']'):  # 微調整 (+10px)
            estimator.focal_length_y += 10.0
            print(f"🔍 焦点距離 fy UP (微調整): {estimator.focal_length_y:.1f} px")
        elif key == ord('['):  # 微調整 (-10px)
            estimator.focal_length_y = max(10.0, estimator.focal_length_y - 10.0)
            print(f"🔍 焦点距離 fy DOWN (微調整): {estimator.focal_length_y:.1f} px")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
