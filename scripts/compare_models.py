#!/usr/bin/env python3
"""
compare_models.py - 3つのモデル（ベースモデル、上部マスク版、クロップ版）の比較検証スクリプト
同一フレームに対して3つのモデルの白線認識結果を横並び（または上下）で合成・保存して比較する。
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

WS_DIR = Path(__file__).resolve().parent.parent
YOLOP_DIR = WS_DIR / "src/ai_formula_oit_2026/perception/yolop"
if str(YOLOP_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOP_DIR))

from lib.config import cfg
from lib.models import get_net
from lib.utils import letterbox_for_img


def load_model(weights_path: str, device: torch.device):
    model = get_net(cfg)
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_lane(model, frame, device, transform, roi_mode="none", top_cut_ratio=0.45):
    h_orig, w_orig = frame.shape[:2]
    cut_y = int(h_orig * top_cut_ratio)

    if roi_mode == "crop_bottom":
        input_frame = frame[cut_y:, :]
        h_input, w_input = input_frame.shape[:2]
    else:
        input_frame = frame
        h_input, w_input = h_orig, w_orig

    img, ratio, pad = letterbox_for_img(input_frame, new_shape=640, auto=True)
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, ll_seg_out = model(img_t)

    pad_x_half, pad_y_half = pad
    ROUNDING_ADJUSTMENT = 0.1
    top, bottom = round(pad_y_half - ROUNDING_ADJUSTMENT), round(pad_y_half + ROUNDING_ADJUSTMENT)
    left, right = round(pad_x_half - ROUNDING_ADJUSTMENT), round(pad_x_half + ROUNDING_ADJUSTMENT)

    height, width = img_t.shape[2:]
    ll_predict = ll_seg_out[:, :, top:(height - bottom), left:(width - right)]
    ll_seg_mask_raw = torch.nn.functional.interpolate(
        ll_predict, size=(h_input, w_input), mode="bilinear"
    )
    _, ll_seg_map = torch.max(ll_seg_mask_raw, dim=1)
    sub_mask = ll_seg_map.int().squeeze().cpu().numpy().astype(np.uint8)

    if roi_mode == "crop_bottom":
        ll_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        ll_mask[cut_y:, :] = sub_mask
    elif roi_mode == "mask_top":
        ll_mask = sub_mask
        ll_mask[:cut_y, :] = 0
    else:
        ll_mask = sub_mask

    return ll_mask


def draw_mask_on_image(frame, mask, color=(0, 255, 0), title=""):
    vis = frame.copy()
    # マスク領域を鮮やかに着色
    vis[mask == 1] = color
    # 元画像と半透明ブレンド
    vis = cv2.addWeighted(frame, 0.4, vis, 0.6, 0)
    # 輪郭を強調
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    # タイトルバー
    cv2.rectangle(vis, (10, 10), (600, 60), (0, 0, 0), -1)
    cv2.putText(vis, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return vis


def main():
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Comparing models on device: {device}")

    # 1. モデルの読み込み
    models_info = [
        {
            "name": "1. Base (shiho-v2)",
            "path": str(WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights/shiho-v2-20251118.pth"),
            "roi_mode": "none",
            "color": (0, 255, 0)
        },
        {
            "name": "2. Method A (shiho_lane_mask)",
            "path": str(WS_DIR / "models/shiho_lane_mask_best.pth"),
            "roi_mode": "mask_top",
            "color": (0, 255, 0)
        },
        {
            "name": "3. Method B (shiho_lane_crop)",
            "path": str(WS_DIR / "models/shiho_lane_crop_best.pth"),
            "roi_mode": "crop_bottom",
            "color": (0, 255, 0)
        }
    ]

    loaded_models = []
    for info in models_info:
        if os.path.exists(info["path"]):
            print(f"Loading {info['name']}...")
            m = load_model(info["path"], device)
            loaded_models.append((info, m))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    video_path = str(WS_DIR / "mp4/shihou_video_2026_08_24_13_51_47.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    output_dir = WS_DIR / "jpeg/comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 比較したい代表的なシーンのフレーム番号
    # 200: 人物の白い服があるシーン
    # 1500: 直進・カーブ
    # 3500: 遠くの白線があるシーン
    # 5500: 白線が二股・交差するシーン
    target_frames = [200, 1200, 2500, 4500]

    for f_idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        results = []
        for info, model in loaded_models:
            mask = infer_lane(model, frame, device, transform, roi_mode=info["roi_mode"])
            vis = draw_mask_on_image(frame, mask, color=info["color"], title=f"{info['name']} (Frame {f_idx})")
            # 縮小して横並びに結合 (各 640x360)
            vis_small = cv2.resize(vis, (640, 360))
            results.append(vis_small)

        # 3つの結果を横並びまたは縦に結合
        combined = np.hstack(results) if len(results) == 3 else np.vstack(results)
        save_path = output_dir / f"compare_frame_{f_idx:05d}.jpg"
        cv2.imwrite(str(save_path), combined)
        print(f"Saved comparison: {save_path}")

    cap.release()
    print("Comparison complete!")


if __name__ == "__main__":
    main()
