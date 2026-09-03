#!/usr/bin/env python3
"""
comprehensive_model_comparison.py
6つのモデル（shiho-v2, 公式YOLOP, shiho_crop, shiho_mask, official_crop, official_mask）を
動画 shihou_video_2026_08_24_13_51_47.mp4 の複数代表フレームで推論・比較する。
結果は 2x3 グリッド画像として jpeg/ に保存する。
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


def draw_mask_panel(frame, mask, title, sub_info, color=(0, 255, 0)):
    vis = frame.copy()
    
    # 白線マスクの着色 & 半透明ブレンド
    vis[mask == 1] = color
    vis = cv2.addWeighted(frame, 0.45, vis, 0.55, 0)
    
    # 白線輪郭線の描画
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    # 上部タイトルヘッダー
    h, w = vis.shape[:2]
    header = np.zeros((70, w, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    cv2.putText(header, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(header, sub_info, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
    
    panel = np.vstack([header, vis])
    return panel


def main():
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Running comparison on device: {device}")

    # 6つのモデル定義
    models_config = [
        {
            "id": "shiho_v2",
            "title": "(1) shiho-v2 (Base Model)",
            "sub_info": "Origin: Past Course Tuned | Mode: Full",
            "path": str(WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights/shiho-v2-20251118.pth"),
            "roi_mode": "none",
            "color": (0, 165, 255) # Orange
        },
        {
            "id": "shiho_mask",
            "title": "(2) shiho-v2 + Mask-Top [Best IoU: 66.15%]",
            "sub_info": "Origin: shiho-v2 | Mode: Top 45% Masked",
            "path": str(WS_DIR / "models/shiho_lane_mask_best.pth"),
            "roi_mode": "mask_top",
            "color": (0, 255, 0) # Green
        },
        {
            "id": "shiho_crop",
            "title": "(3) shiho-v2 + Crop-Bottom [Best IoU: 63.83%]",
            "sub_info": "Origin: shiho-v2 | Mode: Bottom 55% 2x Crop",
            "path": str(WS_DIR / "models/shiho_lane_crop_best.pth"),
            "roi_mode": "crop_bottom",
            "color": (0, 255, 128) # Spring Green
        },
        {
            "id": "official_base",
            "title": "(4) Official YOLOP (BDD100K Untouched)",
            "sub_info": "Origin: Pure Official Weights | Mode: Full",
            "path": str(WS_DIR / "models/pretrained/yolop_official.pth"),
            "roi_mode": "none",
            "color": (255, 100, 0) # Blue-ish / Red
        },
        {
            "id": "official_mask",
            "title": "(5) Official YOLOP + Mask-Top [Best IoU: 61.82%]",
            "sub_info": "Origin: Official YOLOP | Mode: Top 45% Masked",
            "path": str(WS_DIR / "models/yolop_official_mask_best.pth"),
            "roi_mode": "mask_top",
            "color": (255, 200, 0) # Cyan
        },
        {
            "id": "official_crop",
            "title": "(6) Official YOLOP + Crop-Bottom [Best IoU: 59.65%]",
            "sub_info": "Origin: Official YOLOP | Mode: Bottom 55% 2x Crop",
            "path": str(WS_DIR / "models/yolop_official_crop_best.pth"),
            "roi_mode": "crop_bottom",
            "color": (255, 0, 255) # Magenta
        },
    ]

    loaded_models = []
    for cfg_item in models_config:
        if os.path.exists(cfg_item["path"]):
            print(f"Loading {cfg_item['title']}...")
            m = load_model(cfg_item["path"], device)
            loaded_models.append((cfg_item, m))
        else:
            print(f"Warning: {cfg_item['path']} not found!")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    video_path = str(WS_DIR / "mp4/shihou_video_2026_08_24_13_51_47.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    output_dir = WS_DIR / "jpeg/comparison_6models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 代表的フレーム
    test_frames = {
        200: "01_person_noise_frame200 (人物・白い服の誤検知テスト)",
        800: "02_straight_contrast_frame800 (直進路・影とコントラストテスト)",
        1500: "03_curve_continuity_frame1500 (急カーブ旋回・白線連続性テスト)",
        3000: "04_far_lane_frame3000 (遠方白線・逆光解像度テスト)",
        4500: "05_fork_pylons_frame4500 (分岐路・パイロン密集テスト)"
    }

    for f_idx, f_name in test_frames.items():
        print(f"\nProcessing Frame {f_idx} ({f_name})...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {f_idx}")
            continue

        panels = []
        for cfg_item, model in loaded_models:
            mask = infer_lane(model, frame, device, transform, roi_mode=cfg_item["roi_mode"])
            panel = draw_mask_panel(frame, mask, cfg_item["title"], cfg_item["sub_info"], color=cfg_item["color"])
            panel_small = cv2.resize(panel, (640, 380))
            panels.append(panel_small)

        # 2行 x 3列 グリッド作成
        if len(panels) == 6:
            row1 = np.hstack(panels[0:3]) # shiho-v2, shiho_mask, shiho_crop
            row2 = np.hstack(panels[3:6]) # official_base, official_mask, official_crop
            grid = np.vstack([row1, row2])
            
            save_path = output_dir / f"compare_grid_frame_{f_idx:05d}.jpg"
            cv2.imwrite(str(save_path), grid)
            print(f"  -> Saved grid comparison: {save_path}")

            # rootの jpeg/ にも最新の代表比較画像を保存
            root_save_path = WS_DIR / f"jpeg/comparison_6models_frame_{f_idx}.jpg"
            cv2.imwrite(str(root_save_path), grid)

    cap.release()
    print("\n🎉 All 6 models comparison completed! Check images in jpeg/comparison_6models/")


if __name__ == "__main__":
    main()
