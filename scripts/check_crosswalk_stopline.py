#!/usr/bin/env python3
"""
check_crosswalk_stopline.py
shihou_video_2026_08_24_15_38_45.mp4 および shihou_video_2026_08_24_13_51_47.mp4 をスキャンし、
横断歩道や一時停止線があるシーンにおいて、
(1) 初期モデル shiho-v2
(2) 新しい shiho_lane_mask_best.pth
での白線認識マスクを比較・検証する。
"""

import os
import sys
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


def draw_panel(frame, mask, title, color=(0, 255, 0)):
    vis = frame.copy()
    vis[mask == 1] = color
    vis = cv2.addWeighted(frame, 0.45, vis, 0.55, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    h, w = vis.shape[:2]
    header = np.zeros((60, w, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    cv2.putText(header, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return np.vstack([header, vis])


def main():
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    base_path = str(WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights/shiho-v2-20251118.pth")
    new_path = str(WS_DIR / "models/shiho_lane_mask_best.pth")

    print(f"Loading Base: {base_path}")
    base_model = load_model(base_path, device)
    print(f"Loading New: {new_path}")
    new_model = load_model(new_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    out_dir = WS_DIR / "jpeg/crosswalk_stopline_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 対象動画: shihou_video_2026_08_24_13_51_47.mp4
    video_path = str(WS_DIR / "mp4/shihou_video_2026_08_24_13_51_47.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    # 横断歩道、一時停止線、直線、急カーブ、パイロン交差点を含む代表フレーム
    target_frames = [
        200,   # スタート地点・人物・初期ライン
        500,   # 直線進入・路面継ぎ目
        900,   # カーブ手前・減速帯
        1300,  # 急カーブ旋回
        1800,  # 直線路・遠方白線
        2200,  # パイロン密集・交差点エリア
        2600,  # 横向きペイント・停止線エリア
        3000,  # 逆光・白線分岐
        3400,  # 外周ストレート
        3800,  # 横断歩道・ゼブラ模様付近
        4200,  # ヘアピンカーブ
        4600,  # 分岐路・パイロン
        5000,  # ダブルレーン・路肩フェンス
        5500,  # 連続S字コーナー
        6000,  # 終盤ストレート・停止線付近
        6500,  # 最終コーナー
        7000   # ゴール地点・ピットエリア
    ]

    print(f"\nProcessing {len(target_frames)} frames from {Path(video_path).name}...")

    for f in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipping frame {f}")
            continue

        mask_base = infer_lane(base_model, frame, device, transform, roi_mode="none")
        mask_new = infer_lane(new_model, frame, device, transform, roi_mode="mask_top")

        p_base = draw_panel(frame, mask_base, f"Base Model (shiho-v2) | Frame {f}", color=(0, 165, 255))
        p_new = draw_panel(frame, mask_new, f"New (shiho_lane_mask_best) | Frame {f}", color=(0, 255, 0))

        p_base_s = cv2.resize(p_base, (960, 560))
        p_new_s = cv2.resize(p_new, (960, 560))
        combined = np.hstack([p_base_s, p_new_s])
        
        save_file = out_dir / f"shihou135147_frame_{f:05d}.jpg"
        cv2.imwrite(str(save_file), combined)
        print(f"  -> Saved: {save_file.name}")

    cap.release()
    print(f"\n🎉 Finished! All comparison images saved in: {out_dir}")


if __name__ == "__main__":
    main()
