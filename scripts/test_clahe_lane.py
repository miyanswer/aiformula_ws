#!/usr/bin/env python3
"""
test_clahe_lane.py - CLAHE (コントラスト制限付き適応的ヒストグラム平坦化) による
日差し・影エリアでの白線認識改善テストスクリプト (Frame 5500, 6000)
"""

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


def apply_clahe(image_bgr, clip_limit=2.5, tile_grid_size=(8, 8)):
    """LAB色空間のL(輝度)チャンネルにCLAHEを適用して明暗差を平滑化"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def load_model(weights_path: str, device: torch.device):
    model = get_net(cfg)
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_lane(model, frame, device, transform, top_cut_ratio=0.45, threshold=0.5):
    h_orig, w_orig = frame.shape[:2]
    cut_y = int(h_orig * top_cut_ratio)

    img, ratio, pad = letterbox_for_img(frame, new_shape=640, auto=True)
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
        ll_predict, size=(h_orig, w_orig), mode="bilinear"
    )
    
    # 確率マップ（Channel 1: 白線）
    prob = torch.softmax(ll_seg_mask_raw, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask[:cut_y, :] = 0  # 上部マスク

    return mask


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
    model_path = str(WS_DIR / "models/shiho_lane_mask_best.pth")
    model = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    video_path = str(WS_DIR / "mp4/shihou_video_2026_08_24_13_51_47.mp4")
    cap = cv2.VideoCapture(video_path)

    out_dir = WS_DIR / "jpeg/clahe_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_frames = [5500, 5800, 6000]

    for f in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. 通常推論 (Original)
        mask_orig = infer_lane(model, frame, device, transform, threshold=0.5)

        # 2. CLAHE 適用画像での推論
        frame_clahe = apply_clahe(frame, clip_limit=3.0, tile_grid_size=(8, 8))
        mask_clahe = infer_lane(model, frame_clahe, device, transform, threshold=0.45)

        p_orig = draw_panel(frame, mask_orig, f"Original (Before CLAHE) | Frame {f}", color=(0, 165, 255))
        p_clahe = draw_panel(frame_clahe, mask_clahe, f"CLAHE Enhanced (After) | Frame {f}", color=(0, 255, 0))

        p_orig_s = cv2.resize(p_orig, (960, 560))
        p_clahe_s = cv2.resize(p_clahe, (960, 560))
        combined = np.hstack([p_orig_s, p_clahe_s])

        save_path = out_dir / f"clahe_comparison_frame_{f:05d}.jpg"
        cv2.imwrite(str(save_path), combined)
        print(f"Saved: {save_path.name}")

    cap.release()
    print("CLAHE comparison finished!")


if __name__ == "__main__":
    main()
