#!/usr/bin/env python3
"""
eval_lane_yolop.py - 学習済み/ファインチューニング済み YOLOP モデルの動画推論・評価スクリプト

機能:
  - 指定した mp4 動画に対して YOLOP 推論を行い、白線重畳動画またはサンプル画像を生成
  - 改善度（前モデル vs 新モデルの比較）を確認可能

使い方:
  python3 scripts/eval_lane_yolop.py --video mp4/shihou_video_2026_08_24_13_51_47.mp4
  python3 scripts/eval_lane_yolop.py --weights models/shiho_lane_finetuned_best.pth --save-video
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

WS_DIR = Path(__file__).resolve().parent.parent
YOLOP_DIR = WS_DIR / "src/ai_formula_oit_2026/perception/yolop"
if str(YOLOP_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOP_DIR))

try:
    from yolop.lib.config import cfg
    from yolop.lib.models import get_net
    from yolop.lib.utils import letterbox_for_img
except ImportError:
    from lib.config import cfg
    from lib.models import get_net
    from lib.utils import letterbox_for_img


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOP モデルの推論・評価")
    parser.add_argument("--video", type=str, default=str(WS_DIR / "mp4/shihou_video_2026_08_24_13_51_47.mp4"),
                        help="テスト対象の動画パス")
    parser.add_argument("--weights", type=str,
                        default=str(WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights/shiho-v2-20251118.pth"),
                        help="評価するモデルの重みファイル")
    parser.add_argument("--output-dir", type=str, default=str(WS_DIR / "jpeg"),
                        help="結果保存先フォルダ")
    parser.add_argument("--save-video", action="store_true",
                        help="推論結果の動画 (MP4) を書き出す")
    parser.add_argument("--roi-mode", type=str, choices=["none", "mask_top", "crop_bottom"],
                        default="none",
                        help="推論ROIモード: 'mask_top' (方法A), 'crop_bottom' (方法B: 下部クロップ推論), 'none'")
    parser.add_argument("--top-cut-ratio", type=float, default=0.45,
                        help="上部カットの割合 (デフォルト: 0.45)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="処理する最大フレーム数")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def detect_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.weights}")
    print(f"ROI Mode: {args.roi_mode} (Top cut ratio: {args.top_cut_ratio * 100:.1f}%)")
    model = get_net(cfg)
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = None
    if args.save_video:
        video_out_path = output_dir / f"eval_{Path(args.video).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w_orig, h_orig))
        print(f"Saving output video to: {video_out_path}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames_to_process = min(total_frames, args.max_frames)
    print(f"Processing {frames_to_process} frames from {Path(args.video).name} on {device}...")

    cut_y = int(h_orig * args.top_cut_ratio)

    for f_idx in tqdm(range(frames_to_process), desc="Evaluating"):
        ret, frame = cap.read()
        if not ret:
            break

        if args.roi_mode == "crop_bottom":
            # 方法B: 下部領域のみ切り出して推論 (解像度2倍)
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

        if args.roi_mode == "crop_bottom":
            # 元の 1920x1080 に貼り戻す (上部は 0)
            ll_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            ll_mask[cut_y:, :] = sub_mask
        elif args.roi_mode == "mask_top":
            # 方法A: 上部を黒塗り (0)
            ll_mask = sub_mask
            ll_mask[:cut_y, :] = 0
        else:
            ll_mask = sub_mask

        vis = frame.copy()
        vis[ll_mask == 1] = [0, 255, 0]  # Green

        if out_video is not None:
            out_video.write(vis)

        # 代表的なフレームを静止画保存
        if f_idx in [50, 100, 200, 290]:
            img_out_path = output_dir / f"eval_frame_{f_idx}.jpg"
            cv2.imwrite(str(img_out_path), vis)

    cap.release()
    if out_video is not None:
        out_video.release()

    print(f"\n✅ Evaluation complete! Check images in {output_dir}")


if __name__ == "__main__":
    main()
