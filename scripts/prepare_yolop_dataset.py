#!/usr/bin/env python3
"""
prepare_yolop_dataset.py - 動画群からYOLOP白線ファインチューニング用データセットを作成するスクリプト

機能:
  1. mp4/ フォルダ内の全動画（または指定動画）から一定間隔でフレーム画像を抽出
  2. 既存の学習済みモデル（shiho-v2 等）を用いて初期の白線マスク（Pseudo Label）を自動生成
  3. 学習用 (train) と検証用 (val) に自動分割して保存
  4. 生成されたマスク画像 (PNG) は手動修正（白消し・塗り足し）も可能

使い方:
  # 基本（mp4/ の全動画から 1秒に1枚抽出して疑似ラベルを生成）
  python3 scripts/prepare_yolop_dataset.py

  # 特定の動画のみ指定、0.5秒間隔で抽出
  python3 scripts/prepare_yolop_dataset.py --video-dir mp4/ --every-sec 0.5 --max-frames 500

  # GPU / MPS / CPU の指定
  python3 scripts/prepare_yolop_dataset.py --device mps
"""

import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# YOLOP のインポート設定
WS_DIR = Path(__file__).resolve().parent.parent
YOLOP_DIR = WS_DIR / "src/ai_formula_oit_2026/perception/yolop"
if str(YOLOP_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOP_DIR))

try:
    from yolop.lib.config import cfg
    from yolop.lib.models import get_net
    from yolop.lib.utils import letterbox_for_img
except ImportError:
    # ローカルの直接インポートフォールバック
    from lib.config import cfg
    from lib.models import get_net
    from lib.utils import letterbox_for_img


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOP 白線学習用データセット構築ツール")
    parser.add_argument("--video-dir", type=str, default=str(WS_DIR / "mp4"),
                        help="動画フォルダパス (デフォルト: mp4/)")
    parser.add_argument("--video-file", type=str, default="",
                        help="特定の動画ファイルのみ処理する場合に指定")
    parser.add_argument("--output-dir", type=str, default=str(WS_DIR / "data/yolop_dataset"),
                        help="データセット保存先 (デフォルト: data/yolop_dataset)")
    parser.add_argument("--weights", type=str,
                        default=str(WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights/shiho-v2-20251118.pth"),
                        help="疑似ラベル生成に使うベースモデル重み")
    parser.add_argument("--every-sec", type=float, default=1.0,
                        help="動画から何秒ごとに1枚切り出すか (デフォルト: 1.0秒)")
    parser.add_argument("--max-frames-per-video", type=int, default=100,
                        help="1つの動画から抽出する最大フレーム数 (デフォルト: 100)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="検証用データの割合 (デフォルト: 0.2 = 20%)")
    parser.add_argument("--device", type=str, default="auto",
                        help="推論デバイス: 'cuda', 'mps', 'cpu', 'auto'")
    parser.add_argument("--all", action="store_true",
                        help="全ての動画を一括処理する (確認をスキップ)")
    parser.add_argument("--skip-pseudo-label", action="store_true",
                        help="疑似ラベル生成をスキップし、フレーム切り出しのみ行う")
    return parser.parse_args()


def detect_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base_model(weights_path: str, device: torch.device):
    model = get_net(cfg)
    if not os.path.exists(weights_path):
        print(f"Warning: Base weights not found at {weights_path}, using random init for pseudo labeling.")
        model.eval()
        return model.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_lane_mask(model, frame: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """1枚のフレームから白線マスク (ll) と走行領域マスク (da) を推論"""
    h_orig, w_orig = frame.shape[:2]
    img, ratio, pad = letterbox_for_img(frame, new_shape=640, auto=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, da_seg_out, ll_seg_out = model(img_t)

    # 白線マスクの復元
    pad_x_half, pad_y_half = pad
    ROUNDING_ADJUSTMENT = 0.1
    top, bottom = round(pad_y_half - ROUNDING_ADJUSTMENT), round(pad_y_half + ROUNDING_ADJUSTMENT)
    left, right = round(pad_x_half - ROUNDING_ADJUSTMENT), round(pad_x_half + ROUNDING_ADJUSTMENT)

    height, width = img_t.shape[2:]
    ll_predict = ll_seg_out[:, :, top:(height - bottom), left:(width - right)]
    ll_seg_mask_raw = torch.nn.functional.interpolate(
        ll_predict, size=(h_orig, w_orig), mode="bilinear"
    )
    _, ll_seg_map = torch.max(ll_seg_mask_raw, dim=1)
    ll_mask = ll_seg_map.int().squeeze().cpu().numpy().astype(np.uint8)  # 0 or 1

    # 走行領域マスクの復元
    da_predict = da_seg_out[:, :, top:(height - bottom), left:(width - right)]
    da_seg_mask_raw = torch.nn.functional.interpolate(
        da_predict, size=(h_orig, w_orig), mode="bilinear"
    )
    _, da_seg_map = torch.max(da_seg_mask_raw, dim=1)
    da_mask = da_seg_map.int().squeeze().cpu().numpy().astype(np.uint8)  # 0 or 1

    return ll_mask, da_mask


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    # 出力ディレクトリ構造の作成
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "lane_masks" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "drivable_masks" / split).mkdir(parents=True, exist_ok=True)

    # 対象動画の収集
    if args.video_file:
        video_files = [args.video_file] if os.path.exists(args.video_file) else []
    else:
        all_videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")) +
                            glob.glob(os.path.join(args.video_dir, "*.MOV")) +
                            glob.glob(os.path.join(args.video_dir, "*.avi")))
        if not all_videos:
            print(f"Error: No video files found in {args.video_dir}")
            sys.exit(1)

        if args.all or len(all_videos) == 1:
            video_files = all_videos
        else:
            print("\n🎬 利用可能な動画一覧 (mp4/):")
            for idx, vf in enumerate(all_videos, 1):
                size_mb = os.path.getsize(vf) / (1024 * 1024)
                print(f"  [{idx:2d}] {Path(vf).name:<45} ({size_mb:.1f} MB)")

            print(f"  [all] 全ての動画 ({len(all_videos)} 本) を処理")

            try:
                choice = input("\n処理したい動画の番号を入力してください (例: 1 や 1,3 や all) [デフォルト: 1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                sys.exit(0)

            if not choice:
                choice = "1"

            if choice.lower() == "all":
                video_files = all_videos
            else:
                selected_indices = []
                for part in choice.replace(" ", ",").split(","):
                    part = part.strip()
                    if "-" in part:
                        try:
                            start, end = map(int, part.split("-"))
                            selected_indices.extend(range(start, end + 1))
                        except ValueError:
                            pass
                    elif part.isdigit():
                        selected_indices.append(int(part))

                video_files = []
                for idx in selected_indices:
                    if 1 <= idx <= len(all_videos):
                        video_files.append(all_videos[idx - 1])

                if not video_files:
                    print("有効な動画が選択されませんでした。")
                    sys.exit(1)

    print(f"\n=== YOLOP Dataset Preparation ===")
    print(f"Selected {len(video_files)} video(s):")
    for vf in video_files:
        print(f"  - {Path(vf).name}")
    print(f"Output directory: {output_dir}")

    device = detect_device(args.device)
    print(f"Using device: {device}")

    model = None
    if not args.skip_pseudo_label:
        print(f"Loading base model: {args.weights}")
        model = load_base_model(args.weights, device)

    total_train = 0
    total_val = 0
    random.seed(42)

    for vid_idx, video_path in enumerate(video_files):
        vid_name = Path(video_path).stem.replace(" ", "_").replace("(", "").replace(")", "")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            continue

        step = max(1, int(fps * args.every_sec))
        print(f"[{vid_idx+1}/{len(video_files)}] Processing {Path(video_path).name} (FPS: {fps:.1f}, Frames: {total_frames})")

        frame_indices = range(0, total_frames, step)
        if len(frame_indices) > args.max_frames_per_video:
            frame_indices = np.linspace(0, total_frames - 1, args.max_frames_per_video, dtype=int)

        for frame_idx in tqdm(frame_indices, desc="Frames", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            img_filename = f"{vid_name}_f{frame_idx:06d}.jpg"
            mask_filename = f"{vid_name}_f{frame_idx:06d}.png"

            ll_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            da_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            if model is not None:
                ll_mask, da_mask = predict_lane_mask(model, frame, device)

            # Train or Val に即時ディスク保存 (メモリ消費ゼロ化)
            split = "val" if random.random() < args.val_ratio else "train"
            
            img_path = output_dir / "images" / split / img_filename
            ll_path = output_dir / "lane_masks" / split / mask_filename
            da_path = output_dir / "drivable_masks" / split / mask_filename

            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(ll_path), ll_mask * 255)
            cv2.imwrite(str(da_path), da_mask * 255)

            if split == "train":
                total_train += 1
            else:
                total_val += 1

        cap.release()

    print(f"\n✅ Dataset creation complete!")
    print(f"Total images saved: {total_train + total_val}")
    print(f"Train: {total_train} images in {output_dir / 'images/train'}")
    print(f"Val:   {total_val} images in {output_dir / 'images/val'}")
    print(f"\n💡 Mask files (0 or 255) are saved in {output_dir / 'lane_masks'}.")
    print(f"You can now run: python3 scripts/train_lane_yolop.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
