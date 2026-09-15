#!/usr/bin/env python3
"""
crop_honda_dataset.py - honda_data_set を上部45%クロップして honda_cropped_dataset を作成するスクリプト

機能:
  - data/honda_data_set の images, ll_seg_annotations (白線), da_seg_annotations (走行可能領域) を処理
  - det_annotations (物体検出) は参照・出力しない
  - 画像・マスクの上部45%をカット (下部55%を保持, 1080x1920 -> 594x1920)
  - data/honda_cropped_dataset/ 配下に train / val の構造で保存
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm

WS_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="honda_data_set のクロップ処理")
    parser.add_argument("--src-dir", type=str, default=str(WS_DIR / "data/honda_data_set"),
                        help="変換元ディレクトリ (デフォルト: data/honda_data_set)")
    parser.add_argument("--dst-dir", type=str, default=str(WS_DIR / "data/honda_cropped_dataset"),
                        help="出力先ディレクトリ (デフォルト: data/honda_cropped_dataset)")
    parser.add_argument("--top-cut-ratio", type=float, default=0.45,
                        help="上部カットの割合 (デフォルト: 0.45 = 上部45%カット)")
    parser.add_argument("--workers", type=int, default=4,
                        help="並列ワーカー数")
    return parser.parse_args()


def process_sample(item):
    split, img_p, ll_p, da_p, dst_img_dir, dst_ll_dir, dst_da_dir, top_cut_ratio = item
    
    # 画像
    img = cv2.imread(img_p)
    if img is None:
        return False
    h = img.shape[0]
    cut_y = int(h * top_cut_ratio)
    cropped_img = img[cut_y:, :]
    
    stem = Path(img_p).stem
    dst_img_p = dst_img_dir / f"{stem}.jpg"
    cv2.imwrite(str(dst_img_p), cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 白線マスク (ll)
    if os.path.exists(ll_p):
        ll_mask = cv2.imread(ll_p, cv2.IMREAD_UNCHANGED)
        if ll_mask is not None:
            cropped_ll = ll_mask[cut_y:, :]
            dst_ll_p = dst_ll_dir / f"{stem}.png"
            cv2.imwrite(str(dst_ll_p), cropped_ll)
            
    # 走行可能領域マスク (da)
    if os.path.exists(da_p):
        da_mask = cv2.imread(da_p, cv2.IMREAD_UNCHANGED)
        if da_mask is not None:
            cropped_da = da_mask[cut_y:, :]
            dst_da_p = dst_da_dir / f"{stem}.png"
            cv2.imwrite(str(dst_da_p), cropped_da)

    return True


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    
    print("=" * 60)
    print("🚗 Honda Dataset Cropper (Lane & Drivable Only)")
    print(f"Source: {src_dir}")
    print(f"Target: {dst_dir}")
    print(f"Top cut ratio: {args.top_cut_ratio} (Retaining bottom {int((1-args.top_cut_ratio)*100)}%)")
    print("⚠️  Notice: `det_annotations` will be completely ignored.")
    print("=" * 60)

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist.")
        sys.exit(1)

    tasks = []
    for split in ["train", "val"]:
        src_img_dir = src_dir / "images" / split
        src_ll_dir = src_dir / "ll_seg_annotations" / split
        src_da_dir = src_dir / "da_seg_annotations" / split
        
        dst_img_dir = dst_dir / "images" / split
        dst_ll_dir = dst_dir / "lane_masks" / split
        dst_da_dir = dst_dir / "drivable_masks" / split
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_ll_dir.mkdir(parents=True, exist_ok=True)
        dst_da_dir.mkdir(parents=True, exist_ok=True)
        
        img_paths = sorted(glob.glob(str(src_img_dir / "*.jpg")) + glob.glob(str(src_img_dir / "*.png")))
        print(f"Found {len(img_paths)} images in {split} set.")
        
        for img_p in img_paths:
            stem = Path(img_p).stem
            ll_p = str(src_ll_dir / f"{stem}.png")
            da_p = str(src_da_dir / f"{stem}.png")
            tasks.append((split, img_p, ll_p, da_p, dst_img_dir, dst_ll_dir, dst_da_dir, args.top_cut_ratio))

    print(f"\nProcessing total {len(tasks)} samples with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_sample, tasks), total=len(tasks), desc="Cropping"))

    success_count = sum(1 for r in results if r)
    print(f"\n✅ Completed! Successfully processed {success_count}/{len(tasks)} samples.")
    print(f"📁 Cropped dataset saved to: {dst_dir}")


if __name__ == "__main__":
    main()
