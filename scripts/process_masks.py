#!/usr/bin/env python3
"""
process_masks.py - 作成済み白線マスク画像に対する一括ノイズ除去・ROI加工ツール

機能:
  - 方法 A: 上部黒塗り (mask_top)
      画像サイズ・解像度はそのまま、上部（空・人物・遠景）を一括で黒 (0) に塗りつぶす
  - 方法 B: 下半分クロップ (crop_bottom)
      画像およびマスクの下部（路面領域）のみを切り出して新しいデータセットを作成
  - 小面積ノイズ除去 (remove_small_noise)
      白線とは関係ない数ピクセルのゴミ・斑点ノイズを面積フィルタで自動消去

使い方:
  # 方法 A: 既存データセットの上部45%を一括黒塗り（超高速）
  python3 scripts/process_masks.py --mode mask_top --top-cut-ratio 0.45

  # 方法 B: 既存データセットの下部55%を切り出して data/yolop_cropped_dataset を作成
  python3 scripts/process_masks.py --mode crop_bottom --top-cut-ratio 0.45

  # 斑点ノイズ除去も同時に適用
  python3 scripts/process_masks.py --mode mask_top --clean-noise --min-area 50
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

WS_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="白線マスクの一括加工・ノイズ除去ツール")
    parser.add_argument("--data-dir", type=str, default=str(WS_DIR / "data/yolop_dataset"),
                        help="対象のデータセットフォルダ (デフォルト: data/yolop_dataset)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="出力先フォルダ (省略時は上書きまたは cropped フォルダ)")
    parser.add_argument("--mode", type=str, choices=["mask_top", "crop_bottom", "clean_only"],
                        default="mask_top",
                        help="加工モード: 'mask_top' (方法A: 上部黒塗り), 'crop_bottom' (方法B: 下部切り出し), 'clean_only'")
    parser.add_argument("--top-cut-ratio", type=float, default=0.45,
                        help="上部カットの割合 (デフォルト: 0.45 = 上部45%をカット/黒塗り)")
    parser.add_argument("--clean-noise", action="store_true",
                        help="微小な斑点ノイズを面積フィルタで自動除去する")
    parser.add_argument("--min-area", type=int, default=30,
                        help="除去する最小ノイズ面積 (ピクセル数)")
    return parser.parse_args()


def clean_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """微小な孤立ノイズを連結成分解析で除去"""
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"🛠️ Mask Processing Tool")
    print(f"Mode: {args.mode}")
    print(f"Data Dir: {data_dir}")
    print(f"Top Cut Ratio: {args.top_cut_ratio * 100:.1f}%")
    print(f"Noise Filter: {args.clean_noise} (min_area: {args.min_area})")
    print("=" * 60)

    if args.mode == "crop_bottom":
        output_dir = Path(args.output_dir) if args.output_dir else WS_DIR / "data/yolop_cropped_dataset"
        for split in ["train", "val"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "lane_masks" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "drivable_masks" / split).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else data_dir

    total_processed = 0

    for split in ["train", "val"]:
        img_dir = data_dir / "images" / split
        mask_dir = data_dir / "lane_masks" / split
        da_dir = data_dir / "drivable_masks" / split

        mask_files = sorted(glob.glob(str(mask_dir / "*.png")) + glob.glob(str(mask_dir / "*.jpg")))
        if not mask_files:
            continue

        print(f"\nProcessing {split} split ({len(mask_files)} masks)...")

        for mask_path in tqdm(mask_files, desc=split):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape[:2]
            cut_y = int(h * args.top_cut_ratio)

            # 画像ファイルも取得
            img_name = Path(mask_path).stem + ".jpg"
            img_path = img_dir / img_name
            img = cv2.imread(str(img_path)) if img_path.exists() else None

            da_path = da_dir / Path(mask_path).name
            da_mask = cv2.imread(str(da_path), cv2.IMREAD_GRAYSCALE) if da_path.exists() else None

            # ノイズ除去
            if args.clean_noise:
                mask = clean_small_components(mask, args.min_area)

            if args.mode == "mask_top":
                # 方法 A: 上部を黒塗り (値: 0)
                mask[:cut_y, :] = 0
                if da_mask is not None:
                    da_mask[:cut_y, :] = 0

                save_mask_path = output_dir / "lane_masks" / split / Path(mask_path).name
                cv2.imwrite(str(save_mask_path), mask)

                if da_mask is not None:
                    save_da_path = output_dir / "drivable_masks" / split / Path(mask_path).name
                    cv2.imwrite(str(save_da_path), da_mask)

            elif args.mode == "crop_bottom":
                # 方法 B: 下部のみクロップして保存
                mask_cropped = mask[cut_y:, :]
                save_mask_path = output_dir / "lane_masks" / split / Path(mask_path).name
                cv2.imwrite(str(save_mask_path), mask_cropped)

                if img is not None:
                    img_cropped = img[cut_y:, :]
                    save_img_path = output_dir / "images" / split / img_name
                    cv2.imwrite(str(save_img_path), img_cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if da_mask is not None:
                    da_cropped = da_mask[cut_y:, :]
                    save_da_path = output_dir / "drivable_masks" / split / Path(mask_path).name
                    cv2.imwrite(str(save_da_path), da_cropped)

            elif args.mode == "clean_only":
                save_mask_path = output_dir / "lane_masks" / split / Path(mask_path).name
                cv2.imwrite(str(save_mask_path), mask)

            total_processed += 1

    print(f"\n✅ Processing complete! Total processed: {total_processed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
