#!/usr/bin/env python3
"""
extract_frames.py - 動画および静止画像からYOLO学習データセット用画像を抽出・取り込むスクリプト。

機能:
  - 動画ファイル / 動画フォルダ (data/tasks/<task>/raw_videos) からのフレーム切り出し
  - 静止画像ファイル / 画像フォルダ (data/tasks/<task>/raw_images) からのデータ取り込み (インポート)
  - スマホ写真等の EXIF 回転情報の自動補正 (自動正立)
  - 指定フレーム間隔 (例: 15フレーム毎) または 指定秒間隔 (例: 1秒毎) での動画切り出し
  - 差分があまりない重複フレーム/画像の自動スキップ機能 (オプション: --diff-threshold)
  - 最大解像度制限 / リサイズ機能 (オプション: --max-dim)
  - タスク別管理 (--task) 対応
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV", ".m4v", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIFF")


def parse_args():
    parser = argparse.ArgumentParser(description="動画および静止画像からYOLOアノテーション用画像を抽出・取り込むツール")
    parser.add_argument("--task", type=str, default="default",
                        help="タスク名 (例: traffic_light, t_junction, crosswalk, jinmen_dog)")
    parser.add_argument("--source", type=str, default="",
                        help="入力元ファイルまたはフォルダ (動画・画像・混在フォルダ対応)")
    parser.add_argument("--video", type=str, default="",
                        help="動画ファイルパス または 動画フォルダパス (省略時は data/tasks/<task>/raw_videos)")
    parser.add_argument("--images", type=str, default="",
                        help="静止画像ファイルパス または 画像フォルダパス (省略時は data/tasks/<task>/raw_images)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="切り出し・取り込み画像の保存先 (省略時は data/tasks/<task>/extracted_frames)")
    parser.add_argument("--every-sec", type=float, default=1.0,
                        help="[動画用] 何秒ごとに1枚切り出すか (例: 1.0 = 1秒に1枚)")
    parser.add_argument("--frame-interval", type=int, default=0,
                        help="[動画用] フレーム間隔指定 (0の場合は --every-sec を使用)")
    parser.add_argument("--prefix", type=str, default="frame",
                        help="動画切り出し画像のファイル名プレフィックス (例: frame)")
    parser.add_argument("--img-prefix", type=str, default="img",
                        help="静止画像取り込み時のファイル名プレフィックス (例: img)")
    parser.add_argument("--format", type=str, choices=["jpg", "png"], default="jpg",
                        help="保存画像フォーマット (jpg / png)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG保存品質 (1〜100, デフォルト: 95)")
    parser.add_argument("--diff-threshold", type=float, default=0.0,
                        help="直前画像との平均輝度差の最小閾値 (0で無効。静止シーンや重複画像のスキップ)")
    parser.add_argument("--max-dim", type=int, default=0,
                        help="最大長辺サイズ (0でリサイズなし。例: 1920 を指定すると4K写真をFHD相当に縮小)")
    parser.add_argument("--no-auto-orient", action="store_true",
                        help="EXIF回転情報の自動補正を無効化する")
    return parser.parse_args()


def load_image_with_exif_orientation(image_path: str, auto_orient: bool = True) -> np.ndarray:
    """EXIF回転情報を考慮して画像を読み込み、BGRのndarrayとして返す"""
    if HAS_PIL and auto_orient:
        try:
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                img_rgb = np.array(pil_img)
                return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    # フォールバック (OpenCV直接読み込み)
    return cv2.imread(image_path)


def resize_if_needed(img: np.ndarray, max_dim: int) -> np.ndarray:
    """必要に応じてアスペクト比を維持してリサイズ"""
    if max_dim <= 0 or img is None:
        return img
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def save_image(img: np.ndarray, out_path: str, img_format: str, quality: int):
    """画質設定を考慮して画像を保存"""
    if img_format == "jpg":
        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(out_path, img)


def process_video(video_path: str, output_dir: str, args) -> int:
    """動画からフレームを切り出して保存"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[エラー] 動画を開けませんでした: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # 切り出し間隔の決定
    if args.frame_interval > 0:
        step = args.frame_interval
    else:
        step = max(1, int(round(fps * args.every_sec)))

    print(f"\n[動画処理] {os.path.basename(video_path)}")
    print(f"  総フレーム数: {total_frames}, FPS: {fps:.2f}, 切り出し間隔: {step}フレームごと (約{step/fps:.2f}秒ごと)")

    saved_count = 0
    frame_idx = 0
    prev_gray = None

    pbar = tqdm(total=total_frames, desc=video_basename, unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % step == 0:
            should_save = True

            # 差分フィルタ (静止シーンの重複スキップ)
            if args.diff_threshold > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = diff.mean()
                    if mean_diff < args.diff_threshold:
                        should_save = False
                if should_save:
                    prev_gray = gray

            if should_save:
                frame = resize_if_needed(frame, args.max_dim)
                out_filename = f"{args.prefix}_{video_basename}_{frame_idx:06d}.{args.format}"
                out_path = os.path.join(output_dir, out_filename)
                save_image(frame, out_path, args.format, args.quality)
                saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"  -> {saved_count} 枚のフレームを保存しました")
    return saved_count


def process_images(image_files: List[str], output_dir: str, args) -> int:
    """静止画像群を読み込み、EXIF補正・リサイズ等を行って保存"""
    if not image_files:
        return 0

    print(f"\n[静止画取り込み] 対象画像数: {len(image_files)} 枚")
    saved_count = 0
    prev_gray = None

    pbar = tqdm(total=len(image_files), desc="Images Import", unit="img")

    for idx, img_path in enumerate(image_files):
        img = load_image_with_exif_orientation(img_path, auto_orient=not args.no_auto_orient)
        if img is None:
            print(f"\n[警告] 画像の読み込みに失敗しました: {img_path}")
            pbar.update(1)
            continue

        should_save = True
        if args.diff_threshold > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                # サイズが異なる場合はリサイズして比較
                if gray.shape != prev_gray.shape:
                    gray_resized = cv2.resize(gray, (prev_gray.shape[1], prev_gray.shape[0]))
                    diff = cv2.absdiff(gray_resized, prev_gray)
                else:
                    diff = cv2.absdiff(gray, prev_gray)
                if diff.mean() < args.diff_threshold:
                    should_save = False
            if should_save:
                prev_gray = gray

        if should_save:
            img = resize_if_needed(img, args.max_dim)
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            out_filename = f"{args.img_prefix}_{img_basename}.{args.format}"
            out_path = os.path.join(output_dir, out_filename)

            save_image(img, out_path, args.format, args.quality)
            saved_count += 1

        pbar.update(1)

    pbar.close()
    print(f"  -> {saved_count} 枚の静止画像を取り込みました")
    return saved_count


def collect_sources(args) -> Tuple[List[str], List[str]]:
    """動画ファイル一覧と静止画像ファイル一覧を収集"""
    video_files = []
    image_files = []

    def scan_dir(dir_path: str):
        v_list = []
        i_list = []
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1]
                full_path = os.path.join(root, f)
                if ext in VIDEO_EXTENSIONS:
                    v_list.append(full_path)
                elif ext in IMAGE_EXTENSIONS:
                    i_list.append(full_path)
        return v_list, i_list

    # 1. --source が指定されている場合
    if args.source:
        if os.path.isfile(args.source):
            ext = os.path.splitext(args.source)[1]
            if ext in VIDEO_EXTENSIONS:
                video_files.append(args.source)
            elif ext in IMAGE_EXTENSIONS:
                image_files.append(args.source)
        elif os.path.isdir(args.source):
            v, i = scan_dir(args.source)
            video_files.extend(v)
            image_files.extend(i)

    # 2. --video または デフォルトの raw_videos ディレクトリ
    if args.video:
        if os.path.isfile(args.video):
            video_files.append(args.video)
        elif os.path.isdir(args.video):
            v, _ = scan_dir(args.video)
            video_files.extend(v)
    elif not args.source:
        default_video_dir = os.path.join("data", "tasks", args.task, "raw_videos")
        if os.path.isdir(default_video_dir):
            v, _ = scan_dir(default_video_dir)
            video_files.extend(v)

    # 3. --images または デフォルトの raw_images ディレクトリ
    if args.images:
        if os.path.isfile(args.images):
            image_files.append(args.images)
        elif os.path.isdir(args.images):
            _, i = scan_dir(args.images)
            image_files.extend(i)
    elif not args.source:
        default_image_dir = os.path.join("data", "tasks", args.task, "raw_images")
        if os.path.isdir(default_image_dir):
            _, i = scan_dir(default_image_dir)
            image_files.extend(i)

    # 重複排除 & ソート
    video_files = sorted(list(dict.fromkeys(video_files)))
    image_files = sorted(list(dict.fromkeys(image_files)))

    return video_files, image_files


def main():
    args = parse_args()

    # 出力先パスの解決
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("data", "tasks", args.task, "extracted_frames")

    os.makedirs(output_dir, exist_ok=True)

    video_files, image_files = collect_sources(args)

    if not video_files and not image_files:
        print("=" * 60)
        print(f"[警告] 処理対象の動画または画像が見つかりませんでした (タスク: {args.task})")
        print("=" * 60)
        print("💡 以下のいずれかの方法でデータを配置・指定してください:")
        print(f"  1. 動画を配置: data/tasks/{args.task}/raw_videos/")
        print(f"  2. 画像を配置: data/tasks/{args.task}/raw_images/")
        print(f"  3. Webカメラで録画: make record TASK={args.task}")
        print(f"  4. 任意パスを指定: python3 scripts/extract_frames.py --source <動画または画像パス> --task {args.task}")
        print("=" * 60)
        return 1

    print("=" * 60)
    print("=== データ抽出・取り込みツール (動画 & 静止画対応) ===")
    print("=" * 60)
    print(f"🎯 タスク名             : {args.task}")
    print(f"🎬 検出動画数           : {len(video_files)} 本")
    print(f"📸 検出静止画数         : {len(image_files)} 枚")
    print(f"💾 出力先ディレクトリ   : {output_dir}")
    print(f"⚙️ EXIF自動回転補正     : {'無効' if args.no_auto_orient else '有効'}")
    if args.max_dim > 0:
        print(f"📏 最大長辺サイズ制限   : {args.max_dim}px")
    if args.diff_threshold > 0:
        print(f"🔍 差分スキップ閾値     : {args.diff_threshold}")
    print("=" * 60)

    total_video_saved = 0
    for v_path in video_files:
        total_video_saved += process_video(v_path, output_dir, args)

    total_image_saved = 0
    if image_files:
        total_image_saved += process_images(image_files, output_dir, args)

    grand_total = total_video_saved + total_image_saved

    print("\n" + "=" * 60)
    print(f"🎉 [完了] 合計 {grand_total} 枚の画像を出力しました！")
    print(f"  ・動画切り出し : {total_video_saved} 枚")
    print(f"  ・静止画取込   : {total_image_saved} 枚")
    print(f"  ・保存先       : {output_dir}")
    print("=" * 60)
    print(f"\n👉 次のステップ:")
    print(f"  アノテーション (ラベル付け) を開始:")
    print(f"  make label TASK={args.task}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
