#!/usr/bin/env python3
"""
extract_frames.py - 動画および静止画像からYOLO学習データセット用画像を抽出・取り込むスクリプト。

機能:
  - 動画ファイル / 動画フォルダ (data/tasks/<task>/raw_videos) からのフレーム切り出し
  - 静止画像ファイル / 画像フォルダ (data/tasks/<task>/raw_images) からのデータ取り込み (インポート)
  - 道路俯瞰視点変換 (Bird's Eye View / IPM: 逆透視投影変換) 対応 (--bev)
    - ZED 1 物理カメラモデル対応 (地上高56cm, 水平設置 pitch=0°, 消失点y=540)
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
from typing import List, Optional, Tuple

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
                        help="[動画用] 何秒ごとに1枚切り出すか (例: 1.0 = 1秒に1枚, 0.5 = 0.5秒に1枚)")
    parser.add_argument("--frame-interval", type=int, default=0,
                        help="[動画用] フレーム間隔指定 (0の場合は --every-sec を使用)")
    parser.add_argument("--prefix", type=str, default="",
                        help="動画切り出し画像のファイル名プレフィックス (省略時: 通常は frame, BEV時は frame_bev)")
    parser.add_argument("--img-prefix", type=str, default="",
                        help="静止画像取り込み時のファイル名プレフィックス (省略時: 通常は img, BEV時は img_bev)")
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

    # BEV (Bird's Eye View / 逆透視投影変換 IPM) 物理カメラモデルパラメータ
    parser.add_argument("--bev", action="store_true",
                        help="道路面の俯瞰視点 (BEV / IPM) 変換を適用する (1:1 正方形等方性)")
    parser.add_argument("--cam-height", type=float, default=0.56,
                        help="カメラ設置地上高 [m] (デフォルト: 0.56m = 56cm)")
    parser.add_argument("--cam-pitch", type=float, default=0.0,
                        help="カメラ俯仰角 [deg] (0.0 = 地面と水平)")
    parser.add_argument("--cam-focal", type=float, default=1050.0,
                        help="1080p基準の焦点距離 [px] (ZED 1: ~1050px)")
    parser.add_argument("--bev-dist-min", type=float, default=1.15,
                        help="BEV前方最小距離 [m] (バンパー除外: 1.15m)")
    parser.add_argument("--bev-dist-max", type=float, default=10.0,
                        help="BEV前方最大距離 [m] (デフォルト: 10.0m)")
    parser.add_argument("--bev-lateral-max", type=float, default=5.0,
                        help="BEV左右最大半幅 [m] (左右合計 = 2 * lateral_max, 5.0mで横幅10.0m = 縦10mと1:1正方形)")
    parser.add_argument("--bev-origin-bottom", action="store_true", default=True,
                        help="画像の真ん中下端 (Width/2, Height) をロボット/カメラ原点 (X=0, Y=0) に一致させる")
    parser.add_argument("--no-bev-origin-bottom", dest="bev_origin_bottom", action="store_false",
                        help="画像最下端をカメラ原点ではなく最短視認距離 (X_min) に合わせる")
    parser.add_argument("--bev-gamma", type=float, default=1.6,
                        help="BEV変換時のガンマ補正値 (1.0で無補正、1.6で暗い路面・白線を自然に増感)")
    parser.add_argument("--bev-out-w", type=int, default=1080,
                        help="BEV出力横幅 (デフォルト: 1080px 正方形)")
    parser.add_argument("--bev-out-h", type=int, default=1080,
                        help="BEV出力高さ (デフォルト: 1080px 正方形)")
    parser.add_argument("--bev-save-video", action="store_true",
                        help="BEV変換後の動画ファイル (.mp4) も生成・保存する")
    parser.add_argument("--bev-video-dir", type=str, default="",
                        help="BEV動画の保存先 (省略時は data/tasks/<task>/bev_videos)")

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


def build_bev_transform(src_w: int, src_h: int, args) -> Tuple[np.ndarray, Tuple[int, int], Optional[np.ndarray]]:
    """ZED 1 物理カメラモデルに基づき 1:1 実寸等方性 BEV (IPM) 射影変換行列 M、出力サイズ、ガンマ補正用LUTを生成"""
    scale_x = src_w / 1920.0
    scale_y = src_h / 1080.0

    cx = src_w / 2.0
    cy = src_h / 2.0
    f_px = args.cam_focal * scale_x
    h_cam = args.cam_height

    dist_min = max(0.5, args.bev_dist_min)
    dist_max = max(dist_min + 1.0, args.bev_dist_max)
    y_lat = max(0.5, args.bev_lateral_max)

    # 4隅の路面実世界座標 (X: 前方 [m], Y: 左右 [m]) をカメラ画像 (u, v) へ投影
    # u = cx + f * (Y / X)
    # v = cy + f * (h_cam / X)
    src_pts = np.float32([
        [cx - f_px * (y_lat / dist_min), cy + f_px * (h_cam / dist_min)],  # 手前左
        [cx + f_px * (y_lat / dist_min), cy + f_px * (h_cam / dist_min)],  # 手前右
        [cx + f_px * (y_lat / dist_max), cy + f_px * (h_cam / dist_max)],  # 奥右
        [cx - f_px * (y_lat / dist_max), cy + f_px * (h_cam / dist_max)]   # 奥左
    ])

    out_w = args.bev_out_w if args.bev_out_w > 0 else 1080
    out_h = args.bev_out_h if args.bev_out_h > 0 else 1080

    # 画像最下端 (out_w/2, out_h) をカメラ真下の原点 (X=0, Y=0) に配置
    if getattr(args, "bev_origin_bottom", True):
        v_near = (1.0 - (dist_min / dist_max)) * out_h
    else:
        v_near = float(out_h)

    dst_pts = np.float32([
        [0, v_near],
        [out_w, v_near],
        [out_w, 0],
        [0, 0]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    gamma_lut = None
    if args.bev_gamma > 0 and abs(args.bev_gamma - 1.0) > 0.01:
        inv_gamma = 1.0 / args.bev_gamma
        gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return M, (out_w, out_h), gamma_lut


def apply_bev(img: np.ndarray, M: np.ndarray, out_size: Tuple[int, int], gamma_lut: Optional[np.ndarray] = None) -> np.ndarray:
    """画像にBEV変換およびガンマ補正を適用"""
    bev_img = cv2.warpPerspective(img, M, out_size, flags=cv2.INTER_LINEAR)
    if gamma_lut is not None:
        bev_img = cv2.LUT(bev_img, gamma_lut)
    return bev_img


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
    """動画からフレームを切り出して保存 (BEVオプション対応)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[エラー] 動画を開けませんでした: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # 切り出し間隔の決定
    if args.frame_interval > 0:
        step = args.frame_interval
    else:
        step = max(1, int(round(fps * args.every_sec)))

    prefix = args.prefix if args.prefix else ("frame_bev" if args.bev else "frame")

    print(f"\n[動画処理] {os.path.basename(video_path)}")
    print(f"  解像度: {width}x{height}, 総フレーム数: {total_frames}, FPS: {fps:.2f}")
    print(f"  切り出し間隔: {step}フレームごと (約{step/fps:.2f}秒ごと)")
    if args.bev:
        print(f"  🦅 ZED 1 BEV変換: 地上高={args.cam_height}m, 水平 pitch={args.cam_pitch}°, 距離範囲={args.bev_dist_min}m〜{args.bev_dist_max}m, ガンマ={args.bev_gamma}")

    # BEV 変換行列の初期化
    M, out_size, gamma_lut = (None, (width, height), None)
    if args.bev:
        M, out_size, gamma_lut = build_bev_transform(width, height, args)

    # BEV動画の書き出し準備
    video_writer = None
    if args.bev and args.bev_save_video:
        bev_video_dir = args.bev_video_dir if args.bev_video_dir else os.path.join("data", "tasks", args.task, "bev_videos")
        os.makedirs(bev_video_dir, exist_ok=True)
        bev_video_path = os.path.join(bev_video_dir, f"bev_{video_basename}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(bev_video_path, fourcc, fps, out_size)
        print(f"  🎬 BEV動画出力先: {bev_video_path}")

    saved_count = 0
    frame_idx = 0
    prev_gray = None

    pbar = tqdm(total=total_frames, desc=video_basename, unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # BEV変換
        processed_frame = frame
        if args.bev:
            processed_frame = apply_bev(frame, M, out_size, gamma_lut)

        if video_writer is not None:
            video_writer.write(processed_frame)

        if frame_idx % step == 0:
            should_save = True

            # 差分フィルタ (静止シーンの重複スキップ)
            if args.diff_threshold > 0:
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = diff.mean()
                    if mean_diff < args.diff_threshold:
                        should_save = False
                if should_save:
                    prev_gray = gray

            if should_save:
                out_img = resize_if_needed(processed_frame, args.max_dim)
                out_filename = f"{prefix}_{video_basename}_{frame_idx:06d}.{args.format}"
                out_path = os.path.join(output_dir, out_filename)
                save_image(out_img, out_path, args.format, args.quality)
                saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if video_writer is not None:
        video_writer.release()
    print(f"  -> {saved_count} 枚のフレームを保存しました")
    return saved_count


def process_images(image_files: List[str], output_dir: str, args) -> int:
    """静止画像群を読み込み、EXIF補正・BEV・リサイズ等を行って保存"""
    if not image_files:
        return 0

    print(f"\n[静止画取り込み] 対象画像数: {len(image_files)} 枚")
    prefix = args.img_prefix if args.img_prefix else ("img_bev" if args.bev else "img")

    saved_count = 0
    prev_gray = None

    pbar = tqdm(total=len(image_files), desc="Images Import", unit="img")

    for idx, img_path in enumerate(image_files):
        img = load_image_with_exif_orientation(img_path, auto_orient=not args.no_auto_orient)
        if img is None:
            print(f"\n[警告] 画像の読み込みに失敗しました: {img_path}")
            pbar.update(1)
            continue

        processed_img = img
        if args.bev:
            h, w = img.shape[:2]
            M, out_size, gamma_lut = build_bev_transform(w, h, args)
            processed_img = apply_bev(img, M, out_size, gamma_lut)

        should_save = True
        if args.diff_threshold > 0:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
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
            out_img = resize_if_needed(processed_img, args.max_dim)
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            out_filename = f"{prefix}_{img_basename}.{args.format}"
            out_path = os.path.join(output_dir, out_filename)

            save_image(out_img, out_path, args.format, args.quality)
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
    print(f"🦅 BEV (俯瞰視点) 変換  : {'有効' if args.bev else '無効'}")
    if args.bev:
        print(f"📐 ZED 1 カメラモデル   : 地上高 {args.cam_height}m, 傾斜角 {args.cam_pitch}° (水平)")
        print(f"📏 前方距離・左右幅     : 前方 {args.bev_dist_min}m〜{args.bev_dist_max}m, 左右幅 {args.bev_lateral_max*2:.1f}m")
        print(f"💡 ガンマ補正 (明度)    : {args.bev_gamma}")
        if args.bev_save_video:
            print("🎥 BEV動画生成          : 有効")
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
