#!/usr/bin/env python3
"""
録画した動画ファイルから、YOLO学習データセット用のアノテーション画像を抽出・切り出すスクリプト。

機能:
  - 1つの動画またはディレクトリ内の全動画を一括処理
  - 指定フレーム間隔 (例: 15フレーム毎) または 指定秒間隔 (例: 1秒毎) で抽出
  - 差分があまりない静止画の自動スキップ機能 (オプション: --diff-threshold)
  - 切り出し画像のリサイズ・画質設定
"""

import argparse
import glob
import os
import sys
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="動画からYOLOアノテーション用静止画を切り出すツール")
    parser.add_argument("--video", type=str, default="data/raw_videos", help="動画ファイルパス または 動画フォルダパス")
    parser.add_argument("--output-dir", type=str, default="data/extracted_frames", help="切り出し画像の保存先")
    parser.add_argument("--every-sec", type=float, default=1.0, help="何秒ごとに1枚切り出すか (例: 1.0 = 1秒に1枚)")
    parser.add_argument("--frame-interval", type=int, default=0, help="フレーム間隔指定 (0の場合は --every-sec を使用)")
    parser.add_argument("--prefix", type=str, default="frame", help="保存画像のファイル名プレフィックス")
    parser.add_argument("--format", type=str, choices=["jpg", "png"], default="jpg", help="保存画像フォーマット")
    parser.add_argument("--diff-threshold", type=float, default=0.0, help="直前フレームとの平均輝度差の最小閾値 (0で無効。動きがないシーンの重複をスキップ)")
    return parser.parse_args()


def process_video(video_path, output_dir, args):
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

    print(f"\n[処理開始] 動画: {video_path}")
    print(f"  総フレーム数: {total_frames}, FPS: {fps:.2f}, 切り出し間隔: {step}フレームごと (約{step/fps:.2f}秒ごと)")

    saved_count = 0
    frame_idx = 0
    prev_gray = None

    pbar = tqdm(total=total_frames, desc=video_basename, unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
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
                out_filename = f"{args.prefix}_{video_basename}_{frame_idx:06d}.{args.format}"
                out_path = os.path.join(output_dir, out_filename)
                
                if args.format == "jpg":
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(out_path, frame)
                saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"  -> {saved_count} 枚の画像を保存しました: {output_dir}")
    return saved_count


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.video):
        video_files = [args.video]
    elif os.path.isdir(args.video):
        extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.MOV"]
        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(args.video, ext)))
        video_files = sorted(video_files)
    else:
        print(f"[エラー] 指定されたパスが見つかりません: {args.video}")
        return 1

    if not video_files:
        print(f"[警告] 処理対象の動画ファイルが見つかりませんでした: {args.video}")
        return 1

    print(f"=== フレーム切り出しツール ===")
    print(f"対象動画数: {len(video_files)}")
    print(f"出力先: {args.output_dir}")
    print("=============================")

    total_saved = 0
    for v_path in video_files:
        total_saved += process_video(v_path, args.output_dir, args)

    print(f"\n[完了] 全動画から合計 {total_saved} 枚の画像を抽出しました。")
    print(f"次のステップ: アノテーションツール (Roboflow / AnyLabeling / Labelme) でラベル付けを行ってください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
