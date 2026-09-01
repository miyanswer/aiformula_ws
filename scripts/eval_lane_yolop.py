#!/usr/bin/env python3
"""
eval_lane_yolop.py - YOLOP 白線認識モデルの動画インタラクティブ・デバッグツール

機能:
  - 学習済みモデル（.pth）と任意の動画（mp4）を選択し、白線検知のリアルタイムプレビュー＆デバッグ
  - コマ送り・巻き戻し・一時停止・スクリーンショット保存・動画(MP4)書き出し対応
  - 通常モード / 方法A (上部マスク) / 方法B (下部クロップ拡大) を切り替えて比較可能

使い方:
  # 対話的にモデルと動画を選んで起動
  make debug-lane
  # または
  python3 scripts/eval_lane_yolop.py

  # 特定のモデルと動画を指定して起動
  python3 scripts/eval_lane_yolop.py --weights models/shiho_lane_crop_best.pth --video mp4/shihou_video_2026_08_24_13_51_47.mp4

  # GUIなしで推論結果動画 (MP4) を書き出す場合
  python3 scripts/eval_lane_yolop.py --save-video --no-gui
"""

import argparse
import glob
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
    parser = argparse.ArgumentParser(description="YOLOP 白線認識モデルの動画デバッグツール")
    parser.add_argument("--video", type=str, default="",
                        help="テスト対象の動画パス (省略時は一覧から選択)")
    parser.add_argument("--weights", type=str, default="",
                        help="評価するモデルの重みファイル (省略時は一覧から選択)")
    parser.add_argument("--output-dir", type=str, default=str(WS_DIR / "jpeg"),
                        help="結果保存先フォルダ (デフォルト: jpeg/)")
    parser.add_argument("--save-video", action="store_true",
                        help="推論結果の動画 (MP4) を書き出す")
    parser.add_argument("--roi-mode", type=str, choices=["auto", "none", "mask_top", "crop_bottom"],
                        default="auto",
                        help="推論ROIモード: 'auto' (モデル名から自動判定), 'mask_top' (方法A), 'crop_bottom' (方法B), 'none'")
    parser.add_argument("--top-cut-ratio", type=float, default=0.45,
                        help="上部カットの割合 (デフォルト: 0.45 = 上部45%)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="処理する最大フレーム数 (0 で動画の最後まで)")
    parser.add_argument("--no-gui", action="store_true",
                        help="ウィンドウ表示を行わず、ファイル書き出しのみ行う")
    parser.add_argument("--device", type=str, default="auto",
                        help="推論デバイス: 'cuda', 'mps', 'cpu', 'auto'")
    return parser.parse_args()


def detect_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_weights(specified: str) -> str:
    if specified and os.path.exists(specified):
        return specified

    weight_candidates = []
    # 探索先
    search_dirs = [
        WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights",
        WS_DIR / "models"
    ]
    for sdir in search_dirs:
        if sdir.exists():
            weight_candidates.extend(sorted(glob.glob(str(sdir / "*.pth"))))

    # 重複削除
    weight_candidates = list(dict.fromkeys(weight_candidates))

    if not weight_candidates:
        print("Error: No .pth model weights found!")
        sys.exit(1)

    if len(weight_candidates) == 1:
        return weight_candidates[0]

    print("\n🧠 利用可能なモデル重み一覧 (.pth):")
    for idx, wp in enumerate(weight_candidates, 1):
        mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(wp)))
        size_mb = os.path.getsize(wp) / (1024 * 1024)
        print(f"  [{idx:2d}] {Path(wp).name:<38} ({size_mb:4.1f} MB, {mtime})")

    try:
        choice = input(f"\nテストに使用するモデル番号を入力してください [デフォルト: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(0)

    if not choice:
        choice = "1"

    if choice.isdigit() and 1 <= int(choice) <= len(weight_candidates):
        return weight_candidates[int(choice) - 1]

    return weight_candidates[0]


def select_video(specified: str) -> str:
    if specified and os.path.exists(specified):
        return specified

    video_dir = WS_DIR / "mp4"
    video_candidates = sorted(
        glob.glob(str(video_dir / "*.mp4")) +
        glob.glob(str(video_dir / "*.MOV")) +
        glob.glob(str(video_dir / "*.avi"))
    )

    if not video_candidates:
        print(f"Error: No video files found in {video_dir}")
        sys.exit(1)

    if len(video_candidates) == 1:
        return video_candidates[0]

    print("\n🎬 利用可能な動画一覧 (mp4/):")
    for idx, vp in enumerate(video_candidates, 1):
        size_mb = os.path.getsize(vp) / (1024 * 1024)
        print(f"  [{idx:2d}] {Path(vp).name:<45} ({size_mb:5.1f} MB)")

    try:
        choice = input(f"\nテストしたい動画の番号を入力してください [デフォルト: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(0)

    if not choice:
        choice = "1"

    if choice.isdigit() and 1 <= int(choice) <= len(video_candidates):
        return video_candidates[int(choice) - 1]

    return video_candidates[0]


def main():
    args = parse_args()
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = select_weights(args.weights)
    video_path = select_video(args.video)

    # ROIモードの自動判定
    roi_mode = args.roi_mode
    if roi_mode == "auto":
        w_name = Path(weights_path).name.lower()
        if "crop" in w_name:
            roi_mode = "crop_bottom"
        elif "mask" in w_name:
            roi_mode = "mask_top"
        else:
            roi_mode = "none"

    print("=" * 65)
    print(f"🔍 YOLOP Lane Detection Debugger")
    print(f"Model:    {Path(weights_path).name}")
    print(f"Video:    {Path(video_path).name}")
    print(f"ROI Mode: {roi_mode} (Top Cut: {args.top_cut_ratio * 100:.1f}%)")
    print(f"Device:   {device}")
    print("=" * 65)

    # モデルのロード
    model = get_net(cfg)
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = None
    if args.save_video:
        video_out_path = output_dir / f"debug_{Path(weights_path).stem}_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w_orig, h_orig))
        print(f"📹 Recording output to: {video_out_path}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cut_y = int(h_orig * args.top_cut_ratio)

    has_gui = not args.no_gui and ("DISPLAY" in os.environ or sys.platform == "darwin")
    window_name = f"YOLOP Lane Debug: {Path(weights_path).name}"
    if has_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("\n🎮 コントロールキー:")
    print("  [SPACE]   : 一時停止 / 再生")
    print("  [d] / [→] : 1フレーム進む (コマ送り)")
    print("  [a] / [←] : 1フレーム戻る (コマ戻し)")
    print("  [f] / [r] : 5秒早送り / 巻き戻し")
    print("  [m]       : 白線オーバーレイ表示切り替え")
    print("  [s]       : 現在フレームをスクリーンショット保存 (jpeg/)")
    print("  [q] / ESC : 終了\n")

    current_frame = 0
    paused = False
    show_mask = True
    delay = max(1, int(1000 / fps))

    pbar = tqdm(total=total_frames, desc="Processing") if not has_gui else None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nReached end of video.")
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pbar:
                pbar.update(1)

        # 推論
        t0 = time.time()
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

        infer_time_ms = (time.time() - t0) * 1000

        # 可視化画像の作成
        vis = frame.copy()
        if show_mask:
            # 半透明緑色で白線をオーバーレイ
            overlay = vis.copy()
            overlay[ll_mask == 1] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            vis[ll_mask == 1] = [0, 255, 100]  # 白線輪郭を明るく

        # ROI 切断線のガイド表示 (オプション)
        if roi_mode in ["crop_bottom", "mask_top"]:
            cv2.line(vis, (0, cut_y), (w_orig, cut_y), (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"ROI Cut ({args.top_cut_ratio*100:.0f}%)", (20, cut_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 情報テキストの描画
        status_text = "PAUSED" if paused else "PLAYING"
        hud_color = (0, 0, 255) if paused else (0, 255, 0)
        cv2.putText(vis, f"[{status_text}] Frame: {current_frame}/{total_frames} ({(current_frame/fps):.1f}s)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
        cv2.putText(vis, f"Model: {Path(weights_path).name} | Mode: {roi_mode} | Infer: {infer_time_ms:.1f}ms",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if out_video is not None and not paused:
            out_video.write(vis)

        # 代表的な静止画を自動保存
        if current_frame in [50, 200, 500, 1000, 2000]:
            img_out_path = output_dir / f"debug_frame_{current_frame}.jpg"
            cv2.imwrite(str(img_out_path), vis)

        if has_gui:
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(0 if paused else delay) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
            elif key == ord('d') or key == 83:  # 'd' or Right arrow
                paused = True
                ret, frame = cap.read()
                if ret:
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            elif key == ord('a') or key == 81:  # 'a' or Left arrow
                paused = True
                current_frame = max(0, current_frame - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
            elif key == ord('f'):  # Forward 5 sec
                current_frame = min(total_frames - 1, current_frame + int(fps * 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('r'):  # Rewind 5 sec
                current_frame = max(0, current_frame - int(fps * 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('m'):  # Toggle mask
                show_mask = not show_mask
            elif key == ord('s'):  # Screenshot
                screenshot_path = output_dir / f"screenshot_{Path(video_path).stem}_f{current_frame:06d}.jpg"
                cv2.imwrite(str(screenshot_path), vis)
                print(f"\n📸 Screenshot saved: {screenshot_path}")
        else:
            # ヘッドレス環境での制限処理
            if args.max_frames > 0 and current_frame >= args.max_frames:
                break

    cap.release()
    if out_video is not None:
        out_video.release()
    if has_gui:
        cv2.destroyAllWindows()
    if pbar:
        pbar.close()

    print(f"\n✅ Debug session finished!")
    if args.save_video:
        print(f"Saved video to: {video_out_path}")
    print(f"Sample screenshots saved in: {output_dir}")


if __name__ == "__main__":
    main()
