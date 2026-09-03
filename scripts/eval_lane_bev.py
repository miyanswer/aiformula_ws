#!/usr/bin/env python3
"""
eval_lane_bev.py
shiho_lane_mask_v2_best.pth を用いて動画を推論し、
カメラ視点および BEV (Bird's-Eye View / 鳥瞰図) の横並び可視化画像を生成する。
"""

import math
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


# --- カメラ幾何パラメータ (ZED X @ sample_vehicle) ---
CAMERA_Z = 0.54        # 地上高 [m]
CAMERA_X = 0.055       # 車両中心からの前方オフセット [m]
ROLL_DEG = -88.2       # [deg]
PITCH_DEG = -0.2       # [deg]
YAW_DEG = -90.2        # [deg]

# カメラ内部パラメータ (HD1080)
FX = 763.175
FY = 763.175
CX = 990.039
CY = 543.449

# --- t_junction/bev_videos と完全に一致する BEV 物理寸法パラメータ ---
CAM_HEIGHT = 0.56        # カメラ地上高 [m]
CAM_FOCAL = 1050.0       # 1080p基準の焦点距離 [px]
BEV_DIST_MIN = 1.15      # 前方最小距離 [m]
BEV_DIST_MAX = 10.0      # 前方最大距離 [m]
BEV_LATERAL_MAX = 5.0    # 左右最大半幅 [m] (全幅 10.0m)
BEV_OUT_W = 1080         # 出力幅 [px]
BEV_OUT_H = 1080         # 出力高さ [px]


def build_tjunction_bev_transform(src_w: int = 1920, src_h: int = 1080):
    """
    data/tasks/t_junction/bev_videos と完全に同一の 1:1 実寸正方形 BEV 変換行列 M を生成
    """
    cx = src_w / 2.0
    cy = src_h / 2.0
    f_px = CAM_FOCAL * (src_w / 1920.0)

    # 4隅の路面実世界座標をカメラ画像 (u, v) へ投影
    src_pts = np.float32([
        [cx - f_px * (BEV_LATERAL_MAX / BEV_DIST_MIN), cy + f_px * (CAM_HEIGHT / BEV_DIST_MIN)],  # 手前左
        [cx + f_px * (BEV_LATERAL_MAX / BEV_DIST_MIN), cy + f_px * (CAM_HEIGHT / BEV_DIST_MIN)],  # 手前右
        [cx + f_px * (BEV_LATERAL_MAX / BEV_DIST_MAX), cy + f_px * (CAM_HEIGHT / BEV_DIST_MAX)],  # 奥右
        [cx - f_px * (BEV_LATERAL_MAX / BEV_DIST_MAX), cy + f_px * (CAM_HEIGHT / BEV_DIST_MAX)]   # 奥左
    ])

    v_near = (1.0 - (BEV_DIST_MIN / BEV_DIST_MAX)) * BEV_OUT_H
    dst_pts = np.float32([
        [0, v_near],
        [BEV_OUT_W, v_near],
        [BEV_OUT_W, 0],
        [0, 0]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, (BEV_OUT_W, BEV_OUT_H)


def load_yolop_model(weights_path: str, device: torch.device):
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

    prob = torch.softmax(ll_seg_mask_raw, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask[:cut_y, :] = 0
    return mask


def draw_bev_grid(bev_img):
    """BEV 上に距離グリッド線、自車位置、スケール文字を描画 (t_junction BEV 1:1 等方規格)"""
    h, w = bev_img.shape[:2]
    # v_near: 前方 BEV_DIST_MIN (1.15m) のライン
    # 最下端 v=h: 自車カメラ位置 X=0m
    # 最上端 v=0: 前方 BEV_DIST_MAX (10.0m)
    
    # 距離線 (X: 前方 2m, 4m, 6m, 8m, 10m)
    dist_marks = [2.0, 4.0, 6.0, 8.0, 10.0]
    for d in dist_marks:
        v = int((1.0 - (d / BEV_DIST_MAX)) * h)
        if 0 <= v < h:
            color = (0, 180, 255) if d == 10.0 else (90, 90, 90)
            cv2.line(bev_img, (0, v), (w, v), color, 1, cv2.LINE_AA)
            cv2.putText(bev_img, f"{d:.0f}m", (15, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # 左右幅線 (Y: -4m, -3m, -2m, -1m, 0m, 1m, 2m, 3m, 4m)
    for y_val in [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
        u = int((y_val + BEV_LATERAL_MAX) / (2 * BEV_LATERAL_MAX) * w)
        if 0 <= u < w:
            color = (0, 120, 255) if y_val == 0.0 else (60, 60, 60)
            thickness = 2 if y_val == 0.0 else 1
            cv2.line(bev_img, (u, 0), (u, h), color, thickness, cv2.LINE_AA)
            if y_val != 0.0:
                cv2.putText(bev_img, f"{y_val:+.0f}m", (u - 15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    # 自車 (カート) アイコンの描画 (幅 1.0m x 長さ 1.6m)
    kart_w_m, kart_l_m = 1.0, 1.6
    u_center = int(w / 2)
    v_center = h  # 車両原点 (X=0)
    
    px_per_m = h / BEV_DIST_MAX
    kw_px = int((kart_w_m / (2 * BEV_LATERAL_MAX)) * w)
    kl_px = int(kart_l_m * px_per_m)

    pt1 = (u_center - kw_px, v_center - kl_px)
    pt2 = (u_center + kw_px, v_center)
    cv2.rectangle(bev_img, pt1, pt2, (0, 0, 255), -1)  # 赤いカート車体
    cv2.rectangle(bev_img, pt1, pt2, (255, 255, 255), 2)
    # 進行方向矢印
    cv2.arrowedLine(bev_img, (u_center, v_center), (u_center, v_center - kl_px - 35), (0, 255, 255), 3, tipLength=0.3)
    cv2.putText(bev_img, "Kart (Ego)", (u_center - 45, v_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def generate_bev_visual(frame, lane_mask, H_ipm, out_size):
    """カメラ画像 + 白線マスクから BEV 鳥瞰図を生成 (1080x1080)"""
    # 1. カメラ元画像の BEV 投影
    bev_base = cv2.warpPerspective(frame, H_ipm, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
    bev_dark = cv2.addWeighted(bev_base, 0.5, np.zeros_like(bev_base), 0.5, 0)

    # 2. 白線マスクの BEV 投影
    bev_mask = cv2.warpPerspective(lane_mask, H_ipm, out_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. 白線オーバーレイ (鮮やかなネオングリーン)
    bev_overlay = bev_dark.copy()
    bev_overlay[bev_mask == 1] = (0, 255, 0)
    bev_result = cv2.addWeighted(bev_dark, 0.3, bev_overlay, 0.7, 0)

    # 輪郭線を追加
    contours, _ = cv2.findContours(bev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bev_result, contours, -1, (255, 255, 255), 2)

    # 4. グリッドと自車アイコンを描画
    draw_bev_grid(bev_result)
    return bev_result


def main():
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model_path = str(WS_DIR / "models/shiho_lane_mask_v2_best.pth")
    print(f"Loading Model: {model_path} on {device}...")
    model = load_yolop_model(model_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    H_ipm, out_size = build_tjunction_bev_transform(1920, 1080)

    video_path = str(WS_DIR / "mp4/shihou_video_2026_08_24_13_51_47.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    out_dir = WS_DIR / "jpeg/bev_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 代表シーン (直線, カーブ, S字, 分岐, 横断歩道, 停止線)
    target_frames = [
        (500, "Straight (Approach)"),
        (1300, "Sharp Right Turn"),
        (2200, "Cone Chicane Area"),
        (3800, "Split Lane / Pylon"),
        (5000, "Double White Lane Straight"),
        (5500, "Shadow & Sunlight Zone"),
        (6000, "Crosswalk & Stopline Area"),
        (6500, "Final Corner Approach")
    ]

    print(f"Generating {len(target_frames)} BEV evaluation panels with exact t_junction dimensions (1080x1080)...")

    for f_idx, title in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # 白線推論
        mask = infer_lane(model, frame, device, transform, threshold=0.45)

        # カメラ視点オーバーレイ
        cam_vis = frame.copy()
        cam_vis[mask == 1] = (0, 255, 0)
        cam_vis = cv2.addWeighted(frame, 0.45, cam_vis, 0.55, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cam_vis, contours, -1, (255, 255, 255), 2)

        # BEV 鳥瞰図の生成 (1080x1080)
        bev_vis = generate_bev_visual(frame, mask, H_ipm, out_size)

        # 横並びパネル合成
        # カメラ映像 (1920x1080 -> 1080x720) と BEV (1080x1080 -> 720x720) を美しく合成
        h_target = 720
        w_cam = int(1920 * (h_target / 1080))  # 1280
        w_bev = h_target                      # 720 (1:1正方形維持)

        cam_vis_s = cv2.resize(cam_vis, (w_cam, h_target))
        bev_vis_s = cv2.resize(bev_vis, (w_bev, h_target))

        # ヘッダー作成
        header_cam = np.zeros((50, w_cam, 3), dtype=np.uint8)
        header_cam[:] = (35, 35, 35)
        cv2.putText(header_cam, f"Camera View (shiho_lane_mask_v2) | Frame {f_idx}: {title}", 
                    (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        header_bev = np.zeros((50, w_bev, 3), dtype=np.uint8)
        header_bev[:] = (25, 25, 25)
        cv2.putText(header_bev, "BEV (t_junction 1:1 / 10m x 10m)", 
                    (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        left_col = np.vstack([header_cam, cam_vis_s])
        right_col = np.vstack([header_bev, bev_vis_s])
        combined = np.hstack([left_col, right_col])

        out_file = out_dir / f"bev_frame_{f_idx:05d}.jpg"
        cv2.imwrite(str(out_file), combined)
        print(f"  -> Saved: {out_file.name}")

    cap.release()
    print(f"\n🎉 Finished! All BEV evaluation panels saved in: {out_dir}")


if __name__ == "__main__":
    main()
