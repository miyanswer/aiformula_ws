#!/usr/bin/env python3
"""
make_road_masks.py - 学習用の「黒路面 (走れる路面)」の答えを作る (honda_cropped_dataset / yolop_cropped_dataset の画像に)

元のデータセットの drivable_masks は使えない (honda は白線の答えの複製, yolop は緑の歩道を塗ったものが多い) ので,
同じ画像に答えを作り直して data/road_masks/<データセット>/<train|val>/<名前>.png に保存する (元のファイルは変えない):
  1. 公式の YOLOP の重み (models/pretrained/yolop_official.pth) が出す走れる領域を土台に
  2. そのすぐ近くの灰色の路面 (白線を含む) を足し (遠くの灰色の塀や建物は入れない)
  3. 緑 (歩道・島の着色舗装) を除き, 小さな穴を埋め, 画像の下端につながる部分だけ残す
  白飛び・真っ暗の画像と路面の割合が極端な答えは usable.txt に入れない (学習に使わない)
  road_mask_editor.py で手で直した答え (edited.txt) は作り直しても上書きしない
緑は答えを作るためだけに使う (走行中の手がかりには使わない).

使い方:
  python3 scripts/make_road_masks.py [--sheet 確かめ用の画像]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
sys.path.insert(0, str(WS_DIR / "src/aiformula_base/oit_navigation/oit_navigation/yolop"))
import verify_lane_rosbag as V  # noqa: E402
from lib.config import cfg  # noqa: E402
from lib.models import get_net  # noqa: E402

DATASETS = ["honda_cropped_dataset", "yolop_cropped_dataset"]
NEAR_PX = 41               # 公式の走れる領域からこの画素以内の灰色だけ路面に足す (画像は 1920 x 594)


def load_net(path, dev):
    net = get_net(cfg)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("state_dict", ck))
    return net.to(dev).float().eval()


def infer_da(net, dev, proc):
    """切り出し済みの画像 (下 55%) に letterbox をかけて走れる領域 (0/1) を返す (車載と同じ前処理)."""
    mean, std = np.asarray(V.NORM_MEAN, np.float32), np.asarray(V.NORM_STD, np.float32)
    ph, pw = proc.shape[:2]
    r = min(V.INPUT_SIZE / ph, V.INPUT_SIZE / pw)
    nh, nw = int(round(ph * r)), int(round(pw * r))
    top, left = (V.INPUT_SIZE - nh) // 2, (V.INPUT_SIZE - nw) // 2
    canvas = np.full((V.INPUT_SIZE, V.INPUT_SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(proc, (nw, nh), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.ascontiguousarray(((canvas.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])).to(dev)
    with torch.inference_mode():
        _, da, _ = net(x)
    seg = (da[0, 1, top:top + nh, left:left + nw] > da[0, 0, top:top + nh, left:left + nw]).to(torch.uint8).cpu().numpy()
    return cv2.resize(seg, (pw, ph), interpolation=cv2.INTER_NEAREST)


def road_mask(img, da):
    """公式の YOLOP の走れる領域を土台に, そのすぐ近く (NEAR_PX 以内) の灰色の路面と白線を足し, 緑を除く."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] > 35) & (hsv[..., 0] < 95) & (hsv[..., 1] > 35)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    gray = (hsv[..., 1] < 45) & (hsv[..., 2] > 30)
    near = cv2.dilate((da > 0).astype(np.uint8), np.ones((NEAR_PX, NEAR_PX), np.uint8)) > 0
    road = ((da > 0) | (gray & near)) & ~green
    road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # 下端につながる部分だけ残す (遠くの塀などを除く)
    n, lab = cv2.connectedComponents(road)
    keep = set(np.unique(lab[-3:, :])) - {0}
    return np.isin(lab, list(keep)).astype(np.uint8)


def usable(img, m):
    """白飛び・真っ暗の画像と, 路面の割合が極端な答えは学習に使わない."""
    return 25 < img.mean() < 220 and 0.25 < m.mean() < 0.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="")
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = load_net(WS_DIR / "models/pretrained/yolop_official.pth", dev)
    tiles = []
    rng = np.random.default_rng(1)
    for d in DATASETS:
        for split in ("train", "val"):
            out_dir = WS_DIR / "data/road_masks" / d / split
            os.makedirs(out_dir, exist_ok=True)
            files = sorted(glob.glob(str(WS_DIR / "data" / d / "images" / split / "*")))
            show = set(rng.choice(len(files), 3, replace=False)) if split == "val" else set()
            cover, ok_list = [], []
            edited = set((out_dir / "edited.txt").read_text().split()) if (out_dir / "edited.txt").exists() else set()
            prev_ok = set((out_dir / "usable.txt").read_text().split()) if (out_dir / "usable.txt").exists() else set()
            for k, f in enumerate(files):
                img = cv2.imread(f)
                stem = Path(f).stem
                if stem in edited:                          # 手で直した答えはそのまま (学習に使うかも手で決めたもの)
                    m = (cv2.imread(str(out_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                    if stem in prev_ok:
                        ok_list.append(stem)
                else:
                    m = road_mask(img, infer_da(net, dev, img))
                    cv2.imwrite(str(out_dir / f"{stem}.png"), m * 255)
                    if usable(img, m):
                        ok_list.append(stem)
                cover.append(m.mean())
                if k in show:
                    ov = img.copy()
                    ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
                    ov = cv2.resize(ov, (960, 297))
                    cv2.putText(ov, f"{d[:5]} {split} {Path(f).stem[-10:]} road {m.mean() * 100:.0f}%", (8, 24), 0, 0.7,
                                (255, 255, 255), 2)
                    tiles.append(ov)
            (out_dir / "usable.txt").write_text("\n".join(ok_list))
            print(f"{d} {split}: {len(files)} 枚 (学習に使う {len(ok_list)} 枚)  路面の割合 中央値 {np.median(cover) * 100:.0f}%  "
                  f"10% {np.percentile(cover, 10) * 100:.0f}%  -> {out_dir}", flush=True)
    if args.sheet and tiles:
        cv2.imwrite(args.sheet, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]))


if __name__ == "__main__":
    main()
