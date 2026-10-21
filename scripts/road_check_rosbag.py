#!/usr/bin/env python3
"""
road_check_rosbag.py - 黒路面 (走れる路面) の検出を rosbag の映像で確かめる (表示用の画像を作るだけ. 動画は保存しない)

各コマで, 車載と同じ前処理 (上 45% を切り, letterbox) で走れる領域を出し, 次を数える:
  - 基準 (make_road_masks.py と同じ作り方: 公式の重みの走れる領域 + その近くの灰色 - 緑) との IoU
  - 緑の誤り: 路面とした画素のうち, 緑 (歩道・島) の画素の割合 (0 に近いほどよい)
  - 手前の路面: 画像の下 1/5 の帯で, 基準の路面のうち路面と出せた割合
--sheet を付けると, いくつかのコマで「重みごとの路面」を並べた画像を作る.

使い方:
  python3 scripts/road_check_rosbag.py rosbag/mcap/20261012_155111_gamepad.mcap \\
      --models models/honda_shihou_finetuned_best.pth models/honda_shihou_finetuned_road_best.pth --every 2.0 --sheet out.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
import make_road_masks as M  # noqa: E402
import verify_lane_rosbag as V  # noqa: E402
from topo_features import iter_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--every", type=float, default=2.0, help="何秒ごとに数えるか")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--sheet-times", default="", help="画像にするコマ (rosbag の先頭からの秒, カンマ区切り). 省略すると等間隔に 6 コマ")
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nets = [M.load_net(Path(p), dev) for p in args.models]
    official = M.load_net(WS_DIR / "models/pretrained/yolop_official.pth", dev)
    stats = [dict(iou=[], green=[], near=[]) for _ in nets]
    want = [float(x) for x in args.sheet_times.split(",") if x]
    tiles, t0, last = [], None, -1e9
    for t, data in iter_images(args.bag):
        t0 = t if t0 is None else t0
        rt = t - t0
        take_sheet = bool(want) and rt >= want[0]
        if rt - last < args.every and not take_sheet:
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        cut = int(img.shape[0] * V.TOP_CUT_RATIO)
        proc = img[cut:]
        ref = M.road_mask(proc, M.infer_da(official, dev, proc)).astype(bool)
        hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        green = (hsv[..., 0] > 35) & (hsv[..., 0] < 95) & (hsv[..., 1] > 35)
        band = np.zeros_like(ref)
        band[-proc.shape[0] // 5:] = True
        preds = []
        for st, net in zip(stats, nets):
            p = M.infer_da(net, dev, proc).astype(bool)
            preds.append(p)
            if rt - last >= args.every:
                st["iou"].append((p & ref).sum() / max((p | ref).sum(), 1))
                st["green"].append((p & green).sum() / max(p.sum(), 1))
                st["near"].append((p & ref & band).sum() / max((ref & band).sum(), 1))
        if rt - last >= args.every:
            last = rt
        if args.sheet and (take_sheet or (not want and len(tiles) < 6 and int(rt) % 40 < args.every)):
            if want:
                want.pop(0)
            row = []
            for name, p in zip(["基準"] + [Path(m).stem for m in args.models], [ref] + preds):
                ov = proc.copy()
                ov[p] = (0.5 * ov[p] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
                ov = cv2.resize(ov, (640, int(640 * proc.shape[0] / proc.shape[1])))
                # 見出し: 基準は ref, 重みは名前の共通部分を除いたもの (今の車載 = now, road, road_v2 など)
                lab = "ref" if name == "基準" else name.replace("honda_shihou_finetuned", "").replace("_best", "").strip("_") or "now"
                cv2.putText(ov, f"{rt:.1f}s {lab}", (6, 20), 0, 0.6, (0, 0, 0), 4)
                cv2.putText(ov, f"{rt:.1f}s {lab}", (6, 20), 0, 0.6, (255, 255, 255), 2)
                row.append(ov)
            tiles.append(np.hstack(row))
    print(Path(args.bag).name)
    for m, st in zip(args.models, stats):
        print(f"  {Path(m).name:45s} 基準との IoU 中央値 {np.median(st['iou']):.2f} (10% {np.percentile(st['iou'], 10):.2f})"
              f"  緑の誤り {np.median(st['green']) * 100:.1f}%  手前の路面 {np.median(st['near']) * 100:.0f}%  ({len(st['iou'])} コマ)")
    if args.sheet and tiles:
        cv2.imwrite(args.sheet, np.vstack(tiles))
        print("  画像 ->", args.sheet)


if __name__ == "__main__":
    main()
