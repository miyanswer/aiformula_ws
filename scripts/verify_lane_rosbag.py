#!/usr/bin/env python3
"""
verify_lane_rosbag.py - rosbag (mcap) のカメラ映像で YOLOP の白線認識を確かめる (前処理の修正前 / 修正後を並べて表示)

  左: 修正前 = 上部 45% を切り落とし, 残りを縦に引き伸ばして 640x640 (2026-10 までの車載の前処理)
  右: 修正後 = 上部 45% を切り落とし, 縦横比を保って 640x640 に収め余白を 114 で埋める (letterbox.
      学習スクリプト scripts/train_lane_yolop.py と同じ縮尺. 車載の yolop_lane_backend.py もこちらに修正)
  緑 = YOLOP が白線と判定した画素, 黄 = そこから取り出した白線 (1 本ずつ), 下の帯 = ここまでの集計

使い方 (Mac 本体の python3 で動かす. rosbag の読み出しだけ .venv の python に任せる):
  python3 scripts/verify_lane_rosbag.py                                   # rosbag/mcap/ の一覧から選ぶ
  python3 scripts/verify_lane_rosbag.py rosbag/mcap/20261012_153125_6lane.mcap
  python3 scripts/verify_lane_rosbag.py <bag> --start 120 --device cpu

キー: space 一時停止 / 再開, → (d) 一時停止中に 1 コマ進む, f 早送り (3 コマに 1 コマ) 切替,
      r 集計をリセット, q / Esc 終了.  映像は保存しない.
"""

import argparse
import glob
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

WS_DIR = Path(__file__).resolve().parent.parent
TOPIC = "/camera/image/compressed"
TOP_CUT_RATIO = 0.45                   # 車載 lane_detector_params.yaml の top_cut_ratio と同じ
INPUT_SIZE = 640
NORM_MEAN = (0.485, 0.456, 0.406)      # 車載と同じく BGR のまま正規化
NORM_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- rosbag 読み出し (.venv の python で動く側)
def run_reader(bag: str, start: float):
    """rosbag の画像を [ヘッダ長 4B][JSON][JPEG 長 4B][JPEG] で標準出力へ流す (rosbag は読み取りのみ)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    out = sys.stdout.buffer
    with open(bag, "rb") as fp:
        reader = make_reader(fp, decoder_factories=[DecoderFactory()])
        t0 = None
        for _, _, msg, m in reader.iter_decoded_messages(topics=[TOPIC]):
            t = msg.log_time / 1e9
            t0 = t if t0 is None else t0
            if t - t0 < start:
                continue
            head = json.dumps({"t": round(t - t0, 2)}).encode()
            data = bytes(m.data)
            try:
                out.write(struct.pack("<I", len(head)) + head + struct.pack("<I", len(data)) + data)
                out.flush()
            except BrokenPipeError:
                return


# ---------------------------------------------------------------- 白線マスク -> 白線ごとの点列
# 車載 oit_navigation/lane/mask_lines.py と同じ手順 (画像下端から上へ行ごとに白線画素の塊の中心を取り,
# 前の行の点と近いもの同士をつなぐ. 太すぎる塊 = 横断歩道・停止線は捨てる)
def extract_mask_lines(mask, top_ratio=TOP_CUT_RATIO, row_step_ratio=1 / 72, min_run_px_ratio=0.002,
                       max_run_width_ratio=0.12, gate_ratio=0.10, max_gap_rows=4, min_points=4):
    import numpy as np
    h, w = mask.shape[:2]
    top = int(h * top_ratio)
    step = max(1, int(round(h * row_step_ratio)))
    rows = list(range(h - 1, top, -step))
    chains = []
    for ri, v in enumerate(rows):
        depth = (v - top) / max(h - 1 - top, 1)
        scale = 0.4 + 0.6 * depth
        max_run = max_run_width_ratio * w * scale
        gate = max(3.0, gate_ratio * w * scale)
        row = mask[v] > 0
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        centers = [(s + e - 1) / 2.0 for s, e in zip(starts, ends) if min_run_px_ratio * w <= (e - s) <= max_run]
        cands = []
        for ci, ch in enumerate(chains):
            if ri - ch["last_i"] > max_gap_rows:
                continue
            pred = ch["u"][-1]
            if len(ch["u"]) >= 2:
                pred += (ch["u"][-1] - ch["u"][-2]) * (ri - ch["last_i"])
            for k, c in enumerate(centers):
                dist = abs(c - pred)
                if dist <= gate:
                    cands.append((dist, ci, k))
        used_c, used_k = set(), set()
        for dist, ci, k in sorted(cands):
            if ci in used_c or k in used_k:
                continue
            used_c.add(ci)
            used_k.add(k)
            chains[ci]["u"].append(centers[k])
            chains[ci]["v"].append(v)
            chains[ci]["last_i"] = ri
        for k, c in enumerate(centers):
            if k not in used_k:
                chains.append({"u": [c], "v": [v], "last_i": ri})
    return [ch for ch in chains if len(ch["u"]) >= min_points]


# ---------------------------------------------------------------- 表示側 (torch のある python で動く)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", help="rosbag (.mcap). 省略すると rosbag/mcap/ から選ぶ")
    ap.add_argument("--weights", default=str(WS_DIR / "models/honda_shihou_finetuned_best.pth"))
    ap.add_argument("--start", type=float, default=0.0, help="rosbag の先頭から何秒目から再生するか")
    ap.add_argument("--device", default="auto", help="auto / mps / cuda / cpu")
    ap.add_argument("--reader", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--snapshot", default="", help=argparse.SUPPRESS)   # 動作確認用: 最初の 1 コマを画像に書いて終わる
    args = ap.parse_args()

    if args.reader:
        run_reader(args.bag, args.start)
        return

    bag = args.bag
    if not bag:
        bags = sorted(glob.glob(str(WS_DIR / "rosbag/mcap/*.mcap")))
        if not bags:
            sys.exit("rosbag/mcap/ に .mcap がありません. rosbag を引数で指定してください")
        for i, b in enumerate(bags):
            print(f"  [{i}] {os.path.basename(b)}")
        bag = bags[int(input("番号を選んでください: "))]
    bag = os.path.abspath(bag)

    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    ydir = WS_DIR / "src/aiformula_base/oit_navigation/oit_navigation/yolop"
    sys.path.insert(0, str(ydir))
    from lib.config import cfg
    from lib.models import get_net

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if args.device in ("0", "cuda") else args.device)
    net = get_net(cfg)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    net = net.to(device).float().eval()
    mean = np.asarray(NORM_MEAN, np.float32)
    std = np.asarray(NORM_STD, np.float32)
    print(f"重み {args.weights}  device={device}")

    def infer(bgr, letterbox):
        h, w = bgr.shape[:2]
        cut = int(h * TOP_CUT_RATIO)
        proc = bgr[cut:]
        ph, pw = proc.shape[:2]
        if letterbox:
            r = min(INPUT_SIZE / ph, INPUT_SIZE / pw)
            nh, nw = int(round(ph * r)), int(round(pw * r))
            top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
            canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
            canvas[top:top + nh, left:left + nw] = cv2.resize(proc, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            nh, nw, top, left = INPUT_SIZE, INPUT_SIZE, 0, 0
            canvas = cv2.resize(proc, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = (canvas.astype(np.float32) / 255.0 - mean) / std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(device)
        with torch.inference_mode():
            _det, _da, ll = net(x)
        ll = ll[:, :, top:top + nh, left:left + nw].float()
        seg = (ll[0, 1] > ll[0, 0]).to(torch.uint8).cpu().numpy()
        mask = np.zeros((h, w), np.uint8)
        mask[cut:] = cv2.resize(seg, (pw, ph), interpolation=cv2.INTER_NEAREST)
        return mask

    reader = subprocess.Popen([str(WS_DIR / ".venv/bin/python"), os.path.abspath(__file__), bag, "--reader",
                               "--start", str(args.start)], stdout=subprocess.PIPE)
    inp = reader.stdout

    def read_frame():
        b = inp.read(4)
        if len(b) < 4:
            return None
        head = json.loads(inp.read(struct.unpack("<I", b)[0]))
        jpg = inp.read(struct.unpack("<I", inp.read(4))[0])
        return head, jpg

    font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
    F = ImageFont.truetype(font_path, 24) if os.path.exists(font_path) else ImageFont.load_default()
    Fs = ImageFont.truetype(font_path, 19) if os.path.exists(font_path) else ImageFont.load_default()
    PW, PH = 960, 540
    SIDES = [("修正前: 縦に引き伸ばし", False, (255, 150, 150)), ("修正後: 縦横比を保って余白 (学習時と同じ)", True, (150, 255, 150))]
    stats = [dict(frames=0, lines=0, zero=0, px=0) for _ in SIDES]

    def panel(img, mask, lines):
        v = img.copy()
        on = mask > 0
        v[on] = (0.35 * v[on] + 0.65 * np.array([0, 230, 0])).astype(np.uint8)
        for ln in lines:
            pts = np.stack([ln["u"], ln["v"]], 1).astype(np.int32)
            cv2.polylines(v, [pts], False, (0, 255, 255), 2)
        cv2.line(v, (0, int(v.shape[0] * TOP_CUT_RATIO)), (v.shape[1], int(v.shape[0] * TOP_CUT_RATIO)), (90, 90, 90), 1)
        return cv2.resize(v, (PW, PH), interpolation=cv2.INTER_LINEAR)

    WIN = "verify lane (rosbag)"
    if not args.snapshot:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 2 * PW, PH + 200)
    paused = fast = False
    n = 0
    print("キー: space 一時停止/再開, → 1 コマ進む (一時停止中), f 早送り, r 集計リセット, q 終了")
    while True:
        fr = read_frame()
        if fr is None:
            print("rosbag の終わりです. 何かキーで閉じます")
            cv2.waitKey(0)
            break
        head, jpg = fr
        n += 1
        if fast and n % 3:
            continue
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        panels = []
        for k, (_, lb, _) in enumerate(SIDES):
            t0 = time.time()
            mask = infer(img, lb)
            lines = extract_mask_lines(mask)
            st = stats[k]
            st["frames"] += 1
            st["lines"] += len(lines)
            st["zero"] += len(lines) == 0
            st["px"] += int(mask.sum())
            panels.append((panel(img, mask, lines), len(lines), (time.time() - t0) * 1000))
        frame = np.zeros((PH + 200, 2 * PW, 3), np.uint8)
        frame[90:90 + PH] = np.concatenate([p[0] for p in panels], 1)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        for k, (name, _, col) in enumerate(SIDES):
            x0 = 14 + k * PW
            st = stats[k]
            dr.text((x0, 8), f"{name}   線 {panels[k][1]} 本", font=F, fill=col)
            dr.text((x0, 48), f"{os.path.basename(bag)}  t={head['t']:.1f}s  ({panels[k][2]:.0f} ms)", font=Fs, fill=(220, 220, 220))
            f_ = max(1, st["frames"])
            dr.text((x0, 90 + PH + 10),
                    f"ここまで {st['frames']} コマ: 1 コマ平均 {st['lines'] / f_:.2f} 本 / 線 0 本のコマ {100 * st['zero'] / f_:.1f}%",
                    font=Fs, fill=col)
            dr.text((x0, 90 + PH + 40), f"白線と判定した画素 1 コマ平均 {st['px'] / f_:.0f}", font=Fs, fill=(200, 200, 200))
        dr.text((14, 90 + PH + 76), "緑 = YOLOP が白線と判定した画素  黄 = 取り出した白線  灰の線 = これより上は推論に使わない   "
                "space 一時停止  → 1 コマ  f 早送り  r 集計リセット  q 終了", font=Fs, fill=(170, 170, 170))
        if args.snapshot:
            cv2.imwrite(args.snapshot, cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
            break
        cv2.imshow(WIN, cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
        quit_ = False
        while True:
            key = cv2.waitKeyEx(0 if paused else 1)
            if key in (ord("q"), 27):
                quit_ = True
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("f"):
                fast = not fast
            elif key == ord("r"):
                stats = [dict(frames=0, lines=0, zero=0, px=0) for _ in SIDES]
            if not paused or key in (ord("d"), 63235, 2555904):
                break
        if quit_:
            break
    reader.terminate()
    cv2.destroyAllWindows()
    for (name, _, _), st in zip(SIDES, stats):
        f_ = max(1, st["frames"])
        print(f"{name}: {st['frames']} コマ, 1 コマ平均 {st['lines'] / f_:.2f} 本, 線 0 本のコマ {100 * st['zero'] / f_:.1f}%")


if __name__ == "__main__":
    main()
