#!/usr/bin/env python3
"""
auto_relabel.py - 人が手で直して保存した答えから学び, まだ直していない画像の「自動の答え」を作り直す (繰り返し)

1 回の更新:
  1. 手で直した答え (edited.txt) を集め, 5 枚に 1 枚を「確かめ用」に取り分ける (学習には使わない)
  2. 答え作り用のモデル (data/road_masks/label_model.pth. 車載の重みとは別) の走れる領域の出力部分を学習する.
     手で直した答えは EDITED_WEIGHT 回ずつ入れて重く見る. 白線と共通の前段は honda_shihou_finetuned_best.pth のまま
  3. 確かめ用の画像で「人の答え」との一致 (IoU) を, 前のモデルと新しいモデルで比べる. 新しい方が悪ければ使わない
  4. 手で直していない画像の答えを, 新しいモデルの出力 (+ 緑を除く・穴を埋める・下端につながる所だけ) で書き直す.
     エディタで今だれかが開いている画像 (viewing.json) と, 手で直した画像は書き換えない
--watch を付けると, 手で直した答えが --every 枚増えるたびに繰り返す.
最後に更新した時の枚数は relabel_state.json に残すので, 動かし直してもすぐには更新しない.
更新の始まりと終わりは relabel_status.json に書き (エディタの画面に出る), この Mac の通知にも出す.

使い方:
  python3 scripts/auto_relabel.py                 # 1 回だけ
  python3 scripts/auto_relabel.py --watch --every 100
"""

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
import make_road_masks as M  # noqa: E402
import train_road_head as T  # noqa: E402

MASKS = WS_DIR / "data/road_masks"
LABEL_MODEL = MASKS / "label_model.pth"
LOG = MASKS / "relabel_log.txt"
STATE = MASKS / "relabel_state.json"
STATUS = MASKS / "relabel_status.json"   # エディタが読んで, 作業中のみんなに知らせる
EDITED_WEIGHT = 8          # 手で直した答えを何回ずつ学習に入れるか
EPOCHS = 3


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notify(state, msg, **extra):
    """更新の始まり・終わりを relabel_status.json とこの Mac の通知に出す."""
    st = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    if state == "running":
        st = dict(started=time.strftime("%H:%M"))
    else:
        st["finished"] = time.strftime("%H:%M")
    st.update(state=state, msg=msg, **extra)
    STATUS.write_text(json.dumps(st, ensure_ascii=False))
    subprocess.run(["osascript", "-e", f"display notification {json.dumps(msg, ensure_ascii=False)} with title \"黒路面の自動の答え\""],
                   check=False, capture_output=True)


@contextmanager
def file_lock():
    """エディタ (road_mask_editor.py) と同時に一覧 (usable.txt など) を書かないための鍵."""
    with open(MASKS / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_set(p):
    return set(p.read_text().split()) if p.exists() else set()


def edited_all():
    out = []
    for ds in M.DATASETS:
        for split in ("train", "val"):
            out += [(ds, split, s) for s in sorted(read_set(MASKS / ds / split / "edited.txt"))]
    return out


def viewing():
    p = MASKS / "viewing.json"
    if not p.exists():
        return set()
    v = json.loads(p.read_text())
    return set(v["viewing"]) if time.time() - v["updated"] < 120 else set()


def image_of(ds, split, stem):
    return next((WS_DIR / "data" / ds / "images" / split).glob(stem + ".*"), None)


def predict(net, dev, img):
    """答え作り用のモデルの走れる領域を, 答えの形 (緑を除く・穴を埋める・下端につながる所) に整える."""
    da = M.infer_da(net, dev, img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] > 35) & (hsv[..., 0] < 95) & (hsv[..., 1] > 35)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    road = cv2.morphologyEx(((da > 0) & ~green).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab = cv2.connectedComponents(road)
    keep = set(np.unique(lab[-3:, :])) - {0}
    return np.isin(lab, list(keep)).astype(np.uint8)


def iou_on(net, dev, items):
    vals = []
    for ds, split, stem in items:
        ip = image_of(ds, split, stem)
        if ip is None:
            continue
        img = cv2.imread(str(ip))
        human = cv2.imread(str(MASKS / ds / split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        pred = predict(net, dev, img) > 0
        vals.append((pred & human).sum() / max((pred | human).sum(), 1))
    return float(np.mean(vals)) if vals else float("nan")


def cycle():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    edited = edited_all()
    hold = [e for e in edited if zlib.crc32(e[2].encode()) % 5 == 0]
    log(f"更新を始める: 手で直した答え {len(edited)} 枚 (うち確かめ用 {len(hold)} 枚)")
    notify("running", f"学習と書き直しを始めました (手で直した答え {len(edited)} 枚から学習. 15〜20 分)")
    init = LABEL_MODEL if LABEL_MODEL.exists() else WS_DIR / "models/honda_shihou_finetuned_road_best.pth"
    new_path = MASKS / "label_model_new_best.pth"
    args = SimpleNamespace(src=str(WS_DIR / "models/honda_shihou_finetuned_best.pth"), out=str(new_path), epochs=EPOCHS,
                           batch=8, lr=5e-4, init=str(init), edited_weight=EDITED_WEIGHT)
    T.train(args, exclude=set(hold), log=log)
    old_net, new_net = M.load_net(init, dev), M.load_net(new_path, dev)
    old_iou, new_iou = iou_on(old_net, dev, hold), iou_on(new_net, dev, hold)
    log(f"確かめ用 {len(hold)} 枚での人の答えとの一致 (IoU): 前のモデル {old_iou:.3f} -> 新しいモデル {new_iou:.3f}")
    adopted = not (hold and new_iou < old_iou - 0.005)
    if not adopted:
        log("新しいモデルの方が悪いので使わない (前のモデルで書き直す)")
        net = old_net
    else:
        shutil.copyfile(new_path, LABEL_MODEL)
        net = new_net
    # 手で直していない画像の答えを書き直す
    n_new = n_skip = 0
    for ds in M.DATASETS:
        for split in ("train", "val"):
            d = MASKS / ds / split
            files = sorted((WS_DIR / "data" / ds / "images" / split).glob("*"))
            results = {}
            for f in files:
                stem = f.stem
                if stem in read_set(d / "edited.txt") or f"{ds}/{split}/{stem}" in viewing():
                    n_skip += 1
                    continue
                img = cv2.imread(str(f))
                m = predict(net, dev, img)
                results[stem] = (m, M.usable(img, m))
            with file_lock():
                edited_now = read_set(d / "edited.txt")
                usable = read_set(d / "usable.txt")
                for stem, (m, ok) in results.items():
                    if stem in edited_now:          # 書き直している間に人が直した
                        continue
                    cv2.imwrite(str(d / f"{stem}.png"), m * 255)
                    (usable.add if ok else usable.discard)(stem)
                    n_new += 1
                (d / "usable.txt").write_text("\n".join(sorted(usable)))
    log(f"自動の答えを書き直した {n_new} 枚 (手で直した / 開いている画像 {n_skip} 枚はそのまま)")
    notify("done", f"学習と書き直しが終わりました: {n_new} 枚を書き直し (人の答えとの一致 {old_iou:.3f} → {new_iou:.3f}"
                   f"{'' if adopted else ', 新しいモデルは使わず'}). 次に開く画像から新しい答えになります",
           rewritten=n_new, old_iou=old_iou, new_iou=new_iou, adopted=adopted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="手で直した答えが増えるたびに繰り返す")
    ap.add_argument("--every", type=int, default=100, help="手で直した答えが何枚増えたら更新するか")
    args = ap.parse_args()
    if not args.watch:
        cycle()
        return
    last = json.loads(STATE.read_text())["edited"] if STATE.exists() else -10 ** 9
    log(f"見張りを始める: 手で直した答えが {args.every} 枚増えるたびに自動の答えを更新する")
    while True:
        n = len(edited_all())
        if n - last >= args.every:
            last = n
            STATE.write_text(json.dumps({"edited": n}))
            try:
                cycle()
            except Exception as e:                  # 1 回失敗しても見張りは続ける
                log(f"更新に失敗: {e!r}")
                notify("failed", f"学習と書き直しに失敗しました: {e!r}")
        time.sleep(60)


if __name__ == "__main__":
    main()
