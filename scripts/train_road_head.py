#!/usr/bin/env python3
"""
train_road_head.py - 今の白線用の YOLOP (honda_shihou_finetuned_best.pth) の複製に「黒路面 (走れる路面)」の検出を足す

  - 元の重みは変えない. 複製を作り, その「走れる領域」の出力部分 (層 25〜33) だけを学習する.
    共通の前段 (層 0〜24) と白線の出力部分 (層 34〜42) は固定し, BatchNorm の統計も固定するので, 白線の出力は変わらない
  - 画像: honda_cropped_dataset / yolop_cropped_dataset (下 55% に切り出し済み 1920x594)
    答え: data/road_masks (make_road_masks.py で作り, road_mask_editor.py で手で直したもの). usable.txt の画像だけ使う
  - 前処理は車載と同じ letterbox (縦横比を保って 640 に入れ, 余白は 114). 余白は学習しない
  - 最後に検証用の画像で, 白線の出力が元の重みと同じかを確かめる

使い方:
  python3 scripts/train_road_head.py [--epochs 15] [--out models/honda_shihou_finetuned_road_best.pth]
"""

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
sys.path.insert(0, str(WS_DIR / "src/aiformula_base/oit_navigation/oit_navigation/yolop"))
import verify_lane_rosbag as V  # noqa: E402
from lib.config import cfg  # noqa: E402
from lib.models import get_net  # noqa: E402

DATASETS = ["honda_cropped_dataset", "yolop_cropped_dataset"]
HEAD = range(25, 34)       # 走れる領域の出力部分 (YOLOP の seg_out_idx[0] = 33)
IGNORE = 255


def read_set(p):
    return set(p.read_text().split()) if p.exists() else set()


def samples(split, edited_weight=1, exclude=()):
    """学習に使う (usable.txt の) 画像と答え. 手で直した答え (edited.txt) は edited_weight 回入れる. exclude の (ds, split, stem) は除く."""
    out = []
    for ds in DATASETS:
        d = WS_DIR / "data/road_masks" / ds / split
        edited = read_set(d / "edited.txt")
        for stem in sorted(read_set(d / "usable.txt")):
            if (ds, split, stem) in exclude:
                continue
            img = next((p for p in (WS_DIR / "data" / ds / "images" / split).glob(stem + ".*")), None)
            if img is not None:
                out += [(img, d / f"{stem}.png")] * (edited_weight if stem in edited else 1)
    return out


def letterbox(img, mask):
    ph, pw = img.shape[:2]
    r = min(V.INPUT_SIZE / ph, V.INPUT_SIZE / pw)
    nh, nw = int(round(ph * r)), int(round(pw * r))
    top, left = (V.INPUT_SIZE - nh) // 2, (V.INPUT_SIZE - nw) // 2
    canvas = np.full((V.INPUT_SIZE, V.INPUT_SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    m = np.full((V.INPUT_SIZE, V.INPUT_SIZE), IGNORE, np.uint8)
    if mask is not None:
        m[top:top + nh, left:left + nw] = (cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return canvas, m


class RoadSet(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.mean, self.std = np.asarray(V.NORM_MEAN, np.float32), np.asarray(V.NORM_STD, np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.imread(str(ip))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if self.train:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            a, b = np.random.uniform(0.7, 1.3), np.random.uniform(-25, 25)        # 明るさ・コントラスト (逆光や影)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        canvas, m = letterbox(img, mask)
        x = ((canvas.astype(np.float32) / 255 - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(m.astype(np.int64))


def loss_fn(logits, target):
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE)
    valid = target != IGNORE
    p = torch.softmax(logits, 1)[:, 1]
    t = (target == 1).float()
    inter = (p * t * valid).sum()
    dice = 1 - (2 * inter + 1) / ((p * valid).sum() + (t * valid).sum() + 1)
    return ce + dice


def iou(logits, target):
    # 数えるのは CPU で (MPS では学習中に IoU が 1 を超える数え間違いが出たことがある)
    pred, target = logits.argmax(1).cpu(), target.cpu()
    valid = target != IGNORE
    inter = ((pred == 1) & (target == 1) & valid).sum().item()
    union = (((pred == 1) | (target == 1)) & valid).sum().item()
    return inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WS_DIR / "models/honda_shihou_finetuned_best.pth"))
    ap.add_argument("--out", default=str(WS_DIR / "models/honda_shihou_finetuned_road_best.pth"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default="", help="走れる領域の出力部分をこの重みから始める (省略すると --src のまま)")
    ap.add_argument("--edited-weight", type=int, default=1, help="手で直した答えを何回入れるか (多いほど人の答えを重く見る)")
    args = ap.parse_args()
    train(args)


def train(args, exclude=(), log=print):
    """--src の複製の走れる領域だけを学習して --out に保存する. exclude の画像は学習にも検証にも使わない. 一番よい検証の IoU を返す."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 元の重みの複製 (学習前の状態) を作っておく
    copy_path = Path(args.out).with_name(Path(args.out).stem.replace("_best", "") + "_init.pth")
    shutil.copyfile(args.src, copy_path)
    log(f"複製: {args.src} -> {copy_path}")
    net = get_net(cfg)
    ck = torch.load(copy_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("state_dict", ck))
    if getattr(args, "init", ""):
        # 走れる領域の出力部分だけ, 前に学習した重みから始める (白線と共通の前段は --src のまま)
        init = torch.load(args.init, map_location="cpu", weights_only=False)["state_dict"]
        sd = net.state_dict()
        for k in sd:
            if int(k.split(".")[1]) in HEAD and k in init:
                sd[k] = init[k]
        net.load_state_dict(sd)
        log(f"走れる領域の出力部分は {args.init} から始める")
    ref = copy.deepcopy(net).to(dev).eval()                 # 白線の出力が変わらないか比べる元
    net = net.to(dev)
    head_params = []
    for name, p in net.named_parameters():
        li = int(name.split(".")[1])
        p.requires_grad = li in HEAD
        if p.requires_grad:
            head_params.append(p)
    log(f"学習する重み {sum(p.numel() for p in head_params)} / 全部 {sum(p.numel() for p in net.parameters())}")

    def set_mode():
        net.eval()                                          # 固定する所は BatchNorm の統計も変えない
        for i in HEAD:
            net.model[i].train()

    ew = getattr(args, "edited_weight", 1)
    tr, va = samples("train", ew, exclude), samples("val", 1, exclude)
    log(f"学習 {len(tr)} 枚 (手で直した答えは {ew} 回ずつ) / 検証 {len(va)} 枚")
    dl_tr = torch.utils.data.DataLoader(RoadSet(tr, True), batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(RoadSet(va, False), batch_size=args.batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = -1
    for ep in range(args.epochs):
        t0 = time.time()
        set_mode()
        tl = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            _, da, _ = net(x)
            loss = loss_fn(da, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(x)
        sched.step()
        net.eval()
        vi, n = 0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                _, da, _ = net(x)
                vi += iou(da, y) * len(x)
                n += len(x)
        vi /= n
        log(f"epoch {ep + 1:2d}/{args.epochs}  学習の損失 {tl / len(tr):.3f}  検証の IoU {vi:.3f}  ({time.time() - t0:.0f} s)")
        if vi > best:
            best = vi
            torch.save({"state_dict": net.state_dict(), "epoch": ep + 1, "road_val_iou": vi,
                        "note": "honda_shihou_finetuned_best.pth の複製に, 走れる領域 (層 25-33) だけを学習して足したもの"}, args.out)
    log(f"一番よかった検証の IoU {best:.3f} -> {args.out}")

    # 白線の出力が元と同じか
    best_net = get_net(cfg)
    best_net.load_state_dict(torch.load(args.out, map_location="cpu", weights_only=False)["state_dict"])
    best_net = best_net.to(dev).eval()
    diff = 0.0
    with torch.no_grad():
        for k, (x, _) in enumerate(dl_va):
            if k >= 3:
                break
            x = x.to(dev)
            diff = max(diff, (best_net(x)[2] - ref(x)[2]).abs().max().item())
    log(f"白線の出力の差 (元の重みとの最大の差): {diff:.2e}  ({'同じ' if diff < 1e-4 else '違う'})")
    return best


if __name__ == "__main__":
    main()
