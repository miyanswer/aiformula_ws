#!/usr/bin/env python3
"""
train_lane_ufld.py - UFLD (Ultra-Fast-Lane-Detection) 自車レーン検出のファインチューニングスクリプト

背景:
  公式の TuSimple 事前学習済み重み (models/pretrained/ufld_tusimple_r18.pth) は
  Shihou/Honda コースとは全く違う見た目のドメイン(米国高速道路)で学習されているため、
  実車の映像に対しては検出点が白線から外れることがある。
  UFLD 専用のアノテーション(row-anchor形式)は無いが、既存の YOLOP 用の白線マスク
  データセットからそのまま生成できる:
    - 各 row anchor の画像行をスキャンし、車両中心から見て最も近い左右2本を
      自車レーン(lane slot 1, 2)の教師ラベルとして採用する
      (ufld_lane_detector.py の _select_ego_pair と同じ考え方)。
    - 外側レーン(lane slot 0, 3)は今回の白線マスクでは区別できないため、
      常に「レーン無し」クラスとして学習する(推論時にも使わないため実害なし)。
  det_annotations (物体検出) は使用しない。

使い方:
  # 基本 (data/yolop_dataset + data/honda_data_set を使って 20エポック学習)
  python3 scripts/train_lane_ufld.py

  # データセットを明示指定
  python3 scripts/train_lane_ufld.py --data-dirs data/yolop_dataset data/honda_data_set

  # エポック数、バッチサイズ、学習率の指定
  python3 scripts/train_lane_ufld.py --epochs 30 --batch-size 8 --lr 1e-4

  # Mac MPS / NVIDIA CUDA の明示指定
  python3 scripts/train_lane_ufld.py --device mps
"""

import argparse
import glob
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

WS_DIR = Path(__file__).resolve().parent.parent
UFLD_PKG_DIR = WS_DIR / "src/aiformula_base/oit_navigation"
if str(UFLD_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(UFLD_PKG_DIR))

from oit_navigation.ufld.model import ParsingNet  # noqa: E402
from oit_navigation.ufld.decode import encode_lane_label  # noqa: E402
from oit_navigation.ufld.constant import (  # noqa: E402
    TUSIMPLE_ROW_ANCHOR,
    TUSIMPLE_GRIDING_NUM,
    TUSIMPLE_NUM_LANES,
    TUSIMPLE_INPUT_SIZE,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# lane slots, matching TuSimple / UFLD convention (4 lanes)
_FAR_LEFT_SLOT = 0   # 外側左レーン (LL)
_EGO_LEFT_SLOT = 1   # 自車左レーン (L)
_EGO_RIGHT_SLOT = 2  # 自車右レーン (R)
_FAR_RIGHT_SLOT = 3  # 外側右レーン (RR)


def parse_args():
    parser = argparse.ArgumentParser(description="UFLD 4レーン検出 (自車+外側) ファインチューニング")
    parser.add_argument("--data-dirs", nargs="+",
                        default=[str(WS_DIR / "data/yolop_dataset"), str(WS_DIR / "data/honda_data_set")],
                        help="データセットディレクトリ (デフォルト: data/yolop_dataset data/honda_data_set)")
    parser.add_argument("--weights", type=str,
                        default=str(WS_DIR / "models/pretrained/ufld_tusimple_r18.pth"),
                        help="ベースとなる学習済み重みファイル")
    parser.add_argument("--output-name", type=str, default="ufld_honda_4lanes_finetuned",
                        help="保存するモデルの名前プレフィックス")
    parser.add_argument("--epochs", type=int, default=20,
                        help="学習エポック数 (デフォルト: 20)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="バッチサイズ (デフォルト: 8)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学習率 (デフォルト: 1e-4)")
    parser.add_argument("--backbone-lr-ratio", type=float, default=0.1,
                        help="バックボーンの学習率比率 (デフォルト: 0.1 = 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW Weight Decay (デフォルト: 0.01)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="CrossEntropyLoss Label Smoothing (デフォルト: 0.1)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="分類Headの中間Dropout率 (デフォルト: 0.3)")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early Stopping patience (デフォルト: 4)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="バックボーンを固定し、分類Headのみ学習する (データ数が少ない場合に推奨)")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="学習初期のウォームアップエポック数 (デフォルト: 2)")
    parser.add_argument("--device", type=str, default="auto",
                        help="学習デバイス: 'cuda', 'mps', 'cpu', 'auto'")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader ワーカー数")
    return parser.parse_args()


def detect_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_mask_dir(data_dir: Path, split: str) -> Optional[Path]:
    """Supports both the yolop_dataset naming (lane_masks/) and the raw
    honda_data_set naming (ll_seg_annotations/); det_annotations is never touched."""
    for name in ("lane_masks", "ll_seg_annotations"):
        candidate = data_dir / name / split
        if candidate.exists():
            return candidate
    return None


def _row_runs(mask_row: np.ndarray) -> List[float]:
    """Returns the center x of each contiguous on-pixel run in a single mask row."""
    on = np.where(mask_row > 0)[0]
    if len(on) == 0:
        return []
    splits = np.where(np.diff(on) > 1)[0] + 1
    return [float(g.mean()) for g in np.split(on, splits)]


class UFLDRowAnchorDataset(Dataset):
    """
    Converts binary lane masks (YOLOP-style) into UFLD row-anchor classification
    labels on the fly: for each canonical row anchor, lane runs are mapped across
    all 4 lane slots:
      - slot 0: far left lane (outer left)
      - slot 1: ego left lane (nearest left of mid_x)
      - slot 2: ego right lane (nearest right of mid_x)
      - slot 3: far right lane (outer right)
    """

    def __init__(self, data_dirs: List[Path], split: str, is_train: bool):
        self.samples: List[Tuple[str, str]] = []
        for d in data_dirs:
            img_dir = d / "images" / split
            mask_dir = _find_mask_dir(d, split)
            if not img_dir.exists() or mask_dir is None:
                continue
            for img_p in sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))):
                stem = Path(img_p).stem
                mask_p = mask_dir / f"{stem}.png"
                if mask_p.exists():
                    self.samples.append((img_p, str(mask_p)))

        self.is_train = is_train
        self.row_anchor = TUSIMPLE_ROW_ANCHOR
        self.griding_num = TUSIMPLE_GRIDING_NUM
        self.num_lanes = TUSIMPLE_NUM_LANES
        self.net_w, self.net_h = TUSIMPLE_INPUT_SIZE
        self.normalize = transforms.Normalize(mean=_IMAGENET_MEAN.tolist(), std=_IMAGENET_STD.tolist())

    def __len__(self) -> int:
        return len(self.samples)

    def _build_labels(self, mask: np.ndarray) -> torch.Tensor:
        img_h, img_w = mask.shape[:2]
        mid_x = img_w / 2.0
        labels = np.full((len(self.row_anchor), self.num_lanes), self.griding_num, dtype=np.int64)

        for k, ra in enumerate(self.row_anchor):
            y = int(np.clip(round(ra / 288.0 * img_h), 0, img_h - 1))
            centers = _row_runs(mask[y, :])
            if not centers:
                continue

            left_candidates = sorted([c for c in centers if c < mid_x])
            right_candidates = sorted([c for c in centers if c >= mid_x])

            # 自車左 (slot 1) と 外側左 (slot 0)
            if len(left_candidates) >= 1:
                labels[k, _EGO_LEFT_SLOT] = encode_lane_label(left_candidates[-1], img_w, self.griding_num)
            if len(left_candidates) >= 2:
                labels[k, _FAR_LEFT_SLOT] = encode_lane_label(left_candidates[-2], img_w, self.griding_num)

            # 自車右 (slot 2) と 外側右 (slot 3)
            if len(right_candidates) >= 1:
                labels[k, _EGO_RIGHT_SLOT] = encode_lane_label(right_candidates[0], img_w, self.griding_num)
            if len(right_candidates) >= 2:
                labels[k, _FAR_RIGHT_SLOT] = encode_lane_label(right_candidates[1], img_w, self.griding_num)

        return torch.from_numpy(labels)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read sample: {img_path}")
        mask = (mask > 127).astype(np.uint8) * 255

        if self.is_train and np.random.rand() > 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        labels = self._build_labels(mask)

        if self.is_train:
            # 1. 明度・コントラストのジッター
            if np.random.rand() > 0.3:
                alpha = np.random.uniform(0.7, 1.3)
                beta = np.random.uniform(-30, 30)
                image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

            # 2. HSVカラージッター (色相・彩度の揺らぎ)
            if np.random.rand() > 0.5:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 3. 微小ぼかし (Gaussian Blur)
            if np.random.rand() > 0.7:
                k = np.random.choice([3, 5])
                image = cv2.GaussianBlur(image, (k, k), 0)

            # 4. ランダム消去 (Cutout / Erasing): 遮蔽耐性の向上
            if np.random.rand() > 0.6:
                h, w, _ = image.shape
                er_h = np.random.randint(int(h * 0.05), int(h * 0.15))
                er_w = np.random.randint(int(w * 0.05), int(w * 0.20))
                y0 = np.random.randint(0, max(1, h - er_h))
                x0 = np.random.randint(0, max(1, w - er_w))
                image[y0:y0+er_h, x0:x0+er_w] = np.random.randint(0, 100, (er_h, er_w, 3), dtype=np.uint8)

        resized = cv2.resize(image, (self.net_w, self.net_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        img_tensor = self.normalize(img_tensor)

        return img_tensor, labels


class ClsHeadWithDropout(nn.Module):
    """Wraps ParsingNet.cls to add Dropout while preserving exact state_dict keys
    ('cls.0.weight', 'cls.0.bias', 'cls.2.weight', 'cls.2.bias')."""

    def __init__(self, orig_cls: nn.Sequential, p: float = 0.3):
        super().__init__()
        self.add_module("0", orig_cls[0])
        self.add_module("1", orig_cls[1])
        self.dropout = nn.Dropout(p=p)
        self.add_module("2", orig_cls[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        x = self.dropout(x)
        x = self._modules["2"](x)
        return x


def lane_accuracy(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 1) -> float:
    """Fraction of all 4 lane row-anchor cells the model gets right,
    within `tolerance` griding bins for real detections, exact match for 'no lane'."""
    pred_cls = torch.argmax(pred, dim=1)  # (B, num_row_anchors, num_lanes)
    correct = (pred_cls == target) | ((pred_cls - target).abs() <= tolerance)
    return correct.float().mean().item()


def main():
    args = parse_args()
    data_dirs = [Path(d) for d in args.data_dirs if Path(d).exists()]
    if not data_dirs:
        print(f"Error: None of the specified dataset directories exist: {args.data_dirs}")
        sys.exit(1)

    device = detect_device(args.device)
    print("=" * 60)
    print("UFLD 4-Lane Fine-Tuning")
    print(f"Device: {device}")
    print(f"Datasets: {[str(d) for d in data_dirs]}")
    print(f"Base weights: {args.weights}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    train_dataset = UFLDRowAnchorDataset(data_dirs, split="train", is_train=True)
    val_dataset = UFLDRowAnchorDataset(data_dirs, split="val", is_train=False)
    if len(train_dataset) == 0:
        print("Error: no training samples found (check --data-dirs contents).")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = ParsingNet(cls_dim=(TUSIMPLE_GRIDING_NUM + 1, len(TUSIMPLE_ROW_ANCHOR), TUSIMPLE_NUM_LANES))
    if Path(args.weights).exists():
        print(f"Loading checkpoint: {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"Warning: {args.weights} not found, training from scratch!")

    if args.dropout > 0:
        print(f"Adding Dropout (p={args.dropout}) to classification head...")
        model.cls = ClsHeadWithDropout(model.cls, p=args.dropout)

    if args.freeze_backbone:
        print("Freezing backbone layers (model.*), training pool/cls heads only...")
        for name, param in model.named_parameters():
            if name.startswith("model."):
                param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("model.")]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("model.")]
        backbone_lr = args.lr * args.backbone_lr_ratio
        print(f"Differential LR: Backbone LR = {backbone_lr:.2e}, Head LR = {args.lr:.2e}, Weight Decay = {args.weight_decay}")
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=args.weight_decay)

    model.to(device)

    def lr_lambda(epoch_idx):
        if epoch_idx < args.warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch_idx - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    save_dir = WS_DIR / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("\nStarting Training with Regularization & Early Stopping...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)  # (B, griding_num+1, num_row_anchors, num_lanes)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]", leave=False):
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                total_acc += lane_accuracy(out, labels) * imgs.size(0)

        val_loss /= max(1, len(val_dataset))
        val_acc = total_acc / max(1, len(val_dataset))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Lane Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
            no_improve_epochs = 0
            best_save_path = save_dir / f"{args.output_name}_best.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_loss": best_val_loss, "best_acc": best_acc}, str(best_save_path))
            print(f"  ★ New best model saved! Val Loss: {best_val_loss:.4f} (Acc: {best_acc*100:.2f}%) -> {best_save_path}")
        else:
            no_improve_epochs += 1
            print(f"  No improvement in Val Loss for {no_improve_epochs}/{args.patience} epochs (Best: {best_val_loss:.4f} at epoch {best_epoch:02d}).")
            if no_improve_epochs >= args.patience:
                print(f"\n[Early Stopping] Triggered at epoch {epoch:02d}. Stopping training to prevent overfitting.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.1f} minutes! Best Val Loss: {best_val_loss:.4f} (Acc: {best_acc*100:.2f}% at epoch {best_epoch:02d})")

    epochs_run = range(1, len(history["train_loss"]) + 1)
    plot_path = WS_DIR / f"jpeg/ufld_training_curve_{args.output_name}.png"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_run, history["train_loss"], label="Train Loss")
    plt.plot(epochs_run, history["val_loss"], label="Val Loss")
    plt.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.6, label=f"Best Model (Epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({args.output_name})")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_run, [a * 100 for a in history["val_acc"]], label="Val Lane Acc (%)", color="green")
    plt.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.6, label=f"Best Model (Epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Lane Accuracy (Best Val Loss Acc: {best_acc*100:.2f}%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(str(plot_path))
    print(f"Training curve plot saved to: {plot_path}")
    print("To use the fine-tuned model, pass to ufld_lane_detector:")
    print(f"   weight_path:={save_dir / f'{args.output_name}_best.pth'}")


if __name__ == "__main__":
    main()
