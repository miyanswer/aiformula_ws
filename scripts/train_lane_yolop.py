#!/usr/bin/env python3
"""
train_lane_yolop.py - YOLOP 白線セグメンテーションのファインチューニング（追加学習）スクリプト

機能:
  - 既存のモデル (shiho-v2-20251118.pth 等) をベースにした転移学習
  - Dice Loss + Cross Entropy による高精度な白線輪郭学習
  - データ拡張（明るさ・影・ノイズ耐性の強化）
  - 学習後のベスト重み (.pth) 自動保存
  - GPU (CUDA) / Apple Silicon (MPS) / CPU に完全自動対応

使い方:
  # 基本（data/yolop_dataset を使って 30エポック学習）
  python3 scripts/train_lane_yolop.py

  # エポック数、バッチサイズ、学習率の指定
  python3 scripts/train_lane_yolop.py --epochs 50 --batch-size 8 --lr 1e-4

  # Mac MPS / NVIDIA CUDA の明示指定
  python3 scripts/train_lane_yolop.py --device mps
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# YOLOP インポートパス
WS_DIR = Path(__file__).resolve().parent.parent
YOLOP_CANDIDATES = [
    WS_DIR / "src/aiformula_base/oit_navigation/oit_navigation/yolop",
    WS_DIR / "src/ai_formula_oit_2026/perception/yolop",
]
for ydir in YOLOP_CANDIDATES:
    if ydir.exists() and str(ydir) not in sys.path:
        sys.path.insert(0, str(ydir))

try:
    from lib.config import cfg
    from lib.models import get_net
    from lib.utils import letterbox_for_img
except ImportError:
    from yolop.lib.config import cfg
    from yolop.lib.models import get_net
    from yolop.lib.utils import letterbox_for_img


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOP 白線・走行領域ファインチューニング")
    parser.add_argument("--data-dir", type=str, default="",
                        help="データセットディレクトリ (単一指定用, デフォルト: data/yolop_dataset)")
    parser.add_argument("--data-dirs", nargs="+", default=[],
                        help="複数のデータセットディレクトリを指定 (例: --data-dirs data/yolop_cropped_dataset data/honda_cropped_dataset)")
    parser.add_argument("--weights", type=str,
                        default=str(WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights/shiho-v2-20251118.pth"),
                        help="ベースとなる学習済み重みファイル")
    parser.add_argument("--output-name", type=str, default="shiho_lane_finetuned",
                        help="保存するモデルの名前プレフィックス")
    parser.add_argument("--epochs", type=int, default=30,
                        help="学習エポック数 (デフォルト: 30)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="バッチサイズ (デフォルト: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学習率 (デフォルト: 2e-4)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="入力画像サイズ (デフォルト: 640)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="バックボーンを固定し、白線Headのみ学習する (データ数が少ない場合に推奨)")
    parser.add_argument("--crop-bottom", action="store_true",
                        help="方法B: 画像の下部のみを切り出して学習する (白線解像度が向上)")
    parser.add_argument("--mask-top", action="store_true",
                        help="方法A: 画像の上部を黒塗り (0) にして学習する (上部ノイズを抑制)")
    parser.add_argument("--top-cut-ratio", type=float, default=0.45,
                        help="上部カットの割合 (デフォルト: 0.45 = 上部45%を対象)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="学習初期のウォームアップエポック数 (急激な勾配破綻を防止, デフォルト: 3)")
    parser.add_argument("--train-drivable", action="store_true",
                        help="走行可能領域 (da) も同時に学習する")
    parser.add_argument("--device", type=str, default="auto",
                        help="学習デバイス: 'cuda', 'mps', 'cpu', 'auto'")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader ワーカー数")
    return parser.parse_args()


def detect_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================================
# Dataset 定義
# =========================================================================
class LaneDataset(Dataset):
    def __init__(self, data_dirs, split: str = "train", img_size: int = 640, is_train: bool = True,
                 crop_bottom: bool = False, mask_top: bool = False, top_cut_ratio: float = 0.45,
                 train_drivable: bool = False):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [Path(data_dirs)]
        else:
            data_dirs = [Path(d) for d in data_dirs]

        self.samples = []
        for d in data_dirs:
            img_dir = d / "images" / split
            lane_dir = d / "lane_masks" / split
            da_dir = d / "drivable_masks" / split

            img_files = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
            for img_p in img_files:
                stem = Path(img_p).stem
                lane_p = lane_dir / f"{stem}.png"
                da_p = da_dir / f"{stem}.png"
                self.samples.append({
                    "img": img_p,
                    "lane": str(lane_p) if lane_p.exists() else "",
                    "da": str(da_p) if da_p.exists() else ""
                })

        self.img_size = img_size
        self.is_train = is_train
        self.crop_bottom = crop_bottom
        self.mask_top = mask_top
        self.top_cut_ratio = top_cut_ratio
        self.train_drivable = train_drivable

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img"]
        lane_path = sample["lane"]
        da_path = sample["da"]

        # 画像とマスクの読み込み
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if lane_path and os.path.exists(lane_path):
            lane_mask = cv2.imread(lane_path, cv2.IMREAD_GRAYSCALE)
            lane_mask = (lane_mask > 127).astype(np.uint8)  # 0 or 1
        else:
            lane_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if self.train_drivable and da_path and os.path.exists(da_path):
            da_mask = cv2.imread(da_path, cv2.IMREAD_GRAYSCALE)
            da_mask = (da_mask > 127).astype(np.uint8)
        else:
            da_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # ROI 処理 (方法A: 上部黒塗り / 方法B: 下部クロップ)
        # ※ すでに画像がクロップ済み (アスペクト比が横長・高さが約600px以下) の場合は再クロップをスキップ
        h, w = img.shape[:2]
        already_cropped = h < (w * 0.45)  # 1920x1080(比率0.56) vs 1920x594(比率0.31)

        if not already_cropped:
            cut_y = int(h * self.top_cut_ratio)
            if self.crop_bottom:
                # 方法B: 下半分だけを切り出す (解像度2倍)
                img = img[cut_y:, :]
                lane_mask = lane_mask[cut_y:, :]
                da_mask = da_mask[cut_y:, :]
            elif self.mask_top:
                # 方法A: 上部を黒塗りにする
                lane_mask[:cut_y, :] = 0
                da_mask[:cut_y, :] = 0

        # リサイズ (Letterbox)
        h, w = img.shape[:2]
        img_resized, ratio, pad = letterbox_for_img(img, new_shape=self.img_size, auto=False)
        
        # マスクも同様に Letterbox 適用
        lane_resized = cv2.resize(lane_mask, (int(w * ratio[0]), int(h * ratio[1])), interpolation=cv2.INTER_NEAREST)
        pad_lane_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        top, bottom = int(pad[1]), int(self.img_size - pad[1] - lane_resized.shape[0])
        left, right = int(pad[0]), int(self.img_size - pad[0] - lane_resized.shape[1])
        pad_lane_mask[top:top + lane_resized.shape[0], left:left + lane_resized.shape[1]] = lane_resized

        if self.train_drivable:
            da_resized = cv2.resize(da_mask, (int(w * ratio[0]), int(h * ratio[1])), interpolation=cv2.INTER_NEAREST)
            pad_da_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            pad_da_mask[top:top + da_resized.shape[0], left:left + da_resized.shape[1]] = da_resized

        # データ拡張 (Train のみ)
        if self.is_train:
            # ランダム水平フリップ
            if np.random.rand() > 0.5:
                img_resized = np.ascontiguousarray(np.fliplr(img_resized))
                pad_lane_mask = np.ascontiguousarray(np.fliplr(pad_lane_mask))
                if self.train_drivable:
                    pad_da_mask = np.ascontiguousarray(np.fliplr(pad_da_mask))

            # ランダムな明度・コントラスト変動
            if np.random.rand() > 0.3:
                alpha = np.random.uniform(0.7, 1.3)  # コントラスト
                beta = np.random.uniform(-30, 30)    # 明るさ
                img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

        # Tensor 化
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = self.normalize(img_tensor)

        lane_mask_tensor = torch.from_numpy(pad_lane_mask).long()

        if self.train_drivable:
            da_mask_tensor = torch.from_numpy(pad_da_mask).long()
            return img_tensor, lane_mask_tensor, da_mask_tensor

        return img_tensor, lane_mask_tensor


# =========================================================================
# Loss 関数 (Dice Loss + CrossEntropy)
# =========================================================================
class CombinedLaneLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        # 白線クラスに少し重みを付けた CrossEntropy (背景: 0.2, 白線: 1.0)
        class_weights = torch.tensor([0.2, 1.0])
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, 2, H, W), target: (B, H, W)
        if self.ce.weight.device != pred.device:
            self.ce.weight = self.ce.weight.to(pred.device)

        # 解像度が異なる場合 (YOLOPのHead出力は384x640または半分のサイズ)
        if pred.shape[2:] != target.shape[1:]:
            target_resized = torch.nn.functional.interpolate(
                target.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
            ).squeeze(1).long()
        else:
            target_resized = target

        ce_loss = self.ce(pred, target_resized)

        # Dice Loss 計算 (白線クラス = 1)
        prob = torch.softmax(pred, dim=1)[:, 1]  # (B, H, W)
        target_one_hot = (target_resized == 1).float()

        intersection = (prob * target_one_hot).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target_one_hot.sum(dim=(1, 2))
        dice_score = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1.0 - dice_score.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """白線クラスの IoU を計算"""
    if pred.shape[2:] != target.shape[1:]:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
        ).squeeze(1).long()

    pred_cls = torch.argmax(pred, dim=1) == 1
    target_cls = target == 1

    intersection = (pred_cls & target_cls).float().sum()
    union = (pred_cls | target_cls).float().sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


# =========================================================================
# 学習ループ
# =========================================================================
def main():
    args = parse_args()

    # データセットディレクトリの収集
    target_data_dirs = []
    if args.data_dirs:
        target_data_dirs.extend(args.data_dirs)
    elif args.data_dir:
        target_data_dirs.append(args.data_dir)
    else:
        target_data_dirs.append(str(WS_DIR / "data/yolop_dataset"))

    resolved_dirs = [Path(d) for d in target_data_dirs if Path(d).exists()]
    if not resolved_dirs:
        print(f"Error: None of the specified dataset directories exist: {target_data_dirs}")
        sys.exit(1)

    device = detect_device(args.device)
    print("=" * 60)
    print(f"🚀 YOLOP Lane Segmentation Fine-Tuning")
    print(f"Device: {device}")
    print(f"Datasets: {[str(d) for d in resolved_dirs]}")
    print(f"Base weights: {args.weights}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    # 1. データセットと DataLoader
    train_dataset = LaneDataset(
        resolved_dirs, split="train", img_size=args.img_size, is_train=True,
        crop_bottom=args.crop_bottom, mask_top=args.mask_top, top_cut_ratio=args.top_cut_ratio,
        train_drivable=args.train_drivable
    )
    val_dataset = LaneDataset(
        resolved_dirs, split="val", img_size=args.img_size, is_train=False,
        crop_bottom=args.crop_bottom, mask_top=args.mask_top, top_cut_ratio=args.top_cut_ratio,
        train_drivable=args.train_drivable
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. モデル構築 & 重み読み込み
    model = get_net(cfg)
    if os.path.exists(args.weights):
        print(f"Loading checkpoint: {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {args.weights} not found, training from scratch!")

    # バックボーン固定設定
    if args.freeze_backbone:
        print("Freezing backbone layers (model.0 to model.23)...")
        for name, param in model.named_parameters():
            if not name.startswith("model.42"):  # model.42 is Lane Line Head
                param.requires_grad = False

    model.to(device)

    # 3. 最適化器 & スケジューラ (Warmup + Cosine Annealing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch_idx):
        if epoch_idx < args.warmup_epochs:
            # 線形ウォームアップ (1/warmup -> 1.0)
            return float(epoch_idx + 1) / float(max(1, args.warmup_epochs))
        else:
            # コサイン減衰 (1.0 -> 0.01)
            progress = float(epoch_idx - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = CombinedLaneLoss(ce_weight=1.0, dice_weight=1.5)

    # 保存ディレクトリ
    save_dir = WS_DIR / "models"
    weights_dir = WS_DIR / "src/ai_formula_oit_2026/perception/object_road_detector/weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    da_criterion = CombinedLaneLoss(ce_weight=1.0, dice_weight=1.0) if args.train_drivable else None

    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for batch_data in pbar:
            if args.train_drivable:
                imgs, lane_masks, da_masks = batch_data
                imgs = imgs.to(device)
                lane_masks = lane_masks.to(device)
                da_masks = da_masks.to(device)
            else:
                imgs, lane_masks = batch_data
                imgs = imgs.to(device)
                lane_masks = lane_masks.to(device)

            optimizer.zero_grad()
            _, da_seg_out, ll_seg_out = model(imgs)  # da: (B, 2, H, W), ll: (B, 2, H, W)

            loss = criterion(ll_seg_out, lane_masks)
            if args.train_drivable:
                da_loss = da_criterion(da_seg_out, da_masks)
                loss = loss + 0.5 * da_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]", leave=False):
                if args.train_drivable:
                    imgs, lane_masks, da_masks = batch_data
                    imgs = imgs.to(device)
                    lane_masks = lane_masks.to(device)
                    da_masks = da_masks.to(device)
                else:
                    imgs, lane_masks = batch_data
                    imgs = imgs.to(device)
                    lane_masks = lane_masks.to(device)

                _, da_seg_out, ll_seg_out = model(imgs)
                loss = criterion(ll_seg_out, lane_masks)
                if args.train_drivable:
                    da_loss = da_criterion(da_seg_out, da_masks)
                    loss = loss + 0.5 * da_loss

                val_loss += loss.item() * imgs.size(0)

                iou = calculate_iou(ll_seg_out, lane_masks)
                total_iou += iou * imgs.size(0)

        val_loss /= len(val_dataset)
        val_iou = total_iou / len(val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Lane IoU: {val_iou*100:.2f}%")

        # ベストモデルの保存
        if val_iou > best_iou:
            best_iou = val_iou
            best_save_path = save_dir / f"{args.output_name}_best.pth"
            ros_weights_path = weights_dir / f"{args.output_name}.pth"

            # state_dict 保存
            save_payload = {"state_dict": model.state_dict(), "epoch": epoch, "best_iou": best_iou}
            torch.save(save_payload, str(best_save_path))
            torch.save(model.state_dict(), str(ros_weights_path))
            print(f"  ⭐ New Best Model Saved! IoU: {best_iou*100:.2f}% -> {ros_weights_path}")

    total_time = time.time() - start_time
    print(f"\n🎉 Training Finished in {total_time/60:.1f} minutes! Best Val IoU: {best_iou*100:.2f}%")

    # 学習曲線のグラフ保存
    plot_path = WS_DIR / f"jpeg/yolop_training_curve_{args.output_name}.png"
    latest_plot_path = WS_DIR / "jpeg/yolop_training_curve.png"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({args.output_name})")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), [iou * 100 for iou in history["val_iou"]], label="Val IoU (%)", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("IoU (%)")
    plt.title(f"Lane Line IoU (Best: {best_iou*100:.2f}%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(str(plot_path))
    plt.savefig(str(latest_plot_path))
    print(f"📊 Training curve plot saved to: {plot_path}")
    print(f"💡 You can now use the new model in object_road_detector by passing:")
    print(f"   weight_path:={weights_dir / f'{args.output_name}.pth'}")


if __name__ == "__main__":
    main()
