#!/usr/bin/env python3
"""
pseudo_label.py - 学習済みYOLOモデルを活用した未アノテーション画像の高速自動ラベリング (Pseudo-Labeling)

概要:
  初期データ（例: 100枚）で学習したYOLOモデル (models/<TASK>.pt) を使い、
  残りの未アノテーション画像に対して一括推論を行い、AnyLabeling対応の .json
  および YOLO形式の .txt アノテーションファイルを自動生成します。

ワークフロー:
  1. 100枚程度を手動アノテーション (make label TASK=<TASK>)
  2. 初期モデルを学習 (make split -> make train TASK=<TASK>)
  3. 残りの画像に自動ラベリング (make pseudo-label TASK=<TASK>)
  4. AnyLabelingで確認・ズレや見落としを微修正 (make label TASK=<TASK>)
  5. 全データで本番モデルを学習 (make split -> make train TASK=<TASK>)

使い方:
  # 基本（未アノテーション画像に自動でラベル付与）
  make pseudo-label TASK=traffic_light_red

  # 信頼度閾値を調整（デフォルト: 0.30）
  python3 scripts/pseudo_label.py --task traffic_light_red --conf 0.25

  # 特定のモデル重みファイルを指定
  python3 scripts/pseudo_label.py --task traffic_light_red --model models/traffic_light_red.pt

  # 既存のアノテーションも全て上書き再推論する場合
  python3 scripts/pseudo_label.py --task traffic_light_red --overwrite
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import torch


def detect_best_device():
    """実行デバイスの自動検出 (Apple Silicon MPS / NVIDIA CUDA / CPU)"""
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def find_model_path(task: str, explicit_model: str) -> Optional[str]:
    """使用するYOLOモデルファイルの検索"""
    if explicit_model and os.path.isfile(explicit_model):
        return explicit_model

    candidates = []
    if task:
        candidates.extend([
            f"models/{task}.pt",
            f"runs/detect/runs/train/{task}/weights/best.pt",
            f"runs/train/{task}/weights/best.pt",
            f"runs/detect/runs/train/{task}/weights/last.pt",
            f"runs/train/{task}/weights/last.pt",
        ])

    # タスク指定がある場合、まずはタスク一致モデルを探す
    for c in candidates:
        if os.path.isfile(c):
            return c

    # タスク指定がない、または見つからない場合のみ models/ 配下の全 .pt ファイルを探す
    if not task and os.path.exists("models"):
        for f in sorted(os.listdir("models")):
            if f.endswith(".pt") and not f.startswith("."):
                return os.path.join("models", f)

    return None


def find_images_directory(task: str, explicit_source: str) -> Optional[str]:
    """対象画像フォルダの検索"""
    if explicit_source and os.path.isdir(explicit_source):
        return explicit_source

    candidates = []
    if task:
        candidates.extend([
            f"data/tasks/{task}/extracted_frames",
            f"data/tasks/{task}/raw_images",
            f"data/tasks/{task}/annotated",
            f"data/tasks/{task}",
        ])
    else:
        # data/tasks 配下を探索
        tasks_dir = "data/tasks"
        if os.path.exists(tasks_dir):
            for t_name in sorted(os.listdir(tasks_dir)):
                t_path = os.path.join(tasks_dir, t_name)
                if os.path.isdir(t_path) and not t_name.startswith("."):
                    candidates.append(f"data/tasks/{t_name}/extracted_frames")
                    candidates.append(f"data/tasks/{t_name}/raw_images")

        candidates.extend(["data/extracted_frames", "data/raw_images", "data/annotated"])

    for c in candidates:
        if os.path.isdir(c):
            # 画像が存在するか確認
            img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.PNG")
            for ext in img_exts:
                if glob.glob(os.path.join(c, ext)):
                    return c

    return candidates[0] if candidates else None


def parse_args():
    parser = argparse.ArgumentParser(description="学習済みYOLOモデルによる高速自動ラベリング (Pseudo-Labeling)")
    parser.add_argument("--task", type=str, default="",
                        help="タスク名 (例: traffic_light_red, traffic_light_green, jinmen_dog)")
    parser.add_argument("--model", type=str, default="",
                        help="使用するモデルのパス (省略時は models/<task>.pt から自動検出)")
    parser.add_argument("--source", type=str, default="",
                        help="画像フォルダのパス (省略時は data/tasks/<task>/extracted_frames)")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="検出信頼度閾値 (0.0〜1.0、デフォルト: 0.30 推奨: 0.25〜0.35)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU閾値 (デフォルト: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推論画像サイズ (デフォルト: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="実行デバイス (空欄で自動検出: mps / cuda / cpu)")
    parser.add_argument("--overwrite", action="store_true",
                        help="すでにアノテーションが存在する画像も上書き再生成する")
    parser.add_argument("--format", type=str, default="both", choices=["json", "txt", "both"],
                        help="出力フォーマット: json (AnyLabeling互換), txt (YOLO形式), both (両方、デフォルト)")
    return parser.parse_args()


def save_anylabeling_json(json_path: str, img_filename: str, img_w: int, img_h: int, shapes: List[Dict]):
    """AnyLabeling / Labelme 互換の JSON アノテーションファイルを保存"""
    data = {
        "version": "0.4.30",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yolo_txt(txt_path: str, boxes_yolo: List[Tuple[int, float, float, float, float]]):
    """YOLO形式 (.txt) のアノテーションファイルを保存"""
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, nw, nh in boxes_yolo:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")


def main():
    args = parse_args()

    # タスク名の推測
    task_name = args.task
    if not task_name and args.source:
        parts = Path(args.source).parts
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                task_name = parts[idx + 1]

    # モデルパスの検出
    model_path = find_model_path(task_name, args.model)
    if not model_path or not os.path.isfile(model_path):
        print(f"[エラー] 学習済みモデルが見つかりません。")
        print(f"  タスク: {task_name or '(未指定)'}")
        print(f"  探索先: models/{task_name}.pt または runs/train/{task_name}/weights/best.pt")
        print("\n💡 先に100枚程度を AnyLabeling (make label) でアノテーションし、")
        print("   make split && make train を実行して初期モデルを作成してください。")
        return 1

    # 画像フォルダの検出
    source_dir = find_images_directory(task_name, args.source)
    if not source_dir or not os.path.isdir(source_dir):
        print(f"[エラー] 画像フォルダが見つかりません: {source_dir}")
        return 1

    if not task_name:
        parts = Path(source_dir).parts
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                task_name = parts[idx + 1]
    if not task_name:
        task_name = "default"

    # Ultralytics のインポート
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[エラー] 'ultralytics' ライブラリが見つかりません。")
        print("  pip3 install -r requirements-yolo.txt を実行してください。")
        return 1

    device = args.device if args.device else detect_best_device()

    print("=" * 65)
    print("=== 高速疑似ラベリング (Pseudo-Labeling with YOLO) ===")
    print("=" * 65)
    print(f"🎯 タスク名       : {task_name}")
    print(f"🧠 使用モデル     : {model_path}")
    print(f"📁 画像フォルダ   : {source_dir}")
    print(f"🎯 信頼度閾値     : {args.conf:.2f} (見落とし防止のため低めに設定)")
    print(f"⚡ 実行デバイス   : {device}")
    print(f"📄 出力形式       : {args.format.upper()} (AnyLabeling JSON + YOLO TXT)")
    print(f"🔄 既存の上書き   : {'有効 (全て再生成)' if args.overwrite else '無効 (未アノテーションのみ)'}")
    print("=" * 65)

    # 対象画像一覧の収集
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")
    all_imgs = []
    for root, _, files in os.walk(source_dir):
        for f in sorted(files):
            if f.endswith(img_exts):
                all_imgs.append(os.path.join(root, f))

    if not all_imgs:
        print(f"[警告] 画像が見つかりませんでした: {source_dir}")
        return 1

    def has_valid_annotation(img_p: str) -> bool:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        json_p = os.path.join(os.path.dirname(img_p), f"{base_name}.json")
        txt_p = os.path.join(os.path.dirname(img_p), f"{base_name}.txt")

        # 1. JSONチェック (shapes が空でないか)
        if os.path.isfile(json_p):
            try:
                with open(json_p, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                if jdata.get("shapes") and len(jdata["shapes"]) > 0:
                    return True
            except Exception:
                pass

        # 2. TXTチェック (空ファイルでないか)
        if os.path.isfile(txt_p) and os.path.basename(txt_p) != "classes.txt":
            try:
                if os.path.getsize(txt_p) > 0:
                    with open(txt_p, "r", encoding="utf-8") as tf:
                        lines = [l.strip() for l in tf if l.strip()]
                    if lines:
                        return True
            except Exception:
                pass

        return False

    # 処理対象画像のフィルタリング
    target_imgs = []
    skipped_count = 0
    for img_p in all_imgs:
        if not args.overwrite and has_valid_annotation(img_p):
            skipped_count += 1
        else:
            target_imgs.append(img_p)

    print(f"📊 対象画像数     : 全 {len(all_imgs)} 枚中、{len(target_imgs)} 枚を処理 (既存スキップ: {skipped_count} 枚)")
    if not target_imgs:
        print("✨ すべての画像がすでにアノテーション済みです！")
        print("💡 上書きして再推論したい場合は --overwrite を付けて実行してください。")
        return 0

    # モデルのロード
    print(f"\n[1/2] モデル '{model_path}' をロード中...")
    model = YOLO(model_path)
    model_classes = model.names  # {0: 'red', 1: 'green', ...}

    # classes.txt があればクラス名リストを保存
    classes_list = [model_classes[i] for i in sorted(model_classes.keys())]
    classes_txt_path = os.path.join(source_dir, "classes.txt")
    if not os.path.exists(classes_txt_path):
        with open(classes_txt_path, "w", encoding="utf-8") as cf:
            for cname in classes_list:
                cf.write(f"{cname}\n")

    # 推論とアノテーション生成
    print(f"\n[2/2] 未アノテーション画像 ({len(target_imgs)} 枚) に一括推論中...")
    
    total_detected_boxes = 0
    annotated_img_count = 0
    empty_img_count = 0

    batch_size = 16
    for i in range(0, len(target_imgs), batch_size):
        batch_paths = target_imgs[i:i + batch_size]
        results = model.predict(
            source=batch_paths,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )

        for img_p, r in zip(batch_paths, results):
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            dir_name = os.path.dirname(img_p)
            img_filename = os.path.basename(img_p)

            # 画像サイズ
            img_h, img_w = r.orig_shape

            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            boxes_cls = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
            boxes_conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else []

            shapes_json = []
            boxes_yolo = []

            for box, cls_idx, conf_score in zip(boxes_xyxy, boxes_cls, boxes_conf):
                cls_id = int(cls_idx)
                label_name = model_classes.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

                # 1. AnyLabeling 互換 JSON 形状 (rectangle)
                shape_item = {
                    "label": label_name,
                    "text": "",
                    "points": [
                        [round(x1, 1), round(y1, 1)],
                        [round(x2, 1), round(y2, 1)]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "description": f"conf: {conf_score:.2f}"
                }
                shapes_json.append(shape_item)

                # 2. YOLO 形式 (normalized xywh)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                xc = (x1 + bw / 2.0) / img_w
                yc = (y1 + bh / 2.0) / img_h
                nw = bw / img_w
                nh = bh / img_h
                boxes_yolo.append((cls_id, xc, yc, nw, nh))

            # ファイル保存
            if args.format in ("json", "both"):
                json_path = os.path.join(dir_name, f"{base_name}.json")
                save_anylabeling_json(json_path, img_filename, img_w, img_h, shapes_json)

            if args.format in ("txt", "both"):
                txt_path = os.path.join(dir_name, f"{base_name}.txt")
                save_yolo_txt(txt_path, boxes_yolo)

            box_count = len(shapes_json)
            total_detected_boxes += box_count
            if box_count > 0:
                annotated_img_count += 1
            else:
                empty_img_count += 1

        processed = min(i + batch_size, len(target_imgs))
        sys.stdout.write(f"\r  ⚡ 進捗: {processed}/{len(target_imgs)} 枚完了 (検出物体数: {total_detected_boxes} 個)...")
        sys.stdout.flush()

    print("\n\n" + "=" * 65)
    print("🎉 疑似ラベリング (Pseudo-Labeling) 完了！")
    print("=" * 65)
    print(f"  ・処理画像枚数       : {len(target_imgs)} 枚")
    print(f"  ・物体検出された画像 : {annotated_img_count} 枚")
    print(f"  ・物体未検出の画像   : {empty_img_count} 枚 (背景のみ・誤検出なし)")
    print(f"  ・総検出バウンディングボックス : {total_detected_boxes} 個")
    print(f"  ・出力フォルダ       : {source_dir}")
    print("=" * 65)

    print("\n👉 次のステップ (Human-in-the-loop: 確認・微修正):")
    print(f"  1. AnyLabeling を開いて、自動生成されたラベルを確認・微修正します:")
    print(f"     make label TASK={task_name}")
    print(f"     (自動で四角形が配置されているので、ズレ調整やDeleteキーでの誤検知削除だけで済みます！)")
    print(f"\n  2. 修正が完了したら、全データでデータセット分割 & 再学習:")
    print(f"     make split TASK={task_name}")
    print(f"     make train TASK={task_name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
