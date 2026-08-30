#!/usr/bin/env python3
"""
アノテーション済み画像とYOLOラベル(.txt)またはAnyLabelingラベル(.json)を
train / val に自動分割し、YOLO学習用のディレクトリ構造および data.yaml を自動生成するスクリプト。

対応する入力形式 (タスク別完全対応):
  パターンA: data/tasks/<task>/extracted_frames 内に AnyLabeling の .json または .txt がある場合
  パターンB: data/tasks/<task>/annotated 内に .jpg と .txt が混在している場合
  パターンC: 従来の data/annotated や data/extracted_frames
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOデータセットの分割 & data.yaml 生成ツール")
    parser.add_argument("--task", type=str, default="",
                        help="タスク名 (例: traffic_light, t_junction, crosswalk, jinmen_dog)")
    parser.add_argument("--source-dir", type=str, default="",
                        help="アノテーション済みデータがあるフォルダ (省略時は data/tasks/<task>/extracted_frames 等から自動検出)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="分割後のYOLOデータセット出力先 (省略時は data/tasks/<task>/dataset)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="訓練データの割合 (例: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="検証データの割合 (例: 0.2)")
    parser.add_argument("--classes", type=str, default="",
                        help="カンマ区切りのクラス名一覧 (例: 'cone_blue,cone_yellow' や '人面犬')。指定がない場合はラベルから自動検出")
    parser.add_argument("--seed", type=int, default=42,
                        help="ランダムシード")
    return parser.parse_args()


def convert_json_shape_to_yolo(shape, img_w, img_h):
    """AnyLabeling / Labelme JSONのshapeをYOLO正規化座標 (x_center, y_center, w, h) に変換"""
    points = shape.get("points", [])
    if not points or len(points) < 2:
        return None

    pts_x = [p[0] for p in points]
    pts_y = [p[1] for p in points]

    x1, x2 = min(pts_x), max(pts_x)
    y1, y2 = min(pts_y), max(pts_y)

    # クリップ
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    x_center = (x1 + w / 2.0) / img_w
    y_center = (y1 + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    return (x_center, y_center, norm_w, norm_h)


def find_dataset_sources(source_dir_arg: str, task: str):
    """ソースディレクトリとアノテーションファイルの検索および自動検出"""
    if source_dir_arg:
        return source_dir_arg

    candidates = []
    if task:
        candidates.extend([
            f"data/tasks/{task}/extracted_frames",
            f"data/tasks/{task}/raw_images",
            f"data/tasks/{task}/annotated",
            f"data/tasks/{task}",
        ])
    else:
        # data/tasks 配下のフォルダを探索
        tasks_dir = "data/tasks"
        if os.path.exists(tasks_dir):
            for t_name in sorted(os.listdir(tasks_dir)):
                t_path = os.path.join(tasks_dir, t_name)
                if os.path.isdir(t_path) and not t_name.startswith("."):
                    candidates.append(f"data/tasks/{t_name}/extracted_frames")
                    candidates.append(f"data/tasks/{t_name}/raw_images")
                    candidates.append(f"data/tasks/{t_name}/annotated")

        candidates.extend(["data/extracted_frames", "data/raw_images", "data/annotated"])

    for c_dir in candidates:
        if not os.path.exists(c_dir):
            continue

        # 画像ファイルの存在確認
        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
        found_imgs = []
        for ext in img_exts:
            found_imgs.extend(glob.glob(os.path.join(c_dir, "**", ext), recursive=True))

        if not found_imgs:
            continue

        # ラベルファイル (.txt または .json) の存在確認
        found_txts = glob.glob(os.path.join(c_dir, "**", "*.txt"), recursive=True)
        found_txts = [t for t in found_txts if not os.path.basename(t) == "classes.txt"]
        found_jsons = glob.glob(os.path.join(c_dir, "**", "*.json"), recursive=True)

        if found_txts or found_jsons:
            return c_dir

    return candidates[0] if candidates else "data/extracted_frames"


def process_dataset(source_dir: str, explicit_classes: list, task_name: str = ""):
    """画像とアノテーション（.txt または .json）を走査し、YOLO形式ペアとクラスリストを構築"""
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")
    
    # classes.txt があれば読み込み
    classes_file = os.path.join(source_dir, "classes.txt")
    detected_classes = list(explicit_classes) if explicit_classes else []
    if not detected_classes and os.path.isfile(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            detected_classes = [line.strip() for line in f if line.strip()]

    # 画像ファイルの探索
    all_img_paths = []
    for root, _, files in os.walk(source_dir):
        for f in sorted(files):
            if f.endswith(img_exts):
                all_img_paths.append(os.path.join(root, f))

    valid_pairs = []
    unlabeled = []

    # 各画像に対してラベルを探す
    for img_path in all_img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.dirname(img_path)

        # 1. 対応する .txt を探す
        txt_candidates = [
            os.path.join(img_dir, f"{base_name}.txt"),
            os.path.join(source_dir, "labels", f"{base_name}.txt"),
            os.path.join(source_dir, f"{base_name}.txt"),
        ]
        found_txt = None
        for cand in txt_candidates:
            if os.path.isfile(cand) and os.path.basename(cand) != "classes.txt":
                found_txt = cand
                break

        if found_txt:
            valid_pairs.append((img_path, "txt", found_txt))
            continue

        # 2. 対応する .json (AnyLabeling) を探す
        json_candidates = [
            os.path.join(img_dir, f"{base_name}.json"),
            os.path.join(source_dir, f"{base_name}.json"),
        ]
        found_json = None
        for cand in json_candidates:
            if os.path.isfile(cand):
                found_json = cand
                break

        if found_json:
            try:
                with open(found_json, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                shapes = data.get("shapes", [])
                img_w = data.get("imageWidth", 1920)
                img_h = data.get("imageHeight", 1080)

                # クラス名の自動収集
                for s in shapes:
                    lbl = s.get("label", "").strip()
                    if lbl and lbl not in detected_classes:
                        detected_classes.append(lbl)

                if shapes:
                    valid_pairs.append((img_path, "json", (found_json, shapes, img_w, img_h)))
                else:
                    unlabeled.append(img_path)
            except Exception as e:
                print(f"⚠️ JSON読み込みエラー: {found_json} ({e})")
                unlabeled.append(img_path)
        else:
            unlabeled.append(img_path)

    # 実際にJSONやTXTに存在したラベルのみを抽出
    actual_labels = []
    for pair in valid_pairs:
        fmt = pair[1]
        if fmt == "json":
            _, shapes, _, _ = pair[2]
            for s in shapes:
                lbl = s.get("label", "").strip()
                if lbl and lbl not in actual_labels:
                    actual_labels.append(lbl)

    # 明示的なクラス指定(--classes)があればそれを優先、なければ実際に使われたラベル、なければclasses.txt、最後にタスク名のデフォルト
    if explicit_classes:
        final_classes = explicit_classes
    elif actual_labels:
        final_classes = actual_labels
    elif detected_classes:
        final_classes = detected_classes
    elif task_name == "traffic_light_red":
        final_classes = ["traffic_light_red"]
    elif task_name == "traffic_light_green":
        final_classes = ["traffic_light_green"]
    elif task_name == "traffic_light":
        final_classes = ["red", "green"]
    else:
        final_classes = ["object"]

    return valid_pairs, unlabeled, final_classes


def main():
    args = parse_args()
    random.seed(args.seed)

    # 入出力パスの解決
    source_dir = find_dataset_sources(args.source_dir, args.task)
    if not os.path.exists(source_dir):
        print(f"[エラー] ソースディレクトリが見つかりません: {source_dir}")
        print("💡 AnyLabeling (make label) でアノテーションを行うか、画像を配置してください。")
        return 1

    # タスク名の推測（source_dir から）
    task_name = args.task
    if not task_name:
        parts = Path(source_dir).parts
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                task_name = parts[idx + 1]
    if not task_name:
        task_name = "default"

    # 出力先ディレクトリ
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"data/tasks/{task_name}/dataset"

    explicit_classes = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else []
    pairs, unlabeled, classes = process_dataset(source_dir, explicit_classes, task_name)

    print("=" * 60)
    print("=== YOLO データセット自動分割 & 変換ツール ===")
    print("=" * 60)
    print(f"🎯 タスク名           : {task_name}")
    print(f"📁 入力元ディレクトリ : {source_dir}")
    print(f"💾 出力先ディレクトリ : {output_dir}")
    print(f"🏷️ 検出クラス一覧 ({len(classes)} classes): {classes}")
    print(f"⚖️ 分割比率           : Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}")
    print(f"📦 アノテーション済み : {len(pairs)} 枚")
    if unlabeled:
        print(f"⚠️ 未アノテーション   : {len(unlabeled)} 枚 (スキップ)")
    print("=" * 60)

    if not pairs:
        print(f"[エラー] 有効なアノテーション（.txt または .json）が見つかりませんでした。")
        print("💡 AnyLabeling (make label) で保存されているか確認してください。")
        return 1

    # シャッフルして分割
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = max(1, int(n_total * (args.train_ratio / (args.train_ratio + args.val_ratio))))
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    if not val_pairs and len(train_pairs) > 1:
        val_pairs = [train_pairs.pop()]

    print(f"📊 分割結果: Train = {len(train_pairs)} 枚, Val = {len(val_pairs)} 枚")

    # 出力先ディレクトリの初期化（古いキャッシュやゴミファイルを完全クリア）
    for split in ["train", "val"]:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    def export_pairs(split_pairs, split_name):
        for item in split_pairs:
            img_src = item[0]
            fmt_type = item[1]
            base_name = os.path.splitext(os.path.basename(img_src))[0]

            img_dst = os.path.join(output_dir, split_name, "images", os.path.basename(img_src))
            txt_dst = os.path.join(output_dir, split_name, "labels", f"{base_name}.txt")

            shutil.copy2(img_src, img_dst)

            if fmt_type == "txt":
                txt_src = item[2]
                shutil.copy2(txt_src, txt_dst)
            elif fmt_type == "json":
                json_path, shapes, img_w, img_h = item[2]
                with open(txt_dst, "w", encoding="utf-8") as out_f:
                    for s in shapes:
                        lbl = s.get("label", "").strip()
                        class_id = classes.index(lbl) if lbl in classes else 0
                        yolo_box = convert_json_shape_to_yolo(s, img_w, img_h)
                        if yolo_box:
                            out_f.write(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

    export_pairs(train_pairs, "train")
    export_pairs(val_pairs, "val")

    # data.yaml の生成 (絶対パス)
    abs_output_dir = os.path.abspath(output_dir)
    data_yaml_content = {
        "path": abs_output_dir,
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)},
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n🎉 [成功] データセットの自動変換＆分割が完了しました！")
    print(f"  ・data.yaml: {yaml_path}")
    print(f"  ・Train画像: {len(train_pairs)} 枚, Val画像: {len(val_pairs)} 枚")
    print(f"\n👉 次のステップ:")
    print(f"  make train TASK={task_name}")
    print(f"  # または python3 scripts/train_yolo.py --task {task_name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
