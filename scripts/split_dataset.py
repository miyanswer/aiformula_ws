#!/usr/bin/env python3
"""
アノテーション済み画像とYOLOラベル(.txt)を train / val (/ test) に自動分割し、
YOLO学習用のディレクトリ構造および data.yaml を自動生成するスクリプト。

期待する入力ディレクトリ形式 (いずれにも対応):
  パターンA (同一フォルダに混在):
    annotated_data/
      ├── frame_001.jpg
      ├── frame_001.txt
      └── ...
  パターンB (images / labels 分離):
    annotated_data/
      ├── images/ (frame_001.jpg, ...)
      └── labels/ (frame_001.txt, ...)

出力ディレクトリ形式:
  dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
"""

import argparse
import glob
import os
import random
import shutil
import sys
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOデータセットの分割 & data.yaml 生成ツール")
    parser.add_argument("--source-dir", type=str, default="data/annotated", help="アノテーション済みデータがあるフォルダ")
    parser.add_argument("--output-dir", type=str, default="data/dataset", help="分割後のYOLOデータセット出力先")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="訓練データの割合 (例: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="検証データの割合 (例: 0.2)")
    parser.add_argument("--classes", type=str, default="cone_blue,cone_yellow,cone_orange", help="カンマ区切りのクラス名一覧 (例: 'cone_blue,cone_yellow,cone_orange') または classes.txt のパス")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")
    return parser.parse_args()


def load_classes(classes_arg, source_dir):
    # classes.txt が source_dir にあればそれを優先
    classes_file = os.path.join(source_dir, "classes.txt")
    if os.path.isfile(classes_arg):
        with open(classes_arg, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    elif os.path.isfile(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return [c.strip() for c in classes_arg.split(",") if c.strip()]


def find_image_label_pairs(source_dir):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
    pairs = []
    unlabeled_images = []

    # パターンB: images / labels フォルダがあるか
    images_subdir = os.path.join(source_dir, "images")
    labels_subdir = os.path.join(source_dir, "labels")

    if os.path.isdir(images_subdir):
        search_dirs = [images_subdir]
    else:
        search_dirs = [source_dir]

    for s_dir in search_dirs:
        for root, _, files in os.walk(s_dir):
            for file in sorted(files):
                if file.endswith(image_extensions):
                    img_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]

                    # 対応する txt ファイルを探す
                    txt_candidates = [
                        os.path.join(labels_subdir, f"{base_name}.txt"),
                        os.path.join(root, f"{base_name}.txt"),
                        os.path.join(source_dir, f"{base_name}.txt"),
                    ]
                    found_txt = None
                    for cand in txt_candidates:
                        if os.path.isfile(cand):
                            found_txt = cand
                            break

                    if found_txt:
                        pairs.append((img_path, found_txt))
                    else:
                        unlabeled_images.append(img_path)

    return pairs, unlabeled_images


def main():
    args = parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.source_dir):
        print(f"[エラー] ソースディレクトリが存在しません: {args.source_dir}")
        print("アノテーション済みの画像と .txt ファイルをこのフォルダに配置してください。")
        return 1

    classes = load_classes(args.classes, args.source_dir)
    print(f"=== YOLO データセット分割ツール ===")
    print(f"入力元: {args.source_dir}")
    print(f"出力先: {args.output_dir}")
    print(f"検出クラス一覧 ({len(classes)} classes): {classes}")
    print(f"分割比率: Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}")
    print("==================================")

    pairs, unlabeled = find_image_label_pairs(args.source_dir)
    print(f"アノテーション済みペア数: {len(pairs)}")
    if unlabeled:
        print(f"[注意] ラベル(.txt)が見つからなかった画像: {len(unlabeled)}枚 (スキップされます)")

    if not pairs:
        print(f"[エラー] 有効な画像とラベルのペアが見つかりませんでした。")
        return 1

    # シャッフルして分割
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * (args.train_ratio / (args.train_ratio + args.val_ratio)))
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    print(f"分割結果: Train = {len(train_pairs)} 枚, Val = {len(val_pairs)} 枚")

    # 出力先ディレクトリの作成
    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "labels"), exist_ok=True)

    def copy_pairs(split_pairs, split_name):
        for img_src, txt_src in split_pairs:
            img_dst = os.path.join(args.output_dir, split_name, "images", os.path.basename(img_src))
            txt_dst = os.path.join(args.output_dir, split_name, "labels", os.path.basename(txt_src))
            shutil.copy2(img_src, img_dst)
            shutil.copy2(txt_src, txt_dst)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    # data.yaml の生成 (絶対パスまたは相対パス)
    abs_output_dir = os.path.abspath(args.output_dir)
    data_yaml_content = {
        "path": abs_output_dir,
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)},
    }

    yaml_path = os.path.join(args.output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n[成功] データセットの分割が完了しました！")
    print(f"  ・data.yaml: {yaml_path}")
    print(f"次のステップ: 'python3 scripts/train_yolo.py --data {yaml_path}' で学習を開始できます。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
