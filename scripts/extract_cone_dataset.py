#!/usr/bin/env python3
"""
honda_data_set から cone のみを抽出して data/tasks/cone/dataset に配置するスクリプト。

出力内容:
  1. YOLO形式データセット:
     - train/images/*.jpg, train/labels/*.txt
     - val/images/*.jpg, val/labels/*.txt
     - data.yaml
  2. BDD100K形式アノテーション (cone のみ抽出):
     - det_annotations/train/*.json
     - det_annotations/val/*.json
"""

import os
import shutil
import json
from pathlib import Path

SRC_BASE = Path("/Users/miyanswer/aiformula_ws/data/honda_data_set")
DST_BASE = Path("/Users/miyanswer/aiformula_ws/data/tasks/cone/dataset")

IMG_WIDTH = 1920.0
IMG_HEIGHT = 1080.0

def convert_box_to_yolo(box):
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])
    
    x_center = (x1 + x2) / 2.0 / IMG_WIDTH
    y_center = (y1 + y2) / 2.0 / IMG_HEIGHT
    w = (x2 - x1) / IMG_WIDTH
    h = (y2 - y1) / IMG_HEIGHT
    
    # 0.0〜1.0 にクリップ
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h

def process_split(split_name):
    src_det_dir = SRC_BASE / "det_annotations" / split_name
    src_img_dir = SRC_BASE / "images" / split_name
    
    dst_img_dir = DST_BASE / split_name / "images"
    dst_lbl_dir = DST_BASE / split_name / "labels"
    dst_det_dir = DST_BASE / "det_annotations" / split_name
    
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    dst_det_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted([f for f in os.listdir(src_det_dir) if f.endswith(".json")])
    
    total_images = len(json_files)
    total_cone_boxes = 0
    images_with_cone = 0
    
    for fname in json_files:
        base_name = fname.replace(".json", "")
        img_name = f"{base_name}.jpg"
        
        # 1. 画像のコピー
        src_img_path = src_img_dir / img_name
        dst_img_path = dst_img_dir / img_name
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {src_img_path} not found!")
            
        # 2. アノテーションの読み込みと cone の抽出
        with open(src_det_dir / fname, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        cone_objects = []
        yolo_lines = []
        
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") == "cone":
                    cone_objects.append(obj)
                    b = obj.get("box2d")
                    xc, yc, w, h = convert_box_to_yolo(b)
                    yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    total_cone_boxes += 1
                    
        if len(cone_objects) > 0:
            images_with_cone += 1
            
        # 3. YOLOラベルファイル書き込み
        dst_lbl_path = dst_lbl_dir / f"{base_name}.txt"
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)
            
        # 4. cone のみフィルタリングした JSON 書き込み
        filtered_json = {
            "name": data.get("name", base_name),
            "frames": [
                {
                    "timestamp": data.get("frames", [{}])[0].get("timestamp", 0) if data.get("frames") else 0,
                    "objects": cone_objects
                }
            ]
        }
        with open(dst_det_dir / fname, "w", encoding="utf-8") as f:
            json.dump(filtered_json, f, indent=2)
            
    print(f"[{split_name}] 完了: 全画像={total_images}, coneを含む画像={images_with_cone}, 抽出されたcone数={total_cone_boxes}")
    return total_images, images_with_cone, total_cone_boxes

def main():
    print("=== cone データセット抽出開始 ===")
    train_stats = process_split("train")
    val_stats = process_split("val")
    
    # data.yaml の作成
    data_yaml_content = f"""path: {DST_BASE}
train: train/images
val: val/images
names:
  0: cone
"""
    yaml_path = DST_BASE / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    print(f"data.yaml を作成しました: {yaml_path}")
    print("=== 抽出処理完了 ===")

if __name__ == "__main__":
    main()
