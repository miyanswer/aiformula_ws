#!/usr/bin/env python3
"""
auto_annotate.py - 1枚のラベリングから全画像を完全自動でアノテーションするツール

概要:
  最初の1枚（または指定した画像）にユーザーがバウンディングボックスとラベルを設定するだけで、
  後続の何十枚・何百枚もの連続画像（動画フレームなど）に対して、AI / ビジュアルトラッカー
  （OpenCV CSRT + SAM 2 輪郭補正）が物体を自動追跡し、YOLO形式のアノテーション（.txt）を一括自動生成します。

特徴:
  - 完全無料 & 完全ローカル実行（Apple Silicon MPS / CPU / CUDA 対応）
  - 複数物体・複数クラスの同時追跡に対応
  - 1枚目の直感的なGUI指定（マウスドラッグ）または既存YOLOラベル読み込みに対応
  - SAM 2 (Segment Anything 2.1) による高精度な境界線・バウンディングボックス自動補正
  - リアルタイムでの追跡プレビューアニメーション表示
  - YOLO学習パイプライン（data/annotated/ -> make split -> make train）に完全対応

使い方:
  # 基本（GUIで1枚目に枠と名前をつけて自動追跡）
  python3 scripts/auto_annotate.py

  # SAM 2 によるAI高精度補正を有効化
  python3 scripts/auto_annotate.py --use-sam

  # 1枚目をすでに AnyLabeling 等でラベリング済みの場合
  python3 scripts/auto_annotate.py --from-first-label
"""

import os
import sys
import glob
import time
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# カラーパレット（複数クラス表示用）
COLORS = [
    (0, 255, 0),    # 緑
    (255, 0, 0),    # 青
    (0, 0, 255),    # 赤
    (0, 255, 255),  # 黄
    (255, 0, 255),  # マゼンタ
    (255, 255, 0),  # シアン
    (0, 165, 255),  # オレンジ
    (180, 105, 255),# ピンク
    (50, 205, 50),  # ライムグリーン
    (255, 140, 0),  # ダークオレンジ
]


class TrackedObject:
    """追跡対象の物体情報"""
    def __init__(self, obj_id: int, class_name: str, class_id: int, bbox_xywh: Tuple[int, int, int, int], tracker):
        self.obj_id = obj_id
        self.class_name = class_name
        self.class_id = class_id
        self.bbox = bbox_xywh  # (x, y, w, h)
        self.tracker = tracker
        self.color = COLORS[class_id % len(COLORS)]
        self.is_active = True
        self.history: List[Tuple[int, int]] = []  # 中心軌跡


def create_tracker(tracker_type: str = "CSRT"):
    """OpenCVトラッカーの生成"""
    tracker_type = tracker_type.upper()
    if tracker_type == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif tracker_type == "KCF" and hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    elif tracker_type == "MIL" and hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    else:
        # フォールバック
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        else:
            raise RuntimeError("利用可能なOpenCVトラッカーが見つかりません。pip install opencv-contrib-python を実行してください。")


def convert_xywh_to_yolo(bbox_xywh: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """(x, y, w, h) ピクセル座標を YOLO 形式 (x_center, y_center, norm_w, norm_h) に変換"""
    x, y, w, h = bbox_xywh
    # クリップ
    x1 = max(0, min(img_w, x))
    y1 = max(0, min(img_h, y))
    x2 = max(0, min(img_w, x + w))
    y2 = max(0, min(img_h, y + h))

    w_clipped = max(1, x2 - x1)
    h_clipped = max(1, y2 - y1)
    
    x_center = (x1 + w_clipped / 2.0) / img_w
    y_center = (y1 + h_clipped / 2.0) / img_h
    norm_w = w_clipped / img_w
    norm_h = h_clipped / img_h

    return (x_center, y_center, norm_w, norm_h)


def convert_yolo_to_xywh(yolo_box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """YOLO 形式 (x_center, y_center, norm_w, norm_h) を (x, y, w, h) ピクセル座標に変換"""
    x_center, y_center, norm_w, norm_h = yolo_box
    w = int(norm_w * img_w)
    h = int(norm_h * img_h)
    x = int((x_center * img_w) - (w / 2.0))
    y = int((y_center * img_h) - (h / 2.0))
    return (max(0, x), max(0, y), max(1, w), max(1, h))


class InteractiveAnnotator:
    """1枚目の画像に対してマウスで物体を囲み、ラベルを設定するGUI"""
    def __init__(self, image: np.ndarray, window_name: str = "1-Frame Labeler"):
        self.image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        self.boxes: List[Dict] = []
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_box = None
        self.class_history: List[str] = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.current_box = (x, y, 0, 0)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                x1 = min(self.ix, x)
                y1 = min(self.iy, y)
                w = abs(self.ix - x)
                h = abs(self.iy - y)
                self.current_box = (x1, y1, w, h)
                self.redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1 = min(self.ix, x)
                y1 = min(self.iy, y)
                w = abs(self.ix - x)
                h = abs(self.iy - y)
                if w > 5 and h > 5:
                    self.current_box = (x1, y1, w, h)
                    self.prompt_class_name((x1, y1, w, h))
                self.current_box = None
                self.redraw()

    def prompt_class_name(self, bbox: Tuple[int, int, int, int]):
        """クラス名の入力プロンプト"""
        print("\n" + "=" * 60)
        print(f"📦 バウンディングボックスを追加しました: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        
        # 履歴があれば選択肢を表示
        suggested = ""
        if self.class_history:
            print("登録済みクラス候補:")
            for idx, c in enumerate(self.class_history):
                print(f"  [{idx + 1}] {c}")
            suggested = self.class_history[-1]
            print(f"(Enterを押すと前回のクラス名 '{suggested}' を適用)")

        # コンソールからクラス名を入力
        prompt_text = f"👉 この物体のクラス名を入力してください (例: cone_blue): "
        if suggested:
            prompt_text = f"👉 クラス名を入力 [Enter={suggested}]: "
            
        sys.stdout.write(prompt_text)
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()

        class_name = ""
        if user_input.isdigit() and 1 <= int(user_input) <= len(self.class_history):
            class_name = self.class_history[int(user_input) - 1]
        elif user_input == "" and suggested != "":
            class_name = suggested
        elif user_input != "":
            class_name = user_input
        else:
            class_name = "object"

        if class_name not in self.class_history:
            self.class_history.append(class_name)

        self.boxes.append({
            "bbox": bbox,
            "class_name": class_name
        })
        print(f"✅ クラス '{class_name}' として登録しました！（現在 {len(self.boxes)} 個登録済み）")
        print("=" * 60 + "\n")

    def redraw(self):
        self.display_image = self.image.copy()
        h, w = self.display_image.shape[:2]

        # ガイドテキストのオーバーレイ
        header = np.zeros((70, w, 3), dtype=np.uint8)
        header[:] = (30, 30, 30)
        cv2.putText(header, "1. Drag mouse to select objects | 2. Enter class name in terminal", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, "[Space/Enter]: Start Auto-Annotation | [u]: Undo | [c]: Clear | [q]: Cancel", (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # 登録済みボックスの描画
        for idx, item in enumerate(self.boxes):
            bx, by, bw, bh = item["bbox"]
            cname = item["class_name"]
            c_idx = self.class_history.index(cname) if cname in self.class_history else idx
            color = COLORS[c_idx % len(COLORS)]

            # 矩形
            cv2.rectangle(self.display_image, (bx, by), (bx + bw, by + bh), color, 2)
            # ラベル背景
            label_text = f"{cname} (#{idx+1})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(self.display_image, (bx, max(0, by - th - 8)), (bx + tw + 10, by), color, -1)
            cv2.putText(self.display_image, label_text, (bx + 5, max(th + 2, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # ドラッグ中のボックス描画
        if self.current_box:
            bx, by, bw, bh = self.current_box
            cv2.rectangle(self.display_image, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

        combined = np.vstack([header, self.display_image])
        cv2.imshow(self.window_name, combined)

    def run(self) -> List[Dict]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n🖼️ 1枚目の画像ウィンドウが開きました。")
        print("  1. マウスドラッグで物体を囲んでください。")
        print("  2. ターミナルにクラス名を入力してください（複数登録可能）。")
        print("  3. 完了したら画像ウィンドウで [Space] または [Enter] を押すと自動アノテーションが始まります。\n")

        self.redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in [32, 13]:  # Space or Enter
                if not self.boxes:
                    print("⚠️ 物体が1つも登録されていません。マウスで物体を囲んでください。")
                    continue
                break
            elif key == ord('u'):  # Undo
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"↩️ 取り消し: {removed['class_name']}")
                    self.redraw()
            elif key == ord('c'):  # Clear all
                self.boxes.clear()
                print("🗑️ すべてのボックスをクリアしました。")
                self.redraw()
            elif key == ord('q') or key == 27:  # Cancel
                print("❌ キャンセルされました。")
                self.boxes.clear()
                break

        cv2.destroyWindow(self.window_name)
        return self.boxes


def load_from_first_yolo_label(first_img_path: str, classes_list: List[str]) -> List[Dict]:
    """既存のYOLOラベルファイル（.txt）から1枚目のアノテーションを読み込む"""
    img_path = Path(first_img_path)
    label_path = img_path.with_suffix(".txt")

    # labels フォルダにある可能性も考慮
    if not label_path.exists():
        alt_label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if alt_label_path.exists():
            label_path = alt_label_path

    if not label_path.exists():
        raise FileNotFoundError(f"1枚目のラベルファイルが見つかりません: {label_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"画像を読み込めません: {img_path}")
    img_h, img_w = img.shape[:2]

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                norm_w = float(parts[3])
                norm_h = float(parts[4])

                bbox_xywh = convert_yolo_to_xywh((x_center, y_center, norm_w, norm_h), img_w, img_h)
                
                class_name = f"class_{class_id}"
                if class_id < len(classes_list):
                    class_name = classes_list[class_id]

                boxes.append({
                    "bbox": bbox_xywh,
                    "class_name": class_name,
                    "class_id": class_id
                })

    print(f"📄 1枚目のラベルファイル {label_path.name} から {len(boxes)} 個の物体を読み込みました。")
    return boxes


def refine_box_with_sam(sam_model, image: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """SAM 2 を用いてトラッカーの矩形領域から精密なオブジェクト輪郭・バウンディングボックスを再計算"""
    x, y, w, h = bbox_xywh
    img_h, img_w = image.shape[:2]

    # トラッカー領域の少し外側をプロンプトボックスとして渡す
    margin = 5
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)

    try:
        results = sam_model(image, bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            # マスクの輪郭から最小外接矩形を計算
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_cnt)
                if area > 20:  # 極小ノイズは除外
                    rx, ry, rw, rh = cv2.boundingRect(largest_cnt)
                    return (rx, ry, rw, rh)
    except Exception as e:
        # エラー時は元のトラッカー矩形を維持
        pass

    return bbox_xywh


def main():
    parser = argparse.ArgumentParser(description="1枚のアノテーションから全画像を自動追跡・自動アノテーションするツール")
    parser.add_argument("--task", type=str, default="default",
                        help="タスク名 (例: traffic_light, t_junction, crosswalk, jinmen_dog)")
    parser.add_argument("--input-dir", type=str, default="",
                        help="アノテーション対象の画像フォルダ (省略時は data/tasks/<task>/extracted_frames)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="アノテーション結果の保存先フォルダ (省略時は data/tasks/<task>/annotated)")
    parser.add_argument("--tracker", type=str, default="CSRT", choices=["CSRT", "KCF", "MIL"],
                        help="使用するトラッキングアルゴリズム (デフォルト: CSRT)")
    parser.add_argument("--use-sam", action="store_true",
                        help="SAM 2 (Segment Anything 2.1) による輪郭・矩形AI高精度補正を有効化")
    parser.add_argument("--sam-model", type=str, default="sam2.1_t.pt",
                        help="SAM モデル (sam2.1_t.pt / sam2.1_s.pt / sam2.1_b.pt)")
    parser.add_argument("--from-first-label", action="store_true",
                        help="1枚目のGUI入力をスキップし、既存の1枚目ラベル(.txt)を読み込んで自動実行")
    parser.add_argument("--classes", type=str, default="",
                        help="クラス名リスト (カンマ区切り、例: cone_blue,cone_yellow,cone_orange)")
    parser.add_argument("--no-preview", action="store_true",
                        help="追跡プレビュー画面を非表示にして最高速で実行")
    parser.add_argument("--fps-limit", type=int, default=30,
                        help="プレビュー表示の最大FPS (デフォルト: 30)")
    args = parser.parse_args()

    # 入出力パスの解決
    if not args.input_dir:
        input_dir = Path(f"data/tasks/{args.task}/extracted_frames")
        if (not input_dir.exists() or not any(input_dir.glob("*.*"))):
            raw_img_dir = Path(f"data/tasks/{args.task}/raw_images")
            if raw_img_dir.exists() and any(raw_img_dir.glob("*.*")):
                input_dir = raw_img_dir
    else:
        input_dir = Path(args.input_dir)

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/tasks/{args.task}/annotated")
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    # 対象画像の取得
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"]
    image_paths = []
    for ext in image_exts:
        image_paths.extend(sorted(input_dir.glob(ext)))
    # ソート
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"❌ エラー: '{input_dir}' に画像ファイルが見つかりません。")
        print(f"💡 `make extract TASK={args.task}` で動画や静止画像を取り込むか、'data/tasks/{args.task}/raw_images' に画像を配置してください。")
        return

    print("=" * 70)
    print("🚀 1枚ラベリングからの全自動アノテーションツール (One-Shot Auto-Annotator)")
    print("=" * 70)
    print(f"📁 入力フォルダ   : {input_dir} ({len(image_paths)} 枚の画像)")
    print(f"💾 出力フォルダ   : {output_dir}")
    print(f"🎯 トラッカー     : {args.tracker}")
    print(f"✨ SAM 2 AI補正   : {'有効 (' + args.sam_model + ')' if args.use_sam else '無効'}")
    print("=" * 70)

    # 出力フォルダ作成
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # SAM モデルのロード（オプション）
    sam_model = None
    if args.use_sam:
        print("⏳ SAM 2.1 AIモデルを読み込み中...")
        try:
            from ultralytics import SAM
            sam_model = SAM(args.sam_model)
            print("✅ SAM 2.1 モデルを正常にロードしました。")
        except Exception as e:
            print(f"⚠️ SAM の読み込みに失敗しました ({e})。OpenCVトラッカー単体で続行します。")
            sam_model = None

    # クラスリストの初期化
    classes_list = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else []

    # 1枚目の画像読み込み
    first_img_path = str(image_paths[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"❌ 1枚目の画像を読み込めませんでした: {first_img_path}")
        return
    img_h, img_w = first_img.shape[:2]

    # 1枚目のアノテーション取得
    initial_boxes = []
    if args.from_first_label:
        try:
            initial_boxes = load_from_first_yolo_label(first_img_path, classes_list)
        except Exception as e:
            print(f"⚠️ 既存ラベルの読み込みに失敗しました ({e})。GUI対話モードに切り替えます。")
            annotator = InteractiveAnnotator(first_img, window_name="1-Frame Interactive Labeler")
            initial_boxes = annotator.run()
    else:
        annotator = InteractiveAnnotator(first_img, window_name="1-Frame Interactive Labeler")
        initial_boxes = annotator.run()

    if not initial_boxes:
        print("❌ アノテーションが指定されなかったため、終了します。")
        return

    # クラスIDのマッピング
    for item in initial_boxes:
        cname = item["class_name"]
        if cname not in classes_list:
            classes_list.append(cname)
        item["class_id"] = classes_list.index(cname)

    print("\n📋 登録されたアノテーション対象:")
    for idx, item in enumerate(initial_boxes):
        print(f"  [{idx + 1}] クラス: {item['class_name']} (ID: {item['class_id']}) - 初期位置: {item['bbox']}")
    print(f"🏷️ クラス一覧: {classes_list}\n")

    # クラス一覧ファイルの保存
    classes_txt_path = output_dir / "classes.txt"
    with open(classes_txt_path, "w", encoding="utf-8") as f:
        for cname in classes_list:
            f.write(f"{cname}\n")
    print(f"💾 クラス一覧を保存しました: {classes_txt_path}")

    # トラッカーの初期化
    tracked_objects: List[TrackedObject] = []
    for idx, item in enumerate(initial_boxes):
        tracker = create_tracker(args.tracker)
        tracker.init(first_img, item["bbox"])
        obj = TrackedObject(
            obj_id=idx + 1,
            class_name=item["class_name"],
            class_id=item["class_id"],
            bbox_xywh=item["bbox"],
            tracker=tracker
        )
        tracked_objects.append(obj)

    # 1枚目のアノテーション保存
    first_out_img_path = output_images_dir / image_paths[0].name
    first_out_lbl_path = output_labels_dir / f"{image_paths[0].stem}.txt"
    shutil.copy2(first_img_path, first_out_img_path)

    with open(first_out_lbl_path, "w", encoding="utf-8") as f:
        for obj in tracked_objects:
            yolo_box = convert_xywh_to_yolo(obj.bbox, img_w, img_h)
            f.write(f"{obj.class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

    print(f"✅ 1枚目保存完了: {first_out_img_path.name}")
    print("\n🎬 残りのフレームに対する自動トラッキング＆アノテーションを開始します...")
    if not args.no_preview:
        cv2.namedWindow("Auto-Annotating Progress", cv2.WINDOW_AUTOSIZE)

    start_time = time.time()
    annotated_count = 1
    total_images = len(image_paths)

    # 2枚目以降を連続追跡
    for idx in range(1, total_images):
        img_path = str(image_paths[idx])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        current_frame_labels = []

        # 全オブジェクトを追跡
        for obj in tracked_objects:
            if not obj.is_active:
                continue

            success, bbox = obj.tracker.update(frame)
            if success:
                bx, by, bw, bh = [int(v) for v in bbox]

                # SAM 2 によるAI輪郭補正（オプション）
                if sam_model is not None:
                    bx, by, bw, bh = refine_box_with_sam(sam_model, frame, (bx, by, bw, bh))

                # 画面内チェック
                if bw > 5 and bh > 5 and bx + bw > 0 and by + bh > 0 and bx < frame_w and by < frame_h:
                    obj.bbox = (bx, by, bw, bh)
                    obj.history.append((bx + bw // 2, by + bh // 2))
                    if len(obj.history) > 30:
                        obj.history.pop(0)

                    # YOLO形式に変換して追加
                    yolo_box = convert_xywh_to_yolo(obj.bbox, frame_w, frame_h)
                    current_frame_labels.append((obj.class_id, yolo_box, obj.class_name, obj.bbox, obj.color))
                else:
                    # 画面外または消失
                    obj.is_active = False
            else:
                # 追跡失敗（ロスト）
                obj.is_active = False

        # 画像とラベルを保存
        out_img_path = output_images_dir / image_paths[idx].name
        out_lbl_path = output_labels_dir / f"{image_paths[idx].stem}.txt"
        shutil.copy2(img_path, out_img_path)

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            for item in current_frame_labels:
                cid, ybox, _, _, _ = item
                f.write(f"{cid} {ybox[0]:.6f} {ybox[1]:.6f} {ybox[2]:.6f} {ybox[3]:.6f}\n")

        annotated_count += 1

        # プレビュー画面描画
        if not args.no_preview:
            preview_frame = frame.copy()

            # 軌跡とバウンディングボックスの描画
            for obj in tracked_objects:
                if len(obj.history) > 1:
                    for h_idx in range(1, len(obj.history)):
                        cv2.line(preview_frame, obj.history[h_idx - 1], obj.history[h_idx], obj.color, 2)

            for item in current_frame_labels:
                cid, ybox, cname, bbox_xywh, color = item
                bx, by, bw, bh = bbox_xywh
                cv2.rectangle(preview_frame, (bx, by), (bx + bw, by + bh), color, 2)
                
                label_txt = f"{cname}"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(preview_frame, (bx, max(0, by - th - 6)), (bx + tw + 6, by), color, -1)
                cv2.putText(preview_frame, label_txt, (bx + 3, max(th + 2, by - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            # ステータス情報オーバーレイ
            progress_pct = (annotated_count / total_images) * 100
            status_bar = np.zeros((40, frame_w, 3), dtype=np.uint8)
            cv2.putText(status_bar, f"Progress: {annotated_count}/{total_images} ({progress_pct:.1f}%) | Active Objects: {len(current_frame_labels)} | [q]: Stop & Save",
                        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            combined_preview = np.vstack([status_bar, preview_frame])
            cv2.imshow("Auto-Annotating Progress", combined_preview)

            key = cv2.waitKey(max(1, int(1000 / args.fps_limit))) & 0xFF
            if key == ord('q'):
                print("\n⏹️ ユーザーにより中断されました。これまでのアノテーションを保存して終了します。")
                break
            elif key == ord(' '):  # 一時停止
                print("⏸️ 一時停止中... 任意のキーで再開")
                cv2.waitKey(0)

        # コンソール進捗表示
        if annotated_count % 10 == 0 or annotated_count == total_images:
            sys.stdout.write(f"\r⏳ 自動アノテーション進捗: {annotated_count}/{total_images} 枚完了 ({(annotated_count/total_images)*100:.1f}%)")
            sys.stdout.flush()

    if not args.no_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 自動アノテーション完了！ ({annotated_count}/{total_images} 枚処理, 所要時間: {elapsed:.2f}秒, 平均 {annotated_count/max(0.1, elapsed):.1f} fps)")
    print(f"📁 保存先ディレクトリ:")
    print(f"  - 画像ファイル : {output_images_dir}")
    print(f"  - ラベル(.txt) : {output_labels_dir}")
    print(f"  - クラス定義   : {classes_txt_path}")
    print("=" * 70)

    # 次のステップのガイド
    classes_arg = ",".join(classes_list)
    print("\n👉 次のステップ（データセット分割 & YOLO学習）:")
    print(f"  1. データセット分割 & data.yaml 生成:")
    print(f"     python3 scripts/split_dataset.py --classes \"{classes_arg}\"")
    print(f"     # または")
    print(f"     make split")
    print(f"  2. YOLOモデルの学習:")
    print(f"     python3 scripts/train_yolo.py --epochs 50")
    print(f"     # または")
    print(f"     make train\n")


if __name__ == "__main__":
    main()
