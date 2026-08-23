#!/usr/bin/env python3
"""
Ultralytics YOLO (YOLOv8 / YOLOv11) を用いた物体認識モデルの学習スクリプト。

機能:
  - タスク別管理 (--task) 対応
  - Apple Silicon (MPS) / NVIDIA GPU (CUDA) / CPU を自動検出・最適化
  - 転移学習 (事前学習済み重み yolov8n.pt / yolo11n.pt 等から微調整)
  - 学習完了後に models/<task>.pt へ自動コピー集約
  - オプションで ONNX や TorchScript への自動エクスポート
"""

import argparse
import os
import shutil
import sys
import torch
from pathlib import Path


def detect_best_device():
    if torch.cuda.is_available():
        return "0"  # CUDA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"


def find_data_yaml(task: str, explicit_data: str) -> str:
    if explicit_data and os.path.isfile(explicit_data):
        return explicit_data

    candidates = []
    if task:
        candidates.append(f"data/tasks/{task}/dataset/data.yaml")
    
    # data/tasks 配下を探索
    tasks_dir = "data/tasks"
    if os.path.exists(tasks_dir):
        for t_name in sorted(os.listdir(tasks_dir)):
            cand = f"data/tasks/{t_name}/dataset/data.yaml"
            if os.path.isfile(cand):
                candidates.append(cand)

    candidates.append("data/dataset/data.yaml")

    for c in candidates:
        if os.path.isfile(c):
            return c

    return candidates[0] if candidates else "data/dataset/data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO物体認識モデルの学習ツール")
    parser.add_argument("--task", type=str, default="", help="タスク名 (例: traffic_light, t_junction, crosswalk, jinmen_dog)")
    parser.add_argument("--data", type=str, default="", help="data.yamlのパス (省略時は data/tasks/<task>/dataset/data.yaml)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="事前学習モデル (例: yolo11n.pt, yolo11s.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="エポック数 (推奨: 50〜100)")
    parser.add_argument("--imgsz", type=int, default=640, help="学習画像サイズ (デフォルト: 640)")
    parser.add_argument("--batch", type=int, default=16, help="バッチサイズ (-1 でオート)")
    parser.add_argument("--device", type=str, default="", help="実行デバイス (空欄で自動検出: mps / cuda / cpu)")
    parser.add_argument("--project", type=str, default="runs/train", help="学習結果の保存先ディレクトリ")
    parser.add_argument("--name", type=str, default="", help="実験名 (省略時はタスク名)")
    parser.add_argument("--export-onnx", action="store_true", help="学習完了後にONNX形式へエクスポートする")
    return parser.parse_args()


def main():
    args = parse_args()

    # タスク名の推測と設定
    data_yaml = find_data_yaml(args.task, args.data)
    if not os.path.isfile(data_yaml):
        print(f"[エラー] data.yaml が見つかりません: {data_yaml}")
        print("先に 'make split' (または 'python3 scripts/split_dataset.py') を実行してデータセットを作成してください。")
        return 1

    task_name = args.task
    if not task_name:
        parts = Path(data_yaml).parts
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                task_name = parts[idx + 1]
    if not task_name:
        task_name = "default"

    exp_name = args.name if args.name else task_name

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[エラー] 'ultralytics' ライブラリが見つかりません。")
        print("以下を実行してインストールしてください:")
        print("  pip3 install -r requirements-yolo.txt")
        return 1

    device = args.device if args.device else detect_best_device()

    print("=" * 60)
    print("=== YOLO 物体認識モデル学習 ===")
    print("=" * 60)
    print(f"🎯 タスク名     : {task_name}")
    print(f"📄 データ設定   : {data_yaml}")
    print(f"🧠 ベースモデル : {args.model}")
    print(f"🔄 エポック数   : {args.epochs}")
    print(f"🖼️ 画像サイズ   : {args.imgsz}")
    print(f"📦 バッチサイズ : {args.batch}")
    print(f"⚡ 使用デバイス : {device} (PyTorch: {torch.__version__})")
    print(f"💾 出力先       : {args.project}/{exp_name}")
    print("=" * 60)

    # モデルのロード
    print(f"\n[1/3] ベースモデル '{args.model}' をロード中...")
    model = YOLO(args.model)

    # 学習の実行
    print(f"\n[2/3] 学習を開始します ({args.epochs} epochs)...")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=exp_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best_weight = os.path.join(args.project, exp_name, "weights", "best.pt")
    
    # models ディレクトリへの集約保存
    os.makedirs("models", exist_ok=True)
    model_hub_path = os.path.join("models", f"{task_name}.pt")
    if os.path.exists(best_weight):
        shutil.copy2(best_weight, model_hub_path)

    print("\n" + "=" * 60)
    print("🎉 [3/3] 学習完了！")
    print("=" * 60)
    print(f"  ・最高精度モデル (Best Weights) : {best_weight}")
    print(f"  ・モデル集約フォルダ           : {model_hub_path}")
    print(f"  ・学習ログ / グラフ            : {os.path.join(args.project, exp_name)}")
    print("=" * 60)

    # ONNXエクスポート
    if args.export_onnx:
        print("\n[追加処理] ONNX形式へエクスポート中...")
        best_model = YOLO(best_weight)
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz)
        print(f"  ・ONNXモデル: {onnx_path}")

    print("\n👉 次のステップ:")
    print(f"  Webカメラでリアルタイム推論をテスト:")
    print(f"  make detect TASK={task_name}")
    print(f"  # または python3 scripts/detect_webcam.py --task {task_name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
