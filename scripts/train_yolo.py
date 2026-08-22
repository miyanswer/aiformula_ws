#!/usr/bin/env python3
"""
Ultralytics YOLO (YOLOv8 / YOLOv11) を用いた物体認識モデルの学習スクリプト。

機能:
  - Apple Silicon (MPS) / NVIDIA GPU (CUDA) / CPU を自動検出・最適化
  - 転移学習 (事前学習済み重み yolov8n.pt / yolo11n.pt 等から微調整)
  - 学習完了後の検証 (val) & テスト
  - オプションで ONNX や TorchScript への自動エクスポート
"""

import argparse
import os
import sys
import torch


def detect_best_device():
    if torch.cuda.is_available():
        return "0"  # CUDA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO物体認識モデルの学習ツール")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml", help="data.yamlのパス")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="事前学習モデル (例: yolo11n.pt, yolo11s.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="エポック数 (推奨: 50〜100)")
    parser.add_argument("--imgsz", type=int, default=640, help="学習画像サイズ (デフォルト: 640)")
    parser.add_argument("--batch", type=int, default=16, help="バッチサイズ (-1 でオート)")
    parser.add_argument("--device", type=str, default="", help="実行デバイス (空欄で自動検出: mps / cuda / cpu)")
    parser.add_argument("--project", type=str, default="runs/train", help="学習結果の保存先ディレクトリ")
    parser.add_argument("--name", type=str, default="yolo_custom", help="実験名")
    parser.add_argument("--export-onnx", action="store_true", help="学習完了後にONNX形式へエクスポートする")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[エラー] 'ultralytics' ライブラリが見つかりません。")
        print("以下を実行してインストールしてください:")
        print("  pip3 install -r requirements-yolo.txt")
        return 1

    if not os.path.isfile(args.data):
        print(f"[エラー] data.yaml が見つかりません: {args.data}")
        print("先に 'python3 scripts/split_dataset.py' を実行してデータセットを作成してください。")
        return 1

    device = args.device if args.device else detect_best_device()

    print("=== YOLO 物体認識モデル学習 ===")
    print(f"データ設定: {args.data}")
    print(f"ベースモデル: {args.model}")
    print(f"エポック数: {args.epochs}")
    print(f"画像サイズ: {args.imgsz}")
    print(f"バッチサイズ: {args.batch}")
    print(f"使用デバイス: {device} (PyTorch: {torch.__version__})")
    print(f"出力先: {args.project}/{args.name}")
    print("================================")

    # モデルのロード
    print(f"\n[1/3] ベースモデル '{args.model}' をダウンロード/ロード中...")
    model = YOLO(args.model)

    # 学習の実行
    print(f"\n[2/3] 学習を開始します ({args.epochs} epochs)...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best_weight = os.path.join(args.project, args.name, "weights", "best.pt")
    print("\n[3/3] 学習完了！")
    print(f"  ・最高精度モデル (Best Weights): {best_weight}")
    print(f"  ・学習ログ/グラフ: {os.path.join(args.project, args.name)}")

    # ONNXエクスポート
    if args.export_onnx:
        print("\n[追加処理] ONNX形式へエクスポート中...")
        best_model = YOLO(best_weight)
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz)
        print(f"  ・ONNXモデル: {onnx_path}")

    print("\n次のステップ:")
    print(f"  Webカメラでリアルタイム推論をテスト:")
    print(f"  python3 scripts/detect_webcam.py --model {best_weight}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
