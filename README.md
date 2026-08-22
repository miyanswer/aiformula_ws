# aiformula_ws (ROS 2 Humble Development Environment)

Docker / Dev Container 上で動作する ROS 2 Humble の開発環境です。

## 概要

- **ROS 2 ディストリビューション**: ROS 2 Humble Hawksbill (Ubuntu 22.04 LTS)
- **コンテナ内ワークスペースパス**: `/aiformula_ws`
- **ユーザー**: `rosuser` (sudo 権限あり・パスワード不要)
- **対応アーキテクチャ**: Apple Silicon (arm64) / Intel & AMD (x86_64)
- **Web GUI (noVNC)**: ブラウザから RViz2 / rqt などの画面を表示可能 (`http://localhost:8080`)

---

## 使い方

### 1. Dev Containers (VS Code / Antigravity IDE) で開く場合

1. VS Code / IDE でこのワークスペースフォルダを開きます。
2. 左下の緑色のアイコン、またはコマンドパレット (`Cmd + Shift + P`) から **「Dev Containers: Reopen in Container」** を選択します。
3. 自動的に Docker コンテナがビルド・起動し、コンテナ内環境に接続されます。

---

### 2. ターミナル (CLI / Docker Compose) から利用する場合

#### コンテナのビルド & 起動
```bash
# 初回ビルド (またはDockerfile変更時)
make build

# コンテナのバックグラウンド起動
make up
```

#### コンテナ内シェルへの接続
```bash
# 一般ユーザー (rosuser) として接続
make bash

# root ユーザーとして接続したい場合
make root-bash
```

#### コンテナの停止
```bash
make down
```

---

## 🖥️ Web GUI (RViz2 / rqt) の使い方

Mac のブラウザからコンテナの GUI アプリを表示できます。

1. **ブラウザで GUI 画面を開く**:
   ```bash
   make gui
   # またはブラウザで http://localhost:8080 にアクセス
   ```
2. **コンテナ内で GUI アプリを起動**:
   ```bash
   make bash
   rviz2   # RViz2 を起動
   # または
   rqt     # rqt を起動
   ```
3. ブラウザ上に RViz2 / rqt のウィンドウが表示され、Mac からマウスやキーボードで直接操作できます。

---

## コンテナ内での ROS 2 基本操作

コンテナ内に入ると自動的に `/opt/ros/humble/setup.bash` および `/aiformula_ws/install/setup.bash` が読み込まれます。

### 1. 動作確認 (talker / listener)
```bash
# ターミナル1: パブリッシャ起動
ros2 run demo_nodes_cpp talker

# ターミナル2 (別のシェルで make bash して実行): サブスクライバ起動
ros2 run demo_nodes_py listener
```

### 2. 新しいパッケージの作成
```bash
cd /aiformula_ws/src
ros2 pkg create --build-type ament_python my_python_package
# または
ros2 pkg create --build-type ament_cmake my_cpp_package
```

### 3. ワークスペースのビルド
```bash
cd /aiformula_ws
colcon build --symlink-install
source install/setup.bash
```

---

## ディレクトリ構成

```
aiformula_ws/
├── .devcontainer/
│   └── devcontainer.json   # VS Code / Dev Container 設定
├── docker/
│   ├── Dockerfile          # ROS 2 Humble + noVNC Dockerfile
│   └── entrypoint.sh       # コンテナ起動時エントリポイント (GUIサービス自動起動)
├── src/                    # ROS 2 パッケージ配置用ディレクトリ
├── scripts/                # YOLO 物体認識 & データセット用スクリプト集
│   ├── record_video.py     # Webカメラ FHD 15fps 録画ツール
│   ├── extract_frames.py   # 動画からの静止画フレーム切り出しツール
│   ├── split_dataset.py    # YOLOデータセット自動分割 & data.yaml 生成
│   ├── train_yolo.py       # YOLOv8 / YOLOv11 ファインチューニング学習
│   └── detect_webcam.py    # リアルタイムWebカメラ物体認識テスト
├── compose.yaml            # Docker Compose 設定 (ポート8080開放)
├── Makefile                # コマンドショートカット
├── requirements-yolo.txt   # YOLO パイプライン用 Python 依存関係
├── .gitignore
└── README.md
```

---

## 🎯 YOLO 物体認識 学習パイプライン

Webカメラ（FHD 15fps）で撮影した動画から、YOLO（YOLOv8 / YOLOv11）で物体認識モデルを学習・推論する一連のワークフローです。

### 1. 依存ライブラリのインストール (ホスト側 / ローカル)

```bash
pip3 install -r requirements-yolo.txt
# または
make yolo-install
```

---

### 2. Webカメラで 15fps, FHD (1920x1080) 動画を撮影

```bash
python3 scripts/record_video.py
# または
make record
```
- **操作方法**:
  - `[r]`: 録画の開始 / 停止
  - `[s]`: 現在のフレームをスクリーンショット保存
  - `[q]`: 終了
- 撮影された動画は `data/raw_videos/` に自動保存されます。

---

### 3. 動画から学習用フレーム（静止画）を抽出

動画全体を1コマずつアノテーションすると重複が多いため、**1秒に1枚**（または15フレームごと）の間隔で画像を自動抽出します。

```bash
python3 scripts/extract_frames.py --every-sec 1.0
# または
make extract
```
- 切り出された画像は `data/extracted_frames/` に保存されます。

---

### 4. アノテーション（ラベル付け）

抽出した画像に対して、バウンディングボックス（矩形）のラベル付けを行います。

#### おすすめアノテーションツール:
1. **[AnyLabeling](https://github.com/vietanhdev/anylabeling)** (AI自動アノテーション対応・オフラインで超高速)
2. **[Roboflow](https://roboflow.com/)** (Webブラウザでチーム作業・YOLOフォーマット自動出力)
3. **[Labelme](https://github.com/labelmeai/labelme)** / **YOLO-Annotation-Tool**

> [!TIP]
> アノテーション完了後、画像（`.jpg`）とYOLO形式のラベル（`.txt`）を `data/annotated/` ディレクトリに配置してください。

---

### 5. データセットの分割 (Train / Val) & `data.yaml` 生成

アノテーション済みデータを `train` (80%) と `val` (20%) に自動分割し、YOLO設定ファイル `data.yaml` を生成します。

```bash
# 例: カラーコーン (blue, yellow, orange) を認識させたい場合
python3 scripts/split_dataset.py --classes "cone_blue,cone_yellow,cone_orange"
# または
make split
```
- 出力先: `data/dataset/` (`train/`, `val/`, `data.yaml`)

---

### 6. YOLO モデルの学習 (Fine-Tuning)

Apple Silicon (MPS) または NVIDIA GPU (CUDA) を自動検出して高速学習します。

```bash
# デフォルト: YOLO11n, 50エポック
python3 scripts/train_yolo.py --epochs 50 --imgsz 640
# または
make train
```
- 学習が完了すると、最高精度モデルが `runs/train/yolo_custom/weights/best.pt` に保存されます。

---

### 7. 学習済みモデルでリアルタイム Webカメラ物体認識テスト

学習したモデルを使って、Webカメラ映像でリアルタイムに物体認識をテストします。

```bash
python3 scripts/detect_webcam.py --model runs/train/yolo_custom/weights/best.pt
# または
make detect
```
- 画面上に検出枠、クラス名、信頼度スコア、推論ミリ秒/FPSが表示されます。

