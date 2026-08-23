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
├── data/
│   └── tasks/
│       ├── jinmen_dog/          # 人面犬認識タスク
│       │   ├── raw_videos/      # 撮影動画
│       │   ├── extracted_frames/# 抽出画像 & AnyLabeling .json / .txt
│       │   └── dataset/         # 分割済みYOLOデータ (data.yaml, train, val)
│       ├── traffic_light/       # 信号機認識タスク
│       ├── t_junction/          # T字路標示認識タスク
│       └── crosswalk/           # 横断歩道認識タスク
├── models/                      # 学習済みベストモデルの集約ディレクトリ
│   ├── jinmen_dog.pt            # 人面犬の学習済みモデル
│   ├── traffic_light.pt         # 信号機の学習済みモデル (学習後)
│   ├── t_junction.pt            # T字路標示の学習済みモデル (学習後)
│   └── crosswalk.pt             # 横断歩道の学習済みモデル (学習後)
├── scripts/                     # YOLO 物体認識 & データセット用スクリプト集
│   ├── record_video.py          # Webカメラ FHD 15fps 録画ツール (--task 対応)
│   ├── extract_frames.py        # 静止画フレーム切り出しツール (--task 対応)
│   ├── auto_annotate.py         # 1枚ラベリングから全自動アノテーション (--task 対応)
│   ├── split_dataset.py         # YOLOデータセット自動分割 & data.yaml 生成 (--task 対応)
│   ├── train_yolo.py            # YOLOv11 ファインチューニング学習 (--task 対応)
│   └── detect_webcam.py         # リアルタイムWebカメラ物体認識テスト (--task 対応)
├── Makefile                     # コマンドショートカット (TASK=... 対応)
├── requirements-yolo.txt        # YOLO パイプライン用 Python 依存関係
├── .gitignore
└── README.md
```

---

## 🎯 タスク別 YOLO 物体認識 学習パイプライン

信号機（`traffic_light`）、T字路（`t_junction`）、横断歩道（`crosswalk`）、人面犬（`jinmen_dog`）など、**タスク名（`TASK`）を指定するだけ**で、それぞれのデータセットとモデルを完全に分離して独立管理できます。

### 1. 依存ライブラリのインストール (初回のみ)

```bash
make yolo-install
```

---

### 2. Webカメラで動画を撮影

認識させたい対象（例: 信号機 `traffic_light`）をWebカメラで撮影します。

```bash
# 信号機を撮影する場合
make record TASK=traffic_light

# T字路標示を撮影する場合
make record TASK=t_junction

# 横断歩道を撮影する場合
make record TASK=crosswalk
```
- **操作方法**: `[r]` で録画開始/停止、`[s]` でスクショ、`[q]` で終了。
- 動画は自動的に `data/tasks/<TASK>/raw_videos/` に保存されます。

---

### 3. 動画から学習用フレーム（静止画）を抽出

```bash
make extract TASK=traffic_light
```
- 切り出された画像は `data/tasks/<TASK>/extracted_frames/` に自動保存されます。

---

### 4. アノテーション（ラベル付け）

#### ⚡ おすすめ: AnyLabeling (SAM 2 AIアシスト)
```bash
make label TASK=traffic_light
```
1. 上部メニューの **「Auto-Labeling (AI)」** をONにし、**`Segment Anything 2 (Hiera-Tiny)`** を選択。
2. 物体（信号機など）を **左クリック** するとAIが自動で綺麗に輪郭をハイライトします。
3. **`Space`** を押して確定し、クラス名（例: `red`, `green`, `yellow`）を入力。
4. **`D`** キーで次の画像へ進みます（自動保存されます）。

#### (参考) 1枚ラベリングからの全自動トラッキングアノテーション
```bash
make auto-annotate TASK=traffic_light
```

---

### 5. データセットの分割 (Train / Val) & `data.yaml` 生成

AnyLabeling の `.json` や `.txt` を自動検出し、80%の訓練データと20%の検証データに自動分割します。

```bash
make split TASK=traffic_light
```
- 出力先: `data/tasks/<TASK>/dataset/` (`data.yaml`, `train/`, `val/`)

---

### 6. YOLO モデルの学習 (Fine-Tuning)

Apple Silicon GPU (`mps`) で高速に学習（約1〜2分）します。

```bash
make train TASK=traffic_light
```
- 学習が完了すると、最高精度のモデルが **`models/<TASK>.pt`**（例: `models/traffic_light.pt`）に自動保存・集約されます。

---

### 7. 学習済みモデルでリアルタイム Webカメラ認識テスト

```bash
# 信号機モデルでテスト
make detect TASK=traffic_light

# 人面犬モデルでテスト
make detect TASK=jinmen_dog

# TASKを省略すると、作成済みモデル一覧から番号で選べます！
make detect
```

