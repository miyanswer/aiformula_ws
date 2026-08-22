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
├── compose.yaml            # Docker Compose 設定 (ポート8080開放)
├── Makefile                # コマンドショートカット
├── .gitignore
└── README.md
```
