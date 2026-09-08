# aiformula_base (機体ハードウェア基盤 & プラットフォーム層)

AI Formula 実車機体を動かすための **提供ベースフレームワーク（ハードウェア抽象化・センシング・モーター駆動・安全機構）** をまとめた最小構成パッケージ群です。

旧Perception（306MB）や不要なアルゴリズム層を排除し、**純粋な機体プラットフォーム（約74MB）** にスリム化されています。

---

## 🏎️ 含まれるパッケージ一覧

| カテゴリ | パッケージ名 | 役割・説明 |
| :--- | :--- | :--- |
| **`vehicles/`** | `sample_vehicle` | 車両の物理モデル、URDF/Xacro、TF座標系、ZED Xカメラマウント位置 |
| **`sensing/`** | `zed-ros2-wrapper` | ZED X ステレオカメラ 公式ラッパー (RGB / Depth / IMU) |
| | `vectornav` | VectorNav 9軸IMU / GNSS ドライバ |
| | `odometry_publisher` | 車輪エンコーダ ＋ ジャイロ統合オドメトリ |
| | `rear_potentiometer` | ステアリング・後輪角度センサ |
| **`control/`** | `motor_controller` | 速度指令 `Twist` を左右車輪 RPM に変換し CAN 送信 (ID: `0x210`) |
| | `twist_mux` | 安全優先度マルチプレクサ (非常停止・手動介入優先切替) |
| **`launchers/`**| `sample_launchers` | 機体一括起動 Launch (`hardware_bringup.launch.py`) |
| **`common/`** | `aiformula_interfaces` | 共通メッセージ型定義 |
| | `common_python`, `common_cpp` | 共通Launchユーティリティ |
| **`bash/`** | `can_bringup.sh`, `init_sensors.sh` | CANインターフェース (500kbps) 起動・センサ権限設定スクリプト |

---

## 🚀 実機での起動方法

### 1. CAN と センサ権限の初期化
```bash
bash src/aiformula_base/bash/step1.sh
# または
sudo ip link set can0 up type can bitrate 500000
```

### 2. 機体ハードウェア基盤の一括起動
```bash
ros2 launch sample_launchers hardware_bringup.launch.py
```
*(※カメラ、IMU、CAN通信、モーター制御、TF、安全機構が一括で立ち上がり、`/aiformula_control/handle_controller/cmd_vel` の入力を待機します)*

### 3. 頭脳（2027 制御・認識ノード）の起動
```bash
ros2 launch ai_formula_oit_2027 controller.launch.py
ros2 launch ai_formula_oit_2027 traffic_light_detector.launch.py
```
