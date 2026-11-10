#!/usr/bin/env python3
"""
convert_20261027_experiments.py

2026-10-27 の加速・減速・旋回実験の rosbag を Foxglove 用 MCAP にする.
convert_bag_to_mcap.py と同じ統合 (data_0.db3 の全トピック + camera.mp4) に, 確認用の /analysis/* を足す.
元の rosbag は変更しない.

    .venv/bin/python scripts/convert_20261027_experiments.py             # 全回 + 一覧
    .venv/bin/python scripts/convert_20261027_experiments.py 150926      # 1 回だけ
    .venv/bin/python scripts/convert_20261027_experiments.py --overview  # 一覧だけ

出力 (rosbag/mcap/):
    20261027_<時刻>_gamepad.mcap   1 回分. レイアウトは foxglove_layouts/aiformula_20261027_run_layout.json
    20261027_overview.mcap         全回の走行線. レイアウトは foxglove_layouts/aiformula_20261027_overview_layout.json
    20261027_index.md              条件・判定・スピン地点の一覧

判定 (SPIN / DRIFT / OK) は実験者の目視の記録. rosbag からは判定しない.
距離は発進からの odom の走行距離なので, 回をまたいだ「同じ場所」を意味しない (発進場所は途中で変えている).
スピン中の odom の位置は, 車輪速を向きの方向に積分した値で横滑りを含まない.
"""

import argparse
import csv
import heapq
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from mcap.writer import Writer
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.typesys.store import Nodetype

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_bag_to_mcap import encode_cdr_compressed_image, generate_ros2_msgdef, load_stamps  # noqa: E402

ROSBAG_DIR = Path('rosbag')
MCAP_DIR = ROSBAG_DIR / 'mcap'
WHEEL_K = math.pi * 0.254 / 60.0  # rpm -> m/s (vehicles/sample_vehicle/config/wheel.yaml の直径 0.254 m)
RPM_ID = 1809
TREAD = 0.6  # [m] 左右の車輪の間隔
YAW_ONSET = 0.3  # [rad/s] 揺れ始め
YAW_ONSET_SMOOTH = 0.2  # [s] 揺れ始めを見るヨーの平均幅. ZED IMU の瞬間的なノイズ (111822 の 2.9 m など) を拾わないため
YAW_SPIN = 2.0   # [rad/s] スピン地点. 目視でスピンした回は全て超え, しなかった回は超えない

# 種類, 速度 km/h, 加速度, 減速度, 半径, 判定, 実験者のメモ
RUNS = {
    '102050': ('idle', 0, 0, 0, 0, 'NONE', '止まっているだけ'),
    '102229': ('accel', 10, 2.5, 1.0, 0, 'OK', '真っ直ぐ'),
    '102528': ('accel', 20, 2.5, 1.0, 0, 'SPIN', 'スピン'),
    '103040': ('accel', 20, 2.0, 1.0, 0, 'SPIN', 'スピン'),
    '104320': ('accel', 15, 2.5, 1.0, 0, 'DRIFT', 'スピンなし, 右よれ'),
    '104710': ('accel', 15, 3.0, 1.0, 0, 'OK', 'スピンなし, 真っ直ぐ'),
    '105122': ('accel', 15, 3.0, 1.0, 0, 'SPIN', '段差でスピン'),
    '105757': ('accel', 15, 3.0, 1.0, 0, 'OK', 'スピンなし, 真っ直ぐ'),
    '111822': ('accel', 15, 3.5, 1.0, 0, 'SPIN', 'スピン, 白線と黒路面の差によるものか'),
    '112115': ('accel', 15, 3.5, 1.0, 0, 'DRIFT', 'スピンなし, 白線と黒路面の差で右によれた'),
    '112852': ('accel', 15, 3.5, 1.0, 0, 'DRIFT', 'スピンなし, ちょい右よれ, 黒路面だけ'),
    '113221': ('accel', 15, 4.0, 1.0, 0, 'OK', 'スピンなし, 真っ直ぐ'),
    '115207': ('accel', 15, 4.5, 1.0, 0, 'OK', 'スピンなし, ちょい真っ直ぐ'),
    '115443': ('accel', 15, 5.0, 1.0, 0, 'SPIN', '左によって段差で右にスピン'),
    '115826': ('accel', 15, 4.8, 1.0, 0, 'OK', 'スピンなし, ちょい真っ直ぐ'),
    '120236': ('accel', 15, 4.9, 1.0, 0, 'SPIN', '左よれ, 右スピン'),
    '121931': ('brake', 15, 1.0, 4.0, 0, 'OK', 'スピンなし'),
    '122219': ('brake', 15, 1.0, 5.0, 0, 'OK', 'スピンなし, 減速度 4.0 より危ないと感じた'),
    '123216': ('brake', 15, 1.0, 3.0, 0, 'OK', 'スピンなし'),
    '123615': ('brake', 15, 1.0, 3.5, 0, 'OK', 'スピンなし'),
    '150249': ('turn', 5, 1.0, 1.0, 3, 'OK', 'スピンなし'),
    '150407': ('turn', 10, 1.0, 1.0, 10, 'OK', 'スピンなし'),
    '150540': ('turn', 10, 1.0, 1.0, 7, 'OK', 'スピンなし'),
    '150728': ('turn', 10, 1.0, 1.0, 5, 'OK', 'スピンなし'),
    '150926': ('turn', 15, 1.0, 1.0, 10, 'SPIN', 'スピン'),
    '152716': ('turn', 15, 1.0, 1.0, 15, 'SPIN', 'スピン'),
    '152952': ('turn', 15, 1.0, 1.0, 20, 'SPIN', 'スピン'),
    '153329': ('turn', 15, 1.0, 1.0, 40, 'SPIN', 'スピン'),
    '153552': ('turn', 15, 1.0, 1.0, 60, 'SPIN', 'スピン'),
    '155828': ('turn', 15, 1.0, 1.0, 40, 'SPIN', 'スピン (コード変更後)'),
    '160121': ('turn', 15, 1.0, 1.0, 60, 'SPIN', 'スピン (コード変更後)'),
    '160459': ('turn', 13, 1.0, 1.0, 15, 'OK', 'スピンなし (コード変更後)'),
    '160920': ('turn', 14, 1.0, 1.0, 15, 'SPIN', 'スピン (コード変更後)'),
    '161116': ('turn', 13, 1.0, 1.0, 10, 'SPIN', 'スピン (コード変更後)'),
    '161302': ('turn', 13, 1.0, 1.0, 13, 'SPIN', 'スピン (コード変更後)'),
    '161444': ('turn', 13, 1.0, 1.0, 14, 'OK', 'スピンなし (コード変更後)'),
}

TS = get_typestore(Stores.ROS2_HUMBLE)
TS.register(get_types_from_msg(
    'std_msgs/Header header\nuint32 id\nbool is_rtr\nbool is_extended\nbool is_error\nuint8 dlc\nuint8[8] data\n',
    'can_msgs/msg/Frame'))

MARKER = 'visualization_msgs/msg/MarkerArray'
FLOAT64 = 'std_msgs/msg/Float64'
STRING = 'std_msgs/msg/String'
TF = 'tf2_msgs/msg/TFMessage'
LINE_STRIP, SPHERE, TEXT = 4, 2, 9
GREEN, YELLOW, ORANGE, RED = (0.2, 0.85, 0.3), (1.0, 0.85, 0.1), (1.0, 0.5, 0.1), (0.95, 0.15, 0.15)
VERDICT_COLOR = {'OK': GREEN, 'DRIFT': YELLOW, 'SPIN': RED, 'NONE': (0.6, 0.6, 0.6)}
NP_TYPES = {'uint8': np.uint8, 'int8': np.int8, 'float32': np.float32, 'float64': np.float64,
            'uint32': np.uint32, 'int32': np.int32, 'uint16': np.uint16, 'int16': np.int16, 'bool': np.bool_}


def make(typename: str, **overrides):
    """既定値で埋めたメッセージを作る (Marker は使わないフィールドが多いので)."""
    kwargs = {}
    for name, (ntype, val) in TS.FIELDDEFS[typename][1]:
        if name in overrides:
            kwargs[name] = overrides[name]
        elif ntype == Nodetype.BASE:
            kwargs[name] = '' if val[0] == 'string' else (False if val[0] == 'bool' else (0.0 if val[0].startswith('float') else 0))
        elif ntype == Nodetype.NAME:
            kwargs[name] = make(val)
        elif ntype == Nodetype.SEQUENCE:
            sub_type, sub_val = val[0]
            kwargs[name] = np.array([], dtype=NP_TYPES[sub_val[0]]) if sub_type == Nodetype.BASE and sub_val[0] in NP_TYPES else []
        else:
            raise ValueError(f'{typename}.{name}: 既定値がない型')
    return TS.types[typename](**kwargs)


def stamp(t_ns: int):
    return TS.types['builtin_interfaces/msg/Time'](sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))


def header(t_ns: int, frame: str):
    return TS.types['std_msgs/msg/Header'](stamp=stamp(t_ns), frame_id=frame)


def point(x: float, y: float, z: float = 0.0):
    return TS.types['geometry_msgs/msg/Point'](x=float(x), y=float(y), z=float(z))


def rgba(c, a: float = 1.0):
    return TS.types['std_msgs/msg/ColorRGBA'](r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(a))


def vec3(x: float, y: float, z: float):
    return TS.types['geometry_msgs/msg/Vector3'](x=float(x), y=float(y), z=float(z))


def line_marker(t_ns, frame, ns, mid, xy, yaw_abs, width=0.12, solid=None):
    """走行線. solid がなければ |ヨーレート| で色を変える (緑 → 黄 → 橙 → 赤)."""
    colors = [rgba(solid if solid else yaw_color(y)) for y in yaw_abs]
    return make('visualization_msgs/msg/Marker', header=header(t_ns, frame), ns=ns, id=mid, type=LINE_STRIP,
                scale=vec3(width, 0, 0), color=rgba(GREEN), points=[point(x, y) for x, y in xy], colors=colors,
                pose=pose_at(0, 0))


def pose_at(x, y, z=0.0):
    return TS.types['geometry_msgs/msg/Pose'](position=point(x, y, z), orientation=TS.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0))


def sphere_marker(t_ns, frame, ns, mid, x, y, c, size=0.6):
    return make('visualization_msgs/msg/Marker', header=header(t_ns, frame), ns=ns, id=mid, type=SPHERE,
                pose=pose_at(x, y, 0.1), scale=vec3(size, size, size), color=rgba(c))


def text_marker(t_ns, frame, ns, mid, x, y, text, c=(1, 1, 1), size=0.7):
    return make('visualization_msgs/msg/Marker', header=header(t_ns, frame), ns=ns, id=mid, type=TEXT,
                pose=pose_at(x, y, 0.6), scale=vec3(0, 0, size), color=rgba(c), text=text)


def yaw_color(y: float):
    return GREEN if y < YAW_ONSET else YELLOW if y < 1.0 else ORANGE if y < YAW_SPIN else RED


def serialize(msg, typename: str) -> bytes:
    return bytes(TS.serialize_cdr(msg, typename))


def clean_schema(typename: str) -> bytes:
    """同じ型の定義が重複しないように 1 つにまとめる (Marker は Header を複数の経路で含むため)."""
    sep = '=' * 80
    parts = generate_ros2_msgdef(typename).split(f'\n{sep}\n')
    seen, out = set(), [parts[0]]
    for sec in parts[1:]:
        name = sec.split('\n', 1)[0]
        if name not in seen:
            seen.add(name)
            out.append(sec)
    return f'\n{sep}\n'.join(out).encode('utf-8')


def quat_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


@dataclass
class Event:
    t: float
    s: float
    x: float
    y: float
    v: float
    a: float
    yaw: float


@dataclass
class Run:
    tag: str
    kind: str
    kmh: float
    accel: float
    decel: float
    radius: float
    verdict: str
    note: str
    session: Path
    t0: float = 0.0        # 発進
    t_vmax: float = 0.0    # 最高速に達した
    t_turn: float = 0.0    # 切り込み
    t_cut: float = 0.0     # 指令が走行中に切れた (完走なら 0)
    t_end: float = 0.0     # 止まった
    moved: bool = False
    v_max: float = 0.0
    omega_turn: float = 0.0
    start_pose: tuple = (0.0, 0.0, 0.0)
    onset: Optional[Event] = None
    spin: Optional[Event] = None
    yaw_max: float = 0.0
    heading_end: float = 0.0
    odom_heading_end: float = 0.0
    odom: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    series: Dict[str, np.ndarray] = field(default_factory=dict)
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    path_yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    path_s: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def title(self) -> str:
        if self.kind == 'accel':
            return f'{self.kmh:.0f}km/h a{self.accel:.1f}'
        if self.kind == 'brake':
            return f'{self.kmh:.0f}km/h d{self.decel:.1f}'
        if self.kind == 'turn':
            return f'{self.kmh:.0f}km/h R{self.radius:.0f}'
        return 'idle'

    @property
    def kind_jp(self) -> str:
        return {'accel': '加速実験', 'brake': '減速実験', 'turn': '旋回実験', 'idle': '走行なし'}[self.kind]

    @property
    def a_lat(self) -> float:
        return (self.kmh / 3.6) ** 2 / self.radius if self.radius else 0.0


def smooth(t, y, tau):
    """中心移動平均 (不等間隔を一度 100 Hz に直す)."""
    if len(t) < 2:
        return y.copy()
    n = max(1, int(tau * 100))
    grid = np.arange(t[0], t[-1], 0.01)
    if len(grid) < 2:
        return y.copy()
    g = np.interp(grid, t, y)
    g = np.convolve(g, np.ones(n) / n, 'same')
    return np.interp(t, grid, g)


def session_dir(tag: str) -> Path:
    found = sorted(ROSBAG_DIR.glob(f'20261027_{tag}/*/data/metadata.yaml'))
    if not found:
        raise FileNotFoundError(f'20261027_{tag} が見つかりません')
    return found[0].parent.parent


def analyze(tag: str) -> Run:
    run = Run(tag, *RUNS[tag], session=session_dir(tag))
    rows: Dict[str, list] = {k: [] for k in ('mux', 'rpm', 'imu', 'zed', 'odom')}
    with Reader(str(run.session / 'data')) as reader:
        for conn, ts, raw in reader.messages():
            t = ts * 1e-9
            if conn.topic == '/aiformula_control/twist_mux/cmd_vel':
                m = TS.deserialize_cdr(raw, conn.msgtype)
                rows['mux'].append((t, m.linear.x, m.angular.z))
            elif conn.topic == '/aiformula_sensing/vehicle_info':
                m = TS.deserialize_cdr(raw, conn.msgtype)
                if m.id == RPM_ID:
                    rows['rpm'].append((t, *struct.unpack('<ii', bytes(m.data[:8]))))
            elif conn.topic == '/aiformula_sensing/vectornav/imu':
                m = TS.deserialize_cdr(raw, conn.msgtype)
                rows['imu'].append((t, m.linear_acceleration.x, m.linear_acceleration.y))
            elif conn.topic == '/aiformula_sensing/zed_node/imu':
                m = TS.deserialize_cdr(raw, conn.msgtype)
                rows['zed'].append((t, m.angular_velocity.z))
            elif conn.topic == '/aiformula_sensing/gyro_odometry_publisher/odom':
                m = TS.deserialize_cdr(raw, conn.msgtype)
                rows['odom'].append((t, m.pose.pose.position.x, m.pose.pose.position.y, quat_yaw(m.pose.pose.orientation),
                                     m.twist.twist.linear.x, m.twist.twist.angular.z))
    data = {k: np.array(v) if v else np.zeros((0, {'mux': 3, 'rpm': 3, 'imu': 3, 'zed': 2, 'odom': 6}[k])) for k, v in rows.items()}
    odom, mux, rpm, imu, zed = data['odom'], data['mux'], data['rpm'], data['imu'], data['zed']
    run.odom = odom
    moving = np.flatnonzero(np.abs(mux[:, 1]) > 0.02) if len(mux) else np.zeros(0, int)
    if len(moving) == 0:
        run.start_pose = tuple(odom[0, 1:4])
        return run

    # 発進から, 指令が 0.3 秒以上途切れるまでが実験. 以降の手動走行は含めない
    gaps = np.flatnonzero(np.diff(mux[moving, 0]) > 0.3)
    last = moving[gaps[0]] if len(gaps) else moving[-1]
    cmd = mux[moving[0]:last + 1]
    run.moved = True
    run.t0 = cmd[0, 0]
    run.v_max = cmd[:, 1].max()
    top = np.flatnonzero(cmd[:, 1] >= run.v_max - 1e-3)
    run.t_vmax = cmd[top[0], 0]
    if cmd[-1, 1] > 0.2:
        run.t_cut = cmd[-1, 0]
    if run.kind == 'turn':
        steady = np.flatnonzero(np.abs(np.abs(cmd[top, 2]) - abs(cmd[top[-1], 2])) >= 0.003)
        first = top[steady[-1] + 1] if len(steady) else top[0]
        run.t_turn, run.omega_turn = cmd[first, 0], cmd[top[-1], 2]

    # 発進前の静止でゼロ点を取る
    rest = lambda t: (t >= run.t0 - 2.5) & (t < run.t0 - 0.2)
    bias_z = np.median(zed[rest(zed[:, 0]), 1])
    bias_ax = np.median(imu[rest(imu[:, 0]), 1])
    bias_ay = np.median(imu[rest(imu[:, 0]), 2])
    yaw_t, yaw = zed[:, 0], zed[:, 1] - bias_z

    v_t, v_w = rpm[:, 0], (rpm[:, 1] + rpm[:, 2]) * 0.5 * WHEEL_K
    v_s = smooth(v_t, v_w, 0.1)
    a_w = np.gradient(smooth(v_t, v_w, 0.2), v_t) if len(v_t) > 2 else np.zeros_like(v_t)
    a_w = smooth(v_t, a_w, 0.2)
    yaw_wheel = smooth(v_t, (rpm[:, 1] - rpm[:, 2]) * WHEEL_K / TREAD, YAW_ONSET_SMOOTH)  # 左右の車輪の差から計算したヨー

    after = v_t > cmd[-1, 0]
    stopped = np.flatnonzero(after & (v_t < cmd[-1, 0] + 8.0) & (np.abs(v_s) < 0.05))
    run.t_end = v_t[stopped[0]] if len(stopped) else min(cmd[-1, 0] + 8.0, v_t[-1])

    i0 = np.argmin(np.abs(odom[:, 0] - run.t0))
    x0, y0, th0 = odom[i0, 1], odom[i0, 2], odom[i0, 3]
    run.start_pose = (x0, y0, th0)
    c, s = math.cos(-th0), math.sin(-th0)
    rx = c * (odom[:, 1] - x0) - s * (odom[:, 2] - y0)
    ry = s * (odom[:, 1] - x0) + c * (odom[:, 2] - y0)
    in_run = (odom[:, 0] >= run.t0) & (odom[:, 0] <= run.t_end)
    dist = np.r_[0, np.cumsum(np.hypot(np.diff(rx), np.diff(ry)) * (odom[1:, 0] >= run.t0))]

    run.path_xy = np.c_[rx[in_run], ry[in_run]]
    run.path_yaw = np.abs(np.interp(odom[in_run, 0], yaw_t, yaw))
    run.path_s = dist[in_run]

    # 向き: ヨーレートの積分と odom. odom が正しければ一致する
    m = yaw_t >= run.t0
    heading = np.zeros_like(yaw_t)
    heading[m] = np.r_[0, np.cumsum(0.5 * (yaw[m][1:] + yaw[m][:-1]) * np.diff(yaw_t[m]))]
    odom_th = np.unwrap(odom[:, 3]) - np.unwrap(odom[:, 3])[i0]
    run.heading_end = math.degrees(np.interp(run.t_end, yaw_t, heading))
    run.odom_heading_end = math.degrees(np.interp(run.t_end, odom[:, 0], odom_th))

    def event(t):
        return Event(t, float(np.interp(t, odom[:, 0], dist)), float(np.interp(t, odom[:, 0], rx)), float(np.interp(t, odom[:, 0], ry)),
                     float(np.interp(t, v_t, v_s)), float(np.interp(t, v_t, a_w)), float(np.interp(t, yaw_t, yaw)))

    window = (yaw_t >= run.t0) & (yaw_t <= run.t_end)
    run.yaw_max = float(np.abs(yaw[window]).max())
    yaw_s = smooth(yaw_t, yaw, YAW_ONSET_SMOOTH)
    onset = np.flatnonzero(window & (np.abs(yaw_s) > YAW_ONSET))
    if run.kind == 'turn':  # 旋回は指令どおりでもヨーが出るので, 指令より 0.3 rad/s 以上多く回った時点を揺れ始めにする
        # 指令の向きに多く回った分だけを見る. 絶対値にすると, 切り込みでヨーが指令に追いつくまでの差 (指令 0.3 rad/s 以上) を拾う
        cmd_w = np.interp(yaw_t, cmd[:, 0], cmd[:, 2])
        onset = np.flatnonzero(window & (math.copysign(1.0, run.omega_turn) * (yaw_s - cmd_w) > YAW_ONSET))
    if len(onset):
        run.onset = event(yaw_t[onset[0]])
    spin = np.flatnonzero(window & (np.abs(yaw) > YAW_SPIN))
    if len(spin) and run.verdict == 'SPIN':
        run.spin = event(yaw_t[spin[0]])

    def keep(t):
        return (t >= run.t0 - 3.0) & (t <= run.t_end + 2.0)

    t_mask = keep(v_t)
    z_mask = keep(yaw_t)
    z_idx = np.flatnonzero(z_mask)[::4]  # 約 50 Hz
    i_mask = keep(imu[:, 0])
    o_mask = keep(odom[:, 0])
    run.series = {
        'speed_wheel_mps': np.c_[v_t[t_mask], v_w[t_mask]],
        'yaw_rate_wheel_radps': np.c_[v_t[t_mask], yaw_wheel[t_mask]],
        'accel_wheel_mps2': np.c_[v_t[t_mask], a_w[t_mask]],
        'accel_imu_mps2': np.c_[imu[i_mask, 0], imu[i_mask, 1] - bias_ax],
        'lateral_accel_imu_mps2': np.c_[imu[i_mask, 0], imu[i_mask, 2] - bias_ay],
        'yaw_rate_radps': np.c_[yaw_t[z_idx], yaw[z_idx]],
        'yaw_rate_smooth_radps': np.c_[yaw_t[z_idx], yaw_s[z_idx]],
        'heading_imu_deg': np.c_[yaw_t[z_idx], np.degrees(heading[z_idx])],
        'heading_odom_deg': np.c_[odom[o_mask, 0], np.degrees(odom_th[o_mask])],
        'distance_m': np.c_[odom[o_mask, 0], dist[o_mask]],
        'speed_odom_mps': np.c_[odom[o_mask, 0], odom[o_mask, 4]],
        'speed_cmd_mps': np.c_[mux[:, 0], mux[:, 1]],
        'yaw_rate_cmd_radps': np.c_[mux[:, 0], mux[:, 2]],
    }
    return run


def phase_at(run: Run, t: float) -> str:
    if not run.moved or t < run.t0:
        return '待機'
    if t > run.t_end:
        return '停止'
    if run.t_cut and t >= run.t_cut:
        return '指令切断後'
    if t < run.t_vmax:
        return '加速'
    if run.kind == 'turn' and t >= run.t_turn:
        return f'旋回 R{run.radius:.0f}'
    return '一定速/減速'


def run_info(run: Run) -> str:
    cond = f'{run.kind_jp} {run.title} / 判定 {run.verdict} / {run.note}'
    if run.kind == 'turn':
        cond += f' / 指令の横加速度 {run.a_lat:.2f} m/s²'
    if run.onset:
        cond += f' / 揺れ始め: 発進から {run.onset.s:.1f} m, 速度 {run.onset.v:.2f} m/s, 加速度 {run.onset.a:+.2f} m/s²'
    if run.spin:
        cond += (f' / スピン地点: 発進から {run.spin.s:.1f} m ({run.spin.x:.1f}, {run.spin.y:.1f}), '
                 f'速度 {run.spin.v:.2f} m/s, 加速度 {run.spin.a:+.2f} m/s², ヨー {run.spin.yaw:+.2f} rad/s')
    return cond


def analysis_messages(run: Run):
    """/analysis/* を (時刻ns, トピック, 型, CDR) で返す."""
    out = []
    if not run.moved:
        t_ns = int(run.odom[0, 0] * 1e9)
        out.append((t_ns, '/analysis/run_info', STRING, serialize(make(STRING, data=run_info(run)), STRING)))
        return out

    for name, arr in run.series.items():
        topic = f'/analysis/{name}'
        for t, val in arr:
            out.append((int(t * 1e9), topic, FLOAT64, serialize(make(FLOAT64, data=float(val)), FLOAT64)))
    diff_t = run.series['heading_odom_deg'][:, 0]
    diff = run.series['heading_odom_deg'][:, 1] - np.interp(diff_t, *run.series['heading_imu_deg'].T)
    for t, val in zip(diff_t[::5], diff[::5]):
        out.append((int(t * 1e9), '/analysis/heading_odom_minus_imu_deg', FLOAT64, serialize(make(FLOAT64, data=float(val)), FLOAT64)))
    if run.kind == 'turn':
        turn_t = run.series['speed_cmd_mps'][:, 0]
        mask = (turn_t >= run.t_turn) & (turn_t <= (run.t_cut or run.t_end))
        for t in turn_t[mask]:
            out.append((int(t * 1e9), '/analysis/radius_cmd_m', FLOAT64, serialize(make(FLOAT64, data=float(run.radius)), FLOAT64)))

    # 1 秒ごとに同じものを出す (Foxglove でどこへシークしても走行線と条件が出るように)
    t_begin, t_stop = run.odom[0, 0], run.odom[-1, 0]
    x0, y0, th0 = run.start_pose
    c, s = math.cos(-th0), math.sin(-th0)
    tx, ty = -(c * x0 - s * y0), -(s * x0 + c * y0)
    half = -th0 / 2.0
    quat = TS.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
    info = serialize(make(STRING, data=run_info(run)), STRING)
    for t in np.arange(t_begin, t_stop + 1.0, 1.0):
        t_ns = int(t * 1e9)
        out.append((t_ns, '/analysis/run_info', STRING, info))
        out.append((t_ns, '/analysis/markers', MARKER, serialize(run_markers(run, t_ns), MARKER)))
    # run_start -> odom. 発進時の odom を原点・向き 0 に直す枠 (元の odom は変えない)
    for t in np.arange(t_begin, t_stop + 0.5, 0.2):
        t_ns = int(t * 1e9)
        tf = TS.types['geometry_msgs/msg/TransformStamped'](
            header=header(t_ns, 'run_start'), child_frame_id='odom',
            transform=TS.types['geometry_msgs/msg/Transform'](translation=vec3(tx, ty, 0.0), rotation=quat))
        out.append((t_ns, '/analysis/tf_run_start', TF, serialize(TS.types[TF](transforms=[tf]), TF)))

    # 再生中の状態を 10 Hz の文字で出す
    v_t, v = run.series['speed_wheel_mps'].T
    a_t, a = run.series['accel_wheel_mps2'].T
    y_t, y = run.series['yaw_rate_radps'].T
    d_t, d = run.series['distance_m'].T
    for t in np.arange(run.t0 - 1.0, run.t_end + 1.0, 0.1):
        text = (f'{phase_at(run, t)} | 距離 {np.interp(t, d_t, d):5.1f} m | 速度 {np.interp(t, v_t, v):4.2f} m/s '
                f'({np.interp(t, v_t, v) * 3.6:4.1f} km/h) | 加速度 {np.interp(t, a_t, a):+5.2f} m/s² | ヨー {np.interp(t, y_t, y):+5.2f} rad/s')
        out.append((int(t * 1e9), '/analysis/state', STRING, serialize(make(STRING, data=text), STRING)))
    return out


def run_markers(run: Run, t_ns: int):
    frame = 'run_start'
    markers = [line_marker(t_ns, frame, 'path', 0, run.path_xy, run.path_yaw),
               sphere_marker(t_ns, frame, 'start', 0, 0, 0, (0.3, 0.6, 1.0)),
               text_marker(t_ns, frame, 'start', 1, 0, 1.2, f'START {run.title} [{run.verdict}]', (0.5, 0.8, 1.0))]
    if run.kind == 'turn':
        x = float(np.interp(run.t_turn, run.odom[:, 0], rx_of(run)))
        y = float(np.interp(run.t_turn, run.odom[:, 0], ry_of(run)))
        markers += [sphere_marker(t_ns, frame, 'turn_in', 0, x, y, (0.3, 0.9, 0.9), 0.45), text_marker(t_ns, frame, 'turn_in', 1, x, y + 1.0, f'turn-in R{run.radius:.0f}', (0.5, 1, 1), 0.5)]
    if run.onset:
        e = run.onset
        markers += [sphere_marker(t_ns, frame, 'onset', 0, e.x, e.y, YELLOW, 0.45), text_marker(t_ns, frame, 'onset', 1, e.x, e.y - 1.0, f'onset s={e.s:.1f}m v={e.v:.2f} a={e.a:+.2f}', YELLOW, 0.5)]
    if run.spin:
        e = run.spin
        markers += [sphere_marker(t_ns, frame, 'spin', 0, e.x, e.y, RED, 0.7), text_marker(t_ns, frame, 'spin', 1, e.x, e.y + 1.8, f'SPIN s={e.s:.1f}m v={e.v:.2f}m/s a={e.a:+.2f} yaw={e.yaw:+.2f}', RED, 0.6)]
    return TS.types[MARKER](markers=markers)


def rx_of(run):
    x0, y0, th0 = run.start_pose
    return math.cos(-th0) * (run.odom[:, 1] - x0) - math.sin(-th0) * (run.odom[:, 2] - y0)


def ry_of(run):
    x0, y0, th0 = run.start_pose
    return math.sin(-th0) * (run.odom[:, 1] - x0) + math.cos(-th0) * (run.odom[:, 2] - y0)


def camera_messages(run: Run, channel: str):
    video = run.session / 'video'
    mp4, stamps_csv = video / 'camera.mp4', video / 'camera_stamps.csv'
    if not mp4.exists() or not stamps_csv.exists():
        return
    stamps = load_stamps(str(stamps_csv))
    cap = cv2.VideoCapture(str(mp4))
    idx = 0
    while idx < len(stamps):
        ok, frame = cap.read()
        if not ok:
            break
        sec, nanosec = stamps[idx]
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield sec * 1_000_000_000 + nanosec, channel, 'sensor_msgs/msg/CompressedImage', encode_cdr_compressed_image(
            sec, nanosec, 'zed_left_camera_optical_frame', 'jpeg', jpeg.tobytes())
        idx += 1
    cap.release()


class McapOut:
    def __init__(self, path: Path):
        self.f = open(path, 'wb')
        self.w = Writer(self.f)
        self.w.start()
        self.schemas: Dict[str, int] = {}
        self.channels: Dict[str, int] = {}

    def channel(self, topic: str, msgtype: str) -> int:
        if topic not in self.channels:
            if msgtype not in self.schemas:
                self.schemas[msgtype] = self.w.register_schema(name=msgtype, encoding='ros2msg', data=clean_schema(msgtype))
            self.channels[topic] = self.w.register_channel(topic=topic, message_encoding='cdr', schema_id=self.schemas[msgtype])
        return self.channels[topic]

    def add(self, t_ns: int, topic: str, msgtype: str, data: bytes):
        self.w.add_message(channel_id=self.channel(topic, msgtype), log_time=t_ns, publish_time=t_ns, data=data)

    def close(self):
        self.w.finish()
        self.f.close()


def write_run(run: Run, out_path: Path) -> int:
    out = McapOut(out_path)
    count = 0
    with Reader(str(run.session / 'data')) as reader:
        def bag():
            for conn, ts, raw in reader.messages():
                yield ts, conn.topic, conn.msgtype, bytes(raw)
        extra = sorted(analysis_messages(run), key=lambda m: m[0])
        for t_ns, topic, msgtype, data in heapq.merge(bag(), iter(extra), camera_messages(run, '/camera/image/compressed'), key=lambda m: m[0]):
            out.add(t_ns, topic, msgtype, data)
            count += 1
    out.close()
    return count


def overview_layout(runs: List[Run]):
    """(トピック, frame, Run, x, y, ラベル) に並べる."""
    placed = []
    accel = sorted([r for r in runs if r.kind == 'accel'], key=lambda r: (r.kmh, r.accel, r.tag))
    for i, r in enumerate(accel):
        placed.append(('/overview/accel', 'overview_accel', r, 0.0, -14.0 * i, f'{r.title} [{r.verdict}] {r.tag}'))
    brake = sorted([r for r in runs if r.kind == 'brake'], key=lambda r: r.decel)
    for i, r in enumerate(brake):
        placed.append(('/overview/brake', 'overview_brake', r, 0.0, -14.0 * i, f'{r.title} [{r.verdict}] {r.tag}'))
    # 旋回は 速度 × 半径 の表に並べる
    turns = [r for r in runs if r.kind == 'turn']
    speeds = sorted({r.kmh for r in turns})
    radii = sorted({r.radius for r in turns})
    seen: Dict[tuple, int] = {}
    for r in sorted(turns, key=lambda r: r.tag):
        key = (r.kmh, r.radius)
        n = seen.get(key, 0)
        seen[key] = n + 1
        placed.append(('/overview/turn', 'overview_turn', r, 45.0 * speeds.index(r.kmh), -36.0 * radii.index(r.radius) - 12.0 * n,
                       f'{r.title} [{r.verdict}] a_lat={r.a_lat:.2f} {r.tag}'))
    return placed, speeds, radii


def write_overview(runs: List[Run], out_path: Path):
    moved = [r for r in runs if r.moved]
    placed, speeds, radii = overview_layout(moved)
    t_start = min(r.t0 for r in moved)
    out = McapOut(out_path)
    table = index_markdown(runs)
    for sec in range(15):
        t_ns = int((t_start + sec) * 1e9)
        by_topic: Dict[str, list] = {}
        for i, (topic, frame, r, ox, oy, label) in enumerate(placed):
            color = VERDICT_COLOR[r.verdict]
            ms = by_topic.setdefault(topic, [])
            ms.append(line_marker(t_ns, frame, 'path', i, r.path_xy + (ox, oy), r.path_yaw, width=0.25))
            ms.append(sphere_marker(t_ns, frame, 'start', i, ox, oy, (0.3, 0.6, 1.0), 0.9))
            ms.append(text_marker(t_ns, frame, 'label', i, ox + 1.0, oy - 3.5, label, color, 1.6))
            if r.spin:
                ms.append(sphere_marker(t_ns, frame, 'spin', i, r.spin.x + ox, r.spin.y + oy, RED, 1.2))
        for topic, ms in by_topic.items():
            out.add(t_ns, topic, MARKER, serialize(TS.types[MARKER](markers=ms), MARKER))
        out.add(t_ns, '/overview/table', STRING, serialize(make(STRING, data=table), STRING))
        tfs = [TS.types['geometry_msgs/msg/TransformStamped'](
            header=header(t_ns, 'overview_root'), child_frame_id=f,
            transform=TS.types['geometry_msgs/msg/Transform'](translation=vec3(0, 0, 0), rotation=TS.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0)))
            for f in ('overview_accel', 'overview_brake', 'overview_turn')]
        out.add(t_ns, '/overview/tf', TF, serialize(TS.types[TF](transforms=tfs), TF))
    out.close()


def fmt_event(e: Optional[Event]) -> str:
    return f'{e.s:.1f} m | {e.v:.2f} | {e.a:+.2f}' if e else '- | - | -'


def cut_note(run: Run) -> str:
    """スピン地点が指令の切断より後か."""
    if not (run.spin and run.t_cut):
        return '-'
    return f'{run.spin.t - run.t_cut:+.2f} s'


def index_markdown(runs: List[Run]) -> str:
    lines = ['# 2026-10-27 実験の一覧', '',
             '判定は実験者の目視. 距離は発進からの odom の走行距離 (回をまたいだ同じ場所ではない). 速度は車輪速, 加速度は車輪速の微分 (平滑化あり).',
             f'揺れ始めは {YAW_ONSET_SMOOTH} 秒平均の |ヨー| > {YAW_ONSET} rad/s (旋回は 0.2 秒平均のヨーが指令より {YAW_ONSET} rad/s 多い時点). スピン地点は目視でスピンした回の |ヨー| > {YAW_SPIN} rad/s の時点.', '',
             '## 加速実験', '',
             '| rosbag | 速度 km/h | 加速度 指令 | 判定 | 揺れ始め: 距離 | 速度 m/s | 加速度 m/s² | スピン地点: 距離 | 速度 m/s | 加速度 m/s² | 切断からの時間 | メモ |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted((r for r in runs if r.kind == 'accel'), key=lambda r: (r.kmh, r.accel, r.tag)):
        lines.append(f'| {r.tag} | {r.kmh:.0f} | {r.accel:.1f} | {r.verdict} | {fmt_event(r.onset)} | {fmt_event(r.spin)} | {cut_note(r)} | {r.note} |')
    lines += ['', '## 減速実験', '', '| rosbag | 速度 km/h | 減速度 指令 | 判定 | ヨー最大 rad/s | メモ |', '|---|---|---|---|---|---|']
    for r in sorted((r for r in runs if r.kind == 'brake'), key=lambda r: r.decel):
        lines.append(f'| {r.tag} | {r.kmh:.0f} | {r.decel:.1f} | {r.verdict} | {r.yaw_max:.2f} | {r.note} |')
    lines += ['', '## 旋回実験 (全て左旋回)', '', '| rosbag | 速度 km/h | 半径 m | 横加速度 指令 | 判定 | 揺れ始め: 距離 | 速度 m/s | 加速度 m/s² | スピン地点: 距離 | 速度 m/s | 加速度 m/s² | 切断からの時間 | メモ |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted((r for r in runs if r.kind == 'turn'), key=lambda r: (r.kmh, r.radius, r.tag)):
        lines.append(f'| {r.tag} | {r.kmh:.0f} | {r.radius:.0f} | {r.a_lat:.2f} | {r.verdict} | {fmt_event(r.onset)} | {fmt_event(r.spin)} | {cut_note(r)} | {r.note} |')
    lines += ['', '## odom の確認', '', '| rosbag | 発進時 x | y | 向き° | 原点から始まるか | 向きの変化: odom° | IMU積分° | 差° |', '|---|---|---|---|---|---|---|---|']
    for r in sorted(runs, key=lambda r: r.tag):
        x, y, th = r.start_pose
        zero = abs(x) < 0.05 and abs(y) < 0.05 and abs(math.degrees(th)) < 1.0
        lines.append(f'| {r.tag} | {x:.2f} | {y:.2f} | {math.degrees(th):.1f} | {"はい" if zero else "いいえ"} | '
                     f'{r.odom_heading_end:.1f} | {r.heading_end:.1f} | {r.odom_heading_end - r.heading_end:+.1f} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='2026-10-27 の実験 rosbag を Foxglove 用 MCAP にする')
    parser.add_argument('tags', nargs='*', help='時刻 (例 150926). 省略時は全回')
    parser.add_argument('--overview', action='store_true', help='1 回ごとの MCAP を作らず, 一覧だけ作る')
    args = parser.parse_args()
    MCAP_DIR.mkdir(parents=True, exist_ok=True)

    tags = args.tags or list(RUNS)
    all_runs: List[Run] = []
    for tag in (list(RUNS) if not args.tags or args.overview else tags):
        run = analyze(tag)
        all_runs.append(run)
        if args.overview or tag not in tags:
            continue
        path = MCAP_DIR / f'20261027_{tag}_gamepad.mcap'
        count = write_run(run, path)
        print(f'{path}  {count} msgs  {path.stat().st_size / 1e6:.0f} MB  {run_info(run)}', flush=True)
    if len(all_runs) == len(RUNS):
        write_overview(all_runs, MCAP_DIR / '20261027_overview.mcap')
        (MCAP_DIR / '20261027_index.md').write_text(index_markdown(all_runs), encoding='utf-8')
        with open(MCAP_DIR / '20261027_index.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['rosbag', '種類', '速度km/h', '加速度', '減速度', '半径m', '判定', 'メモ', 'スピン距離m', 'スピン速度m/s', 'スピン加速度', 'スピンヨー'])
            for r in all_runs:
                e = r.spin
                w.writerow([r.tag, r.kind_jp, r.kmh, r.accel, r.decel, r.radius, r.verdict, r.note,
                            *([f'{e.s:.2f}', f'{e.v:.2f}', f'{e.a:.2f}', f'{e.yaw:.2f}'] if e else ['', '', '', ''])])
        print(f'{MCAP_DIR}/20261027_overview.mcap / 20261027_index.md / 20261027_index.csv')


if __name__ == '__main__':
    main()
