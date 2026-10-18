#!/usr/bin/env python3
"""
convert_bag_to_mcap.py

ROS 2 SQLite3 bag (data_0.db3) と 車載カメラ映像 (camera.mp4) / パネル映像 (panel.mp4) を統合し、
Foxglove Studio で即座に再生可能な単一の MCAP (.mcap) ファイルへ変換するスクリプト。

主な機能:
  - data_0.db3 の全ROS 2トピック (Odometry, IMU, CAN, TF, Path, Markers, Twist等) の取り込み
  - camera.mp4 のフレームを camera_stamps.csv に記録されたROSナノ秒タイムスタンプと完全同期して
    /camera/image/compressed (sensor_msgs/msg/CompressedImage) として書き込み
  - panel.mp4 (HUD映像) が存在する場合、/panel/image/compressed として同様に書き込み
  - Foxglove Studio がメッセージ内容を完全にパースできるよう、ROS 2 Humble のメッセージ定義スキーマ (ros2msg) を自動埋め込み
  - LZ4 / Zstandard 圧縮による高速かつコンパクトなファイル生成
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
from mcap.writer import Writer
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.store import Nodetype
from tqdm import tqdm


TYPETORE = get_typestore(Stores.ROS2_HUMBLE)


# ---------------------------------------------------------------------------
# Custom Message Definitions (for packages not included in standard ROS 2 Humble)
# ---------------------------------------------------------------------------
CUSTOM_MSG_DEFS = {
    "can_msgs/msg/Frame": """\
std_msgs/Header header
uint32 id
bool is_rtr
bool is_extended
bool is_error
uint8 dlc
uint8[8] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""",
    "sensor_msgs/msg/CompressedImage": """\
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""",
    "aiformula_interfaces/msg/LaneLine": """\
bool valid
bool detected
float64 c0
float64 c1
float64 c2
float64 x_min
float64 x_max
float64[] points_x
float64[] points_y
""",
    "aiformula_interfaces/msg/LaneLines": """\
std_msgs/Header header
aiformula_interfaces/LaneLine left
aiformula_interfaces/LaneLine center
aiformula_interfaces/LaneLine right
float64 lane_width_left
float64 lane_width_right
bool reanchored
================================================================================
MSG: aiformula_interfaces/LaneLine
bool valid
bool detected
float64 c0
float64 c1
float64 c2
float64 x_min
float64 x_max
float64[] points_x
float64[] points_y
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""}


def _msg_ref(typename: str) -> str:
    """ネスト型の参照名を ros2msg 形式 (pkg/Type) にする。rosbags は pkg/msg/Type で返すので 'msg' を除く"""
    parts = typename.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        return f"{parts[0]}/{parts[2]}"
    return typename


def generate_ros2_msgdef(typename: str, visited: Optional[Set[str]] = None) -> str:
    """rosbags の typestore から ros2msg 形式のメッセージ定義テキストを再帰的に生成する"""
    if typename in CUSTOM_MSG_DEFS:
        return CUSTOM_MSG_DEFS[typename]

    if visited is None:
        visited = set()
    if typename in visited:
        return ""
    visited.add(typename)

    if typename not in TYPETORE.FIELDDEFS:
        return ""

    _, fields = TYPETORE.FIELDDEFS[typename]
    lines: List[str] = []
    subtypes: List[str] = []

    for name, (ntype, val) in fields:
        if ntype == Nodetype.BASE:
            bname, _ = val
            lines.append(f"{bname} {name}")
        elif ntype == Nodetype.NAME:
            subname = val
            lines.append(f"{_msg_ref(subname)} {name}")
            subtypes.append(subname)
        elif ntype == Nodetype.SEQUENCE:
            subnode, seq_len = val
            sntype, sval = subnode
            if sntype == Nodetype.BASE:
                bname, _ = sval
                lines.append(f"{bname}[] {name}")
            elif sntype == Nodetype.NAME:
                subname = sval
                lines.append(f"{_msg_ref(subname)}[] {name}")
                subtypes.append(subname)
        elif ntype == Nodetype.ARRAY:
            subnode, arr_len = val
            sntype, sval = subnode
            if sntype == Nodetype.BASE:
                bname, _ = sval
                lines.append(f"{bname}[{arr_len}] {name}")
            elif sntype == Nodetype.NAME:
                subname = sval
                lines.append(f"{_msg_ref(subname)}[{arr_len}] {name}")
                subtypes.append(subname)

    content = "\n".join(lines)
    for sub in subtypes:
        subdef = generate_ros2_msgdef(sub, visited)
        if subdef:
            content += f"\n================================================================================\nMSG: {_msg_ref(sub)}\n" + subdef
    return content


def encode_cdr_compressed_image(stamp_sec: int, stamp_nanosec: int, frame_id: str, format_str: str, jpeg_bytes: bytes) -> bytes:
    """
    sensor_msgs/msg/CompressedImage を little-endian CDR バイナリに直接パックする。
    """
    buf = bytearray()
    # CDR Header (little endian, CDR version 1)
    buf.extend(b"\x00\x01\x00\x00")

    # Header.stamp
    buf.extend(stamp_sec.to_bytes(4, byteorder="little", signed=True))
    buf.extend(stamp_nanosec.to_bytes(4, byteorder="little", signed=False))

    # Header.frame_id (CDR string: 4-byte length including NULL + bytes + NULL + alignment)
    fid_bytes = frame_id.encode("utf-8") + b"\x00"
    buf.extend(len(fid_bytes).to_bytes(4, byteorder="little"))
    buf.extend(fid_bytes)
    while len(buf) % 4 != 0:
        buf.append(0)

    # format (CDR string)
    fmt_bytes = format_str.encode("utf-8") + b"\x00"
    buf.extend(len(fmt_bytes).to_bytes(4, byteorder="little"))
    buf.extend(fmt_bytes)
    while len(buf) % 4 != 0:
        buf.append(0)

    # data (CDR uint8 sequence: 4-byte count + raw bytes)
    buf.extend(len(jpeg_bytes).to_bytes(4, byteorder="little"))
    buf.extend(jpeg_bytes)

    return bytes(buf)


def load_stamps(csv_path: str) -> List[Tuple[int, int]]:
    """CSVから各フレームの (stamp_sec, stamp_nanosec) を取得"""
    stamps = []
    if not os.path.exists(csv_path):
        return stamps
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec_float = float(row["stamp_sec"])
            sec = int(sec_float)
            nanosec = int((sec_float - sec) * 1e9)
            stamps.append((sec, nanosec))
    return stamps


def convert_session(session_dir: str, output_mcap: str, jpeg_quality: int = 80):
    session_path = Path(session_dir).resolve()
    print(f"\n==================================================")
    print(f"Target Session: {session_path}")
    print(f"Output MCAP   : {output_mcap}")
    print(f"==================================================")

    # 1. パス検出
    data_dir = session_path / "data"
    if not (data_dir / "metadata.yaml").exists():
        # もし直下に data がない場合、子ディレクトリを探す (例: 20261012_151105/6lane)
        subdirs = [d for d in session_path.iterdir() if d.is_dir() and (d / "data" / "metadata.yaml").exists()]
        if subdirs:
            session_path = subdirs[0]
            data_dir = session_path / "data"
        else:
            raise FileNotFoundError(f"data/metadata.yaml not found in {session_dir}")

    video_dir = session_path / "video"
    cam_mp4 = video_dir / "camera.mp4"
    cam_stamps_csv = video_dir / "camera_stamps.csv"
    panel_mp4 = video_dir / "panel.mp4"
    panel_stamps_csv = video_dir / "panel_stamps.csv"

    # 2. 出力ディレクトリ作成
    out_path = Path(output_mcap).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f_out:
        writer = Writer(f_out)
        writer.start()

        # スキーマとチャンネルの登録マップ (name -> int ID)
        schema_map: Dict[str, int] = {}
        channel_map: Dict[str, int] = {}

        def get_or_register_schema(msgtype: str) -> int:
            if msgtype not in schema_map:
                msgdef = generate_ros2_msgdef(msgtype)
                schema_id = writer.register_schema(
                    name=msgtype,
                    encoding="ros2msg",
                    data=msgdef.encode("utf-8"),
                )
                schema_map[msgtype] = schema_id
            return schema_map[msgtype]

        def get_or_register_channel(topic: str, msgtype: str) -> int:
            if topic not in channel_map:
                schema_id = get_or_register_schema(msgtype)
                channel_id = writer.register_channel(
                    topic=topic,
                    message_encoding="cdr",
                    schema_id=schema_id,
                )
                channel_map[topic] = channel_id
            return channel_map[topic]

        # 3. ROS 2 bag メッセージの読み込みとチャンネル登録
        print("[1/3] Indexing ROS 2 bag messages...")
        bag_reader = Reader(str(data_dir))
        bag_reader.open()

        for topic, info in bag_reader.topics.items():
            get_or_register_channel(topic, info.msgtype)

        # 4. カメラ映像の準備
        cam_stamps = []
        cam_channel_id = None
        if cam_mp4.exists() and cam_stamps_csv.exists():
            cam_stamps = load_stamps(str(cam_stamps_csv))
            cam_channel_id = get_or_register_channel(
                "/camera/image/compressed", "sensor_msgs/msg/CompressedImage"
            )
            print(f"Found camera video: {cam_mp4.name} ({len(cam_stamps)} stamped frames)")

        # 5. パネル映像の準備
        panel_stamps = []
        panel_channel_id = None
        if panel_mp4.exists() and panel_stamps_csv.exists():
            panel_stamps = load_stamps(str(panel_stamps_csv))
            panel_channel_id = get_or_register_channel(
                "/panel/image/compressed", "sensor_msgs/msg/CompressedImage"
            )
            print(f"Found panel video: {panel_mp4.name} ({len(panel_stamps)} stamped frames)")

        # 6. 書き込み処理
        print("[2/3] Converting and merging video + topics into MCAP...")

        # カメラ動画ジェネレータ
        def video_frame_generator(mp4_path, stamps, channel_id, frame_id):
            if channel_id is None:
                return
            cap = cv2.VideoCapture(str(mp4_path))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or idx >= len(stamps):
                    break
                sec, nanosec = stamps[idx]
                ts_ns = sec * 1_000_000_000 + nanosec
                _, jpeg_data = cv2.imencode(".jpg", frame, encode_params)
                cdr_bytes = encode_cdr_compressed_image(
                    sec, nanosec, frame_id, "jpeg", jpeg_data.tobytes()
                )
                yield ts_ns, channel_id, cdr_bytes
                idx += 1
            cap.release()

        # カメラ・パネル・bagメッセージのイテレータを統合
        cam_gen = video_frame_generator(cam_mp4, cam_stamps, cam_channel_id, "zed_left_camera_optical_frame")
        panel_gen = video_frame_generator(panel_mp4, panel_stamps, panel_channel_id, "panel_frame")

        bag_msg_gen = (
            (ts, channel_map[conn.topic], rawdata)
            for conn, ts, rawdata in bag_reader.messages()
        )

        # 3つのジェネレータから先頭要素を取得してマージソート書き込み (Priority Queue/Peek)
        active_gens = []
        for gen in [bag_msg_gen, cam_gen, panel_gen]:
            try:
                first_item = next(gen)
                active_gens.append([first_item, gen])
            except StopIteration:
                pass

        total_msgs = bag_reader.message_count + len(cam_stamps) + len(panel_stamps)
        pbar = tqdm(total=total_msgs, unit="msg", desc="Writing MCAP")

        while active_gens:
            # 最小タイムスタンプを持つジェネレータを選択
            min_idx = 0
            min_ts = active_gens[0][0][0]
            for i in range(1, len(active_gens)):
                if active_gens[i][0][0] < min_ts:
                    min_ts = active_gens[i][0][0]
                    min_idx = i

            ts_ns, ch_id, data = active_gens[min_idx][0]
            writer.add_message(
                channel_id=ch_id,
                log_time=ts_ns,
                data=data,
                publish_time=ts_ns,
            )
            pbar.update(1)

            # 次のアイテムを取得
            try:
                active_gens[min_idx][0] = next(active_gens[min_idx][1])
            except StopIteration:
                active_gens.pop(min_idx)

        pbar.close()
        bag_reader.close()
        writer.finish()

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[3/3] Done! Created MCAP file:")
    print(f"  Path: {out_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"To open in Foxglove Studio:")
    print(f"  open -a Foxglove \"{out_path}\"")


def main():
    parser = argparse.ArgumentParser(description="ROS 2 bag + Video to Foxglove MCAP converter")
    parser.add_argument("session", type=str, nargs="?", default="", help="Path to rosbag session folder (e.g. rosbag/20261012_151105/6lane)")
    parser.add_argument("--all", action="store_true", help="Convert all sessions found in rosbag/ directory")
    parser.add_argument("--output", "-o", type=str, default="", help="Output MCAP file path (default: rosbag/mcap/<session_name>.mcap)")
    parser.add_argument("--quality", "-q", type=int, default=80, help="JPEG quality for video frames (default: 80)")
    parser.add_argument("--open", action="store_true", help="Automatically open generated MCAP in Foxglove Studio")
    args = parser.parse_args()

    if args.all:
        rosbag_dir = Path("rosbag")
        sessions = sorted([
            p.parent.parent for p in rosbag_dir.glob("2026*/*/data/metadata.yaml")
        ])
        if not sessions:
            print("No valid sessions found in rosbag/ directory.")
            return
        print(f"Found {len(sessions)} sessions to convert:")
        for s in sessions:
            print(f"  - {s}")

        output_dir = Path("rosbag/mcap")
        output_dir.mkdir(parents=True, exist_ok=True)
        for s in sessions:
            name = f"{s.parent.name}_{s.name}"
            out_file = str(output_dir / f"{name}.mcap")
            convert_session(str(s), out_file, args.quality)
        print("\nAll sessions successfully converted!")
        return

    if not args.session:
        parser.print_help()
        print("\nError: Please provide a session path or specify --all")
        return

    session_p = Path(args.session).resolve()
    if not args.output:
        name = session_p.name
        if name in ["6lane", "gamepad"]:
            name = f"{session_p.parent.name}_{name}"
        output_dir = Path("rosbag/mcap")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{name}.mcap")

    convert_session(str(session_p), args.output, args.quality)

    if args.open:
        os.system(f'open -a Foxglove "{args.output}"')


if __name__ == "__main__":
    main()
