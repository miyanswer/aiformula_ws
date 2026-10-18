#!/usr/bin/env python3
"""
diagnose_rosbag_runs.py

全走行セッション (rosbag/2026*/*) のトピックデータを走査し、
「何が起きていたか」「どこで異常・失敗が発生したか」を時系列かつ定量的に診断・集計する。
"""

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores

ts_store = get_typestore(Stores.ROS2_HUMBLE)


def diagnose_session(session_path: str):
    session_p = Path(session_path).resolve()
    data_dir = session_p / "data"
    if not (data_dir / "metadata.yaml").exists():
        subdirs = [d for d in session_p.iterdir() if d.is_dir() and (d / "data" / "metadata.yaml").exists()]
        if subdirs:
            session_p = subdirs[0]
            data_dir = session_p / "data"
        else:
            return None

    name = f"{session_p.parent.name}/{session_p.name}"
    
    # 統計用コンテナ
    phase_counter = Counter()
    total_planner_msgs = 0
    lost_events = [] # (start_time, end_time, max_lost_time)
    reseed_events = [] # (time_sec, val)
    lateral_rejected_count = 0
    tl_reasons = set()
    tl_states = Counter()
    
    current_lost_start = None
    last_lost_time = 0.0
    
    first_ts = None
    last_ts = None
    last_planner_state = None
    last_cmd_vel = None
    last_actual_speed = None
    
    # 車両速度・停止検知
    speed_samples = []
    
    with Reader(str(data_dir)) as reader:
        first_ts = reader.start_time
        last_ts = reader.end_time
        duration_sec = (last_ts - first_ts) / 1e9 if last_ts and first_ts else 0.0

        for conn, timestamp, rawdata in reader.messages():
            t_rel = (timestamp - first_ts) / 1e9

            if conn.topic == "/aiformula_control/six_lane_planner/status":
                total_planner_msgs += 1
                msg = ts_store.deserialize_cdr(rawdata, conn.msgtype)
                try:
                    state = json.loads(msg.data)
                    phase = state.get("phase", "UNKNOWN")
                    phase_counter[phase] += 1
                    last_planner_state = state
                    
                    lost_time = state.get("lost_time", 0.0)
                    if phase == "LOST" or lost_time > 0.5:
                        if current_lost_start is None:
                            current_lost_start = t_rel
                        last_lost_time = max(last_lost_time, lost_time)
                    else:
                        if current_lost_start is not None:
                            lost_events.append((current_lost_start, t_rel, last_lost_time))
                            current_lost_start = None
                            last_lost_time = 0.0
                            
                    if state.get("lateral_rejected", False):
                        lateral_rejected_count += 1
                except Exception:
                    pass

            elif conn.topic == "/aiformula_control/six_lane_planner/lane_reseed":
                msg = ts_store.deserialize_cdr(rawdata, conn.msgtype)
                reseed_events.append((t_rel, msg.data))

            elif conn.topic == "/aiformula_control/traffic_light_stop/status":
                msg = ts_store.deserialize_cdr(rawdata, conn.msgtype)
                try:
                    tl_data = json.loads(msg.data)
                    st = tl_data.get("state", "UNKNOWN")
                    tl_states[st] += 1
                    if "reason" in tl_data:
                        tl_reasons.add(tl_data["reason"])
                except Exception:
                    pass

            elif conn.topic == "/aiformula_control/twist_mux/cmd_vel":
                msg = ts_store.deserialize_cdr(rawdata, conn.msgtype)
                last_cmd_vel = (msg.linear.x, msg.angular.z)
                speed_samples.append((t_rel, msg.linear.x))

            elif conn.topic == "/aiformula_sensing/gyro_odometry_publisher/odom":
                msg = ts_store.deserialize_cdr(rawdata, conn.msgtype)
                last_actual_speed = msg.twist.twist.linear.x

    if current_lost_start is not None:
        lost_events.append((current_lost_start, duration_sec, last_lost_time))

    return {
        "name": name,
        "duration_sec": duration_sec,
        "total_planner_msgs": total_planner_msgs,
        "phase_distribution": dict(phase_counter),
        "reseed_count": len(reseed_events),
        "reseed_events": reseed_events[:10], # 先頭10件
        "lateral_rejected_count": lateral_rejected_count,
        "lost_events": lost_events,
        "tl_states": dict(tl_states),
        "tl_reasons": list(tl_reasons)[:5],
        "last_cmd_vel": last_cmd_vel,
        "last_actual_speed": last_actual_speed,
        "last_planner_state": last_planner_state,
    }


def main():
    sessions = sorted(glob.glob("rosbag/2026*/*"))
    results = []
    for s in sessions:
        if not os.path.isdir(s): continue
        res = diagnose_session(s)
        if res:
            results.append(res)

    print("\n" + "="*80)
    print("           全走行セッション 診断レポート (DIAGNOSTIC REPORT)")
    print("="*80)

    for r in results:
        print(f"\n▶ セッション: 【 {r['name']} 】 (走行時間: {r['duration_sec']:.1f} 秒)")
        print(f"  ・プランナーPhase内訳: {r['phase_distribution']}")
        print(f"  ・白線リシード (Lane Reseed) 発生回数: {r['reseed_count']} 回")
        print(f"  ・白線誤検知・取り違え棄却 (Lateral Rejected): {r['lateral_rejected_count']} 回")
        
        # 白線ロストイベント
        if r["lost_events"]:
            print(f"  ・白線ロスト (LOST) 発生回数: {len(r['lost_events'])} 回")
            for idx, (st, en, max_l) in enumerate(r["lost_events"]):
                print(f"      [{idx+1}] {st:.1f}s 〜 {en:.1f}s (継続: {en-st:.1f}s, 最大ロスト累積: {max_l:.1f}s)")
        else:
            print("  ・白線ロスト: なし (良好)")

        # 信号機認識・停止
        if r["tl_states"]:
            print(f"  ・信号停止制御状態: {r['tl_states']}")
            if r["tl_reasons"]:
                print(f"      主な理由: {r['tl_reasons']}")

        # 最終状態
        print(f"  ・走行終了時の制御指令: {r['last_cmd_vel']}")
        print(f"  ・走行終了時の実測車速: {r['last_actual_speed']:.2f} m/s" if r['last_actual_speed'] is not None else "  ・走行終了時の実測車速: N/A")
        
        if r["last_planner_state"]:
            last_exp = r["last_planner_state"].get("explain", [])
            print(f"  ・走行終了時のプランナー判断理由: {last_exp}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
