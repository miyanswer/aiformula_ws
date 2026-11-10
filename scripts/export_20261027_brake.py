#!/usr/bin/env python3
"""減速実験の図のもとになる値を convert_20261027_experiments.analyze から取り出して保存する.

    .venv/bin/python scripts/export_20261027_brake.py
    python3 scripts/plot_20261027_brake.py     (matplotlib は .venv にないのでシステムの python3)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_20261027_experiments import MCAP_DIR, RUNS, analyze  # noqa: E402

OUT = MCAP_DIR / 'report_20261027_brake'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    for tag in (t for t, r in RUNS.items() if r[0] == 'brake'):
        run = analyze(tag)
        cmd_t, cmd_v = run.series['speed_cmd_mps'].T
        wheel_t, wheel_v = run.series['speed_wheel_mps'].T
        imu_t, imu_a = run.series['accel_imu_mps2'].T
        yaw_t, yaw_s = run.series['yaw_rate_smooth_radps'].T
        head_t, head = run.series['heading_imu_deg'].T
        dist_t, dist = run.series['distance_m'].T
        tv = run.t_vmax
        in_cmd = (cmd_t >= run.t0) & (np.abs(cmd_v) > 0.02)
        end = cmd_t[in_cmd][-1]
        falling = np.flatnonzero((cmd_t > tv) & (cmd_t <= end) & (cmd_v < run.v_max - 0.01))
        ts = cmd_t[falling[0]] if len(falling) else end

        # 指令の減速度: 指令が最高速の 80% から 20% へ落ちる間の傾き
        q = (cmd_t >= ts) & (cmd_t <= end) & (cmd_v < 0.8 * run.v_max) & (cmd_v > 0.2 * run.v_max)
        d_cmd = -np.polyfit(cmd_t[q], cmd_v[q], 1)[0]
        # 実減速度: 車輪速が最高速の 80% から 20% へ落ちる間の平均 (加速実験の実加速度と同じ区間)
        w = (wheel_t >= ts) & (wheel_t <= ts + 4.0)
        tw, vw = wheel_t[w], wheel_v[w]
        i80, i20 = np.argmax(vw < 0.8 * run.v_max), np.argmax(vw < 0.2 * run.v_max)
        i = (imu_t >= tw[i80]) & (imu_t <= tw[i20])

        stop = run.t_end
        o = run.odom[(run.odom[:, 0] >= run.t0) & (run.odom[:, 0] <= run.t_end), 0]
        arrays[f'{tag}_t'] = yaw_t - tv
        arrays[f'{tag}_yaw'] = yaw_s
        arrays[f'{tag}_cmd_t'] = run.series['yaw_rate_cmd_radps'][:, 0] - tv
        arrays[f'{tag}_cmd'] = run.series['yaw_rate_cmd_radps'][:, 1]
        arrays[f'{tag}_xy'] = run.path_xy
        arrays[f'{tag}_path_t'] = o - tv
        arrays[f'{tag}_wheel_t'] = wheel_t - tv
        arrays[f'{tag}_wheel'] = wheel_v
        arrays[f'{tag}_cmd_v_t'] = cmd_t - tv
        arrays[f'{tag}_cmd_v'] = cmd_v
        s_brake = float(np.interp(ts, dist_t, dist))
        s_end = float(np.interp(stop, dist_t, dist))
        m_all = (yaw_t >= run.t0) & (yaw_t <= stop)
        m_brake = (yaw_t >= ts) & (yaw_t <= stop)
        h_end = float(np.interp(stop, head_t, head))
        h_ts = float(np.interp(ts, head_t, head))
        rows.append({
            'tag': tag, 'kmh': run.kmh, 'accel': run.accel, 'decel': run.decel, 'verdict': run.verdict, 'note': run.note,
            'd_cmd': float(d_cmd), 'd_wheel': float((vw[i80] - vw[i20]) / (tw[i20] - tw[i80])), 'd_imu': float(-imu_a[i].mean()),
            'v_brake': float(np.interp(ts, wheel_t, wheel_v)), 'stop_dist': s_end - s_brake, 'stop_time': float(stop - ts),
            's_brake': s_brake, 's_end': s_end, 's_vmax': float(np.interp(tv, dist_t, dist)),
            'heading_brake_start': h_ts, 'heading_end': h_end, 'heading_in_brake': h_end - h_ts,
            'yaw_max': float(np.abs(yaw_s[m_all]).max()), 'yaw_max_brake': float(np.abs(yaw_s[m_brake]).max()),
            'onset_s': run.onset.s if run.onset else None, 'onset_t': run.onset.t - tv if run.onset else None,
            'onset_v': run.onset.v if run.onset else None, 't_start': run.t0 - tv, 't_switch': float(ts - tv),
            't_cmd_end': float(end - tv), 't_stop': float(stop - tv), 'cut': bool(run.t_cut),
        })
    np.savez_compressed(OUT / 'brake_data.npz', **arrays)
    (OUT / 'brake_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in sorted(rows, key=lambda r: r['decel']):
        print(r['tag'], r['decel'], f"cmd={r['d_cmd']:.2f} wheel={r['d_wheel']:.2f} imu={r['d_imu']:.2f} stop={r['stop_dist']:.2f}m {r['stop_time']:.2f}s v={r['v_brake']:.2f} "
              f"head brake_start={r['heading_brake_start']:+.1f} end={r['heading_end']:+.1f} yawmax={r['yaw_max']:.2f} brk={r['yaw_max_brake']:.2f} onset={r['onset_s']} cut={r['cut']} ts={r['t_switch']:.2f} cmdend={r['t_cmd_end']:.2f} stop_t={r['t_stop']:.2f}")


if __name__ == '__main__':
    main()
