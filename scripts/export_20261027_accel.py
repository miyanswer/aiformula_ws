#!/usr/bin/env python3
"""加速実験の図のもとになる値を convert_20261027_experiments.analyze から取り出して npz に保存する.

    .venv/bin/python scripts/export_20261027_accel.py
    python3 scripts/plot_20261027_accel.py     (matplotlib は .venv にないのでシステムの python3)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_20261027_experiments import MCAP_DIR, RUNS, analyze  # noqa: E402

OUT = MCAP_DIR / 'report_20261027_accel'
SWITCH_WINDOW = 0.25  # [s] 減速への切替から この時間以内の揺れ始めを「切替付近」と呼ぶ


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    for tag in (t for t, r in RUNS.items() if r[0] == 'accel'):
        run = analyze(tag)
        cmd_t, cmd_v = run.series['speed_cmd_mps'].T
        wheel_t, wheel_v = run.series['speed_wheel_mps'].T
        imu_t, imu_a = run.series['accel_imu_mps2'].T
        yaw_t, yaw_s = run.series['yaw_rate_smooth_radps'].T
        head_t, head = run.series['heading_imu_deg'].T
        tv = run.t_vmax
        end = run.t_cut or cmd_t[(cmd_t >= tv) & (np.abs(cmd_v) > 0.02)][-1]
        falling = np.flatnonzero((cmd_t > tv) & (cmd_t <= end) & (cmd_v < run.v_max - 0.01))
        t_switch = cmd_t[falling[0]] if len(falling) else end

        # 実加速度: 車輪速が最高速の 20% から 80% になるまでの平均
        w = (wheel_t >= run.t0) & (wheel_t <= tv + 1.0)
        tw, vw = wheel_t[w], wheel_v[w]
        i20, i80 = np.argmax(vw > 0.2 * run.v_max), np.argmax(vw > 0.8 * run.v_max)
        i = (imu_t >= tw[i20]) & (imu_t <= tw[i80])

        phase = '-'
        if run.onset:
            t = run.onset.t
            phase = ('加速中' if t < tv else '切替付近' if abs(t - t_switch) <= SWITCH_WINDOW
                     else '一定速' if t < t_switch else '減速中')

        o = run.odom[(run.odom[:, 0] >= run.t0) & (run.odom[:, 0] <= run.t_end), 0]
        arrays[f'{tag}_t'] = yaw_t - tv
        arrays[f'{tag}_yaw'] = yaw_s
        arrays[f'{tag}_cmd_t'] = run.series['yaw_rate_cmd_radps'][:, 0] - tv
        arrays[f'{tag}_cmd'] = run.series['yaw_rate_cmd_radps'][:, 1]
        arrays[f'{tag}_xy'] = run.path_xy
        arrays[f'{tag}_path_yaw'] = np.abs(np.interp(o, yaw_t, yaw_s))
        arrays[f'{tag}_path_t'] = o - tv
        rows.append({
            'tag': tag, 'kmh': run.kmh, 'accel': run.accel, 'verdict': run.verdict, 'note': run.note,
            'a_wheel': float((vw[i80] - vw[i20]) / (tw[i80] - tw[i20])), 'a_imu': float(imu_a[i].mean()),
            'heading_vmax': float(np.interp(tv, head_t, head)), 'heading_end': run.heading_end,
            'phase': phase, 'onset_s': run.onset.s if run.onset else None,
            'onset_t': run.onset.t - tv if run.onset else None,
            'spin_t': run.spin.t - tv if run.spin else None, 'spin_s': run.spin.s if run.spin else None,
            'onset_v': run.onset.v if run.onset else None, 'spin_v': run.spin.v if run.spin else None,
            't_start': run.t0 - tv, 't_switch': float(t_switch - tv), 't_stop': float(end - tv),
            'cut': bool(run.t_cut), 's_switch': float(np.interp(t_switch, *run.series['distance_m'].T)),
        })
    np.savez_compressed(OUT / 'accel_data.npz', **arrays)
    (OUT / 'accel_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in sorted(rows, key=lambda r: (r['kmh'], r['accel'], r['tag'])):
        print(r['tag'], r['kmh'], r['accel'], r['verdict'], f"{r['a_wheel']:.2f} {r['a_imu']:.2f} {r['heading_vmax']:+.1f} {r['heading_end']:+.0f}",
              r['phase'], r['onset_s'] and round(r['onset_s'], 1), round(r['s_switch'], 1), round(r['t_switch'], 2))


if __name__ == '__main__':
    main()
