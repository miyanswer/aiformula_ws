#!/usr/bin/env python3
"""旋回実験の図のもとになる値を convert_20261027_experiments.analyze から取り出して保存する.

    .venv/bin/python scripts/export_20261027_turn.py
    python3 scripts/plot_20261027_turn.py     (matplotlib は .venv にないのでシステムの python3)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_20261027_experiments import MCAP_DIR, RUNS, analyze  # noqa: E402

OUT = MCAP_DIR / 'report_20261027_turn'
SETTLE = 0.3  # [s] 切り込みの直後は過渡なので, 実ヨーの平均はこの時間のあとから取る


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    for tag in (t for t, r in RUNS.items() if r[0] == 'turn'):
        run = analyze(tag)
        cmd_t, cmd_v = run.series['speed_cmd_mps'].T
        _, cmd_w = run.series['yaw_rate_cmd_radps'].T
        wheel_t, wheel_v = run.series['speed_wheel_mps'].T
        wy_t, wheel_y = run.series['yaw_rate_wheel_radps'].T
        yaw_t, yaw_s = run.series['yaw_rate_smooth_radps'].T
        imu_t, ay = run.series['lateral_accel_imu_mps2'].T
        head_t, head = run.series['heading_imu_deg'].T
        dist_t, dist = run.series['distance_m'].T
        t0, tv, ti = run.t0, run.t_vmax, run.t_turn
        in_cmd = np.flatnonzero((cmd_t >= t0) & (np.abs(cmd_v) > 0.02))
        gaps = np.flatnonzero(np.diff(cmd_t[in_cmd]) > 0.3)
        end = cmd_t[in_cmd[gaps[0]]] if len(gaps) else cmd_t[in_cmd[-1]]
        # 旋回の終わり: 切り込み後, ω が旋回の値から離れる最初の時刻. 走行中に切れた回はそこまで
        off = np.flatnonzero((cmd_t > ti) & (cmd_t <= end) & (np.abs(cmd_w - run.omega_turn) >= 0.003))
        t_turn_end = cmd_t[off[0]] if len(off) else end
        falling = np.flatnonzero((cmd_t > ti) & (cmd_t <= end) & (cmd_v < run.v_max - 0.01))
        t_dec = cmd_t[falling[0]] if len(falling) else end
        cut = bool(run.t_cut)

        a, b = ti + SETTLE, t_turn_end
        mean = lambda t, y: float(y[(t >= a) & (t <= b)].mean()) if ((t >= a) & (t <= b)).any() else float('nan')
        y_imu, y_wheel, v_turn, a_imu = mean(yaw_t, yaw_s), mean(wy_t, wheel_y), mean(wheel_t, wheel_v), mean(imu_t, ay)
        turned = float(np.interp(t_turn_end, head_t, head) - np.interp(ti, head_t, head))

        o = run.odom[(run.odom[:, 0] >= t0) & (run.odom[:, 0] <= run.t_end), 0]
        arrays[f'{tag}_t'] = yaw_t - ti
        arrays[f'{tag}_yaw'] = yaw_s
        arrays[f'{tag}_cmd_t'] = cmd_t - ti
        arrays[f'{tag}_cmd'] = cmd_w
        arrays[f'{tag}_xy'] = run.path_xy
        arrays[f'{tag}_path_t'] = o - ti
        at = lambda t: float(np.interp(t, dist_t, dist))
        rows.append({
            'tag': tag, 'kmh': run.kmh, 'radius': run.radius, 'verdict': run.verdict, 'note': run.note,
            'a_lat': run.a_lat, 'omega': float(run.omega_turn), 'v_turn': v_turn, 'yaw_imu': y_imu, 'yaw_wheel': y_wheel,
            'a_imu': a_imu, 'v_r': v_turn * y_imu, 'window': float(b - a), 'turned': turned,
            'yaw_max': float(np.abs(yaw_s[(yaw_t >= t0) & (yaw_t <= run.t_end)]).max()), 'heading_end': run.heading_end,
            'onset_s': run.onset.s if run.onset else None, 'onset_t': run.onset.t - ti if run.onset else None, 'onset_v': run.onset.v if run.onset else None,
            'spin_s': run.spin.s if run.spin else None, 'spin_t': run.spin.t - ti if run.spin else None, 'spin_v': run.spin.v if run.spin else None,
            't_start': t0 - ti, 't_vmax': tv - ti, 't_turn_end': float(t_turn_end - ti), 't_dec': float(t_dec - ti), 't_end_cmd': float(end - ti), 't_stop': run.t_end - ti,
            'cut': cut, 's_vmax': at(tv), 's_turn': at(ti), 's_turn_end': at(t_turn_end), 's_dec': at(t_dec), 's_cmd_end': at(end), 's_stop': at(run.t_end),
            'code_changed': 'コード変更後' in run.note,
        })
    np.savez_compressed(OUT / 'turn_data.npz', **arrays)
    (OUT / 'turn_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in sorted(rows, key=lambda r: (r['kmh'], r['radius'], r['tag'])):
        print(r['tag'], r['kmh'], r['radius'], r['verdict'], f"ω={r['omega']:.3f} imu={r['yaw_imu']:.3f}({r['yaw_imu'] / r['omega']:.2f}) wheel={r['yaw_wheel']:.3f} v={r['v_turn']:.2f} vr={r['v_r']:.2f} ay={r['a_imu']:.2f} win={r['window']:.2f} 回={r['turned']:.0f}° yawmax={r['yaw_max']:.2f} end={r['heading_end']:.0f} cut={r['cut']} onset_t={r['onset_t']} spin_t={r['spin_t']} tend={r['t_turn_end']:.2f} dec={r['t_dec']:.2f} s_turn={r['s_turn']:.1f} s_stop={r['s_stop']:.1f}")


if __name__ == '__main__':
    main()
