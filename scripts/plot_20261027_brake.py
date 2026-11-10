#!/usr/bin/env python3
"""減速実験の結果の表・グラフ・走行線を PNG にする. 先に export_20261027_brake.py を実行する.

    python3 scripts/plot_20261027_brake.py

出力: rosbag/mcap/report_20261027_brake/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path('rosbag/mcap/report_20261027_brake')
SURFACE, INK, INK2, GRID, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#f0efec'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GOOD = '#0ca30c'
ONSET = 0.3  # [rad/s] 揺れ始めのしきい値 (convert_20261027_experiments.YAW_ONSET)
PHASES = {'加速': BLUE, '一定速': AQUA, '減速': ORANGE}
SWITCH_LABEL = '減速への切替'

plt.rcParams.update({
    'font.family': 'Hiragino Sans', 'font.size': 11, 'axes.unicode_minus': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
})

rows = sorted(json.loads((OUT / 'brake_rows.json').read_text(encoding='utf-8')), key=lambda r: r['decel'])
data = np.load(OUT / 'brake_data.npz')


def title(fig, main, sub=None, y=0.985):
    fig.text(0.02, y, main, fontsize=16, fontweight='bold', va='top')
    if sub:
        fig.text(0.02, y - 0.042 * 10 / fig.get_figheight(), sub, fontsize=10.5, color=INK2, va='top')


def save(fig, name):
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    print(OUT / name)


def verdict_mark(ax, x, y, ms=10, **kw):
    ax.plot(x, y, marker='o', color=GOOD, ms=ms, mec='white', mew=0.6, **kw)


def calc_dist(r):
    return r['v_brake'] ** 2 / (2 * r['decel'])


def fig_table():
    cols = [('rosbag', 0.012, 'l'), ('速度\nkm/h', 0.085, 'c'), ('加速度\n(固定)', 0.14, 'c'), ('指定\n減速度', 0.2, 'c'), ('実減速度\n車輪', 0.265, 'c'), ('実減速度\nIMU', 0.335, 'c'),
            ('停止距離\n実測', 0.41, 'c'), ('停止距離\n計算', 0.485, 'c'), ('減速中の\n向き変化', 0.56, 'c'), ('最大ヨー\nrad/s', 0.635, 'c'), ('揺れ始め', 0.7, 'c'), ('判定', 0.775, 'c'), ('実験者のメモ', 0.835, 'l')]
    n = len(rows)
    fig = plt.figure(figsize=(15, 0.56 * n + 3.3))
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.64])
    ax.set_xlim(0, 1), ax.set_ylim(n + 1.1, 0), ax.axis('off')
    title(fig, '減速実験の結果 (2026-10-27)', '加速 1.0 → 1.5 秒の一定速 → 指定の減速度で停止. 減速度の単位は m/s². 向きは + が左, − が右. 判定は実験者の目視.', y=0.97)
    fig.text(0.02, 0.84, f'揺れ始めの基準: 0.2 秒平均のヨーレートの大きさが {ONSET} rad/s を超えた最初の時点 (ZED IMU, 発進前の静止でゼロ点を補正). 全 4 回で超えなかった.',
             fontsize=10.5, fontweight='bold', va='top')
    for text, x, ha in cols:
        ax.text(x, 0.55, text, ha={'l': 'left', 'c': 'center'}[ha], va='center', fontsize=10.5, color=INK2, fontweight='bold')
    ax.plot([0, 1], [1.1, 1.1], color=INK2, lw=1)
    for i, r in enumerate(rows):
        y = i + 1.65
        vals = [r['tag'], f"{r['kmh']:.0f}", f"{r['accel']:.1f}", f"{r['decel']:.1f}", f"{r['d_wheel']:.2f}", f"{r['d_imu']:.2f}", f"{r['stop_dist']:.2f} m", f"{calc_dist(r):.2f} m",
                f"{r['heading_in_brake']:+.1f}°", f"{r['yaw_max']:.2f}", 'なし', None, r['note'].replace(', ', ',\n')]
        for (_, x, ha), v in zip(cols, vals):
            if v is not None:
                ax.text(x, y, v, ha={'l': 'left', 'c': 'center'}[ha], va='center', fontsize=10.5)
        verdict_mark(ax, 0.742, y)
        ax.text(0.752, y, 'スピンなし', va='center', fontsize=10.5)
        ax.plot([0, 1], [y + 0.5, y + 0.5], color=GRID, lw=0.6)
    fig.text(0.012, 0.015, '実減速度: 車輪速が最高速の 80% から 20% になるまでの平均. 停止距離 実測: 減速の指令に切り替わってから車輪速が止まるまでの距離 (車輪速の積分).\n'
             '停止距離 計算: 切替時の速度 v と指定の減速度 d から v²/(2d). 遅れを含まないので実測より短い. IMU は減速中の車体の沈み込みの影響を受ける可能性がある.', fontsize=9.5, color=INK2)
    save(fig, '01_table.png')


def fig_actual():
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.19)
    title(fig, '指定 5.0 では車輪の実減速度が 4.46 m/s² で, 指定に届かない',
          '1 列が 1 回の実験で, 線で結んだ車輪と IMU は同じ回の値. 黒い横線が指定した減速度で, それより下なら届いていない.\n'
          '実減速度は車輪速が最高速の 80% から 20% になるまでの平均 (約 12〜3 km/h の間). 全回 15 km/h からの減速.')
    for x, r in enumerate(rows):
        ax.plot([x - 0.3, x + 0.3], [r['decel'], r['decel']], color=INK, lw=2.4, solid_capstyle='butt', zorder=1)
        ax.text(x, max(r['decel'], r['d_wheel'], r['d_imu']) + 0.12, f"指定 {r['decel']:.1f}", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        ax.plot([x, x], [r['d_wheel'], r['d_imu']], color=INK2, lw=1.6, zorder=2, solid_capstyle='butt')
        ax.scatter(x, r['d_wheel'], s=120, facecolor=SURFACE, edgecolor=BLUE, marker='o', lw=2.2, zorder=3)
        ax.scatter(x, r['d_imu'], s=60, color=ORANGE, marker='s', edgecolor=SURFACE, lw=0.8, zorder=4)
    ax.set_xticks(range(len(rows)), [r['tag'] for r in rows]), ax.tick_params(axis='x', length=0), ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.6, len(rows) - 0.4), ax.set_ylim(2.5, 5.5), ax.set_ylabel('減速度 [m/s²]')
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.4, label='指定した減速度'),
                        Line2D([], [], marker='o', mfc=SURFACE, mec=BLUE, mew=2.2, ls='', ms=10, label='車輪の実減速度'),
                        Line2D([], [], marker='s', color=ORANGE, ls='', ms=8, label='IMU の実減速度'),
                        Line2D([], [], color=INK2, lw=1.6, label='同じ回の車輪と IMU')],
               frameon=False, ncol=4, loc='lower right', bbox_to_anchor=(0.98, 0.01), fontsize=10.5)
    save(fig, '02_actual_decel.png')


def fig_map():
    """速度 × 指定減速度のマス目. 実験していない速度も未実施として示す."""
    speeds = [20, 15, 10]
    decels = [r['decel'] for r in rows]
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0.01, 0.1, 0.98, 0.68])
    ax.axis('off')
    title(fig, '15 km/h からの減速は, 指定 5.0 まで全回スピンなし',
          '各マスは「その速度から, その減速度で止まった」回. 加速度は全回 1.0. 試したのは 15 km/h の各 1 回だけで, 10・20 km/h は未実施.')
    fig.legend(handles=[Line2D([], [], marker='o', color=GOOD, ls='', ms=10, label='スピンなし')], frameon=False, loc='upper left', bbox_to_anchor=(0.012, 0.9), fontsize=11)
    label_w, col_w = 1.3, 1.0
    h = {20: 0.75, 15: 1.75, 10: 0.75}
    head = 0.55
    ax.set_xlim(0, label_w + col_w * len(decels)), ax.set_ylim(head + sum(h.values()), 0)
    ax.text(0.05, head / 2, '減速する範囲', va='center', fontsize=11, fontweight='bold', color=INK2)
    for k, d in enumerate(decels):
        ax.text(label_w + (k + 0.5) * col_w, head / 2, f'指定 {d:.1f} m/s²', ha='center', va='center', fontsize=11, fontweight='bold', color=INK2)
    ax.plot([0, label_w + col_w * len(decels)], [head, head], color=INK2, lw=1)
    y = head
    for sp in speeds:
        ax.text(0.05, y + h[sp] / 2 - 0.13, f'{sp} km/h → 0', va='center', fontsize=14, fontweight='bold')
        ax.text(0.05, y + h[sp] / 2 + 0.16, f'({sp / 3.6:.2f} m/s)', va='center', fontsize=10, color=INK2)
        for k, r in enumerate(rows):
            x0 = label_w + k * col_w
            if sp != r['kmh']:
                ax.text(x0 + col_w / 2, y + h[sp] / 2, '未実施', ha='center', va='center', fontsize=10, color='#9a9994')
                continue
            ax.fill_between([x0 + 0.015, x0 + col_w - 0.015], y + 0.03, y + h[sp] - 0.03, color='#f3f9f3', lw=0)
            ax.text(x0 + col_w / 2, y + 0.22, 'スピン 0 / 1 回', ha='center', va='center', fontsize=10.5, fontweight='bold')
            verdict_mark(ax, x0 + 0.13, y + 0.55, ms=11)
            ax.text(x0 + 0.25, y + 0.55, r['tag'], va='center', fontsize=11)
            for n, line in enumerate((f"実 {r['d_wheel']:.2f} m/s²", f"停止距離 {r['stop_dist']:.2f} m", f"向き変化 {r['heading_in_brake']:+.1f}°", f"最大ヨー {r['yaw_max']:.2f} rad/s")):
                ax.text(x0 + 0.1, y + 0.82 + 0.21 * n, line, va='center', fontsize=10, color=INK2)
        y += h[sp]
        ax.plot([0, label_w + col_w * len(decels)], [y, y], color=GRID, lw=1)
    fig.text(0.012, 0.03, '実: 車輪の実減速度. 向き変化: 減速の指令に切り替わってから止まるまで. 判定は実験者の目視. 各条件 1 回なので, 限界の値としては使えない.', fontsize=10, color=INK2)
    save(fig, '03_verdict_map.png')


def phase_bands(ax, r):
    ax.axvspan(r['t_start'], 0, color='#e6effa', lw=0, zorder=0)
    ax.axvspan(r['t_switch'], r['t_stop'], color=BAND, lw=0, zorder=0)
    ax.axvline(r['t_switch'], color=INK2, lw=1, zorder=1)


def panel_label(ax, r):
    verdict_mark(ax, 0.012, 1.08, ms=8, transform=ax.transAxes, clip_on=False)
    ax.text(0.045, 1.08, f"指定 {r['decel']:.1f} m/s²", transform=ax.transAxes, va='center', fontsize=11.5, fontweight='bold')
    ax.text(1.0, 1.08, r['tag'], transform=ax.transAxes, va='center', ha='right', fontsize=9.5, color=INK2)


def fig_time():
    """上段に速度, 下段にヨーレート. 4 回を時間をそろえて並べる."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey='row', gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.08, hspace=0.18, wspace=0.05)
    title(fig, 'ヨーは全回 0.3 rad/s 未満. 指定 5.0 だけ, 右への指令に反して左へ 0.19 rad/s 回る', '上段: 速度 (灰色が指令, 黒が車輪速). 下段: 0.2 秒平均のヨーレート (灰色が指令, 黒が実測). + が左, − が右. 時刻 0 が最高速の指令に達した時点.', y=0.985)
    for k, r in enumerate(rows):
        tag = r['tag']
        top, bot = axes[0, k], axes[1, k]
        for ax in (top, bot):
            phase_bands(ax, r)
            ax.set_xlim(-5, 4)
        top.plot(data[f'{tag}_cmd_v_t'], data[f'{tag}_cmd_v'], color='#9a9994', lw=1.4)
        top.plot(data[f'{tag}_wheel_t'], data[f'{tag}_wheel'], color=INK, lw=1.2)
        top.set_ylim(-0.3, 4.9)
        panel_label(top, r)
        bot.axhspan(-ONSET, ONSET, color='#e9f6e9', alpha=0.7, lw=0, zorder=1)
        bot.plot(data[f'{tag}_cmd_t'], data[f'{tag}_cmd'], color='#9a9994', lw=1.3, zorder=2)
        bot.plot(data[f'{tag}_t'], data[f'{tag}_yaw'], color=INK, lw=1.6, zorder=3)
        bot.set_ylim(-0.6, 0.6)
        bot.set_xlabel('最高速の指令に達してからの時間 [s]')
    axes[0, 0].set_ylabel('速度 [m/s]'), axes[1, 0].set_ylabel('ヨー [rad/s]')
    fig.legend(handles=[Patch(color='#e6effa', label='加速'), Patch(color=SURFACE, ec=GRID, label='一定速'), Patch(color=BAND, label='減速'),
                        Patch(color='#e9f6e9', label='揺れ始め未満 (±0.3)'), Line2D([], [], color=INK, lw=1.6, label='実測'), Line2D([], [], color='#9a9994', lw=1.4, label='指令')],
               loc='upper left', bbox_to_anchor=(0.015, 0.915), ncol=6, frameon=False, fontsize=10.5)
    save(fig, '04_speed_yaw_timeline.png')


def fig_paths():
    fig, axes = plt.subplots(4, 1, figsize=(13, 10.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.06, hspace=0.3)
    title(fig, '全 4 回の走行線: 3.0〜4.0 はほぼ真っ直ぐ, 5.0 は減速中に左へ 7.5° 向きが変わる',
          '発進点を原点, 発進時の向きを右向きにそろえた odom の線 (上が左). 色は指令の段階で, 実際の速度は指令より約 0.15 秒遅れる.', y=0.985)
    for ax, r in zip(axes, rows):
        tag = r['tag']
        xy, pt = data[f'{tag}_xy'], data[f'{tag}_path_t']
        t_mid = 0.5 * (pt[:-1] + pt[1:])
        names = np.where(t_mid < 0, '加速', np.where(t_mid < r['t_switch'], '一定速', '減速'))
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=[PHASES[n] for n in names], linewidths=4, capstyle='round', zorder=2))
        ax.plot(0, 0, marker='o', color=INK, mec=SURFACE, ms=7, zorder=4)
        ax.set_xlim(-1, 24), ax.set_ylim(-1.5, 1.5), ax.set_aspect('equal')
        verdict_mark(ax, 0.01, 1.14, ms=9, transform=ax.transAxes, clip_on=False)
        ax.text(0.025, 1.14, f"15 km/h → 0  指定 {r['decel']:.1f} m/s²  スピンなし", transform=ax.transAxes, va='center', fontsize=11.5, fontweight='bold')
        ax.text(1.0, 1.14, tag, transform=ax.transAxes, va='center', ha='right', fontsize=9.5, color=INK2)
        y = 1.0
        for name, a, b in (('加速', 0.0, r['s_vmax']), ('一定速', r['s_vmax'], r['s_brake']), ('減速', r['s_brake'], r['s_end'])):
            ax.plot([19.5, 20.1], [y, y], color=PHASES[name], lw=4, solid_capstyle='round')
            ax.text(20.3, y, f'{name}  {a:.1f}〜{b:.1f} m', va='center', fontsize=10)
            y -= 0.95
        ax.set_ylabel('左 [m]')
    axes[-1].set_xlabel('前方 [m]')
    fig.legend(handles=[Line2D([], [], color=c, lw=4, label=n) for n, c in PHASES.items()] + [Line2D([], [], marker='o', color=INK, ls='', ms=7, label='発進')],
               loc='upper left', bbox_to_anchor=(0.012, 0.935), ncol=4, frameon=False, fontsize=10.5)
    save(fig, '05_paths.png')


for f in (fig_table, fig_actual, fig_map, fig_time, fig_paths):
    f()
