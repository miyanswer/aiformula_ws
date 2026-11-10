#!/usr/bin/env python3
"""加速実験の結果の表・グラフ・走行線を PNG にする. 先に export_20261027_accel.py を実行する.

    python3 scripts/plot_20261027_accel.py

出力: rosbag/mcap/report_20261027_accel/*.png
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

OUT = Path('rosbag/mcap/report_20261027_accel')
SURFACE, INK, INK2, GRID, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#f0efec'
BLUE, ORANGE = '#2a78d6', '#eb6834'
GOOD, WARN, SERIOUS, CRIT = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'
VERDICT = {'OK': ('スピンなし', GOOD, 'o'), 'DRIFT': ('右よれ', WARN, '^'), 'SPIN': ('スピン', CRIT, 'X')}
ONSET = 0.3    # [rad/s] 揺れ始めのしきい値 (convert_20261027_experiments.YAW_ONSET)
YAW_SPIN = 2.0  # [rad/s] スピン地点のしきい値
YAW_BINS = [(0.0, ONSET, GOOD, '0.3 未満'), (ONSET, 1.0, WARN, '0.3〜1'), (1.0, YAW_SPIN, SERIOUS, '1〜2'), (YAW_SPIN, 99.0, CRIT, '2 以上')]

plt.rcParams.update({
    'font.family': 'Hiragino Sans', 'font.size': 11, 'axes.unicode_minus': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
})

rows = sorted(json.loads((OUT / 'accel_rows.json').read_text(encoding='utf-8')), key=lambda r: (r['kmh'], r['accel'], r['tag']))
data = np.load(OUT / 'accel_data.npz')


def title(fig, main, sub=None, y=0.985):
    fig.text(0.02, y, main, fontsize=16, fontweight='bold', va='top')
    if sub:
        fig.text(0.02, y - 0.042 * 10 / fig.get_figheight(), sub, fontsize=10.5, color=INK2, va='top')


def save(fig, name):
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    print(OUT / name)


def fig_table():
    cols = [('rosbag', 0.012, 'l'), ('速度\nkm/h', 0.098, 'c'), ('指定\n加速度', 0.158, 'c'), ('実加速度\n車輪', 0.222, 'c'), ('実加速度\nIMU', 0.292, 'c'),
            ('最高速までの\n向き変化', 0.378, 'c'), ('揺れ始め\n段階', 0.470, 'c'), ('揺れ始め\n距離', 0.548, 'c'), ('判定', 0.625, 'c'),
            ('最後の\n向き変化', 0.705, 'c'), ('実験者のメモ', 0.752, 'l')]
    n = len(rows)
    fig = plt.figure(figsize=(15, 0.46 * n + 2.9))
    ax = fig.add_axes([0.01, 0.06, 0.98, 0.8])
    ax.set_xlim(0, 1), ax.set_ylim(n + 1.1, 0), ax.axis('off')
    title(fig, '加速実験の結果 (2026-10-27)', '指定加速度で加速 → 1.5 秒の一定速 → 減速 1.0. 加速度の単位は m/s². 向きは + が左, − が右. 判定は実験者の目視.')
    fig.text(0.02, 0.905, f'揺れ始めの基準: 0.2 秒平均のヨーレートの大きさが {ONSET} rad/s を超えた最初の時点 (ZED IMU, 発進前の静止でゼロ点を補正). 「-」は超えなかった回. 目視の判定とは別に決めた値.',
             fontsize=10.5, fontweight='bold', va='top')
    fig.text(0.02, 0.878, '段階: 加速中 = 最高速の指令に達する前 / 切替付近 = 減速への切替から ±0.25 秒以内 / 一定速 = それより前 / 減速中 = それより後',
             fontsize=10.5, color=INK2, va='top')
    for text, x, ha in cols:
        ax.text(x, 0.55, text, ha={'l': 'left', 'c': 'center'}[ha], va='center', fontsize=10.5, color=INK2, fontweight='bold')
    ax.plot([0, 1], [1.1, 1.1], color=INK2, lw=1)
    for i, r in enumerate(rows):
        y = i + 1.65
        if r['verdict'] == 'SPIN':
            ax.fill_between([0, 1], y - 0.5, y + 0.5, color='#fbeaea', lw=0)
        label, color, marker = VERDICT[r['verdict']]
        vals = [r['tag'], f"{r['kmh']:.0f}", f"{r['accel']:.1f}", f"{r['a_wheel']:.2f}", f"{r['a_imu']:.2f}", f"{r['heading_vmax']:+.1f}°",
                r['phase'], f"{r['onset_s']:.1f} m" if r['onset_s'] else '-', None, f"{r['heading_end']:+.0f}°", r['note']]
        for (_, x, ha), v in zip(cols, vals):
            if v is not None:
                ax.text(x, y, v, ha={'l': 'left', 'c': 'center'}[ha], va='center', fontsize=10.5)
        ax.plot([0.585], [y], marker=marker, color=color, ms=9, mec='white', mew=0.6)
        ax.text(0.596, y, label, va='center', fontsize=10.5, fontweight='bold' if r['verdict'] == 'SPIN' else 'normal')
        ax.plot([0, 1], [y + 0.5, y + 0.5], color=GRID, lw=0.6)
    fig.text(0.012, 0.02, '実加速度: 車輪速が最高速の 20% から 80% になるまでの平均. 向きは ZED IMU の積分. 切替付近の回でも, 右へ回り始めるのは基準を超える時点より前で, 切替の 0.2〜0.6 秒前.',
             fontsize=9.5, color=INK2)
    save(fig, '01_table.png')


def fig_actual():
    """1 回を 1 列にして, 同じ回の車輪と IMU を線で結ぶ. 列は目標速度 → 指定加速度の順に並べる."""
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.79, bottom=0.2)
    title(fig, '15 km/h では, 指定 4.5 以上で実加速度が 4.0〜4.5 m/s² で頭打ちになる',
          '1 列が 1 回の実験で, 線で結んだ車輪と IMU は同じ回の値. 黒い横線が指定した加速度で, それより下なら届いていない.\n'
          '実加速度は車輪速が最高速の 20% から 80% になるまでの平均 (15 km/h なら約 3〜12 km/h の間).')
    x = 0.0
    ticks, labels = [], []
    GROUP_GAP, SPEED_GAP = 0.7, 2.6
    y_speed = 5.33
    for sp in (10, 15, 20):
        rs = [r for r in rows if r['kmh'] == sp]
        x_first = x
        for a in sorted({r['accel'] for r in rs}):
            same = [r for r in rs if r['accel'] == a]
            x_group = x
            for r in same:
                ax.plot([x, x], [r['a_wheel'], r['a_imu']], color=INK2, lw=1.6, zorder=2, solid_capstyle='butt')
                ax.scatter(x, r['a_wheel'], s=120, facecolor=SURFACE, edgecolor=BLUE, marker='o', lw=2.2, zorder=3)
                ax.scatter(x, r['a_imu'], s=60, color=ORANGE, marker='s', edgecolor=SURFACE, lw=0.8, zorder=4)
                ticks.append(x), labels.append(r['tag'])
                x += 1
            x_end = x - 1
            ax.plot([x_group - 0.42, x_end + 0.42], [a, a], color=INK, lw=2.4, solid_capstyle='butt', zorder=1)
            top = max([a] + [v for r in same for v in (r['a_wheel'], r['a_imu'])])
            ax.text(0.5 * (x_group + x_end), top + 0.12, f'指定 {a:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            x += GROUP_GAP
        x_last = x - GROUP_GAP - 1
        ax.text(x_first - 0.42, y_speed, f'0 → {sp} km/h', fontsize=13, fontweight='bold', va='bottom')
        if sp == 15:
            ax.fill_between([x_first - 0.5, x_last + 0.5], 4.0, 4.5, color=BAND, lw=0, zorder=0)
            ax.text(x_first - 0.35, 4.25, '4.0〜4.5 m/s²', va='center', color=INK2, fontsize=10.5)
        if sp != 20:
            ax.axvline(x - GROUP_GAP + SPEED_GAP / 2 - 0.5, color=GRID, lw=1)
        x += SPEED_GAP - GROUP_GAP
    ax.set_xlim(-0.8, x - SPEED_GAP + 0.3), ax.set_ylim(1.5, 5.3)
    ax.set_xticks(ticks, labels, rotation=90, fontsize=9.5), ax.tick_params(axis='x', length=0), ax.grid(axis='x', visible=False)
    ax.set_ylabel('加速度 [m/s²]')
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.4, label='指定した加速度'),
                        Line2D([], [], marker='o', mfc=SURFACE, mec=BLUE, mew=2.2, ls='', ms=10, label='車輪の実加速度'),
                        Line2D([], [], marker='s', color=ORANGE, ls='', ms=8, label='IMU の実加速度'),
                        Line2D([], [], color=INK2, lw=1.6, label='同じ回の車輪と IMU')],
               frameon=False, ncol=4, loc='lower right', bbox_to_anchor=(0.985, 0.005), fontsize=10.5)
    save(fig, '02_actual_accel.png')


def fig_map():
    """目標速度 × 指定加速度のマス目. 1 回ごとに判定・rosbag・実加速度・最大ヨーを書く."""
    speeds = [20, 15, 10]
    accels = sorted({r['accel'] for r in rows})
    heights = {sp: max(1, max(sum(r['kmh'] == sp and r['accel'] == a for r in rows) for a in accels)) for sp in speeds}
    CELL_H, HEAD_H = 1.0, 0.42  # 1 回分の高さ, マスの見出しの高さ
    fig = plt.figure(figsize=(16, 8.6))
    ax = fig.add_axes([0.01, 0.075, 0.98, 0.735])
    ax.axis('off')
    title(fig, 'スピンは加速度の大きさの順に並ばない',
          '各マスは「0 からその速度まで, その加速度で加速した」回. 15 km/h は同じ指定の 3.0・3.5 でも結果が分かれ (発進場所は 2〜3 か所にまたがる), 4.0〜4.8 はスピンなし, 4.9・5.0 はスピン. 20 km/h は弱い 2.0・2.5 でもスピン.')
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=10, label=l) for l, c, m in VERDICT.values()], frameon=False, ncol=3,
               loc='upper left', bbox_to_anchor=(0.012, 0.9), fontsize=11)
    label_w, col_w = 1.25, 1.0
    total = sum(heights[sp] * CELL_H + HEAD_H for sp in speeds)
    head = 0.55
    ax.set_xlim(0, label_w + col_w * len(accels)), ax.set_ylim(total + head, 0)
    ax.text(0.05, head / 2, '加速する範囲', va='center', fontsize=11, fontweight='bold', color=INK2)
    for k, a in enumerate(accels):
        ax.text(label_w + (k + 0.5) * col_w, head / 2, f'指定 {a:.1f} m/s²', ha='center', va='center', fontsize=11, fontweight='bold', color=INK2)
    ax.plot([0, label_w + col_w * len(accels)], [head, head], color=INK2, lw=1)
    y = head
    for sp in speeds:
        h = heights[sp] * CELL_H + HEAD_H
        ax.text(0.05, y + h / 2 - 0.13, f'0 → {sp} km/h', va='center', fontsize=14, fontweight='bold')
        ax.text(0.05, y + h / 2 + 0.17, f'({sp / 3.6:.2f} m/s)', va='center', fontsize=10, color=INK2)
        for k, a in enumerate(accels):
            x0 = label_w + k * col_w
            same = [r for r in rows if r['kmh'] == sp and r['accel'] == a]
            spins = sum(r['verdict'] == 'SPIN' for r in same)
            if not same:
                ax.text(x0 + col_w / 2, y + h / 2, '未実施', ha='center', va='center', fontsize=9.5, color='#9a9994')
                continue
            ax.fill_between([x0 + 0.015, x0 + col_w - 0.015], y + 0.03, y + h - 0.03, color='#fbeaea' if spins else '#f3f9f3', lw=0)
            ax.text(x0 + col_w / 2, y + HEAD_H / 2 + 0.02, f'スピン {spins} / {len(same)} 回', ha='center', va='center', fontsize=10, fontweight='bold')
            for n, r in enumerate(same):
                _, color, marker = VERDICT[r['verdict']]
                yy = y + HEAD_H + n * CELL_H
                ax.plot(x0 + 0.12, yy + 0.3, marker=marker, color=color, ms=11, mec='white', mew=0.6)
                ax.text(x0 + 0.24, yy + 0.3, r['tag'], va='center', fontsize=10.5)
                peak = np.abs(data[f"{r['tag']}_yaw"]).max()
                ax.text(x0 + 0.08, yy + 0.68, f"実 {r['a_wheel']:.2f}  ヨー {peak:.1f}", va='center', fontsize=9.5, color=INK2)
        y += h
        ax.plot([0, label_w + col_w * len(accels)], [y, y], color=GRID, lw=1)
    fig.text(0.012, 0.03, '実: 車輪速が最高速の 20% から 80% になるまでの平均加速度 [m/s²]. ヨー: 発進から停止後までの 0.2 秒平均ヨーレートの最大 [rad/s]. 判定は実験者の目視. '
             '各条件の回数は 1〜3 回なので, 限界の値としては使えない.', fontsize=10, color=INK2)
    save(fig, '03_verdict_map.png')


def panel_label(ax, r):
    label, color, marker = VERDICT[r['verdict']]
    ax.plot(0.012, 1.075, marker=marker, color=color, ms=8, mec='white', mew=0.5, transform=ax.transAxes, clip_on=False)
    ax.text(0.04, 1.075, f"{r['kmh']:.0f} km/h  指定 {r['accel']:.1f}  {label}", transform=ax.transAxes, va='center', fontsize=10.5, fontweight='bold')
    ax.text(1.0, 1.075, r['tag'], transform=ax.transAxes, va='center', ha='right', fontsize=9, color=INK2)


def fig_yaw():
    fig, axes = plt.subplots(5, 3, figsize=(14, 13.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.885, bottom=0.05, hspace=0.3, wspace=0.05)
    title(fig, '15 km/h のスピンは, 最高速の直後に左へ振れ, 減速への切替より前に右へ回り始める', '黒が 0.2 秒平均の実測ヨーレート, 灰色が指令. + が左, − が右. 時刻 0 が最高速の指令に達した時点. ±1.5 rad/s より外は切っている (スピンは最大 7 rad/s).', y=0.99)
    LIM = 1.5
    for ax, r in zip(axes.flat, rows):
        tag = r['tag']
        t, y = data[f'{tag}_t'], data[f'{tag}_yaw']
        ax.axvspan(r['t_start'], 0, color='#e6effa', lw=0)
        ax.axvspan(r['t_switch'], r['t_stop'], color=BAND, lw=0)
        ax.axvline(r['t_switch'], color=INK2, lw=1)
        ax.axhspan(-ONSET, ONSET, color='#e9f6e9', alpha=0.7, lw=0, zorder=1)
        ax.plot(data[f'{tag}_cmd_t'], np.clip(data[f'{tag}_cmd'], -LIM, LIM), color='#9a9994', lw=1.3, zorder=2)
        ax.plot(t, np.clip(y, -LIM, LIM), color=INK, lw=1.6, solid_capstyle='round', zorder=3)
        if r['onset_t'] is not None:
            oy = np.interp(r['onset_t'], t, y)
            ax.plot(r['onset_t'], oy, marker='D', color=WARN, mec=INK, mew=0.8, ms=8, zorder=4)
        if r['spin_t'] is not None and r['spin_t'] < 4.5:
            ax.plot(r['spin_t'], np.sign(np.interp(r['spin_t'], t, y)) * LIM, marker='X', color=CRIT, mec=SURFACE, ms=11, clip_on=False, zorder=5)
        ax.set_xlim(-2.8, 4.5), ax.set_ylim(-LIM, LIM)
        panel_label(ax, r)
    for ax in axes[-1]:
        ax.set_xlabel('最高速の指令に達してからの時間 [s]')
    for ax in axes[:, 0]:
        ax.set_ylabel('ヨー [rad/s]')
    fig.legend(handles=[Patch(color='#e6effa', label='加速'), Patch(color=SURFACE, ec=GRID, label='一定速'), Patch(color=BAND, label='減速 (切れた回は自動中止まで)'),
                        Patch(color='#e9f6e9', label='揺れ始め未満 (±0.3)'), Line2D([], [], color='#9a9994', lw=1.5, label='指令'), Line2D([], [], marker='D', color=WARN, mec=INK, ls='', ms=8, label='揺れ始め'),
                        Line2D([], [], marker='X', color=CRIT, ls='', ms=10, label='スピン地点 (2 rad/s 超)')],
               loc='upper left', bbox_to_anchor=(0.015, 0.948), ncol=7, frameon=False, fontsize=10)
    save(fig, '04_yaw_timeline.png')


PHASES = {'加速': BLUE, '一定速': '#1baf7a', '減速': ORANGE, '自動中止後': '#6b6a66'}


def fig_paths():
    """全回の走行線を, 指令の段階 (加速・一定速・減速・自動中止後) で色分けする."""
    fig, axes = plt.subplots(5, 3, figsize=(16, 14), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.885, bottom=0.04, hspace=0.27, wspace=0.03)
    title(fig, '全 15 回の走行線: どの段階でよれ, どこでスピンしたか',
          '発進点を原点, 発進時の向きを右向きにそろえた odom の線 (上が左). 色は指令の段階で, 実際の速度は指令より約 0.15 秒遅れる. スピン地点より後は横滑りを含まない参考値なので細線.', y=0.99)
    for ax, r in zip(axes.flat, rows):
        tag = r['tag']
        xy, pt = data[f'{tag}_xy'], data[f'{tag}_path_t']
        s_path = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        t_mid = 0.5 * (pt[:-1] + pt[1:])
        names = np.where(t_mid < 0, '加速', np.where(t_mid < r['t_switch'], '一定速', np.where((t_mid < r['t_stop']) | (not r['cut']), '減速', '自動中止後')))
        after_spin = t_mid > r['spin_t'] if r['spin_t'] is not None else np.zeros(len(t_mid), bool)
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=[PHASES[n] for n in names], linewidths=np.where(after_spin, 1.4, 3.6), capstyle='round', zorder=2))
        ax.plot(0, 0, marker='o', color=INK, mec=SURFACE, ms=7, zorder=4)
        at = lambda t: (np.interp(t, pt, xy[:, 0]), np.interp(t, pt, xy[:, 1]))
        if r['onset_t'] is not None:
            ax.plot(*at(r['onset_t']), marker='D', color=WARN, mec=INK, mew=0.9, ms=8, zorder=5)
        if r['spin_t'] is not None:
            ax.plot(*at(r['spin_t']), marker='X', color=CRIT, mec=SURFACE, ms=13, zorder=6)
        ax.set_xlim(-1.5, 24), ax.set_ylim(-6, 2.5), ax.set_aspect('equal')

        label, color, marker = VERDICT[r['verdict']]
        ax.plot(0.012, 1.07, marker=marker, color=color, ms=9, mec='white', mew=0.5, transform=ax.transAxes, clip_on=False)
        ax.text(0.035, 1.07, f"0 → {r['kmh']:.0f} km/h  指定 {r['accel']:.1f} m/s²  {label}", transform=ax.transAxes, va='center', fontsize=11.5, fontweight='bold')
        ax.text(1.0, 1.07, tag, transform=ax.transAxes, va='center', ha='right', fontsize=9.5, color=INK2)

        # 段階の範囲を距離で書く
        dist = lambda t: float(np.interp(t, pt, s_path))
        spans = [('加速', dist(0)), ('一定速', dist(r['t_switch']))]
        if r['t_stop'] - r['t_switch'] > 0.05 or not r['cut']:
            spans.append(('減速', dist(r['t_stop']) if r['cut'] else s_path[-1]))
        if r['cut']:
            spans.append(('自動中止後', s_path[-1]))
        y, start = -2.75, 0.0
        for name, end in spans:
            ax.plot([-1.1, -0.3], [y, y], color=PHASES[name], lw=3.6, solid_capstyle='round')
            ax.text(0.0, y, f'{name}  {start:.1f}〜{end:.1f} m', va='center', fontsize=9.5)
            y, start = y - 0.85, end
        y = -3.6
        if r['onset_t'] is not None:
            ax.plot(10.7, y, marker='D', color=WARN, mec=INK, mew=0.9, ms=7)
            ax.text(11.3, y, f"揺れ始め {r['onset_s']:.1f} m, {r['onset_v'] * 3.6:.1f} km/h ({r['phase']})", va='center', fontsize=9.5)
            y -= 0.95
        if r['spin_t'] is not None:
            ax.plot(10.7, y, marker='X', color=CRIT, mec=SURFACE, ms=10)
            ax.text(11.3, y, f"スピン {r['spin_s']:.1f} m, {r['spin_v'] * 3.6:.1f} km/h", va='center', fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel('前方 [m]')
    for ax in axes[:, 0]:
        ax.set_ylabel('左 [m]')
    fig.legend(handles=[Line2D([], [], color=c, lw=4, label=n) for n, c in PHASES.items()] +
               [Line2D([], [], marker='o', color=INK, ls='', ms=7, label='発進'),
                Line2D([], [], marker='D', color=WARN, mec=INK, ls='', ms=8, label='揺れ始め (0.2 秒平均のヨー > 0.3 rad/s)'),
                Line2D([], [], marker='X', color=CRIT, ls='', ms=11, label='スピン地点 (ヨー > 2 rad/s)')],
               loc='upper left', bbox_to_anchor=(0.012, 0.952), ncol=7, frameon=False, fontsize=10.5)
    save(fig, '05_paths.png')


for f in (fig_table, fig_actual, fig_map, fig_yaw, fig_paths):
    f()
