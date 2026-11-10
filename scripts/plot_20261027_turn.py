#!/usr/bin/env python3
"""旋回実験の結果の表・グラフ・走行線を PNG にする. 先に export_20261027_turn.py を実行する.

    python3 scripts/plot_20261027_turn.py

出力: rosbag/mcap/report_20261027_turn/*.png
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

OUT = Path('rosbag/mcap/report_20261027_turn')
SURFACE, INK, INK2, GRID, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#f0efec'
BLUE, ORANGE, AQUA, VIOLET, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#6b6a66'
GOOD, WARN, CRIT = '#0ca30c', '#fab219', '#d03b3b'
VERDICT = {'OK': ('スピンなし', GOOD, 'o'), 'SPIN': ('スピン', CRIT, 'X')}
PHASES = {'加速': BLUE, '一定速': AQUA, '旋回': VIOLET, '減速': ORANGE, '自動中止後': GRAY}
ONSET = 0.3  # [rad/s] 揺れ始めのしきい値 (convert_20261027_experiments.YAW_ONSET)

plt.rcParams.update({
    'font.family': 'Hiragino Sans', 'font.size': 11, 'axes.unicode_minus': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
})

rows = sorted(json.loads((OUT / 'turn_rows.json').read_text(encoding='utf-8')), key=lambda r: (r['kmh'], r['radius'], r['tag']))
data = np.load(OUT / 'turn_data.npz')


def title(fig, main, sub=None, y=0.985):
    fig.text(0.02, y, main, fontsize=16, fontweight='bold', va='top')
    if sub:
        fig.text(0.02, y - 0.042 * 10 / fig.get_figheight(), sub, fontsize=10.5, color=INK2, va='top')


def save(fig, name):
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    print(OUT / name)


def fig_table():
    cols = [('rosbag', 0.012, 'l'), ('速度\nkm/h', 0.078, 'c'), ('半径\nm', 0.125, 'c'), ('指定\n横加速度', 0.178, 'c'), ('指令\nヨー', 0.236, 'c'), ('実ヨー\nIMU', 0.29, 'c'), ('実ヨー\n車輪', 0.344, 'c'),
            ('実 ÷ 指令\n(IMU)', 0.403, 'c'), ('旋回で\n回った角度', 0.47, 'c'), ('揺れ始め\n切り込みから', 0.548, 'c'), ('判定', 0.63, 'c'), ('最後の\n向き変化', 0.71, 'c'), ('実験者のメモ', 0.755, 'l')]
    n = len(rows)
    fig = plt.figure(figsize=(15.5, 0.44 * n + 3.1))
    ax = fig.add_axes([0.01, 0.075, 0.98, 0.76])
    ax.set_xlim(0, 1), ax.set_ylim(n + 1.1, 0), ax.axis('off')
    title(fig, '旋回実験の結果 (2026-10-27, 全て左旋回)', '加速 1.0 → 1.5 秒の一定速 → 指定の半径で切り込む. ヨーの単位は rad/s, 横加速度は m/s². 向きは + が左. 判定は実験者の目視.')
    fig.text(0.02, 0.888, f'揺れ始めの基準: 0.2 秒平均のヨーレートが指令より {ONSET} rad/s 多くなった最初の時点 (ZED IMU, 発進前の静止でゼロ点を補正). 「-」は超えなかった回. 目視の判定とは別に決めた値.',
             fontsize=10.5, fontweight='bold', va='top')
    for text, x, ha in cols:
        ax.text(x, 0.55, text, ha={'l': 'left', 'c': 'center'}[ha], va='center', fontsize=10, color=INK2, fontweight='bold')
    ax.plot([0, 1], [1.1, 1.1], color=INK2, lw=1)
    for i, r in enumerate(rows):
        y = i + 1.65
        if r['verdict'] == 'SPIN':
            ax.fill_between([0, 1], y - 0.5, y + 0.5, color='#fbeaea', lw=0)
        label, color, marker = VERDICT[r['verdict']]
        vals = [r['tag'], f"{r['kmh']:.0f}", f"{r['radius']:.0f}", f"{r['a_lat']:.2f}", f"{r['omega']:.2f}", f"{r['yaw_imu']:.2f}", f"{r['yaw_wheel']:.2f}", f"{r['yaw_imu'] / r['omega']:.2f}",
                f"{r['turned']:.0f}°", f"{r['onset_t']:.2f} s" if r['onset_t'] is not None else '-', None, f"{r['heading_end']:+.0f}°", r['note']]
        for (_, x, ha), v in zip(cols, vals):
            if v is not None:
                ax.text(x, y, v, ha={'l': 'left', 'c': 'center'}[ha], va='center', fontsize=10.5)
        ax.plot([0.598], [y], marker=marker, color=color, ms=9, mec='white', mew=0.6)
        ax.text(0.608, y, label, va='center', fontsize=10.5, fontweight='bold' if r['verdict'] == 'SPIN' else 'normal')
        ax.plot([0, 1], [y + 0.5, y + 0.5], color=GRID, lw=0.6)
    fig.text(0.012, 0.015, '実ヨー: 切り込みの 0.3 秒後から旋回の終わりまでの平均. 旋回で回った角度: 切り込みから, 旋回の指令が終わるまで (5 km/h 以外は自動中止で 90° の前に切れた).\n'
             'コード変更後: 90° 回ったあとに戻して 1.5 秒直進する実験コードに変えた後の回. 回った角度が 90° に届く前に切れているので, この変更の部分は走っていない.', fontsize=9.5, color=INK2)
    save(fig, '01_table.png')


def fig_actual():
    fig, ax = plt.subplots(figsize=(15, 7.6))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.78, bottom=0.17)
    title(fig, '10 km/h 以上では, 実ヨーが指令の 1.3〜3.4 倍になる',
          '1 列が 1 回の実験で, 線で結んだ車輪と IMU は同じ回の値. 黒い横線が指令のヨーで, それより上なら指令より多く回っている. 列の上の数字は IMU の実ヨー ÷ 指令.\n'
          '切り込みの 0.3 秒後から旋回の終わりまでの平均. 車輪は左右の車輪の差 ÷ 0.6 m から計算した値で, 実ヨーとほぼ同じなのでタイヤが大きく滑っている様子はない.')
    x, ticks, labels = 0.0, [], []
    for sp in sorted({r['kmh'] for r in rows}):
        rs = [r for r in rows if r['kmh'] == sp]
        x_first = x
        for r in rs:
            ax.plot([x - 0.36, x + 0.36], [r['omega'], r['omega']], color=INK, lw=2.4, solid_capstyle='butt', zorder=1)
            ax.plot([x, x], [r['yaw_wheel'], r['yaw_imu']], color=INK2, lw=1.6, zorder=2)
            ax.scatter(x, r['yaw_wheel'], s=110, facecolor=SURFACE, edgecolor=BLUE, lw=2.1, zorder=3)
            ax.scatter(x, r['yaw_imu'], s=55, color=ORANGE, marker='s', edgecolor=SURFACE, lw=0.8, zorder=4)
            top = max(r['omega'], r['yaw_wheel'], r['yaw_imu'])
            ax.text(x, top + 0.04, f"{r['yaw_imu'] / r['omega']:.2f} 倍", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
            ticks.append(x), labels.append(f"R{r['radius']:.0f}\n{r['tag']}")
            x += 1
        ax.text(x_first - 0.4, 0.98, f'0 → {sp:.0f} km/h', fontsize=13, fontweight='bold', va='bottom')
        if sp != 15:
            ax.axvline(x + 0.25, color=GRID, lw=1)
        x += 1.5
    ax.set_xlim(-0.8, x - 2.0), ax.set_ylim(0, 0.97)
    ax.set_xticks(ticks, labels, fontsize=9), ax.tick_params(axis='x', length=0), ax.grid(axis='x', visible=False)
    ax.set_ylabel('ヨーレート [rad/s]')
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.4, label='指令のヨー'), Line2D([], [], marker='o', mfc=SURFACE, mec=BLUE, mew=2.1, ls='', ms=10, label='車輪の左右差から計算したヨー'),
                        Line2D([], [], marker='s', color=ORANGE, ls='', ms=8, label='IMU の実ヨー'), Line2D([], [], color=INK2, lw=1.6, label='同じ回の車輪と IMU')],
               frameon=False, ncol=4, loc='lower right', bbox_to_anchor=(0.985, 0.005), fontsize=10.5)
    save(fig, '02_actual_yaw.png')


def fig_map():
    speeds = [15, 14, 13, 10, 5]
    radii = sorted({r['radius'] for r in rows})
    heights = {sp: max(1, max(sum(r['kmh'] == sp and r['radius'] == a for r in rows) for a in radii)) for sp in speeds}
    CELL_H, HEAD_H = 1.0, 0.42
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.76])
    ax.axis('off')
    title(fig, '速度の影響が大きい: 15 km/h は R60 でもスピン, 10 km/h は R5 でもスピンなし',
          '各マスは「0 からその速度まで加速して, その半径で切り込んだ」回. 13 km/h は R14 と R13 の間で結果が変わる. 全て左旋回.')
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=10, label=l) for l, c, m in VERDICT.values()], frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.012, 0.91), fontsize=11)
    label_w, col_w = 1.2, 1.0
    total = sum(heights[sp] * CELL_H + HEAD_H for sp in speeds)
    head = 0.55
    ax.set_xlim(0, label_w + col_w * len(radii)), ax.set_ylim(total + head, 0)
    ax.text(0.05, head / 2, '速度', va='center', fontsize=11, fontweight='bold', color=INK2)
    for k, a in enumerate(radii):
        ax.text(label_w + (k + 0.5) * col_w, head / 2, f'半径 {a:.0f} m', ha='center', va='center', fontsize=11, fontweight='bold', color=INK2)
    ax.plot([0, label_w + col_w * len(radii)], [head, head], color=INK2, lw=1)
    y = head
    for sp in speeds:
        h = heights[sp] * CELL_H + HEAD_H
        ax.text(0.05, y + h / 2 - 0.12, f'0 → {sp} km/h', va='center', fontsize=13, fontweight='bold')
        ax.text(0.05, y + h / 2 + 0.16, f'({sp / 3.6:.2f} m/s)', va='center', fontsize=10, color=INK2)
        for k, a in enumerate(radii):
            x0 = label_w + k * col_w
            same = [r for r in rows if r['kmh'] == sp and r['radius'] == a]
            if not same:
                ax.text(x0 + col_w / 2, y + h / 2, '未実施', ha='center', va='center', fontsize=9.5, color='#9a9994')
                continue
            spins = sum(r['verdict'] == 'SPIN' for r in same)
            ax.fill_between([x0 + 0.015, x0 + col_w - 0.015], y + 0.03, y + h - 0.03, color='#fbeaea' if spins else '#f3f9f3', lw=0)
            ax.text(x0 + col_w / 2, y + HEAD_H / 2 + 0.02, f'スピン {spins} / {len(same)} 回', ha='center', va='center', fontsize=9.5, fontweight='bold')
            for n, r in enumerate(same):
                _, color, marker = VERDICT[r['verdict']]
                yy = y + HEAD_H + n * CELL_H
                ax.plot(x0 + 0.11, yy + 0.2, marker=marker, color=color, ms=10, mec='white', mew=0.6)
                ax.text(x0 + 0.21, yy + 0.2, r['tag'], va='center', fontsize=10)
                ax.text(x0 + 0.07, yy + 0.5, f"横 {r['a_lat']:.2f} m/s²", va='center', fontsize=9, color=INK2)
                ax.text(x0 + 0.07, yy + 0.77, f"実÷指令 {r['yaw_imu'] / r['omega']:.2f}", va='center', fontsize=9, color=INK2)
        y += h
        ax.plot([0, label_w + col_w * len(radii)], [y, y], color=GRID, lw=1)
    fig.text(0.012, 0.025, '横: 指定の横加速度 (速度² ÷ 半径). 実÷指令: IMU の実ヨー ÷ 指令のヨー. 判定は実験者の目視. 5 km/h 以外は全回 90° の前に自動中止で切れており, 「スピンなし」は曲がりきれたという意味ではない. 各条件 1〜2 回.', fontsize=10, color=INK2)
    save(fig, '03_verdict_map.png')


def panel_label(ax, r, size=10.5):
    label, color, marker = VERDICT[r['verdict']]
    ax.plot(0.012, 1.075, marker=marker, color=color, ms=8, mec='white', mew=0.5, transform=ax.transAxes, clip_on=False)
    ax.text(0.04, 1.075, f"0 → {r['kmh']:.0f} km/h  R{r['radius']:.0f}  {label}", transform=ax.transAxes, va='center', fontsize=size, fontweight='bold')
    ax.text(1.0, 1.075, r['tag'], transform=ax.transAxes, va='center', ha='right', fontsize=9, color=INK2)


def fig_yaw():
    fig, axes = plt.subplots(4, 4, figsize=(16, 11.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.865, bottom=0.055, hspace=0.27, wspace=0.05)
    title(fig, '切り込み後に実ヨーが指令を超えて増え, 自動中止のあとも増え続けた回がスピンになる',
          '黒が 0.2 秒平均の実測ヨーレート, 灰色が指令. + が左. 時刻 0 が切り込みの時点. 2.5 rad/s より上は切っている (スピンは最大 7 rad/s).', y=0.99)
    LIM = 2.5
    for ax, r in zip(axes.flat, rows):
        tag = r['tag']
        t, y = data[f'{tag}_t'], data[f'{tag}_yaw']
        ax.axvspan(r['t_vmax'], 0, color='#e7f6f0', lw=0, zorder=0)
        ax.axvspan(r['t_start'], r['t_vmax'], color='#e6effa', lw=0, zorder=0)
        ax.axvspan(0, r['t_turn_end'], color='#ebe9fa', lw=0, zorder=0)
        ax.axvline(r['t_turn_end'], color=INK2, lw=1, zorder=1)
        ax.plot(data[f'{tag}_cmd_t'], np.clip(data[f'{tag}_cmd'], -0.5, LIM), color='#9a9994', lw=1.3, zorder=2)
        ax.plot(t, np.clip(y, -0.5, LIM), color=INK, lw=1.6, zorder=3, solid_capstyle='round')
        if r['onset_t'] is not None:
            ax.plot(r['onset_t'], np.interp(r['onset_t'], t, y), marker='D', color=WARN, mec=INK, mew=0.8, ms=7, zorder=5)
        if r['spin_t'] is not None:
            ax.plot(r['spin_t'], LIM, marker='X', color=CRIT, mec=SURFACE, ms=11, clip_on=False, zorder=6)
        ax.set_xlim(-2.5, 5), ax.set_ylim(-0.5, LIM)
        panel_label(ax, r)
    for ax in axes[-1]:
        ax.set_xlabel('切り込みからの時間 [s]')
    for ax in axes[:, 0]:
        ax.set_ylabel('ヨー [rad/s]')
    fig.legend(handles=[Patch(color='#e6effa', label='加速'), Patch(color='#e7f6f0', label='一定速'), Patch(color='#ebe9fa', label='旋回'), Line2D([], [], color=INK2, lw=1, label='旋回の指令の終わり (自動中止)'),
                        Line2D([], [], color='#9a9994', lw=1.5, label='指令'), Line2D([], [], marker='D', color=WARN, mec=INK, ls='', ms=7, label='揺れ始め'), Line2D([], [], marker='X', color=CRIT, ls='', ms=10, label='スピン地点 (2 rad/s 超)')],
               loc='upper left', bbox_to_anchor=(0.012, 0.945), ncol=7, frameon=False, fontsize=10)
    save(fig, '04_yaw_timeline.png')


def fig_paths():
    fig, axes = plt.subplots(6, 3, figsize=(17, 21), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.925, bottom=0.03, hspace=0.2, wspace=0.03)
    title(fig, '全 16 回の走行線: 指定の半径より内側へ切れ込み, 指令が切れたあとにスピンする',
          '発進点を原点, 発進時の向きを右向きにそろえた odom の線 (上が左). 色は指令の段階で, 実際の動きは指令より遅れる. スピン地点より後は横滑りを含まない参考値なので細線. 点線だけは計算で描いた指定の半径.', y=0.99)
    for ax, r in zip(axes.flat, rows):
        tag = r['tag']
        xy, pt = data[f'{tag}_xy'], data[f'{tag}_path_t']
        t_mid = 0.5 * (pt[:-1] + pt[1:])
        names = np.where(t_mid < r['t_vmax'], '加速', np.where(t_mid < 0, '一定速', np.where(t_mid < r['t_turn_end'], '旋回', np.where(r['cut'], '自動中止後', '減速'))))
        after = t_mid > r['spin_t'] if r['spin_t'] is not None else np.zeros(len(t_mid), bool)
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=[PHASES[n] for n in names], linewidths=np.where(after, 1.3, 3.4), capstyle='round', zorder=3))
        # 指定の半径 (計算). 切り込みの位置と向きから 90° 分
        i = int(np.argmin(np.abs(pt)))
        j = int(np.argmin(np.abs(pt + 0.3)))
        th = np.arctan2(xy[i, 1] - xy[j, 1], xy[i, 0] - xy[j, 0])
        c = xy[i] + r['radius'] * np.array([-np.sin(th), np.cos(th)])
        a = np.linspace(-np.pi / 2, 0, 80) + th
        ax.plot(c[0] + r['radius'] * np.cos(a), c[1] + r['radius'] * np.sin(a), color=INK2, lw=1.1, ls=(0, (2, 2)), zorder=2)
        ax.plot(0, 0, marker='o', color=INK, mec=SURFACE, ms=6, zorder=4)
        at = lambda t: (np.interp(t, pt, xy[:, 0]), np.interp(t, pt, xy[:, 1]))
        if r['onset_t'] is not None:
            ax.plot(*at(r['onset_t']), marker='D', color=WARN, mec=INK, mew=0.8, ms=7, zorder=5)
        if r['spin_t'] is not None:
            ax.plot(*at(r['spin_t']), marker='X', color=CRIT, mec=SURFACE, ms=12, zorder=6)
        ax.set_xlim(-2, 36), ax.set_ylim(-11, 13), ax.set_aspect('equal')
        panel_label(ax, r, 11)
        spans = [('加速', 0.0, r['s_vmax']), ('一定速', r['s_vmax'], r['s_turn']), ('旋回', r['s_turn'], r['s_turn_end'])]
        spans.append(('自動中止後', r['s_turn_end'], r['s_stop']) if r['cut'] else ('減速', r['s_turn_end'], r['s_stop']))
        y = -2.8
        for name, s0, s1 in spans:
            ax.plot([-1.2, 0.4], [y, y], color=PHASES[name], lw=3.2, solid_capstyle='round')
            ax.text(0.9, y, f'{name}  {s0:.1f}〜{s1:.1f} m', va='center', fontsize=9.5)
            y -= 2.4
        y = -2.8
        if r['onset_t'] is not None:
            ax.plot(16.6, y, marker='D', color=WARN, mec=INK, mew=0.8, ms=6)
            ax.text(17.6, y, f"揺れ始め {r['onset_s']:.1f} m, {r['onset_v'] * 3.6:.1f} km/h", va='center', fontsize=9.5)
            y -= 2.4
        if r['spin_t'] is not None:
            ax.plot(16.6, y, marker='X', color=CRIT, mec=SURFACE, ms=9)
            ax.text(17.6, y, f"スピン {r['spin_s']:.1f} m, {r['spin_v'] * 3.6:.1f} km/h", va='center', fontsize=9.5)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    axes[-1, 0].tick_params(labelbottom=True), axes[4, 1].tick_params(labelbottom=True), axes[4, 2].tick_params(labelbottom=True)
    axes[-1, 0].set_xlabel('前方 [m]'), axes[4, 1].set_xlabel('前方 [m]'), axes[4, 2].set_xlabel('前方 [m]')
    for ax in axes[:, 0]:
        ax.set_ylabel('左 [m]')
    fig.legend(handles=[Line2D([], [], color=c, lw=4, label=n) for n, c in PHASES.items()] + [Line2D([], [], color=INK2, lw=1.1, ls=(0, (2, 2)), label='指定の半径 (計算)'), Line2D([], [], marker='o', color=INK, ls='', ms=6, label='発進'),
                Line2D([], [], marker='D', color=WARN, mec=INK, ls='', ms=7, label='揺れ始め'), Line2D([], [], marker='X', color=CRIT, ls='', ms=10, label='スピン地点')],
               loc='upper left', bbox_to_anchor=(0.012, 0.968), ncol=9, frameon=False, fontsize=10.5)
    save(fig, '05_paths.png')


for f in (fig_table, fig_actual, fig_map, fig_yaw, fig_paths):
    f()
