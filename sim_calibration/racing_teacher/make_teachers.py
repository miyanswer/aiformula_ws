#!/usr/bin/env python3
"""機体特性を入れた racing 教師を作って NN を学習する. aiformula_machine の中には何も書かない.

    python3 -B make_teachers.py   (-B: リポジトリに __pycache__ を作らない)

リポジトリの teachers.py / train_policy.py は読むだけ. 結果はこのフォルダの policies/ と datasets/ に置く.
"""
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

SRC = os.path.expanduser('~/aiformula_machine/src/oit_navigation/oit_navigation/six_lane')
sys.path.insert(0, SRC)
import train_policy as tp  # noqa: E402
import teachers as T  # noqa: E402

core = tp.core
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'policies')
DATA = os.path.join(HERE, 'datasets')
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

ACCEL, DECEL = 4.5, 4.0  # 実験で確かめた加速, ユーザー指定の減速の上限 [m/s^2]
KMH = 1 / 3.6

# 半径ごとの曲がれる速度 (複数エージェントで検証した表. 信頼度は低い). 曲率 -> 速度
TABLE_K = np.array([0.0, 1 / 60, 1 / 20, 1 / 15, 1 / 10, 1 / 7, 1 / 5, 1 / 3])
TABLE_V = np.array([11, 11, 10, 10, 9, 9, 8, 7]) * KMH


def v_table(k):
    return np.interp(np.abs(k), TABLE_K, TABLE_V)


def v_alat(a):
    return lambda k: np.sqrt(a / np.maximum(np.abs(k), 1e-3))


def plan(ks, ds, v_max, v_of_k, lat_cost=0.0, inner=1, step_m=2.0, start_lane=None):
    """teachers.optimal_lane_plan と同じ動的計画法. 違いは (1) 曲がれる速度を曲率の関数 v_of_k にした
    (2) 減速を 4.0 にした (3) 横移動 1 m あたり lat_cost [s] の費用 (4) 横位置を内側レーン inner の中心までに制限."""
    kap = T._sample_kappa(ks, ds, step_m)
    m = len(kap)
    DN, DV = T.DN, T.DV
    J = int(math.floor(T.lane_center_offset(inner) / DN + 1e-9))
    n_vals = np.arange(-J, J + 1) * DN
    nj = len(n_vals)
    dmax = int(math.floor(step_m * math.tan(math.radians(22.0)) / DN + 1e-9))
    d_vals = np.arange(-dmax, dmax + 1)
    nd = len(d_vals)
    v_vals = np.linspace(0.0, v_max, int(math.ceil(v_max / DV)) + 1)
    nv = len(v_vals)
    v2 = v_vals ** 2
    C = np.full((nj, nd, nv), np.inf)
    C[nj // 2, dmax, :] = 0.0  # 中央から, どの速度でも始められる
    back = np.zeros((m, nj, nd, nv), dtype=np.int16)
    for i in range(1, m):
        scale = np.maximum(1.0 - n_vals * kap[i - 1], 0.05)
        dist = step_m * scale
        ok = ((v2[None, None, :] <= v2[None, :, None] + 2 * ACCEL * dist[:, None, None] + 1e-9)
              & (v2[None, None, :] >= v2[None, :, None] - 2 * DECEL * dist[:, None, None] - 1e-9))
        avg = np.maximum(0.5 * (v_vals[:, None] + v_vals[None, :]), 0.05)
        t_vv = np.where(ok, dist[:, None, None] / avg[None, :, :], np.inf)
        new = np.full((nj, nd, nv), np.inf)
        for b, dp in enumerate(d_vals):
            npp = (dp - d_vals) * DN / (step_m * step_m)
            kpath = kap[i - 1] / scale[:, None] + npp[None, :]
            v_lat = np.minimum(v_max, v_of_k(kpath))
            cost = 1e-6 * abs(dp) + lat_cost * abs(dp) * DN
            base = np.where(v_vals[None, None, :] <= v_lat[:, :, None] + 1e-9, C, np.inf) + cost
            total = base[:, :, :, None] + t_vv[:, None, :, :]
            flat = total.reshape(nj, nd * nv, nv)
            arg = np.argmin(flat, axis=1)
            best = np.take_along_axis(flat, arg[:, None, :], axis=1)[:, 0, :]
            src = np.arange(nj)
            dst = src + dp
            sel = (dst >= 0) & (dst < nj)
            new[dst[sel], b, :] = best[sel]
            back[i, dst[sel], b, :] = arg[sel]
        C = new
    je = int(np.argmin(np.min(C, axis=(1, 2))))
    b, vi = np.unravel_index(int(np.argmin(C[je])), (nd, nv))
    js = np.zeros(m, dtype=int)
    vs = np.zeros(m)
    js[-1], vs[-1] = je, v_vals[vi]
    for i in range(m - 1, 0, -1):
        prev = int(back[i, js[i], b, vi])
        js[i - 1] = js[i] - d_vals[b]
        b, vi = divmod(prev, nv)
        vs[i - 1] = v_vals[vi]
    n_off = n_vals[js]
    return T.offset_to_lane(n_off), vs, n_off


def dataset(p, n, rng, v_of_k, lat_cost, inner, v_set=(2.5, 2.78, 3.06), lead_s=0.5, sigma=0.6, stay=1.5, stats=None):
    X, Y = [], []
    k = np.arange(1, 7, dtype=float)
    step = 2.0
    cnt = n // len(v_set)
    for v_max in v_set:
        pv = replace(p, v_max=v_max)
        ks, ds = tp.random_course(rng, length=3000.0)
        lanes, speeds, n_off = plan(ks, ds, v_max, v_of_k, lat_cost, inner, step)
        if stats is not None:
            kap = T._sample_kappa(ks, ds, step)
            left = kap > 0.02
            right = kap < -0.02
            flat = np.abs(kap) < 0.01
            stats.append({
                'v_max_kmh': round(v_max * 3.6, 1),
                'left_curve_lane_mean': round(float(lanes[left].mean()), 2),
                'right_curve_lane_mean': round(float(lanes[right].mean()), 2),
                'straight_lane_mean': round(float(lanes[flat].mean()), 2),
                'lane_moves_per_km': round(float(np.abs(np.diff(lanes)).sum()) / (len(lanes) * step / 1000), 1),
                'curve_kmh': round(float(speeds[np.abs(kap) > 0.02].mean() * 3.6), 2),
            })
        hi = len(lanes) - 1 - int(math.ceil(v_max * 1.1 * lead_s / step)) - int(math.ceil(p.stations[-1] / step))
        for _ in range(cnt):
            i = int(rng.integers(0, hi))
            s = i * step
            kap_i = [ks[min(len(ks) - 1, int((s + o) / ds))] + rng.normal(0, 0.006) for o in p.stations]
            v = float(np.clip(max(speeds[i], 0.2) * rng.uniform(0.9, 1.1), 0.2, v_max * 1.07))
            F = float(np.clip(3.0 - n_off[i] * 3.0 / T.LANE_WIDTH + rng.uniform(-0.5, 0.5), 0.0, 6.0))
            conf = rng.choice([1 / 3, 2 / 3, 1.0], p=[0.15, 0.35, 0.5])
            target = lanes[i + int(round(v * lead_s / step))]
            cur = core.clamp(F + 0.5, 1.0, 6.0)
            score = -((k - target) ** 2) / (2 * sigma * sigma) - (1.0 - conf) * stay * np.abs(k - cur)
            e = np.exp(score - score.max())
            X.append(core.features(v, kap_i, F, conf, pv))
            Y.append(e / e.sum())
    return np.array(X), np.array(Y)


def rule_dataset(p0, n, rng, home, apex):
    p = replace(p0, home_lane=home, apex_lane=apex)
    return tp.make_dataset(p, n, rng)


CANDIDATES = {
    # 名前: (説明, 作り方)
    'opt_alat06': ('横加速度 0.6 (シミュレータの最速の設定)', dict(v=v_alat(0.6), lat=0.02, inner=1)),
    'opt_alat04': ('横加速度 0.4 (R15 で 9 km/h 相当. 外→内→外が残る)', dict(v=v_alat(0.4), lat=0.02, inner=1)),
    'opt_table': ('半径ごとの曲がれる速度の表', dict(v=v_table, lat=0.02, inner=1)),
    'opt_table_lat': ('表 + 横移動の費用', dict(v=v_table, lat=0.1, inner=1)),
    'opt_table_margin': ('表 + 横移動の費用 + レーン2〜5の範囲', dict(v=v_table, lat=0.1, inner=2)),
    'rule_h3_a2': ('今のルール教師で 直線レーン3・カーブレーン2', dict(home=3, apex=2)),
}


def main():
    summary = {}
    for name, (desc, cfg) in CANDIDATES.items():
        stats = []
        if 'home' in cfg:
            fn = lambda p, n, rng, c=cfg: rule_dataset(p, n, rng, c['home'], c['apex'])  # noqa: E731
        else:
            fn = lambda p, n, rng, c=cfg, st=stats: dataset(p, n, rng, c['v'], c['lat'], c['inner'], stats=st)  # noqa: E731
        metrics = tp.train_policy(None, os.path.join(OUT, name + '.json'), dataset_fn=fn, verbose=False,
                                  extra={'teacher': {'name': name, 'description': desc}})
        # 学習に使ったのと同じ作り方の教師データを 1 万点だけ保存する (確認用)
        X, Y = fn(core.SixLaneParams(), 10000, np.random.default_rng(99))
        np.savez_compressed(os.path.join(DATA, name + '.npz'), X=X, Y=Y)
        summary[name] = {'description': desc, 'metrics': metrics, 'plan': stats[:3]}
        print(name, metrics, json.dumps(stats[:3], ensure_ascii=False), flush=True)
    with open(os.path.join(HERE, 'summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
