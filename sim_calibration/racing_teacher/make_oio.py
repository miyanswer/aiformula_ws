#!/usr/bin/env python3
"""アウト・イン・アウトが残る racing 教師を作る. aiformula_machine の中には何も書かない.

    python3 -B make_oio.py

(1) rule_o{O}_a{A}: 今のルール教師の局面判定で, 外側をレーン O, 内側をレーン A に狭めた教師 (横移動の量を減らす)
(2) opt_oio_*: 最短時間の教師を, 外→内→外が得になる低い横加速度で作った教師
"""
import json
import os

import numpy as np

import make_teachers as M

core = M.core
tp = M.tp
OUT, DATA, HERE = M.OUT, M.DATA, M.HERE


def rule_teacher(outer, apex, sigma=0.6, stay=1.5):
    def fn(v, kappas, F, conf, p):
        phase, s, target, _ = core.classify_phase(v, kappas, p)
        o = outer if s >= 0 else 7 - outer
        a = apex if s >= 0 else 7 - apex
        if s == 0 or phase == core.STRAIGHT:
            t = float(outer)
        elif phase == core.APEX:
            t = float(a)
        elif phase == core.EXIT:
            t = float(o)
        else:  # ENTRY: 元の教師の内外の割合を, 狭めた幅に写す
            full_o, full_i = (6.0, 1.0) if s > 0 else (1.0, 6.0)
            r = (target - full_o) / (full_i - full_o)
            t = o + (a - o) * r
        cur = core.clamp(F + 0.5, 1.0, 6.0)
        k = np.arange(1, 7, dtype=float)
        score = -((k - t) ** 2) / (2 * sigma * sigma) - (1.0 - conf) * stay * np.abs(k - cur)
        e = np.exp(score - score.max())
        return e / e.sum()
    return fn


RULES = {f'rule_o{o}_a{a}': (o, a) for o, a in [(5, 1), (5, 2), (4, 1), (4, 2)]}
OPTS = {'opt_oio_a015': (0.15, 2), 'opt_oio_a02': (0.2, 2), 'opt_oio_a025': (0.25, 2)}


def main():
    summary = {}
    for name, (o, a) in RULES.items():
        fn = rule_teacher(o, a)
        metrics = tp.train_policy(fn, os.path.join(OUT, name + '.json'), verbose=False,
                                  extra={'teacher': {'name': name, 'outer': o, 'apex': a}})
        X, Y = tp.make_dataset(core.SixLaneParams(), 10000, np.random.default_rng(99), fn)
        np.savez_compressed(os.path.join(DATA, name + '.npz'), X=X, Y=Y)
        summary[name] = {'metrics': metrics}
        print(name, metrics, flush=True)
    for name, (al, inner) in OPTS.items():
        stats = []
        fn = lambda p, n, rng, al=al, inner=inner, st=stats: M.dataset(p, n, rng, M.v_alat(al), 0.02, inner, stats=st)  # noqa: E731
        metrics = tp.train_policy(None, os.path.join(OUT, name + '.json'), dataset_fn=fn, verbose=False,
                                  extra={'teacher': {'name': name, 'a_lat': al, 'inner': inner}})
        X, Y = fn(core.SixLaneParams(), 10000, np.random.default_rng(99))
        np.savez_compressed(os.path.join(DATA, name + '.npz'), X=X, Y=Y)
        summary[name] = {'metrics': metrics, 'plan': stats[:3]}
        print(name, metrics, json.dumps(stats[:3], ensure_ascii=False), flush=True)
    with open(os.path.join(HERE, 'summary_oio.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
