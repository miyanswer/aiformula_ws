#!/usr/bin/env python3
"""
role_assignment_eval.py - 白線の役割 (左境界 / 中央 / 右境界) が合っているかを rosbag で数え,
区間の情報で直すと (role_corrector.py) どれだけ良くなるかを見る

車載の役割:
  rosbag に記録があれば (/aiformula_perception/lane_detector/lane_lines) それ.
  なければ (gamepad 走行) 車載と同じ LineTracker (aiformula_machine から読むだけ) を, 同じ白線 (features) にかけて作る
直した役割: role_corrector.RoleCorrector (区間推定のレーンと, 区間の線の種類で直す)
基準の役割 (採点だけに使う):
  --lane-ref があれば, そのコマの範囲は映像で確かめたレーン (inner / outer) の中で, 見えている白線が図面の線
    (その位置に図面で線がある役割だけ) に一番よく重なる横のずれから付けた役割. 範囲の外は採点しない
  なければ, 正解の位置 (data/truth/<bag>/*.npy) の横のずれ ±REF_SEARCH_M の中で同じようにずらしたもの
    (正解のレーンそのものが間違っている所では基準も間違う. gamepad の正解は上の直線や C4 で外側と出ているので
     gamepad では --lane-ref を使う)

使い方:
  python3 scripts/role_assignment_eval.py 20261012_134520_6lane features_134520
  python3 scripts/role_assignment_eval.py 20261012_155111_gamepad features_155111 --start 1334 \\
      --lane-ref 360-388:inner,418-467:inner
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import topo_localize_eval as E
from role_corrector import RoleCorrector, seg_code

WS_DIR = Path(__file__).resolve().parent.parent
MACHINE = Path.home() / "aiformula_machine"
REF_SEARCH_M = 1.2
REF_MAX_RES_M = 0.3
MATCH_M = 0.35
ROLE_JA = {"left": "左境界", "center": "中央", "right": "右境界", "left2": "左の二重線の外側", "right2": "右の二重線の外側", None: "なし"}

READER = r'''
import sys, json
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
out = []
with open(sys.argv[1], "rb") as fp:
    for _, ch, msg, m in make_reader(fp, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=["/aiformula_perception/lane_detector/lane_lines"]):
        row = {"t": msg.log_time / 1e9}
        for role in ("left", "center", "right"):
            l = getattr(m, role)
            if l.detected:
                row[role] = l.c0 + 2 * l.c1 + 4 * l.c2
        out.append(row)
json.dump(out, sys.stdout)
'''


def recorded_roles(bag):
    r = subprocess.run([str(WS_DIR / ".venv/bin/python"), "-c", READER, str(bag)], capture_output=True, text=True, check=True)
    rows = json.loads(r.stdout)
    return rows if any(len(v) > 1 for v in rows) else None


def tracker_roles(R_all):
    """車載と同じ LineTracker (変えずに読むだけ) を features の白線にかけて, コマごとの {役割: y2} を作る."""
    sys.path.insert(0, str(MACHINE / "src/oit_navigation"))
    from oit_navigation.common.geometry import LineFit
    from oit_navigation.lane.line_tracker import LineTracker, LineTrackerParams, ROLES
    tr = LineTracker(LineTrackerParams(x_ref=2.0, lane_width_init=3.5))
    out = {}
    for r in R_all:
        fits = [LineFit(np.array(l["c"], float), l["x0"], l["x1"], 20) for l in r["lines"] if l.get("c")]
        t = tr.update(fits)
        out[r["t"]] = {role: float(t.lines[role].y_at(2.0)) for role in ROLES if t.detected[role]}
    return out


def parse_args(argv=None, extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_name")
    ap.add_argument("features")
    ap.add_argument("--truth", default="old_align_smooth.npy", help="data/truth/<bag>/ の中の正解のファイル")
    ap.add_argument("--start", type=int, default=0, help="features の何コマ目から (gamepad の 2 周目は 1334)")
    ap.add_argument("--road-only", type=float, default=2.7, help="正解の横のずれがこれより外のコマは採点しない")
    ap.add_argument("--lane-ref", default="", help="映像で確かめたレーン: '開始-終了:inner,...' [s] (rosbag の先頭から)")
    ap.add_argument("--show", type=int, default=6, help="直した例をいくつ表示するか")
    ap.add_argument("--tracker", action="store_true",
                    help="rosbag に記録された車載の役割を使わず, 車載と同じ LineTracker を features の白線 (前処理の修正後) にかけて作る."
                         " 2026-10-19 より前の rosbag の記録は前処理の修正前 (縦に引き伸ばし) の白線から作られている")
    if extra:
        extra(ap)
    return ap.parse_args(argv)


def scored_frames(args, corrector=None):
    """採点できるコマごとに dict(r, loc, est, veh, ref_of, fixed, notes) を返す (ref_of(y) は横位置 y の線の基準の役割).
    corrector を渡すと, その correct(loc, est, veh, r) で直した役割を fixed に入れる (省略すると区間の情報で直す).
    最初に dict(src=...) を 1 つ返す."""
    topo = json.load(open(WS_DIR / "data/topological_map/course_topology.json"))
    R_all = json.load(open(WS_DIR / f"data/topological_map/{args.features}.json"))
    R = R_all[args.start:]
    loc = E.TopoLocalizer(topo, True, 1)                          # 発進位置から (gamepad も 2 周目の始め = 発進位置)
    corr = RoleCorrector(topo)
    truth = E.load_truth(str(WS_DIR / f"data/truth/{args.bag_name}/{args.truth}"), R, loc)
    rec = None if args.tracker else recorded_roles(WS_DIR / f"rosbag/mcap/{args.bag_name}.mcap")
    if rec:
        vt = np.array([v["t"] for v in rec])
        src = "rosbag に記録された車載の役割"
    else:
        sim = tracker_roles(R_all)
        src = "車載と同じ LineTracker を同じ白線 (前処理の修正後) にかけた役割" + ("" if args.tracker else " (rosbag に記録なし)")
    yield dict(src=src)
    lane_ref = []
    for part in filter(None, args.lane_ref.split(",")):
        rng, lane = part.split(":")
        a, b = map(float, rng.split("-"))
        lane_ref.append((a, b, lane))

    for r, g in zip(R, truth):
        est = loc.update(r)
        if rec:
            i = int(np.clip(np.searchsorted(vt, r["t"]), 0, len(vt) - 1))
            if abs(vt[i] - r["t"]) > 0.15:
                continue
            veh = {k: v for k, v in rec[i].items() if k != "t"}
        else:
            veh = sim[r["t"]]
        fixed, notes = corrector.correct(loc, est, veh, r) if corrector else corr.correct(loc, est, veh)
        # 基準の横のずれ
        if lane_ref:
            lr = [lane for a, b, lane in lane_ref if a <= r["rt"] <= b]
            if not lr:
                continue
            cand = np.arange(0.9, 2.71, 0.05) * (1 if lr[0] == "inner" else -1)
            i_st = g[0] if g is not None else est
        else:
            if g is None or abs(g[1]) > args.road_only:
                continue
            cand = g[1] + np.arange(-REF_SEARCH_M, REF_SEARCH_M + 1e-6, 0.05)
            i_st = g[0]
        exist = loc.role_type[i_st] != ""
        pos = np.where(exist, loc.role_pos, 99.0)
        ys = np.array(list(veh.values()) + [l["y2"] for l in r["lines"] if abs(l["y2"]) < 4.5])
        if not len(ys):
            continue
        res = np.min(np.abs(ys[None, :, None] + cand[:, None, None] - pos[None, None, :]), 2)
        k = int(np.argmin(np.median(res, 1)))
        if np.median(res[k]) > REF_MAX_RES_M:
            continue
        o_ref = cand[k]

        def ref_of(y, o_ref=o_ref, pos=pos):
            d = np.abs(y + o_ref - pos)
            j = int(np.argmin(d))
            return loc.role_names[j] if d[j] < REF_MAX_RES_M else "?"

        yield dict(r=r, loc=loc, est=est, veh=veh, ref_of=ref_of, fixed=fixed, notes=notes)


def main(argv=None, corrector=None, fix_label="区間の情報で直した"):
    args = parse_args(argv)
    it = scored_frames(args, corrector)
    src = next(it)["src"]
    n = {"veh": [0, 0], "fix": [0, 0]}
    fixed_ok, fixed_bad, examples, bad_examples = 0, 0, [], []
    conf_v, conf_f = {}, {}
    frames = 0
    for f in it:
        r, veh, ref_of, fixed, notes = f["r"], f["veh"], f["ref_of"], f["fixed"], f["notes"]
        frames += 1
        for role, y in veh.items():
            ref = ref_of(y)
            if ref == "?":
                continue
            ok_v = role == ref
            got = [k2 for k2, y2 in fixed.items() if abs(y2 - y) < 1e-6]
            f_role = got[0] if got else None
            ok_f = (f_role == ref) or (f_role is None and ref in ("left2", "right2"))
            n["veh"][0] += ok_v; n["veh"][1] += 1
            n["fix"][0] += ok_f; n["fix"][1] += 1
            if not ok_v:
                conf_v[(ref, role)] = conf_v.get((ref, role), 0) + 1
            if not ok_f:
                conf_f[(ref, f_role)] = conf_f.get((ref, f_role), 0) + 1
            if not ok_v and ok_f:
                fixed_ok += 1
                if len(examples) < args.show:
                    examples.append(f"{r['rt']:6.1f}s  " + " / ".join(notes))
            if ok_v and not ok_f:
                fixed_bad += 1
                if len(bad_examples) < args.show:
                    bad_examples.append(f"{r['rt']:6.1f}s  " + " / ".join(notes))

    print(f"{args.bag_name}  ({src})")
    print(f"  採点したコマ {frames}  基準: {'映像で確かめたレーン ' + args.lane_ref if args.lane_ref else '正解の位置 (' + args.truth + ')'}")
    for key, lab, conf in (("veh", "車載の役割      ", conf_v), ("fix", fix_label, conf_f)):
        a, m = n[key]
        print(f"  {lab}: {a}/{m} 本が正しい ({100 * a / max(m, 1):.1f}%)")
        for (ref, got), c in sorted(conf.items(), key=lambda x: -x[1])[:4]:
            print(f"      取り違え: 本当は {ROLE_JA.get(ref, ref)} を {ROLE_JA.get(got, got)} にした {c} 本")
    print(f"  直して正しくなった {fixed_ok} 本 / 直して間違えた {fixed_bad} 本")
    for e in examples:
        print("    例:", e)
    for e in bad_examples:
        print("    間違えた例:", e)


if __name__ == "__main__":
    main()
