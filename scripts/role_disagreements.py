#!/usr/bin/env python3
"""
role_disagreements.py - 車載の白線の役割と, 区間推定 (+ 区間の情報) が付ける役割が, 中央 <-> 境界 で食い違う場面を探す

  車載の役割: rosbag の /aiformula_perception/lane_detector/lane_lines
  区間推定:   topo_localize_eval.TopoLocalizer + role_corrector.RoleCorrector (レーンが CONF_LANE 以上で決まっているときだけ)
  食い違い:   同じ線 (2 m 先の横位置が近い) に, 片方は中央, もう片方は左境界 / 右境界 と付けている (二重線の件は除く)
近い時刻 (GAP_S 以内) の食い違いを 1 つの場面にまとめ, 場面ごとに 1 枚, カメラ映像に両方の役割を描いた画像を並べる
(どちらが正しいかは映像で確かめる). 画像は --sheet に書く (確かめ用).

使い方:
  python3 scripts/role_disagreements.py 20261012_153125_6lane features_153125 --sheet /tmp/sheet.jpg
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import topo_localize_eval as E
from role_assignment_eval import recorded_roles
from role_corrector import RoleCorrector, seg_code
from topo_features import iter_images

WS_DIR = Path(__file__).resolve().parent.parent
GAP_S = 1.5
SWAP = {("center", "left"), ("left", "center"), ("center", "right"), ("right", "center")}
FX, CX, CY, H, CAMX, PIT = 733.26 / 3, 980.22 / 3, 516.63 / 3, 0.56, 0.055, math.radians(0.55)
SHORT = {"left": "L", "center": "C", "right": "R", None: "-"}


def to_image(x, y):
    xr = x - CAMX
    zc = math.cos(PIT) * xr + math.sin(PIT) * H
    yd = -(math.sin(PIT) * xr - math.cos(PIT) * H)
    return CX + FX * (-y) / zc, CY + FX * yd / zc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_name")
    ap.add_argument("features")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--unknown-start", action="store_true", help="発進位置から始めない (位置が分からない状態から)")
    args = ap.parse_args()
    topo = json.load(open(WS_DIR / "data/topological_map/course_topology.json"))
    R = json.load(open(WS_DIR / f"data/topological_map/{args.features}.json"))
    rec = recorded_roles(WS_DIR / f"rosbag/mcap/{args.bag_name}.mcap")
    vt = np.array([v["t"] for v in rec])
    loc = E.TopoLocalizer(topo, not args.unknown_start, 1)
    corr = RoleCorrector(topo)
    hits = []
    n_frames = n_conf = 0
    for r in R:
        est = loc.update(r)
        i = int(np.clip(np.searchsorted(vt, r["t"]), 0, len(vt) - 1))
        if abs(vt[i] - r["t"]) > 0.15:
            continue
        veh = {k: v for k, v in rec[i].items() if k != "t"}
        if not veh:
            continue
        n_frames += 1
        fixed, notes = corr.correct(loc, est, veh)
        if not notes:
            inner = float(loc.po[E.OFFSETS > 0].sum())
            n_conf += max(inner, 1 - inner) >= 0.9
            continue
        n_conf += 1
        swaps = []
        for role, y in veh.items():
            new = [k for k, y2 in fixed.items() if abs(y2 - y) < 1e-6]
            if new and (role, new[0]) in SWAP:
                swaps.append((role, new[0], y))
        if swaps:
            hits.append(dict(t=r["t"], rt=r["rt"], seg=seg_code(topo["segments"][int(loc.seg_of[est])]), s=float(loc.s_arr[est]),
                             swaps=swaps, veh=veh, fixed=fixed, lines=r["lines"], lane="inner" if loc.offset() > 0 else "outer"))
    # 場面にまとめる
    scenes = []
    for h in hits:
        if scenes and h["rt"] - scenes[-1][-1]["rt"] <= GAP_S:
            scenes[-1].append(h)
        else:
            scenes.append([h])
    print(f"{args.bag_name}: 車載の役割があるコマ {n_frames}, 区間推定がレーンを決めていたコマ {n_conf}, "
          f"中央 <-> 境界 の食い違い {len(hits)} コマ / {len(scenes)} 場面")
    for k, sc in enumerate(scenes):
        sw = {f"{SHORT[a]}->{SHORT[b]}" for h in sc for a, b, _ in h["swaps"]}
        print(f"  場面 {k}: {sc[0]['rt']:6.1f}〜{sc[-1]['rt']:6.1f} s ({len(sc)} コマ)  区間 {sc[0]['seg']} (s={sc[0]['s']:.0f} m)"
              f"  区間推定のレーン {sc[0]['lane']}  車載->区間の情報: {', '.join(sorted(sw))}")
    if not args.sheet or not scenes:
        return
    want = {sc[len(sc) // 2]["t"]: (k, sc[len(sc) // 2]) for k, sc in enumerate(scenes)}
    tiles = []
    keys = np.array(sorted(want))
    for t, data in iter_images(WS_DIR / f"rosbag/mcap/{args.bag_name}.mcap"):
        j = int(np.clip(np.searchsorted(keys, t), 0, len(keys) - 1))
        near = [x for x in (keys[j], keys[max(j - 1, 0)]) if abs(x - t) < 2e-3]
        if not near:
            continue
        k, h = want.pop(near[0])
        keys = np.array(sorted(want)) if want else np.array([1e18])
        img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (640, 360))
        for role, y in h["veh"].items():
            l = min(h["lines"], key=lambda l: abs(l["y2"] - y))
            if abs(l["y2"] - y) > 0.35:
                continue
            c0, c1, c2 = l["c"]
            pts = [to_image(x, c0 + c1 * x + c2 * x * x) for x in np.arange(max(l["x0"], 0.9), min(l["x1"], 12), 0.2)]
            pts = np.array([(int(u), int(v)) for u, v in pts if 0 <= u < 640 and 0 <= v < 360])
            if len(pts) < 2:
                continue
            new = [kk for kk, y2 in h["fixed"].items() if abs(y2 - y) < 1e-6]
            changed = not new or new[0] != role
            cv2.polylines(img, [pts], False, (0, 0, 255) if changed else (0, 255, 0), 2)
            u, v = pts[0]
            txt = f"V:{SHORT[role]}" + (f" fix:{SHORT[new[0]] if new else '-'}" if changed else "")
            cv2.putText(img, txt, (min(max(u, 4), 560), max(v - 8, 14)), 0, 0.55, (0, 0, 0), 4)
            cv2.putText(img, txt, (min(max(u, 4), 560), max(v - 8, 14)), 0, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"#{k} {h['rt']:.1f}s {h['seg']} lane:{h['lane']}", (8, 24), 0, 0.7, (0, 0, 0), 4)
        cv2.putText(img, f"#{k} {h['rt']:.1f}s {h['seg']} lane:{h['lane']}", (8, 24), 0, 0.7, (0, 255, 255), 2)
        tiles.append((k, img))
        if not want:
            break
    tiles = [im for _, im in sorted(tiles, key=lambda x: x[0])]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(args.sheet, sheet)
    print("  画像 ->", args.sheet)


if __name__ == "__main__":
    main()
