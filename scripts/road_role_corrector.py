#!/usr/bin/env python3
"""
road_role_corrector.py - 車載が付けた白線の役割 (左境界 / 中央 / 右境界) を, 線の外側に「走れる路面」が続くかで直す

路面は honda_shihou_finetuned_road_v2_best.pth の走れる領域 (白線の出力は車載の重みと同じなので, 1 回の推論で両方出る).
features の各白線の lines[].road (topo_features.py --add-road) を使う.

調べる所 (手前の範囲):
  前方 NEAR_X = 2.0〜5.0 m (0.5 m ごと 7 点) × 線から外へ OUT_D = 1.0〜2.0 m (0.2 m ごと 6 点) を, 線の左右それぞれで.
  画面の外の点は数えず, 42 点のうち MIN_CELLS 点以上見えたときだけ使う (画面の端の数点だけでは, 端の路面の外に当たる).
  路面の点の割合が NO_ROAD 未満なら「その側に路面なし」.
  - 前方 2.0 m より手前: 車線の幅 3.66 m の隣の線 (±1.8 m) から外へ 1〜2 m (横 2.8〜3.8 m) は画面の外 (画面に入る横幅は
    前方 x m で左 1.34x m / 右 1.28x m) なので点がほとんど取れない
  - 前方 5.0 m より先: 画像 1 画素が前後 0.18 m 以上になり (5 m で 0.18 m, 8 m で 0.47 m), 車の揺れ (ピッチ 0.5 度で
    5 m 先が約 0.4 m ずれる) と線の式の外挿の誤りも大きくなる
  - 外へ 1.0 m から: 二重線は内側と外側の線が 0.8 m 離れていて, その間は路面と出る. 1.0 m より近くを見ると二重線の
    内側の線が「外に路面あり」になる
  - 外へ 2.0 m まで: 車線の幅 3.66 m より十分狭く, 中央線から見て隣の車線の中に収まる
  rosbag で確かめた基準 (134520 と gamepad 2 周目の映像で確かめた区間) では, 「路面なし」と出た線が境界でない誤りは
  0〜1.8%. 逆に境界線の外に路面がある (コースの外が舗装されている, 逆光など) ことは 6〜42% あるので,
  「路面あり -> 境界ではない」は使わない.

直し方 (どれも「路面なし」の向きだけを使う):
  A. 車載が中央と呼んだ線の左に路面なしが PERSIST コマ (0.6 s) 続いた -> その線は本当は左境界. 車載の役割が 1 本
     ずれているので 中央 -> 左境界, 右境界 -> 中央, 車載の左境界 (それより外の線) は役割を外す. 右なら左右を逆に.
     続くことを求めるのは, コーンなどの物が路面を隠したときや逆光の一瞬で直さないため
  B. 二重線: 車載が左境界と呼んだ線の左すぐ (0.2〜0.6 m) に路面がなく, その 0.8 m 内側 (右) に別の線があって
     その線の左 (1.0〜2.0 m) にも路面がない -> 車載の線は二重線の外側. 内側の線を左境界にする. 右も同じ

aiformula_machine の車載コード (LineTracker) は変えない. 入出力は role_corrector.py と同じく 役割 -> 横位置 の辞書.

使い方 (採点は role_assignment_eval.py と同じ):
  python3 scripts/road_role_corrector.py 20261012_134520_6lane features_134520
  python3 scripts/road_role_corrector.py 20261012_155111_gamepad features_155111 --start 1334 \\
      --lane-ref 360-388:inner,418-467:inner
  --with-segments を付けると, 区間の情報で直した (role_corrector.py) 後に路面で直す
"""

import sys

import numpy as np

from topo_features import ROAD_DS, ROAD_XS

NEAR_X = (2.0, 5.0)        # 前方のどこを見るか [m]
OUT_D = (1.0, 2.0)         # 線から外へどこを見るか [m]
GAP_D = (0.2, 0.6)         # 二重線の間 (内側と外側の線の間 0.8 m) を見る所 [m]
MIN_CELLS = 15             # 画面に入った点 (42 点のうち) がこれ未満なら使わない
PERSIST = 3                # 中央の外の「路面なし」が何コマ続いたら直すか (1 コマ 0.2 s)
NO_ROAD = 0.3              # 路面の点の割合がこれ未満なら「路面なし」
MATCH_M = 0.35             # 車載の線と features の線を同じ線とみなす横位置の差 [m]
DOUBLE_GAP_M = 0.8         # 二重線の内側と外側の線の間 [m]
DOUBLE_TOL_M = 0.15
ROLE_JA = {"left": "左境界", "center": "中央", "right": "右境界"}

_XS, _DS = np.array(ROAD_XS), np.array(ROAD_DS)


def road_frac(line, side, d_rng, x_rng=NEAR_X, min_cells=MIN_CELLS):
    """線の左 (side=0) / 右 (side=1) の, 前方 x_rng × 外へ d_rng の路面の割合. 見えた点が MIN_CELLS 未満なら None."""
    s = line.get("road")
    if not s:
        return None
    g = np.frombuffer(s.encode(), np.uint8).reshape(len(_XS), 2, len(_DS))
    sub = g[(_XS >= x_rng[0] - 1e-6) & (_XS <= x_rng[1] + 1e-6)][:, side][:, (_DS >= d_rng[0] - 1e-6) & (_DS <= d_rng[1] + 1e-6)]
    seen = sub != ord("-")
    if seen.sum() < min_cells:
        return None
    return float((sub == ord("1")).sum() / seen.sum())


def no_road(line, side, d_rng=OUT_D, min_cells=MIN_CELLS):
    f = road_frac(line, side, d_rng, min_cells=min_cells)
    return f is not None and f < NO_ROAD


class RoadRoleCorrector:
    def __init__(self, base=None):
        self.base = base          # 先に区間の情報で直すとき role_corrector.RoleCorrector
        self.streak = [0, 0]      # 中央と呼んだ線の 左 / 右 に「路面なし」が続いたコマ数

    def correct(self, loc, est, vehicle, r):
        """vehicle: {役割: y2}, r: そのコマの features. 返り値: (直した {役割: y2}, 直したことの説明のリスト)."""
        notes = []
        if self.base is not None:
            vehicle, notes = self.base.correct(loc, est, vehicle)
        lines = [l for l in r["lines"] if l.get("road")]

        def match(y):
            if not lines:
                return None
            l = min(lines, key=lambda l: abs(l["y2"] - y))
            return l if abs(l["y2"] - y) < MATCH_M else None

        out = dict(vehicle)
        # A. 中央と呼んだ線の外に路面なし -> 役割が 1 本ずれている
        c = match(out["center"]) if "center" in out else None
        for side in (0, 1):
            self.streak[side] = self.streak[side] + 1 if c is not None and no_road(c, side) else 0
        if self.streak[0] >= PERSIST:
            notes.append(f"中央と付けた線 (y={out['center']:+.2f}) の左 {OUT_D[0]}〜{OUT_D[1]} m に路面なし -> 左境界にずらす")
            new = {"left": out["center"]}
            if "right" in out:
                new["center"] = out["right"]
            out = new
        elif self.streak[1] >= PERSIST:
            notes.append(f"中央と付けた線 (y={out['center']:+.2f}) の右 {OUT_D[0]}〜{OUT_D[1]} m に路面なし -> 右境界にずらす")
            new = {"right": out["center"]}
            if "left" in out:
                new["center"] = out["left"]
            out = new
        # B. 二重線の外側を境界と呼んでいたら内側の線に替える
        for role, side, inward in (("left", 0, -1), ("right", 1, 1)):
            b = match(out[role]) if role in out else None
            if b is None or not no_road(b, side, GAP_D, min_cells=6):
                continue
            inner = [l for l in lines if abs(l["y2"] - (b["y2"] + inward * DOUBLE_GAP_M)) < DOUBLE_TOL_M]
            if inner and no_road(inner[0], side):
                notes.append(f"{ROLE_JA[role]}と付けた線 (y={out[role]:+.2f}) は二重線の外側 -> 内側の線 (y={inner[0]['y2']:+.2f}) を{ROLE_JA[role]}に")
                out[role] = inner[0]["y2"]
        return out, notes


if __name__ == "__main__":
    import json
    from pathlib import Path

    import role_assignment_eval as RA
    from role_corrector import RoleCorrector
    argv = sys.argv[1:]
    with_seg = "--with-segments" in argv
    argv = [a for a in argv if a != "--with-segments"]
    base = None
    if with_seg:
        base = RoleCorrector(json.load(open(Path(__file__).resolve().parent.parent / "data/topological_map/course_topology.json")))
    RA.main(argv, corrector=RoadRoleCorrector(base), fix_label="区間 + 路面で直した" if with_seg else "路面で直した    ")
