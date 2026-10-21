#!/usr/bin/env python3
"""
build_topological_map.py - 公式図面からコース外周のトポロジカルマップ (区間の並び) を作る

外周の中央線に沿って, 走る向き (反時計回り, 発進位置から) に STEP_M ごとに図面を調べ,
  - 形: 直線 / 左カーブ / 右カーブ (中央線の曲率)
  - 線の種類: 左境界 (内側) / 中央 / 右境界 (外側) それぞれ 実線 / 破線 / なし.
    course.glb の線のかけら (破線は 1 本ずつ別のかけら) の長さと位置から決める (図面の画素を平均すると切り替わりが
    ずれたり, 曲がって離れていく線を破線と読み違えたりするため)
  - 二重線: 境界線の外側 0.4〜1.2 m にもう 1 本あるか
  - 緑: 境界線の外側 0.3〜1.5 m が緑 (着色舗装) か
  - 目印: 横断歩道 / 停止線 / 信号 (MyLaps)
を読み取り, 図面と実際のコースが違う所 (FIELD_FIXES) を直してから, 同じ属性が続く所を 1 つの区間にまとめる. 座標 (地図) ではなく区間の並びとして持つ.

入力 (読むだけ): aiformula_machine の公式図面 drawing.png と course.glb (外周 3 本の白線)
出力: data/topological_map/course_topology.json, course_topology.png (図面に区間を重ねた図)

使い方:
  .venv/bin/python scripts/build_topological_map.py
  .venv/bin/python scripts/build_topological_map.py --machine ~/aiformula_machine
  (data/topological_map/segments_manual.json があれば, 区間はその境目で作る. segment_editor.py で手で直して保存したもの)
"""

import argparse
import json
import math
import os
import struct
from pathlib import Path

import cv2
import numpy as np

WS_DIR = Path(__file__).resolve().parent.parent
PXM = 5.711                 # 図面 1 m あたりの画素数 (course_from_drawing と同じ)
LANE_W = 3.655              # 外周の線の間隔 [m] (図面: 車道 3.5 m + 白線 0.15 m)
STEP_M = 0.5                # 調べる間隔 [m]
CURVE_KAPPA = 1.0 / 40.0    # これより曲率が大きければカーブ [1/m] (外周のカーブは半径 15〜25 m で 0.04〜0.07, 直線は 0.015 以下)
MIN_SEG_M = 3.0             # これより短い区間は前の区間にまとめる (図面の描き方の細かな揺れを消す)
PIECE_LAT_M = 0.45          # 線のかけらが役割の横位置 (左 +LANE_W / 中央 0 / 右 -LANE_W) からこれ以内なら, その役割の線
DASH_MAX_M = 4.5            # これより短いかけらは破線の 1 本 (図面: 中央の破線 3.6 m, 左境界の破線 2.0 m)
DOUBLE_SNAP_M = 1.0         # 二重線の端を線の種類の変わり目へそろえる距離
DOUBLE_MIN_M = 3.0          # これより短い二重線は消す (図面の画素の読み違い)
DASH_GAP_MAX_M = 3.5        # 破線どうしの間がこれ以下なら, 間も破線 (図面: 途切れ 1.2〜2.8 m)
KAPPA_HALF_M = 4.0          # 曲率は前後 ±KAPPA_HALF_M で測る (半径の大きいカーブで直線と行き来しないように)
# 発進位置 (aiformula_machine web_simulator/js/course.js START_POSE) と MyLaps 信号 (course_props.js, odom 座標)
START_POSE = (93.175, -50.868, math.radians(-91.49))
MYLAPS_ODOM = (4.747, 82.523)
TYPE_JA = {"solid": "実線", "dashed": "破線", "none": "なし"}
# 図面と実際のコースが違う所の直し (図面 drawing.png は変えず, 読み取った後に直す).
# (役割, 図面での種類, 実際の種類, 目印, その目印からの距離 [m], 理由)
# 信号の交差点の左境界 (内側) の案内破線は, 図面 drawing.png そのものに描き足した (2026-10-20) ので, ここでは直さない
FIELD_FIXES = []


# ---------------------------------------------------------------- course.glb (追加ライブラリなしで読む)
def load_glb_lines(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    J = json.loads(b[20:20 + n])
    off = 20 + n
    blen = struct.unpack("<I", b[off:off + 4])[0]
    BIN = b[off + 8:off + 8 + blen]

    def acc(i):
        a = J["accessors"][i]
        v = J["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        o = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        arr = np.frombuffer(BIN, dt, a["count"] * nc, o)
        return arr.reshape(a["count"], nc) if nc > 1 else arr

    out = {"center": [], "outer": [], "inner": []}
    for m in J["meshes"]:
        name = m["name"]
        key = "center" if name.startswith("Line_Center") else "outer" if name.startswith("Line_Outer") else \
            "inner" if name.startswith("Line_Block_Loop") else None
        if key is None:
            continue
        P = acc(m["primitives"][0]["attributes"]["POSITION"]).astype(np.float64)
        out[key].append(np.stack([P[:, 0], -P[:, 2]], 1))   # glTF (Y-up) -> 図面座標 (x, y)
    return {k: np.concatenate(v) for k, v in out.items()}


def load_glb_pieces(path):
    """course.glb の外周 3 本を, かけら (メッシュ) ごとの図面座標の点で返す: [(役割, 名前, 点 (n, 2))].
    中央の破線や内側の境界の破線は 1 本ずつ別のメッシュになっている."""
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    J = json.loads(b[20:20 + n])
    off = 20 + n
    blen = struct.unpack("<I", b[off:off + 4])[0]
    BIN = b[off + 8:off + 8 + blen]
    out = []
    for m in J["meshes"]:
        name = m["name"]
        role = "center" if name.startswith("Line_Center") else "right" if name.startswith("Line_Outer") else \
            "left" if name.startswith("Line_Block_Loop") else None
        if role is None:
            continue
        prim = m["primitives"][0]
        a = J["accessors"][prim["attributes"]["POSITION"]]
        v = J["bufferViews"][a["bufferView"]]
        P = np.frombuffer(BIN, np.float32, a["count"] * 3, v.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(-1, 3)
        ai = J["accessors"][prim["indices"]]
        vi = J["bufferViews"][ai["bufferView"]]
        dt = {5125: np.uint32, 5123: np.uint16}[ai["componentType"]]
        F = np.frombuffer(BIN, dt, ai["count"], vi.get("byteOffset", 0) + ai.get("byteOffset", 0)).reshape(-1, 3)
        out.append((role, name, np.stack([P[:, 0], -P[:, 2]], 1).astype(np.float64), F.astype(np.int64)))
    return out


def piece_line_types(pieces, loop, step):
    """ステーションごとに, 左境界 / 中央 / 右境界 の線が 実線 / 破線 / なし か を, course.glb の線のかけらから決める.
    かけらの三角形を外周の中央線に沿った距離 (s) と横位置に直し, その役割の横位置 (±PIECE_LAT_M) にある三角形が
    覆う s を線とする. かけらの長さが DASH_MAX_M 未満なら破線の 1 本, それ以上なら実線.
    破線の 1 本の上か, 破線どうしの間 (DASH_GAP_MAX_M 以内) なら破線, 実線の上なら実線, どちらでもなければなし."""
    N = len(loop)
    lap = N * step
    t = np.roll(loop, -2, 0) - np.roll(loop, 2, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nl = np.stack([-t[:, 1], t[:, 0]], 1)
    role_off = {"left": LANE_W, "center": 0.0, "right": -LANE_W}
    spans = {r: {"solid": [], "dash": []} for r in role_off}
    for role, name, P, F in pieces:
        d = np.hypot(P[:, None, 0] - loop[None, :, 0], P[:, None, 1] - loop[None, :, 1])
        i = np.argmin(d, 1)
        rel = P - loop[i]
        along = i * step + np.einsum("ij,ij->i", rel, t[i])
        lat = np.einsum("ij,ij->i", rel, nl[i])
        near = np.abs(lat - role_off[role]) < PIECE_LAT_M
        # その役割の横位置にある三角形ごとに, s の範囲 (周回の継ぎ目をまたぐものは継ぎ目の先へ伸ばす)
        iv = []
        for f in F[near[F].all(1)]:
            a = along[f]
            a = (a - a[0] + lap / 2) % lap + a[0] - lap / 2
            iv.append((float(a.min()) % lap, float(a.max() - a.min())))
        if not iv:
            continue
        cov = np.zeros(N, bool)
        for lo, ln in iv:
            cov[((np.arange(N) * step - lo) % lap) <= ln] = True
        # かけらの長さ = 覆っているステーションの数 x 間隔. 1 本の破線は 1 つの続きになる
        length = cov.sum() * step
        spans[role]["dash" if length < DASH_MAX_M else "solid"].append(cov)
    out = {}
    for role, sp in spans.items():
        kind = np.array(["none"] * N, object)
        dash = np.any(sp["dash"], 0) if sp["dash"] else np.zeros(N, bool)
        solid = np.any(sp["solid"], 0) if sp["solid"] else np.zeros(N, bool)
        # 破線の途切れ: 前後 DASH_GAP_MAX_M 以内に線があり, 少なくとも片方が破線なら破線 (実線から破線へ変わる所の途切れも)
        g = int(round(DASH_GAP_MAX_M / step))
        line = dash | solid
        filled = dash.copy()
        for j in range(N):
            if not line[j]:
                back = [k for k in range(1, g + 1) if line[(j - k) % N]]
                fwd = [k for k in range(1, g + 1) if line[(j + k) % N]]
                if back and fwd and (dash[(j - back[0]) % N] or dash[(j + fwd[0]) % N]):
                    filled[j] = True
        kind[filled] = "dashed"
        kind[solid] = "solid"
        out[role] = kind
    return out


def center_loop(pts, n_bins=720):
    """中央線の点を重心まわりの角度で並べて 1 周の折れ線にする (破線の切れ目は補間)."""
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    k = ((ang + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins
    loop = np.full((n_bins, 2), np.nan)
    for i in range(n_bins):
        m = k == i
        if m.any():
            loop[i] = pts[m].mean(0)
    ok = ~np.isnan(loop[:, 0])
    idx = np.arange(n_bins)
    for d in (0, 1):
        loop[:, d] = np.interp(idx, idx[ok], loop[ok, d], period=n_bins)
    return loop          # 角度が増える向き = 反時計回り


def resample(loop, step):
    P = np.vstack([loop, loop[:1]])
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1), s[-1]


# ---------------------------------------------------------------- 図面の画素
class Drawing:
    def __init__(self, path):
        im = cv2.imread(str(path))[..., ::-1].astype(np.int16)
        self.h, self.w = im.shape[:2]
        self.white = im.mean(2) > 185
        self.green = (im[..., 1] - np.maximum(im[..., 0], im[..., 2])) > 30
        self.img = im.astype(np.uint8)

    def px(self, xy):
        u = np.round(np.asarray(xy)[..., 0] * PXM).astype(int)
        v = np.round(-np.asarray(xy)[..., 1] * PXM).astype(int)
        ok = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        return np.clip(u, 0, self.w - 1), np.clip(v, 0, self.h - 1), ok

    def frac(self, mask, xy):
        u, v, ok = self.px(xy)
        return float(mask[v, u][ok].mean()) if ok.any() else 0.0


def line_type(dw, p, t, n, offset, search=0.45, half=2.0):
    """位置 p の接線 t 方向に ±half m, 横 offset 付近で一番白い所の白の割合から 実線 / 破線 / なし を決める."""
    along = np.arange(-half, half + 1e-9, 0.15)
    best, best_off = 0.0, offset
    for d in np.arange(-search, search + 1e-9, 0.08):
        xy = p + np.outer(along, t) + (offset + d) * n
        f = dw.frac(dw.white, xy)
        if f > best:
            best, best_off = f, offset + d
    kind = "solid" if best > 0.8 else "dashed" if best > 0.15 else "none"
    return kind, best_off, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default=str(Path.home() / "aiformula_machine"), help="aiformula_machine のパス (読むだけ)")
    ap.add_argument("--out", default=str(WS_DIR / "data/topological_map"))
    ap.add_argument("--segments", default=str(WS_DIR / "data/topological_map/segments_manual.json"),
                    help="手で直した区間 (segment_editor.py の保存先). ファイルがなければ属性から自動で区間を作る. '' で自動")
    args = ap.parse_args()
    machine = Path(args.machine).expanduser()
    drawing_path = machine / "web_simulator/tools/course_from_drawing/drawing.png"
    glb_path = machine / "web_simulator/models/course.glb"
    os.makedirs(args.out, exist_ok=True)

    lines = load_glb_lines(glb_path)
    dw = Drawing(drawing_path)
    loop, total = resample(center_loop(lines["center"]), STEP_M)
    # 発進位置にいちばん近い点を 0 m にする (走る向き = 反時計回り)
    i0 = int(np.argmin(np.hypot(*(loop - START_POSE[:2]).T)))
    loop = np.roll(loop, -i0, axis=0)
    N = len(loop)

    # 横断歩道・停止線: 図面の白のうち太い所 (3x3 で開いても残る) を塊ごとに
    thick = cv2.morphologyEx(dw.white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(thick, np.ones((9, 9), np.uint8)))
    marks = []
    for i in range(1, nlab):
        x, y, w, h, _a = st[i]
        if (thick[lab == i] > 0).sum() < 20:
            continue
        box = (x / PXM, (x + w) / PXM, -(y + h) / PXM, -y / PXM)
        marks.append(("crosswalk" if (box[1] - box[0]) * (box[3] - box[2]) > 15 else "stop_line", box))
    c, s_ = math.cos(START_POSE[2]), math.sin(START_POSE[2])
    sig = np.array([START_POSE[0] + c * MYLAPS_ODOM[0] - s_ * MYLAPS_ODOM[1],
                    START_POSE[1] + s_ * MYLAPS_ODOM[0] + c * MYLAPS_ODOM[1]])

    types = piece_line_types(load_glb_pieces(glb_path), loop, STEP_M)
    rows = []
    for i in range(N):
        p = loop[i]
        hw = int(round(KAPPA_HALF_M / STEP_M))
        a, b = loop[(i - hw) % N], loop[(i + hw) % N]         # ±KAPPA_HALF_M で接線と曲率
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])                            # 左向き
        h1 = math.atan2(*(loop[i] - a)[::-1]); h2 = math.atan2(*(b - loop[i])[::-1])
        kappa = ((h2 - h1 + math.pi) % (2 * math.pi) - math.pi) / (STEP_M * hw)   # 2 本の弦の向きの差 / 弦の長さ
        shape = "straight" if abs(kappa) < CURVE_KAPPA else ("left" if kappa > 0 else "right")
        r = {"s": round(i * STEP_M, 1), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2),
             "kappa": round(kappa, 4), "shape": shape}
        for role, off, out_sign in (("left", LANE_W, 1), ("center", 0.0, 0), ("right", -LANE_W, -1)):
            _, found, f = line_type(dw, p, t, n, off, half=0.5)
            kind = str(types[role][i])                             # 線の種類は course.glb の線のかけらから
            r[role] = kind
            if out_sign:
                # 境界線の外側: もう 1 本 (二重線. 前後 ±0.5 m だけ見る) / 緑
                kind2, _, f2 = line_type(dw, p, t, n, found + out_sign * 0.8, search=0.4, half=0.5)
                r[role + "_double"] = kind != "none" and f2 > 0.5
                band = [p + np.outer(np.arange(-1.5, 1.51, 0.3), t) + (found + out_sign * o) * n for o in np.arange(0.3, 2.51, 0.2)]
                r[role + "_green"] = dw.frac(dw.green, np.concatenate(band)) > 0.4
        lm = []
        for kind, (x0, x1, y0, y1) in marks:
            # 車道 (左右 3.7 m 以内) で, この位置の前後 1 m にかかる目印
            for ds in (-1.0, 0.0, 1.0):
                for dl in np.arange(-3.6, 3.61, 0.6):
                    q = p + ds * t + dl * n
                    if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                        lm.append(kind)
                        break
                else:
                    continue
                break
        if np.hypot(*(sig - p)) < 6.0 and (sig - p) @ n < 0:     # 信号は右側
            lm.append("signal")
        r["landmarks"] = sorted(set(lm))
        rows.append(r)

    # 二重線は図面の画素で見ているので, 線の端や曲がり始めで短く出たり切れたりする. DOUBLE_MIN_M より短い続きは消す
    for role in ("left", "right"):
        flag = np.array([r[role + "_double"] for r in rows])
        i = 0
        while i < N:
            j = i
            while j < N and flag[j] == flag[i]:
                j += 1
            if flag[i] and (j - i) * STEP_M < DOUBLE_MIN_M and not (i == 0 and flag[-1]) and not (j == N and flag[0]):
                for k in range(i, j):
                    rows[k][role + "_double"] = False
            i = j

    # 二重線の端が, 同じ側の線の種類の変わり目 (course.glb から正確に決めたもの) から DOUBLE_SNAP_M 以内なら, そこへそろえる
    for role in ("left", "right"):
        kind = [r[role] for r in rows]
        change = [i for i in range(N) if kind[i] != kind[i - 1]]
        flag = [r[role + "_double"] for r in rows]
        edges = [i for i in range(N) if flag[i] != flag[i - 1]]
        w = int(round(DOUBLE_SNAP_M / STEP_M))
        for e in edges:
            near = [c for c in change if 0 < abs((c - e + N // 2) % N - N // 2) <= w]
            if not near:
                continue
            c = min(near, key=lambda c: abs((c - e + N // 2) % N - N // 2))
            val = flag[e]                                  # 端の先 (e 以降) の値
            lo, hi = (c, e) if (e - c) % N < N // 2 else (e, c)
            k = lo
            while k != hi:
                rows[k][role + "_double"] = val if (e - c) % N < N // 2 else not val
                k = (k + 1) % N

    # 図面と実際が違う所を直す
    for role, frm, to, mark, dist, why in FIELD_FIXES:
        s_mark = [r["s"] for r in rows if mark in r["landmarks"]]
        n_fixed = 0
        for r in rows:
            near = any(abs((r["s"] - sm + N * STEP_M / 2) % (N * STEP_M) - N * STEP_M / 2) <= dist for sm in s_mark)
            if near and r[role] == frm:
                r[role] = to
                r.setdefault("fixed", []).append(f"{role}: {frm} -> {to}")
                n_fixed += 1
        print(f"図面の直し: {role} {frm} -> {to} ({mark} の前後 {dist} m) {n_fixed} か所 / {why}")

    # 同じ属性が続く所を区間にまとめる (短すぎる区間は前にまとめる)
    keys = ("shape", "left", "center", "right", "left_double", "right_double", "left_green", "right_green")
    segs = []
    for r in rows:
        sig_ = tuple(r[k] for k in keys)
        if segs and segs[-1]["_sig"] == sig_:
            segs[-1]["s_end"] = r["s"] + STEP_M
            segs[-1]["landmarks"] |= set(r["landmarks"])
        else:
            segs.append({"_sig": sig_, "s_start": r["s"], "s_end": r["s"] + STEP_M, "landmarks": set(r["landmarks"]),
                         **{k: r[k] for k in keys}})
    changed = True
    while changed:
        changed = False
        for j in range(1, len(segs)):
            if segs[j]["s_end"] - segs[j]["s_start"] < MIN_SEG_M:
                segs[j - 1]["s_end"] = segs[j]["s_end"]
                segs[j - 1]["landmarks"] |= segs[j]["landmarks"]
                del segs[j]
                changed = True
                break
        for j in range(1, len(segs)):        # まとめた結果, 前後が同じ属性になったらつなぐ
            if segs[j]["_sig"] == segs[j - 1]["_sig"]:
                segs[j - 1]["s_end"] = segs[j]["s_end"]
                segs[j - 1]["landmarks"] |= segs[j]["landmarks"]
                del segs[j]
                changed = True
                break
    # 1 周は輪: 最初と最後の区間が同じ属性ならつなぐ (発進位置がカーブの途中にあるため)
    if len(segs) > 1 and segs[0]["_sig"] == segs[-1]["_sig"]:
        first = segs.pop(0)
        segs[-1]["s_end"] = round(total + first["s_end"], 1)
        segs[-1]["landmarks"] |= first["landmarks"]
    for k, sg in enumerate(segs):
        sg["id"] = k
        sg["length"] = round(sg["s_end"] - sg["s_start"], 1)
        sg["landmarks"] = sorted(sg["landmarks"])
        del sg["_sig"]

    # 手で直した区間 (segment_editor.py で保存したもの) があれば, その境目で区間を作り直す.
    # 各区間の属性は, 区間の中のステーションで一番多いもの
    seg_src = "auto"
    manual = Path(args.segments).expanduser() if args.segments else None
    if manual and manual.exists():
        from collections import Counter
        ms = sorted(json.load(open(manual))["segments"], key=lambda x: x["s_start"])
        segs = []
        for k, m in enumerate(ms):
            i0 = int(round(m["s_start"] / STEP_M)) % N
            i1 = int(round(ms[(k + 1) % len(ms)]["s_start"] / STEP_M)) % N
            n_ = (i1 - i0) % N or N
            rr = [rows[(i0 + j) % N] for j in range(n_)]
            sg = {key: Counter(r[key] for r in rr).most_common(1)[0][0] for key in keys}
            sg.update(s_start=round(i0 * STEP_M, 1), s_end=round(i0 * STEP_M + n_ * STEP_M, 1),
                      landmarks=sorted({l for r in rr for l in r["landmarks"]}), name=m.get("name", ""), note=m.get("note", ""))
            segs.append(sg)
        for k, sg in enumerate(segs):
            sg["id"] = k
            sg["length"] = round(sg["s_end"] - sg["s_start"], 1)
        seg_src = str(manual)
        print(f"区間は手で直したものを使う: {manual} ({len(segs)} 区間)")

    meta = {"source": {"drawing": str(drawing_path), "course_glb": str(glb_path)}, "direction": "counterclockwise (from start pose)",
            "lap_length_m": round(total, 1), "step_m": STEP_M, "lane_width_m": LANE_W, "curve_kappa": CURVE_KAPPA,
            "roles": "left = inner boundary, center, right = outer boundary (driving counterclockwise)",
            "field_fixes": [dict(role=f[0], drawing=f[1], actual=f[2], near=f[3], within_m=f[4], reason=f[5]) for f in FIELD_FIXES],
            "segments_source": seg_src}
    json.dump({"meta": meta, "segments": segs, "stations": rows},
              open(os.path.join(args.out, "course_topology.json"), "w"), ensure_ascii=False, indent=1)

    # 図: 図面に区間を色分けして番号を付ける
    vis = cv2.cvtColor(dw.img, cv2.COLOR_RGB2BGR).copy()
    rng = np.random.default_rng(3)
    for sg in segs:
        col = tuple(int(c) for c in rng.integers(40, 255, 3))
        ii = [i for i, r in enumerate(rows) if sg["s_start"] <= r["s"] < sg["s_end"]]
        pts = np.array([[rows[i]["x"] * PXM, -rows[i]["y"] * PXM] for i in ii], np.int32)
        cv2.polylines(vis, [pts], False, col, 4)
        m = pts[len(pts) // 2]
        cv2.putText(vis, str(sg["id"]), (int(m[0]) + 6, int(m[1]) - 6), 0, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, str(sg["id"]), (int(m[0]) + 6, int(m[1]) - 6), 0, 0.5, col, 1)
    u, v, _ = dw.px(loop[0])
    cv2.circle(vis, (int(u), int(v)), 6, (0, 0, 255), -1)
    u, v, _ = dw.px(sig)
    cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_STAR, 14, 2)
    cv2.imwrite(os.path.join(args.out, "course_topology.png"), vis[120:720, 460:1080])

    print(f"1 周 {total:.1f} m, 区間 {len(segs)} -> {args.out}/course_topology.json")
    for sg in segs:
        dbl = ("右二重 " if sg["right_double"] else "") + ("左二重 " if sg["left_double"] else "")
        grn = ("左緑 " if sg["left_green"] else "") + ("右緑" if sg["right_green"] else "")
        print(f"  {sg['id']:2d}  {sg['s_start']:6.1f}-{sg['s_end']:6.1f} m ({sg['length']:5.1f})  "
              f"{dict(straight='直線', left='左カーブ', right='右カーブ')[sg['shape']]:4s} "
              f"左:{TYPE_JA[sg['left']]} 中:{TYPE_JA[sg['center']]} 右:{TYPE_JA[sg['right']]}  {dbl}{grn}  "
              f"{' '.join(sg['landmarks'])}")


if __name__ == "__main__":
    main()
