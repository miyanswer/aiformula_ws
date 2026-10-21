#!/usr/bin/env python3
"""
build_truth.py - rosbag の各コマで, 車 (カメラ) が図面のどこにいたかの「正解」を作る (採点と表示だけに使う)

前の正解 (old_align_smooth.npy) は, オドメトリの軌跡を図面に合わせたあと, 1 コマごとに一番白線が重なる補正を
独立に選んでいた. そのため二重線のもう 1 本に合わせてしまったり (0.8 m 跳ぶ), 数 m 跳んだり, 道路の外に出たりした.

ここでは:
  1. 基準の姿勢 = オドメトリの軌跡 (gamepad_poses_refined.npy) を図面へ写したもの (gp_to_dwg.json, heading_corr.npy)
  2. 各コマで, 画像の白い画素を地面に投げ, 補正 (横 dx, dy ±SEARCH_M, 向き ±SEARCH_DEG) ごとに
     図面の外周 3 本 (course.glb から作った格子 answer_grid.npz) に重なる割合を出し, よい候補を CANDS 個残す
  3. 全コマの候補から, 「よく重なる」「補正が急に変わらない (オドメトリのずれはゆっくり)」
     「車が道路の中 (中央線から ROAD_HALF_M 以内) にいる」を同時に満たす並びを動的計画法で選ぶ
  4. 選んだ候補のまわりを細かく探して仕上げる
出力は前と同じ形 (t, x, y, 向き, 重なり, 根拠のコマまでの時間 [s], 根拠になったか) の npy.

使い方 (Mac 本体の python3. rosbag の読み出しは .venv の python に任せる):
  python3 scripts/build_truth.py rosbag/mcap/20261012_155111_gamepad.mcap data/truth/20261012_155111_gamepad
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, uniform_filter

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
from topo_features import iter_images  # noqa: E402

# カメラ (車載と同じ. 640x360)
FX, CX, CY = 733.26 / 3, 980.22 / 3, 516.63 / 3
CAM_HEIGHT, CAM_X, PITCH = 0.56, 0.055, math.radians(0.55)
FPS = 7.5                  # 正解を作る間隔 [コマ/秒] (区間推定は 5 コマ/秒)
SEARCH_M = 1.5             # 補正を探す横の範囲 [m]
SEARCH_DEG = 3.0           # 補正を探す向きの範囲 [度]
CANDS = 15                 # 1 コマで残す候補の数
HIT_M = 0.06               # 白い画素が図面の線からこの距離以内なら「重なった」
MOVE_COST = 2.0            # 補正が 1 コマで 1 m 変わるときの罰 (重なりの割合と同じ単位)
TURN_COST = 0.3            # 補正の向きが 1 度変わるときの罰
ROAD_HALF_M = 2.7          # 車が中央線からこれより離れたら道路の外 (車道 3.5 m + 白線 + 余裕)
OFFROAD_COST = 1.0         # 道路の外へ 1 m 出るごとの罰
GOOD_SCORE = 0.35          # これ以上重なったコマを「根拠」とする


def to_ground(u, v):
    s_, c_ = math.sin(PITCH), math.cos(PITCH)
    xc = (u - CX) / FX
    yc = (v - CY) / FX
    fwd = c_ - yc * s_
    down = s_ + yc * c_
    t = CAM_HEIGHT / np.maximum(down, 1e-6)
    return t * fwd, -t * xc, down > 0.03


def main():
    bag, d = sys.argv[1], Path(sys.argv[2])
    g = np.load(WS_DIR / "data/truth/answer_grid.npz")
    grid = g["grid"]
    X0, Y0, RES = float(g["X0"]), float(g["Y0"]), float(g["RES"])
    ny, nx = grid.shape
    DT = (distance_transform_edt(grid == 0) * RES).astype(np.float32)
    A = json.load(open(d / "gp_to_dwg.json"))
    Am, At = np.array(A["A"]), np.array([A["tx"], A["ty"]])
    rot, SC = math.atan2(Am[1, 0], Am[0, 0]), math.sqrt(abs(np.linalg.det(Am)))
    P = np.load(d / "gamepad_poses_refined.npy")
    yu = np.unwrap(P[:, 3])
    HC = np.load(d / "heading_corr.npy")
    topo = json.load(open(WS_DIR / "data/topological_map/course_topology.json"))
    cl = np.array([[r["x"], r["y"]] for r in topo["stations"]])          # 外周の中央線 (図面座標)

    def base_pose(t):
        x, y = np.interp(t, P[:, 0], P[:, 1]), np.interp(t, P[:, 0], P[:, 2])
        yaw = np.interp(t, P[:, 0], yu) + rot + np.interp(t, HC[:, 0], HC[:, 1])
        p = Am @ [x, y] + At
        return p[0] + CAM_X * SC * math.cos(yaw), p[1] + CAM_X * SC * math.sin(yaw), yaw

    def score(px, py, yaw, loc, dx, dy, da):
        yy = yaw + da
        c, s = np.cos(yy), np.sin(yy)
        wx = px + dx[:, None] + c[:, None] * loc[None, :, 0] - s[:, None] * loc[None, :, 1]
        wy = py + dy[:, None] + s[:, None] * loc[None, :, 0] + c[:, None] * loc[None, :, 1]
        ix = ((wx - X0) / RES).astype(np.int32)
        iy = ((wy - Y0) / RES).astype(np.int32)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        dd = np.full(ix.shape, 9.0, np.float32)
        dd[ok] = DT[iy[ok], ix[ok]]
        return (dd < HIT_M).mean(1)

    ax = np.arange(-SEARCH_M, SEARCH_M + 1e-6, 0.1)
    aa = np.radians(np.arange(-SEARCH_DEG, SEARCH_DEG + 1e-6, 0.5))
    DX, DY, DA = np.meshgrid(ax, ax, aa, indexing="ij")
    rng = np.random.default_rng(0)

    # ---- 1, 2: 各コマの候補
    frames = []
    last_t = -1e9
    for t, data in iter_images(bag):
        if t < P[0, 0] or t > P[-1, 0] or t - last_t < 1.0 / FPS - 1e-3:
            continue
        last_t = t
        px, py, yaw = base_pose(t)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.int16)
        brf = img.mean(2)
        bg = uniform_filter(brf, size=(9, 41))
        sat = img.max(2) - img.min(2)
        wv, wu = np.nonzero(((brf - bg) > 28) & (sat < 45))
        k = wv > 185
        gx, gy, ok = to_ground(wu[k].astype(float), wv[k].astype(float))
        k = ok & (gx > 1.5) & (gx < 8) & (np.abs(gy) < 6)
        loc = np.stack([gx[k], gy[k]], 1)
        cands = np.zeros((0, 4))
        if len(loc) >= 40:
            loc = loc[rng.choice(len(loc), min(300, len(loc)), replace=False)]
            sc = score(px, py, yaw, loc, DX.ravel(), DY.ravel(), DA.ravel()).reshape(DX.shape)
            peak = (sc == maximum_filter(sc, size=3, mode="nearest")) & (sc > 0.05)
            idx = np.argwhere(peak)
            vals = sc[peak]
            order = np.argsort(-vals)[:CANDS]
            cands = np.array([[DX[tuple(i)], DY[tuple(i)], DA[tuple(i)], v] for i, v in zip(idx[order], vals[order])])
        frames.append(dict(t=t, px=px, py=py, yaw=yaw, loc=loc, cands=cands))
        if len(frames) % 500 == 0:
            print(len(frames), "frames", flush=True)

    # ---- 3: 動的計画法 (候補のない / 少ないコマは「補正そのまま」の候補を 1 つ足す)
    def offroad(f, c):
        x, y = f["px"] + c[:, 0], f["py"] + c[:, 1]
        dist = np.min(np.hypot(cl[None, :, 0] - x[:, None], cl[None, :, 1] - y[:, None]), 1)
        return OFFROAD_COST * np.maximum(dist - ROAD_HALF_M, 0)

    HOLD = 5                   # 候補なし用に, 前のコマで点のよかった状態をこれだけ引き継ぐ
    states, best, back = [], [], []
    for i, f in enumerate(frames):
        c = f["cands"]
        if i:
            top = np.argsort(-best[-1])[:HOLD]
            hold = np.c_[states[-1][top, :3], np.zeros(len(top))]        # 前の補正のまま (重なり 0)
            c = np.vstack([c, hold]) if len(c) else hold
        elif not len(c):
            c = np.zeros((1, 4))
        gain = c[:, 3] - offroad(f, c)
        if i:
            a = states[-1]
            cost = MOVE_COST * np.hypot(c[:, None, 0] - a[None, :, 0], c[:, None, 1] - a[None, :, 1]) \
                + TURN_COST * np.degrees(np.abs(c[:, None, 2] - a[None, :, 2]))
            tot = best[-1][None, :] - cost
            j = np.argmax(tot, 1)
            best.append(tot[np.arange(len(c)), j] + gain)
            back.append(j)
        else:
            best.append(gain)
            back.append(np.zeros(len(c), int))
        states.append(c)
    n = len(frames)
    path = [int(np.argmax(best[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path = path[::-1]

    # ---- 4: 選んだ候補のまわりを細かく
    fine = np.arange(-0.1, 0.1001, 0.025)
    fa = np.radians(np.arange(-0.5, 0.501, 0.125))
    FX_, FY_, FA_ = (v.ravel() for v in np.meshgrid(fine, fine, fa, indexing="ij"))
    out = []
    for f, st, j in zip(frames, states, path):
        dx, dy, da, sc = st[j]
        if len(f["loc"]) >= 40:
            s2 = score(f["px"], f["py"], f["yaw"], f["loc"], dx + FX_, dy + FY_, da + FA_)
            k = int(np.argmax(s2))
            dx, dy, da, sc = dx + FX_[k], dy + FY_[k], da + FA_[k], float(s2[k])
        else:
            sc = 0.0
        out.append((f["t"], f["px"] + dx, f["py"] + dy, f["yaw"] + da, sc, len(f["loc"])))
    o = np.array(out)
    good = (o[:, 4] >= GOOD_SCORE) & (o[:, 5] >= 80)
    gi = np.where(good)[0]
    near = np.array([np.min(np.abs(o[gi, 0] - ti)) for ti in o[:, 0]]) if len(gi) else np.full(len(o), 99.0)
    res = np.column_stack([o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4], near, good])
    np.save(d / "truth.npy", res)

    step = np.hypot(*np.diff(res[:, 1:3], axis=0).T)
    dist = np.min(np.hypot(cl[None, :, 0] - res[:, None, 1], cl[None, :, 1] - res[:, None, 2]), 1)
    print(f"frames {n}  根拠 {good.mean() * 100:.1f}%  1 コマで 0.8 m 以上跳んだ {(step > 0.8).sum()} 回 (最大 {step.max():.2f} m)"
          f"  道路の外 (中央線から {ROAD_HALF_M} m 超) {(dist > ROAD_HALF_M).sum()} コマ  -> {d / 'truth.npy'}")


if __name__ == "__main__":
    main()
