#!/usr/bin/env python3
"""
topo_visual_motion.py - カメラ映像だけから, 前のコマからどれだけ前へ進んだかを測る (区間推定の「進み幅」に使う)

やり方:
  1. 画像の下側 (地面) を真上から見た図に直す (カメラの高さ 0.56 m と内部パラメータは車載と同じ. 0.85〜4.0 m 先, 2 cm/画素)
  2. 大きな明るさのむら (影, 逆光) を引き, 白飛び (太陽, フレア) を隠して, 細かい模様 (白線の縁, 横断歩道, 路面) だけにする
  3. 今のコマの手前側を, 前のコマで「どれだけ遠くにあったか」(0.04〜0.8 m) を総当たりで探す
  4. 一番よく合うずらし量が, 他のずらし量よりはっきり良いときだけ「測れた」とする
     (破線の端や横断歩道が見えているとき. 進む向きに伸びる実線だけでは前後のずれが分からない)
  ずらさない (0) ときの合い方 c0 も返す. 止まっていればここが一番よく合う

位置を足し合わせることはしない (1 コマごとの進み幅だけ). 天候や時間帯は, 0.2 秒前の自分の画像と比べるので関係しにくい.
"""

import math

import cv2
import numpy as np

# カメラ (車載 lane_detector_params.yaml と同じ. 640x360)
FX, CX, CY, H, CAMX, PIT = 733.26 / 3, 980.22 / 3, 516.63 / 3, 0.56, 0.055, math.radians(0.55)
RES = 0.02                      # 真上から見た図の 1 画素 [m]
X0, X1, YW = 0.85, 4.0, 2.4     # 使う地面の範囲: 前 0.85〜4.0 m, 左右 ±2.4 m
MAX_SHIFT = 40                  # 1 コマで探す最大の進み [画素] (0.8 m)
MIN_SHIFT = 2                   # 0 付近 (画像に居座る模様と区別できない) は探さない
SIDE = 10                       # 横のずれを探す幅 [画素]


def to_image(x, y):
    xr = x - CAMX
    zc = math.cos(PIT) * xr + math.sin(PIT) * H
    yd = -(math.sin(PIT) * xr - math.cos(PIT) * H)
    return CX + FX * (-y) / zc, CY + FX * yd / zc


BW, BH = int(2 * YW / RES), int((X1 - X0) / RES)
_PTS = [(X0, -1.5), (X0, 1.5), (X1, -1.5), (X1, 1.5)]
_HM = cv2.getPerspectiveTransform(np.float32([to_image(*p) for p in _PTS]),
                                  np.float32([((YW - y) / RES, (X1 - x) / RES) for x, y in _PTS]))
_VALID = cv2.erode((cv2.warpPerspective(np.full((360, 640), 255, np.uint8), _HM, (BW, BH)) > 0).astype(np.uint8),
                   np.ones((9, 9), np.uint8))


def prepare(bgr):
    """1 コマを, 真上から見た細かい模様の図とその有効範囲にする."""
    g = cv2.cvtColor(cv2.resize(bgr, (640, 360)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.dilate((g > 230).astype(np.uint8), np.ones((15, 15), np.uint8))
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    g = g - cv2.GaussianBlur(g, (0, 0), 6)
    b = cv2.warpPerspective(g, _HM, (BW, BH), flags=cv2.INTER_AREA)
    m = _VALID * cv2.warpPerspective(1 - sat, _HM, (BW, BH), flags=cv2.INTER_NEAREST)
    return (b * m).astype(np.float32), m


def measure(prev, cur):
    """前のコマ -> 今のコマで前へ進んだ量 dx [m] と, その確からしさ (sharp), ずらさないときの合い方 c0."""
    a, ma = prev
    b, mb = cur
    m = (ma * mb).astype(bool)
    if m.sum() < 2000:
        return None
    c0 = float(np.corrcoef(a[m], b[m])[0, 1])
    tpl = b[MAX_SHIFT:BH - 2, SIDE:BW - SIDE]
    if tpl.std() < 1e-3:
        return None
    res = cv2.matchTemplate(a[2:BH - 2, :], tpl, cv2.TM_CCOEFF_NORMED)
    s_of = MAX_SHIFT - 2 - np.arange(res.shape[0])          # res の行 -> 進み [画素]
    keep = s_of >= MIN_SHIFT
    sub, s_sub = res[keep], s_of[keep]
    r, c = np.unravel_index(int(np.argmax(sub)), sub.shape)
    best = float(sub[r, c])
    s = float(s_sub[r])
    if 0 < r < sub.shape[0] - 1:                             # 放物線で小数まで
        c_0, c_1, c_2 = sub[r - 1, c], sub[r, c], sub[r + 1, c]
        den = c_0 - 2 * c_1 + c_2
        if abs(den) > 1e-6:
            s += 0.5 * (c_0 - c_2) / den
    prof = sub[:, c]
    far = np.concatenate([prof[:max(0, r - 3)], prof[r + 4:]])
    sharp = float(best - np.median(far)) if len(far) else 0.0
    return dict(dx=round(s * RES, 3), best=round(best, 3), sharp=round(sharp, 3), c0=round(c0, 3))
