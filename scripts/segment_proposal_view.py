#!/usr/bin/env python3
"""
segment_proposal_view.py - 区間の分け方の案を, 今の区間と並べて図面の上に描く (議論用)

案の分け方 (0.5 m ごとのステーションの属性から):
  - カーブ / 直線
  - 左境界が破線か / 中央が破線か
  緑・二重線・横断歩道・信号では分けない (信号と横断歩道は区間に書くだけ).
  区間の境目は, 線の種類 (course.glb の線のかけらから決めたもの) と形が変わる所そのまま. 2 m 未満のかけらだけ隣へ.

使い方:
  python3 scripts/segment_proposal_view.py [出力 png]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS_DIR = Path(__file__).resolve().parent.parent
PXM = 5.711
MIN_SEG_M = 2.0            # これより短いかけら (形の境目と線の境目がわずかにずれてできたもの) だけ隣にまとめる
FONT = "/Library/Fonts/Arial Unicode.ttf"
SHAPE_JA = {"straight": "直線", "left": "左カーブ", "right": "右カーブ"}


def cross_of(r):
    return any(k in r["landmarks"] for k in ("crosswalk", "stop_line"))


def proposal(topo):
    """カーブ / 直線 と, 左境界が破線か, 中央が破線か で分ける. MIN_SEG_M より短い所は隣の長い方へまとめる."""
    st = topo["stations"]
    N, step = len(st), topo["meta"]["step_m"]
    s = np.arange(N) * step
    cross = np.array([any(k in r["landmarks"] for k in ("crosswalk", "stop_line")) for r in st])
    signal = np.array(["signal" in r["landmarks"] for r in st])
    # 横断歩道で線が切れている所 (なし) は, 手前の種類が続いているとみなす (横断歩道だけの短い区間を作らない)
    keys, prev = [], {}
    for r in st:
        kinds = {}
        for role in ("left", "center"):
            kinds[role] = prev.get(role, r[role]) if (r[role] == "none" and cross_of(r)) else r[role]
        prev = kinds
        keys.append(("curve" if r["shape"] != "straight" else "straight", kinds["left"] == "dashed", kinds["center"] == "dashed"))
    # 同じ属性の並び (1 周つながり) -> [key, 始めの番号, 長さ (ステーション数)]
    runs = []
    for i, k in enumerate(keys):
        if runs and runs[-1][0] == k:
            runs[-1][2] += 1
        else:
            runs.append([k, i, 1])
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1], runs[0][2] = runs[-1][1], runs[0][2] + runs[-1][2]
        runs.pop()

    def join_same():
        k = 0
        while len(runs) > 1 and k < len(runs):
            n2 = (k + 1) % len(runs)
            if runs[k][0] == runs[n2][0]:
                runs[k][2] += runs[n2][2]
                runs.pop(n2)
                k = 0
                continue
            k += 1

    while True:
        cand = [i for i, r in enumerate(runs) if r[2] * step < MIN_SEG_M]
        if not cand or len(runs) <= 1:
            break
        i = min(cand, key=lambda i: runs[i][2])
        a, b = (i - 1) % len(runs), (i + 1) % len(runs)
        if runs[a][2] >= runs[b][2]:
            runs[a][2] += runs[i][2]
        else:
            runs[b][1], runs[b][2] = runs[i][1], runs[b][2] + runs[i][2]
        runs.pop(i)
        join_same()
    # 発進位置 (s=0) の直後から始まる区間を 0 番にする
    runs.sort(key=lambda r: r[1])
    out = []
    lap = N * step
    for k, (key, i, n) in enumerate(runs):
        dashed = [n for n, f in (("左境界", key[1]), ("中央", key[2])) if f]
        label = ("カーブ" if key[0] == "curve" else "直線") + (" 破線: " + "・".join(dashed) if dashed else "")
        idx = [(i + j) % N for j in range(n)]
        notes = (["信号"] if signal[idx].any() else []) + (["横断歩道"] if cross[idx].any() else [])
        if notes:
            label += " (" + "・".join(notes) + ")"
        out.append(dict(id=k, start=i, n=n, s_start=s[i], s_end=(s[i] + n * step) % lap or lap, label=label))
    return out


JA = {"solid": "実線", "dashed": "破線", "none": "なし"}


def details(topo, sg):
    """区間の中身: 3 本の線の種類と長さ, 二重線, カーブの半径, 目印, 内側 / 外側レーンから破線が見えるか."""
    from collections import Counter
    st = topo["stations"]
    N, step = len(st), topo["meta"]["step_m"]
    rs = [st[(sg["start"] + j) % N] for j in range(sg["n"])]

    def comp(role):
        c = Counter(r[role] for r in rs)
        return "・".join(f"{JA[k]}{v * step:.1f}m" for k, v in c.most_common())

    curv = [abs(r["kappa"]) for r in rs if abs(r["kappa"]) >= 1 / 40]
    rad = f"半径 約{1 / np.median(curv):.0f} m" if len(curv) > len(rs) / 2 else "直線"
    dbl = [f"{name}{sum(r[k] for r in rs) * step:.1f}m" for name, k in (("左", "left_double"), ("右", "right_double"))
           if any(r[k] for r in rs)]
    lm = Counter(l for r in rs for l in r["landmarks"])
    lm_ja = {"signal": "信号", "crosswalk": "横断歩道", "stop_line": "停止線"}
    marks = "・".join(f"{lm_ja.get(k, k)}{v * step:.1f}m" for k, v in lm.items()) or "なし"
    # 内側レーンからは左境界と中央, 外側レーンからは中央と右境界が隣の線
    inner = [n for n, role in (("左境界", "left"), ("中央", "center")) if any(r[role] == "dashed" for r in rs)]
    outer = [n for n, role in (("中央", "center"), ("右境界", "right")) if any(r[role] == "dashed" for r in rs)]
    return [f"左境界 {comp('left')} / 中央 {comp('center')} / 右境界 {comp('right')}",
            f"{rad} / 二重線 {'・'.join(dbl) or 'なし'} / 目印 {marks}",
            f"隣に見える破線  内側レーン: {'・'.join(inner) or 'なし'}   外側レーン: {'・'.join(outer) or 'なし'}"]


def detail_table(topo, segs, cols, width):
    f_h, f_l = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 16)
    rows = [(sg, details(topo, sg)) for sg in segs]
    img = Image.new("RGB", (width, 40 + 92 * len(rows)), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((12, 8), "案の区間の詳細", font=ImageFont.truetype(FONT, 22), fill=(255, 255, 255))
    for k, ((sg, lines), col) in enumerate(zip(rows, cols)):
        y = 42 + 92 * k
        d.rectangle([12, y + 4, 30, y + 20], fill=col[::-1])
        d.text((40, y), f"{sg['id']}   {sg['s_start']:.1f} 〜 {sg['s_end']:.1f} m ({sg['n'] * 0.5:.1f} m)   {sg['label']}",
               font=f_h, fill=(255, 255, 255))
        for j, t in enumerate(lines):
            d.text((60, y + 24 + 21 * j), t, font=f_l, fill=(205, 205, 205))
    return img


def seg_colors(n):
    cols = []
    for k in range(n):
        h = (k * 0.61803) % 1.0
        c = cv2.cvtColor(np.uint8([[[int(h * 180), 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cols.append(tuple(int(v) for v in c))
    return cols


def seg_of_now(topo):
    st = topo["stations"]
    N, step = len(st), topo["meta"]["step_m"]
    out = []
    for sg in topo["segments"]:
        i = int(round(sg["s_start"] / step)) % N
        n = int(round((sg["s_end"] - sg["s_start"]) / step))
        lab = SHAPE_JA[sg["shape"]] + f" 左{ {'solid': '実', 'dashed': '破', 'none': '無'}[sg['left']]} 中{ {'solid': '実', 'dashed': '破', 'none': '無'}[sg['center']]}"
        out.append(dict(id=sg["id"], start=i, n=n, s_start=sg["s_start"], s_end=sg["s_end"], label=lab))
    return out


def panel(topo, segs, title, size=760, margin=46):
    xy = np.array([[r["x"], r["y"]] for r in topo["stations"]])
    N = len(xy)
    lo, hi = xy.min(0), xy.max(0)
    sc = (size - 2 * margin) / max(hi - lo)
    M = np.float32([[sc / PXM, 0, margin - lo[0] * sc], [0, sc / PXM, size - margin + lo[1] * sc]])
    dw = cv2.imread(topo["meta"]["source"]["drawing"])
    img = cv2.warpAffine(dw, M, (size, size), flags=cv2.INTER_AREA, borderValue=(40, 40, 40))
    img = (img.astype(np.float32) * 0.55).astype(np.uint8)
    px = np.array([[margin + (x - lo[0]) * sc, size - margin - (y - lo[1]) * sc] for x, y in xy]).astype(int)
    rng = np.random.default_rng(3)
    cols = []
    for k in range(len(segs)):
        h = (k * 0.61803) % 1.0
        c = cv2.cvtColor(np.uint8([[[int(h * 180), 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cols.append(tuple(int(v) for v in c))
    for sg, col in zip(segs, cols):
        idx = [(sg["start"] + k) % N for k in range(sg["n"] + 1)]
        cv2.polylines(img, [px[idx]], False, col, 7, cv2.LINE_AA)
        cv2.circle(img, tuple(px[sg["start"]]), 5, (0, 0, 0), -1)
        cv2.circle(img, tuple(px[sg["start"]]), 3, (255, 255, 255), -1)
    pil = Image.fromarray(img[..., ::-1])
    d = ImageDraw.Draw(pil)
    f_num, f_t = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 24)
    ctr = px.mean(0)
    for sg, col in zip(segs, cols):
        im = (sg["start"] + sg["n"] // 2) % N
        p = px[im].astype(float)
        v = p - ctr
        v /= max(np.linalg.norm(v), 1)
        q = p + v * 28
        d.text((q[0], q[1]), str(sg["id"]), font=f_num, fill=col[::-1], anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    d.text((12, 8), title, font=f_t, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    # 一覧
    lst = Image.new("RGB", (size, 30 + 24 * len(segs)), (24, 24, 24))
    dl = ImageDraw.Draw(lst)
    f_l = ImageFont.truetype(FONT, 17)
    for k, (sg, col) in enumerate(zip(segs, cols)):
        y = 10 + 24 * k
        dl.rectangle([12, y + 4, 30, y + 18], fill=col[::-1])
        dl.text((40, y), f"{sg['id']:2d}   {sg['s_start']:6.1f} 〜 {sg['s_end']:6.1f} m  ({sg['n'] * 0.5:5.1f} m)   {sg['label']}",
                font=f_l, fill=(230, 230, 230))
    out = Image.new("RGB", (size, size + lst.height), (24, 24, 24))
    out.paste(pil, (0, 0))
    out.paste(lst, (0, size))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(WS_DIR / "data/topological_map/segments_proposal.png")
    topo = json.load(open(WS_DIR / "data/topological_map/course_topology.json"))
    now = seg_of_now(topo)
    prop = proposal(topo)
    a = panel(topo, now, f"今の区間 ({len(now)} 区間)")
    b = panel(topo, prop, f"案の区間 ({len(prop)} 区間)")
    t = detail_table(topo, prop, seg_colors(len(prop)), a.width + b.width + 10)
    h = max(a.height, b.height)
    img = Image.new("RGB", (a.width + b.width + 10, h + 10 + t.height), (24, 24, 24))
    img.paste(a, (0, 0))
    img.paste(b, (a.width + 10, 0))
    img.paste(t, (0, h + 10))
    img.save(out_path)
    print(len(now), "->", len(prop), "区間 ->", out_path)
    for sg in prop:
        print(f"  {sg['id']:2d} {sg['s_start']:6.1f}-{sg['s_end']:6.1f} m ({sg['n'] * 0.5:5.1f}) {sg['label']}")


if __name__ == "__main__":
    main()
