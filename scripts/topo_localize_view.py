#!/usr/bin/env python3
"""
topo_localize_view.py - rosbag の走行映像と「今どの区間と推定しているか」を並べて再生する (表示だけ. 保存はしない)

左: カメラ映像. YOLOP で取った白線 (topo_features.py の結果) を重ねる (実線=白, 破線=水色, 不明=灰.
    種類は区間推定が使ったもの = 同じ線を数コマ続けて見て決めた種類)
    推定した横のずれから決めた各線の役割 (L = 左境界, C = 中央, R = 右境界, L2 / R2 = 二重線の外側) を線の横に書く
右: 公式図面 (aiformula_machine の drawing.png) の上にトポロジカルマップを重ねる. 区間ごとに色分けして番号を振り,
    推定の確率 (赤の濃さ) と推定位置 (赤丸. 横のずれも反映) を描く.
    --truth を渡すと図面に合わせた正解の位置 (緑丸) も描く (推定には使わない)
--road-roles: 路面 (honda_shihou_finetuned_road_v2_best.pth の走れる領域) を映像に紫で重ね, 各線の外の調べる所
    (前方 2.0〜5.0 m × 線の外 1.0〜2.0 m. 青 = 左, 橙 = 右) を点で描き, 車載の役割を路面で直した役割
    (road_role_corrector.py) を線の横に書く. --vehicle-roles と一緒に付けると 区間 + 路面 で直した役割
--no-localize: 区間推定をしない. 地図の図と区間推定の役割 (黄色の L/C/R) を出さず, 車載の役割と, それを路面だけで直した
    役割を出す (--road-roles も付く). 線は区間推定が滑らかにしたものではなく, そのコマの YOLOP の白線

推定は topo_localize_eval.py の TopoLocalizer をそのまま使う (地図は区間の並びだけ).
1 コマの進み幅は, 既定ではカメラ映像だけで測った進み (topo_visual_motion.py) を使う (--motion twist で前の方法).

使い方 (Mac 本体の python3. rosbag の読み出しは .venv の python に任せる):
  python3 scripts/topo_localize_view.py rosbag/mcap/20261012_155111_gamepad.mcap data/topological_map/features_155111.json \
      --start 266.8 [--known-start] [--truth <align_smooth.npy>] [--speed 2] [--vehicle-roles] [--road-roles]
  キー: スペース=一時停止, q / Esc=終了 (最後まで再生したら, キーを押すまで最後のコマを出しておく)
"""

import argparse
import colorsys
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
from topo_localize_eval import ADVANCE_MAX, TopoLocalizer  # noqa: E402

CAM_W, CAM_H = 640, 360
VIEW_H = 720                                  # 表示の高さ (カメラは 2 倍に拡大)
KIND_COLOR = {"solid": (255, 255, 255), "dashed": (255, 220, 0), "unknown": (150, 150, 150)}
ROLE_LABEL = {"left": "L", "center": "C", "right": "R", "left2": "L2", "right2": "R2"}
PXM = 5.711                                   # 図面 1 m あたりの画素数 (build_topological_map.py と同じ)

# カメラ (車載 lane_detector_params.yaml と同じ. topo_features.py と同じ値)
FX, CX, CY, H, CAMX, PIT = 733.26 / 3, 980.22 / 3, 516.63 / 3, 0.56, 0.055, math.radians(0.55)


def seg_code(sg):
    """区間の短い名前: 手で付けた名前の中の英数字の記号 (例 "L1の破線（L1-2）" -> "L1-2"). なければ番号."""
    m = re.findall(r"[A-Z]\d+(?:-\d+)?", sg.get("name", ""))
    return m[-1] if m else str(sg["id"])


def to_image(x, y):
    xr = x - CAMX
    zc = math.cos(PIT) * xr + math.sin(PIT) * H
    yd = -(math.sin(PIT) * xr - math.cos(PIT) * H)
    return CX + FX * (-y) / zc, CY + FX * yd / zc


class MapPanel:
    """下絵 (公式図面 + 区間の色と番号) を 1 回だけ作り, 毎コマ推定を上に描く."""

    def __init__(self, topo, loc, size=VIEW_H, margin=40):
        self.loc, self.topo = loc, topo
        xy = loc.sxy
        lo, hi = xy.min(0), xy.max(0)
        self.scale = (size - 2 * margin) / max(hi - lo)
        self.size, self.lo, self.hi, self.margin = size, lo, hi, margin
        self.px = np.array([self.to_px(x, y) for x, y in xy])
        n_seg = len(topo["segments"])
        self.seg_color = {}
        for sg in topo["segments"]:
            r, g, b = colorsys.hsv_to_rgb((sg["id"] * 0.61803) % 1.0, 0.55, 0.85)
            self.seg_color[sg["id"]] = (int(b * 255), int(g * 255), int(r * 255))
        # 公式図面を同じ縮尺・位置に貼る (図面の画素 (u, v) = (x, -y) * PXM). 区間の色が見えるよう少し暗くする
        base = np.full((size, size, 3), 30, np.uint8)
        dw = cv2.imread(topo["meta"]["source"]["drawing"])
        if dw is not None:
            a = self.scale / PXM
            M = np.float32([[a, 0, margin - lo[0] * self.scale], [0, a, size - margin + lo[1] * self.scale]])
            base = cv2.warpAffine(dw, M, (size, size), flags=cv2.INTER_AREA, borderValue=(30, 30, 30))
            base = (base.astype(np.float32) * 0.75).astype(np.uint8)
        for i in range(loc.N):
            j = (i + 1) % loc.N
            cv2.line(base, tuple(self.px[i]), tuple(self.px[j]), self.seg_color[int(loc.seg_of[i])], 5, cv2.LINE_AA)
        tan = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
        tan /= np.linalg.norm(tan, axis=1)[:, None]
        self.nl = np.stack([-tan[:, 1], tan[:, 0]], 1)        # 左向きの法線 (横のずれの向き)
        # 区間の境目に短い線, 中ほどに番号 (コースの外側へ少しずらす)
        ctr = self.px.mean(0)
        for sg in topo["segments"]:
            i0 = int(round(sg["s_start"] / loc.step)) % loc.N
            cv2.circle(base, tuple(self.px[i0]), 3, (0, 0, 0), -1)
            im = int(round(((sg["s_start"] + sg["s_end"]) / 2) / loc.step)) % loc.N
            p = self.px[im].astype(float)
            d = p - ctr
            d /= max(np.linalg.norm(d), 1)
            q = (p + d * 26).astype(int)
            txt = seg_code(sg)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(base, txt, (q[0] - tw // 2, q[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(base, txt, (q[0] - tw // 2, q[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.seg_color[sg["id"]], 2, cv2.LINE_AA)
        cv2.putText(base, f"lap {topo['meta']['lap_length_m']:.0f} m, {n_seg} segments, counterclockwise",
                    (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        self.base = base

    def to_px(self, x, y):
        u = self.margin + (x - self.lo[0]) * self.scale
        v = self.size - self.margin - (y - self.lo[1]) * self.scale        # 北を上に
        return int(round(u)), int(round(v))

    def draw(self, est, truth_xy=None):
        img = self.base.copy()
        p = self.loc.ps
        pm = p.max()
        if pm > 0:
            for i in np.nonzero(p > pm * 0.05)[0]:
                a = p[i] / pm
                cv2.circle(img, tuple(self.px[i]), 2 + int(4 * a), (0, 0, int(80 + 175 * a)), -1)
        if truth_xy is not None:
            cv2.circle(img, self.to_px(*truth_xy), 9, (0, 220, 0), 2, cv2.LINE_AA)
        e = self.to_px(*(self.loc.sxy[est] + self.nl[est] * self.loc.offset()))
        cv2.circle(img, e, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, e, 8, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def move_text(loc):
    """このコマの進み幅をどう決めたか (camera = 映像で測れた, hold = 直前に測れた進みを使い続け, stopped, unknown)."""
    if loc.adv_mode == "visual":
        return f"camera {loc.vo_dx:.2f} m/frame"
    if loc.adv_mode == "hold":
        return f"hold {loc.vo_dx:.2f} m/frame"
    return {"stopped": "stopped", "uniform": "unknown (0-0.5 m)"}.get(loc.adv_mode, "-")


ROLE_SHORT = {"left": "L", "center": "C", "right": "R", "left2": "L2", "right2": "R2", None: "-"}


def draw_road(img, da, lines):
    """路面 (0/1, CAM_W x CAM_H) を紫で重ね, 各線の外の調べる所 (road_role_corrector の NEAR_X x OUT_D) を点で描く."""
    from road_role_corrector import NEAR_X, OUT_D
    m = da > 0
    img[m] = (0.6 * img[m] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    for l in lines:
        if not l.get("c"):
            continue
        c0, c1, c2 = l["c"]
        for x in np.arange(NEAR_X[0], NEAR_X[1] + 1e-6, 0.5):
            yl = c0 + c1 * x + c2 * x * x
            for side, col in ((1, (255, 128, 0)), (-1, (0, 128, 255))):
                for d in np.arange(OUT_D[0], OUT_D[1] + 1e-6, 0.2):
                    u, v = to_image(x, yl + side * d)
                    if 0 <= u < CAM_W and 0 <= v < CAM_H:
                        cv2.circle(img, (int(u), int(v)), 1, col, -1)


def draw_lines(img, lines, roles, veh_roles=None, fixed_roles=None):
    """線と役割を描く. veh_roles があれば, 線の横に 車載の役割 (V:) と, 直した役割 (->) を書く."""
    for k, (l, role) in enumerate(zip(lines, roles)):
        if not l.get("c"):
            continue
        c0, c1, c2 = l["c"]
        pts = []
        for x in np.arange(max(l["x0"], 0.9), min(l["x1"], 12.0), 0.2):
            u, v = to_image(x, c0 + c1 * x + c2 * x * x)
            if 0 <= u < CAM_W and 0 <= v < CAM_H:
                pts.append((int(u), int(v)))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts)], False, KIND_COLOR.get(l["kind"], (150, 150, 150)), 2, cv2.LINE_AA)
            u, v = pts[0]
            txt = ROLE_LABEL.get(role, "?")
            u = min(max(u, 4), CAM_W - 30)
            cv2.putText(img, txt, (u, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (u, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if role else (150, 150, 150), 2, cv2.LINE_AA)
            if veh_roles is not None and (veh_roles[k] is not None or fixed_roles[k] is not None):
                vr, fr = veh_roles[k], fixed_roles[k]
                t2 = f"V:{ROLE_SHORT[vr]}" + ("" if fr == vr else f"->{ROLE_SHORT[fr]}")
                col = (255, 255, 255) if fr == vr else (0, 0, 255)
                cv2.putText(img, t2, (u, v + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, t2, (u, v + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("features", help="topo_features.py の出力 (同じ rosbag から作ったもの)")
    ap.add_argument("--topology", default=str(WS_DIR / "data/topological_map/course_topology.json"))
    ap.add_argument("--truth", default="", help="正解の位置の npy (表示だけ. 推定には使わない)")
    ap.add_argument("--start", type=float, default=0.0, help="rosbag の先頭から何秒目から推定と再生を始めるか")
    ap.add_argument("--known-start", action="store_true", help="始めの位置を発進位置 (s=0) とする")
    ap.add_argument("--advance-max", type=int, default=1, help=f"1 コマで進める最大のステーション数 (評価の既定 {ADVANCE_MAX})")
    ap.add_argument("--motion", default="visual", choices=["visual", "twist"], help="1 コマの進み幅の決め方")
    ap.add_argument("--vehicle-roles", action="store_true",
                    help="車載の役割 (rosbag の記録. なければ車載と同じ LineTracker で作る) と, 区間の情報で直した役割も描く")
    ap.add_argument("--road-roles", action="store_true",
                    help="路面を重ね, 車載の役割を路面で直した役割を描く (--vehicle-roles と一緒なら 区間 + 路面)")
    ap.add_argument("--no-localize", action="store_true",
                    help="区間推定をせず, 車載の役割を路面だけで直した役割を描く (地図の図は出さない)")
    ap.add_argument("--speed", type=float, default=1.0, help="再生の速さ (1 = 実時間)")
    args = ap.parse_args()
    if args.no_localize:
        args.road_roles, args.vehicle_roles = True, False     # 区間の情報で直すには区間推定がいる

    topo = json.load(open(args.topology))
    loc = TopoLocalizer(topo, args.known_start, args.advance_max, motion=args.motion)
    seg_by_id = {sg["id"]: sg for sg in topo["segments"]}
    panel = MapPanel(topo, loc)
    all_rows = json.load(open(args.features))
    rows = [r for r in all_rows if r["rt"] >= args.start]
    veh_of = corr = road_net = None
    if args.vehicle_roles or args.road_roles:
        import role_assignment_eval as RA
        from road_role_corrector import RoadRoleCorrector
        from role_corrector import RoleCorrector
        corr = RoleCorrector(topo) if args.vehicle_roles else None
        if args.road_roles:
            import torch
            import make_road_masks as M
            import verify_lane_rosbag as V
            from topo_features import ROAD_MODEL
            road_dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            road_net = M.load_net(ROAD_MODEL, road_dev)
            corr = RoadRoleCorrector(corr)
        fix_by = {(True, False): "segment info", (False, True): "road", (True, True): "segment info + road"}[
            (args.vehicle_roles, args.road_roles)]
        rec = RA.recorded_roles(Path(args.bag))
        if rec:
            vt = np.array([v["t"] for v in rec])

            def veh_of(r):
                i = int(np.clip(np.searchsorted(vt, r["t"]), 0, len(vt) - 1))
                return {k: v for k, v in rec[i].items() if k != "t"} if abs(vt[i] - r["t"]) < 0.15 else {}
        else:
            sim = RA.tracker_roles(all_rows)

            def veh_of(r):
                return sim.get(r["t"], {})
    if not rows:
        raise SystemExit("--start 以降のフレームがありません")
    T = np.load(args.truth) if args.truth else None

    reader = subprocess.Popen([str(WS_DIR / ".venv/bin/python"), str(WS_DIR / "scripts/topo_features.py"), "--reader",
                               os.path.abspath(args.bag)], stdout=subprocess.PIPE)
    inp = reader.stdout
    win = "topological localization (space: pause, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    k = 0
    wall0, rt0 = None, rows[0]["rt"]
    quit_by_key = False
    try:
        while k < len(rows):
            b = inp.read(4)
            if len(b) < 4:
                break
            head = json.loads(inp.read(struct.unpack("<I", b)[0]))
            data = inp.read(struct.unpack("<I", inp.read(4))[0])
            if head["k"] != "img" or abs(head["t"] - rows[k]["t"]) > 2e-3:
                continue
            r = rows[k]
            k += 1
            est = None if args.no_localize else loc.update(r)
            cam = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if road_net is not None:
                cut = int(cam.shape[0] * V.TOP_CUT_RATIO)          # 車載と同じく上を切って推論
                da = np.zeros(cam.shape[:2], np.uint8)
                da[cut:] = M.infer_da(road_net, road_dev, cam[cut:])
            cam = cv2.resize(cam, (CAM_W, CAM_H))
            # 線は区間推定が使ったもの (前後のコマで滑らかにした位置と, 数コマ続けて見て決めた種類)
            shown = r["lines"] if args.no_localize else (loc.lines or r["lines"])
            vr = fr = None
            n_fix = 0
            if road_net is not None:
                draw_road(cam, cv2.resize(da, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST), r["lines"])
            if veh_of is not None:
                veh = veh_of(r)
                fixed, notes = corr.correct(loc, est, veh, r) if args.road_roles else corr.correct(loc, est, veh)
                n_fix = len(notes)
                vr, fr = [], []
                for l in shown:                      # 描く線 (区間推定が滑らかにした線) ごとに, 近い車載の線の役割
                    m = [role for role, y in veh.items() if abs(y - l["y2"]) < 0.35]
                    vr.append(m[0] if m else None)
                    # 直した役割: 車載の線の位置のまま役割が変わったもの, または別の線 (二重線の内側) に移ったもの
                    f = [role for role, y in fixed.items()
                         if (m and abs(y - veh[m[0]]) < 1e-6) or (not m and abs(y - l["y2"]) < 0.35)]
                    fr.append(f[0] if f else None)
            # 線の上の役割: 区間推定なしなら 路面で直した役割, ありなら 区間推定のレーンから付けた役割
            draw_lines(cam, shown, fr if args.no_localize else loc.roles_of(shown, est), vr, fr)
            cam = cv2.resize(cam, (VIEW_H * CAM_W // CAM_H, VIEW_H))

            if args.no_localize:
                info = [f"t={r['rt']:6.1f}s  v={abs(r['v']):.1f} m/s",
                        "no localization: roles = vehicle (LineTracker) fixed by road only",
                        f"vehicle roles fixed: {n_fix}" + ("  (red = fixed by road)" if n_fix else "")]
            else:
                sid = int(loc.seg_of[est])
                sg = seg_by_id[sid]
                truth_xy, tsid = None, None
                if T is not None:
                    i = int(np.clip(np.searchsorted(T[:, 0], r["t"]), 0, len(T) - 1))
                    truth_xy = (T[i, 1], T[i, 2])
                    d = np.hypot(loc.sxy[:, 0] - T[i, 1], loc.sxy[:, 1] - T[i, 2])
                    j = int(np.argmin(d))
                    if d[j] < 5.0 and T[i, 5] <= 1.0:
                        tsid = int(loc.seg_of[j])
                mp = panel.draw(est, truth_xy)

                # 文字 (カメラ映像の上): 推定区間と, その区間の図面上の特徴, 今の手がかり
                info = [f"t={r['rt']:6.1f}s  v={abs(r['v']):.1f} m/s",
                        f"estimate: {seg_code(sg)}  (s={loc.s_arr[est]:.1f} m, p={loc.ps[est] * 100:.0f}%)"
                        f"  lane {'inner' if loc.offset() > 0 else 'outer'} ({loc.offset():+.2f} m)",
                        f"  {sg['shape']}  L:{sg['left']}  C:{sg['center']}  R:{sg['right']}"
                        + ("  " + ",".join(sg["landmarks"]) if sg["landmarks"] else ""),
                        "cue: curvature " + (f"{loc.kobs:+.3f}" if loc.kobs is not None else "-")
                        + "  signal " + (f"{loc.sig_d:.1f} m" if loc.sig_d is not None else "-"),
                        "move: " + move_text(loc)]
                if veh_of is not None:
                    info.append(f"vehicle roles fixed: {n_fix}" + (f"  (red = fixed by {fix_by})" if n_fix else ""))
                if T is not None:
                    info.append(f"truth: {seg_code(seg_by_id[tsid]) if tsid is not None else '-'}"
                                + ("" if tsid is None else ("  OK" if tsid == sid else ("  near" if abs(tsid - sid) <= 1 else "  NG"))))
            y = 34
            for s in info:
                cv2.putText(cam, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(cam, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                y += 34
            cv2.imshow(win, cam if args.no_localize else np.hstack([cam, mp]))

            # 実時間 (x speed) に合わせて待つ
            if wall0 is None:
                wall0 = time.time()
            wait = (r["rt"] - rt0) / args.speed - (time.time() - wall0)
            key = cv2.waitKey(max(1, int(wait * 1000))) & 0xFF
            if key in (ord("q"), 27):
                quit_by_key = True
                break
            if key == ord(" "):
                t_pause = time.time()
                while (cv2.waitKey(50) & 0xFF) != ord(" "):
                    pass
                wall0 += time.time() - t_pause
        if not quit_by_key and k:
            # 最後まで再生したら, 最後のコマを出したまま待つ (すぐ閉じると見逃す)
            print("最後まで再生しました. 何かキーを押すと閉じます", flush=True)
            cv2.waitKey(0)
    finally:
        reader.kill()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
