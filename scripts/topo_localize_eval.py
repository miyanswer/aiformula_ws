#!/usr/bin/env python3
"""
topo_localize_eval.py - トポロジカルマップ (build_topological_map.py) の上で「今どの区間か」を追い, 正解と比べる

位置の持ち方: 1 周を 0.5 m ごとの位置 (ステーション) に分け, さらに「道路の中央線から車が左右にどれだけずれているか」
  (OFFSETS. 左が正) を組にして, 各組にいる確率を持つ (隠れマルコフモデル).
  - 進み方 (--motion):
      visual (既定): カメラ映像だけで測った 1 コマの進み (topo_visual_motion.py) が測れたコマはその量 ± VO_SIGMA だけ進め,
        測れないコマは直前に測れた進みを VO_HOLD 秒まで使い続ける. それもなければ下の twist と同じ.
        位置は足し合わせない (毎コマの進み幅としてだけ使う)
      twist: 1 コマ (0.2 s) で 0〜ADVANCE_MAX 個先へ進む (同じ確率). rosbag の車速は「動いているか」だけに使う.
        (rosbag の車速は実際より小さく, 動いているのに 0 の時間もある)
    横のずれは 1 コマで隣へ少し広がる (レーン移動を許す)
  - 手がかり (その瞬間の値だけ):
      白線の役割: 見えている白線ごとに, 車の横のずれから「図面の左境界 / 中央 / 右境界 (/ 二重線の外側) のどれか」を決め,
        その位置に線があるはずか (位置の近さ) と, その役割の線の種類 (実線 / 破線) が図面と合うかを見る.
        左の破線 1 本しか見えていなくても, 横のずれ (前のコマから引き継ぐ) から左境界か中央かが決まり, 区間を絞れる
        線は前後のコマで追いかけ (LineKindTracker), 種類は数コマ (6.5 m 進む間) 続けて見て決め,
        位置と曲がり具合は前後のコマで滑らかにする. 横断歩道・停止線の前後 3 m では種類を使わず位置だけを使う
      白線の曲がり具合 (前後 3 m 以上見えている線の, 3 m 先での曲率の中央値) と, 図面の 3 m 先の曲率との近さ
        (車の旋回の速さは, 6 レーン走行のレーン移動で大きく揺れてコースの形を表さないので使わない)
      信号: 車載と同じ検出器の枠の高さから距離 d を出し, 「信号の位置から d 手前」に強く寄せる (位置の較正).
        見えていないときは, 信号の 25 m 手前〜2 m 先である確率を少し下げる
  - 出力: 一番確率の高い位置 -> 区間. 正解 (図面に合わせた位置) があれば区間が合っていた割合を出す

使い方:
  python3 scripts/topo_localize_eval.py data/topological_map/features_134520.json [--truth <align_smooth.npy>] [--known-start]
      [--no-roles] [--no-signal-dist]   (比較用: 白線の役割 / 信号の距離 の手がかりを使わない)
"""

import argparse
import json
from pathlib import Path

import numpy as np

WS_DIR = Path(__file__).resolve().parent.parent
ADVANCE_MAX = 2            # 1 コマで進める最大のステーション数 (0.5 m 単位. 5 コマ/秒で 1 m = 5 m/s まで)
KAPPA_SIGMA = 0.02         # 曲がり具合の観測のばらつき [1/m]
KAPPA_AHEAD = 3.0          # 白線の曲がり具合を測る距離 (車の前 3 m) [m]
SIGNAL_BEFORE, SIGNAL_AFTER = 25.0, 2.0
OFFSETS = np.round(np.arange(-2.7, 2.71, 0.45), 2)   # 車の横のずれ [m] (道路の中央線から, 左が正)
OFFSET_KERNEL = (0.15, 0.7, 0.15)                    # 動いているとき 1 コマで横のずれが隣へ移る確率
DOUBLE_GAP = 0.8           # 二重線の外側の線は境界から 0.8 m 外 (build_topological_map.py と同じ)
LINE_SIGMA = 0.4           # 白線の横位置の観測のばらつき [m]
LINE_CLUTTER = 0.15        # どの役割にも当たらない線 (誤検出, 縁石など) が出る割合
# 種類の合いやすさ: KIND_LIKE[見えた種類][図面の種類]
KIND_LIKE = {"solid": {"solid": 1.0, "dashed": 0.3}, "dashed": {"solid": 0.1, "dashed": 1.0}}
# 映像で測った進み (topo_features.py --add-motion) の使い方
VO_SHARP = 0.2             # これより確からしいときだけ「測れた」とする
VO_MIN_DX = 0.07           # これより小さい進みは, 止まっている / 画像に居座る模様と区別できないので使わない [m]
VO_SIGMA = 0.1             # 測れたコマの進みのばらつき [m]
VO_HOLD = 3.0              # 測れないコマで, 直前に測れた進みを使い続ける時間 [s]
VO_HOLD_SIGMA = 0.15       # そのときの進みのばらつき [m]
VO_STOP_C0 = 0.93          # rosbag の車速が 0 で, ずらさないときの一致がこれ以上なら止まっている

# 白線の種類を数コマ続けて決める (中央の破線は 1 本 3.6 m・途切れ 2.8 m と長く, 1 コマでは実線に見えやすい)
KIND_WINDOW = 6.5          # これだけ進む間 (中央の破線の 1 周期) ずっと白なら実線 [m]
KIND_GAP = 6               # 手前の白さの並びで, これだけ (0.1 m 単位) 続けて白くなければ途切れ
GAP_MIN_FRAMES = 2         # 途切れがこれだけのコマで見えたら破線 (横断歩道などの 1 回きりの切れ目で破線にしない)
KIND_IGNORE_M = 3.0        # 横断歩道・停止線からこの距離 [m] 以内では白線の種類を使わない
KIND_TRACK_GATE = 0.6      # 前のコマの線と同じ線とみなす横位置の差 [m]
KIND_TRACK_KEEP = 1.0      # 見えなくなった線を覚えておく時間 [s]
ADV_UNKNOWN = 0.25         # 進みが分からないコマで仮に進んだとみなす量 [m]
# 同じ白線の位置と曲がり具合を前後のコマで滑らかにする (1 コマごとの検出は揺れる. 特に曲がり具合)
SMOOTH_Y = 0.5             # 横位置: 予測 (前の位置 + 横の速さ x 時間) と今の検出を混ぜる割合 (1 = 今の検出だけ)
SMOOTH_VY = 0.2            # 横の速さ (レーン移動で線が横へ動く) を直す割合
SMOOTH_C = 0.5             # 向きと曲がり具合 (c1, c2) を今の検出へ寄せる割合
SMOOTH_VY_MAX = 3.0        # 横の速さの上限 [m/s]


def has_gap(near, x0):
    """手前の白さの並びに, 白線の途切れ (KIND_GAP 以上続けて白くない所) があるか.
    途切れとみなすのは, (1) YOLOP が線を見つけ始める所 x0 より手前にある (線の手前の端がまだ来ていない. 中央の破線の
    2.8 m の途切れはこう見える) か, (2) 前後どちらにも白い所がある とき. 画面の端や曲がりで白さを取り損ねた所
    (YOLOP は線を見ているのに白くない) は途切れにしない."""
    bits = near["bits"]
    i = 0
    while i < len(bits):
        if bits[i] != "0":
            i += 1
            continue
        j = i
        while j < len(bits) and bits[j] == "0":
            j += 1
        if j - i >= KIND_GAP:
            before_x0 = near["x_a"] + 0.1 * j <= x0 + 0.1
            inside = "111" in bits[:i] and "111" in bits[j:]
            if before_x0 or inside:
                return True
        i = j
    return False


class LineKindTracker:
    """同じ白線を数コマ追いかける.
      - 種類: 手前 (画面に入る所から 2.5 m) を途切れが通ったか で実線 / 破線を決める.
        1 コマで長く (KIND_WINDOW 以上) 見えている線は, そのコマの判定 (topo_features.py の kind) をそのまま使う
      - 位置: 2 m 先の横位置は「前の位置 + 横の速さ x 時間」の予測と今の検出を混ぜ (レーン移動にもついていく),
        向きと曲がり具合 (c1, c2) は前のコマとの間をとる (smooth=False なら今の検出のまま)"""

    def __init__(self, smooth=True):
        self.tracks = []          # {y, vy, c, t, hist: [(進んだ距離の累計, 途切れがあったか, 全部白か)]}
        self.travel = 0.0
        self.smooth = smooth

    def update(self, t, lines, adv):
        """各線の種類と, 滑らかにした線 (y2, c を置き換えたもの) を返す."""
        self.travel += adv
        self.tracks = [tr for tr in self.tracks if t - tr["t"] < KIND_TRACK_KEEP]
        kinds, out = [], []
        used = set()
        for l in lines:
            near = l.get("near")
            k_frame = l["kind"]
            span = min(l["x1"], 9.0) - max(l["x0"], 1.0)
            # 前のコマの同じ線 (横の速さで今の位置を予測して比べる)
            best, bd = None, KIND_TRACK_GATE
            for i, tr in enumerate(self.tracks):
                y_pred = tr["y"] + tr["vy"] * (t - tr["t"])
                if i not in used and abs(y_pred - l["y2"]) < bd:
                    best, bd = i, abs(y_pred - l["y2"])
            if best is None:
                self.tracks.append(dict(y=l["y2"], vy=0.0, c=list(l["c"]), t=t, hist=[]))
                best = len(self.tracks) - 1
            used.add(best)
            tr = self.tracks[best]
            dt = t - tr["t"]
            if self.smooth and dt > 1e-3:
                y_pred = tr["y"] + tr["vy"] * dt
                e = l["y2"] - y_pred
                tr["y"] = y_pred + SMOOTH_Y * e
                tr["vy"] = float(np.clip(tr["vy"] + SMOOTH_VY * e / dt, -SMOOTH_VY_MAX, SMOOTH_VY_MAX))
                c1 = (1 - SMOOTH_C) * tr["c"][1] + SMOOTH_C * l["c"][1]
                c2 = (1 - SMOOTH_C) * tr["c"][2] + SMOOTH_C * l["c"][2]
                tr["c"] = [tr["y"] - 2 * c1 - 4 * c2, c1, c2]       # 2 m 先で滑らかにした横位置を通る
            else:
                tr["y"], tr["c"] = l["y2"], list(l["c"])
            tr["t"] = t
            out.append(dict(l, y2=tr["y"], c=tr["c"]))
            if near and len(near["bits"].replace("-", "")) >= 10:
                tr["hist"].append((self.travel, has_gap(near, l["x0"]), near["bits"].count("0") <= 2))
            tr["hist"] = [h for h in tr["hist"] if self.travel - h[0] <= KIND_WINDOW + 0.5]
            h = tr["hist"]
            if k_frame == "dashed" or sum(x[1] for x in h) >= GAP_MIN_FRAMES:
                kind = "dashed"
            elif k_frame == "solid" and span >= KIND_WINDOW:
                kind = "solid"
            elif len(h) >= 5 and self.travel - h[0][0] >= KIND_WINDOW and all(x[2] for x in h):
                kind = "solid"
            else:
                kind = "unknown"
            kinds.append(kind)
        return kinds, out


# 信号の距離 = SIGNAL_K / (枠の高さ / 画像の高さ). SIGNAL_S0 = 距離 0 になる位置 [m] (図面の信号の位置から).
# どちらも正解の位置で較正した値 (topo_localize_eval.py --calibrate-signal)
SIGNAL_K = 0.3026
SIGNAL_S0 = -0.78


def signal_distance(r):
    """そのコマで一番近い信号までの距離 [m] (検出がなければ None)."""
    ds = [SIGNAL_K / (d["h"] / d["H"]) for d in r.get("tl", []) if d["h"] > 0]
    ds = [d for d in ds if d < 40]
    return min(ds) if ds else None


class TopoLocalizer:
    """トポロジカルマップの上で, (ステーション, 横のずれ) にいる確率を 1 コマずつ更新する (評価と表示の道具で共通)."""

    def __init__(self, topo, known_start=False, advance_max=ADVANCE_MAX, use_speed=False, roles=True, signal_dist=True,
                 motion="visual", temporal_kind=True, smooth=True):
        st = topo["stations"]
        self.N = N = len(st)
        self.step = step = topo["meta"]["step_m"]
        self.lap = topo["meta"]["lap_length_m"]
        W = topo["meta"]["lane_width_m"]
        self.sxy = np.array([[r["x"], r["y"]] for r in st])
        self.kap = np.array([r["kappa"] for r in st])
        # 見えている白線の曲がり具合は KAPPA_AHEAD m 先で測るので, 図面も同じだけ先の曲率と比べる
        self.kap_ahead = np.roll(self.kap, -int(round(KAPPA_AHEAD / topo["meta"]["step_m"])))
        self.s_arr = s_arr = np.arange(N) * step
        sig_idx = [i for i, r in enumerate(st) if "signal" in r["landmarks"]]
        self.s_sig = None
        self.near_sig = np.zeros(N, bool)
        if sig_idx:
            self.s_sig = s_arr[sig_idx].mean()
            self.near_sig = (s_arr >= self.s_sig - SIGNAL_BEFORE) & (s_arr <= self.s_sig + SIGNAL_AFTER)
        self.seg_of = np.zeros(N, int)
        for sg in topo["segments"]:
            for i in range(N):
                ss = s_arr[i] if s_arr[i] >= sg["s_start"] else s_arr[i] + self.lap
                if sg["s_start"] <= ss < sg["s_end"]:
                    self.seg_of[i] = sg["id"]
        # 各ステーションで線がある横位置 (道路の中央線から, 左が正) と, その線の役割・種類
        self.role_names = ["left", "center", "right", "left2", "right2"]
        self.role_pos = np.array([W, 0.0, -W, W + DOUBLE_GAP, -W - DOUBLE_GAP])
        self.role_type = np.full((N, 5), "", object)        # "" = その位置に線はない
        for i, r in enumerate(st):
            for k, name in enumerate(["left", "center", "right"]):
                if r[name] in ("solid", "dashed"):
                    self.role_type[i, k] = r[name]
            if r["left_double"]:
                self.role_type[i, 3] = "solid"
            if r["right_double"]:
                self.role_type[i, 4] = "solid"
        # 見えた種類ごとの「種類の合いやすさ」を先に表にしておく (N, 5).
        # 横断歩道・停止線の前後 KIND_IGNORE_M では種類を使わない (図面は縞を線の種類と読み違え,
        # 実際の線も横断歩道で切れるので数コマの判定も破線になる). 線の位置だけを使う
        mark = np.array([any(k in r["landmarks"] for k in ("crosswalk", "stop_line")) for r in st])
        w = int(round(KIND_IGNORE_M / step))
        self.kind_ignore = np.convolve(np.r_[mark[N - w:], mark, mark[:w]].astype(float), np.ones(2 * w + 1), "valid") > 0
        self.kind_like = {kind: np.where(self.kind_ignore[:, None], 1.0,
                                         np.array([[tab.get(t, 1.0) for t in row] for row in self.role_type]))
                          for kind, tab in KIND_LIKE.items()}
        self.O = len(OFFSETS)
        self.p = np.zeros((self.O, N))
        if known_start:
            self.p[:, 0] = 1.0 / self.O
        else:
            self.p[:] = 1.0 / (self.O * N)
        self.ps, self.po = self.p.sum(0), self.p.sum(1)
        self.kernel = np.ones(advance_max + 1) / (advance_max + 1)
        self.use_speed, self.roles, self.use_signal_dist = use_speed, roles, signal_dist
        self.prev_t = None
        self.motion = motion
        self.vo_dx, self.vo_t = None, -1e9                    # 直前に映像で測れた進み [m/コマ] と時刻
        self.adv_mode = ""
        self.adv_m = 0.0                                      # このコマで進んだとみなした量 [m] (線の種類の判定用)
        self.kind_tracker = LineKindTracker(smooth) if (temporal_kind or smooth) else None
        self.lines = []
        self.temporal_kind = temporal_kind
        self.line_kinds = []
        self.kobs = self.sig_d = None
        self.dash_obs = False

    def line_like(self, y, kind):
        """横位置 y [m] (車から, 左が正) に種類 kind の線が見えたときの, (横のずれ, ステーション) ごとの尤度."""
        pos = y + OFFSETS[:, None]                            # 道路の中央線からの位置 (横のずれごと)  (O, 1)
        like = np.full((self.O, self.N), LINE_CLUTTER)
        kl_all = self.kind_like.get(kind)
        for k in range(5):
            has = self.role_type[:, k] != ""                  # (N,)
            g = np.exp(-0.5 * ((pos - self.role_pos[k]) / LINE_SIGMA) ** 2)   # (O, 1)
            kl = kl_all[:, k] if kl_all is not None else np.ones(self.N)
            like = np.maximum(like, np.where(has[None, :], (1 - LINE_CLUTTER) * g * kl[None, :] + LINE_CLUTTER, LINE_CLUTTER))
        return like

    def update(self, r):
        """1 コマ分 (topo_features.py の 1 行) を取り込み, 一番確率の高いステーション番号を返す."""
        p = self.p
        v = abs(r["v"])
        # 進む (止まっていれば進まない: 車速はその瞬間の「動いているか」だけに使う)
        dt = 0.2 if self.prev_t is None else r["t"] - self.prev_t
        self.prev_t = r["t"]
        if self.use_speed:
            # 比較用 (オドメトリあり相当): 車速 x 時間 [ステーション] だけ進め, 前後 ±1 にぼかす
            adv = v * dt / self.step; k0 = int(np.floor(adv)); f = adv - k0
            q = (1 - f) * np.roll(p, k0, 1) + f * np.roll(p, k0 + 1, 1)
            p = 0.25 * np.roll(q, -1, 1) + 0.5 * q + 0.25 * np.roll(q, 1, 1)
        else:
            kernel = self.advance_kernel(r, v)
            if kernel is not None:
                q = np.zeros_like(p)
                for k, wk in enumerate(kernel):
                    if wk > 1e-4:
                        q += wk * np.roll(p, k, 1)
                p = q
        if self.adv_mode != "stopped":
            a, b, c = OFFSET_KERNEL
            q = b * p
            q[1:] += a * p[:-1]; q[:-1] += c * p[1:]
            q[0] += a * p[0]; q[-1] += c * p[-1]              # 端では外へ出ない
            p = q

        # 白線の種類: 数コマ続けて見た判定に置き換える
        if self.kind_tracker is not None:
            kinds, tracked = self.kind_tracker.update(r["t"], r["lines"], self.adv_m if not self.use_speed else v * dt)
            if not self.temporal_kind:
                kinds = [l["kind"] for l in r["lines"]]
            lines = [dict(l, kind=k) for l, k in zip(tracked, kinds)]
        else:
            lines = r["lines"]
        self.line_kinds = [l["kind"] for l in lines]
        self.lines = lines                                    # 推定に使った線 (滑らかにした位置, 種類)

        like = np.ones(self.N)
        ks = []
        for l in lines:
            if l.get("c") and l["x1"] - l["x0"] >= 3.0 and abs(l["y2"]) < 6:
                c0, c1, c2 = l["c"]; d1 = c1 + 2 * c2 * 3.0
                ks.append(2 * c2 / (1 + d1 * d1) ** 1.5)
        self.kobs = float(np.median(ks)) if ks else None
        if ks:
            like *= np.exp(-0.5 * ((self.kobs - self.kap_ahead) / KAPPA_SIGMA) ** 2) + 1e-3

        self.dash_obs = any(l["kind"] == "dashed" and abs(l["y2"]) < 4.5 for l in lines)
        like2 = np.ones((self.O, self.N))
        if self.roles:
            for l in lines:
                if abs(l["y2"]) < 6:
                    like2 *= self.line_like(l["y2"], l["kind"])
        else:
            dashed_exp = (self.role_type[:, 0] == "dashed") | (self.role_type[:, 1] == "dashed")
            like *= np.where(dashed_exp, 0.5, 0.05 if self.dash_obs else 0.95)

        self.sig_d = signal_distance(r) if "tl" in r else None
        seen = bool(r.get("tl")) if "tl" in r else r["signal"]
        if self.use_signal_dist and self.sig_d is not None and self.s_sig is not None:
            # 信号の位置から d 手前 (周回をまたぐ差に注意) に寄せる. 遠いほど距離の誤差は大きい
            diff = (self.s_arr - (self.s_sig + SIGNAL_S0 - self.sig_d) + self.lap / 2) % self.lap - self.lap / 2
            sig = 1.5 + 0.1 * self.sig_d
            like *= np.exp(-0.5 * (diff / sig) ** 2) + 0.05     # 誤検出 (信号以外の赤い物など) でも壊れないよう下限を持つ
        else:
            like *= np.where(self.near_sig, 0.9 if seen else 0.6, 0.02 if seen else 0.98)

        p = p * like[None, :] * like2
        self.p = p / p.sum()
        self.ps = self.p.sum(0)                               # ステーションごとの確率 (横のずれをまとめる)
        self.po = self.p.sum(1)                               # 横のずれごとの確率
        return int(np.argmax(self.ps))

    def advance_kernel(self, r, v):
        """このコマで何ステーション進むかの確率 (None = 進まない). self.adv_mode に使った方法を残す."""
        vo = r.get("vo") if self.motion == "visual" else None
        ok = vo is not None and vo["sharp"] > VO_SHARP and vo["dx"] > VO_MIN_DX and vo["best"] > vo["c0"]
        stopped = v <= 0.1 and (self.motion != "visual" or vo is None or vo["c0"] >= VO_STOP_C0)
        if ok:
            self.vo_dx, self.vo_t = vo["dx"], r["t"]
            self.adv_mode, self.adv_m = "visual", vo["dx"]
            return self.gauss_kernel(vo["dx"], VO_SIGMA)
        if stopped:
            self.adv_mode, self.adv_m = "stopped", 0.0
            return None
        if self.motion == "visual" and self.vo_dx is not None and r["t"] - self.vo_t < VO_HOLD:
            self.adv_mode, self.adv_m = "hold", self.vo_dx
            return self.gauss_kernel(self.vo_dx, VO_HOLD_SIGMA)
        self.adv_mode, self.adv_m = "uniform", ADV_UNKNOWN
        return self.kernel

    def gauss_kernel(self, dx, sigma):
        k = np.arange(int((dx + 3 * sigma) / self.step) + 2)
        w = np.exp(-0.5 * ((k * self.step - dx) / sigma) ** 2)
        return w / w.sum()

    def offset(self):
        """一番確率の高い横のずれ [m] (道路の中央線から, 左が正)."""
        return float(OFFSETS[int(np.argmax(self.po))])

    def roles_of(self, lines, est):
        """推定 (ステーション est, 横のずれ) で, 各線がどの役割に当たるか (当たらなければ None)."""
        o = self.offset()
        out = []
        for l in lines:
            pos = l["y2"] + o
            gap = np.abs(pos - self.role_pos) + np.where(self.role_type[est] != "", 0, 99)
            k = int(np.argmin(gap))
            out.append(self.role_names[k] if gap[k] < 1.0 else None)
        return out


def load_truth(path, R, loc):
    """正解の位置 (図面に合わせたもの) を, 各コマの (ステーション, 横のずれ [m]) にする. 採点だけに使う."""
    T = np.load(path)
    sxy = loc.sxy
    tan = np.roll(sxy, -1, 0) - np.roll(sxy, 1, 0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    nl = np.stack([-tan[:, 1], tan[:, 0]], 1)
    truth = []
    for r in R:
        i = int(np.clip(np.searchsorted(T[:, 0], r["t"]), 0, len(T) - 1))
        cx, cy, gap = T[i, 1], T[i, 2], T[i, 5]
        d = np.hypot(sxy[:, 0] - cx, sxy[:, 1] - cy)
        j = int(np.argmin(d))
        ok = d[j] < 5.0 and gap <= 1.0                        # 外周から離れている・位置が不確かなら採点しない
        truth.append((j, float((np.array([cx, cy]) - sxy[j]) @ nl[j])) if ok else None)
    return truth


def moving_mask(path, R):
    """正解で, そのコマまでの 1 秒に 0.3 m 以上動いているか (採点を走っているコマに絞るのに使う)."""
    T = np.load(path)
    out = []
    for r in R:
        i = int(np.clip(np.searchsorted(T[:, 0], r["t"]), 0, len(T) - 1))
        j = int(np.clip(np.searchsorted(T[:, 0], r["t"] - 1.0), 0, len(T) - 1))
        out.append(float(np.hypot(*(T[i, 1:3] - T[j, 1:3]))) > 0.3)
    return out


def calibrate_signal(R, truth, loc):
    """正解の位置で, 信号の枠の高さ -> 距離 の係数と位置のずれを最小二乗で決める: s_sig - s = K * (H / h) - S0."""
    xs, ys = [], []
    for r, g in zip(R, truth):
        if g is None or not r.get("tl"):
            continue
        h = max(d["h"] / d["H"] for d in r["tl"])
        ds = (loc.s_sig - loc.s_arr[g[0]] + loc.lap / 2) % loc.lap - loc.lap / 2
        if -5 < ds < 45:
            xs.append(1 / h); ys.append(ds)
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5:
        print("信号の検出が少なく較正できません")
        return
    A = np.stack([xs, -np.ones_like(xs)], 1)
    (k, s0), *_ = np.linalg.lstsq(A, ys, rcond=None)
    res = ys - A @ [k, s0]
    print(f"信号の較正: {len(xs)} コマ  SIGNAL_K={k:.4f}  SIGNAL_S0={s0:+.2f} m  残差 中央値 {np.median(np.abs(res)):.2f} m  90% {np.percentile(np.abs(res), 90):.2f} m")
    cur = ys - (SIGNAL_K * xs - SIGNAL_S0)
    print(f"  今の値 (SIGNAL_K={SIGNAL_K}, SIGNAL_S0={SIGNAL_S0}) での誤差 中央値 {np.median(np.abs(cur)):.2f} m  (推定 - 正解 の中央値 {np.median(-cur):+.2f} m)")
    for lo, hi in [(-5, 8), (8, 15), (15, 25), (25, 45)]:
        m = (ys >= lo) & (ys < hi)
        if m.any():
            print(f"  正解の距離 {lo:3d}〜{hi:2d} m: {m.sum():4d} コマ  較正後の差 中央値 {np.median(np.abs(res[m])):.2f} m  今の値での差 {np.median(np.abs(cur[m])):.2f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--topology", default=str(WS_DIR / "data/topological_map/course_topology.json"))
    ap.add_argument("--truth", default="", help="正解の位置 (t, カメラ x, y, 向き, ...) の npy. 採点だけに使う")
    ap.add_argument("--known-start", action="store_true", help="発進位置 (s=0) から始める (省略すると 1 周のどこか分からない状態から)")
    ap.add_argument("--advance-max", type=int, default=ADVANCE_MAX, help="1 コマで進める最大のステーション数")
    ap.add_argument("--use-speed", action="store_true", help="比較用: その瞬間の車速 x 時間だけ進める (オドメトリありに相当)")
    ap.add_argument("--from-lap", type=int, default=0, help="正解で発進位置を n 回目に通った所から追い始める (2 = 2 周目から. 正解が必要)")
    ap.add_argument("--no-roles", action="store_true", help="比較用: 白線の役割を使わず, 破線が見えるかだけを使う (前の版)")
    ap.add_argument("--no-signal-dist", action="store_true", help="比較用: 信号の距離を使わず, 見えたかだけを使う (前の版)")
    ap.add_argument("--motion", default="visual", choices=["visual", "twist"], help="1 コマの進み幅の決め方")
    ap.add_argument("--frame-kind", action="store_true", help="比較用: 白線の種類を 1 コマだけで決める (前の版)")
    ap.add_argument("--no-smooth", action="store_true", help="比較用: 白線の位置と曲がり具合を前後のコマで滑らかにしない (前の版)")
    ap.add_argument("--moving-only", action="store_true", help="正解で動いているコマ (1 秒に 0.3 m 以上) だけで採点する (長い停止で点数が偏らないように)")
    ap.add_argument("--calibrate-signal", action="store_true", help="正解の位置で信号の距離の係数を求めて表示する")
    args = ap.parse_args()

    topo = json.load(open(args.topology))
    loc = TopoLocalizer(topo, args.known_start, args.advance_max, args.use_speed, not args.no_roles, not args.no_signal_dist,
                        args.motion, not args.frame_kind, not args.no_smooth)
    s_arr, seg_of = loc.s_arr, loc.seg_of

    R = json.load(open(args.features))
    truth = load_truth(args.truth, R, loc) if args.truth else None
    if args.calibrate_signal:
        calibrate_signal(R, truth, loc)
        return

    if args.from_lap:
        # 正解の位置が 1 周の終わり (s > 240 m) から始まり (s < 30 m) へ移った回数を数えて, n 周目の始まりを探す
        laps, start_k = 1, None
        for k in range(1, len(R)):
            a, b = truth[k - 1], truth[k]
            if a is not None and b is not None and s_arr[a[0]] > 240 and s_arr[b[0]] < 30:
                laps += 1
                if laps == args.from_lap:
                    start_k = k
                    break
        if start_k is None:
            raise SystemExit(f"{args.from_lap} 周目の始まりが見つかりません")
        print(f"{args.from_lap} 周目の始まり: t={R[start_k]['rt']:.1f}s")
        R, truth = R[start_k:], truth[start_k:]
    if args.moving_only:
        truth = [g if m else None for g, m in zip(truth, moving_mask(args.truth, R))]
    est, off = [], []
    for r in R:
        est.append(loc.update(r))
        off.append(loc.offset())

    if truth is None:
        for r, e in list(zip(R, est))[::25]:
            print(f"t={r['rt']:6.1f}s  推定 s={s_arr[e]:6.1f} m  区間 {seg_of[e]}")
        return
    ok = [(e, g[0]) for e, g in zip(est, truth) if g is not None]
    seg_ok = np.mean([seg_of[e] == seg_of[g] for e, g in ok])
    near_ok = np.mean([abs(int(seg_of[e]) - int(seg_of[g])) <= 1 or {seg_of[e], seg_of[g]} == {0, len(topo['segments']) - 1} for e, g in ok])
    ds = np.array([((s_arr[e] - s_arr[g] + loc.lap / 2) % loc.lap) - loc.lap / 2 for e, g in ok])
    print(f"採点したフレーム {len(ok)} / {len(R)}  ({'発進位置から' if args.known_start else '位置が分からない状態から'})")
    print(f"  区間が合っていた: {100 * seg_ok:.1f}%   隣の区間まで含めると: {100 * near_ok:.1f}%")
    print(f"  位置の誤差 [m]: 中央値 {np.median(np.abs(ds)):.1f}  90% {np.percentile(np.abs(ds), 90):.1f}  最大 {np.abs(ds).max():.1f}")
    near = [abs(d) for d, (e, g) in zip(ds, ok) if loc.near_sig[g]]
    if near:
        print(f"  信号の 25 m 手前〜2 m 先での位置の誤差 [m]: 中央値 {np.median(near):.1f}  90% {np.percentile(near, 90):.1f}  ({len(near)} コマ)")
    # 走っているレーン (正解の横のずれが 0.9〜2.8 m のコマだけ. 左が正 = 内側のレーン).
    # 正解が道路の外 (> 2.8 m) のコマは, 図面合わせの取り違え (右の直線の二重線) なので採点しない
    lane = [(np.sign(o), np.sign(g[1])) for o, g in zip(off, truth) if g is not None and 0.9 < abs(g[1]) < 2.8]
    if lane:
        print(f"  レーン (内側 / 外側) が合っていた: {100 * np.mean([a == b for a, b in lane]):.1f}%  ({len(lane)} コマ)")
    last = None
    for r, e, g in zip(R, est, truth):
        if g is None:
            continue
        row = (int(seg_of[e]), int(seg_of[g[0]]))
        if row != last:
            print(f"   t={r['rt']:6.1f}s  推定 区間 {row[0]:2d} (s={s_arr[e]:5.1f})  正解 区間 {row[1]:2d} (s={s_arr[g[0]]:5.1f})")
            last = row


if __name__ == "__main__":
    main()
