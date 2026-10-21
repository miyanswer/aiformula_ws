#!/usr/bin/env python3
"""
lane_role_compare.py - rosbag の走行映像で, YOLOP の前処理の修正前 / 修正後 の白線の役割づけ (左境界 / 中央 / 右境界) を比べる

車載の白線処理 (aiformula_machine/src/oit_navigation の lane_detector.py) を, 変えずに読むだけで同じ順にかける:
  YOLOP (前処理だけ 2 通り) -> mask_lines.extract_mask_lines -> geometry.project_to_ground -> geometry.fit_line
  -> line_tracker.LineTracker.update
  - 修正前: 上部 45% を切り, 残りを縦に引き伸ばして 640x640 (2026-10-19 の修正 95faaf2 より前の車載)
  - 修正後: 上部 45% を切り, 縦横比を保って 640 に収め余白を 114 で埋める (letterbox. 学習時と同じ. 今の車載)
  重み (models/honda_shihou_finetuned_best.pth) と, カメラ・LineTracker のパラメータ (lane_detector_params.yaml) は両方同じ.
  カメラ映像の全コマ (約 15 コマ/秒. 車載の max_inference_hz 15 と同じ) にかける.
  6 レーン走行の rosbag では, 走行中に車載が受け取った lane_reseed (6 レーン計画からの横位置) を同じ時刻に両方へ入れる
  (--no-reseed で入れない). 6 レーン走行の rosbag に記録された車載の役割 (修正前) とも, どれだけ一致するかを数える
  (この再現が車載と同じに動いているかの確かめ).

役割の正しさはこの道具では決めない (人が映像で判断する). 出すもの:
  1. 計算: data/lane_role_compare/<bag>.json に, コマごとの両方の線と役割を残す (数値だけ. 映像は保存しない)
     と, 役割ごとの見えた割合 / 修正前と修正後で役割が食い違った時間帯の一覧 (<bag>_summary.txt)
  2. --view: 修正前 (左) と修正後 (右) を並べて再生する. 線を役割の色で描き (L 青 / C 黄 / R 赤, 補完した線は細線,
     役割の付かなかった線は灰),
     同じ白線に違う役割が付いたコマは ROLE DIFF (修正前の役割 -> 修正後の役割) と出す. 下の帯は役割が食い違った時間帯 (赤)
     と今の位置. 線の横の文字は役割 (? が付くのは見えていない線を他の線から補完したもの).
     信号の枠 (白の四角) と, 信号の真下の地面 = 中央線があるはずの所 (紫の x. 使えたコマだけ) も描く.
     信号で中央を置き直したコマは SIGNAL RE-ANCHOR と出す
     キー: スペース = 一時停止, n = 次の食い違いの少し前まで早送り, . = (一時停止中) 1 コマ進む,
           q / Esc / 窓の x = 終了 (キーは映像の窓を選んでから押す)

信号で付け直す (--signal): 信号機は中央線の真上にある. 車載の LineTracker.update(fits, signal) (aiformula_machine に
  2026-10-21 に足した信号機での置き直し) をそのまま呼ぶ. 信号機の位置は lane_detector.py と同じ求め方:
  信号機の検出器 (車載と同じ models/traffic_light.pt, conf 0.35, imgsz 640) の枠のうち確からしさが一番高いもの
  (signal_conf_min 以上) から, 距離 d = real_height * focal_length_y / 360 / (枠の高さ / 画像の高さ)
  (traffic_light_distance_node と同じ), 前方 x = camera_x + d, 横 y = d * (cx - 枠の中心 u) / fx.
  YOLOP はかけ直さない: 計算済みの修正後の白線の式 (fits) に LineTracker をかけ直す. 6 レーン走行の rosbag の
  lane_reseed は記録したときの古い役割 (修正前) から 6 レーン走行が出したものなので入れない. 比べやすいように
  信号機を渡さない版 (after_nr = 修正後で lane_reseed なし) も作る. --view --pair after_nr,signal で並べる

使い方 (Mac 本体の python3. rosbag の読み出しは .venv の python に任せる):
  python3 scripts/lane_role_compare.py rosbag/mcap/20261012_155111_gamepad.mcap           # 計算 (数分)
  python3 scripts/lane_role_compare.py rosbag/mcap/20261012_155111_gamepad.mcap --view [--start 120] [--speed 0.5]
  python3 scripts/lane_role_compare.py rosbag/mcap/20261012_155111_gamepad.mcap --signal          # 信号の検出と付け直し
  python3 scripts/lane_role_compare.py rosbag/mcap/20261012_155111_gamepad.mcap --view --pair after_nr,signal
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

WS_DIR = Path(__file__).resolve().parent.parent
MACHINE_PKG = Path.home() / "aiformula_machine/src/oit_navigation"
PARAMS_YAML = MACHINE_PKG / "oit_navigation/lane/config/lane_detector_params.yaml"
OUT_DIR = WS_DIR / "data/lane_role_compare"
WEIGHTS = WS_DIR / "models/honda_shihou_finetuned_best.pth"     # 車載の既定と同じファイル
TOPICS = ["/camera/image/compressed", "/aiformula_control/six_lane_planner/lane_reseed",
          "/aiformula_perception/lane_detector/lane_lines"]
VARIANTS = ("before", "after")
ROLES = ("left", "center", "right")
DIFF_M = 0.5               # 同じ役割の線の 2 m 先の横位置がこれ以上違えば「見え方の違い」 [m] (車線の幅 3.66 m の約 1/7)
SAME_LINE_M = 0.3          # 修正前と修正後の線の 2 m 先の横位置がこれ以内なら同じ白線とみなす [m]
SEG_JOIN_S = 0.5           # 食い違いのコマがこの時間以内なら 1 つの時間帯にまとめる
SEG_MIN_S = 0.3            # これより短い食い違いの時間帯は一覧に出さない
LABELS = {"before": "before: stretch (old)", "after": "after: letterbox (now)",
          "after_nr": "after (no lane_reseed)", "signal": "after + signal (no lane_reseed)"}
TL_PARAMS_YAML = MACHINE_PKG / "oit_navigation/traffic_light/config/traffic_light_params.yaml"


def run_reader(bag, start=0.0):
    """rosbag を時刻順に [ヘッダ長][JSON][データ長][データ] で標準出力へ (読み取りのみ).
    start [s] があれば, rosbag の先頭からその秒数より前は読まない (先頭から読み飛ばすと待たされるので).
    最初にヘッダ {"k": "t0", "t": rosbag の先頭の時刻} を 1 つ出す."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    out = sys.stdout.buffer
    with open(bag, "rb") as fp:
        reader = make_reader(fp, decoder_factories=[DecoderFactory()])
        t0_ns = reader.get_summary().statistics.message_start_time
        h = json.dumps({"k": "t0", "t": t0_ns / 1e9}).encode()
        out.write(struct.pack("<I", len(h)) + h + struct.pack("<I", 0))
        for _, ch, msg, m in reader.iter_decoded_messages(topics=TOPICS, start_time=t0_ns + int(start * 1e9)):
            t = msg.log_time / 1e9
            data = b""
            if ch.topic.endswith("compressed"):
                head, data = {"k": "img", "t": t}, bytes(m.data)
            elif ch.topic.endswith("lane_reseed"):
                head = {"k": "reseed", "t": t, "v": float(m.data)}
            else:
                head = {"k": "rec", "t": t, "roles": {r: [getattr(m, r).c0, getattr(m, r).c1, getattr(m, r).c2]
                                                      for r in ROLES if getattr(m, r).detected}}
            h = json.dumps(head).encode()
            out.write(struct.pack("<I", len(h)) + h + struct.pack("<I", len(data)) + data)


def messages(bag, start=0.0):
    """rosbag の (ヘッダ, データ) を時刻順に返す. ヘッダに rt (rosbag の先頭からの秒) を足す."""
    reader = subprocess.Popen([str(WS_DIR / ".venv/bin/python"), os.path.abspath(__file__), "--reader", os.path.abspath(bag),
                               str(max(start - 1.0, 0.0))], stdout=subprocess.PIPE)
    inp, t0 = reader.stdout, None
    try:
        while True:
            b = inp.read(4)
            if len(b) < 4:
                break
            head = json.loads(inp.read(struct.unpack("<I", b)[0]))
            data = inp.read(struct.unpack("<I", inp.read(4))[0])
            if head["k"] == "t0":
                t0 = head["t"]
                continue
            head["rt"] = head["t"] - t0
            if head["rt"] >= start:
                yield head, data
    finally:
        reader.kill()


def tracker_params(P, LineTrackerParams, signal=True):
    """lane_detector.py と同じく lane_detector_params.yaml から LineTrackerParams を作る (signal=False で信号機を使わない)."""
    return LineTrackerParams(
        x_ref=P["tracker_x_ref"], lane_width_init=P["lane_width"], gate_ratio=P["tracker_gate_ratio"],
        width_tolerance=P["tracker_width_tolerance"], max_heading_diff=P["tracker_max_heading_diff"],
        init_offset=P["tracker_init_offset"], anchor_tolerance=P["tracker_anchor_tolerance"],
        anchor_frames=P["tracker_anchor_frames"], anchor_max_x_min=P["tracker_anchor_max_x_min"],
        anchor_heading_diff=P["tracker_anchor_heading_diff"],
        use_signal_anchor=signal and P.get("signal_anchor_enable", True),
        signal_x_min=P.get("signal_x_min", 4.0), signal_x_max=P.get("signal_x_max", 15.0),
        signal_match=P.get("signal_match", 0.9), signal_frames=P.get("signal_frames", 3),
        signal_distance_tolerance=P.get("signal_distance_tolerance", 0.2))


class VehicleLane:
    """車載 lane_detector.py の YOLOP より後 (白線の取り出し -> 地面へ -> 当てはめ -> 役割) を, 車載のコードのまま動かす."""

    def __init__(self, P):
        sys.path.insert(0, str(MACHINE_PKG))
        import numpy as np
        from oit_navigation.common.geometry import CameraModel, fit_line, project_to_ground
        from oit_navigation.lane.line_tracker import LineTracker, LineTrackerParams
        from oit_navigation.lane.mask_lines import MaskLinesParams, extract_mask_lines
        self.np, self.fit_line, self.project, self.extract = np, fit_line, project_to_ground, extract_mask_lines
        self.mlp = MaskLinesParams(top_ratio=P["top_cut_ratio"])
        self.fit_x_max = P["fit_x_max"]
        self.cam = CameraModel(fx=P["camera_fx"], fy=P["camera_fy"], cx=P["camera_cx"], cy=P["camera_cy"],
                               ref_width=P["camera_ref_width"], ref_height=P["camera_ref_height"],
                               cam_height=P["camera_height"], cam_x=P["camera_x"],
                               pitch_down=np.deg2rad(P["camera_pitch_down_deg"]))
        self.tracker = LineTracker(tracker_params(P, LineTrackerParams))

    def process(self, mask):
        h, w = mask.shape[:2]
        fits = []
        for lane in self.extract(mask, self.mlp):
            x, y, valid = self.project(self.cam, lane["u"], lane["v"], w, h)
            f = self.fit_line(x[valid], y[valid], x_max_fit=self.fit_x_max)
            if f is not None:
                fits.append(f)
        tr = self.tracker.update(fits)

        def enc(f):
            return [round(float(v), 4) for v in f.coeffs] + [round(f.x_min, 2), round(f.x_max, 2)]
        roles = {}
        for r in ROLES:
            f = tr.lines.get(r)
            if f is not None:
                roles[r] = dict(f=enc(f), det=bool(tr.detected.get(r)), inf=bool(f.inferred))
        return dict(fits=[enc(f) for f in fits], roles=roles, re=bool(tr.reanchored))


def y2(f):
    return f[0] + 2 * f[1] + 4 * f[2]


def role_conflict(a, b):
    """同じ白線 (2 m 先の横位置が SAME_LINE_M 以内) に, 修正前 a と修正後 b で違う役割が付いた組 [(a の役割, b の役割)].
    実際に見えた線 (補完でない) だけを比べる. 片方だけに見えた線は数えない (見え方の違い)."""
    out = []
    for ra in ROLES:
        if not a.get(ra, {}).get("det"):
            continue
        for rb in ROLES:
            if rb != ra and b.get(rb, {}).get("det") and abs(y2(a[ra]["f"]) - y2(b[rb]["f"])) < SAME_LINE_M:
                out.append((ra, rb))
    return out


def role_diff(a, b):
    """2 つの役割づけ (role -> dict(f=...)) が食い違う役割のリスト (見えた / 見えないの違い, または横位置が DIFF_M 以上違う).
    補完した線 (実際には見えていない) は比べない."""
    out = []
    for r in ROLES:
        ra = a.get(r) if a.get(r, {}).get("det") else None
        rb = b.get(r) if b.get(r, {}).get("det") else None
        if ra is None and rb is None:
            continue
        if ra is None or rb is None or abs(y2(ra["f"]) - y2(rb["f"])) >= DIFF_M:
            out.append(r)
    return out


def compute(args):
    import cv2
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(WS_DIR / "scripts"))
    import verify_lane_rosbag as V
    sys.path.insert(0, str(WS_DIR / "src/aiformula_base/oit_navigation/oit_navigation/yolop"))
    from lib.config import cfg
    from lib.models import get_net

    P = yaml.safe_load(open(PARAMS_YAML))["lane_detector"]["ros__parameters"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = get_net(cfg)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("state_dict", ck))
    net = net.to(dev).float().eval()
    mean, std = np.asarray(V.NORM_MEAN, np.float32), np.asarray(V.NORM_STD, np.float32)
    S = V.INPUT_SIZE

    def infer(bgr, letterbox):
        """車載 yolop_lane_backend.py の infer_mask と同じ (letterbox=False は修正前, True は修正後)."""
        h, w = bgr.shape[:2]
        cut = int(h * P["top_cut_ratio"])
        proc = bgr[cut:]
        ph, pw = proc.shape[:2]
        if letterbox:
            r = min(S / ph, S / pw)
            nh, nw = int(round(ph * r)), int(round(pw * r))
            top, left = (S - nh) // 2, (S - nw) // 2
            canvas = np.full((S, S, 3), 114, np.uint8)
            canvas[top:top + nh, left:left + nw] = cv2.resize(proc, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            nh, nw, top, left = S, S, 0, 0
            canvas = cv2.resize(proc, (S, S), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(((canvas.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])).to(dev)
        with torch.inference_mode():
            _, _, ll = net(x)
        ll = ll[:, :, top:top + nh, left:left + nw].float()
        seg = (ll[0, 1] > ll[0, 0]).to(torch.uint8).cpu().numpy()
        mask = np.zeros((h, w), np.uint8)
        mask[cut:] = cv2.resize(seg, (pw, ph), interpolation=cv2.INTER_NEAREST)
        return mask

    veh = {v: VehicleLane(P) for v in VARIANTS}
    frames, rec, n_reseed = [], [], 0
    t_start = time.time()
    for head, data in messages(args.bag):
        if head["k"] == "reseed":
            if not args.no_reseed:
                for v in VARIANTS:
                    veh[v].tracker.seed_lane_position(head["v"])
                n_reseed += 1
            continue
        if head["k"] == "rec":
            rec.append(dict(t=head["t"], roles=head["roles"]))
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        row = dict(t=round(head["t"], 3), rt=round(head["rt"], 3), size=[img.shape[1], img.shape[0]])
        for v, lb in zip(VARIANTS, (False, True)):
            row[v] = veh[v].process(infer(img, lb))
        frames.append(row)
        if len(frames) % 500 == 0:
            print(f"{len(frames)} コマ ({time.time() - t_start:.0f} s)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(args.bag).stem
    json.dump(dict(bag=name, reseed=not args.no_reseed, n_reseed=n_reseed, frames=frames, recorded=rec),
              open(OUT_DIR / f"{name}.json", "w"))
    summary(name)


def signal_of(tl, W, P, TP):
    """そのコマの信号機の検出 -> 信号機の位置 (前方 x [m], 横 y [m], 左正). lane_detector.py の _signal_position と同じ.
    tl: 検出 [{conf, u (枠の中心), h (枠の高さ), H (画像の高さ)}], W: 画像の幅, P: lane_detector の, TP: traffic_light の パラメータ."""
    tl = [d for d in tl if d["conf"] >= P.get("signal_conf_min", 0.5) and d["h"] > 0]
    if not tl:
        return None
    d0 = max(tl, key=lambda d: d["conf"])
    d = TP["real_height_m"] * TP["focal_length_y"] / TP["reference_image_height"] / (d0["h"] / d0["H"])
    s = W / P["camera_ref_width"]
    return P["camera_x"] + d, d * (P["camera_cx"] * s - d0["u"]) / (P["camera_fx"] * s)


def compute_signal(args):
    """信号の検出 (各コマ) と lane_reseed を rosbag から読み, 修正後の白線に LineTracker (+ 信号で付け直し) をかけ直す."""
    import cv2
    import numpy as np
    import yaml
    from ultralytics import YOLO
    sys.path.insert(0, str(MACHINE_PKG))
    from oit_navigation.common.geometry import LineFit
    name = Path(args.bag).stem
    path = OUT_DIR / f"{name}.json"
    D = json.load(open(path))
    F = D["frames"]
    P = yaml.safe_load(open(PARAMS_YAML))["lane_detector"]["ros__parameters"]
    if "reseeds" not in D or any("tl" not in f for f in F):
        model = YOLO(str(WS_DIR / "models/traffic_light.pt"))
        by_t = {round(f["t"], 3): f for f in F}
        reseeds, k, t_start = [], 0, time.time()
        for head, data in messages(args.bag):
            if head["k"] == "reseed":
                reseeds.append([head["t"], head["v"]])
                continue
            f = by_t.get(round(head["t"], 3)) if head["k"] == "img" else None
            if f is None:
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            res = model.predict(img, conf=0.35, iou=0.45, imgsz=640, verbose=False)[0]
            f["tl"] = [dict(cls=model.names[int(b.cls[0])].replace("traffic_light_", ""), conf=round(float(b.conf[0]), 3),
                            u=round(float(b.xyxy[0][0] + b.xyxy[0][2]) / 2, 1), v0=round(float(b.xyxy[0][1]), 1),
                            h=round(float(b.xyxy[0][3] - b.xyxy[0][1]), 1), H=img.shape[0]) for b in res.boxes]
            k += 1
            if k % 1000 == 0:
                print(f"信号の検出 {k} コマ ({time.time() - t_start:.0f} s)", flush=True)
        D["reseeds"] = reseeds
    # 計算済みの修正後の白線に, 車載の LineTracker をかけ直す:
    #   check = lane_reseed あり・信号機なし (修正後の役割と同じになるかの確かめ), after_nr = lane_reseed なし・信号機なし,
    #   signal = lane_reseed なし・信号機あり
    from oit_navigation.lane.line_tracker import LineTracker, LineTrackerParams
    TP = yaml.safe_load(open(TL_PARAMS_YAML))
    TP = next(iter(TP.values()))["ros__parameters"]
    check, plain = LineTracker(tracker_params(P, LineTrackerParams, False)), LineTracker(tracker_params(P, LineTrackerParams, False))
    sig = LineTracker(tracker_params(P, LineTrackerParams, True))
    rs = D["reseeds"] if D["reseed"] else []
    j, same, n_anchor = 0, 0, 0
    for f in F:
        while j < len(rs) and rs[j][0] <= f["t"]:
            check.seed_lane_position(rs[j][1])
            j += 1
        fits = [LineFit(np.array(c[:3]), c[3], c[4], 10) for c in f["after"]["fits"]]
        tp = check.update(fits)
        tn = plain.update(fits)
        s_ = signal_of(f.get("tl", []), f["size"][0], P, TP)
        ts = sig.update(fits, s_)

        def enc(tr):
            return {r: dict(f=[round(float(v), 4) for v in tr.lines[r].coeffs] + [round(tr.lines[r].x_min, 2), round(tr.lines[r].x_max, 2)],
                            det=bool(tr.detected.get(r)), inf=bool(tr.lines[r].inferred)) for r in ROLES if tr.lines.get(r) is not None}
        same += not role_diff(enc(tp), f["after"]["roles"]) and not role_conflict(enc(tp), f["after"]["roles"])
        f["after_nr"] = dict(fits=f["after"]["fits"], roles=enc(tn), re=bool(tn.reanchored))
        sig_anchor = bool(ts.signal_anchored)
        n_anchor += sig_anchor
        f["signal"] = dict(fits=f["after"]["fits"], roles=enc(ts), re=bool(ts.reanchored), sig=s_, sig_anchor=sig_anchor)
    json.dump(D, open(path, "w"))
    n_sig = sum(f["signal"]["sig"] is not None for f in F)
    nd = sum(bool(role_conflict(f["after_nr"]["roles"], f["signal"]["roles"])) for f in F)
    print(f"{name}: 信号機の位置が出たコマ {n_sig} / {len(F)}. 信号機で中央を置き直したコマ {n_anchor}."
          f" 信号機あり / なし (lane_reseed なし) で役割が食い違ったコマ {nd}")
    print(f"  確かめ: 保存した白線の式に LineTracker (信号機なし, lane_reseed あり) をかけ直した役割が, 修正後と同じだったコマ {same} / {len(F)}"
          f" ({100 * same / len(F):.1f}%)")
    signal_check(name)


def signal_check(name):
    """信号が本当に中央線の上に見えているか: 信号の横位置と, 修正後の各役割の線を前方 d m まで延ばした横位置の差."""
    import numpy as np
    D = json.load(open(OUT_DIR / f"{name}.json"))
    res = {r: [] for r in ROLES}
    for f in D["frames"]:
        s_ = f.get("signal", {}).get("sig")
        if s_ is None:
            continue
        d, ys = s_
        for r in ROLES:
            ro = f["after"]["roles"].get(r)
            if ro and ro["det"]:
                c = ro["f"]
                res[r].append(c[0] + c[1] * d + c[2] * d * d - ys)
    JA = {"left": "左境界", "center": "中央", "right": "右境界"}
    print("  信号の横位置 と 修正後の線を信号の距離まで延ばした横位置 の差 (線 - 信号):")
    for r in ROLES:
        v = np.array(res[r])
        if len(v):
            print(f"    {JA[r]}: 中央値 {np.median(v):+.2f} m, |差| < 0.9 m が {100 * np.mean(np.abs(v) < 0.9):.0f}% ({len(v)} コマ)")


def diff_segments(frames, a="before", b="after"):
    """修正前と修正後で役割が食い違った (同じ白線に違う役割) 時間帯 [(始め rt, 終わり rt, 修正前の役割 -> 修正後の役割 の集合)]."""
    segs = []
    for fr in frames:
        if a not in fr or b not in fr:
            continue
        d = {f"{x}->{y}" for x, y in role_conflict(fr[a]["roles"], fr[b]["roles"])}
        if not d:
            continue
        if segs and fr["rt"] - segs[-1][1] <= SEG_JOIN_S:
            segs[-1][1] = fr["rt"]
            segs[-1][2].update(d)
        else:
            segs.append([fr["rt"], fr["rt"], set(d)])
    return [s for s in segs if s[1] - s[0] >= SEG_MIN_S]


def summary(name):
    import numpy as np
    D = json.load(open(OUT_DIR / f"{name}.json"))
    F = D["frames"]
    n = len(F)
    dur = F[-1]["rt"] - F[0]["rt"]
    L = [f"{name}: {n} コマ ({dur:.0f} s, {n / max(dur, 1e-6):.1f} コマ/秒)  lane_reseed "
         + (f"{D['n_reseed']} 回入れた" if D["reseed"] else "入れない")]
    L.append(f"  {'':10s}{'白線が 1 本以上':>14s}{'左境界':>10s}{'中央':>10s}{'右境界':>10s}{'1 コマの線の数':>16s}   (役割は実際に見えた線だけ)")
    for v, lab in zip(VARIANTS, ("修正前", "修正後")):
        any_line = np.mean([len(f[v]["fits"]) > 0 for f in F]) * 100
        det = [np.mean([f[v]["roles"].get(r, {}).get("det", False) for f in F]) * 100 for r in ROLES]
        nl = np.mean([len(f[v]["fits"]) for f in F])
        L.append(f"  {lab:10s}{any_line:13.1f}%" + "".join(f"{d:9.1f}%" for d in det) + f"{nl:15.2f}")
    JA = {"left": "左境界", "center": "中央", "right": "右境界"}
    nc, kinds = 0, {}
    for f in F:
        c = role_conflict(f["before"]["roles"], f["after"]["roles"])
        nc += bool(c)
        for k in c:
            kinds[k] = kinds.get(k, 0) + 1
    L.append(f"  役割の食い違い (同じ白線に違う役割): {nc} / {n} コマ ({100 * nc / n:.1f}%)")
    for (ra, rb), c in sorted(kinds.items(), key=lambda x: -x[1]):
        L.append(f"      修正前 {JA[ra]} -> 修正後 {JA[rb]}: {c} コマ")
    nd = [f for f in F if role_diff(f["before"]["roles"], f["after"]["roles"])]
    L.append(f"  見え方の違い (同じ役割の線が片方だけ見えた, または 2 m 先の横位置が {DIFF_M} m 以上違う): "
             f"{len(nd)} / {n} コマ ({100 * len(nd) / n:.1f}%)")
    if D["recorded"]:
        # この再現が車載と同じに動いているか: 修正前の再現と, rosbag に記録された車載の役割 (修正前) の一致
        rt_ = np.array([r["t"] for r in D["recorded"]])
        same = tot = 0
        for f in F:
            i = int(np.clip(np.searchsorted(rt_, f["t"]), 0, len(rt_) - 1))
            if abs(rt_[i] - f["t"]) > 0.05:
                continue
            recd = {r: dict(f=c + [0, 0], det=True) for r, c in D["recorded"][i]["roles"].items()}
            tot += 1
            same += not role_diff(f["before"]["roles"], recd)
        L.append(f"  確かめ: 修正前の再現と, rosbag に記録された車載の役割 (修正前) が一致したコマ {same} / {tot}"
                 f" ({100 * same / max(tot, 1):.1f}%. 車載は約 11 回/秒しか処理していないので, 0.05 s 以内のコマだけ比べる)")
    segs = diff_segments(F)
    L.append(f"  役割が食い違った時間帯 ({SEG_MIN_S} s 以上, {len(segs)} 個). --view --start <始め - 2> で見られる:")
    for a, b, rs in segs:
        L.append(f"    {a:7.1f} - {b:7.1f} s  ({b - a:4.1f} s)  "
                 + " / ".join(f"修正前 {JA[x.split('->')[0]]} -> 修正後 {JA[x.split('->')[1]]}" for x in sorted(rs)))
    text = "\n".join(L)
    print(text)
    (OUT_DIR / f"{name}_summary.txt").write_text(text + "\n")


def view(args):
    import cv2
    import numpy as np
    sys.path.insert(0, str(MACHINE_PKG))
    import yaml
    from oit_navigation.common.geometry import CameraModel, ground_to_image
    name = Path(args.bag).stem
    D = json.load(open(OUT_DIR / f"{name}.json"))
    F = D["frames"]
    by_t = {round(f["t"], 3): f for f in F}
    A, B = args.pair.split(",")
    if any(B not in f or A not in f for f in F[:1]):
        raise SystemExit(f"{A} / {B} の結果がありません (signal は先に --signal で作る)")
    segs = diff_segments(F, A, B)
    P = yaml.safe_load(open(PARAMS_YAML))["lane_detector"]["ros__parameters"]
    cam = CameraModel(fx=P["camera_fx"], fy=P["camera_fy"], cx=P["camera_cx"], cy=P["camera_cy"],
                      ref_width=P["camera_ref_width"], ref_height=P["camera_ref_height"], cam_height=P["camera_height"],
                      cam_x=P["camera_x"], pitch_down=np.deg2rad(P["camera_pitch_down_deg"]))
    W, H = 640, 360
    COL = {"left": (255, 160, 40), "center": (0, 220, 255), "right": (60, 60, 255)}
    LAB = {"left": "L", "center": "C", "right": "R"}
    rt_end = F[-1]["rt"]

    def draw(img, res, title):
        # 役割の付いていない線 (灰) -> 役割の付いた線 (色. 補完は細線)
        def poly(f, col, th):
            xs = np.arange(max(f[3], 0.5), min(f[4], 15.0), 0.2)
            if len(xs) < 2:
                return None
            u, v, ok = ground_to_image(cam, xs, f[0] + f[1] * xs + f[2] * xs * xs, W, H)
            pts = np.stack([u[ok], v[ok]], 1).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
                return tuple(pts[np.argmax(pts[:, 1])])
            return None
        for f in res["fits"]:
            poly(f, (170, 170, 170), 1)
        for r in ROLES:
            if r not in res["roles"]:
                continue
            ro = res["roles"][r]
            f = ro["f"] if not ro["inf"] else ro["f"][:3] + [1.0, 8.0]
            p = poly(f, COL[r], 1 if ro["inf"] else 3)
            if p is not None:
                u, v = int(np.clip(p[0], 8, W - 40)), int(np.clip(p[1] - 10, 20, H - 6))
                t = LAB[r] + ("?" if ro["inf"] else "")
                cv2.putText(img, t, (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(img, t, (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL[r], 2, cv2.LINE_AA)
        cv2.putText(img, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_signal(img, fr, anchored):
        """信号の枠 (縦線) と, 信号の真下の地面 (中央線があるはずの所. 前方 d m, 横 y_s) の印 (使えたコマだけ)."""
        for t in fr.get("tl", []):
            u, v0, h = int(t["u"] * W / fr["size"][0]), int(t["v0"] * H / fr["size"][1]), int(t["h"] * H / fr["size"][1])
            cv2.rectangle(img, (u - max(h // 3, 3), v0), (u + max(h // 3, 3), v0 + h), (255, 255, 255), 1)
        s_ = fr.get("signal", {}).get("sig")
        if s_ is not None:
            u, v, ok = ground_to_image(cam, np.array([s_[0]]), np.array([s_[1]]), W, H)
            if ok[0]:
                cv2.drawMarker(img, (int(u[0]), int(v[0])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 3)
        if anchored:
            cv2.putText(img, "SIGNAL RE-ANCHOR", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    def timeline(width, rt):
        bar = np.full((22, width, 3), 40, np.uint8)
        for a, b, _ in segs:
            cv2.rectangle(bar, (int(a / rt_end * width), 3), (max(int(b / rt_end * width), int(a / rt_end * width) + 1), 18), (60, 60, 230), -1)
        x = int(rt / rt_end * width)
        cv2.line(bar, (x, 0), (x, 21), (255, 255, 255), 2)
        return bar

    win = f"lane roles: {A} | {B}   space: pause, n: next diff, .: step, q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    wall0 = rt0 = None
    paused, skip_to = False, None
    for head, data in messages(args.bag, args.start):
        if head["k"] != "img":
            continue
        fr = by_t.get(round(head["t"], 3))
        if fr is None:
            continue
        if skip_to is not None and fr["rt"] < skip_to:
            continue
        if skip_to is not None:
            skip_to, wall0 = None, None
        img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (W, H))
        a, b = img.copy(), img.copy()
        draw(a, fr[A], LABELS[A])
        draw(b, fr[B], LABELS[B])
        for im, v in ((a, A), (b, B)):
            draw_signal(im, fr, fr[v].get("sig_anchor", False))
        d = role_conflict(fr[A]["roles"], fr[B]["roles"])
        top = np.full((34, 2 * W, 3), 25, np.uint8)
        cv2.putText(top, f"{name}  t={fr['rt']:7.2f} s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if d:
            txt = "ROLE DIFF: " + ", ".join(f"{LAB[a]}->{LAB[b]}" for a, b in d)
            cv2.putText(top, txt, (2 * W - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
        frame = np.vstack([top, np.hstack([a, b]), timeline(2 * W, fr["rt"])])
        cv2.imshow(win, frame)
        if wall0 is None:
            wall0, rt0 = time.time(), fr["rt"]
        wait = (fr["rt"] - rt0) / args.speed - (time.time() - wall0)
        key = cv2.waitKey(1 if paused else max(1, int(wait * 1000))) & 0xFF
        while True:
            # q / Esc, または窓の x で閉じたら終わる
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
                wall0 = None
            if key == ord("n"):
                nxt = [s[0] for s in segs if s[0] > fr["rt"] + 2.0]
                if nxt:
                    skip_to, paused = nxt[0] - 2.0, False
                break
            if not paused or key == ord("."):
                break
            key = cv2.waitKey(50) & 0xFF
    print("最後まで再生しました. 何かキーを押すと閉じます", flush=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--reader":
        try:
            run_reader(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--view", action="store_true", help="計算済みの結果を映像に重ねて, 修正前と修正後を並べて再生する")
    ap.add_argument("--summary", action="store_true", help="計算済みの結果の集計だけ出す")
    ap.add_argument("--no-reseed", action="store_true", help="6 レーン走行の lane_reseed を入れない")
    ap.add_argument("--signal", action="store_true", help="信号の検出と, 信号で中央線を置き直す版 (signal) を計算する")
    ap.add_argument("--pair", default="before,after", help="--view: 左右に並べる役割づけ (before / after / signal)")
    ap.add_argument("--start", type=float, default=0.0, help="--view: rosbag の先頭から何秒目から")
    ap.add_argument("--speed", type=float, default=1.0, help="--view: 再生の速さ (1 = 実時間)")
    args = ap.parse_args()
    if args.view:
        try:
            view(args)
        except KeyboardInterrupt:            # Ctrl-C で止めたときは静かに終わる
            pass
    elif args.signal:
        compute_signal(args)
    elif args.summary:
        summary(Path(args.bag).stem)
    else:
        compute(args)


if __name__ == "__main__":
    main()
