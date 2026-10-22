#!/usr/bin/env python3
"""
distance_check_rosbag.py - rosbag で, コーンと信号機までの距離 (車載の検出器が出すもの) が合っているかを確かめる

車載の距離の出し方:
  コーン   (cone_detector_node.py)          : 枠の下の辺の中央を地面 (平ら) に写した点 (base_link の x, y). カメラの傾きに敏感
  信号機   (traffic_light_distance_node.py) : 枠の高さから. 距離 = real_height_m * focal_length_y / 360 / (枠の高さ / 画像の高さ)

本当の距離の代わり (地図や巻尺はなくてよい): コーンと信号機は止まっているので, 車の位置 (正解の位置 data/truth/<bag>/truth.npy,
なければ gyro_odometry の odom) で検出を図面 (世界) の座標に置くと, 距離が正しければどの距離から見ても同じ所に来る.
  1. 近くで見えたとき (コーン NEAR_CONE, 信号機 NEAR_SIGNAL. 近いほど距離の誤差は小さい) の世界の位置の中央値を, その物の位置とする
  2. 遠くで見えたときの「その物の位置までの本当の距離」と「検出器の距離」を比べ, 距離ごとの比 (検出 / 本当) と差を出す
     比が 1 より小さい = 検出器は距離を短く出している. これは近くの検出も正しいと仮定した, 距離どうしの比べ
  信号機は正解の位置があれば, 図面の信号機の位置 (course_topology.json の signal の印の中央線上の点) までの距離とも比べる
  (こちらは近くの検出を使わない絶対の比べ. 図面の位置と正解の位置の誤差 (0.5 m ほど) を含む).
  odom は走った距離を短く出す (正解の位置の 0.73〜0.80 倍. 2026-10-12) ので, 正解の位置がない rosbag の結果は目安にならない
使う検出: 6 レーン走行の rosbag は走行中に記録された車載の出力 (cone_detector/status, traffic_light/status).
  gamepad の rosbag には記録がないので, 信号機は lane_role_compare.py --signal で車載と同じ検出器をかけた結果を使う
  (距離は traffic_light_distance_node と同じ式. 時間方向のならし (smoothing) はかけない).

使い方:
  python3 scripts/distance_check_rosbag.py rosbag/mcap/20261012_134520_6lane.mcap
  python3 scripts/distance_check_rosbag.py rosbag/mcap/20261012_153125_6lane.mcap        # 正解の位置がなければ odom
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

WS_DIR = Path(__file__).resolve().parent.parent
TOPICS = ["/aiformula_perception/cone_detector/status", "/aiformula_perception/traffic_light/status",
          "/aiformula_sensing/gyro_odometry_publisher/odom"]
NEAR_CONE = (1.5, 4.0)       # この距離 [m] のコーンの検出で位置を決める
NEAR_SIGNAL = (4.0, 7.0)     # この距離 [m] の信号機の検出で位置を決める
CLUSTER_M = 1.2              # 世界の座標でこれ以内の近くの検出は同じコーン
MATCH_M = 2.0                # 遠くの検出 (の世界の位置) が物の位置からこれ以内なら同じ物とみなす (距離の誤差を許す幅)
GAP_S = 3.0                  # 同じ物を続けて見ている (この時間以上空いたら別の通過)
CAM_X = 0.055                # base_link からカメラの前方オフセット [m]
TL = dict(real_height_m=0.40, focal_length_y=244.42, reference_image_height=360)   # traffic_light_params.yaml と同じ
CAM = dict(fx=733.26, cx=980.22, ref_width=1920)

READER = r'''
import sys, json
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
topics = json.loads(sys.argv[2])
out = []
with open(sys.argv[1], "rb") as fp:
    for _, ch, msg, m in make_reader(fp, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=topics):
        t = msg.log_time / 1e9
        if ch.topic.endswith("odom"):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            out.append(["odom", t, p.x, p.y, 2 * __import__("math").atan2(q.z, q.w)])
        else:
            out.append([ch.topic.split("/")[-2], t, json.loads(m.data)])
json.dump(out, sys.stdout)
'''


def read_bag(bag):
    r = subprocess.run([str(WS_DIR / ".venv/bin/python"), "-c", READER, str(bag), json.dumps(TOPICS)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


class Pose:
    """時刻 -> 車 (base_link) の世界の位置と向き. 正解の位置 (図面の座標) か odom."""

    def __init__(self, t, x, y, yaw, name):
        self.t, self.x, self.y, self.name = np.asarray(t), np.asarray(x), np.asarray(y), name
        self.yaw = np.unwrap(np.asarray(yaw))

    def heading_check(self):
        """向き (yaw) と, 位置の動きから求めた進む向きの差の中央値 [deg] (動いているところだけ)."""
        dx, dy = np.diff(self.x), np.diff(self.y)
        mv = np.hypot(dx, dy) > 0.02
        d = (np.arctan2(dy, dx)[mv] - self.yaw[1:][mv] + np.pi) % (2 * np.pi) - np.pi
        return float(np.degrees(np.median(d)))

    def at(self, t):
        if t < self.t[0] or t > self.t[-1]:
            return None
        return (np.interp(t, self.t, self.x), np.interp(t, self.t, self.y), np.interp(t, self.t, self.yaw))

    def to_world(self, t, bx, by):
        p = self.at(t)
        if p is None:
            return None
        x, y, yaw = p
        c, s = math.cos(yaw), math.sin(yaw)
        return x + c * bx - s * by, y + s * bx + c * by

    def path_length(self):
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))


def signal_detections(rows, offline_json):
    """[(t, 距離 [m], 横の向き tan = 横 / 前)] (確からしさが一番高い 1 つ). 記録 (traffic_light/status) があればそれ."""
    out = []
    for kind, t, d in ((r[0], r[1], r[2]) for r in rows if r[0] == "traffic_light"):
        dets = [x for x in d.get("detections", []) if x.get("distance_m") is not None]
        if not dets:
            continue
        b = max(dets, key=lambda x: x["confidence"])
        H = b["bbox_height_px"] / b["occupancy_ratio"] if b.get("occupancy_ratio") else 360.0
        W = H * 16 / 9                         # 記録に画像の幅がない (2026-10-21 より前) ので 16:9 とする
        s = W / CAM["ref_width"]
        u = 0.5 * (b["bbox"][0] + b["bbox"][2])
        out.append((t, b["distance_m"], (CAM["cx"] * s - u) / (CAM["fx"] * s)))
    if out or not offline_json or not offline_json.exists():
        return out, "rosbag に記録された車載の出力"
    D = json.load(open(offline_json))
    k = TL["real_height_m"] * TL["focal_length_y"] / TL["reference_image_height"]
    for f in D["frames"]:
        dets = [x for x in f.get("tl", []) if x["h"] > 0]
        if not dets:
            continue
        b = max(dets, key=lambda x: x["conf"])
        W = f["size"][0]
        s = W / CAM["ref_width"]
        out.append((f["t"], k / (b["h"] / b["H"]), (CAM["cx"] * s - b["u"]) / (CAM["fx"] * s)))
    return out, "車載と同じ検出器をかけ直した結果 (lane_role_compare.py --signal)"


def cone_detections(rows):
    """[(t, base_link の x, y, 距離)]"""
    out = []
    for r in rows:
        if r[0] == "cone_detector":
            for c in r[2].get("cones", []):
                out.append((r[1], c["x"], c["y"], c["distance"]))
    return out


def clusters(points):
    """世界の座標の点 [(x, y)] を CLUSTER_M 以内でまとめる -> 各まとまりの点の添字のリスト."""
    groups = []
    for i, (x, y) in enumerate(points):
        for g in groups:
            if math.hypot(x - g["c"][0], y - g["c"][1]) < CLUSTER_M:
                g["i"].append(i)
                g["c"] = np.median([points[j] for j in g["i"]], 0)
                break
        else:
            groups.append(dict(i=[i], c=np.array([x, y])))
    return groups


def signal_on_drawing():
    """図面の信号機の位置 (signal の印の付いた中央線上の点の平均) [x, y]. なければ None."""
    st = json.load(open(WS_DIR / "data/topological_map/course_topology.json"))["stations"]
    pts = [(r["x"], r["y"]) for r in st if "signal" in r["landmarks"]]
    return np.mean(pts, 0) if pts else None


def table(rows, title):
    """rows: [(本当の距離, 検出器の距離)] を距離ごとにまとめて表にする."""
    rows = np.array(rows)
    print(f"    {title} ({len(rows)} 回):")
    print(f"    {'本当の距離':>10s} {'回数':>5s} {'比 (検出/本当) 中央値':>20s} {'差 (検出-本当) 中央値':>20s} {'差の 10〜90%':>16s}")
    for a, b in ((1.5, 3), (3, 5), (5, 7), (7, 10), (10, 13), (13, 16), (16, 20), (20, 30)):
        m = (rows[:, 0] >= a) & (rows[:, 0] < b)
        if m.sum() < 3:
            continue
        r, dd = rows[m, 1] / rows[m, 0], rows[m, 1] - rows[m, 0]
        print(f"    {a:4.1f}〜{b:4.1f} m {m.sum():5d} {np.median(r):20.2f} {np.median(dd):+19.2f} m"
              f"   {np.percentile(dd, 10):+.2f}〜{np.percentile(dd, 90):+.2f}")


def true_range(pose, t, obj, from_camera):
    x, y, yaw = pose.at(t)
    dx, dy = obj[0] - x, obj[1] - y
    fx_, fy_ = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.hypot(fx_ - (CAM_X if from_camera else 0.0), fy_)


def compare_absolute(dets, pose, obj):
    """信号機: 図面の位置までの本当の距離と比べる (信号機が見えている向きの ±MATCH_M 以内の検出だけ)."""
    rows = []
    for t, bx, by, d in dets:
        w = pose.to_world(t, bx, by)
        if w is None or math.hypot(w[0] - obj[0], w[1] - obj[1]) > MATCH_M + 0.25 * d:
            continue
        rows.append((true_range(pose, t, obj, True), d))
    if rows:
        table(rows, "図面の信号機の位置までの本当の距離ごとの, 検出器の距離との比べ")


def compare(dets, pose, near, label, ref=None):
    """dets: [(t, base_link の x, y, 検出器の距離)]. 近くの検出で物の位置を決め, 遠くの検出の距離と比べる."""
    world = [pose.to_world(t, x, y) for t, x, y, _ in dets]
    ok = [i for i, w in enumerate(world) if w is not None]
    near_i = [i for i in ok if near[0] <= dets[i][3] <= near[1]]
    groups = [g for g in clusters([world[i] for i in near_i]) if len(g["i"]) >= 2]
    objs = [g["c"] for g in groups]
    print(f"  {label}: 検出 {len(dets)} 回, 近く ({near[0]}〜{near[1]} m) で位置が決まった物 {len(objs)} 個")
    if ref is not None and objs:
        print("    近くの検出で決めた位置と, 図面の信号機の位置の差: "
              + ", ".join(f"{math.hypot(o[0] - ref[0], o[1] - ref[1]):.2f} m" for o in objs))
    if not objs:
        return
    rows = []
    for i in ok:
        t, bx, by, d = dets[i]
        w = world[i]
        k = int(np.argmin([math.hypot(w[0] - o[0], w[1] - o[1]) for o in objs]))
        if math.hypot(w[0] - objs[k][0], w[1] - objs[k][1]) > MATCH_M + 0.15 * d:
            continue
        rows.append((true_range(pose, t, objs[k], label.startswith("信号")), d))
    table(rows, "本当の距離 (近くの検出で決めた位置まで) ごとの, 検出器の距離との比べ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    args = ap.parse_args()
    name = Path(args.bag).stem
    rows = read_bag(args.bag)
    od = np.array([r[1:] for r in rows if r[0] == "odom"])
    truth_p = WS_DIR / f"data/truth/{name}/truth.npy"
    odom = Pose(od[:, 0], od[:, 1], od[:, 2], od[:, 3], "odom") if len(od) else None
    if truth_p.exists():
        T = np.load(truth_p)
        pose = Pose(T[:, 0], T[:, 1], T[:, 2], T[:, 3], "正解の位置 (truth.npy)")
        if odom is not None:
            a, b = max(T[0, 0], od[0, 0]), min(T[-1, 0], od[-1, 0])
            mt, mo = (T[:, 0] >= a) & (T[:, 0] <= b), (od[:, 0] >= a) & (od[:, 0] <= b)
            lt = Pose(T[mt, 0], T[mt, 1], T[mt, 2], T[mt, 3], "").path_length()
            lo = Pose(od[mo, 0], od[mo, 1], od[mo, 2], od[mo, 3], "").path_length()
            print(f"  走った距離: 正解の位置 {lt:.1f} m, odom {lo:.1f} m (odom / 正解 = {lo / lt:.3f})")
    else:
        pose = odom
    print(f"{name}  車の位置: {pose.name}  (向きと進む向きの差の中央値 {pose.heading_check():+.1f} 度)")
    sig, src = signal_detections(rows, WS_DIR / f"data/lane_role_compare/{name}.json")
    print(f"  信号機の検出: {src}")
    if sig:
        dets = [(t, d + CAM_X, d * tn, d) for t, d, tn in sig]
        ref = signal_on_drawing() if pose.name != "odom" else None
        compare(dets, pose, NEAR_SIGNAL, "信号機", ref)
        if ref is not None:
            compare_absolute(dets, pose, ref)
    cones = cone_detections(rows)
    if cones:
        compare(cones, pose, NEAR_CONE, "コーン")
    else:
        print("  コーン: rosbag に検出の記録なし")


if __name__ == "__main__":
    main()
