#!/usr/bin/env python3
"""
topo_features.py - rosbag から, トポロジカルマップでの位置推定に使う「その瞬間の手がかり」を取り出す

毎フレーム (FPS コマ/秒に間引く):
  - v, yaw_rate: その瞬間の車速 [m/s] と旋回の速さ [rad/s] (gyro_odometry の twist. 位置 (積算) は使わない)
  - kappa: その瞬間の車の曲がり具合 = yaw_rate / v (遅いときは None). レーン移動で揺れるのでコースの形には使いにくい
  - lines: YOLOP (車載と同じ letterbox の前処理) で取った白線ごとに
        y2: 前方 2 m での横位置 [m] (左正), x0/x1: 見えている前後の範囲 [m], c: y = c0 + c1 x + c2 x^2 の係数
            (白線の曲がり具合 = 道路の形. 車がレーンを移っても変わらない),
        cover: 線に沿って画像で白く見える割合 (破線は途切れるので小さい), kind: solid / dashed / unknown
  - signal: 信号 (traffic_light) の検出があり 25 m 以内か (rosbag に記録された検出を使う. gamepad の rosbag には記録がない)
  - tl: --add-signal で後から足す. 車載と同じ信号の検出器 (models/traffic_light.pt, conf 0.35, imgsz 640) を
        その場で映像にかけた検出 [{cls: red/green, conf, h: 枠の高さ [px], H: 画像の高さ [px]}].
        距離への換算 (係数) は topo_localize_eval.py 側で持つ (rosbag に記録がなくても使えるようにするため)
  - lines[].near: --add-near で後から足す. 各白線の, 画面に入る一番手前から NEAR_LEN m 先までを 0.1 m ごとに
        白く見えるか (1/0/-1 = 画面外) の並び {x_a: 始めの距離 [m], bits: "1101..."}.
        中央の破線は 1 本 3.6 m・途切れ 2.8 m と長く, 1 コマでは途切れが写らないことが多い. 区間推定の側で
        数コマ (6.5 m ほど進む間) 続けて見て, 途切れが通れば破線, ずっと白なら実線と決めるのに使う
  - lines[].road: --add-road で後から足す. 各白線の左右が「走れる路面」か を, 前方 ROAD_XS [m] × 線からの横の距離
        ROAD_DS [m] の格子で並べた文字列 (1 = 路面, 0 = 路面でない, - = 画面の外. 並びは x ごとに 左の ROAD_DS, 右の ROAD_DS).
        路面は honda_shihou_finetuned_road_v2_best.pth の走れる領域 (白線の出力は車載の重みと同じ). 役割づけの手がかり
        (境界線の外側は路面が続かない, 中央線は両側に路面がある) に使う
  - vo: --add-motion で後から足す. 前のコマ (0.2 s 前) からカメラ映像だけで測った進み (topo_visual_motion.py)
        {dx: 進み [m], sharp: 確からしさ, best: 一番よく合ったときの一致, c0: ずらさないときの一致}

使い方 (Mac 本体の python3. rosbag の読み出しは .venv の python に任せる):
  python3 scripts/topo_features.py rosbag/mcap/20261012_134520_6lane.mcap data/topological_map/features_134520.json
  python3 scripts/topo_features.py --add-signal rosbag/mcap/20261012_134520_6lane.mcap data/topological_map/features_134520.json
  python3 scripts/topo_features.py --add-near rosbag/mcap/20261012_134520_6lane.mcap data/topological_map/features_134520.json
  python3 scripts/topo_features.py --add-road rosbag/mcap/20261012_134520_6lane.mcap data/topological_map/features_134520.json
  python3 scripts/topo_features.py --add-motion rosbag/mcap/20261012_134520_6lane.mcap data/topological_map/features_134520.json
"""

import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

WS_DIR = Path(__file__).resolve().parent.parent
FPS = 5.0
TOPICS = ["/camera/image/compressed", "/aiformula_sensing/gyro_odometry_publisher/odom",
          "/aiformula_perception/traffic_light/status"]


def run_reader(bag):
    """rosbag を時刻順に [ヘッダ長][JSON][データ長][データ] で標準出力へ (読み取りのみ)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    out = sys.stdout.buffer
    with open(bag, "rb") as fp:
        for _, ch, msg, m in make_reader(fp, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=TOPICS):
            t = msg.log_time / 1e9
            if ch.topic.endswith("compressed"):
                head, data = {"k": "img", "t": t}, bytes(m.data)
            elif ch.topic.endswith("odom"):
                head, data = {"k": "odom", "t": t, "v": m.twist.twist.linear.x, "w": m.twist.twist.angular.z}, b""
            else:
                d = json.loads(m.data)
                near = [x for x in d.get("detections", []) if (x.get("distance_m") or 99) < 25]
                head, data = {"k": "tl", "t": t, "n": len(near)}, b""
            h = json.dumps(head).encode()
            try:
                out.write(struct.pack("<I", len(h)) + h + struct.pack("<I", len(data)) + data)
                out.flush()
            except BrokenPipeError:
                return


def iter_images(bag):
    """rosbag の画像を (時刻, JPEG のバイト列) で順に返す (読み出しは .venv の python)."""
    reader = subprocess.Popen([str(WS_DIR / ".venv/bin/python"), os.path.abspath(__file__), "--reader", os.path.abspath(bag)],
                              stdout=subprocess.PIPE)
    inp = reader.stdout
    try:
        while True:
            b = inp.read(4)
            if len(b) < 4:
                return
            head = json.loads(inp.read(struct.unpack("<I", b)[0]))
            data = inp.read(struct.unpack("<I", inp.read(4))[0])
            if head["k"] == "img":
                yield head["t"], data
    finally:
        reader.kill()


def add_signal(bag, path):
    """features の各コマに, 車載と同じ信号の検出器をかけた結果 (tl) を足して上書きする."""
    import numpy as np
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(WS_DIR / "models/traffic_light.pt"))
    rows = json.load(open(path))
    k = 0
    for t, data in iter_images(bag):
        if k >= len(rows):
            break
        if abs(t - rows[k]["t"]) > 2e-3:
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        res = model.predict(img, conf=0.35, iou=0.45, imgsz=640, verbose=False)[0]
        rows[k]["tl"] = [dict(cls=model.names[int(b.cls[0])].replace("traffic_light_", ""), conf=round(float(b.conf[0]), 3),
                              h=round(float(b.xyxy[0][3] - b.xyxy[0][1]), 1), H=img.shape[0]) for b in res.boxes]
        k += 1
        if k % 500 == 0:
            print(k, "frames", flush=True)
    json.dump(rows, open(path, "w"))
    print("frames", k, "/", len(rows), "->", path, " 信号を検出したコマ", sum(bool(r.get("tl")) for r in rows))


def add_motion(bag, path):
    """features の各コマに, 前のコマからカメラ映像だけで測った進み (vo) を足して上書きする."""
    import numpy as np
    import cv2
    import topo_visual_motion as M
    rows = json.load(open(path))
    k, prev = 0, None
    for t, data in iter_images(bag):
        if k >= len(rows):
            break
        if abs(t - rows[k]["t"]) > 2e-3:
            continue
        cur = M.prepare(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
        rows[k]["vo"] = M.measure(prev, cur) if prev is not None else None
        prev = cur
        k += 1
        if k % 500 == 0:
            print(k, "frames", flush=True)
    json.dump(rows, open(path, "w"))
    print("frames", k, "/", len(rows), "->", path)


NEAR_LEN = 2.5     # 白く見えるかを記録する長さ [m] (画面に入る一番手前から)


def near_bits(L, c, to_image, fx, cx):
    """白線 y = c0 + c1 x + c2 x^2 の, 画面に入る一番手前から NEAR_LEN m を 0.1 m ごとに白く見えるか (1/0, 画面外 -1).
    白さの判定は line_features と同じ (線の上が横 ±12 px の中央値より 12 以上明るい)."""
    import numpy as np
    y2 = c[0] + 2 * c[1] + 4 * c[2]
    x_a = max(1.0, abs(y2) * fx / (cx - 14) + 0.2)          # 横に離れた線は手前が画面の外に出る
    bits = []
    for xx in np.arange(x_a, x_a + NEAR_LEN, 0.1):
        u, v = to_image(xx, c[0] + c[1] * xx + c[2] * xx * xx)
        u, v = int(round(u)), int(round(v))
        if not (14 <= u < 626 and 0 <= v < 360):
            bits.append("-")
            continue
        row = L[v, u - 14:u + 15]
        bits.append("1" if row[10:19].max() - np.median(row) > 12 else "0")
    return dict(x_a=round(float(x_a), 2), bits="".join(bits))


ROAD_XS = [1.0 + 0.5 * i for i in range(15)]      # 前方 1.0〜8.0 m
ROAD_DS = [0.2 * (i + 1) for i in range(18)]       # 線から 0.2〜3.6 m 外 (車線の幅 3.66 m まで)
ROAD_MODEL = WS_DIR / "models/honda_shihou_finetuned_road_v2_best.pth"


def road_grid(R, c, to_image):
    """白線 y = c0 + c1 x + c2 x^2 の左右の路面を ROAD_XS × ROAD_DS の格子で読む (R: 640x360 の路面 0/1, 上を切った所は -1)."""
    out = []
    for x in ROAD_XS:
        yl = c[0] + c[1] * x + c[2] * x * x
        for side in (1, -1):                       # 左 (+), 右 (-)
            for d in ROAD_DS:
                u, v = to_image(x, yl + side * d)
                u, v = int(round(u)), int(round(v))
                if not (0 <= u < 640 and 0 <= v < 360) or R[v, u] < 0:
                    out.append("-")
                else:
                    out.append("1" if R[v, u] else "0")
    return "".join(out)


def add_road(bag, path):
    """features の各白線に, 左右の路面の格子 (road) を足して上書きする."""
    import numpy as np
    import cv2
    import torch
    sys.path.insert(0, str(WS_DIR / "scripts"))
    import make_road_masks as M
    import verify_lane_rosbag as V
    FX, CX, CY, H, CAMX, PIT = 733.26 / 3, 980.22 / 3, 516.63 / 3, 0.56, 0.055, math.radians(0.55)

    def to_image(x, y):
        xr = x - CAMX; zc = math.cos(PIT) * xr + math.sin(PIT) * H; yd = -(math.sin(PIT) * xr - math.cos(PIT) * H)
        return CX + FX * (-y) / zc, CY + FX * yd / zc

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = M.load_net(ROAD_MODEL, dev)
    rows = json.load(open(path))
    k = 0
    for t, data in iter_images(bag):
        if k >= len(rows):
            break
        if abs(t - rows[k]["t"]) > 2e-3:
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        cut = int(img.shape[0] * V.TOP_CUT_RATIO)
        full = np.full(img.shape[:2], -1, np.int8)            # 車載と同じく上を切った所は見ていない
        full[cut:] = M.infer_da(net, dev, img[cut:])
        R = cv2.resize(full, (640, 360), interpolation=cv2.INTER_NEAREST)
        for l in rows[k]["lines"]:
            if l.get("c"):
                l["road"] = road_grid(R, l["c"], to_image)
        k += 1
        if k % 500 == 0:
            print(k, "frames", flush=True)
    json.dump(rows, open(path, "w"))
    print("frames", k, "/", len(rows), "->", path)


def add_near(bag, path):
    """features の各白線に, 手前 NEAR_LEN m の白さの並び (near) を足して上書きする."""
    import numpy as np
    import cv2
    FX, CX, CY, H, CAMX, PIT = 733.26 / 3, 980.22 / 3, 516.63 / 3, 0.56, 0.055, math.radians(0.55)

    def to_image(x, y):
        xr = x - CAMX; zc = math.cos(PIT) * xr + math.sin(PIT) * H; yd = -(math.sin(PIT) * xr - math.cos(PIT) * H)
        return CX + FX * (-y) / zc, CY + FX * yd / zc

    rows = json.load(open(path))
    k = 0
    for t, data in iter_images(bag):
        if k >= len(rows):
            break
        if abs(t - rows[k]["t"]) > 2e-3:
            continue
        img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (640, 360))
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        for l in rows[k]["lines"]:
            if l.get("c"):
                l["near"] = near_bits(L, l["c"], to_image, FX, CX)
        k += 1
        if k % 500 == 0:
            print(k, "frames", flush=True)
    json.dump(rows, open(path, "w"))
    print("frames", k, "/", len(rows), "->", path)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--add-near":
        add_near(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--add-road":
        add_road(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--add-motion":
        sys.path.insert(0, str(WS_DIR / "scripts"))
        add_motion(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--reader":
        run_reader(sys.argv[2])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--add-signal":
        add_signal(sys.argv[2], sys.argv[3])
        return
    bag, out_path = sys.argv[1], sys.argv[2]
    import numpy as np
    import cv2
    import torch
    sys.path.insert(0, str(WS_DIR / "scripts"))
    import verify_lane_rosbag as V           # 前処理 (letterbox) と白線の取り出しは確認用の道具と同じもの
    sys.path.insert(0, str(WS_DIR / "src/aiformula_base/oit_navigation/oit_navigation/yolop"))
    from lib.config import cfg
    from lib.models import get_net

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = get_net(cfg)
    ck = torch.load(WS_DIR / "models/honda_shihou_finetuned_best.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    net = net.to(dev).float().eval()
    mean, std = np.asarray(V.NORM_MEAN, np.float32), np.asarray(V.NORM_STD, np.float32)

    def infer(bgr):
        h, w = bgr.shape[:2]
        cut = int(h * V.TOP_CUT_RATIO)
        proc = bgr[cut:]
        ph, pw = proc.shape[:2]
        r = min(V.INPUT_SIZE / ph, V.INPUT_SIZE / pw)
        nh, nw = int(round(ph * r)), int(round(pw * r))
        top, left = (V.INPUT_SIZE - nh) // 2, (V.INPUT_SIZE - nw) // 2
        canvas = np.full((V.INPUT_SIZE, V.INPUT_SIZE, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(proc, (nw, nh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(((canvas.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])).to(dev)
        with torch.inference_mode():
            _, _, ll = net(x)
        seg = (ll[0, 1, top:top + nh, left:left + nw] > ll[0, 0, top:top + nh, left:left + nw]).to(torch.uint8).cpu().numpy()
        m = np.zeros((h, w), np.uint8)
        m[cut:] = cv2.resize(seg, (pw, ph), interpolation=cv2.INTER_NEAREST)
        return m

    # カメラ (車載 lane_detector_params.yaml と同じ. 640x360)
    FX, CX, CY, H, CAMX, PIT = 733.26 / 3, 980.22 / 3, 516.63 / 3, 0.56, 0.055, math.radians(0.55)

    def to_ground(u, v):
        xc = (u - CX) / FX; yc = (v - CY) / FX
        fwd = math.cos(PIT) - yc * math.sin(PIT); down = math.sin(PIT) + yc * math.cos(PIT)
        t = np.where(down > 0.02, H / np.maximum(down, 1e-6), np.nan)
        return CAMX + t * fwd, -t * xc

    def to_image(x, y):
        xr = x - CAMX; zc = math.cos(PIT) * xr + math.sin(PIT) * H; yd = -(math.sin(PIT) * xr - math.cos(PIT) * H)
        return CX + FX * (-y) / zc, CY + FX * yd / zc

    def line_features(img, lane):
        x, y = to_ground(np.asarray(lane["u"], float), np.asarray(lane["v"], float))
        ok = np.isfinite(x) & (x > 0.8) & (x < 12)
        if ok.sum() < 4 or np.ptp(x[ok]) < 0.8:
            return None
        A = np.stack([np.ones(ok.sum()), x[ok], x[ok] ** 2], 1); w = 1 / np.maximum(x[ok], 1)
        c = np.linalg.lstsq(A * w[:, None], y[ok] * w, rcond=None)[0]
        x0, x1 = float(x[ok].min()), float(x[ok].max())
        # 線に沿って (x0〜min(x1, 9) m を 0.1 m ごと) 画像で白く見えるか: 周り (横 ±12 px) より明るい
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        xs = np.arange(max(x0, 1.0), min(x1, 9.0), 0.1)
        if len(xs) < 10:
            return dict(y2=float(c[0] + 2 * c[1] + 4 * c[2]), x0=x0, x1=x1, c=[float(v) for v in c], cover=None, kind="unknown")
        hit = []
        for xx in xs:
            u, v = to_image(xx, c[0] + c[1] * xx + c[2] * xx * xx)
            u, v = int(round(u)), int(round(v))
            if not (14 <= u < 626 and 0 <= v < 360):
                continue
            row = L[v, u - 14:u + 15]
            hit.append(row[10:19].max() - np.median(row) > 12)
        if len(hit) < 10:
            return dict(y2=float(c[0] + 2 * c[1] + 4 * c[2]), x0=x0, x1=x1, c=[float(v) for v in c], cover=None, kind="unknown")
        cover = float(np.mean(hit)); span = xs[-1] - xs[0]
        kind = "solid" if cover > 0.8 else ("dashed" if cover < 0.65 and span >= 3.0 else "unknown")
        return dict(y2=float(c[0] + 2 * c[1] + 4 * c[2]), x0=x0, x1=x1, c=[float(v) for v in c], cover=round(cover, 2), kind=kind)

    reader = subprocess.Popen([str(WS_DIR / ".venv/bin/python"), os.path.abspath(__file__), "--reader", os.path.abspath(bag)],
                              stdout=subprocess.PIPE)
    inp = reader.stdout
    v = w = 0.0
    tl_t = -1e9
    last_img_t = -1e9
    rows = []
    t0 = None
    while True:
        b = inp.read(4)
        if len(b) < 4:
            break
        head = json.loads(inp.read(struct.unpack("<I", b)[0]))
        data = inp.read(struct.unpack("<I", inp.read(4))[0])
        t = head["t"]
        t0 = t if t0 is None else t0
        if head["k"] == "odom":
            v, w = head["v"], head["w"]
            continue
        if head["k"] == "tl":
            if head["n"]:
                tl_t = t
            continue
        if t - last_img_t < 1.0 / FPS - 1e-3:
            continue
        last_img_t = t
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        lines = [f for f in (line_features(img, l) for l in V.extract_mask_lines(infer(img))) if f]
        rows.append(dict(t=round(t, 3), rt=round(t - t0, 2), v=round(v, 3), yaw_rate=round(w, 4),
                         kappa=round(w / v, 4) if v > 0.3 else None, signal=(t - tl_t) < 0.6, lines=lines))
        if len(rows) % 200 == 0:
            print(len(rows), "frames", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(rows, open(out_path, "w"))
    print("frames", len(rows), "->", out_path)


if __name__ == "__main__":
    main()
