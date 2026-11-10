"""lane2 の 2 本の rosbag を Foxglove 用の MCAP にする (元の rosbag は読むだけ).

    ~/aiformula_ws/.venv/bin/python ~/aiformula_ws/scripts/convert_lane2_review_mcap.py <session>/6lane <out.mcap>

入れるもの:
  - rosbag の全トピック (data_0.db3 を sqlite で直接読む. metadata.yaml がなくてよい)
  - /camera/image/compressed, /panel/image/compressed (mp4 + stamps.csv から)
  - /analysis/planner, /analysis/traffic_light, /analysis/cones : status の JSON をほどいた数値 (Plot で見られる)
  - /analysis/image_annotations : 信号とコーンの検出の枠をカメラ画像に重ねる (foxglove.ImageAnnotations)
  - /analysis/events : 調べた出来事の説明 (foxglove.Log)
"""
import csv, json, os, sqlite3, sys
from pathlib import Path

import cv2
from mcap.writer import Writer

sys.path.insert(0, os.path.expanduser('~/aiformula_ws/scripts'))
from convert_bag_to_mcap import generate_ros2_msgdef, encode_cdr_compressed_image  # noqa: E402

from rosbags.typesys import get_typestore, Stores  # noqa: E402
TS = get_typestore(Stores.ROS2_HUMBLE)

# 調べた出来事 (記録開始からの秒). 2026-11-08 の調査結果
EVENTS = {
    '135955': [
        (93.4, 3, 'コーン: 右前方の3本でレーン3,4が塞がれる'),
        (95.2, 3, 'コーン: 塞がれたレーンが 2,3,4 に増え, 左端のレーン1へ逃げる'),
        (99.0, 4, 'コーン: レーン1も塞がれ, コーン列の向こうのレーン5を目標にして約2m手前で右へ横切ろうとする'),
        (100.0, 4, 'コーン: 反応的回避は「左へ回避」, 目標レーンは右 → 逆向き. 減速 (最小 0.29 m/s)'),
        (104.9, 2, 'コーン: 通過して基準のレーン2へ戻る'),
        (116.6, 4, '信号: フェンス際の人・物を赤信号と誤検出 (高さ約6px, 信頼度0.36, 距離15.6mと誤算)'),
        (116.8, 4, '信号: 2回連続で赤 → 減速開始. 以後は赤が見えなくても停止まで減速を続ける'),
        (135.4, 3, '信号: 誤検出で完全に停止 → 赤が3秒見えないので発進'),
        (157.2, 2, '信号: 本物の信号へ接近 (正常)'),
        (175.1, 2, '信号: 本物の赤で停止 (高さ約18px, 5.3m, 信頼度0.56〜0.7)'),
        (194.6, 2, '信号: 青を確認して発進 (正常)'),
    ],
    '154757': [
        (116.8, 2, 'コーン: レーン3が塞がれレーン1へ. 通過後に戻る'),
        (133.0, 4, '信号: フェンス際のピンクの柵あたりを赤と誤検出 (高さ約6px, 信頼度0.35, 15.6m)'),
        (133.1, 4, '信号: 2回連続で赤 → 減速開始 (0.09 m/s まで落ちる)'),
        (140.1, 3, '信号: 赤が見えない → 発進'),
        (148.7, 3, '信号: 赤の誤検出 (短い減速)'),
        (150.8, 3, 'コーン: 左前方の2本でレーン2が塞がれ, レーン3へ'),
        (154.0, 4, 'コーン: レーン3も塞がれ, 目標がレーン1へ反転. 目の前のコーンを横切る向き (最小 1.10 m). 減速 0.41 m/s'),
        (157.2, 2, 'コーン: 通過して基準のレーン2へ戻る'),
        (166.6, 3, '信号: 本物の青と同時に, 別の場所を赤と誤検出'),
        (193.6, 2, 'コーン: レーン2が塞がれレーン3へ (正常)'),
        (219.6, 4, '信号: 逆光の中の駐車車両・建物を赤と誤検出 → 減速 (0.25 m/s まで)'),
        (225.6, 3, '信号: 赤が見えない → 発進'),
    ],
}
TL_CODE = {'NORMAL': 0, 'APPROACH': 1, 'STOPPED': 2, 'RESUME': 3}


def stamp(ns):
    return {'sec': ns // 1_000_000_000, 'nsec': ns % 1_000_000_000}


def main(session, out):
    session = Path(session)
    key = '154757' if '154757' in str(session) else '135955'
    db = sqlite3.connect(f"file:{session / 'data' / 'data_0.db3'}?mode=ro", uri=True)
    topics = {r[0]: (r[1], r[2]) for r in db.execute('select id,name,type from topics')}
    t0 = db.execute('select min(timestamp) from messages').fetchone()[0]

    f = open(out, 'wb')
    w = Writer(f)
    w.start(profile='ros2', library='lane2-review')
    schemas, chans = {}, {}

    def ros_chan(topic, typ):
        if topic not in chans:
            if typ not in schemas:
                schemas[typ] = w.register_schema(name=typ, encoding='ros2msg', data=generate_ros2_msgdef(typ).encode())
            chans[topic] = w.register_channel(topic=topic, message_encoding='cdr', schema_id=schemas[typ])
        return chans[topic]

    def json_chan(topic, name):
        if topic not in chans:
            if name not in schemas:
                schemas[name] = w.register_schema(name=name, encoding='jsonschema', data=json.dumps({'type': 'object'}).encode())
            chans[topic] = w.register_channel(topic=topic, message_encoding='json', schema_id=schemas[name])
        return chans[topic]

    msgs = []  # (log_time_ns, channel, bytes)
    for tid, (name, typ) in topics.items():
        ch = ros_chan(name, typ)
        for ts, data in db.execute('select timestamp,data from messages where topic_id=?', (tid,)):
            msgs.append((ts, ch, data))
            if not typ.startswith('std_msgs/msg/String'):
                continue
            m = TS.deserialize_cdr(data, typ)
            try:
                j = json.loads(m.data)
            except Exception:
                continue
            if name == '/aiformula_control/six_lane_planner/status':
                o = {k: j.get(k) for k in ('phase', 'current_lane', 'target_lane', 'confidence', 'kappa_path', 'speed_limit', 'driving_mode', 'lost_time')}
                o.update(v_cmd=j.get('v'), v_meas=j.get('v_meas'), blocked=j.get('blocked', []), blocked_count=len(j.get('blocked', [])),
                         lane_pos=None if j.get('F') is None else j['F'] + 0.5, avoid=j.get('avoid'), explain=j.get('explain'))
                msgs.append((ts, json_chan('/analysis/planner', 'lane2_review.Planner'), json.dumps(o, ensure_ascii=False).encode()))
            elif name == '/aiformula_control/traffic_light_stop/status':
                o = {k: j.get(k) for k in ('state', 'v_cap', 'red_distance', 'green_distance', 'distance', 'red_count', 'reason')}
                o['state_code'] = TL_CODE.get(j.get('state'), -1)
                msgs.append((ts, json_chan('/analysis/traffic_light', 'lane2_review.TrafficLight'), json.dumps(o, ensure_ascii=False).encode()))
            elif name == '/aiformula_perception/traffic_light/status' or name == '/aiformula_perception/cone_detector/status':
                is_tl = 'traffic_light' in name
                dets = j.get('detections', []) if is_tl else j.get('cones', [])
                pts, txt = [], []
                for d in dets:
                    x1, y1, x2, y2 = d['bbox']
                    red = is_tl and 'red' in d.get('class_name', '')
                    col = {'r': 1, 'g': 0.1, 'b': 0.1, 'a': 1} if red else ({'r': 0.1, 'g': 0.9, 'b': 0.2, 'a': 1} if is_tl else {'r': 1, 'g': 0.6, 'b': 0, 'a': 1})
                    pad = 4 if is_tl else 0
                    pts.append({'timestamp': stamp(ts), 'type': 2, 'thickness': 2, 'outline_color': col,
                                'points': [{'x': x1 - pad, 'y': y1 - pad}, {'x': x2 + pad, 'y': y1 - pad}, {'x': x2 + pad, 'y': y2 + pad}, {'x': x1 - pad, 'y': y2 + pad}]})
                    label = (f"{'赤' if red else '青'} {d['confidence']:.2f} {d.get('distance_m') or 0:.1f}m h{d.get('bbox_height_px', 0):.0f}px" if is_tl
                             else f"cone {d['conf']:.2f} ({d['x']:.1f},{d['y']:+.1f})")
                    txt.append({'timestamp': stamp(ts), 'position': {'x': x1, 'y': max(10, y1 - 8)}, 'text': label, 'font_size': 11,
                                'text_color': col, 'background_color': {'r': 0, 'g': 0, 'b': 0, 'a': 0.6}})
                topic = '/analysis/annotations/traffic_light' if is_tl else '/analysis/annotations/cones'
                msgs.append((ts, json_chan(topic, 'foxglove.ImageAnnotations'), json.dumps({'circles': [], 'points': pts, 'texts': txt}, ensure_ascii=False).encode()))
                if not is_tl:
                    near = min(dets, key=lambda d: d['x']) if dets else None
                    o = {'num': len(dets), 'nearest_x': near and near['x'], 'nearest_y': near and near['y']}
                    msgs.append((ts, json_chan('/analysis/cones', 'lane2_review.Cones'), json.dumps(o).encode()))

    for sec, level, text in EVENTS[key]:
        ts = t0 + int(sec * 1e9)
        o = {'timestamp': stamp(ts), 'level': level, 'message': f'[{sec:.1f}s] {text}', 'name': 'lane2-review', 'file': '', 'line': 0}
        msgs.append((ts, json_chan('/analysis/events', 'foxglove.Log'), json.dumps(o, ensure_ascii=False).encode()))

    video = session / 'video'
    for mp4, csvf, topic, frame in [('camera.mp4', 'camera_stamps.csv', '/camera/image/compressed', 'zed_left_camera_optical_frame'),
                                    ('panel.mp4', 'panel_stamps.csv', '/panel/image/compressed', 'panel_frame')]:
        if not (video / mp4).exists():
            continue
        stamps = [float(r['stamp_sec']) for r in csv.DictReader(open(video / csvf))]
        ch = ros_chan(topic, 'sensor_msgs/msg/CompressedImage')
        cap = cv2.VideoCapture(str(video / mp4)); i = 0
        while i < len(stamps):
            ok, img = cap.read()
            if not ok:
                break
            ns = int(round(stamps[i] * 1e9))
            jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            msgs.append((ns, ch, encode_cdr_compressed_image(ns // 1_000_000_000, ns % 1_000_000_000, frame, 'jpeg', jpg)))
            i += 1

    msgs.sort(key=lambda m: m[0])
    for ts, ch, data in msgs:
        w.add_message(channel_id=ch, log_time=ts, publish_time=ts, data=data)
    w.finish(); f.close()
    print(out, len(msgs), 'messages', f'{os.path.getsize(out) / 1e6:.0f} MB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
