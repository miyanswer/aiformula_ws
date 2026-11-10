"""rosbag の入力で 6レーンのプランナー (+ コーンの反応的回避 + 信号停止) を動かし直す (開ループ).

six_lane_planner_node._control_step と同じ順序・同じ計算を ROS なしで再現する. 車は記録どおりに動くので,
判断を変えても車の動きは変わらない (記録の時刻ごとの判断だけを比べる).

    python replay.py <bag dir (…/6lane)> <src root (oit_navigation のある src/oit_navigation)> <params.yaml> <out.json> [--patch name]

入力の作り方 (rosbag に LaneLines が記録されていないので近似):
  - 白線: lane_line_publisher/lane_lines/{left,center,right} (nav_msgs/Path, 当てはめた線を 0.5 m おきに並べたもの) を
    2 次式に当てはめ直す. x が 1.0〜8.0 ちょうどなら補完線 (detected=false), それ以外は検出線とし,
    点群 (px, py) には Path の点を使う (本物の点群ではない).
  - 付け直し (reanchored): 記録された lane_reseed の直後の白線で true.
  - 車速: CAN 車輪速 + vectornav の前後加速度 (SpeedEstimator). ヨーレートも vectornav.
  - 速度上限・走行モード: 記録された status の値を時刻どおりに入れる.
"""
import bisect, json, math, sqlite3, struct, sys
import numpy as np
import yaml
from rosbags.typesys import get_typestore, Stores

TS = get_typestore(Stores.ROS2_HUMBLE)
FRAME_DEF = 'std_msgs/Header header\nuint32 id\nbool is_rtr\nbool is_extended\nbool is_error\nuint8 dlc\nuint8[8] data\n'
from rosbags.typesys import get_types_from_msg  # noqa: E402
TS.register(get_types_from_msg(FRAME_DEF, 'can_msgs/msg/Frame'))


def load(bag, src, params_yaml):
    sys.path.insert(0, src)
    global core, TrafficLightStop, TrafficLightStopParams, ReactiveAvoider
    from oit_navigation.six_lane import six_lane_core as core
    from oit_navigation.traffic_light.traffic_light_stop import TrafficLightStop, TrafficLightStopParams
    from oit_navigation.cone.cone_avoidance import ReactiveAvoider
    db = sqlite3.connect(f'file:{bag}/data/data_0.db3?mode=ro', uri=True)
    tp = {r[1]: (r[0], r[2]) for r in db.execute('select id,name,type from topics')}

    def rows(name):
        tid, typ = tp[name]
        for ts, d in db.execute('select timestamp,data from messages where topic_id=? order by timestamp', (tid,)):
            yield ts * 1e-9, TS.deserialize_cdr(d, typ)
    stamp = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
    ev = []  # (受信時刻, 種類, 中身)
    paths = {}
    for role in core.ROLES:
        for t, m in rows(f'/aiformula_perception/lane_line_publisher/lane_lines/{role}'):
            paths.setdefault(stamp(m.header), {})[role] = (t, [(p.pose.position.x, p.pose.position.y) for p in m.poses])
    for st, d in paths.items():
        if len(d) == 3:
            ev.append((max(v[0] for v in d.values()), 'lines', (st, {r: d[r][1] for r in core.ROLES})))
    for t, m in rows('/aiformula_perception/cone_detector/cones'):
        ev.append((t, 'cones', (stamp(m.header), [(p.position.x, p.position.y) for p in m.poses])))
    for t, m in rows('/aiformula_sensing/vehicle_info'):
        if m.id == 1809 and len(m.data) >= 8:
            r, l = struct.unpack('<ii', bytes(m.data[:8]))
            ev.append((t, 'can', 0.5 * (l + r) / 60.0 * math.pi * 0.254))
    for t, m in rows('/aiformula_sensing/vectornav/imu'):
        ev.append((t, 'imu', (m.linear_acceleration.x, m.angular_velocity.z)))
    for t, m in rows('/aiformula_perception/traffic_light/red_distance'):
        ev.append((t, 'red', float(m.data)))
    for t, m in rows('/aiformula_perception/traffic_light/green_distance'):
        ev.append((t, 'green', float(m.data)))
    for t, m in rows('/aiformula_control/six_lane_planner/lane_reseed'):
        ev.append((t, 'reseed', None))
    rec = []
    for t, m in rows('/aiformula_control/six_lane_planner/status'):
        j = json.loads(m.data); rec.append((t, j))
        ev.append((t, 'mode', (j.get('driving_mode'), j.get('speed_limit'))))
    ev.sort(key=lambda e: e[0])
    t0 = db.execute('select min(timestamp) from messages').fetchone()[0] * 1e-9
    y = next(iter(yaml.safe_load(open(params_yaml)).values()))['ros__parameters']
    return ev, rec, t0, y


def line_obs(pts):
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts]); yv = np.array([p[1] for p in pts])
    c = np.polyfit(x, yv, 2) if len(pts) >= 3 else np.append(np.polyfit(x, yv, 1), 0.0)[[0, 1, 2]]
    c = [float(c[2]), float(c[1]), float(c[0])] if len(pts) >= 3 else [float(c[1]), float(c[0]), 0.0]
    inferred = abs(x[0] - 1.0) < 1e-6 and abs(x[-1] - 8.0) < 1e-6
    return core.LineObs(c=c, x_min=float(x[0]), x_max=float(x[-1]), detected=not inferred,
                        px=None if inferred else list(map(float, x)), py=None if inferred else list(map(float, yv)))


def run(bag, src, params_yaml, patch=None):
    ev, rec, t0, y = load(bag, src, params_yaml)
    p = core.SixLaneParams()
    for name in p.__dataclass_fields__:
        if name in y and name not in ('v_max', 'driving_mode'):
            default = getattr(p, name)
            v = y[name]
            setattr(p, name, tuple(float(a) for a in v) if isinstance(default, tuple) else type(default)(v))
    import os
    net = core.LanePolicyNet.load(os.path.join(src, 'oit_navigation/six_lane/six_lane_policy.json'))
    planner = core.SixLanePlanner(net, p)
    tlp = TrafficLightStopParams()
    for k, v in y.items():
        if k.startswith('traffic_light_stop.') and hasattr(tlp, k.split('.', 1)[1]):
            f = k.split('.', 1)[1]; setattr(tlp, f, type(getattr(tlp, f))(v))
    tl = TrafficLightStop(tlp)
    avoider = ReactiveAvoider(enabled=bool(y.get('cone_avoidance.reactive', True)))
    if patch:
        patch(planner, tl, avoider)
    est = core.SpeedEstimator(tau=float(y.get('speed_estimator_tau', 1.0)), scale=float(y.get('wheel_speed_scale', 1.0)))
    gate_l, gate_c = core.LatencyGate(p.latency_max), core.LatencyGate(p.latency_max)
    S = dict(v=0.0, v_wheel=0.0, yaw=0.0, imu_t=-1e9, acc=[], last_est=None, pending=None, lstamp=None, rean=False,
             cones=[], cones_t=-1e9, cstamp=None, last_lines_t=None, last_step=None, raw=(0.0, 0.0), last_cmd=None,
             red=None, green=None, reseed=False)
    cones_timeout = float(y.get('cones_timeout', 0.5)); rate = float(y.get('control_rate', 20.0))
    out = []
    next_ctrl = ev[0][0]
    i = 0

    def publish(now, raw, cones):
        dt = 0.0 if S['last_cmd'] is None else min(now - S['last_cmd'], 0.5); S['last_cmd'] = now
        v, om = avoider.step(now, dt, raw[0], raw[1], cones, core.effective_control(planner.p, S['v'])['react_scale'])
        if S['red'] is not None or S['green'] is not None:
            t = max(x[0] for x in (S['red'], S['green']) if x is not None)
            tl.observe(t, S['red'][1] if S['red'] else None, S['green'][1] if S['green'] else None)
            S['red'] = S['green'] = None
        v, om = tl.apply(now, dt, S['v'], v, om)
        return v, om

    def cones_now(now):
        cs = list(S['cones']) if now - S['cones_t'] < cones_timeout else []
        if not cs:
            return cs
        age, ok = gate_c.check(None if S['cstamp'] is None else now - S['cstamp'])
        omega = S['yaw'] if now - S['imu_t'] < 0.3 else S['raw'][1]
        return core.compensate_points(cs, S['v'], omega, age, p.latency_max) if ok else []

    while i < len(ev):
        t, kind, d = ev[i]
        if t >= next_ctrl:  # 制御周期 (20 Hz). その時刻までに届いたものだけで判断する
            now = next_ctrl; next_ctrl += 1.0 / rate
            acc = sum(S['acc']) / len(S['acc']) if S['acc'] else None; S['acc'] = []
            dt = 0.0 if S['last_est'] is None else now - S['last_est']; S['last_est'] = now
            S['v'] = est.update(S['v_wheel'], acc if now - S['imu_t'] < 0.3 else None, dt)
            omega = S['yaw'] if now - S['imu_t'] < 0.3 else S['raw'][1]
            lines, S['pending'] = S['pending'], None
            rean, S['rean'] = S['rean'], False
            hold = min(0.3, p.hold_distance / max(S['v'], 0.1))
            if lines is not None:
                age, ok = gate_l.check(None if S['lstamp'] is None else now - S['lstamp'])
                if ok:
                    if all(lines[r] is not None for r in core.ROLES):
                        lines = core.compensate_lines(lines, S['v'], omega, age, p.latency_max)
                    S['last_lines_t'] = now
                else:
                    lines = None
            cn = cones_now(now)
            if lines is None and S['last_lines_t'] is not None and now - S['last_lines_t'] < hold:
                vo, oo = publish(now, S['raw'], cn)
                continue
            dts = 0.0 if S['last_step'] is None else min(now - S['last_step'], 0.5); S['last_step'] = now
            usable = lines if lines is not None and all(lines[r] is not None for r in core.ROLES) else None
            planner._replay_t = now - t0
            st = planner.step(dts, usable, S['v'], cn, rean)
            S['raw'] = (float(st['v']), float(st['omega']))
            vo, oo = publish(now, S['raw'], cn)
            if st.get('lane_reseed'):
                pass
            out.append(dict(t=now - t0, phase=st['phase'], target=st.get('target_lane'), current=st.get('current_lane'),
                            blocked=sorted(st.get('blocked', [])), F=st.get('F'), F_meas=st.get('F_meas'),
                            rej=st.get('lateral_rejected'), v_plan=st['v'], v_out=vo, tl=tl.state, v_meas=S['v'],
                            hold={int(k): round(float(v), 2) for k, v in planner.s.block_hold.items()}, margin=planner.p.cone_pass_margin))
            continue
        i += 1
        if kind == 'lines':
            st_, roles = d
            S['pending'] = {r: line_obs(roles[r]) for r in core.ROLES}; S['lstamp'] = st_
            if S['reseed']:
                S['rean'] = True; S['reseed'] = False
        elif kind == 'cones':
            S['cstamp'], S['cones'] = d; S['cones_t'] = t
        elif kind == 'can':
            S['v_wheel'] = d
        elif kind == 'imu':
            S['acc'].append(d[0]); S['yaw'] = d[1]; S['imu_t'] = t
        elif kind == 'red':
            S['red'] = (t, d)
        elif kind == 'green':
            S['green'] = (t, d)
        elif kind == 'reseed':
            S['reseed'] = True
        elif kind == 'mode':
            mode, lim = d
            if mode: planner.set_driving_mode(mode)
            if lim: planner.set_speed_limit(lim)
    return out, [(t - t0, j) for t, j in rec]


if __name__ == '__main__':
    out, rec = run(sys.argv[1], sys.argv[2], sys.argv[3])
    json.dump(dict(replay=out, record=[dict(t=t, target=j['target_lane'], blocked=j['blocked'], phase=j['phase'], F=j['F'], rej=j['lateral_rejected'], v=j['v']) for t, j in rec]),
              open(sys.argv[4], 'w'))
    print(len(out), 'steps')
