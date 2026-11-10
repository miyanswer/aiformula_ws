"""再生結果から失敗を数える (開ループなので, 判断が変わった後の車の動きは記録のまま. 判断の瞬間だけを見る)."""
import json, sys
CROSS = 4.0      # [m] これより近いコーンで塞がれたレーンを越えて目標にしたら「横切る判断」
TRUE_TL = {'135955': (148, 196), '154757': (160, 170)}  # 本物の信号の時間帯 (154757 は青で通過)


def evaluate(out, bag):
    out = [o for o in out if o['t'] > 51]
    cross = []
    for o in out:
        cur, tgt = o['current'], o['target']
        if cur is None or tgt is None or cur == tgt:
            continue
        lo, hi = sorted((cur, tgt))
        near = [j for j in range(lo + 1, hi) if str(j) in map(str, o['hold']) and o['hold'][j if j in o['hold'] else str(j)] - o['margin'] < CROSS]
        if near:
            cross.append(o['t'])
    # 横切る判断のかたまり
    ep = []
    for t in cross:
        if ep and t - ep[-1][1] < 1.0: ep[-1][1] = t
        else: ep.append([t, t])
    # 信号: APPROACH に入った回数 (本物の時間帯の外 = 誤り), 本物で STOPPED になったか
    a, z = TRUE_TL[bag]
    entries, prev = [], 'NORMAL'
    for o in out:
        if o['tl'] == 'APPROACH' and prev != 'APPROACH': entries.append(round(o['t'], 1))
        prev = o['tl']
    false_ap = [t for t in entries if not (a <= t <= z)]
    stopped_true = any(o['tl'] == 'STOPPED' and a <= o['t'] <= z for o in out)
    stops_false = []
    for o in out:
        if o['v_out'] < 0.3 * max(o['v_plan'], 0.01) and o['tl'] in ('APPROACH', 'STOPPED') and not (a <= o['t'] <= z):
            stops_false.append(o['t'])
    switches = sum(1 for p, q in zip(out, out[1:]) if p['target'] is not None and q['target'] is not None and p['target'] != q['target'])
    return dict(cross_episodes=[(round(x, 1), round(y, 1)) for x, y in ep], false_approach=false_ap, true_stop=stopped_true,
                false_slow_s=round(sum(1 for o in out if o['tl'] in ('APPROACH','STOPPED') and not (a <= o['t'] <= z)) / 20, 1), switches=switches)


if __name__ == '__main__':
    for f in sys.argv[1:]:
        bag = '154757' if '154757' in f else '135955'
        print(f, evaluate(json.load(open(f))['replay'], bag))
