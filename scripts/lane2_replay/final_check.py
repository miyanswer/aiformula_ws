import sys, json; sys.path.insert(0, '.'); sys.argv = ['x']
import replay, metrics
for label, S in [('直す前 (コミット済み HEAD)', 'headsrc_clean/src/oit_navigation'), ('直した後 (今の作業ツリー)', '/Users/miyanswer/aiformula_machine/src/oit_navigation')]:
    Y = S + '/oit_navigation/six_lane/config/six_lane_params.yaml'
    for b in ['135955', '154757']:
        out, _ = replay.run(f'/Users/miyanswer/aiformula_ws/rosbag/20261027_{b}/6lane', S, Y)
        m = metrics.evaluate(out, b)
        ent = next((o for o in out if o['tl'] == 'APPROACH' and metrics.TRUE_TL[b][0] <= o['t'] <= metrics.TRUE_TL[b][1]), None)
        print(f"{label} {b}: 横切る判断 {m['cross_episodes']} / 誤った接近 {m['false_approach']} / 誤検出による減速・停止 {m['false_slow_s']} s / 本物の赤で停止 {m['true_stop']} (接近開始 {ent and round(ent['t'],1)}) / 目標の変更 {m['switches']} 回")
    for k in list(sys.modules):
        if k.startswith('oit_navigation'): del sys.modules[k]
    sys.path.remove(S)
