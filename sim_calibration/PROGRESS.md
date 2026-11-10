# シミュレータへの機体特性の搭載: 進捗

状態: 完了 (下の追加依頼は別. 定期タスクなし)
最終更新: 2026-11-04 (この行は作業の区切りごとに必ず更新する)

## ユーザーの依頼 (原文)
「これらの実験結果から機体特性をシミュレーションに搭載してください。そして、実機と同じ挙動するようにリアルとのギャップを埋めていってください。
検証も忘れずに、間違っているところがないかの確認も全て終わった後に１回行ってください。トークンが切れても、制限が復活したら自動で続けるようにしてください」

## 守ること
- 日本語で答える。コードを変えたらどこをどう変えたか説明する (/Users/miyanswer/aiformula_machine/CLAUDE.md に従う)。
- シミュレータ専用の上書きは作らない。実機の物理・パラメータをそのまま再現する。
- ユーザーに頼まれるまでコミット・プッシュしない。rosbag を変更しない。
- 実験データ: ~/aiformula_ws/rosbag/20261027_* (135955, 144614, 154757 は無視)。
- 判定 (スピン/よれ/OK) は実験者の目視。~/aiformula_ws/scripts/convert_20261027_experiments.py の RUNS にある。

## 方針
- web_simulator/js/vehicle_physics.js を, 指令 → motor_controller の制限 → モーターの速度制御 → タイヤの力 → 車体 (vx, vy, r) の平面モデルにする。
- 機体: 前 2 輪駆動, 後ろはオムニホイール (横に自由)。重心は前軸の後ろ。輪荷重 右 25.0, 左 23.4, 後 23.2 kg。
- 同定は Python (web_simulator/tools/vehicle_id/), 同じ式を JS に移し, web_simulator/test/ のブラウザのページで一致を確認する (node はない)。

## 手順と状況
- [x] 1. rosbag から同定用データを作る
- [x] 2. Python のモデルと同定
- [x] 3. vehicle_physics.js に搭載 (test/vehicle_physics.html で model.py と一致)
- [x] 4. 実機との比較でギャップを詰める (全 35 回, 実験コードの閉ループも含む)
- [x] 5. 既存テスト (pytest, parity.html, research_test.html) とシミュレータの動作確認
- [x] 6. 全部終わった後の最終確認 (間違いがないか全体を見直す) → 完了したら状態を「完了」にして定期タスク resume-sim-vehicle-model を削除

## メモ
- 追加依頼 2 完了: round8 を JS に反映, テスト全部 OK。lane2 は 10 km/h まで逸脱なし, racing は 8 km/h まで。コミットしない。
- 同定 8 回目を採用 (params.json=params_round8.json, MU_Y 固定, 不感帯, ブライアー 0.122)。JS 反映中。外部映像の横位置は精度不足で定性確認のみ。
- 追加依頼 2 (2026-11-05): rosbag詳細.docx, ~/aiformula_ws/mts の外部映像, 駆動輪の横摩擦 35.50 kg を使う. モーター応答も入れる. コミットしない。model.py: MU_Y=35.50/48.4 固定 (前 2 輪を引いたと解釈, 報告で確認), 不感帯 V_ON=.30, V_OFF=.20, 同定 8 回目 (params_round8.json). JS 未反映。外部映像は手持ちで横位置の測定を試す。
- 追加依頼の結果: 同定 7 回目を JS に反映済み。golden, compare, pytest 実施。実コース lane2/racing は 10 km/h まで逸脱なし, 12 km/h は逸脱ありで 3 回中 2 回未完走。σ は 0.25 のまま (0.35 は減速のヨーが大きくなりすぎる)。コミットしない。
- 同定 7 回目を採用 (params.json = params_round7.json)。lane2 のヨー誤差 0.050→0.024, 0.061→0.028, ブライアー 0.144。JS へ反映中 (scrub, scrub_k, a_r)。
- 同定 6 回目は不採用 (scrub 一定の形では lane2 が改善せず)。格子探索: 曲率に応じた scrub (10 N m, 0.2/m) で lane2 のヨー誤差は半減するが旋回が弱くなる。同定 7 回目 (params_round7.json) で scrub_k と後輪の滑り角 a_r を加え, lane2 の重みを上げた。model.py 変更済み, JS 未反映。
- 追加依頼 (2026-11-04): 135955 (5 km/h lane2), 154757 (8 km/h lane2) も使って差を埋める. コミットしない。lane2.py で比較。実機は小さなヨー指令で実ヨーが 0.64〜0.72 倍, モデルは 0.93〜0.99 倍。scrub (一定の回転抵抗) を追加して同定 6 回目 (params_round6.json)。採用の基準: lane2 の残りの区間と 10/12 の走行が良くなり, スピン率が悪くならない。採用したら JS に scrub を移す。
- 最終確認中: 荷重移動の入力が平均値になっていた点を修正 (_ax/_ay)。全テストと実コース走行を再実行する。
- 同定 5 回目は不採用 (holdout の傾きは 0.72→0.74 とほぼ同じ, スピン率のブライアースコアは 0.137→0.170 に悪化)。params.json = round4 のまま。低速のヨーの効きが実機より約 1 割小さいのは限界として報告する。
- 残り: 本体の動作確認, compare.py 再実行, 最終確認。
- 比較用の一時ファイルは削除済み。実コース: 旧物理は 12 km/h まで逸脱なし, 新物理は 8 km/h まで逸脱なし, 10 km/h で逸脱あり, 12 km/h 完走できず (報告する)。
- holdout.py: 同定に使っていない 10/12 の 6 レーン走行でヨー誤差 0.03〜0.05 rad/s。低速のヨーの効きが 1 割ほど小さいので, 同定 5 回目 (params_round5.json) で 134520, 155111 の各 40 s を加えた。MC が悪化せず holdout が良くなれば params.json と vehicle_physics.js の VEHICLE_DYNAMICS を更新して gen_golden.py を実行する。
- 一時ファイル web_simulator/test/_old_physics_tmp.js は比較用. 終わったら必ず削除する。
- 実コース (baseline): 5 km/h は完走, 10 km/h は逸脱 2〜3 回で 1 回停止, 12 km/h 以上は完走できない。旧物理との比較中。プランナーの調整は頼まれていないので変えず報告する。
- テスト済み: pytest 133, research_test 13/13, parity 15364/15364, vehicle_physics.html ALL PASS。残り: 実コース走行の確認 (CarRunner), compare.py の図, simulator 本体の動作確認, 最終確認。
- 採用: params.json = params_round4.json, グリップのばらつき σ=0.25, τ=0.3 s (ブライアースコア 0.137)。旋回の表と 20 km/h の直線スピンは再現, 15 km/h の直線スピン率は実機 4/12 に対し約 0.1 と低い (報告に書く)。
- 同定 3 回目 (params_round3.json): 開ループでスピンなしの回は車輪 0.05 m/s, ヨー 0.05 rad/s ほどの誤差。閉ループの一致 25/35。
- 残り: 直線のスピンは左右対称だと起きないので, グリップのばらつき (montecarlo.py) を入れ, 各条件のスピン率で比べる。同定 4 回目は params_round4.json。
- 実験の中止条件などは closed_loop.py が実験コード本体を使う。
- 同定: python3 web_simulator/tools/vehicle_id/fit.py ~/aiformula_ws/sim_calibration 60 → params.json, fit.log。閉ループ: closed_loop.py。
