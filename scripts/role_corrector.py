#!/usr/bin/env python3
"""
role_corrector.py - 車載が付けた白線の役割 (左境界 / 中央 / 右境界) を, 区間推定 (今どの区間・どのレーンか) と
地図の区間の情報 (その区間にどの線があり, 実線か破線か) に照らして直す

  入力: 車載の役割 {役割: 2 m 先の横位置 [m]} と, 同じコマで TopoLocalizer.update() を済ませた区間推定
  1. 区間推定のレーン (横のずれ) から, 見えている各線が地図のどの線か を決める (TopoLocalizer.roles_of)
  2. 車載の役割と違い, かつ区間推定がレーンを CONF_LANE 以上の確からしさで決めていて, さらに
     「ここ数秒に見えた白線」がそのレーンを支えている (EVIDENCE_MIN 以上) ときだけ直す.
     区間推定のレーンは前から引き継いだもので, 左右とも実線の区間などでは白線からは決まらない. 長く止まったり
     手で押されたりして一度ずれると, 確からしさは高いまま間違い続けるので, 今見えている白線の裏付けを求める
  3. 車載が境界と呼んだ線が, 地図では二重線の外側の 1 本なら, その役割を外す (内側の 1 本が本当の境界)
  直した理由 (区間名, その区間の線の種類, 推定したレーン) も返す.

aiformula_machine の車載コード (LineTracker) は変えない. 後で車載へ移せるよう, 入出力は役割 -> 横位置の辞書だけにしている.
"""

import re

import numpy as np

import topo_localize_eval as E

CONF_LANE = 0.9            # 区間推定のレーン (内側 / 外側) の確からしさがこれ以上のときだけ直す
EVIDENCE_MIN = 1.4         # ここ数秒の白線が, 区間推定のレーンをもう一方より何倍 (対数) 支えていれば直すか (log 4)
EVIDENCE_DECAY = 0.93      # 白線の裏付けの減り方 (1 コマ 0.2 s. 約 3 秒で半分)
MATCH_M = 0.35             # 車載の線と区間推定の線を同じ線とみなす横位置の差 [m]
ROLE_JA = {"left": "左境界", "center": "中央", "right": "右境界", "left2": "左の二重線の外側", "right2": "右の二重線の外側"}
TYPE_JA = {"solid": "実線", "dashed": "破線", "": "なし"}


def seg_code(sg):
    m = re.findall(r"[A-Z]\d+(?:-\d+)?", sg.get("name", ""))
    return m[-1] if m else str(sg["id"])


class RoleCorrector:
    def __init__(self, topo):
        self.seg = {sg["id"]: sg for sg in topo["segments"]}
        self.evidence = 0.0       # 内側レーンを支える白線の裏付け (対数尤度比の減衰つきの和. 正 = 内側)

    def update_evidence(self, loc, est):
        """このコマの白線だけで, 内側 / 外側のどちらがよく説明できるか を足し込む (推定位置の前後 ±2 m で)."""
        lines = [l for l in loc.lines if abs(l["y2"]) < 6]
        ev = 0.0
        if lines:
            idx = [(est + k) % loc.N for k in range(-4, 5)]
            lik = np.ones((len(E.OFFSETS), len(idx)))
            for l in lines:
                lik *= loc.line_like(l["y2"], l["kind"])[:, idx]
            inner = lik[E.OFFSETS > 0].max()
            outer = lik[E.OFFSETS < 0].max()
            ev = float(np.log(inner + 1e-9) - np.log(outer + 1e-9))
        self.evidence = EVIDENCE_DECAY * self.evidence + ev
        return self.evidence

    def correct(self, loc, est, vehicle):
        """vehicle: {役割: y2}. 返り値: (直した {役割: y2}, 直したことの説明のリスト)."""
        lanes_p = loc.po
        inner = float(lanes_p[E.OFFSETS > 0].sum())
        conf = max(inner, 1 - inner)
        lane = "内側" if inner >= 0.5 else "外側"
        support = self.update_evidence(loc, est) * (1 if inner >= 0.5 else -1)
        lines = [dict(y2=y, kind="unknown") for y in vehicle.values()]
        roles = loc.roles_of(lines, est) if lines else []
        mine = dict(zip(vehicle.keys(), roles))
        if conf < CONF_LANE or support < EVIDENCE_MIN:
            return dict(vehicle), []
        sg = self.seg[int(loc.seg_of[est])]
        types = loc.role_type[est]
        why_seg = (f"{seg_code(sg)} (左境界 {TYPE_JA[types[0]]} / 中央 {TYPE_JA[types[1]]} / 右境界 {TYPE_JA[types[2]]}"
                   + (" / 左二重線" if types[3] else "") + (" / 右二重線" if types[4] else "") + f") で{lane}レーン ({conf * 100:.0f}%, 白線の裏付け {support:.1f})")
        out, notes = {}, []
        for role, y in vehicle.items():
            m = mine.get(role)
            if m == role or m is None:
                if role not in out:
                    out[role] = y
                continue
            if m in ("left2", "right2"):
                notes.append(f"{ROLE_JA[role]}と付けた線 (y={y:+.2f}) は {why_seg} の {ROLE_JA[m]} -> 役割を外す")
                continue
            notes.append(f"{ROLE_JA[role]}と付けた線 (y={y:+.2f}) は {why_seg} の {ROLE_JA[m]} -> {ROLE_JA[m]}に直す")
            out[m] = y
        return out, notes
