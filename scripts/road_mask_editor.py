#!/usr/bin/env python3
"""
road_mask_editor.py - 学習用の「黒路面」の答え (data/road_masks) を確かめて, 手で直す道具 (ブラウザで開く)

画像に答えを重ねて 1 枚ずつ見て, ブラシで塗る / 消しゴムで消す / 多角形で塗る・消す ことができる.
「学習に使う」のオン / オフも切り替えられる. 保存すると data/road_masks/<データセット>/<train|val>/<名前>.png と
usable.txt (学習に使う画像), edited.txt (手で直した画像. make_road_masks.py で作り直しても上書きしない) に書く.

2 人以上で作業するとき (--share): 同じネットワークの別のパソコンから http://<この Mac の IP>:<ポート>/?k=<合言葉> で開ける.
  合言葉は起動のたびに作り直す (--token で自分で決めることもできる. なければ開けない). 同じ画像を 2 人が直したときは, 後から保存した人に知らせて上書きしない.
  誰が直したかは edited_by.json に残る.
退屈しないように (ドパガキ仕様): 保存ごとの点数 (変更があった保存で, 1 枚につき 1 回だけ)・コンボ (45 秒以内に続けて保存)・フィーバー (5 コンボで 2 倍)・レベル・
紙吹雪・効果音・2 人のランキング・相手が保存したときの知らせ. 「落ち着きモード」で音と演出を切れる.
エディタを直して起動し直すと, 開いているページは (保存してから) 自動で読み込み直し, 同じ画像に戻る.

使い方:
  python3 scripts/road_mask_editor.py [8766] [--share] [--token 合言葉] [--tailscale-only]
"""

import argparse
import base64
import fcntl
import json
import secrets
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

WS_DIR = Path(__file__).resolve().parent.parent
DATASETS = ["honda_cropped_dataset", "yolop_cropped_dataset"]
MASKS = WS_DIR / "data/road_masks"
TOKEN = ""                 # --share のときの合言葉 (空なら確かめない = この Mac だけで開くとき)
LOCK = threading.Lock()    # 保存 (答えと一覧) を 1 人ずつにする


def read_list(p):
    return set(p.read_text().split()) if p.exists() else set()


def image_path(ds, split, stem):
    for ext in (".jpg", ".png", ".jpeg"):
        p = WS_DIR / "data" / ds / "images" / split / (stem + ext)
        if p.exists():
            return p
    return None


def items():
    out = []
    for ds in DATASETS:
        for split in ("train", "val"):
            d = MASKS / ds / split
            if not d.exists():
                continue
            usable, edited = read_list(d / "usable.txt"), read_list(d / "edited.txt")
            for m in sorted(d.glob("*.png")):
                out.append(dict(ds=ds, split=split, stem=m.stem, usable=m.stem in usable, edited=m.stem in edited))
    return out


VERSION = str(Path(__file__).stat().st_mtime_ns)   # このファイルが変わったら (起動し直したら), 開いているページを読み込み直させる


WORKERS = MASKS / "workers.json"


def name_key(name):
    """名前の重なりを調べるための形: 全角・半角や大文字・小文字, 前後や続いた空白の違いを無くす."""
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", name).split()).lower()


def load_workers():
    """作業者の名簿 {名前: {score, best, today, day, last, rewarded}}. 無ければ, これまでに直した人の名前から作る."""
    if WORKERS.exists():
        return json.loads(WORKERS.read_text())
    w = {}
    for p in MASKS.glob("*/*/edited_by.json"):
        for who in json.loads(p.read_text()).values():
            if who and who not in w:
                w[who] = dict(score=0, best=0, today=0, day="", last="", rewarded=[])
    return w


def save_workers(w):
    WORKERS.write_text(json.dumps(w, ensure_ascii=False, indent=0))


def find_worker(w, name):
    k = name_key(normalize_name(name))
    return next((n for n in w if name_key(n) == k), None)


def normalize_name(who):
    """作業者名をそろえる: data/road_masks/name_alias.json の [{"prefix": ..., "name": ...}] で始まる名前はその名前に (長すぎる名前を短く)."""
    p = MASKS / "name_alias.json"
    for a in (json.loads(p.read_text()) if p.exists() else []):
        if who.startswith(a["prefix"]):
            return a["name"]
    return who[:40]


VIEWING = {}               # 作業者 (接続元) -> (開いている画像, 時刻)


def note_viewing(who, key):
    """開いている画像を data/road_masks/viewing.json に書く. auto_relabel.py はここにある画像の答えを書き換えない."""
    import time
    with LOCK:
        VIEWING[who] = (key, time.time())
        now = time.time()
        active = sorted({k for k, t in VIEWING.values() if now - t < 60})
        (MASKS / "viewing.json").write_text(json.dumps(dict(updated=now, viewing=active)))


def stats():
    """みんなの進み具合: 全部の枚数, 手で直した枚数, 作業者ごとの枚数, エディタの版."""
    total = edited = 0
    by = {}
    for ds in DATASETS:
        for split in ("train", "val"):
            d = MASKS / ds / split
            if not d.exists():
                continue
            total += len(list(d.glob("*.png")))
            edited += len(read_list(d / "edited.txt"))
            if (d / "edited_by.json").exists():
                for who in json.loads((d / "edited_by.json").read_text()).values():
                    by[who] = by.get(who, 0) + 1
    st = MASKS / "relabel_status.json"            # auto_relabel.py の学習と書き直しの始まり・終わり
    relabel = json.loads(st.read_text()) if st.exists() else {}
    return dict(total=total, edited=edited, by=by, version=VERSION, relabel=relabel)


HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>黒路面ペイント</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Mochiy+Pop+One&display=swap" rel="stylesheet">
<style>
:root { --bg:#f6f6f4; --panel:#fff; --ink:#1d1d1f; --sub:#5f6368; --line:#d9d9d6; --accent:#2f6fde; --warn:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --panel:#202124; --ink:#e8e8e8; --sub:#a0a4a8; --line:#3a3b3e; --accent:#7aa7ff; --warn:#ff7b6b; } }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif; background:var(--bg); color:var(--ink); }
#bg { position:fixed; inset:0; z-index:-1; }
body.anim { background:transparent; }
body.anim header, body.anim #hud, body.anim .help, body.anim #info { background:color-mix(in srgb, var(--bg) 72%, transparent); backdrop-filter:blur(3px); }
#stage-name { position:fixed; left:14px; bottom:12px; z-index:5; font-size:12px; font-weight:700; color:#fff; text-shadow:0 0 6px #000; pointer-events:none; }
header { display:flex; gap:8px; align-items:center; padding:8px 14px; border-bottom:1px solid var(--line); flex-wrap:wrap; font-size:13px; }
header h1 { font-size:15px; margin:0 10px 0 0; }
button, select, input[type=range] { font:inherit; font-size:13px; }
button, select { padding:5px 10px; border-radius:6px; border:1px solid var(--line); background:var(--panel); color:var(--ink); cursor:pointer; }
button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.sep { width:1px; height:22px; background:var(--line); margin:0 4px; }
#info { padding:6px 14px; font-size:13px; color:var(--sub); display:flex; gap:14px; flex-wrap:wrap; align-items:center; }
#info b { color:var(--ink); }
#wrap { padding:0 14px 14px; }
#stage { position:relative; width:100%; touch-action:none; }
#stage canvas { position:absolute; left:0; top:0; width:100%; }
#base { position:relative !important; }
.warn { color:var(--warn); }
.help { font-size:12px; color:var(--sub); padding:0 14px 6px; line-height:1.6; }
/* ---- ドパガキ仕様 ---- */
#hud { display:flex; gap:14px; align-items:center; padding:8px 14px; flex-wrap:wrap;
  background:linear-gradient(90deg,#ff3cac22,#784ba022,#2b86c522); border-bottom:1px solid var(--line); }
#hud .big { font-size:26px; font-weight:900; letter-spacing:.02em; font-variant-numeric:tabular-nums; }
#hud .lab { font-size:11px; color:var(--sub); display:block; }
#combo { color:#ff3cac; transition:transform .15s; display:inline-block; }
#combo.pop { transform:scale(1.6) rotate(-6deg); }
#lvbar { width:160px; height:10px; border-radius:6px; background:var(--line); overflow:hidden; }
#lvbar i { display:block; height:100%; width:0; background:linear-gradient(90deg,#f9d423,#ff4e50); transition:width .4s; }
#prog { flex:1; min-width:180px; }
#progbar { height:14px; border-radius:8px; background:var(--line); overflow:hidden; position:relative; }
#progbar i { display:block; height:100%; width:0; background:linear-gradient(90deg,#00f260,#0575e6); transition:width .6s; }
#progbar span { position:absolute; inset:0; font-size:10px; text-align:center; line-height:14px; color:#fff; font-weight:700; text-shadow:0 0 3px #000; }
#rank { font-size:12px; line-height:1.5; }
#rank b { font-size:13px; }
#fx { position:fixed; inset:0; pointer-events:none; z-index:50; }
#toast { position:fixed; left:50%; top:120px; transform:translateX(-50%); z-index:60; pointer-events:none; text-align:center; }
.shout { font-size:54px; font-weight:900; color:#fff; -webkit-text-stroke:2px #000; text-shadow:0 4px 0 #ff3cac,0 8px 18px #0008;
  animation:shout 1.1s ease-out forwards; white-space:nowrap; }
.shout.small { font-size:28px; }
@keyframes shout { 0% { transform:scale(.2) rotate(-12deg); opacity:0 } 15% { transform:scale(1.25) rotate(4deg); opacity:1 }
  30% { transform:scale(1) rotate(0) } 80% { opacity:1 } 100% { transform:translateY(-60px) scale(.9); opacity:0 } }
body.fever #stage { box-shadow:0 0 0 4px #ff3cac, 0 0 30px 6px #f9d423; animation:fever .6s linear infinite alternate; }
@keyframes fever { from { box-shadow:0 0 0 4px #ff3cac,0 0 30px 6px #f9d423 } to { box-shadow:0 0 0 4px #2b86c5,0 0 40px 10px #00f260 } }
body.shake { animation:shake .35s; }
@keyframes shake { 20% { transform:translate(-6px,2px) } 40% { transform:translate(5px,-3px) } 60% { transform:translate(-4px,3px) } 80% { transform:translate(3px,-2px) } }
#update { position:fixed; right:14px; bottom:14px; z-index:70; padding:10px 14px; border-radius:8px; background:#2f6fde; color:#fff; font-size:13px; display:none; }
@media (prefers-reduced-motion: reduce) { .shout, body.fever #stage, body.shake { animation:none !important; } }

/* ===== インク塗りゲーム風 ===== */
:root { --c-brush:#b6ff00; --c-erase:#00e5ff; --c-bucket:#ff7a00; --c-poly:#ff2fa8; --c-polyerase:#8b5cff; --tool:var(--c-brush); }
body[data-tool="erase"] { --tool:var(--c-erase); } body[data-tool="bucket"] { --tool:var(--c-bucket); }
body[data-tool="poly"] { --tool:var(--c-poly); } body[data-tool="polyerase"] { --tool:var(--c-polyerase); }
body { font-family:"Mochiy Pop One","Hiragino Maru Gothic ProN","Hiragino Sans",sans-serif; }
header.bar { display:block; padding:10px 14px; background:#16121f; color:#fff; border-bottom:4px solid var(--tool);
  background-image:radial-gradient(circle at 8% 120%, #ff2fa833 0 60px, transparent 61px), radial-gradient(circle at 92% -20%, #b6ff0033 0 70px, transparent 71px); }
header.bar .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
header.bar .row + .row { margin-top:10px; }
.grow { flex:1; }
.logo { margin:0 8px 0 0; font-size:22px; line-height:1; transform:rotate(-3deg); letter-spacing:.02em; }
.logo span { display:inline-block; padding:4px 8px; background:#000; color:#fff; border-radius:8px; }
.logo .pop { background:var(--c-poly); color:#000; margin-left:-4px; transform:rotate(4deg) translateY(2px); box-shadow:3px 3px 0 #000; }
kbd { font-family:inherit; font-size:10px; padding:1px 5px; border-radius:5px; background:#0008; color:#fff; border:1px solid #fff5; margin-left:4px; }
.pill-in { background:#2a2338 !important; color:#fff !important; border:2px solid #4b3f63 !important; border-radius:999px !important; padding:5px 12px !important; }
.chunky { font-family:inherit; font-size:14px; color:#000; background:#fff; border:3px solid #000; border-radius:14px; padding:6px 14px;
  box-shadow:0 4px 0 #000; cursor:pointer; transition:transform .08s, box-shadow .08s; }
.chunky:active { transform:translateY(3px); box-shadow:0 1px 0 #000; }
.chunky.save { font-size:20px; padding:8px 26px; background:var(--c-brush); transform:rotate(-2deg); }
.chunky.save:hover { background:#d5ff5a; }
.nav { display:inline-flex; gap:6px; align-items:center; }
.tools { display:flex; gap:10px; }
.tool { position:relative; width:74px; height:74px; border-radius:22px; border:3px solid #000; background:#2a2338; color:#fff; cursor:pointer;
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:2px; font-family:inherit; font-size:10px;
  box-shadow:0 5px 0 #000; transition:transform .12s, background .12s; }
.tool svg { width:34px; height:34px; }
.tool .tname { white-space:nowrap; font-size:9.5px; }
.tool kbd { position:absolute; top:-8px; right:-8px; background:#000; border:2px solid #fff; font-size:11px; }
.tool:hover { transform:translateY(-2px); }
.tool-brush { color:var(--c-brush); } .tool-erase { color:var(--c-erase); } .tool-bucket { color:var(--c-bucket); }
.tool-poly { color:var(--c-poly); } .tool-polyerase { color:var(--c-polyerase); }
.tool.on { color:#000; transform:scale(1.15) rotate(-4deg); z-index:2; animation:boing .35s; }
.tool-brush.on { background:var(--c-brush); } .tool-erase.on { background:var(--c-erase); } .tool-bucket.on { background:var(--c-bucket); }
.tool-poly.on { background:var(--c-poly); } .tool-polyerase.on { background:var(--c-polyerase); color:#fff; }
.tool.on::before { content:""; position:absolute; inset:-16px; z-index:-1; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, currentColor 0 0, transparent 0),
    radial-gradient(circle at 20% 25%, var(--tool) 0 9px, transparent 10px), radial-gradient(circle at 85% 20%, var(--tool) 0 7px, transparent 8px),
    radial-gradient(circle at 80% 85%, var(--tool) 0 10px, transparent 11px), radial-gradient(circle at 15% 80%, var(--tool) 0 6px, transparent 7px),
    radial-gradient(circle, var(--tool) 0 46%, transparent 47%); opacity:.55; filter:blur(.3px); }
@keyframes boing { 0% { transform:scale(.8) rotate(6deg) } 60% { transform:scale(1.25) rotate(-6deg) } 100% { transform:scale(1.15) rotate(-4deg) } }
.opts { display:flex; gap:10px; flex-wrap:wrap; }
.opt { display:flex; align-items:center; gap:6px; background:#2a2338; padding:6px 10px; border-radius:14px; border:2px solid #4b3f63; font-size:12px; }
.opt b { min-width:26px; text-align:right; color:var(--tool); }
.opt input[type=range] { accent-color:var(--tool); width:120px; }
.toggles { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-left:auto; }
.tog { display:flex; align-items:center; gap:4px; padding:6px 12px; border-radius:999px; background:#2a2338; border:2px solid #4b3f63; cursor:pointer; font-size:12px; }
.tog:has(input:checked) { background:var(--c-erase); color:#000; border-color:#000; }
.tog input { display:none; }
#toolpill { position:absolute; left:12px; top:12px; z-index:3; display:flex; align-items:center; gap:8px; padding:8px 16px 8px 10px;
  background:var(--tool); color:#000; border:3px solid #000; border-radius:999px; box-shadow:0 4px 0 #000; font-size:16px; pointer-events:none;
  transform:rotate(-2deg); transition:background .15s; }
#toolpill svg { width:26px; height:26px; }
#toolpill.bump { animation:boing .35s; }
#stage { border:4px solid var(--tool); border-radius:10px; overflow:hidden; transition:border-color .15s; }
#hud { background:linear-gradient(90deg,#ff2fa833,#8b5cff33,#00e5ff33) !important; }
body.anim header.bar { background-color:#16121fe8; }

/* ===== 画像の上には何も載せない ===== */
#statusrow { display:flex; align-items:center; gap:12px; min-height:56px; margin:8px 0; flex-wrap:wrap; }
#toolpill { position:static !important; }
#stage-name { position:static; font-size:13px; color:#fff; text-shadow:0 0 6px #000; }
#toast { position:relative; left:auto; top:auto; transform:none; flex:1; min-width:200px; height:48px; text-align:left; }
#toast .shout { position:absolute; left:0; top:0; font-size:30px; }
#toast .shout.small { font-size:22px; }
#update { position:static; }
#relabel { display:none; padding:4px 10px; border-radius:999px; font-size:12px; white-space:nowrap; cursor:pointer; }
#relabel.running { background:#ff7a00; color:#000; }
#relabel.done { background:#b6ff00; color:#000; }
#relabel.failed { background:#ff3b3b; color:#fff; }
#work { display:flex; gap:14px; align-items:flex-start; }
#stage { flex:1; min-width:0; width:auto; }
#loupedock { display:none; flex:none; width:226px; position:sticky; top:8px; text-align:center; color:#fff; font-size:12px; }
#loupedock canvas { display:block; border-radius:50%; box-shadow:0 0 0 3px #fff, 0 0 0 5px #000, 0 6px 20px #0008; background:#000; margin:3px; }
body.loupe-on #loupedock { display:block; }

/* ===== 1 画面にまとめる (スクロールしない) ===== */
html, body { height:100%; }
body { display:flex; flex-direction:column; overflow:hidden; }
header.bar.top { flex:none; padding:8px 12px; }
header.bar.top .logo { font-size:18px; }
#app { flex:1; min-height:0; display:grid; grid-template-columns:auto minmax(0,1fr) 300px; gap:10px; padding:10px; }
body.side-hidden #app { grid-template-columns:auto minmax(0,1fr) 0; }
body.side-hidden #side { display:none; }
#toolbar { overflow-y:auto; }
#toolbar .tools { flex-direction:column; gap:14px; padding:10px 8px; }
#toolbar .tool { width:66px; height:66px; }
#center { min-width:0; min-height:0; display:flex; flex-direction:column; }
#info { padding:2px 4px; font-size:12px; }
#statusrow { min-height:44px; margin:4px 0; }
#work { flex:1; min-height:0; display:block; }
#stage { flex:none; margin:0 auto; }
#side { min-height:0; overflow-y:auto; display:flex; flex-direction:column; gap:10px; }
#side .card { background:#16121fe8; color:#fff; border:2px solid #4b3f63; border-radius:16px; padding:10px; }
#side h3 { margin:0 0 8px; font-size:13px; color:var(--tool); }
#side .opts { flex-direction:column; }
#side .opt { justify-content:space-between; }
#side .opt input[type=range] { flex:1; width:auto; min-width:60px; }
#side .toggles { margin-left:0; }
#loupecard { display:none; }
body.loupe-on #loupecard { display:block; }
#loupedock { display:block !important; position:static; width:auto; margin:0 auto; }
#loupedock canvas { margin:3px auto; }
#hud { display:grid !important; grid-template-columns:1fr 1fr; gap:8px; padding:0 !important; background:none !important; border:none !important; }
#hud .big { font-size:22px; }
#hud #prog, #hud #rank, #hud label { grid-column:1 / -1; }
#lvbar { width:100%; }
#side details summary { cursor:pointer; font-size:13px; }
#side .help { padding:8px 0 0; color:#ddd; background:none !important; }
</style></head><body>
<header class="bar top"><div class="row">
    <h1 class="logo"><span>黒路面</span><span class="pop">ペイント!</span></h1>
    <select id="whosel" class="pill-in" title="自分の名前を選ぶと, 前の続きから始まる"><option value="">👤 名前を選ぶ</option></select><input id="who" type="hidden">
    <select id="dsf" class="pill-in"><option value="">両方のデータセット</option><option value="honda_cropped_dataset">honda</option><option value="yolop_cropped_dataset">yolop</option></select>
    <select id="filter" class="pill-in"><option value="all">すべて</option><option value="usable">学習に使う</option><option value="unused">学習に使わない</option><option value="edited">手で直した</option><option value="notedited">まだ直していない</option></select>
    <span class="nav"><button id="prev" class="chunky">◀ 前 <kbd>A</kbd></button><input id="jump" class="pill-in" type="number" min="1" style="width:70px" title="番号へ移動"><button id="next" class="chunky">次 ▶ <kbd>D</kbd></button></span>
    <span class="grow"></span>
    <button id="sidebtn" class="chunky" title="右のパネルを出す / しまう (G)">☰ パネル <kbd>G</kbd></button>
    <button id="save" class="chunky save">保存! <kbd>S</kbd></button>
  </div></header>
<canvas id="bg"></canvas><canvas id="fx"></canvas>
<div id="app">
  <nav id="toolbar"><div class="tools"><button id="t-brush" class="tool tool-brush on" title="ブラシ (Q)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3l7 7-8.5 8.5c-1 1-2.5 1-3.5 0l-3.5-3.5c-1-1-1-2.5 0-3.5z" fill="currentColor"/><path d="M5.5 15.5c-2 .5-3 2-3 4.5 2.5 0 4-1 4.5-3z" fill="currentColor"/></svg><span class="tname">ブラシ</span><kbd>Q</kbd></button>
      <button id="t-erase" class="tool tool-erase" title="消しゴム (W)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 3l6 6-10 10H6l-3-3c-1-1-1-2.5 0-3.5z" fill="currentColor"/><path d="M3 21h18" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg><span class="tname">消しゴム</span><kbd>W</kbd></button>
      <button id="t-bucket" class="tool tool-bucket" title="バケツ (E)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9l7-6 8 8-7 7c-1 1-2.5 1-3.5 0L4 13.5C3 12.5 3 10 4 9z" fill="currentColor"/><path d="M19.5 14c1.5 2 2.5 3.5 2.5 4.5a2.5 2.5 0 01-5 0c0-1 1-2.5 2.5-4.5z" fill="currentColor"/></svg><span class="tname">バケツ</span><kbd>E</kbd></button>
      <button id="t-poly" class="tool tool-poly" title="多角形で塗る (R)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18L7 5l11 2 3 10-9 4z" fill="currentColor" fill-opacity=".45" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"/><circle cx="4" cy="18" r="2" fill="currentColor"/><circle cx="7" cy="5" r="2" fill="currentColor"/><circle cx="18" cy="7" r="2" fill="currentColor"/><circle cx="21" cy="17" r="2" fill="currentColor"/><circle cx="12" cy="21" r="2" fill="currentColor"/></svg><span class="tname">多角形で塗る</span><kbd>R</kbd></button>
      <button id="t-polyerase" class="tool tool-polyerase" title="多角形で消す (T)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18L7 5l11 2 3 10-9 4z" fill="none" stroke="currentColor" stroke-width="2.4" stroke-dasharray="3 2.5" stroke-linejoin="round"/><path d="M9 10l6 6M15 10l-6 6" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/></svg><span class="tname">多角形で消す</span><kbd>T</kbd></button></div></nav>
  <main id="center">
    <div id="info"></div>
    <div id="statusrow"><div id="toolpill"></div><div id="stage-name"></div><div id="toast"></div><div id="update"></div><div id="relabel" title="クリックで消す"></div></div>
    <div id="work"><div id="stage"><canvas id="base"></canvas><canvas id="over"></canvas><canvas id="ui"></canvas></div></div>
  </main>
  <aside id="side">
    <section class="card"><h3>道具の設定</h3><div class="opts">
      <label class="opt"><span id="sizelab">ブラシの大きさ</span><input id="size" type="range" min="4" max="200" value="40"><b id="sizeval">40</b><kbd>1</kbd><kbd>2</kbd></label>
      <label class="opt" title="バケツ: クリックした所と色が近い, つながった所を塗る (Shift+クリックで消す)"><span>色の近さ</span><input id="tol" type="range" min="2" max="80" value="18"><b id="tolval">18</b></label>
      <label class="opt"><span>重ね</span><input id="alpha" type="range" min="0" max="100" value="45"><kbd>C</kbd></label>
    </div></section>
    <section class="card"><div class="toggles">
      <label class="tog" title="ブラシ・消しゴムの場所を拡大して見る"><input id="loupe" type="checkbox"><span>🔍 ルーペ</span><kbd>X</kbd></label>
      <select id="loupez" class="pill-in"><option value="2">2 倍</option><option value="3" selected>3 倍</option><option value="4">4 倍</option><option value="6">6 倍</option></select>
      <label class="tog"><input id="usable" type="checkbox"><span>✅ 学習に使う</span><kbd>F</kbd></label>
      <button id="undo" class="chunky">↩ 元に戻す <kbd>Z</kbd></button>
    </div></section>
    <section class="card" id="loupecard"><div id="loupedock"><canvas id="loupec" width="220" height="220"></canvas><div class="lab">ルーペ <kbd>X</kbd></div></div></section>
    <section class="card"><h3>スコア</h3><div id="hud">
  <div><span class="lab">スコア</span><span class="big" id="score">0</span></div>
  <div><span class="lab">レベル <b id="lv">1</b></span><div id="lvbar"><i></i></div></div>
  <div><span class="lab">コンボ</span><span class="big" id="combo">0</span></div>
  <div><span class="lab">今日の保存</span><span class="big" id="today">0</span></div>
  <div id="prog"><span class="lab">みんなで直した <b id="progtxt"></b></span><div id="progbar"><i></i><span id="progpct"></span></div></div>
  <div id="rank"></div>
  <label style="font-size:12px"><input id="calm" type="checkbox"> 落ち着きモード (音と演出を切る)</label>
</div></section>
    <details class="card"><summary>⌨️ ショートカットと使い方</summary><div class="help"><b>左手のショートカット:</b> Q ブラシ / W 消しゴム / E バケツ / R 多角形で塗る / T 多角形で消す ・ A 前 / S 保存 / D 次 / F 学習に使う ・
Z 元に戻す / X ルーペ / C 重ねを隠す・戻す / V ブラシ⇔消しゴム / G 右のパネル ・ 1 小さく / 2 大きく ・ Space 多角形を確定 / Esc やめる ・ Shift+クリック (バケツ) で消す<br>
ブラシ・消しゴム: ドラッグ (画像の上でマウスホイールを回すと大きさが変わる) / バケツ: クリックした所と色が近い, つながった所を塗る / 多角形: クリックで点を置き, ダブルクリックか Space で確定。
←→ か A / D で前後の画像 (未保存の変更があれば先に保存するか聞きます)。紫が「走れる路面」。緑の歩道・島・白線の外・車体・塀は路面に含めないでください (白線の上は路面)。</div></details>
  </aside>
</div>
<script>
const K = new URLSearchParams(location.search).get('k') || '';
const q = (u) => u + (u.includes('?') ? '&' : '?') + 'k=' + encodeURIComponent(K);
let ITEMS = [], view = [], cur = -1, tool = 'brush', dirty = false, poly = [], hist = [], mtime = 0;
// 作業者: 名簿から選ぶ (新しい人だけ登録. 同じ名前は登録できない). 選ぶと前の続き (最後の画像・点数) から始まる
const NEW_NAME = '__new__';
let ME = null;      // 名簿にある自分の情報 {name, score, best, today, day, last, rewarded}
async function loadWorkers(select) { const r = await fetch(q('api/workers')); if (!r.ok) return [];
  const ws = await r.json(), sel = document.getElementById('whosel');
  sel.innerHTML = '<option value="">👤 名前を選ぶ</option>' + ws.map(w => `<option value="${w.name.replace(/"/g, '&quot;')}">${w.name} (${w.count} 枚)</option>`).join('') +
    `<option value="${NEW_NAME}">＋ 新しく名前を登録</option>`;
  if (select) sel.value = select; return ws; }
const base = document.getElementById('base'), over = document.getElementById('over'), ui = document.getElementById('ui');
const bctx = base.getContext('2d'), octx = over.getContext('2d'), uctx = ui.getContext('2d');
let mask = null, mctx = null, W = 0, H = 0;   // mask: 元の大きさの 1 チャンネル相当 (アルファで持つ)
const $ = id => document.getElementById(id);
// ブラシと消しゴムは大きさを別々に持つ (ブラウザに覚える)
const SIZES = (() => { try { return Object.assign({brush: 40, erase: 40}, JSON.parse(localStorage.getItem('sizes') || '{}')); } catch (e) { return {brush: 40, erase: 40}; } })();
function showSize() { const k = tool === 'erase' ? 'erase' : 'brush'; $('size').value = SIZES[k]; $('sizeval').textContent = SIZES[k];
  $('sizelab').textContent = k === 'erase' ? '消しゴムの大きさ' : 'ブラシの大きさ'; }
$('size').oninput = () => { const k = tool === 'erase' ? 'erase' : 'brush'; SIZES[k] = +$('size').value; $('sizeval').textContent = SIZES[k];
  try { localStorage.setItem('sizes', JSON.stringify(SIZES)); } catch (e) {} };
const TOOL_NAME = {brush: 'ブラシ', erase: '消しゴム', bucket: 'バケツ', poly: '多角形で塗る', polyerase: '多角形で消す'};
function setTool(t) { tool = t; for (const k of ['brush','erase','bucket','poly','polyerase']) $('t-' + k).classList.toggle('on', k === t);
  document.body.dataset.tool = t;
  const pill = $('toolpill'); pill.innerHTML = $('t-' + t).querySelector('svg').outerHTML + TOOL_NAME[t];
  pill.classList.remove('bump'); void pill.offsetWidth; pill.classList.add('bump');
  poly = []; showSize(); drawUI(); }
showSize();
['brush','erase','bucket','poly','polyerase'].forEach(k => $('t-' + k).onclick = () => setTool(k));
setTool('brush');
try { $('tol').value = localStorage.getItem('tol') || 18; } catch (e) {} $('tolval').textContent = $('tol').value;
$('tol').oninput = () => { $('tolval').textContent = $('tol').value; try { localStorage.setItem('tol', $('tol').value); } catch (e) {} };
// バケツ: クリックした所 (5x5 の平均の色) と色の差が「色の近さ」以内で, 上下左右につながった画素を塗る (erase なら消す)
function bucketFill(x0, y0, erase) {
  x0 = Math.round(x0); y0 = Math.round(y0); if (x0 < 0 || y0 < 0 || x0 >= W || y0 >= H) return;
  const img = bctx.getImageData(0, 0, W, H).data, tol = +$('tol').value;
  let sr = 0, sg = 0, sb = 0, n = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const x = x0 + dx, y = y0 + dy;
    if (x < 0 || y < 0 || x >= W || y >= H) continue; const i = (y * W + x) * 4; sr += img[i]; sg += img[i + 1]; sb += img[i + 2]; n++; }
  sr /= n; sg /= n; sb /= n;
  const seen = new Uint8Array(W * H), stack = [y0 * W + x0]; seen[y0 * W + x0] = 1;
  const m = mctx.getImageData(0, 0, W, H), md = m.data; let count = 0;
  while (stack.length) { const p = stack.pop(), i = p * 4;
    if (Math.max(Math.abs(img[i] - sr), Math.abs(img[i + 1] - sg), Math.abs(img[i + 2] - sb)) > tol) continue;
    md[i] = 255; md[i + 1] = 0; md[i + 2] = 255; md[i + 3] = erase ? 0 : 255; count++;
    const x = p % W, y = (p - x) / W;
    if (x > 0 && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); } if (x < W - 1 && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
    if (y > 0 && !seen[p - W]) { seen[p - W] = 1; stack.push(p - W); } if (y < H - 1 && !seen[p + W]) { seen[p + W] = 1; stack.push(p + W); } }
  mctx.putImageData(m, 0, 0); dirty = true; drawOver(); info();
  return count; }
async function loadList() { const r = await fetch(q('api/list')); if (!r.ok) { document.getElementById('info').textContent = '開けません (合言葉付きの URL で開いてください)'; return; }
  ITEMS = await r.json(); applyFilter(0); }
function applyFilter(keepStem) {
  const f = $('filter').value, dsf = $('dsf').value;
  view = ITEMS.map((it, i) => i).filter(i => { const it = ITEMS[i];
    if (dsf && it.ds !== dsf) return false;
    return f === 'all' || (f === 'usable' && it.usable) || (f === 'unused' && !it.usable) || (f === 'edited' && it.edited) || (f === 'notedited' && !it.edited); });
  if (!view.length) { $('info').textContent = 'この絞り込みに当てはまる画像はありません'; return; }
  let start = 0;
  try { const rs = sessionStorage.getItem('resume'); if (rs) { sessionStorage.removeItem('resume');
    const k = view.findIndex(i => `${ITEMS[i].ds}/${ITEMS[i].split}/${ITEMS[i].stem}` === rs); if (k >= 0) start = k; } } catch (e) {}
  go(start);
}
$('filter').onchange = () => { if (confirmLeave()) applyFilter(); };
$('dsf').onchange = () => { if (confirmLeave()) applyFilter(); };
function confirmLeave() { if (!dirty) return true; if (confirm('未保存の変更があります。保存しますか? (キャンセルで保存せずに移動)')) { save(); } dirty = false; return true; }
async function go(k) {
  if (!view.length) return; k = (k + view.length) % view.length; cur = k; const it = ITEMS[view[k]];
  const meta = await (await fetch(q(`api/meta/${it.ds}/${it.split}/${it.stem}`))).json();
  Object.assign(it, meta); mtime = meta.mtime;
  const [img, m] = await Promise.all([loadImg(q(`img/${it.ds}/${it.split}/${it.stem}`)), loadImg(q(`mask/${it.ds}/${it.split}/${it.stem}?t=${Date.now()}`))]);
  W = img.naturalWidth; H = img.naturalHeight;
  for (const c of [base, over, ui]) { c.width = W; c.height = H; }
  fitStage();
  bctx.drawImage(img, 0, 0);
  mask = document.createElement('canvas'); mask.width = W; mask.height = H; mctx = mask.getContext('2d');
  // 答え (白 = 路面) をアルファに直す
  const tmp = document.createElement('canvas'); tmp.width = W; tmp.height = H; const tctx = tmp.getContext('2d'); tctx.drawImage(m, 0, 0);
  const src = tctx.getImageData(0, 0, W, H), dst = mctx.createImageData(W, H);
  for (let i = 0; i < src.data.length; i += 4) { const v = src.data[i] > 127 ? 255 : 0; dst.data[i] = 255; dst.data[i+1] = 0; dst.data[i+2] = 255; dst.data[i+3] = v; }
  mctx.putImageData(dst, 0, 0);
  hist = []; poly = []; dirty = false; $('usable').checked = it.usable; drawOver(); drawUI(); info();
  if (typeof refreshStats === 'function') refreshStats();
}
function loadImg(src) { return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; }); }
function cover() { const d = mctx.getImageData(0, 0, W, H).data; let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 127; return n / (W * H); }
function info() { const it = ITEMS[view[cur]]; const c = cover();
  $('info').innerHTML = `<b>${cur + 1} / ${view.length}</b> <span>${it.ds} / ${it.split} / ${it.stem}</span> <span>路面 ${(c * 100).toFixed(0)}%</span>` +
    `<span>${it.usable ? '学習に使う' : '<span class="warn">学習に使わない</span>'}</span><span>${it.edited ? '手で直した' + (it.by ? ' (' + it.by + ')' : '') : '自動の答え'}</span>` + (dirty ? '<span class="warn">未保存</span>' : ''); }
function drawOver() { octx.clearRect(0, 0, W, H); octx.globalAlpha = $('alpha').value / 100; octx.drawImage(mask, 0, 0); octx.globalAlpha = 1; }
$('alpha').oninput = drawOver;
function drawUI(px, py) { uctx.clearRect(0, 0, W, H);
  if (poly.length) { uctx.strokeStyle = tool === 'poly' ? '#ff0' : '#0ff'; uctx.lineWidth = 3; uctx.beginPath(); poly.forEach(([x, y], i) => i ? uctx.lineTo(x, y) : uctx.moveTo(x, y));
    if (px !== undefined) uctx.lineTo(px, py); uctx.stroke(); poly.forEach(([x, y]) => { uctx.fillStyle = '#fff'; uctx.fillRect(x - 4, y - 4, 8, 8); }); }
  else if (px !== undefined && (tool === 'brush' || tool === 'erase')) { uctx.strokeStyle = tool === 'brush' ? '#f0f' : '#fff'; uctx.lineWidth = 2; uctx.beginPath(); uctx.arc(px, py, $('size').value / 2, 0, Math.PI * 2); uctx.stroke(); } }
function pos(e) { const r = ui.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]; }
function pushHist() { hist.push(mctx.getImageData(0, 0, W, H)); if (hist.length > 30) hist.shift(); }
function stroke(x0, y0, x1, y1) { mctx.globalCompositeOperation = tool === 'brush' ? 'source-over' : 'destination-out';
  mctx.strokeStyle = '#f0f'; mctx.lineWidth = +$('size').value; mctx.lineCap = 'round'; mctx.beginPath(); mctx.moveTo(x0, y0); mctx.lineTo(x1, y1); mctx.stroke();
  mctx.globalCompositeOperation = 'source-over'; }
let drag = null;
ui.addEventListener('pointerdown', e => { const [x, y] = pos(e);
  if (tool === 'brush' || tool === 'erase') { pushHist(); drag = [x, y]; stroke(x, y, x, y); drawOver(); dirty = true; ui.setPointerCapture(e.pointerId); }
  else if (tool === 'bucket') { pushHist(); bucketFill(x, y, e.shiftKey); }
  else { poly.push([x, y]); drawUI(x, y); } });
ui.addEventListener('pointermove', e => { const [x, y] = pos(e);
  if (drag) { stroke(drag[0], drag[1], x, y); drag = [x, y]; drawOver(); }
  drawUI(x, y); });
ui.addEventListener('pointerup', () => { if (drag) { drag = null; info(); } });
// ルーペ: カーソルのまわりを拡大して, 画像・路面の重ね・ブラシの輪を, 画像の右の列に描く (画像の上には何も載せない)
const loupe = document.getElementById('loupec'), lctx = loupe.getContext('2d');
try { $('loupe').checked = localStorage.getItem('loupe') === '1'; $('loupez').value = localStorage.getItem('loupez') || '3'; } catch (e) {}
function dockLoupe() { document.body.classList.toggle('loupe-on', $('loupe').checked); dispatchEvent(new Event('resize')); }
$('loupe').onchange = () => { try { localStorage.setItem('loupe', $('loupe').checked ? '1' : '0'); } catch (e) {} dockLoupe(); };
dockLoupe();
$('loupez').onchange = () => { try { localStorage.setItem('loupez', $('loupez').value); } catch (e) {} };
function drawLoupe(e) {
  if (!$('loupe').checked || !W) return;
  const [x, y] = pos(e), z = +$('loupez').value, L = loupe.width, r = ui.getBoundingClientRect();
  const scr = W / r.width;                     // 画像の 1 画素 = 画面の 1/scr 画素
  const half = L / 2 * scr / z;                // 拡大して L 画素に入る, 画像の範囲の半分
  lctx.save(); lctx.clearRect(0, 0, L, L); lctx.beginPath(); lctx.arc(L / 2, L / 2, L / 2, 0, 6.28); lctx.clip();
  lctx.fillStyle = '#000'; lctx.fillRect(0, 0, L, L);
  lctx.imageSmoothingEnabled = false;
  lctx.drawImage(base, x - half, y - half, half * 2, half * 2, 0, 0, L, L);
  lctx.globalAlpha = $('alpha').value / 100; lctx.drawImage(mask, x - half, y - half, half * 2, half * 2, 0, 0, L, L); lctx.globalAlpha = 1;
  const k = tool === 'erase' ? 'erase' : 'brush';
  if (tool === 'brush' || tool === 'erase') { const rr = SIZES[k] / 2 * L / (half * 2);   // 紫の上でも見えるよう黒で縁取る
    lctx.lineWidth = 4; lctx.strokeStyle = '#000'; lctx.beginPath(); lctx.arc(L / 2, L / 2, rr, 0, 6.28); lctx.stroke();
    lctx.lineWidth = 2; lctx.strokeStyle = tool === 'brush' ? '#ffd400' : '#fff'; lctx.beginPath(); lctx.arc(L / 2, L / 2, rr, 0, 6.28); lctx.stroke(); }
  lctx.strokeStyle = '#ffff00'; lctx.lineWidth = 1; lctx.beginPath(); lctx.moveTo(L / 2 - 8, L / 2); lctx.lineTo(L / 2 + 8, L / 2); lctx.moveTo(L / 2, L / 2 - 8); lctx.lineTo(L / 2, L / 2 + 8); lctx.stroke();
  lctx.restore(); }
ui.addEventListener('pointermove', drawLoupe);
ui.addEventListener('pointerdown', drawLoupe);
ui.addEventListener('pointerleave', () => { lctx.clearRect(0, 0, loupe.width, loupe.height); });
ui.addEventListener('wheel', drawLoupe);
// 画像の上でマウスホイール: 今の道具 (ブラシ / 消しゴム) の大きさを変える (画面はスクロールしない)
ui.addEventListener('wheel', e => { e.preventDefault(); const k = tool === 'erase' ? 'erase' : 'brush';
  const step = Math.max(1, Math.round(SIZES[k] * 0.1));
  SIZES[k] = Math.max(+$('size').min, Math.min(+$('size').max, SIZES[k] + (e.deltaY < 0 ? step : -step)));
  try { localStorage.setItem('sizes', JSON.stringify(SIZES)); } catch (er) {}
  showSize(); const [x, y] = pos(e); drawUI(x, y); }, {passive: false});
ui.addEventListener('dblclick', e => { e.preventDefault(); finishPoly(); });
function finishPoly() { if (poly.length < 3) { poly = []; drawUI(); return; } pushHist();
  mctx.globalCompositeOperation = tool === 'poly' ? 'source-over' : 'destination-out'; mctx.fillStyle = '#f0f'; mctx.beginPath();
  poly.forEach(([x, y], i) => i ? mctx.lineTo(x, y) : mctx.moveTo(x, y)); mctx.closePath(); mctx.fill(); mctx.globalCompositeOperation = 'source-over';
  poly = []; dirty = true; drawOver(); drawUI(); info(); }
function undo() { if (!hist.length) return; mctx.putImageData(hist.pop(), 0, 0); dirty = true; drawOver(); info(); }
$('undo').onclick = undo;
$('usable').onchange = () => { dirty = true; info(); };
// 点数は「変更があった保存」で, 1 枚につき 1 回だけ (保存を連打しても増えない)
const REWARDED = new Set();      // 点数をもらった画像 (名簿にその人の分を持つ)
async function save() { if (!ME) { alert('先に左上で自分の名前を選んでください (新しい人は「新しく名前を登録」)'); document.getElementById('whosel').focus(); return; }
  const it = ITEMS[view[cur]]; const changed = dirty, key = `${it.ds}/${it.split}/${it.stem}`;
  // アルファ -> 白黒の png
  const out = document.createElement('canvas'); out.width = W; out.height = H; const c = out.getContext('2d');
  const s = mctx.getImageData(0, 0, W, H), d = c.createImageData(W, H);
  for (let i = 0; i < s.data.length; i += 4) { const v = s.data[i + 3] > 127 ? 255 : 0; d.data[i] = d.data[i+1] = d.data[i+2] = v; d.data[i+3] = 255; }
  c.putImageData(d, 0, 0);
  const r = await fetch(q('api/save'), {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ds: it.ds, split: it.split, stem: it.stem, usable: $('usable').checked, png: out.toDataURL('image/png'),
                          mtime: mtime, who: $('who').value || ''})});
  if (r.ok) { const j = await r.json(); mtime = j.mtime; it.usable = $('usable').checked; it.edited = true; it.by = $('who').value || ''; dirty = false; info();
    if (changed && !REWARDED.has(key)) { REWARDED.add(key); pendingRewarded.push(key); reward(); }
    else { shout(changed ? '保存! (この画像の点数はもらい済み)' : '保存! (変更なし)', true); strokes = 0; }
    if ($('update').style.display === 'block') autoUpdate(); }
  else if (r.status === 409) { const j = await r.json();
    if (confirm(`この画像は ${j.by || '別の人'} が先に保存しています。先の人の答えを読み込み直しますか? (キャンセルで今の画面のまま)`)) { dirty = false; go(cur); } }
  else alert('保存できませんでした'); }
$('save').onclick = save;
$('prev').onclick = () => { if (confirmLeave()) go(cur - 1); };
$('next').onclick = () => { if (confirmLeave()) go(cur + 1); };
$('jump').onchange = () => { if (confirmLeave()) go(+$('jump').value - 1); };
window.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT' && ['text', 'number'].includes(e.target.type)) return;   // 名前・番号の入力中だけ効かない
  if (e.target.tagName === 'SELECT') e.target.blur();
  const k = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) { if (k === 'z') { e.preventDefault(); undo(); } return; }
  // 左手だけで使えるように: Q W E R T = 道具 / A S D F = 前・保存・次・学習に使う / Z X C V = 元に戻す・ルーペ・重ね・ブラシ⇔消しゴム
  const act = {
    q: () => setTool('brush'), w: () => setTool('erase'), e: () => setTool('bucket'), r: () => setTool('poly'), t: () => setTool('polyerase'),
    a: () => { if (confirmLeave()) go(cur - 1); }, d: () => { if (confirmLeave()) go(cur + 1); },
    arrowleft: () => { if (confirmLeave()) go(cur - 1); }, arrowright: () => { if (confirmLeave()) go(cur + 1); },
    s: () => save(), f: () => { $('usable').checked = !$('usable').checked; dirty = true; info(); },
    z: () => undo(), x: () => { $('loupe').checked = !$('loupe').checked; $('loupe').onchange(); },
    c: () => { const a = $('alpha'); if (+a.value > 0) { a.dataset.prev = a.value; a.value = 0; } else a.value = a.dataset.prev || 45; drawOver(); },
    v: () => setTool(tool === 'erase' ? 'brush' : 'erase'), g: () => toggleSide(),
    '1': () => resize(-1), '2': () => resize(+1),
    ' ': () => finishPoly(), enter: () => finishPoly(), escape: () => { poly = []; drawUI(); },
  }[k];
  if (act) { e.preventDefault(); act(); } });
function resize(dir) { const k = tool === 'erase' ? 'erase' : 'brush', step = Math.max(1, Math.round(SIZES[k] * 0.15));
  SIZES[k] = Math.max(+$('size').min, Math.min(+$('size').max, SIZES[k] + dir * step));
  try { localStorage.setItem('sizes', JSON.stringify(SIZES)); } catch (er) {} showSize(); }
// 画像は中央の幅と窓の残りの高さの両方に収まる大きさにする (スクロールしない)
function fitStage() { if (!W) return; const st = $('stage'), c = $('work');
  const availW = c.clientWidth, availH = innerHeight - c.getBoundingClientRect().top - 12;
  const w = Math.max(200, Math.min(availW, availH * W / H)); st.style.width = w + 'px'; st.style.height = (w * H / W) + 'px'; }
window.addEventListener('resize', fitStage);
// 右のパネルを出す / しまう (G). しまうと画像が大きくなる
try { document.body.classList.toggle('side-hidden', localStorage.getItem('side') === '0'); } catch (e) {}
function toggleSide() { document.body.classList.toggle('side-hidden');
  try { localStorage.setItem('side', document.body.classList.contains('side-hidden') ? '0' : '1'); } catch (e) {} fitStage(); }
$('sidebtn').onclick = toggleSide;
window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

// =================== ドパガキ仕様 ===================
const fx = document.getElementById('fx'), fctx = fx.getContext('2d'); let parts = [];
function fitFx() { fx.width = innerWidth; fx.height = innerHeight; } fitFx(); addEventListener('resize', fitFx);
const calm = () => $('calm').checked;
try { $('calm').checked = localStorage.getItem('calm') === '1'; } catch (e) {}
$('calm').onchange = () => { try { localStorage.setItem('calm', $('calm').checked ? '1' : '0'); } catch (e) {} };
let AC = null;
function tone(f, t0, dur, type = 'square', vol = 0.06) { if (calm()) return; AC = AC || new (window.AudioContext || window.webkitAudioContext)();
  const o = AC.createOscillator(), g = AC.createGain(); o.type = type; o.frequency.value = f; const t = AC.currentTime + t0;
  g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur); o.connect(g).connect(AC.destination); o.start(t); o.stop(t + dur); }
const sfx = {
  save: (c) => { const b = 523 * Math.pow(1.06, Math.min(c, 24)); [0, 4, 7, 12].forEach((n, i) => tone(b * Math.pow(2, n / 12), i * 0.06, 0.18)); },
  level: () => { [0, 4, 7, 12, 16, 19, 24].forEach((n, i) => tone(392 * Math.pow(2, n / 12), i * 0.08, 0.3, 'triangle', 0.08)); },
  pop: () => tone(880 + Math.random() * 400, 0, 0.04, 'sine', 0.02),
  rival: () => { tone(220, 0, 0.15, 'sawtooth', 0.05); tone(196, 0.15, 0.25, 'sawtooth', 0.05); },
};
function burst(x, y, n, big) { if (calm()) return; for (let i = 0; i < n; i++) { const a = Math.random() * 6.28, v = (big ? 6 : 3) + Math.random() * (big ? 9 : 4);
  parts.push({x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - (big ? 6 : 2), life: 60 + Math.random() * 40, c: `hsl(${Math.random() * 360} 95% 60%)`, s: big ? 6 + Math.random() * 6 : 3 }); } }
function tick() { fctx.clearRect(0, 0, fx.width, fx.height);
  const sr = $('stage').getBoundingClientRect(); fctx.save(); fctx.beginPath(); fctx.rect(0, 0, fx.width, fx.height);
  fctx.rect(sr.left, sr.top, sr.width, sr.height); fctx.clip('evenodd');      // 画像の所を抜いて描く
  parts = parts.filter(p => p.life > 0); for (const p of parts) { p.x += p.vx; p.y += p.vy; p.vy += 0.35; p.vx *= 0.99; p.life--;
    fctx.globalAlpha = Math.min(1, p.life / 30); fctx.fillStyle = p.c; fctx.fillRect(p.x, p.y, p.s, p.s * 0.6); }
  fctx.globalAlpha = 1; fctx.restore(); requestAnimationFrame(tick); }
tick();
function shout(text, small) { if (calm()) return; const d = document.createElement('div'); d.className = 'shout' + (small ? ' small' : ''); d.textContent = text;
  $('toast').appendChild(d); setTimeout(() => d.remove(), 1200); }
function shake() { if (calm()) return; document.body.classList.remove('shake'); void document.body.offsetWidth; document.body.classList.add('shake'); }
const PRAISE = ['ナイス!', 'うまい!', '神塗り!', 'その調子!', '天才か?', '職人芸!', 'キレてる!', '最高!', 'えらい!', 'プロの犯行'];
const G = {score: 0, best: 0, day: '', today: 0};      // 名前を選ぶと名簿の値 (と, このブラウザに前から残っていた値の大きい方) になる
G.score = G.score || 0; G.best = G.best || 0; G.day = G.day || ''; G.today = G.today || 0;
const todayKey = new Date().toISOString().slice(0, 10); if (G.day !== todayKey) { G.day = todayKey; G.today = 0; }
let combo = 0, lastSave = 0;
const lvOf = sc => Math.floor(Math.sqrt(sc / 300)) + 1, lvNeed = lv => 300 * (lv - 1) * (lv - 1);
function hud() { $('score').textContent = G.score.toLocaleString(); const lv = lvOf(G.score); $('lv').textContent = lv;
  $('lvbar').firstElementChild.style.width = (100 * (G.score - lvNeed(lv)) / (lvNeed(lv + 1) - lvNeed(lv))) + '%';
  $('combo').textContent = combo ? combo + ' COMBO' : '0'; $('today').textContent = G.today; }
let pendingRewarded = [];
function persist() { if (!ME) return;
  const it = ITEMS[view[cur]];
  fetch(q('api/progress'), {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: ME.name, score: G.score, best: G.best, today: G.today, day: G.day,
                          last: it ? `${it.ds}/${it.split}/${it.stem}` : '', rewarded_add: pendingRewarded})}).then(() => { pendingRewarded = []; }); }
let strokes = 0;   // この画像で塗った回数 (たくさん直すほど点が高い)
function reward() {
  const now = Date.now(); combo = (now - lastSave < 45000) ? combo + 1 : 1; lastSave = now; G.best = Math.max(G.best, combo);
  const fever = combo >= 5; document.body.classList.toggle('fever', fever && !calm());
  const pts = Math.round((100 + Math.min(strokes, 20) * 15) * (fever ? 2 : 1) * (1 + combo * 0.1));
  const before = lvOf(G.score); G.score += pts; G.today += 1; persist(); hud();
  const r = $('save').getBoundingClientRect(); burst(r.left + r.width / 2, r.top + r.height / 2, fever ? 70 : 35, false);
  shout(`+${pts}  ${PRAISE[Math.floor(Math.random() * PRAISE.length)]}`, true); sfx.save(combo);
  const c = $('combo'); c.classList.add('pop'); setTimeout(() => c.classList.remove('pop'), 150);
  if (combo === 5) setTimeout(() => shout('FEVER!! 点数 2 倍'), 300);
  if (G.today % 10 === 0) setTimeout(() => { shout(`今日 ${G.today} 枚!!`); burst(innerWidth / 2, innerHeight / 3, 160, true); }, 500);
  const after = lvOf(G.score);
  if (after > before) setTimeout(() => { shout(`LEVEL UP!! Lv.${after}`); burst(innerWidth / 2, innerHeight / 3, 220, true); shake(); sfx.level();
    setTimeout(() => shout(`ステージ ${after}: ${THEMES[(after - 1) % THEMES.length]}`, true), 1100); }, 700);
  strokes = 0; refreshStats();
}
// 塗るたびに小さな火花と音
ui.addEventListener('pointerdown', e => { strokes++; burst(e.clientX, e.clientY, 6, false); sfx.pop(); });
let lastSpark = 0;
ui.addEventListener('pointermove', e => { if (e.buttons && Date.now() - lastSpark > 60 && (tool === 'brush' || tool === 'erase')) { lastSpark = Date.now(); burst(e.clientX, e.clientY, 2, false); } });
// みんなの進み具合とランキング (10 秒ごと)
let prevCounts = null;
async function refreshStats() { try { const it0 = ITEMS[view[cur]];
  const r = await fetch(q('api/stats' + (it0 ? `?viewing=${encodeURIComponent(it0.ds + '/' + it0.split + '/' + it0.stem)}` + (ME ? `&who=${encodeURIComponent(ME.name)}` : '') : ''))); if (!r.ok) return; const st = await r.json();
  $('progtxt').textContent = `${st.edited} / ${st.total} 枚`; const pct = 100 * st.edited / Math.max(st.total, 1);
  $('progbar').firstElementChild.style.width = pct + '%'; $('progpct').textContent = pct.toFixed(1) + '%';
  const me = $('who').value || '';
  const rows = Object.entries(st.by).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const medal = ['👑', '🥈', '🥉', '　'];
  $('rank').innerHTML = rows.map(([n, c], i) => `${medal[i]} <b>${n || '(名前なし)'}</b> ${c} 枚${n === me ? ' ← あなた' : ''}`).join('<br>');
  if (prevCounts) for (const [n, c] of Object.entries(st.by)) if (n !== me && c > (prevCounts[n] || 0)) { shout(`${n || 'ライバル'} が +${c - (prevCounts[n] || 0)} 枚! 負けるな!`, true); sfx.rival(); }
  prevCounts = st.by;
  if (st.version && VERSION && st.version !== VERSION) autoUpdate();
  showRelabel(st.relabel || {});
} catch (e) {} }
// 自動の答えの学習と書き直し (auto_relabel.py) の始まり・終わりを知らせる
let prevRelabel = null, hiddenRelabel = '';
function showRelabel(rl) { const e = $('relabel'), key = (rl.state || '') + (rl.started || '');
  if (!rl.state || key === hiddenRelabel) { e.style.display = 'none'; prevRelabel = key; return; }
  e.className = rl.state; e.title = (rl.msg || '') + ' (クリックで消す)'; e.style.display = 'block';
  e.textContent = rl.state === 'running' ? `🤖 ${rl.started}〜 自動の答えを学習・書き直し中`
    : rl.state === 'done' ? `✅ ${rl.finished} 書き直し完了 ${rl.rewritten} 枚 (次の画像から新しい答え)` : `⚠ ${rl.finished} 書き直しに失敗`;
  if (prevRelabel !== null && key !== prevRelabel) shout(rl.state === 'running' ? '🤖 学習スタート!' : rl.state === 'done' ? '✅ 書き直し完了!' : '⚠ 書き直し失敗', true);
  prevRelabel = key; }
document.getElementById('relabel').onclick = () => { const st = $('relabel'); hiddenRelabel = prevRelabel; st.style.display = 'none'; };
// 自動更新: エディタが新しくなったら, 保存してから読み込み直す (今の画像に戻る)
let VERSION = '';
fetch(q('api/stats')).then(r => r.json()).then(st => { VERSION = st.version; });
function autoUpdate() { const u = $('update'); u.style.display = 'block';
  if (dirty) { u.textContent = 'エディタが新しくなりました。保存すると読み込み直します'; return; }
  u.textContent = 'エディタが新しくなりました。読み込み直します…';
  try { const it = ITEMS[view[cur]]; if (it) sessionStorage.setItem('resume', `${it.ds}/${it.split}/${it.stem}`); } catch (e) {}
  setTimeout(() => location.reload(), 1200); }
setInterval(refreshStats, 10000); hud(); refreshStats();

// =================== レベルごとに変わる背景 ===================
const bgc = document.getElementById('bg'), bctx2 = bgc.getContext('2d');
function fitBg() { bgc.width = innerWidth; bgc.height = innerHeight; } fitBg(); addEventListener('resize', fitBg);
const THEMES = ['夜明けの道', '星空', 'ネオングリッド', '桜吹雪', 'オーロラ', 'ワープ', '虹のプラズマ'];
const rnd = (a, b) => a + Math.random() * (b - a);
const stars = Array.from({length: 220}, () => ({x: Math.random(), y: Math.random(), z: Math.random(), p: Math.random() * 6.28}));
const petals = Array.from({length: 90}, () => ({x: Math.random(), y: Math.random(), r: rnd(3, 7), s: rnd(0.3, 1), w: Math.random() * 6.28}));
let bgT = 0, lastBg = performance.now();
function level() { return lvOf(G.score); }
function speed() { return (1 + Math.min(combo, 20) * 0.15) * (combo >= 5 ? 1.8 : 1) * (1 + Math.floor((level() - 1) / THEMES.length) * 0.5); }
function drawBg(now) {
  const dt = Math.min(50, now - lastBg) / 1000; lastBg = now;
  const on = !calm() && !matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.body.classList.toggle('anim', on);
  if (!on) { bctx2.clearRect(0, 0, bgc.width, bgc.height); $('stage-name').textContent = ''; requestAnimationFrame(drawBg); return; }
  bgT += dt * speed(); const t = bgT, W2 = bgc.width, H2 = bgc.height, c = bctx2, th = (level() - 1) % THEMES.length;
  $('stage-name').textContent = `ステージ ${level()}: ${THEMES[th]}` + (combo >= 5 ? '  FEVER!!' : '');
  if (th === 0) {               // 夜明けの道: 空の色がゆっくり移り, 道の白線が流れる
    const h = (t * 8) % 360, g = c.createLinearGradient(0, 0, 0, H2);
    g.addColorStop(0, `hsl(${(h + 220) % 360} 60% 18%)`); g.addColorStop(0.55, `hsl(${(h + 20) % 360} 80% 55%)`); g.addColorStop(0.56, '#2a2a2e'); g.addColorStop(1, '#151517');
    c.fillStyle = g; c.fillRect(0, 0, W2, H2);
    c.fillStyle = `hsl(${(h + 40) % 360} 100% 70%)`; c.beginPath(); c.arc(W2 * 0.75, H2 * 0.55, H2 * 0.12, Math.PI, 0); c.fill();
    c.strokeStyle = '#fff8'; c.lineWidth = 6;
    for (let k = 0; k < 12; k++) { const z = ((k / 12 + t * 0.25) % 1), y = H2 * 0.56 + z * z * H2 * 0.44, w = 4 + z * 30;
      c.fillStyle = '#ffffffaa'; c.fillRect(W2 / 2 - w / 2, y, w, 6 + z * 30); }
  } else if (th === 1) {        // 星空: またたく星と流れ星
    c.fillStyle = '#070b1e'; c.fillRect(0, 0, W2, H2);
    for (const st of stars) { const a = 0.4 + 0.6 * Math.abs(Math.sin(t * 2 + st.p)); c.fillStyle = `rgba(255,255,255,${a})`;
      c.fillRect(st.x * W2, st.y * H2, 1 + st.z * 2, 1 + st.z * 2); }
    const m = (t * 0.6) % 1; c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.moveTo(W2 * (1 - m), H2 * m * 0.5); c.lineTo(W2 * (1 - m) + 120, H2 * m * 0.5 - 60); c.stroke();
  } else if (th === 2) {        // ネオングリッド: 奥から流れる格子
    const g = c.createLinearGradient(0, 0, 0, H2); g.addColorStop(0, '#12002b'); g.addColorStop(0.5, '#ff2a6d'); g.addColorStop(0.5, '#05001a'); g.addColorStop(1, '#000');
    c.fillStyle = g; c.fillRect(0, 0, W2, H2); c.strokeStyle = `hsl(${(t * 30) % 360} 100% 60%)`; c.lineWidth = 2;
    for (let k = -20; k <= 20; k++) { c.beginPath(); c.moveTo(W2 / 2, H2 / 2); c.lineTo(W2 / 2 + k * W2 / 8, H2); c.stroke(); }
    for (let k = 0; k < 14; k++) { const z = (k / 14 + t * 0.3) % 1, y = H2 / 2 + z * z * H2 / 2; c.beginPath(); c.moveTo(0, y); c.lineTo(W2, y); c.stroke(); }
  } else if (th === 3) {        // 桜吹雪
    const g = c.createLinearGradient(0, 0, 0, H2); g.addColorStop(0, '#ffd1dc'); g.addColorStop(1, '#a3c4f3'); c.fillStyle = g; c.fillRect(0, 0, W2, H2);
    for (const pt of petals) { pt.y += dt * pt.s * 0.12 * speed(); pt.x += Math.sin(t + pt.w) * 0.0008 * speed(); if (pt.y > 1.05) { pt.y = -0.05; pt.x = Math.random(); }
      c.save(); c.translate(pt.x * W2, pt.y * H2); c.rotate(t * 2 + pt.w); c.fillStyle = '#ff9eb5'; c.beginPath(); c.ellipse(0, 0, pt.r, pt.r * 0.6, 0, 0, 6.28); c.fill(); c.restore(); }
  } else if (th === 4) {        // オーロラ
    c.fillStyle = '#020814'; c.fillRect(0, 0, W2, H2);
    for (let b = 0; b < 4; b++) { c.beginPath(); c.moveTo(0, H2);
      for (let x = 0; x <= W2; x += 20) c.lineTo(x, H2 * (0.35 + b * 0.08) + Math.sin(x * 0.004 + t * (0.8 + b * 0.3)) * 60 + Math.sin(x * 0.011 - t) * 25);
      c.lineTo(W2, H2); c.closePath(); c.fillStyle = `hsla(${(130 + b * 40 + t * 10) % 360} 90% 55% / .25)`; c.fill(); }
  } else if (th === 5) {        // ワープ: 中心から星が飛んでくる
    c.fillStyle = 'rgba(0,0,10,.35)'; c.fillRect(0, 0, W2, H2);
    for (const st of stars) { st.z -= dt * 0.4 * speed(); if (st.z <= 0.02) { st.z = 1; st.x = Math.random(); st.y = Math.random(); }
      const x = W2 / 2 + (st.x - 0.5) * W2 / st.z * 0.5, y = H2 / 2 + (st.y - 0.5) * H2 / st.z * 0.5;
      c.fillStyle = `hsl(${(st.p * 60 + t * 40) % 360} 100% 75%)`; c.fillRect(x, y, 2 / st.z, 2 / st.z); }
  } else {                      // 虹のプラズマ
    const n = 24, cw = W2 / n, ch = H2 / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = Math.sin(i * 0.4 + t) + Math.sin(j * 0.5 - t * 1.3) + Math.sin((i + j) * 0.3 + t * 0.7);
      c.fillStyle = `hsl(${(v * 60 + t * 50) % 360} 90% 50%)`; c.fillRect(i * cw, j * ch, cw + 1, ch + 1); }
  }
  if (combo >= 5) { c.fillStyle = `hsla(${(t * 200) % 360} 100% 60% / .12)`; c.fillRect(0, 0, W2, H2); }
  requestAnimationFrame(drawBg);
}
requestAnimationFrame(drawBg);

async function login(name, fromSaved) {
  const r = await fetch(q('api/worker'), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'login', name})});
  if (!r.ok) { await loadWorkers(''); return; }
  ME = await r.json(); $('who').value = ME.name; try { localStorage.setItem('who', ME.name); } catch (e) {}
  // このブラウザに前から残っていた点数 (同じ名前で使っていたとき) と大きい方
  let old = {}; try { if (fromSaved) old = JSON.parse(localStorage.getItem('game') || '{}'); } catch (e) {}
  G.score = Math.max(ME.score || 0, old.score || 0); G.best = Math.max(ME.best || 0, old.best || 0);
  G.day = todayKey; G.today = (ME.day === todayKey ? ME.today : 0); if (old.day === todayKey) G.today = Math.max(G.today, old.today || 0);
  REWARDED.clear(); (ME.rewarded || []).forEach(k => REWARDED.add(k));
  if (fromSaved) try { JSON.parse(localStorage.getItem('rewarded') || '[]').forEach(k => { if (!REWARDED.has(k)) { REWARDED.add(k); pendingRewarded.push(k); } }); } catch (e) {}
  try { localStorage.removeItem('rewarded'); localStorage.removeItem('game'); } catch (e) {}    // これからは名簿に持つ
  hud(); persist(); await loadWorkers(ME.name);
  // 前の続き (最後に開いていた画像) へ
  if (ME.last && !dirty) { const [ds, split, stem] = ME.last.split('/');
    let k = view.findIndex(i => ITEMS[i].ds === ds && ITEMS[i].split === split && ITEMS[i].stem === stem);
    if (k < 0) { $('filter').value = 'all'; $('dsf').value = ''; applyFilter(); k = view.findIndex(i => ITEMS[i].ds === ds && ITEMS[i].split === split && ITEMS[i].stem === stem); }
    if (k >= 0) { go(k); shout(`おかえり ${ME.name}! 続きから`, true); } }
  refreshStats(); }
document.getElementById('whosel').onchange = async e => { const v = e.target.value; if (!v) return;
  if (dirty && !confirm('未保存の変更があります。名前を変えると保存せずに移ります。よいですか?')) { e.target.value = ME ? ME.name : ''; return; }
  dirty = false;
  if (v === NEW_NAME) { const name = (prompt('新しく登録する名前 (ほかの人と同じ名前は使えません)') || '').trim();
    if (!name) { e.target.value = ME ? ME.name : ''; return; }
    const r = await fetch(q('api/worker'), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'register', name})});
    const j = await r.json();
    if (r.status === 409) { alert(`「${j.name}」はもう登録されています。自分の名前なら一覧から選んでください (別の人なら別の名前にしてください)`); await loadWorkers(ME ? ME.name : ''); return; }
    if (!r.ok) { alert(j.error || '登録できませんでした'); return; }
    await login(j.name); return; }
  await login(v); };
(async () => { await loadList(); await loadWorkers('');
  let saved = ''; try { saved = localStorage.getItem('who') || ''; } catch (e) {}
  if (saved) await login(saved, true);      // 前にこのブラウザで使っていた名前なら, その点数も引き継ぐ
  if (!ME) shout('左上で自分の名前を選んでね', true); })();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _parts(self, prefix):
        ds, split, stem = unquote(self.path[len(prefix):].split("?")[0]).split("/")
        if ds not in DATASETS or split not in ("train", "val") or "/" in stem or ".." in stem:
            raise ValueError
        return ds, split, stem

    def _ok(self):
        if not TOKEN:
            return True
        from urllib.parse import parse_qs, urlparse
        k = parse_qs(urlparse(self.path).query).get("k", [""])[0]
        return secrets.compare_digest(k, TOKEN)

    def do_GET(self):
        self.raw_path = self.path
        if not self._ok():
            self._send(403, "合言葉付きの URL で開いてください".encode(), "text/plain; charset=utf-8")
            return
        self.path = self.path.split("?")[0] if not self.path.startswith("/mask/") else self.path
        try:
            if self.path.startswith("/api/meta/"):
                ds, split, stem = self._parts("/api/meta/")
                d = MASKS / ds / split
                by = json.loads((d / "edited_by.json").read_text()).get(stem, "") if (d / "edited_by.json").exists() else ""
                self._send(200, json.dumps(dict(mtime=str((d / f"{stem}.png").stat().st_mtime_ns), usable=stem in read_list(d / "usable.txt"),
                                                edited=stem in read_list(d / "edited.txt"), by=by)).encode(), "application/json")
            elif self.path == "/api/workers":
                w = load_workers()
                counts = {}
                for p in MASKS.glob("*/*/edited_by.json"):
                    for who in json.loads(p.read_text()).values():
                        counts[who] = counts.get(who, 0) + 1
                self._send(200, json.dumps(sorted([dict(name=n, count=counts.get(n, 0), score=v.get("score", 0)) for n, v in w.items()],
                                                  key=lambda x: -x["count"]), ensure_ascii=False).encode(), "application/json")
            elif self.path == "/api/stats":
                from urllib.parse import parse_qs, urlparse
                v = parse_qs(urlparse(self.raw_path).query).get("viewing", [""])[0]
                if v:
                    note_viewing(self.client_address[0], v)
                    who = parse_qs(urlparse(self.raw_path).query).get("who", [""])[0]
                    if who:
                        with LOCK:
                            w = load_workers()
                            n = find_worker(w, who)
                            if n and w[n].get("last") != v:
                                w[n]["last"] = v
                                save_workers(w)
                self._send(200, json.dumps(stats(), ensure_ascii=False).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/list":
                self._send(200, json.dumps(items()).encode(), "application/json")
            elif self.path.startswith("/img/"):
                p = image_path(*self._parts("/img/"))
                self._send(200, p.read_bytes(), "image/jpeg" if p.suffix != ".png" else "image/png")
            elif self.path.startswith("/mask/"):
                ds, split, stem = self._parts("/mask/")
                self._send(200, (MASKS / ds / split / f"{stem}.png").read_bytes(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")
        except (ValueError, FileNotFoundError, AttributeError):
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._ok():
            self._send(403, b"forbidden", "text/plain")
            return
        self.path = self.path.split("?")[0]
        if self.path in ("/api/worker", "/api/progress"):
            self._worker_post(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            return
        if self.path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ds, split, stem = b["ds"], b["split"], b["stem"]
        if ds not in DATASETS or split not in ("train", "val") or "/" in stem or ".." in stem:
            self._send(400, b"bad", "text/plain")
            return
        d = MASKS / ds / split
        png = base64.b64decode(b["png"].split(",", 1)[1])
        m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            self._send(400, b"bad image", "text/plain")
            return
        with LOCK, open(MASKS / ".lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)          # auto_relabel.py と同時に一覧を書かない
            path = d / f"{stem}.png"
            by_path = d / "edited_by.json"
            by = json.loads(by_path.read_text()) if by_path.exists() else {}
            # 更新時刻はナノ秒の 19 桁. ブラウザの数値では丸められるので文字列で受け渡す
            # 開いた後に変わっていても, auto_relabel.py が自動の答えを書き直しただけなら人の保存を優先する
            if str(path.stat().st_mtime_ns) != str(b.get("mtime", "")) and stem in read_list(d / "edited.txt"):
                # 開いた後に誰かが保存した: 上書きしない
                self._send(409, json.dumps({"by": by.get(stem, "")}, ensure_ascii=False).encode(), "application/json")
                return
            cv2.imwrite(str(path), np.where(m > 127, 255, 0).astype(np.uint8))
            usable, edited = read_list(d / "usable.txt"), read_list(d / "edited.txt")
            (usable.add if b["usable"] else usable.discard)(stem)
            edited.add(stem)
            wk = load_workers()
            by[stem] = find_worker(wk, b.get("who", "")) or normalize_name(b.get("who", ""))
            (d / "usable.txt").write_text("\n".join(sorted(usable)))
            (d / "edited.txt").write_text("\n".join(sorted(edited)))
            by_path.write_text(json.dumps(by, ensure_ascii=False, indent=0))
            mt = str(path.stat().st_mtime_ns)
        print("saved", ds, split, stem, "usable" if b["usable"] else "unused", "by", b.get("who", ""), flush=True)
        self._send(200, json.dumps({"ok": True, "mtime": mt}).encode(), "application/json")

    def _worker_post(self, b):
        """作業者の登録 (同じ名前は登録できない) / 続きから (名簿の名前で入る) / 進み具合 (点数・最後の画像・点をもらった画像) の保存."""
        with LOCK:
            w = load_workers()
            name = (b.get("name") or "").strip()
            n = find_worker(w, name)
            if self.path == "/api/worker" and b.get("action") == "register":
                name = normalize_name(" ".join(name.split()))[:40]
                if not name:
                    self._send(400, json.dumps({"error": "名前が空です"}, ensure_ascii=False).encode(), "application/json")
                    return
                if n:
                    self._send(409, json.dumps({"error": "その名前はもう使われています", "name": n}, ensure_ascii=False).encode(), "application/json")
                    return
                w[name] = dict(score=0, best=0, today=0, day="", last="", rewarded=[])
                save_workers(w)
                print("registered", name, flush=True)
                self._send(200, json.dumps(dict(name=name, **w[name]), ensure_ascii=False).encode(), "application/json")
                return
            if not n:
                self._send(404, json.dumps({"error": "名簿にない名前です"}, ensure_ascii=False).encode(), "application/json")
                return
            if self.path == "/api/progress":
                v = w[n]
                v["score"] = max(v.get("score", 0), int(b.get("score", 0)))
                v["best"] = max(v.get("best", 0), int(b.get("best", 0)))
                if b.get("day"):
                    v["today"] = max(v.get("today", 0), int(b.get("today", 0))) if v.get("day") == b["day"] else int(b.get("today", 0))
                    v["day"] = b["day"]
                if b.get("last"):
                    v["last"] = b["last"]
                v["rewarded"] = sorted(set(v.get("rewarded", [])) | set(b.get("rewarded_add", [])))
                save_workers(w)
            self._send(200, json.dumps(dict(name=n, **w[n]), ensure_ascii=False).encode(), "application/json")

    def log_message(self, *a):
        pass


def tailscale_ip():
    """この Mac の Tailscale の住所 (100.64.0.0/10). なければ空."""
    import subprocess
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        return out[0] if out else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8766)
    ap.add_argument("--share", action="store_true", help="同じネットワークの別のパソコンからも開けるようにする (合言葉つき)")
    ap.add_argument("--token", default="", help="合言葉を自分で決める (省略すると起動のたびに作る)")
    ap.add_argument("--tailscale-only", action="store_true", help="Tailscale の住所 (100.x.y.z) だけで待つ (研究室のネットワークや外からは開けない)")
    args = ap.parse_args()
    if args.share:
        TOKEN = args.token or secrets.token_urlsafe(12)
        ts = tailscale_ip()
        host = "0.0.0.0"
        if args.tailscale_only:
            if not ts:
                sys.exit("Tailscale の住所が見つかりません (Tailscale につながっているか確かめてください)")
            host = ts
        else:
            print(f"黒路面の答えエディタ (共有): http://{lan_ip()}:{args.port}/?k={TOKEN}", flush=True)
        if ts:
            print(f"  Tailscale では: http://{ts}:{args.port}/?k={TOKEN}", flush=True)
        if not args.tailscale_only:
            print(f"  この Mac では: http://localhost:{args.port}/?k={TOKEN}", flush=True)
    else:
        print(f"黒路面の答えエディタ: http://localhost:{args.port}/", flush=True)
        host = "127.0.0.1"
    try:
        server = ThreadingHTTPServer((host, args.port), Handler)
    except OSError as e:
        if e.errno == 48:
            sys.exit(f"ポート {args.port} はもう使われています. エディタがすでに動いていないか確かめてください "
                     f"(動いていればその URL をそのまま使えます. 止めるにはそのターミナルで Ctrl-C)")
        raise
    server.serve_forever()


if __name__ == "__main__":
    main()
