#!/usr/bin/env python3
"""
segment_editor.py - 区間を手で直す道具 (ブラウザで開く)

図面の上に外周と区間の境目を描き, 境目をドラッグで動かす / コースをダブルクリックで足す / 選んで Delete で消す.
区間ごとに名前とメモも書ける. 「保存」で data/topological_map/segments_manual.json に書く.
最初の区間は, segments_manual.json があればそれ, なければ案 (segment_proposal_view.py の proposal).

使い方:
  python3 scripts/segment_editor.py [ポート (既定 8765)]
  -> http://localhost:8765/ を開く
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS_DIR / "scripts"))
import segment_proposal_view as V  # noqa: E402

TOPO = WS_DIR / "data/topological_map/course_topology.json"
OUT = WS_DIR / "data/topological_map/segments_manual.json"
PXM = 5.711


def initial():
    topo = json.load(open(TOPO))
    if OUT.exists():
        saved = json.load(open(OUT))
        return topo, saved["segments"], "segments_manual.json"
    segs = [dict(s_start=float(sg["s_start"]), name="", note="") for sg in V.proposal(topo)]
    return topo, segs, "案 (segment_proposal_view.py)"


def page_data():
    topo, segs, src = initial()
    st = [dict(s=r["s"], x=r["x"], y=r["y"], shape=r["shape"], left=r["left"], center=r["center"], right=r["right"],
               ld=r["left_double"], rd=r["right_double"], lm=r["landmarks"]) for r in topo["stations"]]
    return dict(stations=st, segments=segs, source=src, lap=topo["meta"]["lap_length_m"], step=topo["meta"]["step_m"],
                pxm=PXM)


HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>区間エディタ</title>
<style>
:root { --bg:#f6f6f4; --panel:#ffffff; --ink:#1d1d1f; --sub:#5f6368; --line:#d9d9d6; --accent:#2f6fde; --warn:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --panel:#202124; --ink:#e8e8e8; --sub:#a0a4a8; --line:#3a3b3e; --accent:#7aa7ff; --warn:#ff7b6b; } }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif; background:var(--bg); color:var(--ink); }
header { display:flex; gap:12px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 8px 0 0; }
button { font:inherit; font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--line); background:var(--panel); color:var(--ink); cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button:disabled { opacity:.45; cursor:default; }
#status { font-size:12px; color:var(--sub); }
main { display:grid; grid-template-columns: minmax(0,1fr) 420px; height:calc(100vh - 52px); }
@media (max-width: 900px) { main { grid-template-columns: 1fr; height:auto; } #map { height:70vh; } }
#map { position:relative; overflow:hidden; background:#2b2b2b; touch-action:none; }
svg { width:100%; height:100%; display:block; cursor:grab; user-select:none; }
#side { overflow:auto; border-left:1px solid var(--line); background:var(--panel); padding:10px 12px; }
.help { font-size:12px; color:var(--sub); line-height:1.6; margin:0 0 10px; }
.seg { border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin-bottom:8px; }
.seg.sel { outline:2px solid var(--accent); }
.seg .head { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; }
.sw { width:14px; height:14px; border-radius:3px; flex:none; }
.seg .auto { font-size:12px; color:var(--sub); margin:4px 0 6px; line-height:1.5; }
.seg input, .seg textarea { width:100%; font:inherit; font-size:12px; padding:4px 6px; border:1px solid var(--line); border-radius:4px; background:var(--bg); color:var(--ink); }
.seg textarea { height:40px; resize:vertical; margin-top:4px; }
.short { color:var(--warn); }
</style></head>
<body>
<header>
  <h1>区間エディタ</h1>
  <button id="save" class="primary">保存</button>
  <button id="del" disabled>選んだ境目を消す</button>
  <button id="reset">案に戻す</button>
  <button id="fit">全体を表示</button>
  <span id="status"></span>
</header>
<main>
  <div id="map"><svg id="svg"></svg></div>
  <div id="side">
    <p class="help">境目 (白い丸) をドラッグして動かす (0.5 m ごと) / コースの線をダブルクリックして境目を足す /
    丸をクリックして選び Delete キーで消す / ホイールで拡大・ドラッグで移動。区間の番号は発進位置から反時計回り。</p>
    <div id="list"></div>
  </div>
</main>
<script>
const D = __DATA__;
const ST = D.stations, N = ST.length, STEP = D.step, LAP = D.lap, PXM = D.pxm;
const svg = document.getElementById('svg');
const NS = 'http://www.w3.org/2000/svg';
let bounds = D.segments.map(s => Math.round(s.s_start / STEP) % N);       // 境目 = 区間の始めのステーション番号
let meta = {}; D.segments.forEach(s => meta[Math.round(s.s_start / STEP) % N] = {name: s.name || '', note: s.note || ''});
let sel = null, dirty = false;
const TYPE = {solid:'実線', dashed:'破線', none:'なし'};
const SHAPE = {straight:'直線', left:'左カーブ', right:'右カーブ'};

// 図面座標 (m) -> 図面画素
const px = i => [ST[i].x * PXM, -ST[i].y * PXM];
const xs = ST.map(s => s.x * PXM), ys = ST.map(s => -s.y * PXM);
const box = [Math.min(...xs) - 40, Math.min(...ys) - 40, Math.max(...xs) - Math.min(...xs) + 80, Math.max(...ys) - Math.min(...ys) + 80];
let vb = box.slice();
function setVB() { svg.setAttribute('viewBox', vb.join(' ')); }
setVB();
const img = document.createElementNS(NS, 'image');
img.setAttribute('href', 'drawing.png'); img.setAttribute('x', 0); img.setAttribute('y', 0);
img.setAttribute('width', 1200); img.setAttribute('height', 848); img.setAttribute('opacity', 0.55);
svg.appendChild(img);
const gSeg = document.createElementNS(NS, 'g'), gHit = document.createElementNS(NS, 'g'), gB = document.createElementNS(NS, 'g'), gLab = document.createElementNS(NS, 'g');
svg.append(gSeg, gHit, gLab, gB);
// コースをダブルクリックするための太い透明線
const hit = document.createElementNS(NS, 'polyline');
hit.setAttribute('points', ST.map((_, i) => px(i).join(',')).concat([px(0).join(',')]).join(' '));
hit.setAttribute('fill', 'none'); hit.setAttribute('stroke', 'transparent'); hit.setAttribute('stroke-width', 14);
gHit.appendChild(hit);

function colorOf(k) { return `hsl(${(k * 0.61803 * 360) % 360} 85% 58%)`; }
function sorted() { return [...new Set(bounds)].sort((a, b) => a - b); }
function segsNow() {
  const b = sorted();
  return b.map((st, k) => { const en = b[(k + 1) % b.length]; const n = ((en - st + N - 1) % N) + 1; return {k, st, n}; });
}
function nearest(clientX, clientY) {
  const p = svg.createSVGPoint(); p.x = clientX; p.y = clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  let bi = 0, bd = 1e18;
  for (let i = 0; i < N; i++) { const [x, y] = px(i); const d = (x - q.x) ** 2 + (y - q.y) ** 2; if (d < bd) { bd = d; bi = i; } }
  return bi;
}
function describe(st, n) {
  const cnt = (f) => { const c = {}; for (let j = 0; j < n; j++) { const v = f(ST[(st + j) % N]); c[v] = (c[v] || 0) + 1; } return c; };
  const fmt = (c, map) => Object.entries(c).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${map ? map[k] : k} ${(v * STEP).toFixed(1)}m`).join('・');
  const lm = {}; for (let j = 0; j < n; j++) for (const l of ST[(st + j) % N].lm) lm[l] = (lm[l] || 0) + 1;
  const LM = {signal:'信号', crosswalk:'横断歩道', stop_line:'停止線'};
  const dbl = []; let l = 0, r = 0; for (let j = 0; j < n; j++) { l += ST[(st + j) % N].ld; r += ST[(st + j) % N].rd; }
  if (l) dbl.push(`左 ${(l * STEP).toFixed(1)}m`); if (r) dbl.push(`右 ${(r * STEP).toFixed(1)}m`);
  return `形 ${fmt(cnt(s => s.shape), SHAPE)}<br>左境界 ${fmt(cnt(s => s.left), TYPE)} / 中央 ${fmt(cnt(s => s.center), TYPE)} / 右境界 ${fmt(cnt(s => s.right), TYPE)}` +
         `<br>二重線 ${dbl.join('・') || 'なし'} / 目印 ${Object.entries(lm).map(([k, v]) => `${LM[k] || k} ${(v * STEP).toFixed(1)}m`).join('・') || 'なし'}`;
}
function draw() {
  gSeg.replaceChildren(); gB.replaceChildren(); gLab.replaceChildren();
  const S = segsNow();
  const cx = xs.reduce((a, b) => a + b) / N, cy = ys.reduce((a, b) => a + b) / N;
  for (const s of S) {
    const pts = []; for (let j = 0; j <= s.n; j++) pts.push(px((s.st + j) % N).join(','));
    const pl = document.createElementNS(NS, 'polyline');
    pl.setAttribute('points', pts.join(' ')); pl.setAttribute('fill', 'none'); pl.setAttribute('stroke', colorOf(s.k));
    pl.setAttribute('stroke-width', 5); pl.setAttribute('stroke-linecap', 'butt'); pl.style.pointerEvents = 'none';
    gSeg.appendChild(pl);
    const [mx, my] = px((s.st + Math.floor(s.n / 2)) % N); const vx = mx - cx, vy = my - cy, vl = Math.hypot(vx, vy) || 1;
    const t = document.createElementNS(NS, 'text');
    t.setAttribute('x', mx + vx / vl * 22); t.setAttribute('y', my + vy / vl * 22 + 5); t.setAttribute('text-anchor', 'middle');
    t.setAttribute('font-size', 15); t.setAttribute('font-weight', 700); t.setAttribute('fill', colorOf(s.k));
    t.setAttribute('stroke', '#000'); t.setAttribute('stroke-width', 3); t.setAttribute('paint-order', 'stroke'); t.textContent = s.k;
    t.style.pointerEvents = 'none'; gLab.appendChild(t);
  }
  for (const b of sorted()) {
    const [x, y] = px(b); const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', b === sel ? 7 : 5);
    c.setAttribute('fill', b === sel ? '#ffd400' : '#fff'); c.setAttribute('stroke', '#000'); c.setAttribute('stroke-width', 2);
    c.style.cursor = 'grab'; c.dataset.b = b;
    c.addEventListener('pointerdown', ev => startDrag(ev, b));
    gB.appendChild(c);
  }
  // 一覧
  const list = document.getElementById('list'); list.replaceChildren();
  for (const s of S) {
    const m = meta[s.st] || (meta[s.st] = {name: '', note: ''});
    const div = document.createElement('div'); div.className = 'seg' + (s.st === sel ? ' sel' : '');
    const len = s.n * STEP, en = ((s.st + s.n) % N) * STEP;
    div.innerHTML = `<div class="head"><span class="sw" style="background:${colorOf(s.k)}"></span>区間 ${s.k}　${(s.st * STEP).toFixed(1)} 〜 ${en.toFixed(1)} m　` +
      `<span class="${len < 3 ? 'short' : ''}">(${len.toFixed(1)} m)</span></div><div class="auto">${describe(s.st, s.n)}</div>`;
    const name = document.createElement('input'); name.placeholder = '名前 (例: 信号手前の直線)'; name.value = m.name;
    name.oninput = () => { m.name = name.value; markDirty(); };
    const note = document.createElement('textarea'); note.placeholder = 'メモ'; note.value = m.note;
    note.oninput = () => { m.note = note.value; markDirty(); };
    div.append(name, note);
    div.querySelector('.head').onclick = () => { sel = s.st; draw(); };
    list.appendChild(div);
  }
  document.getElementById('del').disabled = sel === null;
  document.getElementById('status').textContent = `${S.length} 区間 / 読み込み元: ${D.source}` + (dirty ? ' / 未保存の変更あり' : '');
}
function markDirty() { dirty = true; document.getElementById('status').textContent = `${segsNow().length} 区間 / 未保存の変更あり`; }
// 境目のドラッグ (0.5 m ごとに吸い付く)
function startDrag(ev, b) {
  ev.stopPropagation(); ev.preventDefault(); sel = b; let cur = b;
  const move = e => { const i = nearest(e.clientX, e.clientY); if (i !== cur && !bounds.includes(i)) {
      bounds = bounds.map(x => x === cur ? i : x); meta[i] = meta[cur] || {name: '', note: ''}; if (i !== cur) delete meta[cur];
      cur = i; sel = i; dirty = true; draw(); } };
  const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); draw(); };
  window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  draw();
}
hit.addEventListener('dblclick', ev => { const i = nearest(ev.clientX, ev.clientY); if (!bounds.includes(i)) { bounds.push(i); sel = i; dirty = true; draw(); } });
function delSel() { if (sel === null || bounds.length <= 1) return; bounds = bounds.filter(x => x !== sel); delete meta[sel]; sel = null; dirty = true; draw(); }
document.getElementById('del').onclick = delSel;
window.addEventListener('keydown', e => { if ((e.key === 'Delete' || e.key === 'Backspace') && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); delSel(); } });
// 拡大・移動
svg.addEventListener('wheel', e => { e.preventDefault(); const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse()); const f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  vb = [q.x - (q.x - vb[0]) * f, q.y - (q.y - vb[1]) * f, vb[2] * f, vb[3] * f]; setVB(); }, {passive: false});
let pan = null;
svg.addEventListener('pointerdown', e => { if (e.target.tagName === 'circle') return; pan = {x: e.clientX, y: e.clientY, vb: vb.slice()}; svg.style.cursor = 'grabbing'; });
window.addEventListener('pointermove', e => { if (!pan) return; const k = vb[2] / svg.clientWidth;
  vb = [pan.vb[0] - (e.clientX - pan.x) * k, pan.vb[1] - (e.clientY - pan.y) * k, vb[2], vb[3]]; setVB(); });
window.addEventListener('pointerup', () => { pan = null; svg.style.cursor = 'grab'; });
document.getElementById('fit').onclick = () => { vb = box.slice(); setVB(); };
document.getElementById('reset').onclick = async () => { if (!confirm('案の区間に戻します (保存するまでファイルは変わりません)')) return;
  const r = await fetch('api/proposal'); const p = await r.json(); bounds = p.map(s => Math.round(s.s_start / STEP) % N); meta = {}; sel = null; dirty = true; draw(); };
document.getElementById('save').onclick = async () => {
  const segs = segsNow().map(s => ({id: s.k, s_start: s.st * STEP, s_end: ((s.st + s.n) % N) * STEP, length_m: s.n * STEP,
    name: (meta[s.st] || {}).name || '', note: (meta[s.st] || {}).note || ''}));
  const r = await fetch('api/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({segments: segs})});
  const j = await r.json(); dirty = false; document.getElementById('status').textContent = `${segs.length} 区間を保存しました -> ${j.path}`;
};
window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });
draw();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.replace("__DATA__", json.dumps(page_data(), ensure_ascii=False)).encode(), "text/html; charset=utf-8")
        elif self.path == "/drawing.png":
            topo = json.load(open(TOPO))
            self._send(200, open(topo["meta"]["source"]["drawing"], "rb").read(), "image/png")
        elif self.path == "/api/proposal":
            topo = json.load(open(TOPO))
            self._send(200, json.dumps([dict(s_start=float(sg["s_start"])) for sg in V.proposal(topo)]).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = dict(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"), topology=str(TOPO), segments=body["segments"])
        json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("saved", len(body["segments"]), "segments ->", OUT, flush=True)
        self._send(200, json.dumps({"path": str(OUT.relative_to(WS_DIR))}).encode(), "application/json")

    def log_message(self, *a):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"区間エディタ: http://localhost:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
