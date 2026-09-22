#!/usr/bin/env python3
"""
verification_gui.py - モデル・動画選択式 検証GUI (ブラウザ版)

概要:
    学習済みモデル (models/*.pt, *.pth) と検証動画 (mp4/*.mp4) をブラウザから選択し、
    `ros2 launch` を起動して RViz 上で認識結果をリアルタイムに目視検証できるWebツール。
    標準ライブラリのみ (http.server) で動作し、ホスト側ブラウザ (http://localhost:8090) で開く。

対応モデル:
    1. YOLO 物体検出 (*.pt):
       - cone.pt, traffic_light.pt, jinmen_dog.pt 等
       - 起動: traffic_light_video_test.launch.py (model_path:=...)
       - 描画: バウンディングボックス、クラス名、信頼度、推定距離
    2. YOLOP 白線・走路認識 (*.pth):
       - shiho_lane_mask_v2_best.pth, yolop_official_*.pth 等
       - 起動: yolop_video_test.launch.py (weight_path:=...)
       - 描画: 白線境界線・走行可能エリアオーバーレイ
    3. UFLD 白線認識 (ufld*.pth):
       - ufld_honda_finetuned_best.pth 等
       - 起動: ufld_video_test.launch.py (weight_path:=...)
       - 描画: UFLD レーンキーポイント描画
"""

import glob
import json
import os
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = int(os.environ.get("VERIFICATION_GUI_PORT", "8090"))
DEFAULT_MP4_DIR = os.environ.get("VERIFICATION_MP4_DIR", "/aiformula_ws/mp4")
DEFAULT_MODELS_DIR = os.environ.get("VERIFICATION_MODELS_DIR", "/aiformula_ws/models")

DEVICES = ["cpu", "mps", "0"]


def resolve_dir_fallback(target_dir: str, fallback_candidates: list) -> str:
    """指定ディレクトリが存在しない場合、ホストやローカルの候補から探索して返す。"""
    if os.path.isdir(target_dir):
        return target_dir
    for cand in fallback_candidates:
        if os.path.isdir(cand):
            return cand
    return target_dir


def scan_models(models_dir: str) -> list:
    """models_dir 配下の .pt / .pth モデルを探索してメタ情報付きで返す。"""
    resolved_dir = resolve_dir_fallback(
        models_dir,
        ["models", "/aiformula_ws/models", os.path.expanduser("~/aiformula_ws/models")]
    )
    if not os.path.isdir(resolved_dir):
        return []

    model_files = []
    for ext in ["*.pt", "*.pth"]:
        model_files.extend(glob.glob(os.path.join(resolved_dir, ext)))
        model_files.extend(glob.glob(os.path.join(resolved_dir, "*", ext)))

    model_files = sorted(list(set(model_files)))
    results = []
    for path in model_files:
        name = os.path.basename(path)
        if name.endswith(".pt"):
            m_type = "yolo"
            type_label = "YOLO 物体検出"
            desc = "バウンディングボックス + 距離推定"
        elif "ufld" in name.lower():
            m_type = "ufld"
            type_label = "UFLD 白線認識"
            desc = "Ultra-Fast-Lane-Detection"
        else:
            m_type = "yolop"
            type_label = "YOLOP 白線・走路"
            desc = "マルチタスク走行可能領域 & 白線"

        results.append({
            "name": name,
            "path": path,
            "type": m_type,
            "type_label": type_label,
            "desc": desc,
            "label": f"{name} [{type_label}]"
        })
    return results


def scan_videos(mp4_dir: str) -> list:
    """mp4_dir 配下の動画を探索して返す。"""
    resolved_dir = resolve_dir_fallback(
        mp4_dir,
        ["mp4", "/aiformula_ws/mp4", os.path.expanduser("~/aiformula_ws/mp4"),
         "data/tasks/cone/raw_videos", "/aiformula_ws/data/tasks/cone/raw_videos"]
    )
    if not os.path.isdir(resolved_dir):
        return []

    paths = sorted(glob.glob(os.path.join(resolved_dir, "*.mp4")) + glob.glob(os.path.join(resolved_dir, "*.avi")))
    return [{"name": os.path.basename(p), "path": p} for p in paths]


class LaunchRunner:
    """`ros2 launch` を1つだけ実行・停止・ログ収集する。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.label = ""
        self.logs: list[str] = []
        self.returncode = None

    def status(self):
        with self._lock:
            running = self.proc is not None and self.proc.poll() is None
            return {"running": running, "label": self.label, "returncode": self.returncode}

    def logs_since(self, offset: int):
        with self._lock:
            return self.logs[offset:], len(self.logs)

    def start(self, model_path: str, video_path: str, device: str, fps: str, loop: bool, run_integration: bool = False):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("既にlaunchが実行中です。先に停止してください。")

            model_name = os.path.basename(model_path)
            if model_name.endswith(".pt"):
                m_type = "yolo"
            elif "ufld" in model_name.lower():
                m_type = "ufld"
            else:
                m_type = "yolop"

            if run_integration:
                # 統合パイプライン (白線 + YOLO + PurePursuit)
                launch_file = "video_test.launch.py"
                cmd = [
                    "ros2", "launch", "oit_navigation", launch_file,
                    f"video_path:={video_path}",
                    f"fps:={fps}",
                    f"loop:={'true' if loop else 'false'}",
                    f"use_device:={device}",
                    "traffic_light:=true",
                    f"traffic_light_model_path:={model_path if m_type == 'yolo' else '/aiformula_ws/models/traffic_light.pt'}",
                    "rviz:=true",
                ]
                if m_type in ("yolop", "ufld"):
                    cmd.append(f"weight_path:={model_path}")
                label = f"統合走行検証 [{model_name}]"
            elif m_type == "yolo":
                # YOLO モデル単体検証 (cone, traffic_light 等)
                launch_file = "traffic_light_video_test.launch.py"
                cmd = [
                    "ros2", "launch", "oit_navigation", launch_file,
                    f"video_path:={video_path}",
                    f"model_path:={model_path}",
                    f"device:={device}",
                    f"fps:={fps}",
                    f"loop:={'true' if loop else 'false'}",
                    "rviz:=true",
                ]
                label = f"YOLO検証 [{model_name}]"
            elif m_type == "ufld":
                # UFLD 白線モデル単体検証
                launch_file = "ufld_video_test.launch.py"
                cmd = [
                    "ros2", "launch", "oit_navigation", launch_file,
                    f"video_path:={video_path}",
                    f"weight_path:={model_path}",
                    f"use_device:={device}",
                    f"fps:={fps}",
                    f"loop:={'true' if loop else 'false'}",
                    "rviz:=true",
                ]
                label = f"UFLD白線検証 [{model_name}]"
            else:
                # YOLOP 白線モデル単体検証
                launch_file = "yolop_video_test.launch.py"
                cmd = [
                    "ros2", "launch", "oit_navigation", launch_file,
                    f"video_path:={video_path}",
                    f"weight_path:={model_path}",
                    f"use_device:={device}",
                    f"fps:={fps}",
                    f"loop:={'true' if loop else 'false'}",
                    "rviz:=true",
                ]
                label = f"YOLOP白線検証 [{model_name}]"

            self.logs = [f"$ {' '.join(cmd)}"]
            self.label = label
            self.returncode = None

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
            proc = self.proc

        threading.Thread(target=self._pump_output, args=(proc,), daemon=True).start()
        return {"ok": True, "label": label}

    def _pump_output(self, proc: subprocess.Popen):
        if proc.stdout is not None:
            for line in proc.stdout:
                with self._lock:
                    self.logs.append(line.rstrip("\n"))
        proc.wait()
        with self._lock:
            self.returncode = proc.returncode
            self.logs.append(f"[launch終了] returncode={proc.returncode}")

    def stop(self):
        with self._lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return {"ok": True, "already_stopped": True}
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            return {"ok": True}

        def _force_kill():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        threading.Timer(10.0, _force_kill).start()
        return {"ok": True}


RUNNER = LaunchRunner()

INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OIT Navigation モデル検証GUI</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Hiragino Sans", sans-serif; background:#0f172a; color:#f1f5f9; margin:0; padding:24px; }
  .container { max-width: 960px; margin: 0 auto; }
  h1 { font-size:22px; font-weight:700; margin:0 0 4px; display:flex; align-items:center; gap:8px; }
  .subtitle { font-size:13px; color:#94a3b8; margin-bottom:20px; }
  .panel { background:#1e293b; border-radius:10px; padding:20px; margin-bottom:20px; border:1px solid #334155; box-shadow:0 4px 6px -1px rgba(0,0,0,0.1); }
  label { display:block; font-size:13px; font-weight:600; color:#cbd5e1; margin-bottom:6px; margin-top:14px; }
  label:first-child { margin-top:0; }
  select, input[type=text], input[type=number] { width:100%; box-sizing:border-box; padding:8px 10px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:#f8fafc; font-size:14px; transition:border 0.2s; }
  select:focus, input:focus { outline:none; border-color:#38bdf8; }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .row > div { flex:1; min-width:180px; }
  .flex-row { display:flex; gap:8px; align-items:flex-end; }
  .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:700; letter-spacing:0.5px; }
  .badge-yolo { background:#0369a1; color:#e0f2fe; }
  .badge-yolop { background:#047857; color:#d1fae5; }
  .badge-ufld { background:#6d28d9; color:#ede9fe; }
  .model-meta { font-size:12px; color:#94a3b8; margin-top:6px; display:flex; align-items:center; gap:8px; min-height:22px; }
  .checkbox-row { display:flex; align-items:center; gap:8px; margin-top:16px; background:#0f172a; padding:8px 12px; border-radius:6px; border:1px solid #334155; }
  .checkbox-row label { margin:0; cursor:pointer; font-weight:normal; }
  button { padding:9px 18px; border-radius:6px; border:none; cursor:pointer; font-size:14px; font-weight:600; transition:all 0.15s; }
  #startBtn { background:#0284c7; color:white; }
  #startBtn:hover:not(:disabled) { background:#0369a1; }
  #startBtn:disabled { background:#334155; color:#64748b; cursor:not-allowed; }
  #stopBtn { background:#e11d48; color:white; }
  #stopBtn:hover:not(:disabled) { background:#be123c; }
  #stopBtn:disabled { background:#334155; color:#64748b; cursor:not-allowed; }
  .btn-secondary { background:#334155; color:#f1f5f9; padding:8px 12px; font-size:13px; }
  .btn-secondary:hover { background:#475569; }
  .btn-bar { display:flex; align-items:center; gap:12px; margin-top:20px; }
  #status { font-size:13px; font-weight:600; color:#94a3b8; }
  #status.running { color:#38bdf8; }
  #log { background:#020617; border-radius:8px; padding:14px; height:400px; overflow-y:auto; white-space:pre-wrap; font-family:ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size:12px; line-height:1.6; color:#94a3b8; border:1px solid #1e293b; }
</style>
</head>
<body>
<div class="container">
  <h1><span>🎯</span> OIT Navigation モデル動画検証GUI</h1>
  <div class="subtitle">モデル (.pt / .pth) と検証動画 (.mp4) を選択して ros2 launch を起動します。RViz (:8080) で認識枠を目視確認できます。</div>

  <div class="panel">
    <!-- モデル選択 -->
    <div class="flex-row">
      <div style="flex:1;">
        <label for="modelsdir">モデルフォルダ</label>
        <input type="text" id="modelsdir" value="/aiformula_ws/models">
      </div>
      <div>
        <button id="reloadModelsBtn" class="btn-secondary">モデル再読込</button>
      </div>
    </div>

    <label for="model">検証モデル (models/*.pt, *.pth)</label>
    <select id="model"></select>
    <div id="modelMeta" class="model-meta">
      <span id="modelBadge" class="badge"></span>
      <span id="modelDesc"></span>
    </div>

    <hr style="border:0; border-top:1px solid #334155; margin:18px 0;">

    <!-- 動画選択 -->
    <div class="flex-row">
      <div style="flex:1;">
        <label for="mp4dir">動画フォルダ</label>
        <input type="text" id="mp4dir" value="/aiformula_ws/mp4">
      </div>
      <div>
        <button id="reloadVideosBtn" class="btn-secondary">動画再読込</button>
      </div>
    </div>

    <label for="video">検証動画 (.mp4)</label>
    <select id="video"></select>

    <!-- オプション設定 -->
    <div class="row" style="margin-top:6px;">
      <div>
        <label for="device">推論デバイス</label>
        <select id="device"></select>
      </div>
      <div>
        <label for="fps">再生FPS</label>
        <input type="number" id="fps" value="15.0" step="0.5">
      </div>
    </div>

    <div class="row">
      <div class="checkbox-row">
        <input type="checkbox" id="loop" checked>
        <label for="loop">動画ループ再生</label>
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="integration">
        <label for="integration">統合走行モード (白線+PurePursuit経路生成も同時に回す)</label>
      </div>
    </div>

    <!-- アクションボタン -->
    <div class="btn-bar">
      <button id="startBtn">検証開始 (Launch)</button>
      <button id="stopBtn" disabled>停止</button>
      <span id="status">停止中</span>
    </div>
  </div>

  <div class="panel" style="padding:14px;">
    <div style="font-size:12px; font-weight:600; color:#cbd5e1; margin-bottom:8px;">実行ログ (stdout / stderr)</div>
    <div id="log"></div>
  </div>
</div>

<script>
let logOffset = 0;
let pollTimer = null;
let currentModels = [];

async function loadModels() {
  const dir = document.getElementById('modelsdir').value;
  const res = await fetch('/api/models?dir=' + encodeURIComponent(dir));
  currentModels = await res.json();
  const sel = document.getElementById('model');
  sel.innerHTML = '';
  if (currentModels.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(モデルが見つかりません)';
    sel.appendChild(opt);
    updateModelMeta();
    return;
  }
  currentModels.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.path;
    opt.textContent = m.label;
    sel.appendChild(opt);
  });
  updateModelMeta();
}

function updateModelMeta() {
  const sel = document.getElementById('model');
  const badgeEl = document.getElementById('modelBadge');
  const descEl = document.getElementById('modelDesc');
  const selected = currentModels.find(m => m.path === sel.value);
  if (!selected) {
    badgeEl.style.display = 'none';
    descEl.textContent = '';
    return;
  }
  badgeEl.style.display = 'inline-block';
  badgeEl.className = 'badge badge-' + selected.type;
  badgeEl.textContent = selected.type_label;
  descEl.textContent = selected.desc;
}

document.getElementById('model').addEventListener('change', updateModelMeta);

async function loadVideos() {
  const dir = document.getElementById('mp4dir').value;
  const res = await fetch('/api/videos?dir=' + encodeURIComponent(dir));
  const videos = await res.json();
  const sel = document.getElementById('video');
  sel.innerHTML = '';
  if (videos.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(動画が見つかりません)';
    sel.appendChild(opt);
    return;
  }
  videos.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.path;
    opt.textContent = v.name;
    sel.appendChild(opt);
  });
}

async function loadDevices() {
  const res = await fetch('/api/devices');
  const devices = await res.json();
  const sel = document.getElementById('device');
  sel.innerHTML = '';
  devices.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = d === '0' ? '0 (CUDA GPU)' : d.toUpperCase();
    sel.appendChild(opt);
  });
}

function appendLog(lines) {
  const box = document.getElementById('log');
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 16;
  lines.forEach(l => { box.textContent += l + "\\n"; });
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function pollLogsAndStatus() {
  try {
    const [logRes, statusRes] = await Promise.all([
      fetch('/api/logs?since=' + logOffset),
      fetch('/api/status'),
    ]);
    const logData = await logRes.json();
    appendLog(logData.lines);
    logOffset = logData.next;

    const st = await statusRes.json();
    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (st.running) {
      statusEl.className = 'running';
      statusEl.textContent = '実行中: ' + st.label;
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } else {
      statusEl.className = '';
      statusEl.textContent = '停止中';
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  } catch (e) {
    // サーバー停止中などは無視
  }
}

document.getElementById('reloadModelsBtn').addEventListener('click', loadModels);
document.getElementById('reloadVideosBtn').addEventListener('click', loadVideos);

document.getElementById('startBtn').addEventListener('click', async () => {
  const modelPath = document.getElementById('model').value;
  const videoPath = document.getElementById('video').value;
  if (!modelPath) { alert('検証モデルを選択してください'); return; }
  if (!videoPath) { alert('検証動画を選択してください'); return; }

  const body = {
    model_path: modelPath,
    video_path: videoPath,
    device: document.getElementById('device').value,
    fps: document.getElementById('fps').value,
    loop: document.getElementById('loop').checked,
    run_integration: document.getElementById('integration').checked,
  };

  const res = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json();
    alert('起動失敗: ' + (err.error || '不明なエラー'));
  }
});

document.getElementById('stopBtn').addEventListener('click', async () => {
  await fetch('/api/stop', { method: 'POST' });
});

(async function init() {
  await Promise.all([loadModels(), loadVideos(), loadDevices()]);
  pollTimer = setInterval(pollLogsAndStatus, 1000);
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # ポーリングAPIでターミナルが埋まらないよう抑制

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/models":
            models_dir = query.get("dir", [DEFAULT_MODELS_DIR])[0]
            self._send_json(scan_models(models_dir))
            return

        if path == "/api/videos":
            mp4_dir = query.get("dir", [DEFAULT_MP4_DIR])[0]
            self._send_json(scan_videos(mp4_dir))
            return

        if path == "/api/devices":
            self._send_json(DEVICES)
            return

        if path == "/api/status":
            self._send_json(RUNNER.status())
            return

        if path == "/api/logs":
            since = int(query.get("since", ["0"])[0])
            lines, next_offset = RUNNER.logs_since(since)
            self._send_json({"lines": lines, "next": next_offset})
            return

        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path == "/api/start":
            try:
                body = self._read_json_body()
                result = RUNNER.start(
                    model_path=str(body["model_path"]),
                    video_path=str(body["video_path"]),
                    device=str(body.get("device", "cpu")),
                    fps=str(body.get("fps", "15.0")),
                    loop=bool(body.get("loop", True)),
                    run_integration=bool(body.get("run_integration", False)),
                )
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if self.path == "/api/stop":
            self._send_json(RUNNER.stop())
            return

        self._send_json({"error": "not found"}, status=404)


def main():
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", DEFAULT_PORT), Handler)
    print(f"[verification_gui] http://localhost:{DEFAULT_PORT} をホスト側ブラウザで開いてください")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
