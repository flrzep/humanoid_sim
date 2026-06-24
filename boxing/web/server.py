"""Boxing game server: runs the two-fighter sim in Python, streams qpos + game
state to the browser (which renders the combined model in MuJoCo-WASM).

Endpoints:
    GET  /            the game UI
    GET  /boxing_app.js
    GET  /model/manifest        {scene, files}
    GET  /model/file?path=       arena.xml or a mesh (bytes)
    GET  /meta                   {gloves: {"0": [...geom ids], "1": [...]}}
    GET  /poses                  SSE: {qpos, tel} at the frame rate
    POST /command                move | punch | reset
"""
from __future__ import annotations

import json
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco

from ..arena import RING_DIR, ROBOT_XML
from ..game import GameSim

HERE = Path(__file__).resolve().parent


class BoxingWorker(threading.Thread):
    """Runs the match in real time; keeps the latest qpos + telemetry."""

    def __init__(self, fps: int = 50):
        super().__init__(daemon=True)
        self.fps = fps
        self.lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self._commands: list[dict] = []
        self.latest_qpos: list[float] = []
        self.telemetry: dict = {}
        self.running = True
        self.game = GameSim()

    def post_command(self, cmd: dict) -> None:
        with self.cmd_lock:
            self._commands.append(cmd)

    def snapshot_poses(self):
        with self.lock:
            return list(self.latest_qpos), dict(self.telemetry)

    def stop(self) -> None:
        self.running = False

    def _apply_commands(self) -> bool:
        with self.cmd_lock:
            cmds, self._commands = self._commands, []
        resync = False
        for c in cmds:
            action = c.get("action")
            if action == "reset":
                self.game.reset()
                resync = True
            elif action == "move":
                self.game.set_move(int(c.get("player", 0)), c.get("vx", 0.0),
                                   c.get("vy", 0.0), c.get("yaw", 0.0))
            elif action == "punch":
                self.game.punch(int(c.get("player", 0)), int(c.get("side", 1)))
        return resync

    def run(self) -> None:
        dt = self.game.arena.dt
        wall0 = time.time()
        last = None
        while self.running:
            if self._apply_commands():
                wall0 = time.time() - self.game.t
            target = time.time() - wall0
            if target - self.game.t > 0.5:        # too far behind: resync
                wall0 = time.time() - self.game.t
                target = self.game.t
            guard, max_steps = 0, int(0.5 / dt)
            while self.game.t < target and guard < max_steps:
                last = self.game.step()
                guard += 1
            if last is not None:
                with self.lock:
                    self.telemetry = last
                    self.latest_qpos = self.game.arena.data.qpos.tolist()
            time.sleep(0.001)


def _glove_geoms(model) -> dict:
    """Geom ids of each fighter's gloves, keyed by player index ("0"/"1").

    The gloves are mesh geoms whose mesh name contains "glove" (the attach prefix
    ``r1_``/``r2_`` tells us which fighter). The browser tints these per-player and
    leaves every other geom its original colour.
    """
    out = {"0": [], "1": []}
    for gi in range(model.ngeom):
        did = int(model.geom_dataid[gi])
        if did < 0:
            continue
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, did) or ""
        if "glove" not in mesh.lower():
            continue
        body = int(model.geom_bodyid[gi])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        out["0" if bname.startswith("r1_") else "1"].append(gi)
    return out


def _resolve_meshes(xml: str) -> dict:
    """Map each mesh file name referenced by the XML to a real file on disk.

    The combined XML uses bare filenames (no meshdir). Disk names may differ in case
    (e.g. ``glove_l.stl`` vs an uppercase ref), so resolve case-insensitively.
    """
    root = ET.fromstring(xml)
    wanted = {el.get("file") for el in root.iter("mesh") if el.get("file")}
    # Meshes live under the robot description dir and the boxing ring asset dir.
    by_lower: dict[str, Path] = {}
    for base in (ROBOT_XML.parent, RING_DIR):
        for p in base.rglob("*"):
            if p.suffix.lower() in (".stl", ".obj"):
                by_lower.setdefault(p.name.lower(), p)
    out = {}
    for name in wanted:
        p = by_lower.get(name.lower())
        if p is not None:
            out[name] = p
    return out


class Handler(BaseHTTPRequestHandler):
    worker: BoxingWorker = None
    arena_xml: str = ""
    meshes: dict = {}
    meta: dict = {}

    def log_message(self, *a):
        pass

    def _bytes(self, data: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        self._bytes(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/boxing.html"):
            self._bytes((HERE / "boxing.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/boxing_app.js":
            self._bytes((HERE / "boxing_app.js").read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/model/manifest":
            self._json({"scene": "arena.xml", "files": ["arena.xml"] + sorted(self.meshes)})
        elif path == "/meta":
            self._json(self.meta)
        elif path == "/model/file":
            self._model_file()
        elif path == "/poses":
            self._sse()
        else:
            self.send_error(404)

    def _model_file(self):
        from urllib.parse import parse_qs, urlparse
        rel = parse_qs(urlparse(self.path).query).get("path", [""])[0]
        if rel == "arena.xml":
            self._bytes(self.arena_xml.encode(), "application/xml")
        elif rel in self.meshes:
            self._bytes(self.meshes[rel].read_bytes(), "model/stl")
        else:
            self.send_error(404)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while self.worker.running:
                qpos, tel = self.worker.snapshot_poses()
                if qpos:
                    self.wfile.write(f"data: {json.dumps({'qpos': qpos, 'tel': tel})}\n\n".encode())
                    self.wfile.flush()
                time.sleep(1.0 / self.worker.fps)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if self.path != "/command":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            cmd = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"ok": False})
            return
        self.worker.post_command(cmd)
        self._json({"ok": True})


def serve(host="127.0.0.1", port=8002, fps=50) -> None:
    worker = BoxingWorker(fps=fps)
    Handler.worker = worker
    Handler.arena_xml = worker.game.arena.spec.to_xml()
    Handler.meshes = _resolve_meshes(Handler.arena_xml)
    Handler.meta = {"gloves": _glove_geoms(worker.game.arena.model)}
    worker.start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Boxing game at http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        httpd.shutdown()
