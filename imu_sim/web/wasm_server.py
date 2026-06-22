"""Server for the browser (MuJoCo-WASM) demo.

Unlike ``server.py`` (which renders frames and streams MJPEG), this server does
**not** render at all. It runs the physics + IMU + control sim in Python and
streams the generalized positions (``qpos``) to the browser, where MuJoCo-WASM +
three.js render on the client GPU. It also serves the model XML + mesh files so the
browser can compile the same model in its WASM virtual filesystem.

Endpoints:
    GET  /            the WASM single-page UI
    GET  /wasm_app.js the client module
    GET  /meta        robots, IMU presets, current selection
    GET  /model/manifest?robot=  files the browser must load into MEMFS
    GET  /model/file?robot=&path=  one model file (XML or mesh), bytes
    GET  /poses       Server-Sent Events: {qpos, telemetry} at the frame rate
    POST /command     reset | set_imu | set_robot | push
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import model as M
from ..imu import presets
from .server import SimWorker

HERE = Path(__file__).resolve().parent


def collect_model_files(robot: str) -> tuple[Path, list[str]]:
    """Return (root_dir, [relative paths]) of every file the model needs.

    Follows ``<include>`` recursively and gathers ``<mesh>``/``<texture>``/
    ``<hfield>`` asset files (resolving ``<compiler meshdir/texturedir>``). All of
    our models keep their assets under the scene file's directory, which becomes the
    MEMFS root in the browser.
    """
    scene = Path(M.ROBOTS[robot]["path"]).resolve()
    root = scene.parent
    needed: set[Path] = set()

    def visit(xml_path: Path) -> None:
        xml_path = xml_path.resolve()
        if xml_path in needed or not xml_path.exists():
            if xml_path not in needed:
                return
        needed.add(xml_path)
        tree = ET.parse(xml_path)
        node = tree.getroot()
        base = xml_path.parent
        comp = node.find("compiler")
        meshdir = (comp.get("meshdir") if comp is not None else None) or "."
        texdir = (comp.get("texturedir") if comp is not None else None) or "."
        for inc in node.iter("include"):
            visit(base / inc.get("file"))
        for tag, d in (("mesh", meshdir), ("texture", texdir), ("hfield", meshdir)):
            for el in node.iter(tag):
                f = el.get("file")
                if f:
                    p = (base / d / f).resolve()
                    if p.exists():
                        needed.add(p)

    visit(scene)
    rels = sorted(str(p.relative_to(root)).replace("\\", "/") for p in needed)
    return root, rels


class Handler(BaseHTTPRequestHandler):
    worker: SimWorker = None

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html", "/wasm.html"):
            self._file(HERE / "wasm.html", "text/html; charset=utf-8")
        elif url.path == "/wasm_app.js":
            self._file(HERE / "wasm_app.js", "text/javascript; charset=utf-8")
        elif url.path == "/meta":
            self._json({"robots": list(M.ROBOTS), "imus": presets.list_presets(),
                        "robot": self.worker.robot_name, "imu": self.worker.imu_name})
        elif url.path == "/model/manifest":
            robot = q.get("robot", [self.worker.robot_name])[0]
            _, files = collect_model_files(robot)
            scene = Path(M.ROBOTS[robot]["path"]).name
            self._json({"robot": robot, "scene": scene, "files": files})
        elif url.path == "/model/file":
            self._model_file(q)
        elif url.path == "/poses":
            self._sse()
        else:
            self.send_error(404)

    def _model_file(self, q):
        robot = q.get("robot", [self.worker.robot_name])[0]
        rel = q.get("path", [""])[0]
        root, files = collect_model_files(robot)
        if rel not in files:                      # whitelist guards path traversal
            self.send_error(404)
            return
        ctype = "model/stl" if rel.lower().endswith(".stl") else "application/octet-stream"
        if rel.lower().endswith(".xml"):
            ctype = "application/xml"
        self._file((root / rel), ctype)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while self.worker.running:
                qpos, tel = self.worker.snapshot_poses()
                if qpos:
                    payload = json.dumps({"qpos": qpos, "tel": tel})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                time.sleep(1.0 / self.worker.fps)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if urlparse(self.path).path != "/command":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            cmd = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"ok": False}, code=400)
            return
        self.worker.post_command(cmd)
        self._json({"ok": True})


def serve(robot="g1", imu="ideal", host="127.0.0.1", port=8001, fps=50) -> None:
    worker = SimWorker(robot=robot, imu=imu, fps=fps, render=False)
    worker.start()
    Handler.worker = worker
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"IMU_Sim WASM demo running at http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        httpd.shutdown()
