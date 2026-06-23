"""A tiny browser demonstrator for the balancing humanoid + IMU testbed.

Inspired by the NVIDIA GEAR-SONIC web demo (watch the robot, shove it, see it
recover). That demo runs MuJoCo-WASM in the browser; here we instead run *our*
existing Python sim (LQR balancer + IMU error models + estimator) on a background
thread, render frames offscreen, and stream them to the browser as MJPEG over plain
HTTP. No async, no websockets, no extra services — and the IMU models stay in Python.

Endpoints:
    GET  /            the single-page UI
    GET  /meta        JSON: available robots, IMU presets, current selection
    GET  /stream      multipart/x-mixed-replace MJPEG video of the sim
    GET  /telemetry   JSON: tilt (true/est), temperature, status
    POST /command     JSON: {action: reset|set_imu|set_robot|push, ...}
"""
from __future__ import annotations

import io
import json
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .. import model as M
from ..control.factory import make_controller
from ..estimator.complementary import ComplementaryFilter
from ..imu import presets
from ..sim.loop import BalanceSim

RAD = 180.0 / np.pi
HERE = Path(__file__).resolve().parent
PUSH_DURATION = 0.1  # seconds a disturbance impulse is held


class SimWorker(threading.Thread):
    """Runs the balancing sim in real time and keeps the latest JPEG + telemetry."""

    def __init__(self, robot="h1", imu="ideal", width=640, height=480, fps=30, render=True):
        super().__init__(daemon=True)
        self.width, self.height, self.fps = width, height, fps
        self.render = render            # False -> stream qpos only (client renders)
        self.frame_lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self._commands: list[dict] = []
        self.latest_jpeg = b""
        self.latest_qpos: list[float] = []
        self.telemetry: dict = {}
        self.running = True

        self.robot_name = robot
        self.imu_name = imu
        self._pending_robot: str | None = None
        self._build(robot, imu)

    # --- construction -----------------------------------------------------
    def _build(self, robot: str, imu: str) -> None:
        self.hm = M.load_robot(robot)
        spec = M.ROBOTS[robot]
        self.policy_options = M.policy_options(robot)
        self.policy_name = self.policy_options[0]
        self.controller = make_controller(self.hm, robot, self.policy_name)
        self.estimator = ComplementaryFilter(**spec.get("estimator", {}))
        self.sim = BalanceSim(self.hm, self.controller, presets.build_imu(imu, randomize=True), self.estimator)
        self.base_body = int(self.hm.model.site_bodyid[self.hm.imu_site_id])
        self.controlling = True
        self.push_steps = 0
        self.push_force = np.zeros(3)
        self.push_body = self.base_body
        self.robot_name, self.imu_name = robot, imu

    def _make_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.base_body
        cam.distance = 3.2
        cam.azimuth = 120.0
        cam.elevation = -15.0
        return cam

    @staticmethod
    def _make_scene_option() -> mujoco.MjvOption:
        # Hide collision geom groups (0, 3) so each high-poly mesh is drawn once,
        # not twice (visual + collision). Roughly halves render cost on the G1/H1.
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0
        opt.geomgroup[3] = 0
        return opt

    # --- public API (thread-safe) ----------------------------------------
    def post_command(self, cmd: dict) -> None:
        with self.cmd_lock:
            self._commands.append(cmd)

    def snapshot(self) -> tuple[bytes, dict]:
        with self.frame_lock:
            return self.latest_jpeg, dict(self.telemetry)

    def snapshot_poses(self) -> tuple[list, dict]:
        """Latest generalized positions (qpos) + telemetry, for client-side rendering."""
        with self.frame_lock:
            return list(self.latest_qpos), dict(self.telemetry)

    def stop(self) -> None:
        self.running = False

    # --- main loop --------------------------------------------------------
    def run(self) -> None:
        if self.render:
            self.renderer = mujoco.Renderer(self.hm.model, self.height, self.width)
            self.camera = self._make_camera()
            self.scene_opt = self._make_scene_option()
        render_interval = 1.0 / self.fps
        wall0 = time.time()          # wall-clock origin for sim time
        next_render = wall0
        last_tm = None
        try:
            while self.running:
                if self._apply_commands():       # reset / robot change -> resync clock
                    wall0 = time.time() - self.sim.t
                dt = self.hm.dt

                # Step physics to catch up to wall-clock so the sim runs in real time
                # regardless of render cost. Bounded so a slow frame can't death-spiral.
                target = time.time() - wall0
                if target - self.sim.t > 0.5:    # too far behind: accept slowdown, resync
                    wall0 = time.time() - self.sim.t
                    target = self.sim.t
                guard, max_steps = 0, int(0.5 / dt)
                while self.sim.t < target and guard < max_steps:
                    self.hm.data.xfrc_applied[:] = 0.0
                    if self.push_steps > 0:
                        self.hm.data.xfrc_applied[self.push_body, :3] = self.push_force
                        self.push_steps -= 1
                    last_tm = self.sim.step(control=self.controlling)
                    if self.controlling and last_tm.fallen:
                        self.controlling = False
                    guard += 1

                # Publish state every iteration (cheap); render JPEG only if enabled
                # and at the capped frame rate, decoupled from the physics rate.
                if last_tm is not None:
                    self._update_state(last_tm)
                    now = time.time()
                    if self.render and now >= next_render:
                        self._render_jpeg()
                        next_render = max(now, next_render) + render_interval

                time.sleep(0.001)
        finally:
            if self.render:
                # Close the GL context on the thread that created it (avoids a crash
                # if it were left to garbage collection on another thread at exit).
                self.renderer.close()

    def _apply_commands(self) -> bool:
        """Apply queued commands. Returns True if the sim clock should resync."""
        with self.cmd_lock:
            cmds, self._commands = self._commands, []
        resync = False
        for cmd in cmds:
            action = cmd.get("action")
            if action == "reset":
                self.sim.reset()
                self.controlling = True
                self.push_steps = 0
                resync = True
            elif action == "set_imu":
                name = cmd.get("imu", self.imu_name)
                if name in presets.list_presets():
                    self.imu_name = name
                    self.sim.imu = presets.build_imu(name, randomize=True)
                    self.sim.reset()
                    self.controlling = True
                    resync = True
            elif action == "set_robot":
                name = cmd.get("robot")
                if name in M.ROBOTS and name != self.robot_name:
                    self._rebuild_robot(name)
                    resync = True
            elif action == "push":
                self._start_push(cmd.get("fx", 0.0), cmd.get("fy", 0.0),
                                 cmd.get("fz", 0.0), cmd.get("body"))
            elif action == "move":
                self.controller.set_command(cmd.get("vx", 0.0), cmd.get("vy", 0.0),
                                            cmd.get("yaw", 0.0))
            elif action == "set_policy":
                name = cmd.get("policy")
                if name in self.policy_options and name != self.policy_name:
                    self.policy_name = name
                    self.controller = make_controller(self.hm, self.robot_name, name)
                    self.sim.controller = self.controller
                    self.sim.reset()
                    self.controlling = True
                    resync = True
        return resync

    def _rebuild_robot(self, name: str) -> None:
        imu = self.imu_name
        self._build(name, imu)
        if self.render:
            # Renderer is bound to a model, so rebuild it for the new robot. Close the
            # old one here (worker thread) rather than leaving it to be GC'd elsewhere.
            self.renderer.close()
            self.renderer = mujoco.Renderer(self.hm.model, self.height, self.width)
            self.camera = self._make_camera()

    def _start_push(self, fx: float, fy: float, fz: float = 0.0, body=None) -> None:
        self.push_force = np.array([float(fx), float(fy), float(fz)])
        nbody = int(self.hm.model.nbody)
        self.push_body = self.base_body if body is None else max(1, min(int(body), nbody - 1))
        self.push_steps = max(1, round(PUSH_DURATION / self.hm.dt))

    @staticmethod
    def _vec(v, nd):
        return None if v is None else [round(float(x), nd) for x in v]

    def _update_state(self, tm) -> None:
        """Publish the latest qpos + telemetry (cheap; called every iteration)."""
        gyro_err = None if tm.imu_gyro is None else round(
            float(np.linalg.norm(tm.imu_gyro - tm.true_gyro)), 3)
        accel_err = None if tm.imu_accel is None else round(
            float(np.linalg.norm(tm.imu_accel - tm.true_accel)), 2)
        tel = {
            "t": round(tm.t, 2),
            "robot": self.robot_name,
            "imu": self.imu_name,
            "temperature": None if np.isnan(tm.temperature) else round(tm.temperature, 1),
            "true_tilt": round(float(np.linalg.norm(tm.true_tilt)) * RAD, 1),
            "est_tilt": round(float(np.linalg.norm(tm.est_tilt)) * RAD, 1),
            "tilt_err": round(tm.tilt_err * RAD, 2),
            # Signed (roll, pitch) in degrees, for the attitude chart.
            "true_rp": self._vec(tm.true_tilt * RAD, 2),
            "est_rp": self._vec(tm.est_tilt * RAD, 2),
            "status": "balancing" if self.controlling else ("fallen" if tm.fallen else "off"),
            # Raw IMU readout: ground truth vs IMU-corrupted (body frame).
            "true_gyro": self._vec(tm.true_gyro, 3),
            "imu_gyro": self._vec(tm.imu_gyro, 3),
            "true_accel": self._vec(tm.true_accel, 2),
            "imu_accel": self._vec(tm.imu_accel, 2),
            "gyro_err": gyro_err,
            "accel_err": accel_err,
        }
        qpos = self.hm.data.qpos.tolist()
        with self.frame_lock:
            self.telemetry = tel
            self.latest_qpos = qpos

    def _render_jpeg(self) -> None:
        self.renderer.update_scene(self.hm.data, camera=self.camera, scene_option=self.scene_opt)
        img = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=70)
        with self.frame_lock:
            self.latest_jpeg = buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    worker: SimWorker = None  # set by serve()

    def log_message(self, *args):  # silence per-request logging
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/meta":
            self._send_json({
                "robots": list(M.ROBOTS),
                "imus": presets.list_presets(),
                "robot": self.worker.robot_name,
                "imu": self.worker.imu_name,
            })
        elif self.path == "/telemetry":
            self._send_json(self.worker.snapshot()[1])
        elif self.path == "/stream":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while self.worker.running:
                frame, _ = self.worker.snapshot()
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                time.sleep(1.0 / self.worker.fps)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if self.path != "/command":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            cmd = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "bad json"}, code=400)
            return
        self.worker.post_command(cmd)
        self._send_json({"ok": True})


def serve(robot="g1", imu="ideal", host="127.0.0.1", port=8000,
          width=640, height=480, fps=30) -> None:
    worker = SimWorker(robot=robot, imu=imu, width=width, height=height, fps=fps)
    worker.start()
    Handler.worker = worker
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"IMU_Sim web demo running at {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        worker.join(timeout=3.0)  # let it close the GL context in-thread
        httpd.shutdown()
