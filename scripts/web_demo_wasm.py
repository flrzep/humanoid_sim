"""Launch the browser MuJoCo-WASM render demo.

    python scripts/web_demo_wasm.py                 # G1 + ideal IMU, opens a browser
    python scripts/web_demo_wasm.py --robot h1 --imu consumer_mems
    python scripts/web_demo_wasm.py --port 8080 --no-browser

Physics + IMU + control run in Python and stream qpos over SSE; the browser compiles
the same model in MuJoCo-WASM and renders it on the client GPU with three.js. Pick an
IMU, shove the robot (buttons or Shift-click), and watch it balance or fall.

Needs an internet connection on first load: three.js and the MuJoCo-WASM module are
fetched from a CDN (see the import map in imu_sim/web/wasm.html).
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imu_sim import model as M
from imu_sim.imu import presets
from imu_sim.web.wasm_server import serve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="g1", choices=list(M.ROBOTS))
    p.add_argument("--imu", default="ideal", choices=presets.list_presets())
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = p.parse_args()

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    serve(robot=args.robot, imu=args.imu, host=args.host, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
