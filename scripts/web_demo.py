"""Launch the browser demonstrator.

    python scripts/web_demo.py                 # H1 + ideal IMU, opens a browser
    python scripts/web_demo.py --robot classic --imu consumer_mems
    python scripts/web_demo.py --port 8080 --no-browser

Then pick an IMU in the page, shove the robot, and watch it balance or fall.
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
from imu_sim.web.server import serve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="g1", choices=list(M.ROBOTS))
    p.add_argument("--imu", default="ideal", choices=presets.list_presets())
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = p.parse_args()

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    serve(robot=args.robot, imu=args.imu, host=args.host, port=args.port,
          width=args.width, height=args.height, fps=args.fps)


if __name__ == "__main__":
    main()
