"""Launch the 1v1 G1 boxing game (browser, MuJoCo-WASM render).

    python scripts/boxing_demo.py                 # opens a browser
    python scripts/boxing_demo.py --port 8080 --no-browser

Player 1: WASD/QE move, F/G punch. Player 2: arrows + ,/. + K/L. Or plug in PS4
controllers (gamepad i -> player i): left stick moves, L1/R1 punch, Options rematch.
Knock the other fighter down to win the round; first internet load fetches three.js
and MuJoCo-WASM from a CDN.
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxing.web.server import serve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = p.parse_args()

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    serve(host=args.host, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
