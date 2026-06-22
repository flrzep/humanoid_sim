"""Watch the humanoid balance with a chosen IMU, or evaluate it headless.

Examples
--------
    python scripts/demo_balance.py --imu ideal
    python scripts/demo_balance.py --imu consumer_mems --push 120
    python scripts/demo_balance.py --imu industrial --headless --duration 30

The 3D window shows the robot; a live status line in the console reports the true
vs estimated torso tilt, the IMU temperature, and the survival status. On exit a
one-line summary is printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imu_sim import model as M
from imu_sim.control.factory import make_controller
from imu_sim.estimator.complementary import ComplementaryFilter
from imu_sim.imu import presets
from imu_sim.sim.loop import BalanceSim

RAD = 180.0 / np.pi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="classic", choices=list(M.ROBOTS),
                   help="which humanoid to balance (default: classic; 'h1' = Unitree H1)")
    p.add_argument("--imu", default="consumer_mems",
                   help=f"IMU preset: {', '.join(presets.list_presets())} (default: consumer_mems)")
    p.add_argument("--duration", type=float, default=60.0, help="max sim seconds (default 60)")
    p.add_argument("--push", type=float, default=0.0,
                   help="lateral push force (N) applied for 0.1 s every few seconds (default 0)")
    p.add_argument("--push-interval", type=float, default=4.0, help="seconds between pushes")
    p.add_argument("--seed", type=int, default=None, help="override the preset RNG seed")
    p.add_argument("--true-state", action="store_true",
                   help="bypass the IMU and use ground-truth attitude (reference)")
    p.add_argument("--headless", action="store_true", help="no window; just run and print the summary")
    args = p.parse_args()
    return args


class Pusher:
    """Applies brief lateral impulses of constant direction on a fixed schedule."""

    def __init__(self, hm, force: float, interval: float, seed: int = 0, dwell: float = 0.1):
        self.body = hm.model.site_bodyid[hm.imu_site_id]
        self.d = hm.data
        self.force = force
        self.interval = interval
        self.dwell = dwell
        self.rng = np.random.default_rng(seed)
        self.next_push = interval
        self.active_until = -1.0
        self.dir = np.zeros(3)

    def update(self, t: float) -> None:
        if self.force <= 0:
            return
        if t >= self.next_push and t > self.active_until:
            ang = self.rng.uniform(0, 2 * np.pi)
            self.dir = np.array([np.cos(ang), np.sin(ang), 0.0])
            self.active_until = t + self.dwell
            self.next_push = t + self.interval
        self.d.xfrc_applied[self.body, :3] = self.force * self.dir if t < self.active_until else 0.0


def status_line(tm, imu_name):
    flag = "FALLEN" if tm.fallen else "balancing"
    temp = " n/a " if np.isnan(tm.temperature) else f"{tm.temperature:5.1f}"
    return (f"\r[{imu_name:13s}] t={tm.t:6.2f}s  T={temp}C  "
            f"h={tm.torso_height:4.2f}m  tilt true={np.linalg.norm(tm.true_tilt)*RAD:5.1f} "
            f"est={np.linalg.norm(tm.est_tilt)*RAD:5.1f} err={tm.tilt_err*RAD:5.2f}deg  {flag}   ")


def build_sim(args):
    hm = M.load_robot(args.robot)
    spec = M.ROBOTS[args.robot]
    controller = make_controller(hm, args.robot)
    imu = presets.build_imu(args.imu, seed=args.seed)
    estimator = ComplementaryFilter(**spec.get("estimator", {}))
    sim = BalanceSim(hm, controller, imu, estimator, use_estimator=not args.true_state)
    return hm, sim


def run_headless(hm, sim, args):
    pusher = Pusher(hm, args.push, args.push_interval)
    n = int(args.duration / hm.dt)
    last_tm = None
    for _ in range(n):
        pusher.update(sim.t)
        last_tm = sim.step()
        if last_tm.fallen:
            break
    print(status_line(last_tm, args.imu))
    return last_tm


def run_viewer(hm, sim, args):
    import mujoco
    import mujoco.viewer

    pusher = Pusher(hm, args.push, args.push_interval)
    reset_request = {"do": False}

    def key_callback(keycode):
        # 'R' resets the simulation for another test (GLFW letter codes are ASCII).
        if keycode == ord("R"):
            reset_request["do"] = True

    print("Interactive disturbance: double-click a body to select it, then hold Ctrl "
          "and drag with the RIGHT mouse button to shove it (Ctrl + LEFT drag to twist). "
          "Release to let go. Press F1 for the full control list.")
    print("On a fall the controller switches OFF and the robot drops. Press R to reset "
          "and balance again.")

    with mujoco.viewer.launch_passive(hm.model, hm.data, key_callback=key_callback) as viewer:
        start = time.time()
        last_print = 0.0
        controlling = True
        last_tm = None
        while viewer.is_running() and sim.t < args.duration:
            if reset_request["do"]:
                sim.reset()
                pusher = Pusher(hm, args.push, args.push_interval)
                controlling = True
                reset_request["do"] = False
                start = time.time()
                last_print = 0.0
                sys.stdout.write("\n" + " " * 100 + "\rSimulation reset. Balancing again.\n")

            # Clear external forces each step (the perturb apply does not clear on
            # release), then apply the optional scripted push and the user's mouse drag.
            hm.data.xfrc_applied[:] = 0.0
            pusher.update(sim.t)
            mujoco.mjv_applyPerturbForce(hm.model, hm.data, viewer.perturb)

            last_tm = sim.step(control=controlling)
            viewer.sync()

            if controlling and last_tm.fallen:
                # Failure: stop the controller, let the robot fall, wait for reset.
                controlling = False
                sys.stdout.write(status_line(last_tm, args.imu) + "\n")
                print("Robot fell — controller OFF. Press R to reset, or close the window to exit.")
                last_print = last_tm.t
            elif last_tm.t - last_print > 0.1:
                sys.stdout.write(status_line(last_tm, args.imu))
                sys.stdout.flush()
                last_print = last_tm.t

            # real-time pacing (sim.t restarts at 0 after a reset, so does `start`)
            lag = sim.t - (time.time() - start)
            if lag > 0:
                time.sleep(lag)
        return last_tm


def main():
    args = parse_args()
    hm, sim = build_sim(args)
    label = "true-state (no IMU)" if args.true_state else f"IMU preset '{args.imu}'"
    print(f"Balancing '{args.robot}' with {label}. Pushes: "
          f"{f'{args.push:.0f} N every {args.push_interval:.0f}s' if args.push > 0 else 'none'}.")

    tm = run_headless(hm, sim, args) if args.headless else run_viewer(hm, sim, args)

    print()
    if tm is not None:
        verdict = f"FELL after {tm.t:.2f}s" if tm.fallen else f"stayed up for {tm.t:.2f}s"
        temp = "n/a" if np.isnan(tm.temperature) else f"{tm.temperature:.1f} C"
        print(f"Summary [{args.imu}{' / true-state' if args.true_state else ''}]: {verdict}; "
              f"final tilt-est error {tm.tilt_err * RAD:.2f} deg, temperature {temp}")


if __name__ == "__main__":
    main()
