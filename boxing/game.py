"""The 1v1 boxing match: two fighters, physics, knockdown detection.

No IMU anywhere — the fighters read ground-truth state. A round ends when a
fighter is knocked down (base height below ``KO_HEIGHT``); the other wins. Reset
starts a fresh round.
"""
from __future__ import annotations

import mujoco
import numpy as np

from imu_sim import model as M
from .arena import build_arena
from .fighter import FighterController, load_policy

KO_HEIGHT = 0.45          # base height (m) below which a fighter is "down"


class GameSim:
    def __init__(self):
        self.cfg = M.ROBOTS["g1_boxer"]["policy"]
        self.arena = build_arena()
        # Each fighter gets its OWN policy instance: the TorchScript module carries
        # internal state, so two fighters sharing one module corrupt each other (both
        # destabilise within seconds). Separate instances each balance independently.
        self.fighters = []
        for f in self.arena.fighters:
            policy, torch_mod = load_policy(self.cfg["path"])
            self._torch = torch_mod
            self.fighters.append(FighterController(policy, torch_mod, f, self.cfg))
        self.reset()

    def reset(self) -> None:
        self.arena.reset()
        for c in self.fighters:
            c.reset()
        self.t = 0.0
        self.status = "fighting"      # "fighting" | "ko"
        self.winner = None            # 0, 1, or None (draw)

    # --- input -----------------------------------------------------------
    def set_move(self, player: int, vx, vy, yaw) -> None:
        if 0 <= player < 2:
            self.fighters[player].set_move(vx, vy, yaw)

    def punch(self, player: int, side: int) -> None:
        if self.status == "fighting" and 0 <= player < 2:
            self.fighters[player].punch(side)

    # --- step ------------------------------------------------------------
    def step(self) -> dict:
        a = self.arena
        m, d = a.model, a.data
        if self.status == "fighting":
            for c in self.fighters:
                c.control_step(m, d, a.dt)
        else:
            d.ctrl[:] = 0.0           # knocked out: go limp
        mujoco.mj_step(m, d)
        self.t += a.dt

        if self.status == "fighting":
            downs = [i for i, f in enumerate(a.fighters) if a.base_height(f) < KO_HEIGHT]
            if downs:
                self.status = "ko"
                self.winner = (1 - downs[0]) if len(downs) == 1 else None
        return self.telemetry()

    def telemetry(self) -> dict:
        a = self.arena
        fs = []
        for i, (f, c) in enumerate(zip(a.fighters, self.fighters)):
            xy = a.base_xy(f)
            fs.append({
                "height": round(a.base_height(f), 2),
                "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2),
                "punching": bool(c.punching),
                "down": bool(a.base_height(f) < KO_HEIGHT),
            })
        return {"t": round(self.t, 2), "status": self.status, "winner": self.winner,
                "fighters": fs}
