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

KO_HEIGHT = 0.35          # base height (m) below which a fighter is "down"
# A punch lands only on real glove->opponent contact (see _resolve_punches). The arm
# swing itself shoves ~350 N, which the policy shrugs off, so a landed glove also deals
# a knockback impulse — the "damage". Set HIT_FORCE = 0 for pure-physics shoving (no KO).
HIT_FORCE = 800           # knockback force on a clean glove contact (N) — ~1-punch KO = 1800
HIT_LIFT = 400.0          # upward component (N)
HIT_STEPS = 18            # steps the knockback impulse is applied (~0.036 s)


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
        self._build_geom_sets()
        self.reset()

    def _build_geom_sets(self) -> None:
        """Geom ids that strike (each arm) and that can be struck (whole robot)."""
        m = self.arena.model
        def body(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        self._strike = []   # per fighter: {+1: left-arm geoms, -1: right-arm geoms}
        self._victim = []   # per fighter: all of that robot's geoms
        for f in self.arena.fighters:
            # Each arm strikes with its upper arm + forearm (the forearm holds the glove).
            lbodies = {body(f.prefix + "left_arm"), body(f.prefix + "left_forearm")}
            rbodies = {body(f.prefix + "right_arm"), body(f.prefix + "right_forearm")}
            left, right, allg = set(), set(), set()
            for gi in range(m.ngeom):
                b = int(m.geom_bodyid[gi])
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if b in lbodies:
                    left.add(gi)
                if b in rbodies:
                    right.add(gi)
                if bn.startswith(f.prefix):
                    allg.add(gi)
            self._strike.append({1: left, -1: right})
            self._victim.append(allg)

    def reset(self) -> None:
        self.arena.reset()
        for c in self.fighters:
            c.reset()
        self.t = 0.0
        self.status = "fighting"      # "fighting" | "ko"
        self.winner = None            # 0, 1, or None (draw)
        self._impact = [np.zeros(3), np.zeros(3)]   # active knockback force per fighter
        self._impact_steps = [0, 0]
        self._hit = [False, False]    # did each fighter get hit since last telemetry

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
        d.xfrc_applied[:] = 0.0
        if self.status == "fighting":
            for c in self.fighters:
                c.control_step(m, d, a.dt)
            self._resolve_punches()
        else:
            d.ctrl[:] = 0.0           # knocked out: go limp
        # apply any active knockback impulses to the fighters' bases
        for j, f in enumerate(a.fighters):
            if self._impact_steps[j] > 0:
                d.xfrc_applied[f.base_body, :3] = self._impact[j]
                self._impact_steps[j] -= 1
        mujoco.mj_step(m, d)
        self.t += a.dt

        if self.status == "fighting":
            downs = [i for i, f in enumerate(a.fighters) if a.base_height(f) < KO_HEIGHT]
            if downs:
                self.status = "ko"
                self.winner = (1 - downs[0]) if len(downs) == 1 else None
        return self.telemetry()

    def _resolve_punches(self) -> None:
        """A punch lands when the *active glove* physically contacts the opponent.

        The arm swing is real physics (it already shoves the opponent ~350 N), but
        that alone can't topple the robust policy — so a confirmed glove contact also
        deals a knockback impulse (HIT_FORCE) as the "hit". You must actually land the
        correct glove on the opponent, not just be near them.
        """
        a, d = self.arena, self.arena.data
        for i, c in enumerate(self.fighters):
            if not (c.punching and not c.landed):
                continue
            strike = self._strike[i][1 if c.punch_side > 0 else -1]
            victim = self._victim[1 - i]
            if not self._contact_between(d, strike, victim):
                continue
            j = 1 - i
            c.landed = True
            self._hit[j] = True
            delta = a.base_xy(a.fighters[j]) - a.base_xy(a.fighters[i])
            dist = float(np.linalg.norm(delta))
            dirv = delta / dist if dist > 1e-6 else np.array([a.fighters[i].facing, 0.0])
            self._impact[j] = np.array([dirv[0] * HIT_FORCE, dirv[1] * HIT_FORCE, HIT_LIFT])
            self._impact_steps[j] = HIT_STEPS

    @staticmethod
    def _contact_between(d, geoms_a, geoms_b) -> bool:
        for ci in range(d.ncon):
            g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
            if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
                return True
        return False

    def telemetry(self) -> dict:
        a = self.arena
        fs = []
        for i, (f, c) in enumerate(zip(a.fighters, self.fighters)):
            xy = a.base_xy(f)
            fs.append({
                "height": round(a.base_height(f), 2),
                "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2),
                "punching": bool(c.punching),
                "hit": bool(self._hit[i]),
                "down": bool(a.base_height(f) < KO_HEIGHT),
            })
        self._hit = [False, False]
        return {"t": round(self.t, 2), "status": self.status, "winner": self.winner,
                "fighters": fs}
