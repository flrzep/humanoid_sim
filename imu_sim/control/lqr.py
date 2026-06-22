"""LQR balancing controller for the humanoid.

The model is linearized about the standing equilibrium with MuJoCo's
finite-difference transition (``mjd_transitionFD``), and a discrete-time LQR gain
is solved with SciPy. The control law is ``ctrl = ctrl0 - K @ dx``.

The cost weights the horizontal centre-of-mass (so the robot keeps its mass over
its feet), joint posture, and velocities. This is the standard recipe from the
official MuJoCo LQR humanoid tutorial, adapted for a two-legged stance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from ..estimator.complementary import orientation_error
from ..model import HumanoidModel
from .base import Controller


@dataclass
class LQRCost:
    """Diagonal-ish LQR cost configuration."""

    balance: float = 1000.0        # weight on horizontal CoM excursion
    balance_joint: float = 3.0     # posture weight on leg joints (balance-relevant)
    other_joint: float = 0.3       # posture weight on all other joints
    velocity: float = 0.1          # weight on every velocity DOF
    r_scale: float = 1.0           # control penalty (R = r_scale * I)
    # Joint-name substrings treated as balance-relevant (heavier posture weight).
    # Covers both the classic humanoid (abdomen/*) and Unitree H1 (torso, *_hip_*).
    balance_joints: tuple[str, ...] = (
        "hip", "knee", "ankle", "abdomen", "torso", "waist",
    )


class LQRController(Controller):
    def __init__(self, hm: HumanoidModel, cost: LQRCost | None = None):
        self.hm = hm
        self.cost = cost or LQRCost()
        self.ctrl0 = hm.ctrl0.copy()
        self.K = self._solve_gain()

    def _solve_gain(self) -> np.ndarray:
        m, d = self.hm.model, self.hm.data
        nv, nu = m.nv, m.nu

        # Put data exactly at the equilibrium operating point.
        d.qpos[:] = self.hm.qpos0
        d.qvel[:] = 0
        d.ctrl[:] = self.ctrl0
        mujoco.mj_forward(m, d)

        # --- linearize: discrete A (2nv x 2nv), B (2nv x nu) ---------------
        A = np.zeros((2 * nv, 2 * nv))
        B = np.zeros((2 * nv, nu))
        eps = 1e-6
        mujoco.mjd_transitionFD(m, d, eps, True, A, B, None, None)

        # --- cost matrices --------------------------------------------------
        Q = self._build_Q()
        R = self.cost.r_scale * np.eye(nu)

        # --- discrete LQR ---------------------------------------------------
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return K

    def _build_Q(self) -> np.ndarray:
        m, d = self.hm.model, self.hm.data
        nv = m.nv

        # Horizontal centre-of-mass balance cost via the whole-body CoM Jacobian
        # (subtree CoM of the floating-base body = the entire robot's CoM).
        jac_com = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, d, jac_com, self.hm.base_body_id)
        Qbalance = self.cost.balance * (jac_com[:2].T @ jac_com[:2])

        # Posture cost: 0 on the free joint, heavier on balance-relevant joints.
        Qjoint = np.zeros((nv, nv))
        for jid in range(m.njnt):
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof = m.jnt_dofadr[jid]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            weight = (
                self.cost.balance_joint
                if any(s in name for s in self.cost.balance_joints)
                else self.cost.other_joint
            )
            Qjoint[dof, dof] = weight

        Qpos = Qbalance + Qjoint
        Qvel = self.cost.velocity * np.eye(nv)
        Q = np.block([
            [Qpos, np.zeros((nv, nv))],
            [np.zeros((nv, nv)), Qvel],
        ])
        return Q

    def compute(self, dx: np.ndarray) -> np.ndarray:
        ctrl = self.ctrl0 - self.K @ dx
        return ctrl

    def control_step(self, hm, est_quat, est_omega, t, dt) -> None:
        m, d = hm.model, hm.data
        nv = m.nv
        # True tangent state for everything except the base attitude/rate, which
        # come from the IMU estimate (the only IMU-affected feedback channel).
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, hm.qpos0, d.qpos)
        dx = np.concatenate([dq, d.qvel])
        dx[3:6] = orientation_error(hm.qpos0[3:7], est_quat)
        dx[nv + 3:nv + 6] = est_omega
        ctrl = self.compute(dx)
        d.ctrl[:] = np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def state_error(hm: HumanoidModel, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Tangent-space state error dx = [differentiatePos(qpos, qpos0), qvel]."""
    m = hm.model
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, hm.qpos0, qpos)
    return np.concatenate([dq, qvel])
