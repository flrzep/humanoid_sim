"""Drop-in RL locomotion controller using Unitree's pretrained policy.

Wraps the TorchScript policy from ``unitree_rl_gym`` (G1) behind the same
``Controller`` interface as the LQR. The observation it builds mirrors Unitree's
``deploy_mujoco.py`` exactly, with one change that is the whole point of this repo:
the two IMU-derived observation channels — base angular velocity (``obs[0:3]``,
the gyro) and projected gravity (``obs[3:6]``, the orientation) — are taken from
our IMU model + estimator, not from ground truth. Everything else (joint
positions/velocities) comes from clean encoders. So a worse IMU degrades a real,
push-recovering RL policy the same way it would on hardware.

The policy outputs joint position targets at 50 Hz; a PD law converts them to the
motor torques applied every physics step.
"""
from __future__ import annotations

import numpy as np

from ..model import HumanoidModel
from .base import Controller


def gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Gravity direction in the base frame from a (w,x,y,z) quaternion.

    Identical to Unitree's ``get_gravity_orientation`` in deploy_mujoco.py.
    """
    qw, qx, qy, qz = quat
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


class PolicyController(Controller):
    def __init__(self, hm: HumanoidModel, cfg: dict):
        import torch  # lazy: only needed for policy-driven robots

        # A 47-dim MLP runs fastest single-threaded; the default thread pool just
        # adds per-call overhead and oversubscribes the CPU the sim/render need.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self._torch = torch
        self.policy = torch.jit.load(str(cfg["path"]))
        self.policy.eval()

        self.nu = hm.nu
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.cmd = np.array(cfg.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.num_obs = cfg["num_obs"]
        self.num_actions = cfg["num_actions"]
        self.decimation = cfg["decimation"]
        self.gait_period = cfg.get("gait_period", 0.8)
        self.reset()

    def reset(self) -> None:
        self.counter = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target = self.default_angles.copy()

    def _build_obs(self, hm, est_quat, est_omega) -> np.ndarray:
        d = hm.data
        qj = (d.qpos[7:7 + self.num_actions] - self.default_angles) * self.dof_pos_scale
        dqj = d.qvel[6:6 + self.num_actions] * self.dof_vel_scale
        omega = np.asarray(est_omega, dtype=np.float32) * self.ang_vel_scale   # IMU gyro
        grav = gravity_orientation(np.asarray(est_quat, dtype=np.float32))     # IMU attitude

        count = self.counter * hm.dt
        phase = (count % self.gait_period) / self.gait_period
        sin_cos = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

        obs = np.zeros(self.num_obs, dtype=np.float32)
        n = self.num_actions
        obs[0:3] = omega
        obs[3:6] = grav
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9:9 + n] = qj
        obs[9 + n:9 + 2 * n] = dqj
        obs[9 + 2 * n:9 + 3 * n] = self.action
        obs[9 + 3 * n:9 + 3 * n + 2] = sin_cos
        return obs

    def control_step(self, hm, est_quat, est_omega, t, dt) -> None:
        self.counter += 1
        if self.counter % self.decimation == 0:
            obs = self._build_obs(hm, est_quat, est_omega)
            with self._torch.no_grad():
                out = self.policy(self._torch.from_numpy(obs).unsqueeze(0))
            self.action = out.detach().numpy().squeeze()
            self.target = self.action * self.action_scale + self.default_angles

        # PD -> motor torque, every physics step
        data = hm.data
        qj = data.qpos[7:7 + self.num_actions]
        dqj = data.qvel[6:6 + self.num_actions]
        data.ctrl[:] = (self.target - qj) * self.kps - dqj * self.kds
