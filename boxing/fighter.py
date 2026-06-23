"""Per-fighter controller: Unitree locomotion policy on a robot slice + scripted punch.

Reuses the pretrained G1 locomotion policy (the same TorchScript checkpoint as the
IMU demo) but reads the base orientation/rate as **ground truth** — there is no IMU
in the boxing game. The policy balances and walks; a "punch" is a scripted committed
lunge toward the opponent (the boxer model's arms are rigid, so the punch is a
whole-body charge — left/right pick the lead/strafe side). When articulated arms are
added later, ``punch`` can drive an arm trajectory instead without touching the rest.
"""
from __future__ import annotations

import numpy as np

from imu_sim.control.policy import gravity_orientation
from .arena import Fighter

# Lunge "punch": a short, hard forward command toward the opponent.
PUNCH_DURATION = 0.35      # seconds
LUNGE_VX = 0.8             # forward speed during a punch (m/s, robot frame +x = opponent)
LUNGE_VY = 0.35           # lateral lead for left/right punches (m/s)


def load_policy(path):
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    policy = torch.jit.load(str(path))
    policy.eval()
    return policy, torch


class FighterController:
    def __init__(self, policy, torch_mod, fighter: Fighter, cfg: dict):
        self.policy = policy
        self._torch = torch_mod
        self.f = fighter
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.num_obs = cfg["num_obs"]
        self.num_actions = cfg["num_actions"]
        self.decimation = cfg["decimation"]
        self.gait_period = cfg.get("gait_period", 0.8)
        self._dt = 0.002                                # updated each control_step
        self.move_cmd = np.zeros(3, dtype=np.float32)   # player walk command
        self.punch_steps = 0
        self.punch_side = 0
        self.reset()

    def reset(self) -> None:
        self.counter = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target = self.default.copy()
        self.cmd = np.zeros(3, dtype=np.float32)
        self.move_cmd = np.zeros(3, dtype=np.float32)
        self.punch_steps = 0

    def set_move(self, vx, vy, yaw) -> None:
        self.move_cmd = np.clip(np.array([vx, vy, yaw], dtype=np.float32),
                                [-0.8, -0.5, -0.8], [0.8, 0.5, 0.8])

    def punch(self, side: int) -> None:
        """Trigger a lunge punch. side: -1 right, +1 left."""
        self.punch_steps = max(self.punch_steps, int(round(PUNCH_DURATION / self._dt)))
        self.punch_side = 1 if side >= 0 else -1

    @property
    def punching(self) -> bool:
        return self.punch_steps > 0

    def _effective_cmd(self) -> np.ndarray:
        if self.punch_steps > 0:
            self.punch_steps -= 1
            return np.array([LUNGE_VX, self.punch_side * LUNGE_VY, 0.0], dtype=np.float32)
        return self.move_cmd

    def _obs(self, data) -> np.ndarray:
        f = self.f
        quat = data.qpos[f.base_quat]                 # ground-truth base orientation
        omega = data.qvel[f.base_avel]                # body-frame angular velocity
        qj = (data.qpos[f.jpos] - self.default) * self.dof_pos_scale
        dqj = data.qvel[f.jvel] * self.dof_vel_scale
        grav = gravity_orientation(np.asarray(quat, dtype=np.float32))

        phase = (self.counter * self._dt % self.gait_period) / self.gait_period
        sin_cos = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

        n = self.num_actions
        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3] = np.asarray(omega, dtype=np.float32) * self.ang_vel_scale
        obs[3:6] = grav
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9:9 + n] = qj
        obs[9 + n:9 + 2 * n] = dqj
        obs[9 + 2 * n:9 + 3 * n] = self.action
        obs[9 + 3 * n:9 + 3 * n + 2] = sin_cos
        return obs

    def control_step(self, model, data, dt) -> None:
        self._dt = dt
        self.counter += 1
        self.cmd = self._effective_cmd()
        if self.counter % self.decimation == 0:
            obs = self._obs(data)
            with self._torch.no_grad():
                out = self.policy(self._torch.from_numpy(obs).unsqueeze(0))
            self.action = out.detach().numpy().squeeze()
            self.target = self.action * self.action_scale + self.default

        f = self.f
        qj = data.qpos[f.jpos]
        dqj = data.qvel[f.jvel]
        data.ctrl[f.ctrl] = (self.target - qj) * self.kps - dqj * self.kds
