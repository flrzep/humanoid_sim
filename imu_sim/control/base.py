"""Controller interface.

A controller turns the robot state into actuator commands. The floating-base
attitude and angular rate are passed in *as estimated from the IMU* (or ground
truth in the reference case); everything else the controller reads from the clean
joint encoders in ``hm.data``. Implementations:

* ``LQRController`` (lqr.py) — analytic balance controller for the classic/H1 models.
* ``PolicyController`` (policy.py) — Unitree's pretrained RL locomotion policy (G1).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    @abstractmethod
    def control_step(self, hm, est_quat: np.ndarray, est_omega: np.ndarray,
                     t: float, dt: float) -> None:
        """Write actuator commands into ``hm.data.ctrl`` for this physics step.

        ``est_quat`` is the estimated base orientation (w, x, y, z) and
        ``est_omega`` the estimated base angular velocity (rad/s, body frame) —
        both derived from the IMU. The controller reads joint positions/velocities
        directly from ``hm.data`` (clean encoders).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal controller state."""
