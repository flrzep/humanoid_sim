"""Build the right controller backend for a registered robot."""
from __future__ import annotations

from .. import model as M
from ..model import HumanoidModel
from .base import Controller
from .lqr import LQRController, LQRCost


def make_controller(hm: HumanoidModel, robot: str) -> Controller:
    """Construct the controller specified by the robot's registry entry.

    ``control: "lqr"`` -> :class:`LQRController`; ``control: "policy"`` ->
    :class:`PolicyController` (imported lazily so torch is only required for it).
    """
    spec = M.ROBOTS[robot]
    kind = spec.get("control", "lqr")
    if kind == "policy":
        from .policy import PolicyController  # lazy: pulls in torch
        return PolicyController(hm, spec["policy"])
    return LQRController(hm, LQRCost(**spec.get("lqr_cost", {})))
