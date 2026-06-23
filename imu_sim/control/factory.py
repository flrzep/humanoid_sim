"""Build the right controller backend for a registered robot."""
from __future__ import annotations

from .. import model as M
from ..model import HumanoidModel
from .base import Controller
from .lqr import LQRController, LQRCost


def make_controller(hm: HumanoidModel, robot: str, policy: str | None = None) -> Controller:
    """Construct the controller specified by the robot's registry entry.

    ``control: "lqr"`` -> :class:`LQRController`; ``control: "policy"`` ->
    :class:`PolicyController` (imported lazily so torch is only required for it).
    ``policy`` selects a named entry from the spec's ``policies`` map when present
    (the runtime policy switcher); otherwise the single bundled policy is used.
    """
    spec = M.ROBOTS[robot]
    kind = spec.get("control", "lqr")
    if kind == "policy":
        from .policy import PolicyController  # lazy: pulls in torch
        pols = spec.get("policies")
        cfg = pols[policy] if (pols and policy in pols) else spec["policy"]
        return PolicyController(hm, cfg)
    return LQRController(hm, LQRCost(**spec.get("lqr_cost", {})))
