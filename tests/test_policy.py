"""Tests for the Unitree RL policy controller on the G1.

Skipped automatically if torch or the vendored policy checkpoint is unavailable.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imu_sim import model as M
from imu_sim.control.factory import make_controller
from imu_sim.control.policy import gravity_orientation
from imu_sim.estimator.complementary import ComplementaryFilter
from imu_sim.imu import presets
from imu_sim.sim.loop import BalanceSim

_HAS_G1 = M.ROBOTS["g1"]["policy"]["path"].exists()
pytestmark = pytest.mark.skipif(not _HAS_G1, reason="G1 policy checkpoint not vendored")


def test_gravity_orientation_upright():
    # Upright base: gravity points straight down in the body frame.
    np.testing.assert_allclose(gravity_orientation(np.array([1.0, 0, 0, 0])), [0, 0, -1], atol=1e-9)


def test_g1_loads_as_policy_robot():
    hm = M.load_robot("g1")
    assert hm.nu == 12
    np.testing.assert_allclose(hm.ctrl0, 0.0)  # policy robot: no LQR equilibrium control


def test_g1_policy_balances_and_imu_matters():
    hm = M.load_robot("g1")
    est = M.ROBOTS["g1"]["estimator"]
    ideal = BalanceSim(hm, make_controller(hm, "g1"),
                       presets.build_imu("ideal"), ComplementaryFilter(**est)).run(8.0)
    consumer = BalanceSim(hm, make_controller(hm, "g1"),
                          presets.build_imu("consumer_mems"), ComplementaryFilter(**est)).run(8.0)
    assert not ideal["fell"]                                   # policy balances with a clean IMU
    assert consumer["rms_tilt_err"] >= ideal["rms_tilt_err"]   # worse IMU -> worse attitude estimate
