"""Tests for the model equilibrium, LQR gain, and closed-loop balancing."""
import numpy as np
import pytest

from imu_sim import model as M
from imu_sim.control.lqr import LQRController, LQRCost, state_error
from imu_sim.estimator.complementary import ComplementaryFilter
from imu_sim.imu import presets
from imu_sim.sim.loop import BalanceSim


@pytest.fixture(scope="module")
def hm():
    return M.load()


def test_equilibrium_is_upright(hm):
    assert 1.0 < hm.qpos0[2] < 1.4               # reasonable standing height
    np.testing.assert_allclose(hm.qpos0[3:7], [1, 0, 0, 0], atol=1e-6)  # upright
    assert np.all(np.abs(hm.ctrl0) <= 1.0 + 1e-9)  # within motor ctrl range


def test_lqr_gain_well_formed(hm):
    ctl = LQRController(hm)
    assert ctl.K.shape == (hm.nu, 2 * hm.nv)
    assert np.all(np.isfinite(ctl.K))


def test_balances_on_true_state(hm):
    ctl = LQRController(hm)
    sim = BalanceSim(hm, ctl, presets.build_imu("ideal"), use_estimator=False)
    res = sim.run(8.0)
    assert not res["fell"]
    assert res["survived_s"] == pytest.approx(8.0)


def test_ideal_imu_matches_true_state(hm):
    ctl = LQRController(hm)
    sim = BalanceSim(hm, ctl, presets.build_imu("ideal"), ComplementaryFilter())
    res = sim.run(8.0)
    assert not res["fell"]
    assert res["rms_tilt_err"] < 0.01  # near-perfect estimate from an ideal IMU


def test_consumer_imu_is_worse_than_industrial(hm):
    ctl = LQRController(hm)
    industrial = BalanceSim(hm, ctl, presets.build_imu("industrial"), ComplementaryFilter()).run(15.0)
    consumer = BalanceSim(hm, ctl, presets.build_imu("consumer_mems"), ComplementaryFilter()).run(15.0)
    assert consumer["rms_tilt_err"] > industrial["rms_tilt_err"]


def test_h1_balances_and_imu_quality_orders_survival():
    """The real Unitree H1 balances with a clean IMU, and survival degrades with IMU grade."""
    hm = M.load_robot("h1")
    cost = LQRCost(**M.ROBOTS["h1"].get("lqr_cost", {}))
    ctl = LQRController(hm, cost)
    est = lambda: ComplementaryFilter(**M.ROBOTS["h1"]["estimator"])

    ideal = BalanceSim(hm, ctl, presets.build_imu("ideal"), est()).run(30.0)
    consumer = BalanceSim(hm, ctl, presets.build_imu("consumer_mems"), est()).run(30.0)

    assert not ideal["fell"]                              # clean IMU keeps H1 up
    assert consumer["survived_s"] < ideal["survived_s"]   # worse IMU -> shorter survival
