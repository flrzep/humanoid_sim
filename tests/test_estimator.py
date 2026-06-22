"""Convergence tests for the complementary attitude filter.

Inputs are synthetic clean IMU signals (specific force = body-frame gravity, zero
rate) so we test the filter math directly, independent of the simulator.
"""
import mujoco
import numpy as np
import pytest

from imu_sim.estimator.complementary import ComplementaryFilter, _quat_conj, _rot_vec

G = 9.81


def quat(axis, angle):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * ax)])


def synth_accel(q):
    """Specific force a static, tilted IMU reads: R^T @ [0,0,g]."""
    return _rot_vec(_quat_conj(q), np.array([0.0, 0.0, G]))


def true_err(q):
    dq = np.zeros(4)
    vel = np.zeros(3)
    mujoco.mju_mulQuat(dq, _quat_conj(np.array([1.0, 0, 0, 0])), q)
    mujoco.mju_quat2Vel(vel, dq, 1.0)
    return vel


@pytest.mark.parametrize("axis,angle", [
    ([1, 0, 0], 0.15),
    ([0, 1, 0], 0.15),
    ([1, 0, 0], -0.20),
    ([0, 1, 0], -0.20),
    ([1, 0, 0], 0.40),
])
def test_converges_to_true_tilt(axis, angle):
    q = quat(axis, angle)
    acc = synth_accel(q)
    cf = ComplementaryFilter(kp=2.0, ki=0.0)
    for _ in range(3000):
        cf.update(acc, np.zeros(3), 0.005)
    np.testing.assert_allclose(cf.orientation_error(), true_err(q), atol=1e-2)


def test_gyro_integration_tracks_rotation():
    """With a constant rate and no tilt error, the rate estimate matches the gyro."""
    cf = ComplementaryFilter(kp=1.0, ki=0.0)
    gyro = np.array([0.0, 0.0, 0.3])  # yaw rate, leaves gravity reference unchanged
    acc = np.array([0.0, 0.0, G])
    for _ in range(10):
        _, omega = cf.update(acc, gyro, 0.005)
    np.testing.assert_allclose(omega, gyro, atol=1e-6)


def test_bias_estimation_removes_constant_gyro_bias():
    """A constant gyro bias on a static sensor is learned and removed from the rate."""
    cf = ComplementaryFilter(kp=2.0, ki=0.5)
    bias = np.array([0.05, -0.03, 0.0])
    acc = np.array([0.0, 0.0, G])
    for _ in range(20000):
        _, omega = cf.update(acc + 0 * bias, bias, 0.005)
    # estimated rate should converge back toward zero (true motion) despite the bias
    assert np.linalg.norm(omega) < 0.02
