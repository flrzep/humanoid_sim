"""Complementary (Mahony) attitude filter.

Fuses the gyroscope (good short-term, drifts long-term) with the accelerometer
(noisy short-term, but a stable long-term gravity reference) into a torso
orientation estimate. This is the *only* path by which the controller learns the
torso attitude, so the IMU's error directly shapes the estimate.

Yaw about gravity is unobservable from an accelerometer alone; that is fine here
because balancing depends on tilt (roll/pitch) and angular rate.
"""
from __future__ import annotations

import mujoco
import numpy as np


def _quat_conj(q: np.ndarray) -> np.ndarray:
    c = q.copy()
    c[1:] *= -1.0
    return c


def _rot_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (world = q applied to body vector)."""
    res = np.zeros(3)
    mujoco.mju_rotVecQuat(res, v, q)
    return res


class ComplementaryFilter:
    """Mahony-style complementary attitude filter.

    Parameters
    ----------
    kp : proportional gain pulling the estimate toward the accelerometer gravity.
    ki : integral gain that estimates and removes the gyro bias.
    accel_reject : fractional band around g over which the accelerometer is still
        trusted. When the sensed |a| departs from g (the body is accelerating),
        the gravity correction is faded out and the filter coasts on the gyro.
        This keeps a stiff, high-gain balancer from chasing dynamic accelerations.
    gravity : expected gravity magnitude (m/s^2).
    """

    UP = np.array([0.0, 0.0, 1.0])  # accelerometer specific force points "up" at rest

    def __init__(self, kp: float = 2.0, ki: float = 0.1,
                 accel_reject: float = 0.1, gravity: float = 9.81):
        self.kp = kp
        self.ki = ki
        self.accel_reject = accel_reject
        self.gravity = gravity
        self.reset()

    def reset(self, q0: np.ndarray | None = None) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else np.array(q0, float)
        self.gyro_bias = np.zeros(3)
        self.omega = np.zeros(3)

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the filter one step; return (quaternion, angular-rate estimate)."""
        accel = np.asarray(accel, float)
        gyro = np.asarray(gyro, float)

        # Accelerometer correction, faded out when |a| departs from g so that
        # dynamic (non-gravity) acceleration does not corrupt the attitude.
        norm = np.linalg.norm(accel)
        err = np.zeros(3)
        if norm > 1e-6:
            v_meas = accel / norm                       # measured up, body frame
            v_pred = _rot_vec(_quat_conj(self.q), self.UP)  # world up in body frame
            trust = np.exp(-0.5 * ((norm - self.gravity) / (self.gravity * self.accel_reject)) ** 2)
            err = trust * np.cross(v_meas, v_pred)

        # Integral term estimates the slowly-varying gyro bias.
        self.gyro_bias = self.gyro_bias - self.ki * err * dt
        omega = gyro - self.gyro_bias + self.kp * err
        self.omega = gyro - self.gyro_bias  # bias-corrected rate estimate (no P kick)

        # Integrate quaternion: q_dot = 0.5 * q (x) [0, omega].
        qdot = np.zeros(4)
        mujoco.mju_mulQuat(qdot, self.q, np.array([0.0, *omega]))
        self.q = self.q + 0.5 * qdot * dt
        mujoco.mju_normalize4(self.q)
        return self.q.copy(), self.omega.copy()

    def orientation_error(self, q_ref: np.ndarray | None = None) -> np.ndarray:
        """Rotation vector from the reference orientation to the current estimate."""
        return orientation_error(q_ref, self.q)


def orientation_error(q_ref, q) -> np.ndarray:
    """Rotation vector (body frame) from ``q_ref`` to ``q``.

    Matches MuJoCo's ``mj_differentiatePos`` convention for a free joint's
    rotational DOFs, so it can be dropped straight into the LQR state error.
    """
    q_ref = np.array([1.0, 0.0, 0.0, 0.0]) if q_ref is None else np.asarray(q_ref, float)
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, _quat_conj(q_ref), np.asarray(q, float))
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, dq, 1.0)
    return vel
