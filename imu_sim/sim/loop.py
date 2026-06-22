"""Closed-loop balancing simulation: IMU -> estimator -> LQR -> physics.

This is where the IMU enters the control loop. The torso orientation and angular
rate fed to the controller come *only* from the IMU + estimator; all other state
(joint encoders, base position/linear velocity) is taken as clean. So switching
IMU presets changes exactly one thing: the quality of the attitude estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..control.base import Controller
from ..estimator.complementary import ComplementaryFilter, orientation_error
from ..imu.base import IMUModel
from ..model import HumanoidModel

FALL_HEIGHT = 0.7  # torso height (m) below which we declare a fall


@dataclass
class Telemetry:
    t: float
    temperature: float
    torso_height: float
    true_tilt: np.ndarray   # (roll, pitch) ground truth, rad
    est_tilt: np.ndarray    # (roll, pitch) from estimator, rad
    tilt_err: float         # |est - true| over roll/pitch, rad
    fallen: bool
    true_accel: np.ndarray          # ground-truth accelerometer (m/s^2), body frame
    true_gyro: np.ndarray           # ground-truth gyro (rad/s), body frame
    imu_accel: np.ndarray | None    # IMU-corrupted accel; None when controller is off
    imu_gyro: np.ndarray | None     # IMU-corrupted gyro; None when controller is off


class BalanceSim:
    def __init__(
        self,
        hm: HumanoidModel,
        controller: Controller,
        imu: IMUModel,
        estimator: ComplementaryFilter | None = None,
        use_estimator: bool = True,
        calibrate_samples: int = 500,
    ):
        self.hm = hm
        self.controller = controller
        self.imu = imu
        self.estimator = estimator or ComplementaryFilter()
        self.use_estimator = use_estimator
        self.calibrate_samples = calibrate_samples
        self.q_ref = hm.qpos0[3:7].copy()
        # Declare a fall when the base drops well below its standing height.
        self.fall_height = max(0.4, 0.6 * float(hm.qpos0[2]))
        self.reset()

    def reset(self, qpos: np.ndarray | None = None, qvel: np.ndarray | None = None) -> None:
        m, d = self.hm.model, self.hm.data
        d.qpos[:] = self.hm.qpos0 if qpos is None else qpos
        d.qvel[:] = 0 if qvel is None else qvel
        d.ctrl[:] = self.hm.ctrl0
        mujoco.mj_forward(m, d)
        self.imu.reset()
        self.controller.reset()
        # Start the filter aligned with the true attitude (a real filter converges fast).
        self.estimator.reset(q0=self.hm.true_quat())
        self._calibrate_gyro_bias()
        self.t = 0.0

    def _calibrate_gyro_bias(self) -> None:
        """Power-on stationary gyro-bias calibration, as a real IMU rig performs.

        With the robot held still, the average gyro reading is its bias. Seeding the
        estimator with it removes the constant turn-on bias, so what remains to
        degrade balancing is the *uncompensated* part: noise, bias random walk, and
        especially temperature-dependent drift as the sensor self-heats.
        """
        if not self.use_estimator or self.calibrate_samples <= 0:
            return
        dt = self.hm.dt
        acc0, gyr0 = self.hm.true_accel(), self.hm.true_gyro()
        samples = [self.imu.measure(acc0, gyr0, 0.0, dt).gyro for _ in range(self.calibrate_samples)]
        self.estimator.gyro_bias = np.mean(samples, axis=0)
        self.imu.reset()  # restore deterministic state for the run proper

    def step(self, control: bool = True) -> Telemetry:
        """Advance one physics step.

        With ``control=False`` the controller (and IMU/estimator) are bypassed and
        zero torque is applied, so the robot ragdolls under gravity. The caller uses
        this to "switch the controller off" once a fall is detected.
        """
        m, d = self.hm.model, self.hm.data
        dt = self.hm.dt

        true_quat = d.qpos[3:7].copy()
        true_omega = d.qvel[3:6].copy()
        true_tilt = orientation_error(self.q_ref, true_quat)[:2]

        # Ground-truth IMU signals (reading sensors has no side effects, so always
        # capture them for the readout, even with the controller off).
        true_accel = self.hm.true_accel()
        true_gyro = self.hm.true_gyro()

        # --- IMU -> estimator -> base attitude/rate fed to the controller ---
        if control and self.use_estimator:
            reading = self.imu.measure(true_accel, true_gyro, self.t, dt)
            self.estimator.update(reading.accel, reading.gyro, dt)
            est_quat = self.estimator.q.copy()
            est_omega = self.estimator.omega.copy()
            temperature = reading.temperature
            est_tilt = orientation_error(self.q_ref, est_quat)[:2]
            imu_accel, imu_gyro = reading.accel, reading.gyro
        else:
            est_quat, est_omega = true_quat, true_omega
            temperature = float("nan")
            est_tilt = true_tilt
            imu_accel = imu_gyro = None

        # --- control + physics ---
        if control:
            self.controller.control_step(self.hm, est_quat, est_omega, self.t, dt)
        else:
            d.ctrl[:] = 0.0  # controller off: let it fall
        mujoco.mj_step(m, d)
        self.t += dt

        height = float(d.qpos[2])
        return Telemetry(
            t=self.t,
            temperature=temperature,
            torso_height=height,
            true_tilt=true_tilt,
            est_tilt=est_tilt,
            tilt_err=float(np.linalg.norm(est_tilt - true_tilt)),
            fallen=height < self.fall_height,
            true_accel=true_accel,
            true_gyro=true_gyro,
            imu_accel=imu_accel,
            imu_gyro=imu_gyro,
        )

    def run(self, duration: float) -> dict:
        """Run until ``duration`` or a fall; return a small metrics summary."""
        n = int(duration / self.hm.dt)
        tilt_errs, max_tilt = [], 0.0
        survived = duration
        for _ in range(n):
            tm = self.step()
            tilt_errs.append(tm.tilt_err)
            max_tilt = max(max_tilt, float(np.linalg.norm(tm.true_tilt)))
            if tm.fallen:
                survived = tm.t
                break
        return {
            "survived_s": survived,
            "fell": survived < duration,
            "rms_tilt_err": float(np.sqrt(np.mean(np.square(tilt_errs)))) if tilt_errs else 0.0,
            "max_true_tilt": max_tilt,
            "final_temperature": tm.temperature,
        }
