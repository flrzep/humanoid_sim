"""Core IMU types: the measurement, the per-channel error config, and the ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImuReading:
    """A single (possibly corrupted) IMU sample."""

    accel: np.ndarray   # specific force, m/s^2, body frame
    gyro: np.ndarray    # angular velocity, rad/s, body frame
    temperature: float  # deg C
    t: float            # simulation time, s


@dataclass
class ChannelConfig:
    """Error parameters for one 3-axis channel (accelerometer or gyroscope).

    Units follow the channel: accel in m/s^2, gyro in rad/s. Noise densities are
    per-sqrt-Hz (VRW for accel, ARW for gyro). All errors default to "off" so an
    all-default channel is an ideal sensor.
    """

    # deterministic gain errors
    scale: float = 0.0              # scale-factor error (fraction, e.g. 0.01 = 1%)
    misalignment: float = 0.0       # cross-axis coupling (rad)
    tempco_scale: float = 0.0       # scale change per deg C (fraction/deg)
    # bias
    bias0: float = 0.0              # constant bias (units)
    tempco_bias: float = 0.0        # bias change per deg C (units/deg)
    bias_rw: float = 0.0            # bias instability random-walk rate (units/sqrt(s))
    # stochastic
    noise_density: float = 0.0      # white noise density (units/sqrt(Hz))
    # gyro-only: sensitivity to specific force (units per m/s^2)
    g_sensitivity: float = 0.0
    # signal conditioning
    bandwidth_hz: float = 0.0       # first-order low-pass cutoff (0 = off)
    quant_lsb: float = 0.0          # quantization step (0 = off)
    range: float = 0.0              # saturation limit, +/- (0 = off)
    delay_s: float = 0.0            # transport delay (s)

    # reference temperature for tempco terms
    t_ref: float = 25.0


class IMUModel(ABC):
    """Maps the true motion at the IMU site to a measured reading."""

    @abstractmethod
    def measure(
        self,
        true_accel: np.ndarray,
        true_gyro: np.ndarray,
        t: float,
        dt: float,
    ) -> ImuReading:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state (bias walk, filters, delay buffers)."""
