"""Composable IMU error stages and the error-injecting IMU model.

Each stage transforms a 3-vector reading and may hold internal state (a bias
random walk, a low-pass filter, a delay buffer). The stages are applied in
physical order:

    true -> scale & misalignment -> g-sensitivity (gyro) -> temperature bias
         -> bias-instability random walk -> white noise -> bandwidth low-pass
         -> quantization -> saturation -> transport delay
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import ChannelConfig, IMUModel, ImuReading
from .temperature import Constant, TemperatureProfile


@dataclass
class StageContext:
    dt: float
    t: float
    temperature: float
    rng: np.random.Generator
    accel_true: np.ndarray  # true specific force, for gyro g-sensitivity


class Stage:
    def apply(self, x: np.ndarray, ctx: StageContext) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        pass


class ScaleMisalignment(Stage):
    """Multiply by (I*(1+scale(T)) + misalignment off-diagonals)."""

    def __init__(self, scale: float, misalignment: float, tempco_scale: float, t_ref: float):
        self.scale = scale
        self.tempco_scale = tempco_scale
        self.t_ref = t_ref
        # Fixed off-diagonal cross-axis coupling (deterministic, small).
        m = misalignment
        self.offdiag = np.array([
            [0.0, m, -m],
            [-m, 0.0, m],
            [m, -m, 0.0],
        ])

    def apply(self, x, ctx):
        s = self.scale + self.tempco_scale * (ctx.temperature - self.t_ref)
        M = np.eye(3) * (1.0 + s) + self.offdiag
        return M @ x


class GSensitivity(Stage):
    """Gyro-only: add coupling proportional to specific force (g-sensitivity)."""

    def __init__(self, g_sensitivity: float):
        self.g = g_sensitivity

    def apply(self, x, ctx):
        return x + self.g * ctx.accel_true


class TemperatureBias(Stage):
    """Add bias0 + tempco_bias*(T - T_ref) on every axis."""

    def __init__(self, bias0: float, tempco_bias: float, t_ref: float):
        self.bias0 = bias0
        self.tempco_bias = tempco_bias
        self.t_ref = t_ref

    def apply(self, x, ctx):
        b = self.bias0 + self.tempco_bias * (ctx.temperature - self.t_ref)
        return x + b


class RandomWalkBias(Stage):
    """Bias-instability: bias performs a random walk and is added to the signal."""

    def __init__(self, rw_rate: float):
        self.rw_rate = rw_rate
        self.bias = np.zeros(3)

    def apply(self, x, ctx):
        self.bias = self.bias + self.rw_rate * math.sqrt(ctx.dt) * ctx.rng.standard_normal(3)
        return x + self.bias

    def reset(self):
        self.bias = np.zeros(3)


class WhiteNoise(Stage):
    """Additive white noise from a per-sqrt-Hz density (discrete std = density/sqrt(dt))."""

    def __init__(self, density: float):
        self.density = density

    def apply(self, x, ctx):
        std = self.density / math.sqrt(ctx.dt)
        return x + std * ctx.rng.standard_normal(3)


class Bandwidth(Stage):
    """First-order low-pass with cutoff fc (Hz)."""

    def __init__(self, fc: float):
        self.fc = fc
        self.y = None

    def apply(self, x, ctx):
        if self.y is None:
            self.y = x.copy()
            return self.y
        tau = 1.0 / (2 * math.pi * self.fc)
        alpha = ctx.dt / (tau + ctx.dt)
        self.y = self.y + alpha * (x - self.y)
        return self.y

    def reset(self):
        self.y = None


class Quantize(Stage):
    """Round to the nearest multiple of the LSB."""

    def __init__(self, lsb: float):
        self.lsb = lsb

    def apply(self, x, ctx):
        return np.round(x / self.lsb) * self.lsb


class Saturate(Stage):
    """Clip to +/- range."""

    def __init__(self, rng_limit: float):
        self.limit = rng_limit

    def apply(self, x, ctx):
        return np.clip(x, -self.limit, self.limit)


class Delay(Stage):
    """Transport delay of round(delay_s/dt) samples."""

    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.buf: deque | None = None
        self.n = None

    def apply(self, x, ctx):
        if self.n is None:
            self.n = max(0, round(self.delay_s / ctx.dt))
            self.buf = deque([x.copy()] * self.n, maxlen=self.n) if self.n else None
        if not self.n:
            return x
        out = self.buf[0]
        self.buf.append(x.copy())
        return out

    def reset(self):
        self.buf = None
        self.n = None


def build_channel(cfg: ChannelConfig, is_gyro: bool) -> list[Stage]:
    """Build the ordered stage list for one channel, skipping disabled stages."""
    stages: list[Stage] = []
    if cfg.scale or cfg.misalignment or cfg.tempco_scale:
        stages.append(ScaleMisalignment(cfg.scale, cfg.misalignment, cfg.tempco_scale, cfg.t_ref))
    if is_gyro and cfg.g_sensitivity:
        stages.append(GSensitivity(cfg.g_sensitivity))
    if cfg.bias0 or cfg.tempco_bias:
        stages.append(TemperatureBias(cfg.bias0, cfg.tempco_bias, cfg.t_ref))
    if cfg.bias_rw:
        stages.append(RandomWalkBias(cfg.bias_rw))
    if cfg.noise_density:
        stages.append(WhiteNoise(cfg.noise_density))
    if cfg.bandwidth_hz:
        stages.append(Bandwidth(cfg.bandwidth_hz))
    if cfg.quant_lsb:
        stages.append(Quantize(cfg.quant_lsb))
    if cfg.range:
        stages.append(Saturate(cfg.range))
    if cfg.delay_s:
        stages.append(Delay(cfg.delay_s))
    return stages


class ErrorIMU(IMUModel):
    """IMU model that applies a configurable error pipeline to accel and gyro."""

    def __init__(
        self,
        accel: ChannelConfig,
        gyro: ChannelConfig,
        temperature: TemperatureProfile | None = None,
        seed: int = 0,
        name: str = "custom",
    ):
        self.accel_cfg = accel
        self.gyro_cfg = gyro
        self.temperature = temperature or Constant()
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.accel_stages = build_channel(accel, is_gyro=False)
        self.gyro_stages = build_channel(gyro, is_gyro=True)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        for s in self.accel_stages + self.gyro_stages:
            s.reset()

    def measure(self, true_accel, true_gyro, t, dt) -> ImuReading:
        temp = self.temperature(t)
        ctx = StageContext(dt=dt, t=t, temperature=temp, rng=self.rng, accel_true=np.asarray(true_accel))
        a = np.asarray(true_accel, dtype=float)
        for s in self.accel_stages:
            a = s.apply(a, ctx)
        g = np.asarray(true_gyro, dtype=float)
        for s in self.gyro_stages:
            g = s.apply(g, ctx)
        return ImuReading(accel=a, gyro=g, temperature=temp, t=t)
