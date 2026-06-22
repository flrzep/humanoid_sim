"""Temperature profiles T(t) in degrees Celsius.

Temperature drives the temperature-dependent bias and scale errors in the IMU
model. Real MEMS IMUs self-heat after power-on and drift with ambient changes;
these profiles let us reproduce those scenarios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class TemperatureProfile:
    """Base class: callable returning temperature (deg C) at time t (s)."""

    def __call__(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Constant(TemperatureProfile):
    value: float = 25.0

    def __call__(self, t: float) -> float:
        return self.value


@dataclass
class Ramp(TemperatureProfile):
    """Linear ramp from ``start`` at ``rate`` deg/s, clamped at ``stop``."""

    start: float = 25.0
    rate: float = 0.5
    stop: float = 60.0

    def __call__(self, t: float) -> float:
        return min(self.stop, self.start + self.rate * t)


@dataclass
class SelfHeating(TemperatureProfile):
    """Exponential approach to a steady operating temperature (power-on warm-up)."""

    ambient: float = 25.0
    rise: float = 30.0      # total temperature rise at steady state
    tau: float = 120.0      # heating time constant (s)

    def __call__(self, t: float) -> float:
        return self.ambient + self.rise * (1.0 - math.exp(-t / self.tau))


@dataclass
class Sinusoid(TemperatureProfile):
    """Sinusoidal ambient variation around ``mean``."""

    mean: float = 25.0
    amplitude: float = 10.0
    period: float = 60.0

    def __call__(self, t: float) -> float:
        return self.mean + self.amplitude * math.sin(2 * math.pi * t / self.period)


@dataclass
class Step(TemperatureProfile):
    """Step from ``before`` to ``after`` at time ``t_step``."""

    before: float = 25.0
    after: float = 45.0
    t_step: float = 5.0

    def __call__(self, t: float) -> float:
        return self.after if t >= self.t_step else self.before


PROFILES = {
    "constant": Constant,
    "ramp": Ramp,
    "self_heating": SelfHeating,
    "sinusoid": Sinusoid,
    "step": Step,
}


def build_profile(spec: dict | None) -> TemperatureProfile:
    """Build a temperature profile from a ``{type: ..., **params}`` dict."""
    if not spec:
        return Constant()
    spec = dict(spec)
    kind = spec.pop("type", "constant")
    if kind not in PROFILES:
        raise ValueError(f"unknown temperature profile {kind!r}; choices: {list(PROFILES)}")
    return PROFILES[kind](**spec)
