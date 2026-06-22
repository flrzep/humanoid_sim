"""Build IMU models from YAML presets."""
from __future__ import annotations

from pathlib import Path

import yaml

from .base import ChannelConfig
from .stages import ErrorIMU
from .temperature import TemperatureProfile, build_profile

DEFAULT_PRESETS = Path(__file__).resolve().parent.parent.parent / "configs" / "imu_presets.yaml"


def load_presets(path: str | Path = DEFAULT_PRESETS) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_presets(path: str | Path = DEFAULT_PRESETS) -> list[str]:
    return list(load_presets(path).keys())


def build_imu(
    name: str,
    path: str | Path = DEFAULT_PRESETS,
    seed: int | None = None,
    temperature: TemperatureProfile | None = None,
) -> ErrorIMU:
    """Construct an :class:`ErrorIMU` from a named preset.

    ``seed`` and ``temperature`` override the preset's defaults when given.
    """
    presets = load_presets(path)
    if name not in presets:
        raise KeyError(f"unknown IMU preset {name!r}; choices: {list(presets)}")
    spec = presets[name]
    accel = ChannelConfig(**(spec.get("accel") or {}))
    gyro = ChannelConfig(**(spec.get("gyro") or {}))
    temp = temperature if temperature is not None else build_profile(spec.get("temperature"))
    seed = spec.get("seed", 0) if seed is None else seed
    return ErrorIMU(accel=accel, gyro=gyro, temperature=temp, seed=seed, name=name)
