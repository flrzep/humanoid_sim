"""Sanity tests for IMU error stages, temperature, and presets."""
import math

import numpy as np
import pytest

from imu_sim.imu.base import ChannelConfig
from imu_sim.imu.stages import (
    Bandwidth,
    Delay,
    ErrorIMU,
    Quantize,
    RandomWalkBias,
    Saturate,
    ScaleMisalignment,
    StageContext,
    TemperatureBias,
    WhiteNoise,
)
from imu_sim.imu.temperature import Ramp, SelfHeating, build_profile
from imu_sim.imu import presets


def ctx(dt=0.01, t=0.0, temp=25.0, seed=0, accel=None):
    return StageContext(
        dt=dt, t=t, temperature=temp,
        rng=np.random.default_rng(seed),
        accel_true=np.array(accel if accel is not None else [0.0, 0.0, 9.81]),
    )


def test_ideal_passthrough():
    imu = ErrorIMU(ChannelConfig(), ChannelConfig())
    a = np.array([0.1, 0.2, 9.8])
    g = np.array([0.01, -0.02, 0.03])
    r = imu.measure(a, g, t=0.0, dt=0.01)
    np.testing.assert_allclose(r.accel, a)
    np.testing.assert_allclose(r.gyro, g)


def test_constant_bias():
    s = TemperatureBias(bias0=0.5, tempco_bias=0.0, t_ref=25.0)
    out = s.apply(np.zeros(3), ctx())
    np.testing.assert_allclose(out, 0.5)


def test_temperature_bias_scales_with_temp():
    s = TemperatureBias(bias0=0.0, tempco_bias=0.01, t_ref=25.0)
    out = s.apply(np.zeros(3), ctx(temp=45.0))
    np.testing.assert_allclose(out, 0.01 * 20.0)


def test_scale_factor():
    s = ScaleMisalignment(scale=0.1, misalignment=0.0, tempco_scale=0.0, t_ref=25.0)
    out = s.apply(np.array([1.0, 0.0, 0.0]), ctx())
    np.testing.assert_allclose(out, [1.1, 0.0, 0.0])


def test_tempco_scale():
    s = ScaleMisalignment(scale=0.0, misalignment=0.0, tempco_scale=0.01, t_ref=25.0)
    out = s.apply(np.array([1.0, 0.0, 0.0]), ctx(temp=35.0))
    np.testing.assert_allclose(out[0], 1.0 + 0.01 * 10.0)


def test_saturation():
    s = Saturate(2.0)
    out = s.apply(np.array([5.0, -5.0, 1.0]), ctx())
    np.testing.assert_allclose(out, [2.0, -2.0, 1.0])


def test_quantization():
    s = Quantize(0.5)
    out = s.apply(np.array([0.24, 0.26, 1.1]), ctx())
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_delay_holds_initial():
    s = Delay(0.03)  # 3 samples at dt=0.01
    c = ctx(dt=0.01)
    first = s.apply(np.array([1.0, 1.0, 1.0]), c)
    # buffer is pre-filled with the first sample, so output stays at it for n steps
    np.testing.assert_allclose(first, [1.0, 1.0, 1.0])
    s.apply(np.array([2.0, 2.0, 2.0]), c)
    s.apply(np.array([3.0, 3.0, 3.0]), c)
    out4 = s.apply(np.array([4.0, 4.0, 4.0]), c)  # output[k] == input[k-3] == first sample
    np.testing.assert_allclose(out4, [1.0, 1.0, 1.0])
    out5 = s.apply(np.array([5.0, 5.0, 5.0]), c)  # now the second sample emerges
    np.testing.assert_allclose(out5, [2.0, 2.0, 2.0])


def test_bandwidth_lowpass_attenuates_step_initially():
    s = Bandwidth(fc=5.0)
    c = ctx(dt=0.01)
    y0 = s.apply(np.array([1.0, 0.0, 0.0]), c)  # first call seeds the filter
    y1 = s.apply(np.array([0.0, 0.0, 0.0]), c)
    # output should lag the input change, not jump immediately to 0
    assert 0.0 < y1[0] < 1.0


def test_white_noise_density_matches():
    s = WhiteNoise(density=0.1)
    c = ctx(dt=0.01, seed=42)
    samples = np.array([s.apply(np.zeros(3), c)[0] for _ in range(20000)])
    expected_std = 0.1 / math.sqrt(0.01)
    assert abs(samples.std() - expected_std) / expected_std < 0.05


def test_random_walk_grows():
    s = RandomWalkBias(rw_rate=0.05)
    c = ctx(dt=0.01, seed=1)
    early = np.array([np.linalg.norm(s.apply(np.zeros(3), c)) for _ in range(50)]).mean()
    for _ in range(5000):
        s.apply(np.zeros(3), c)
    late = np.array([np.linalg.norm(s.apply(np.zeros(3), c)) for _ in range(50)]).mean()
    assert late > early


def test_temperature_profiles():
    assert Ramp(start=25, rate=1.0, stop=100)(10) == 35
    assert Ramp(start=25, rate=1.0, stop=30)(10) == 30  # clamped
    sh = SelfHeating(ambient=25, rise=30, tau=100)
    assert sh(0) == pytest.approx(25.0)
    assert sh(1e6) == pytest.approx(55.0, abs=0.1)
    assert build_profile(None)(123.0) == 25.0


def test_presets_load_and_build():
    names = presets.list_presets()
    assert {"ideal", "consumer_mems", "industrial"} <= set(names)
    imu = presets.build_imu("consumer_mems")
    r = imu.measure(np.array([0, 0, 9.81]), np.zeros(3), t=0.0, dt=0.005)
    # consumer IMU should not return exactly the truth
    assert not np.allclose(r.accel, [0, 0, 9.81])


def test_reset_is_repeatable():
    imu = presets.build_imu("consumer_mems", seed=7)
    a = np.array([0.0, 0.0, 9.81])
    first = [imu.measure(a, np.zeros(3), t=i * 0.005, dt=0.005).accel.copy() for i in range(10)]
    imu.reset()
    second = [imu.measure(a, np.zeros(3), t=i * 0.005, dt=0.005).accel.copy() for i in range(10)]
    for x, y in zip(first, second):
        np.testing.assert_allclose(x, y)
