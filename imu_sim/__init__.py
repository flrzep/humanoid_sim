"""IMU_Sim: a MuJoCo testbed for evaluating IMU quality on a balancing humanoid.

The controller balances the humanoid using only an IMU-derived estimate of the
torso attitude. Swapping in different IMU error models (noise, bias drift,
temperature effects) shows how IMU quality affects balancing.
"""

__version__ = "0.1.0"
