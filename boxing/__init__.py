"""1v1 boxing game built on the humanoid_sim foundation.

Two Unitree G1 boxers in one MuJoCo scene, each driven by the pretrained
locomotion policy (fed ground truth — no IMU here). Players move with WASD /
gamepad and trigger scripted lunge "punches"; a knockdown (a fighter falling)
ends the round. Shares only the model assets and policy weights with the IMU
testbed; none of the IMU simulation or its UI is used.
"""
