"""Model loading, sensor access, and standing-equilibrium helpers.

Everything that is specific to *how a humanoid XML is wired* lives here, so that
swapping one robot for another (classic humanoid <-> Unitree H1) only touches this
module. New robots are added to ``ROBOTS`` and must expose an ``imu`` site with
``imu_accel``/``imu_gyro``/``imu_quat`` sensors and a standing keyframe.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_MODELS = Path(__file__).resolve().parent.parent / "models"

# Registry of supported robots: name -> (xml path, standing keyframe, imu site).
ROBOTS = {
    "classic": {
        "path": _MODELS / "humanoid.xml",
        "keyframe": "stand",
        "imu_site": "imu",
        # No lean needed: the all-zero stance already has the CoM near the ankles.
        "lean_joints": (),
    },
    "h1": {
        "path": _MODELS / "menagerie" / "unitree_h1" / "scene_imusim.xml",
        "keyframe": "stand",
        "imu_site": "imu_pelvis",
        "control": "lqr",
        "lean_joints": ("left_ankle", "right_ankle"),
        # H1 is heavy and tall; needs more velocity damping and stiffer posture/
        # balance weighting than the classic model to keep a usable stability margin.
        "lqr_cost": {"velocity": 5.0, "balance": 5000.0,
                     "balance_joint": 50.0, "other_joint": 5.0},
        # Its stiff, high-gain loop can't chase the accelerometer, so trust the
        # gyro more (low kp) and reject dynamic acceleration aggressively.
        "estimator": {"kp": 0.4, "ki": 0.05, "accel_reject": 0.08},
    },
    "g1": {
        # Unitree G1 (12-DOF legs) driven by Unitree's pretrained RL locomotion policy.
        "path": _MODELS / "unitree_rl_gym" / "resources" / "robots" / "g1_description" / "scene_imusim.xml",
        "keyframe": "stand",
        "imu_site": "imu_pelvis",
        "control": "policy",   # no LQR equilibrium; the policy balances actively
        "estimator": {"kp": 0.5, "ki": 0.05, "accel_reject": 0.1},
        "policy": {
            "path": _MODELS / "unitree_rl_gym" / "pre_train" / "g1" / "motion.pt",
            "kps": [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
            "kds": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
            "default_angles": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                               -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
            "ang_vel_scale": 0.25, "dof_pos_scale": 1.0, "dof_vel_scale": 0.05,
            "action_scale": 0.25, "cmd_scale": [2.0, 2.0, 0.25],
            "cmd_init": [0.0, 0.0, 0.0],   # stand in place
            "num_obs": 47, "num_actions": 12, "decimation": 10, "gait_period": 0.8,
        },
    },
    "g1_boxer": {
        # Unitree G1 (12-DOF legs) driven by Unitree's pretrained RL locomotion policy.
        "path": _MODELS / "unitree_rl_gym" / "resources" / "robots" / "g1_boxer_description" / "scene_imusim.xml",
        "keyframe": "stand",
        "imu_site": "imu_pelvis",
        "control": "policy",   # no LQR equilibrium; the policy balances actively
        "estimator": {"kp": 0.5, "ki": 0.05, "accel_reject": 0.1},
        "policy": {
            "path": _MODELS / "unitree_rl_gym" / "pre_train" / "g1" / "motion.pt",
            "kps": [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
            "kds": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
            "default_angles": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                               -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
            "ang_vel_scale": 0.25, "dof_pos_scale": 1.0, "dof_vel_scale": 0.05,
            "action_scale": 0.25, "cmd_scale": [2.0, 2.0, 0.25],
            "cmd_init": [0.0, 0.0, 0.0],   # stand in place
            "num_obs": 47, "num_actions": 12, "decimation": 10, "gait_period": 0.8,
        },
    }
}

# Default model shipped with the repo.
DEFAULT_MODEL = ROBOTS["classic"]["path"]


def policy_options(robot: str) -> list[str]:
    """Display names of the controllers/policies available for a robot.

    A robot spec may carry a ``policies`` dict {name: policy_cfg} for swappable RL
    policies; otherwise there is a single backend (the bundled locomotion policy, or
    the LQR balancer). This is the extension point: drop another ``.pt`` in, add an
    entry to ``policies``, and it appears in the UI switcher. (No alternative drop-in
    G1 policy is currently available to download — others use a different obs spec.)
    """
    spec = ROBOTS[robot]
    pols = spec.get("policies")
    if pols:
        return list(pols)
    return ["Unitree RL (locomotion)"] if spec.get("control") == "policy" else ["LQR balancer"]


@dataclass
class HumanoidModel:
    """A loaded MuJoCo humanoid plus cached indices and an equilibrium."""

    model: mujoco.MjModel
    data: mujoco.MjData
    qpos0: np.ndarray          # equilibrium configuration (nq,)
    ctrl0: np.ndarray          # equilibrium control that holds qpos0 (nu,)
    imu_site_id: int
    base_body_id: int          # the floating-base body (root of the kinematic tree)

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nu(self) -> int:
        return self.model.nu

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    # --- sensor access -----------------------------------------------------
    def sensor(self, name: str) -> np.ndarray:
        """Return the current value of a named sensor as a copy."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise KeyError(f"no sensor named {name!r}")
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return np.array(self.data.sensordata[adr : adr + dim])

    def true_accel(self) -> np.ndarray:
        """Ground-truth specific force at the IMU site (m/s^2), body frame."""
        return self.sensor("imu_accel")

    def true_gyro(self) -> np.ndarray:
        """Ground-truth angular velocity at the IMU site (rad/s), body frame."""
        return self.sensor("imu_gyro")

    def true_quat(self) -> np.ndarray:
        """Ground-truth orientation of the IMU site (w, x, y, z)."""
        return self.sensor("imu_quat")

    def torso_height(self) -> float:
        return float(self.data.qpos[2])


def load(
    path: str | Path = DEFAULT_MODEL,
    keyframe: str = "stand",
    imu_site: str = "imu",
    lean_joints: tuple[str, ...] = (),
    compute_equilibrium: bool = True,
) -> HumanoidModel:
    """Load a humanoid from an explicit XML path.

    For LQR robots an exact standing equilibrium (qpos0, ctrl0) is computed. For
    policy-driven robots (``compute_equilibrium=False``) the keyframe pose is used
    as the nominal stance and ctrl0 is zero — the policy balances actively.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if compute_equilibrium:
        qpos0, ctrl0 = find_standing_equilibrium(model, data, keyframe, lean_joints)
    else:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise KeyError(f"no keyframe named {keyframe!r}")
        qpos0 = np.array(model.key_qpos[key_id])
        ctrl0 = np.zeros(model.nu)
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site)
    if imu_site_id < 0:
        raise KeyError(f"no IMU site named {imu_site!r}")
    base_body_id = _find_base_body(model)
    # Leave data sitting at the nominal pose, ready to simulate.
    data.qpos[:] = qpos0
    data.qvel[:] = 0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)
    return HumanoidModel(model, data, qpos0, ctrl0, imu_site_id, base_body_id)


def load_robot(name: str = "classic") -> HumanoidModel:
    """Load a registered robot (see :data:`ROBOTS`) by name."""
    if name not in ROBOTS:
        raise KeyError(f"unknown robot {name!r}; choices: {list(ROBOTS)}")
    spec = ROBOTS[name]
    return load(
        spec["path"],
        keyframe=spec["keyframe"],
        imu_site=spec["imu_site"],
        lean_joints=spec.get("lean_joints", ()),
        compute_equilibrium=spec.get("control", "lqr") != "policy",
    )


def _find_base_body(model: mujoco.MjModel) -> int:
    """Return the body carrying the free joint (root of the floating-base tree)."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j])
    raise ValueError("model has no free joint; expected a floating-base humanoid")


def find_standing_equilibrium(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    keyframe: str = "stand",
    lean_joints: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Find a static standing equilibrium (qpos0, ctrl0) for the given keyframe.

    Two residuals from inverse dynamics must vanish for a genuine fixed point:

    * the **vertical** root force, zeroed by bisecting the torso height so the
      contact forces exactly support the robot's weight;
    * the **pitch** root moment, zeroed (when ``lean_joints`` such as the ankle
      pitch joints are given) by leaning those joints to bring the centre of mass
      over the feet.

    Without the pitch correction a heavy robot like the H1 has no holding control
    and topples immediately. The control that holds the pose is then recovered
    from inverse dynamics.
    """
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise KeyError(f"no keyframe named {keyframe!r}")

    base_h = float(model.key_qpos[key_id][2])
    lean_adr = []
    for name in lean_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            lean_adr.append(model.jnt_qposadr[jid])

    def root_residual(lean: float, height: float) -> np.ndarray:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        for adr in lean_adr:
            data.qpos[adr] += lean
        data.qpos[2] = height
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        return np.array(data.qfrc_inverse)

    def height_for_vertical_zero(lean: float) -> float:
        # Bracket: high -> floats (residual = +weight), low -> penetrates (negative).
        # Residual increases with height (high = floating = +weight, low =
        # penetrating = negative), so a positive residual means go lower.
        lo, hi = base_h - 0.12, base_h + 0.12
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if root_residual(lean, mid)[2] > 0:
                hi = mid          # unsupported, go lower
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # Outer solve: lean the ankles until the pitch root moment is zero.
    lean = 0.0
    height = height_for_vertical_zero(lean)
    if lean_adr:
        def pitch(lean: float) -> tuple[float, float]:
            h = height_for_vertical_zero(lean)
            return float(root_residual(lean, h)[4]), h

        a0, a1 = -0.3, 0.3
        p0, _ = pitch(a0)
        p1, h1 = pitch(a1)
        for _ in range(60):
            if abs(p1) < 1e-3 or p1 == p0:
                break
            a2 = a1 - p1 * (a1 - a0) / (p1 - p0)
            p2, h1 = pitch(a2)
            a0, p0, a1, p1 = a1, p1, a2, p2
        lean, height = a1, h1

    qfrc0 = root_residual(lean, height)
    qpos0 = np.array(data.qpos)

    # Map required generalized forces to actuator commands (least squares).
    moment = actuator_moment_dense(model, data)  # (nu, nv)
    ctrl0 = qfrc0 @ np.linalg.pinv(moment)

    return qpos0, ctrl0


def actuator_moment_dense(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense (nu, nv) actuator moment matrix.

    MuJoCo stores ``actuator_moment`` sparsely; expand it with the official helper.
    """
    dense = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        dense,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    return dense
