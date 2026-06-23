"""Build a two-fighter boxing arena from one G1 boxer model.

MuJoCo's ``MjSpec`` lets us attach the same robot twice into one scene with name
prefixes (``r1_`` / ``r2_``), facing each other. We compile that for the Python sim
and also emit an XML + flat mesh manifest so the browser (MuJoCo-WASM) can compile the
identical model and render it.

The stock ``g1_12dof`` boxer has 12 actuated leg joints; the arms/gloves are rigid
geoms on the torso. To let the fighters punch, ``articulate_arms`` splits each arm off
into its own body with a **shoulder-pitch hinge + position servo** (the geoms are moved
onto it). The arm bodies are appended *after* the legs, so the 12 leg joints keep their
positions and the locomotion policy is untouched; the two arm joints are driven
separately for punches. Indices are looked up by name, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from imu_sim import model as M

ROBOT_XML = M.ROBOTS["g1_boxer"]["path"].parent / "g1_12dof.xml"

# Leg joints in the policy's action/observation order (left leg, then right leg).
LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
ARM_JOINTS = ["left_shoulder_punch", "right_shoulder_punch"]   # added by articulate_arms

DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float64)
STAND_HEIGHT = 0.793
START_X = 0.7

# Arm articulation: shoulder pivot (in pelvis frame), servo gains, punch reach.
SHOULDER = (0.0, 0.10, 0.292)     # (x, |y|, z)
ARM_KP, ARM_KD = 80.0, 4.0
ARM_RANGE = (-2.0, 0.4)           # rad; negative = swung forward (punch)
PUNCH_ANGLE = -1.3                # target shoulder angle at full extension


def articulate_arms(spec: mujoco.MjSpec) -> None:
    """Split each rigid arm into a hinged body with a position-servo actuator."""
    pelvis = spec.body("pelvis")
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        pivot = np.array([SHOULDER[0], sgn * SHOULDER[1], SHOULDER[2]])
        arm = pelvis.add_body(name=f"{side}_arm", pos=pivot.tolist())
        # Explicit inertial so the body has mass even though the visual geoms are
        # density 0; CoM out along the arm. Kept light + stiff so it behaves almost
        # rigidly at rest (the policy barely notices) but can swing for a punch.
        arm.explicitinertial = True
        arm.mass = 1.6
        arm.ipos = [0.06, 0.0, -0.10]
        arm.inertia = [0.02, 0.02, 0.008]
        arm.add_joint(name=f"{side}_shoulder_punch", type=mujoco.mjtJoint.mjJNT_HINGE,
                      axis=[0, 1, 0], range=list(ARM_RANGE))

        # Move this side's geoms (|y| beyond the torso) onto the arm body.
        movers = [g for g in pelvis.geoms if g.pos[1] * sgn > 0.08]
        for g in movers:
            ng = arm.add_geom()
            ng.type = g.type
            ng.size = g.size
            ng.pos = (np.array(g.pos) - pivot).tolist()
            ng.quat = g.quat
            ng.rgba = g.rgba
            ng.group = g.group
            ng.contype = g.contype
            ng.conaffinity = g.conaffinity
            ng.condim = g.condim
            ng.density = 0.0          # mass comes from the explicit inertial above
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                ng.meshname = g.meshname
            spec.delete(g)

        act = spec.add_actuator(name=f"{side}_shoulder_punch")
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = f"{side}_shoulder_punch"
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = [ARM_KP] + [0.0] * 9
        act.biasprm = [0.0, -ARM_KP, -ARM_KD] + [0.0] * 7
        act.ctrllimited = 1
        act.ctrlrange = list(ARM_RANGE)


@dataclass
class Fighter:
    prefix: str
    facing: float
    base_body: int
    qadr: int                 # free-joint qpos address
    vadr: int                 # free-joint dof address
    leg_q: np.ndarray         # 12 qpos indices (policy order)
    leg_v: np.ndarray         # 12 qvel indices
    leg_c: np.ndarray         # 12 leg actuator indices
    arm_c: np.ndarray         # 2 arm actuator indices [left, right]

    @property
    def base_quat(self):
        return slice(self.qadr + 3, self.qadr + 7)

    @property
    def base_avel(self):
        return slice(self.vadr + 3, self.vadr + 6)


@dataclass
class Arena:
    model: mujoco.MjModel
    data: mujoco.MjData
    spec: mujoco.MjSpec
    fighters: list[Fighter] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def stand_pose(self) -> np.ndarray:
        qpos = np.zeros(self.model.nq)
        for f in self.fighters:
            x = -START_X if f.facing > 0 else START_X
            quat = (1, 0, 0, 0) if f.facing > 0 else (0, 0, 0, 1)
            qpos[f.qadr:f.qadr + 3] = (x, 0.0, STAND_HEIGHT)
            qpos[f.qadr + 3:f.qadr + 7] = quat
            qpos[f.leg_q] = DEFAULT_ANGLES
        return qpos

    def reset(self) -> None:
        self.data.qpos[:] = self.stand_pose()
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def base_xy(self, f: Fighter) -> np.ndarray:
        return self.data.qpos[f.qadr:f.qadr + 2].copy()

    def base_height(self, f: Fighter) -> float:
        return float(self.data.qpos[f.qadr + 2])


def _idx(model, prefix, names, kind):
    obj = mujoco.mjtObj.mjOBJ_JOINT if kind in ("q", "v") else mujoco.mjtObj.mjOBJ_ACTUATOR
    out = []
    for n in names:
        i = mujoco.mj_name2id(model, obj, prefix + n)
        if kind == "q":
            out.append(int(model.jnt_qposadr[i]))
        elif kind == "v":
            out.append(int(model.jnt_dofadr[i]))
        else:
            out.append(i)
    return np.array(out, dtype=int)


def build_arena() -> Arena:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    world = spec.worldbody
    world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [6, 6, 0.1]
    floor.rgba = [0.27, 0.30, 0.36, 1.0]

    layout = [("r1_", +1.0, -START_X, 0.0), ("r2_", -1.0, +START_X, 180.0)]
    for prefix, _facing, x, yaw in layout:
        child = mujoco.MjSpec.from_file(str(ROBOT_XML))
        articulate_arms(child)
        frame = world.add_frame(pos=[x, 0, 0.0], euler=[0, 0, yaw])
        spec.attach(child, prefix=prefix, frame=frame)

    model = spec.compile()
    data = mujoco.MjData(model)

    fighters = []
    for prefix, facing, _x, _yaw in layout:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        fighters.append(Fighter(
            prefix=prefix, facing=facing,
            base_body=int(model.jnt_bodyid[jid]),
            qadr=int(model.jnt_qposadr[jid]),
            vadr=int(model.jnt_dofadr[jid]),
            leg_q=_idx(model, prefix, LEG_JOINTS, "q"),
            leg_v=_idx(model, prefix, LEG_JOINTS, "v"),
            leg_c=_idx(model, prefix, LEG_JOINTS, "c"),
            arm_c=_idx(model, prefix, ARM_JOINTS, "c"),
        ))

    arena = Arena(model=model, data=data, spec=spec, fighters=fighters)
    arena.reset()
    return arena
