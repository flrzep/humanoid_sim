"""Build a two-fighter boxing arena from one G1 boxer model.

MuJoCo's ``MjSpec`` lets us attach the same robot twice into one scene with name
prefixes (``r1_`` / ``r2_``), facing each other. We compile that for the Python
sim and also emit an XML + flat mesh manifest so the browser (MuJoCo-WASM) can
compile the identical model and render it.

The boxer (``g1_12dof``) has 12 actuated leg joints per robot; the arms/gloves are
rigid. So each fighter is a free base (7 qpos / 6 qvel) + 12 leg joints, laid out
contiguously: robot ``k`` owns ``qpos[k*19:(k+1)*19]``, ``qvel[k*18:(k+1)*18]`` and
``ctrl[k*12:(k+1)*12]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = (_ROOT / "models" / "unitree_rl_gym" / "resources" / "robots"
             / "g1_boxer_description" / "g1_12dof.xml")

# Standing leg pose (matches the locomotion policy's default_angles), in joint order
# left[hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll] then right[...].
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float64)
STAND_HEIGHT = 0.793
START_X = 0.7              # each fighter this far from centre (1.4 m apart)
NJ = 12                   # actuated joints per fighter
NQ_R = 7 + NJ             # qpos per fighter (free base + joints)
NV_R = 6 + NJ             # qvel per fighter


@dataclass
class Fighter:
    """Index bookkeeping for one robot inside the combined model."""
    prefix: str
    qadr: int             # free-joint qpos address (base x is qadr, height qadr+2)
    vadr: int             # free-joint dof address
    ctrl0: int            # first actuator index
    base_body: int        # body id of the floating base (for pushes / pose read)
    facing: float         # +1 faces +x, -1 faces -x

    @property
    def jpos(self):       # slice of the 12 leg joint angles
        return slice(self.qadr + 7, self.qadr + 7 + NJ)

    @property
    def jvel(self):
        return slice(self.vadr + 6, self.vadr + 6 + NJ)

    @property
    def ctrl(self):
        return slice(self.ctrl0, self.ctrl0 + NJ)

    @property
    def base_quat(self):  # (w,x,y,z) qpos slice
        return slice(self.qadr + 3, self.qadr + 7)

    @property
    def base_avel(self):  # base angular velocity (world) dof slice
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
        """Full qpos placing both fighters upright, facing each other."""
        qpos = np.zeros(self.model.nq)
        for f in self.fighters:
            x = -START_X if f.facing > 0 else START_X
            quat = (1, 0, 0, 0) if f.facing > 0 else (0, 0, 0, 1)  # 180° about z
            qpos[f.qadr:f.qadr + 3] = (x, 0.0, STAND_HEIGHT)
            qpos[f.qadr + 3:f.qadr + 7] = quat
            qpos[f.jpos] = DEFAULT_ANGLES
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


def build_arena() -> Arena:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    world = spec.worldbody
    world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [6, 6, 0.1]
    floor.rgba = [0.27, 0.30, 0.36, 1.0]

    layout = [("r1_", +1, -START_X, 0.0), ("r2_", -1, +START_X, 180.0)]
    for prefix, _facing, x, yaw in layout:
        child = mujoco.MjSpec.from_file(str(ROBOT_XML))
        frame = world.add_frame(pos=[x, 0, 0.0], euler=[0, 0, yaw])
        spec.attach(child, prefix=prefix, frame=frame)

    model = spec.compile()
    data = mujoco.MjData(model)

    fighters = []
    for prefix, facing, _x, _yaw in layout:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + "left_hip_pitch_joint")
        fighters.append(Fighter(
            prefix=prefix,
            qadr=int(model.jnt_qposadr[jid]),
            vadr=int(model.jnt_dofadr[jid]),
            ctrl0=int(aid),
            base_body=int(model.jnt_bodyid[jid]),
            facing=float(facing),
        ))

    arena = Arena(model=model, data=data, spec=spec, fighters=fighters)
    arena.reset()
    return arena


def arena_xml(spec: mujoco.MjSpec) -> str:
    """The combined model as XML, for the browser to compile in MuJoCo-WASM."""
    return spec.to_xml()
