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
from pathlib import Path

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
# Arm joints added by articulate_arms, in actuator order (left arm, then right arm).
ARM_JOINTS = ["left_shoulder_punch", "left_elbow_punch",
              "right_shoulder_punch", "right_elbow_punch"]

DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float64)
STAND_HEIGHT = 0.793
START_X = 0.7

# --- Arm articulation tunables (tweak these to change how a punch looks/reaches) ---
SHOULDER = (0.0, 0.10, 0.292)      # shoulder pivot in pelvis frame (x, |y|, z)
ELBOW = (0.0158, 0.1468, 0.1052)   # elbow pivot in pelvis frame (x, |y|, z)
ARM_KP, ARM_KD = 80.0, 4.0         # position-servo gains for both arm joints
ARM_RANGE = (-2.4, 1.3)            # rad joint limits; negative = swung forward
SHOULDER_PUNCH_ANGLE = -1.5        # shoulder target at full extension
ELBOW_PUNCH_ANGLE = 1.1           # elbow target at full extension (forearm snap)
# Geoms (by mesh name) that belong to the forearm; everything else on that side is
# upper arm. The glove lives on the forearm so the elbow extension carries it forward.
_FOREARM_MESHES = {"left": {"left_elbow_link", "glove_l"},
                   "right": {"right_elbow_link", "glove_r"}}

# --- Ring tunables ---------------------------------------------------------------
# The ring's *looks* come from a mesh you control: edit boxing/assets/ring.stl,
# regenerate it (scripts/make_ring_placeholder.py), or drop in your own STL. The mesh
# is COSMETIC by default — MuJoCo collides meshes by their convex hull, so a hollow
# ring would turn into a solid block you couldn't stand inside. Containment instead
# comes from four invisible primitive "rope walls" sized by RING_HALF below. Set
# RING_MESH_COLLIDE = True only if your STL is convex (e.g. a flat platform).
RING_DIR = Path(__file__).resolve().parent / "assets"
RING_STL = "ring.stl"                  # mesh file inside RING_DIR (swap freely)
RING_POS = (0.0, 0.0, 0.0)             # ring origin in the world (x, y, z)
RING_SCALE = 1.0                       # uniform scale applied to the mesh
RING_RGBA = (0.85, 0.80, 0.52, 1.0)    # mesh colour (browser tints the whole mesh)
RING_MESH_COLLIDE = False              # make the mesh itself a collider (convex hull!)

RING_WALLS = True                     # build the collision boundary (rope walls)
RING_HALF = 2.45                      # half-width of the square fighting area (m)
WALL_HEIGHT = 0.55                     # rope-wall height (m)
WALL_THICK = 0.04                      # rope-wall thickness (m)


def _rgba(c) -> str:
    return " ".join(f"{v:g}" for v in c)


def add_ring(spec: mujoco.MjSpec) -> None:
    """Attach the ring: a cosmetic mesh + (optionally) primitive collision walls.

    Built as its own child spec (like the robots) so the mesh resolves to a bare
    filename with no global ``meshdir`` — which keeps the exported XML loadable in the
    browser. The ring body is static (no joint), so it adds nothing to the robots' DOFs.
    """
    geoms = [
        f'<geom type="mesh" mesh="ring" group="2" '
        f'contype="{int(RING_MESH_COLLIDE)}" conaffinity="{int(RING_MESH_COLLIDE)}" '
        f'pos="0 0 0" rgba="{_rgba(RING_RGBA)}"/>'
    ]
    if RING_WALLS:
        h, t, zc = RING_HALF, WALL_THICK, WALL_HEIGHT / 2
        # Four boundary walls (collision only, group 3 = hidden in the browser).
        for sx, sy, hx, hy in ((0, h, h, t), (0, -h, h, t), (h, 0, t, h), (-h, 0, t, h)):
            geoms.append(
                f'<geom type="box" size="{hx:g} {hy:g} {zc:g}" pos="{sx:g} {sy:g} {zc:g}" '
                f'group="3" contype="1" conaffinity="1" condim="3"/>'
            )
    meshdir = str(RING_DIR).replace("\\", "/")
    xml = (
        f'<mujoco model="ring">\n'
        f'  <compiler meshdir="{meshdir}"/>\n'
        f'  <asset><mesh name="ring" file="{RING_STL}" '
        f'scale="{RING_SCALE:g} {RING_SCALE:g} {RING_SCALE:g}"/></asset>\n'
        f'  <worldbody><body name="ring">\n    ' + "\n    ".join(geoms) +
        f'\n  </body></worldbody>\n</mujoco>'
    )
    ring = mujoco.MjSpec.from_string(xml)
    frame = spec.worldbody.add_frame(pos=list(RING_POS))
    spec.attach(ring, prefix="ring_", frame=frame)


def _add_servo(spec, name):
    act = spec.add_actuator(name=name)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.target = name
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm = [ARM_KP] + [0.0] * 9
    act.biasprm = [0.0, -ARM_KP, -ARM_KD] + [0.0] * 7
    act.ctrllimited = 1
    act.ctrlrange = list(ARM_RANGE)


def _move_geom(g, body, pivot, spec):
    ng = body.add_geom()
    ng.type, ng.size, ng.quat, ng.rgba = g.type, g.size, g.quat, g.rgba
    ng.group, ng.contype, ng.conaffinity, ng.condim = g.group, g.contype, g.conaffinity, g.condim
    ng.pos = (np.array(g.pos) - pivot).tolist()
    ng.density = 0.0               # mass comes from each body's explicit inertial
    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
        ng.meshname = g.meshname
    spec.delete(g)


def articulate_arms(spec: mujoco.MjSpec) -> None:
    """Split each rigid arm into a hinged upper arm + forearm (shoulder + elbow).

    Both joints are stiff position servos so the arm is near-rigid at rest (the policy
    barely notices); a punch drives the shoulder *and* elbow forward so the forearm
    snaps out, giving the glove extra reach and forward momentum.
    """
    pelvis = spec.body("pelvis")
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        shoulder = np.array([SHOULDER[0], sgn * SHOULDER[1], SHOULDER[2]])
        elbow = np.array([ELBOW[0], sgn * ELBOW[1], ELBOW[2]])

        upper = pelvis.add_body(name=f"{side}_arm", pos=shoulder.tolist())
        upper.explicitinertial = True
        upper.mass = 1.0
        upper.ipos = [0.0, 0.0, -0.09]
        upper.inertia = [0.012, 0.012, 0.005]
        upper.add_joint(name=f"{side}_shoulder_punch", type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=[0, 1, 0], range=list(ARM_RANGE))

        fore = upper.add_body(name=f"{side}_forearm", pos=(elbow - shoulder).tolist())
        fore.explicitinertial = True
        fore.mass = 0.7
        fore.ipos = [0.07, 0.0, -0.03]
        fore.inertia = [0.006, 0.006, 0.003]
        fore.add_joint(name=f"{side}_elbow_punch", type=mujoco.mjtJoint.mjJNT_HINGE,
                       axis=[0, 1, 0], range=list(ARM_RANGE))

        for g in [g for g in pelvis.geoms if g.pos[1] * sgn > 0.08]:
            on_forearm = (g.type == mujoco.mjtGeom.mjGEOM_MESH
                          and g.meshname in _FOREARM_MESHES[side])
            if on_forearm:
                _move_geom(g, fore, elbow, spec)
            else:
                _move_geom(g, upper, shoulder, spec)

        _add_servo(spec, f"{side}_shoulder_punch")
        _add_servo(spec, f"{side}_elbow_punch")


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

    add_ring(spec)

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
