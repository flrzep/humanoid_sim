# IMU_Sim — Humanoid Balancing IMU Evaluation (MVP)

A MuJoCo testbed for evaluating how **IMU quality affects a balancing humanoid**.

A humanoid balances under an LQR controller. The catch: the controller never sees the
true torso orientation. It only sees what a simulated IMU reports, fused by an attitude
estimator. Swap in a different IMU error model — noise, bias drift, temperature effects
— and watch the robot wobble more, or fall.

Three robots are supported (`--robot`), with two controller backends:
- **`classic`** — 27-DOF MuJoCo humanoid, **LQR** balancer (light, forgiving).
- **`h1`** — **Unitree H1** (MuJoCo Menagerie), **LQR** balancer. Heavy and IMU-sensitive.
- **`g1`** — **Unitree G1** driven by **Unitree's pretrained RL locomotion policy**
  (`unitree_rl_gym`). A real, push-recovering whole-body policy — the most realistic
  setting for judging how much the IMU actually matters.

The controller backend is per-robot (`imu_sim/control/factory.py`); both consume the
torso attitude + angular rate **only** through the IMU + estimator, so the IMU is the
single variable under test regardless of controller.

```
 MuJoCo physics (ground truth)
        │ true accel + gyro at torso site
        ▼
   IMU error model  ──►  corrupted accel + gyro
        │                       │
        │            complementary filter (estimator)
        │                       │  torso tilt + angular rate
        ▼                       ▼
   clean joint encoders ──► LQR controller ──► motor torques ──► (loop)
```

Only the torso **attitude** channel is corrupted by the IMU; joint encoders and base
position are treated as clean. So changing the IMU preset changes exactly one thing:
the quality of the attitude estimate the controller relies on.

## Install

Requires Python 3.10+ (developed on 3.14). All model assets and policy weights are in
the repo, and the browser demos fetch three.js / mujoco-js from a CDN, so there is
nothing else to download.

**Windows (one step):** from a clean machine, just run the setup script — it finds a
Python 3.10+, builds `.venv`, installs everything, and smoke-tests the install.

```powershell
.\setup.ps1            # or double-click setup.bat
.\setup.ps1 -Recreate  # rebuild the venv from scratch
```

**Manual (any OS):**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

## Run

```bash
# Watch the classic humanoid balance with different IMUs (3D window + live status line):
python scripts/demo_balance.py --imu ideal
python scripts/demo_balance.py --imu consumer_mems

# The real Unitree H1 (LQR):
python scripts/demo_balance.py --robot h1 --imu ideal

# The Unitree G1 driven by Unitree's pretrained RL policy (needs torch):
python scripts/demo_balance.py --robot g1 --imu ideal
python scripts/demo_balance.py --robot g1 --imu consumer_mems

# Headless evaluation (no window), prints a one-line verdict:
python scripts/demo_balance.py --robot h1 --imu consumer_mems --headless --duration 120

# Reference run that bypasses the IMU and uses ground-truth attitude:
python scripts/demo_balance.py --robot h1 --imu ideal --headless --true-state

# Optional: scripted periodic lateral pushes:
python scripts/demo_balance.py --imu industrial --push 100
```

**Interactive disturbance (in the 3D window):** double-click a body to select it, then
hold **Ctrl** and **right-drag** to shove it (Ctrl + left-drag to twist). Release to let
go. This applies a real external force the IMU must ride through — a good way to feel how
recovery degrades with a worse IMU. Press **F1** for the full viewer control list.

### Browser demo

A web demonstrator (inspired by NVIDIA's GEAR-SONIC demo) lets you watch the robot, pick
an IMU, and shove it — all from a browser, with the IMU selector in the UI:

```bash
python scripts/web_demo.py                 # Unitree H1 + ideal IMU, opens a browser
python scripts/web_demo.py --robot classic --imu consumer_mems
python scripts/web_demo.py --port 8080 --no-browser
```

It runs *our* Python sim (balancer + IMU error models + estimator) on a background
thread, renders frames offscreen, and streams them to the page as MJPEG over plain HTTP
(no WASM, no websockets). The page is responsive (the view scales to the window) and
offers a **robot** and **IMU** dropdown, a **reset** button (or press **R**), directional
+ random **push** buttons with a force slider, click-on-view to shove, and a live
telemetry panel (true vs estimated tilt, IMU temperature, status). On a fall the
controller switches off and the robot drops; reset to try again.

**Performance:** the physics + control loop is locked to wall-clock (catch-up stepping),
so the simulation always runs in **real time**; only the video frame rate degrades if
rendering is slow. Render speed is dominated by offscreen OpenGL on the CPU here — the G1's
high-poly meshes cost the most, so collision geoms are hidden (each mesh drawn once) and
torch is pinned to one thread. Expect ~real-time physics with ~20–30 fps video depending
on the machine's GL. (NVIDIA's demo renders on the client GPU via MuJoCo-WASM, which is
why it's smoother; moving rendering client-side, e.g. three.js, is the path to higher fps.)

### Browser demo — client-side render (MuJoCo-WASM)

A second browser demo moves *rendering* to the client GPU, like NVIDIA's demo:

```bash
python scripts/web_demo_wasm.py            # Unitree G1 + ideal IMU, opens a browser
python scripts/web_demo_wasm.py --robot h1 --imu consumer_mems
python scripts/web_demo_wasm.py --port 8080 --no-browser
```

Physics + IMU + control still run in **Python** (so the IMU study is unchanged); the server
streams generalized positions (`qpos`) over Server-Sent Events instead of video. The browser
compiles the **same model** in [MuJoCo-WASM](https://www.npmjs.com/package/mujoco-js)
(`mujoco-js`, the official Google DeepMind WebAssembly build, Apache-2.0), writes each
streamed `qpos` into `mjData`, runs `mj_forward`, and draws the resulting world geom
transforms with [three.js](https://threejs.org/). The WASM module is used purely as a
forward-kinematics + geometry source — no physics runs in the browser — so the render matches
the Python sim exactly. Same UI as the MJPEG demo (robot/IMU dropdowns, reset, push pad +
slider, telemetry); orbit with drag, zoom with scroll, **Shift-click** the robot to shove it.

`three.js` and `mujoco-js` load from a CDN (import map in
[`imu_sim/web/wasm.html`](imu_sim/web/wasm.html)), so the first load needs internet. Because
the GPU does the drawing, frame rate no longer depends on the server's OpenGL. This is also
the reusable foundation for moving physics fully client-side later (MuJoCo-WASM can step, not
just `mj_forward`).

## Learned RL policy (Unitree G1)

For a realistic, push-recovering controller, the `g1` robot is driven by **Unitree's
pretrained RL locomotion policy** (`unitree_rl_gym`, a TorchScript checkpoint) — no
training required. It's wired behind the same `Controller` interface as the LQR
(`imu_sim/control/policy.py`), reproducing Unitree's `deploy_mujoco` observation exactly,
with one deliberate change: the two **IMU-derived** observation channels —
`obs[0:3]` (base angular velocity = gyro) and `obs[3:6]` (projected gravity = orientation)
— are fed from our IMU model + estimator instead of ground truth. Joint pos/vel come from
clean encoders. So a worse IMU degrades a *real deployed RL policy* the way it would on
hardware.

**On realism / how much the IMU matters:** "IMU importance" is a property of the *control
stack*, not the robot alone — a brittle standing LQR falls from a tiny gyro bias; a robust
RL policy tolerates much more (and so does real hardware). This testbed gives trustworthy
**relative** answers (consumer vs industrial vs ideal), **sensitivity trends** (vs noise
density / temperature drift), and **plausible thresholds** (largest push survived per IMU
grade). Absolute "good enough for robot X" numbers require matching X's actual estimator,
control rate, and policy training.

> Why not NVIDIA's GEAR-SONIC policy itself? It's a motion-*tracking* policy for the G1
> (trained in Isaac Lab, expects a reference-motion stream + SMPL encoders) under a
> non-commercial license — heavy to integrate for a balance study. Unitree's locomotion
> policy gives the same realism win (a real push-recovering RL controller consuming
> IMU-derived observations) with a clean flat observation and a permissive BSD-3 license.

> Note: `torch.jit.load` warns as deprecated on Python 3.14 but currently works.

### Representative results (quiet standing, 120 s)

**Unitree H1** (real robot — sensitive):

| IMU preset      | outcome          | tilt-est error | end temp |
|-----------------|------------------|----------------|----------|
| `ideal`         | stands 120 s     | ~0°            | 25 °C    |
| `industrial`    | **falls ~20 s**  | ~0.9°          | 26.8 °C  |
| `consumer_mems` | **falls ~5 s**   | ~1.7°          | 27.0 °C  |

**Classic humanoid** (light — forgiving):

| IMU preset      | outcome          | tilt-est error | end temp |
|-----------------|------------------|----------------|----------|
| `ideal`         | stands 120 s     | ~0°            | 25 °C    |
| `industrial`    | stands 120 s     | ~0.2°          | 32.6 °C  |
| `consumer_mems` | **falls ~110 s** | ~21°           | 49.7 °C  |

Better IMU → longer balance, on both robots. The **H1 is dramatically more sensitive**:
a clean IMU holds it indefinitely, but a consumer MEMS sensor topples it within seconds.
This is realistic — a heavy humanoid balancing on ankle torque has little stability
margin, so IMU noise and (uncompensated) temperature/bias drift in the attitude estimate
quickly become fatal. The estimator tracks the true tilt closely right up to the fall;
what kills balance is the small, persistent error the controller cannot reject.

To make the comparison fair, the sim performs a **power-on stationary gyro-bias
calibration** (averaging the gyro while the robot is held still), exactly as a real IMU
rig does. This removes each sensor's *constant* turn-on bias, leaving noise, bias random
walk, and **temperature-dependent drift** as the differentiators — so as the IMU
self-heats during the run, a high-tempco consumer sensor drifts while a low-tempco
industrial one holds.

## IMU error model

IMU presets live in [`configs/imu_presets.yaml`](configs/imu_presets.yaml) and are fully
configurable. Each channel (accelerometer, gyroscope) passes through a pipeline of
error stages, applied in physical order:

```
true → scale & misalignment → g-sensitivity (gyro) → temperature bias
     → bias-instability random walk → white noise → bandwidth low-pass
     → quantization → saturation → transport delay
```

Temperature is driven by a profile T(t) (constant, ramp, self-heating, sinusoid, step);
bias and scale shift with temperature via tempco terms. Add a new preset by copying a
block in the YAML and editing the numbers.

## Layout

```
models/humanoid.xml          classic humanoid + torso IMU site/sensors + standing keyframe
models/menagerie/unitree_h1/ Unitree H1 (MuJoCo Menagerie) + scene_imusim.xml wrapper
models/unitree_rl_gym/       Unitree G1 model + meshes + pretrained policy (pre_train/g1/motion.pt)
configs/imu_presets.yaml     ideal / consumer_mems / industrial presets
imu_sim/
  model.py                  ROBOTS registry, sensor access, standing-equilibrium finder
  control/factory.py        picks LQR vs policy controller per robot
  control/lqr.py            linearize (mjd_transitionFD) + discrete LQR (SciPy)
  control/policy.py         Unitree RL policy controller (TorchScript + IMU-injected obs)
  imu/                       error stages, temperature profiles, preset loader
  estimator/complementary.py complementary (Mahony) attitude filter
  sim/loop.py               closed-loop IMU→estimator→controller→physics + bias calibration
  web/server.py             background sim worker + MJPEG HTTP server for the browser demo
  web/index.html            single-page UI (robot/IMU selectors, push controls, telemetry)
  web/wasm_server.py        SSE qpos stream + model/mesh serving for the client-render demo
  web/wasm.html             MuJoCo-WASM + three.js single-page UI (client-side render)
  web/wasm_app.js           browser client: compile model in WASM, mj_forward, draw with three.js
scripts/demo_balance.py     native viewer entry point (--robot, viewer + headless)
scripts/web_demo.py         browser demo entry point (MJPEG / server render)
scripts/web_demo_wasm.py    browser demo entry point (MuJoCo-WASM / client render)
tests/                      IMU stages, estimator, LQR balancing, G1 policy
```

### Adding another robot

A robot is one entry in `ROBOTS` in [imu_sim/model.py](imu_sim/model.py): the XML path, a
standing `keyframe`, the `imu_site` name (mount it on the floating-base body so the
measured attitude matches the controller's state), optional `lean_joints` (ankle pitch
joints, used to put the CoM over the feet for a true equilibrium), and optional per-robot
`lqr_cost` / `estimator` gains. The model must expose `imu_accel`/`imu_gyro`/`imu_quat`
sensors at that site. Position-actuated models (e.g. Unitree G1) would need a
position-control variant of the LQR and are not supported yet.

Run the tests with `python -m pytest`.

## Scope

Done so far: classic + Unitree H1 (LQR) and Unitree G1 (pretrained RL policy), full IMU
error suite, complementary-filter estimator, native + browser demos with disturbances.
Deliberately **not** included yet: training our own policy, a statistical episode/sweep
harness with plots (e.g. "largest push survived per IMU grade"), EKF estimators,
integral/LQI control for the LQR robots, and the NVIDIA GEAR-SONIC motion-tracking policy.
The `Controller`/`IMUModel` interfaces and the `ROBOTS` registry are the seams for these.

## Attribution

`models/humanoid.xml` is derived from the MuJoCo humanoid model by DeepMind
Technologies, licensed under Apache 2.0, with an IMU site, sensors, and a standing
keyframe added.

`models/menagerie/unitree_h1/` is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Unitree H1, BSD-3-Clause; see its `LICENSE`). The only local change is a pelvis IMU
site added to `h1.xml`, plus the new `scene_imusim.xml` wrapper (sensors + standing
keyframe). Obtained via a sparse checkout of the `unitree_h1` directory.

`models/unitree_rl_gym/` (G1 model, meshes, and the pretrained `motion.pt` policy) is
from [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) by Unitree
Robotics, **BSD-3-Clause**. Local changes: a pelvis IMU site added to `g1_12dof.xml` and
the new `scene_imusim.xml` wrapper. The policy is loaded read-only for inference.
