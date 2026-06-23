# G1 Boxing (1v1)

A two-player fighting game built on the `humanoid_sim` foundation, on the `boxing`
branch. Two Unitree G1 boxers share one MuJoCo scene; each is driven by the pretrained
locomotion policy and balances actively. Knock the other fighter down to win the round.

**No IMU here.** The IMU error simulation and its UI are not used — the fighters read
ground-truth state. This branch reuses only the model assets and the policy weights.

## Run

```bash
python scripts/boxing_demo.py            # opens a browser at http://127.0.0.1:8002/
python scripts/boxing_demo.py --no-browser --port 8080
```

Physics + policies run in Python; the browser compiles the combined two-robot model in
MuJoCo-WASM and renders it (first load fetches three.js + `mujoco-js` from a CDN).

## Controls

| | Player 1 (blue) | Player 2 (orange) |
|---|---|---|
| Move / turn | `W A S D` | arrow keys |
| Strafe | `Q` `E` | `,` `.` |
| Punch L / R | `F` / `G` | `K` / `L` |
| Rematch | `Enter` / `R` | `Enter` / `R` |

**PS4 controllers** (connect over Bluetooth, then press a button so the browser sees it):
gamepad *i* drives player *i* — left stick moves, right stick turns, **L1 / R1** (or
Square / Circle) punch left / right, **Options** rematches.

## How a "punch" works (and a model limitation)

The `g1_12dof` boxer is **leg-only**: 12 actuated leg joints, but the arms and gloves are
rigid (no shoulder/elbow joints). So a scripted *arm* swing isn't possible without adding
articulated arms to the model. Instead a **punch is a scripted committed lunge** — a hard
forward charge toward the opponent (left/right pick the lead/strafe side); the rigid
gloves/body make contact and a knockdown (base height below `KO_HEIGHT`) ends the round.
The punch is wired through `FighterController.punch()`, so when articulated arms are added
later it can drive an arm trajectory instead with no other changes.

## Architecture

```
boxing/
  arena.py        two G1 boxers in one model via MjSpec attach (r1_/r2_ prefixes);
                  per-fighter qpos/qvel/ctrl slices; exports XML for the browser
  fighter.py      locomotion policy on a robot slice (ground truth) + lunge punch
  game.py         GameSim: step both fighters, knockdown detection, round/winner state
  web/server.py   BoxingWorker (real-time stepping) + SSE qpos/state + model serving
  web/boxing.html, boxing_app.js   two-robot render, 2-player keyboard + PS4 gamepad, KO
scripts/boxing_demo.py             launcher
```
