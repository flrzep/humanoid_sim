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

## How a punch works

The stock `g1_12dof` boxer is **leg-only** — the arms/gloves are rigid geoms with no
joints. So `arena.articulate_arms()` adds a **shoulder-pitch hinge + position servo per
arm** (moving that arm's geoms onto a new hinged body) at model-build time. The arms are
appended *after* the legs, so the 12 leg joints keep their indices and the locomotion
policy is untouched; a stiff servo holds the arms at rest so balance is barely affected.

A punch (`FighterController.punch(side)`) swings the chosen shoulder forward (the visible
jab) **and** triggers a committed forward lunge. The swing is real physics — the glove
genuinely collides with the opponent (~350 N of shove) — but the robust locomotion policy
recovers from that alone. So a punch only "lands" when the **active glove actually
contacts the opponent** (`_resolve_punches` checks MuJoCo's contact list), and a landed
glove deals a **knockback impulse** (`HIT_FORCE`) — the damage that topples them. You have
to get the right glove *on* the opponent (gap ≲ 0.45 m, which the lunge closes), not just
be nearby. A knockdown (base height below `KO_HEIGHT`) ends the round.

Tuning: `HIT_FORCE` (impulse on a landed glove; **0 = pure-physics shoving, no KO**),
`PUNCH_ANGLE` / `ARM_KP` (how far/hard the arm swings), `LUNGE_VX` (how far a punch closes).

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
