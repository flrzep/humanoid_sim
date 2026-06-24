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

The fighters keep the G1's original colours; each player's **gloves** are tinted so you
can tell them apart (Player 1 blue, Player 2 red). The view fills the window. A
**Controls** button (top-left) opens a modal with the full control scheme and a live
overview of which input device each player is using — it also opens automatically on load
as the start screen.

## Controls

| | Player 1 (blue gloves) | Player 2 (red gloves) |
|---|---|---|
| Move / turn | `W A S D` | arrow keys |
| Strafe | `Q` `E` | `,` `.` |
| Punch L / R | `F` / `G` | `K` / `L` |
| Rematch | `Enter` / `R` | `Enter` / `R` |

**PS4 controllers** (connect over Bluetooth, then press a button so the browser sees it):
gamepad *i* drives player *i* — left stick moves, right stick turns, **L1 / R1** (or
Square / Circle) punch left / right, **Options** rematches. The Controls modal shows a
green dot per player when their gamepad is detected.

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

## The ring

The ring is split into **looks** and **physics**, because MuJoCo collides meshes by their
*convex hull* — a hollow ring mesh would become a solid block you couldn't stand inside.

- **Looks:** a cosmetic mesh geom loaded from `boxing/assets/ring.stl`. Drop in your own
  STL, edit the placeholder, or regenerate it:
  `python scripts/make_ring_placeholder.py [--half 1.4 --rope-top 0.6]`. The mesh never
  collides (unless you set `RING_MESH_COLLIDE = True`, only sensible for a convex STL).
- **Physics:** four invisible primitive **rope walls** (`RING_WALLS`) that actually
  contain the fighters, sized by `RING_HALF`. They're in render group 3 so the browser
  hides them; only your STL is visible.

All knobs are the `# --- Ring tunables ---` block in `arena.py`: `RING_STL`, `RING_POS`,
`RING_SCALE`, `RING_RGBA`, `RING_HALF`, `WALL_HEIGHT`. The ring body is static (no joint),
so it adds nothing to the fighters' DOFs and leaves the locomotion policy untouched. Keep
`RING_HALF` matched to your STL's rope footprint so the visible ropes line up with the
walls.

**What MuJoCo needs for a collision object** (your STL, in data-structure terms): a *mesh
asset* — a vertex array + triangle-index array parsed from the STL and registered once —
plus a *geom* that references it by name with a transform (pos/quat/scale) and collision
attributes (`contype`/`conaffinity` bitmasks, `condim`, friction). `add_ring()` builds
both as a small child spec so the mesh resolves to a bare filename (browser-loadable).

## Architecture

```
boxing/
  arena.py        two G1 boxers in one model via MjSpec attach (r1_/r2_ prefixes);
                  per-fighter qpos/qvel/ctrl slices; add_ring(); exports XML for browser
  assets/ring.stl placeholder ring mesh (swap / edit / regenerate)
  fighter.py      locomotion policy on a robot slice (ground truth) + lunge punch
  game.py         GameSim: step both fighters, knockdown detection, round/winner state
  web/server.py   BoxingWorker (real-time stepping) + SSE qpos/state + model serving
  web/boxing.html, boxing_app.js   two-robot render, 2-player keyboard + PS4 gamepad, KO
scripts/boxing_demo.py             launcher
scripts/make_ring_placeholder.py   regenerate assets/ring.stl
```
