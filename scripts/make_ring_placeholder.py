"""Generate a placeholder boxing-ring mesh: boxing/assets/ring.stl.

The ring is purely cosmetic in the sim (MuJoCo collides meshes by their convex hull,
so a hollow ring can't be a collider directly — containment is handled by primitive
"rope walls" in boxing/arena.py). This script just gives you something ring-shaped to
look at and a starting point to edit. Swap in any STL you like, or re-run this with
different sizes.

The shape is a square canvas (mat) with four corner posts and three rope levels, built
as one watertight mesh of axis-aligned boxes. The mat's top sits at z = 0 (the floor),
so the fighters stand on it.

    python scripts/make_ring_placeholder.py                 # default 1.15 m half-width
    python scripts/make_ring_placeholder.py --half 1.4 --rope-top 0.6
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "boxing" / "assets" / "ring.stl"

# Unit cube corners and the 12 triangles (2 per face), wound outward.
_CORNERS = np.array([(sx, sy, sz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], float)
_FACES = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
          (2, 6, 7), (2, 7, 3), (1, 3, 7), (1, 7, 5), (0, 4, 6), (0, 6, 2)]


def _box(tris: list, center, half):
    """Append the 12 triangles of an axis-aligned box to ``tris``."""
    c = np.asarray(center, float)
    h = np.asarray(half, float)
    verts = _CORNERS * h + c
    for a, b, d in _FACES:
        va, vb, vd = verts[a], verts[b], verts[d]
        n = np.cross(vb - va, vd - va)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        tris.append((n, va, vb, vd))


def build(half=1.15, mat_thick=0.05, post=0.06, post_h=0.55,
          rope_t=0.02, rope_levels=(0.20, 0.36, 0.52)) -> list:
    """Return the ring's triangle list. ``half`` is the rope-square half-width (m)."""
    tris: list = []
    m = half + 0.10                                   # mat overhangs the ropes a little
    _box(tris, (0, 0, -mat_thick / 2), (m, m, mat_thick / 2))           # canvas / mat
    for sx in (-1, 1):                                                  # four corner posts
        for sy in (-1, 1):
            _box(tris, (sx * half, sy * half, post_h / 2), (post, post, post_h / 2))
    for z in rope_levels:                                              # ropes on all 4 sides
        _box(tris, (0,  half, z), (half, rope_t, rope_t))
        _box(tris, (0, -half, z), (half, rope_t, rope_t))
        _box(tris, ( half, 0, z), (rope_t, half, rope_t))
        _box(tris, (-half, 0, z), (rope_t, half, rope_t))
    return tris


def write_stl(path: Path, tris: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)                            # 80-byte header
        f.write(struct.pack("<I", len(tris)))
        for n, a, b, c in tris:
            f.write(struct.pack("<3f", *n))
            for v in (a, b, c):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate boxing/assets/ring.stl placeholder.")
    ap.add_argument("--half", type=float, default=1.15, help="rope-square half-width (m)")
    ap.add_argument("--post-height", type=float, default=0.55)
    ap.add_argument("--rope-top", type=float, default=0.52, help="height of the top rope (m)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    levels = tuple(np.linspace(0.20, args.rope_top, 3))
    tris = build(half=args.half, post_h=args.post_height, rope_levels=levels)
    write_stl(args.out, tris)
    print(f"wrote {args.out}  ({len(tris)} triangles, half={args.half} m)")


if __name__ == "__main__":
    main()
