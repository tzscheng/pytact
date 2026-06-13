#!/usr/bin/env -S uv run python
"""Generate extras/envs/hf1.bin — a 10x10 m walkable height field for legged robots.

One data file, two scene definitions: extras/envs/hf1.yml (tact, same dir) and
extras/mjcf/hf1.xml (mujoco, ../envs/hf1.bin) both reference this output.
Heights are in METERS (the YAML uses size: [5, 5, 1], i.e. sz=1, so no extra scaling).
Grid is 101x101 (0.1 m spacing). Gentle rolling hills (typical slope <12 deg, peaks
~15 deg) with a flat ~1 m-radius spawn disc at the center so a robot can stand up before
walking onto the terrain. Re-run to regenerate:  ./extras/envs/hf1_gen.py
"""
import os
import numpy as np

N    = 101          # grid nodes per side (0.1 m spacing over 10 m)
HALF = 5.0          # half-extent (m); spans [-5, 5] x [-5, 5]

xs = np.linspace(-HALF, HALF, N)          # X along columns (j)
ys = np.linspace(-HALF, HALF, N)          # Y along rows (i)
X, Y = np.meshgrid(xs, ys)                # X[i,j]=xs[j], Y[i,j]=ys[i] -> data[i*N+j]

# Rolling terrain: two low-frequency components keep slopes gentle and walkable.
z  = 0.12 * np.sin(2*np.pi*X/5.0) * np.cos(2*np.pi*Y/5.0)        # big hills (~9 deg)
z += 0.06 * np.sin(2*np.pi*X/2.5 + 0.7) * np.sin(2*np.pi*Y/3.0)  # medium undulation (~9 deg)

# Flat spawn disc at center: smoothstep mask 0 (r<1 m) -> 1 (r>2.5 m).
r = np.sqrt(X**2 + Y**2)
t = np.clip((r - 1.0) / (2.5 - 1.0), 0.0, 1.0)
z *= t*t*(3.0 - 2.0*t)

z = np.ascontiguousarray(z, dtype=np.float32)

# MuJoCo custom hfield binary (the ONE shared data file — read by both tact's
# YAML loader and MuJoCo's <hfield file=...>): int32 nrow, int32 ncol,
# float32 data[nrow*ncol], row-major data[i*ncol+j], row i along +Y, col j
# along +X. Values are raw meters; tact uses them as-is (size sz=1), MuJoCo
# normalizes to [0,1] — the mjenv scene must carry size[2]=range and a geom z
# offset of min (printed below; update tact/extras/mjcf/hf1.xml if regenerated).
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf1.bin')
with open(out, 'wb') as f:
    np.array(z.shape, dtype=np.int32).tofile(f)   # nrow, ncol
    z.tofile(f)

# Report walkability stats + the constants the mjenv scene file needs.
dx = 2*HALF/(N-1)
gy, gx = np.gradient(z.astype(np.float64), dx)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
print(f"wrote {out}  grid={z.shape}  spacing={dx:.3f} m")
print(f"height range = [{z.min():+.6f}, {z.max():+.6f}] m  (range {z.max()-z.min():.6f})")
print(f"mjenv: <hfield size=\"{HALF} {HALF} {z.max()-z.min():.6f} ...\"/>, geom pos z = {z.min():+.6f}")
print(f"slope: mean={slope_deg.mean():.1f} deg, max={slope_deg.max():.1f} deg")
