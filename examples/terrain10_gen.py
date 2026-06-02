#!/usr/bin/env -S uv run python
"""Generate examples/terrain10.npy — a 10x10 m walkable height field for legged robots.

Heights are in METERS (the YAML uses size: [5, 5, 1], i.e. sz=1, so no extra scaling).
Grid is 101x101 (0.1 m spacing). Gentle rolling hills (typical slope <12 deg, peaks
~15 deg) with a flat ~1 m-radius spawn disc at the center so a robot can stand up before
walking onto the terrain. Re-run to regenerate:  ./examples/terrain10_gen.py
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

z = np.ascontiguousarray(z, dtype=np.float64)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'terrain10.npy')
np.save(out, z)

# Report walkability stats.
dx = 2*HALF/(N-1)
gy, gx = np.gradient(z, dx)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
print(f"wrote {out}  grid={z.shape}  spacing={dx:.3f} m")
print(f"height range = [{z.min():+.3f}, {z.max():+.3f}] m")
print(f"slope: mean={slope_deg.mean():.1f} deg, max={slope_deg.max():.1f} deg")
