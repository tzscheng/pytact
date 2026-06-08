#!/usr/bin/env -S uv run python
"""Headless stability analysis for the box_wall scene (tests/scenes/box_wall.yml).

Runs the brick wall for `T_SIM` seconds with zero input, sampling kinetic
energy and per-brick pose drift on a coarse schedule. Reports:

  - row-level settle depth (mean z drift vs q0 — LCP's CFM regularization
    allows sub-mm penetration per brick under load)
  - max lateral (xy) drift across all bricks (under Phase 2 box-box manifold:
    ~μm scale; under Phase 1 single-point MPR baseline: was ~12 mm at 30 s)
  - max tilt angle (acos of body-z dotted with world-z, from the free joint's
    Euler angles)
  - kinetic energy trajectory (should monotonically decay)
  - final verdict against tolerances (set loose enough to allow either
    Phase 1 or Phase 2 contact solver paths)

Run from anywhere — script chdirs into fg/tact so the relative YAML path
resolves and the C-side mesh loader sees `demos/...`:
    uv run python /home/ubuntu/uv/fg/tact/tests/box_wall_stability.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                # → fg/tact
sys.path.insert(0, os.path.dirname(ROOT))   # → fg (so `import tact` works)
os.chdir(ROOT)

import tact

T_SIM       = 3.0     # seconds of sim time
SAMPLE_EVERY = 0.1    # seconds between samples
TOL_Z_DRIFT  = 5e-3   # 5 mm settling compression per brick is fine
TOL_XY_DRIFT = 5e-3   # 5 mm lateral wander (Phase 1: 12 mm at 30 s; Phase 2: μm)
TOL_TILT_DEG = 2.0    # 2° tilt
# CFM-regularized LCP contacts don't reach exact rest — there's a low-amplitude
# residual KE from constraint-projection PGS jitter. Decay (last quartile mean
# strictly less than first quartile mean) + boundedness (no KE sample exceeds
# the initial drop spike) is the right criterion. penalty path has similar
# residual but from spring-damper chatter at √(k_n/m) eigenfrequency.
TOL_KE_BOUNDED = 5e-3 # J — anything bigger means the wall is exploding


def euler_to_body_z(rx, ry, rz):
    """Tact's free joint uses ZYX-intrinsic Euler (matches model default).
    Returns the world-frame body z-axis after R = Rz(rz) Ry(ry) Rx(rx).
    The z-axis is the third column."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # third column of Rz Ry Rx:
    return np.array([cz*sy*cx + sz*sx,
                     sz*sy*cx - cz*sx,
                     cy*cx])


def main():
    env = tact.Env('tests/scenes/box_wall', render=False)   # frozen fixture (see tests/scenes/)
    nb_bricks = (len(env.q)) // 6                       # all bodies are free here
    dt = env.m.dt
    n_steps = int(T_SIM / dt)
    sample_stride = int(SAMPLE_EVERY / dt)

    q0 = env.q.copy()
    u  = np.zeros(env.dof)

    print(f"{'─'*72}")
    print(f"  box_wall stability analysis")
    print(f"{'─'*72}")
    print(f"  bricks       : {nb_bricks}")
    print(f"  nq / dof     : {len(env.q)} / {env.dof}")
    print(f"  dt           : {dt*1e3:.2f} ms")
    print(f"  T_SIM        : {T_SIM:.2f} s ({n_steps} steps)")
    print(f"  sample every : {SAMPLE_EVERY*1e3:.0f} ms")
    print()

    ke_trace, t_trace = [], []
    t0 = time.perf_counter()
    for k in range(n_steps + 1):
        if k % sample_stride == 0:
            qd = env.qd
            # Free-joint qd layout: [vx, vy, vz, wx, wy, wz] per brick.
            # KE_translational = ½ m v². Each brick mass = 0.2 (from YAML).
            # We use a quick translational-only estimate — rotational
            # contribution is small for stationary-ish bricks but if you
            # want exact KE, hit the model.mass_matrix path.
            v2 = qd.reshape(nb_bricks, 6)[:, :3] ** 2
            ke = 0.5 * 0.2 * v2.sum()
            ke_trace.append(ke)
            t_trace.append(k * dt)
        if k < n_steps:
            env.step(u)
    wall = time.perf_counter() - t0

    q_final = env.q.copy()
    dq = q_final.reshape(nb_bricks, 6) - q0.reshape(nb_bricks, 6)

    # per-brick metrics
    z_drift  = dq[:, 2]                                 # signed (compression = negative)
    xy_drift = np.linalg.norm(dq[:, :2], axis=1)
    # tilt: angle between final body-z and world-z
    tilt_deg = np.array([
        np.degrees(np.arccos(np.clip(
            euler_to_body_z(*q_final.reshape(nb_bricks, 6)[i, 3:6])[2], -1, 1)))
        for i in range(nb_bricks)
    ])

    # group by row using initial z (0.04, 0.12, 0.20, 0.28)
    row_z = q0.reshape(nb_bricks, 6)[:, 2]
    row_id = np.round((row_z - 0.04) / 0.08).astype(int)

    print(f"  ke trace (J):")
    for t, e in zip(t_trace, ke_trace):
        bar = '█' * int(min(60, max(0, np.log10(e + 1e-12) + 6) * 6))
        print(f"    t={t:5.2f}s  KE={e:.3e}  {bar}")
    print()

    print(f"  per-row mean z-drift (settling compression, m):")
    for r in range(int(row_id.max()) + 1):
        m = row_id == r
        print(f"    row {r} ({m.sum():d} bricks, z0={row_z[m][0]:.3f}):"
              f"  mean Δz = {z_drift[m].mean()*1e3:+7.3f} mm   "
              f"max |Δxy| = {xy_drift[m].max()*1e3:6.3f} mm   "
              f"max tilt = {tilt_deg[m].max():5.2f}°")
    print()

    ke_arr = np.array(ke_trace)
    q = len(ke_arr) // 4
    ke_q1 = ke_arr[1:q+1].mean()        # skip t=0 (KE=0)
    ke_q4 = ke_arr[-q:].mean() if q else ke_arr[-1]

    z_ok    = np.all(np.abs(z_drift)  < TOL_Z_DRIFT)
    xy_ok   = np.all(xy_drift         < TOL_XY_DRIFT)
    tilt_ok = np.all(tilt_deg         < TOL_TILT_DEG)
    ke_bounded = ke_arr.max() < TOL_KE_BOUNDED
    ke_decay   = ke_q4 < ke_q1

    print(f"  global maxima:")
    print(f"    max |Δz|       = {np.abs(z_drift).max()*1e3:7.3f} mm   "
          f"(tol {TOL_Z_DRIFT*1e3:.1f} mm)   {'OK' if z_ok else 'FAIL'}")
    print(f"    max |Δxy|      = {xy_drift.max()*1e3:7.3f} mm   "
          f"(tol {TOL_XY_DRIFT*1e3:.1f} mm)   {'OK' if xy_ok else 'FAIL'}")
    print(f"    max tilt       = {tilt_deg.max():7.3f}°    "
          f"(tol {TOL_TILT_DEG:.1f}°)    {'OK' if tilt_ok else 'FAIL'}")
    print(f"    peak KE        = {ke_arr.max():.3e} J   "
          f"(tol {TOL_KE_BOUNDED:.1e} J)  {'OK' if ke_bounded else 'FAIL'}")
    print(f"    KE Q1 mean     = {ke_q1:.3e} J")
    print(f"    KE Q4 mean     = {ke_q4:.3e} J   "
          f"({'decaying ✓' if ke_decay else 'growing ✗'})")
    print()

    verdict = z_ok and xy_ok and tilt_ok and ke_bounded and ke_decay
    print(f"  verdict: {'STABLE ✓' if verdict else 'UNSTABLE ✗'}")
    print(f"  wall-clock for {n_steps} steps: {wall:.2f}s "
          f"(realtime factor {T_SIM/wall:.1f}×)")
    print(f"{'─'*72}")
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
