#!/usr/bin/env -S uv run python
"""Capture trajectory baselines for the regression test suite.

For each YAML in `SCENARIOS`, runs a deterministic headless sim of `N_STEPS`
zero-input steps and saves the q trajectory to `baseline/<name>.npy`. The
test runner (`test_traj.py`) re-runs the same sims and compares against
these files via `np.allclose`.

Run manually whenever a deliberate physics-affecting change lands AND has
been reviewed → capture new baseline, commit alongside the change. Don't
auto-update on every test failure (that defeats the safety net).

    uv run python tests/regression/capture_baseline.py            # all
    uv run python tests/regression/capture_baseline.py box_wall    # one
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TACT_ROOT = os.path.dirname(os.path.dirname(HERE))   # → fg/tact
sys.path.insert(0, os.path.dirname(TACT_ROOT))       # → fg (so `import tact` works)
os.chdir(TACT_ROOT)

import tact

# (yml, n_steps) — n_steps small enough that the whole suite runs in seconds,
# large enough that contact transients have engaged. 500 steps = 0.5 s at dt=1ms.
SCENARIOS = [
    ('box_wall',                   500),   # 4-row brick wall, exercises box-box manifold
    ('mini_wall_box',              500),   # 3-brick mini wall
    ('wall5_box',                  500),   # 5-brick 2-row wall
    ('obj1',                       500),   # mesh+box contact (mesh path → MPR fallback)
    ('sphere_test',                500),   # sphere on floor
    ('cartpole/cartpole',          500),   # no contact, pure dynamics
    ('rb5/rb5',                    500),   # 6-DoF arm, no contact
    ('rb10/rb10',                  500),   # 6-DoF arm, no contact
    ('arm2',                       500),   # 2-DoF arm, no contact
    ('arm3',                       500),   # 3-DoF arm, no contact
]


def run_one(yml, n_steps):
    # Load from tests/scenes/ — FROZEN copies of the example YAMLs. Test inputs must
    # not drift when someone tweaks examples/, so the suite owns its own fixtures.
    # See tests/scenes/README.md. (Copies are byte-identical at capture time.)
    env = tact.Env(f'tests/scenes/{yml}', render=False)
    nq = len(env.q)
    if nq == 0:
        # No movable bodies (pure environment scene) — nothing to record.
        return np.zeros((0,))
    u = np.zeros(env.dof)
    traj = np.empty((n_steps + 1, nq))
    traj[0] = env.q
    for k in range(n_steps):
        env.step(u)
        traj[k + 1] = env.q
    return traj


def main():
    requested = set(sys.argv[1:])
    out_dir = os.path.join(HERE, 'baseline')
    os.makedirs(out_dir, exist_ok=True)

    print(f"  {'scenario':30s}  {'nq':>4}  {'n_steps':>7}  result")
    print(f"  {'-'*30}  {'-'*4}  {'-'*7}  {'-'*40}")
    for yml, n in SCENARIOS:
        if requested and yml not in requested:
            continue
        traj = run_one(yml, n)
        out_path = os.path.join(out_dir, yml.replace('/', '__') + '.npy')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, traj)
        print(f"  {yml:30s}  {traj.shape[1] if traj.ndim>1 else 0:4d}  {n:7d}  → {out_path}")


if __name__ == '__main__':
    main()
