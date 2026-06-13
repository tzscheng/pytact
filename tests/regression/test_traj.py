#!/usr/bin/env -S uv run python
"""Regression test: re-run the scenarios captured by capture_baseline.py and
verify the resulting q trajectory matches the .npy baseline element-wise.

Pytest-runnable:
    uv run pytest tests/regression/test_traj.py -v
or standalone:
    uv run python tests/regression/test_traj.py

A failure means physics changed since the baseline was captured — could be a
bug, could be intentional (then re-run capture_baseline.py to update). Do NOT
auto-update baselines on red.

ATOL is 1e-12 (essentially bit-identical) for WELL-CONDITIONED scenes — the bar
that makes a refactor's no-op-ness checkable. A FEW marginally-stable scenes
(SCENE_ATOL below) are checked at a looser, physically-meaningful bar instead:
their bit-exactness is dominated by compiler FP reassociation, not physics. The
release build uses -ffast-math, so code edits NEAR the PGS hot loop (even
runtime-dead ones) can reassociate its summation and shift a chaotic stack like
box_wall by ~1e-6 m while the physics is provably unchanged (verified by a
strict-IEEE `-fno-fast-math -ffp-contract=off` A/B — see
docs/design-joint-friction.md "Phase 3 -ffast-math finding"). Pinning those two
scenes at bit-level would test the compiler, not the solver, so they get a
~0.1 mm tolerance: tight enough to catch a brick visibly moving (cm scale),
loose enough to ignore reassociation noise.

(The durable fix — decoupling the bit-exact reference from the -ffast-math
release build, and growing analytic/correctness tests like tests/test_joint_friction.py
— is deferred to the next test-infra forcing function, e.g. a compiler bump.)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TACT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(TACT_ROOT))

# Reuse the SCENARIOS list + run_one helper from capture_baseline so the two
# stay in sync.
sys.path.insert(0, HERE)
from capture_baseline import SCENARIOS, run_one


ATOL = 1e-12
RTOL = 0.0
BASELINE_DIR = os.path.join(HERE, 'baseline')

# Per-scene atol override for marginally-stable scenes whose bit-exactness is
# compiler-FP-reassociation-dominated (not physics). ~0.1 mm: ≫ the ~1e-6 m
# -ffast-math noise, ≪ a real collapse (cm). See module docstring. Everything
# not listed here uses the strict 1e-12 bar.
SCENE_ATOL = {
    'box_wall':  1e-4,   # 4-row brick wall — marginally-stable stack
    'wall5_box': 1e-4,   # 5-brick 2-row wall — same
}


def compare_one(yml, n_steps):
    baseline_path = os.path.join(BASELINE_DIR, yml.replace('/', '__') + '.npy')
    if not os.path.exists(baseline_path):
        return False, f"baseline missing: {baseline_path} (run capture_baseline.py)"
    baseline = np.load(baseline_path)
    os.chdir(TACT_ROOT)
    traj = run_one(yml, n_steps)
    if traj.shape != baseline.shape:
        return False, f"shape mismatch: got {traj.shape}, baseline {baseline.shape}"
    if traj.size == 0:
        return True, "trivial (no q-bodies)"
    atol = SCENE_ATOL.get(yml, ATOL)
    if not np.allclose(traj, baseline, atol=atol, rtol=RTOL):
        max_diff = float(np.max(np.abs(traj - baseline)))
        first_diff_step = int(np.argmax(np.any(np.abs(traj - baseline) > atol, axis=1)))
        return False, (f"max|Δq|={max_diff:.3e} (atol {atol:.1e}), first diff at step {first_diff_step}")
    note = "" if atol == ATOL else f" (loose atol {atol:.0e})"
    return True, "OK" + note


def test_regression():
    """Pytest entry point — fails on any baseline mismatch."""
    failures = []
    for yml, n_steps in SCENARIOS:
        ok, msg = compare_one(yml, n_steps)
        if not ok:
            failures.append(f"  {yml}: {msg}")
    if failures:
        raise AssertionError("regression(s):\n" + "\n".join(failures))


def main():
    print(f"  {'scenario':30s}  result")
    print(f"  {'-'*30}  {'-'*50}")
    n_fail = 0
    for yml, n_steps in SCENARIOS:
        ok, msg = compare_one(yml, n_steps)
        status = '✓' if ok else '✗'
        print(f"  {status} {yml:28s}  {msg}")
        if not ok: n_fail += 1
    print()
    if n_fail:
        print(f"  {n_fail}/{len(SCENARIOS)} scenarios regressed")
        sys.exit(1)
    print(f"  all {len(SCENARIOS)} scenarios pass")
    sys.exit(0)


if __name__ == '__main__':
    main()
