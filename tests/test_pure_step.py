#!/usr/bin/env -S uv run python
"""Property tests for Model.step() referential transparency — the `ctx` /
SolverState pure-function interface (Phase 1, 2026-05-25).

Unlike tests/regression/test_traj.py (golden-value comparison; needs a reviewed
baseline re-capture on legitimate numeric changes), these assert STRUCTURAL
invariants — determinism, ctx immutability, warm==Env equivalence, warm/cold
fixpoint agreement — so they are self-validating: no golden baseline, no
re-capture. They guard the pure contract against future refactors (workspace
externalization, sparse-A, etc.) that could silently break it.

Run:  uv run python /home/ubuntu/uv/fg/tact/tests/test_pure_step.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # → fg/tact
sys.path.insert(0, os.path.dirname(ROOT))    # → fg (so `import tact` works)
os.chdir(ROOT)                               # example YAML paths + C mesh loader

import tact
from tact import SolverState

SCENE = 'tests/scenes/box_wall'   # frozen fixture (see tests/scenes/); contact-rich
N     = 200                       # settle steps

def _model(use_c=True):
    m = tact.Model(SCENE); m.use_c = use_c
    return m

def run_env(n=N):
    e = tact.Env(SCENE, render=False)
    u = np.zeros(e.dof)
    for _ in range(n): e.step(u)
    return e.q.copy()

def run_threaded(use_c=True, n=N):
    m = _model(use_c); q, qd = m.q0.copy(), m.qd0.copy(); tau = np.zeros(len(q)); ctx = None
    for _ in range(n): q, qd, y, ctx = m.step(q, qd, tau, ctx=ctx)
    return q, ctx

def run_cold(n=N):
    m = _model(); q, qd = m.q0.copy(), m.qd0.copy(); tau = np.zeros(len(q))
    for _ in range(n): q, qd, y, _ = m.step(q, qd, tau, ctx=None)   # cold every step
    return q

CHECKS = []
def check(name, ok, detail=''):
    CHECKS.append((name, bool(ok), detail))

def main():
    # 1. zero_state sizing
    m = _model(); npair = len(m.cpair); zs = m.zero_state()
    check('zero_state len = 6*MAX_PTS_PER_PAIR*n_pair',
          len(zs.lam) == 6 * tact.MAX_PTS_PER_PAIR * max(npair, 1),
          f'len={len(zs.lam)}')

    # 2. cold-start determinism — referential transparency from inputs alone
    d = np.abs(run_cold() - run_cold()).max()
    check('cold determinism (Δq == 0)', d == 0.0, f'max|Δq|={d:.2e}')

    # 3. warm-threaded Model == Env  → bit-identical (so no baseline re-capture)
    mq, _ = run_threaded(use_c=True); eq = run_env()
    d = np.abs(mq - eq).max()
    check('Model(ctx thread) == Env (bit-identical)', d == 0.0, f'max|Δq|={d:.2e}')

    # 4. ctx immutability + fork safety — step must not mutate the input ctx
    m = _model(); q, qd = m.q0.copy(), m.qd0.copy(); tau = np.zeros(len(q)); ctx = None
    for _ in range(50): q, qd, y, ctx = m.step(q, qd, tau, ctx=ctx)
    snap = ctx.lam.copy()
    _, _, _, cA = m.step(q, qd,       tau, ctx=ctx)
    _, _, _, cB = m.step(q, qd + 0.1, tau, ctx=ctx)
    check('ctx immutable after step (fork-safe)', np.array_equal(ctx.lam, snap))
    check('fork → independent carries', not np.array_equal(cA.lam, cB.lam))

    # 5. warm/cold reach the SAME LCP solution given enough iters (warm-start is not a
    #    numeric cheat — at default iters=20 cold PGS is truncated mid-converge so they
    #    differ). Use a single-contact scene (sphere on floor) so PGS converges fully;
    #    box_wall's 138 coupled contacts converge too slowly for a clean fixed-iter test.
    ms = tact.Model('tests/scenes/sphere_test'); ms.use_c = True
    q, qd = ms.q0.copy(), ms.qd0.copy(); tau = np.zeros(len(q)); ctx = None
    for _ in range(600): q, qd, y, ctx = ms.step(q, qd, tau, ctx=ctx)   # settle onto floor
    ms.iters = 500                                                      # crank PGS to solution
    _, qd_w, _, _ = ms.step(q, qd, tau, ctx=ctx)     # warm
    _, qd_c, _, _ = ms.step(q, qd, tau, ctx=None)    # cold
    d = np.abs(qd_w - qd_c).max()
    check('warm/cold same solution @ high iters (1-contact)', d < 1e-6, f'max|Δqd|={d:.2e}')

    # 6. python step path threads ctx too, and tracks the C path (not bit-identical
    #    across py/C, but ~1e-10 over this trajectory)
    pq, _ = run_threaded(use_c=False)
    d = np.abs(pq - mq).max()
    check('python-path threading tracks C path', d < 1e-5, f'max|Δq|={d:.2e}')

    # report
    print(f"\n  {'check':52} result")
    print(f"  {'-'*52} {'-'*26}")
    nfail = 0
    for name, ok, detail in CHECKS:
        if not ok: nfail += 1
        print(f"  {'✓' if ok else '✗'} {name:50} {'OK' if ok else 'FAIL':4} {detail}")
    print()
    if nfail:
        print(f"  {nfail}/{len(CHECKS)} checks FAILED"); sys.exit(1)
    print(f"  all {len(CHECKS)} checks pass")

if __name__ == '__main__':
    main()
