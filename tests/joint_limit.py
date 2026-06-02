#!/usr/bin/env -S uv run python
"""Joint range limits (MuJoCo-style) — correctness tests.

A joint limit is solved as a one-sided LCP constraint row (a "contact normal" on the
joint coordinate): when q reaches a bound it adds a row with Jacobian ±e_j, λ≥0, and
Baumgarte push-out — the second concrete instance of the row-table generalization the
friction work built (see docs/design-lcp-perf.md I1–I5).

Analytic / cross-check tests (no captured baseline, never go stale):
  1. holds at the stop   — a torque driving a joint into its limit stops it at the bound
                          (both upper and lower), with only a tiny transient overshoot.
  2. unlimited passes     — lo==hi (unlimited) lets the joint accelerate freely past.
  3. inactive in-range    — a wide limit doesn't perturb interior motion.
  4. C vs Python A/B      — both solver paths agree to ~machine ε with a limit active.
  5. referential transparency — double-step at the stop is identical; ctx carries limit λ.

Pytest-runnable:  uv run pytest tests/joint_limit.py -v
or standalone:    uv run python tests/joint_limit.py

NOTE: `limit` in YAML is DEGREES (rev) / m (lin), same convention as `q0`; internally
stored as rad/m. These tests use a rev joint, so the bound is checked in radians.
"""
import os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TACT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(TACT_ROOT))
import tact

_TMP = tempfile.mkdtemp(prefix='jlim_')

# Single rev rotor about local Z, point mass at the joint origin (joint inertia = Izz);
# no gravity/damping so a constant torque drives it cleanly into the stop.
_YAML = """
sim: {{solver: lcp, dt: 0.001, g: [0, 0, 0]}}
view: {{target: [0,0,0], distance: 3}}
materials:
    m: {{normal: [20000, 50], tangent: [20000, 50, 0.8], spin: [100,1,0.02], roll: [100,1,0.005], restitution: 0.0}}
bodies:
  - name: root
    shapes: [{{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [-1,0,0,1]}}]
  - name: link
    joint: {{type: rev, parent: root, limit: [{lo}, {hi}]}}
    inertial: {{mass: 1.0, tensor: [diag, 0.01, 0.01, 0.01], pos: [0,0,0]}}
    shapes: [{{type: box, param: [0.2,0.05,0.05], contact: [-1, m], rgba: [0.3,0.3,0.8,1]}}]
"""
# YAML `limit` is in DEGREES (rev); the joint stops at the radian equivalent.
LO_DEG, HI_DEG = -30.0, 30.0
LO_RAD, HI_RAD = np.deg2rad(LO_DEG), np.deg2rad(HI_DEG)


def _run(lo_deg, hi_deg, tau, nsteps=2000, use_c=True):
    open(os.path.join(_TMP, 's.yml'), 'w').write(_YAML.format(lo=lo_deg, hi=hi_deg))
    env = tact.Env(os.path.join(_TMP, 's'), render=False)
    env.m.use_c = use_c
    traj = np.empty(nsteps)
    for k in range(nsteps):
        env.step(np.array([tau]))
        traj[k] = env.q[0]                 # internal q is radians
    return traj


def check_holds_at_stop():
    bad = []
    qp = _run(LO_DEG, HI_DEG, +1.0)
    if not (abs(qp[-1] - HI_RAD) < 2e-3 and qp.max() < HI_RAD + 1e-2):
        bad.append(f"upper: final={qp[-1]:.5f} max={qp.max():.5f}")
    qn = _run(LO_DEG, HI_DEG, -1.0)
    if not (abs(qn[-1] - LO_RAD) < 2e-3 and qn.min() > LO_RAD - 1e-2):
        bad.append(f"lower: final={qn[-1]:.5f} min={qn.min():.5f}")
    return not bad, "holds at upper/lower stop (limit=30°→0.524 rad)", "; ".join(bad) or f"q→{qp[-1]:+.4f}/{qn[-1]:+.4f} rad"


def check_unlimited_passes():
    q = _run(0, 0, +1.0, nsteps=500)     # lo==hi → unlimited
    return q[-1] > 1.0, "unlimited (lo==hi) accelerates freely", f"q={q[-1]:.2f} after 500 steps"


def check_inactive_in_range():
    q = _run(-1000, 1000, +0.5, nsteps=200)  # very wide (±17 rad), never reached
    return q[-1] > 0.05, "wide limit inactive in interior", f"q={q[-1]:.4f} rad"


def check_c_vs_python():
    bad = []
    for tau in (+1.0, -1.0):
        d = float(np.max(np.abs(_run(LO_DEG, HI_DEG, tau, use_c=True) - _run(LO_DEG, HI_DEG, tau, use_c=False))))
        if d > 1e-9:
            bad.append(f"tau={tau:+}: |C−Py|={d:.2e}")
    return not bad, "C vs Python solver A/B", "; ".join(bad) or "agree <1e-9"


def check_purity():
    open(os.path.join(_TMP, 's.yml'), 'w').write(_YAML.format(lo=LO_DEG, hi=HI_DEG))
    bad = []
    for use_c in (True, False):
        m = tact.Env(os.path.join(_TMP, 's'), render=False); m.m.use_c = use_c; m = m.m
        q, qd, ctx = np.zeros(1), np.zeros(1), m.zero_state()
        for _ in range(800):                                       # drive into +limit
            q, qd, y, ctx = m.step(q, qd, tau=np.array([1.0]), ctx=ctx)
        lam_before = ctx.lam_limit.copy()
        q1, qd1, y1, c1 = m.step(q, qd, tau=np.array([1.0]), ctx=ctx)
        q2, qd2, y2, c2 = m.step(q, qd, tau=np.array([1.0]), ctx=ctx)
        identical = np.array_equal(q1, q2) and np.array_equal(c1.lam_limit, c2.lam_limit)
        immutable = np.array_equal(ctx.lam_limit, lam_before)
        carried   = np.any(np.abs(ctx.lam_limit) > 0)
        if not (identical and immutable and carried):
            bad.append(f"use_c={use_c}: id={identical} immut={immutable} carried={carried}")
    return not bad, "referential transparency (limit λ via ctx)", "; ".join(bad) or "pure on both paths"


CHECKS = [check_holds_at_stop, check_unlimited_passes, check_inactive_in_range,
          check_c_vs_python, check_purity]


def test_joint_limit():
    fails = []
    for chk in CHECKS:
        ok, label, detail = chk()
        if not ok:
            fails.append(f"  {label}: {detail}")
    assert not fails, "joint-limit check(s) failed:\n" + "\n".join(fails)


def main():
    n_fail = 0
    for chk in CHECKS:
        ok, label, detail = chk()
        print(f"  {'✓' if ok else '✗'} {label:42s}  {detail}")
        n_fail += not ok
    print()
    if n_fail:
        print(f"  {n_fail}/{len(CHECKS)} checks FAILED"); sys.exit(1)
    print(f"  all {len(CHECKS)} checks pass"); sys.exit(0)


if __name__ == '__main__':
    main()
