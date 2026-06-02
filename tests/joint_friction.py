#!/usr/bin/env -S uv run python
"""Joint Coulomb friction (MuJoCo `frictionloss`) — correctness tests.

Unlike tests/regression/test_traj.py (a bit-exact trajectory snapshot), these are
ANALYTIC / cross-check tests: they assert against closed-form physics and against
the C↔Python solver agreement, so they need NO captured baseline and never go
stale under a compiler/flag/hardware change. This is the durable kind of test the
friction work seeds (see docs/design-joint-friction.md).

Coverage:
  1. prismatic static-hold      floss > m·g  → joint held (q ≡ 0), the case the
                                removed penalty solver could not do.
  2. prismatic break-away       floss < m·g  → constant a=(m·g−floss)/m, q vs analytic.
  3. prismatic free-fall ctrl   floss = 0    → q = −½ g t².
  4. prismatic monotonic        more floss → less slide.
  5. revolute static-hold/swing exercises the jtype==1 path (lin is jtype==2).
  6. C vs Python A/B            both solver paths agree to ~machine ε on a friction scene.
  7. referential transparency   double-step from the same (q,qd,ctx) is identical and
                                ctx is immutable — on BOTH paths — with friction λ carried.

Pytest-runnable:  uv run pytest tests/joint_friction.py -v
or standalone:    uv run python tests/joint_friction.py
"""
import os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TACT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(TACT_ROOT))      # → fg (so `import tact` works)
import tact

_TMP = tempfile.mkdtemp(prefix='jfric_')

# Vertical prismatic slider (lin → local Z = world Z with euler 0), driven purely
# by gravity; frictionloss resists. No viscous damping → clean constant-accel analysis.
_LIN = """
sim: {{solver: lcp, dt: 0.001, g: [0, 0, -9.81]}}
view: {{target: [0,0,0], distance: 4}}
materials:
    m: {{normal: [20000, 50], tangent: [20000, 50, 0.8], spin: [100,1,0.02], roll: [100,1,0.005], restitution: 0.0}}
bodies:
  - name: root
    shapes: [{{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [-1,0,0,1]}}]
  - name: slider
    joint: {{type: lin, parent: root, euler: [0,0,0], frictionloss: {floss}}}
    inertial: {{mass: 1.0, tensor: [diag, 0.01, 0.01, 0.01]}}
    shapes: [{{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [0.3,0.3,0.8,1]}}]
"""

# Revolute pendulum about world Y (euler [90,0,0] maps rev's local Z → world Y);
# mass offset +X gives gravity torque m·g·L at q0=0 (the max-torque pose).
_REV = """
sim: {{solver: lcp, dt: 0.001, g: [0, 0, -9.81]}}
view: {{target: [0,0,0], distance: 4}}
materials:
    m: {{normal: [20000, 50], tangent: [20000, 50, 0.8], spin: [100,1,0.02], roll: [100,1,0.005], restitution: 0.0}}
bodies:
  - name: root
    shapes: [{{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [-1,0,0,1]}}]
  - name: arm
    joint: {{type: rev, parent: root, euler: [90,0,0], q0: 0, frictionloss: {floss}}}
    inertial: {{mass: 1.0, tensor: [diag, 0.01, 0.01, 0.01], pos: [0.5, 0, 0]}}
    shapes: [{{type: box, param: [0.05,0.5,0.05], contact: [-1, m], rgba: [0.3,0.8,0.3,1]}}]
"""

G, M, DT = 9.81, 1.0, 0.001


def _env(tmpl, floss, use_c=True):
    path = os.path.join(_TMP, 'scene')
    open(path + '.yml', 'w').write(tmpl.format(floss=floss))
    env = tact.Env(path, render=False)
    env.m.use_c = use_c
    return env


def _run(tmpl, floss, nsteps=1000, use_c=True):
    env = _env(tmpl, floss, use_c)
    dof = int(np.sum(env.m.active))
    for _ in range(nsteps):
        env.step(np.zeros(dof))
    return float(env.q[0])


# ---- the checks (each returns (ok, label, detail)) --------------------------
def check_static_hold():
    q = _run(_LIN, 20.0)                       # floss=20 > m·g=9.81
    return abs(q) < 1e-3, "prismatic static-hold (floss > m·g)", f"q={q:+.6f} (expect 0)"

def check_breakaway():
    bad = []
    for floss in (5.0, 2.0):
        a = (M * G - floss) / M
        q_an = -0.5 * a * (DT * 1000) ** 2
        q = _run(_LIN, floss)
        if abs(q - q_an) > 5e-3:
            bad.append(f"floss={floss}: q={q:.4f} vs {q_an:.4f}")
    return not bad, "prismatic break-away (a=(m·g−floss)/m)", "; ".join(bad) or "matches analytic <5e-3"

def check_freefall():
    q = _run(_LIN, 0.0)
    q_an = -0.5 * G * 1.0 ** 2
    return abs(q - q_an) < 5e-3, "prismatic free-fall (floss=0)", f"q={q:.4f} vs {q_an:.4f}"

def check_monotonic():
    finals = [_run(_LIN, fl) for fl in (0, 2, 5, 8, 20)]
    mono = all(finals[i] <= finals[i+1] + 1e-9 for i in range(len(finals)-1))
    return mono, "prismatic monotonic (more floss → less slide)", str([f"{x:+.3f}" for x in finals])

def check_revolute():
    qh = _run(_REV, 10.0)                       # floss=10 > m·g·L≈4.9 → hold at q0=0
    qf = np.rad2deg(_run(_REV, 0.0))            # floss=0 → swings
    ok = abs(np.rad2deg(qh)) < 0.1 and abs(qf) > 20
    return ok, "revolute (jtype==1) hold vs swing", f"hold→{np.rad2deg(qh):.3f}°, free→{qf:.1f}°"

def check_c_vs_python():
    bad = []
    for floss in (5.0, 20.0, 0.0):
        qc = [_run(_LIN, floss, use_c=True)]
        qp = [_run(_LIN, floss, use_c=False)]
        # compare full trajectory, not just final
        ec = _env(_LIN, floss, True);  ep = _env(_LIN, floss, False)
        tc = np.array([ (ec.step(np.zeros(1)), ec.q.copy())[1] for _ in range(1000)])
        tp = np.array([ (ep.step(np.zeros(1)), ep.q.copy())[1] for _ in range(1000)])
        d = float(np.max(np.abs(tc - tp)))
        if d > 1e-9:
            bad.append(f"floss={floss}: max|C−Py|={d:.2e}")
    return not bad, "C vs Python solver A/B (~machine ε)", "; ".join(bad) or "agree <1e-9"

def check_purity():
    bad = []
    for use_c in (True, False):
        m = _env(_LIN, 5.0, use_c).m
        q, qd, ctx = np.zeros(1), np.zeros(1), m.zero_state()
        for _ in range(10):                              # build a non-trivial friction λ in ctx
            q, qd, y, ctx = m.step(q, qd, ctx=ctx)
        lam_fric_before = ctx.lam_fric.copy()
        q1, qd1, y1, c1 = m.step(q, qd, ctx=ctx)
        q2, qd2, y2, c2 = m.step(q, qd, ctx=ctx)         # same inputs → must be identical
        identical = (np.array_equal(q1, q2) and np.array_equal(qd1, qd2)
                     and np.array_equal(c1.lam_fric, c2.lam_fric))
        immutable = np.array_equal(ctx.lam_fric, lam_fric_before)
        carried   = np.any(np.abs(ctx.lam_fric) > 0)
        if not (identical and immutable and carried):
            bad.append(f"use_c={use_c}: id={identical} immut={immutable} carried={carried}")
    return not bad, "referential transparency (friction λ via ctx)", "; ".join(bad) or "pure on both paths"


CHECKS = [check_static_hold, check_breakaway, check_freefall, check_monotonic,
          check_revolute, check_c_vs_python, check_purity]


def test_joint_friction():
    """Pytest entry point — fails on any failed check."""
    fails = []
    for chk in CHECKS:
        ok, label, detail = chk()
        if not ok:
            fails.append(f"  {label}: {detail}")
    assert not fails, "joint-friction check(s) failed:\n" + "\n".join(fails)


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
