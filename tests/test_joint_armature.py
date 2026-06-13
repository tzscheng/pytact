#!/usr/bin/env -S uv run python
"""Joint armature (MuJoCo `armature`) — correctness tests.

Armature is rotor/reflected inertia: a per-DoF constant added to the diagonal of the
generalized inertia matrix M (and to the ABA articulated inertia `d`), matching
MuJoCo's definition exactly. Reflected inertia = rotor_inertia · gear_ratio².

Analytic / cross-check tests (no captured baseline, never go stale):
  1. effective inertia  — a single revolute rotor under constant torque τ accelerates
                          at qdd = τ/(I + armature), to machine precision, for a sweep
                          of armature values.
  2. armature=0 no-op   — adding armature plumbing must not change a zero-armature model.
  3. C vs Python A/B    — both solver paths agree exactly with armature active.

Pytest-runnable:  uv run pytest tests/test_joint_armature.py -v
or standalone:    uv run python tests/test_joint_armature.py
"""
import os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TACT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(TACT_ROOT))
import tact

_TMP = tempfile.mkdtemp(prefix='jarm_')

# Single revolute rotor about local Z, point mass at the joint origin so the joint-
# space inertia is exactly Izz; no gravity/damping → constant-torque test is clean.
IZZ, TAU, DT, N = 0.01, 1.0, 0.001, 100
_YAML = f"""
sim: {{{{solver: lcp, dt: {DT}, g: [0, 0, 0]}}}}
view: {{{{target: [0,0,0], distance: 3}}}}
materials:
    m: {{{{normal: [20000, 50], tangent: [20000, 50, 0.8], spin: [100,1,0.02], roll: [100,1,0.005], restitution: 0.0}}}}
bodies:
  - name: root
    shapes: [{{{{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [-1,0,0,1]}}}}]
  - name: rotor
    joint: {{{{type: rev, parent: root, armature: {{arm}}}}}}
    inertial: {{{{mass: 1.0, tensor: [diag, {IZZ}, {IZZ}, {IZZ}], pos: [0,0,0]}}}}
    shapes: [{{{{type: box, param: [0.2,0.05,0.05], contact: [-1, m], rgba: [0.3,0.3,0.8,1]}}}}]
"""


def _qdd(arm, use_c=True):
    open(os.path.join(_TMP, 'scene.yml'), 'w').write(_YAML.format(arm=arm))
    env = tact.Env(os.path.join(_TMP, 'scene'), render=False)
    env.m.use_c = use_c
    for _ in range(N):
        env.step(np.array([TAU]))                 # constant torque from rest
    return env.qd[0] / (N * DT)                    # semi-implicit Euler: qd_N = N·qdd·dt


def check_effective_inertia():
    bad = []
    for arm in (0.0, 0.01, 0.04, 0.09, 0.99):
        qdd = _qdd(arm)
        an = TAU / (IZZ + arm)
        if abs(qdd - an) / an > 1e-9:
            bad.append(f"arm={arm}: qdd={qdd:.4f} vs {an:.4f}")
    return not bad, "qdd = τ/(Izz+armature) to machine ε", "; ".join(bad) or "matches analytic <1e-9"


def check_zero_noop():
    # armature 0 must reproduce the no-armature qdd = τ/Izz exactly.
    qdd = _qdd(0.0)
    return abs(qdd - TAU / IZZ) < 1e-9, "armature=0 reproduces τ/Izz", f"qdd={qdd:.4f} (expect {TAU/IZZ:.1f})"


def check_c_vs_python():
    bad = []
    for arm in (0.0, 0.04, 0.99):
        d = abs(_qdd(arm, True) - _qdd(arm, False))
        if d > 1e-9:
            bad.append(f"arm={arm}: |C−Py|={d:.2e}")
    return not bad, "C vs Python solver A/B", "; ".join(bad) or "agree <1e-9"


CHECKS = [check_effective_inertia, check_zero_noop, check_c_vs_python]


def test_joint_armature():
    fails = []
    for chk in CHECKS:
        ok, label, detail = chk()
        if not ok:
            fails.append(f"  {label}: {detail}")
    assert not fails, "joint-armature check(s) failed:\n" + "\n".join(fails)


def main():
    n_fail = 0
    for chk in CHECKS:
        ok, label, detail = chk()
        print(f"  {'✓' if ok else '✗'} {label:38s}  {detail}")
        n_fail += not ok
    print()
    if n_fail:
        print(f"  {n_fail}/{len(CHECKS)} checks FAILED"); sys.exit(1)
    print(f"  all {len(CHECKS)} checks pass"); sys.exit(0)


if __name__ == '__main__':
    main()
