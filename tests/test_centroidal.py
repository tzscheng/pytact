"""Verify com / com_jacob / com_inertia / jacob_dot_qd added to tact.Model.

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_centroidal.py
"""
import sys; sys.path.insert(0, '/home/ubuntu/uv/fg')
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0

def check(cond, label, detail=''):
    global fails
    tag = PASS if cond else FAIL
    print(f'  [{tag}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

m = tact.Model('dog', fixed_base=False)
q0  = m.q0.copy()
qd0 = m.qd0.copy()
nq, nv = len(q0), len(qd0)
mtot = float(sum(m.m))

print(f'\nModel: dog (floating-base)  nq={nq} nv={nv}  M_total={mtot:.4f} kg')

# ----- 1) com sanity -----
print('\n[1] com(q0)')
r_com = m.com(q0)
r_base = q0[:3]  # base origin (jtype=3 has q[:3] = pos)
print(f'    r_com  = {r_com}')
print(f'    r_base = {r_base}')
check(r_com.shape == (3,),                 'shape == (3,)')
check(r_com[2] > 0,                        'z > 0 (above ground)')
check(np.linalg.norm(r_com - r_base) < 0.5,'within 0.5m of base origin')

# ----- 2) com_jacob sanity -----
print('\n[2] com_jacob(q0)')
Jc = m.com_jacob(q0)
print(f'    J_com shape = {Jc.shape}')
print(f'    J_com[:, :6] (base cols) =\n{Jc[:, :6]}')
check(Jc.shape == (3, nv),                                  'shape == (3, nv)')
# qd0=0 → v_com=0 trivially. Strong check: J_com base linear cols = I (qd[:3]=v_body in body frame, base R=I at q0).
R_base = tact.expmap_so3(q0[3:6])
check(np.allclose(Jc[:, :3], R_base, atol=1e-9),            'J_com[:, :3] == R_base (linear-v_body cols)')

# ----- 3) com_jacob finite-diff cross-check -----
print('\n[3] com_jacob: analytic vs finite-diff')
rng = np.random.default_rng(0)
qd_test = rng.standard_normal(nv) * 0.1
dt = 1e-6
# integrate q (axis-angle q[3:6] vector-update is O(dt) accurate enough at this scale)
q1 = q0 + dt * qd_test
v_com_analytic = Jc @ qd_test
v_com_fd       = (m.com(q1) - m.com(q0)) / dt
err = np.linalg.norm(v_com_analytic - v_com_fd)
print(f'    analytic = {v_com_analytic}')
print(f'    fd       = {v_com_fd}')
print(f'    |err|    = {err:.3e}')
check(err < 1e-4, '|J_com·qd − dr_com/dt| < 1e-4')

# ----- 4) com_inertia sanity -----
print('\n[4] com_inertia(q0)')
Ig = m.com_inertia(q0)
print(f'    Ig =\n{Ig}')
eig = np.linalg.eigvalsh(Ig)
print(f'    eigvals = {eig}')
check(Ig.shape == (3, 3),                  'shape == (3, 3)')
check(np.allclose(Ig, Ig.T, atol=1e-12),  'symmetric')
check(np.all(eig > 0),                     'positive-definite (all eigvals > 0)')

# ----- 5) Ig parallel-axis sanity (single body case via construction) -----
# Build a fictitious 1-body system: simply compare against parallel-axis identity
# I_about_com = Σ ( R_i I_i R_iᵀ + m_i (||d||² I − d dᵀ) ),  d = r_i_world − r_com.
# Recompute manually for dog at q0 and compare to Ig.
print('\n[5] Ig: independent recomputation from m, c, I, FK')
T_all = tact.rbd._fk(m.Ti, m.parent, m.jtype, q0)
Ig_manual = np.zeros((3, 3))
for i in range(len(T_all)):
    R_i = T_all[i, :3, :3]
    r_i = (T_all[i] @ np.array([m.c[i][0], m.c[i][1], m.c[i][2], 1.0]))[:3]
    d   = r_i - r_com
    Ig_manual += R_i @ m.I[i] @ R_i.T + m.m[i] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
err = np.linalg.norm(Ig - Ig_manual, ord='fro')
check(err < 1e-12, f'matches independent recomputation', f'|err_F|={err:.3e}')

# ----- 6) jacob_dot_qd sanity: zero qd → zero -----
print('\n[6] jacob_dot_qd: qd=0 → 0')
foot_keys = {'foot1':'3d', 'foot2':'3d', 'foot3':'3d', 'foot4':'3d'}
Jdq_zero = m.jacob_dot_qd(foot_keys, q0, np.zeros(nv))
print(f'    |Jdot·qd| @ qd=0 = {np.linalg.norm(Jdq_zero):.3e}')
check(np.linalg.norm(Jdq_zero) < 1e-9, '|Jdot·qd| < 1e-9 when qd=0')

# ----- 7) jacob_dot_qd scaling: ε·(Jdot·qd) ≈ J(q+ε qd) qd − J(q) qd -----
print('\n[7] jacob_dot_qd scaling consistency')
qd = rng.standard_normal(nv) * 0.5
Jdq = m.jacob_dot_qd(foot_keys, q0, qd, dt=1e-4)
J0 = m.jacob(foot_keys, q0)
eps_list = [1e-3, 1e-4, 1e-5]
for eps in eps_list:
    lhs = (m.jacob(foot_keys, q0 + eps * qd) - J0) @ qd
    rhs = eps * Jdq
    ratio = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-12)
    print(f'    eps={eps:.0e}: |LHS−RHS|/|RHS| = {ratio:.3e}')
# at eps comparable to internal dt (1e-4) ratio should be tiny; at eps=1e-3 still <0.1
lhs1 = (m.jacob(foot_keys, q0 + 1e-4 * qd) - J0) @ qd
rhs1 = 1e-4 * Jdq
r1 = np.linalg.norm(lhs1 - rhs1) / max(np.linalg.norm(rhs1), 1e-12)
check(r1 < 1e-2, 'finite-diff scaling holds at eps=1e-4 (ratio<1e-2)')

# ----- 8) cross-check w/ M(q) base-block structure -----
# Floating-base M has the well-known structure (qd[:3]=v_body, qd[3:6]=w_body):
#   M_locked = [[ m_tot I,           -m_tot [c_b]×       ],
#               [ m_tot [c_b]×,      Ig_body + m_tot [c_b]×ᵀ [c_b]×]]
# where c_b = R_baseᵀ (r_com − r_base) is the CoM offset in body frame, and
# Ig_body = R_baseᵀ Ig R_base.
print('\n[8] M(q0) base 6×6 block matches centroidal identity')
M = m.inertia(q0)
M_bb = M[:6, :6]
c_b = R_base.T @ (r_com - r_base)
Ig_body = R_base.T @ Ig @ R_base
cx = np.array([[0, -c_b[2], c_b[1]], [c_b[2], 0, -c_b[0]], [-c_b[1], c_b[0], 0]])
M_expected = np.block([
    [mtot * np.eye(3),    -mtot * cx],
    [mtot * cx,            Ig_body + mtot * cx.T @ cx],
])
print(f'    |M_bb − M_expected|_F = {np.linalg.norm(M_bb - M_expected, ord="fro"):.3e}')
err = np.linalg.norm(M_bb - M_expected, ord='fro')
check(err < 1e-8, 'M[:6,:6] matches [m·I, -m·[c]×; m·[c]×, Ig_body + m·[c]×ᵀ[c]×]', f'|err|={err:.3e}')

# ----- summary -----
print()
if fails == 0:
    print(f'\033[32mAll checks passed.\033[0m')
    sys.exit(0)
else:
    print(f'\033[31m{fails} check(s) failed.\033[0m')
    sys.exit(1)
