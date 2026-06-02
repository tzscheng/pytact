"""Verify floating-base wiring: pack helpers + standing-balance dynamics identity.

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_floating_base.py
"""
import sys; sys.path.insert(0, '/home/ubuntu/uv/fg')
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

m_fb = tact.Model('dog', fixed_base=False)
nq = len(m_fb.q0); nv = len(m_fb.qd0)
mtot = float(sum(m_fb.m))
g = np.array(m_fb.g) if any(m_fb.g) else np.array([0, 0, -9.81])
if not any(m_fb.g): m_fb.g = list(g)   # ensure bias() sees gravity (Model defaults g=[0,0,0])
print(f'\nModel: dog (floating-base)  nq={nq} nv={nv}  m_tot={mtot:.4f}  g={tuple(m_fb.g)}')

# ----- 1) pack round-trip -----
print('\n[1] pack_q_fb / pack_qd_fb round-trip')
rng = np.random.default_rng(1)
q_joint = rng.standard_normal(12) * 0.2
qd_joint = rng.standard_normal(12) * 0.3
# random small rotation (stay in branch where logmap_so3 is single-valued)
axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
angle = 0.5
R = tact.expmap_so3(axis * angle)
p = np.array([0.1, -0.2, 0.5])
w_body = rng.standard_normal(3) * 0.4
v_world = rng.standard_normal(3) * 0.3

q_fb  = m_fb.pack_q_fb(q_joint, R, p)
qd_fb = m_fb.pack_qd_fb(qd_joint, R, w_body, v_world)

# extract back
p_back     = q_fb[:3]
R_back     = tact.expmap_so3(q_fb[3:6])
q_back     = q_fb[6:]
v_body_back= qd_fb[:3]
w_back     = qd_fb[3:6]
qd_back    = qd_fb[6:]

check(np.allclose(p_back, p),                      'p round-trip')
check(np.allclose(R_back, R, atol=1e-12),          'R round-trip (via logmap/expmap)')
check(np.allclose(q_back, q_joint),                'q_joint round-trip')
check(np.allclose(v_body_back, R.T @ v_world),     'v_body = Rᵀ v_world')
check(np.allclose(w_back, w_body),                 'w_body round-trip')
check(np.allclose(qd_back, qd_joint),              'qd_joint round-trip')

# fixed-base Model should reject pack
m_fix = tact.Model('dog', fixed_base=True)
try:
    m_fix.pack_q_fb(q_joint, R, p); rejected = False
except AssertionError: rejected = True
check(rejected, 'pack_q_fb rejects fixed-base Model')

# ----- 2) Standing balance dynamics identity -----
# Equation under qd=0, q̈=0:  b_fb = Sᵀ·τ + Σ J_iᵀ·λ_i.
# Base rows (no actuator):    b_fb[:6] = Σ J_i[:, :6]ᵀ · λ_i.
# Solving λ_i (12 unknowns) from 6 base equations is under-determined — we take
# the min-norm solution and verify it is physically reasonable (z>0, Σ λ_z ≈ m·g,
# friction-cone feasible).
print('\n[2] Standing balance: solve λ from b_fb[:6] and check physical sanity')

q_home = np.array([0, -0.45, 0.7,  0, -0.45, 0.7,  0, -0.45, 0.7,  0, -0.45, 0.7])
R0 = np.eye(3)
foot_keys = {'foot1':'3d', 'foot2':'3d', 'foot3':'3d', 'foot4':'3d'}
foot_radius = 0.025

# Place feet on ground (z = foot_radius), find base height that makes that happen.
p0 = np.array([0, 0, 0.0])
q_fb_probe = m_fb.pack_q_fb(q_home, R0, p0)
foot_pos_probe = m_fb.fk(foot_keys, q_fb_probe).reshape(4, 3)
p0 = np.array([0, 0, foot_radius - foot_pos_probe[:, 2].min()])
q_fb0 = m_fb.pack_q_fb(q_home, R0, p0)
qd_fb0 = np.zeros(nv)

r_com   = m_fb.com(q_fb0)
J_feet  = m_fb.jacob(foot_keys, q_fb0)        # (12, 18)
b_fb    = m_fb.bias(q_fb0, qd_fb0)
foot_pos = m_fb.fk(foot_keys, q_fb0).reshape(4, 3)
print(f'    base p   = {p0}')
print(f'    r_com    = {r_com}')
print(f'    feet xyz =\n{foot_pos}')
print(f'    b_fb[:6] = {b_fb[:6]}')

# Solve  J_base.T @ λ = b_fb[:6]    (J_base.T has shape (6, 12))
J_base_T = J_feet[:, :6].T
lam_sol, *_ = np.linalg.lstsq(J_base_T, b_fb[:6], rcond=None)
lam_per_foot = lam_sol.reshape(4, 3)
print(f'    λ per foot (world-frame N) =\n{lam_per_foot}')

residual = J_base_T @ lam_sol - b_fb[:6]
print(f'    base residual |J^T λ − b[:6]| = {np.linalg.norm(residual):.3e}')

check(np.linalg.norm(residual) < 1e-8,
      'min-norm λ satisfies J_baseᵀ λ = b_fb[:6]',
      f'|res|={np.linalg.norm(residual):.3e}')

# z-components positive (no pull-into-ground)
check(np.all(lam_per_foot[:, 2] > 0),
      'all foot λ_z > 0 (compression)')

# total vertical force ≈ m·g (gravity carried by contacts)
total_lam_z = lam_per_foot[:, 2].sum()
expected_mg = mtot * (-g[2])
check(abs(total_lam_z - expected_mg) < 1e-6,
      'Σ λ_z ≈ m·g',
      f'Σλ_z={total_lam_z:.3f}, mg={expected_mg:.3f}')

# friction cone (μ=0.8 typical) — should hold for a reasonable standing pose
mu = 0.8
fric_ok = all(np.linalg.norm(lam_per_foot[i, :2]) <= mu * lam_per_foot[i, 2] for i in range(4))
check(fric_ok, f'friction cone |λ_xy| ≤ μ·λ_z (μ={mu})')

# ----- 3) Pure-gravity cross-check: b_fb base rows ↔ centroidal wrench -----
# Under qd=0, b_fb is the joint torques to hold against gravity *in air* (no contact).
# Base rows = base-frame wrench that gravity exerts (with RNEA sign so τ = -wrench).
# Independent: w_grav_world = (m·g_world, r_com × m·g_world).
# Map to body frame (R=I here so just transpose): linear = m·g_world, angular = (r_com − r_base) × (m·g_world).
print('\n[3] Cross-check: b_fb[:6] vs analytic gravity wrench at base')
g_world = np.array(m_fb.g)
F_grav_world = mtot * g_world
M_grav_about_base = np.cross(r_com - p0, F_grav_world)
# RNEA bias returns the torque required to oppose gravity → equals minus the gravity wrench.
expected = -np.concatenate([R0.T @ F_grav_world, R0.T @ M_grav_about_base])
print(f'    analytic = {expected}')
print(f'    b_fb[:6] = {b_fb[:6]}')
err = np.linalg.norm(b_fb[:6] - expected)
check(err < 1e-6, 'b_fb[:6] matches analytic −w_grav,base', f'|err|={err:.3e}')

# ----- summary -----
print()
if fails == 0:
    print(f'\033[32mAll checks passed.\033[0m')
    sys.exit(0)
else:
    print(f'\033[31m{fails} check(s) failed.\033[0m')
    sys.exit(1)
