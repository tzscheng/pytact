"""Standing-balance verification for tact.WBC.

Pose dog in its home stance, four feet planted on flat ground, and ask WBC for τ.
Expected:
  - OSQP returns 'solved'
  - q̈ ≈ 0 (we ARE in steady state, so the controller shouldn't accelerate us)
  - Σ λ_z ≈ m·g (vertical equilibrium)
  - |λ_xy| ≤ μ λ_z per foot (friction cone)
  - |τ| ≤ τ_max
  - Sanity: τ produces the correct joint torques to hold against gravity given the GRFs
    (i.e. the dynamics equality is satisfied to QP tolerance).

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_wbc_standing.py
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
m_fb.g = [0.0, 0.0, -9.81]
nv = len(m_fb.qd0); n_act = nv - 6
mtot = float(sum(m_fb.m))
print(f'\nDog floating-base: nv={nv}  n_act={n_act}  m_tot={mtot:.4f} kg  g=({m_fb.g[0]:.2f}, {m_fb.g[1]:.2f}, {m_fb.g[2]:.2f})')

# Standing pose: dog home joints, base R=I, base height so feet rest at z = foot_radius.
q_home = np.array([0,-0.45,0.7, 0,-0.45,0.7, 0,-0.45,0.7, 0,-0.45,0.7])
R0 = np.eye(3)
foot_keys = {'foot1':'3d', 'foot2':'3d', 'foot3':'3d', 'foot4':'3d'}
foot_radius = 0.025
q_fb_probe = m_fb.pack_q_fb(q_home, R0, np.zeros(3))
zmin = m_fb.fk(foot_keys, q_fb_probe).reshape(4, 3)[:, 2].min()
p0 = np.array([0.0, 0.0, foot_radius - zmin])
q_fb0  = m_fb.pack_q_fb(q_home, R0, p0)
qd_fb0 = np.zeros(nv)
print(f'  base p0 = {p0}')
print(f'  feet z  = {m_fb.fk(foot_keys, q_fb0).reshape(4,3)[:,2]}')

# WBC instance
wbc = tact.WBC(m_fb, foot_frames=['foot1','foot2','foot3','foot4'],
               mu=0.8, tau_max=40.0,
               w_body_lin=100.0, w_body_ang=100.0,
               w_swing=200.0, w_post=1.0, w_tau=1e-3,
               verbose=False)

# Tasks: tell WBC "we want to stay exactly here".
body_task = tact.BodyTask(Kp_lin=400.0, Kd_lin=20.0, Kp_ang=100.0, Kd_ang=10.0)
body_task.p_ref = p0.copy(); body_task.R_ref = R0.copy()
body_task.v_ref = np.zeros(3); body_task.w_ref = np.zeros(3)

swing_tasks = [tact.SwingTask() for _ in range(4)]   # unused (all in contact)
posture = tact.PostureTask(q_ref=q_home, Kp=10.0, Kd=2.0)

contact_mask = np.array([True, True, True, True])

# ----- Solve -----
print('\n[1] WBC solve()')
tau, res = wbc.solve(q_fb0, qd_fb0, contact_mask, body_task, swing_tasks, posture)
print(f'    OSQP status: {res.info.status}  iters: {res.info.iter}  obj: {res.info.obj_val:.6f}')
check(res.info.status in ('solved', 'solved inaccurate'), 'OSQP solved')

z = res.x
qdd_sol = z[wbc.qdd0:wbc.qdd0 + nv]
lam_sol = z[wbc.lam0:wbc.lam0 + 3*4].reshape(4, 3)
tau_sol = z[wbc.tau0:wbc.tau0 + n_act]

print(f'    |q̈| max  = {np.max(np.abs(qdd_sol)):.4e}')
print(f'    λ per foot (N):\n{lam_sol}')
print(f'    Σ λ_z    = {lam_sol[:, 2].sum():.4f}  vs  m·g = {mtot * (-m_fb.g[2]):.4f}')
print(f'    τ (Nm)   = {tau_sol}')
print(f'    |τ| max  = {np.max(np.abs(tau_sol)):.4f}  (limit {wbc.tau_max[0]:.1f})')

# ----- Checks -----
print('\n[2] Steady-state and equilibrium')
check(np.max(np.abs(qdd_sol)) < 0.05, '|q̈| ≈ 0 (no commanded acceleration)',
      f'max |q̈|={np.max(np.abs(qdd_sol)):.4e}')

mg = mtot * (-m_fb.g[2])
check(abs(lam_sol[:, 2].sum() - mg) < 1e-3,
      'Σ λ_z = m·g', f'Σλ_z={lam_sol[:, 2].sum():.4f}, mg={mg:.4f}')

check(np.all(lam_sol[:, 2] > 0), 'all λ_z > 0 (compression)')

mu = wbc.mu
fric_ok = all(abs(lam_sol[i,0]) <= mu*lam_sol[i,2] + 1e-6 and abs(lam_sol[i,1]) <= mu*lam_sol[i,2] + 1e-6
              for i in range(4))
check(fric_ok, f'friction cone |λ_xy| ≤ μ λ_z  (μ={mu})')

check(np.all(np.abs(tau_sol) <= wbc.tau_max + 1e-6),
      '|τ| ≤ τ_max', f'max|τ|={np.max(np.abs(tau_sol)):.3f}')

# Dynamics residual: M q̈ + b − Sᵀ τ − Σ Jᵀ λ = 0
M = m_fb.inertia(q_fb0)
b = m_fb.bias(q_fb0, qd_fb0)
J_all = m_fb.jacob(foot_keys, q_fb0)
S_T = np.zeros((nv, n_act)); S_T[6:, :] = np.eye(n_act)
rhs = S_T @ tau_sol
for i in range(4):
    rhs += J_all[3*i:3*(i+1), :].T @ lam_sol[i]
lhs = M @ qdd_sol + b
print(f'\n[3] Dynamics residual')
print(f'    |M q̈ + b − Sᵀτ − Σ Jᵀλ| = {np.linalg.norm(lhs - rhs):.4e}')
check(np.linalg.norm(lhs - rhs) < 1e-3,
      'dynamics equality satisfied',
      f'|res|={np.linalg.norm(lhs - rhs):.4e}')

# ----- summary -----
print()
if fails == 0:
    print('\033[32mAll checks passed.\033[0m'); sys.exit(0)
else:
    print(f'\033[31m{fails} check(s) failed.\033[0m'); sys.exit(1)
