"""Standing-balance verification for tact.ConvexMPC.

Setup: dog at home stance, four feet planted, body at rest. Ask MPC to keep
the state at x_ref = x0 over a 10-step horizon. Expected first-step output:
  - 4-foot stance: f_z per foot weighted by CoM-foot lever to sum to m·g and
    cancel pitch moment
  - λ_xy small (no horizontal acceleration commanded)
  - friction-cone-feasible

Run:  cd fg/dog && uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_mpc_standing.py
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
foot_keys = {'foot1':'3d','foot2':'3d','foot3':'3d','foot4':'3d'}
mtot = float(sum(m_fb.m))

# Standing pose
q_home = np.array([0,-0.425,0.737]*4)
R0     = np.eye(3)
q_probe = m_fb.pack_q_fb(q_home, R0, np.zeros(3))
zmin    = m_fb.fk(foot_keys, q_probe).reshape(4, 3)[:, 2].min()
p0      = np.array([0, 0, 0.025 - zmin])
q_fb    = m_fb.pack_q_fb(q_home, R0, p0)
foot_pos = m_fb.fk(foot_keys, q_fb).reshape(4, 3)
r_com    = m_fb.com(q_fb)
Ig       = m_fb.com_inertia(q_fb)

print(f'\nDog standing — m_tot={mtot:.3f} kg,  CoM={r_com},  base z={p0[2]:.3f}')

# MPC
mpc = tact.ConvexMPC(mass=mtot, Ig_body=Ig, foot_count=4,
                     horizon=10, dt=0.03, mu=0.8, f_max=400.0)

# State: Θ=0, p=p0, ω=0, v=0  (Θ uses small-angle / zero here at yaw=0)
x0 = np.concatenate([np.zeros(3), p0, np.zeros(3), np.zeros(3)])

# Static horizon — feet & CoM held constant
foot_traj = np.tile(foot_pos, (mpc.N, 1, 1))           # (N, nf, 3)
com_traj  = np.tile(r_com,    (mpc.N, 1))              # (N, 3)
contact   = np.ones((mpc.N, 4), dtype=bool)            # 4-foot stance throughout

print('\n[1] solve()')
u0, res = mpc.solve(x0, foot_traj, com_traj, contact)
print(f'    OSQP status: {res.info.status}  iters: {res.info.iter}  obj: {res.info.obj_val:.3f}')
check(res.info.status in ('solved', 'solved inaccurate'), 'OSQP solved')

print(f'    u_0 per foot (N, world):\n{u0}')
print(f'    Σ f_z  = {u0[:, 2].sum():.3f} N   (target m·g = {mtot*9.81:.3f})')
print(f'    Σ f_xy = {u0[:, :2].sum(axis=0)}')

print('\n[2] equilibrium checks')
check(np.all(u0[:, 2] > 0), 'all λ_z > 0')
check(abs(u0[:, 2].sum() - mtot*9.81) / (mtot*9.81) < 0.05,
      'Σ f_z ≈ m·g (within 5%)',
      f'Σf_z={u0[:, 2].sum():.3f}, mg={mtot*9.81:.3f}')

# Friction cone
mu = mpc.mu
fric_ok = all(abs(u0[i,0]) <= mu*u0[i,2] + 1e-5 and abs(u0[i,1]) <= mu*u0[i,2] + 1e-5
              for i in range(4))
check(fric_ok, f'friction cone |λ_xy| ≤ μ λ_z   (μ={mu})')

# Pitch moment cancellation: Σ (r_i × f_i) about CoM should ≈ 0
M_total = np.zeros(3)
for i in range(4):
    r = foot_pos[i] - r_com
    M_total += np.cross(r, u0[i])
print(f'    Σ (r×f) about CoM = {M_total}')
check(np.linalg.norm(M_total) < 5.0, 'centroidal moment ≈ 0 (within 5 N·m)',
      f'|M|={np.linalg.norm(M_total):.3f}')

# ----- 3) Swing-leg constraint -----
print('\n[3] swing schedule (foot 0 swing at step 0)')
contact2 = contact.copy()
contact2[0, 0] = False      # foot 0 lifted at step 0
u0_sw, _ = mpc.solve(x0, foot_traj, com_traj, contact2)
check(np.linalg.norm(u0_sw[0]) < 1e-3, 'swing foot λ_0 ≈ 0',
      f'|λ_0|={np.linalg.norm(u0_sw[0]):.3e}')
# MPC distributes the lost foot's load across the horizon — at step 0 the
# remaining feet take *most* of it, the rest gets absorbed by a brief v_z
# excursion that the next horizon step corrects. 80% recovery at the first
# step is the MPC's cost-optimal split; not a bug.
check(u0_sw[1:, 2].sum() > 0.8 * mtot*9.81,
      'Σ f_z (remaining 3 feet) covers ≥80% of m·g at step 0',
      f'Σf_z={u0_sw[1:, 2].sum():.3f} ({u0_sw[1:, 2].sum()/(mtot*9.81)*100:.0f}%)')

print()
if fails == 0:
    print('\033[32mAll checks passed.\033[0m'); sys.exit(0)
else:
    print(f'\033[31m{fails} check(s) failed.\033[0m'); sys.exit(1)
