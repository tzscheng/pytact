"""Verify FootstepPlanner (Raibert) + BezierSwing.

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_planner_swing.py
"""
import sys; sys.path.insert(0, '/home/ubuntu/uv/fg')
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

# ----- 1) FootstepPlanner: hip anchoring at zero velocity -----
print('\n[1] FootstepPlanner: at v=0, v_des=0 the step lies exactly at hip projection')
hips = np.array([
    [+0.22,  +0.13, -0.38],   # FL
    [+0.22,  -0.13, -0.38],   # FR
    [-0.34,  +0.13, -0.38],   # RL
    [-0.34,  -0.13, -0.38],   # RR
])
fp = tact.FootstepPlanner(hips, k=0.03, foot_radius=0.025)
R = np.eye(3); p = np.array([0.0, 0.0, 0.5])
v = np.zeros(3); vd = np.zeros(3)
for i in range(4):
    p_step = fp.target(i, R, p, v, vd, T_stance=0.25)
    expected = hips[i] + p
    check(np.allclose(p_step, expected), f'foot{i+1} hip projection',
          f'p_step={p_step}, expected={expected}')

# ----- 2) FootstepPlanner: forward velocity advances the step -----
print('\n[2] FootstepPlanner: forward v adds 0.5·v·T_stance')
v = np.array([0.4, 0.0, 0.0]); vd = np.zeros(3)
T_stance = 0.25
p_step = fp.target(0, R, p, v, vd, T_stance=T_stance)
expected = hips[0] + p + 0.5 * v * T_stance + 0.03 * (v - vd)
check(np.allclose(p_step, expected), 'forward step shift',
      f'p_step={p_step}, expected={expected}')

# ----- 3) FootstepPlanner: velocity-error correction term -----
print('\n[3] FootstepPlanner: v_des cancels v exactly when equal → no error term')
v = np.array([0.4, 0.0, 0.0]); vd = v.copy()
p_step = fp.target(0, R, p, v, vd, T_stance=T_stance)
expected = hips[0] + p + 0.5 * v * T_stance     # k·(v−vd)=0
check(np.allclose(p_step, expected), 'v == v_des ⇒ correction = 0')

# ----- 4) FootstepPlanner: rotation maps hip correctly -----
print('\n[4] FootstepPlanner: body rotation rotates the hip anchor')
# 90° yaw: x-body → y-world
R_yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
v = np.zeros(3); vd = np.zeros(3)
p_step = fp.target(0, R_yaw90, p, v, vd, T_stance=T_stance)
expected = R_yaw90 @ hips[0] + p
check(np.allclose(p_step, expected), 'hip rotated by yaw=90°')

# ----- 5) FootstepPlanner: env.get_z overrides z -----
print('\n[5] FootstepPlanner: env terrain snaps z')
class FlatGround:
    def get_z(self, x, y): return 0.1   # flat at 0.1
v = np.zeros(3); vd = np.zeros(3)
p_step = fp.target(0, R, p, v, vd, T_stance=T_stance, env=FlatGround())
check(abs(p_step[2] - (0.1 + 0.025)) < 1e-12, 'z = get_z + foot_radius',
      f'z={p_step[2]}, expected={0.1+0.025}')

# ----- 6) BezierSwing: endpoints exact -----
print('\n[6] BezierSwing: endpoints')
sw = tact.BezierSwing(height=0.06)
p0 = np.array([0.2, 0.13, 0.025])
p1 = np.array([0.3, 0.13, 0.025])
check(np.allclose(sw.eval(0.0, p0, p1), p0), 'B(0) == start')
check(np.allclose(sw.eval(1.0, p0, p1), p1), 'B(1) == end')

# ----- 7) BezierSwing: midpoint apex height -----
# B(0.5) = 0.25·p0 + 0.5·apex + 0.25·p1, apex = mid(p0,p1)+(0,0,h) so
# B(0.5) = mid(p0,p1) + (0, 0, h/2)
print('\n[7] BezierSwing: midpoint = mid + (0,0,h/2)')
mid = sw.eval(0.5, p0, p1)
expected_mid = 0.5*(p0+p1) + np.array([0, 0, 0.03])
check(np.allclose(mid, expected_mid), 'B(0.5) = mid + (0,0,h/2)',
      f'B(0.5)={mid}, expected={expected_mid}')

# ----- 8) BezierSwing: derivative consistency -----
print('\n[8] BezierSwing: eval_vel matches finite-diff')
T_swing = 0.25
for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
    v_analytic = sw.eval_vel(s, T_swing, p0, p1)
    # finite-diff at t = s · T_swing
    eps = 1e-6
    s_lo = max(0.0, s - eps); s_hi = min(1.0, s + eps)
    v_fd = (sw.eval(s_hi, p0, p1) - sw.eval(s_lo, p0, p1)) / ((s_hi - s_lo) * T_swing)
    err = np.linalg.norm(v_analytic - v_fd)
    check(err < 1e-4, f'eval_vel @ s={s} matches finite-diff', f'|err|={err:.3e}')

# Endpoint velocities have nice closed forms:
# B'(0) = 2(apex − start) ⇒ v(0) = 2·(apex − start)/T_swing
# B'(1) = 2(end − apex)   ⇒ v(1) = 2·(end − apex)/T_swing
apex = 0.5*(p0+p1) + np.array([0,0,0.06])
v0 = 2*(apex - p0)/T_swing
v1 = 2*(p1 - apex)/T_swing
check(np.allclose(sw.eval_vel(0.0, T_swing, p0, p1), v0), 'v(0) closed-form')
check(np.allclose(sw.eval_vel(1.0, T_swing, p0, p1), v1), 'v(1) closed-form')

# ----- summary -----
print()
if fails == 0:
    print('\033[32mAll checks passed.\033[0m'); sys.exit(0)
else:
    print(f'\033[31m{fails} check(s) failed.\033[0m'); sys.exit(1)
