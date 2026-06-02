"""Verify tact.GaitScheduler for the standard quadruped + biped gaits.

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_gait_scheduler.py
"""
import sys; sys.path.insert(0, '/home/ubuntu/uv/fg')
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

# ----- 1) Trot semantics -----
print('\n[1] Trot  period=0.5  duty=0.5  offsets=[0, 0.5, 0.5, 0]')
g = tact.GaitScheduler(0.5, 0.5, [0.0, 0.5, 0.5, 0.0])

# At t=0: FL (offset 0) and RR (offset 0) start stance; FR and RL start swing.
c, sp = g.update(0.0)
print(f'    t=0.000  contact={c}  swing_phase={sp}')
check(c.tolist() == [True, False, False, True],         't=0: FL,RR stance / FR,RL swing')
check(np.allclose(sp[[0, 3]], 0.0),                      't=0: stance feet swing_phase = 0')
check(np.allclose(sp[[1, 2]], 0.0),                      't=0: swing feet just started (phase 0)')

# Mid-cycle (t=0.125): everyone is mid-phase
c, sp = g.update(0.125)
print(f'    t=0.125  contact={c}  swing_phase={sp}')
check(np.allclose(sp[1], 0.5) and np.allclose(sp[2], 0.5), 'swing feet at swing_phase=0.5 mid-swing')

# At t=0.25: roles swap
c, sp = g.update(0.25)
print(f'    t=0.250  contact={c}  swing_phase={sp}')
check(c.tolist() == [False, True, True, False],          't=0.25: roles swapped')

# Periodicity
c0, sp0 = g.update(0.123)
c1, sp1 = g.update(0.123 + 0.5)
check(np.array_equal(c0, c1) and np.allclose(sp0, sp1),  'periodicity update(t) == update(t+period)')

# At every instant, trot has exactly 2 feet in contact
for t in np.linspace(0, 1, 41):
    c, _ = g.update(t)
    if c.sum() != 2:
        check(False, f'trot has 2 contacts at all t (failed @ t={t:.3f}: {c})'); break
else:
    check(True,  'trot has exactly 2 feet in contact at every t')

# stance/swing durations
check(np.allclose(g.stance_duration(), 0.25) and np.allclose(g.swing_duration(), 0.25),
      'stance_duration = swing_duration = 0.25 s')

# time_in_phase + time_to_next_event = (current phase duration)
t = 0.18
elapsed = g.time_in_phase(t)
to_next = g.time_to_next_event(t)
phase_dur = np.where(g.update(t)[0], g.stance_duration(), g.swing_duration())
check(np.allclose(elapsed + to_next, phase_dur),
      'time_in_phase + time_to_next_event = current phase duration')

# ----- 2) Pace -----
print('\n[2] Pace  period=0.5  duty=0.5  offsets=[0, 0.5, 0, 0.5]  (FL+RL together)')
g = tact.GaitScheduler(0.5, 0.5, [0.0, 0.5, 0.0, 0.5])
c, _ = g.update(0.0)
check(c.tolist() == [True, False, True, False], 't=0: left side stance, right side swing')

# ----- 3) Bound -----
print('\n[3] Bound  period=0.4  duty=0.5  offsets=[0, 0, 0.5, 0.5]  (front pair / rear pair)')
g = tact.GaitScheduler(0.4, 0.5, [0.0, 0.0, 0.5, 0.5])
c, _ = g.update(0.0)
check(c.tolist() == [True, True, False, False], 't=0: front stance, rear swing')

# ----- 4) Static crawl (1 foot at a time) -----
print('\n[4] Crawl  period=2.0  duty=0.75  offsets=[0, 0.5, 0.75, 0.25]')
g = tact.GaitScheduler(2.0, 0.75, [0.0, 0.5, 0.75, 0.25])
# duty=0.75 → swing fraction 0.25, four offsets spaced by 0.25 → exactly one swing at a time
fail_t = None
for t in np.linspace(0, 4, 401):
    c, _ = g.update(t)
    if c.sum() != 3:
        # at exact boundary it can be 4 (two feet at touchdown/lift-off simultaneously)
        if c.sum() == 4: continue
        fail_t = t; break
check(fail_t is None, 'crawl has 3 feet in contact almost everywhere',
      f'(violation at t={fail_t})' if fail_t is not None else '')

# ----- 5) Per-foot heterogeneous params -----
print('\n[5] Heterogeneous per-foot params')
g = tact.GaitScheduler(period=[0.4, 0.4, 0.6, 0.6],
                       duty=[0.5, 0.5, 0.6, 0.6],
                       offsets=[0.0, 0.5, 0.5, 0.0])
sd = g.stance_duration()
check(np.allclose(sd, [0.2, 0.2, 0.36, 0.36]), 'per-foot stance_duration', f'={sd}')

# ----- 6) Edge case: duty=1 always-stance -----
print('\n[6] Edge: duty=1.0 ⇒ always stance')
g = tact.GaitScheduler(0.5, 1.0, [0.0])
for t in np.linspace(0, 3, 31):
    c, sp = g.update(t)
    if not c[0] or sp[0] != 0.0:
        check(False, 'duty=1 always-stance', f'(t={t}, c={c}, sp={sp})'); break
else:
    check(True, 'duty=1 ⇒ contact=True, swing_phase=0 for all t')

# ----- 7) Biped (2-leg) -----
print('\n[7] Biped (2-leg, walking gait — overlapping double support)')
g = tact.GaitScheduler(period=1.0, duty=0.6, offsets=[0.0, 0.5])
# duty=0.6 means stance lasts 60% — two feet overlap in stance for 0.6+0.6-1.0=0.2 fraction
# At t=0: foot0 starts stance, foot1 is at phase 0.5 → 0.5 < 0.6 ⇒ both stance (double support)
c, _ = g.update(0.0)
check(c.tolist() == [True, True], 'biped t=0: double support')
c, _ = g.update(0.4)
# foot0 phase 0.4 (stance); foot1 phase 0.9 (swing) — single support
check(c.tolist() == [True, False], 'biped t=0.4: single support (foot1 swing)')

# ----- summary -----
print()
if fails == 0:
    print('\033[32mAll checks passed.\033[0m'); sys.exit(0)
else:
    print(f'\033[31m{fails} check(s) failed.\033[0m'); sys.exit(1)
