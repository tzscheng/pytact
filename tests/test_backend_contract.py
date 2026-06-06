"""Backend core-contract conformance test (docs/backend-interface.md).

Self-validating, no baseline. Asserts:
  1. the CORE marker in docs/backend-interface.md matches this file's CORE list
  2. tact Env + a fake-cdll CEnv provide all core members with the right kinds
     (callable / attribute type), per the N=6 contract
  3. step/reset return contract (shape, dtype), reset zero-step guarantee,
     and the dt=None path for backends without get_dt
  4. ledger classification: CEnv has NO sensor-publishing members (absence is
     legitimate — no empty stubs), and add/delete/groups fail loudly

Run:  uv run --no-project python tact/tests/test_backend_contract.py
"""
import sys, os, re, ctypes
# fg dir (= parent of the tact package this file lives in), NOT a hardcoded
# workspace path — the repo root differs per machine.
TACT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TACT))
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

CORE = ['step', 'reset', 'finish', 'backend', 'has_pd', 'dt']
SENSOR_NAMES = ['cameras', 'lidars', 'camera_frames', 'lidar_frames']
TACT_ONLY = SENSOR_NAMES + ['add', 'delete', 'groups']   # CEnv: absent AND blocked from cdll forwarding

# 1) doc <-> code: the ledger doc's CORE marker is the single source of truth
print('\n[doc marker]')
doc = open(os.path.join(TACT, 'docs', 'backend-interface.md')).read()
m = re.search(r'<!-- CORE: (.*?) -->', doc)
check(m is not None and m.group(1).split() == CORE,
      'docs/backend-interface.md CORE marker == test CORE list',
      m.group(1) if m else 'marker missing')

# 2) tact Env satisfies the core contract
print('\n[Env (tact)]')
env = tact.Env(f'{tact.pkg_dir}/examples/arm2', render=False)
for n in CORE:
    check(hasattr(env, n), f'Env has {n!r}')
check(env.backend == 'tact' and isinstance(env.backend, str), 'backend is the str label')
check(isinstance(env.has_pd, (bool, np.bool_)), 'has_pd is bool')
check(isinstance(env.dt, float) and env.dt == env.m.dt, 'dt is float (= m.dt)', f'dt={env.dt}')
y0 = env.reset()
check(isinstance(y0, np.ndarray) and y0.dtype == np.float64, 'reset() returns float64 ndarray y')
check(np.array_equal(env.q, env.m.q0) and env.cnt == 0,
      'reset zero-step guarantee (q == q0, cnt == 0 — integrator not advanced)')
y1 = env.step()
check(y1.shape == y0.shape and y1.dtype == y0.dtype,
      'step() y matches reset() y in shape/dtype', f'shape={y1.shape}')
check(env.finish() is None, 'finish() is callable (tact no-op)')
# tact-only capabilities present (ledger: 비목표/단일 backend rows)
check(all(hasattr(env, n) for n in SENSOR_NAMES + ['add', 'delete', 'height_scan']),
      'tact-only capabilities live on Env')
# get_z removed 2026-06-06 (sim-trick reduction): absolute world-z has no
# real-robot counterpart — height_scan (base-relative) is the only terrain query.
check(not hasattr(env, 'get_z'), 'get_z REMOVED from Env (absolute-z oracle)')

# 3) CEnv satisfies the core contract — fake cdll, no .so needed.
#    The fake's reset writes a sentinel and counts step calls, so the zero-step
#    guarantee is observable (reset must NOT advance via step).
print('\n[CEnv (fake cdll)]')
N_Y, N_U = 5, 2
calls = {'step': 0, 'reset': 0}

def fake_step(tau, q_ref, qd_ref, ybuf):
    calls['step'] += 1
    for i in range(N_Y): ybuf[i] = 10.0 + i
    return 0

def fake_reset(ybuf):
    calls['reset'] += 1
    for i in range(N_Y): ybuf[i] = 1.0 + i

class FakeCdll:                      # attribute-assignable like ctypes.CDLL,
    step, reset = staticmethod(fake_step), staticmethod(fake_reset)
    # but raises AttributeError for undeclared names (= dlsym failure): no
    # get_z/get_dt/set_redraw -> CEnv's init probes must take the absent path.

cenv = tact.CEnv(FakeCdll(), n_y=N_Y, n_u=N_U, backend='real', has_pd=False)
for n in CORE:
    check(hasattr(cenv, n), f'CEnv has {n!r}')
check(cenv.backend == 'real' and cenv.has_pd is False, 'backend/has_pd reflect ctor args')
check(cenv.dt is None, 'dt is None when backend exports no get_dt (contract allows None)')
y0 = cenv.reset()
check(isinstance(y0, np.ndarray) and y0.dtype == np.float64 and np.array_equal(y0, 1.0 + np.arange(N_Y)),
      'reset() returns the C-side post-reset y')
check(calls['step'] == 0, 'reset zero-step guarantee (no step() call behind reset)')
y1 = cenv.step()
check(y1.shape == y0.shape and np.array_equal(y1, 10.0 + np.arange(N_Y)),
      'step() y matches reset() y in shape', f'shape={y1.shape}')
check(calls['step'] == 1, 'step() called the backend exactly once')

# 4) ledger classification on CEnv: tact-only names are ABSENT (hasattr False —
#    no stubs) and BLOCKED from raw cdll forwarding (a future backend .so
#    exporting a same-named symbol must not become a silent wrong call).
print('\n[CEnv capability classification]')
for n in TACT_ONLY:
    check(not hasattr(cenv, n),
          f'{n!r} ABSENT on CEnv (hasattr False; absence is legitimate)')
try:
    cenv.add; check(False, 'blocked names raise a pointered AttributeError')
except AttributeError as e:
    check('tact-only' in str(e), 'blocked names raise a pointered AttributeError', str(e)[:64])
# get_z is blocked from forwarding (mjenv.so still exports the C symbol — a stale
# caller must fail fast, not silently garbage-marshal through an argtypes-less ptr)
try:
    cenv.get_z; check(False, 'get_z blocked on CEnv with a height_scan pointer')
except AttributeError as e:
    check('height_scan' in str(e), 'get_z blocked on CEnv with a height_scan pointer', str(e)[:60])
# forwarding check: a non-blocked attribute that EXISTS on the cdll comes through
# (set_abf = a real per-robot eio command, the canonical example)
FakeCdll.set_abf = staticmethod(lambda: 42)
check(cenv.set_abf() == 42, 'per-robot cdll symbols still forward via __getattr__ (eio channel)')

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
