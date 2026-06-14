"""GOS finger surface-taxel tactile smoke test.

Uses the 2x8 surface-taxel arrays in gos.yml. Direct finger-to-finger face
contact should distribute force across the whole pad; a narrower probe box
should produce a partial taxel footprint.

Run: UV_CACHE_DIR=/tmp/uv-cache uv run --offline python tests/test_gos_tactile_sensor.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0


def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond:
        fails += 1


GOS_YML = '/home/ubuntu/fg/gos/gos'
if not os.path.exists(GOS_YML + '.yml'):
    print(f'SKIP: {GOS_YML}.yml not found')
    raise SystemExit(0)


probe = os.path.join(tempfile.gettempdir(), 'gos_tactile_probe')
with open(probe + '.yml', 'w') as f:
    f.write("""materials:
  mat2: {normal: [30000, 300], tangent: [20000, 200, 0.8], spin: [100, 1, 0.02], roll: [100, 1, 0.005], restitution: 0.0}
bodies:
  - name: root
  - name: probe
    joint: {type: free, parent: root, q0: [0, -0.3533, 1.4971, 0, 0, 0]}
    inertial: {mass: 0.1, tensor: [box, 0.19, 0.08, 0.08]}
    shapes: [{type: box, param: [0.095, 0.04, 0.04], contact: [1, mat2], rgba: [0.2, 0.6, 0.9, 1]}]
""")

env = tact.Env(GOS_YML, render=False)

check([t['name'] for t in env.tactiles] == ['finger1_taxel', 'finger2_taxel'],
      'GOS tactile specs registered')
check(all(t['n'] == 16 for t in env.tactiles), 'each finger has 2x8 = 16 taxels')
check(all('cell' in t and np.allclose(t['cell'], [0.01875, 0.02]) for t in env.tactiles),
      'each finger tactile has surface cell geometry')

env.reset()
env.q[6] = 0.0
env.q[7] = 0.0
env.qd[:] = 0.0
env.step()
contacts, _ = env.m.contact_reports()
check(len(contacts) >= 4, 'direct finger contact reports a face manifold',
      f'n_contacts={len(contacts)}')

env.cnt = 10
frames = dict(env.tactile_frames())
for name in ('finger1_taxel', 'finger2_taxel'):
    check(name in frames, f'{name} direct-contact frame published')
    arr = np.frombuffer(frames[name], '<f4').reshape(16, 1)
    normal = arr[:, 0]
    check(arr.shape == (16, 1), f'{name} payload shape is (16,1)')
    check(float(normal.sum()) > 0.0 and int((normal > 1e-6).sum()) == 16,
          f'{name} distributes face contact across all taxels',
          f'sum={normal.sum():.3f} max={normal.max():.3f} nonzero={(normal > 1e-6).sum()}')

env = tact.Env(GOS_YML, render=False)
env.add(probe)
env.step()
contacts, _ = env.m.contact_reports()
check(len(contacts) >= 2, 'probe contacts both finger pads', f'n_contacts={len(contacts)}')

env.cnt = 10
frames = dict(env.tactile_frames())
for name in ('finger1_taxel', 'finger2_taxel'):
    check(name in frames, f'{name} frame published')
    arr = np.frombuffer(frames[name], '<f4').reshape(16, 1)
    normal = arr[:, 0]
    check(arr.shape == (16, 1), f'{name} payload shape is (16,1)')
    check(float(normal.sum()) > 0.0 and 4 <= int((normal > 1e-6).sum()) < 16,
          f'{name} sees a partial surface footprint',
          f'sum={normal.sum():.3f} max={normal.max():.3f} nonzero={(normal > 0).sum()}')

if fails:
    raise SystemExit(fails)
print('\nOK')
