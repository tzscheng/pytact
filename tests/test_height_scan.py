"""Verify Env.height_scan(): the sim-only GT terrain scan, contract-compatible
with tact.MiniElevationMap.sample() (yaw-frame offsets, under-base-relative
heights, invalid -> default, validity in .last).

Run:  uv run --no-project python tact/tests/test_height_scan.py
"""
import sys, os.path
# fg dir (= parent of the tact package this file lives in), NOT a hardcoded
# workspace path — the repo root differs per machine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import os, tempfile
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

# Scene: floor (top z=0) + two steps + a raycast:false "robot" slab hovering at
# (0,0) — the scan below it must read the floor, not the slab.
STEP1, STEP2 = 0.06, 0.10                    # box tops
YML = f"""materials:
    mat1: {{normal: [20000, 200], tangent: [20000, 200, 1.0], spin: [200, 2, 0.05], roll: [200, 2, 0.02], restitution: 0.0}}
bodies:
  - name: root
    shapes:
      - {{type: box, pos: [0, 0, -0.5], param: [10, 10, 0.5], rgba: [0.7, 0.7, 0.7, 1.0]}}
      - {{type: box, pos: [1.0, 0, {STEP1 / 2}], param: [0.15, 2.0, {STEP1 / 2}], rgba: [0.8, 0.8, 0.8, 1.0]}}
      - {{type: box, pos: [2.0, 0, {STEP2 / 2}], param: [0.15, 2.0, {STEP2 / 2}], rgba: [0.8, 0.8, 0.8, 1.0]}}
      - {{type: box, pos: [0, 0, 0.5], param: [0.3, 0.1, 0.05], rgba: [0.3, 0.3, 0.6, 1.0], raycast: false}}
"""
base = os.path.join(tempfile.gettempdir(), 'height_scan_test')
open(base + '.yml', 'w').write(YML)
env = tact.Env(base, render=False)
env.reset()


def terrain(x):                              # analytic GT of the scene above
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x - 1.0) <= 0.15, STEP1,
                    np.where(np.abs(x - 2.0) <= 0.15, STEP2, 0.0))


gx, gy = np.meshgrid([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
                     [-0.3, -0.15, 0.0, 0.15, 0.3], indexing='ij')
OFFSETS = np.stack([gx.ravel(), gy.ravel()], axis=-1)

# ----- 1) flat-base scan vs analytic (exact) -----
print('\n[1] scan vs analytic terrain')
h = env.height_scan([0.7, 0.0], 0.0, OFFSETS)
gt = terrain(0.7 + OFFSETS[:, 0]) - terrain(0.7)
check(bool(env.last['valid'].all()) and env.last['base_valid'], 'all points + base valid')
check(np.allclose(h, gt, atol=1e-9), 'heights match analytic exactly',
      f'max|e|={np.abs(h - gt).max():.2e}')

# ----- 2) yawed scan -----
print('\n[2] yawed scan (0.5 rad)')
yaw = 0.5
h = env.height_scan([0.7, 0.0], yaw, OFFSETS)
c, s = np.cos(yaw), np.sin(yaw)
gt = terrain(0.7 + c * OFFSETS[:, 0] - s * OFFSETS[:, 1]) - terrain(0.7)
check(np.allclose(h, gt, atol=1e-9), 'yawed heights match analytic',
      f'max|e|={np.abs(h - gt).max():.2e}')

# ----- 3) under-base relativization: base on a step -> floor reads negative -----
print('\n[3] base on step2 -> relative heights')
h = env.height_scan([2.0, 0.0], 0.0, np.array([[0.0, 0.0], [0.5, 0.0]]))
check(abs(h[0]) < 1e-9 and abs(h[1] + STEP2) < 1e-9,
      'under-base 0, floor reads -step2', f'h={h}')

# ----- 4) raycast:false robot slab is invisible -----
print('\n[4] robot exclusion (raycast: false)')
h = env.height_scan([0.0, 0.0], 0.0, np.array([[0.0, 0.0]]))
check(abs(h[0]) < 1e-9 and env.last['base_valid'],
      'scan under the hovering slab reads the floor', f'h={h[0]:+.3f}')

# ----- 5) miss handling: floor-less scene -----
print('\n[5] floor-less scene -> invalid + default')
open(base + '_void.yml', 'w').write(YML.split('shapes:')[0] + 'shapes:\n'
      + '      - {type: box, pos: [0, 0, 0.5], param: [0.3, 0.1, 0.05], rgba: [0.3, 0.3, 0.6, 1.0], raycast: false}\n')
env_v = tact.Env(base + '_void', render=False)
env_v.reset()
h = env_v.height_scan([0.0, 0.0], 0.0, OFFSETS, default=0.0)
check(not env_v.last['valid'].any() and not env_v.last['base_valid'],
      'all invalid, base invalid')
check(bool((h == 0.0).all()), 'invalid points return default')

# ----- 6) provider-contract parity with MiniElevationMap.sample -----
print('\n[6] parity with MiniElevationMap (same consumer, two providers)')
m = tact.MiniElevationMap(cell_size=0.04, sigma_meas=0.001)
rng = np.random.default_rng(0)
px = rng.uniform(-0.5, 3.0, 120000); py = rng.uniform(-1.0, 1.0, 120000)
m.move_to([0.7, 0.0])
m.insert(np.stack([px, py, terrain(px)], axis=-1))      # noise-free GT cloud
h_map = m.sample([0.7, 0.0], 0.3, OFFSETS)
h_gt = env.height_scan([0.7, 0.0], 0.3, OFFSETS)
edge = np.minimum.reduce([np.abs(np.abs((0.7 + np.cos(0.3) * OFFSETS[:, 0]
        - np.sin(0.3) * OFFSETS[:, 1]) - cx0) - 0.15) for cx0 in (1.0, 2.0)])
far = edge > 0.06                                        # off step edges (cell quantization)
check(bool(m.last['valid'].all()) and bool(env.last['valid'].all()),
      'both providers fully valid')
check(np.allclose(h_map[far], h_gt[far], atol=0.02),
      'map and GT providers agree off-edge',
      f'max|e|={np.abs(h_map - h_gt)[far].max():.4f}')
check(set(m.last) >= {'valid', 'base_valid', 'ref'} and
      set(env.last) >= {'valid', 'base_valid', 'ref'},
      'identical .last contract keys')

# ----- 7) hfield terrain (hf1) -----
print('\n[7] hfield scan (hf1 vs grid bilinear)')
env_h = tact.Env(f'{tact.pkg_dir}/examples/hf1', render=False)
env_h.reset()
H = tact.load_hfield(f'{tact.pkg_dir}/hfields/hf1.bin')   # (101,101), 0.1 m, [-5,5]
def npy_z(x, y):
    gx, gy = (np.asarray(x) + 5) / 0.1, (np.asarray(y) + 5) / 0.1
    i0, j0 = np.floor(gy).astype(int), np.floor(gx).astype(int)
    fy, fx = gy - i0, gx - j0
    return (H[i0, j0] * (1 - fx) * (1 - fy) + H[i0, j0 + 1] * fx * (1 - fy)
            + H[i0 + 1, j0] * (1 - fx) * fy + H[i0 + 1, j0 + 1] * fx * fy)
h = env_h.height_scan([1.5, -0.8], 0.7, OFFSETS)
c, s = np.cos(0.7), np.sin(0.7)
wx = 1.5 + c * OFFSETS[:, 0] - s * OFFSETS[:, 1]
wy = -0.8 + s * OFFSETS[:, 0] + c * OFFSETS[:, 1]
gt = npy_z(wx, wy) - npy_z(1.5, -0.8)
check(bool(env_h.last['valid'].all()), 'all points valid on hfield')
check(np.allclose(h, gt, atol=0.03), 'matches grid bilinear (triangulation tol)',
      f'max|e|={np.abs(h - gt).max():.4f}')

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
