"""Verify tact.MiniElevationMap: per-cell Kalman fusion, robot-following shift,
drift inflation, gating, and the yaw-frame relative height scan (the live
counterpart of dog/rl/dog_stairs_env._height_scan).

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_elevation_map.py
"""
import sys; sys.path.insert(0, '/home/ubuntu/uv/fg')
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

rng = np.random.default_rng(0)
CELL, NOISE = 0.04, 0.01

# Synthetic graded staircase along +x (dog_stairs_env-style: run-up, 0.3 m treads).
RUN_START, STEP_DEPTH, STEP_H, N_STEPS = 1.0, 0.3, 0.04, 10
def terrain(x):
    k = np.clip(np.ceil((np.asarray(x, dtype=float) - RUN_START) / STEP_DEPTH),
                0, N_STEPS)
    return STEP_H * k

def edge_dist(x):
    """Distance from x to the nearest step edge (for excluding quantization cells)."""
    edges = RUN_START + STEP_DEPTH * np.arange(N_STEPS + 1)
    return np.abs(np.asarray(x, dtype=float)[..., None] - edges).min(axis=-1)

def sense(m, bx, by=0.0, n_pts=3000, span_x=(-0.6, 1.0), span_y=(-1.0, 1.0)):
    """Simulated depth frame: noisy points on the true terrain around the robot."""
    x = bx + rng.uniform(*span_x, n_pts)
    y = by + rng.uniform(*span_y, n_pts)
    z = terrain(x) + rng.normal(0.0, NOISE, n_pts)
    m.insert(np.stack([x, y, z], axis=-1))

# dog_stairs_env's 7x5 scan grid
gx, gy = np.meshgrid([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
                     [-0.3, -0.15, 0.0, 0.15, 0.3], indexing='ij')
OFFSETS = np.stack([gx.ravel(), gy.ravel()], axis=-1)

def analytic_scan(bx, by, yaw):
    """Ground truth via the same math as dog_stairs_env._height_scan."""
    c, s = np.cos(yaw), np.sin(yaw)
    wx = bx + c * OFFSETS[:, 0] - s * OFFSETS[:, 1]
    return terrain(wx) - terrain(bx), wx

# ----- 1) Flat ground: convergence + full validity -----
print('\n[1] Flat ground fusion')
m = tact.MiniElevationMap(cell_size=CELL)
for _ in range(10):
    m.move_to([0.0, 0.0]); sense(m, 0.0)
h = m.sample([0.0, 0.0], 0.0, OFFSETS)
check(bool(m.last['valid'].all()), 'all 35 scan points valid')
check(float(np.abs(h).max()) < 0.01, 'flat -> |h| < 1 cm', f'max={np.abs(h).max():.4f}')
check(float(m.P[m.valid].mean()) < NOISE**2, 'fused variance < single-shot noise^2',
      f'meanP={m.P[m.valid].mean():.2e}')

# ----- 2) Stairs walkthrough vs analytic _height_scan -----
print('\n[2] Stairs walkthrough (vs analytic scan, edge cells excluded)')
m = tact.MiniElevationMap(cell_size=CELL)
worst = 0.0
for bx in np.arange(0.0, 3.501, 0.05):
    m.move_to([bx, 0.0]); sense(m, bx)
    if bx >= 1.0 and edge_dist(bx) > 1.5 * CELL:   # base ref cell off the edges too
        gt, wx = analytic_scan(bx, 0.0, 0.0)
        h = m.sample([bx, 0.0], 0.0, OFFSETS)
        far = edge_dist(wx) > 1.5 * CELL           # quantization-free comparison set
        worst = max(worst, float(np.abs(h - gt)[far].max()))
        if not m.last['valid'].all():
            check(False, f'validity lost at bx={bx:.2f}'); break
else:
    check(True, 'all scan points valid along the climb')
check(worst < 0.02, 'scan matches analytic within 2 cm off-edge', f'worst={worst:.4f}')

# rear points (-0.4 m) come from map memory, not the current frame
m2 = tact.MiniElevationMap(cell_size=CELL)
for bx in np.arange(0.0, 2.501, 0.05):
    m2.move_to([bx, 0.0]); sense(m2, bx, span_x=(0.0, 1.0))   # forward-only sensor
h = m2.sample([2.5, 0.0], 0.0, OFFSETS)
gt, wx = analytic_scan(2.5, 0.0, 0.0)
rear = OFFSETS[:, 0] < 0
far = edge_dist(wx) > 2.0 * CELL    # rear cells are stale (drift-inflated) -> looser
check(bool(m2.last['valid'][rear].all()), 'rear points valid from memory (forward-only sensor)')
check(float(np.abs(h - gt)[rear & far].max()) < 0.03, 'rear heights correct from memory',
      f'max={np.abs(h - gt)[rear & far].max():.4f}')

# ----- 3) Yawed query -----
print('\n[3] Yawed scan (yaw=0.5 rad)')
gt, wx = analytic_scan(2.0, 0.0, 0.5)
h = m.sample([2.0, 0.0], 0.5, OFFSETS)
far = edge_dist(wx) > 1.5 * CELL
check(bool(m.last['valid'].all()), 'all points valid under yaw')
check(float(np.abs(h - gt)[far].max()) < 0.02, 'yawed scan matches analytic',
      f'worst={np.abs(h - gt)[far].max():.4f}')

# ----- 4) z-drift immunity (common-mode cancellation) -----
print('\n[4] Constant odom z-offset cancels in the relative scan')
ma, mb = tact.MiniElevationMap(cell_size=CELL), tact.MiniElevationMap(cell_size=CELL)
for bx in np.arange(1.5, 2.501, 0.05):
    rng_state = rng.bit_generator.state
    ma.move_to([bx, 0.0]); sense(ma, bx)
    rng.bit_generator.state = rng_state            # identical clouds, shifted in z
    x = bx + rng.uniform(-0.6, 1.0, 3000); y = rng.uniform(-1.0, 1.0, 3000)
    z = terrain(x) + rng.normal(0.0, NOISE, 3000) + 0.5
    mb.move_to([bx, 0.0]); mb.insert(np.stack([x, y, z], axis=-1))
ha = ma.sample([2.5, 0.0], 0.0, OFFSETS)
hb = mb.sample([2.5, 0.0], 0.0, OFFSETS)
check(np.allclose(ha, hb, atol=1e-9), 'relative scan identical under +0.5 m z drift',
      f'maxdiff={np.abs(ha - hb).max():.2e}')

# ----- 5) Map shift: stale cells scroll off, unseen ahead -> default + invalid -----
print('\n[5] Shift / invalidation semantics')
m3 = tact.MiniElevationMap(cell_size=CELL, map_size=2.0)
m3.move_to([0.0, 0.0]); sense(m3, 0.0)
m3.move_to([10.0, 0.0])                            # jump past the whole map, no sensing
h = m3.sample([10.0, 0.0], 0.0, OFFSETS, default=0.0)
check(not m3.last['valid'].any() and not m3.last['base_valid'],
      'fully unseen region -> all invalid')
check(bool((h == 0.0).all()), 'invalid points return default')

# ----- 6) Drift inflation + gated re-init -----
print('\n[6] Variance inflation and outlier gate')
m4 = tact.MiniElevationMap(cell_size=CELL)
m4.move_to([0.0, 0.0]); sense(m4, 0.0)
p_before = float(m4.P[m4.valid].mean())
m4.move_to([0.5, 0.0])                             # move without sensing
p_after = float(m4.P[m4.valid].mean())
check(p_after > p_before, 'variance grows with traveled distance',
      f'{p_before:.2e} -> {p_after:.2e}')
# terrain suddenly changes by 0.5 m: gate must re-init, not slowly average
patch = np.stack([rng.uniform(-0.08, 0.08, 400), rng.uniform(-0.08, 0.08, 400),
                  np.full(400, 0.5)], axis=-1)
m4.insert(patch)
habs, ok = m4._interp(np.array([0.0]), np.array([0.0]))
check(bool(ok[0]) and abs(float(habs[0]) - 0.5) < 0.02,
      'gated re-init tracks a 0.5 m terrain change in one frame',
      f'h={float(habs[0]):.3f}  n_reinit={m4.last["n_reinit"]}')

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
