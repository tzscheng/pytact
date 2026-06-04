"""Verify Env.raycloud(): the 3D point-cloud twin of Env.raymap().

The Python-side ray reconstruction must mirror tact_raymap_query's (tact.c)
per-pixel ray generation exactly — angular & pinhole projections and the -90°
optical roll. The decisive check is geometric: points back-projected from
raymap ranges and transformed by the frame's PLAIN fkh pose must land exactly
on known world surfaces (a floor plane, an offset box). Any ray-direction or
roll mismatch pushes them off-surface.

Run:  uv run --no-project python /home/ubuntu/uv/fg/tact/tests/test_raycloud.py
"""
import sys; sys.path.insert(0, '/home/ubuntu/uv/fg')
import os, tempfile
import numpy as np, tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0
def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond: fails += 1

# Scene: floor slab (top at z=0) + a small box ahead-left (+x, +y) for an
# orientation-asymmetry check, + a forward-looking lidar pitched 30° down.
BOX_C, BOX_HS = np.array([1.5, 0.8, 0.25]), np.array([0.2, 0.2, 0.25])
YML = f"""materials:
    mat1: {{normal: [20000, 200], tangent: [20000, 200, 1.0], spin: [200, 2, 0.05], roll: [200, 2, 0.02], restitution: 0.0}}
bodies:
  - name: root
    shapes:
      - {{type: box, pos: [0, 0, -0.5], param: [10, 10, 0.5], rgba: [0.7, 0.7, 0.7, 1.0]}}
      - {{type: box, pos: [{BOX_C[0]}, {BOX_C[1]}, {BOX_C[2]}], param: [{BOX_HS[0]}, {BOX_HS[1]}, {BOX_HS[2]}], rgba: [0.8, 0.3, 0.3, 1.0]}}
lidars:
  - {{name: lid, type: 2d, body: root, pos: [0, 0, 0.6], euler: [0, -60, 0], eulerseq: xyz, res: [64, 48], dth: 1.0, fps: 30}}
"""
base = os.path.join(tempfile.gettempdir(), 'raycloud_test')
open(base + '.yml', 'w').write(YML)
env = tact.Env(base, render=False)
env.reset()
W, H, DTH = 64, 48, 1.0
Te = env.m.fkh(['lid'], env.q)[0]


def world(pts):
    return pts @ Te[:3, :3].T + Te[:3, 3]


def on_box(pw, tol=1e-6):
    return (np.abs(pw - BOX_C) <= BOX_HS + tol).all(axis=1)


for mode, pinhole in (('angular', False), ('pinhole', True)):
    print(f'\n[{mode}]')
    D = env.raymap('lid', W, H, DTH, pinhole=pinhole)
    pts = env.raycloud('lid', W, H, DTH, pinhole=pinhole)
    d = D.ravel()
    hit = d >= 0.0

    # 1) reprojection identity: same hit count, |point| == range (order-preserving)
    check(len(pts) == int(hit.sum()), f'hit count {len(pts)} == raymap non-(-1) count')
    check(np.allclose(np.linalg.norm(pts, axis=1), d[hit], atol=1e-12),
          '|point| equals raymap range per pixel')

    # 2) world-surface test: every point lies on the floor plane (z=0) or the box
    pw = world(pts)
    box = on_box(pw)
    zerr = np.abs(pw[~box, 2])
    check(bool(box.any()), 'box is seen', f'{int(box.sum())} pts')
    check(float(zerr.max()) < 1e-9, 'all non-box points on floor plane z=0',
          f'max|z|={zerr.max():.2e}  n={int((~box).sum())}')

    # 3) orientation asymmetry: above-floor points exist ONLY inside the +x,+y box
    above = pw[:, 2] > 1e-6
    check(bool(above.any()) and bool(box[above].all()),
          'above-floor points all inside the box (+x,+y) — roll/orientation correct',
          f'{int(above.sum())} pts, y_mean={pw[above, 1].mean():+.2f}')

    # 4) max_range filter
    pts_r = env.raycloud('lid', W, H, DTH, pinhole=pinhole, max_range=2.0)
    check(len(pts_r) < len(pts) and
          float(np.linalg.norm(pts_r, axis=1).max()) <= 2.0 + 1e-12,
          'max_range=2.0 drops far hits', f'{len(pts)} -> {len(pts_r)}')

# 5) ray cache: repeated calls reuse the cached grid (same object)
r1 = env._raycloud_rays(W, H, DTH, False)
r2 = env._raycloud_rays(W, H, DTH, False)
check(r1 is r2, 'ray grid cached per (w,h,dth,pinhole)')

# 6) end-to-end: raycloud -> MiniElevationMap.insert -> sample reads the scene.
# Single-frame semantics: near field is dense (1 deg/px ~ a few cm on the ground),
# the FAR field is sparser than the 4 cm grid (grazing rays) -> correctly invalid,
# and the ground under a forward-looking sensor is out of FoV -> base_valid False.
# Multi-frame fusion while moving (the map's whole point) fills both.
print('\n[raycloud -> MiniElevationMap]')
m = tact.MiniElevationMap(cell_size=0.04, sigma_meas=0.005)
m.move_to([0.0, 0.0])
m.insert(world(env.raycloud('lid', W, H, DTH)))
probe = np.array([[0.6, -0.2], [1.0, 0.0],                      # near floor (dense)
                  [BOX_C[0] - 0.1, BOX_C[1]],                   # box
                  [2.5, -0.5]])                                  # far floor (sparse)
h = m.sample([0.0, 0.0], 0.0, probe)
v = m.last['valid']
check(bool(v[:3].all()), 'near-field + box probes valid from one frame')
check(abs(h[0]) < 0.01 and abs(h[1]) < 0.01, 'floor probes read ~0',
      f'h={h[0]:+.4f}, {h[1]:+.4f}')
check(h[2] > 0.3, 'box probe reads raised terrain', f'h={h[2]:+.3f} (top=0.5)')
check(not v[3] and h[3] == 0.0,
      'far grazing probe correctly invalid -> default (needs multi-frame fusion)')
check(not m.last['base_valid'],
      'under-base out of forward FoV in a single frame (map memory fills it while walking)')

# 7) wire format: lidars `type: 3d` -> lidar_frames() -> zstd (N,3) f32 roundtrip
print('\n[3d (pointcloud) wire type]')
import zstandard
YML_PC = YML.replace(
    "{name: lid, type: 2d, body: root, pos: [0, 0, 0.6], euler: [0, -60, 0], eulerseq: xyz, res: [64, 48], dth: 1.0, fps: 30}",
    "{name: lid, type: 3d, body: root, pos: [0, 0, 0.6], euler: [0, -60, 0], eulerseq: xyz, res: [64, 48], dth: 1.0, fps: 30, max_range: 2.0}")
base_pc = os.path.join(tempfile.gettempdir(), 'raycloud_test_pc')
open(base_pc + '.yml', 'w').write(YML_PC)
env_pc = tact.Env(base_pc, render=False)
env_pc.reset()
frames = dict(env_pc.lidar_frames())
check('lid' in frames, "lidar_frames() publishes the pointcloud lidar")
dec = np.frombuffer(zstandard.decompress(frames['lid']), '<f4').reshape(-1, 3)
ref = env_pc.raycloud('lid', W, H, DTH, max_range=2.0).astype('<f4')
check(dec.shape == ref.shape and np.array_equal(dec, ref),
      'wire roundtrip == raycloud(max_range from spec), f32-exact',
      f'N={len(dec)}')
check(float(np.linalg.norm(dec.astype(np.float64), axis=1).max()) <= 2.0 + 1e-6,
      'spec max_range honored on the wire')
try:
    open(base_pc + '_bad.yml', 'w').write(YML.replace('type: 2d', 'type: voxel'))
    tact.Env(base_pc + '_bad', render=False)
    check(False, 'unknown lidar type rejected at parse')
except ValueError:
    check(True, 'unknown lidar type rejected at parse')

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
