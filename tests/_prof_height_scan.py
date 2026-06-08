#!/usr/bin/env -S uv run python
"""height_scan oracle cost measurement — raycast phase 2 pre-work.

The only hot height_scan consumer is dog's StairsPolicy oracle mode: one
7x5(+1 ref) scan per 50 Hz policy tick = 36 tact_raycast_query calls, each
re-running _fk + shape-pose cache build. Question: is batching that loop
into one C call (per-ray-origin tact_raycast_world) worth doing?

Measures, on the live `./start dog -e stairs` scene:
  (a) one full height_scan (current 36-single-query loop), flat + stairs region
  (b) single raycast cost (loop body)
  (c) batch amortization: tact_raycast_frame at n=36 vs n=360 from a frame
      -> slope = per-ray cost, intercept ~= fk + cache + call overhead
      -> predicted batched-scan cost ~= intercept + 36*slope
  (d) env.step for budget context (50 Hz tick = 20 steps at dt=1ms)
"""
import sys, time, pathlib, ctypes
FG = pathlib.Path(__file__).resolve().parents[2]   # .../fg
sys.path.insert(0, str(FG))
import os; os.chdir(FG / 'dog')                    # dog.yml relative paths
import numpy as np, tact
from tact._clib import clib, _DBL

env = tact.Env('dog', render=False)
env.add(f'{tact.pkg_dir}/envs/stairs')

# StairsPolicy grid (dog_stairs_policy.npz scan_x/scan_y)
gx, gy = np.meshgrid([-0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8],
                     [-0.3, -0.15, 0., 0.15, 0.3], indexing='ij')
offsets = np.stack([gx.ravel(), gy.ravel()], axis=-1)   # (35, 2)

def t_ms(f, reps):
    f()                                   # warm
    t0 = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t0) / reps * 1e3

n_rc = int(np.count_nonzero(env.m._build_craycast))
print(f"scene: nb={len(env.m.parent)} nq={len(env.q)} n_shape={len(env.m.ctype)} "
      f"raycast-on={n_rc}")

# (a) full scan: flat region (over the big floor box) and mid-stairs
for tag, xy in [('flat  xy=(0,0)', (0.0, 0.0)), ('stairs xy=(8,0)', (8.0, 0.0))]:
    ms = t_ms(lambda xy=xy: env.height_scan(xy, 0.0, offsets), 200)
    print(f"(a) height_scan {tag}: {ms:7.4f} ms  ({env.last['n_valid']}/35 valid)")

# (b) single ray — the old per-query loop body (n=1 tact_raycast_world call;
# Env.raycast was inlined into height_scan 2026-06-06, so go via clib)
R0 = np.ascontiguousarray([[8.0, 0.0, 100.0]]); Rd = np.ascontiguousarray([[0.0, 0.0, -1.0]])
t1 = np.empty(1)
def one_ray():
    q = np.ascontiguousarray(env.q)
    clib.tact_raycast_world(env.m._h, q.ctypes.data_as(_DBL),
                            R0.ctypes.data_as(_DBL), Rd.ctypes.data_as(_DBL),
                            ctypes.c_int(1), t1.ctypes.data_as(_DBL))
ms1 = t_ms(one_ray, 2000)
print(f"(b) single raycast:        {ms1:7.4f} ms  (x36 = {36*ms1:7.4f} ms)")

# (c) batch slope/intercept through an existing frame (foot1). Dirs in a small
# cone so the frustum cull stays active like a real sensor batch.
# (Env._raycast_frame was inlined into lidar_frames 2026-06-06 — go via clib.)
rng = np.random.default_rng(0)
def cone_dirs(n):
    v = rng.normal(size=(n, 3)) * 0.15 + np.array([0.0, 0.0, -1.0])
    return v / np.linalg.norm(v, axis=1, keepdims=True)
d36, d360 = cone_dirs(36), cone_dirs(360)
def frame_rays(dirs):
    t = np.empty(len(dirs))
    q = np.ascontiguousarray(env.q)
    clib.tact_raycast_frame(env.m._h, q.ctypes.data_as(_DBL),
                            ctypes.c_int(env.m.fdict['foot1']),
                            dirs.ctypes.data_as(_DBL), ctypes.c_int(len(dirs)),
                            t.ctypes.data_as(_DBL))
    return t
b36  = t_ms(lambda: frame_rays(d36), 500)
b360 = t_ms(lambda: frame_rays(d360), 200)
slope = (b360 - b36) / (360 - 36)
icept = b36 - 36 * slope
print(f"(c) batch n=36: {b36:7.4f} ms   n=360: {b360:7.4f} ms")
print(f"    per-ray {slope*1e3:6.2f} us | fk+cache+call {icept*1e3:6.2f} us"
      f" | predicted batched scan ~ {icept + 36*slope:7.4f} ms")

# (d) sim step for budget context
u = np.zeros(env.dof)
for _ in range(300): env.step(u)          # settle on the floor
ms_step = t_ms(lambda: env.step(u), 1000)
tick = 20 * ms_step                       # 50 Hz policy tick = 20 steps @ 1 ms
scan = t_ms(lambda: env.height_scan((8.0, 0.0), 0.0, offsets), 200)
print(f"(d) env.step: {ms_step:7.4f} ms -> 50 Hz tick = {tick:7.4f} ms sim")
print(f"    oracle scan overhead/tick: {scan:7.4f} ms = "
      f"{scan / (tick + scan) * 100:5.1f}% of tick "
      f"(batched est {icept + 36*slope:6.4f} ms = "
      f"{(icept + 36*slope) / (tick + icept + 36*slope) * 100:5.1f}%)")
