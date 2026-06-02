#!/usr/bin/env -S uv run python
"""Multi-biped (zen) LCP scaling — box feet → box-box manifold, higher nc/robot."""
import sys, os, time
sys.path.insert(0, '/home/ubuntu/uv/fg')
os.chdir('/home/ubuntu/uv/fg/zen')
import numpy as np, tact

ZEN = '/home/ubuntu/uv/fg/zen/zen'
FLOOR = '/tmp/bigfloor'

for K in [1, 2, 4, 8, 12]:    # MAX_NB=256: zen=15 bodies → up to ~17 zen
    env = tact.Env(FLOOR, render=False)
    for i in range(K):
        env.add(ZEN, name=f'zen{i}', offset=[i*0.8 - (K-1)*0.4, 0, 0, 0, 0, 0])
    u = np.zeros(env.dof)
    for _ in range(300):
        env.step(u)
    t0 = time.perf_counter()
    for _ in range(1000):
        env.step(u)
    dt = (time.perf_counter() - t0) / 1000 * 1e3
    print(f"K={K:2d} zen | nq={len(env.q):4d} n_pair={env.m.cpair.shape[0]:4d} | {dt:8.4f} ms/step")
