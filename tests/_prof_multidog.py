#!/usr/bin/env -S uv run python
"""Multi-robot LCP scaling test. Loads a big floor + K dogs, settles, times.
With dense Delassus the per-step cost should grow ~K^3; the question is whether
the real multi-walking-robot use case actually hits that regime."""
import sys, os, time
sys.path.insert(0, '/home/ubuntu/uv/fg')
os.chdir('/home/ubuntu/uv/fg/dog')   # so dog.yml relative paths resolve
import numpy as np, tact

DOG = '/home/ubuntu/uv/fg/dog/dog'
FLOOR = '/tmp/bigfloor'

for K in [1, 2, 4, 8, 12]:   # MAX_NB=256: dog=17 bodies → up to ~15 dogs
    env = tact.Env(FLOOR, render=False)
    for i in range(K):
        env.add(DOG, name=f'dog{i}', offset=[i*0.8 - (K-1)*0.4, 0, 0, 0, 0, 0])
    u = np.zeros(env.dof)
    for _ in range(300):                      # settle
        env.step(u)
    t0 = time.perf_counter()
    for _ in range(1000):
        env.step(u)
    dt = (time.perf_counter() - t0) / 1000 * 1e3
    print(f"K={K:2d} dogs | nq={len(env.q):4d} n_pair={env.m.cpair.shape[0]:4d} | {dt:8.4f} ms/step")
