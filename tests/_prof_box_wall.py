#!/usr/bin/env -S uv run python
"""Throwaway profiling harness for box_wall LCP bottleneck analysis.
Runs N headless steps; prints n_pair, mean/max nc, mean iters, and wall time.
Intended to be run under `perf record` for a symbol-level breakdown."""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                # → fg/tact
sys.path.insert(0, os.path.dirname(ROOT))   # → fg
os.chdir(ROOT)

import tact

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

env = tact.Env('tests/scenes/box_wall', render=False)   # frozen fixture (see tests/scenes/)
u = np.zeros(env.dof)
print(f"nq={len(env.q)} dof={env.dof} n_pair={env.m.cpair.shape[0]}")

# warm up one step (JIT-free C, but settle handle)
env.step(u)

t0 = time.perf_counter()
for k in range(N):
    env.step(u)
dt_wall = time.perf_counter() - t0
print(f"{N} steps in {dt_wall:.3f}s  ->  {dt_wall/N*1e3:.4f} ms/step")
