# Regression tests

Trajectory-baseline tests that catch unintended physics changes. Each baseline
is a `(n_steps + 1, nq)` numpy array of `q` values from a deterministic
zero-input headless sim of a bundled example YAML.

## Usage

```bash
# Run the regression check (after rebuild). Exits 1 on mismatch.
uv run python tests/regression/test_traj.py

# Or via pytest (same logic, prettier output)
uv run pytest tests/regression/test_traj.py -v

# Re-capture baselines (do this only when an intentional, reviewed change
# alters physics — commit the new .npy files alongside the change).
uv run python tests/regression/capture_baseline.py

# Re-capture a single scenario
uv run python tests/regression/capture_baseline.py box_wall
```

## What's covered

`SCENARIOS` in `capture_baseline.py` lists the bundled YAMLs included:

| scenario | nq | what it tests |
|---|---|---|
| `box_wall` | 108 | 4-row brick wall — box-box manifold (Phase 2) |
| `mini_wall_box` | 18 | 3-brick mini wall — box-box manifold edge cases |
| `wall5_box` | 30 | 5-brick 2-row wall — box-box manifold |
| `obj1` | 6 | floating mesh+box on inclined plane — mesh narrowphase (MPR fallback) |
| `sphere_test` | 6 | sphere on floor — sphere narrowphase (MPR) |
| `cartpole/cartpole` | 2 | cart-pole — no contact, pure rigid-body |
| `arm2`, `arm3` | 2, 3 | small arms — gravity-driven swing |

500 steps × dt=1ms = 0.5 s sim each. Full suite runs in ~5 s.

## ATOL

`test_traj.py` uses `np.allclose(atol=1e-12, rtol=0)`. This is essentially
bit-identical — any meaningful float-arithmetic reordering will fail the
check. The intent is: changes that shouldn't affect physics (refactors,
performance tweaks that preserve algorithms) must produce identical
trajectories. If your change is intentional, re-capture the baseline.

## When to re-capture

Run `capture_baseline.py` and commit the resulting `.npy` files alongside
your code change in these cases:

- Physics algorithm change (new integrator, new contact model, new narrowphase
  for a shape combo, etc.)
- Numerical tweak (CFM scale, Baumgarte ERP, PGS iters, friction cone discretization)
- Bug fix that corrects previously-wrong dynamics

Don't re-capture for:

- Performance optimization that's supposed to preserve algorithm (then test
  must pass — failure means you broke it)
- Refactor that's supposed to be bit-identical (same reason)
- Test infrastructure changes (just verify the suite still runs)

## How baselines were initially captured

Phase 2 box-box manifold landed and stability was verified. Baselines were
captured at that point. Phase 1 → Phase 2 transition was the largest
physics-affecting change since these baselines exist; future changes are
expected to be small.
