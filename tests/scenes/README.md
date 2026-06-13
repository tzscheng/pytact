# tests/scenes — frozen test fixtures

These YAMLs (and `objs/` meshes) are the **frozen scene fixtures** for the test
suite. They are copies of files from the live scene trees (`demos/`, `extras/envs/`),
but deliberately decoupled:

- `demos/*.yml` / `extras/envs/*.yml` are **mutable** — meant to be tweaked, retuned, and changed.
- `tests/scenes/*.yml` are **test inputs** — they must NOT drift, or golden
  baselines (`tests/regression/baseline/*.npy`) silently break and the tests
  stop meaning what they meant.

Consumers (all load from here, not `demos/`/`extras/envs/`):

- `tests/regression/capture_baseline.py` + `test_traj.py` — bit-identical golden
  regression over `SCENARIOS`.
- `tests/test_pure_step.py` — Model.step pure-function invariants.

## Rules

1. **Do not edit these to "fix" a test.** If a test fails, fix the code.
2. **Changing a fixture is a deliberate act**: edit the YAML here, then re-run
   `uv run python tests/regression/capture_baseline.py <name>` to re-capture the
   golden baseline, and commit both together (reviewed). Never auto-recapture.
3. To pull a fresh copy of a scene after improving it in `demos/`/`extras/envs/`, copy it
   here explicitly and re-capture — that is the only way the test set tracks an
   example change.

Mesh note: `obj1.yml` references `objs/5.obj` and `objs/big_disk.obj`, resolved
against this directory — so `objs/` lives here too.
