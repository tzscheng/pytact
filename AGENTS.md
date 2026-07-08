# AGENTS.md

This file provides guidance to Codex and Claude Code when working with code in this repository.

This is the core toolkit guide. Keep it as boot context only; deep design detail
lives in `docs/`, and per-change history lives in git.

## What this is

`tact` is the simulator identity and native engine. Python distribution:
`pytact`; import namespace: `tact`.

Layout:

- outer `tact/`: dev/build/test/docs and package metadata
- inner `tact/tact/`: import package, runtime assets, `bin/libtact.so`
- `native/`: C sources/headers built by `make` to `native/lib/libtact.so`
- `extras/`: repo-local launchers/tools and MuJoCo backend assets that are not
  part of the Python package (`start`, `mjenv.cpp`, `mjcf/`, `envs/`)

Python modules:

- `_clib.py`: package-local native loader and ctypes bindings
- `rbd.py`: pure math/dynamics/contact/raycast helpers
- `sim.py`: `Model`, `Env`, `CEnv`
- `control.py`: controllers, gait helpers, estimators

Consumer robot projects (in `/home/ubuntu/fg`) provide `Controller` + YAML/XML
and run through the shared `start` launcher, which lives at the consumer repo
root (`fg/start`) — not here. It imports `tact` and locates this repo's assets
(`extras/mjenv.so`, `extras/mjcf/`, `extras/envs/`) via `tact.pkg_dir`. Active
backends: built-in tact sim, MuJoCo, real hardware CEnv/eio. Chrono `-c` is
removed.

## Working Preferences

1. Korean technical conversation with English code terms.
2. Honest trade-off assessment over advocacy.
3. Show scope + risk before destructive or broad multi-file changes.
4. Prefer incremental milestones with verification.
5. Be precise about competing tools; hedge when unsure.
6. When asked strategically, answer whether this is worth doing now.

## Build / run

```bash
make                         # native/lib/libtact.so + native/demos/basic/bin-test
make debug                   # no-opt/gdb-friendly build
extras/build.sh              # extras/mjenv.so for start -m
tact/demos/basic/runner.py arm2
tests/perf/build.sh          # matmul microbenchmarks
```

This repo is its own uv project (`pyproject.toml`/`uv.lock` here, packaged as
`pytact`) — run Python through `uv` from `/home/ubuntu/tact`. `start` uses
`#!/usr/bin/env -S uv run python`, so it discovers whichever workspace encloses
the launch directory: this one when run from inside tact, or a consumer's (e.g.
`/home/ubuntu/fg`, which reaches tact via its `fg/tact` symlink) when launched
from a robot project there. After cache warmup, prefer:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline python ...
```

`_clib.py` should resolve `libtact.so` from inner-package
`<package_dir>/bin/libtact.so`; no system install is expected in this checkout.

## Shared `start` (lives at `fg/start` in the consumer repo)

Run from a project directory:

```bash
./start module                 # built-in tact sim
./start -m module              # MuJoCo, module.xml or xml/module.xml
./start -x [arg] module        # real hardware via ./eio.so
./start -y other module        # override YAML name
./start -e env module          # add environment scene
./start -d dispatch.txt module # replay timestamped commands
./start -f 4 module            # physics ticks per controller update
./start -t 16 module           # redraw/render interval in physics ticks
./start -l module              # headless/no render window
```

Current `start` has no `-s` speed flag or wall-clock sleep pacing. Real hardware
pacing lives in eio/backend code; sim advances as fast as Python/render allows.

The argv tail names the Python module. It must define:

```python
class Controller:
    n_u: int
    n_y: int
    def __init__(self, env, ymlname, prefix='', rate=None, verbose=False): ...
    def msgproc(self, w): ...
    def update(self, y): ...  # (tau, q_ref, qd_ref, kp, kd)
```

The 5-tuple is the general joint command. Torque-only returns
`(tau, None, None, None, None)`. References require matching gains; missing
gains raise in `Model.step`/`CEnv.step`.

For tact `Env`, `rate = int(round(1/env.dt)) // arg.f`. For CEnv, `rate=None`.

## IPC

`start` binds ZMQ IPC sockets under `/dev/shm`:

- PULL `ipc:///dev/shm/default`: commands; `quit` and `reset` are handled in
  `start`, everything else goes to `controller.msgproc(w)`.
- PUB `ipc:///dev/shm/proprio` with `ZMQ_CONFLATE`: float32-packed feed `y`.
- PUB `ipc:///dev/shm/<camera-name>` / `<lidar-name>` for declared sensors.
- Without `-d`, commands are logged to `/dev/shm/out.txt` for replay.

Full wire format: `docs/runtime.md`.

## YAML / Assets

Full schema: `docs/yaml-schema.md`.

Quick rules:

- top-level: `sim`, `view`, `lights`, `materials`, `bodies`, optional
  `cameras`, `lidars`, `feeds`
- joint params include `damping`, `spring`, `frictionloss`, `armature`, `limit`,
  `q0`; PD gains are control outputs, not YAML plant params
- mesh `file:` paths resolve relative to the YAML file directory
- reusable repo-local env scenes live in `extras/envs/`; packaged demos in
  inner-package `demos/`

Runtime assets are addressed via `tact.pkg_dir/<asset>`.

## Code Routing

- Native engine: `native/`; make output `native/lib/libtact.so`; Python
  packaging separately builds package-local `tact/bin/libtact.so`.
- Contact/narrowphase details: `docs/design-contact.md`,
  `docs/design-hfield.md`, `docs/design-lcp-perf.md`,
  `docs/design-joint-friction.md`.
- Pure-step/ctx design: `docs/design-pure-step.md`.
- Backend contract: `docs/backend-interface.md`.
- Runtime/IPC: `docs/runtime.md`.
- Tests: `tests/`; frozen test scenes: `tests/scenes/`.

When editing:

- `native/lcp.c` owns production LCP/PGS contact solve.
- `native/narrow.c`, `box_box.c`, `mpr.c`, `shape.c`, `ray.c` own collision and
  raycast behavior.
- `native/render.c` owns GLFW/EGL rendering and camera encoding.
- `extras/mjenv.cpp` owns MuJoCo backend behavior.
- `tact/rbd.py`, `sim.py`, `control.py` own Python math/sim/control.

## Conventions

- `lcp` is production; `minimal` is sphere/ground test-only; removed `penalty`
  YAML should raise migration errors.
- `Model.step` should remain referentially transparent; solver warm-start state
  is threaded through `ctx`/`SolverState`.
- Dynamic `add`/`delete`/`groups` are tact-`Env` only, not CEnv. Demo:
  `demos/topology/demo_delete.py`.
- Per-body DoF count is not one slot per body: fixed=0, rev/lin=1, free=6.
- Controllers speak active-only vectors.
- Mesh `cshape[i][0]` is an internal slot id, not a filename.
- `_/` is archive/scratch; do not edit unless asked.
