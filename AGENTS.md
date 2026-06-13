# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a navigation + conventions doc. Deep design detail lives in `docs/design-*.md`; per-change history lives in git. Point to those rather than duplicating them here.

## What this is

`tact` is the simulator identity and native engine: **tact: a high contact/tactile fidelity dynamics simulator**. The Python distribution name is **`pytact`** (`pip install pytact`), while the Python import namespace remains **`tact`** (`import tact`). Today the Python package wraps and ships the package-local C shared library (`libtact.so`) for collision detection (GJK/CCD), contact dynamics, and OpenGL/EGL rendering. Long-term, `tact` is intended to also stand as a usable native C library with a stable C API, and `pytact` remains the Python wrapper/distribution package.

Current installable layout: the inner `tact/` directory holds the `.py` package files plus runtime assets such as `bin/`, `envs`, and `demos`; the outer project directory holds dev/build/test/docs tooling plus a small root-import shim. If/when the outer directory is renamed, the intended shape is `fg/pytact/tact/`: outer `pytact` = Python distribution project, inner `tact` = import package/native wrapper assets. The **C/C sources + headers live under outer `native/`** and build directly to `tact/bin/libtact.so`. The package implements rigid-body dynamics (Featherstone / Siciliano / Craig), forward kinematics, controllers, and a YAML-driven scene/robot loader, split by file but exposing a flat API via `__init__.py` re-exports:

- `_clib.py` — internal: `libtact.so` loader + ctypes argtypes/restypes (not exposed at top level)
- `rbd.py` — pure math/dynamics primitives: rotation/quaternion helpers, spatial-vector building blocks, Featherstone/Siciliano/Craig algorithms, integrators, contact, ray casts. No state. C-side counterpart is `rbd.c`.
- `sim.py` — simulation classes only: `Model`, `Env`, `CEnv`. Does `from .rbd import *` so inline references stay flat.
- `control.py` — controllers & planning (see Code map). Depends on `rbd` for transform helpers (single-direction: `control → rbd ← sim → _clib`).
- `__init__.py` — `from .rbd import *; from .sim import *; from .control import *`, so callers see a flat `tact.Env`, `tact.PIDController`, `tact.crm`, etc. Explicit-domain access also works: `tact.sim.Env`, `tact.rbd.crm` (same object).

It is consumed by per-robot projects: a bundled example under `demos/cartpole/` and external siblings (`../dg5`, `../dog`, `../gos`, `../h9`, `../h12`, `../kida`, `../mk1`, `../mk2`, `../mk3`, `../zen`). Each supplies its own `Controller` class + YAML and is launched through a uniform `start` script that swaps the backend (built-in sim, MuJoCo, or real hardware) behind a common loop. The Chrono `-c` backend was removed (all chrono projects deactivated). The controller is a low-level execution controller (receives mode commands from outside at low freq, runs the inner control loop at sim rate); high-level control lives outside via ZMQ.

## Working preferences

User works on this project from multiple machines, so preferences live here (in the repo) rather than only in per-machine memory. These shape *how* to interact:

1. **Korean technical conversation with English code terms.** All discussion in Korean. C/Python identifiers, algorithm names (Featherstone, PGS, LCP, MPR, SAT, etc.), and English technical terms ("manifold", "convex", "kinematic tree", "warm-start") stay English. Prefer tables and bullet structures for comparisons.
2. **Honest trade-off assessment over advocacy.** Present trade-offs with skepticism, especially for speculative future capabilities (autodiff, GPU). Don't pile up justifications for architectural commitments unless warranted. When uncertain whether work is worthwhile, ask "is this the right time" rather than enthusiastically endorsing.
3. **Explicit scope + risk before destructive/multi-file changes.** Don't start bulk YAML migrations, deprecations, or refactors touching many files without first showing scope + risk and getting confirmation.
4. **Incremental milestones with verification.** Prefer `(1) → (2) → (3)` ordering with completion checkpoints; run regression / verify after each. Pause for go/no-go between phases.
5. **Be precise about the competing-tools landscape.** Don't oversimplify ("Bullet uses X", "RaiSim uses Y") without checking — the user knows the field. Hedge when unsure.
6. **Strategic over tactical when asked.** When user asks "should we do this AT ALL?", give an honest "is this worth it for *this* project" answer with concrete reasoning.

## Build / run

```bash
make                         # builds tact/bin/libtact.so and build/tools/bin-test
make debug                   # no-opt/gdb-friendly libtact.so + bin-test
make mjenv                   # builds mjenv/mjenv.so from mjenv/mjenv.cpp (auxiliary MuJoCo backend for
                             # cross-checking tact's dynamics; -I/usr/local/include/mujoco -lmujoco -lGL -lglfw)
./yml-test arm2              # quick test loading a YAML model + running env.step in a window
perf/build.sh                # builds C/C++ matmul benchmarks (cmm, eigmm)
```

`libtact.so` goes to package-local `tact/bin/`, and `_clib.py` loads it from `<package_dir>/bin/libtact.so`. The installable distribution is named `pytact`, but it provides `import tact`. The auxiliary MuJoCo backend is intentionally separate: `mjenv/mjenv.so` is loaded only by the repo-local `start -m` path and is not part of the installable package.

Python is run via `uv` from the workspace root `/home/ubuntu/fg` (see `fg/pyproject.toml`, `requires-python >=3.12`). The `start` scripts use `#!/usr/bin/env -S uv run python`, so `uv run` from any project walks up and discovers `fg/pyproject.toml` as the root. `uv.lock` is committed for reproducible dev/test resolution; after a machine has warmed its uv cache once, prefer `UV_CACHE_DIR=/tmp/uv-cache uv run --offline python ...` for tests to avoid network prompts.

`_clib.py` resolves `libtact.so` in order: (1) **next to itself** — `<package_dir>/bin/libtact.so` where `<package_dir>` is the inner `tact` package (`fg/pytact/tact` after the outer project rename, currently `fg/tact/tact` in this checkout); (2) `./bin/libtact.so` in cwd (legacy/ad-hoc development build); (3) the OS loader via `ctypes.util.find_library('tact')`. In the current monorepo layout path (1) succeeds, so no system install is required.

Asset layout (runtime assets under the inner `tact/` import package, discoverable as `tact.pkg_dir/<asset>`):
- `bin/libtact.so` — installable native library (`make` / `make package-lib` output)
- `mjenv/mjenv.cpp`, `mjenv/mjenv.so` — internal MuJoCo backend source/output for `start -m` (not packaged)
- `envs/*.yml` — **background-environment** scene YAMLs (passive terrain a robot is dropped onto: `1.yml`–`5.yml` ground planes, `d3.yml` stepping stones, `hf1.yml` hfield, `stairs.yml`, `box1.yml`/`desk1.yml`). Loaded by `start -e <name>` (searches `envs` then `demos`) and composed by per-project RL envs (`sim.add(f'{tact.pkg_dir}/envs/<name>')`). Height-field **grid data + generators** live here too: `hf1.bin` (MuJoCo's custom hfield binary — int32 nrow, int32 ncol, float32 data; raw meters, row i→+Y col j→+X) ← `hf1_gen.py`. ONE data file per terrain, referenced by BOTH the tact YAML (`type: hfield, file: hf1.bin` → raw meters × `sz`) and the MuJoCo env xml (`mjenv/hf1.xml`, `<hfield file=../tact/envs/hf1.bin>`; MuJoCo-normalized to [0,1] so the xml carries data-derived `size[2]` + geom z, printed by the generator)
- `demos/*.yml`, `demos/*.py` — **tact feature/physics demos** (manipulators `arm2`–`arm4`/`fv`, contact-manifold brick walls `box_wall`/`wall5_box`/`mini_wall_box`, solver tests `sphere_test`/`min_test`, floating objects `obj1`/`obj2`, driver scripts `ball_throw.py`/`raymap_demo.py`/`demo_delete.py`)
- `demos/meshes/*.obj` — shared mesh assets. A mesh shape's YAML `file:` path is resolved by the Python loader against the **YAML's own directory** and passed to the C side (`set_mesh_path` → `load_obj` in `shape.c`; render-side `load_obj_as_mesh` in `render.c`) as an absolute path — so a `demos/` scene references these as `file: meshes/<name>.obj`, and a project can ship its own local mesh dir.
- `demos/cartpole/` — bundled example project

## Per-project run pattern

Inside an example project (e.g. `demos/cartpole/`) or any sibling project:

```bash
./start cartpole             # built-in tact sim, controller module = cartpole.py, yml = cartpole.yml
./start -m cartpole          # MuJoCo backend (mjenv/mjenv.so): robot model = cartpole.xml next to the
                             # controller (xml/ transitional fallback); combine with -e for an environment
./start -x [arg] cartpole    # real hardware via ./eio.so
./start -y other_yml cartpole         # override YAML name
./start -e env_yml cartpole           # add an environment scene on top of robot (tact: yml from cwd /
                                      # yml/ / envs/ / demos/; -m: xml from cwd then the shared tact/mjenv/)
./start -d dispatch.txt cartpole      # replay timestamped commands from a file
./start -k 4 cartpole                 # physics ticks per control update (default 1; ZOH between)
./start -s 2 cartpole                 # 2x realtime (works headless AND GUI)
./start -s 2 -f 30 cartpole           # 2x sim, window redraws at ~30 Hz
./start -l -s 0 cartpole              # headless, free-run (no pacing) — RL/batch
```

**Speed and render rate are two orthogonal knobs** (tact backend): `-s` SPEED (default 1.0, `0`=free-run) sets sim speed; `-f` (default 60) sets the GUI render rate. The window redraw cadence is **derived, never user-set**: `redraw = ceil(speed/(dt*fps))`. Both the tact and mujoco backends derive it this way — mjenv exposes `get_dt`/`set_redraw` so `start` drives its cadence (and uses its dt) identically.

**Real-time pacing**: the main loop sleep-paces wall-clock to `-s`× (anchored to an absolute target; re-anchors if >100 ms behind). Paces both headless and GUI; window vsync stays **on** (`glfwSwapInterval(1)`) and `start` sets `__GL_YIELD=USLEEP` (before GL init) to stop NVIDIA's vblank busy-spin. mujoco paced identically; Chrono/real pace via their own loop/HW. Full rationale → `docs/runtime.md`.

The argv tail (`cartpole`) names a Python module to import — it must define `class Controller: __init__(env, ymlname, prefix='', rate=None, verbose=False)`, `msgproc(w)`, and `update(y) -> (tau, q_ref, qd_ref, kp, kd)` — the **5-tuple command** (the general affine joint feedback law, same shape as the MIT-mode/Unitree MotorCmd actuator packet; servo modes are projections — torque-only returns `tau, None, None, None, None`). Plus `n_u`/`n_y` attributes. Controllers branch on `env.has_pd` (typically cached as `self.has_pd` at init) to choose torque (`tau`) vs position reference (`q_ref`). See `demos/cartpole/cartpole.py`.

## IPC contract (`start` script)

Bound by `start` over ZMQ IPC on `/dev/shm`:
- PULL `ipc:///dev/shm/default` — incoming commands (whitespace-split string). Words `quit`/`reset` handled in `start`; everything else forwarded to `controller.msgproc(w)`.
- PUB `ipc:///dev/shm/proprio` (CONFLATE) — float32 packed proprioceptive vector `y`, sent every 20 sim steps.
- PUB `ipc:///dev/shm/<camera-name>` / `<lidar-name>` per `cameras:`/`lidars:` entry — `env.camera_frames()`/`lidar_frames()` own the rate-gating (`fps` vs `env.cnt`) and `type`→encoder dispatch, yielding `(name, bytes)` for due sensors: rgb→JPEG, depth-camera→zstd float32 (C-side), lidar 2d/3d→**raw float32**. Sim core has no zmq (sockets/send stay in the runner). CEnv backends publish nothing. Full wire format → `docs/runtime.md`.
- Without `-d`, every received command is logged to `/dev/shm/out.txt` with the step counter.

The cartpole subdir has its own (identical) `start`; new projects typically copy this script.

## Code map

### C side (`libtact.so`)

All installable C sources + headers live under **`native/`** (headers beside sources, so `#include "tact.h"` needs no `-I`), built into `tact/bin/libtact.so`. Link deps are only `-lm -ldl`; GL/EGL/GLFW/turbojpeg/zstd are dlopen/dlsym'd on render paths.

- `tact.h` — public interface + constants (`MAX_NB=256`, `MAX_PTS_PER_PAIR=4`, shape codes `MESH=100, BOX=101, SPHERE=102, CYL=103, CAPSULE=104, HFIELD=105`, Euler conventions `EULER_{EXT,INT}_{XYZ,ZYX}`).
- `rbd.c` — linear algebra (matmul, transforms, Euler conversions) + spatial-vector dynamics (ABA, CRB, RNE, LDLᵀ, jacob_whitney) + `choose_rotation`. Python `rbd.py` mirrors it 1:1.
- `shape.h` / `shape.c` — shared shape-asset **slot storage** (collision side): mesh slots (loaded lazily from `.obj` by `load_obj`, set by `set_mesh_path`) and height-field slots (pushed from Python by `set_hfield_data`, no lazy load). `MAX_MESH=64`, `MAX_HFIELD=16` in `shape.h`. Consumed by `mpr.c`, `narrow.c`, `ray.c`, `render.c`. Bounding-sphere helpers `mesh_local_radius`/`hfield_local_radius` live here too.
- `narrow.c` — **narrow-phase dispatch + analytic detectors** per shape pair (box-box, sphere/capsule/cylinder/box closed-forms, full hfield matrix), MPR/EPA fallback (`mpr.c`) for the rest. `collision_check(...)→n_points`, 7 doubles/point `[p(3), n(3), depth]`. Dispatch table → `docs/design-contact.md`.
- `box_box.c` — **box-box contact manifold via SAT (15 axes) + Sutherland-Hodgman face clipping**, exposing `box_box_manifold` (called from `narrow.c`). Algorithm + 4-point selection → `docs/design-contact.md`.
- `mpr.c` — generic **convex narrowphase fallback** (MPR + EPA portal, libccd-derived) for shape pairs with no dedicated detector. Holds the per-shape support function, `size_of_param`, `collision_check_mpr`.
- `ray.c` — **ray-primitive intersections** (`ray_intersects_{triangle,mesh_slot,hfield,box,sphere,cylinder,capsule}`); each returns forward t along Rd or -1. Mesh/hfield read `shape.c` slots (ray transformed into shape-local frame). Hfield uses a **2D DDA grid walk** (first-hit = nearest). Mirrors `rbd.py:ray_intersects_*`.
- `lcp.c` — `contact_lcp`: Stewart-Trinkle/Anitescu PGS, 4 contact cones (normal + tangent disk + spin + roll) + non-contact constraint rows (joint friction/limit), CFM-regularized + Baumgarte. Row-table A-build, warm-start λ per (cpair_idx, sub_id), block-sparsity opts. See `docs/design-lcp-perf.md`, `docs/design-joint-friction.md`.
- `tact.c` — opaque `tact_t` handle (single arena malloc for all parallel arrays + workspaces). `tact_create` packs everything; `tact_step_lcp` runs one timestep; `tact_destroy` frees the arena. The handle holds NO solver state across steps: q/qd are passed in by the caller and the PGS warm-start λ is caller-threaded through `tact_step_lcp(lam_in, lam_out)` — one unified vector `[contact | fric | limit]` (the `SolverState` layout).
- `render.c` — OpenGL rendering. Two entry points: `win_render(...)` (GLFW window, from `env._win_render`) and `egl_render(..., out_buf, opt, req_width, req_height, fovy_deg)` (offscreen EGL; `opt==1`→RGB JPEG via libjpeg-turbo, `opt==2`→linear eye-space metric depth as zstd float32; size-dependent buffers reallocated grow-only; called inline from `env.camera_frames`). Packet header (`MAX_PACKET=16384, HEADER_SIZE=16`) mirrored in Python's `stream_assembler`. Per-slot mesh fingerprints (`prev_type[i]`, `prev_shape[i]`) so dynamic add/delete only rebuilds changed GPU mesh slots.

### Auxiliary MuJoCo backend (`mjenv/mjenv.cpp`)
Thin C++ shim around MuJoCo (`mjModel`/`mjData`, `mjv*` viz, GLFW window). Built into `mjenv/mjenv.so` and dlopened from `start -m`. Its only role is to cross-check `tact`'s dynamics against MuJoCo on the same YAML/XML pair (no per-robot code to host).
- `start -m` loads the robot model as `<module>.xml` next to the controller (`xml/<module>.xml` transitional fallback); `-e <name>` optionally names a **robot-independent environment xml** — searched in the project dir, then shared **`tact/mjenv/`** (`0.xml` = infinite checker plane, floor geom group 1 per the raycast/height_scan convention).
- `init(robot_xml, env_xml, render)` composes them via mjSpec (`mj_parseXML` ×2 + `mjs_attach` of the env worldbody into the robot's world frame, prefix `env_`, then `mj_compile`; env_xml NULL → plain `mj_loadXML`). Robot is the parent spec, so its `<option>`/`<visual>` win and its names stay unprefixed — env files carry assets + worldbody only. The MJX loaders (`dog/rl/dog_mjx.py`, `zen/rl/zen_mjx.py`) mirror this in Python (`spec.attach(..., prefix="env_")`).
- Render cadence + sim speed driven by `start` via exported **`get_dt()`** + **`set_redraw(n)`**, so `-s`/`-f` behave identically to the tact backend.
- **XML convention:** every XML used here must have matched motor↔position actuator pairs per joint (dual-actuator). `start` passes `has_pd=True` for the mujoco backend on that assumption; a torque-only XML requires the controller to set `env.has_pd = False`.
- **Per-step PD gains:** `step(tau, q_ref, qd_ref, kp, kd, y)` — same control-input set as tact's `Env.step`; each step statelessly writes the caller's kp/kd into the position actuators' gainprm/biasprm. The XML's kp/kv are structural placeholders the runtime never reads. Numerical caveat: tact's PD is implicit (unconditionally stable) while MuJoCo's is integrator-dependent explicit, so the same numbers do NOT transfer 1:1 — interface shared, values are the controller's per-backend responsibility.

### Python side (`tact/` package — flat API)
Pure rigid-body math/dynamics in `rbd.py`; stateful simulator classes in `sim.py`; controllers in `control.py`. `rbd.py` exports:
- Rotation/Euler/quaternion/homogeneous conversions. Convention: lowercase eulerseq = extrinsic, uppercase = intrinsic (`xyz`/`XYZ`/`zyx`/`ZYX`). YAML loader default is `XYZ` intrinsic, degrees.
- Spatial-vector building blocks: `crm`, `crf`, `jcalc`, `get_spatial_inertia`, `get_spatial_transform`, `_fk`.
- Dynamics: `crb_featherstone` (mass matrix), `rne_featherstone`/`rne_lwp` (inverse dynamics — spatial vs Luh-Walker-Paul body-frame), `aba_featherstone` (forward dynamics — also folds semi-implicit `ff`/`sk`/joint-PD terms and `armature` rotor inertia into articulated inertia `d`), `inertia_lagrange`, `cc_finitediff`, `gravity_lagrange`, `jacob_whitney`. **`armature`** (MuJoCo-style per-DoF reflected inertia) is added to the diagonal of `d` in ABA and to M after CRB; armature=0 is bit-identical.
- Integrators: `euler_step`, `euler_step2`, `rk4_step`.
- Contact: `contact_lcp` (production solver, `solver: lcp` — calls multi-point `clib.collision_check`, builds J/A/b, runs PGS over 4 cones). `contact_ground_sphere` is **test-only** (`solver: minimal`): explicit spring-damper ground contact for SPHERE shapes only.
- Ray casts: `ray_intersects_{triangle,mesh,box,sphere,cylinder,capsule}` — used by the terrain oracle `env.height_scan` (G+1 vertical rays in one `tact_raycast_world` call) and the lidar publish path (`env._ray_grid` + `tact_raycast_frame`). Single-threaded, brute-force per ray (no broadphase/BVH).

Key classes:
- `Model` — YAML → kinematic tree. Holds parallel arrays `jtype, parent, active, fixed, Ti, m, c, I, q0, qd0, ff, sk` + contact arrays `ctype, cbody, cshape, cparam, ctran, crgba`. Frame registry (`fdict, fbody, ftran`) from `frames:` blocks. `step(q, qd, tau=None, q_ref=None, qd_ref=None, kp=None, kd=None, ctx=None)` runs contact + integrator, returns `(q_next, qd_next, y, ctx_next)`. `kp`/`kd` are per-step implicit joint-PD gains (control inputs, not plant parameters — YAML `k:` removed; the model carries no gains). A reference without its gain raises (`q_ref`→`kp`, `qd_ref`→`kd`); gains without references are inert; `q_ref`+`kp` without `kd` = P-only. **Referentially transparent (pure)**: the PGS warm-start λ is threaded explicitly via `ctx` (a `SolverState` namedtuple holding ONE unified λ vector `[contact | joint-friction | joint-limit]` + read-only block views `lam_contact`/`lam_fric`/`lam_limit`; `ctx=None` = cold start, immutable so fork/branch is safe). `Model.zero_state()` builds a cold ctx. `Env` keeps one in `self._ctx`. Single-thread only. See `docs/design-pure-step.md`. Helpers: `fk`, `fkh`, `jacob`, `error`.
- `Env` — wraps `Model`, owns the render window + an EGL image buffer (1024×768×4). `step(u)` expands `u` (active-only) into joint torques, calls `model.step`, optionally redraws every `redraw` ticks. `add(..., name=)` composes multiple YAMLs into one tree (robot + env scene); `delete(name)` removes a group; `env.groups` lists active groups. (See Dynamic add/delete.)
- Controllers (`control.py`): `PIDController` (vectorized), `JacobianTransposeController` (task-space PD via Jacobian transpose), `JacobianTransposeForce` (task-space feed-forward), `ComputedTorqueController`, `HybridForcePositionController` (selection matrix from contact force normal), `MovingAverageWaypointSmoother` (sliding-window target ramp — see `cartpole.py`), `Envelope` (fixed-pose holder), `StepGenerator2` (biped) / `StepGenerator4` (quadruped). Pinching primitives `pinch2`, `pinch3p`, `pinch3_icra2012`, `pinch3_piony`. Terrain handling consumes only base-relative `env.height_scan` deltas anchored to a stance foot's FK z — no controller may read absolute terrain height (`env.get_z` removed; no real-robot counterpart).
- `stream_assembler` — reassembles fragmented UDP camera frames matching `render.c`'s 16-byte header (`<IHHII`); 200 ms stale-frame timeout.

### YAML scene format

Full schema → **`docs/yaml-schema.md`**. Quick orientation: top-level `sim` (`{solver: lcp, dt, g}` + flat LCP knobs `erp`/`slop`/`cfm_scale`/`v_rest_thresh`/`iters`/`tol`), `view`, `lights`, `materials`, `bodies`, optional `cameras`/`lidars`/`feeds`. Body named `root` is the world (`base != 'root'` prefixes it `*`). Body/`joint` params: `damping`/`spring`/`frictionloss`/`armature`/`limit`/`q0` (`k:` removed, PD gains pass per step). Shape `param` is MuJoCo-style half-extents/half-lengths (box/sphere/cyl/capsule/mesh/hfield). Default Euler `XYZ` intrinsic, degrees. Worked examples: `demos/arm2.yml`, `demos/cartpole/cartpole.yml`, `fg/gos/gos.yml` (cameras), `fg/dog/dog.yml` (lidars), `envs/hf1.yml` (hfield).

### Conventions worth knowing

- **Controller rate (control-loop ticks/sec)** — every fg Controller takes `rate=None` and stores `self.rate`. The runner is the single source of truth: `start` computes `rate = (1/env.dt) // arg.frameskip` for tact.Env (None for CEnv since Python has no dt); project-specific runners (e.g. `kida/kida.run` → 240) pass an explicit value. `rate=None` → controller falls back to its own HW pacing (kida family → 240). Frameskip: physics steps run at `1/env.dt`, controllers are called every `frameskip` steps, last `(tau, q_ref, qd_ref)` is held between (ZOH — matches real HW eio). Controllers that don't use `self.rate` ignore it.
- `model.add(...)` composes multiple YAMLs into one tree (`offset`/`q0` overrides). `start -e` uses this to drop the robot into an environment scene. `fixed_base=True` strips a free-joint root before merging.
- Per-jtype DoF count (`nq_per_body[i]`): fixed=0, rev/lin=1, free=6. `nq = sum(nq_per_body)`, so `nq != nb` is the norm (free pushes up, fixed pulls down). The `q, qd, qdd, tau, ff, sk, floss, armature, jnt_lo, jnt_hi, Kp_j, Kd_j` arrays are all length `nq`. `active` is the per-DoF mask of length `nq` (fixed→absent, rev/lin→[1], free→[0]×6). Controllers speak the active-only vector of length `sum(active) = dof`; `env.step(u)` expands it. No `extend`/`compress` between forms — fixed joints simply have no q-slots.
- Mesh objects use `file:` in YAML (resolved against the YAML's directory when relative); Python registers each unique path with `set_mesh_path()`. The `cshape[i][0]` integer is an internal slot id, not the filename.
- `start -d <file>` replays a log produced previously to `/dev/shm/out.txt` — deterministic re-runs of an interactive session.

### Contact solver

**`lcp` is the production contact solver** — `contact_lcp` / `tact_step_lcp`. Stewart-Trinkle PGS with friction cones (normal + tangent disk + spin + roll). Semi-implicit Euler. (`solver: minimal` = test-only sphere/ground spring-damper; not for production. A YAML with the removed `solver: penalty` now raises a migration error.) Implicit joint-PD (`q_ref`/`qd_ref` + per-step `kp`/`kd`, the only gain channel) flows into ABA's articulated inertia. Persistent state: the PGS warm-start λ, ONE unified vector `[contact (per (cpair_idx, sub_id) slot) | joint-friction (per-DoF) | joint-limit (per-DoF)]` threaded via `ctx`/`SolverState`.

**Joint Coulomb friction (`frictionloss`)** and **joint range limits (`limit`)** are solved as constraint **rows** in the same PGS: one row per active 1-DoF rev/lin DoF. Friction = 1D box clamp `±floss·dt` (Coulomb is nonsmooth → can't fold into ABA like viscous `damping`; this is what holds a joint static). Limit = one-sided `λ≥0` + Baumgarte, active only at the bound. Each has a per-DoF ctx-threaded warm-start block (`ctx.lam_fric`/`ctx.lam_limit`). Full design: `docs/design-joint-friction.md`, `docs/design-lcp-perf.md`. Multi-point manifold + narrowphase dispatch → `docs/design-contact.md`; hfield detail → `docs/design-hfield.md`.

### Dynamic add/delete

Each `add()` call is tracked as a named **group** so the same set of bodies/shapes/frames can be removed later. Pattern follows MuJoCo's mjSpec (edit spec → recompile → migrate state), but at group granularity with automatic state migration.

- `env.add(src, name=...)` / `model.add(src, name=...)` — `name` defaults to `prefix` (if given) else `modelname`; auto-suffixed `_1, _2, ...` on collision. Explicit `name=` collision raises `ValueError`. Group metadata (insertion ranges into every parallel array) recorded in `model.groups`.
- `env.delete(name)` / `model.delete(name)` — splices the group's slots out of every parallel array, shifts trailing indices down, rebuilds `X`/`I6`/`cpair` and recreates the C handle. `_ctx` reset to None (cpair size change invalidates the λ layout).
- `env.groups` — currently-active group names in insertion order.
- **Deletion ordering**: a group can be deleted iff no surviving body has its parent inside the group's body range. Root-attached groups (free objects on floor, items on conveyor, independent humanoids) have no inbound dependencies → arbitrary-order delete works. Groups added with `base=` pointing at another group create cross-group dependencies, removed in dependency-reverse order. Violations raise `RuntimeError` naming the dependent body.
- **State preservation in `Env.add()`**: existing q/qd kept and new body's q0/qd0 appended, so mid-sim `add()` (conveyor, etc.) preserves the rest of the scene's state.
- **Render handling**: `win_render`/`egl_render` keep a per-slot mesh fingerprint (`prev_type[i]`, `prev_shape[i]`) and rebuild only changed GPU mesh slots, with cleanup when `n_obj` shrinks.
- **Cost** (dt=1ms): ~0.7–1.2ms per add/delete for free-object groups, ~3ms for a 10-link arm; step is ~0.17ms. Sub-1Hz conveyor scenarios have negligible overhead. See `_bench_delete.py`.
- **Backend scope**: tact-backend (`Env`) only. On `CEnv` (mujoco/chrono/real wrappers) `add`/`delete`/`groups` are ABSENT per the capability ledger (`docs/backend-interface.md`) — `hasattr` probes False; the names sit on `CEnv._NO_FORWARD`, the `__getattr__` blocklist that stops the forward-to-cdll footgun (forwarding itself stays for per-robot eio commands like `unlock`/`set_abf`).
- **Demo**: `demos/demo_delete.py`.

## Subdirectories

- `native/` — installable C sources + headers (`rbd.c`, `shape.c`/`shape.h`, `mpr.c`, `narrow.c`, `ray.c`, `lcp.c`, `tact.c`/`tact.h`, `model.c`, `render.c`) plus native tools under `native/tools/`. `make` compiles these → `tact/bin/libtact.so` and `build/tools/bin-test`. Headers beside sources so `#include` needs no `-I`; nothing references them at runtime except the package loader.
- `mjenv/` — internal MuJoCo backend (`mjenv.cpp` → `mjenv.so`) plus shared MuJoCo environment XMLs (`0.xml`, `hf1.xml`, etc.) used only by repo-local `start -m`; excluded from packaging.
- `docs/` — reference + design docs: **`yaml-schema.md`** (full YAML scene/robot schema), **`runtime.md`** (`start` speed/render pacing + ZMQ IPC wire detail), **`backend-interface.md`** (backend core contract N=6 — `step/reset/finish/backend/has_pd/dt` — + capability ledger; enforced by `tests/test_backend_contract.py`), `design-c-state.md` (C handle lifecycle), `design-lcp-perf.md` (LCP block-sparsity perf + constraint-row invariants), `design-contact.md` (narrowphase dispatch, multi-point manifold), `design-joint-friction.md` (frictionloss/armature/limit as constraint rows), `design-pure-step.md` (Model.step referential transparency), `design-hfield.md` (height-field terrain — read before extending hfield).
- `tests/` — **all test assets live here**. `regression/` — bit-identical golden suite: `capture_baseline.py` (captures `baseline/*.npy`), `test_traj.py` (re-runs, compares at atol 1e-12; capture is a deliberate reviewed act, never auto-update). Core smoke/contract tests live as `test_*.py`; analytic joint checks live as `test_joint_*.py`. **`tests/scenes/`** — FROZEN copies of the YAMLs + meshes the suite loads, decoupled from the mutable `demos/`/`envs/` trees (`tests/scenes/README.md`).
- `_/` (top-level and inside projects) — archive/scratch: older copies of scripts and previous `start`/`tact.py` versions. Do not edit unless asked.
- `envs/` — **background-environment scene YAMLs** (`start -e` and per-project-RL-env target): `1.yml`–`5.yml` (ground planes), `d3.yml` (stepping stones), `hf1.yml` (10×10 m height-field; data in `envs/hf1.bin` ← `envs/hf1_gen.py`; mujoco twin `mjenv/hf1.xml`), `stairs.yml`, `box1.yml`/`desk1.yml`. Composed via `self.sim.add(f'{tact.pkg_dir}/envs/<name>')` (e.g. `mk1/hop0.py`, `mk2/hop0.py`).
- `demos/` — **tact feature/physics demos**: sample model YAMLs (`arm2`–`arm4`, `fv`, `obj1`/`obj2`, `sphere_test`/`min_test`, `box_wall`/`wall5_box`/`mini_wall_box`, used by `yml-test`); demo scripts (`ball_throw.py`, `raymap_demo.py`, `demo_delete.py`); the `cartpole/` example project (own `./start` symlinking `../../start`, `tact -> ../..`); shared mesh assets in `demos/meshes/*.obj`, referenced by mesh-shape YAMLs as `file: meshes/<name>.obj`. (The Rainbow rb5/rb10 projects moved to `fg/fgx/_/` — deactivated.)
- `perf/` — micro-benchmarks comparing matmul in C (`cmm.c`), Eigen (`eigmm.cpp`), NumPy (`npmm.py`).
- `fg/fgx/_/rb5/`, `fg/fgx/_/rb10/` — Rainbow Robotics arm projects, **deactivated**. Each `basic.c` is a TCP client to the robot controller; `rb10/chenv.cpp` is the legacy Chrono backend (no longer launchable via `start`).
