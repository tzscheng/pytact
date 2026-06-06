# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tact` is a robotics simulation + control toolkit. It pairs a C shared library (`libtact.so`) for collision detection (GJK/CCD) and OpenGL/EGL rendering with a Python **package** at `fg/tact/` (flat layout: the `.py` package files sit directly in the project directory alongside `build.sh`, `start`, example projects, and asset folders; the **C/C++ sources + headers live under `native/`** and build to `bin/libtact.so`). The package implements rigid-body dynamics (Featherstone / Siciliano / Craig algorithms), forward kinematics, controllers, and a YAML-driven scene/robot loader. It is split into two domains by file but exposes a flat API via `__init__.py` re-exports:

- `_clib.py` — internal: `libtact.so` loader + ctypes argtypes/restypes (not exposed at top level)
- `rbd.py` — pure math/dynamics primitives: rotation/quaternion helpers, spatial-vector building blocks, Featherstone/Siciliano/Craig algorithms, integrators, contact, ray casts. No state. C-side counterpart is `rbd.c`.
- `sim.py` — simulation classes only: `Model`, `Env`, `CEnv`. Does `from .rbd import *` so inline references stay flat.
- `control.py` — controllers & planning: `PIDController`, `JacobianTransposeController`, `JacobianTransposeForce`, `ComputedTorqueController`, `HybridForcePositionController`, `MovingAverageWaypointSmoother`, `Envelope`, `StepGenerator2/4`, pinch primitives. Depends on `rbd` for transform helpers (single-direction: `control → rbd ← sim → _clib`).
- `__init__.py` — `from .rbd import *; from .sim import *; from .control import *`, so callers see a flat `tact.Env`, `tact.PIDController`, `tact.crm`, etc. Optional explicit-domain access also works: `tact.sim.Env`, `tact.rbd.crm`, `tact.control.PIDController` (same object — `tact.Env is tact.sim.Env` → `True`).

It is consumed by per-robot projects: bundled example projects under `examples/` (`cartpole/`, `rb5/`, `rb10/`) and external siblings under `../dg5`, `../dog`, `../gos`, `../h9`, `../h12`, `../kida`, `../mk1`, `../mk2`, `../mk3`, `../zen`. Each project supplies its own `Controller` class and YAML and is launched through a uniform `start` script that swaps the simulation backend (built-in sim, MuJoCo, Chrono, or real hardware) behind a common loop. The class is a low-level execution controller (receives mode commands from outside at low freq, runs the inner control loop at sim rate). It was named `agent` historically but doesn't act as an autonomous agent — actual high-level control lives outside via ZMQ.

## Working preferences

User works on this project from multiple machines, so preferences live here (in the repo) rather than only in per-machine memory. These shape *how* to interact, not *what* to do:

1. **Korean technical conversation with English code terms.** All discussion in Korean. C/Python identifiers, algorithm names (Featherstone, PGS, LCP, MPR, SAT, etc.), and English technical terms ("manifold", "convex", "kinematic tree", "warm-start") stay English. Prefer tables and bullet structures for comparisons.

2. **Honest trade-off assessment over advocacy.** Present trade-offs with skepticism, especially for speculative future capabilities (autodiff, GPU, etc.). Don't pile up justifications for architectural commitments unless warranted. When uncertain about whether work is worthwhile, ask "is this the right time" rather than enthusiastically endorsing.

3. **Explicit scope + risk before destructive/multi-file changes.** Don't start bulk YAML migrations, deprecations, or refactors touching many files without first showing scope + risk and getting confirmation. Always show what *will* be touched before touching it.

4. **Incremental milestones with verification.** Prefer `(1) → (2) → (3)` ordering with completion checkpoints. After each milestone: run regression / verify. Don't start grand refactors without phase plan; pause for go/no-go between phases.

5. **Be precise about the competing-tools landscape.** Don't oversimplify ("Bullet uses X", "RaiSim uses Y") without checking — the user knows the field and will correct. Hedge when unsure.

6. **Strategic over tactical when asked.** When user asks "should we do this AT ALL?", give an honest "is this worth it for *this* project's use case" answer with concrete reasoning, not just "how to do it".

## Build / run

```bash
./build.sh                   # builds bin/libtact.so from native/{rbd,shape,mpr,narrow,box_box,ray,lcp,tact,render}.c
                             # link deps: -lm -lEGL -lGL -lGLEW -lGLU -lglfw -lturbojpeg -lzstd
                             # also builds bin/mjenv.so from native/mjenv.cpp (auxiliary MuJoCo backend
                             # used for cross-checking tact's dynamics; -I/usr/local/include/mujoco
                             # -lmujoco -lGL -lglfw)
./yml-test arm2              # quick test loading a YAML model + running env.step in a window
perf/build.sh                # builds C/C++ matmul benchmarks (cmm, eigmm)
```

All build artifacts go to `bin/` (created by `build.sh` if missing). Loaders look for them at `<package_dir>/bin/<name>` — `_clib.py` for `libtact.so`, the `start` script for `mjenv.so`.

Python is run via `uv` from the workspace root at `/home/ubuntu/uv1` (see `/home/ubuntu/uv1/pyproject.toml`). The `start` scripts use the shebang `#!/usr/bin/env -S uv run python`.

`tact/_clib.py` resolves `libtact.so` in this order: (1) **next to itself** — `<package_dir>/bin/libtact.so`, where `<package_dir>` is the resolved path of the `_clib.py` file, which follows the per-project symlink `<proj>/tact` → `fg/tact/`; (2) `./bin/libtact.so` in the current working directory (development build); (3) the OS dynamic loader via `ctypes.util.find_library('tact')` (ldconfig / `LD_LIBRARY_PATH` / `/usr/lib` / `/usr/local/lib` on Linux, `DYLD_LIBRARY_PATH` on macOS, `PATH` on Windows). In the current monorepo layout, path (1) succeeds because each project has a `tact` symlink pointing at `fg/tact/` and `bin/libtact.so` lives there alongside the package files — so no system install is required.

Asset layout (all under `fg/tact/`, discoverable as `tact.pkg_dir/<asset>`):
- `tact.pkg_dir/bin/libtact.so`, `tact.pkg_dir/bin/mjenv.so` — native libraries (build.sh outputs)
- `tact.pkg_dir/examples/*.yml` — standard scene + sample model YAMLs (loaded by `start -e <name>` and per-project RL envs)
- `tact.pkg_dir/examples/{cartpole,rb5,rb10}/` — bundled example projects
- `tact.pkg_dir/examples/objs/*.obj` — mesh assets. C-side `load_obj` (in `shape.c`; the render-side equivalent `load_obj_as_mesh` is in `render.c`) tries CWD-relative `objs/<idx>.obj` first, then `examples/objs/<idx>.obj` — so `yml-test` from `tact/` finds them automatically, and projects that want their own meshes can supply a local `objs/` directory.

## Per-project run pattern

Inside an example project (e.g. `examples/cartpole/`) or any sibling project:

```bash
./start cartpole             # built-in tact sim, controller module = cartpole.py, yml = cartpole.yml
./start -m <xml> cartpole    # MuJoCo backend (loads bin/mjenv.so, built from native/mjenv.cpp)
./start -c <arg> cartpole    # Chrono backend (./chenv.so in the project dir)
./start -x [arg] cartpole    # real hardware via ./eio.so
./start -y other_yml cartpole         # override YAML name
./start -e env_yml cartpole           # add an environment scene on top of robot
./start -d dispatch.txt cartpole      # replay timestamped commands from a file
./start -k 4 cartpole                 # physics ticks per control update (default 1; ZOH between)
./start -s 2 cartpole                 # 2x realtime (works headless AND GUI)
./start -s 2 -f 30 cartpole           # 2x sim, window redraws at ~30 Hz
./start -l -s 0 cartpole              # headless, free-run (no pacing) — RL/batch
```

**Speed and render rate are two orthogonal knobs** (tact backend): `-s` SPEED (default 1.0, `0`=free-run) sets sim speed; `-f` (default 60) sets the GUI render rate. The window redraw cadence is **derived, never user-set**: `redraw = ceil(speed/(dt*fps))`, so the sim advances at `-s`× realtime while the window redraws at ~`-f` Hz, independently. **Both the tact and mujoco backends** derive it this way — mjenv exposes `get_dt`/`set_redraw` so `start` drives its cadence (and uses its dt) identically. (There is no longer a raw steps-per-render flag.)

**Real-time pacing**: the main loop sleep-paces wall-clock to `-s`× (anchored to an absolute target; re-anchors if >100 ms behind). Paces both headless and GUI; window vsync stays **on** (`glfwSwapInterval(1)`) and `start` sets `__GL_YIELD=USLEEP` (before GL init) to stop NVIDIA's vblank busy-spin. mujoco backend paced identically (via `get_dt`/`set_redraw`); Chrono/real pace via their own loop/HW. Full rationale (vsync-vs-sleep trade-offs, off-thread renderer) → `docs/runtime.md`.

The argv tail (`cartpole`) names a Python module to import — that module must define `class Controller: __init__(env, ymlname, prefix='', rate=None, verbose=False)`, `msgproc(w)`, and `update(y) -> (tau, q_ref, qd_ref)`, plus `n_u` / `n_y` attributes. Controllers can branch on `env.has_pd` (typically cached as `self.has_pd` at `__init__`, optionally overridden under the controller's own responsibility) to choose between emitting torque (`tau`) vs position reference (`q_ref`). See `examples/cartpole/cartpole.py` and `examples/rb10/rb10.py` for the reference shape.

## IPC contract (`start` script)

Bound by `start` over ZMQ IPC on `/dev/shm`:
- PULL `ipc:///dev/shm/default` — incoming commands (whitespace-split string). Words `quit` and `reset` are handled in `start`; everything else is forwarded to `controller.msgproc(w)`.
- PUB `ipc:///dev/shm/proprio` (CONFLATE) — float32 packed proprioceptive vector `y`, sent every 20 sim steps.
- PUB `ipc:///dev/shm/<camera-name>` / `<lidar-name>` per `cameras:`/`lidars:` entry — `env.camera_frames()`/`lidar_frames()` own the rate-gating (`fps` vs `env.cnt`) and `type`→encoder dispatch, yielding `(name, bytes)` for due sensors: rgb→JPEG, depth-camera→zstd float32 (C-side), lidar 2d/3d→**raw float32** (Python-side zstd removed 2026-06-06 — floats compress ~×1.5 at real sim-loop cost vs ~1000× IPC headroom). Sim core has no zmq (sockets/send stay in the runner). CEnv backends publish nothing. Full wire format → `docs/runtime.md`.
- Without `-d`, every received command is logged to `/dev/shm/out.txt` with the step counter.

The cartpole subdir has its own (identical) `start`; new projects typically copy this script.

## Code map

### C side (`libtact.so`)

All C/C++ sources + headers live under **`native/`** (headers sit beside the sources, so `#include "tact.h"` needs no `-I`), built into `bin/libtact.so` by `build.sh`. Linked deps: `-lm -lEGL -lGL -lGLEW -lGLU -lglfw -lturbojpeg -lzstd`. (File names below are under `native/`.)

- `tact.h` — public interface + common constants (`MAX_NB=256`, `MAX_PTS_PER_PAIR=4`, shape codes `MESH=100, BOX=101, SPHERE=102, CYL=103, CAPSULE=104, HFIELD=105`, Euler conventions `EULER_{EXT,INT}_{XYZ,ZYX}` for the rbd helpers).
- `rbd.c` — linear algebra (matmul, transforms, Euler conversions) + spatial-vector dynamics (ABA, CRB, RNE, LDLᵀ, jacob_whitney) + `choose_rotation`. Python `rbd.py` mirrors this 1:1.
- **`shape.h` / `shape.c`** — shared shape-asset **slot storage** (the collision side's common data): mesh slots (`vertex`/`face`/`num_*`/`mesh_path`, loaded lazily from `.obj` by `load_obj`, set by `set_mesh_path`) and height-field slots (`hf_nrow`/`hf_ncol`/`hf_sx`/`hf_sy`/`hf_minh`/`hf_maxh`/`hf_data`, pushed from Python by `set_hfield_data` — no lazy load). `MAX_MESH=64`, `MAX_HFIELD=16` live in `shape.h` (data-only header; function prototypes stay in `tact.h`). Consumed by `mpr.c` (mesh support), `narrow.c` (hfield contact), `ray.c` (ray casts), `render.c`. `mesh_local_radius` / `hfield_local_radius` (bounding-sphere for the raycast broad phase) live here too.
- `narrow.c` — **narrow-phase dispatch + dedicated analytic detectors** per shape pair (box-box via `box_box.c` SAT+clip ≤4 pts; sphere/capsule/cylinder/box analytic closed-forms; full hfield matrix Tier 2), MPR/EPA fallback (`mpr.c`) for the rest. `collision_check(...)→n_points`, 7 doubles/point `[p(3), n(3), depth]`. Full dispatch table → `docs/design-contact.md`. (Split from former `ccd.c` 2026-05-28; hfield matrix done 2026-05-29.)
- `box_box.c` — **box-box contact manifold via SAT (15 axes) + Sutherland-Hodgman face clipping**. Exposes `box_box_manifold` called from `narrow.c`'s dispatcher. Algorithm: (1) SAT over 3+3 face normals + 9 edge crosses, face-bias picks face wins on coplanar ties; (2) face-axis winner → clip incident face polygon against reference face planes, prune to MAX_PTS_PER_PAIR via Bullet 4-point selection (deepest+farthest+max-area+opposite-side), polar-angle sub_id sort; (3) edge-cross winner → closest-pair on the two supporting edges (single point). Originally separate, merged into `narrow.c` 2026-05-28, re-split 2026-05-28 for code-map clarity.
- `mpr.c` — generic **convex narrowphase fallback** (MPR + EPA portal, libccd-derived `ccd_*` core) used by `collision_check` for shape pairs with no dedicated detector. Holds the per-shape support function (`ccd_support`, reads mesh vertices from `shape.c`), `size_of_param`, and `collision_check_mpr`.
- `ray.c` — **ray-primitive intersections** (`ray_intersects_{triangle,mesh_slot,hfield,box,sphere,cylinder,capsule}`). Each returns forward t along Rd or -1. Mesh/hfield variants read `shape.c`'s slot storage (ray transformed into shape-local frame first). `ray_intersects_hfield` uses a **2D DDA grid walk** (only cells the ray crosses, first-hit = nearest, 3D-AABB clip folds in footprint + height-slab culling) — ~870× faster than brute force on a 101×101 grid. Mirrors `rbd.py:ray_intersects_*`. (`narrow.c`/`mpr.c`/`ray.c`/`shape.c` were split out of the former `ccd.c` + `box_box.c` 2026-05-28; pure reorg, regression bit-identical.)
- `lcp.c` — `contact_lcp`: Stewart-Trinkle/Anitescu PGS, 4 contact cones (normal + tangent disk + spin + roll) + non-contact constraint rows (joint friction/limit), CFM-regularized + Baumgarte. Row-table A-build (`Mrow` + per-row `row_blocks[]`, `M2=6·Pm+2·nq`). Warm-start λ per (cpair_idx, sub_id). Block-sparsity opts S1 + S2 win(a) shipped 2026-05-25 (box_wall 4.4×, zen K=12 10.8×). See `docs/design-lcp-perf.md`, `docs/design-joint-friction.md`.
- `tact.c` — opaque `tact_t` handle (single arena malloc for all parallel arrays + workspaces). `tact_create` packs everything into one block; `tact_step_lcp` runs one timestep; `tact_destroy` frees the arena. Persistent state across steps: `lam_prev` (LCP warm-start), q/qd (passed in by caller).
- `render.c` — OpenGL rendering. Two entry points consumed from Python: `win_render(...)` (GLFW window, called from `env._win_render`) and `egl_render(..., out_buf, opt, req_width, req_height, fovy_deg)` (offscreen EGL; `opt==1` → RGB JPEG via libjpeg-turbo, `opt==2` → linear eye-space metric depth as zstd-compressed float32; sized to `req_width`×`req_height` with the size-dependent GL/CPU buffers reallocated grow-only; called inline from `env.camera_frames`). The packet header layout (`MAX_PACKET=16384, HEADER_SIZE=16`) is mirrored in Python's `stream_assembler`. Maintains per-slot mesh fingerprints (`prev_type[i]`, `prev_shape[i]`) so dynamic add/delete only rebuilds changed GPU mesh slots each frame.

### Auxiliary MuJoCo backend (`native/mjenv.cpp`)
- `native/mjenv.cpp` — thin C++ shim around MuJoCo (`mjModel`/`mjData`, `mjv*` viz, GLFW window). Built into `bin/mjenv.so` by `build.sh` and dlopened from `start -m`. Kept here (not as a sibling project) because its only role is to cross-check `tact`'s built-in dynamics against MuJoCo on the same YAML/XML pair — there's no per-robot code to host. `start -m <name>` resolves the XML as `<name>.xml` or `xml/<name>.xml` in the project dir (the shared MuJoCo models live under `fg/../fgx/mujoco/models/`). `init(xml, window_flag)`'s 2nd arg is just 0/1 (headless vs GLFW window); the render cadence + sim speed are driven by `start` via the exported **`get_dt()`** (timestep, so `start` can derive `redraw` and sleep-pace) and **`set_redraw(n)`** — so `-s`/`-f` behave identically to the tact backend. **XML convention:** every MuJoCo XML used with this backend must have matched motor↔position actuator pairs per joint (dual-actuator). `start` passes `has_pd=True` for the mujoco backend on that assumption. If a torque-only XML is used, the controller must override `env.has_pd = False` before stepping (otherwise `mjenv` silently ignores q_ref).

### Python side (`tact/` package — flat API)
Three modules with a flat re-exported namespace. Pure rigid-body math/dynamics primitives live in `rbd.py`; stateful simulator classes in `sim.py`; controllers in `control.py`. `rbd.py` exports:
- Rotation/Euler/quaternion/homogeneous conversions. Convention: lowercase eulerseq = extrinsic, uppercase = intrinsic (`xyz`/`XYZ`/`zyx`/`ZYX`). Default in YAML loader is `XYZ` intrinsic, degrees.
- Spatial-vector building blocks: `crm`, `crf`, `jcalc`, `get_spatial_inertia`, `get_spatial_transform`, `_fk`.
- Dynamics algorithms: `crb_featherstone` (mass matrix), `rne_featherstone` / `rne_lwp` (inverse dynamics — spatial vs Luh-Walker-Paul body-frame RNE), `aba_featherstone` (forward dynamics / articulated-body — also folds the semi-implicit `ff`/`sk`/joint-PD terms and the `armature` rotor inertia into its articulated inertia `d`), `inertia_lagrange`, `cc_finitediff`, `gravity_lagrange`, `jacob_whitney` (geometric Jacobian). **`armature`** (MuJoCo-style per-DoF rotor/reflected inertia) is added to the diagonal of `d` in ABA and to the M diagonal after CRB (the LCP step does `M += diag(armature)`), keeping the free predictor and the contact solve on the same effective inertia; armature=0 is bit-identical.
- Integrators: `euler_step`, `euler_step2`, `rk4_step`.
- Contact: `contact_lcp` (calls multi-point `clib.collision_check`, builds J / A / b, runs PGS over 4 cones; the production solver, `solver: lcp`). `contact_ground_sphere` is a **test-only** minimal solver (`solver: minimal`): explicit spring-damper ground (z=0 plane) contact for SPHERE shapes only, fed as f_ext into ABA — Python step path, no Coulomb cone / box / mesh.
- Ray casts: `ray_intersects_{triangle,mesh,box,sphere,cylinder,capsule}` — used by `env.raycast` and the lidar publish path (`env._ray_grid` + `tact_raycast_frame`; the public `raymap`/`raycloud` wrappers were inlined into `lidar_frames` 2026-06-06). Single-threaded; cost is O(pixels × raycast-shapes), brute-force per ray (no broadphase/BVH — meshes are O(triangles)/ray).

Key classes:
- `Model` (1079) — YAML → kinematic tree. Holds parallel arrays `jtype, parent, active, fixed, Ti, m, c, I, q0, qd0, ff, sk` and contact arrays `ctype, cbody, cshape, cparam, ctran, crgba`. Frame registry (`fdict, fbody, ftran`) is populated from `frames:` blocks. `step(q, qd, tau=None, q_ref=None, qd_ref=None, ctx=None)` runs contact + integrator and returns `(q_next, qd_next, y, ctx_next)`. **It is referentially transparent (pure): the LCP warm-start λ (contact `lam` + per-DoF joint-friction `lam_fric`) — the only hidden state since penalty was removed — is threaded explicitly via `ctx` (a `SolverState` namedtuple, `ctx=None` = cold start, `ctx` left immutable so fork/branch is safe). `Model.zero_state()` builds a cold ctx for external callers. `Env` keeps one in `self._ctx` so Env users are unaffected, and it is bit-identical to the pre-pure path (no baseline re-capture — `test_traj.py` 10/10). Single-thread only; thread-safety/batched/autodiff need workspace externalization (not done). See `docs/design-pure-step.md`.** Helpers: `fk` (returns concatenated 3d/6d frame poses per a `{frame: '3d'|'6d'}` dict), `fkh` (homogeneous), `jacob`, `error`.
- `Env` (1811) — wraps `Model`, owns the render window and an EGL image buffer (1024×768×4). `step(u)` expands `u` (active-only) into joint torques, calls `model.step`, optionally redraws every `redraw` ticks. `add(..., name=)` composes multiple YAMLs into one tree (e.g. robot + env scene); `delete(name)` removes a previously added group (see "Dynamic add/delete" below). `env.groups` lists active group names.
- Controllers: `PIDController` (vectorized), `JacobianTransposeController` (task-space PD via Jacobian transpose), `JacobianTransposeForce` (task-space feed-forward), `ComputedTorqueController` (computed torque), `HybridForcePositionController` (hybrid force/position with selection matrix from contact force normal). `MovingAverageWaypointSmoother` is a sliding-window way-point smoother used to ramp targets — see `cartpole.py` for use with `PIDController`.
- Pinching primitives: `pinch2`, `pinch3p`, `pinch3_icra2012`, `pinch3_piony`. `Envelope` is a fixed-pose holder.
- Step generators: `StepGenerator2` (biped) and `StepGenerator4` (quadruped). Terrain handling is trick-free as of 2026-06-06: edge-detect ("line_adjust") and foothold z consume only base-relative `env.height_scan` deltas anchored to a stance foot's FK z — `env.get_z` (absolute world-z oracle) was REMOVED; no controller may read absolute terrain height (no real-robot counterpart).
- `stream_assembler` (2459) — reassembles fragmented UDP camera frames matching `camera.c`'s 16-byte header (`<IHHII`); 200 ms stale-frame timeout.

### YAML scene format

Full schema → **`docs/yaml-schema.md`** (top-level keys; body/`joint` params incl. `damping`/`spring`/`frictionloss`/`armature`/`limit`/`q0`/`k`; `inertial`; `shapes` incl. `rgba` translucency; `materials` library; `cameras` / `lidars` libraries + sensor-frame convention; global LCP knobs `erp`/`slop`/`cfm_scale`/`v_rest_thresh`/`iters`/`tol`; shape `param` conventions box/sphere/cyl/capsule/mesh/hfield).

Quick orientation: top-level `sim` (`{solver: lcp, dt, g}` + flat LCP knobs), `view`, `lights`, `materials`, `bodies`, optional `cameras`/`lidars`/`feeds`. Body named `root` is the world (`base != 'root'` prefixes it `*`). Shape `param` is MuJoCo-style half-extents/half-lengths. Default Euler is `XYZ` intrinsic, degrees. Worked examples: `examples/arm2.yml`, `examples/cartpole/cartpole.yml`, `fg/gos/gos.yml` (cameras), `fg/dog/dog.yml` (lidars), `examples/terrain10.yml` (hfield).

### Conventions worth knowing

- **Controller rate (control-loop ticks/sec)** — every fg Controller takes a `rate=None` kwarg in `__init__` and stores `self.rate`. The runner is the single source of truth: `start` computes `rate = (1/env.dt) // arg.frameskip` for tact.Env (None for CEnv since Python has no dt); project-specific runners (e.g. `kida/kida.run` → 240) pass an explicit value. When the controller receives `rate=None` (direct construction, or start+CEnv), it falls back to its own HW pacing (kida family → 240). Frameskip is the inverse policy: physics steps run at `1/env.dt`, controllers are called every `frameskip` steps, last `(tau, q_ref, qd_ref)` is held between (ZOH — matches real HW eio behavior). bt was the old name for this; now uniformly `rate`. Controllers that don't yet use `self.rate` simply ignore it — usage (e.g. trajectory durations `4 * self.rate * e_eff`) can be added per project incrementally.
- `model.add(...)` allows composing multiple YAMLs into one tree (`offset` / `q0` overrides). `start -e` uses this to drop the robot into an environment scene. `fixed_base=True` strips a free-joint root before merging.
- Per-jtype DoF count (`nq_per_body[i]`): fixed=0, rev/lin=1, free=6. `nq = sum(nq_per_body)`, so `nq != nb` is the norm now (free pushes it up, fixed pulls it down). The `q, qd, qdd, tau, ff, sk, floss, armature, jnt_lo, jnt_hi, Kp_j, Kd_j` arrays are all length `nq`. `active` is the per-DoF mask of length `nq` (fixed→absent, rev/lin→[1], free→[0]×6). Controllers speak the active-only vector of length `sum(active) = dof`; `env.step(u)` expands it to the `nq`-length internal vector. There is no longer an `extend`/`compress` between fixed and non-fixed forms — fixed joints simply don't have q-slots.
- Mesh objects use `file:` in YAML (resolved against the YAML's directory when relative); Python registers each unique path with the C-side `set_mesh_path()` slot table. The `cshape[i][0]` integer is an internal slot id, not the filename.
- `start -d <file>` replays a log produced previously to `/dev/shm/out.txt` — useful for deterministic re-runs of an interactive session.

### Contact solver

**`lcp` is the production contact solver** — `contact_lcp` / `tact_step_lcp`. Stewart-Trinkle PGS with friction cones (normal + tangent disk + spin + roll). Semi-implicit Euler integrator. (`solver: minimal` = test-only sphere/ground spring-damper via `rbd.contact_ground_sphere` on the Python step path; not for production.) Implicit joint-PD (`Kp_j`/`Kd_j`/`q_ref`/`qd_ref`) flows into ABA's articulated inertia. Persistent state: `lam_prev` (per (cpair_idx, sub_id) slot warm-start) and `lam_fric` (per-DoF joint-friction warm-start) — both threaded via `ctx`/`SolverState` for referential transparency.

**Joint Coulomb friction (`frictionloss`)** and **joint range limits (`limit`)** are solved as constraint **rows** in the same PGS (the row-table generalization, I1/I2): one row per active 1-DoF rev/lin DoF. Friction = 1D box clamp `±floss·dt` (Coulomb is nonsmooth → can't fold into ABA like viscous `damping`; this is what holds a joint static, the failure that retired penalty). Limit = one-sided `λ≥0` + Baumgarte, a "contact normal" on the joint coordinate, active only at the bound. Each has its own per-DoF ctx-threaded warm-start (`lam_fric`/`lam_limit`); row buffers use `M2 = 6·Pm + 2·nq`. Mirrored C↔Python to ~machine ε; `tests/joint_{friction,limit}.py`. Zero-friction/limit is bit-identical (the `-ffast-math` box_wall/wall5_box ~1e-6 m shift is checked at 0.1 mm in `test_traj.py`, rest at 1e-12). Full design: `docs/design-joint-friction.md`, `docs/design-lcp-perf.md`.

The legacy `penalty` spring-damper + brush-friction solver was **removed 2026-05-24** (`penalty.c` archived to `_/`). It was retired because its brush friction could not hold a planted foot (a box on a 20° incline with μ=0.9 slid 500 mm instead of sticking; LCP sticks to 0.004 mm), and box stacking diverged at dt=1ms (needed dt≤0.2ms, still 100×+ less accurate than LCP). All sibling projects were migrated `solver: penalty → lcp`. A YAML with `solver: penalty` now raises a migration error.

**Multi-point contact manifold + narrowphase dispatch** → **`docs/design-contact.md`**: `collision_check(...)→n_points` (≤`MAX_PTS_PER_PAIR=4` pts/cpair), the per-shape-pair dedicated-detector dispatch table (box-box SAT+clip ≤4; analytic sphere/capsule/cylinder/box; full hfield Tier-2 matrix; MPR/EPA fallback), the `slot = cpair_idx·MAX_PTS_PER_PAIR + sub_id` indexing + polar-angle sub_id ordering, and the box_wall drift/cost result. hfield detail: `docs/design-hfield.md`.

### Dynamic add/delete

Each `add()` call is tracked as a named **group** so the same set of bodies/shapes/frames can be removed later. Pattern follows MuJoCo's mjSpec (edit spec → recompile → migrate state), but at group granularity rather than per-element, and state migration is automatic.

- `env.add(src, name=...)` / `model.add(src, name=...)` — `name` defaults to `prefix` (if given) else `modelname`; auto-suffixed with `_1, _2, ...` on collision. Explicit `name=` collision raises `ValueError`. Group metadata (insertion ranges into every parallel array) is recorded in `model.groups`.
- `env.delete(name)` / `model.delete(name)` — splices the group's slots out of every parallel array, shifts trailing indices down (`parent`, `cbody`, `fbody`, `fixed`, `feeds[*]`, `fdict[*]`), pops fdict keys, rebuilds `X`/`I6`/`cpair` and recreates the C handle. `lam_prev` (LCP warm-start) reset to None — cpair size change invalidates it.
- `env.groups` — list of currently-active group names in insertion order.
- **Deletion ordering**: a group can be deleted iff no surviving body has its parent inside the group's body range. Root-attached groups (typical "free objects on floor", "items on conveyor", "independent humanoids") have no inbound dependencies → arbitrary-order delete works. Groups added with `base=` pointing at another group's body create cross-group dependencies; those must be removed in dependency-reverse order (LIFO-like). Violations raise `RuntimeError` naming the dependent body.
- **State preservation in `Env.add()`**: existing q/qd values are kept and new body's q0/qd0 appended (vs. pre-feature behavior that reset `self.q = self.m.q0` on every add). All pre-existing sibling call sites are init-time `add()` chains where this is bit-identical; mid-sim `add()` (conveyor, etc.) now correctly preserves the rest of the scene's state.
- **Render handling**: `win_render`/`egl_render` in `render.c` keep a per-slot mesh fingerprint (`prev_type[i]`, `prev_shape[i]`) and rebuild only the changed GPU mesh slots each frame, with cleanup when `n_obj` shrinks. Without this, dynamically added shapes have no GPU resources, and post-delete shape-index shifts would draw the wrong mesh per slot.
- **Cost** (measured, dt=1ms): ~0.7–1.2ms per add or delete for free-object groups; ~3ms for a 10-link arm. Step is ~0.17ms. Sub-1Hz conveyor scenarios (5–10/min) have negligible overhead. Break-even with step-rate sits around tens of ops/sec; optimization path (deferred rebuild, single C handle create per step) only matters above that. See `_bench_delete.py`.
- **Backend scope**: tact-backend (`Env`) only. On `CEnv` (mujoco/chrono/real wrappers) `add`/`delete`/`groups` are ABSENT per the capability ledger (`docs/backend-interface.md`) — `hasattr` probes False, and the names sit on `CEnv._TACT_ONLY`, the `__getattr__` blocklist that stops the forward-to-cdll footgun if a future backend happens to export a same-named C symbol (forwarding itself stays: it's the channel for per-robot eio commands like `unlock`/`set_abf`). mujoco's mjSpec recompile path could be wired up later but isn't a priority.
- **Demo**: `examples/demo_delete.py` drops shapes onto a floor in time order, then deletes in arbitrary order (middle-aged before older), demonstrating cross-group state preservation and the render path fix.

## Subdirectories

- `native/` — all C/C++ sources + headers (`rbd.c`, `shape.c`/`shape.h`, `mpr.c`, `narrow.c`, `ray.c`, `lcp.c`, `tact.c`/`tact.h`, `render.c`, `mjenv.cpp`). `build.sh` compiles these → `bin/`. Headers live beside sources so `#include` needs no `-I`; nothing references them at runtime (the package loads `bin/libtact.so` only).

- `docs/` — reference + design docs (deeper than this file): **`yaml-schema.md`** (full YAML scene/robot schema — bodies/joints/shapes/materials/cameras/lidars/global knobs/shape params), **`runtime.md`** (`start` speed/render pacing + ZMQ IPC wire detail), **`backend-interface.md`** (backend core contract N=6 — `step/reset/finish/backend/has_pd/dt` — + capability ledger: per-backend optional surface, caller-side guard conventions, parity goals; enforced by `tests/test_backend_contract.py`; sensor publishing = tact-only capability, CEnv has NO sensor stubs), `design-c-state.md` (C handle lifecycle/state), `design-friction.md` (contact friction model), `design-lcp-perf.md` (LCP block-sparsity perf S1/S2 + constraint-row forward-compat invariants), `design-contact.md` (narrowphase dispatch table, multi-point manifold, box_wall drift), `design-joint-friction.md` (frictionloss/armature/limit as constraint rows — the row-table generalization, phases + verification), `design-pure-step.md` (Model.step referential transparency via `ctx`/`SolverState`), `design-hfield.md` (height-field terrain: data model, 2D-DDA raycast, contact-fidelity roadmap — read before extending hfield).
- `tests/` — **all test assets live here** (so they aren't mistaken for editable examples). `regression/` — bit-identical golden suite: `capture_baseline.py` (SCENARIOS + `run_one`, captures `baseline/*.npy`), `test_traj.py` (re-runs, compares at atol 1e-12; capture is a deliberate reviewed act, never auto-update). `test_pure_step.py` — Model.step pure-function invariants (self-validating, no baseline). `box_wall_stability.py` — headless box_wall stability analysis. `_prof_*.py` — perf profilers (`_prof_box_wall` = frozen fixture; `_prof_multizen`/`_prof_multidog` profile **live** sibling models `fg/zen`,`fg/dog`). **`tests/scenes/`** — FROZEN copies (byte-identical at capture time) of the YAMLs + meshes the suite loads, decoupled from mutable `examples/`; all test/prof loaders read here, not `examples/`. Edit a fixture only deliberately, then re-capture (`tests/scenes/README.md`).
- `_/` (top-level and inside projects) — archive/scratch: older copies of scripts (`render.py`, `vsim`, `vcloud.py`, `vdepth.py`, `svs.py`, encoders `png2gif`/`png2mp4`/`raw2mp4`) and previous `start`/`tact.py` versions. Do not edit unless asked.
- `examples/` — bundles three things in one tree:
  - **Scene YAMLs**: `1.yml`–`5.yml`, `box1.yml`, `desk1.yml`, `terrain10.yml` (10×10 m walkable height-field terrain for legged robots — `terrain10.npy` grid + `terrain10_gen.py` generator). Loaded via `start -e <name>`.
  - **Sample model YAMLs**: `arm2`, `arm3`, `arm4`, `fv`, `obj1`, `obj2`, `sphere_test`. Used by `yml-test` for quick model inspection.
  - **Example projects**: `cartpole/`, `rb5/`, `rb10/`. Each is a self-contained agent + YAML pair launched via its own `./start` (symlinks to `../../start`, with `tact -> ../..` pointing back at the package).
  - **Mesh assets**: `objs/*.obj`. Referenced by `param: [<idx>]` in mesh-shape YAMLs (`obj1`, `obj2`).
  - Per-project RL envs (e.g. `mk1/hop0.py`, `mk2/hop0.py`) compose scene YAMLs via `self.sim.add(f'{tact.pkg_dir}/examples/<name>')`.
- `perf/` — micro-benchmarks comparing matmul in C (`cmm.c` against `tact.h`), Eigen (`eigmm.cpp`), and NumPy (`npmm.py`).
- `examples/rb5/`, `examples/rb10/` — Rainbow Robotics arm projects. Each `basic.c` is a TCP client to the actual robot controller (ports 5000 cmd / 5001 data, hard-coded `192.168.0.8`). `rb10/chenv.cpp` is the optional Chrono backend (built via `rb10/build.sh`, requires Chrono+Irrlicht+Eigen+Bullet headers).
- `examples/cartpole/` — minimal RL example. `train` uses Stable-Baselines3 PPO over a Gymnasium wrapper of `tact.Env`; `cartpole.py` controller supports states `home`, `zero`, `test1` (MPC via `scipy.optimize.minimize`), `test2` (PPO policy).
