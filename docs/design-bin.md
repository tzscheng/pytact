# bin v1

`bin` is the compiled-model handoff format for the standalone C API.
It keeps YAML parsing and schema evolution in Python while letting C programs
load a model without importing Python.

Intended flow:

```bash
python -m tact.compile robot.yml -o robot.bin
```

```c
#include "tact.h"

tact_t *h = NULL;
tact_ctx_t *ctx = NULL, *ctx_next = NULL;

tact_load("robot.bin", &h);
tact_create_ctx(h, &ctx);
tact_create_ctx(h, &ctx_next);
tact_step(h, q, qd, tau, ctx, q_next, qd_next, y, ctx_next);
tact_destroy_ctx(ctx);
tact_destroy_ctx(ctx_next);
tact_destroy(h);
```

The v1 exporter serializes the same compiled arrays that `Model._create_c_handle`
passes to the C handle. The C loader can therefore reconstruct the same
low-level handle without implementing a YAML loader in C.

## Object Split

The public C API keeps immutable model data separate from caller-owned runtime
state:

| tact | Owns |
| --- | --- |
| `tact_t` | immutable compiled topology, geometry, frames, feeds, defaults, reusable scratch |
| `tact_ctx_t` | caller-threaded warm-start lambda |
| caller arrays | mutable `q`, `qd`, `q_next`, `qd_next`, `y`, control inputs |

This mirrors Python `Model.step(q, qd, tau, ctx=...)`: `q`/`qd` are explicit
inputs, and warm-start state is explicit through `ctx`.

## Public C Contract

`tact_load(path, &h)` owns all arrays decoded from the bin file plus the
scratch used by step/render calls. The caller releases it with
`tact_destroy(h)`.
`tact_create_ctx(h, &ctx)` allocates a warm-start lambda buffer for that handle.
Destroy contexts before or after destroying their handle only if they are no
longer passed to tact APIs; normal ownership is `tact_destroy_ctx(ctx)`.

The main accessors are:

| Function | Contract |
| --- | --- |
| `tact_info(h, &info)` | fills counts and `dt`; no ownership transfer |
| `tact_q0(h)`, `tact_qd0(h)` | borrowed default-state pointers owned by `h` |
| `tact_frame_id(h, name)` | resolves a frame name to the integer id used by query APIs |
| `tact_frame_count(h)`, `tact_frame_name(h, id)` | enumerate loaded frame names |
| `tact_create_ctx(h, &ctx)` | allocates one warm-start lambda buffer |
| `tact_ctx_lam(ctx)` | borrowed lambda pointer owned by `ctx` |
| `tact_step(h, q, qd, tau, ctx_in, q_out, qd_out, y_out, ctx_out)` | one zero-PD step; `tau == NULL` means zero torque |
| `tact_step_pd(h, q, qd, tau, q_ref, qd_ref, kp, kd, ctx_in, q_out, qd_out, y_out, ctx_out)` | one step with optional implicit joint PD |
| `tact_render(h, q)` | renders the loaded model at `q` in a GLFW window |

All control arrays are `nq`-length when non-NULL. `tact_step_pd` activates the
P term only when `q_ref && kp` are non-NULL, and the D term only when
`kd && (q_ref || qd_ref)` are non-NULL. This mirrors `Model.step`. Implicit PD is
only a supported public contract for 1-DoF `rev`/`lin` joints; free-joint gains
should remain zero/NULL in standalone C callers.

Use separate `ctx_in` and `ctx_out` for referentially transparent stepping.
Passing NULL `ctx_in` cold-starts; passing NULL `ctx_out` discards warm-start.

## File Format

All integers are little-endian. All float arrays are `float64`. The file starts
with:

```c
struct bin_header {
    char     magic[8];     // "TACTMDL\0"
    uint32_t version;      // 1
    uint32_t n_chunks;
};
```

Then `n_chunks` records:

```c
struct bin_chunk_header {
    char     tag[16];      // NUL-padded ASCII
    uint32_t dtype;        // 1=int32, 2=float64, 3=utf8
    uint32_t ndim;         // 0..4
    uint64_t shape[4];     // unused trailing dims are zero
    uint64_t nbytes;
    uint8_t  payload[nbytes];
};
```

No padding is inserted between chunks.

The C loader treats chunk headers as part of the ABI: `dtype`, `ndim`, `shape`,
and `nbytes` must match exactly for known chunks. Required scalar chunks
(`dims_i32`, `sim_f64`, `sim_i32`) reject duplicates. Generic required arrays
also reject duplicates when their pointer has already been filled. Unknown
chunks are ignored so future debug metadata can be added without breaking old
loaders, but a missing or malformed required chunk makes `tact_load`
fail.

## Required Chunks

The v1 required chunks are:

| Chunk | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `dims_i32` | int32 | `(8,)` | `nb, nq, n_shape, n_pair, n_frame, n_feed, y_size, lam_size` |
| `sim_f64` | float64 | `(6,)` | `dt, erp, slop, cfm_scale, v_rest_thresh, tol` |
| `sim_i32` | int32 | `(2,)` | `integrator, iters` |
| `parent` | int32 | `(nb,)` | body parent indices, root parent = -1 |
| `jtype` | int32 | `(nb,)` | joint type codes |
| `X` | float64 | `(nb, 6, 6)` | spatial transforms |
| `I6` | float64 | `(nb, 6, 6)` | spatial inertias |
| `Ti` | float64 | `(nb, 4, 4)` | joint transforms |
| `ff`, `sk`, `floss`, `armature`, `jnt_lo`, `jnt_hi` | float64 | `(nq,)` | per-DoF joint parameters |
| `g` | float64 | `(3,)` | gravity |
| `ctype`, `cbody`, `craycast` | int32 | `(n_shape,)` | shape metadata |
| `cshape` | float64 | `(n_shape, 3)` | shape parameters |
| `ctran` | float64 | `(n_shape, 4, 4)` | shape transforms |
| `cparam` | float64 | `(n_shape, 13)` | contact material parameters |
| `cpair` | int32 | `(n_pair, 2)` | collidable shape pairs |
| `q0`, `qd0` | float64 | `(nq,)` | default state |
| `feed_kinds` | int32 | `(n_feed,)` | feedback kind codes |
| `feed_offsets` | int32 | `(n_feed + 1,)` | packed feed index offsets |
| `feed_idx` | int32 | `(n_feed_idx,)` | packed frame indices referenced by feeds |
| `fbody` | int32 | `(n_frame,)` | frame body indices |
| `ftran`, `ftran_inv` | float64 | `(n_frame, 4, 4)` | frame transforms |

Optional chunks:

| Chunk | Type | Meaning |
| --- | --- | --- |
| `meta_json` | utf8 | debug metadata and name maps; a minimal C loader may ignore it |
| `frame_names` | utf8 | NUL-separated frame names in frame-id order; enables `tact_frame_id`/`tact_frame_name` |
| `mesh_slots_i32` | int32 | mesh slot ids in the same order as `mesh_paths` |
| `mesh_paths` | utf8 | NUL-separated absolute mesh paths; C loader calls `set_mesh_path(slot, path)` |
| `hfield_meta_i32` | int32 | `(n_hfield, 3)` rows `[slot, nrow, ncol]` |
| `hfield_size_f64` | float64 | `(n_hfield, 2)` rows `[sx, sy]` |
| `hfield_offsets` | int32 | `(n_hfield + 1,)` offsets into `hfield_data_f64` |
| `hfield_data_f64` | float64 | packed row-major height samples in meters |

## Compatibility

The file is a compiled ABI artifact, not a source scene format. A `bin`
version bump is expected whenever the compiled model array contract changes.
YAML remains the source format owned by `pytact`.

## Current Test Gates

`tests/test_bin_smoke.py` covers:

- header sanity and malformed file rejection: bad version, missing required
  chunk, duplicate required chunk, bad chunk shape, inconsistent `lam_size`,
  truncated payload, missing mesh paths, bad hfield offsets;
- C API vs Python wrapper parity for arm dynamics, implicit PD, feedback `y`,
  free-body gravity, single-contact settling, multi-body box manifold contact,
  mesh contact, joint friction, and joint limit;
- mesh and hfield bin load paths.

`tests/test_packaging_smoke.py` additionally checks that the installed package
exports the bin C API from `tact/bin/libtact.so`, that
`python -m tact.compile` works from package assets, and that a
`ctypes`-loaded bin state steps with Python parity. With
`TACT_PACKAGING_INSTALLED=1`, it avoids injecting the checkout into
`sys.path`, so the same test can run against a wheel installed into a target
directory.
