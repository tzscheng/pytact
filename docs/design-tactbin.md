# tactbin v1

`tactbin` is the compiled-model handoff format for the standalone C API.
It keeps YAML parsing and schema evolution in Python while letting C programs
load a model without importing Python.

Intended flow:

```bash
python -m tact.compile_model robot.yml -o robot.tactbin
```

```c
#include "tact.h"

tact_model_t *m = NULL;
tact_state_t *s = NULL;

tact_load_model("robot.tactbin", &m);
tact_create_state(m, &s);
tact_step(m, s, tau);
tact_destroy_state(s);
tact_destroy_model(m);
```

The v1 exporter serializes the same compiled arrays that `Model._create_c_handle`
passes to `tact_create`. The first C loader can therefore reconstruct the current
low-level handle without implementing a YAML loader in C.

## Object Split

The public C API should look like MuJoCo's `mjModel` / `mjData` split:

| tact | Owns |
| --- | --- |
| `tact_model_t` | immutable compiled topology, geometry, frames, feeds, defaults |
| `tact_state_t` | mutable `q`, `qd`, warm-start lambda, outputs, scratch/handle |

The first implementation may keep the existing internal `tact_t` inside each
`tact_state_t`. That avoids a large internal split while presenting the right
public API.

## Public C Contract

`tact_load_model(path, &m)` owns all arrays decoded from the tactbin file. The
caller releases that immutable model with `tact_destroy_model(m)`.
`tact_create_state(m, &s)` allocates mutable state for that exact model: `q`,
`qd`, the two warm-start lambda buffers, and the internal `tact_t` handle used
to step. Destroy states before destroying their model.

The main accessors are:

| Function | Contract |
| --- | --- |
| `tact_model_info(m, &info)` | fills counts and `dt`; no ownership transfer |
| `tact_q(s)`, `tact_qd(s)`, `tact_y(s)` | borrowed pointers owned by `s`; re-read after each step |
| `tact_step(m, s, tau)` | one zero-PD step; `tau == NULL` means zero torque |
| `tact_step_pd(m, s, tau, q_ref, qd_ref, kp, kd)` | one step with optional implicit joint PD |

All control arrays are `nq`-length when non-NULL. `tact_step_pd` activates the
P term only when `q_ref && kp` are non-NULL, and the D term only when
`kd && (q_ref || qd_ref)` are non-NULL. This mirrors `Model.step`. Implicit PD is
only a supported public contract for 1-DoF `rev`/`lin` joints; free-joint gains
should remain zero/NULL in standalone C callers.

`tact_state_t` carries warm-start lambda internally and flips between two
buffers on each step. That makes the standalone stateful API convenient while
preserving the lower-level `tact_step_lcp` contract, where callers explicitly
thread immutable `lam_in` / `lam_out`.

## File Format

All integers are little-endian. All float arrays are `float64`. The file starts
with:

```c
struct tactbin_header {
    char     magic[8];     // "TACTBIN\0"
    uint32_t version;      // 1
    uint32_t n_chunks;
};
```

Then `n_chunks` records:

```c
struct tactbin_chunk_header {
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
loaders, but a missing or malformed required chunk makes `tact_load_model`
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
| `mesh_slots_i32` | int32 | mesh slot ids in the same order as `mesh_paths` |
| `mesh_paths` | utf8 | NUL-separated absolute mesh paths; C loader calls `set_mesh_path(slot, path)` |
| `hfield_meta_i32` | int32 | `(n_hfield, 3)` rows `[slot, nrow, ncol]` |
| `hfield_size_f64` | float64 | `(n_hfield, 2)` rows `[sx, sy]` |
| `hfield_offsets` | int32 | `(n_hfield + 1,)` offsets into `hfield_data_f64` |
| `hfield_data_f64` | float64 | packed row-major height samples in meters |

## Compatibility

The file is a compiled ABI artifact, not a source scene format. A `tactbin`
version bump is expected whenever `tact_create`'s array contract changes.
YAML remains the source format owned by `pytact`.

## Current Test Gates

`tests/test_tactbin_smoke.py` covers:

- header sanity and malformed file rejection: bad version, missing required
  chunk, duplicate required chunk, bad chunk shape, inconsistent `lam_size`,
  truncated payload, missing mesh paths, bad hfield offsets;
- C API vs Python wrapper parity for arm dynamics, implicit PD, feedback `y`,
  free-body gravity, single-contact settling, multi-body box manifold contact,
  mesh contact, joint friction, and joint limit;
- mesh and hfield tactbin load paths.

`tests/test_packaging_smoke.py` additionally checks that the installed package
exports the tactbin C API from `tact/bin/libtact.so`, that
`python -m tact.compile_model` works from package assets, and that a
`ctypes`-loaded tactbin state steps with Python parity. With
`TACT_PACKAGING_INSTALLED=1`, it avoids injecting the checkout into
`sys.path`, so the same test can run against a wheel installed into a target
directory.
