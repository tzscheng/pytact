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
| `mesh_json` | utf8 | mesh slot to absolute path table; v1 loader can use it to call `set_mesh_path` |

Height-field asset data is not frozen in v1. The current Python loader pushes
hfield grids directly into C slots and does not retain the full grids on `Model`.
Before supporting C-loaded hfields, store those grids in the compiled model and
add binary hfield chunks.

## Compatibility

The file is a compiled ABI artifact, not a source scene format. A `tactbin`
version bump is expected whenever `tact_create`'s array contract changes.
YAML remains the source format owned by `pytact`.
