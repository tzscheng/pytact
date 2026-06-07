# design-hfield.md — height-field terrain (collision, raycast, render) + contact roadmap

Scope: the `hfield` shape type (`HFIELD=105`) — a regular grid of heights forming a
non-convex terrain surface. Covers what shipped (slot storage, sphere/box contact, DDA
raycast, render), and — the durable part — the **contact-fidelity roadmap** (Tier 2 →
2.5 → full SAT) with the stability/accuracy/cost tradeoffs that decide how far to go.
Read this before extending hfield contact. Companion to `narrow.c` (`sphere_hf_contact`/
`box_hf_contact` + dispatch), `ray.c` (`ray_intersects_hfield`), `shape.c` (slot storage),
`tact.c` (raycast dispatch), `render.c` (`make_hfield`), `sim.py` (`hfield` parse),
`_clib.py` (`set_hfield_data`).

## 1. Data model (shipped)

An hfield is a slot of grid data, mirroring the mesh slot table (`shape.c` / `shape.h`):
`hf_nrow/hf_ncol/hf_sx/hf_sy/hf_minh/hf_maxh/hf_data[MAX_HFIELD=16]`. The grid spans
local `[-sx,sx]×[-sy,sy]` in XY with height along +Z; `hf_data[i*ncol+j]` is the height at
node (row i along +Y, col j along +X). Each cell is two triangles split on the p00→p11
diagonal: A=(p00,p10,p11), B=(p00,p11,p01). Python loads/scales the grid and pushes it via
`set_hfield_data` (no lazy C-side load, unlike meshes). YAML: `{type: hfield, file: x.bin
| data: [[...]], size: [sx, sy, sz]}`; `cshape[0]=slot`. `file:` is **MuJoCo's custom hfield
binary** (int32 nrow, int32 ncol, float32 data; raw meters here) read by `tact.load_hfield`
— ONE data file under `tact/hfields/` serves both backends (2026-06-08; .npy retired). tact
uses raw values × `sz`; MuJoCo normalizes the same file to [0,1], so the mjcf twin scene
carries `size[2]`=range + geom z=min (data-derived, printed by the generator; row order
verified identical via mj_ray probes at asymmetric grid nodes, max err ~1e-8). Worked
example: `examples/hf1.yml` ↔ `mjcf/hf1.xml` (+ `hfields/hf1.bin`,
`hfields/hf1_gen.py`).

Per-shape pose is the usual `ctran` (body-relative homogeneous transform), so an hfield can
sit on any body at any orientation; all narrowphase/raycast works in hfield-local frame and
maps results back to world.

## 2. Raycast — 2D DDA (shipped, replaces brute force)

`ray_intersects_hfield` (ray.c) walks only the cells the ray crosses (Amanatides-Woo), in
increasing-t order, with **first-hit = nearest early-exit**. Setup clips the ray to the
hfield 3D AABB (`[-sx,sx]×[-sy,sy]×[min_h,max_h]`), which folds in footprint rejection and
height-slab trimming. Plugged into `tact.c::raycast_cached` (case 105) + bounding sphere in
`rc_build_cache`. So `get_z` / lidar / depth cameras see the terrain.

- Perf (dog lidar 320×240 on hf1 (then terrain10) 101×101, 2026-05-28): **8916 ms → 10.2 ms/frame (874×)**;
  `get_z` 119 µs → 7.6 µs (15.7×). Bit-identical to brute force (lidar max|Δ|=0; get_z 3.6e-15).
- **Vertical-ray gotcha (fixed):** `get_z` shoots straight down → XY projection is a point.
  At grid-aligned query coords, FP in `floor((x+sx)/dx)` vs the reconstructed vertex `-sx+j*dx`
  can place the point in cell (j-1)/(i-1) while floor says (j)/(i), so a single-cell test
  misses. Fix: the vertical branch tests the **2×2 cell block** around (i,j) — only
  footprint-containing cells can be hit by a vertical ray, so this is brute-force-equivalent.
  Oblique rays self-heal (DDA steps into the adjacent cell), so only the single-point path
  needed it.
- Cost: O(cells-crossed) ≈ O(grid side) per ray. Not parallelized — `raycast_cached` is
  documented thread-safe (no per-ray shared writes); **OpenMP over the raymap pixel loop is a
  deferred, easy N-core multiplier** (after DDA, lidar is already realtime, so low priority).

## 3. Contact — current state and the per-point-normal invariant

Dispatch in `collision_check` (narrow.c), normal convention param1→param2 (flip when the
shape is param1):
- **sphere-hfield** (Tier 1, shipped): `sphere_hf_contact` — searches the cells overlapping
  the sphere's XY footprint, `closest_point_on_triangle` (Ericson) over those cells'
  triangles, single contact. Tunneling caveat: contact found only while the center stays
  within r of the surface (no signed/inside test); normal forced to local +Z when the center
  is at/below the surface. Single-point is correct for a sphere (no flat feature, like
  `box_sphere_contact`); it under-detects only in a concave crease/groove/pit tighter than
  ~2r, where the sphere should touch 2+ faces but only the nearest is reported (rare in
  terrain, degrades gracefully — the LCP still settles it). Multi-pointing it (emit one point
  per *penetrating* footprint triangle, then prune ≤4, mirroring `box_hf_contact`) is a clean
  additive change, deferred until a concave-wedging use case needs it.
- **box-hfield** (Tier 2, shipped): `box_hf_contact` — see §4.
- **capsule/cylinder/mesh-hfield**: **guarded** — an hfield is non-convex and must never reach
  the convex MPR fallback (it would collide against the terrain's convex hull, a giant dome),
  so any unhandled hfield combo returns -1 (no contact). Tier 3.

**Key enabling invariant (verified, `lcp.c:303-311`):** the solver reads each contact
point's own normal `out[3..5]` and builds an independent tangent frame / Jacobian / material
per point (`choose_rotation` per point). So contact points in one pair may carry **different
normals** — a box straddling cells of different slope, or any per-point-normal manifold, is
handled natively. This is what makes the box-on-terrain manifold work; do not assume a single
shared normal per pair.

## 4. Tier 2 (shipped): box vertices vs local terrain plane

`box_hf_contact` tests the box's 8 vertices: each vertex is located in its cell, the cell
triangle it lies in is chosen by the diagonal, and the **signed distance to that triangle's
plane** decides contact (point = vertex projected onto the plane, normal = the cell triangle's
up-normal, depth = penetration). Up to `MAX_PTS_PER_PAIR=4` deepest are returned. Cost is
**O(8), independent of box size** (each vertex maps to one cell).

Verified (2026-05-28): narrowphase BOX↔HFIELD yields a 4-point manifold (the bottom corners);
box rests flat at z≈hz with tilt 0 (stable, no jitter); box rests flush on a 26.6° slope at
tilt = slope angle, |v|=|ω|=0, height = hz/cos (exact); no tunneling (signed plane distance,
unlike sphere_hf); regression `test_traj.py` 10/10 bit-identical; dog/zen load with terrain
and feet form contact pairs.

### What Tier 2 misses
- **Interior peaks (failure mode A):** a terrain peak under the box *center* (between the
  sampled corners) is not seen → the box sinks until its corners hit lower surrounding terrain
  and the peak interpenetrates. Not tunneling (corners still catch); the resting height /
  contact timing is just wrong. Matters when terrain features are smaller than the box.
  Pathological sub-case: a box perched over a peak narrower than the box (all 4 corners hang
  over lower terrain) → **no contact at all → box drops suddenly** until corners catch.
- **Edge-edge (also missed):** a box edge crossing a terrain ridge where the deepest point is
  mid-edge on both — no vertex involved. (Closed only by full SAT, §6.)
- **>4 contact points:** a large box over wavy terrain capped at 4 (shared box-box limit).

Quantitatively the interior-peak error ≈ how far the terrain's interior maximum (within the
box footprint) rises above the corner-defined plane. For `hf1` (gentle, 0.1 m cells,
shortest wavelength ~2.5 m) a footplate-sized box bows ~2–3 mm → negligible. It bites on
sharp sub-foot features (rocks, stair nosings, ridges).

## 5. Tier 2.5 (not built): add terrain vertices vs box

Add a **second pass**: iterate the terrain grid vertices under the box footprint and test
each against the box (transform into box-local — axis-aligned — check half-extents, penetration
= min face distance, normal = that box-face axis). Feed both passes into the same candidate
buffer + deepest-≤4 pruning. This is the **box-vertex-vs-face + terrain-vertex-vs-face**
two-way vertex method (both vertex-face directions of the convex-convex feature set).

- **Closes** failure mode A (a peak's grid vertex now pokes into the box face → contact) and
  the sudden-drop sub-case. So the "ground height the robot feels" becomes correct.
- **Still misses edge-edge** (§4) — that needs full SAT.
- **Keeps the stability properties** of Tier 2 (vertex-based, plane/box-axis normals → no
  internal-edge artifacts, §6), at the cost of mild contact-set variation as terrain vertices
  enter/leave the footprint (slightly less "fixed" than Tier 2's 4 corners).

### Architecture: 2 → 2.5 is purely additive
The box-vertex pass stays unchanged; you add a parallel pass feeding the same buffer/pruning.
The scaffolding (dispatch wiring, cell-overlap range, triangle generation, frame transforms,
≤4 pruning, the per-point-normal LCP path) is shared. ~30–50 lines added, nothing removed.

## 6. Full SAT/clipping (not built): box vs each overlapping triangle

The production-engine approach (see §8): for each overlapping cell triangle, run a full
**box-triangle SAT** (box faces + triangle face + 9 box-edge×triangle-edge cross axes) +
Sutherland-Hodgman clipping to build a per-triangle manifold; merge across triangles; reduce
to ≤4. This is geometrically complete (vertex-face **and** edge-edge), so it also closes the
ridge-balance case.

- **Cost it adds:** the **internal-edge problem** — shared edges between adjacent triangles
  emit spurious normals that snag a sliding/planted foot (jitter, phantom forces). Must be
  corrected with precomputed edge adjacency (Bullet `btTriangleInfoMap` /
  `btAdjustInternalEdgeContacts`; PhysX "active edges"). New machinery with no precursor in
  the hfield code (but the SAT+clip kernel itself can borrow patterns from narrow.c's box-box section).
- **Contact flood:** generates up to ~2·(cells) raw contacts → reduction cost + risk of a
  fatter LCP if not well-reduced (cf. box_wall 1→4-point manifold made LCP ~7× slower).

### Architecture: 2.5 → full SAT is "re-engine", not "continue"
The scaffolding (dispatch, cell overlap, triangle gen, transforms, pruning, per-point-normal
LCP) carries over (~half the lines). But the **contact kernel is swapped** (vertex tests →
SAT+clip), the Tier 2/2.5 vertex passes are **discarded** (SAT subsumes their vertex-face
cases; keeping both double-counts), and internal-edge correction is **net new**. Not "undo and
redo" — the frame stays — but the algorithmic heart is replaced. Tier 2.5 is not wasted if you
later go full SAT (scaffolding survives), but plan it as a swap, not an extension.

## 7. Tradeoffs: stability vs accuracy vs cost

**Stability (contact smoothness, jitter, warm-start).** Tier 2's fixed 4 corners give a
maximally-spread support polygon with clean plane normals and a stable contact-point identity
→ best warm-start, no internal-edge snag → **best for walking on gentle terrain**. Full SAT is
**worst** here (internal-edge artifacts + contact-set churn as the foot moves over triangles)
— which is exactly why engines bolt on internal-edge correction. Tier 2.5 ≈ Tier 2 minus a
little (terrain vertices enter/leave). Caveat: Tier 2's stability is partly from
*under-detecting* — it's "stable but the felt ground height is wrong" on sub-foot features,
and "stable until the sudden-drop pathological case."

**Accuracy.** full SAT > 2.5 > 2. Tier 2.5 recovers ground-height correctness (peaks) without
the internal-edge instability; full SAT adds only edge-edge (rare ridge-balance).

**Cost (analytical, per box-hfield pair; K = cells covered).**

| | narrowphase scale | per-cell constant | sqrt | contacts → LCP |
|---|---|---|---|---|
| Tier 2 | O(8) fixed | — | 8 | ≤4 |
| Tier 2.5 | O(8) + O(K) | ~28 flop, 0 sqrt | 8 | ≤4 (LCP unchanged) |
| full SAT | O(2K) | ~700 flop, 2–4 sqrt + edge corr. | ~2K | up to ~2K → reduce (LCP grows) |

- 2 vs 2.5: ~2× flops for a small foot (K≈6 → ~300 vs ~640 flop), growing to ~10× for a large
  box / fine grid; **same sqrt count, same LCP cost** (both prune to ≤4). In absolute terms
  this is noise — box-hfield narrowphase for a few pairs is nanoseconds vs the LCP/dynamics of
  a step. **So compute does not decide 2-vs-2.5; the stability/accuracy tradeoff does.**
- full SAT: ~20–30× heavier per overlapping cell than 2.5's terrain pass, *plus* internal-edge
  machinery, *plus* a heavier LCP from the contact flood. The real cost is downstream, not the
  narrowphase flops.

## 8. How production engines do it

All collide the geom against **each overlapping terrain triangle** (convex-vs-triangle), not
vertex sampling — so interior peaks are caught (the peak's triangle collides), and they pay
the internal-edge cost:
- **MuJoCo:** colliding geom vs the overlapping hfield triangle **prisms** (base-closed → no
  tunneling) via its `mjc` collision functions; recommends modest hfield resolution for cost.
- **Bullet** (`btHeightfieldTerrainShape`): concave shape, `processTriangle` runs
  convex-vs-triangle per overlapping cell; internal edges fixed via `btTriangleInfoMap` /
  `btAdjustInternalEdgeContacts`.
- **PhysX / Isaac:** first-class heightfield narrowphase + PCM; "active edges" for internal
  edges. Common for legged-robot RL on rough terrain.
- **ODE** (`dHeightfield`): build overlapping triangles, collide geom against each, merge.

tact's Tier 2 deliberately takes the *opposite* trade (vertex sampling) to get clean normals
and no internal-edge artifacts cheaply — at the cost of sub-box features. Tier 2.5 recovers
most of the accuracy gap while keeping that artifact-free property.

## 9. Decision (deferred 2026-05-28)

Staying on **Tier 2** while the feature gets real use. Guidance for the next decision:
- **Gentle terrain + flat feet + "develop a stable walker"** → Tier 2 is already a sweet spot
  (stable, cheap, artifact-free). Don't add complexity for its own sake.
- **Rougher terrain, or controller senses foot-ground geometry (touchdown height/timing), or
  the sudden-drop case appears** → do **Tier 2.5** (additive, ~2× of a negligible cost, LCP
  unchanged, no new instability). This is the recommended next step if accuracy is needed.
- **Need ridge-balance / edge-edge fidelity and can manage internal-edge correction** → full
  SAT (re-engine; consider skipping 2.5 since SAT subsumes its vertex passes).

Also deferred: Tier 3 (capsule/cylinder/mesh-hfield), OpenMP raymap (§2), CLAUDE.md hfield
schema note, and a `tests/scenes/` frozen hfield scene + baseline (baseline capture is a
deliberate, reviewed act).
