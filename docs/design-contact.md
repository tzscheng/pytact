# Contact narrowphase: multi-point manifold + dispatch

Narrowphase dispatch table, the multi-point contact manifold (sub_id slotting),
and the box_wall stability/drift result. Moved out of CLAUDE.md 2026-05-31 to keep
it as orientation. Related: `design-hfield.md` (hfield detectors + fidelity roadmap),
`design-lcp-perf.md` (the A-matrix cost behind the box_wall slowdown).

**Multi-point contact manifold (Phase 2 contact-point refactor):**

The narrowphase API `collision_check(t1, p1, t2, p2, out, max_pts) → n_points` lets a single cpair produce up to `MAX_PTS_PER_PAIR=4` (currently — `tact.h` `#define`) contact points. Box-box points share one normal; box-hfield points may each carry their own normal (the LCP builds a per-point tangent frame — see `docs/design-hfield.md`). Indexing convention throughout LCP (cdata, A matrix rows, `lam_prev`):

```
slot = cpair_idx * MAX_PTS_PER_PAIR + sub_id    # sub_id ∈ [0, n_points)
```

**Dispatch (current, as of 2026-05-29):** all hfield pairs now have dedicated detectors (the previous `guarded -1` for capsule/cyl/mesh-hfield is gone).

| Pair | Detector | Manifold | Notes |
|---|---|---:|---|
| box-box | `box_box_manifold` (box_box.c) | ≤4 | SAT + face clipping, polar-angle sub_id |
| box-sphere | `box_sphere_contact` | 1 | analytic clamp-to-box |
| sphere-sphere | `sphere_sphere_contact` | 1 | closed-form |
| capsule-sphere | `capsule_sphere_contact` | 1 | segment closest pt + sphere-sphere |
| capsule-capsule | `capsule_capsule_contact` | 1 | Ericson segment-segment closest pair |
| cylinder-sphere | `cylinder_sphere_contact` | 1 | 4-region classification (side/cap/edge/inside) |
| box-capsule | `box_capsule_contact` | ≤2 | 3-sample + 2-pt manifold via footprint clip |
| sphere-hfield | `sphere_hf_contact` | 1 | closest point over footprint cells |
| box-hfield | `box_hf_contact` | ≤4 | Tier 2: 8 box vertices vs cell planes |
| capsule-hfield | `capsule_hf_contact` | ≤2 | 3-sample sphere_hf_local + footprint clip |
| cylinder-hfield | `cylinder_hf_contact` | ≤4 | Tier 2: 82 surface samples (cap+rim+side) |
| mesh-hfield | `mesh_hf_contact` | ≤4 | Tier 2: vertex sampling, streaming top-N |
| any other | `collision_check_mpr` (mpr.c) | 1 | MPR/EPA single-point witness |

Height-field shape/raycast/contact + the contact-fidelity roadmap (Tier 2 → 2.5 → full SAT) are documented in `docs/design-hfield.md`. **Remaining narrowphase gap**: `box-plane` (no plane shape type yet — hfield acts as plane), and cyl-vs-{box,cyl,cap,mesh} plus mesh-vs-{box,cyl,cap,mesh} still use MPR (where it works well).

**Canonical sub_id ordering:** when n_points ≥ 2, the narrowphase sorts contact points by polar angle around their centroid in the tangent plane (CCW). Stable across frames as long as the contact polygon topology doesn't change → warm-start λ keeps its meaning. Topology changes (polygon vertex count shift) cost one frame of cold-start PGS convergence.

**Drift observed on `demos/box_wall.yml`** (4-row 18-brick running-bond wall, 30 s sim):

| approach | edge drift @ 30 s | wall time |
|---|---|---|
| `solver: lcp`, single-point MPR (Phase 1 baseline) | ~12 mm (bottom row spreads outward) | 33 s |
| `solver: lcp`, 4-point box-box manifold (Phase 2) | ~0.001 mm | 230 s |

The 7× LCP slowdown comes from the larger A matrix (`6·MAX_PTS_PER_PAIR·n_pair` upper bound); narrowphase itself got faster (SAT is 1.5–2× faster than MPR per call). See `tests/box_wall_stability.py` for the harness.
