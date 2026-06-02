# design-lcp-perf.md — LCP solver performance (block-diagonal exploitation)

Scope: performance of `contact_lcp` / `tact_step_lcp` (`lcp.c`). The contact
solver is the dominant cost for multi-robot and large-pile scenes. This doc
records the profiling result, the **S1** optimization that shipped, the planned
**S2**, and — importantly — the invariants that keep S2 compatible with future
general constraints (joint limits, loop closure / weld). It is the durable,
in-repo companion to the work; read it before touching the Delassus build/solve.

## 1. Bottleneck (profiled 2026-05-24, `-DLCP_PROF`)

The LCP step builds the Delassus system `A = J·M⁻¹·Jᵀ` (dense), then PGS-solves
it. Two stages dominate, and box-pile vs multi-robot have *opposite* sub-bottlenecks:

| scene | nc / F | `factorY` (M⁻¹ factor + M⁻¹Jᵀ) | `Amat` (A = J·Yᵀ) |
|---|---|---|---|
| box_wall (large nc, F=108) | 131 / 108 | 31% | 63% |
| multi-robot (small nc, F grows with K) | small / 6·K·dof | 52–60% | 39–45% |

Multi-robot scaling was **O(K³)** (zen K=4/8/12 = 0.69 / 4.05 / 12.75 ms/step;
real-time at dt=1ms broke around zen K=4). Root cause: the code treated `M` and
`J` as dense, but:

- **`M` is block-diagonal by kinematic connected component.** `M[a][b]` (CRB) is
  structurally nonzero only when bodies a,b are in ancestor–descendant relation,
  so two moving DoF couple iff one body is a (possibly fixed-link-spanning)
  ancestor of the other. The static world doesn't couple independent subtrees →
  each robot / free body / pile component is its own dense block. `crb_featherstone`
  leaves cross-block entries at the `memset`-0 they start at.
- **`J` is sparse** — each contact row touches only its two bodies' DoF (≤12 of F).

Chosen fix: **S1 (block-diagonal M factor/solve)** + **S2 (sparse-J A build)**.
Not "islands" (contact-graph union-find): islands only help *disconnected*
components and give nothing inside a connected pile (box_wall is one island).
S1+S2 exploit sparsity *within* a component too, so S1+S2 ⊋ islands.

## 2. S1 — block-diagonal M factor/solve (SHIPPED 2026-05-25)

Implementation in `lcp.c`:

- **Cached partition** (`g_part`, static): connected components from
  `parent[]`/`jtype[]` via union-find (merge each moving body with its nearest
  moving ancestor). Topology-only → changes solely on add/delete, so it is
  computed once and reused, validated by an exact `(nb, parent, jtype)` compare
  (`lcp_ensure_partition`). `nblk ≤ nb`.
- **PASS 3**: for each block, gather its dense sub-matrix from `M_full` into the
  packed buffer `Mpack` (the former dense `Mf` slot — no arena growth, since
  `Σ s_b² ≤ F²`) and `ldlt_factor` it. Then build `Y = M⁻¹Jᵀ` solving **only the
  ≤2 blocks each contact touches** (`lcp_block_solve` per involved block).
- **PASS 5**: `dqd = M⁻¹·(Jᵀλ)` solves every block (each covers disjoint DoF).

### Why it is bit-identical to the dense path

The cross-block entries of `M` are exactly `0.0`, so a dense LDLᵀ produces no
fill-in across blocks and every dropped term is `x ± 0.0`. Factoring blocks
independently is therefore the same arithmetic as the dense factor restricted to
each block. Verified: regression `tests/regression/test_traj.py` 10/10 (atol
1e-12), plus a direct zen K=2 / K=8 block-vs-dense run over 400 steps =
**1.9e-16 / 2.7e-16** max|Δq| (machine epsilon; the only residual is `-ffast-math`
reduction reordering — the block factor sums the same terms as the dense factor
but without the interspersed structural zeros that change vectorized grouping).

### Gotcha (cost a debugging cycle — keep the test in mind)

A **contact body can be a fixed body** (`jtype=0`), e.g. zen's rigidly-mounted
box feet. Its contact Jacobian (`jacob_whitney`) is nonzero in its *moving
ancestors'* block, not its own (it has no DoF). The first cut mapped fixed bodies
to `blk_of_body = -1` and skipped the solve → the robot fell (Y off by ~150,
diverged at step 1). Fix: a final pass in `lcp_build_partition` maps every fixed
body to its **nearest-moving-ancestor's block** (truly static bodies, with no
moving ancestor, stay −1 — their Jacobian row is all zero, correctly skipped).
The regression suite did **not** catch this: its only contact scenarios
(box_wall / obj1 / sphere) use *moving* contact bodies = single 6×6 free-body
blocks; arms/cartpole have no contact. A free-base robot with fixed feet (18×18
articulated blocks) was the discriminating case — **test one if you touch this.**

### Measured S1-alone speedup (`-DLCP_PROF`, vs dense baseline)

| scene | `factorY` | total ms/step | `Amat` share now |
|---|---|---|---|
| box_wall | 2392 → 84 µs (**28×**) | 8.45 → 6.23 (**1.36×**) | 66% → 91% |
| zen K=8 | 2006 → 89 µs (**23×**) | 4.04 → 2.10 (**1.93×**) | 45% → 90% |

zen scaling K=4/8/12: 0.69/4.05/12.75 → 0.44/2.10/6.44 ms (1.6/1.9/2.0×, grows
with K). S1 alone removes `factorY`; `Amat` (now ~90% everywhere) is the S2 target.
The "combined S1+S2 ≈ 8–9×" estimate requires S2.

## 3. S2 — sparse-J A build (SHIPPED 2026-05-25, win(a) only)

`A[ri][rj]` is a structural zero unless constraint rows ri,rj share an M-block;
otherwise skipped. `A` stays **dense-stored** so PGS is unchanged. Reuses S1's
partition via a **per-row** block-support table `row_blocks[]` (generalized from the
former per-contact `cblk` on 2026-05-31 when the row-table refactor — Phase 2 of
`docs/design-joint-friction.md` — made the A-build row-driven per I2; bit-identical,
all 10 regression scenarios). Implementation in `lcp.c`: `memset(A,0)`, then for each
sharing row pair compute `A[ri][rj]` with the unchanged full-F `dotN`.

### Two possible wins — only (a) is bit-identical

- **win(a) — skip non-sharing pairs (SHIPPED).** A non-sharing pair's dense dot is a
  sum of exact-`0.0` products = exactly `0.0` = the memset, so skipping is
  **bit-identical**; sharing entries use the unchanged full-F dot. Whole A matrix is
  bit-identical → regression 10/10, no re-capture. For K independent robots,
  cross-robot contacts never share → Amat goes O(K³)→O(K).
- **win(b) — column-restrict the dot to the shared block (DEFERRED).** Cuts per-entry
  cost O(F)→O(block). NOT bit-identical: it reassociates the float sum vs the dense
  multi-accumulator `dotN` (`-ffast-math`) — a ~1e-12 diff that **box_wall's
  marginally-stable stack amplifies past the 1e-12 regression bound** (1.5e-6 by step
  80, on the A diagonal PGS divides by). Skipping zeros is exact; restricting a nonzero
  sum is not. Measured incremental gain over win(a) is only **~1.3–1.6×** (not the ~10×
  the element count suggests — see resume kit §3.1). Deferred 2026-05-25 (modest gain
  vs. losing the exact-match safety net); revisit with a reviewed baseline re-capture
  if huge piles / many-robot real-time become the bottleneck.

### Measured combined S1+S2 (win a) speedup (`-DLCP_PROF`, vs dense baseline)

| scene | Amat | total ms/step | total speedup |
|---|---|---|---|
| box_wall | 5658 → 1273 µs (4.4×) | 8.45 → 1.90 | **4.4×** |
| zen K=8 | 1729 → 240 µs (7.2×) | 4.04 → 0.65 | **6.2×** |

zen scaling K=4/8/12: dense 0.69/4.05/12.75 → 0.28/0.65/1.18 ms (**2.5×/6.2×/10.8×**,
grows with K; ~K^2.6 → ~K^1.3). Real-time (dt=1ms) for 12 zen was broken (12.75ms),
now ~1.18ms. The ~8–9× combined estimate assumed win(a)+win(b); win(a) alone lands
4.4–10.8× bit-identically, the remainder is win(b)'s (non-exact) column restriction.

### 3.1 win(b) resume kit (DEFERRED — pick up here if needed)

**What to change:** in `lcp.c`'s A build (the `for (b=0; b<6; b++)` inner loop of the
sharing-pair block), replace the full-F dot

```c
Arow[b] = dotN(Ji, Y + (size_t)(6*kj + b) * F, F);
```

with a dot restricted to contact kj's block columns. Two variants were measured:

```c
/* (i) gather — works for ANY block layout (contiguous or interleaved) */
const double *Yj = Y + (size_t)(6*kj + b) * F;
double s = 0.0;
if (c0 >= 0) { int o=g_part.blk_off[c0],e=g_part.blk_off[c0+1];
               for(int t=o;t<e;t++){int cc=g_part.blk_dof[t]; s+=Ji[cc]*Yj[cc];} }
if (c1 >= 0) { int o=g_part.blk_off[c1],e=g_part.blk_off[c1+1];
               for(int t=o;t<e;t++){int cc=g_part.blk_dof[t]; s+=Ji[cc]*Yj[cc];} }
Arow[b] = s;

/* (ii) contiguous — faster (keeps SIMD) but ASSUMES each block's DoF are a
        contiguous range; only valid when blk_dof[blk_off[b]..] is consecutive.
        Needs a per-block "contiguous?" flag + fall back to (i) otherwise. */
const double *Yj = Y + (size_t)(6*kj + b) * F;
double s = 0.0;
if (c0 >= 0) { int o=g_part.blk_off[c0],lo=g_part.blk_dof[o]; s += dotN(Ji+lo, Yj+lo, g_part.blk_off[c0+1]-o); }
if (c1 >= 0) { int o=g_part.blk_off[c1],lo=g_part.blk_dof[o]; s += dotN(Ji+lo, Yj+lo, g_part.blk_off[c1+1]-o); }
Arow[b] = s;
```

(`c0,c1` = `row_blocks[2*rj], row_blocks[2*rj+1]`, already in scope.)

**Measured (wall ms/step), build `-DLCP_PROF`:**

| scene | win(a) (shipped) | +win(b) gather | +win(b) contiguous |
|---|---|---|---|
| box_wall | 1.90 | 1.30 | 1.19 (Amat 1273→597 µs) |
| zen K=8 | 0.65 | 0.56 | 0.50 (Amat 240→83 µs) |

So win(b) reaches the original estimate (box_wall ~7×, zen K=8 ~8× combined vs dense),
but the **incremental** gain over the shipped win(a) is only ~1.3–1.6×: Amat is the
biggest single stage (~50%) but not the whole step, and the gather variant loses SIMD
(why (ii) is faster). The element-count "~10×" never materializes at the step level.

**Why it's blocked (must resolve before merging win(b)):**
1. NOT bit-identical → `tests/regression/test_traj.py` fails on box_wall / wall5_box /
   obj1 (the box-box / multi-point scenarios; ~1e-12 reassoc amplifies). Requires a
   deliberate, reviewed `capture_baseline.py` re-run — against the project rule that
   perf opts must pass the existing baseline. Get sign-off first.
2. Variant (ii) needs a per-block contiguity flag (compute once in `lcp_build_partition`)
   + gather fallback for interleaved-body components; (i) alone is correct but slower.
3. PGS's `w` update is still dense O(Mrow) per λ change — once Amat shrinks it becomes a
   larger share, so a sparse-PGS pass may matter more than win(b) for huge nc.

**Decision rationale (2026-05-25):** deferred. win(a) already gives exact O(K³)→O(K)
for multi-robot (the primary use case) and 4.4× on box_wall; win(b)'s ~1.4× isn't worth
losing the bit-identical guard + the added complexity. Reconsider only if profiling a
real workload (dozens of robots at dt=1ms, or a very large single pile) shows the A
build dominating after win(a).

### Forward-compat with future general constraints (held by shipped S2)

No sibling project has closed chains today and no YAML has constraints — but
future joint limits (MuJoCo-style) and loop-closure / weld constraints (DART-style)
are anticipated. The shipped S2 is built so those are a **pure addition** with no S2
rework, at ~0 extra cost, because **robot–robot contact already forces the
"one row spans 2 blocks" case** — the same design is constraint-ready for free.

Invariants the shipped S2 holds (any extension must keep them):

- **I1 — A-sparsity = per-row BLOCK-SUPPORT-SET intersection.** Compute `A[r][s]`
  iff `support(r) ∩ support(s) ≠ ∅`. **Never** reduce a row to a single
  component-id, and never "skip cross-component pairs" — a J row can legitimately
  span 2 blocks (robot–robot contact; weld/loop-closure constraint). S1 already
  carries a 2-block support `(b1,b2)` per contact; S2 extends it to row-pairs.
- **I2 — A-build generic over row count/type.** Do not bake "6 rows per contact"
  into the sparsity/build logic. Joint limit = 1 row, weld = 6, revolute cut = 5.
  Drive the build off `Mrow` + a `row_blocks[]` table.
- **I3 — constraints never enter M.** `M` is the spanning-tree mass matrix. Joint
  limits and cut joints don't touch `parent[]`/`jtype[]`, so S1 is untouched and
  the S1 partition cache key `(nb, parent, jtype)` stays valid when constraints
  are added.
- **I4 — cross-component coupling is mediated by the CONSTRAINT ROW.**
  `A[constraint][contactX] ≠ 0`, but `A[c1][c2]` across two components stays 0.
  Support-intersection produces exactly this — don't try to merge components or be
  cleverer than the rule.
- **I5 — A stays dense-stored; keep PGS projection separate from the A-build.**
  Constraint rows just append rows/cols. Future projection types (1D clamp for a
  joint limit, no-clamp for a bilateral constraint) plug into PGS only. (Dense A
  is also the autodiff/IFT-adjoint–friendly choice; do **not** drift to a
  matrix-free "S3" without re-examining the adjoint.)

Data structure: generalize S1's `(b1,b2)` to a per-entity block list (≤2 in
practice), derived from `blk_of_body` (which already maps fixed → nearest moving
ancestor). When constraints land, they add a `constraint_type_t`-style row
provider (J rows + body set → block support) plus a PGS projection branch; S1/S2
already accommodate them.

**Landed 2026-05-31 — first concrete instance: joint Coulomb friction
(`frictionloss`).** The A-build was generalized from the per-contact 6×6-block tiling
to row-driven off `Mrow` + a per-row `row_blocks[]` table (the I2 realization;
`cblk` → `row_blocks`), bit-identical to the contacts-only build. Joint-friction rows
then slot in exactly as predicted here: one `e_{fj}` row per frictive DoF, singleton
block-support `{blk_of_body[body]}` (I1/I4 hold), never entering `M` (I3 — the S1
partition cache key `(nb,parent,jtype)` is untouched), with a 1D PGS box clamp kept
separate from the A-build (I5). So the invariants above are now exercised, not just
asserted. See `docs/design-joint-friction.md`.

**Second instance: joint range limits (`limit`, also 2026-05-31).** A one-sided
position constraint posed at the velocity level like a contact normal — Jacobian
`±e_{fj}`, `λ≥0`, Baumgarte push-out — active only when the joint reaches a bound.
Same singleton block-support, same I1–I5 reuse; the PGS projection is literally the
contact-normal `max(0,·)`. This drove a sizing fix: the row-sized buffers now use
`M2 = 6·Pm + 2·nq` (was `6·Pm`) so models with more frictive/limited DoFs than
contact-row capacity don't overflow — a latent bug the friction-only path could already
hit. Weld / loop-closure rows follow the same path. See `docs/design-joint-friction.md`.

## 4. How to profile / verify

```bash
# stage timers (off by default, zero-cost when off)
gcc ... -DLCP_PROF ...                      # or edit build.sh CFLAGS
TACT_LCP_PROF=1 uv run python tests/_prof_box_wall.py 2000
TACT_LCP_PROF=1 uv run python tests/_prof_multizen.py   # multi-robot scaling

# bit-identical guard (must stay 10/10 for any algorithm-preserving change)
uv run python tests/regression/test_traj.py
```

`/tmp/bigfloor.yml` is the large floor used by the multi-robot harnesses. uv root
is `/home/ubuntu/uv` (`cd /home/ubuntu/uv && uv run python ...`).

## 5. Related decisions

- Closed-chain support: contact-only now, door left open at ~0 cost via I1–I5.
- The legacy `penalty` solver was removed 2026-05-24; `lcp` is the only solver.
- Autodiff substrate is the regularized convex LCP (not penalty); S1/S2 align with
  an IFT-adjoint differentiable path (dense A + reusable block M factor).
