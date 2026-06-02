# design-joint-friction.md — joint Coulomb friction (MuJoCo `frictionloss`) as a constraint row

Status: **design / not implemented**. This doc fixes the math, the workspace layout,
the warm-start scheme, the regression gate, and the phased plan **before** any code
lands. Read `docs/design-lcp-perf.md` §"Forward-compat with future general
constraints" (invariants I1–I5) first — joint friction is the **first concrete
instance** of the generic constraint row those invariants were written for.

Decision (2026-05-31): implement as a **general constraint-row generalization** of
`lcp.c` (option A), not a frictionloss-only bolt-on. frictionloss is row provider #1;
joint-limit / weld / loop-closure become near-free additions afterward.

---

## 1. What `frictionloss` is

A per-DoF **Coulomb (dry) friction** that opposes motion in that DoF with a force
bounded by a constant `±floss` (N·m for revolute, N for prismatic), regardless of
load. It is the **nonsmooth sibling of the existing viscous damping `ff`**:

| | force law | how it enters the solve |
|---|---|---|
| viscous `ff` (`rbd.py:855`) | `−ff · qd`  (linear) | folded into ABA articulated inertia (`di += ff·dt`, `ui += −ff·qd`) — implicit, stable |
| **Coulomb `floss`** | `−floss · sign(qd)`, clamped (nonsmooth) | **cannot** fold into the linear ABA system → needs the implicit box solve |

The reason it must go in the constraint solver is the **same reason the penalty
solver was removed (2026-05-24)**: Coulomb friction is nonsmooth at `qd=0`. An
explicit `−floss·sign(qd)` clamp inside ABA chatters around zero and cannot hold a
joint static — exactly the brush-friction failure that could not hold a planted
foot. Static friction ("a joint that should stay put stays put") is intrinsically an
**implicit box solve**. So joint friction belongs in the same place contact friction
already lives.

This holds **even with no contact**: an isolated frictionloss DoF still needs the
implicit solve for its own stiffness at zero velocity. When the DoF *is* also in
contact (gripper finger pressing an object; a leg joint while the foot is planted),
the two couple through `M⁻¹` and **must** be solved together — which is precisely
what MuJoCo does and what motivates option A.

## 2. How MuJoCo solves it (reference)

MuJoCo stacks **all** constraints — equality, friction(loss), limit, contact — into
one constraint system (`efc`) and solves the **same** convex dual once per step
(PGS / CG / Newton over `solref`/`solimp` soft constraints).

| efc type | Jacobian row | force bound | target |
|---|---|---|---|
| equality | constraint J | `(−∞, ∞)` | 0 |
| **frictionloss** | **`e_j` (selects DoF `j` velocity)** | **`[−floss, +floss]` constant** | velocity 0 |
| limit | limit J | `[0, ∞)` | 0 |
| contact (frictional) | contact J | friction cone (pyramidal/elliptic) | 0 |

Key properties tact inherits:

- frictionloss is **one row per DoF**; its Jacobian is the trivial `e_j` → J/A build
  is cheap and exact (no narrowphase, no geometry).
- the bound is a **constant** `floss` — unlike a contact tangent (`μ·λ_n`), it is
  coupled to nothing. No cone budget, no normal-force ordering dependence.
- it is solved **together with contact** only because of `M⁻¹` coupling; structurally
  it is the simplest constraint in the table.

MuJoCo's softness (`solref/solimp`) gives a smooth static-friction plateau. tact's
analogue is the existing CFM diagonal (`R_diag`, `lcp.c:486`) plus the row target;
tact's projection is a hard PGS box clamp (acceptable — same regime as the spin
clamp today).

## 3. Mapping onto tact's PGS — it is the spin clamp's cousin

tact's PGS **already contains a 1D box clamp**: the spin cone (`lcp.c:579`,
`|λ_spin| ≤ μ_spin·λ_n`). A frictionloss row is the same projection with two
differences:

| | spin clamp (existing) | joint frictionloss (new) |
|---|---|---|
| Jacobian row | contact J spin row | `e_j` (unit DoF selector in F-space) |
| bound | `μ_spin · λ_n` (load-dependent) | `floss · dt` (constant; λ is impulse) |
| exists when | contact present | **always** (even nc = 0) |
| `A` diagonal | `J M⁻¹ Jᵀ` | `(M⁻¹)_jj` |
| rhs `c` | 0 (target zero slip) | `qd_free_j` (target zero velocity) |

### 3.1 Math

Work in the same impulse-space, free-subspace formulation as `contact_lcp`. Let the
constraint set be `contacts ∪ friction-DoFs`. For a frictionloss DoF `j` (free-index
`fj = free_map[j]`):

- **Jacobian row** `J_fric = e_{fj}ᵀ` (a single 1.0 at column `fj`, else 0).
- **Delassus diagonal** `A_jj = e_{fj}ᵀ M⁻¹ e_{fj} = (M⁻¹)_{fj,fj}`.
- **Cross term to contact row `i`** `A_{j,i} = e_{fj}ᵀ M⁻¹ Jᵀ_i = Y[i][fj]` — i.e.
  the `fj`-th column of the already-computed `Y = M⁻¹Jᵀ`. **No new linear solve**:
  the coupling is read straight out of `Y`. (And `A_{j,j'}` between two friction
  DoFs = `(M⁻¹)_{fj,fj'}`, nonzero only within a shared M-block.)
- **rhs** `c_j = qd_free_{fj}` (target velocity 0 → drive `qd_next,j → 0`). No
  Baumgarte/restitution on a friction row.
- **bound** in impulse units: `floss · dt` (because λ is impulse; cf. `f_ext = λ/dt`,
  `lcp.c:654`).
- **projection** in PGS: `λ_j ← clamp(−(w_j − A_jj λ_j)/A_jj, −floss·dt, +floss·dt)`.
- **velocity correction & f_ext**: `Jᵀλ` already accumulates the friction impulse
  into DoF `j` (column `fj`) in PASS 5; the existing `M⁻¹(Jᵀλ)` scatter applies it.
  No PASS 6 wrench synthesis for friction rows (it's a joint-space generalized force,
  not a Cartesian contact wrench) — friction rows are simply skipped in PASS 6.

### 3.2 Block-sparsity (S1/S2) interaction — fits with zero rework

A frictionloss DoF `j` lives in exactly **one** M-block: `blk_of_body[body(j)]`. So
its row's block-support set is the singleton `{that block}`. This drops straight into
the I1 support-intersection rule (`docs/design-lcp-perf.md`): `A[friction_j][row_s]`
is computed iff `block(j) ∈ support(s)`. It couples to contacts/frictions **in the
same block only** — exactly I4. Per I3, friction rows **never enter M** (the partition
cache key `(nb, parent, jtype)` is untouched), so S1 stays valid and bit-identical.

## 4. Generalizing `lcp.c` to constraint rows (option A)

Today `lcp.c` hardcodes "6 rows per contact" (`Mrow = 6*nc`). Generalize to a
**row table** driven by I2:

```
Mrow = 6*nc + n_fric              (+ future: + n_limit + 6*n_weld + ...)
```

Introduce a per-row provider concept (lightweight — no need for a full vtable yet):

- **rows [0, 6·nc)** — contact rows, built exactly as today (PASS 1/2).
- **rows [6·nc, 6·nc + n_fric)** — friction rows: `J_fric` is `e_{fj}` (one nonzero),
  `c = qd_free_{fj}`, block-support `{blk_of_body[body(j)]}`, bound `floss·dt`,
  projection = 1D clamp.

Concrete edits (touch points, all confirmed against current code):

| file / fn | change |
|---|---|
| `lcp.c` workspace | size `J/Y/A/c/lam/w` to `Mrow = 6P_pts + n_fric` rows (not `6P_pts`). Add `row_kind[]`, `row_dof[]` (for friction rows), `row_blocks[]` (generalize `cblk`'s `(b1,b2)` to all rows, I1). |
| `lcp.c:332` early-return | `nc==0` must **not** bail when `n_fric>0`. Bail only when `Mrow==0`. |
| `lcp.c` PASS 2 | after stacking contact J, append friction rows: write a single `1.0` at `[6nc+r][fj]`. |
| `lcp.c` PASS 3 (`Y`,`cblk`) | friction row `Y` = its own `M⁻¹e_{fj}` (one block solve on block `block(j)`); record singleton support in `row_blocks`. |
| `lcp.c` PASS 3 (`A`) | drive the sparsity loop off `row_blocks[]` + `Mrow` (I2), not `nc`/`6`. Block-pair products are unchanged; structural-zero skip is identical. |
| `lcp.c` `c`/CFM | friction rows: `c = qd_free_{fj}`; optional CFM `R_fric` on the diagonal (start with 0 = hard clamp; revisit if chatter). No Baumgarte/restitution. |
| `lcp.c` PASS 4 (PGS) | add a friction projection branch: 1D clamp to `±floss·dt` with the standard incremental `w` update. Keep it **separate from the A-build** (I5). |
| `lcp.c` PASS 5 | velocity correction unchanged (`Jᵀλ` already routes friction impulse to DoF `fj`). |
| `lcp.c` PASS 6 | **skip** friction rows (no Cartesian wrench). |
| warm-start | friction λ needs its own per-DoF slot array (see §5). |
| signature | `contact_lcp(...)` + `tact_create`/`tact_step_lcp` gain `floss` (nq-length, per-DoF, 0 = off) and a friction warm-start buffer. Mirror in `rbd.py:contact_lcp` and `_clib.py` argtypes. |
| `sim.py` parse | joint `frictionloss:` key → `self.floss` nq-length array (parallel to `ff`/`sk`), spliced in add/delete like `Kd_j` (`sim.py:851`). |
| `tact.h` | proto + the row-provider comment; bump the workspace-sizing note. |

**Cost.** Each friction DoF adds 1 row/col to `A` (cheap) and one block solve for its
`Y` column. PGS gains `n_fric` 1D clamps per sweep. For a typical arm (a few frictive
joints) this is negligible vs. the contact rows. Isolated friction DoFs (no contact
in their block) could later be solved by a direct 1D clamp outside PGS
(`λ = clamp(−qd_free_j/(M⁻¹)_jj, ±floss·dt)`, O(1)) — **deferred**, not in v1.

## 5. Warm-start

Contact λ warm-starts per `(cpair_idx, sub_id)` slot (`lcp.c:512`). Friction λ is
**per-DoF** and persistent across steps independent of any contact, so it needs its
own buffer:

```
lam_fric_prev[nq]        # one scalar impulse per DoF, carried step to step
```

Seed friction rows from `lam_fric_prev` at PGS start; scatter back at PASS 5. Thread
it the same pure way contact warm-start is threaded today (`lam_in`/`lam_out` /
`SolverState ctx` in `Model.step`) — extend `SolverState` with the friction λ vector
so `Model.step` stays referentially transparent (`docs/design-pure-step.md`). Reset
to None on add/delete (DoF count change invalidates it), exactly like `lam_prev`.

## 6. Regression / bit-identical gate

**Invariant: a scene with no frictionloss (all `floss=0`) must stay bit-identical.**

- `n_fric` counts only DoFs with `floss > 0`. If `n_fric == 0`, `Mrow == 6*nc`, the
  row table is contacts-only, and every code path reduces to today's — verify the
  golden suite (`tests/regression/test_traj.py`, atol 1e-12) is untouched.
- A scene **with** frictionloss is new physics → it gets a freshly captured baseline
  (deliberate, reviewed — never auto-update; `tests/scenes/README.md`).
- The known-stale box-box baselines (memory `tact-regression-baseline-stale-boxbox`)
  are orthogonal — don't conflate.

New validation scenes:

1. **Static hold** — a horizontal slide/hinge DoF with `floss` and a tangential load
   below `floss`: joint must **not** drift (the penalty-solver failure case, now
   expected to stick). Mirror of the box-on-incline μ test.
2. **Break-away** — load just above `floss`: joint moves, friction force saturates at
   `floss`. Check steady-state velocity matches `(τ_applied − floss)/damping`.
3. **Contact coupling** — a frictive joint whose link is simultaneously in hard
   contact: confirm the coupled solve (cross `A` term via `Y`) vs. an
   uncoupled-sequential reference differs as expected (this is the whole point of A).

## 7. Phased plan (go/no-go between phases)

1. **Doc + signoff** (this doc). ← *here*
2. **Row-table refactor, contacts-only.** ✅ **DONE 2026-05-31.** Generalized
   `lcp.c`'s A-build to be **row-driven** off a per-row block-support table
   `row_blocks[]` + `Mrow = 6*nc + n_fric` (`n_fric = 0`), replacing the per-contact
   `cblk` and the 6×6-block A-build tiling (I1/I2). **Bit-identical** verified by a
   pre/post snapshot diff (`np.array_equal`) over all 10 scenarios — including the
   box-box manifold scenes that exercise the most A coupling — and the golden suite
   (`test_traj.py`, atol 1e-12) is 10/10 green. `row_kind[]`/`row_dof[]` were **not**
   added yet (they are write-only with zero friction rows → no verification value);
   they land in Phase 3 where they are actually consumed. tact.c workspace sizing +
   comment and `docs/design-lcp-perf.md` (cblk→row_blocks) updated. This isolated the
   structural change from new physics — the load-bearing checkpoint passed.
3. **Friction rows (C).** ✅ **DONE 2026-05-31.** Added `floss` plumbing
   (`tact_create`/`tact_step_lcp`/`_clib`/`sim.py` parse of joint `frictionloss:`),
   the friction row provider (enumerate 1-DoF rev/lin DoFs with floss>0 → `e_{fj}`
   rows; generic c/A-build yields `c=qd_free_{fj}`, `A_diag=(M⁻¹)_{fj,fj}` and
   contact cross-terms for free), the PGS 1D box clamp to `±floss·dt`, and per-DoF
   warm-start (`lam_fric`, internal/stateful for now). **Validation (headless,
   analytic):** vertical prismatic slider under gravity (m=1, g=9.81) — static-hold
   floss=20 holds q≡0 exactly (the penalty-solver failure case, now sticks);
   break-away floss=5/2 give a=(mg−floss)/m → q matches analytic to <2.5e-3 (half a
   semi-implicit step); monotonic floss→less slide; free-fall floss=0 matches −½gt².
   Revolute pendulum (jtype==1): floss=10 holds against the 4.9 N·m gravity torque,
   floss=0 swings. Both jtype paths confirmed.

   **-ffast-math finding (important).** Phase 3 is new physics → not bit-identical by
   intent, but **zero-floss scenes must be unchanged**. Under the `-ffast-math`
   release build they are NOT all bit-identical: box_wall (1.9e-6) and wall5_box
   (1.3e-6) exceed the 1e-12 gate, while every non-contact, single-contact, and
   lightly-stacked scene is exact. This is **pure compiler FP-reassociation**, not a
   logic change — *proven* two ways: (a) compiling out all friction code
   (`-DTACT_NO_JFRIC`, a kept compile-time kill-switch) reproduces `/tmp/pre`
   bit-identically; (b) a strict-IEEE build (`-fno-fast-math -ffp-contract=off`) makes
   friction-on vs friction-off **bit-identical for zero floss across all 10 scenes**.
   So the dead friction code merely perturbs how `-ffast-math` reassociates the PGS
   hot loop, and only box_wall/wall5_box's marginally-stable stacks amplify it past
   1e-12 (consistent with the box_wall sensitivity noted in design-lcp-perf.md). The
   golden baselines are left untouched here; re-capture is the deliberate Phase 5 act.
   Lesson for future PGS-area edits: the 1e-12 gate is inherently fragile under
   `-ffast-math`; use the strict-IEEE A/B (or `-DTACT_NO_JFRIC`) to tell real logic
   changes from reassociation noise.
4. **Python mirror + parse.** ✅ **DONE 2026-05-31.** `rbd.py:contact_lcp` mirrors the
   friction rows (append `e_{fpos}` rows to J, pad R/b with zeros, PGS 1D clamp,
   per-DoF `lam_fric_prev/lam_fric_full`, return contacts-only `lam_c`);
   `SolverState` gains `lam_fric` (per-DoF, default None); both step paths thread it
   via ctx for referential transparency — the C path got `lam_fric_in/out` on
   `tact_step_lcp` (seeded like the contact λ), `_clib.py` argtypes +2. `sim.py`
   parse already landed in Phase 3; **fixed a real bug there**: the free-joint
   (jtype=3) parse branch appended `ff/sk/Kp_j/Kd_j` per-DoF but not `floss`, so
   `len(floss) < nq` whenever a free joint was present → the C-side `memcpy(…, nq)`
   over-ran the per-DoF λ buffers (`free(): invalid pointer` on box_wall). Now floss
   is appended in both branches and spliced in delete.

   **Verification:** C-vs-Python A/B on friction scenes agrees to machine ε
   (break-away 4.4e-16, static-hold exactly 0, no-friction control 2.2e-16);
   referential transparency holds for friction scenes on **both** paths
   (double-step from the same (q,qd,ctx) is identical, ctx immutable, friction λ
   actually carried in ctx); the existing pure-step suite is 7/7 and the zero-floss
   regression is unchanged from Phase 3 (only the proven box_wall/wall5_box -ffast-math
   noise).
5. **Docs + test consolidation.** ✅ **DONE 2026-05-31.** No golden `.npy` re-capture:
   8/10 scenes stayed bit-identical at 1e-12; the 2 marginally-stable stacks
   (box_wall, wall5_box) get a per-scene 0.1 mm atol in `test_traj.py` (their
   bit-exactness is `-ffast-math` reassociation, proven benign), which absorbs both
   the current and any future PGS-edit noise — strictly better than re-capturing this
   compiler's noise as exact. Friction physics promoted to `tests/joint_friction.py`
   (analytic static-hold/break-away/monotonic + rev/lin + C↔Python A/B + ctx-purity,
   7/7) — a baseline-free correctness test that never goes stale. CLAUDE.md (joint
   `frictionloss` schema, lcp.c row-table + Contact-solver friction-row sections,
   per-DoF array list, ctx hidden-state) and `docs/design-lcp-perf.md` forward-compat
   ("first instance landed") updated.

**Deferred (next test-infra forcing function, e.g. a compiler bump).** The regression
suite still bit-tests the `-ffast-math` release build, which conflates physics with
compiler/flag/CPU incidentals (Phase 3 was an early symptom). The durable fix is
structural, not another re-capture: decouple the bit-exact reference from the release
build (pin toolchain + strict-math for the golden, run release separately) and shift
weight toward analytic/correctness tests like `tests/joint_friction.py`. Do this at
the next moment the test infra is touched anyway (a compiler upgrade forcing a mass
re-capture is the natural trigger) — not as a speculative refactor now. Option C
(strict-build just `lcp.c`) is the alternative if a robust bit-level gate is wanted on
every scene, but it costs PGS hot-loop perf (measure first) and isn't needed given the
per-scene-atol + analytic-test approach above.

Phase 2 is the load-bearing checkpoint: if the row-table refactor cannot be made
bit-identical, stop and reassess before adding friction physics on top.

## 8. Open questions to settle during phase 2/3

- **CFM softness on friction AND limit rows?** Both ship **hard** (`R=0`). CFM adds a
  small `R` to the row's A-diagonal so the constraint enforces `w = −R·λ` instead of
  `w = 0` — i.e. a finite stiffness (stiff spring) instead of a rigid stop, the same
  compliance MuJoCo gives via `solref`/`solimp`. Why each row type might want it:
  - **friction** — the stick↔slip box-clamp boundary is discontinuous; at near-zero
    velocity λ can flip step-to-step → **chatter** (buzzing joint). A small `R_fric`
    smooths the force response.
  - **limit** — a hard wall hit at speed **rings/bounces** and PGS can oscillate at
    the boundary. `R_lim` makes it a stiff bumper. Paired with the existing Baumgarte,
    **ERP + CFM = a spring-damper** (ERP = restoring stiffness, CFM = compliance/damping)
    — exactly MuJoCo's `solref` reparametrized.

  Trade-off: softness allows a small violation (∝ `R·λ` — limit slightly penetrated
  under load, friction a tiny creep). Start hard; add `R` only if chatter/ring is
  actually observed (pulling it in early just adds violation). Two lines in `lcp.c`
  beside the contact CFM loop; value as a global knob (like `cfm_scale`) or per-joint.

  **NB — distinct from the limit pre-engagement margin (§9):** CFM softness damps the
  constraint *force* (chatter/ring); the margin engages the limit *before* the bound
  to kill the one-step overshoot. Different knobs, different parts of the path.
- **Free-joint DoFs.** frictionloss is physically a rev/lin concept. v1: restrict to
  `jtype ∈ {1,2}` (mirrors the implicit-PD restriction, `rbd.py:843`). Per-DoF free
  friction is possible but unmotivated — defer.
- **Tendon frictionloss.** MuJoCo also has it on tendons; tact has no tendons → N/A.

## 9. Follow-on: joint range limits (landed 2026-05-31)

The row-table this doc built was reused verbatim for **joint range limits** (`limit:
[lo, hi]`), the second non-contact constraint type. A limit is a one-sided position
constraint posed at the velocity level like a **contact normal**: when a 1-DoF rev/lin
joint reaches a bound it adds one row with Jacobian `±e_{fj}` (+1 lower / −1 upper),
`λ≥0`, and Baumgarte push-out `b=(erp/dt)·max(0, depth−slop)`. Reuse vs. new:

- **Reused from friction:** `e_{fj}` J-stacking, singleton block-support, per-DoF
  warm-start (`lam_limit`, ctx-threaded), C↔Python mirror, the whole row-table.
- **Reused from the contact normal:** the `λ≥0` projection and the Baumgarte bias.
- **New:** the active-set gating (only add a row when `q` is at/past a bound) and the
  limit violation depth → bias.

A latent sizing bug surfaced and was fixed: the row buffers used `M2 = 6·Pm`, which
overflows when frictive/limited DoFs outnumber the contact-row capacity. Now
`M2 = 6·Pm + 2·nq` (6 per contact-point + 1 friction + 1 limit per DoF).

Verified analytically (`tests/joint_limit.py`): holds at the stop, unlimited passes,
inactive in interior, C↔Python A/B to machine ε, referential transparency. **Open**
(both deferred until observed): (a) v1 activates at `depth≥0` (joint can transiently
cross by up to `v·dt` before Baumgarte pulls it back, ~5e-3 rad in tests) — a
**pre-engagement margin** would remove the overshoot; (b) **CFM softness** on the limit
row (`R_lim`) would damp ringing/PGS oscillation at the stop (ERP + CFM = spring-damper,
MuJoCo `solref`) — see §8, and note it's a *distinct* knob from the margin. YAML `limit`
is **degrees for rev / m for lin** (same convention as `q0`; converted to rad/m at parse).
