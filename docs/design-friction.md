# design-friction.md — contact friction model

> **Status (stale — penalty era).** This documents the **penalty** contact solver's friction
> model (`contact_ccd`), removed **2026-05-24** when `lcp` became the sole contact solver. The
> implementation described below — `contact_ccd` in `tact.py`/`ccd.c`, the
> `s_t_world`/`s_phi_world`/`s_roll_world` brush state, the `use_c` path, hardcoded `k_t`/`mu`/… —
> **no longer exists**; the collision side now lives in `narrow.c`/`mpr.c`/`ray.c`/`shape.c`, and
> LCP friction params are **per-material** (YAML `materials:`; see CLAUDE.md + `design-lcp-perf.md`).
> Kept as a design record: the **4-cone model** (normal/tangent/spin/roll) carried into LCP, and
> the rationale here — why single-point contact needs spin friction, and that a **contact
> manifold is the cleaner fix** (§"Why the gate is the right scope") — is exactly what the later
> box-box / height-field manifold work realizes.

Mirrors what `contact_ccd` does in both `tact.py` and `ccd.c`. Notes the
non-obvious decisions and the empirically-discovered limitation that motivated
the current shape.

## Stages

The contact wrench at one MPR-derived contact point is the sum of four
independent contributions:

| Stage | Name             | DOF                       | State (per pair)       | Hardcoded params                                |
| ----- | ---------------- | ------------------------- | ---------------------- | ----------------------------------------------- |
| 1.1   | normal           | n̂ (no-pull spring + damp) | —                      | k_n, d_n (per-shape, from YAML `contact:`)      |
| 1.2   | tangential slide | tangent plane (2 DOF)     | `s_t_world` (3D world) | `k_t = 20000`, `mu = 0.8`, `d_t` (YAML)         |
| 1.3   | torsional spin   | about n̂ (1 DOF)           | `s_phi_world` (scalar) | `k_spin = 100`, `d_spin = 1.0`, `mu_spin = 0.02`|
| 1.4   | rolling          | tangent plane (2 DOF)     | `s_roll_world` (3D)    | `k_roll = 100`, `d_roll = 1.0`, `mu_roll = 0.005`|

All four use the same template: brush spring + viscous damper integrated each
step, with a Coulomb cone `|τ| ≤ μ·fz` enforcing the slip limit and a snap-back
of the brush state when clamped.

`fz` (normal force, no-pull) is computed once in Stage 1.1 and reused by all
three friction stages — i.e. when contact opens up (`fz=0`), all friction
saturates to zero, brush states are reset to zero on next geometric-miss.

## Why spin friction is gated to world contacts

**The gate**: spin (Stage 1.3) is computed only when at least one of the two
bodies in the contact pair is the world (`cbody < 0`). For body-body pairs
spin is skipped, `m_world` for that pair gets zero spin contribution, and
`s_phi_world[n]` is reset to 0 to avoid stale stick state.

Rolling (Stage 1.4) and the other stages are unaffected — they apply to all
contact pairs.

### Why it exists

Single-point contact via MPR cannot generate a yaw-restoring torque around the
contact normal (a single force has zero moment along its own line of action).
For flat-on-flat resting contacts — typical case: a box on the ground — this
means the body has zero resistance to spinning about the contact normal, and
tiny numerical noise plus contact-point jitter from MPR's witness selection
slowly pumps a visible yaw drift over time.

Spin friction adds a per-pair brush-spring + Coulomb-cone torsional dissipator
about n̂. The "stick" sub-model (spring side, below Coulomb saturation) is
what kills slow drift: the spring captures the relative yaw at the moment of
contact and resists any subsequent yaw motion until the Coulomb cone is broken.

This was confirmed to work as intended for box-on-ground.

### Why it broke body-body grasps

The same "stick" sub-model that suppresses yaw drift against the world also
**couples the two bodies' yaws** through the captured spring. When both bodies
move (i.e. neither is the world), the spring acts as a constraint that tries to
hold their relative yaw at whatever it was when contact began.

In a parallel-jaw pick (observed in `gos/`), this happens:

1. The gripper closes on an object with the pinch axis roughly aligned with
   the object's yaw axis (so n̂ is approximately ±ẑ_object, and yaw rotation
   becomes ω_n at the contact, not ω_t).
2. As fingers contact, `s_phi` captures the initial relative yaw.
3. Normal-force × moment-arm would naturally rotate the object into alignment
   with the finger face. The Stage-1.3 stick spring resists this.
4. The arm has small yaw wobbles (PID corrections, contact reactions, link
   dynamics). Through the stick spring, these get communicated to the object
   as a couple along the contact normal.
5. The two finger contacts contribute non-symmetric spin couples (different
   normals, different fz), and their imbalance — plus the arm's transient
   yaw motion — can rotate the object in a direction unrelated to alignment.

Empirically, setting `mu_spin = 0` made the artifact disappear in `gos/`, and
the box-on-plane yaw-drift suppression still worked for `cbody = world` pairs
under the gate. Disabling rolling friction (`mu_roll = 0`) had no effect on
the gos artifact — confirming spin is the cause.

### Why the gate is the right scope

- World contacts are essentially asymmetric: only one body has dynamics, so
  the "stick spring couples two yaws" mechanism reduces to "stick spring
  couples one body's yaw to a fixed reference" — which is exactly the
  drift-suppression we want, with no cross-body artifacts.
- Body-body contacts are where the coupling causes problems, and they are
  also the case where a properly resolved manifold contact would have
  multiple contact points whose normal forces already produce restoring
  torques — single-point spin friction is the wrong tool there anyway.

A cleaner fix (contact manifold with multiple points per pair) would remove
the need for spin friction entirely against the world too. Until that exists,
the gate is the minimum change that keeps spin's benefits where they're real
and removes its artifacts where they aren't.

## Why rolling is *not* gated

Rolling friction has the same brush-spring structure as spin, so the same
"coupling" concern applies in principle. In practice the magnitudes work out
differently:

- For typical body-body grasps with horizontal pinch (n̂ horizontal), the
  object's yaw axis is in the tangent plane → handled by rolling, not spin.
  In this case rolling friction must oppose alignment rotation, but the
  alignment torque from normal force × moment arm dominates
  (mu_roll = 0.005 vs ~0.05 effective for normal × arm).
- For spherical/cylindrical contacts on the ground, rolling friction does the
  job spin friction does for boxes — preventing perpetual roll-out from
  numerical noise.

If a body-body grasp artifact analogous to the spin one is later observed
through rolling, the same gating treatment will apply.

## Tuning notes

Defaults are conservative for the typical mass/dt ranges in this repo
(0.1–10 kg bodies, dt = 1ms). Stiffness values were lowered from initial
guesses for explicit-integrator stability margin against light masses:

- `k_t`: 50000 → 20000  (matches YAML k_n distribution average; slip threshold
  still ~0.4mm for fz = 10N — visually rigid)
- `k_spin`: 200 → 100   (improves dt margin for I ≈ 1e-4 kg·m² objects)
- `k_roll`: 200 → 100   (same reason)

Coulomb coefficients (`mu_spin = 0.02`, `mu_roll = 0.005`) are larger than
MuJoCo defaults but consistent with this codebase's higher `k_t` and explicit
integration; they can be tuned per-scene later via YAML if anisotropic needs
emerge. For now they are hardcoded in `contact_ccd` (both `tact.py:898+` and
`ccd.c:952+`).

## Files

- `tact.py` `contact_ccd` (Python reference path, `use_c=False`)
- `ccd.c`   `contact_ccd` (production path, `use_c=True`)
- `tact.c`  arena for `s_t_world`, `s_phi_world`, `s_roll_world`
- `tact.h`  function prototype

The two `contact_ccd` implementations must stay in sync — any change to one
needs the mirrored change in the other. The gate in §"Why spin friction is
gated to world contacts" is one such mirrored change.
