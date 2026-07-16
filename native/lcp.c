/* lcp.c — Stewart-Trinkle / Anitescu time-stepping LCP contact solver.
 *
 * Companion layer on top of narrow.c (narrow-phase) and rbd.c (CRB + LDLᵀ).
 * Mirrors rbd.py:contact_lcp 1:1. lcp is the only contact solver (the legacy
 * penalty solver was removed 2026-05-24). Public surface in tact.h. */
#include "core.h"

/* ----- optional stage profiling (compile with -DLCP_PROF) -------------------- */
#ifdef LCP_PROF
#include <time.h>
static double g_prof[7];          /* narrow, jac, factorY, Amat, pgs, post, TOTAL */
static long   g_prof_calls;
static long   g_prof_nc_sum, g_prof_nc_max, g_prof_F_last;
static inline double prof_now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
__attribute__((destructor)) static void lcp_prof_dump(void){
    if (!getenv("TACT_LCP_PROF") || g_prof_calls == 0) return;
    const char *nm[7] = {"narrow","jac(P2)","factorY","Amat","pgs","post","TOTAL"};
    fprintf(stderr, "\n[LCP PROF] %ld solver calls | nc mean=%.1f max=%ld | F=%ld\n",
            g_prof_calls, (double)g_prof_nc_sum/g_prof_calls, g_prof_nc_max, g_prof_F_last);
    for (int i = 0; i < 7; i++)
        fprintf(stderr, "  %-8s %9.4f s  %8.3f us/call  %5.1f%%\n",
                nm[i], g_prof[i], g_prof[i]/g_prof_calls*1e6,
                g_prof[6] > 0 ? 100.0*g_prof[i]/g_prof[6] : 0.0);
}
#define PROF_TS(v) double v = prof_now()
#define PROF_ADD(i, dt) g_prof[i] += (dt)
#else
#define PROF_TS(v)
#define PROF_ADD(i, dt)
#endif

/* ----- workspace layout ------------------------------------------------------
 * P = n_pair (upper bound on nc), F = n_free (jtype>0 count, upper bound = nb).
 *
 *   p_world  [3*P]              contact point (world)
 *   R_tan    [9*P]              orthonormal frame, R_tan[:,2] = n̂
 *   depth    [P]
 *   mat      [12*P]             k_n,d_n,k_t,d_t,mu,k_sp,d_sp,mu_sp,k_rl,d_rl,mu_rl,e_rest
 *   ci, cj, cp_idx, sub_id      parallel int arrays (4*P)
 *   free_map [nb]               int, jtype>0 → 0..F-1, else -1
 * Row-sized buffers use M2 = 6P + 3·nq = max constraint rows (6 per contact-point +
 * 1 friction + 1 limit + 1 actuator per DoF), so non-contact rows never overflow the
 * contact-row capacity 6P:
 *   row_blocks [2·M2]           S2: ≤2 M-block ids per CONSTRAINT ROW (block-support
 *                               set). Per-row (not per-contact) so non-contact rows
 *                               — joint friction, limits — drive the same A-build (I2).
 *   J        [M2 · F]           Jacobian rows on free subspace (row-major)
 *   Mpack    [F · F]            packed per-block sub-matrices of M (Σ s_b² ≤ F²),
 *                               each factored in-place (LDLᵀ) — see S1 above
 *   Y        [M2 · F]           M⁻¹·Jᵀ stored column-by-column (treated as M2 × F row-major)
 *   A        [M2 · M2]          Delassus, dense
 *   c_vec/lam/w  [M2] each      bias rhs / current λ / (A·λ + c) maintained in PGS
 *   tmp      [F]                per-column solve scratch
 *   J6       [6·nb]             jacob_whitney output for one body
 *   act_b/act_vstar [nq each]   actuator-row effective damping Kp·dt+Kd / target v*
 *
 * Total doubles: ≈ M2² + 2·M2·F + F² + 3·M2 + 25P + 9·nq  (sized in tact.c).
 * Total ints:    4P (ci,cj,cp_idx,sub_id) + nq (free_map) + 2·M2 (row_blocks)
 *                + 2·nq (fric_dof,fric_body) + 3·nq (lim_dof,lim_sign,lim_body)
 *                + 2·nq (act_dof,act_body).
 * ----------------------------------------------------------------------------- */

static inline double dotN(const double *a, const double *b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ----- S1: block-diagonal M factor/solve ------------------------------------
 * The free-subspace mass matrix M is block-diagonal by KINEMATIC CONNECTED
 * COMPONENT: M[a][b] (CRB) is structurally nonzero only when bodies a,b are in
 * ancestor-descendant relation, so two moving DoF couple iff one body is a
 * (possibly fixed-link-spanning) ancestor of the other. The static world root
 * (jtype=0, no DoF) does not couple its independent subtrees → each robot /
 * free object / pile component is its own dense block; crb_featherstone leaves
 * cross-block entries at the memset-0 they were initialized to.
 *
 * Factoring blocks independently is BIT-IDENTICAL to a dense LDLᵀ of the whole
 * matrix: the cross-block entries being exactly 0.0 means no fill-in and every
 * dropped term is `x ± 0.0`. Cost drops from O(F³) to Σ_b s_b³ (≈ K² less for
 * K equal blocks); each contact's M⁻¹Jᵀ solve touches only the ≤2 blocks of its
 * two bodies (a contact Jacobian row is structurally 0 outside those blocks,
 * since jacob_whitney's nonzero columns are the body→root chain ⊆ its block).
 *
 * The partition depends only on topology (nb, parent[], jtype[]), which is
 * fixed per handle (add/delete rebuilds the handle) → built once on the
 * first solve into core->lcp_part; no per-call validation needed.
 *
 * Rationale, measured speedups, the fixed-foot gotcha, and the planned S2
 * (sparse-J Delassus build) with its forward-compat invariants for future
 * general constraints (joint limits / loop closure): see docs/design-lcp-perf.md. */
#define LCP_MAXF (6 * TACT_MAX_NB)              /* max free DoF (every body free) */

/* Partition lives per-handle in core->lcp_part (lcp_partition_t, core.h) —
 * built once on the first solve; (nb, parent, jtype) are fixed per handle. */

static int uf_find(int *uf, int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; }

static void lcp_build_partition(lcp_partition_t *pt, int nb, const int *parent, const int *jtype)
{
    /* union-find: merge each moving body with its nearest moving ancestor */
    int uf[TACT_MAX_NB];
    for (int i = 0; i < nb; i++) uf[i] = i;
    for (int i = 0; i < nb; i++) {
        if (jtype[i] == 0) continue;       /* fixed: no DoF, no block */
        int a = parent[i];
        while (a >= 0 && jtype[a] == 0) a = parent[a];   /* skip fixed links */
        if (a >= 0) {
            int ri = uf_find(uf, i), ra = uf_find(uf, a);
            if (ri != ra) uf[ri] = ra;
        }
    }
    /* per-body DoF base (== F-base, free_map is identity since fixed→0 slots) */
    int q_base[TACT_MAX_NB], nq = 0;
    for (int i = 0; i < nb; i++) {
        q_base[i] = nq;
        nq += (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
    }
    /* dense block ids assigned in body order; accumulate sizes */
    int root2blk[TACT_MAX_NB];
    for (int i = 0; i < nb; i++) root2blk[i] = -1;
    int nblk = 0;
    for (int b = 0; b < nb; b++) pt->blk_size[b] = 0;
    for (int i = 0; i < nb; i++) {
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        if (nvi == 0) { pt->blk_of_body[i] = -1; continue; }
        int r = uf_find(uf, i);
        if (root2blk[r] < 0) root2blk[r] = nblk++;
        int b = root2blk[r];
        pt->blk_of_body[i] = b;
        pt->blk_size[b] += nvi;
    }
    /* prefix sums into blk_dof and into the packed block-matrix buffer */
    pt->blk_off[0] = 0;
    int mat_off = 0;
    for (int b = 0; b < nblk; b++) {
        pt->blk_off[b + 1] = pt->blk_off[b] + pt->blk_size[b];
        pt->blk_mat_off[b] = mat_off;
        mat_off += pt->blk_size[b] * pt->blk_size[b];
    }
    /* fill DoF index lists (ascending: bodies visited in order, DoF consecutive) */
    int cur[TACT_MAX_NB];
    for (int b = 0; b < nblk; b++) cur[b] = pt->blk_off[b];
    for (int i = 0; i < nb; i++) {
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        if (nvi == 0) continue;
        int b = pt->blk_of_body[i];
        for (int k = 0; k < nvi; k++) pt->blk_dof[cur[b]++] = q_base[i] + k;
    }
    /* A contact's Jacobian (jacob_whitney) is nonzero on the body→root chain, so
       a contact on a FIXED body (e.g. a rigidly-mounted foot) lands in the block
       of its nearest MOVING ancestor. Map every fixed body to that block so the
       involved-block Y solve covers it; truly static bodies (no moving ancestor)
       stay -1 (their Jacobian row is all zero → correctly skipped). */
    for (int i = 0; i < nb; i++) {
        if (pt->blk_of_body[i] >= 0) continue;       /* moving — own block set */
        int a = parent[i];
        while (a >= 0 && jtype[a] == 0) a = parent[a];
        pt->blk_of_body[i] = (a >= 0) ? pt->blk_of_body[a] : -1;
    }
    pt->nblk = nblk;
    pt->valid = 1;
}

/* Solve M⁻¹·x in place for block b only: gather x at the block's DoF, run the
 * cached LDLᵀ factor, scatter back. Off-block entries are untouched (M⁻¹ is
 * block-diagonal). `pack` is the packed factored block-matrix buffer. */
static void lcp_block_solve(const lcp_partition_t *pt, int b, double *x, const double *pack)
{
    int s = pt->blk_size[b];
    const int *d = pt->blk_dof + pt->blk_off[b];
    const double *Mb = pack + pt->blk_mat_off[b];
    double sb[LCP_MAXF];
    for (int r = 0; r < s; r++) sb[r] = x[d[r]];
    ldlt_solve(Mb, s, sb);
    for (int r = 0; r < s; r++) x[d[r]] = sb[r];
}

void contact_lcp(tact_t *h, double *q, double *lam_in)
{
    tact_core_t *core = h->core;
    /* Hoist handle fields into locals once per call (ns-scale) — the solve
       body below is unchanged from the flat-argument version, so inner-loop
       codegen is identical. Reads/writes contract: core.h. */
    int     nb       = h->nb;
    int     n_pair   = h->n_pair;
    double  dt       = h->dt;
    double *T        = core->T;
    int    *parent   = core->parent, *jtype = core->jtype;
    int    *cpair    = core->cpair, *ctype = core->ctype, *cbody = core->cbody;
    double *ctran    = core->ctran, *cshape = core->cshape, *cparam = core->cparam;
    double *qd_free  = core->qd_free_buf;
    double *M_full   = core->M_buf;
    double  erp      = core->erp, slop = core->slop, cfm_scale = core->cfm_scale;
    double  v_rest_thresh = core->v_rest_thresh, tol = core->tol;
    int     iters    = core->iters;
    double *floss    = core->floss, *jnt_lo = core->jnt_lo, *jnt_hi = core->jnt_hi;
    double *taulim   = core->taulim;
    const double *act_kp   = core->act_kp,   *act_kd    = core->act_kd;
    const double *act_qref = core->act_qref, *act_qdref = core->act_qdref;
    double *workspace = core->lcp_ws;

    /* ctx layout [contact | fric | limit | act] — owned here, mirrored by Python
       SolverState. lam_in (NULL = cold) is read-only; the fric/limit/act blocks
       are seeded into ctx_next and updated in place by the solve. */
    int C_ofs = 6 * TACT_MAX_PTS_PER_PAIR * (n_pair > 0 ? n_pair : 1);
    double *lam_contact_prev = lam_in;
    double *lam_contact_out  = h->ctx_next;
    double *lam_fric         = h->ctx_next + C_ofs;
    double *lam_limit        = lam_fric + h->nq;
    double *lam_act          = lam_limit + h->nq;
    if (lam_in) memcpy(lam_fric, lam_in + C_ofs, 3 * h->nq * sizeof(double));
    else        memset(lam_fric, 0,              3 * h->nq * sizeof(double));

    double *dqd_out   = core->qdd;
    double *f_ext_out = core->f_ext;
    int    *nc_out    = &core->lcp_nc;
    int    *iters_out = &core->lcp_iters;
    double *residual_out = &core->lcp_residual;
    int    *contact_count_out = &h->contact_count;
    int    *contact_i_out     = core->contact_i;
    double *contact_d_out     = core->contact_d;

    int P  = n_pair > 0 ? n_pair : 1;
    int Pm = TACT_MAX_PTS_PER_PAIR * P;                 /* upper bound on contact-point count nc */

    /* Compute nq = sum(nv[i]) with nv=6 for jtype=3 (free), 0 for fixed, else 1.
     * Per-DoF free_map: position in compressed free vector. With fixed=0 slots,
     * every nq slot belongs to a movable DoF, so free_map is just identity. */
    int q_base_local[TACT_MAX_NB];
    int nq = 0;
    for (int i = 0; i < nb; i++) {
        q_base_local[i] = nq;
        nq += (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
    }

    /* Max constraint rows = 6 per contact-point + 1 joint-friction row per DoF +
     * 1 joint-limit row per DoF (a DoF can have both). All M2-sized buffers (J, Y,
     * A, c_vec, lam, w, row_blocks) use this so non-contact rows never overflow even
     * when frictive/limited/actuated DoFs outnumber the contact-row capacity 6·Pm. */
    int M2 = 6 * Pm + 3 * nq;

    int F = 0;
    /* slice workspace — doubles first, ints at the tail. Sizes use Pm (=MAX_PTS·P)
     * for per-contact-point arrays so a cpair may contribute up to TACT_MAX_PTS_PER_PAIR
     * contacts. Per-cpair arrays (ci, cj) stay sized P (one entry per cpair, holds
     * the shape indices); per-point arrays (cp_idx, sub_id) sized Pm. */
    double *p_world = workspace;                    /* 3*Pm */
    double *R_tan   = p_world + 3*Pm;               /* 9*Pm */
    double *depth   = R_tan   + 9*Pm;               /* Pm   */
    double *mat     = depth   + Pm;                 /* 12*Pm */
    double *J       = mat     + 12*Pm;              /* M2 * nq upper bound */
    double *Mpack   = J       + (size_t)M2 * nq;    /* nq*nq upper bound (packed blocks) */
    double *Y       = Mpack   + (size_t)nq * nq;    /* M2 * nq upper bound */
    double *A       = Y       + (size_t)M2 * nq;    /* M2 * M2 */
    double *c_vec   = A       + (size_t)M2 * M2;    /* M2 */
    double *lam     = c_vec   + M2;                 /* M2 */
    double *w       = lam     + M2;                 /* M2 */
    double *tmp     = w       + M2;                 /* nq */
    double *J6      = tmp     + nq;                 /* 6*nq */
    double *act_b   = J6      + 6*nq;               /* n_act≤nq — effective PD damping Kp·dt+Kd */
    double *act_vstar = act_b + nq;                 /* n_act≤nq — PD target velocity v* */
    int    *ci_arr  = (int*)(act_vstar + nq);       /* Pm — shape idx i of contact k's owner cpair */
    int    *cj_arr  = ci_arr  + Pm;                 /* Pm — shape idx j */
    int    *cp_idx  = cj_arr  + Pm;                 /* Pm — cpair_idx of contact k */
    int    *sub_id  = cp_idx  + Pm;                 /* Pm — sub-index 0..TACT_MAX_PTS_PER_PAIR-1 within cpair */
    int    *free_map= sub_id  + Pm;                 /* nq (per-DoF) */
    int    *row_blocks = free_map + nq;             /* 2*M2 — per-CONSTRAINT-ROW block-
                                                       support set: ≤2 distinct M-block
                                                       ids (-1 pad). Generalizes the old
                                                       per-contact `cblk` to per-row so
                                                       the A-build is driven off Mrow +
                                                       this table (I2, design-lcp-perf.md);
                                                       non-contact rows (joint friction /
                                                       limits) append with no A-build change. */
    int    *fric_dof  = row_blocks + 2*M2;          /* n_fric≤nq — free-DoF index of each
                                                       joint-friction row (== nq-index, free_map
                                                       being identity) */
    int    *fric_body = fric_dof  + nq;             /* n_fric≤nq — owning body (for block lookup) */
    int    *lim_dof   = fric_body + nq;             /* n_limit≤nq — DoF index of each joint-limit row */
    int    *lim_sign  = lim_dof   + nq;             /* n_limit≤nq — +1 (lower bound) / −1 (upper bound) */
    int    *lim_body  = lim_sign  + nq;             /* n_limit≤nq — owning body (for block lookup) */
    int    *act_dof   = lim_body  + nq;             /* n_act≤nq — DoF index of each actuator row */
    int    *act_body  = act_dof   + nq;             /* n_act≤nq — owning body (for block lookup) */

    /* Per-DoF free_map: each body contributes nv[i] consecutive entries.
     * Fixed=0 slots → no entries. All other slots are free. */
    for (int i = 0; i < nb; i++) {
        int qb = q_base_local[i];
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        for (int k = 0; k < nvi; k++) {
            free_map[qb + k] = F++;
        }
    }

    /* ---- defaults / scalar init -------------------------------------------- */
    memset(f_ext_out, 0, 6*nb*sizeof(double));
    memset(dqd_out,   0,    nq*sizeof(double));
    if (contact_count_out) *contact_count_out = 0;
    /* lam_contact_out carries previous values forward on inactive (cpair, sub_id) slots, so
       initialize from lam_contact_prev (zero if NULL). lam_contact_prev and lam_contact_out may alias —
       when they do, the seed copy is already done (and a self-memcpy would be UB).
       Length: 6 * TACT_MAX_PTS_PER_PAIR * n_pair (one 6-vec per (cpair_idx, sub_id) slot). */
    if (!lam_contact_prev)                          memset(lam_contact_out, 0, 6*TACT_MAX_PTS_PER_PAIR*n_pair*sizeof(double));
    else if (lam_contact_prev != lam_contact_out)      memcpy(lam_contact_out, lam_contact_prev, 6*TACT_MAX_PTS_PER_PAIR*n_pair*sizeof(double));

    /* Newton restitution: e is per-contact (min-blended material, mat[12*k+11]);
     * v_rest_thresh is a global numerical gate, now passed in (was hardcoded 3e-2). */

    /* ---- PASS 1: collision detection → active contact set ------------------ *
     * Each cpair may produce up to TACT_MAX_PTS_PER_PAIR contact points (box-box
     * manifold yields up to 4; other type combinations yield 1). Per-point
     * cdata gets sub_id ∈ [0, n_points-1] from the narrowphase output, with
     * the ordering convention defined in narrow.c's box-box manifold (polar angle around
     * centroid in tangent plane). The tact_collision_check out-buffer is laid out
     * 7 doubles per point. */
    int nc = 0;
    double out_buf[7 * TACT_MAX_PTS_PER_PAIR];
    PROF_TS(t_p0);
    for (int n = 0; n < n_pair; n++) {
        int si = cpair[2*n + 0];
        int sj = cpair[2*n + 1];

        double T1[16], T2[16];
        if (cbody[si] < 0) memcpy(T1, ctran + 16*si, 16*sizeof(double));
        else               matmul(T1, T + 16*cbody[si], ctran + 16*si, 4, 4, 4);
        if (cbody[sj] < 0) memcpy(T2, ctran + 16*sj, 16*sizeof(double));
        else               matmul(T2, T + 16*cbody[sj], ctran + 16*sj, 4, 4, 4);

        double param1[9], param2[9];
        homogeneous_to_xyzeuler(T1, param1, "xyz");
        homogeneous_to_xyzeuler(T2, param2, "xyz");
        for (int k = 0; k < 3; k++) {
            param1[6+k] = cshape[3*si + k];
            param2[6+k] = cshape[3*sj + k];
        }

        int npts = tact_collision_check(ctype[si], param1, ctype[sj], param2,
                                   out_buf, TACT_MAX_PTS_PER_PAIR);
        if (npts <= 0) continue;             /* < 0: separating; 0: touching */

        for (int s = 0; s < npts; s++) {
            double *out = out_buf + 7*s;
            double dvec[3] = {out[3], out[4], out[5]};
            double dnorm = sqrt(dvec[0]*dvec[0] + dvec[1]*dvec[1] + dvec[2]*dvec[2]);
            if (dnorm < 1e-4) continue;

            int k = nc;
            p_world[3*k+0] = out[0]; p_world[3*k+1] = out[1]; p_world[3*k+2] = out[2];
            choose_rotation(dvec, R_tan + 9*k);
            depth[k]       = out[6];
            mat[12*k+ 0] = 0.5*(cparam[13*si+ 1] + cparam[13*sj+ 1]);  /* k_n     */
            mat[12*k+ 1] = 0.5*(cparam[13*si+ 2] + cparam[13*sj+ 2]);  /* d_n     */
            mat[12*k+ 2] = 0.5*(cparam[13*si+ 3] + cparam[13*sj+ 3]);  /* k_t     */
            mat[12*k+ 3] = 0.5*(cparam[13*si+ 4] + cparam[13*sj+ 4]);  /* d_t     */
            mat[12*k+ 4] = 0.5*(cparam[13*si+ 5] + cparam[13*sj+ 5]);  /* mu      */
            mat[12*k+ 5] = 0.5*(cparam[13*si+ 6] + cparam[13*sj+ 6]);  /* k_spin  */
            mat[12*k+ 6] = 0.5*(cparam[13*si+ 7] + cparam[13*sj+ 7]);  /* d_spin  */
            mat[12*k+ 7] = 0.5*(cparam[13*si+ 8] + cparam[13*sj+ 8]);  /* mu_spin */
            mat[12*k+ 8] = 0.5*(cparam[13*si+ 9] + cparam[13*sj+ 9]);  /* k_roll  */
            mat[12*k+ 9] = 0.5*(cparam[13*si+10] + cparam[13*sj+10]);  /* d_roll  */
            mat[12*k+10] = 0.5*(cparam[13*si+11] + cparam[13*sj+11]);  /* mu_roll */
            mat[12*k+11] = fmin(cparam[13*si+12],  cparam[13*sj+12]);  /* restitution e (min blend) */
            ci_arr[k] = si; cj_arr[k] = sj; cp_idx[k] = n;
            sub_id[k] = s;                   /* manifold sub-index from narrowphase */
            nc++;
        }
    }

    *nc_out = nc;

    /* ---- joint Coulomb friction rows (MuJoCo frictionloss) ----------------- *
     * Enumerate 1-DoF (rev/lin) joints with floss>0: each adds ONE constraint
     * row whose Jacobian is e_{fj} (selects that DoF's velocity), target 0, and a
     * CONSTANT box bound ±floss·dt in impulse units. Free joints (jtype=3) are
     * intentionally excluded in v1 (see docs/design-joint-friction.md §8). floss
     * is static, so the friction-row set is identical every step → its warm-start
     * λ (lam_fric, per-DoF) carries cleanly.
     *
     * TACT_NO_JFRIC compiles out every joint-FRICTION addition (limits stay) — used
     * to prove the zero-floss path is logically a no-op (any release-build delta is
     * then pure -ffast-math reassociation, not a logic change; see
     * docs/design-joint-friction.md Phase 3 notes). */
    int n_fric = 0;
#ifndef TACT_NO_JFRIC
    if (floss) {
        for (int i = 0; i < nb; i++) {
            if (jtype[i] != 1 && jtype[i] != 2) continue;   /* 1-DoF rev/lin only */
            int dof = q_base_local[i];
            if (floss[dof] > 0.0) {
                fric_dof[n_fric]  = dof;
                fric_body[n_fric] = i;
                n_fric++;
            }
        }
    }
#endif

    /* ---- joint limit rows (MuJoCo-style range) ---------------------------- *
     * One-sided position constraint on a 1-DoF rev/lin joint, posed at the velocity
     * level exactly like a contact normal: lower limit (q≥lo) activates when q≤lo
     * with Jacobian +e_j (push +q, "separating"); upper limit (q≤hi) when q≥hi with
     * −e_j. λ≥0, Baumgarte b=(erp/dt)·max(0, depth−slop). A DoF is limited iff
     * lo<hi; only the violated bound is active (≤1 row/DoF). rev/lin only (v1). */
    int n_limit = 0;
    if (q && jnt_lo && jnt_hi) {
        for (int i = 0; i < nb; i++) {
            if (jtype[i] != 1 && jtype[i] != 2) continue;
            int dof = q_base_local[i];
            if (!(jnt_lo[dof] < jnt_hi[dof])) continue;     /* lo<hi → limited */
            int sign = 0;
            if      (q[dof] <= jnt_lo[dof]) sign = +1;       /* lower bound active */
            else if (q[dof] >= jnt_hi[dof]) sign = -1;       /* upper bound active */
            if (sign != 0) {
                lim_dof[n_limit]  = dof;
                lim_sign[n_limit] = sign;
                lim_body[n_limit] = i;
                n_limit++;
            }
        }
    }

    /* ---- actuator rows (box-bounded implicit joint PD) --------------------- *
     * The implicit PD torque τ = Kp(qref − q − dt·qd⁺) + Kd(qdref − qd⁺) rewrites
     * as τ = b·(v* − qd⁺) with b = Kp·dt + Kd, v* = (Kp(qref−q) + Kd·qdref)/b —
     * a compliant velocity constraint toward v* with CFM 1/(b·dt), box-bounded by
     * the actuator limit |λ| ≤ taulim·dt. Unsaturated, the PGS solution equals the
     * ABA implicit-PD fold (same linear relation); saturated it delivers exactly
     * ±taulim, coupled consistently with contacts in the same solve. Rows exist
     * only for DoFs with taulim > 0 whose PD is active this step (staged by
     * tact_step_lcp, which masks those gains out of the ABA predictor). Kp counts
     * only with q_ref (same rule as ABA); qdref defaults to 0. rev/lin only (v1),
     * matching the ABA fold's 1-DoF-only implicit PD. */
    int n_act = 0;
    if (taulim && (act_qref || act_qdref) && (act_kp || act_kd)) {
        for (int i = 0; i < nb; i++) {
            if (jtype[i] != 1 && jtype[i] != 2) continue;
            int dof = q_base_local[i];
            if (taulim[dof] <= 0.0) continue;
            double Kp_i = (act_kp && act_qref) ? act_kp[dof] : 0.0;
            double Kd_i = act_kd ? act_kd[dof] : 0.0;
            double b_i  = Kp_i * dt + Kd_i;
            if (b_i <= 0.0) continue;
            double qr_i  = act_qref  ? act_qref[dof]  : 0.0;
            double qdr_i = act_qdref ? act_qdref[dof] : 0.0;
            act_dof[n_act]   = dof;
            act_body[n_act]  = i;
            act_b[n_act]     = b_i;
            act_vstar[n_act] = (Kp_i * (qr_i - q[dof]) + Kd_i * qdr_i) / b_i;
            n_act++;
        }
    }

    /* Constraint rows = 6·nc contact + n_fric friction + n_limit limit + n_act
       actuator (generic over row count/type, I2: everything below is driven off
       Mrow + row_blocks). */
    int Mrow = 6 * nc + n_fric + n_limit + n_act;
    if (Mrow == 0) {                               /* no contacts, no friction, no limit, no act */
        *iters_out    = 0;
        *residual_out = 0.0;
        return;
    }
    PROF_TS(t_p1);

    /* ---- PASS 2: stack J on the free subspace (Mrow × F) ------------------- */
    /* J row layout per contact k: [n, t1, t2, spin, r1, r2] = R_tan cols [2,0,1] for
       both linear and angular Jacobian rows of (body_j - body_i) relative motion. */
    for (int k = 0; k < Mrow * F; k++) J[k] = 0.0;

    for (int k = 0; k < nc; k++) {
        int si = ci_arr[k], sj = cj_arr[k];
        double *Rk = R_tan + 9*k;
        double *pc = p_world + 3*k;
        /* JvB - JvA  (relative linear Jacobian) accumulated implicitly via
           sign: +B body, -A body. Same for Jw rows. */
        for (int side = 0; side < 2; side++) {
            int body = (side == 0) ? cbody[si] : cbody[sj];
            if (body < 0) continue;
            double sign = (side == 0) ? -1.0 : 1.0;
            double Tp[16];
            memcpy(Tp, T + 16*body, 16*sizeof(double));
            Tp[3] = pc[0]; Tp[7] = pc[1]; Tp[11] = pc[2];
            jacob_whitney(J6, nb, T, Tp, parent, jtype, body);   /* 6 × nq row-major */
            /* Compress nq columns → F columns and accumulate projected rows. */
            for (int j = 0; j < nq; j++) {
                int jc = free_map[j];
                if (jc < 0) continue;
                double Jv0 = J6[0*nq + j], Jv1 = J6[1*nq + j], Jv2 = J6[2*nq + j];
                double Jw0 = J6[3*nq + j], Jw1 = J6[4*nq + j], Jw2 = J6[5*nq + j];
                /* dot 3-vector R column with the linear/angular triple */
                double t1v = Rk[0]*Jv0 + Rk[3]*Jv1 + Rk[6]*Jv2;
                double t2v = Rk[1]*Jv0 + Rk[4]*Jv1 + Rk[7]*Jv2;
                double nv  = Rk[2]*Jv0 + Rk[5]*Jv1 + Rk[8]*Jv2;
                double t1w = Rk[0]*Jw0 + Rk[3]*Jw1 + Rk[6]*Jw2;
                double t2w = Rk[1]*Jw0 + Rk[4]*Jw1 + Rk[7]*Jw2;
                double nw  = Rk[2]*Jw0 + Rk[5]*Jw1 + Rk[8]*Jw2;
                size_t base = (size_t)(6*k) * F + jc;
                J[base + 0*F] += sign * nv;        /* normal  */
                J[base + 1*F] += sign * t1v;       /* tan1    */
                J[base + 2*F] += sign * t2v;       /* tan2    */
                J[base + 3*F] += sign * nw;        /* spin    */
                J[base + 4*F] += sign * t1w;       /* roll1   */
                J[base + 5*F] += sign * t2w;       /* roll2   */
            }
        }
    }

    /* friction rows: J[6nc+r] = e_{fj} — a single 1.0 at the DoF's free column.
       The generic c-build (J·qd_f) then yields c=qd_free_{fj} and the generic A-build
       yields A_diag=(M⁻¹)_{fj,fj} and cross-terms with contacts, all for free. */
#ifndef TACT_NO_JFRIC
    for (int r = 0; r < n_fric; r++) {
        int fj = free_map[fric_dof[r]];
        J[(size_t)(6*nc + r) * F + fj] = 1.0;
    }
#endif

    /* limit rows: J[6nc+n_fric+r] = sign·e_{fj} (+1 lower bound / −1 upper). The
       generic c-build then gives c = sign·qd_free_{fj}; the limit Baumgarte bias is
       subtracted below, and PGS projects λ≥0 (one-sided, like a contact normal). */
    for (int r = 0; r < n_limit; r++) {
        int fj  = free_map[lim_dof[r]];
        int row = 6*nc + n_fric + r;
        J[(size_t)row * F + fj] = (double)lim_sign[r];
    }

    /* actuator rows: J[6nc+n_fric+n_limit+r] = e_{fj}, same shape as friction. The
       generic c-build gives c = qd_free_{fj}; the PD target bias −v* and the CFM
       1/(b·dt) are added below, and PGS clamps λ to the ±taulim·dt box. */
    for (int r = 0; r < n_act; r++) {
        int fj  = free_map[act_dof[r]];
        int row = 6*nc + n_fric + n_limit + r;
        J[(size_t)row * F + fj] = 1.0;
    }

    PROF_TS(t_p2);

    /* ---- PASS 3: factor M block-by-block; build Y = M⁻¹·Jᵀ; build A and c --- */
    /* S1: M is block-diagonal by kinematic connected component (free_map is the
       identity since fixed bodies own 0 slots, so M_full IS the free-subspace M).
       Gather each block's dense sub-matrix from M_full and factor it in place in
       the packed buffer; cross-block coupling is exactly 0 → bit-identical to a
       dense LDLᵀ of the whole matrix (see S1 note above). */
    lcp_partition_t *pt = &core->lcp_part;   /* per-handle; built on first solve */
    if (!pt->valid) lcp_build_partition(pt, nb, parent, jtype);
    int nblk = pt->nblk;
    for (int b = 0; b < nblk; b++) {
        int s = pt->blk_size[b];
        const int *d = pt->blk_dof + pt->blk_off[b];
        double *Mb = Mpack + pt->blk_mat_off[b];
        for (int r = 0; r < s; r++) {
            const double *Mr = M_full + (size_t)d[r] * nq;
            double *Mbr = Mb + (size_t)r * s;
            for (int c = 0; c < s; c++) Mbr[c] = Mr[d[c]];
        }
        if (ldlt_factor(Mb, s) != 0) {
            *iters_out    = 0;
            *residual_out = 0.0;
            return;
        }
    }

    /* Y[i,:] = M⁻¹ · J[i,:]ᵀ, contact by contact. The 6 rows of contact k are
       structurally 0 outside the (≤2) blocks of its two bodies — a fixed contact
       body (e.g. a foot) maps to the block of its nearest moving ancestor — so
       M⁻¹ only needs those blocks (the rest stays 0, as copied from J). */
    for (int k = 0; k < nc; k++) {
        int b1 = (cbody[ci_arr[k]] >= 0) ? pt->blk_of_body[cbody[ci_arr[k]]] : -1;
        int b2 = (cbody[cj_arr[k]] >= 0) ? pt->blk_of_body[cbody[cj_arr[k]]] : -1;
        for (int i = 6*k; i < 6*k + 6; i++) {
            double *Yi = Y + (size_t)i * F;
            memcpy(Yi, J + (size_t)i * F, F * sizeof(double));
            if (b1 >= 0)              lcp_block_solve(pt, b1, Yi, Mpack);
            if (b2 >= 0 && b2 != b1)  lcp_block_solve(pt, b2, Yi, Mpack);
        }
        /* record this contact's ≤2 distinct non-negative block ids (the support
           set used by the S2 sparse A build below). A contact owns 6 rows, all with
           the same support — replicate across them so the A-build reads a uniform
           per-row table (future friction rows fill their own single-row entries). */
        int n0 = -1, n1 = -1;
        if (b1 >= 0) n0 = b1;
        if (b2 >= 0 && b2 != b1) { if (n0 < 0) n0 = b2; else n1 = b2; }
        for (int a = 0; a < 6; a++) {
            row_blocks[2*(6*k + a) + 0] = n0;
            row_blocks[2*(6*k + a) + 1] = n1;
        }
    }

    /* friction rows: Y = M⁻¹·e_{fj} = column fj of M⁻¹ (restricted to the DoF's
       block). row_blocks is the singleton {block of the owning body}. */
#ifndef TACT_NO_JFRIC
    for (int r = 0; r < n_fric; r++) {
        int row = 6*nc + r;
        int b   = pt->blk_of_body[fric_body[r]];
        double *Yi = Y + (size_t)row * F;
        memcpy(Yi, J + (size_t)row * F, F * sizeof(double));   /* e_{fj} */
        if (b >= 0) lcp_block_solve(pt, b, Yi, Mpack);
        row_blocks[2*row + 0] = b;
        row_blocks[2*row + 1] = -1;
    }
#endif

    /* limit rows: Y = M⁻¹·(sign·e_{fj}); singleton block-support, same as friction. */
    for (int r = 0; r < n_limit; r++) {
        int row = 6*nc + n_fric + r;
        int b   = pt->blk_of_body[lim_body[r]];
        double *Yi = Y + (size_t)row * F;
        memcpy(Yi, J + (size_t)row * F, F * sizeof(double));   /* sign·e_{fj} */
        if (b >= 0) lcp_block_solve(pt, b, Yi, Mpack);
        row_blocks[2*row + 0] = b;
        row_blocks[2*row + 1] = -1;
    }

    /* actuator rows: Y = M⁻¹·e_{fj}; singleton block-support, same as friction. */
    for (int r = 0; r < n_act; r++) {
        int row = 6*nc + n_fric + n_limit + r;
        int b   = pt->blk_of_body[act_body[r]];
        double *Yi = Y + (size_t)row * F;
        memcpy(Yi, J + (size_t)row * F, F * sizeof(double));   /* e_{fj} */
        if (b >= 0) lcp_block_solve(pt, b, Yi, Mpack);
        row_blocks[2*row + 0] = b;
        row_blocks[2*row + 1] = -1;
    }

    PROF_TS(t_p3);

    /* A = J · Yᵀ  (Mrow × Mrow, dense-stored). ROW-DRIVEN (I2): A[ri][rj] is a
       STRUCTURAL ZERO unless rows ri and rj share an M-block — a J row is nonzero
       only on its block-support, so non-overlapping supports make every product term
       0 and the dense dot exactly 0.0. Skip those (memset leaves them 0, bit-identical
       to the dense result); compute the rest with the same full-F dot as before — so
       the whole A matrix is bit-identical, only the structural zeros are skipped.
       Sparsity is driven by per-ROW block-support SETS (row_blocks), NOT a single
       component-id and NOT a per-contact 6×6 tiling, so robot-robot contacts AND
       non-contact constraint rows (joint friction, limits) are handled uniformly.
       See docs/design-lcp-perf.md (I1/I2). For contacts-only input this is
       bit-identical to the former per-contact 6×6-block build: all 6 rows of a
       contact carry the same support, so share(ri,rj) == share(ri/6,rj/6) and the
       same cells are computed vs. skipped, each via the identical dotN.

       NB: column-restricting the dot to the shared block (a further ~10× on Amat)
       was tried and reverted — it reassociates the float sum vs the dense
       multi-accumulator dotN (~1e-12), which box_wall's marginally-stable stack
       amplifies past the regression's 1e-12 bound. Skipping is exact; restricting
       is not. Revisit only with a deliberate, reviewed baseline re-capture. */
    memset(A, 0, (size_t)Mrow * Mrow * sizeof(double));
    for (int ri = 0; ri < Mrow; ri++) {
        int a0 = row_blocks[2*ri], a1 = row_blocks[2*ri + 1];
        const double *Ji = J + (size_t)ri * F;
        double *Arow = A + (size_t)ri * Mrow;
        for (int rj = 0; rj < Mrow; rj++) {
            int c0 = row_blocks[2*rj], c1 = row_blocks[2*rj + 1];
            int share = (a0 >= 0 && (a0 == c0 || a0 == c1)) ||
                        (a1 >= 0 && (a1 == c0 || a1 == c1));
            if (!share) continue;
            Arow[rj] = dotN(Ji, Y + (size_t)rj * F, F);
        }
    }

    /* Build c = J · qd_free - b  (only normal entries of b are nonzero).
       For each contact k:
         v_n_pre = J[6k+0, :] · qd_free_f
         b_baum  = (erp/dt)·max(0, depth − slop)
         b_rest  = −e_rest·v_n_pre  if v_n_pre < −v_rest_thresh else 0
         b[6k+0] = max(b_baum, b_rest)
       Tangent/spin/roll rows: b = 0 (target zero slip). */
    /* compress qd_free (nq) to F-vector → reuse tmp */
    for (int i = 0, fi = 0; i < nq; i++) if (free_map[i] >= 0) tmp[fi++] = qd_free[i];
    /* J_f · qd_f → into c_vec (full Mrow vector) */
    for (int i = 0; i < Mrow; i++) c_vec[i] = dotN(J + (size_t)i*F, tmp, F);
    /* add −b on the normal rows; subtract restitution / Baumgarte */
    for (int k = 0; k < nc; k++) {
        double v_n_pre = c_vec[6*k + 0];           /* J_n · qd_f, before b applied */
        double b_baum  = (erp / dt) * (depth[k] - slop > 0.0 ? depth[k] - slop : 0.0);
        double e_rest  = mat[12*k+11];
        double b_rest  = (v_n_pre < -v_rest_thresh) ? -e_rest * v_n_pre : 0.0;
        double b_n     = b_baum > b_rest ? b_baum : b_rest;
        c_vec[6*k + 0] -= b_n;
    }
    /* limit rows: Baumgarte push-out (same form as the contact normal). depth =
       penetration past the active bound; the c-build already put sign·qd_free on the
       row, so subtracting b makes w = A·λ + c the post-velocity in the separating
       direction, projected λ≥0 below. No restitution/CFM (hard one-sided limit). */
    for (int r = 0; r < n_limit; r++) {
        int row = 6*nc + n_fric + r;
        int dof = lim_dof[r];
        double depth = (lim_sign[r] > 0) ? (jnt_lo[dof] - q[dof])   /* lower bound */
                                         : (q[dof] - jnt_hi[dof]);  /* upper bound */
        double b_lim = (erp / dt) * (depth - slop > 0.0 ? depth - slop : 0.0);
        c_vec[row] -= b_lim;
    }
    /* actuator rows: bias toward the PD target velocity v* (c already holds
       qd_free on the row from the generic c-build), so w = A·λ + c is the
       post-velocity error against the servo target. */
    for (int r = 0; r < n_act; r++) {
        int row = 6*nc + n_fric + n_limit + r;
        c_vec[row] -= act_vstar[r];
    }

    /* Add CFM diagonal regularization: R_diag = cfm_scale / (k·dt² + d·dt + ε).
       Folded directly into A's diagonal so PGS sees A_reg without an extra array. */
    for (int k = 0; k < nc; k++) {
        double k_n  = mat[12*k+0], d_n  = mat[12*k+1];
        double k_t  = mat[12*k+2], d_t  = mat[12*k+3];
        double k_sp = mat[12*k+5], d_sp = mat[12*k+6];
        double k_rl = mat[12*k+8], d_rl = mat[12*k+9];
        double R_n  = cfm_scale / (k_n *dt*dt + d_n *dt + 1e-12);
        double R_t  = cfm_scale / (k_t *dt*dt + d_t *dt + 1e-12);
        double R_sp = cfm_scale / (k_sp*dt*dt + d_sp*dt + 1e-12);
        double R_rl = cfm_scale / (k_rl*dt*dt + d_rl*dt + 1e-12);
        size_t base = (size_t)(6*k);
        A[base*Mrow + base + 0] += R_n;
        A[(base+1)*Mrow + (base+1)] += R_t;
        A[(base+2)*Mrow + (base+2)] += R_t;
        A[(base+3)*Mrow + (base+3)] += R_sp;
        A[(base+4)*Mrow + (base+4)] += R_rl;
        A[(base+5)*Mrow + (base+5)] += R_rl;
    }
    /* actuator rows: the compliance 1/(b·dt) IS the implicit PD (not a CFM
       tweak) — it makes the unsaturated PGS fixed point solve
       λ·(1/(b·dt)) = v* − qd⁺, i.e. τ = b·(v* − qd⁺), the same relation the
       ABA fold applies. No cfm_scale here; the regularization is physical. */
    for (int r = 0; r < n_act; r++) {
        size_t row = (size_t)(6*nc + n_fric + n_limit + r);
        A[row*Mrow + row] += 1.0 / (act_b[r] * dt);
    }

    PROF_TS(t_p4);

    /* ---- PASS 4: warm-start λ and PGS sweep with 4 cones ------------------- */
    /* warm-start: gather from lam_contact_out (already seeded from lam_contact_prev). Active
       contacts' slots in lam_contact_out will be overwritten in pass 5 — read first.
       Slot index: cp_idx[k] * TACT_MAX_PTS_PER_PAIR + sub_id[k]. */
    for (int k = 0; k < nc; k++) {
        int slot = cp_idx[k] * TACT_MAX_PTS_PER_PAIR + sub_id[k];
        memcpy(lam + 6*k, lam_contact_out + 6*slot, 6*sizeof(double));
    }
    /* friction rows warm-start from the per-DoF lam_fric carry (0 if cold). */
#ifndef TACT_NO_JFRIC
    for (int r = 0; r < n_fric; r++) {
        lam[6*nc + r] = lam_fric ? lam_fric[fric_dof[r]] : 0.0;
    }
#endif
    /* limit rows warm-start from the per-DoF lam_limit carry (0 if cold). */
    for (int r = 0; r < n_limit; r++) {
        lam[6*nc + n_fric + r] = lam_limit ? lam_limit[lim_dof[r]] : 0.0;
    }
    /* actuator rows warm-start from the per-DoF lam_act carry (0 if cold). */
    for (int r = 0; r < n_act; r++) {
        lam[6*nc + n_fric + n_limit + r] = lam_act ? lam_act[act_dof[r]] : 0.0;
    }

    /* w = A · λ + c  (initial; incrementally maintained inside PGS) */
    for (int i = 0; i < Mrow; i++) {
        w[i] = c_vec[i] + dotN(A + (size_t)i*Mrow, lam, Mrow);
    }

    double residual = 0.0;
    int it = 0;
    for (it = 0; it < iters; it++) {
        residual = 0.0;
        for (int k = 0; k < nc; k++) {
            double mu      = mat[12*k+ 4];
            double mu_spin = mat[12*k+ 7];
            double mu_roll = mat[12*k+10];
            int i_n  = 6*k + 0;
            int i_t1 = 6*k + 1, i_t2 = 6*k + 2;
            int i_sp = 6*k + 3;
            int i_r1 = 6*k + 4, i_r2 = 6*k + 5;

            double A_n  = A[(size_t)i_n *Mrow + i_n ];
            double A_t1 = A[(size_t)i_t1*Mrow + i_t1];
            double A_t2 = A[(size_t)i_t2*Mrow + i_t2];
            double A_sp = A[(size_t)i_sp*Mrow + i_sp];
            double A_r1 = A[(size_t)i_r1*Mrow + i_r1];
            double A_r2 = A[(size_t)i_r2*Mrow + i_r2];

            /* (1) normal: project to [0, ∞) */
            double row_n   = w[i_n] - A_n * lam[i_n];
            double new_n   = -row_n / A_n;
            if (new_n < 0.0) new_n = 0.0;
            double d_n_lam = new_n - lam[i_n];
            if (d_n_lam != 0.0) {
                double *Ai = A + (size_t)i_n * Mrow;       /* row i_n (= column by symmetry) */
                for (int r = 0; r < Mrow; r++) w[r] += Ai[r] * d_n_lam;
                lam[i_n] = new_n;
            }
            double res_n = d_n_lam < 0 ? -d_n_lam : d_n_lam;
            if (res_n > residual) residual = res_n;

            /* (2) linear tangent disk: ‖(λ_t1, λ_t2)‖ ≤ μ·λ_n
               compute both v1, v2 from current w (using each other's OLD value
               through w), then joint disk project, then column-update both. */
            double row_t1 = w[i_t1] - A_t1 * lam[i_t1];
            double row_t2 = w[i_t2] - A_t2 * lam[i_t2];
            double v1 = -row_t1 / A_t1;
            double v2 = -row_t2 / A_t2;
            double bound_t = mu * new_n;
            double mag_t   = sqrt(v1*v1 + v2*v2);
            if (mag_t > bound_t && mag_t > 1e-12) {
                double s = bound_t / mag_t;
                v1 *= s; v2 *= s;
            }
            double d1 = v1 - lam[i_t1];
            double d2 = v2 - lam[i_t2];
            if (d1 != 0.0 || d2 != 0.0) {
                double *Ai1 = A + (size_t)i_t1 * Mrow;
                double *Ai2 = A + (size_t)i_t2 * Mrow;
                for (int r = 0; r < Mrow; r++) w[r] += Ai1[r]*d1 + Ai2[r]*d2;
                lam[i_t1] = v1; lam[i_t2] = v2;
            }
            double r1a = d1 < 0 ? -d1 : d1; if (r1a > residual) residual = r1a;
            double r2a = d2 < 0 ? -d2 : d2; if (r2a > residual) residual = r2a;

            /* (3) spin (1D clamp): |λ_spin| ≤ μ_spin·λ_n */
            double row_sp = w[i_sp] - A_sp * lam[i_sp];
            double v_sp = -row_sp / A_sp;
            double bound_sp = mu_spin * new_n;
            if      (v_sp >  bound_sp) v_sp =  bound_sp;
            else if (v_sp < -bound_sp) v_sp = -bound_sp;
            double d_sp_lam = v_sp - lam[i_sp];
            if (d_sp_lam != 0.0) {
                double *Ai = A + (size_t)i_sp * Mrow;
                for (int r = 0; r < Mrow; r++) w[r] += Ai[r] * d_sp_lam;
                lam[i_sp] = v_sp;
            }
            double res_sp = d_sp_lam < 0 ? -d_sp_lam : d_sp_lam;
            if (res_sp > residual) residual = res_sp;

            /* (4) roll disk */
            double row_r1 = w[i_r1] - A_r1 * lam[i_r1];
            double row_r2 = w[i_r2] - A_r2 * lam[i_r2];
            double u1 = -row_r1 / A_r1;
            double u2 = -row_r2 / A_r2;
            double bound_r = mu_roll * new_n;
            double mag_r   = sqrt(u1*u1 + u2*u2);
            if (mag_r > bound_r && mag_r > 1e-12) {
                double s = bound_r / mag_r;
                u1 *= s; u2 *= s;
            }
            double e1 = u1 - lam[i_r1];
            double e2 = u2 - lam[i_r2];
            if (e1 != 0.0 || e2 != 0.0) {
                double *Ai1 = A + (size_t)i_r1 * Mrow;
                double *Ai2 = A + (size_t)i_r2 * Mrow;
                for (int r = 0; r < Mrow; r++) w[r] += Ai1[r]*e1 + Ai2[r]*e2;
                lam[i_r1] = u1; lam[i_r2] = u2;
            }
            double r3a = e1 < 0 ? -e1 : e1; if (r3a > residual) residual = r3a;
            double r4a = e2 < 0 ? -e2 : e2; if (r4a > residual) residual = r4a;
        }

        /* (5) joint-friction rows: 1D box clamp to ±floss·dt (constant bound, no
           normal-force coupling). Same incremental w-update as the cones. Kept a
           separate pass (not folded into the contact loop) per I5 — projection types
           plug into PGS only. With no CFM on these rows A_rr = (M⁻¹)_{fj,fj} > 0. */
#ifndef TACT_NO_JFRIC
        for (int r = 0; r < n_fric; r++) {
            int row = 6*nc + r;
            double A_rr  = A[(size_t)row * Mrow + row];
            double bound = floss[fric_dof[r]] * dt;       /* impulse units */
            double row_excl = w[row] - A_rr * lam[row];
            double v = -row_excl / A_rr;
            if      (v >  bound) v =  bound;
            else if (v < -bound) v = -bound;
            double dl = v - lam[row];
            if (dl != 0.0) {
                double *Ai = A + (size_t)row * Mrow;
                for (int i = 0; i < Mrow; i++) w[i] += Ai[i] * dl;
                lam[row] = v;
            }
            double ra = dl < 0 ? -dl : dl; if (ra > residual) residual = ra;
        }
#endif

        /* (6) joint-limit rows: one-sided clamp λ≥0 (identical to the contact normal
           projection — a "contact" on the joint coordinate). Sign is already baked
           into J/c, so the projection is just max(0, ...). */
        for (int r = 0; r < n_limit; r++) {
            int row = 6*nc + n_fric + r;
            double A_rr = A[(size_t)row * Mrow + row];
            double row_excl = w[row] - A_rr * lam[row];
            double v = -row_excl / A_rr;
            if (v < 0.0) v = 0.0;
            double dl = v - lam[row];
            if (dl != 0.0) {
                double *Ai = A + (size_t)row * Mrow;
                for (int i = 0; i < Mrow; i++) w[i] += Ai[i] * dl;
                lam[row] = v;
            }
            double ra = dl < 0 ? -dl : dl; if (ra > residual) residual = ra;
        }

        /* (7) actuator rows: 1D box clamp to ±taulim·dt (constant bound, like joint
           friction). The 1/(b·dt) compliance on A_rr makes the unclamped solve the
           implicit PD; the projection is the torque limit. */
        for (int r = 0; r < n_act; r++) {
            int row = 6*nc + n_fric + n_limit + r;
            double A_rr  = A[(size_t)row * Mrow + row];
            double bound = taulim[act_dof[r]] * dt;       /* impulse units */
            double row_excl = w[row] - A_rr * lam[row];
            double v = -row_excl / A_rr;
            if      (v >  bound) v =  bound;
            else if (v < -bound) v = -bound;
            double dl = v - lam[row];
            if (dl != 0.0) {
                double *Ai = A + (size_t)row * Mrow;
                for (int i = 0; i < Mrow; i++) w[i] += Ai[i] * dl;
                lam[row] = v;
            }
            double ra = dl < 0 ? -dl : dl; if (ra > residual) residual = ra;
        }

        if (residual < tol) break;
    }
    *iters_out    = it + (it < iters ? 1 : 0);
    *residual_out = residual;
    PROF_TS(t_p5);

    /* ---- PASS 5: velocity correction --------------------------------------- */
    /* dqd_f = M⁻¹ · (Jᵀ · λ). Jᵀ·λ has size F: tmp[c] = Σ_i J[i,c] · λ[i]. */
    for (int c = 0; c < F; c++) {
        double s = 0.0;
        for (int i = 0; i < Mrow; i++) s += J[(size_t)i*F + c] * lam[i];
        tmp[c] = s;
    }
    /* M⁻¹·(Jᵀλ): block-diagonal solve over all blocks (each covers disjoint DoF) */
    for (int b = 0; b < nblk; b++) lcp_block_solve(pt, b, tmp, Mpack);
    /* scatter back to full nq-vector */
    for (int i = 0, fi = 0; i < nq; i++) {
        if (free_map[i] >= 0) dqd_out[i] = tmp[fi++];
    }
    /* scatter λ to lam_contact_out at (cpair_idx, sub_id) slot for next step's warm-start */
    for (int k = 0; k < nc; k++) {
        int slot = cp_idx[k] * TACT_MAX_PTS_PER_PAIR + sub_id[k];
        memcpy(lam_contact_out + 6*slot, lam + 6*k, 6*sizeof(double));
    }
    /* scatter friction λ to the per-DoF lam_fric carry (in-place warm-start). */
#ifndef TACT_NO_JFRIC
    if (lam_fric)
        for (int r = 0; r < n_fric; r++) lam_fric[fric_dof[r]] = lam[6*nc + r];
#endif
    /* scatter limit λ to the per-DoF lam_limit carry (in-place warm-start). */
    if (lam_limit)
        for (int r = 0; r < n_limit; r++) lam_limit[lim_dof[r]] = lam[6*nc + n_fric + r];
    /* scatter actuator λ to the per-DoF lam_act carry (in-place warm-start). */
    if (lam_act)
        for (int r = 0; r < n_act; r++) lam_act[act_dof[r]] = lam[6*nc + n_fric + n_limit + r];

    /* ---- PASS 6: body-frame f_ext synthesis -------------------------------- */
    /* For each active contact:
         f_local = (λ_t1, λ_t2, λ_n) / dt           contact-frame force
         m_local = (λ_r1, λ_r2, λ_spin) / dt        contact-frame couple
         cf0 = R_tan · f_local                       world-frame force
         m0  = R_tan · m_local                       world-frame couple
       Body ci gets (−cf0, −m0). Body cj gets (+cf0, +m0).
       Body wrench layout = [moment(3), force(3)]: arm × force is moment from
       offset of application point relative to body origin. */
    for (int k = 0; k < nc; k++) {
        int si = ci_arr[k], sj = cj_arr[k];
        double *Rk = R_tan + 9*k;
        double *pw = p_world + 3*k;
        double f_loc[3] = { lam[6*k+1]/dt, lam[6*k+2]/dt, lam[6*k+0]/dt };
        double m_loc[3] = { lam[6*k+4]/dt, lam[6*k+5]/dt, lam[6*k+3]/dt };
        double cf0[3], m0[3];
        /* R_tan · f_loc */
        cf0[0] = Rk[0]*f_loc[0] + Rk[1]*f_loc[1] + Rk[2]*f_loc[2];
        cf0[1] = Rk[3]*f_loc[0] + Rk[4]*f_loc[1] + Rk[5]*f_loc[2];
        cf0[2] = Rk[6]*f_loc[0] + Rk[7]*f_loc[1] + Rk[8]*f_loc[2];
        m0[0]  = Rk[0]*m_loc[0] + Rk[1]*m_loc[1] + Rk[2]*m_loc[2];
        m0[1]  = Rk[3]*m_loc[0] + Rk[4]*m_loc[1] + Rk[5]*m_loc[2];
        m0[2]  = Rk[6]*m_loc[0] + Rk[7]*m_loc[1] + Rk[8]*m_loc[2];

        if (contact_i_out && contact_d_out) {
            contact_i_out[4*k + 0] = si;
            contact_i_out[4*k + 1] = sj;
            contact_i_out[4*k + 2] = cbody[si];
            contact_i_out[4*k + 3] = cbody[sj];
            contact_d_out[10*k + 0] = pw[0];
            contact_d_out[10*k + 1] = pw[1];
            contact_d_out[10*k + 2] = pw[2];
            contact_d_out[10*k + 3] = Rk[2];
            contact_d_out[10*k + 4] = Rk[5];
            contact_d_out[10*k + 5] = Rk[8];
            contact_d_out[10*k + 6] = cf0[0];
            contact_d_out[10*k + 7] = cf0[1];
            contact_d_out[10*k + 8] = cf0[2];
            contact_d_out[10*k + 9] = depth[k];
        }

        for (int side = 0; side < 2; side++) {
            int body = (side == 0) ? cbody[si] : cbody[sj];
            if (body < 0) continue;
            double sign = (side == 0) ? -1.0 : 1.0;
            double *Tb = T + 16*body;
            double Tb_inv[16];
            matinv4_affine(Tb_inv, Tb);
            /* ef = sign · Rᵀ · cf0  (Rᵀ extracts body-frame force) */
            double ef[3];
            ef[0] = sign * (Tb[0]*cf0[0] + Tb[4]*cf0[1] + Tb[ 8]*cf0[2]);
            ef[1] = sign * (Tb[1]*cf0[0] + Tb[5]*cf0[1] + Tb[ 9]*cf0[2]);
            ef[2] = sign * (Tb[2]*cf0[0] + Tb[6]*cf0[1] + Tb[10]*cf0[2]);
            /* rp = (Tb_inv · [pw,1])[:3] */
            double rp[3];
            rp[0] = Tb_inv[0]*pw[0] + Tb_inv[1]*pw[1] + Tb_inv[ 2]*pw[2] + Tb_inv[ 3];
            rp[1] = Tb_inv[4]*pw[0] + Tb_inv[5]*pw[1] + Tb_inv[ 6]*pw[2] + Tb_inv[ 7];
            rp[2] = Tb_inv[8]*pw[0] + Tb_inv[9]*pw[1] + Tb_inv[10]*pw[2] + Tb_inv[11];
            /* em = rp × ef + sign · Rᵀ · m0 */
            double em[3];
            cross3(rp[0], rp[1], rp[2], ef[0], ef[1], ef[2], em);
            em[0] += sign * (Tb[0]*m0[0] + Tb[4]*m0[1] + Tb[ 8]*m0[2]);
            em[1] += sign * (Tb[1]*m0[0] + Tb[5]*m0[1] + Tb[ 9]*m0[2]);
            em[2] += sign * (Tb[2]*m0[0] + Tb[6]*m0[1] + Tb[10]*m0[2]);
            f_ext_out[6*body + 0] += em[0];
            f_ext_out[6*body + 1] += em[1];
            f_ext_out[6*body + 2] += em[2];
            f_ext_out[6*body + 3] += ef[0];
            f_ext_out[6*body + 4] += ef[1];
            f_ext_out[6*body + 5] += ef[2];
        }
    }
    if (contact_count_out) *contact_count_out = nc;
#ifdef LCP_PROF
    {
        PROF_TS(t_p6);
        PROF_ADD(0, t_p1 - t_p0);   /* narrowphase  */
        PROF_ADD(1, t_p2 - t_p1);   /* J stacking   */
        PROF_ADD(2, t_p3 - t_p2);   /* Mf factor + Y = M⁻¹Jᵀ */
        PROF_ADD(3, t_p4 - t_p3);   /* A = J·Yᵀ + c + CFM */
        PROF_ADD(4, t_p5 - t_p4);   /* PGS sweep    */
        PROF_ADD(5, t_p6 - t_p5);   /* dqd + scatter + f_ext */
        PROF_ADD(6, t_p6 - t_p0);   /* TOTAL        */
        g_prof_calls++;
        g_prof_nc_sum += nc;
        if (nc > g_prof_nc_max) g_prof_nc_max = nc;
        g_prof_F_last = F;
    }
#endif
}
