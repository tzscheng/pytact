/* tact.c — high-level handle API.
 *
 * Owns tact_t (per-instance state object), orchestrates the collision side
 * (narrow.c / mpr.c / ray.c / shape.c) and rbd.c (dynamics) + lcp.c (contact)
 * into a single ctypes-friendly entry surface for the
 * Python package (sim.py drives it). Build / runtime split + lifecycle
 * invariants documented in docs/design-c-state.md §3.
 *
 * Public surface declared in tact.h. Struct definition is private to this
 * file — Python only sees an opaque void* handle. */
#include "tact.h"

/* ============================================================================
 * tact_t struct — owns dynamics arena (static + dynamic buffers) + a separate
 * feedback arena (resettable via tact_set_feedback).
 * ============================================================================ */
struct tact_t {
    /* scalars */
    int     nb;
    int     nq;                 /* total q vector length = sum(nq_per_body) */
    int     nv;                 /* total v vector length = sum(nv_per_body) */
                                /* Under axis-angle convention nq == nv; quat would
                                 * make them diverge by one extra slot per free joint. */
    int     n_shape, n_pair;
    int     integrator;         /* 1=euler, 2=rk4 — generic integrator selector (vestigial: the lcp path uses its own semi-implicit Euler) */
    double  dt;
    double  g[3];

    /* LCP solver knobs (set at create from YAML sim:, constant per sim). erp/slop/
     * cfm_scale = Baumgarte/deadband/CFM regularization; v_rest_thresh = restitution
     * velocity gate; iters/tol = PGS budget. Passed straight into contact_lcp(). */
    double  erp, slop, cfm_scale, v_rest_thresh, tol;
    int     iters;

    /* static (build-time copies) */
    int    *parent;             /* nb */
    int    *jtype;              /* nb */
    int    *q_base;             /* nb — first q index of body i (cumsum of nq_per_body) */
    int    *v_base;             /* nb — first v index of body i (cumsum of nv_per_body) */
    int    *nq_per_body;        /* nb — per-body q-state slot count */
    int    *nv_per_body;        /* nb — per-body v-state DoF count (currently == nq_per_body) */
    double *X;                  /* 36*nb */
    double *I6;                 /* 36*nb */
    double *Ti;                 /* 16*nb */
    double *ff;                 /* nq — joint damping coefficient (viscous) */
    double *sk;                 /* nq — joint spring stiffness */
    double *floss;              /* nq — joint Coulomb friction bound (frictionloss); 0 = off */
    double *armature;           /* nq — joint rotor/reflected inertia (added to M diagonal & ABA d); 0 = off */
    double *jnt_lo;             /* nq — joint lower limit (rev: rad, lin: m); limited iff lo < hi */
    double *jnt_hi;             /* nq — joint upper limit */
    int    *ctype;              /* n_shape */
    int    *cbody;              /* n_shape */
    double *cshape;             /* 3*n_shape */
    double *ctran;              /* 16*n_shape */
    double *cparam;             /* 13*n_shape — [pair_id, k_n, d_n, k_t, d_t, mu, k_spin, d_spin, mu_spin, k_roll, d_roll, mu_roll, restitution] */
    int    *craycast;           /* n_shape — 1: included in raycast, 0: invisible to ray (still renders + collides) */
    int    *cpair;              /* 2*n_pair */

    /* dynamic (overwritten each step) */
    double *T;                  /* 16*nb */
    double *f_ext;              /* 6*nb */
    double *f, *a, *v;          /* 6*nb each */
    double *qdd;                /* nb */
    double *q_next, *qd_next;   /* nb each */
    double *tau_p;              /* nb — scratch: tau - ff*qd - sk*q for integrator (explicit damping/spring) */
    double *workspace;          /* 120*nb (shared by euler_step / rk4_step / tact_gravity_query / tact_inertia_query / tact_bias_query / crb_featherstone) */

    /* LCP path (method=2) state — persistent across steps for warm-start */
    double *lam_prev;           /* 6 * MAX_PTS_PER_PAIR * max(n_pair,1) — λ from previous step
                                 * (indexed by slot = cpair_idx * MAX_PTS_PER_PAIR + sub_id) */
    double *lam_fric_prev;      /* nq — joint-friction λ warm-start (per-DoF, in-place carry).
                                 * Phase 3 keeps this internal/stateful; Phase 4 promotes it to
                                 * the ctx/SolverState-threaded pure-step buffer (design-joint-friction.md). */
    double *lam_limit_prev;     /* nq — joint-limit λ warm-start (per-DoF, in-place carry; ctx-threaded) */
    double *qd_free_buf;        /* nb — predictor velocity (qd + qdd_free * dt) */
    double *M_buf;              /* nb*nb — joint-space mass matrix (CRB output) */
    double *lcp_ws;             /* contact_lcp workspace; sized in tact_create */

    void   *arena;              /* root malloc — destroy() frees this + struct */

    /* Phase 2: feedback (filled by tact_set_feedback; separate arena allocated lazily) */
    int     fb_set;             /* 1 once feedback has been wired up */
    int     n_feeds;
    int    *feed_kinds;         /* n_feeds */
    int    *feed_offsets;       /* n_feeds + 1 — start of each feed's idx range */
    int    *feed_idx;           /* total count = feed_offsets[n_feeds] (frame indices) */
    int     n_frames;
    int    *fbody;              /* n_frames — frame_idx → body_idx, or -1 for root */
    double *ftran;              /* 16*n_frames */
    double *ftran_inv;          /* 16*n_frames — pre-computed for case 14 */
    int     y_size;
    double *y_buf;              /* y_size */
    void   *fb_arena;
};

tact_t *tact_create(int nb, int *parent, int *jtype, double *X, double *I6, double *Ti, double *ff, double *sk, double *floss, double *armature, double *jnt_lo, double *jnt_hi, double *g, double dt, int integrator, int n_shape, int n_pair, int *ctype, int *cbody, double *cshape, double *ctran, double *cparam, int *craycast, int *cpair, double erp, double slop, double cfm_scale, double v_rest_thresh, int iters, double tol)
{
    int npair_max = n_pair > 0 ? n_pair : 1;
    /* Per-body indexing. nq_per_body[i] = q slots, nv_per_body[i] = velocity DoFs.
     * Under axis-angle: 6 for jtype=3 (free), 0 for jtype=0 (fixed — no state),
     * else 1. d_total = sum(nv²) sizes the d-block array in aba/crb/rne workspace. */
    int nq = 0, nv_total = 0, d_total = 0;
    for (int i = 0; i < nb; ++i) {
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        int nqi = nvi;     /* axis-angle: nq_per_body == nv_per_body */
        nq       += nqi;
        nv_total += nvi;
        d_total  += nvi * nvi;
    }

    /* LCP workspace size (matches slice layout in lcp.c). The max constraint-row
     * count is M2 = 6·Pm + 2·nq (6 per contact-point + 1 friction + 1 limit per DoF),
     * so the row-sized buffers (J, Y = M2·nq; A = M2²; c_vec/lam/w = 3·M2; row_blocks
     * = 2·M2 ints) must use M2, NOT 6·Pm — otherwise a model with more frictive/limited
     * DoFs than contact-row capacity overflows. The int tail (ci,cj,cp_idx,sub_id=4·Pm;
     * free_map=nq; row_blocks=2·M2; fric_dof/body + limit_dof/sign/body = 5·nq) is
     * folded in (as doubles) with slack below. */
    size_t Pm_max = (size_t)MAX_PTS_PER_PAIR * (size_t)npair_max;
    size_t M2_max = (size_t)6*Pm_max + (size_t)2*nq;     /* max constraint rows */
    size_t lcp_ws_doubles = M2_max*M2_max               /* A */
                          + (size_t)2*M2_max*nq          /* J + Y */
                          + (size_t)nq*nq                /* Mpack */
                          + (size_t)3*M2_max             /* c_vec, lam, w */
                          + (size_t)25*Pm_max            /* p_world, R_tan, depth, mat */
                          + (size_t)10*nq                /* tmp, J6, slack */
                          + (size_t)(2*Pm_max + M2_max + 4*nq)  /* int tail as doubles (overestimate) */
                          + 128;

    /* aba/crb/rne workspace (flat layout, see rbd.c aba_featherstone):
     *   Xup 36nb + S 6nq + vv 6nb + cc 6nb + IA 36nb + pA 6nb + U 6nq + d d_total + u nq
     *   = 90nb + 13nq + d_total + slack.
     * rk4_step layers 10*nq intermediate (qdd1..4 / qk1..3 / qdk1..3) + nq tau_dummy
     * after the aba slice. Total: 90nb + 24nq + d_total + slack.
     * For non-free (nq=nb, d_total=nb): 115nb. We keep 120nb floor for legacy. */
    size_t aba_ws_size = (size_t)90*nb + (size_t)24*nq + (size_t)d_total + 64;
    if (aba_ws_size < (size_t)120*nb) aba_ws_size = (size_t)120*nb;
    /* size doubles + ints separately; place doubles first so the carve stays 8-aligned */
    size_t bytes_dbl = sizeof(double) * (
        36*nb + 36*nb + 16*nb              /* X, I6, Ti */
      + nq + nq + nq + nq                   /* ff, sk, floss, armature (per-DoF) */
      + nq + nq                             /* jnt_lo, jnt_hi (per-DoF) */
      + 3*n_shape + 16*n_shape + 13*n_shape /* cshape, ctran, cparam */
      + 16*nb                              /* T */
      + 6*nb + 6*nb + 6*nb + 6*nb          /* f_ext, f, a, v */
      + nq + nq + nq + nq                   /* qdd, q_next, qd_next, tau_p (per-DoF) */
      + aba_ws_size                         /* workspace (sized for aba/crb/rne incl. free) */
      + 6*MAX_PTS_PER_PAIR*npair_max         /* lam_prev (LCP warm-start, slot-indexed) */
      + nq                                  /* lam_fric_prev (joint-friction warm-start, per-DoF) */
      + nq                                  /* lam_limit_prev (joint-limit warm-start, per-DoF) */
      + nq                                  /* qd_free_buf (LCP predictor, per-DoF) */
      + nq*nq                               /* M_buf (joint-space mass matrix) */
      + lcp_ws_doubles                      /* lcp_ws (contact_lcp workspace) */
    );
    /* ints: parent, jtype, q_base, v_base, nq_per_body, nv_per_body (6*nb)
     *     + ctype, cbody, craycast (3*n_shape) + cpair (2*n_pair) */
    size_t bytes_int = sizeof(int) * (6*nb + 3*n_shape + 2*n_pair);

    tact_t *h = (tact_t*)calloc(1, sizeof(tact_t));
    h->arena = malloc(bytes_dbl + bytes_int);

    h->nb = nb; h->nq = nq; h->nv = nv_total;
    h->n_shape = n_shape; h->n_pair = n_pair;
    h->integrator = integrator; h->dt = dt;
    h->erp = erp; h->slop = slop; h->cfm_scale = cfm_scale;
    h->v_rest_thresh = v_rest_thresh; h->iters = iters; h->tol = tol;
    h->g[0] = g[0]; h->g[1] = g[1]; h->g[2] = g[2];

    char *p = (char*)h->arena;
    #define CARVE_DBL(field, n) do { h->field = (double*)p; p += (size_t)(n)*sizeof(double); } while (0)
    CARVE_DBL(X,        36*nb);
    CARVE_DBL(I6,       36*nb);
    CARVE_DBL(Ti,       16*nb);
    CARVE_DBL(ff,       nq);
    CARVE_DBL(sk,       nq);
    CARVE_DBL(floss,    nq);
    CARVE_DBL(armature, nq);
    CARVE_DBL(jnt_lo,   nq);
    CARVE_DBL(jnt_hi,   nq);
    CARVE_DBL(cshape,   3*n_shape);
    CARVE_DBL(ctran,    16*n_shape);
    CARVE_DBL(cparam,   13*n_shape);
    CARVE_DBL(T,        16*nb);
    CARVE_DBL(f_ext,    6*nb);
    CARVE_DBL(f,        6*nb);
    CARVE_DBL(a,        6*nb);
    CARVE_DBL(v,        6*nb);
    CARVE_DBL(qdd,      nq);
    CARVE_DBL(q_next,   nq);
    CARVE_DBL(qd_next,  nq);
    CARVE_DBL(tau_p,    nq);
    CARVE_DBL(workspace, aba_ws_size);
    CARVE_DBL(lam_prev,    6*MAX_PTS_PER_PAIR*npair_max);
    CARVE_DBL(lam_fric_prev, nq);
    CARVE_DBL(lam_limit_prev, nq);
    CARVE_DBL(qd_free_buf, nq);
    CARVE_DBL(M_buf,       nq*nq);
    CARVE_DBL(lcp_ws,      lcp_ws_doubles);
    #undef CARVE_DBL
    #define CARVE_INT(field, n) do { h->field = (int*)p; p += (size_t)(n)*sizeof(int); } while (0)
    CARVE_INT(parent,       nb);
    CARVE_INT(jtype,        nb);
    CARVE_INT(q_base,       nb);
    CARVE_INT(v_base,       nb);
    CARVE_INT(nq_per_body,  nb);
    CARVE_INT(nv_per_body,  nb);
    CARVE_INT(ctype,        n_shape);
    CARVE_INT(cbody,        n_shape);
    CARVE_INT(craycast,     n_shape);
    CARVE_INT(cpair,        2*n_pair);
    #undef CARVE_INT

    /* copy static data */
    memcpy(h->parent, parent, nb*sizeof(int));
    memcpy(h->jtype,  jtype,  nb*sizeof(int));
    /* fill q_base / v_base / nq_per_body / nv_per_body from jtype */
    int q_offset = 0, v_offset = 0;
    for (int i = 0; i < nb; ++i) {
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        int nqi = nvi;     /* axis-angle convention */
        h->q_base[i]      = q_offset;
        h->v_base[i]      = v_offset;
        h->nq_per_body[i] = nqi;
        h->nv_per_body[i] = nvi;
        q_offset         += nqi;
        v_offset         += nvi;
    }
    memcpy(h->X,      X,      36*nb*sizeof(double));
    memcpy(h->I6,     I6,     36*nb*sizeof(double));
    memcpy(h->Ti,     Ti,     16*nb*sizeof(double));
    memcpy(h->ff,     ff,        nq*sizeof(double));
    memcpy(h->sk,     sk,        nq*sizeof(double));
    if (floss) memcpy(h->floss, floss, nq*sizeof(double));
    else       memset(h->floss, 0,     nq*sizeof(double));
    if (armature) memcpy(h->armature, armature, nq*sizeof(double));
    else          memset(h->armature, 0,        nq*sizeof(double));
    if (jnt_lo) memcpy(h->jnt_lo, jnt_lo, nq*sizeof(double)); else memset(h->jnt_lo, 0, nq*sizeof(double));
    if (jnt_hi) memcpy(h->jnt_hi, jnt_hi, nq*sizeof(double)); else memset(h->jnt_hi, 0, nq*sizeof(double));
    if (n_shape > 0) {
        memcpy(h->ctype,  ctype,  n_shape*sizeof(int));
        memcpy(h->cbody,  cbody,  n_shape*sizeof(int));
        memcpy(h->cshape, cshape, 3*n_shape*sizeof(double));
        memcpy(h->ctran,  ctran,  16*n_shape*sizeof(double));
        memcpy(h->cparam, cparam, 13*n_shape*sizeof(double));
        if (craycast) memcpy(h->craycast, craycast, n_shape*sizeof(int));
        else for (int i = 0; i < n_shape; i++) h->craycast[i] = 1;  /* default: all visible to raycast */
    }
    if (n_pair > 0) memcpy(h->cpair, cpair, 2*n_pair*sizeof(int));

    memset(h->lam_prev,     0, 6*MAX_PTS_PER_PAIR*npair_max*sizeof(double));
    memset(h->lam_fric_prev, 0, nq*sizeof(double));
    memset(h->lam_limit_prev, 0, nq*sizeof(double));
    return h;
}

void tact_destroy(tact_t *h)
{
    if (!h) return;
    free(h->arena);
    if (h->fb_arena) free(h->fb_arena);
    free(h);
}

/* Phase 2: pre-marshal feedback descriptors into the handle. Allocates a separate
 * arena (h->fb_arena) so it can be re-set without disturbing the dynamics arena.
 * After this is called, tact_step() will fill h->y_buf at the end of each step. */
void tact_set_feedback(tact_t *h, int n_feeds, int *kinds, int *offsets, int *idx, int n_frames, int *fbody, double *ftran, double *ftran_inv, int y_size)
{
    if (h->fb_arena) { free(h->fb_arena); h->fb_arena = NULL; }

    int n_idx = (n_feeds > 0) ? offsets[n_feeds] : 0;
    int y_alloc = y_size > 0 ? y_size : 1;   /* always alloc ≥1 to keep ptr non-null */
    int frame_alloc = n_frames > 0 ? n_frames : 1;

    size_t bytes_dbl = sizeof(double) * (16*frame_alloc + 16*frame_alloc + y_alloc);
    size_t bytes_int = sizeof(int) * (
        (n_feeds > 0 ? n_feeds : 1)         /* feed_kinds */
      + (n_feeds + 1)                        /* feed_offsets */
      + (n_idx > 0 ? n_idx : 1)              /* feed_idx */
      + frame_alloc);                        /* fbody */
    h->fb_arena = malloc(bytes_dbl + bytes_int);

    char *p = (char*)h->fb_arena;
    h->ftran     = (double*)p; p += 16*frame_alloc*sizeof(double);
    h->ftran_inv = (double*)p; p += 16*frame_alloc*sizeof(double);
    h->y_buf     = (double*)p; p += y_alloc*sizeof(double);
    h->feed_kinds   = (int*)p; p += (n_feeds > 0 ? n_feeds : 1)*sizeof(int);
    h->feed_offsets = (int*)p; p += (n_feeds + 1)*sizeof(int);
    h->feed_idx     = (int*)p; p += (n_idx > 0 ? n_idx : 1)*sizeof(int);
    h->fbody        = (int*)p;

    if (n_frames > 0) {
        memcpy(h->fbody,     fbody,     n_frames*sizeof(int));
        memcpy(h->ftran,     ftran,     16*n_frames*sizeof(double));
        memcpy(h->ftran_inv, ftran_inv, 16*n_frames*sizeof(double));
    }
    if (n_feeds > 0) {
        memcpy(h->feed_kinds,   kinds,   n_feeds*sizeof(int));
        memcpy(h->feed_offsets, offsets, (n_feeds + 1)*sizeof(int));
        if (n_idx > 0) memcpy(h->feed_idx, idx, n_idx*sizeof(int));
    }
    h->n_feeds  = n_feeds;
    h->n_frames = n_frames;
    h->y_size   = y_size;
    h->fb_set   = 1;
}

/* In-place update of inertia buffers (X, I6, Ti). Topology (nb) must be unchanged.
 * Arena is preserved, so any numpy views from tact_get_* remain valid (cf. §3.5).
 * For topology changes, the caller must destroy + create a fresh handle. */
void tact_edit_model(tact_t *h, double *X, double *I6, double *Ti)
{
    memcpy(h->X,  X,  36*h->nb*sizeof(double));
    memcpy(h->I6, I6, 36*h->nb*sizeof(double));
    memcpy(h->Ti, Ti, 16*h->nb*sizeof(double));
}

/* Phase 2: feedback in C — mirrors sim.py:Model.feedback() 14 cases.
 * Reads from h->{T,v,a,f,f_ext}, q, qd, tau (raw actuation, for case 3).
 * Writes to h->y_buf (length h->y_size). */
static void tact_feedback(tact_t *h, double *q, double *qd, double *tau)
{
    double *y = h->y_buf;
    int yi = 0;

    for (int fi = 0; fi < h->n_feeds; fi++) {
        int kind  = h->feed_kinds[fi];
        int start = h->feed_offsets[fi];
        int end   = h->feed_offsets[fi + 1];

        for (int k = start; k < end; k++) {
            int frame_idx = h->feed_idx[k];
            int body_idx  = h->fbody[frame_idx];
            double *Tb    = h->T     + 16*body_idx;   /* 4×4 row-major */
            double *vb    = h->v     + 6 *body_idx;   /* [w(3); v0(3)] body frame */
            double *ab    = h->a     + 6 *body_idx;
            double *fb    = h->f     + 6 *body_idx;
            /* f_ext (external force) was previously used by case 14 — kept the
             * pointer commented for reference. Replaced by `fb` (propagated
             * wrench) so the FT readout is mass-correct, not just for massless
             * bodies. */
            /* double *fe = h->f_ext + 6 *body_idx; */
            double *Tf    = h->ftran + 16*frame_idx;
            double *Tfi   = h->ftran_inv + 16*frame_idx;

            switch (kind) {
            /* Cases 1/2/3 are only meaningful for 1-DoF joints. q uses q_base
             * (position state), qd/tau use v_base (velocity state). For 1-DoF
             * q_base[i] == v_base[i] always. Fixed bodies have nv_per_body=0
             * so reading q[q_base] would alias the next body's slot — emit 0
             * instead (callers shouldn't request joint state from a fixed body
             * but YAML 'jointpos: fixed_body' shouldn't crash either). */
            case 1: y[yi++] = h->nq_per_body[body_idx] ? q   [h->q_base[body_idx]] : 0.0; break;
            case 2: y[yi++] = h->nv_per_body[body_idx] ? qd  [h->v_base[body_idx]] : 0.0; break;
            case 3: y[yi++] = h->nv_per_body[body_idx] ? tau [h->v_base[body_idx]] : 0.0; break;
		
            case 4: { /* framepos: (T_body @ Tf)[:3,3] */
                double Tw[16];
                matmul(Tw, Tb, Tf, 4, 4, 4);
                y[yi++] = Tw[3]; y[yi++] = Tw[7]; y[yi++] = Tw[11];
                break;
            }
		
            case 5: { /* framequat: rotation_to_quat((T_body @ Tf)[:3,:3]); Python returns [w,x,y,z] */
                /* BUG FIX: include the frame's own rotation (Tf's R block) — matches
                   mujoco's <framequat objtype="site"> semantics. For frames with
                   identity Tf rotation (most "site" cases) old and new agree.
                   OLD: double R[9] = {Tb[0],Tb[1],Tb[2], Tb[4],Tb[5],Tb[6], Tb[8],Tb[9],Tb[10]}; */
                double Tw[16];
                matmul(Tw, Tb, Tf, 4, 4, 4);
                double R[9] = {Tw[0],Tw[1],Tw[2], Tw[4],Tw[5],Tw[6], Tw[8],Tw[9],Tw[10]};
                double qxyzw[4];
                rotation_to_quat(R, qxyzw); /* C convention: [x,y,z,w] */
                y[yi++] = qxyzw[3]; y[yi++] = qxyzw[0]; y[yi++] = qxyzw[1]; y[yi++] = qxyzw[2];
                break;
            }
		
            case 6: { /* framelinvel: R_body @ (v0 + w × r), r=Tf[:3,3] */
                double r[3]  = {Tf[3], Tf[7], Tf[11]};
                double w[3]  = {vb[0], vb[1], vb[2]};
                double v0[3] = {vb[3], vb[4], vb[5]};
                double cx[3]; cross3(w[0],w[1],w[2], r[0],r[1],r[2], cx);
                double v1[3] = {v0[0]+cx[0], v0[1]+cx[1], v0[2]+cx[2]};
                y[yi++] = Tb[0]*v1[0] + Tb[1]*v1[1] + Tb[2]*v1[2];
                y[yi++] = Tb[4]*v1[0] + Tb[5]*v1[1] + Tb[6]*v1[2];
                y[yi++] = Tb[8]*v1[0] + Tb[9]*v1[1] + Tb[10]*v1[2];
                break;
            }
		
            case 7: { /* frameangvel: R_body @ w */
                y[yi++] = Tb[0]*vb[0] + Tb[1]*vb[1] + Tb[2]*vb[2];
                y[yi++] = Tb[4]*vb[0] + Tb[5]*vb[1] + Tb[6]*vb[2];
                y[yi++] = Tb[8]*vb[0] + Tb[9]*vb[1] + Tb[10]*vb[2];
                break;
            }
		
            case 8:    /* framelinacc: R_body @ s + g_world; s = a0 + ẇ×r + w×(w×r) */
            case 12: { /* accelerometer: same s, no R_body @ , no +g */
                /* BUG FIX: rne returns _a[3:] = a^F + ω×v0 = proper acceleration
                   of body origin (= accelerometer reading at body origin). So:
                   - Accelerometer at point r:  proper(r)_body = a0 + ẇ×r + ω×(ω×r)
                   - Framelinacc (world kinematic): R @ proper(r)_body + g_world
                   The old `s` had an extra w×v0 term (double-counting rne's
                   correction) and case 8 was missing the +g term. At rest
                   ω=v0=0 so the bug was hidden, but during dynamic motion the
                   error was ~|ω||v0| (a few % of g during walking).
                   OLD: s = a0 + w×v0 + ẇ×r + w×(w×r), case 8 had no +g */
                double r[3]  = {Tf[3], Tf[7], Tf[11]};
                double w[3]  = {vb[0], vb[1], vb[2]};
                double a0[3] = {ab[3], ab[4], ab[5]};
                double wd[3] = {ab[0], ab[1], ab[2]};
                /* double v0[3] = {vb[3], vb[4], vb[5]}; */     /* unused after fix */
                /* double c1[3]; cross3(w,v0,c1); */            /* removed: extra ω×v0 */
                double c2[3], c3[3], c4[3];
                cross3(wd[0],wd[1],wd[2],r[0], r[1], r[2],  c2);
                cross3(w[0],w[1],w[2],   r[0], r[1], r[2],  c3);
                cross3(w[0],w[1],w[2],   c3[0],c3[1],c3[2], c4);
                double s[3] = {a0[0]+c2[0]+c4[0], a0[1]+c2[1]+c4[1], a0[2]+c2[2]+c4[2]};
                if (kind == 8) {
                    y[yi++] = Tb[0]*s[0] + Tb[1]*s[1] + Tb[2]*s[2]  + h->g[0];
                    y[yi++] = Tb[4]*s[0] + Tb[5]*s[1] + Tb[6]*s[2]  + h->g[1];
                    y[yi++] = Tb[8]*s[0] + Tb[9]*s[1] + Tb[10]*s[2] + h->g[2];
                } else {
                    y[yi++] = s[0]; y[yi++] = s[1]; y[yi++] = s[2];
                }
                break;
            }
		
            case 9: { /* frameangacc: R_body @ ẇ */
                y[yi++] = Tb[0]*ab[0] + Tb[1]*ab[1] + Tb[2]*ab[2];
                y[yi++] = Tb[4]*ab[0] + Tb[5]*ab[1] + Tb[6]*ab[2];
                y[yi++] = Tb[8]*ab[0] + Tb[9]*ab[1] + Tb[10]*ab[2];
                break;
            }
		
            case 10:  /* velocimeter */
		y[yi++] = vb[3]; y[yi++] = vb[4]; y[yi++] = vb[5];
		break; 
		
            case 11: /* gyro */
		y[yi++] = vb[0]; y[yi++] = vb[1]; y[yi++] = vb[2];
		break;
		
            case 13: { /* ft-sensor: [f_lin; f_ang] = [f[3:6]; f[0:3]] */
                y[yi++] = fb[3]; y[yi++] = fb[4]; y[yi++] = fb[5];
                y[yi++] = fb[0]; y[yi++] = fb[1]; y[yi++] = fb[2];
                break;
            }
		
            case 14: { /* contact force (frame-transformed FT sensor).
                          Force:  f1   = R₁ᵀ · f_lin
                          Moment: m1   = R₁ᵀ · f_mom + t₁ × f1
                          where (R₁,t₁) = ftran_inv (R₁ rotates body→frame,
                          t₁ is body origin in frame coords), and (f_lin, f_mom)
                          is the ABA/RNE-propagated spatial wrench transmitted
                          from parent through joint to body. Matches mujoco/
                          Drake/Bullet constraint-Lagrange-multiplier semantics —
                          includes inertial terms (I·a + crf·I·v) so the reading
                          is correct for any mass distribution. For mass=0 with
                          no velocity, f = -f_ext, so the f_ext-based formula
                          gave the same output in that special case. */
                /* OLDEST (contact force only, exact only for massless body):
                   double f_lin[3] = {fe[3], fe[4], fe[5]};
                   double f_mom[3] = {fe[0], fe[1], fe[2]};
                   (-Tfi[...]·f_lin and -Tfi[...]·(f_mom + t₁ × f_lin))         */
                /* OLD (f-based but wrong moment-transfer when frame rotates):
                   the cross was done with t₁ (frame coords) and f_lin (body
                   coords), so the moment-arm shift was wrong whenever ftran
                   has a non-identity rotation R block.
                   double cx[3]; cross3(t1, f_lin, cx);
                   double s[3] = {f_mom + cx};
                   y = Tfi[:3,:3] @ s;                                            */
                double f_lin[3] = {fb[3], fb[4], fb[5]};
                double f_mom[3] = {fb[0], fb[1], fb[2]};
                double t1[3]    = {Tfi[3], Tfi[7], Tfi[11]};
                /* f1 = R₁ᵀ · f_lin  (force rotated to frame coords) */
                double f1[3];
                f1[0] = Tfi[0]*f_lin[0] + Tfi[1]*f_lin[1] + Tfi[2] *f_lin[2];
                f1[1] = Tfi[4]*f_lin[0] + Tfi[5]*f_lin[1] + Tfi[6] *f_lin[2];
                f1[2] = Tfi[8]*f_lin[0] + Tfi[9]*f_lin[1] + Tfi[10]*f_lin[2];
                y[yi++] = f1[0]; y[yi++] = f1[1]; y[yi++] = f1[2];
                /* m1 = R₁ᵀ · f_mom + t₁ × f1  (moment-arm shift in frame coords) */
                double m_rot[3];
                m_rot[0] = Tfi[0]*f_mom[0] + Tfi[1]*f_mom[1] + Tfi[2] *f_mom[2];
                m_rot[1] = Tfi[4]*f_mom[0] + Tfi[5]*f_mom[1] + Tfi[6] *f_mom[2];
                m_rot[2] = Tfi[8]*f_mom[0] + Tfi[9]*f_mom[1] + Tfi[10]*f_mom[2];
                double cx[3]; cross3(t1[0],t1[1],t1[2], f1[0],f1[1],f1[2], cx);
                y[yi++] = m_rot[0] + cx[0];
                y[yi++] = m_rot[1] + cx[1];
                y[yi++] = m_rot[2] + cx[2];
                break;
            }
            }
        }
    }
}

/* LCP path: _fk → ABA(no contact) → CRB → contact_lcp → semi-implicit Euler → feedback.
   Self-contained. Reads q, qd, tau. Writes h->{T, f_ext (contact wrench), f, a, v,
   qdd, q_next, qd_next, M_buf, qd_free_buf, lcp_ws (transient), y_buf,
   lam_prev (warm-start carry)}. */
void tact_step_lcp(tact_t *h, double *q, double *qd, double *tau, double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref, double *lam_in, double *lam_out, double *lam_fric_in, double *lam_fric_out, double *lam_limit_in, double *lam_limit_out)
{
    /* Stage 1: forward kinematics */
    _fk(h->T, h->nb, h->Ti, h->parent, h->jtype, q);

    /* free predictor: aba_featherstone(f_ext=0). Output qdd is no-contact joint
       accel. f/a/v are overwritten by RNE below (post-contact spatial dynamics).
       Kp_j/Kd_j/q_ref/qd_ref pass through to aba's implicit PD path
       (NULLs = inactive). full=0 — we only need qdd here. */
    memset(h->f_ext, 0, 6*h->nb*sizeof(double));
    aba_featherstone(h->nb, h->X, h->I6, h->parent, h->jtype, q, qd, tau,
                     h->f_ext, h->g, h->qdd, h->f, h->a, h->v, h->workspace,
                     h->ff, h->sk, h->armature, h->dt, Kp_j, Kd_j, q_ref, qd_ref, /*full=*/0);
    for (int i = 0; i < h->nq; i++) h->qd_free_buf[i] = qd[i] + h->qdd[i] * h->dt;

    /* joint-space mass matrix at q. Add armature to the diagonal (rotor/reflected
       inertia, MuJoCo-style) so the contact solve sees the same effective inertia as
       the ABA predictor above. armature=0 → no-op (bit-identical). */
    crb_featherstone(h->nb, h->X, h->I6, h->parent, h->jtype, q, h->M_buf, h->workspace);
    for (int i = 0; i < h->nq; i++) h->M_buf[(size_t)i*h->nq + i] += h->armature[i];

    /* LCP solve: writes dqd into h->qdd (reused as scratch), fills h->f_ext with
       contact wrench, and updates h->lam_prev in place (warm-start carry). */
    int nc_out = 0, iters_out = 0;
    double residual_out = 0.0;
    /* Explicit warm-start carry (referential transparency): caller supplies lam_in
       and receives lam_out. NULL → fall back to the handle's internal h->lam_prev
       (legacy stateful behavior). contact_lcp seeds lam_out from lam_in when they
       differ, so lam_in is left untouched (caller's ctx stays immutable). */
    double *lin  = lam_in  ? lam_in  : h->lam_prev;
    double *lout = lam_out ? lam_out : h->lam_prev;
    /* joint-friction warm-start, threaded for referential transparency just like the
       contact λ: read from lam_fric_in, write to lam_fric_out (seeded from in so the
       in-place contact_lcp update leaves lam_fric_in untouched). NULL → internal
       h->lam_fric_prev (legacy stateful carry). */
    double *lfin  = lam_fric_in  ? lam_fric_in  : h->lam_fric_prev;
    double *lfout = lam_fric_out ? lam_fric_out : h->lam_fric_prev;
    if (lfout != lfin) memcpy(lfout, lfin, h->nq * sizeof(double));
    /* joint-limit warm-start, threaded the same immutable-in/out way. */
    double *llin  = lam_limit_in  ? lam_limit_in  : h->lam_limit_prev;
    double *llout = lam_limit_out ? lam_limit_out : h->lam_limit_prev;
    if (llout != llin) memcpy(llout, llin, h->nq * sizeof(double));
    contact_lcp(h->nb, h->T, h->parent, h->jtype,
                h->n_pair, h->cpair, h->ctype, h->cbody,
                h->ctran, h->cshape, h->cparam,
                h->qd_free_buf, h->M_buf, h->dt,
                h->erp, h->slop, h->cfm_scale, h->v_rest_thresh,
                h->iters, h->tol,
                lin,                        /* in: previous λ (warm-start) */
                h->floss, lfout,            /* joint Coulomb friction + its warm-start (in-place on lfout) */
                q, h->jnt_lo, h->jnt_hi, llout,  /* joint limits (q for activation) + warm-start (in-place on llout) */
                h->qdd,                     /* out: dqd (velocity correction) */
                lout,                       /* out: λ_full (next warm-start) */
                h->f_ext,                   /* out: per-body contact wrench */
                &nc_out, &iters_out, &residual_out,
                h->lcp_ws);

    /* semi-implicit Euler: qd_next = qd_free + dqd; q_next = q_step(q, qd_next, dt).
       q_step handles SO(3) integration for free bodies (translation R·v_body·dt
       and rotation via exp map). For non-free it reduces to q + qd_next·dt.
       qdd is overwritten with the realized effective acceleration for feedback. */
    for (int i = 0; i < h->nq; i++) h->qd_next[i] = h->qd_free_buf[i] + h->qdd[i];
    q_step(h->nb, h->jtype, q, h->qd_next, h->dt, h->q_next);
    for (int i = 0; i < h->nq; i++) h->qdd[i] = (h->qd_next[i] - qd[i]) / h->dt;

    /* Kinematic forward pass (RNE) at (q, qd, qdd_actual) to populate h->{f,a,v}
       with post-contact spatial dynamics. Unlike re-running ABA, this is purely
       kinematic — given the realized joint accel it computes the body's spatial
       accel directly, so the accelerometer-like feeds (cases 8/12) report what
       the body physically experiences rather than the no-contact ABA prediction.
       tau output is discarded (we reuse tau_p as a scratch sink). */
    rne_featherstone(h->nb, h->X, h->I6, h->parent, h->jtype, q, qd, h->qdd,
                     h->f_ext, h->g, h->tau_p, h->f, h->a, h->v, h->workspace);

    /* Stage 4: feedback. raw tau (pre ff/sk/PID) is what case 3 reads. */
    if (h->fb_set) tact_feedback(h, q, qd, tau);
}

/* ============================================================================
 * Phase 4: query functions — fk / error / jacob with the per-frame loop in C.
 * Caller (Python) resolves frames-dict string keys to integer frame indices
 * once via fdict, then passes flat int arrays. mode[k]: 0=3d, 1=6d.
 *
 * Quirks preserved for parity with the Python reference path:
 *   fk    : body_idx<0 (root frame) → Te = ftran[fi]   (no T multiplication)
 *   error : body_idx<0              → Te = T[nb+body_idx] @ ftran[fi]   (Python negative indexing)
 *   jacob : body_idx<0              → same as error, AND jacob_whitney walks from nb+body_idx
 * ============================================================================ */

void tact_fk_query(tact_t *h, double *q, int n, int *frame_idx, int *mode, const char *eulerseq, double *out)
{
    _fk(h->T, h->nb, h->Ti, h->parent, h->jtype, q);
    int oi = 0;
    for (int k = 0; k < n; k++) {
        int fi = frame_idx[k];
        int bi = h->fbody[fi];
        double Te[16];
        if (bi < 0) memcpy(Te, h->ftran + 16*fi, 16*sizeof(double));
        else        matmul(Te, h->T + 16*bi, h->ftran + 16*fi, 4, 4, 4);
        if (mode[k] == 0) { out[oi++] = Te[3]; out[oi++] = Te[7]; out[oi++] = Te[11]; }
        else              { homogeneous_to_xyzeuler(Te, out + oi, eulerseq); oi += 6; }
    }
}

void tact_error_query(tact_t *h, double *q, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double *out)
{
    _fk(h->T, h->nb, h->Ti, h->parent, h->jtype, q);
    int oi = 0;
    for (int k = 0; k < n; k++) {
        int fi = frame_idx[k];
        int bi = h->fbody[fi];
        if (bi < 0) bi = h->nb + bi;        /* match Python error()'s negative indexing */
        double Te[16];
        matmul(Te, h->T + 16*bi, h->ftran + 16*fi, 4, 4, 4);
        if (mode[k] == 0) {
            out[oi+0] = x_d[oi+0] - Te[3];
            out[oi+1] = x_d[oi+1] - Te[7];
            out[oi+2] = x_d[oi+2] - Te[11];
            oi += 3;
        } else {
            /* Python homogeneous_error layout = [e_translation(3), e_orientation(3)];
             * C homogeneous_error returns the opposite order, so build directly here. */
            double Td[16];
            xyzeuler_to_homogeneous(x_d + oi, Td, eulerseq);
            out[oi+0] = Td[3]  - Te[3];
            out[oi+1] = Td[7]  - Te[7];
            out[oi+2] = Td[11] - Te[11];
            double R1[9] = {Td[0],Td[1],Td[2], Td[4],Td[5],Td[6], Td[8],Td[9],Td[10]};
            double R2[9] = {Te[0],Te[1],Te[2], Te[4],Te[5],Te[6], Te[8],Te[9],Te[10]};
            rotation_error(R1, R2, out + oi + 3);
            oi += 6;
        }
    }
}

/* J_out shape: (total_rows, nb), row-major, where total_rows = sum(3 if mode==0 else 6).
 * caller (Python) computes total_rows and pre-allocates. */
void tact_jacob_query(tact_t *h, double *q, int n, int *frame_idx, int *mode, double *J_out)
{
    int nb = h->nb, nq = h->nq;
    _fk(h->T, nb, h->Ti, h->parent, h->jtype, q);
    double *J_temp = h->workspace;          /* 6*nq scratch (fits in workspace) */
    int row_off = 0;
    for (int k = 0; k < n; k++) {
        int fi = frame_idx[k];
        int bi = h->fbody[fi];
        int idx_c = (bi < 0) ? (nb + bi) : bi;
        double Te[16];
        matmul(Te, h->T + 16*idx_c, h->ftran + 16*fi, 4, 4, 4);
        jacob_whitney(J_temp, nb, h->T, Te, h->parent, h->jtype, idx_c);
        int rows = (mode[k] == 0) ? 3 : 6;
        memcpy(J_out + (size_t)row_off*nq, J_temp, (size_t)rows*nq*sizeof(double));
        row_off += rows;
    }
}

/* Centroidal linear Jacobian J_com (3 × nq) at q. Mass-weighted average of the
 * per-body world-linear Jacobian (top 3 rows of jacob_whitney at body i's CoM
 * point B_i = T[i] · T_trans(c_i)).
 *
 * Inputs:  m_in (nb)       — body masses
 *          c_in (3*nb r-m) — body-frame CoM offsets, row-major
 * Output:  J_out (3*nq r-m) — caller pre-allocated.
 *
 * The two mass/offset arrays are passed in by the caller because tact_t stores
 * only the 6×6 spatial inertia I6 (not raw (m_i, c_i)) — see tact.h note. */
void tact_com_jacob_query(tact_t *h, double *q, double *m_in, double *c_in, double *J_out)
{
    int nb = h->nb, nq = h->nq;
    _fk(h->T, nb, h->Ti, h->parent, h->jtype, q);
    double *J_temp = h->workspace;  /* 6*nq — same scratch jacob_query uses */

    memset(J_out, 0, (size_t)3*nq*sizeof(double));

    double mtot = 0.0;
    for (int i = 0; i < nb; i++) mtot += m_in[i];

    for (int i = 0; i < nb; i++) {
        /* B_i = T[i] · T_trans(c_i). Translation column shifts by R·c, rotation block
         * unchanged. T is row-major 4×4: indices 0..15 with row r col c at 4r+c. */
        double Bi[16];
        double *Ti = h->T + 16*i;
        memcpy(Bi, Ti, 16*sizeof(double));
        double cx = c_in[3*i+0], cy = c_in[3*i+1], cz = c_in[3*i+2];
        Bi[3]  += Ti[0]*cx + Ti[1]*cy + Ti[2 ]*cz;
        Bi[7]  += Ti[4]*cx + Ti[5]*cy + Ti[6 ]*cz;
        Bi[11] += Ti[8]*cx + Ti[9]*cy + Ti[10]*cz;

        jacob_whitney(J_temp, nb, h->T, Bi, h->parent, h->jtype, i);

        /* J_out += (m_i / mtot) · J_temp[0:3, :] */
        double w = m_in[i] / mtot;
        for (int row = 0; row < 3; row++) {
            const double *src = J_temp + (size_t)row * nq;
            double       *dst = J_out  + (size_t)row * nq;
            for (int col = 0; col < nq; col++) dst[col] += w * src[col];
        }
    }
}

/* CoM world position. r_out[0..2] = Σ m_i · (T_i · c_i)[:3] / Σ m_i.
 * Same (m_in, c_in) inputs as tact_com_jacob_query — see header for why. */
void tact_com_query(tact_t *h, double *q, double *m_in, double *c_in, double *r_out)
{
    int nb = h->nb;
    _fk(h->T, nb, h->Ti, h->parent, h->jtype, q);
    double mtot = 0.0;
    for (int i = 0; i < nb; i++) mtot += m_in[i];
    r_out[0] = r_out[1] = r_out[2] = 0.0;
    for (int i = 0; i < nb; i++) {
        double *Ti = h->T + 16*i;
        double cx = c_in[3*i+0], cy = c_in[3*i+1], cz = c_in[3*i+2];
        /* world point = Ti @ [cx, cy, cz, 1].
         * Ti is row-major 4×4; r_world[k] = Ti[4k]*cx + Ti[4k+1]*cy + Ti[4k+2]*cz + Ti[4k+3]. */
        double mi = m_in[i];
        r_out[0] += mi * (Ti[0]*cx + Ti[1]*cy + Ti[2 ]*cz + Ti[3 ]);
        r_out[1] += mi * (Ti[4]*cx + Ti[5]*cy + Ti[6 ]*cz + Ti[7 ]);
        r_out[2] += mi * (Ti[8]*cx + Ti[9]*cy + Ti[10]*cz + Ti[11]);
    }
    double inv = 1.0 / mtot;
    r_out[0] *= inv;
    r_out[1] *= inv;
    r_out[2] *= inv;
}

/* gravity-only inverse dynamics. Internally zeros qd/qdd/f_ext in workspace and
 * dispatches to rne_featherstone. Caller pre-allocates b_out (length nb).
 * If g_override is non-NULL it replaces h->g (used by walk2.py-style body-frame gravity). */
void tact_gravity_query(tact_t *h, double *q, double *g_override, double *b_out)
{
    int nb = h->nb, nq = h->nq;
    double *zero_qd   = h->workspace;          /* nq */
    double *zero_qdd  = zero_qd  + nq;         /* nq */
    double *zero_fext = zero_qdd + nq;         /* 6*nb */
    double *rne_ws    = zero_fext + 6*nb;      /* rne needs 36*nb + 6*nq — fits in workspace */
    memset(h->workspace, 0, (size_t)(2*nq + 6*nb)*sizeof(double));
    rne_featherstone(nb, h->X, h->I6, h->parent, h->jtype, q, zero_qd, zero_qdd, zero_fext, g_override ? g_override : h->g, b_out, h->f, h->a, h->v, rne_ws);
}

/* joint-space mass matrix at q (mirrors Model.inertia). Thin wrapper over
 * crb_featherstone; caller pre-allocates H_out (nq*nq, row-major). The Python
 * caller is responsible for the np.delete(self.fixed, axis=0/1) post-processing. */
void tact_inertia_query(tact_t *h, double *q, double *H_out)
{
    crb_featherstone(h->nb, h->X, h->I6, h->parent, h->jtype, q, H_out, h->workspace);
}

/* bias = C(q,qd)·qd + g(q) − Jᵀf_ext (mirrors Model.bias). Calls rne_featherstone
 * with qdd=0. Caller pre-allocates b_out (length nq). f_ext_in may be NULL → treat
 * as zero (matches Python `f_ext=None` default); else it is a 6*nb body-frame wrench. */
void tact_bias_query(tact_t *h, double *q, double *qd, double *f_ext_in, double *b_out)
{
    int nb = h->nb, nq = h->nq;
    double *zero_qdd  = h->workspace;          /* nq */
    double *zero_fext = zero_qdd + nq;         /* 6*nb (used only when f_ext_in==NULL) */
    double *rne_ws    = zero_fext + 6*nb;      /* rne needs 36*nb + 6*nq — fits in workspace */
    memset(h->workspace, 0, (size_t)(nq + 6*nb)*sizeof(double));
    double *fext = f_ext_in ? f_ext_in : zero_fext;
    rne_featherstone(nb, h->X, h->I6, h->parent, h->jtype, q, qd, zero_qdd, fext, h->g, b_out, h->f, h->a, h->v, rne_ws);
}

/* ============================================================================
 * Raycast — single ray vs all collision shapes. Caller (Python) handles the
 * env interface; here we dispatch per ctype to the matching ray_intersects_*.
 *
 * tact_raycast_query: one shot. Recomputes _fk(q), then walks shapes.
 * tact_raycast_batch: n rays from a sensor frame (directions generated in
 *   Python — sim.py Env._ray_grid) over the shared per-shape loop (single
 *   _fk for the whole batch).
 *
 * Mesh raycast transforms the ray into shape-local frame (rotation transpose +
 * translation diff) so we don't have to transform every vertex per ray.
 * ============================================================================ */
/* Per-frame precomputed shape pose cache. tact_raycast_batch fires n rays through a
 * single _fk(q) state, so each shape's world transform (body T @ ctran) is identical for
 * the whole batch — recomputing it per ray was pure waste (n matmuls per shape). We
 * hoist it: build this cache ONCE per frame (rc_build_cache), then every ray reads it
 * (raycast_cached). The cache is frame-local and read-only during the ray loop — the
 * frustum cull (rc_frustum_cull) compacts it once per frame, and the same property is the
 * seam a future BVH / OpenMP plug into (no per-ray writes to shared state). */
typedef struct {
    double p[3];      /* world position */
    double R[9];      /* world rotation, row-major (R[0..2] = first row) */
    double z[3];      /* shape local +z in world (cylinder/capsule axis) */
    int    type;      /* ctype */
    const double *sh; /* -> cshape + 3*i (box half-size / sphere r / cyl r,hh) */
    int    slot;      /* mesh slot for type 100, else -1 */
    double bsr;       /* bounding-sphere radius around p (broad phase); <0 = no bound / never skip */
} rc_shape;

/* Fill `cache` with the world pose of every raycast-on shape; returns the count.
 * Requires h->T populated by _fk(q). `cache` must hold at least h->n_shape entries. */
static int rc_build_cache(tact_t *h, rc_shape *cache)
{
    int n = 0;
    for (int i = 0; i < h->n_shape; i++) {
        if (h->craycast[i] == 0) continue;  /* shape opted out of raycast (YAML raycast: false) */
        double Tw[16];
        if (h->cbody[i] < 0) memcpy(Tw, h->ctran + 16*i, 16*sizeof(double));
        else matmul(Tw, h->T + 16*h->cbody[i], h->ctran + 16*i, 4, 4, 4);
        rc_shape *s = &cache[n++];
        s->p[0]=Tw[3];  s->p[1]=Tw[7];  s->p[2]=Tw[11];
        s->R[0]=Tw[0];  s->R[1]=Tw[1];  s->R[2]=Tw[2];
        s->R[3]=Tw[4];  s->R[4]=Tw[5];  s->R[5]=Tw[6];
        s->R[6]=Tw[8];  s->R[7]=Tw[9];  s->R[8]=Tw[10];
        s->z[0]=Tw[2];  s->z[1]=Tw[6];  s->z[2]=Tw[10];
        s->type = h->ctype[i];
        s->sh   = h->cshape + 3*i;
        s->slot = (s->type == 100 || s->type == 105) ? (int)h->cshape[3*i] : -1;  /* mesh / hfield slot */
        /* Bounding-sphere radius around p (= shape center for these primitives) for the
         * broad phase (frustum cull + ray-sphere reject). bsr < 0 means "no bound, never
         * skip" (only unloadable meshes / unknown types). */
        double s0 = s->sh[0], s1 = s->sh[1], s2 = s->sh[2];
        switch (s->type) {
        case 101: s->bsr = sqrt(s0*s0 + s1*s1 + s2*s2); break;  /* box: |half-extents| */
        case 102: s->bsr = s0;                          break;  /* sphere: r */
        case 103: s->bsr = sqrt(s1*s1 + s0*s0);         break;  /* cylinder: sqrt(hh^2+r^2) */
        case 104: s->bsr = s1 + s0;                     break;  /* capsule: hh + r */
        case 100: s->bsr = mesh_local_radius(s->slot);  break;  /* mesh: max|vertex| (-1 if unloadable → never cull) */
        case 105: s->bsr = hfield_local_radius(s->slot);break;  /* hfield: sqrt(sx²+sy²+max|h|²) (-1 if empty → never cull) */
        default:  s->bsr = -1.0;                        break;  /* unknown: never cull */
        }
    }
    return n;
}

/* Per-frame frustum cull: compact `cache` in place to the shapes whose bounding sphere
 * could be seen by the sensor, returning the kept count. The sensor's rays all lie within
 * the cone of half-angle `max_half_angle` about `fwd` (the diagonal corner angle, which
 * contains the rectangular frustum), so a sphere entirely outside that cone is unhittable
 * by any ray and safe to drop — the cull is conservative and the image is unchanged. All
 * rays then share this reduced list (N -> N_visible). `cache` is frame-local scratch, so
 * compacting it (vs. an index array) is fine and keeps the per-ray loop branch-free. */
static int rc_frustum_cull(rc_shape *cache, int n, const double *R0,
                           const double *fwd, double max_half_angle)
{
    int m = 0;
    for (int k = 0; k < n; k++) {
        rc_shape s = cache[k];
        if (s.bsr < 0.0) { cache[m++] = s; continue; }           /* never-cull (mesh) */
        double d[3] = {s.p[0]-R0[0], s.p[1]-R0[1], s.p[2]-R0[2]};
        double dist2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        if (dist2 <= s.bsr*s.bsr) { cache[m++] = s; continue; }  /* sensor inside sphere → keep */
        double dist = sqrt(dist2);
        double cosang = (d[0]*fwd[0] + d[1]*fwd[1] + d[2]*fwd[2]) / dist;
        if (cosang > 1.0) cosang = 1.0; else if (cosang < -1.0) cosang = -1.0;
        double angle_center = acos(cosang);
        double ang_radius   = asin(s.bsr / dist);
        if (angle_center - ang_radius <= max_half_angle) cache[m++] = s;  /* overlaps cone → keep */
    }
    return m;
}

/* One ray vs the precomputed shape cache: bounding-sphere broad test, then the matching
 * ray_intersects_* on the cached world pose. Returns nearest forward hit distance, or -1.0
 * on miss. All per-ray temporaries are stack-local (no writes to shared state), so this is
 * safe to call from many threads over one shared cache. */
static double raycast_cached(const rc_shape *cache, int n, const double *R0, const double *Rd)
{
    double best = -1.0;
    for (int k = 0; k < n; k++) {
        const rc_shape *s = &cache[k];
        const double *p = s->p, *R = s->R, *z = s->z;
        double t = -1.0;

        /* Bounding-sphere broad test before the (possibly expensive) primitive. The sphere
         * encloses the primitive, so missing it ⇒ missing the primitive (skip), and its
         * near entry t_near = tca - thc is a lower bound on any real hit ⇒ if that already
         * exceeds `best`, this shape can't win (skip). Both are exact-conservative, so the
         * image is unchanged. Biggest win for meshes (skips the triangle loop); cheap
         * primitives still pay only a few flops. Rd is unit (caller guarantees). bsr<0
         * (unloadable mesh / unknown) ⇒ no sphere, fall through and test directly. */
        if (s->bsr >= 0.0) {
            double L[3] = {p[0]-R0[0], p[1]-R0[1], p[2]-R0[2]};
            double tca = L[0]*Rd[0] + L[1]*Rd[1] + L[2]*Rd[2];
            double d2  = (L[0]*L[0] + L[1]*L[1] + L[2]*L[2]) - tca*tca;
            double r2  = s->bsr * s->bsr;
            if (d2 > r2) continue;                            /* ray misses sphere → misses primitive */
            double thc = sqrt(r2 - d2);
            if (tca + thc < 0.0) continue;                    /* sphere entirely behind ray origin */
            if (best >= 0.0 && tca - thc >= best) continue;   /* nearest possible hit already occluded */
        }

        switch (s->type) {
        case 100: {  /* MESH: ray into mesh-local frame: R0_loc = Rᵀ(R0-p), Rd_loc = Rᵀ Rd */
            double dp[3] = {R0[0]-p[0], R0[1]-p[1], R0[2]-p[2]};
            double R0l[3] = {R[0]*dp[0]+R[3]*dp[1]+R[6]*dp[2],
                             R[1]*dp[0]+R[4]*dp[1]+R[7]*dp[2],
                             R[2]*dp[0]+R[5]*dp[1]+R[8]*dp[2]};
            double Rdl[3] = {R[0]*Rd[0]+R[3]*Rd[1]+R[6]*Rd[2],
                             R[1]*Rd[0]+R[4]*Rd[1]+R[7]*Rd[2],
                             R[2]*Rd[0]+R[5]*Rd[1]+R[8]*Rd[2]};
            t = ray_intersects_mesh_slot(R0l, Rdl, s->slot);
            break;
        }
        case 101: {  /* BOX */
            double hs[3] = {s->sh[0], s->sh[1], s->sh[2]};
            t = ray_intersects_box(R0, Rd, p, R, hs);
            break;
        }
        case 102:    /* SPHERE */
            t = ray_intersects_sphere(R0, Rd, p, s->sh[0]);
            break;
        case 103: {  /* CYLINDER: sh = [r, hh, _]; endpoints = p ± hh·z */
            double r = s->sh[0], hh = s->sh[1];
            double P1[3] = {p[0]+hh*z[0], p[1]+hh*z[1], p[2]+hh*z[2]};
            double P2[3] = {p[0]-hh*z[0], p[1]-hh*z[1], p[2]-hh*z[2]};
            t = ray_intersects_cylinder(R0, Rd, P1, P2, r);
            break;
        }
        case 104: {  /* CAPSULE */
            double r = s->sh[0], hh = s->sh[1];
            double P1[3] = {p[0]+hh*z[0], p[1]+hh*z[1], p[2]+hh*z[2]};
            double P2[3] = {p[0]-hh*z[0], p[1]-hh*z[1], p[2]-hh*z[2]};
            t = ray_intersects_capsule(R0, Rd, P1, P2, r);
            break;
        }
        case 105: {  /* HFIELD: ray into hfield-local frame, same transform as mesh */
            double dp[3] = {R0[0]-p[0], R0[1]-p[1], R0[2]-p[2]};
            double R0l[3] = {R[0]*dp[0]+R[3]*dp[1]+R[6]*dp[2],
                             R[1]*dp[0]+R[4]*dp[1]+R[7]*dp[2],
                             R[2]*dp[0]+R[5]*dp[1]+R[8]*dp[2]};
            double Rdl[3] = {R[0]*Rd[0]+R[3]*Rd[1]+R[6]*Rd[2],
                             R[1]*Rd[0]+R[4]*Rd[1]+R[7]*Rd[2],
                             R[2]*Rd[0]+R[5]*Rd[1]+R[8]*Rd[2]};
            t = ray_intersects_hfield(R0l, Rdl, s->slot);
            break;
        }
        }
        if (t >= 0.0 && (best < 0.0 || t < best)) best = t;
    }
    return best;
}

double tact_raycast_query(tact_t *h, double *q, double *R0, double *Rd)
{
    _fk(h->T, h->nb, h->Ti, h->parent, h->jtype, q);
    rc_shape cache[h->n_shape > 0 ? h->n_shape : 1];
    int n = rc_build_cache(h, cache);
    return raycast_cached(cache, n, R0, Rd);
}

/* Batched raycast from a sensor frame. `dirs` = n unit ray directions in the
 * frame's REGISTERED coordinates (YAML pos/euler) — ray GENERATION lives in
 * Python (sim.py Env._ray_grid, the single source); this side only intersects.
 * The old tact_raymap_query (in-C angular/pinhole pixel loops + the -90deg
 * optical roll baked into the frame pose) was removed 2026-06-06: raycloud had
 * to bit-mirror that ray generation in Python to back-project ranges, a fragile
 * duplication. perpendicular mode also moved to Python (t *= -dir_z, one line).
 *
 * One _fk + shape-pose cache + frustum cull for the whole batch, then one
 * raycast_cached per ray. The cull cone is computed FROM the input rays:
 * axis = normalized mean direction, half-angle = max ray-to-axis angle
 * (+margin) — contains every input ray by construction, so it never culls a
 * hittable shape; a degenerate mean (rays spanning >~180deg) disables it.
 * t_out[k] = forward range along dirs[k]; -1 = no hit. */
void tact_raycast_batch(tact_t *h, double *q, int frame_idx, double *dirs, int n, double *t_out)
{
    _fk(h->T, h->nb, h->Ti, h->parent, h->jtype, q);
    /* Hoist per-shape world transforms out of the ray loop (single _fk -> poses
     * fixed for the whole batch). Frame-local, read-only during the loop. */
    rc_shape cache[h->n_shape > 0 ? h->n_shape : 1];
    int ncache = rc_build_cache(h, cache);

    int bi = h->fbody[frame_idx];
    double Twf[16];
    if (bi < 0) memcpy(Twf, h->ftran + 16*frame_idx, 16*sizeof(double));
    else matmul(Twf, h->T + 16*bi, h->ftran + 16*frame_idx, 4, 4, 4);

    double R0[3] = {Twf[3], Twf[7], Twf[11]};
    double Rc[9] = {Twf[0],Twf[1],Twf[2], Twf[4],Twf[5],Twf[6], Twf[8],Twf[9],Twf[10]};
    double PI = 3.14159265358979323846;

    if (n > 0) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int k = 0; k < n; k++) { ax += dirs[3*k]; ay += dirs[3*k+1]; az += dirs[3*k+2]; }
        double an = sqrt(ax*ax + ay*ay + az*az);
        if (an > 1e-9) {
            ax /= an; ay /= an; az /= an;
            double cmin = 1.0;
            for (int k = 0; k < n; k++) {
                double c = ax*dirs[3*k] + ay*dirs[3*k+1] + az*dirs[3*k+2];
                if (c < cmin) cmin = c;
            }
            if (cmin > 1.0) cmin = 1.0;
            if (cmin > -1.0 + 1e-9) {
                double max_half_angle = acos(cmin) + 1e-4;
                if (max_half_angle < PI) {
                    double fwd[3] = {   /* cone axis: registered frame -> world */
                        Rc[0]*ax + Rc[1]*ay + Rc[2]*az,
                        Rc[3]*ax + Rc[4]*ay + Rc[5]*az,
                        Rc[6]*ax + Rc[7]*ay + Rc[8]*az,
                    };
                    ncache = rc_frustum_cull(cache, ncache, R0, fwd, max_half_angle);
                }
            }
        }
    }

    for (int k = 0; k < n; k++) {
        double dx = dirs[3*k], dy = dirs[3*k+1], dz = dirs[3*k+2];
        double Rd[3] = {
            Rc[0]*dx + Rc[1]*dy + Rc[2]*dz,
            Rc[3]*dx + Rc[4]*dy + Rc[5]*dz,
            Rc[6]*dx + Rc[7]*dy + Rc[8]*dz,
        };
        t_out[k] = raycast_cached(cache, ncache, R0, Rd);
    }
}

/* ---------------------------------------------------------------------------
 * Cholesky helpers — small, dense, SPD m×m matrices (m ≤ ~6*MAX_NB in practice).
 * Used by tact_ik2_query to solve (J Jᵀ + λ²I) x = e_x. λ²>0 guarantees SPD.
 * In-place factorization writes L into the lower triangle of A.
 * --------------------------------------------------------------------------- */
static int chol_factor(double *A, int m) {
    for (int j = 0; j < m; j++) {
        double s = A[j*m + j];
        for (int k = 0; k < j; k++) s -= A[j*m + k] * A[j*m + k];
        if (s <= 0.0) return -1;
        double Ljj = sqrt(s);
        A[j*m + j] = Ljj;
        double inv = 1.0 / Ljj;
        for (int i = j+1; i < m; i++) {
            double t = A[i*m + j];
            for (int k = 0; k < j; k++) t -= A[i*m + k] * A[j*m + k];
            A[i*m + j] = t * inv;
        }
    }
    return 0;
}

static void chol_solve(double *A, int m, double *b, double *x, double *y) {
    /* L y = b */
    for (int i = 0; i < m; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*m + k] * y[k];
        y[i] = s / A[i*m + i];
    }
    /* Lᵀ x = y */
    for (int i = m-1; i >= 0; i--) {
        double s = y[i];
        for (int k = i+1; k < m; k++) s -= A[k*m + i] * x[k];
        x[i] = s / A[i*m + i];
    }
}

/* tact_ik2_query — Damped Least Squares IK, mirrors fixed sim.py:Model.ik2().
 * Algorithm (per iter): e=error(q); if |e|<tol return; J=jacob(q); A=JJᵀ+λ²I;
 * solve A x=e; q += advance · Jᵀ x.
 *
 * Fixed-joint columns of J are zeroed before forming A — algebraically identical
 * to Python's np.delete(J, self.fixed, axis=1) since zero columns contribute nothing
 * to JJᵀ and produce zero updates on those q rows.
 *
 * Workspace layout (in h->workspace, 120·nb doubles): q_full(nb) | dq(nb) | J(m·nb)
 *                                                   | A(m·m) | e(m) | bx(m) | yvec(m).
 * Caller responsibility: ensure m ≤ ~6·MAX_NB so the carve fits. */
int tact_ik2_query(tact_t *h, double *q_in, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double advance, double tolerance, double damping, int max_iter, double *q_out)
{
    int nb = h->nb, nq = h->nq;
    int m  = 0;
    for (int k = 0; k < n; k++) m += (mode[k] == 0) ? 3 : 6;

    /* per-DoF (nq) layout. q_full/dq are nq; J is m × nq; J_block scratch needs 6*nq */
    double *q_full = h->workspace;             /* nq */
    double *dq     = q_full + nq;              /* nq */
    double *J      = dq     + nq;              /* m * nq */
    double *J_blk  = J      + (size_t)m*nq;    /* 6 * nq  (jacob_whitney scratch) */
    double *A      = J_blk  + (size_t)6*nq;    /* m * m   */
    double *e_x    = A      + (size_t)m*m;     /* m */
    double *bx     = e_x    + m;               /* m */
    double *yvec   = bx     + m;               /* m */

    memcpy(q_full, q_in, nq*sizeof(double));
    double lam2 = damping * damping;

    int iter = 0;
    double e_norm = 0.0;
    while (iter < max_iter) {
        /* error: _fk + per-frame Te + 3d/6d residual (parity with tact_error_query) */
        _fk(h->T, nb, h->Ti, h->parent, h->jtype, q_full);
        int oi = 0;
        for (int k = 0; k < n; k++) {
            int fi = frame_idx[k];
            int bi = h->fbody[fi];
            if (bi < 0) bi = nb + bi;
            double Te[16];
            matmul(Te, h->T + 16*bi, h->ftran + 16*fi, 4, 4, 4);
            if (mode[k] == 0) {
                e_x[oi+0] = x_d[oi+0] - Te[3];
                e_x[oi+1] = x_d[oi+1] - Te[7];
                e_x[oi+2] = x_d[oi+2] - Te[11];
                oi += 3;
            } else {
                double Td[16];
                xyzeuler_to_homogeneous(x_d + oi, Td, eulerseq);
                e_x[oi+0] = Td[3]  - Te[3];
                e_x[oi+1] = Td[7]  - Te[7];
                e_x[oi+2] = Td[11] - Te[11];
                double R1[9] = {Td[0],Td[1],Td[2], Td[4],Td[5],Td[6], Td[8],Td[9],Td[10]};
                double R2[9] = {Te[0],Te[1],Te[2], Te[4],Te[5],Te[6], Te[8],Te[9],Te[10]};
                rotation_error(R1, R2, e_x + oi + 3);
                oi += 6;
            }
        }
        e_norm = 0.0;
        for (int i = 0; i < m; i++) e_norm += e_x[i]*e_x[i];
        e_norm = sqrt(e_norm);
        if (e_norm < tolerance) {
            memcpy(q_out, q_full, nq*sizeof(double));
            return iter;
        }

        /* jacob: stack jacob_whitney blocks into J(m, nq) row-major */
        int row_off = 0;
        for (int k = 0; k < n; k++) {
            int fi = frame_idx[k];
            int bi = h->fbody[fi];
            int idx_c = (bi < 0) ? (nb + bi) : bi;
            double Te[16];
            matmul(Te, h->T + 16*idx_c, h->ftran + 16*fi, 4, 4, 4);
            jacob_whitney(J_blk, nb, h->T, Te, h->parent, h->jtype, idx_c);
            int rows = (mode[k] == 0) ? 3 : 6;
            memcpy(J + (size_t)row_off*nq, J_blk, (size_t)rows*nq*sizeof(double));
            row_off += rows;
        }
        /* Fixed joints contribute zero columns to J because nv_per_body=0 — no
         * explicit zeroing needed. */

        /* A = J Jᵀ + λ²I  (m × m, symmetric SPD) */
        for (int i = 0; i < m; i++) {
            double *Ji = J + (size_t)i*nq;
            for (int j = 0; j <= i; j++) {
                double *Jj = J + (size_t)j*nq;
                double s = 0.0;
                for (int k = 0; k < nq; k++) s += Ji[k] * Jj[k];
                if (i == j) s += lam2;
                A[i*m + j] = s;
                A[j*m + i] = s;
            }
        }

        /* Cholesky-solve A x = e_x → bx */
        if (chol_factor(A, m) != 0) {
            memcpy(q_out, q_full, nq*sizeof(double));
            return -(iter + 1);
        }
        chol_solve(A, m, e_x, bx, yvec);

        /* dq = Jᵀ bx  (length nq; fixed cols of J are zero → dq[fixed]=0) */
        for (int i = 0; i < nq; i++) {
            double s = 0.0;
            for (int k = 0; k < m; k++) s += J[(size_t)k*nq + i] * bx[k];
            dq[i] = s;
        }
        /* IK update: free-joint slots (jtype=3) need SO(3)-aware step on the
         * rotation block; rest is linear q += advance·dq. We integrate as if
         * advance·dq were a velocity over unit time. */
        q_step(nb, h->jtype, q_full, dq, advance, q_full);

        iter++;
    }

    memcpy(q_out, q_full, nq*sizeof(double));
    return -max_iter;
}

/* arena read-only accessors */
double *tact_get_f       (tact_t *h) { return h->f; }
double *tact_get_a       (tact_t *h) { return h->a; }
double *tact_get_v       (tact_t *h) { return h->v; }
double *tact_get_q_next  (tact_t *h) { return h->q_next; }
double *tact_get_qd_next (tact_t *h) { return h->qd_next; }
double *tact_get_y       (tact_t *h) { return h->y_buf; }
