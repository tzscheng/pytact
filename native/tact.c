/* tact.c — high-level handle API.
 *
 * Owns tact_t (per-instance state object), orchestrates the collision side
 * (narrow.c / mpr.c / ray.c / shape.c) and rbd.c (dynamics) + lcp.c (contact)
 * into a single ctypes-friendly entry surface for the
 * Python package (sim.py drives it). Build / runtime split + lifecycle
 * invariants documented in docs/design-c-state.md §3.
 *
 * Public surface declared in tact.h: the tact_t head exposes read-only
 * dimensions and step-output views; everything else lives behind the private
 * tact_core_t (core.h), allocated together with the head here. */
#include "core.h"


int tact_create_from_arrays(int nb, int *parent, int *jtype, double *X, double *I6, double *Ti, double *ff, double *sk, double *floss, double *armature, double *jnt_lo, double *jnt_hi, double *g, double dt, int integrator, int n_shape, int n_pair, int *ctype, int *cbody, double *cshape, double *ctran, double *cparam, int *craycast, int *cpair, double erp, double slop, double cfm_scale, double v_rest_thresh, int iters, double tol, tact_t **out)
{
    if (!out) return -1;
    *out = NULL;
    /* count-gated NULL checks: a body-less scene (nb=0, e.g. world-attached
     * terrain only) is valid and passes NULL body/DoF arrays */
    if (nb < 0 || n_shape < 0 || n_pair < 0 || !g) return -2;
    if (nb > 0 && (!parent || !jtype || !X || !I6 || !Ti)) return -2;
    if (n_shape > 0 && (!ctype || !cbody || !cshape || !ctran || !cparam)) return -3;
    if (n_pair > 0 && !cpair) return -4;

    int npair_max = n_pair > 0 ? n_pair : 1;
    /* Per-body indexing. nq_per_body[i] = q slots, nv_per_body[i] = velocity DoFs.
     * Under axis-angle: 6 for jtype=3 (free), 0 for jtype=0 (fixed — no state),
     * else 1. d_total = sum(nv²) sizes the d-block array in aba/crb/rne workspace. */
    int nq = 0, d_total = 0;
    for (int i = 0; i < nb; ++i) {
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        int nqi = nvi;     /* axis-angle: nq_per_body == nv_per_body */
        nq       += nqi;
        d_total  += nvi * nvi;
    }
    if (nq > 0 && (!ff || !sk)) return -2;

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
      + nq                                  /* qd_free_buf (LCP predictor, per-DoF) */
      + nq*nq                               /* M_buf (joint-space mass matrix) */
      + 10*Pm_max                           /* contact_d reports: p,n,f,depth */
      + 2*(6*Pm_max + 2*nq)                 /* ctx_next + ctx_prev (= ctx_size warm-start λ each) */
      + lcp_ws_doubles                      /* lcp_ws (contact_lcp workspace) */
    );
    /* ints: parent, jtype, q_base, v_base, nq_per_body, nv_per_body (6*nb)
     *     + ctype, cbody, craycast (3*n_shape) + cpair (2*n_pair)
     *     + contact_i reports (4*Pm_max) */
    size_t bytes_int = sizeof(int) * (6*nb + 3*n_shape + 2*n_pair + 4*Pm_max);

    /* public head + private core in one allocation; h->core points past the head */
    tact_t *h = (tact_t*)calloc(1, sizeof(tact_t) + sizeof(tact_core_t));
    if (!h) return -5;
    tact_core_t *c = h->core = (tact_core_t*)(h + 1);
    c->arena = malloc(bytes_dbl + bytes_int);
    if (!c->arena) {
        tact_destroy(h);
        return -6;
    }

    (void)integrator;
    h->nb = nb; h->nq = nq;
    h->n_shape = n_shape; h->n_pair = n_pair;
    h->contact_count = 0;
    h->dt = dt;
    c->erp = erp; c->slop = slop; c->cfm_scale = cfm_scale;
    c->v_rest_thresh = v_rest_thresh; c->iters = iters; c->tol = tol;
    h->ctx_size = (int)(6 * MAX_PTS_PER_PAIR * (n_pair > 0 ? n_pair : 1) + 2 * nq);
    c->g[0] = g[0]; c->g[1] = g[1]; c->g[2] = g[2];

    char *p = (char*)c->arena;
    #define CARVE_DBL(dst, n) do { dst = (double*)p; p += (size_t)(n)*sizeof(double); } while (0)
    CARVE_DBL(c->X,        36*nb);
    CARVE_DBL(c->I6,       36*nb);
    CARVE_DBL(c->Ti,       16*nb);
    CARVE_DBL(c->ff,       nq);
    CARVE_DBL(c->sk,       nq);
    CARVE_DBL(c->floss,    nq);
    CARVE_DBL(c->armature, nq);
    CARVE_DBL(c->jnt_lo,   nq);
    CARVE_DBL(c->jnt_hi,   nq);
    CARVE_DBL(c->cshape,   3*n_shape);
    CARVE_DBL(c->ctran,    16*n_shape);
    CARVE_DBL(c->cparam,   13*n_shape);
    CARVE_DBL(c->T,        16*nb);
    CARVE_DBL(c->f_ext,    6*nb);
    CARVE_DBL(h->f,        6*nb);
    CARVE_DBL(h->a,        6*nb);
    CARVE_DBL(h->v,        6*nb);
    CARVE_DBL(c->qdd,      nq);
    CARVE_DBL(h->q_next,   nq);
    CARVE_DBL(h->qd_next,  nq);
    CARVE_DBL(c->tau_p,    nq);
    CARVE_DBL(c->workspace, aba_ws_size);
    CARVE_DBL(c->qd_free_buf, nq);
    CARVE_DBL(c->M_buf,       nq*nq);
    CARVE_DBL(c->contact_d,   10*Pm_max);
    CARVE_DBL(h->ctx_next,    h->ctx_size);
    CARVE_DBL(c->ctx_prev,    h->ctx_size);
    CARVE_DBL(c->lcp_ws,      lcp_ws_doubles);
    #undef CARVE_DBL
    #define CARVE_INT(dst, n) do { dst = (int*)p; p += (size_t)(n)*sizeof(int); } while (0)
    CARVE_INT(c->parent,       nb);
    CARVE_INT(c->jtype,        nb);
    CARVE_INT(c->q_base,       nb);
    CARVE_INT(c->v_base,       nb);
    CARVE_INT(c->nq_per_body,  nb);
    CARVE_INT(c->nv_per_body,  nb);
    CARVE_INT(c->ctype,        n_shape);
    CARVE_INT(c->cbody,        n_shape);
    CARVE_INT(c->craycast,     n_shape);
    CARVE_INT(c->cpair,        2*n_pair);
    CARVE_INT(c->contact_i,    4*Pm_max);
    #undef CARVE_INT

    /* copy static data */
    if (nb > 0) {
        memcpy(c->parent, parent, nb*sizeof(int));
        memcpy(c->jtype,  jtype,  nb*sizeof(int));
        memcpy(c->X,      X,      36*nb*sizeof(double));
        memcpy(c->I6,     I6,     36*nb*sizeof(double));
        memcpy(c->Ti,     Ti,     16*nb*sizeof(double));
    }
    /* fill q_base / v_base / nq_per_body / nv_per_body from jtype */
    int q_offset = 0, v_offset = 0;
    for (int i = 0; i < nb; ++i) {
        int nvi = (jtype[i] == 3) ? 6 : (jtype[i] == 0 ? 0 : 1);
        int nqi = nvi;     /* axis-angle convention */
        c->q_base[i]      = q_offset;
        c->v_base[i]      = v_offset;
        c->nq_per_body[i] = nqi;
        c->nv_per_body[i] = nvi;
        q_offset         += nqi;
        v_offset         += nvi;
    }
    if (nq > 0) {
        memcpy(c->ff, ff, nq*sizeof(double));
        memcpy(c->sk, sk, nq*sizeof(double));
    }
    if (floss) memcpy(c->floss, floss, nq*sizeof(double));
    else       memset(c->floss, 0,     nq*sizeof(double));
    if (armature) memcpy(c->armature, armature, nq*sizeof(double));
    else          memset(c->armature, 0,        nq*sizeof(double));
    if (jnt_lo) memcpy(c->jnt_lo, jnt_lo, nq*sizeof(double)); else memset(c->jnt_lo, 0, nq*sizeof(double));
    if (jnt_hi) memcpy(c->jnt_hi, jnt_hi, nq*sizeof(double)); else memset(c->jnt_hi, 0, nq*sizeof(double));
    if (n_shape > 0) {
        memcpy(c->ctype,  ctype,  n_shape*sizeof(int));
        memcpy(c->cbody,  cbody,  n_shape*sizeof(int));
        memcpy(c->cshape, cshape, 3*n_shape*sizeof(double));
        memcpy(c->ctran,  ctran,  16*n_shape*sizeof(double));
        memcpy(c->cparam, cparam, 13*n_shape*sizeof(double));
        if (craycast) memcpy(c->craycast, craycast, n_shape*sizeof(int));
        else for (int i = 0; i < n_shape; i++) c->craycast[i] = 1;  /* default: all visible to raycast */
    }
    if (n_pair > 0) memcpy(c->cpair, cpair, 2*n_pair*sizeof(int));

    *out = h;
    return 0;
}

void tact_destroy(tact_t *h)
{
    if (!h) return;
    tact_core_t *c = h->core;
    free(c->arena);
    if (c->fb_arena) free(c->fb_arena);
    free((void*)h->q0);
    free((void*)h->qd0);
    free(c->crgba);
    free(c->view);
    free(c->light0);
    free(c->frame_names);
    free(h);
}

void tact_contact_reports(tact_t *h, int *contact_i_out, double *contact_d_out)
{
    if (!h) return;
    tact_core_t *c = h->core;
    int n = h->contact_count;
    if (contact_i_out && n > 0) memcpy(contact_i_out, c->contact_i, 4*n*sizeof(int));
    if (contact_d_out && n > 0) memcpy(contact_d_out, c->contact_d, 10*n*sizeof(double));
}

/* Phase 2: pre-marshal feedback descriptors into the handle. Allocates a separate
 * arena (c->fb_arena) so it can be re-set without disturbing the dynamics arena.
 * After this is called, tact_step_lcp() will fill h->y at the end of each step. */
void tact_set_feedback(tact_t *h, int n_feeds, int *kinds, int *offsets, int *idx, int n_frames, int *fbody, double *ftran, double *ftran_inv, int y_size)
{
    tact_core_t *c = h->core;
    if (c->fb_arena) { free(c->fb_arena); c->fb_arena = NULL; }

    int n_idx = (n_feeds > 0) ? offsets[n_feeds] : 0;
    int y_alloc = y_size > 0 ? y_size : 1;   /* always alloc ≥1 to keep ptr non-null */
    int frame_alloc = n_frames > 0 ? n_frames : 1;

    size_t bytes_dbl = sizeof(double) * (16*frame_alloc + 16*frame_alloc + y_alloc);
    size_t bytes_int = sizeof(int) * (
        (n_feeds > 0 ? n_feeds : 1)         /* feed_kinds */
      + (n_feeds + 1)                        /* feed_offsets */
      + (n_idx > 0 ? n_idx : 1)              /* feed_idx */
      + frame_alloc);                        /* fbody */
    c->fb_arena = malloc(bytes_dbl + bytes_int);

    char *p = (char*)c->fb_arena;
    c->ftran     = (double*)p; p += 16*frame_alloc*sizeof(double);
    c->ftran_inv = (double*)p; p += 16*frame_alloc*sizeof(double);
    h->y     = (double*)p; p += y_alloc*sizeof(double);
    c->feed_kinds   = (int*)p; p += (n_feeds > 0 ? n_feeds : 1)*sizeof(int);
    c->feed_offsets = (int*)p; p += (n_feeds + 1)*sizeof(int);
    c->feed_idx     = (int*)p; p += (n_idx > 0 ? n_idx : 1)*sizeof(int);
    c->fbody        = (int*)p;

    if (n_frames > 0) {
        memcpy(c->fbody,     fbody,     n_frames*sizeof(int));
        memcpy(c->ftran,     ftran,     16*n_frames*sizeof(double));
        memcpy(c->ftran_inv, ftran_inv, 16*n_frames*sizeof(double));
    }
    if (n_feeds > 0) {
        memcpy(c->feed_kinds,   kinds,   n_feeds*sizeof(int));
        memcpy(c->feed_offsets, offsets, (n_feeds + 1)*sizeof(int));
        if (n_idx > 0) memcpy(c->feed_idx, idx, n_idx*sizeof(int));
    }
    c->n_feeds  = n_feeds;
    h->n_frames = n_frames;
    h->y_size   = y_size;
    c->fb_set   = 1;
}

/* In-place update of inertia buffers (X, I6, Ti). Topology (nb) must be unchanged.
 * Arena is preserved, so any numpy views from tact_* remain valid (cf. §3.5).
 * For topology changes, the caller must destroy + create a fresh handle. */
void tact_edit_model(tact_t *h, double *X, double *I6, double *Ti)
{
    tact_core_t *c = h->core;
    memcpy(c->X,  X,  36*h->nb*sizeof(double));
    memcpy(c->I6, I6, 36*h->nb*sizeof(double));
    memcpy(c->Ti, Ti, 16*h->nb*sizeof(double));
}

/* Phase 2: feedback in C — mirrors sim.py:Model.feedback() 14 cases.
 * Reads from h->{T,v,a,f,f_ext}, q, qd, tau (raw actuation, for case 3).
 * Writes to h->y (length h->y_size). */
static void tact_feedback(tact_t *h, double *q, double *qd, double *tau)
{
    tact_core_t *c = h->core;
    double *y = h->y;
    int yi = 0;

    for (int fi = 0; fi < c->n_feeds; fi++) {
        int kind  = c->feed_kinds[fi];
        int start = c->feed_offsets[fi];
        int end   = c->feed_offsets[fi + 1];

        for (int k = start; k < end; k++) {
            int frame_idx = c->feed_idx[k];
            int body_idx  = c->fbody[frame_idx];
            double *Tb    = c->T     + 16*body_idx;   /* 4×4 row-major */
            double *vb    = h->v     + 6 *body_idx;   /* [w(3); v0(3)] body frame */
            double *ab    = h->a     + 6 *body_idx;
            double *fb    = h->f     + 6 *body_idx;
            /* f_ext (external force) was previously used by case 14 — kept the
             * pointer commented for reference. Replaced by `fb` (propagated
             * wrench) so the FT readout is mass-correct, not just for massless
             * bodies. */
            /* double *fe = c->f_ext + 6 *body_idx; */
            double *Tf    = c->ftran + 16*frame_idx;
            double *Tfi   = c->ftran_inv + 16*frame_idx;

            switch (kind) {
            /* Cases 1/2/3 are only meaningful for 1-DoF joints. q uses q_base
             * (position state), qd/tau use v_base (velocity state). For 1-DoF
             * q_base[i] == v_base[i] always. Fixed bodies have nv_per_body=0
             * so reading q[q_base] would alias the next body's slot — emit 0
             * instead (callers shouldn't request joint state from a fixed body
             * but YAML 'jointpos: fixed_body' shouldn't crash either). */
            case 1: y[yi++] = c->nq_per_body[body_idx] ? q   [c->q_base[body_idx]] : 0.0; break;
            case 2: y[yi++] = c->nv_per_body[body_idx] ? qd  [c->v_base[body_idx]] : 0.0; break;
            case 3: y[yi++] = c->nv_per_body[body_idx] ? tau [c->v_base[body_idx]] : 0.0; break;
		
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
                    y[yi++] = Tb[0]*s[0] + Tb[1]*s[1] + Tb[2]*s[2]  + c->g[0];
                    y[yi++] = Tb[4]*s[0] + Tb[5]*s[1] + Tb[6]*s[2]  + c->g[1];
                    y[yi++] = Tb[8]*s[0] + Tb[9]*s[1] + Tb[10]*s[2] + c->g[2];
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
   qdd, q_next, qd_next, M_buf, qd_free_buf, lcp_ws (transient), y}.

   ctx: caller-owned solver context (read-only input; NULL = cold start, zero λ).
   Its payload is the PGS warm-start λ vector — ONE vector holding every row
   type, blocks in row-table order (the SolverState layout, the C↔Python ABI):
       [contact (6·MAX_PTS_PER_PAIR·max(n_pair,1), slot-indexed)
        | joint-friction (nq) | joint-limit (nq)]
   The next warm-start is written to the engine-owned h->ctx_next (same idiom
   as q_next/qd_next). Three ways to drive it:
     - pure threading: copy h->ctx_next into your own ctx buffer each step and
       pass that (caller buffers are never mutated);
     - stateful convenience: pass h->ctx_next itself — "continue from my own
       last ctx". The engine stages it into an internal buffer first, so the
       solve never reads and writes the same memory;
     - NULL: cold start (zero λ).
   A future constraint-row type appends a block here instead of growing this
   signature. (The separate ctx_out arg was folded into h->ctx_next when the
   public tact_t head was introduced.) */
int tact_step_lcp(tact_t *h, double *q, double *qd, double *tau, double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref, double *ctx)
{
    if (!h || !q || !qd || !tau) return -1;
    tact_core_t *c = h->core;
    double *lam_in = ctx;                /* NULL = cold start */
    double *lam_out = h->ctx_next;
    if (ctx == h->ctx_next) {
        /* internal warm-start mode: stage last output so in/out never alias */
        memcpy(c->ctx_prev, h->ctx_next, (size_t)h->ctx_size * sizeof(double));
        lam_in = c->ctx_prev;
    }

    /* Stage 1: forward kinematics */
    _fk(c->T, h->nb, c->Ti, c->parent, c->jtype, q);

    /* free predictor: aba_featherstone(f_ext=0). Output qdd is no-contact joint
       accel. f/a/v are overwritten by RNE below (post-contact spatial dynamics).
       Kp_j/Kd_j/q_ref/qd_ref pass through to aba's implicit PD path
       (NULLs = inactive). full=0 — we only need qdd here. */
    memset(c->f_ext, 0, 6*h->nb*sizeof(double));
    aba_featherstone(h->nb, c->X, c->I6, c->parent, c->jtype, q, qd, tau,
                     c->f_ext, c->g, c->qdd, h->f, h->a, h->v, c->workspace,
                     c->ff, c->sk, c->armature, h->dt, Kp_j, Kd_j, q_ref, qd_ref, /*full=*/0);
    for (int i = 0; i < h->nq; i++) c->qd_free_buf[i] = qd[i] + c->qdd[i] * h->dt;

    /* joint-space mass matrix at q. Add armature to the diagonal (rotor/reflected
       inertia, MuJoCo-style) so the contact solve sees the same effective inertia as
       the ABA predictor above. armature=0 → no-op (bit-identical). */
    crb_featherstone(h->nb, c->X, c->I6, c->parent, c->jtype, q, c->M_buf, c->workspace);
    for (int i = 0; i < h->nq; i++) c->M_buf[(size_t)i*h->nq + i] += c->armature[i];

    /* LCP solve: writes dqd into c->qdd (reused as scratch), fills c->f_ext with
       contact wrench, and writes the next warm-start λ into lam_out. */
    int nc_out = 0, iters_out = 0;
    double residual_out = 0.0;
    /* Slice the unified λ vector into contact_lcp's per-type pointers (offset
       arithmetic mirrors SolverState / Model.step). Contact block: contact_lcp
       reads lin and seeds/writes lout itself. Per-DoF blocks (fric, limit —
       adjacent in the layout): contact_lcp updates them in place, so seed lam_out
       from lam_in here; lam_in stays untouched (caller's ctx immutable). */
    int C = 6 * MAX_PTS_PER_PAIR * (h->n_pair > 0 ? h->n_pair : 1);
    double *lin   = lam_in;
    double *lout  = lam_out;
    double *lfout = lam_out + C;            /* joint-friction block */
    double *llout = lam_out + C + h->nq;    /* joint-limit block */
    if (lam_in) memcpy(lfout, lam_in + C, 2 * h->nq * sizeof(double));
    else        memset(lfout, 0,          2 * h->nq * sizeof(double));
    contact_lcp(h->nb, c->T, c->parent, c->jtype,
                h->n_pair, c->cpair, c->ctype, c->cbody,
                c->ctran, c->cshape, c->cparam,
                c->qd_free_buf, c->M_buf, h->dt,
                c->erp, c->slop, c->cfm_scale, c->v_rest_thresh,
                c->iters, c->tol,
                lin,                        /* in: previous λ (warm-start) */
                c->floss, lfout,            /* joint Coulomb friction + its warm-start (in-place on lfout) */
                q, c->jnt_lo, c->jnt_hi, llout,  /* joint limits (q for activation) + warm-start (in-place on llout) */
                c->qdd,                     /* out: dqd (velocity correction) */
                lout,                       /* out: λ_full (next warm-start) */
                c->f_ext,                   /* out: per-body contact wrench */
                &nc_out, &iters_out, &residual_out,
                &h->contact_count, c->contact_i, c->contact_d,
                c->lcp_ws);

    /* semi-implicit Euler: qd_next = qd_free + dqd; q_next = q_step(q, qd_next, dt).
       q_step handles SO(3) integration for free bodies (translation R·v_body·dt
       and rotation via exp map). For non-free it reduces to q + qd_next·dt.
       qdd is overwritten with the realized effective acceleration for feedback. */
    for (int i = 0; i < h->nq; i++) h->qd_next[i] = c->qd_free_buf[i] + c->qdd[i];
    q_step(h->nb, c->jtype, q, h->qd_next, h->dt, h->q_next);
    for (int i = 0; i < h->nq; i++) c->qdd[i] = (h->qd_next[i] - qd[i]) / h->dt;

    /* Kinematic forward pass (RNE) at (q, qd, qdd_actual) to populate h->{f,a,v}
       with post-contact spatial dynamics. Unlike re-running ABA, this is purely
       kinematic — given the realized joint accel it computes the body's spatial
       accel directly, so the accelerometer-like feeds (cases 8/12) report what
       the body physically experiences rather than the no-contact ABA prediction.
       tau output is discarded (we reuse tau_p as a scratch sink). */
    rne_featherstone(h->nb, c->X, c->I6, c->parent, c->jtype, q, qd, c->qdd,
                     c->f_ext, c->g, c->tau_p, h->f, h->a, h->v, c->workspace);

    /* Stage 4: feedback. raw tau (pre ff/sk/PID) is what case 3 reads. */
    if (c->fb_set) tact_feedback(h, q, qd, tau);
    return 0;
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

void tact_fk(tact_t *h, double *q, int n, int *frame_idx, int *mode, const char *eulerseq, double *out)
{
    tact_core_t *c = h->core;
    _fk(c->T, h->nb, c->Ti, c->parent, c->jtype, q);
    int oi = 0;
    for (int k = 0; k < n; k++) {
        int fi = frame_idx[k];
        int bi = c->fbody[fi];
        double Te[16];
        if (bi < 0) memcpy(Te, c->ftran + 16*fi, 16*sizeof(double));
        else        matmul(Te, c->T + 16*bi, c->ftran + 16*fi, 4, 4, 4);
        if (mode[k] == 0) { out[oi++] = Te[3]; out[oi++] = Te[7]; out[oi++] = Te[11]; }
        else              { homogeneous_to_xyzeuler(Te, out + oi, eulerseq); oi += 6; }
    }
}

void tact_error(tact_t *h, double *q, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double *out)
{
    tact_core_t *c = h->core;
    _fk(c->T, h->nb, c->Ti, c->parent, c->jtype, q);
    int oi = 0;
    for (int k = 0; k < n; k++) {
        int fi = frame_idx[k];
        int bi = c->fbody[fi];
        if (bi < 0) bi = h->nb + bi;        /* match Python error()'s negative indexing */
        double Te[16];
        matmul(Te, c->T + 16*bi, c->ftran + 16*fi, 4, 4, 4);
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
void tact_jacob(tact_t *h, double *q, int n, int *frame_idx, int *mode, double *J_out)
{
    tact_core_t *c = h->core;
    int nb = h->nb, nq = h->nq;
    _fk(c->T, nb, c->Ti, c->parent, c->jtype, q);
    double *J_temp = c->workspace;          /* 6*nq scratch (fits in workspace) */
    int row_off = 0;
    for (int k = 0; k < n; k++) {
        int fi = frame_idx[k];
        int bi = c->fbody[fi];
        int idx_c = (bi < 0) ? (nb + bi) : bi;
        double Te[16];
        matmul(Te, c->T + 16*idx_c, c->ftran + 16*fi, 4, 4, 4);
        jacob_whitney(J_temp, nb, c->T, Te, c->parent, c->jtype, idx_c);
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
void tact_com_jacob(tact_t *h, double *q, double *m_in, double *c_in, double *J_out)
{
    tact_core_t *c = h->core;
    int nb = h->nb, nq = h->nq;
    _fk(c->T, nb, c->Ti, c->parent, c->jtype, q);
    double *J_temp = c->workspace;  /* 6*nq — same scratch jacob_query uses */

    memset(J_out, 0, (size_t)3*nq*sizeof(double));

    double mtot = 0.0;
    for (int i = 0; i < nb; i++) mtot += m_in[i];

    for (int i = 0; i < nb; i++) {
        /* B_i = T[i] · T_trans(c_i). Translation column shifts by R·c, rotation block
         * unchanged. T is row-major 4×4: indices 0..15 with row r col c at 4r+c. */
        double Bi[16];
        double *Ti = c->T + 16*i;
        memcpy(Bi, Ti, 16*sizeof(double));
        double cx = c_in[3*i+0], cy = c_in[3*i+1], cz = c_in[3*i+2];
        Bi[3]  += Ti[0]*cx + Ti[1]*cy + Ti[2 ]*cz;
        Bi[7]  += Ti[4]*cx + Ti[5]*cy + Ti[6 ]*cz;
        Bi[11] += Ti[8]*cx + Ti[9]*cy + Ti[10]*cz;

        jacob_whitney(J_temp, nb, c->T, Bi, c->parent, c->jtype, i);

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
 * Same (m_in, c_in) inputs as tact_com_jacob — see header for why. */
void tact_com(tact_t *h, double *q, double *m_in, double *c_in, double *r_out)
{
    tact_core_t *c = h->core;
    int nb = h->nb;
    _fk(c->T, nb, c->Ti, c->parent, c->jtype, q);
    double mtot = 0.0;
    for (int i = 0; i < nb; i++) mtot += m_in[i];
    r_out[0] = r_out[1] = r_out[2] = 0.0;
    for (int i = 0; i < nb; i++) {
        double *Ti = c->T + 16*i;
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
 * If g_override is non-NULL it replaces c->g (used by walk2.py-style body-frame gravity). */
void tact_gravity(tact_t *h, double *q, double *g_override, double *b_out)
{
    tact_core_t *c = h->core;
    int nb = h->nb, nq = h->nq;
    double *zero_qd   = c->workspace;          /* nq */
    double *zero_qdd  = zero_qd  + nq;         /* nq */
    double *zero_fext = zero_qdd + nq;         /* 6*nb */
    double *rne_ws    = zero_fext + 6*nb;      /* rne needs 36*nb + 6*nq — fits in workspace */
    memset(c->workspace, 0, (size_t)(2*nq + 6*nb)*sizeof(double));
    rne_featherstone(nb, c->X, c->I6, c->parent, c->jtype, q, zero_qd, zero_qdd, zero_fext, g_override ? g_override : c->g, b_out, h->f, h->a, h->v, rne_ws);
}

/* joint-space mass matrix at q (mirrors Model.inertia). Thin wrapper over
 * crb_featherstone; caller pre-allocates H_out (nq*nq, row-major). The Python
 * caller is responsible for the np.delete(self.fixed, axis=0/1) post-processing. */
void tact_inertia(tact_t *h, double *q, double *H_out)
{
    tact_core_t *c = h->core;
    crb_featherstone(h->nb, c->X, c->I6, c->parent, c->jtype, q, H_out, c->workspace);
}

/* bias = C(q,qd)·qd + g(q) − Jᵀf_ext (mirrors Model.bias). Calls rne_featherstone
 * with qdd=0. Caller pre-allocates b_out (length nq). f_ext_in may be NULL → treat
 * as zero (matches Python `f_ext=None` default); else it is a 6*nb body-frame wrench. */
void tact_bias(tact_t *h, double *q, double *qd, double *f_ext_in, double *b_out)
{
    tact_core_t *c = h->core;
    int nb = h->nb, nq = h->nq;
    double *zero_qdd  = c->workspace;          /* nq */
    double *zero_fext = zero_qdd + nq;         /* 6*nb (used only when f_ext_in==NULL) */
    double *rne_ws    = zero_fext + 6*nb;      /* rne needs 36*nb + 6*nq — fits in workspace */
    memset(c->workspace, 0, (size_t)(nq + 6*nb)*sizeof(double));
    double *fext = f_ext_in ? f_ext_in : zero_fext;
    rne_featherstone(nb, c->X, c->I6, c->parent, c->jtype, q, qd, zero_qdd, fext, c->g, b_out, h->f, h->a, h->v, rne_ws);
}

/* ============================================================================
 * Raycast — single ray vs all collision shapes. Caller (Python) handles the
 * env interface; here we dispatch per ctype to the matching ray_intersects_*.
 *
 * tact_raycast_query: one shot. Recomputes _fk(q), then walks shapes.
 * tact_raycast_frame: n rays from a sensor frame (directions generated in
 *   Python — sim.py Env._ray_grid) over the shared per-shape loop (single
 *   _fk for the whole batch).
 *
 * Mesh raycast transforms the ray into shape-local frame (rotation transpose +
 * translation diff) so we don't have to transform every vertex per ray.
 * ============================================================================ */
/* Per-frame precomputed shape pose cache. tact_raycast_frame fires n rays through a
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
 * Requires c->T populated by _fk(q). `cache` must hold at least h->n_shape entries. */
static int rc_build_cache(tact_t *h, rc_shape *cache)
{
    tact_core_t *c = h->core;
    int n = 0;
    for (int i = 0; i < h->n_shape; i++) {
        if (c->craycast[i] == 0) continue;  /* shape opted out of raycast (YAML raycast: false) */
        double Tw[16];
        if (c->cbody[i] < 0) memcpy(Tw, c->ctran + 16*i, 16*sizeof(double));
        else matmul(Tw, c->T + 16*c->cbody[i], c->ctran + 16*i, 4, 4, 4);
        rc_shape *s = &cache[n++];
        s->p[0]=Tw[3];  s->p[1]=Tw[7];  s->p[2]=Tw[11];
        s->R[0]=Tw[0];  s->R[1]=Tw[1];  s->R[2]=Tw[2];
        s->R[3]=Tw[4];  s->R[4]=Tw[5];  s->R[5]=Tw[6];
        s->R[6]=Tw[8];  s->R[7]=Tw[9];  s->R[8]=Tw[10];
        s->z[0]=Tw[2];  s->z[1]=Tw[6];  s->z[2]=Tw[10];
        s->type = c->ctype[i];
        s->sh   = c->cshape + 3*i;
        s->slot = (s->type == 100 || s->type == 105) ? (int)c->cshape[3*i] : -1;  /* mesh / hfield slot */
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

/* n world-frame rays, per-ray origins — the general primitive (height_scan's
 * vertical scan grid, single-shot queries). One _fk + shape cache for the
 * batch, then raycast_cached per ray. No cone cull: that optimization assumes
 * all rays share one origin (sensor batch below); here origins differ per ray
 * and the per-shape bounding-sphere broad test inside raycast_cached is the
 * broad phase. (Replaced the single-ray tact_raycast_query 2026-06-06 —
 * height_scan's G+1-query loop re-ran _fk + cache per ray, 36x the fixed
 * cost per scan.) */
void tact_raycast_world(tact_t *h, double *q, double *R0s, double *Rds, int n, double *t_out)
{
    tact_core_t *c = h->core;
    _fk(c->T, h->nb, c->Ti, c->parent, c->jtype, q);
    rc_shape cache[h->n_shape > 0 ? h->n_shape : 1];
    int ncache = rc_build_cache(h, cache);
    for (int k = 0; k < n; k++)
        t_out[k] = raycast_cached(cache, ncache, R0s + 3*k, Rds + 3*k);
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
void tact_raycast_frame(tact_t *h, double *q, int frame_idx, double *dirs, int n, double *t_out)
{
    tact_core_t *c = h->core;
    _fk(c->T, h->nb, c->Ti, c->parent, c->jtype, q);
    /* Hoist per-shape world transforms out of the ray loop (single _fk -> poses
     * fixed for the whole batch). Frame-local, read-only during the loop. */
    rc_shape cache[h->n_shape > 0 ? h->n_shape : 1];
    int ncache = rc_build_cache(h, cache);

    int bi = c->fbody[frame_idx];
    double Twf[16];
    if (bi < 0) memcpy(Twf, c->ftran + 16*frame_idx, 16*sizeof(double));
    else matmul(Twf, c->T + 16*bi, c->ftran + 16*frame_idx, 4, 4, 4);

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
 * Used by tact_ik2 to solve (J Jᵀ + λ²I) x = e_x. λ²>0 guarantees SPD.
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

/* tact_ik2 — Damped Least Squares IK, mirrors fixed sim.py:Model.ik2().
 * Algorithm (per iter): e=error(q); if |e|<tol return; J=jacob(q); A=JJᵀ+λ²I;
 * solve A x=e; q += advance · Jᵀ x.
 *
 * Fixed-joint columns of J are zeroed before forming A — algebraically identical
 * to Python's np.delete(J, self.fixed, axis=1) since zero columns contribute nothing
 * to JJᵀ and produce zero updates on those q rows.
 *
 * Workspace layout (in c->workspace, 120·nb doubles): q_full(nb) | dq(nb) | J(m·nb)
 *                                                   | A(m·m) | e(m) | bx(m) | yvec(m).
 * Caller responsibility: ensure m ≤ ~6·MAX_NB so the carve fits. */
int tact_ik2(tact_t *h, double *q_in, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double advance, double tolerance, double damping, int max_iter, double *q_out)
{
    tact_core_t *c = h->core;
    int nb = h->nb, nq = h->nq;
    int m  = 0;
    for (int k = 0; k < n; k++) m += (mode[k] == 0) ? 3 : 6;

    /* per-DoF (nq) layout. q_full/dq are nq; J is m × nq; J_block scratch needs 6*nq */
    double *q_full = c->workspace;             /* nq */
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
        /* error: _fk + per-frame Te + 3d/6d residual (parity with tact_error) */
        _fk(c->T, nb, c->Ti, c->parent, c->jtype, q_full);
        int oi = 0;
        for (int k = 0; k < n; k++) {
            int fi = frame_idx[k];
            int bi = c->fbody[fi];
            if (bi < 0) bi = nb + bi;
            double Te[16];
            matmul(Te, c->T + 16*bi, c->ftran + 16*fi, 4, 4, 4);
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
            int bi = c->fbody[fi];
            int idx_c = (bi < 0) ? (nb + bi) : bi;
            double Te[16];
            matmul(Te, c->T + 16*idx_c, c->ftran + 16*fi, 4, 4, 4);
            jacob_whitney(J_blk, nb, c->T, Te, c->parent, c->jtype, idx_c);
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
        q_step(nb, c->jtype, q_full, dq, advance, q_full);

        iter++;
    }

    memcpy(q_out, q_full, nq*sizeof(double));
    return -max_iter;
}
