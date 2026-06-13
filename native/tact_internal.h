#ifndef TACT_INTERNAL_H
#define TACT_INTERNAL_H

#include "tact.h"

struct tact_t {
    int     nb;
    int     nq;
    int     nv;
    int     n_shape, n_pair;
    int     integrator;
    double  dt;
    double  g[3];

    double  erp, slop, cfm_scale, v_rest_thresh, tol;
    int     iters;

    int    *parent;
    int    *jtype;
    int    *q_base;
    int    *v_base;
    int    *nq_per_body;
    int    *nv_per_body;
    double *X;
    double *I6;
    double *Ti;
    double *ff;
    double *sk;
    double *floss;
    double *armature;
    double *jnt_lo;
    double *jnt_hi;
    int    *ctype;
    int    *cbody;
    double *cshape;
    double *ctran;
    double *cparam;
    int    *craycast;
    int    *cpair;

    double *T;
    double *f_ext;
    double *f, *a, *v;
    double *qdd;
    double *q_next, *qd_next;
    double *tau_p;
    double *workspace;

    double *qd_free_buf;
    double *M_buf;
    double *lcp_ws;

    void   *arena;

    int     fb_set;
    int     n_feeds;
    int    *feed_kinds;
    int    *feed_offsets;
    int    *feed_idx;
    int     n_frames;
    int    *fbody;
    double *ftran;
    double *ftran_inv;
    int     y_size;
    double *y_buf;
    void   *fb_arena;

    int     lam_size;
    double *q0;
    double *qd0;
    double *zero_tau;
    double *zero_lam;
    double *crgba;
    double *view;
    double *light0;
    char   *frame_names;
};

#endif /* TACT_INTERNAL_H */
