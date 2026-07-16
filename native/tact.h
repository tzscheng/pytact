#ifndef TACT_H
#define TACT_H

#include <stdint.h>

/* libtact.so is built with -fvisibility=hidden: ONLY declarations carrying
 * TACT_API are exported. Everything else (internal math/collision helpers,
 * vendored libccd) stays private to the library, so generic internal names
 * like matmul/normalize can never collide with other libraries in the
 * process. */
#ifndef TACT_API
#  if defined(__GNUC__)
#    define TACT_API __attribute__((visibility("default")))
#  else
#    define TACT_API
#  endif
#endif

/* Common public constants. */
#define TACT_MAX_NB 256
#define TACT_EPS 1e-12
#define TACT_MAX_PTS_PER_PAIR 4

/* Shape type codes, kept in sync with the Python YAML loader. */
#define TACT_MESH    100
#define TACT_BOX     101
#define TACT_SPHERE  102
#define TACT_CYL     103
#define TACT_CAPSULE 104
#define TACT_HFIELD  105

typedef struct tact_core_t tact_core_t;   /* private engine state, defined in core.h */

/* Public handle. Always engine-allocated (tact_create_*); never allocate or
 * copy it yourself. All fields are read-only for user code — mutate state only
 * through tact_ functions. Pointer fields are views into engine-owned storage,
 * valid until tact_destroy().
 *
 * ABI rule: new public fields are appended at the end of their section, before
 * `core`, so field offsets compiled into consumers stay valid across releases. */
typedef struct tact_t {
    /* -- model dimensions / parameters: fixed after create -- */
    int     nb;                 /* bodies */
    int     nq;                 /* position/velocity DoF slots (fixed=0, rev/lin=1, free=6) */
    int     n_shape, n_pair;    /* collision shapes / candidate pairs */
    int     n_frames;           /* frames registered via tact_set_feedback (tact_fk index range) */
    int     y_size;             /* feedback vector length (0 until tact_set_feedback) */
    int     ctx_size;           /* solver ctx length for tact_step_lcp */
    int     contact_count;      /* step output: active contacts after tact_step_lcp */
    double  dt;

    const double *q0, *qd0;     /* initial state, length nq (bin models; else NULL) */

    /* -- engine-owned step outputs: read after tact_step_lcp -- */
    double *f, *a, *v;          /* per-body spatial wrench / accel / vel, 6*nb */
    double *q_next, *qd_next;   /* integrated next state, nq */
    double *y;                  /* feedback vector, y_size (NULL until tact_set_feedback) */
    double *ctx_next;           /* next warm-start ctx, ctx_size. Pass ctx_next itself
                                 * as tact_step_lcp's ctx to continue from the engine's
                                 * own last warm-start (stateful convenience), or copy
                                 * it into your own ctx buffer and pass that (pure
                                 * threading, the q/q_next idiom); NULL = cold start */

    /* -- private engine state: do not access -- */
    tact_core_t *core;
} tact_t;

/* Functions prefixed with tact_ are the supported public API — and with
 * -fvisibility=hidden they are the ONLY symbols libtact.so exports. */

TACT_API int tact_create_from_arrays(int nb, int *parent, int *jtype, double *X, double *I6, double *Ti, double *ff, double *sk, double *floss, double *armature, double *taulim, double *jnt_lo, double *jnt_hi,
			    double *g, double dt, int integrator, int n_shape, int n_pair, int *ctype, int *cbody, double *cshape, double *ctran, double *cparam, int *craycast, int *cpair,
			    double erp, double slop, double cfm_scale, double v_rest_thresh, int iters, double tol, tact_t **out);

TACT_API int  tact_create_from_bin(const char *path, tact_t **out);
TACT_API void tact_destroy(tact_t *h);
TACT_API int  tact_step_lcp(tact_t *h, double *q, double *qd, double *tau, double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref, double *ctx);
TACT_API void tact_set_feedback(tact_t *h, int n_feeds, int *kinds, int *offsets, int *idx, int n_frames, int *fbody, double *ftran, double *ftran_inv, int y_size);
TACT_API void tact_edit_model(tact_t *h, double *X, double *I6, double *Ti);

TACT_API void tact_fk(tact_t *h, double *q, int n, int *frame_idx, int *mode, const char *eulerseq, double *out);
TACT_API void tact_error(tact_t *h, double *q, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double *out);
TACT_API void tact_jacob(tact_t *h, double *q, int n, int *frame_idx, int *mode, double *J_out);
TACT_API int  tact_ik2(tact_t *h, double *q_in, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double advance, double tolerance, double damping, int max_iter, double *q_out);
TACT_API void tact_com_jacob(tact_t *h, double *q, double *m_in, double *c_in, double *J_out);
TACT_API void tact_com(tact_t *h, double *q, double *m_in, double *c_in, double *r_out);
TACT_API void tact_gravity(tact_t *h, double *q, double *g_override, double *b_out);
TACT_API void tact_inertia(tact_t *h, double *q, double *H_out);
TACT_API void tact_bias(tact_t *h, double *q, double *qd, double *f_ext_in, double *b_out);

TACT_API void tact_raycast_world(tact_t *h, double *q, double *R0s, double *Rds, int n, double *t_out);
TACT_API void tact_raycast_frame(tact_t *h, double *q, int frame_idx, double *dirs, int n, double *t_out);

TACT_API int  tact_render(const tact_t *h, const double *q);
TACT_API void tact_render_set_light(float pos[3], float target[3], float ortho, int shadow_enabled);

/* Scene-array renderers (the Python Env render path binds these directly;
 * tact_render is the handle-based convenience wrapper over tact_win_render).
 * objpose is column-major 4x4 per object; campose = [target(3), dist, yaw, pitch].
 * tact_win_render: GLFW window; returns -1 when the window is closed / ESC.
 * tact_egl_render: headless EGL to out_buf (opt selects rgb-JPEG / depth-zstd);
 * returns the encoded byte count. */
TACT_API int tact_win_render(int n_obj, int *obj_type, float *shape, float *objcolor, float *objpose, float *campose);
TACT_API int tact_egl_render(int n_obj, int *obj_type, float *shape, float *objcolor, float *objpose, float *campose,
                    char *out_buf, int opt, int req_width, int req_height, float fovy_deg);

TACT_API int  tact_collision_check(int type1, double *param1, int type2, double *param2, double *out, int max_pts);
TACT_API int  tact_collision_check_mpr(int type1, double *param1, int type2, double *param2, double *out);
TACT_API int  tact_box_box_manifold(const double *param1, const double *param2, double *out, int max_pts);
TACT_API void tact_set_mesh_path(int idx, const char *path);
TACT_API void tact_set_hfield_data(int slot, int nrow, int ncol, double sx, double sy, const double *data);
TACT_API void tact_contact_reports (tact_t *h, int *contact_i_out, double *contact_d_out);

TACT_API int tact_frame_count(const tact_t *h);
TACT_API const char *tact_frame_name(const tact_t *h, int frame_id);
TACT_API int tact_frame_id(const tact_t *h, const char *name);


#endif /* TACT_H */
