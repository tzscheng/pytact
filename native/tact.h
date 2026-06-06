#ifndef TACT_H
#define TACT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =============================================================================
 * Common constants
 * ============================================================================= */
#define MAX_NB 256
#define EPS 1e-12

/* Contact manifold capacity per cpair. Currently the narrowphase still returns 1
 * point per pair (sub_id always 0), so slots 1..3 are unused. Phase 2+: box-box
 * SAT/clipping narrowphase will fill 0..MAX_PTS_PER_PAIR-1. Cost: the contact
 * warm-start block and LCP A workspace scale as MAX_PTS² in the worst case (currently dominated by
 * other terms since most slots stay empty). Indexing convention: warm-start slot =
 * cpair_idx * MAX_PTS_PER_PAIR + sub_id. */
#define MAX_PTS_PER_PAIR 4

/* shape type codes (collision) — kept in sync with sim.py (YAML loader) + _clib.py */
#define MESH    100
#define BOX     101
#define SPHERE  102
#define CYL     103
#define CAPSULE 104
#define HFIELD  105   /* height field — regular grid of heights, non-convex terrain */


/* =============================================================================
 * rbd.c — linear algebra
 * ============================================================================= */
void cross3(double a1, double a2, double a3, double b1, double b2, double b3, double *v);
void normalize(double *n, double *v, int row);
void matscalar(double *m2, double a, double *m1, int row, int col);
void matmul(double *m3, double *m1, double *m2, int r1, int c1r2, int c2);
void matsum(double *m3, double *m1, double *m2, int row, int col);
void matsub(double *m3, double *m1, double *m2, int row, int col);
void matinv3(double *Minv, double *M);
void matinv4_affine(double *Tinv, const double *T);
void matover(double *m2, int r2, int c2, int a, int b, double *m1, int r1, int c1);
void identity(double *m, int size);
void diagonal(double *m, double *v, int size);
void transpose(double *m2, double *m1, int r1, int c1);
void matprint(double *m, int row, int col);
void randmat(double *m, int row, int col);


/* =============================================================================
 * rbd.c — euler / quaternion / homogeneous transforms
 * Convention: lowercase eulerseq = extrinsic ("xyz"/"zyx"/...),
 *             uppercase = intrinsic ("XYZ"/"ZYX"/...).
 * ============================================================================= */
void euler_to_rotation     (double *e, double *R, const char *eulerseq);
void rotation_to_euler     (double *R, double *e, const char *eulerseq);
void quat_to_rotation      (double *q, double *R);
void rotation_to_quat      (double *R, double *q);
void euler_to_quat         (double *e, double *q, const char *eulerseq);
void quat_to_euler         (double *q, double *e, const char *eulerseq);
void xyzeuler_to_xyzquat   (double *x6, double *x7, const char *eulerseq);
void xyzquat_to_xyzeuler   (double *x7, double *x6, const char *eulerseq);
void homogeneous_to_xyzeuler(double *T, double *x, const char *eulerseq);
void xyzeuler_to_homogeneous(double *x, double *T, const char *eulerseq);
void xyzquat_to_homogeneous(double *x, double *T);
void homogeneous_to_xyzquat(double *T, double *x);
void xyheading_to_homogeneous(double x, double y, double heading, double *T);
void rotxyz_to_homogeneous (double *R, double *x, double *T);
void homogeneous_error     (double *T1, double *T2, double *e);
void rotation_error        (double *R1, double *R2, double *e);

/* SO(3) exp/log — used by free (jtype=3) joint with axis-angle rotation. */
void skew3        (double *K, const double *v);
void expmap_so3   (const double *w, double *R);
void logmap_so3   (const double *R, double *w);
void integrate_so3(const double *w, const double *omega, double dt, double *w_next);

/* basic 3×3 / 4×4 / 6×6 axis-aligned builders */
void rot_x  (double *R, double th); void rot_y  (double *R, double th); void rot_z  (double *R, double th);
void T_rot_x(double *T, double th); void T_rot_y(double *T, double th); void T_rot_z(double *T, double th);
void T_trans(double *T, double x, double y, double z);
void X_rot_x(double *X, double th); void X_rot_y(double *X, double th); void X_rot_z(double *X, double th);
void X_trans(double *X, double x, double y, double z);


/* =============================================================================
 * rbd.c — spatial-vector dynamics (Featherstone / Siciliano)
 * ============================================================================= */
void crm  (double *M, double *v);
void crf  (double *M, double *v);
void jcalc(double *XJ, double *S, int jtype, double q);
void jcalc6(double *XJ, double *S, const double *q6);
void _fk  (double *T, int nb, double *Ti, int *parent, int *jtype, double *q);
void jacob_whitney(double *J, int nb, double *T, double *_T, int *parent, int *jtype, int idx);

/* full=1 also fills f_out/a_out/v_out (the "spatial" tuple).
 * ff/sk/dt_imp: optional semi-implicit damping/spring (NULL/0 = legacy explicit).
 * Kp_j/Kd_j/q_ref/qd_ref: optional joint-space implicit PD (capability+activation —
 * all four arrays length nb, NULL = inactive). Kp acts only when both Kp_j and q_ref
 * are non-NULL; Kd acts whenever Kd_j is non-NULL and either q_ref or qd_ref is.
 * Result equivalent to: qdd = (tau - bias - ff·qd - sk·q - Kp·(q-qref) - Kp·dt·qd
 *                              - Kd·(qd-qdref)) / (M + (ff+Kd)·dt + (sk+Kp)·dt²) */
void aba_featherstone(int nb, double *X, double *I6, int *parent, int *jtype,
                      double *q, double *qd, double *tau, double *f_ext, double *g,
                      double *qdd, double *f_out, double *a_out, double *v_out,
                      double *workspace,
                      double *ff, double *sk, double *armature, double dt_imp,
                      double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref,
                      int full);
void rne_featherstone(int nb, double *X, double *I6, int *parent, int *jtype,
                      double *q, double *qd, double *qdd, double *f_ext, double *g,
                      double *tau, double *f_out, double *a_out, double *v_out,
                      double *workspace);

/* Composite Rigid-Body Algorithm — joint-space mass matrix H (row-major, nb*nb).
 * Workspace size: 78*nb doubles. Fixed-joint (jtype==0) rows/cols are zero. */
void crb_featherstone(int nb, double *X, double *I6, int *parent, int *jtype, double *q, double *H, double *workspace);

/* General dense SPD LDL^T (in-place). A is n×n row-major. After factor: lower
 * triangle holds L (unit-diag implicit), diagonal holds D, upper triangle is
 * unchanged. ldlt_factor returns 0 on success or -(k+1) when pivot D[k]≤EPS. */
int  ldlt_factor(double *A, int n);
void ldlt_solve (const double *A, int n, double *b);

/* q_step: per-jtype manifold-aware q update. Mirrors rbd.py:_q_step. For free
 * (jtype=3) body, q[0:3] is integrated via R(w)·v_body·dt (body→world frame)
 * and q[3:6] (rotation vector) via SO(3) exp map composition. */
void q_step(int nb, int *jtype, const double *q, const double *qd, double dt, double *q_next);

/* integrator=1 → euler_step, integrator=2 → rk4_step */
void euler_step(int nb, double *X, double *I6, int *parent, int *jtype,
                double *q, double *qd, double *tau, double *f_ext, double *g, double dt,
                double *q_next, double *qd_next, double *qdd,
                double *f_out, double *a_out, double *v_out,
                double *workspace);
void rk4_step  (int nb, double *X, double *I6, int *parent, int *jtype,
                double *q, double *qd, double *tau, double *f_ext, double *g, double dt,
                double *q_next, double *qd_next, double *qdd,
                double *f_out, double *a_out, double *v_out,
                double *workspace);


/* =============================================================================
 * narrow.c (dispatch + analytic detectors) / box_box.c (box-box SAT+clipping) /
 * mpr.c (convex fallback) / ray.c (ray casts) / shape.c (mesh + hfield slot
 * storage) — collision side
 * ============================================================================= */

/* Multi-point collision detection.
 *
 * Output buffer `out` carries up to `max_pts` contact points, each laid out as
 * 7 consecutive doubles:
 *   out[7*k + 0..2]  contact point in world coords
 *   out[7*k + 3..5]  depth vector  =  n_hat * depth (world; direction by which
 *                                     param1's body is pushed apart from param2)
 *   out[7*k + 6]     depth scalar (≥ 0)
 *
 * Return value:
 *   ≥ 0 : number of contact points written (0 = colliding but no points met
 *         the depth threshold, ≥ 1 = active contacts).
 *   < 0 : separating axis found / no overlap; `out` not modified.
 *
 * Dispatch (current implementation):
 *   - box-box pairs:  box_box_manifold() — SAT + face clipping, up to max_pts pts.
 *   - all other type combinations: existing MPR (single point), wrapped to return
 *     n_pts = 1 (depth > 0) or n_pts = 0 (touching, no penetration). */
int  collision_check(int type1, double *param1, int type2, double *param2,
                     double *out, int max_pts);

/* Single-point MPR fallback (mpr.c) — used by collision_check for convex pairs
 * with no dedicated detector. out[0..6] = [point, normal, depth]; returns intersect. */
int  collision_check_mpr(int type1, double *param1, int type2, double *param2, double *out);

/* box_box.c — SAT + face-clipping manifold for two BOX shapes.
 * Same return / out-layout convention as collision_check.
 *
 * Half-extents in param[6..8], position in param[0..2], extrinsic-xyz Euler in
 * param[3..5]. Up to min(max_pts, MAX_PTS_PER_PAIR) contact points returned,
 * sub-id'd 0..n-1 by polar angle around manifold centroid in tangent plane. */
int  box_box_manifold(const double *param1, const double *param2,
                      double *out, int max_pts);
int  load_obj(int mesh_idx);
/* Ray intersection primitives (ray.c) — sphere needs |Rd|=1, box returns -1 when
 * origin inside, triangle/mesh are double-sided. All return forward distance t along
 * Rd, or -1.0 on miss. ray_intersects_mesh_slot pulls vertex/face data from shape.c's
 * global slot storage (loaded lazily). */
double ray_intersects_triangle (const double *R0, const double *Rd, const double *v0, const double *v1, const double *v2);
double ray_intersects_mesh_slot(const double *R0, const double *Rd, int mesh_idx);
double ray_intersects_box      (const double *R0, const double *Rd, const double *center, const double *R, const double *hs);
double ray_intersects_sphere   (const double *R0, const double *Rd, const double *C, double r);
double ray_intersects_cylinder (const double *R0, const double *Rd, const double *P1, const double *P2, double r);
double ray_intersects_capsule  (const double *R0, const double *Rd, const double *P1, const double *P2, double r);
/* Ray vs height-field slot `slot`, R0/Rd in the hfield's local frame (grid in the
 * local XY plane, height along +Z). 2D DDA grid walk (only the cells the ray crosses,
 * first-hit = nearest); returns forward t along Rd or -1.0 on miss. */
double ray_intersects_hfield   (const double *R0, const double *Rd, int slot);
/* Bounding-sphere radius of mesh slot `idx` about its local origin (max |vertex|), cached
 * per slot. -1 if invalid/unloadable. Used by the raycast broad phase (frustum cull /
 * ray-sphere reject) to bound a mesh without touching its triangles. */
double mesh_local_radius(int idx);
/* Bounding-sphere radius of hfield slot `slot` about its local origin:
 * sqrt(sx² + sy² + max(|min_h|,|max_h|)²). -1 if the slot is empty. Raycast broad phase. */
double hfield_local_radius(int slot);
/* Register the filesystem path for mesh slot `idx`. Called from Python's
 * Model.build() after tact_create() so subsequent collision/render mesh
 * loads know which `.obj` file to read. Path must be readable from the
 * process's current working directory or absolute. */
void set_mesh_path(int idx, const char* path);
/* Register a height-field grid into slot `slot`. Called from Python's Model.build()
 * (analogous to set_mesh_path, but the grid is pushed directly rather than read from a
 * file C-side). `data` is nrow*ncol heights in meters, row-major: data[i*ncol + j] is
 * the height at grid node (row i along local +Y, col j along local +X). The grid spans
 * local [-sx, sx] × [-sy, sy]. Copies the data; safe to free `data` after the call. */
void set_hfield_data(int slot, int nrow, int ncol, double sx, double sy, const double* data);
/* 3×3 orthonormal frame whose third column = normalized z_in. Row-major, R[3*r+c]. */
void choose_rotation(double *z_in, double *R);


/* =============================================================================
 * lcp.c — Stewart-Trinkle / Anitescu LCP contact solver with 4 cones
 * (normal, tangent disk, spin clamp, roll disk). Mirrors rbd.py:contact_lcp.
 *
 * Inputs (caller-allocated):
 *   nb, T[16*nb], parent[nb], jtype[nb]               kinematics at current q
 *   n_pair, cpair[2*n_pair], ctype[nshape], cbody[nshape]
 *   ctran[16*nshape], cshape[3*nshape], cparam[13*nshape]   contact geometry/material
 *   qd_free[nb], M_full[nb*nb], dt                    pre-contact predictor + mass matrix
 *   erp, slop, cfm_scale, v_rest_thresh, iters, tol   solver parameters
 *   lam_contact_prev[6*n_pair] or NULL                        warm-start (cpair-indexed); NULL=cold
 *   floss[nq] or NULL                                 per-DoF joint Coulomb bound (N·m / N);
 *                                                     0/NULL = no friction row (jtype 1/2 only)
 *   lam_fric[nq] or NULL                              per-DoF friction warm-start, in-place carry
 *   q[nq], jnt_lo[nq], jnt_hi[nq] or NULL             joint positions + per-DoF limit range
 *                                                     (limited iff lo<hi, jtype 1/2 only); NULL = no limits
 *   lam_limit[nq] or NULL                             per-DoF limit warm-start, in-place carry
 *   workspace                                         see lcp.c for sizing (≈ 6P·F + 36P² + ...)
 *
 * Outputs:
 *   dqd_out[nb]            velocity correction (qd_next = qd_free + dqd)
 *   lam_contact_out[6*n_pair] λ scattered back to cpair index (warm-start carry)
 *   f_ext_out[6*nb]        body-frame wrench (zero on bodies with no contact)
 *   *nc_out                number of active contacts this step
 *   *iters_out             actual PGS iterations consumed (≤ iters)
 *   *residual_out          max |Δλ| at last iteration
 * ============================================================================= */
void contact_lcp(int nb, double *T, int *parent, int *jtype,
                 int n_pair, int *cpair, int *ctype, int *cbody,
                 double *ctran, double *cshape, double *cparam,
                 double *qd_free, double *M_full, double dt,
                 double erp, double slop, double cfm_scale,
                 double v_rest_thresh,
                 int iters, double tol,
                 double *lam_contact_prev,
                 double *floss, double *lam_fric,
                 double *q, double *jnt_lo, double *jnt_hi, double *lam_limit,
                 double *dqd_out, double *lam_contact_out, double *f_ext_out,
                 int *nc_out, int *iters_out, double *residual_out,
                 double *workspace);


/* =============================================================================
 * tact.c — high-level handle API (orchestrates ccd + rbd, owns per-instance state)
 * Build / runtime split + lifecycle invariants: see docs/design-c-state.md §3.
 * ============================================================================= */
typedef struct tact_t tact_t;

tact_t *tact_create(int nb, int *parent, int *jtype,
                    double *X, double *I6, double *Ti, double *ff, double *sk,
                    double *floss, double *armature, double *jnt_lo, double *jnt_hi,
                    double *g, double dt, int integrator,
                    int n_shape, int n_pair,
                    int *ctype, int *cbody, double *cshape, double *ctran, double *cparam,
                    int *craycast, int *cpair,
                    double erp, double slop, double cfm_scale,
                    double v_rest_thresh, int iters, double tol);
void    tact_destroy(tact_t *h);

void    tact_set_feedback(tact_t *h,
                          int n_feeds, int *kinds, int *offsets, int *idx,
                          int n_frames, int *fbody, double *ftran, double *ftran_inv,
                          int y_size);

/* In-place updates that preserve the arena → tact_get_* views stay valid.
 * Topology (nb / n_shape / n_pair) must be unchanged; for topology changes,
 * destroy + create a fresh handle. */
void    tact_edit_model (tact_t *h, double *X, double *I6, double *Ti);

/* hot path — self-contained: _fk → contact solve → integrate → feedback.
 * tact_step_lcp: Kp_j/Kd_j/q_ref/qd_ref are optional joint-space implicit PD inputs
 * (all length nb, NULL = inactive — bit-identical to pre-PD behavior). See
 * aba_featherstone's contract for semantics. */
/* lam_in / lam_out (REQUIRED, non-NULL): caller-threaded PGS warm-start λ — ONE
 * vector per direction, every row type, blocks in row-table order (= the Python
 * SolverState layout, the C↔Python ABI):
 *     [contact (6·MAX_PTS_PER_PAIR·max(n_pair,1)) | joint-friction (nq) | joint-limit (nq)]
 * lam_in is read-only (seeds lam_out); pass distinct buffers for a pure,
 * immutable-input step. A future constraint-row type appends a layout block
 * instead of growing this signature. */
void    tact_step_lcp    (tact_t *h, double *q, double *qd, double *tau,
                          double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref,
                          double *lam_in, double *lam_out);

/* multi-frame query API — caller resolves names→indices once; mode[k]: 0=3d, 1=6d */
void    tact_fk_query     (tact_t *h, double *q, int n, int *frame_idx, int *mode, const char *eulerseq, double *out);
void    tact_error_query  (tact_t *h, double *q, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double *out);
void    tact_jacob_query  (tact_t *h, double *q, int n, int *frame_idx, int *mode, double *J_out);
/* tact_com_jacob_query: 3 × nq CoM linear Jacobian at q.
 * m_in (nb) and c_in (3*nb row-major) are body masses and body-frame CoM offsets
 * — passed in because tact_t holds only the 6×6 spatial inertia I6, not the
 * raw (m_i, c_i) tuples. Algorithm mirrors Python's com_jacob_lagrange: mass-
 * weighted average of per-body world-linear Jacobians (top 3 rows of
 * jacob_whitney evaluated at each body's world-frame CoM point). */
void    tact_com_jacob_query(tact_t *h, double *q, double *m_in, double *c_in, double *J_out);
/* tact_com_query: 3-vector CoM world position. Same (m_in, c_in) inputs as
 * tact_com_jacob_query for the same reason — tact_t lacks the raw (m_i, c_i). */
void    tact_com_query    (tact_t *h, double *q, double *m_in, double *c_in, double *r_out);
void    tact_gravity_query(tact_t *h, double *q, double *g_override, double *b_out);
/* tact_inertia_query: joint-space mass matrix at q. H_out is row-major (nq*nq). */
void    tact_inertia_query(tact_t *h, double *q, double *H_out);
/* tact_bias_query: C(q,qd)·qd + g(q) − Jᵀf_ext. f_ext_in NULL → treated as zero. */
void    tact_bias_query   (tact_t *h, double *q, double *qd, double *f_ext_in, double *b_out);

/* Damped Least Squares IK. q_in/q_out are nq-length per-DoF vectors (fixed joints
 * contribute 0 slots so no extend/compress is needed by the caller).
 * Returns iter count if converged (|e|<tolerance), -iter count if max_iter hit. */
int     tact_ik2_query    (tact_t *h, double *q_in, double *x_d,
                           int n, int *frame_idx, int *mode, const char *eulerseq,
                           double advance, double tolerance, double damping, int max_iter,
                           double *q_out);

/* Raycast: n world-frame rays vs all collision shapes — the general primitive.
 * R0s/Rds = n per-ray origins/unit directions (3 doubles each). One _fk +
 * shape-pose cache for the whole batch; no cone cull (it assumes a shared
 * origin — the consumers here, height_scan / single-shot, fire from distinct
 * origins). t_out[k] = forward range along Rds[k], -1 = no hit.
 * (Replaced the single-ray tact_raycast_query 2026-06-06.) */
void    tact_raycast_world      (tact_t *h, double *q, double *R0s, double *Rds,
                           int n, double *t_out);
/* Batched raycast from a sensor frame: `dirs` = n unit ray directions in the
 * frame's registered coordinates (ray generation lives in Python — single
 * source; replaced tact_raymap_query 2026-06-06). Shared origin → cone
 * frustum cull. t_out[k] = forward range along dirs[k], -1 = no hit. */
void    tact_raycast_frame(tact_t *h, double *q, int frame_idx,
                           double *dirs, int n, double *t_out);

/* arena read-only accessors — Python wraps as numpy views.
 * Views are valid only until the next tact_destroy on this handle (§3.5 invariant). */
double *tact_get_f       (tact_t *h);
double *tact_get_a       (tact_t *h);
double *tact_get_v       (tact_t *h);
double *tact_get_q_next  (tact_t *h);
double *tact_get_qd_next (tact_t *h);
double *tact_get_y       (tact_t *h);

#endif /* TACT_H */
