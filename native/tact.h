#ifndef TACT_H
#define TACT_H

#include <stdint.h>

/* Common public constants. */
#define MAX_NB 256
#define EPS 1e-12
#define MAX_PTS_PER_PAIR 4

/* Shape type codes, kept in sync with the Python YAML loader. */
#define MESH    100
#define BOX     101
#define SPHERE  102
#define CYL     103
#define CAPSULE 104
#define HFIELD  105

typedef struct tact_t tact_t;

/* Functions prefixed with tact_ are the supported public API. Other exported
 * symbols are internal/unstable and may change without notice. */

int tact_create_from_arrays(int nb, int *parent, int *jtype, double *X, double *I6, double *Ti, double *ff, double *sk, double *floss, double *armature, double *jnt_lo, double *jnt_hi,
			    double *g, double dt, int integrator, int n_shape, int n_pair, int *ctype, int *cbody, double *cshape, double *ctran, double *cparam, int *craycast, int *cpair,
			    double erp, double slop, double cfm_scale, double v_rest_thresh, int iters, double tol, tact_t **out);

int  tact_create_from_bin(const char *path, tact_t **out);
void tact_destroy(tact_t *h);
int  tact_step_lcp(tact_t *h, double *q, double *qd, double *tau, double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref, double *ctx_in, double *ctx_out);
void tact_set_feedback(tact_t *h, int n_feeds, int *kinds, int *offsets, int *idx, int n_frames, int *fbody, double *ftran, double *ftran_inv, int y_size);
void tact_edit_model(tact_t *h, double *X, double *I6, double *Ti);

void tact_fk(tact_t *h, double *q, int n, int *frame_idx, int *mode, const char *eulerseq, double *out);
void tact_error(tact_t *h, double *q, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double *out);
void tact_jacob(tact_t *h, double *q, int n, int *frame_idx, int *mode, double *J_out);
int  tact_ik2(tact_t *h, double *q_in, double *x_d, int n, int *frame_idx, int *mode, const char *eulerseq, double advance, double tolerance, double damping, int max_iter, double *q_out);
void tact_com_jacob(tact_t *h, double *q, double *m_in, double *c_in, double *J_out);
void tact_com(tact_t *h, double *q, double *m_in, double *c_in, double *r_out);
void tact_gravity(tact_t *h, double *q, double *g_override, double *b_out);
void tact_inertia(tact_t *h, double *q, double *H_out);
void tact_bias(tact_t *h, double *q, double *qd, double *f_ext_in, double *b_out);

void tact_raycast_world(tact_t *h, double *q, double *R0s, double *Rds, int n, double *t_out);
void tact_raycast_frame(tact_t *h, double *q, int frame_idx, double *dirs, int n, double *t_out);

int  tact_render(const tact_t *h, const double *q);
void tact_render_set_light(float pos[3], float target[3], float ortho, int shadow_enabled);

int  tact_collision_check(int type1, double *param1, int type2, double *param2, double *out, int max_pts);
int  tact_collision_check_mpr(int type1, double *param1, int type2, double *param2, double *out);
int  tact_box_box_manifold(const double *param1, const double *param2, double *out, int max_pts);
void tact_set_mesh_path(int idx, const char *path);
void tact_set_hfield_data(int slot, int nrow, int ncol, double sx, double sy, const double *data);
int  tact_contact_count(tact_t *h);
void tact_contact_reports (tact_t *h, int *contact_i_out, double *contact_d_out);

int tact_nb(const tact_t *h);
int tact_nq(const tact_t *h);
int tact_n_shape(const tact_t *h);
int tact_n_pair(const tact_t *h);
int tact_y_size(const tact_t *h);
int tact_ctx_size(const tact_t *h);
double tact_dt(const tact_t *h);

const double *tact_q0(const tact_t *h);
const double *tact_qd0(const tact_t *h);
int tact_frame_count(const tact_t *h);
const char *tact_frame_name(const tact_t *h, int frame_id);
int tact_frame_id(const tact_t *h, const char *name);

double *tact_f(tact_t *h);
double *tact_a(tact_t *h);
double *tact_v(tact_t *h);
double *tact_q_next(tact_t *h);
double *tact_qd_next(tact_t *h);
double *tact_y(tact_t *h);


#endif /* TACT_H */
