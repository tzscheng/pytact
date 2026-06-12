#include "tact.h"

enum {
    TACTBIN_DTYPE_I32 = 1,
    TACTBIN_DTYPE_F64 = 2,
    TACTBIN_DTYPE_UTF8 = 3
};

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t n_chunks;
} tactbin_header_t;

typedef struct {
    char tag[16];
    uint32_t dtype;
    uint32_t ndim;
    uint64_t shape[4];
    uint64_t nbytes;
} tactbin_chunk_header_t;

struct tact_model_t {
    int nb, nq, n_shape, n_pair, n_frame, n_feed, y_size, lam_size;
    int integrator, iters;
    double dt, erp, slop, cfm_scale, v_rest_thresh, tol;

    int *parent, *jtype;
    double *X, *I6, *Ti;
    double *ff, *sk, *floss, *armature, *jnt_lo, *jnt_hi, *g;
    int *ctype, *cbody, *craycast, *cpair;
    double *cshape, *ctran, *cparam;
    double *q0, *qd0;
    int *feed_kinds, *feed_offsets, *feed_idx, *fbody;
    double *ftran, *ftran_inv;
};

struct tact_state_t {
    const tact_model_t *m;
    tact_t *h;
    double *q, *qd, *tau;
    double *lam[2];
    int lam_active;
};

static void free_ptr(void **p)
{
    if (*p) {
        free(*p);
        *p = NULL;
    }
}

static int read_exact(FILE *f, void *ptr, size_t n)
{
    return fread(ptr, 1, n, f) == n ? 0 : -1;
}

static int take_chunk(tact_model_t *m, const char *tag, uint32_t dtype, void *buf, uint64_t nbytes)
{
    if (strcmp(tag, "dims_i32") == 0) {
        if (dtype != TACTBIN_DTYPE_I32 || nbytes < 8 * sizeof(int32_t)) return -1;
        int32_t *v = (int32_t*)buf;
        m->nb = v[0]; m->nq = v[1]; m->n_shape = v[2]; m->n_pair = v[3];
        m->n_frame = v[4]; m->n_feed = v[5]; m->y_size = v[6]; m->lam_size = v[7];
        free(buf); return 0;
    }
    if (strcmp(tag, "sim_f64") == 0) {
        if (dtype != TACTBIN_DTYPE_F64 || nbytes < 6 * sizeof(double)) return -1;
        double *v = (double*)buf;
        m->dt = v[0]; m->erp = v[1]; m->slop = v[2];
        m->cfm_scale = v[3]; m->v_rest_thresh = v[4]; m->tol = v[5];
        free(buf); return 0;
    }
    if (strcmp(tag, "sim_i32") == 0) {
        if (dtype != TACTBIN_DTYPE_I32 || nbytes < 2 * sizeof(int32_t)) return -1;
        int32_t *v = (int32_t*)buf;
        m->integrator = v[0]; m->iters = v[1];
        free(buf); return 0;
    }

    #define TAKE_I32(name) do { if (strcmp(tag, #name) == 0) { \
        if (dtype != TACTBIN_DTYPE_I32) return -1; \
        m->name = (int*)buf; return 0; \
    } } while (0)
    #define TAKE_F64(name) do { if (strcmp(tag, #name) == 0) { \
        if (dtype != TACTBIN_DTYPE_F64) return -1; \
        m->name = (double*)buf; return 0; \
    } } while (0)

    TAKE_I32(parent); TAKE_I32(jtype); TAKE_I32(ctype); TAKE_I32(cbody);
    TAKE_I32(craycast); TAKE_I32(cpair); TAKE_I32(feed_kinds);
    TAKE_I32(feed_offsets); TAKE_I32(feed_idx); TAKE_I32(fbody);

    TAKE_F64(X); TAKE_F64(I6); TAKE_F64(Ti); TAKE_F64(ff); TAKE_F64(sk);
    TAKE_F64(floss); TAKE_F64(armature); TAKE_F64(jnt_lo); TAKE_F64(jnt_hi);
    TAKE_F64(g); TAKE_F64(cshape); TAKE_F64(ctran); TAKE_F64(cparam);
    TAKE_F64(q0); TAKE_F64(qd0); TAKE_F64(ftran); TAKE_F64(ftran_inv);

    #undef TAKE_I32
    #undef TAKE_F64

    free(buf);
    return 0;
}

static int model_ready(const tact_model_t *m)
{
    return m && m->nb > 0 && m->nq >= 0 && m->lam_size > 0 &&
           m->parent && m->jtype && m->X && m->I6 && m->Ti &&
           m->ff && m->sk && m->floss && m->armature && m->jnt_lo && m->jnt_hi &&
           m->g && m->q0 && m->qd0 && m->feed_offsets && m->fbody && m->ftran && m->ftran_inv;
}

int tact_load_model(const char *path, tact_model_t **out)
{
    if (!path || !out) return -1;
    *out = NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return -2;

    tactbin_header_t hdr;
    if (read_exact(f, &hdr, sizeof(hdr)) != 0 ||
        memcmp(hdr.magic, "TACTBIN\0", 8) != 0 || hdr.version != 1) {
        fclose(f);
        return -3;
    }

    tact_model_t *m = (tact_model_t*)calloc(1, sizeof(tact_model_t));
    if (!m) { fclose(f); return -4; }

    for (uint32_t i = 0; i < hdr.n_chunks; ++i) {
        tactbin_chunk_header_t ch;
        if (read_exact(f, &ch, sizeof(ch)) != 0) {
            tact_destroy_model(m); fclose(f); return -5;
        }
        void *buf = NULL;
        if (ch.nbytes > 0) {
            if (ch.nbytes > (uint64_t)SIZE_MAX) {
                tact_destroy_model(m); fclose(f); return -6;
            }
            buf = malloc((size_t)ch.nbytes);
            if (!buf) { tact_destroy_model(m); fclose(f); return -7; }
            if (read_exact(f, buf, (size_t)ch.nbytes) != 0) {
                free(buf); tact_destroy_model(m); fclose(f); return -8;
            }
        }

        char tag[17];
        memcpy(tag, ch.tag, 16);
        tag[16] = '\0';
        char *nul = memchr(tag, '\0', 16);
        if (nul) *nul = '\0';

        if (take_chunk(m, tag, ch.dtype, buf, ch.nbytes) != 0) {
            if (buf) free(buf);
            tact_destroy_model(m); fclose(f); return -9;
        }
    }
    fclose(f);

    if (!model_ready(m)) {
        tact_destroy_model(m);
        return -10;
    }

    *out = m;
    return 0;
}

void tact_destroy_model(tact_model_t *m)
{
    if (!m) return;
    free_ptr((void**)&m->parent); free_ptr((void**)&m->jtype);
    free_ptr((void**)&m->X); free_ptr((void**)&m->I6); free_ptr((void**)&m->Ti);
    free_ptr((void**)&m->ff); free_ptr((void**)&m->sk); free_ptr((void**)&m->floss);
    free_ptr((void**)&m->armature); free_ptr((void**)&m->jnt_lo); free_ptr((void**)&m->jnt_hi);
    free_ptr((void**)&m->g);
    free_ptr((void**)&m->ctype); free_ptr((void**)&m->cbody); free_ptr((void**)&m->craycast);
    free_ptr((void**)&m->cpair); free_ptr((void**)&m->cshape); free_ptr((void**)&m->ctran);
    free_ptr((void**)&m->cparam);
    free_ptr((void**)&m->q0); free_ptr((void**)&m->qd0);
    free_ptr((void**)&m->feed_kinds); free_ptr((void**)&m->feed_offsets);
    free_ptr((void**)&m->feed_idx); free_ptr((void**)&m->fbody);
    free_ptr((void**)&m->ftran); free_ptr((void**)&m->ftran_inv);
    free(m);
}

int tact_model_info(const tact_model_t *m, tact_model_info_t *out)
{
    if (!m || !out) return -1;
    out->nb = m->nb; out->nq = m->nq; out->n_shape = m->n_shape; out->n_pair = m->n_pair;
    out->n_frame = m->n_frame; out->n_feed = m->n_feed; out->y_size = m->y_size;
    out->lam_size = m->lam_size; out->dt = m->dt;
    return 0;
}

int tact_create_state(const tact_model_t *m, tact_state_t **out)
{
    if (!model_ready(m) || !out) return -1;
    *out = NULL;

    tact_state_t *s = (tact_state_t*)calloc(1, sizeof(tact_state_t));
    if (!s) return -2;
    s->m = m;

    s->q = (double*)malloc((size_t)m->nq * sizeof(double));
    s->qd = (double*)malloc((size_t)m->nq * sizeof(double));
    s->tau = (double*)calloc((size_t)m->nq, sizeof(double));
    s->lam[0] = (double*)calloc((size_t)m->lam_size, sizeof(double));
    s->lam[1] = (double*)calloc((size_t)m->lam_size, sizeof(double));
    if (!s->q || !s->qd || !s->tau || !s->lam[0] || !s->lam[1]) {
        tact_destroy_state(s);
        return -3;
    }
    memcpy(s->q, m->q0, (size_t)m->nq * sizeof(double));
    memcpy(s->qd, m->qd0, (size_t)m->nq * sizeof(double));

    s->h = tact_create(
        m->nb, m->parent, m->jtype, m->X, m->I6, m->Ti, m->ff, m->sk,
        m->floss, m->armature, m->jnt_lo, m->jnt_hi, m->g, m->dt, m->integrator,
        m->n_shape, m->n_pair, m->ctype, m->cbody, m->cshape, m->ctran, m->cparam,
        m->craycast, m->cpair, m->erp, m->slop, m->cfm_scale, m->v_rest_thresh,
        m->iters, m->tol);
    if (!s->h) {
        tact_destroy_state(s);
        return -4;
    }
    tact_set_feedback(s->h, m->n_feed, m->feed_kinds, m->feed_offsets, m->feed_idx,
                      m->n_frame, m->fbody, m->ftran, m->ftran_inv, m->y_size);

    *out = s;
    return 0;
}

void tact_destroy_state(tact_state_t *s)
{
    if (!s) return;
    if (s->h) tact_destroy(s->h);
    free(s->q); free(s->qd); free(s->tau);
    free(s->lam[0]); free(s->lam[1]);
    free(s);
}

double *tact_q(tact_state_t *s)  { return s ? s->q : NULL; }
double *tact_qd(tact_state_t *s) { return s ? s->qd : NULL; }
double *tact_y(tact_state_t *s)  { return (s && s->h) ? tact_get_y(s->h) : NULL; }

int tact_step(const tact_model_t *m, tact_state_t *s, const double *tau)
{
    if (!m || !s || s->m != m || !s->h) return -1;
    const double *u = tau ? tau : s->tau;
    double *lam_in = s->lam[s->lam_active];
    double *lam_out = s->lam[1 - s->lam_active];

    tact_step_lcp(s->h, s->q, s->qd, (double*)u, NULL, NULL, NULL, NULL, lam_in, lam_out);
    memcpy(s->q, tact_get_q_next(s->h), (size_t)m->nq * sizeof(double));
    memcpy(s->qd, tact_get_qd_next(s->h), (size_t)m->nq * sizeof(double));
    s->lam_active = 1 - s->lam_active;
    return 0;
}
