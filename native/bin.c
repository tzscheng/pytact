#include "core.h"

enum {
    BIN_DTYPE_I32 = 1,
    BIN_DTYPE_F64 = 2,
    BIN_DTYPE_UTF8 = 3
};

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t n_chunks;
} bin_header_t;

typedef struct {
    char tag[16];
    uint32_t dtype;
    uint32_t ndim;
    uint64_t shape[4];
    uint64_t nbytes;
} bin_chunk_header_t;

typedef struct bin_model_t {
    int nb, nq, n_shape, n_pair, n_frame, n_feed, y_size, lam_size;
    int integrator, iters;
    double dt, erp, slop, cfm_scale, v_rest_thresh, tol;
    int have_dims, have_sim_f64, have_sim_i32;
    tact_t *h;

    int *parent, *jtype;
    double *X, *I6, *Ti;
    double *ff, *sk, *floss, *armature, *jnt_lo, *jnt_hi, *g;
    int *ctype, *cbody, *craycast, *cpair;
    double *cshape, *ctran, *cparam;
    double *crgba, *view, *light0;
    double *q0, *qd0;
    int *feed_kinds, *feed_offsets, *feed_idx, *fbody;
    double *ftran, *ftran_inv;
    char *frame_names;
    int n_mesh;
    int *mesh_slots;
    char *mesh_paths;
    int n_hfield;
    int n_hfield_offsets, n_hfield_values;
    int *hfield_meta, *hfield_offsets;
    double *hfield_size, *hfield_data;
} bin_model_t;

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

static int shape_is(const bin_chunk_header_t *ch, uint32_t ndim,
                    uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    return ch->ndim == ndim &&
           ch->shape[0] == d0 && ch->shape[1] == d1 &&
           ch->shape[2] == d2 && ch->shape[3] == d3;
}

static int require_dims(const bin_model_t *m)
{
    return m->have_dims ? 0 : -1;
}

static int take_chunk(bin_model_t *m, const char *tag,
                      const bin_chunk_header_t *ch, void *buf)
{
    uint32_t dtype = ch->dtype;
    uint64_t nbytes = ch->nbytes;

    if (strcmp(tag, "dims_i32") == 0) {
        if (m->have_dims || dtype != BIN_DTYPE_I32 ||
            !shape_is(ch, 1, 8, 0, 0, 0) ||
            nbytes != 8 * sizeof(int32_t)) return -1;
        int32_t *v = (int32_t*)buf;
        m->nb = v[0]; m->nq = v[1]; m->n_shape = v[2]; m->n_pair = v[3];
        m->n_frame = v[4]; m->n_feed = v[5]; m->y_size = v[6]; m->lam_size = v[7];
        m->have_dims = 1;
        free(buf); return 0;
    }
    if (strcmp(tag, "sim_f64") == 0) {
        if (m->have_sim_f64 || dtype != BIN_DTYPE_F64 ||
            !shape_is(ch, 1, 6, 0, 0, 0) ||
            nbytes != 6 * sizeof(double)) return -1;
        double *v = (double*)buf;
        m->dt = v[0]; m->erp = v[1]; m->slop = v[2];
        m->cfm_scale = v[3]; m->v_rest_thresh = v[4]; m->tol = v[5];
        m->have_sim_f64 = 1;
        free(buf); return 0;
    }
    if (strcmp(tag, "sim_i32") == 0) {
        if (m->have_sim_i32 || dtype != BIN_DTYPE_I32 ||
            !shape_is(ch, 1, 2, 0, 0, 0) ||
            nbytes != 2 * sizeof(int32_t)) return -1;
        int32_t *v = (int32_t*)buf;
        m->integrator = v[0]; m->iters = v[1];
        m->have_sim_i32 = 1;
        free(buf); return 0;
    }
    if (strcmp(tag, "mesh_slots_i32") == 0) {
        if (m->mesh_slots || dtype != BIN_DTYPE_I32 ||
            !shape_is(ch, 1, ch->shape[0], 0, 0, 0) ||
            nbytes != ch->shape[0] * sizeof(int32_t)) return -1;
        m->n_mesh = (int)(nbytes / sizeof(int32_t));
        m->mesh_slots = (int*)buf;
        return 0;
    }
    if (strcmp(tag, "mesh_paths") == 0) {
        if (m->mesh_paths || dtype != BIN_DTYPE_UTF8 ||
            !shape_is(ch, 1, ch->shape[0], 0, 0, 0) ||
            nbytes != ch->shape[0]) return -1;
        char *s = (char*)malloc((size_t)nbytes + 1);
        if (!s) return -1;
        if (nbytes > 0) memcpy(s, buf, (size_t)nbytes);
        s[nbytes] = '\0';
        free(buf);
        m->mesh_paths = s;
        return 0;
    }
    if (strcmp(tag, "frame_names") == 0) {
        if (m->frame_names || dtype != BIN_DTYPE_UTF8 ||
            !shape_is(ch, 1, ch->shape[0], 0, 0, 0) ||
            nbytes != ch->shape[0]) return -1;
        char *s = (char*)malloc((size_t)nbytes + 1);
        if (!s) return -1;
        if (nbytes > 0) memcpy(s, buf, (size_t)nbytes);
        s[nbytes] = '\0';
        free(buf);
        m->frame_names = s;
        return 0;
    }
    if (strcmp(tag, "hfield_meta_i32") == 0) {
        if (m->hfield_meta || dtype != BIN_DTYPE_I32 ||
            !shape_is(ch, 2, ch->shape[0], 3, 0, 0) ||
            nbytes != ch->shape[0] * 3 * sizeof(int32_t)) return -1;
        m->n_hfield = (int)(nbytes / (3 * sizeof(int32_t)));
        m->hfield_meta = (int*)buf;
        return 0;
    }
    if (strcmp(tag, "hfield_offsets") == 0) {
        if (m->hfield_offsets || dtype != BIN_DTYPE_I32 ||
            !shape_is(ch, 1, ch->shape[0], 0, 0, 0) ||
            nbytes != ch->shape[0] * sizeof(int32_t)) return -1;
        m->n_hfield_offsets = (int)ch->shape[0];
        m->hfield_offsets = (int*)buf;
        return 0;
    }
    if (strcmp(tag, "hfield_size_f64") == 0) {
        if (m->hfield_size || dtype != BIN_DTYPE_F64 ||
            !shape_is(ch, 2, ch->shape[0], 2, 0, 0) ||
            nbytes != ch->shape[0] * 2 * sizeof(double)) return -1;
        m->hfield_size = (double*)buf;
        return 0;
    }
    if (strcmp(tag, "hfield_data_f64") == 0) {
        if (m->hfield_data || dtype != BIN_DTYPE_F64 ||
            !shape_is(ch, 1, ch->shape[0], 0, 0, 0) ||
            nbytes != ch->shape[0] * sizeof(double)) return -1;
        m->n_hfield_values = (int)ch->shape[0];
        m->hfield_data = (double*)buf;
        return 0;
    }

    #define TAKE_I32(name) do { if (strcmp(tag, #name) == 0) { \
        if (require_dims(m) != 0 || m->name || dtype != BIN_DTYPE_I32) return -1; \
        m->name = (int*)buf; return 0; \
    } } while (0)
    #define TAKE_F64(name) do { if (strcmp(tag, #name) == 0) { \
        if (require_dims(m) != 0 || m->name || dtype != BIN_DTYPE_F64) return -1; \
        m->name = (double*)buf; return 0; \
    } } while (0)

    if (strcmp(tag, "parent") == 0 || strcmp(tag, "jtype") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 1, (uint64_t)m->nb, 0, 0, 0) ||
            nbytes != (uint64_t)m->nb * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "X") == 0 || strcmp(tag, "I6") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 3, (uint64_t)m->nb, 6, 6, 0) ||
            nbytes != (uint64_t)m->nb * 36 * sizeof(double)) return -1;
    } else if (strcmp(tag, "Ti") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 3, (uint64_t)m->nb, 4, 4, 0) ||
            nbytes != (uint64_t)m->nb * 16 * sizeof(double)) return -1;
    } else if (strcmp(tag, "ff") == 0 || strcmp(tag, "sk") == 0 ||
               strcmp(tag, "floss") == 0 || strcmp(tag, "armature") == 0 ||
               strcmp(tag, "jnt_lo") == 0 || strcmp(tag, "jnt_hi") == 0 ||
               strcmp(tag, "q0") == 0 || strcmp(tag, "qd0") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 1, (uint64_t)m->nq, 0, 0, 0) ||
            nbytes != (uint64_t)m->nq * sizeof(double)) return -1;
    } else if (strcmp(tag, "g") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 1, 3, 0, 0, 0) ||
            nbytes != 3 * sizeof(double)) return -1;
    } else if (strcmp(tag, "ctype") == 0 || strcmp(tag, "cbody") == 0 ||
               strcmp(tag, "craycast") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 1, (uint64_t)m->n_shape, 0, 0, 0) ||
            nbytes != (uint64_t)m->n_shape * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "cshape") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 2, (uint64_t)m->n_shape, 3, 0, 0) ||
            nbytes != (uint64_t)m->n_shape * 3 * sizeof(double)) return -1;
    } else if (strcmp(tag, "ctran") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 3, (uint64_t)m->n_shape, 4, 4, 0) ||
            nbytes != (uint64_t)m->n_shape * 16 * sizeof(double)) return -1;
    } else if (strcmp(tag, "cparam") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 2, (uint64_t)m->n_shape, 13, 0, 0) ||
            nbytes != (uint64_t)m->n_shape * 13 * sizeof(double)) return -1;
    } else if (strcmp(tag, "crgba") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 2, (uint64_t)m->n_shape, 4, 0, 0) ||
            nbytes != (uint64_t)m->n_shape * 4 * sizeof(double)) return -1;
    } else if (strcmp(tag, "cpair") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 2, (uint64_t)m->n_pair, 2, 0, 0) ||
            nbytes != (uint64_t)m->n_pair * 2 * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "view") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 1, 6, 0, 0, 0) ||
            nbytes != 6 * sizeof(double)) return -1;
    } else if (strcmp(tag, "light0") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 1, 8, 0, 0, 0) ||
            nbytes != 8 * sizeof(double)) return -1;
    } else if (strcmp(tag, "feed_kinds") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 1, (uint64_t)m->n_feed, 0, 0, 0) ||
            nbytes != (uint64_t)m->n_feed * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "feed_offsets") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 1, (uint64_t)m->n_feed + 1, 0, 0, 0) ||
            nbytes != ((uint64_t)m->n_feed + 1) * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "feed_idx") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 1, ch->shape[0], 0, 0, 0) ||
            nbytes != ch->shape[0] * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "fbody") == 0) {
        if (dtype != BIN_DTYPE_I32 || !shape_is(ch, 1, (uint64_t)m->n_frame, 0, 0, 0) ||
            nbytes != (uint64_t)m->n_frame * sizeof(int32_t)) return -1;
    } else if (strcmp(tag, "ftran") == 0 || strcmp(tag, "ftran_inv") == 0) {
        if (dtype != BIN_DTYPE_F64 || !shape_is(ch, 3, (uint64_t)m->n_frame, 4, 4, 0) ||
            nbytes != (uint64_t)m->n_frame * 16 * sizeof(double)) return -1;
    }

    TAKE_I32(parent); TAKE_I32(jtype); TAKE_I32(ctype); TAKE_I32(cbody);
    TAKE_I32(craycast); TAKE_I32(cpair); TAKE_I32(feed_kinds);
    TAKE_I32(feed_offsets); TAKE_I32(feed_idx); TAKE_I32(fbody);

    TAKE_F64(X); TAKE_F64(I6); TAKE_F64(Ti); TAKE_F64(ff); TAKE_F64(sk);
    TAKE_F64(floss); TAKE_F64(armature); TAKE_F64(jnt_lo); TAKE_F64(jnt_hi);
    TAKE_F64(g); TAKE_F64(cshape); TAKE_F64(ctran); TAKE_F64(cparam);
    TAKE_F64(crgba); TAKE_F64(q0); TAKE_F64(qd0); TAKE_F64(view); TAKE_F64(light0);
    TAKE_F64(ftran); TAKE_F64(ftran_inv);

    #undef TAKE_I32
    #undef TAKE_F64

    free(buf);
    return 0;
}

static int model_ready(const bin_model_t *m)
{
    int expect_lam = 0;
    if (m && m->have_dims && m->nq >= 0 && m->n_pair >= 0) {
        expect_lam = 6 * MAX_PTS_PER_PAIR * (m->n_pair > 0 ? m->n_pair : 1) + 2 * m->nq;
    }
    return m && m->have_dims && m->have_sim_f64 && m->have_sim_i32 &&
           m->nb >= 0 && m->nq >= 0 && m->n_shape >= 0 && m->n_pair >= 0 &&
           m->n_frame >= 0 && m->n_feed >= 0 && m->lam_size > 0 &&
           m->lam_size == expect_lam &&
           m->dt > 0.0 &&
           (m->nb == 0 || (m->parent && m->jtype && m->X && m->I6 && m->Ti)) &&
           (m->nq == 0 || (m->ff && m->sk && m->floss && m->armature &&
                           m->jnt_lo && m->jnt_hi && m->q0 && m->qd0)) &&
           (m->n_shape == 0 || (m->ctype && m->cbody && m->cshape && m->ctran && m->cparam)) &&
           (m->n_pair == 0 || m->cpair) &&
           (m->n_feed == 0 || m->feed_kinds) &&
           (m->n_frame == 0 || (m->fbody && m->ftran && m->ftran_inv)) &&
           m->g && m->feed_offsets;
}

static int register_mesh_paths(const bin_model_t *m)
{
    if (!m->mesh_paths) return m->n_mesh == 0 ? 0 : -1;
    const char *p = m->mesh_paths;
    for (int i = 0; i < m->n_mesh; ++i) {
        if (!m->mesh_slots) return -1;
        if (*p == '\0') return -1;
        tact_set_mesh_path(m->mesh_slots[i], p);
        p += strlen(p) + 1;
    }
    if (*p != '\0') return -1;
    return 0;
}

static int register_hfield_data(const bin_model_t *m)
{
    if (m->n_hfield == 0) return 0;
    if (!m->hfield_meta || !m->hfield_offsets || !m->hfield_size || !m->hfield_data) return -1;
    if (m->n_hfield_offsets != m->n_hfield + 1) return -1;
    for (int i = 0; i < m->n_hfield; ++i) {
        int slot = m->hfield_meta[3 * i + 0];
        int nrow = m->hfield_meta[3 * i + 1];
        int ncol = m->hfield_meta[3 * i + 2];
        int lo = m->hfield_offsets[i];
        int hi = m->hfield_offsets[i + 1];
        if (slot < 0 || nrow < 2 || ncol < 2 || lo < 0 || hi < lo ||
            hi > m->n_hfield_values || hi - lo != nrow * ncol) return -1;
        tact_set_hfield_data(slot, nrow, ncol, m->hfield_size[2 * i], m->hfield_size[2 * i + 1],
                        m->hfield_data + lo);
    }
    if (m->hfield_offsets[m->n_hfield] != m->n_hfield_values) return -1;
    return 0;
}

static void destroy_bin_model(bin_model_t *m)
{
    if (!m) return;
    if (m->h) tact_destroy(m->h);
    free_ptr((void**)&m->parent); free_ptr((void**)&m->jtype);
    free_ptr((void**)&m->X); free_ptr((void**)&m->I6); free_ptr((void**)&m->Ti);
    free_ptr((void**)&m->ff); free_ptr((void**)&m->sk); free_ptr((void**)&m->floss);
    free_ptr((void**)&m->armature); free_ptr((void**)&m->jnt_lo); free_ptr((void**)&m->jnt_hi);
    free_ptr((void**)&m->g);
    free_ptr((void**)&m->ctype); free_ptr((void**)&m->cbody); free_ptr((void**)&m->craycast);
    free_ptr((void**)&m->cpair); free_ptr((void**)&m->cshape); free_ptr((void**)&m->ctran);
    free_ptr((void**)&m->cparam); free_ptr((void**)&m->crgba);
    free_ptr((void**)&m->q0); free_ptr((void**)&m->qd0);
    free_ptr((void**)&m->view); free_ptr((void**)&m->light0);
    free_ptr((void**)&m->feed_kinds); free_ptr((void**)&m->feed_offsets);
    free_ptr((void**)&m->feed_idx); free_ptr((void**)&m->fbody);
    free_ptr((void**)&m->ftran); free_ptr((void**)&m->ftran_inv);
    free_ptr((void**)&m->frame_names);
    free_ptr((void**)&m->mesh_slots); free_ptr((void**)&m->mesh_paths);
    free_ptr((void**)&m->hfield_meta); free_ptr((void**)&m->hfield_offsets);
    free_ptr((void**)&m->hfield_size); free_ptr((void**)&m->hfield_data);
    free(m);
}

static int create_handle(bin_model_t *m)
{
    if (!model_ready(m)) return -1;
    int rc = tact_create_from_arrays(
        m->nb, m->parent, m->jtype, m->X, m->I6, m->Ti, m->ff, m->sk,
        m->floss, m->armature, m->jnt_lo, m->jnt_hi, m->g, m->dt, m->integrator,
        m->n_shape, m->n_pair, m->ctype, m->cbody, m->cshape, m->ctran, m->cparam,
        m->craycast, m->cpair, m->erp, m->slop, m->cfm_scale, m->v_rest_thresh,
        m->iters, m->tol, &m->h);
    if (rc != 0) return -1;
    tact_set_feedback(m->h, m->n_feed, m->feed_kinds, m->feed_offsets, m->feed_idx,
                      m->n_frame, m->fbody, m->ftran, m->ftran_inv, m->y_size);
    m->h->q0 = m->q0; m->q0 = NULL;
    m->h->qd0 = m->qd0; m->qd0 = NULL;
    m->h->crgba = m->crgba; m->crgba = NULL;
    m->h->view = m->view; m->view = NULL;
    m->h->light0 = m->light0; m->light0 = NULL;
    m->h->frame_names = m->frame_names; m->frame_names = NULL;
    return 0;
}

int tact_create_from_bin(const char *path, tact_t **out)
{
    if (!path || !out) return -1;
    *out = NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return -2;

    bin_header_t hdr;
    if (read_exact(f, &hdr, sizeof(hdr)) != 0 ||
        memcmp(hdr.magic, "TACTMDL\0", 8) != 0 || hdr.version != 1) {
        fclose(f);
        return -3;
    }

    bin_model_t *m = (bin_model_t*)calloc(1, sizeof(bin_model_t));
    if (!m) { fclose(f); return -4; }

    for (uint32_t i = 0; i < hdr.n_chunks; ++i) {
        bin_chunk_header_t ch;
        if (read_exact(f, &ch, sizeof(ch)) != 0) {
            destroy_bin_model(m); fclose(f); return -5;
        }
        void *buf = NULL;
        if (ch.nbytes > 0) {
            if (ch.nbytes > (uint64_t)SIZE_MAX) {
                destroy_bin_model(m); fclose(f); return -6;
            }
            buf = malloc((size_t)ch.nbytes);
            if (!buf) { destroy_bin_model(m); fclose(f); return -7; }
            if (read_exact(f, buf, (size_t)ch.nbytes) != 0) {
                free(buf); destroy_bin_model(m); fclose(f); return -8;
            }
        }

        char tag[17];
        memcpy(tag, ch.tag, 16);
        tag[16] = '\0';
        char *nul = memchr(tag, '\0', 16);
        if (nul) *nul = '\0';

        if (take_chunk(m, tag, &ch, buf) != 0) {
            if (buf) free(buf);
            destroy_bin_model(m); fclose(f); return -9;
        }
    }
    fclose(f);

    if (!model_ready(m)) {
        destroy_bin_model(m);
        return -10;
    }
    if (register_mesh_paths(m) != 0) {
        destroy_bin_model(m);
        return -11;
    }
    if (register_hfield_data(m) != 0) {
        destroy_bin_model(m);
        return -12;
    }
    if (create_handle(m) != 0) {
        destroy_bin_model(m);
        return -13;
    }

    *out = m->h;
    m->h = NULL;
    destroy_bin_model(m);
    return 0;
}

int tact_frame_count(const tact_t *h)
{
    return (h && h->frame_names) ? h->n_frames : 0;
}

const char *tact_frame_name(const tact_t *h, int frame_id)
{
    if (!h || !h->frame_names || frame_id < 0 || frame_id >= h->n_frames) return NULL;
    const char *p = h->frame_names;
    for (int i = 0; i < frame_id; ++i) p += strlen(p) + 1;
    return *p ? p : NULL;
}

int tact_frame_id(const tact_t *h, const char *name)
{
    if (!h || !h->frame_names || !name) return -1;
    const char *p = h->frame_names;
    for (int i = 0; i < h->n_frames; ++i) {
        if (strcmp(p, name) == 0) return i;
        p += strlen(p) + 1;
    }
    return -1;
}

const double *tact_q0(const tact_t *h)
{
    return h ? h->q0 : NULL;
}

const double *tact_qd0(const tact_t *h)
{
    return h ? h->qd0 : NULL;
}

int tact_render(const tact_t *h, const double *q)
{
    if (!h || !q) return -1;

    int n_obj = h->n_shape;
    size_t n_alloc = (size_t)(n_obj > 0 ? n_obj : 1);
    double *T = (double*)malloc((size_t)(h->nb > 0 ? h->nb : 1) * 16 * sizeof(double));
    float *shape = (float*)calloc(n_alloc * 8, sizeof(float));
    float *objcolor = (float*)malloc(n_alloc * 4 * sizeof(float));
    float *objpose = (float*)malloc(n_alloc * 16 * sizeof(float));
    if (!T || !shape || !objcolor || !objpose) {
        free(T); free(shape); free(objcolor); free(objpose);
        return -2;
    }

    if (h->nb > 0) _fk(T, h->nb, h->Ti, h->parent, h->jtype, (double*)q);

    for (int i = 0; i < n_obj; ++i) {
        for (int k = 0; k < 3; ++k) shape[8*i + k] = (float)h->cshape[3*i + k];

        if (h->crgba) {
            for (int k = 0; k < 4; ++k) objcolor[4*i + k] = (float)h->crgba[4*i + k];
        } else {
            objcolor[4*i + 0] = 0.7f;
            objcolor[4*i + 1] = 0.7f;
            objcolor[4*i + 2] = 0.7f;
            objcolor[4*i + 3] = 1.0f;
        }

        double Tw[16];
        if (h->cbody[i] < 0) memcpy(Tw, h->ctran + 16*i, 16 * sizeof(double));
        else                matmul(Tw, T + 16*h->cbody[i], h->ctran + 16*i, 4, 4, 4);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                objpose[16*i + 4*c + r] = (float)Tw[4*r + c];
            }
        }
    }

    float campose[6] = {0.0f, 0.0f, 0.0f, 3.0f, 45.0f, 20.0f};
    if (h->view) {
        for (int i = 0; i < 6; ++i) campose[i] = (float)h->view[i];
    }

    float light_pos[3] = {7.0f, 7.0f, 7.0f};
    float light_target[3] = {0.0f, 0.0f, 0.0f};
    float light_ortho = 5.0f;
    int shadow_enabled = 1;
    if (h->light0) {
        for (int i = 0; i < 3; ++i) light_pos[i] = (float)h->light0[i];
        for (int i = 0; i < 3; ++i) light_target[i] = (float)h->light0[3 + i];
        light_ortho = (float)h->light0[6];
        shadow_enabled = h->light0[7] != 0.0;
    }
    tact_render_set_light(light_pos, light_target, light_ortho, shadow_enabled);
    int rc = win_render(n_obj, h->ctype, shape, objcolor, objpose, campose);

    free(T);
    free(shape);
    free(objcolor);
    free(objpose);
    return rc;
}
