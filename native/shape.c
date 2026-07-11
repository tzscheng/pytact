/* shape.c — shared shape-asset storage: mesh slot table (vertices/faces loaded
 * from .obj) and height-field slot table (grids pushed from Python). Defines the
 * extern globals declared in shape.h; consumed by mpr.c (mesh support), narrow.c
 * (hfield contact), ray.c (ray casts), render.c. (Split out of ccd.c.) */
#include "core.h"
#include "shape.h"
#include <limits.h>

// Per-slot vertex + face storage. `vertex[idx]` is a malloc'd array of
// `num_vertex[idx]` vec3s; `face[idx]` is a malloc'd array of `num_face[idx]`
// integer triangles (0-indexed into vertex[idx]). Both populated lazily on
// first collision/raycast query by load_mesh(). BSS-zero-initialized.
//
// Vertices feed CCD support functions; faces feed ray_intersects_mesh().
int       num_vertex[MAX_MESH];
double (*vertex[MAX_MESH])[3];
int       num_face[MAX_MESH];
int     (*face  [MAX_MESH])[3];
double    mesh_radius[MAX_MESH];   // cached bounding-sphere radius (about local origin); 0 = uncomputed

// Mesh path table populated by tact_set_mesh_path() from Python during build().
// Indexed by the same slot id stored in cshape[i][0] for mesh shapes (cytpe==100).
// Empty string means "no path registered" — collision/render will error.
char mesh_path[MAX_MESH][MAX_PATH_LEN];

void tact_set_mesh_path(int idx, const char* path) {
    if (idx < 0 || idx >= MAX_MESH) {
        fprintf(stderr, "tact_set_mesh_path: idx %d out of range [0, %d)\n", idx, MAX_MESH);
        return;
    }
    strncpy(mesh_path[idx], path, MAX_PATH_LEN - 1);
    mesh_path[idx][MAX_PATH_LEN - 1] = '\0';
    // Invalidate cached data so the next load picks up the new path. Free old
    // buffers to avoid leaks if Python re-registers the slot.
    if (vertex[idx]) { free(vertex[idx]); vertex[idx] = NULL; }
    if (face  [idx]) { free(face  [idx]); face  [idx] = NULL; }
    num_vertex[idx] = 0;
    num_face  [idx] = 0;
    mesh_radius[idx] = 0.0;  /* invalidate cached bounding-sphere radius */
}

// Per-slot height-field storage. `hf_data[slot]` is a malloc'd array of
// hf_nrow[slot]*hf_ncol[slot] heights (meters), row-major: hf_data[i*ncol + j] is the
// height at grid node (row i along local +Y, col j along local +X). The grid spans local
// [-sx, sx] × [-sy, sy]. min/max height are cached for the bounding-sphere broad phase.
// Pushed directly from Python via tact_set_hfield_data() during build() (no lazy C-side load,
// unlike meshes). Indexed by the slot id stored in cshape[i][0] for hfield shapes
// (ctype==105). hf_data[slot]==NULL means "no grid registered".
int      hf_nrow[MAX_HFIELD];
int      hf_ncol[MAX_HFIELD];
double   hf_sx[MAX_HFIELD], hf_sy[MAX_HFIELD];   // half-extents in local X / Y
double   hf_minh[MAX_HFIELD], hf_maxh[MAX_HFIELD];
double  *hf_data[MAX_HFIELD];

void tact_set_hfield_data(int slot, int nrow, int ncol, double sx, double sy, const double* data) {
    if (slot < 0 || slot >= MAX_HFIELD) {
        fprintf(stderr, "tact_set_hfield_data: slot %d out of range [0, %d)\n", slot, MAX_HFIELD);
        return;
    }
    if (nrow < 2 || ncol < 2) {
        fprintf(stderr, "tact_set_hfield_data: grid %dx%d too small (need >=2 each)\n", nrow, ncol);
        return;
    }
    if (hf_data[slot]) { free(hf_data[slot]); hf_data[slot] = NULL; }  // re-register: drop old grid
    int n = nrow * ncol;
    hf_data[slot] = (double*)malloc((size_t)n * sizeof(double));
    double lo = data[0], hi = data[0];
    for (int k = 0; k < n; k++) {
        double h = data[k];
        hf_data[slot][k] = h;
        if (h < lo) lo = h;
        if (h > hi) hi = h;
    }
    hf_nrow[slot] = nrow;
    hf_ncol[slot] = ncol;
    hf_sx[slot]   = sx;
    hf_sy[slot]   = sy;
    hf_minh[slot] = lo;
    hf_maxh[slot] = hi;
}

double hfield_local_radius(int slot) {
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1.0;
    double az = fabs(hf_minh[slot]) > fabs(hf_maxh[slot]) ? fabs(hf_minh[slot]) : fabs(hf_maxh[slot]);
    double sx = hf_sx[slot], sy = hf_sy[slot];
    return sqrt(sx*sx + sy*sy + az*az);
}

/* Bounding-sphere radius of mesh `slot` about its local origin (max |vertex|), for the
 * raycast broad phase. Loads the mesh on first use and caches the radius per slot
 * (geometry is constant; only the world pose changes per frame). Returns -1 if the slot
 * is invalid or fails to load (caller then treats it as "never cull"). tact_set_mesh_path()
 * invalidates the cache (mesh_radius[idx]=0) when a slot is re-pathed. */
double mesh_local_radius(int slot)
{
    if (slot < 0 || slot >= MAX_MESH) return -1.0;
    if (mesh_radius[slot] > 0.0) return mesh_radius[slot];
    if (num_vertex[slot] == 0) num_vertex[slot] = load_mesh(slot);
    int nv = num_vertex[slot];
    if (nv == 0) return -1.0;
    double r2 = 0.0;
    for (int i = 0; i < nv; i++) {
        double *v = vertex[slot][i];
        double n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        if (n2 > r2) r2 = n2;
    }
    return (mesh_radius[slot] = sqrt(r2));
}

typedef struct {
    double (*v)[3];
    int n;
    int cap;
} obj_verts_t;

typedef struct {
    int (*f)[3];
    int n;
    int cap;
} obj_faces_t;

static void obj_push_vert(obj_verts_t *a, double x, double y, double z)
{
    if (a->n >= a->cap) {
        int new_cap = (a->cap == 0) ? 4096 : a->cap * 2;
        double (*nv)[3] = (double (*)[3])realloc(a->v, (size_t)new_cap * sizeof(*a->v));
        if (!nv) {
            fprintf(stderr, "load_obj: malloc failed while growing vertex array\n");
            free(a->v);
            exit(0);
        }
        a->v = nv;
        a->cap = new_cap;
    }
    a->v[a->n][0] = x;
    a->v[a->n][1] = y;
    a->v[a->n][2] = z;
    a->n++;
}

static void obj_push_face(obj_faces_t *a, int i0, int i1, int i2)
{
    if (a->n >= a->cap) {
        int new_cap = (a->cap == 0) ? 4096 : a->cap * 2;
        int (*nf)[3] = (int (*)[3])realloc(a->f, (size_t)new_cap * sizeof(*a->f));
        if (!nf) {
            fprintf(stderr, "load_obj: malloc failed while growing face array\n");
            free(a->f);
            exit(0);
        }
        a->f = nf;
        a->cap = new_cap;
    }
    a->f[a->n][0] = i0;
    a->f[a->n][1] = i1;
    a->f[a->n][2] = i2;
    a->n++;
}

static int obj_parse_vertex_index(const char *tok, int vcount)
{
    char buf[64];
    int k = 0;
    while (tok[k] && tok[k] != '/' && k < (int)sizeof(buf) - 1) {
        buf[k] = tok[k];
        k++;
    }
    buf[k] = '\0';

    int idx = atoi(buf);
    if (idx == 0) return -1;
    if (idx < 0) idx = vcount + idx;
    else         idx = idx - 1;
    if (idx < 0 || idx >= vcount) return -1;
    return idx;
}

int load_obj(int mesh_idx) {
    // Path must have been registered via tact_set_mesh_path() before any mesh-shape
    // collision/render/raycast call. Python's Model.build() does this after
    // tact_create_from_arrays(). Loads both vertices (for CCD support) and triangle faces
    // (for ray_intersects_mesh). Returns vertex count.
    if (mesh_idx < 0 || mesh_idx >= MAX_MESH || mesh_path[mesh_idx][0] == '\0') {
        fprintf(stderr, "load_obj: no path registered for mesh slot %d "
                        "(was YAML missing `file:` on the mesh shape?)\n", mesh_idx);
        exit(0);
    }
    FILE* file = fopen(mesh_path[mesh_idx], "r");
    if (!file) {
        fprintf(stderr, "load_obj: cannot open %s (mesh slot %d)\n",
                mesh_path[mesh_idx], mesh_idx);
        exit(0);
    }

    if (vertex[mesh_idx]) free(vertex[mesh_idx]);
    if (face  [mesh_idx]) free(face  [mesh_idx]);
    vertex[mesh_idx] = NULL;
    face[mesh_idx] = NULL;
    num_face[mesh_idx] = 0;

    obj_verts_t verts = {0};
    obj_faces_t faces = {0};
    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\r' || *s == '\n' || *s == '\0') continue;

        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) {
            double x, y, z;
            if (sscanf(s + 1, "%lf %lf %lf", &x, &y, &z) == 3) {
                obj_push_vert(&verts, x, y, z);
            }
        } else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            int idx[128];
            int n = 0;
            char *tok = strtok(s + 1, " \t\r\n");
            for (; tok && n < (int)(sizeof(idx) / sizeof(idx[0])); tok = strtok(NULL, " \t\r\n")) {
                int vi = obj_parse_vertex_index(tok, verts.n);
                if (vi >= 0) idx[n++] = vi;
            }
            if (n >= 3) {
                for (int i = 1; i + 1 < n; i++) {
                    obj_push_face(&faces, idx[0], idx[i], idx[i + 1]);
                }
            }
        }
    }
    fclose(file);

    vertex[mesh_idx] = verts.v;
    face[mesh_idx] = faces.f;
    num_face[mesh_idx] = faces.n;
    return verts.n;
}

static int load_stl(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= MAX_MESH || mesh_path[mesh_idx][0] == '\0') {
        fprintf(stderr, "load_stl: no path registered for mesh slot %d\n", mesh_idx);
        exit(0);
    }
    FILE* file = fopen(mesh_path[mesh_idx], "rb");
    if (!file) {
        fprintf(stderr, "load_stl: cannot open %s (mesh slot %d)\n",
                mesh_path[mesh_idx], mesh_idx);
        exit(0);
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "load_stl: cannot seek %s\n", mesh_path[mesh_idx]);
        fclose(file);
        exit(0);
    }
    long fsize = ftell(file);
    if (fsize < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "load_stl: cannot measure %s\n", mesh_path[mesh_idx]);
        fclose(file);
        exit(0);
    }

    unsigned char header[84];
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        fprintf(stderr, "load_stl: truncated header in %s\n", mesh_path[mesh_idx]);
        fclose(file);
        exit(0);
    }
    uint32_t ntri = ((uint32_t)header[80]) |
                    ((uint32_t)header[81] << 8) |
                    ((uint32_t)header[82] << 16) |
                    ((uint32_t)header[83] << 24);
    if ((uint64_t)fsize != 84ull + 50ull * (uint64_t)ntri) {
        fprintf(stderr, "load_stl: unsupported or malformed STL %s "
                        "(only binary STL is supported)\n", mesh_path[mesh_idx]);
        fclose(file);
        exit(0);
    }
    if (ntri > (uint32_t)(INT_MAX / 3)) {
        fprintf(stderr, "load_stl: too many triangles in %s\n", mesh_path[mesh_idx]);
        fclose(file);
        exit(0);
    }

    if (vertex[mesh_idx]) free(vertex[mesh_idx]);
    if (face  [mesh_idx]) free(face  [mesh_idx]);
    vertex[mesh_idx] = (double (*)[3])malloc((size_t)ntri * 3 * sizeof(*vertex[mesh_idx]));
    face[mesh_idx]   = (int    (*)[3])malloc((size_t)ntri * sizeof(*face[mesh_idx]));
    if (!vertex[mesh_idx] || !face[mesh_idx]) {
        fprintf(stderr, "load_stl: malloc failed for %u triangles (slot %d)\n", ntri, mesh_idx);
        fclose(file);
        exit(0);
    }

    for (uint32_t t = 0; t < ntri; t++) {
        unsigned char rec[50];
        if (fread(rec, 1, sizeof(rec), file) != sizeof(rec)) {
            fprintf(stderr, "load_stl: truncated triangle data in %s\n", mesh_path[mesh_idx]);
            fclose(file);
            exit(0);
        }
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                uint32_t u = ((uint32_t)rec[12 + 12*j + 4*k]) |
                             ((uint32_t)rec[13 + 12*j + 4*k] << 8) |
                             ((uint32_t)rec[14 + 12*j + 4*k] << 16) |
                             ((uint32_t)rec[15 + 12*j + 4*k] << 24);
                float x;
                memcpy(&x, &u, sizeof(x));
                vertex[mesh_idx][3*t + j][k] = (double)x;
            }
        }
        face[mesh_idx][t][0] = (int)(3*t + 0);
        face[mesh_idx][t][1] = (int)(3*t + 1);
        face[mesh_idx][t][2] = (int)(3*t + 2);
    }

    fclose(file);
    num_face[mesh_idx] = (int)ntri;
    return (int)(3 * ntri);
}

static int has_ext(const char *path, const char *ext)
{
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    while (*dot && *ext) {
        char a = *dot++;
        char b = *ext++;
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return *dot == '\0' && *ext == '\0';
}

int load_mesh(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= MAX_MESH || mesh_path[mesh_idx][0] == '\0') {
        fprintf(stderr, "load_mesh: no path registered for mesh slot %d\n", mesh_idx);
        exit(0);
    }
    if (has_ext(mesh_path[mesh_idx], ".obj")) return load_obj(mesh_idx);
    if (has_ext(mesh_path[mesh_idx], ".stl")) return load_stl(mesh_idx);
    fprintf(stderr, "load_mesh: unsupported mesh extension in %s\n", mesh_path[mesh_idx]);
    exit(0);
}
