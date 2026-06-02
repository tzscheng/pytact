/* shape.c — shared shape-asset storage: mesh slot table (vertices/faces loaded
 * from .obj) and height-field slot table (grids pushed from Python). Defines the
 * extern globals declared in shape.h; consumed by mpr.c (mesh support), narrow.c
 * (hfield contact), ray.c (ray casts), render.c. (Split out of ccd.c.) */
#include "tact.h"
#include "shape.h"

// Per-slot vertex + face storage. `vertex[idx]` is a malloc'd array of
// `num_vertex[idx]` vec3s; `face[idx]` is a malloc'd array of `num_face[idx]`
// integer triangles (0-indexed into vertex[idx]). Both populated lazily on
// first collision/raycast query by load_obj(). BSS-zero-initialized.
//
// Vertices feed CCD support functions; faces feed ray_intersects_mesh().
int       num_vertex[MAX_MESH];
double (*vertex[MAX_MESH])[3];
int       num_face[MAX_MESH];
int     (*face  [MAX_MESH])[3];
double    mesh_radius[MAX_MESH];   // cached bounding-sphere radius (about local origin); 0 = uncomputed

// Mesh path table populated by set_mesh_path() from Python during build().
// Indexed by the same slot id stored in cshape[i][0] for mesh shapes (cytpe==100).
// Empty string means "no path registered" — collision/render will error.
char mesh_path[MAX_MESH][MAX_PATH_LEN];

void set_mesh_path(int idx, const char* path) {
    if (idx < 0 || idx >= MAX_MESH) {
        fprintf(stderr, "set_mesh_path: idx %d out of range [0, %d)\n", idx, MAX_MESH);
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
// Pushed directly from Python via set_hfield_data() during build() (no lazy C-side load,
// unlike meshes). Indexed by the slot id stored in cshape[i][0] for hfield shapes
// (ctype==105). hf_data[slot]==NULL means "no grid registered".
int      hf_nrow[MAX_HFIELD];
int      hf_ncol[MAX_HFIELD];
double   hf_sx[MAX_HFIELD], hf_sy[MAX_HFIELD];   // half-extents in local X / Y
double   hf_minh[MAX_HFIELD], hf_maxh[MAX_HFIELD];
double  *hf_data[MAX_HFIELD];

void set_hfield_data(int slot, int nrow, int ncol, double sx, double sy, const double* data) {
    if (slot < 0 || slot >= MAX_HFIELD) {
        fprintf(stderr, "set_hfield_data: slot %d out of range [0, %d)\n", slot, MAX_HFIELD);
        return;
    }
    if (nrow < 2 || ncol < 2) {
        fprintf(stderr, "set_hfield_data: grid %dx%d too small (need >=2 each)\n", nrow, ncol);
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
 * is invalid or fails to load (caller then treats it as "never cull"). set_mesh_path()
 * invalidates the cache (mesh_radius[idx]=0) when a slot is re-pathed. */
double mesh_local_radius(int slot)
{
    if (slot < 0 || slot >= MAX_MESH) return -1.0;
    if (mesh_radius[slot] > 0.0) return mesh_radius[slot];
    if (num_vertex[slot] == 0) num_vertex[slot] = load_obj(slot);
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

int load_obj(int mesh_idx) {
    // Path must have been registered via set_mesh_path() before any mesh-shape
    // collision/render/raycast call. Python's Model.build() does this after
    // tact_create(). Loads both vertices (for CCD support) and triangle faces
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

    // Two-pass: count `v ` and `f ` lines first to size both allocs exactly,
    // then re-read and fill. Avoids realloc growth.
    char line[256];
    int nv = 0, nf = 0;
    while (fgets(line, sizeof(line), file)) {
        if      (strncmp(line, "v ", 2) == 0) nv++;
        else if (strncmp(line, "f ", 2) == 0) nf++;
    }
    rewind(file);

    if (vertex[mesh_idx]) free(vertex[mesh_idx]);
    if (face  [mesh_idx]) free(face  [mesh_idx]);
    vertex[mesh_idx] = (double (*)[3]) malloc((size_t)nv * 3 * sizeof(double));
    face  [mesh_idx] = (nf > 0) ? (int (*)[3]) malloc((size_t)nf * 3 * sizeof(int)) : NULL;
    if (!vertex[mesh_idx] || (nf > 0 && !face[mesh_idx])) {
        fprintf(stderr, "load_obj: malloc failed for %d verts / %d faces (slot %d)\n",
                nv, nf, mesh_idx);
        fclose(file);
        exit(0);
    }

    double tmp[3];
    int vi[3];
    int vcount = 0, fcount = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            if (sscanf(line + 2, "%lf %lf %lf", &tmp[0], &tmp[1], &tmp[2]) == 3) {
                vertex[mesh_idx][vcount][0] = tmp[0];
                vertex[mesh_idx][vcount][1] = tmp[1];
                vertex[mesh_idx][vcount][2] = tmp[2];
                vcount++;
            }
        } else if (strncmp(line, "f ", 2) == 0) {
            /* Accept both `f a b c` and `f a/vt/vn ...`: split on whitespace and take the
             * leading vertex index of each of the first 3 tokens (atoi stops at '/').
             * The old sscanf("%d%*[/0-9] ...") failed on plain `f a b c` — the suppressed
             * scanset needs >=1 char but hit the space, so plain-format OBJs parsed 0 faces
             * and were invisible to mesh raycast. Convert 1-indexed → 0-indexed. */
            char *tok = strtok(line + 2, " \t\r\n");
            int got = 0;
            for (; tok && got < 3; tok = strtok(NULL, " \t\r\n"))
                vi[got++] = atoi(tok) - 1;
            if (got == 3) {
                face[mesh_idx][fcount][0] = vi[0];
                face[mesh_idx][fcount][1] = vi[1];
                face[mesh_idx][fcount][2] = vi[2];
                fcount++;
            }
        }
    }
    fclose(file);
    num_face[mesh_idx] = fcount;
    return vcount;
}
