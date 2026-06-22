#ifndef SHAPE_H
#define SHAPE_H
/* shape.h — shared shape-asset storage (the "공용 데이터" of the collision side).
 * Slot tables for meshes (.obj vertices/faces) and height fields (grids), defined
 * once in shape.c and read across modules: mpr.c (mesh support function), narrow.c
 * (hfield contact), ray.c (ray casts), render.c (GPU mesh build). Function
 * prototypes for the storage API (set_mesh_path/set_hfield_data/load_obj/
 * mesh_local_radius/hfield_local_radius) live in tact.h. */

#define MAX_MESH        64
#define MAX_HFIELD      16
#define MAX_PATH_LEN    256

/* Per-slot mesh storage. `vertex[idx]`/`face[idx]` are malloc'd by load_mesh() on
 * first use; `mesh_path[idx]` is set by set_mesh_path() from Python at build time. */
extern int      num_vertex[MAX_MESH];
extern double (*vertex[MAX_MESH])[3];
extern int      num_face[MAX_MESH];
extern int    (*face  [MAX_MESH])[3];
extern double   mesh_radius[MAX_MESH];                 /* cached bounding-sphere radius */
extern char     mesh_path[MAX_MESH][MAX_PATH_LEN];

/* Per-slot height-field storage. Pushed from Python via set_hfield_data().
 * `hf_data[slot]` is hf_nrow*hf_ncol heights (row-major, meters). */
extern int      hf_nrow[MAX_HFIELD];
extern int      hf_ncol[MAX_HFIELD];
extern double   hf_sx[MAX_HFIELD], hf_sy[MAX_HFIELD];  /* half-extents in local X / Y */
extern double   hf_minh[MAX_HFIELD], hf_maxh[MAX_HFIELD];
extern double  *hf_data[MAX_HFIELD];

#endif /* SHAPE_H */
