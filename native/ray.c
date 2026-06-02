/* ray.c — ray-primitive intersections (triangle/mesh/hfield/box/sphere/cylinder/
 * capsule). Each returns forward distance t along Rd, or -1 on miss. Mesh/hfield
 * variants read shape.c's slot storage (transform the ray into shape-local frame
 * first). Mirrors rbd.py:ray_intersects_*. (Split out of ccd.c.) */
#include "tact.h"
#include "shape.h"

/* ============================================================================
 * Ray-primitive intersections — mirrors rbd.py:ray_intersects_* (verified
 * against 27 analytical + 82 cross-validation cases). All functions return
 * the forward distance t along Rd (so hit point = R0 + t*Rd), or -1 on miss
 * (parallel, behind ray, outside surface bounds, etc.).
 *
 * Conventions preserved from the Python originals:
 *   - sphere assumes |Rd|=1 (other primitives are |Rd|-scale-invariant)
 *   - box returns -1 when ray origin is inside (t_min<0); slab algorithm
 *   - triangle/mesh are double-sided (no backface culling)
 *   - cylinder/capsule cap intersections clamp to the radial test
 *
 * `ray_intersects_mesh_slot` consumes shape.c's global vertex/face storage by slot
 * index; transform the ray into mesh-local frame before calling (cheaper than
 * transforming every vertex).
 * ============================================================================ */
#define RAY_EPS 1e-6

double ray_intersects_triangle(const double *R0, const double *Rd,
                               const double *v0, const double *v1, const double *v2)
{
    double e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    double e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    double h[3];  cross3(Rd[0],Rd[1],Rd[2], e2[0],e2[1],e2[2], h);
    double a = e1[0]*h[0] + e1[1]*h[1] + e1[2]*h[2];
    if (a > -RAY_EPS && a < RAY_EPS) return -1.0;  /* parallel */
    double f = 1.0 / a;
    double s[3] = {R0[0]-v0[0], R0[1]-v0[1], R0[2]-v0[2]};
    double u = f * (s[0]*h[0] + s[1]*h[1] + s[2]*h[2]);
    if (u < 0.0 || u > 1.0) return -1.0;
    double q[3];  cross3(s[0],s[1],s[2], e1[0],e1[1],e1[2], q);
    double v = f * (Rd[0]*q[0] + Rd[1]*q[1] + Rd[2]*q[2]);
    if (v < 0.0 || u + v > 1.0) return -1.0;
    double t = f * (e2[0]*q[0] + e2[1]*q[1] + e2[2]*q[2]);
    return (t > RAY_EPS) ? t : -1.0;
}

double ray_intersects_mesh_slot(const double *R0, const double *Rd, int mesh_idx)
{
    if (mesh_idx < 0 || mesh_idx >= MAX_MESH) return -1.0;
    if (num_vertex[mesh_idx] == 0) num_vertex[mesh_idx] = load_obj(mesh_idx);
    int nf = num_face[mesh_idx];
    if (nf == 0) return -1.0;
    double best = -1.0;
    for (int i = 0; i < nf; i++) {
        int a = face[mesh_idx][i][0], b = face[mesh_idx][i][1], c = face[mesh_idx][i][2];
        double t = ray_intersects_triangle(R0, Rd, vertex[mesh_idx][a],
                                                   vertex[mesh_idx][b],
                                                   vertex[mesh_idx][c]);
        if (t > 0.0 && (best < 0.0 || t < best)) best = t;
    }
    return best;
}

/* Ray vs height-field slot. R0/Rd in the hfield's local frame (grid in local XY,
 * height along +Z). 2D DDA grid traversal (Amanatides-Woo): walks only the cells the
 * ray crosses, in increasing-t order, so the first cell that yields a triangle hit gives
 * the nearest hit (early-exit). The setup clips the ray to the hfield's 3D AABB
 * ([-sx,sx]×[-sy,sy]×[min_h,max_h]), which both rejects rays that miss the footprint and
 * trims the traversal to the height slab. O(cells-crossed) ≈ O(grid side) per ray vs the
 * old O(cells) brute force. Returns nearest forward t along Rd, or -1.0 on miss. */
double ray_intersects_hfield(const double *R0, const double *Rd, int slot)
{
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1.0;
    int nr = hf_nrow[slot], nc = hf_ncol[slot];
    const double *H = hf_data[slot];
    double sx = hf_sx[slot], sy = hf_sy[slot];
    double dx = 2.0*sx/(nc-1), dy = 2.0*sy/(nr-1);

    /* Clip ray to the 3D AABB → [tmin, tmax]. Folds in footprint + height-slab culling. */
    double lo[3] = {-sx, -sy, hf_minh[slot]};
    double hi[3] = { sx,  sy, hf_maxh[slot]};
    double tmin = 0.0, tmax = 1e300;     /* forward ray: t >= 0 */
    for (int d = 0; d < 3; d++) {
        if (fabs(Rd[d]) < 1e-12) {
            if (R0[d] < lo[d] || R0[d] > hi[d]) return -1.0;   /* parallel & outside slab */
        } else {
            double inv = 1.0/Rd[d];
            double t1 = (lo[d]-R0[d])*inv, t2 = (hi[d]-R0[d])*inv;
            if (t1 > t2) { double tmp=t1; t1=t2; t2=tmp; }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return -1.0;
        }
    }

    /* Entry point in XY → starting cell (clamped to the [0, n-2] cell range). */
    double ex = R0[0] + tmin*Rd[0];
    double ey = R0[1] + tmin*Rd[1];
    int j = (int)floor((ex + sx)/dx);
    int i = (int)floor((ey + sy)/dy);
    if (j < 0) j = 0; else if (j > nc-2) j = nc-2;
    if (i < 0) i = 0; else if (i > nr-2) i = nr-2;

    /* Test cell (I,J)'s two triangles; fold the nearer hit into `best`. */
    #define HF_TEST_CELL(I,J) do {                                                          \
        double y0_=-sy+(I)*dy, y1_=-sy+((I)+1)*dy, x0_=-sx+(J)*dx, x1_=-sx+((J)+1)*dx;       \
        double p00[3]={x0_,y0_,H[(I)*nc+(J)]},     p10[3]={x1_,y0_,H[(I)*nc+(J)+1]};         \
        double p01[3]={x0_,y1_,H[((I)+1)*nc+(J)]}, p11[3]={x1_,y1_,H[((I)+1)*nc+(J)+1]};     \
        double tt_;                                                                         \
        tt_ = ray_intersects_triangle(R0,Rd,p00,p10,p11);                                   \
        if (tt_ > 0.0 && (best < 0.0 || tt_ < best)) best = tt_;                             \
        tt_ = ray_intersects_triangle(R0,Rd,p00,p11,p01);                                   \
        if (tt_ > 0.0 && (best < 0.0 || tt_ < best)) best = tt_;                             \
    } while(0)

    double best = -1.0;

    /* (Near-)vertical ray: XY projection is a point → test the 2×2 cell block around
     * (i,j). FP in floor((x+sx)/dx) vs the reconstructed vertex -sx+j*dx can place a
     * grid-aligned query in cell (j-1)/(i-1) instead of (j)/(i); only cells whose
     * footprint contains (x,y) can be hit, so this block is brute-force-equivalent and
     * never spuriously hits a non-containing cell. (Oblique rays self-heal by stepping
     * into the adjacent cell, so only this single-point path needs the redundancy.) */
    if (fabs(Rd[0]) < 1e-12 && fabs(Rd[1]) < 1e-12) {
        int i0 = (i > 0) ? i-1 : i, j0 = (j > 0) ? j-1 : j;
        for (int ii = i0; ii <= i; ii++)
            for (int jj = j0; jj <= j; jj++)
                HF_TEST_CELL(ii, jj);
        return best;
    }

    /* DDA setup over the XY grid. */
    int stepX = (Rd[0] > 0.0) ? 1 : ((Rd[0] < 0.0) ? -1 : 0);
    int stepY = (Rd[1] > 0.0) ? 1 : ((Rd[1] < 0.0) ? -1 : 0);
    double tDeltaX = (stepX != 0) ? fabs(dx/Rd[0]) : 1e300;
    double tDeltaY = (stepY != 0) ? fabs(dy/Rd[1]) : 1e300;
    double tMaxX = (stepX > 0) ? ((-sx+(j+1)*dx)-R0[0])/Rd[0]
                 : (stepX < 0) ? ((-sx+ j   *dx)-R0[0])/Rd[0] : 1e300;
    double tMaxY = (stepY > 0) ? ((-sy+(i+1)*dy)-R0[1])/Rd[1]
                 : (stepY < 0) ? ((-sy+ i   *dy)-R0[1])/Rd[1] : 1e300;

    for (;;) {
        HF_TEST_CELL(i, j);
        double tcell_exit = (tMaxX < tMaxY) ? tMaxX : tMaxY;   /* far boundary of this cell */
        if (best >= 0.0 && best <= tcell_exit) break;          /* nearest hit is in this cell */
        if (tMaxX < tMaxY) { j += stepX; tMaxX += tDeltaX; }
        else               { i += stepY; tMaxY += tDeltaY; }
        if (i < 0 || i > nr-2 || j < 0 || j > nc-2) break;     /* left the grid */
        if (tcell_exit > tmax) break;                          /* past the AABB exit (height slab) */
    }
    #undef HF_TEST_CELL
    return best;
}

double ray_intersects_box(const double *R0, const double *Rd,
                          const double *center, const double *R, const double *hs)
{
    /* R is 3x3 row-major. axis[i] = R[:, i] = (R[0+i], R[3+i], R[6+i]). */
    double t_min = -1e300, t_max = 1e300;
    double p[3] = {center[0]-R0[0], center[1]-R0[1], center[2]-R0[2]};
    for (int i = 0; i < 3; i++) {
        double ax[3] = {R[i], R[3+i], R[6+i]};
        double e = ax[0]*p[0]  + ax[1]*p[1]  + ax[2]*p[2];
        double f = ax[0]*Rd[0] + ax[1]*Rd[1] + ax[2]*Rd[2];
        if (f > RAY_EPS || f < -RAY_EPS) {
            double t1 = (e + hs[i]) / f;
            double t2 = (e - hs[i]) / f;
            if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }
            if (t1 > t_min) t_min = t1;
            if (t2 < t_max) t_max = t2;
            if (t_min > t_max) return -1.0;
        } else {
            /* ray parallel to slab; origin must be inside slab (|e| ≤ hs) */
            if (-e - hs[i] > 0.0 || -e + hs[i] < 0.0) return -1.0;
        }
    }
    return (t_min < 0.0) ? -1.0 : t_min;
}

double ray_intersects_sphere(const double *R0, const double *Rd,
                             const double *C, double r)
{
    /* Assumes |Rd|=1 (caller responsibility). */
    double L[3]  = {C[0]-R0[0], C[1]-R0[1], C[2]-R0[2]};
    double tca   = L[0]*Rd[0] + L[1]*Rd[1] + L[2]*Rd[2];
    double L2    = L[0]*L[0] + L[1]*L[1] + L[2]*L[2];
    double d2    = L2 - tca*tca;
    double r2    = r*r;
    if (d2 > r2) return -1.0;
    double thc   = sqrt(r2 - d2);
    double t0    = tca - thc;
    double t1    = tca + thc;
    if (t0 >= 0.0) return t0;
    if (t1 >= 0.0) return t1;
    return -1.0;
}

double ray_intersects_cylinder(const double *R0, const double *Rd,
                               const double *P1, const double *P2, double r)
{
    double axis[3] = {P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]};
    double L       = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (L < RAY_EPS) return -1.0;
    double v[3] = {axis[0]/L, axis[1]/L, axis[2]/L};
    double m[3] = {R0[0]-P1[0], R0[1]-P1[1], R0[2]-P1[2]};
    double md   = m[0]*v[0] + m[1]*v[1] + m[2]*v[2];
    double nd   = Rd[0]*v[0] + Rd[1]*v[1] + Rd[2]*v[2];
    double a    = (Rd[0]*Rd[0] + Rd[1]*Rd[1] + Rd[2]*Rd[2]) - nd*nd;
    double b    = (Rd[0]*m[0] + Rd[1]*m[1] + Rd[2]*m[2]) - nd*md;
    double c    = (m[0]*m[0] + m[1]*m[1] + m[2]*m[2]) - md*md - r*r;
    double best = -1.0;
    if (a > RAY_EPS*RAY_EPS) {
        double disc = b*b - a*c;
        if (disc >= 0.0) {
            double sd = sqrt(disc);
            double t12[2] = {(-b - sd)/a, (-b + sd)/a};
            for (int k = 0; k < 2; k++) {
                double t = t12[k];
                if (t < 0.0) continue;
                double hp[3] = {R0[0]+t*Rd[0], R0[1]+t*Rd[1], R0[2]+t*Rd[2]};
                double h = (hp[0]-P1[0])*v[0] + (hp[1]-P1[1])*v[1] + (hp[2]-P1[2])*v[2];
                if (h >= 0.0 && h <= L && (best < 0.0 || t < best)) best = t;
            }
        }
    }
    /* caps: plane ⊥ v at P1 and P2 */
    const double *caps[2] = {P1, P2};
    for (int k = 0; k < 2; k++) {
        double denom = Rd[0]*v[0] + Rd[1]*v[1] + Rd[2]*v[2];
        if (denom > RAY_EPS || denom < -RAY_EPS) {
            double dp[3] = {caps[k][0]-R0[0], caps[k][1]-R0[1], caps[k][2]-R0[2]};
            double t = (dp[0]*v[0] + dp[1]*v[1] + dp[2]*v[2]) / denom;
            if (t < 0.0) continue;
            double hp[3]  = {R0[0]+t*Rd[0], R0[1]+t*Rd[1], R0[2]+t*Rd[2]};
            double rel[3] = {hp[0]-caps[k][0], hp[1]-caps[k][1], hp[2]-caps[k][2]};
            double axdot  = rel[0]*v[0] + rel[1]*v[1] + rel[2]*v[2];
            double rad[3] = {rel[0]-axdot*v[0], rel[1]-axdot*v[1], rel[2]-axdot*v[2]};
            double rad2   = rad[0]*rad[0] + rad[1]*rad[1] + rad[2]*rad[2];
            if (rad2 <= r*r && (best < 0.0 || t < best)) best = t;
        }
    }
    return best;
}

double ray_intersects_capsule(const double *R0, const double *Rd,
                              const double *P1, const double *P2, double r)
{
    double axis[3] = {P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]};
    double L       = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (L < RAY_EPS) return ray_intersects_sphere(R0, Rd, P1, r);
    double v[3] = {axis[0]/L, axis[1]/L, axis[2]/L};
    double m[3] = {R0[0]-P1[0], R0[1]-P1[1], R0[2]-P1[2]};
    double md   = m[0]*v[0] + m[1]*v[1] + m[2]*v[2];
    double nd   = Rd[0]*v[0] + Rd[1]*v[1] + Rd[2]*v[2];
    double a    = (Rd[0]*Rd[0] + Rd[1]*Rd[1] + Rd[2]*Rd[2]) - nd*nd;
    double b    = (Rd[0]*m[0] + Rd[1]*m[1] + Rd[2]*m[2]) - nd*md;
    double c    = (m[0]*m[0] + m[1]*m[1] + m[2]*m[2]) - md*md - r*r;
    double best = -1.0;
    if (a > RAY_EPS*RAY_EPS) {
        double disc = b*b - a*c;
        if (disc >= 0.0) {
            double sd = sqrt(disc);
            double t12[2] = {(-b - sd)/a, (-b + sd)/a};
            for (int k = 0; k < 2; k++) {
                double t = t12[k];
                if (t < 0.0) continue;
                double hp[3] = {R0[0]+t*Rd[0], R0[1]+t*Rd[1], R0[2]+t*Rd[2]};
                double h = (hp[0]-P1[0])*v[0] + (hp[1]-P1[1])*v[1] + (hp[2]-P1[2])*v[2];
                if (h >= 0.0 && h <= L && (best < 0.0 || t < best)) best = t;
            }
        }
    }
    /* hemispherical caps at P1, P2 */
    const double *caps[2] = {P1, P2};
    for (int k = 0; k < 2; k++) {
        double t = ray_intersects_sphere(R0, Rd, caps[k], r);
        if (t >= 0.0 && (best < 0.0 || t < best)) best = t;
    }
    return best;
}
