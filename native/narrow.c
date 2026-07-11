/* narrow.c — narrow-phase contact dispatch + dedicated analytic detectors.
 * tact_collision_check routes each shape pair to tact_box_box_manifold (box_box.c, SAT +
 * clipping), the analytic box-sphere / sphere-hfield / box-hfield detectors here,
 * or the MPR fallback (mpr.c) for the remaining convex pairs. out[] layout +
 * param1->param2 normal convention: see tact.h. */
#include "core.h"
#include "shape.h"

/* Analytical box-sphere narrowphase. A sphere has no flat feature, so box-sphere
 * is always a single contact point — faster and more robust than MPR. Common for
 * walker robots (sphere foot on box ground).
 *   box_param = [pos(3), euler_xyz(3), half-extents(3)]
 *   sph_param = [pos(3), euler_xyz(3), radius, _, _]   (orientation unused)
 * Writes out[7] = [contact point (world, on box surface), unit normal (world,
 * pointing box→sphere), depth]. Returns 1 = penetrating, 0 = touching, -1 = miss. */
static int box_sphere_contact(double *box_param, double *sph_param, double *out)
{
    double R[9];
    euler_to_rotation(box_param + 3, R, "xyz");   /* box orientation, row-major */
    const double *cb = box_param;                 /* box center (world) */
    const double *h  = box_param + 6;             /* half-extents */
    const double *cs = sph_param;                 /* sphere center (world) */
    double r = sph_param[6];

    /* sphere center in box-local frame: p = Rᵀ (cs − cb) */
    double d[3] = { cs[0]-cb[0], cs[1]-cb[1], cs[2]-cb[2] };
    double p[3] = { R[0]*d[0] + R[3]*d[1] + R[6]*d[2],
                    R[1]*d[0] + R[4]*d[1] + R[7]*d[2],
                    R[2]*d[0] + R[5]*d[1] + R[8]*d[2] };

    /* closest point on the box (local), clamped to the half-extents */
    double q[3];
    for (int i = 0; i < 3; i++)
        q[i] = p[i] < -h[i] ? -h[i] : (p[i] > h[i] ? h[i] : p[i]);

    double e[3] = { p[0]-q[0], p[1]-q[1], p[2]-q[2] };
    double dist2 = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];

    double n_local[3], depth;
    if (dist2 > 1e-18) {
        /* sphere center outside the box (the common case) */
        double dist = sqrt(dist2);
        if (dist > r) return -1;                  /* separated */
        n_local[0] = e[0]/dist; n_local[1] = e[1]/dist; n_local[2] = e[2]/dist;
        depth = r - dist;
    } else {
        /* sphere center inside the box: push out along the least-penetrated face */
        int axis = 0; double best = h[0] - fabs(p[0]);
        for (int i = 1; i < 3; i++) { double pen = h[i] - fabs(p[i]); if (pen < best) { best = pen; axis = i; } }
        n_local[0] = n_local[1] = n_local[2] = 0.0;
        n_local[axis] = (p[axis] >= 0.0) ? 1.0 : -1.0;
        q[axis]       = n_local[axis] * h[axis];  /* project contact onto that face */
        depth = r + best;
    }

    /* world-frame: cp = R·q + cb,  n = R·n_local */
    out[0] = R[0]*q[0] + R[1]*q[1] + R[2]*q[2] + cb[0];
    out[1] = R[3]*q[0] + R[4]*q[1] + R[5]*q[2] + cb[1];
    out[2] = R[6]*q[0] + R[7]*q[1] + R[8]*q[2] + cb[2];
    out[3] = R[0]*n_local[0] + R[1]*n_local[1] + R[2]*n_local[2];
    out[4] = R[3]*n_local[0] + R[4]*n_local[1] + R[5]*n_local[2];
    out[5] = R[6]*n_local[0] + R[7]*n_local[1] + R[8]*n_local[2];
    out[6] = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Analytical sphere-sphere narrowphase. Spheres have no flat features, so contact
 * is a single point on the line between centers — fully closed-form, no iteration.
 *   sph1_param = [pos(3), euler_xyz(3), radius, _, _]   (orientation unused)
 *   sph2_param = [pos(3), euler_xyz(3), radius, _, _]   (orientation unused)
 * Writes out[7] = [contact point (world, on sphere1 surface), unit normal (world,
 * pointing sphere1 → sphere2), depth]. Returns 1 = penetrating, 0 = touching, -1 = miss.
 *
 * Degenerate case: when centers coincide (dist² < 1e-18), the normal direction is
 * undefined; we pick +x to avoid a NaN and let the solver push them apart on the
 * next step. depth = r1+r2 in that case. */
static int sphere_sphere_contact(double *sph1_param, double *sph2_param, double *out)
{
    const double *c1 = sph1_param;
    const double *c2 = sph2_param;
    double r1 = sph1_param[6];
    double r2 = sph2_param[6];

    double d[3]  = { c2[0]-c1[0], c2[1]-c1[1], c2[2]-c1[2] };
    double dist2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    double rsum  = r1 + r2;
    if (dist2 > rsum*rsum) return -1;            /* separated */

    double n[3], depth;
    if (dist2 > 1e-18) {
        double dist = sqrt(dist2);
        n[0] = d[0]/dist; n[1] = d[1]/dist; n[2] = d[2]/dist;
        depth = rsum - dist;
    } else {
        n[0] = 1.0; n[1] = 0.0; n[2] = 0.0;      /* centers coincide: arbitrary axis */
        depth = rsum;
    }

    /* Contact point on sphere1's surface (matches the param1-surface convention used
     * by box_sphere_contact / sphere_hf_contact). */
    out[0] = c1[0] + r1*n[0];
    out[1] = c1[1] + r1*n[1];
    out[2] = c1[2] + r1*n[2];
    out[3] = n[0]; out[4] = n[1]; out[5] = n[2];
    out[6] = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Analytical capsule-sphere narrowphase. A capsule's surface is the locus of points
 * at distance r_cap from its axis segment, so contact with a sphere reduces to a
 * closest-point-on-segment query followed by sphere-sphere logic with the segment
 * point treated as a "sphere center" of radius r_cap. Closed-form, no iteration.
 *   cap_param = [pos(3), euler_xyz(3), r_cap, hh, _]   (axis = capsule-local +z;
 *                                                       hh = half-length of cylindrical section)
 *   sph_param = [pos(3), euler_xyz(3), radius, _, _]   (orientation unused)
 * Writes out[7] = [contact point (world, on capsule surface), unit normal (world,
 * pointing capsule → sphere), depth]. Returns 1 = penetrating, 0 = touching, -1 = miss.
 *
 * Degenerate case: sphere center lies exactly on the capsule axis (dist² < 1e-18) —
 * the radial normal is undefined; pick the capsule's local +x to avoid NaN. */
static int capsule_sphere_contact(double *cap_param, double *sph_param, double *out)
{
    double R[9];
    euler_to_rotation(cap_param + 3, R, "xyz");
    const double *cc = cap_param;     /* capsule center (world) */
    const double *cs = sph_param;     /* sphere center (world) */
    double r_cap = cap_param[6];
    double hh    = cap_param[7];
    double r_sph = sph_param[6];

    /* Capsule axis direction = +z column of R (row-major: rows 0/3/6 give col indices) */
    double z[3] = { R[2], R[5], R[8] };

    /* Project sphere center onto capsule axis: t = (cs - cc) · z, clamp to [-hh, +hh].
     * Beyond the cylindrical section the closest axis point is just the endpoint, which
     * makes the rest of the math behave like a sphere-sphere test against the hemisphere
     * cap centered there — exactly what we want. */
    double d[3] = { cs[0]-cc[0], cs[1]-cc[1], cs[2]-cc[2] };
    double t = d[0]*z[0] + d[1]*z[1] + d[2]*z[2];
    if (t >  hh) t =  hh;
    if (t < -hh) t = -hh;

    /* Closest point on capsule's axis segment to sphere center */
    double q[3] = { cc[0] + t*z[0], cc[1] + t*z[1], cc[2] + t*z[2] };

    /* From axis point to sphere center */
    double e[3] = { cs[0]-q[0], cs[1]-q[1], cs[2]-q[2] };
    double dist2 = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    double rsum  = r_cap + r_sph;
    if (dist2 > rsum*rsum) return -1;     /* separated */

    double n[3], depth;
    if (dist2 > 1e-18) {
        double dist = sqrt(dist2);
        n[0] = e[0]/dist; n[1] = e[1]/dist; n[2] = e[2]/dist;
        depth = rsum - dist;
    } else {
        /* sphere center on axis: pick capsule-local +x to avoid NaN */
        n[0] = R[0]; n[1] = R[3]; n[2] = R[6];
        depth = rsum;
    }

    /* Contact point on the capsule's surface (cylindrical side or one of the cap hemispheres) */
    out[0] = q[0] + r_cap*n[0];
    out[1] = q[1] + r_cap*n[1];
    out[2] = q[2] + r_cap*n[2];
    out[3] = n[0]; out[4] = n[1]; out[5] = n[2];
    out[6] = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Analytical capsule-capsule narrowphase. Both surfaces are loci of points at
 * distance r_cap from their axis segments, so contact reduces to closest-pair
 * on two line segments (Ericson, Real-Time Collision Detection §5.1.9) followed
 * by sphere-sphere logic with the closest segment points treated as "sphere
 * centers" of radii r1, r2. Closed-form, no iteration.
 *   cap1/cap2_param = [pos(3), euler_xyz(3), r_cap, hh, _]   (axis = local +z)
 * Writes out[7] = [contact point (world, on cap1 surface), unit normal (world,
 * pointing cap1 → cap2), depth]. Returns 1 = penetrating, 0 = touching, -1 = miss.
 *
 * Closest-pair derivation: minimize F(s, t) = |u + s·d1 − t·d2|² with u = c1 − c2,
 * unit axes d1, d2, and (s, t) ∈ [−hh1, hh1] × [−hh2, hh2]. ∂F/∂s = ∂F/∂t = 0 gives
 * s_opt = (b·q − p) / (1 − b²), t_opt = s·b + q  where b = d1·d2, p = u·d1, q = u·d2.
 * Standard clamp-and-resolve handles boundary cases; parallel axes (|b|≈1, denom≈0)
 * collapse the line of solutions and the t-clamp pass alone picks an optimal pair.
 *
 * Degenerate case: closest points on the two axes coincide (dist² < 1e-18) — the
 * radial normal is undefined; pick cap1-local +x to avoid NaN. */
static int capsule_capsule_contact(double *cap1_param, double *cap2_param, double *out)
{
    double R1[9], R2[9];
    euler_to_rotation(cap1_param + 3, R1, "xyz");
    euler_to_rotation(cap2_param + 3, R2, "xyz");
    const double *c1 = cap1_param;
    const double *c2 = cap2_param;
    double r1 = cap1_param[6], hh1 = cap1_param[7];
    double r2 = cap2_param[6], hh2 = cap2_param[7];

    /* Axis directions (unit, world): +z column of each R */
    double d1[3] = { R1[2], R1[5], R1[8] };
    double d2[3] = { R2[2], R2[5], R2[8] };

    double u[3] = { c1[0]-c2[0], c1[1]-c2[1], c1[2]-c2[2] };
    double b = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2];
    double p = u[0]*d1[0] + u[1]*d1[1] + u[2]*d1[2];
    double q = u[0]*d2[0] + u[1]*d2[1] + u[2]*d2[2];
    double denom = 1.0 - b*b;

    /* Pass 1: pick s from the unconstrained optimum (or s=0 for parallel axes); then
     *         compute the matching t. */
    double s, t;
    if (denom < 1e-12) {
        s = 0.0;
        t = q;          /* t_opt at s=0 */
    } else {
        s = (b*q - p) / denom;
        if (s >  hh1) s =  hh1;
        if (s < -hh1) s = -hh1;
        t = s*b + q;
    }
    /* Pass 2: clamp t to its segment and re-solve s if t was clipped (yields the
     *         constrained optimum; second clamp on s catches the diagonal corner). */
    if (t > hh2) {
        t = hh2;
        s = t*b - p;
        if (s >  hh1) s =  hh1;
        if (s < -hh1) s = -hh1;
    } else if (t < -hh2) {
        t = -hh2;
        s = t*b - p;
        if (s >  hh1) s =  hh1;
        if (s < -hh1) s = -hh1;
    }

    /* Closest points on each axis */
    double qA[3] = { c1[0] + s*d1[0], c1[1] + s*d1[1], c1[2] + s*d1[2] };
    double qB[3] = { c2[0] + t*d2[0], c2[1] + t*d2[1], c2[2] + t*d2[2] };

    /* Sphere-sphere on (qA, r1) and (qB, r2) — normal = qB − qA = cap1 → cap2 */
    double e[3]  = { qB[0]-qA[0], qB[1]-qA[1], qB[2]-qA[2] };
    double dist2 = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    double rsum  = r1 + r2;
    if (dist2 > rsum*rsum) return -1;

    double n[3], depth;
    if (dist2 > 1e-18) {
        double dist = sqrt(dist2);
        n[0] = e[0]/dist; n[1] = e[1]/dist; n[2] = e[2]/dist;
        depth = rsum - dist;
    } else {
        n[0] = R1[0]; n[1] = R1[3]; n[2] = R1[6];   /* axes coincide: cap1 local +x */
        depth = rsum;
    }

    /* Contact point on cap1's surface */
    out[0] = qA[0] + r1*n[0];
    out[1] = qA[1] + r1*n[1];
    out[2] = qA[2] + r1*n[2];
    out[3] = n[0]; out[4] = n[1]; out[5] = n[2];
    out[6] = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Analytical cylinder-sphere narrowphase. A finite cylinder has three external
 * contact regions for a sphere: the curved side, the two flat end caps, and the two
 * circular edge rings where side meets cap. Plus an internal "sphere center inside
 * cylinder" case that pushes out along the least-penetrated face. Closed-form
 * classification + clamp in the cylinder's local frame; no iteration.
 *   cyl_param = [pos(3), euler_xyz(3), r_cyl, hh, _]   (axis = local +z, hh = half-height)
 *   sph_param = [pos(3), euler_xyz(3), r_sph, _, _]    (orientation unused)
 * Writes out[7] = [contact point (world, on cylinder surface), unit normal (world,
 * pointing cylinder → sphere), depth]. Returns 1 = penetrating, 0 = touching, -1 = miss.
 *
 * Region classification (cylinder-local, with rho = √(px² + py²)):
 *   |z| ≤ hh, rho ≤ r       INSIDE — push out along least-penetrated face (side or cap)
 *   |z| ≤ hh, rho >  r       beside the curved side — closest point on side
 *   |z| >  hh, rho ≤ r       above/below a flat cap — closest point on cap face
 *   |z| >  hh, rho >  r       outside the corner — closest point on the edge ring
 *
 * Degenerate cases:
 *   - Sphere on cylinder axis AND inside (rho < 1e-9, the side-push branch): pick
 *     cylinder-local +x to avoid division by zero.
 *   - Sphere center exactly on the edge ring (dist² < 1e-18 in the edge-ring branch):
 *     fall through to depth = r_sph + tiny; normal computed via the radial direction. */
static int cylinder_sphere_contact(double *cyl_param, double *sph_param, double *out)
{
    double R[9];
    euler_to_rotation(cyl_param + 3, R, "xyz");
    const double *cc = cyl_param;
    const double *cs = sph_param;
    double r_cyl = cyl_param[6];
    double hh    = cyl_param[7];
    double r_sph = sph_param[6];

    /* sphere center in cylinder-local frame: p = Rᵀ (cs - cc) */
    double d[3] = { cs[0]-cc[0], cs[1]-cc[1], cs[2]-cc[2] };
    double p[3] = { R[0]*d[0] + R[3]*d[1] + R[6]*d[2],
                    R[1]*d[0] + R[4]*d[1] + R[7]*d[2],
                    R[2]*d[0] + R[5]*d[1] + R[8]*d[2] };
    double rho2 = p[0]*p[0] + p[1]*p[1];
    double rho  = sqrt(rho2);
    double z    = p[2];
    double az   = fabs(z);

    double q[3], n_local[3], depth;
    int outside_axial  = (az > hh);
    int outside_radial = (rho > r_cyl);

    if (outside_axial && outside_radial) {
        /* Edge-ring case: closest point on the circle at z=±hh, radius r_cyl */
        double z_face  = (z > 0.0) ? hh : -hh;
        double inv_rho = 1.0 / rho;          /* rho > r_cyl > 0, safe */
        q[0] = r_cyl * p[0] * inv_rho;
        q[1] = r_cyl * p[1] * inv_rho;
        q[2] = z_face;
        double drho = rho - r_cyl;
        double dz   = az  - hh;
        double dist2 = drho*drho + dz*dz;
        if (dist2 > r_sph*r_sph) return -1;
        double dist = sqrt(dist2);
        double inv  = (dist > 1e-9) ? (1.0/dist) : 0.0;
        n_local[0] = (p[0]-q[0]) * inv;
        n_local[1] = (p[1]-q[1]) * inv;
        n_local[2] = (p[2]-q[2]) * inv;
        depth = r_sph - dist;
    } else if (outside_axial) {
        /* Above/below a flat cap — closest point projected onto cap face */
        double z_face = (z > 0.0) ? hh : -hh;
        double dz     = az - hh;
        if (dz > r_sph) return -1;
        q[0] = p[0]; q[1] = p[1]; q[2] = z_face;
        n_local[0] = 0.0; n_local[1] = 0.0; n_local[2] = (z > 0.0) ? 1.0 : -1.0;
        depth = r_sph - dz;
    } else if (outside_radial) {
        /* Beside the curved side */
        double inv_rho = 1.0 / rho;          /* rho > r_cyl > 0, safe */
        double drho    = rho - r_cyl;
        if (drho > r_sph) return -1;
        q[0] = r_cyl * p[0] * inv_rho;
        q[1] = r_cyl * p[1] * inv_rho;
        q[2] = z;
        n_local[0] = p[0]*inv_rho; n_local[1] = p[1]*inv_rho; n_local[2] = 0.0;
        depth = r_sph - drho;
    } else {
        /* Sphere center INSIDE the cylinder — push out along least-penetrated face */
        double pen_side = r_cyl - rho;
        double pen_top  = hh - z;
        double pen_bot  = z + hh;
        if (pen_side <= pen_top && pen_side <= pen_bot) {
            if (rho > 1e-9) {
                double inv_rho = 1.0 / rho;
                n_local[0] = p[0]*inv_rho; n_local[1] = p[1]*inv_rho; n_local[2] = 0.0;
            } else {
                n_local[0] = 1.0; n_local[1] = 0.0; n_local[2] = 0.0;   /* on axis */
            }
            q[0] = r_cyl * n_local[0]; q[1] = r_cyl * n_local[1]; q[2] = z;
            depth = r_sph + pen_side;
        } else if (pen_top <= pen_bot) {
            n_local[0] = 0.0; n_local[1] = 0.0; n_local[2] = 1.0;
            q[0] = p[0]; q[1] = p[1]; q[2] = hh;
            depth = r_sph + pen_top;
        } else {
            n_local[0] = 0.0; n_local[1] = 0.0; n_local[2] = -1.0;
            q[0] = p[0]; q[1] = p[1]; q[2] = -hh;
            depth = r_sph + pen_bot;
        }
    }

    /* world: cp = R·q + cc, n = R·n_local (cylinder surface → sphere center) */
    out[0] = R[0]*q[0] + R[1]*q[1] + R[2]*q[2] + cc[0];
    out[1] = R[3]*q[0] + R[4]*q[1] + R[5]*q[2] + cc[1];
    out[2] = R[6]*q[0] + R[7]*q[1] + R[8]*q[2] + cc[2];
    out[3] = R[0]*n_local[0] + R[1]*n_local[1] + R[2]*n_local[2];
    out[4] = R[3]*n_local[0] + R[4]*n_local[1] + R[5]*n_local[2];
    out[5] = R[6]*n_local[0] + R[7]*n_local[1] + R[8]*n_local[2];
    out[6] = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Analytical box-capsule narrowphase. A capsule's surface is the locus of points at
 * distance r_cap from its axis segment, so contact reduces to "sphere-box at points
 * along the capsule axis" with radius r_cap.
 *   box_param = [pos(3), euler_xyz(3), half-extents(3)]
 *   cap_param = [pos(3), euler_xyz(3), r_cap, hh, _]    (axis = local +z, hh = half-length
 *                                                        of cylindrical section)
 * Writes out[7·k] = [contact point (world, on box surface), unit normal (world, pointing
 * box → capsule), depth] for k ∈ [0, n_points). Returns n_points ≥ 1 = penetrating,
 * 0 = touching, -1 = miss.
 *
 * Strategy: sphere-box at three samples — the capsule endpoints (t = ±hh) and the segment
 * point closest to the box center (t* = clamp(−c_loc·z_loc, ±hh)) — then choose between:
 *   (A) Footprint-clip 2-point: when the center anchor's normal is axis-aligned (a box
 *       face) and the capsule axis is perpendicular to that normal (parallel to the face),
 *       clip the segment range [-hh, +hh] to the box footprint on the OTHER two in-face
 *       axes. The two clipped t-values become sub_id 0/1. Handles the long-capsule case
 *       where endpoints fall outside the box footprint and the contact patch is interior
 *       to the segment.
 *   (B) Endpoint 2-point: both endpoints contact the box with nearly-parallel normals
 *       (n_A · n_B > 0.99). Slight-tilt fallback for cases (A) rejected.
 *   (C) Single deepest: corner/edge/penetrating case, or 2-point checks both fail.
 *
 * The 3-sample-and-pick-deepest approach replaced a fragile piecewise-quadratic search
 * over the segment — at infinite precision the latter still picks "closest to box SURFACE"
 * rather than "deepest INSIDE box" when the segment penetrates, missing the maximum-pen
 * point. Sampling endpoints + center anchor catches both cases robustly.
 *
 * Limitation: in rare diagonal-grazing configurations where the deepest contact lies in
 * the segment interior between samples (e.g., long capsule passing diagonally through a
 * box's top-edge region), depth can be under-estimated by ~15%; the LCP still pushes
 * apart correctly and converges over subsequent steps. */
static int box_capsule_contact(double *box_param, double *cap_param, double *out, int max_pts)
{
    double Rb[9], Rc[9];
    euler_to_rotation(box_param + 3, Rb, "xyz");
    euler_to_rotation(cap_param + 3, Rc, "xyz");
    const double *cb = box_param;
    const double *cc = cap_param;
    const double *h  = box_param + 6;
    double r_cap = cap_param[6];
    double hh    = cap_param[7];

    /* Capsule center and axis in box-local frame: c_loc = Rbᵀ(cc − cb), z_loc = RbᵀRc[:,2] */
    double dc[3] = { cc[0]-cb[0], cc[1]-cb[1], cc[2]-cb[2] };
    double c_loc[3] = { Rb[0]*dc[0]+Rb[3]*dc[1]+Rb[6]*dc[2],
                        Rb[1]*dc[0]+Rb[4]*dc[1]+Rb[7]*dc[2],
                        Rb[2]*dc[0]+Rb[5]*dc[1]+Rb[8]*dc[2] };
    double zw[3]    = { Rc[2], Rc[5], Rc[8] };   /* capsule +z column (world) */
    double z_loc[3] = { Rb[0]*zw[0]+Rb[3]*zw[1]+Rb[6]*zw[2],
                        Rb[1]*zw[0]+Rb[4]*zw[1]+Rb[7]*zw[2],
                        Rb[2]*zw[0]+Rb[5]*zw[1]+Rb[8]*zw[2] };

    /* Three sample t-values along segment: endpoints A (k=0), B (k=1), and center anchor (k=2) */
    double t_center = -(c_loc[0]*z_loc[0] + c_loc[1]*z_loc[1] + c_loc[2]*z_loc[2]);
    if (t_center >  hh) t_center =  hh;
    if (t_center < -hh) t_center = -hh;
    double ts[3] = { -hh, +hh, t_center };

    /* Per-sample results (box-local frame) */
    double q_arr[3][3], n_arr[3][3], d_arr[3] = {0, 0, 0};
    int    valid[3]   = { 0, 0, 0 };

    for (int k = 0; k < 3; k++) {
        double t = ts[k];
        double p[3] = { c_loc[0]+t*z_loc[0], c_loc[1]+t*z_loc[1], c_loc[2]+t*z_loc[2] };
        double q[3] = { p[0], p[1], p[2] };
        for (int i = 0; i < 3; i++) {
            if (q[i] >  h[i]) q[i] =  h[i];
            if (q[i] < -h[i]) q[i] = -h[i];
        }
        double e[3] = { p[0]-q[0], p[1]-q[1], p[2]-q[2] };
        double dist2 = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];

        double n_local[3], depth;
        if (dist2 > 1e-18) {
            double dist = sqrt(dist2);
            if (dist > r_cap) continue;   /* this sample misses */
            n_local[0] = e[0]/dist; n_local[1] = e[1]/dist; n_local[2] = e[2]/dist;
            depth = r_cap - dist;
        } else {
            int axis = 0; double best_pen = h[0] - fabs(p[0]);
            for (int i = 1; i < 3; i++) {
                double pen = h[i] - fabs(p[i]);
                if (pen < best_pen) { best_pen = pen; axis = i; }
            }
            n_local[0] = n_local[1] = n_local[2] = 0.0;
            n_local[axis] = (p[axis] >= 0.0) ? 1.0 : -1.0;
            q[axis] = n_local[axis] * h[axis];
            depth = r_cap + best_pen;
        }
        q_arr[k][0]=q[0]; q_arr[k][1]=q[1]; q_arr[k][2]=q[2];
        n_arr[k][0]=n_local[0]; n_arr[k][1]=n_local[1]; n_arr[k][2]=n_local[2];
        d_arr[k] = depth;
        valid[k] = 1;
    }

    if (!valid[0] && !valid[1] && !valid[2]) return -1;

    /* 2-point manifold via footprint clipping (handles long-capsule case where endpoints
     * fall outside the box footprint). When the center anchor has a face-aligned normal
     * (one ±1 component) AND the capsule axis is perpendicular to that normal (parallel
     * to the face), the contact patch is a 1-D segment in the face plane. We find its
     * extent by clipping the capsule's t-range to the box footprint on the OTHER two
     * axes; the two clipped t-values become sub_id 0/1 contact points. */
    if (max_pts >= 2 && valid[2]) {
        int    face_axis = -1;
        double face_sign = 0.0;
        for (int i = 0; i < 3; i++) {
            if (fabs(n_arr[2][i]) > 0.999) {
                face_axis = i;
                face_sign = (n_arr[2][i] > 0.0) ? +1.0 : -1.0;
                break;
            }
        }
        if (face_axis >= 0 && fabs(z_loc[face_axis]) < 1e-3) {
            /* Capsule axis parallel to the contact face — clip [-hh, +hh] to the
             * box footprint on the two in-face axes. */
            double t_lo = -hh, t_hi = +hh;
            int    feasible = 1;
            for (int i = 0; i < 3; i++) {
                if (i == face_axis) continue;
                if (fabs(z_loc[i]) > 1e-12) {
                    double t1 = ( h[i] - c_loc[i]) / z_loc[i];
                    double t2 = (-h[i] - c_loc[i]) / z_loc[i];
                    double a = t1 < t2 ? t1 : t2;
                    double b = t1 < t2 ? t2 : t1;
                    if (a > t_lo) t_lo = a;
                    if (b < t_hi) t_hi = b;
                } else if (c_loc[i] >= h[i] || c_loc[i] <= -h[i]) {
                    /* Axis stuck outside footprint on this axis — no parallel contact */
                    feasible = 0;
                    break;
                }
            }
            if (feasible && t_lo < t_hi - 1e-9) {
                double pts_t[2] = { t_lo, t_hi };          /* sub_id 0 = lower t */
                for (int k = 0; k < 2; k++) {
                    double t = pts_t[k];
                    double p[3] = { c_loc[0]+t*z_loc[0], c_loc[1]+t*z_loc[1], c_loc[2]+t*z_loc[2] };
                    double q[3] = { p[0], p[1], p[2] };
                    q[face_axis] = face_sign * h[face_axis];
                    double dist = fabs(p[face_axis] - q[face_axis]);
                    double depth = r_cap - dist;
                    if (depth < 0.0) depth = 0.0;
                    double n_local[3] = {0, 0, 0};
                    n_local[face_axis] = face_sign;
                    double *o = out + 7*k;
                    o[0] = Rb[0]*q[0] + Rb[1]*q[1] + Rb[2]*q[2] + cb[0];
                    o[1] = Rb[3]*q[0] + Rb[4]*q[1] + Rb[5]*q[2] + cb[1];
                    o[2] = Rb[6]*q[0] + Rb[7]*q[1] + Rb[8]*q[2] + cb[2];
                    o[3] = Rb[0]*n_local[0] + Rb[1]*n_local[1] + Rb[2]*n_local[2];
                    o[4] = Rb[3]*n_local[0] + Rb[4]*n_local[1] + Rb[5]*n_local[2];
                    o[5] = Rb[6]*n_local[0] + Rb[7]*n_local[1] + Rb[8]*n_local[2];
                    o[6] = depth;
                }
                return 2;
            }
        }
    }

    /* Endpoint-based 2-point fallback (slight-tilt cases the parallel check rejected):
     * both endpoints have valid contacts with nearly-parallel normals. */
    if (max_pts >= 2 && valid[0] && valid[1]) {
        double dot_n = n_arr[0][0]*n_arr[1][0] + n_arr[0][1]*n_arr[1][1] + n_arr[0][2]*n_arr[1][2];
        if (dot_n > 0.99) {
            for (int k = 0; k < 2; k++) {
                double *o = out + 7*k;
                o[0] = Rb[0]*q_arr[k][0] + Rb[1]*q_arr[k][1] + Rb[2]*q_arr[k][2] + cb[0];
                o[1] = Rb[3]*q_arr[k][0] + Rb[4]*q_arr[k][1] + Rb[5]*q_arr[k][2] + cb[1];
                o[2] = Rb[6]*q_arr[k][0] + Rb[7]*q_arr[k][1] + Rb[8]*q_arr[k][2] + cb[2];
                o[3] = Rb[0]*n_arr[k][0] + Rb[1]*n_arr[k][1] + Rb[2]*n_arr[k][2];
                o[4] = Rb[3]*n_arr[k][0] + Rb[4]*n_arr[k][1] + Rb[5]*n_arr[k][2];
                o[5] = Rb[6]*n_arr[k][0] + Rb[7]*n_arr[k][1] + Rb[8]*n_arr[k][2];
                o[6] = d_arr[k];
            }
            return 2;
        }
    }

    /* Single deepest sample (corner/edge/penetrating case) */
    int best_k = 0;
    double best_d = -1e300;
    for (int k = 0; k < 3; k++) {
        if (valid[k] && d_arr[k] > best_d) { best_d = d_arr[k]; best_k = k; }
    }
    out[0] = Rb[0]*q_arr[best_k][0] + Rb[1]*q_arr[best_k][1] + Rb[2]*q_arr[best_k][2] + cb[0];
    out[1] = Rb[3]*q_arr[best_k][0] + Rb[4]*q_arr[best_k][1] + Rb[5]*q_arr[best_k][2] + cb[1];
    out[2] = Rb[6]*q_arr[best_k][0] + Rb[7]*q_arr[best_k][1] + Rb[8]*q_arr[best_k][2] + cb[2];
    out[3] = Rb[0]*n_arr[best_k][0] + Rb[1]*n_arr[best_k][1] + Rb[2]*n_arr[best_k][2];
    out[4] = Rb[3]*n_arr[best_k][0] + Rb[4]*n_arr[best_k][1] + Rb[5]*n_arr[best_k][2];
    out[5] = Rb[6]*n_arr[best_k][0] + Rb[7]*n_arr[best_k][1] + Rb[8]*n_arr[best_k][2];
    out[6] = best_d;
    return best_d > 0.0 ? 1 : 0;
}


/* Closest point on triangle (a,b,c) to point p, via Voronoi-region tests
 * (Ericson, Real-Time Collision Detection §5.1.5). Writes the closest point to out. */
static void closest_point_on_triangle(const double *p, const double *a, const double *b, const double *c, double *out)
{
    double ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    double ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    double ap[3] = {p[0]-a[0], p[1]-a[1], p[2]-a[2]};
    double d1 = ab[0]*ap[0]+ab[1]*ap[1]+ab[2]*ap[2];
    double d2 = ac[0]*ap[0]+ac[1]*ap[1]+ac[2]*ap[2];
    if (d1 <= 0.0 && d2 <= 0.0) { out[0]=a[0]; out[1]=a[1]; out[2]=a[2]; return; }       /* vertex a */

    double bp[3] = {p[0]-b[0], p[1]-b[1], p[2]-b[2]};
    double d3 = ab[0]*bp[0]+ab[1]*bp[1]+ab[2]*bp[2];
    double d4 = ac[0]*bp[0]+ac[1]*bp[1]+ac[2]*bp[2];
    if (d3 >= 0.0 && d4 <= d3) { out[0]=b[0]; out[1]=b[1]; out[2]=b[2]; return; }         /* vertex b */

    double vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {                                            /* edge ab */
        double v = d1 / (d1 - d3);
        out[0]=a[0]+v*ab[0]; out[1]=a[1]+v*ab[1]; out[2]=a[2]+v*ab[2]; return;
    }

    double cp[3] = {p[0]-c[0], p[1]-c[1], p[2]-c[2]};
    double d5 = ab[0]*cp[0]+ab[1]*cp[1]+ab[2]*cp[2];
    double d6 = ac[0]*cp[0]+ac[1]*cp[1]+ac[2]*cp[2];
    if (d6 >= 0.0 && d5 <= d6) { out[0]=c[0]; out[1]=c[1]; out[2]=c[2]; return; }         /* vertex c */

    double vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {                                            /* edge ac */
        double w = d2 / (d2 - d6);
        out[0]=a[0]+w*ac[0]; out[1]=a[1]+w*ac[1]; out[2]=a[2]+w*ac[2]; return;
    }

    double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0) {                                  /* edge bc */
        double w = (d4-d3) / ((d4-d3)+(d5-d6));
        out[0]=b[0]+w*(c[0]-b[0]); out[1]=b[1]+w*(c[1]-b[1]); out[2]=b[2]+w*(c[2]-b[2]); return;
    }

    double denom = 1.0/(va+vb+vc);                                                       /* inside face */
    double v = vb*denom, w = vc*denom;
    out[0]=a[0]+ab[0]*v+ac[0]*w;
    out[1]=a[1]+ab[1]*v+ac[1]*w;
    out[2]=a[2]+ab[2]*v+ac[2]*w;
}

/* Core sphere-hfield narrowphase, computed entirely in the hfield's local frame.
 * Takes the sphere center already transformed to hfield-local (p_local), the radius,
 * and the hfield slot; emits the contact in hfield-local frame (cp_local, n_local,
 * depth). Pulled out so capsule_hf_contact can call this 3× without redoing the
 * hfield rotation matrix or the world↔local transforms.
 * Returns 1 = penetrating, 0 = touching, -1 = miss / outside grid.
 *
 * Tier-1 limitations (inherited from sphere_hf_contact): contact is found only while
 * the sphere center stays within `r` of the surface (deeper than that tunnels — no
 * signed/inside test); the normal is forced to local +Z when the center is at/below
 * the surface (heightfield "outside" is the +Z side), which keeps a resting foot
 * stable but is approximate against near-vertical features. */
static int sphere_hf_local(const double *p, double r, int slot,
                           double *cp_local, double *n_local, double *depth_out)
{
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1;
    int nr = hf_nrow[slot], nc = hf_ncol[slot];
    const double *H = hf_data[slot];
    double sx = hf_sx[slot], sy = hf_sy[slot];
    double dx = 2.0*sx/(nc-1), dy = 2.0*sy/(nr-1);

    /* cell range overlapping the sphere's XY footprint, clamped to the grid */
    int j0 = (int)floor((p[0]-r+sx)/dx), j1 = (int)floor((p[0]+r+sx)/dx);
    int i0 = (int)floor((p[1]-r+sy)/dy), i1 = (int)floor((p[1]+r+sy)/dy);
    if (j0 < 0) j0 = 0;
    if (j1 > nc-2) j1 = nc-2;
    if (i0 < 0) i0 = 0;
    if (i1 > nr-2) i1 = nr-2;
    if (j0 > j1 || i0 > i1) return -1;

    double best_d2 = r*r;                        /* only surface points within r matter */
    double best_q[3] = {0,0,0};
    int hit = 0;
    for (int i = i0; i <= i1; i++) {
        double y0 = -sy + i*dy, y1 = -sy + (i+1)*dy;
        for (int j = j0; j <= j1; j++) {
            double x0 = -sx + j*dx, x1 = -sx + (j+1)*dx;
            double p00[3] = { x0, y0, H[ i   *nc + j  ] };
            double p10[3] = { x1, y0, H[ i   *nc + j+1] };
            double p01[3] = { x0, y1, H[(i+1)*nc + j  ] };
            double p11[3] = { x1, y1, H[(i+1)*nc + j+1] };
            double q[3], e[3], dd;
            closest_point_on_triangle(p, p00, p10, p11, q);   /* tri A */
            e[0]=p[0]-q[0]; e[1]=p[1]-q[1]; e[2]=p[2]-q[2];
            dd = e[0]*e[0]+e[1]*e[1]+e[2]*e[2];
            if (dd < best_d2) { best_d2=dd; best_q[0]=q[0]; best_q[1]=q[1]; best_q[2]=q[2]; hit=1; }
            closest_point_on_triangle(p, p00, p11, p01, q);   /* tri B */
            e[0]=p[0]-q[0]; e[1]=p[1]-q[1]; e[2]=p[2]-q[2];
            dd = e[0]*e[0]+e[1]*e[1]+e[2]*e[2];
            if (dd < best_d2) { best_d2=dd; best_q[0]=q[0]; best_q[1]=q[1]; best_q[2]=q[2]; hit=1; }
        }
    }
    if (!hit) return -1;

    double dist = sqrt(best_d2);
    double depth = r - dist;
    if (dist > 1e-9) {
        n_local[0] = (p[0]-best_q[0])/dist;
        n_local[1] = (p[1]-best_q[1])/dist;
        n_local[2] = (p[2]-best_q[2])/dist;
        if (n_local[2] <= 0.0) {                 /* at/below surface → push straight up */
            n_local[0] = 0.0; n_local[1] = 0.0; n_local[2] = 1.0;
        }
    } else {
        n_local[0] = 0.0; n_local[1] = 0.0; n_local[2] = 1.0;
    }
    cp_local[0] = best_q[0];
    cp_local[1] = best_q[1];
    cp_local[2] = best_q[2];
    *depth_out = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Analytical sphere-hfield narrowphase. Thin wrapper: transform sphere center to
 * hfield-local frame, call sphere_hf_local, transform contact back to world.
 *   sph_param = [pos(3), euler_xyz(3), radius, _, _]  (orientation unused)
 *   hf_param  = [pos(3), euler_xyz(3), slot,   _, _]
 * Writes out[7] = [contact point (world, on terrain surface), unit normal (world,
 * pointing terrain→sphere), depth]. Returns 1 = penetrating, 0 = touching, -1 = miss. */
static int sphere_hf_contact(double *sph_param, double *hf_param, double *out)
{
    int slot = (int)hf_param[6];
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1;

    double R[9];
    euler_to_rotation(hf_param + 3, R, "xyz");
    const double *o = hf_param;

    /* sphere center in hfield-local: p = Rᵀ (cs − o) */
    double d[3] = { sph_param[0]-o[0], sph_param[1]-o[1], sph_param[2]-o[2] };
    double p[3] = { R[0]*d[0] + R[3]*d[1] + R[6]*d[2],
                    R[1]*d[0] + R[4]*d[1] + R[7]*d[2],
                    R[2]*d[0] + R[5]*d[1] + R[8]*d[2] };

    double cp_local[3], n_local[3], depth;
    int rc = sphere_hf_local(p, sph_param[6], slot, cp_local, n_local, &depth);
    if (rc < 0) return -1;

    /* world: cp = R·cp_local + o, n = R·n_local (points terrain surface → sphere) */
    out[0] = R[0]*cp_local[0] + R[1]*cp_local[1] + R[2]*cp_local[2] + o[0];
    out[1] = R[3]*cp_local[0] + R[4]*cp_local[1] + R[5]*cp_local[2] + o[1];
    out[2] = R[6]*cp_local[0] + R[7]*cp_local[1] + R[8]*cp_local[2] + o[2];
    out[3] = R[0]*n_local[0] + R[1]*n_local[1] + R[2]*n_local[2];
    out[4] = R[3]*n_local[0] + R[4]*n_local[1] + R[5]*n_local[2];
    out[5] = R[6]*n_local[0] + R[7]*n_local[1] + R[8]*n_local[2];
    out[6] = depth;
    return depth > 0.0 ? 1 : 0;
}


/* Box-hfield narrowphase (Tier 2). Tests the box's 8 vertices against the local terrain
 * triangle plane each falls on: a vertex below its cell's surface is a contact (point =
 * vertex projected onto the plane, normal = that cell triangle's up-normal, depth =
 * penetration). A box resting flat yields its 4 bottom corners → a stable manifold; each
 * point carries its own normal, so a box straddling cells of different slope is handled
 * (the LCP builds per-point contact rows). Up to max_pts (≤4) deepest points returned.
 *   box_param = [pos(3), euler_xyz(3), half-extents(3)]   (world pose)
 *   hf_param  = [pos(3), euler_xyz(3), slot, _, _]         (world pose)
 * Writes out[7·k] = [contact point (world), unit normal (world, terrain→box), depth].
 * Returns point count (0 = no penetrating vertex), or -1 if the box misses the grid.
 *
 * Tier-2 limitations: only box VERTICES are sampled — a terrain peak poking into the box's
 * bottom face between vertices, or a box edge across a sharp ridge, is missed (fine for
 * terrain varying gently relative to the box; a terrain-vertex-vs-box pass would close it). */
static int box_hf_contact(double *box_param, double *hf_param, double *out, int max_pts)
{
    int slot = (int)hf_param[6];
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1;
    int nr = hf_nrow[slot], nc = hf_ncol[slot];
    const double *H = hf_data[slot];
    double sx = hf_sx[slot], sy = hf_sy[slot];
    double dx = 2.0*sx/(nc-1), dy = 2.0*sy/(nr-1);

    double Rhf[9], Rb[9];
    euler_to_rotation(hf_param + 3, Rhf, "xyz");   /* hfield world rotation */
    euler_to_rotation(box_param + 3, Rb,  "xyz");  /* box world rotation */
    const double *o  = hf_param;                   /* hfield origin (world) */
    const double *cb = box_param;                  /* box center (world) */
    const double *h  = box_param + 6;              /* box half-extents */

    /* box center in hfield-local: p_c = Rhfᵀ (cb − o) */
    double d[3] = { cb[0]-o[0], cb[1]-o[1], cb[2]-o[2] };
    double pc[3] = { Rhf[0]*d[0]+Rhf[3]*d[1]+Rhf[6]*d[2],
                     Rhf[1]*d[0]+Rhf[4]*d[1]+Rhf[7]*d[2],
                     Rhf[2]*d[0]+Rhf[5]*d[1]+Rhf[8]*d[2] };
    /* box axes in hfield-local: columns of Rbl = Rhfᵀ·Rb. axis a = Rbl[:,a]. */
    double ax[3][3];
    for (int a = 0; a < 3; a++) {
        double c0=Rb[0+a], c1=Rb[3+a], c2=Rb[6+a];   /* a-th column of Rb (world) */
        ax[a][0] = Rhf[0]*c0 + Rhf[3]*c1 + Rhf[6]*c2;
        ax[a][1] = Rhf[1]*c0 + Rhf[4]*c1 + Rhf[7]*c2;
        ax[a][2] = Rhf[2]*c0 + Rhf[5]*c1 + Rhf[8]*c2;
    }

    int n_out = 0;
    double cand[8*7];          /* candidate contacts (local): [px,py,pz, nx,ny,nz, depth] */
    int    ncand = 0;

    for (int vtx = 0; vtx < 8; vtx++) {
        double s0 = (vtx & 1) ? 1.0 : -1.0;
        double s1 = (vtx & 2) ? 1.0 : -1.0;
        double s2 = (vtx & 4) ? 1.0 : -1.0;
        double v[3] = {
            pc[0] + s0*h[0]*ax[0][0] + s1*h[1]*ax[1][0] + s2*h[2]*ax[2][0],
            pc[1] + s0*h[0]*ax[0][1] + s1*h[1]*ax[1][1] + s2*h[2]*ax[2][1],
            pc[2] + s0*h[0]*ax[0][2] + s1*h[1]*ax[1][2] + s2*h[2]*ax[2][2] };

        /* cell containing the vertex's XY; skip vertices off the grid footprint */
        int j = (int)floor((v[0]+sx)/dx);
        int i = (int)floor((v[1]+sy)/dy);
        if (j < 0 || j > nc-2 || i < 0 || i > nr-2) continue;

        double x0=-sx+j*dx, x1=-sx+(j+1)*dx, y0=-sy+i*dy, y1=-sy+(i+1)*dy;
        double p00[3]={x0,y0,H[i*nc+j]},     p10[3]={x1,y0,H[i*nc+j+1]};
        double p01[3]={x0,y1,H[(i+1)*nc+j]}, p11[3]={x1,y1,H[(i+1)*nc+j+1]};

        /* pick the triangle the vertex's XY lies in: diagonal p00-p11 splits the cell
         * into A=(p00,p10,p11) for local t<=s and B=(p00,p11,p01) for t>=s. */
        double sloc = (v[0]-x0)/dx, tloc = (v[1]-y0)/dy;
        double *a3, *b3, *c3;
        if (tloc <= sloc) { a3=p00; b3=p10; c3=p11; }   /* tri A */
        else              { a3=p00; b3=p11; c3=p01; }   /* tri B */

        /* triangle up-normal (+z) and signed distance of the vertex above the plane */
        double e1[3]={b3[0]-a3[0],b3[1]-a3[1],b3[2]-a3[2]};
        double e2[3]={c3[0]-a3[0],c3[1]-a3[1],c3[2]-a3[2]};
        double nrm[3]; cross3(e1[0],e1[1],e1[2], e2[0],e2[1],e2[2], nrm);
        double nl = sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1]+nrm[2]*nrm[2]);
        if (nl < 1e-15) continue;
        nrm[0]/=nl; nrm[1]/=nl; nrm[2]/=nl;
        if (nrm[2] < 0.0) { nrm[0]=-nrm[0]; nrm[1]=-nrm[1]; nrm[2]=-nrm[2]; }   /* orient up */

        double sd = nrm[0]*(v[0]-a3[0]) + nrm[1]*(v[1]-a3[1]) + nrm[2]*(v[2]-a3[2]);
        if (sd >= 0.0) continue;                 /* vertex above the surface → no contact */

        double depth = -sd;
        double *cd = cand + 7*ncand;
        cd[0]=v[0]-sd*nrm[0]; cd[1]=v[1]-sd*nrm[1]; cd[2]=v[2]-sd*nrm[2];   /* project onto plane */
        cd[3]=nrm[0]; cd[4]=nrm[1]; cd[5]=nrm[2];
        cd[6]=depth;
        ncand++;
    }
    if (ncand == 0) return 0;

    /* Keep up to max_pts deepest candidates (selection by repeated max). */
    int cap = (max_pts < MAX_PTS_PER_PAIR) ? max_pts : MAX_PTS_PER_PAIR;
    int used[8] = {0};
    for (n_out = 0; n_out < cap && n_out < ncand; n_out++) {
        int best=-1; double bd=-1.0;
        for (int c=0; c<ncand; c++)
            if (!used[c] && cand[7*c+6] > bd) { bd=cand[7*c+6]; best=c; }
        used[best]=1;
        double *cd=cand+7*best, *ot=out+7*n_out;
        /* local → world: cp = Rhf·cp_l + o, n = Rhf·n_l */
        ot[0]=Rhf[0]*cd[0]+Rhf[1]*cd[1]+Rhf[2]*cd[2]+o[0];
        ot[1]=Rhf[3]*cd[0]+Rhf[4]*cd[1]+Rhf[5]*cd[2]+o[1];
        ot[2]=Rhf[6]*cd[0]+Rhf[7]*cd[1]+Rhf[8]*cd[2]+o[2];
        ot[3]=Rhf[0]*cd[3]+Rhf[1]*cd[4]+Rhf[2]*cd[5];
        ot[4]=Rhf[3]*cd[3]+Rhf[4]*cd[4]+Rhf[5]*cd[5];
        ot[5]=Rhf[6]*cd[3]+Rhf[7]*cd[4]+Rhf[8]*cd[5];
        ot[6]=cd[6];
    }
    return n_out;
}


/* Helper for cylinder-hfield (and any future feature-sampling detector): in the
 * hfield's local frame, check whether a single point `p` lies below the terrain
 * triangle of the cell containing its xy projection. If yes, fills cd[7] with the
 * projected contact point (on triangle plane, local frame), the triangle's
 * up-normal, and the penetration depth (-signed distance) and returns 1. Returns 0
 * if p is above the surface or its xy falls off the grid. Mirrors the per-vertex
 * check used inline by box_hf_contact. */
static inline int point_below_hf_triangle(const double *p,
                                          int nr, int nc, const double *H,
                                          double sx, double sy, double dx, double dy,
                                          double *cd)
{
    int j = (int)floor((p[0]+sx)/dx);
    int i = (int)floor((p[1]+sy)/dy);
    if (j < 0 || j > nc-2 || i < 0 || i > nr-2) return 0;
    double x0=-sx+j*dx, x1=-sx+(j+1)*dx, y0=-sy+i*dy, y1=-sy+(i+1)*dy;
    double p00[3]={x0,y0,H[i*nc+j]},     p10[3]={x1,y0,H[i*nc+j+1]};
    double p01[3]={x0,y1,H[(i+1)*nc+j]}, p11[3]={x1,y1,H[(i+1)*nc+j+1]};
    double sloc = (p[0]-x0)/dx, tloc = (p[1]-y0)/dy;
    double *a3, *b3, *c3;
    if (tloc <= sloc) { a3=p00; b3=p10; c3=p11; }
    else              { a3=p00; b3=p11; c3=p01; }
    double e1[3]={b3[0]-a3[0],b3[1]-a3[1],b3[2]-a3[2]};
    double e2[3]={c3[0]-a3[0],c3[1]-a3[1],c3[2]-a3[2]};
    double nrm[3]; cross3(e1[0],e1[1],e1[2], e2[0],e2[1],e2[2], nrm);
    double nl = sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1]+nrm[2]*nrm[2]);
    if (nl < 1e-15) return 0;
    nrm[0]/=nl; nrm[1]/=nl; nrm[2]/=nl;
    if (nrm[2] < 0.0) { nrm[0]=-nrm[0]; nrm[1]=-nrm[1]; nrm[2]=-nrm[2]; }   /* orient up */
    double sd = nrm[0]*(p[0]-a3[0]) + nrm[1]*(p[1]-a3[1]) + nrm[2]*(p[2]-a3[2]);
    if (sd >= 0.0) return 0;
    cd[0]=p[0]-sd*nrm[0]; cd[1]=p[1]-sd*nrm[1]; cd[2]=p[2]-sd*nrm[2];   /* project onto plane */
    cd[3]=nrm[0]; cd[4]=nrm[1]; cd[5]=nrm[2];
    cd[6]=-sd;
    return 1;
}


/* Capsule-hfield narrowphase. A capsule's surface is the locus of points at distance
 * r_cap from its axis segment, so contact reduces to "sphere-hfield at points along
 * the capsule axis" with radius r_cap. Mirrors box_capsule_contact's structure.
 *   cap_param = [pos(3), euler_xyz(3), r_cap, hh, _]   (axis = local +z)
 *   hf_param  = [pos(3), euler_xyz(3), slot,  _, _]
 * Writes out[7·k] = [contact point (world, on terrain), unit normal (world, pointing
 * terrain → capsule), depth] for k ∈ [0, n). Returns n ≥ 1 = penetrating, 0 = touching,
 * -1 = miss / no axis sample reaches the grid.
 *
 * Strategy: sphere-hfield at three capsule-axis samples — endpoints (t = ±hh) and the
 * segment point closest to the hfield XY origin (t* = clamp(−(c·z)_xy / |z|²_xy, ±hh)).
 *   • If both endpoints contact terrain with nearly-parallel terrain normals
 *     (n_A · n_B > 0.99) → emit a 2-point manifold (sub_id 0 = endpoint A, 1 = endpoint B).
 *     Damps roll about the capsule axis for a capsule resting on flat-ish terrain.
 *   • Else emit the single deepest sample.
 *
 * Inherits all Tier-1 limitations of sphere_hf_local (no signed inside test, normal
 * forced to +Z near surface). Diagonal-grazing where the deepest contact lies in the
 * segment interior between samples (e.g., long capsule crossing a terrain peak) is the
 * same family of edge cases as box_capsule_contact; depth can be under-estimated and
 * the LCP catches up over subsequent steps. */
static int capsule_hf_contact(double *cap_param, double *hf_param, double *out, int max_pts)
{
    int slot = (int)hf_param[6];
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1;

    double Rhf[9], Rc[9];
    euler_to_rotation(hf_param + 3, Rhf, "xyz");
    euler_to_rotation(cap_param + 3, Rc,  "xyz");
    const double *o  = hf_param;
    const double *cc = cap_param;
    double r_cap = cap_param[6];
    double hh    = cap_param[7];

    /* Capsule center and axis in hfield-local frame */
    double dc[3] = { cc[0]-o[0], cc[1]-o[1], cc[2]-o[2] };
    double c_loc[3] = { Rhf[0]*dc[0]+Rhf[3]*dc[1]+Rhf[6]*dc[2],
                        Rhf[1]*dc[0]+Rhf[4]*dc[1]+Rhf[7]*dc[2],
                        Rhf[2]*dc[0]+Rhf[5]*dc[1]+Rhf[8]*dc[2] };
    double zw[3]    = { Rc[2], Rc[5], Rc[8] };   /* capsule +z column (world) */
    double z_loc[3] = { Rhf[0]*zw[0]+Rhf[3]*zw[1]+Rhf[6]*zw[2],
                        Rhf[1]*zw[0]+Rhf[4]*zw[1]+Rhf[7]*zw[2],
                        Rhf[2]*zw[0]+Rhf[5]*zw[1]+Rhf[8]*zw[2] };

    /* Center anchor: segment point whose XY projection is closest to the hfield's XY
     * origin (terrain center). For a nearly-horizontal capsule above an extended grid,
     * this picks the most "interesting" middle sample; for a vertical capsule the
     * formula degenerates and we fall back to t = 0 (capsule center). */
    double zxy_sq = z_loc[0]*z_loc[0] + z_loc[1]*z_loc[1];
    double t_center;
    if (zxy_sq > 1e-12) {
        t_center = -(c_loc[0]*z_loc[0] + c_loc[1]*z_loc[1]) / zxy_sq;
        if (t_center >  hh) t_center =  hh;
        if (t_center < -hh) t_center = -hh;
    } else {
        t_center = 0.0;
    }
    double ts[3] = { -hh, +hh, t_center };

    double q_arr[3][3], n_arr[3][3], d_arr[3] = {0, 0, 0};
    int    valid[3]   = {0, 0, 0};

    for (int k = 0; k < 3; k++) {
        double t = ts[k];
        double p_local[3] = { c_loc[0]+t*z_loc[0], c_loc[1]+t*z_loc[1], c_loc[2]+t*z_loc[2] };
        double cp_local[3], n_local[3], depth;
        int rc = sphere_hf_local(p_local, r_cap, slot, cp_local, n_local, &depth);
        if (rc < 0) continue;
        q_arr[k][0]=cp_local[0]; q_arr[k][1]=cp_local[1]; q_arr[k][2]=cp_local[2];
        n_arr[k][0]=n_local[0];  n_arr[k][1]=n_local[1];  n_arr[k][2]=n_local[2];
        d_arr[k] = depth;
        valid[k] = 1;
    }

    if (!valid[0] && !valid[1] && !valid[2]) return -1;

    /* 2-point manifold: both endpoints valid with nearly-parallel terrain normals
     * (capsule resting on flat or uniformly-sloped terrain entirely within footprint). */
    if (max_pts >= 2 && valid[0] && valid[1]) {
        double dot_n = n_arr[0][0]*n_arr[1][0] + n_arr[0][1]*n_arr[1][1] + n_arr[0][2]*n_arr[1][2];
        if (dot_n > 0.99) {
            for (int k = 0; k < 2; k++) {
                double *out_p = out + 7*k;
                out_p[0] = Rhf[0]*q_arr[k][0] + Rhf[1]*q_arr[k][1] + Rhf[2]*q_arr[k][2] + o[0];
                out_p[1] = Rhf[3]*q_arr[k][0] + Rhf[4]*q_arr[k][1] + Rhf[5]*q_arr[k][2] + o[1];
                out_p[2] = Rhf[6]*q_arr[k][0] + Rhf[7]*q_arr[k][1] + Rhf[8]*q_arr[k][2] + o[2];
                out_p[3] = Rhf[0]*n_arr[k][0] + Rhf[1]*n_arr[k][1] + Rhf[2]*n_arr[k][2];
                out_p[4] = Rhf[3]*n_arr[k][0] + Rhf[4]*n_arr[k][1] + Rhf[5]*n_arr[k][2];
                out_p[5] = Rhf[6]*n_arr[k][0] + Rhf[7]*n_arr[k][1] + Rhf[8]*n_arr[k][2];
                out_p[6] = d_arr[k];
            }
            return 2;
        }
    }

    /* Footprint-clip 2-point manifold (long-capsule case): capsule axis extends past
     * the hfield's xy footprint so endpoint samples missed. Center anchor caught the
     * terrain and the capsule axis is parallel to the xy plane (|z_loc[2]| ≈ 0) — clip
     * the capsule t-range to the footprint rectangle [-sx, sx] × [-sy, sy] on the
     * in-plane (x, y) axes. Sloped terrain is handled by re-evaluating sphere_hf_local
     * at each clipped t (depth and normal computed per-point), so this works on slopes
     * too — only the xy-extent of the capsule above the grid matters here. */
    if (max_pts >= 2 && valid[2] && fabs(z_loc[2]) < 1e-3) {
        double sx = hf_sx[slot], sy = hf_sy[slot];
        double bounds[2] = { sx, sy };
        double t_lo = -hh, t_hi = +hh;
        int    feasible = 1;
        for (int i = 0; i < 2; i++) {
            if (fabs(z_loc[i]) > 1e-12) {
                double t1 = ( bounds[i] - c_loc[i]) / z_loc[i];
                double t2 = (-bounds[i] - c_loc[i]) / z_loc[i];
                double a = t1 < t2 ? t1 : t2;
                double b = t1 < t2 ? t2 : t1;
                if (a > t_lo) t_lo = a;
                if (b < t_hi) t_hi = b;
            } else if (c_loc[i] >= bounds[i] || c_loc[i] <= -bounds[i]) {
                feasible = 0;
                break;
            }
        }
        if (feasible && t_lo < t_hi - 1e-9) {
            /* Re-evaluate sphere_hf_local at each clipped t to get accurate terrain
             * elevation and depth at that exact axis point. */
            double pts_t[2] = { t_lo, t_hi };
            double q2[2][3], n2[2][3], d2[2];
            int v2[2] = {0, 0};
            for (int k = 0; k < 2; k++) {
                double t = pts_t[k];
                double p_local[3] = { c_loc[0]+t*z_loc[0], c_loc[1]+t*z_loc[1], c_loc[2]+t*z_loc[2] };
                int rc = sphere_hf_local(p_local, r_cap, slot, q2[k], n2[k], &d2[k]);
                v2[k] = (rc >= 0);
            }
            if (v2[0] && v2[1]) {
                for (int k = 0; k < 2; k++) {
                    double *out_p = out + 7*k;
                    out_p[0] = Rhf[0]*q2[k][0] + Rhf[1]*q2[k][1] + Rhf[2]*q2[k][2] + o[0];
                    out_p[1] = Rhf[3]*q2[k][0] + Rhf[4]*q2[k][1] + Rhf[5]*q2[k][2] + o[1];
                    out_p[2] = Rhf[6]*q2[k][0] + Rhf[7]*q2[k][1] + Rhf[8]*q2[k][2] + o[2];
                    out_p[3] = Rhf[0]*n2[k][0] + Rhf[1]*n2[k][1] + Rhf[2]*n2[k][2];
                    out_p[4] = Rhf[3]*n2[k][0] + Rhf[4]*n2[k][1] + Rhf[5]*n2[k][2];
                    out_p[5] = Rhf[6]*n2[k][0] + Rhf[7]*n2[k][1] + Rhf[8]*n2[k][2];
                    out_p[6] = d2[k];
                }
                return 2;
            }
            /* If clip endpoints didn't reach the grid (boundary edge case), fall through */
        }
    }

    /* Single deepest sample */
    int best_k = 0;
    double best_d = -1e300;
    for (int k = 0; k < 3; k++) {
        if (valid[k] && d_arr[k] > best_d) { best_d = d_arr[k]; best_k = k; }
    }
    out[0] = Rhf[0]*q_arr[best_k][0] + Rhf[1]*q_arr[best_k][1] + Rhf[2]*q_arr[best_k][2] + o[0];
    out[1] = Rhf[3]*q_arr[best_k][0] + Rhf[4]*q_arr[best_k][1] + Rhf[5]*q_arr[best_k][2] + o[1];
    out[2] = Rhf[6]*q_arr[best_k][0] + Rhf[7]*q_arr[best_k][1] + Rhf[8]*q_arr[best_k][2] + o[2];
    out[3] = Rhf[0]*n_arr[best_k][0] + Rhf[1]*n_arr[best_k][1] + Rhf[2]*n_arr[best_k][2];
    out[4] = Rhf[3]*n_arr[best_k][0] + Rhf[4]*n_arr[best_k][1] + Rhf[5]*n_arr[best_k][2];
    out[5] = Rhf[6]*n_arr[best_k][0] + Rhf[7]*n_arr[best_k][1] + Rhf[8]*n_arr[best_k][2];
    out[6] = best_d;
    return best_d > 0.0 ? 1 : 0;
}


/* Cylinder-hfield narrowphase (Tier 2, dense feature sampling). A cylinder has three
 * contact features against terrain that need distinct treatment:
 *   • Two flat cap disks at z = ±hh (cylinder-local), radius r_cyl
 *   • One curved side surface at radius r_cyl, |z| ≤ hh
 *   • Two circular edge rims where caps meet side (z = ±hh, radius r_cyl)
 *
 * Unlike the capsule (where sphere_hf_local at axis points correctly reproduces the
 * spherical-cap-+-cylindrical-side surface), a real cylinder has FLAT caps. Treating
 * cylinder ≈ capsule mislocates standing-cylinder contacts by r_cyl. So we sample the
 * cylinder surface densely at distinct feature points and check each against the
 * terrain triangle below its xy projection — the same vertex-vs-plane logic box_hf
 * uses for box vertices, generalized to cylinder surface samples.
 *
 * Sample set (in cylinder-local frame, transformed to hfield-local for the check):
 *   • 2 cap centers (axis endpoints, z=±hh, ρ=0)
 *   • 2·N_ANG cap-rim points (z=±hh, ρ=r_cyl, angular θ ∈ {2πk/N_ANG})
 *   • 1·N_ANG side middle samples (z=0, ρ=r_cyl) — covers lying-cylinder side contact
 *   • 2·N_ANG side off-center samples (z = ±hh/2, ρ=r_cyl) — denser side coverage
 *                                                          for long lying cylinders
 * With N_ANG=16 the total is 2 + 32 + 16 + 32 = 82 samples per pair. Each is a single
 * vertex-vs-cell-plane check (~30 ops); selection picks the deepest ≤ MAX_PTS_PER_PAIR.
 *
 *   cyl_param = [pos(3), euler_xyz(3), r_cyl, hh, _]   (axis = local +z)
 *   hf_param  = [pos(3), euler_xyz(3), slot,  _, _]
 * Writes out[7·k] = [contact point (world, on terrain triangle), unit normal (world,
 * pointing terrain → cylinder), depth] per emitted point. Returns N ≥ 1 = penetrating
 * with N points, 0 = colliding but no penetrating sample, -1 = slot invalid.
 *
 * Tier-2 limitations (inherited from sphere/box-hf): no signed inside test (deep
 * tunneling beyond r_cyl misses); angular sampling discretization (a terrain spike
 * between two side samples within one cell can be missed); sampled cap interior is
 * sparse (cap-center only, no radial samples) so a terrain peak under the cap face
 * between center and rim is approximate. */
static int cylinder_hf_contact(double *cyl_param, double *hf_param, double *out, int max_pts)
{
    int slot = (int)hf_param[6];
    if (slot < 0 || slot >= MAX_HFIELD || !hf_data[slot]) return -1;
    int nr = hf_nrow[slot], nc = hf_ncol[slot];
    const double *H = hf_data[slot];
    double sx = hf_sx[slot], sy = hf_sy[slot];
    double dx = 2.0*sx/(nc-1), dy = 2.0*sy/(nr-1);

    double Rhf[9], Rc[9];
    euler_to_rotation(hf_param + 3, Rhf, "xyz");
    euler_to_rotation(cyl_param + 3, Rc,  "xyz");
    const double *o = hf_param;
    const double *cc = cyl_param;
    double r_cyl = cyl_param[6];
    double hh    = cyl_param[7];

    /* Cylinder center in hfield-local */
    double dc[3] = { cc[0]-o[0], cc[1]-o[1], cc[2]-o[2] };
    double pc[3] = { Rhf[0]*dc[0]+Rhf[3]*dc[1]+Rhf[6]*dc[2],
                     Rhf[1]*dc[0]+Rhf[4]*dc[1]+Rhf[7]*dc[2],
                     Rhf[2]*dc[0]+Rhf[5]*dc[1]+Rhf[8]*dc[2] };
    /* Cylinder local axes in hfield-local: a-th column = Rhfᵀ·Rc[:,a].
     * ax[0] = local x, ax[1] = local y, ax[2] = local z (cylinder axis). */
    double ax[3][3];
    for (int a = 0; a < 3; a++) {
        double c0 = Rc[0+a], c1 = Rc[3+a], c2 = Rc[6+a];
        ax[a][0] = Rhf[0]*c0 + Rhf[3]*c1 + Rhf[6]*c2;
        ax[a][1] = Rhf[1]*c0 + Rhf[4]*c1 + Rhf[7]*c2;
        ax[a][2] = Rhf[2]*c0 + Rhf[5]*c1 + Rhf[8]*c2;
    }

    /* Sample cylinder surface, check each vs local terrain triangle */
    #define N_ANG 16
    enum { N_CAND_MAX = 2 + 2*N_ANG + 3*N_ANG };  /* caps + rims + 3 axial side rings */
    double cand[N_CAND_MAX * 7];
    int ncand = 0;

    /* Pre-compute angular table (sin/cos) once */
    double co[N_ANG], si[N_ANG];
    for (int k = 0; k < N_ANG; k++) {
        double th = 2.0*M_PI*k/N_ANG;
        co[k] = cos(th); si[k] = sin(th);
    }

    /* (1) Cap centers (axis endpoints, ρ=0) */
    for (int s = 0; s < 2; s++) {
        double sgn = s ? +1.0 : -1.0;
        double p[3] = { pc[0] + sgn*hh*ax[2][0],
                        pc[1] + sgn*hh*ax[2][1],
                        pc[2] + sgn*hh*ax[2][2] };
        if (point_below_hf_triangle(p, nr, nc, H, sx, sy, dx, dy, cand + 7*ncand))
            ncand++;
    }

    /* (2) Cap rims: at z = ±hh, radius r_cyl */
    for (int s = 0; s < 2; s++) {
        double sgn = s ? +1.0 : -1.0;
        for (int k = 0; k < N_ANG; k++) {
            double rx = r_cyl * (co[k]*ax[0][0] + si[k]*ax[1][0]);
            double ry = r_cyl * (co[k]*ax[0][1] + si[k]*ax[1][1]);
            double rz = r_cyl * (co[k]*ax[0][2] + si[k]*ax[1][2]);
            double p[3] = { pc[0] + sgn*hh*ax[2][0] + rx,
                            pc[1] + sgn*hh*ax[2][1] + ry,
                            pc[2] + sgn*hh*ax[2][2] + rz };
            if (point_below_hf_triangle(p, nr, nc, H, sx, sy, dx, dy, cand + 7*ncand))
                ncand++;
        }
    }

    /* (3) Side surface: rings at z = -hh/2, 0, +hh/2 (interior axial positions) */
    const double tz_arr[3] = { -0.5*hh, 0.0, +0.5*hh };
    for (int kt = 0; kt < 3; kt++) {
        double tz = tz_arr[kt];
        for (int k = 0; k < N_ANG; k++) {
            double rx = r_cyl * (co[k]*ax[0][0] + si[k]*ax[1][0]);
            double ry = r_cyl * (co[k]*ax[0][1] + si[k]*ax[1][1]);
            double rz = r_cyl * (co[k]*ax[0][2] + si[k]*ax[1][2]);
            double p[3] = { pc[0] + tz*ax[2][0] + rx,
                            pc[1] + tz*ax[2][1] + ry,
                            pc[2] + tz*ax[2][2] + rz };
            if (point_below_hf_triangle(p, nr, nc, H, sx, sy, dx, dy, cand + 7*ncand))
                ncand++;
        }
    }
    #undef N_ANG

    if (ncand == 0) return 0;

    /* Select up to max_pts deepest candidates (selection by repeated max; small cap). */
    int cap_n = (max_pts < MAX_PTS_PER_PAIR) ? max_pts : MAX_PTS_PER_PAIR;
    int used[N_CAND_MAX] = {0};
    int n_out = 0;
    for (n_out = 0; n_out < cap_n && n_out < ncand; n_out++) {
        int best = -1;
        double bd = -1.0;
        for (int c = 0; c < ncand; c++)
            if (!used[c] && cand[7*c+6] > bd) { bd = cand[7*c+6]; best = c; }
        if (best < 0) break;
        used[best] = 1;
        double *cd = cand + 7*best;
        double *ot = out  + 7*n_out;
        /* hfield-local → world: cp = Rhf·cp_local + o, n = Rhf·n_local */
        ot[0] = Rhf[0]*cd[0] + Rhf[1]*cd[1] + Rhf[2]*cd[2] + o[0];
        ot[1] = Rhf[3]*cd[0] + Rhf[4]*cd[1] + Rhf[5]*cd[2] + o[1];
        ot[2] = Rhf[6]*cd[0] + Rhf[7]*cd[1] + Rhf[8]*cd[2] + o[2];
        ot[3] = Rhf[0]*cd[3] + Rhf[1]*cd[4] + Rhf[2]*cd[5];
        ot[4] = Rhf[3]*cd[3] + Rhf[4]*cd[4] + Rhf[5]*cd[5];
        ot[5] = Rhf[6]*cd[3] + Rhf[7]*cd[4] + Rhf[8]*cd[5];
        ot[6] = cd[6];
    }
    return n_out;
}


/* Mesh-hfield narrowphase (Tier 2, vertex sampling). For each vertex in the mesh's
 * .obj data, transform it to hfield-local frame and check against the local terrain
 * triangle (same vertex-vs-cell-plane pattern as box_hf and cylinder_hf, but with
 * mesh.obj vertices instead of generated samples). Streaming top-N selection keeps
 * memory bounded at MAX_PTS_PER_PAIR regardless of mesh size, so this handles small
 * (tetrahedron) and large (2000+ vertex) meshes alike.
 *
 *   mesh_param = [pos(3), euler_xyz(3), slot, _, _]   (slot indexes shape.c mesh storage)
 *   hf_param   = [pos(3), euler_xyz(3), slot, _, _]
 * Writes out[7·k] = [contact point (world, on terrain triangle), unit normal (world,
 * pointing terrain → mesh), depth] per emitted point. Returns N ≥ 1 with N points
 * (sorted deepest-first for stable sub_id), 0 = colliding but no penetrating vertex,
 * -1 = slot invalid / mesh data unloadable.
 *
 * Tier-2 limitations:
 *   • Vertex sampling only — a terrain spike poking up between two mesh vertices, or
 *     a mesh edge dipping below terrain without any vertex doing so, is missed. For
 *     coarse meshes this can be significant; for fine meshes it's bounded by the
 *     terrain cell / mesh edge length ratio.
 *   • Cost is O(V). A 1000-vertex mesh takes ~10× a box_hf check. BVH or mesh AABB
 *     prefilter could cut this if mesh-hf becomes a hot path (deferred).
 *   • Inherits sphere/box-hf no-signed-inside limitation (deep tunneling misses). */
static int mesh_hf_contact(double *mesh_param, double *hf_param, double *out, int max_pts)
{
    int hf_slot = (int)hf_param[6];
    if (hf_slot < 0 || hf_slot >= MAX_HFIELD || !hf_data[hf_slot]) return -1;
    int mesh_slot = (int)mesh_param[6];
    if (mesh_slot < 0 || mesh_slot >= MAX_MESH) return -1;
    if (num_vertex[mesh_slot] == 0) {
        num_vertex[mesh_slot] = load_mesh(mesh_slot);
        if (num_vertex[mesh_slot] <= 0) return -1;
    }
    int nv = num_vertex[mesh_slot];
    double (*verts)[3] = vertex[mesh_slot];

    int nr = hf_nrow[hf_slot], nc = hf_ncol[hf_slot];
    const double *H = hf_data[hf_slot];
    double sx = hf_sx[hf_slot], sy = hf_sy[hf_slot];
    double dx = 2.0*sx/(nc-1), dy = 2.0*sy/(nr-1);

    double Rhf[9], Rm[9];
    euler_to_rotation(hf_param + 3, Rhf, "xyz");
    euler_to_rotation(mesh_param + 3, Rm, "xyz");
    const double *o  = hf_param;
    const double *mc = mesh_param;

    /* Streaming top-N: keep deepest MAX_PTS_PER_PAIR candidates as we scan vertices.
     * For each new penetrating vertex, if it's deeper than current shallowest, replace.
     * Memory cost: O(MAX_PTS_PER_PAIR), regardless of nv. */
    double top[MAX_PTS_PER_PAIR][7];
    int    n_top = 0;
    int    idx_min = 0;     /* index of shallowest in top[] (valid when n_top == MAX) */

    for (int vi = 0; vi < nv; vi++) {
        /* mesh-local vertex → world: w = mc + Rm·v_local */
        double vl0 = verts[vi][0], vl1 = verts[vi][1], vl2 = verts[vi][2];
        double w[3] = {
            Rm[0]*vl0 + Rm[1]*vl1 + Rm[2]*vl2 + mc[0],
            Rm[3]*vl0 + Rm[4]*vl1 + Rm[5]*vl2 + mc[1],
            Rm[6]*vl0 + Rm[7]*vl1 + Rm[8]*vl2 + mc[2],
        };
        /* world → hfield-local: p = Rhfᵀ(w − o) */
        double d[3] = { w[0]-o[0], w[1]-o[1], w[2]-o[2] };
        double p[3] = { Rhf[0]*d[0] + Rhf[3]*d[1] + Rhf[6]*d[2],
                        Rhf[1]*d[0] + Rhf[4]*d[1] + Rhf[7]*d[2],
                        Rhf[2]*d[0] + Rhf[5]*d[1] + Rhf[8]*d[2] };

        double cd[7];
        if (!point_below_hf_triangle(p, nr, nc, H, sx, sy, dx, dy, cd)) continue;

        if (n_top < MAX_PTS_PER_PAIR) {
            for (int j = 0; j < 7; j++) top[n_top][j] = cd[j];
            n_top++;
            if (n_top == MAX_PTS_PER_PAIR) {
                /* Initialize idx_min */
                idx_min = 0;
                for (int k = 1; k < MAX_PTS_PER_PAIR; k++)
                    if (top[k][6] < top[idx_min][6]) idx_min = k;
            }
        } else if (cd[6] > top[idx_min][6]) {
            for (int j = 0; j < 7; j++) top[idx_min][j] = cd[j];
            /* Recompute idx_min (constant 4 comparisons) */
            idx_min = 0;
            for (int k = 1; k < MAX_PTS_PER_PAIR; k++)
                if (top[k][6] < top[idx_min][6]) idx_min = k;
        }
    }

    if (n_top == 0) return 0;

    /* Emit selected candidates, deepest first for stable sub_id ordering. */
    int cap_n = (max_pts < MAX_PTS_PER_PAIR) ? max_pts : MAX_PTS_PER_PAIR;
    int n_emit = (n_top < cap_n) ? n_top : cap_n;
    int used[MAX_PTS_PER_PAIR] = {0};
    for (int k = 0; k < n_emit; k++) {
        int best = -1;
        double bd = -1.0;
        for (int c = 0; c < n_top; c++)
            if (!used[c] && top[c][6] > bd) { bd = top[c][6]; best = c; }
        if (best < 0) break;
        used[best] = 1;
        double *cd = top[best];
        double *ot = out + 7*k;
        /* hfield-local → world */
        ot[0] = Rhf[0]*cd[0] + Rhf[1]*cd[1] + Rhf[2]*cd[2] + o[0];
        ot[1] = Rhf[3]*cd[0] + Rhf[4]*cd[1] + Rhf[5]*cd[2] + o[1];
        ot[2] = Rhf[6]*cd[0] + Rhf[7]*cd[1] + Rhf[8]*cd[2] + o[2];
        ot[3] = Rhf[0]*cd[3] + Rhf[1]*cd[4] + Rhf[2]*cd[5];
        ot[4] = Rhf[3]*cd[3] + Rhf[4]*cd[4] + Rhf[5]*cd[5];
        ot[5] = Rhf[6]*cd[3] + Rhf[7]*cd[4] + Rhf[8]*cd[5];
        ot[6] = cd[6];
    }
    return n_emit;
}


/* Multi-point dispatcher. See tact.h for out[] layout and return-value convention.
 *
 * box-box pairs route through tact_box_box_manifold (SAT + face clipping, up to 4
 * coplanar contact points). box-sphere / sphere-sphere / sphere-hfield route
 * through closed-form analytic detectors (single point). box-hfield uses a
 * Tier-2 multi-point detector (≤4 points). All other type combinations fall
 * back to MPR for a single-point witness — once those shapes get dedicated
 * detectors (Phase 3+ for capsule, mesh-hfield) they'd dispatch here too. */
int tact_collision_check(int type1, double* param1, int type2, double *param2, double* out, int max_pts)
{
    if (max_pts <= 0) return 0;

    if (type1 == BOX && type2 == BOX) {
        return tact_box_box_manifold(param1, param2, out, max_pts);
    }
    /* box-sphere (either order). out normal convention is param1→param2, while
     * box_sphere_contact emits box→sphere; flip when the sphere is param1. */
    if (type1 == BOX && type2 == SPHERE) {
        return box_sphere_contact(param1, param2, out);
    }
    if (type1 == SPHERE && type2 == BOX) {
        int rc = box_sphere_contact(param2, param1, out);   /* box=param2, sphere=param1 */
        if (rc >= 0) { out[3] = -out[3]; out[4] = -out[4]; out[5] = -out[5]; }
        return rc;
    }
    /* sphere-sphere: symmetric, no swap needed. Normal is already param1→param2. */
    if (type1 == SPHERE && type2 == SPHERE) {
        return sphere_sphere_contact(param1, param2, out);
    }
    /* capsule-sphere (either order). capsule_sphere_contact emits capsule→sphere; flip
     * the normal when the sphere is param1 (so the capsule is param2). */
    if (type1 == CAPSULE && type2 == SPHERE) {
        return capsule_sphere_contact(param1, param2, out);
    }
    if (type1 == SPHERE && type2 == CAPSULE) {
        int rc = capsule_sphere_contact(param2, param1, out);   /* cap=param2, sphere=param1 */
        if (rc >= 0) { out[3] = -out[3]; out[4] = -out[4]; out[5] = -out[5]; }
        return rc;
    }
    /* capsule-capsule: symmetric (closest-pair on two segments), no swap needed.
     * Normal is already param1 → param2 from the function. */
    if (type1 == CAPSULE && type2 == CAPSULE) {
        return capsule_capsule_contact(param1, param2, out);
    }
    /* cylinder-sphere (either order). cylinder_sphere_contact emits cylinder→sphere; flip
     * the normal when the sphere is param1 (so the cylinder is param2). */
    if (type1 == CYL && type2 == SPHERE) {
        return cylinder_sphere_contact(param1, param2, out);
    }
    if (type1 == SPHERE && type2 == CYL) {
        int rc = cylinder_sphere_contact(param2, param1, out);   /* cyl=param2, sphere=param1 */
        if (rc >= 0) { out[3] = -out[3]; out[4] = -out[4]; out[5] = -out[5]; }
        return rc;
    }
    /* box-capsule (either order). box_capsule_contact emits box→capsule per point; flip the
     * normal of every point when the capsule is param1 (so the box is param2). May return
     * up to 2 contact points (parallel-resting manifold). */
    if (type1 == BOX && type2 == CAPSULE) {
        return box_capsule_contact(param1, param2, out, max_pts);
    }
    if (type1 == CAPSULE && type2 == BOX) {
        int n = box_capsule_contact(param2, param1, out, max_pts);   /* box=param2, cap=param1 */
        for (int k = 0; k < n; k++) {
            out[7*k+3] = -out[7*k+3]; out[7*k+4] = -out[7*k+4]; out[7*k+5] = -out[7*k+5];
        }
        return n;
    }
    /* sphere-hfield (either order). sphere_hf_contact emits terrain→sphere; that is
     * param1→param2 when the hfield is param1, so flip only when the sphere is param1. */
    if (type1 == HFIELD && type2 == SPHERE) {
        return sphere_hf_contact(param2, param1, out);          /* hf=param1, sphere=param2 */
    }
    if (type1 == SPHERE && type2 == HFIELD) {
        int rc = sphere_hf_contact(param1, param2, out);        /* sphere=param1, hf=param2 */
        if (rc >= 0) { out[3] = -out[3]; out[4] = -out[4]; out[5] = -out[5]; }
        return rc;
    }
    /* box-hfield (either order, Tier 2). box_hf_contact emits terrain→box per point; that
     * is param1→param2 when the hfield is param1, so flip every point when box is param1. */
    if (type1 == HFIELD && type2 == BOX) {
        return box_hf_contact(param2, param1, out, max_pts);    /* hf=param1, box=param2 */
    }
    if (type1 == BOX && type2 == HFIELD) {
        int n = box_hf_contact(param1, param2, out, max_pts);   /* box=param1, hf=param2 */
        for (int k = 0; k < n; k++) {
            out[7*k+3] = -out[7*k+3]; out[7*k+4] = -out[7*k+4]; out[7*k+5] = -out[7*k+5];
        }
        return n;
    }
    /* capsule-hfield (either order). capsule_hf_contact emits terrain→capsule per point;
     * that is param1→param2 when the hfield is param1, so flip every point when capsule
     * is param1. May return up to 2 contact points (parallel-resting manifold). */
    if (type1 == HFIELD && type2 == CAPSULE) {
        return capsule_hf_contact(param2, param1, out, max_pts);    /* cap=param2, hf=param1 */
    }
    if (type1 == CAPSULE && type2 == HFIELD) {
        int n = capsule_hf_contact(param1, param2, out, max_pts);   /* cap=param1, hf=param2 */
        for (int k = 0; k < n; k++) {
            out[7*k+3] = -out[7*k+3]; out[7*k+4] = -out[7*k+4]; out[7*k+5] = -out[7*k+5];
        }
        return n;
    }
    /* cylinder-hfield (either order, Tier 2). cylinder_hf_contact emits terrain→cylinder
     * per point; flip every point when cylinder is param1. May return up to MAX_PTS
     * (rim/side dense-sample manifold). */
    if (type1 == HFIELD && type2 == CYL) {
        return cylinder_hf_contact(param2, param1, out, max_pts);    /* cyl=param2, hf=param1 */
    }
    if (type1 == CYL && type2 == HFIELD) {
        int n = cylinder_hf_contact(param1, param2, out, max_pts);   /* cyl=param1, hf=param2 */
        for (int k = 0; k < n; k++) {
            out[7*k+3] = -out[7*k+3]; out[7*k+4] = -out[7*k+4]; out[7*k+5] = -out[7*k+5];
        }
        return n;
    }
    /* mesh-hfield (either order, Tier 2). mesh_hf_contact emits terrain→mesh per
     * point; flip every point when mesh is param1. Vertex-sampling based; up to
     * MAX_PTS_PER_PAIR deepest vertices kept. */
    if (type1 == HFIELD && type2 == MESH) {
        return mesh_hf_contact(param2, param1, out, max_pts);        /* mesh=param2, hf=param1 */
    }
    if (type1 == MESH && type2 == HFIELD) {
        int n = mesh_hf_contact(param1, param2, out, max_pts);       /* mesh=param1, hf=param2 */
        for (int k = 0; k < n; k++) {
            out[7*k+3] = -out[7*k+3]; out[7*k+4] = -out[7*k+4]; out[7*k+5] = -out[7*k+5];
        }
        return n;
    }
    /* Safety guard: every hfield pair is now handled above. If a future shape type
     * is added without an hfield-specific detector, prevent the convex MPR fallback
     * from colliding against the hfield's convex hull (a giant dome). */
    if (type1 == HFIELD || type2 == HFIELD) return -1;

    /* Fallback: single-point MPR. Translate its return convention (0 = hit,
     * <0 = miss) into the multi-point convention (n_pts written, or -1 = miss). */
    int rc = tact_collision_check_mpr(type1, param1, type2, param2, out);
    if (rc < 0) return -1;
    /* MPR populated out[0..6]. n_pts = 1 if depth > 0, else 0 (touching). */
    return (out[6] > 0.0) ? 1 : 0;
}
