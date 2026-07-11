/* box_box.c — box-box contact manifold via SAT + Sutherland-Hodgman face
 * clipping. Returns up to MAX_PTS_PER_PAIR contact points sharing one contact
 * normal (the smallest-penetration separating axis from SAT). Dispatched into
 * from tact_collision_check (narrow.c) for BOX/BOX pairs.
 *
 * Algorithm follows the well-known ODE / Bullet box-box detector:
 *   1. SAT over 15 candidate axes (3 face normals of A, 3 of B, 9 edge crosses).
 *      Track the axis with smallest positive penetration. Any separating axis
 *      (negative penetration) returns -1 immediately. Face axes carry a small
 *      bias so coplanar overlaps prefer a face winner over a near-tied edge
 *      cross (avoids single-point jitter on flat stacks).
 *   2. Face axis winner (best_code 0..5): face-face / face-vertex contact —
 *      clip the incident box's contact face polygon against the four side
 *      planes of the reference face (Sutherland-Hodgman in the reference face's
 *      tangent plane, depth interpolated along clipped edges). Vertices with
 *      depth < 0 are dropped; if more than MAX_PTS_PER_PAIR survive, Bullet's
 *      4-point pruning (deepest + farthest + max-area + opposite-side) selects
 *      the subset that best spans the contact patch.
 *   3. Edge-cross winner (best_code 6..14): edge-edge contact — closest-pair
 *      formula on the two supporting edges (single contact point; intrinsic
 *      to an edge crossing, no manifold expansion).
 *
 * Canonical sub_id ordering: polar angle in the tangent plane around the
 * manifold centroid, CCW. Yields stable sub_ids 0..n-1 across frames as long
 * as the polygon topology is preserved — warm-start λ retains its meaning.
 * A topology change (vertex count shift) costs one frame of cold-start PGS.
 *
 * out[] layout (per contact point, 7 doubles):
 *   [0..2]  contact point in world coordinates (on reference face plane)
 *   [3..5]  unit normal direction (world) — same for every point in this manifold;
 *           oriented from param1 toward param2 (matches MPR convention)
 *   [6]     scalar penetration depth (≥ 0)
 *
 * Param layout (matches the BOX entries elsewhere in tact):
 *   param[0..2]  box center in world
 *   param[3..5]  extrinsic-xyz Euler angles (radians)
 *   param[6..8]  half-extents along local x, y, z
 */
#include "core.h"

/* ===== tiny inline helpers ================================================ */

static inline double dot3(const double *a, const double *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void sub3(double *out, const double *a, const double *b) {
    out[0] = a[0]-b[0]; out[1] = a[1]-b[1]; out[2] = a[2]-b[2];
}

static inline void cross3v(double *out, const double *a, const double *b) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/* Row-major 3x3 column accessor — R columns are basis vectors. */
static inline void Rcol(const double *R, int j, double *out) {
    out[0] = R[0+j]; out[1] = R[3+j]; out[2] = R[6+j];
}


/* ===== SAT ================================================================ */

/* Penetration along a candidate axis L (need not be unit; we divide by |L|).
 * Returns: (r1 + r2 - |D·L|) / |L|. Positive = overlap; negative = separated. */
static double sat_penetration(const double *L, const double *D,
                              const double *R1, const double *h1,
                              const double *R2, const double *h2,
                              double L2) {
    double a0[3], a1[3], a2[3], b0[3], b1[3], b2[3];
    Rcol(R1, 0, a0); Rcol(R1, 1, a1); Rcol(R1, 2, a2);
    Rcol(R2, 0, b0); Rcol(R2, 1, b1); Rcol(R2, 2, b2);
    double r1 = fabs(dot3(a0, L))*h1[0] + fabs(dot3(a1, L))*h1[1] + fabs(dot3(a2, L))*h1[2];
    double r2 = fabs(dot3(b0, L))*h2[0] + fabs(dot3(b1, L))*h2[1] + fabs(dot3(b2, L))*h2[2];
    return (r1 + r2 - fabs(dot3(D, L))) / sqrt(L2);
}


/* ===== Sutherland-Hodgman 2D polygon clipping ============================= *
 *
 * Each vertex is a triple (u, v, depth). We clip in the (u, v) plane and let
 * depth interpolate linearly along clipped edges. Output buffer must be ≥ 2×
 * input capacity (clipping can momentarily grow polygons before later passes
 * reduce them); we cap at MAX_CLIP_VERTS = 8 which is enough for box-box face.
 */
#define MAX_CLIP_VERTS 8

static int clip_half(const double poly_in[][3], int n_in,
                     double poly_out[][3], int axis, double bound, int keep_above)
{
    int n_out = 0;
    for (int i = 0; i < n_in; i++) {
        int j = (i + 1) % n_in;
        double a_coord = poly_in[i][axis];
        double b_coord = poly_in[j][axis];
        int a_in = keep_above ? (a_coord >= bound) : (a_coord <= bound);
        int b_in = keep_above ? (b_coord >= bound) : (b_coord <= bound);
        if (a_in) {
            if (b_in) {
                if (n_out < MAX_CLIP_VERTS)
                    memcpy(poly_out[n_out++], poly_in[j], 3*sizeof(double));
            } else {
                double dc = b_coord - a_coord;
                double t  = (fabs(dc) > 1e-30) ? (bound - a_coord) / dc : 0.0;
                if (n_out < MAX_CLIP_VERTS) {
                    poly_out[n_out][0] = poly_in[i][0] + t * (poly_in[j][0] - poly_in[i][0]);
                    poly_out[n_out][1] = poly_in[i][1] + t * (poly_in[j][1] - poly_in[i][1]);
                    poly_out[n_out][2] = poly_in[i][2] + t * (poly_in[j][2] - poly_in[i][2]);
                    n_out++;
                }
            }
        } else if (b_in) {
            double dc = b_coord - a_coord;
            double t  = (fabs(dc) > 1e-30) ? (bound - a_coord) / dc : 0.0;
            if (n_out < MAX_CLIP_VERTS) {
                poly_out[n_out][0] = poly_in[i][0] + t * (poly_in[j][0] - poly_in[i][0]);
                poly_out[n_out][1] = poly_in[i][1] + t * (poly_in[j][1] - poly_in[i][1]);
                poly_out[n_out][2] = poly_in[i][2] + t * (poly_in[j][2] - poly_in[i][2]);
                n_out++;
            }
            if (n_out < MAX_CLIP_VERTS)
                memcpy(poly_out[n_out++], poly_in[j], 3*sizeof(double));
        }
    }
    return n_out;
}


/* ===== reference / incident face geometry ================================= */

/* Given the SAT-winning face axis (best_code ∈ 0..5) and the contact normal n
 * (world, pointing param1 → param2), choose which box is the reference and
 * which face on that box is the contact face.
 *
 * For best_code ∈ 0..2 (face normal of box A):
 *   ref = A. ref_axis = best_code. ref_sign = +1 if n agrees with A's +axis,
 *   else -1 (the +/- face of A is the one whose outward normal aligns with n).
 *
 * For best_code ∈ 3..5 (face normal of box B):
 *   ref = B. ref_axis = best_code - 3. ref_sign = -1 if n agrees with B's
 *   +axis (because B's contact face has outward normal ≈ -n).
 */
static void choose_reference_face(int best_code, const double *n,
                                  const double *R1, const double *R2,
                                  int *ref_box, int *ref_axis, int *ref_sign)
{
    if (best_code <= 2) {
        *ref_box  = 0;
        *ref_axis = best_code;
        double ax[3]; Rcol(R1, *ref_axis, ax);
        *ref_sign = (dot3(n, ax) > 0.0) ? +1 : -1;
    } else {
        *ref_box  = 1;
        *ref_axis = best_code - 3;
        double ax[3]; Rcol(R2, *ref_axis, ax);
        *ref_sign = (dot3(n, ax) > 0.0) ? -1 : +1;
    }
}


/* Of the 6 faces of the "incident" box, pick the one whose outward normal is
 * most antiparallel to ref_outward (= ref_sign * R_ref[:, ref_axis], world).
 * Sets inc_axis ∈ {0,1,2} and inc_sign ∈ {±1}. */
static void choose_incident_face(const double *ref_outward,
                                 const double *R_inc,
                                 int *inc_axis, int *inc_sign)
{
    double best_dot = 1e30;                  /* we want the most-negative dot */
    int best_axis = 0, best_sign = +1;
    for (int j = 0; j < 3; j++) {
        double axis[3]; Rcol(R_inc, j, axis);
        double d = dot3(axis, ref_outward);  /* if d>0, +face faces away from ref */
        if ( d < best_dot) { best_dot =  d; best_axis = j; best_sign = +1; }
        if (-d < best_dot) { best_dot = -d; best_axis = j; best_sign = -1; }
    }
    *inc_axis = best_axis;
    *inc_sign = best_sign;
}


/* World-frame corners of a box face. The face is at center + sign·R[:,axis]·h[axis];
 * the four corners span ±h[u] and ±h[v] along the other two axes (u=axis+1, v=axis+2,
 * both mod 3). CCW order in (u,v) when viewed from the face's outward side. */
static void box_face_corners(const double *c, const double *R, const double *h,
                             int axis, int sign, double corners[4][3])
{
    int u = (axis + 1) % 3, v = (axis + 2) % 3;
    double n_ax[3], e_u[3], e_v[3];
    Rcol(R, axis, n_ax); Rcol(R, u, e_u); Rcol(R, v, e_v);
    double base[3] = { c[0] + sign*n_ax[0]*h[axis],
                       c[1] + sign*n_ax[1]*h[axis],
                       c[2] + sign*n_ax[2]*h[axis] };
    /* (u, v) sign combinations in CCW order — when viewed from +n_ax × ?,
     * ordering subtleties are absorbed by the polar-angle sub_id sort later. */
    const double su[4] = {-1, +1, +1, -1};
    const double sv[4] = {-1, -1, +1, +1};
    for (int k = 0; k < 4; k++) {
        double cu = h[u] * su[k];
        double cv = h[v] * sv[k];
        corners[k][0] = base[0] + cu*e_u[0] + cv*e_v[0];
        corners[k][1] = base[1] + cu*e_u[1] + cv*e_v[1];
        corners[k][2] = base[2] + cu*e_u[2] + cv*e_v[2];
    }
}


/* ===== sub_id assignment via polar-angle ordering ========================= */

/* Sort `n` (≤ MAX_CLIP_VERTS) 3D points (in_poly: u, v, depth) CCW by polar
 * angle around the centroid. order_out[i] = original index that ends up at
 * sorted position i. Stable for small n. */
static void polar_sort_indices(const double poly[][3], int n, int order_out[MAX_CLIP_VERTS])
{
    if (n <= 1) {
        for (int i = 0; i < n; i++) order_out[i] = i;
        return;
    }
    double cu = 0, cv = 0;
    for (int i = 0; i < n; i++) { cu += poly[i][0]; cv += poly[i][1]; }
    cu /= n; cv /= n;
    double ang[MAX_CLIP_VERTS];
    for (int i = 0; i < n; i++) {
        ang[i] = atan2(poly[i][1] - cv, poly[i][0] - cu);
        order_out[i] = i;
    }
    /* selection sort by ang (n ≤ 8, O(n²) fine) */
    for (int i = 0; i < n - 1; i++) {
        int min_j = i;
        for (int j = i + 1; j < n; j++)
            if (ang[order_out[j]] < ang[order_out[min_j]]) min_j = j;
        int tmp = order_out[i]; order_out[i] = order_out[min_j]; order_out[min_j] = tmp;
    }
}


/* ===== prune ≥4 candidates down to max_pts via Bullet-style 4-point select
 *
 * When clipping produces more candidates than the manifold capacity, pick the
 * subset that best spans the contact patch:
 *
 *   1. Deepest vertex (largest depth → carries the most normal load)
 *   2. Vertex farthest from #1 in (u,v) — spans the longest axis of the patch
 *   3. Vertex maximizing |signed triangle area| with (#1, #2) — spans the
 *      perpendicular axis on one side of segment 1-2
 *   4. Vertex maximizing |triangle area| with (#1, #2) on the OPPOSITE side
 *      of segment 1-2 — ensures the resulting quadrilateral is convex
 *
 * For n_kept ≤ max_pts the input is copied straight through. Output is not
 * sub_id-sorted yet (the caller does polar-angle sort afterwards).
 */
static int prune_to_max(const double kept[MAX_CLIP_VERTS][3], int n_kept,
                        double pruned[MAX_CLIP_VERTS][3], int max_pts)
{
    if (n_kept <= max_pts) {
        for (int k = 0; k < n_kept; k++)
            memcpy(pruned[k], kept[k], 3*sizeof(double));
        return n_kept;
    }

    int chosen[4] = {0};
    int n_out = 0;
    int taken[MAX_CLIP_VERTS] = {0};

    /* (1) deepest vertex */
    int best = 0;
    for (int k = 1; k < n_kept; k++)
        if (kept[k][2] > kept[best][2]) best = k;
    chosen[n_out++] = best; taken[best] = 1;
    if (n_out >= max_pts) goto emit;

    /* (2) farthest from chosen[0] in (u,v) */
    {
        double best_d2 = -1.0;
        int bk = -1;
        for (int k = 0; k < n_kept; k++) {
            if (taken[k]) continue;
            double du = kept[k][0] - kept[chosen[0]][0];
            double dv = kept[k][1] - kept[chosen[0]][1];
            double d2 = du*du + dv*dv;
            if (d2 > best_d2) { best_d2 = d2; bk = k; }
        }
        if (bk < 0) goto emit;
        chosen[n_out++] = bk; taken[bk] = 1;
    }
    if (n_out >= max_pts) goto emit;

    /* (3) max |signed area| triangle with chosen[0], chosen[1]; remember the sign */
    double signed_area_3 = 0.0;
    {
        double best_a = -1.0;
        int bk = -1;
        double u01 = kept[chosen[1]][0] - kept[chosen[0]][0];
        double v01 = kept[chosen[1]][1] - kept[chosen[0]][1];
        for (int k = 0; k < n_kept; k++) {
            if (taken[k]) continue;
            double uk = kept[k][0] - kept[chosen[0]][0];
            double vk = kept[k][1] - kept[chosen[0]][1];
            double sa = u01*vk - v01*uk;
            double aa = sa < 0 ? -sa : sa;
            if (aa > best_a) { best_a = aa; bk = k; signed_area_3 = sa; }
        }
        if (bk < 0) goto emit;
        chosen[n_out++] = bk; taken[bk] = 1;
    }
    if (n_out >= max_pts) goto emit;

    /* (4) max |triangle area| with chosen[0,1] on the OPPOSITE side of segment 1-2
     * from chosen[2]. If no opposite-side candidate exists (all on the same side),
     * fall back to the largest-area same-side point. */
    {
        double best_a_opp = -1.0;
        int bk_opp = -1;
        double best_a_any = -1.0;
        int bk_any = -1;
        double u01 = kept[chosen[1]][0] - kept[chosen[0]][0];
        double v01 = kept[chosen[1]][1] - kept[chosen[0]][1];
        for (int k = 0; k < n_kept; k++) {
            if (taken[k]) continue;
            double uk = kept[k][0] - kept[chosen[0]][0];
            double vk = kept[k][1] - kept[chosen[0]][1];
            double sa = u01*vk - v01*uk;
            double aa = sa < 0 ? -sa : sa;
            int opp = (signed_area_3 > 0) ? (sa < 0) : (sa > 0);
            if (opp && aa > best_a_opp) { best_a_opp = aa; bk_opp = k; }
            if (aa > best_a_any) { best_a_any = aa; bk_any = k; }
        }
        int bk = (bk_opp >= 0) ? bk_opp : bk_any;
        if (bk >= 0) { chosen[n_out++] = bk; taken[bk] = 1; }
    }

emit:
    for (int k = 0; k < n_out; k++)
        memcpy(pruned[k], kept[chosen[k]], 3*sizeof(double));
    return n_out;
}


/* ===== face-face manifold (the meat) ====================================== */

static int face_face_manifold(int best_code, const double *n,
                              const double *c1, const double *R1, const double *h1,
                              const double *c2, const double *R2, const double *h2,
                              double *out, int max_pts)
{
    int ref_box, ref_axis, ref_sign;
    choose_reference_face(best_code, n, R1, R2, &ref_box, &ref_axis, &ref_sign);

    /* Resolve reference vs incident box pointers. */
    const double *c_ref, *R_ref, *h_ref;
    const double *c_inc, *R_inc, *h_inc;
    if (ref_box == 0) {
        c_ref = c1; R_ref = R1; h_ref = h1;
        c_inc = c2; R_inc = R2; h_inc = h2;
    } else {
        c_ref = c2; R_ref = R2; h_ref = h2;
        c_inc = c1; R_inc = R1; h_inc = h1;
    }

    /* Reference face outward normal (world) — used to choose incident face. */
    double n_ref_ax[3]; Rcol(R_ref, ref_axis, n_ref_ax);
    double ref_outward[3] = { ref_sign*n_ref_ax[0], ref_sign*n_ref_ax[1], ref_sign*n_ref_ax[2] };

    int inc_axis, inc_sign;
    choose_incident_face(ref_outward, R_inc, &inc_axis, &inc_sign);

    /* Build incident face corners in world coords. */
    double inc_corners[4][3];
    box_face_corners(c_inc, R_inc, h_inc, inc_axis, inc_sign, inc_corners);

    /* Express incident corners in reference face's local frame:
     *   For each world corner p, local_p = R_ref^T (p - c_ref).
     *   The two in-plane axes are u_ax = (ref_axis+1)%3, v_ax = (ref_axis+2)%3.
     *   Depth into the reference box = h_ref[ref_axis] - ref_sign · local_p[ref_axis].
     *
     * Polygon vertex storage: (u, v, depth). Depth > 0 = below the reference
     * face (penetrating into ref box).
     */
    int u_ax = (ref_axis + 1) % 3, v_ax = (ref_axis + 2) % 3;
    double eu[3]; Rcol(R_ref, u_ax,   eu);
    double ev[3]; Rcol(R_ref, v_ax,   ev);
    double en[3]; Rcol(R_ref, ref_axis, en);

    double poly_a[MAX_CLIP_VERTS][3];
    double poly_b[MAX_CLIP_VERTS][3];
    int n_poly = 4;
    for (int k = 0; k < 4; k++) {
        double dp[3]; sub3(dp, inc_corners[k], c_ref);
        double lu = dot3(dp, eu);
        double lv = dot3(dp, ev);
        double ln = dot3(dp, en);
        poly_a[k][0] = lu;
        poly_a[k][1] = lv;
        poly_a[k][2] = h_ref[ref_axis] - ref_sign * ln;        /* depth */
    }

    /* Clip against 4 sides of the reference rectangle:
     *   u ∈ [-h_ref[u_ax], +h_ref[u_ax]]
     *   v ∈ [-h_ref[v_ax], +h_ref[v_ax]]
     * Alternate poly_a / poly_b as ping-pong buffers. */
    n_poly = clip_half(poly_a, n_poly, poly_b, 0, -h_ref[u_ax], /*keep_above=*/1);
    n_poly = clip_half(poly_b, n_poly, poly_a, 0, +h_ref[u_ax], /*keep_above=*/0);
    n_poly = clip_half(poly_a, n_poly, poly_b, 1, -h_ref[v_ax], /*keep_above=*/1);
    n_poly = clip_half(poly_b, n_poly, poly_a, 1, +h_ref[v_ax], /*keep_above=*/0);
    /* After 4 clips, surviving polygon is in poly_a. */

    /* Filter: keep only vertices with depth ≥ 0 (actually penetrating). */
    double kept[MAX_CLIP_VERTS][3];
    int n_kept = 0;
    for (int k = 0; k < n_poly; k++) {
        if (poly_a[k][2] >= 0.0) {
            kept[n_kept][0] = poly_a[k][0];
            kept[n_kept][1] = poly_a[k][1];
            kept[n_kept][2] = poly_a[k][2];
            n_kept++;
        }
    }
    if (n_kept == 0) return 0;

    /* Prune to the smaller of max_pts and MAX_PTS_PER_PAIR via Bullet-style
     * 4-point selection (deepest + farthest + max-area + opposite-side). For
     * the typical 4-corner-overlap case n_kept ≤ 4, this is a straight copy. */
    int cap = max_pts;
    if (cap > MAX_PTS_PER_PAIR) cap = MAX_PTS_PER_PAIR;
    double pruned[MAX_CLIP_VERTS][3];
    n_kept = prune_to_max(kept, n_kept, pruned, cap);

    /* Canonical sub_id ordering: polar angle around manifold centroid in (u,v). */
    int order[MAX_CLIP_VERTS];
    polar_sort_indices(pruned, n_kept, order);

    /* Convert each kept vertex back to world coords (projection onto reference
     * face plane) and emit. Normal is shared and pre-oriented (n → out[3..5]).
     * Contact point world = c_ref + ref_sign·h_ref[ref_axis]·en + u·eu + v·ev. */
    double face_base[3] = {
        c_ref[0] + ref_sign * h_ref[ref_axis] * en[0],
        c_ref[1] + ref_sign * h_ref[ref_axis] * en[1],
        c_ref[2] + ref_sign * h_ref[ref_axis] * en[2],
    };
    for (int k = 0; k < n_kept; k++) {
        int src = order[k];
        double lu    = pruned[src][0];
        double lv    = pruned[src][1];
        double depth = pruned[src][2];
        double *o = out + 7*k;
        o[0] = face_base[0] + lu*eu[0] + lv*ev[0];
        o[1] = face_base[1] + lu*eu[1] + lv*ev[1];
        o[2] = face_base[2] + lu*eu[2] + lv*ev[2];
        o[3] = n[0]; o[4] = n[1]; o[5] = n[2];   /* unit normal */
        o[6] = depth;
    }
    return n_kept;
}


/* ===== edge-edge contact ================================================ *
 *
 * For SAT codes 6..14, the smallest-penetration axis is L = R1[:,i] × R2[:,j]
 * for some (i, j). The contact is geometrically a single point at the closest
 * pair on the two relevant edges:
 *   - On box A: the edge parallel to R1[:,i] whose midpoint is farthest in
 *     direction +n (= the "supporting" edge of A toward B).
 *   - On box B: the edge parallel to R2[:,j] whose midpoint is farthest in
 *     direction -n (the supporting edge of B toward A).
 *
 * Each edge is parameterized from its center: point = c_edge + s · (h · axis),
 * with s ∈ [-1, +1] (so s=±1 are the endpoints). Closest-pair on two non-
 * parallel infinite lines is the 2×2 solve of (dA·dA)s - (dA·dB)t = -dA·w,
 * (dA·dB)s - (dB·dB)t = -dB·w with w = cA - cB; clamp s,t to [-1, +1] for
 * segments. (Parallel case is unreachable here because SAT skipped axes with
 * |L|² < 1e-12.)
 *
 * Contact point = midpoint of the closest pair. Normal and depth come from the
 * already-computed SAT axis + penetration. Returns 1 contact point (edge-edge
 * is intrinsically single-point — no manifold expansion). */
static int edge_edge_contact(int best_code, const double *n, double depth,
                             const double *c1, const double *R1, const double *h1,
                             const double *c2, const double *R2, const double *h2,
                             double *out, int max_pts)
{
    if (max_pts < 1) return 0;

    int i = (best_code - 6) / 3;             /* axis of A's edge   */
    int j = (best_code - 6) % 3;             /* axis of B's edge   */
    int i_p1 = (i + 1) % 3, i_p2 = (i + 2) % 3;
    int j_p1 = (j + 1) % 3, j_p2 = (j + 2) % 3;

    /* A's supporting edge: pick the corner offset (±h_a[i_p1], ±h_a[i_p2])
     * that maximizes its dot with +n (closest to box B). Edge runs ± h_a[i]
     * along R1[:,i] from that corner. */
    double R1_i  [3], R1_p1[3], R1_p2[3];
    Rcol(R1, i,    R1_i  );
    Rcol(R1, i_p1, R1_p1 );
    Rcol(R1, i_p2, R1_p2 );
    double s1 = (dot3(R1_p1, n) > 0.0) ? +h1[i_p1] : -h1[i_p1];
    double s2 = (dot3(R1_p2, n) > 0.0) ? +h1[i_p2] : -h1[i_p2];
    double cA[3] = {                         /* edge midpoint on A */
        c1[0] + s1*R1_p1[0] + s2*R1_p2[0],
        c1[1] + s1*R1_p1[1] + s2*R1_p2[1],
        c1[2] + s1*R1_p1[2] + s2*R1_p2[2],
    };
    double dA[3] = {                         /* half-edge vector — endpoint = cA ± dA */
        h1[i] * R1_i[0],
        h1[i] * R1_i[1],
        h1[i] * R1_i[2],
    };

    /* B's supporting edge: same idea but maximize dot with -n. */
    double R2_j  [3], R2_p1[3], R2_p2[3];
    Rcol(R2, j,    R2_j  );
    Rcol(R2, j_p1, R2_p1 );
    Rcol(R2, j_p2, R2_p2 );
    double t1 = (dot3(R2_p1, n) > 0.0) ? -h2[j_p1] : +h2[j_p1];
    double t2 = (dot3(R2_p2, n) > 0.0) ? -h2[j_p2] : +h2[j_p2];
    double cB[3] = {
        c2[0] + t1*R2_p1[0] + t2*R2_p2[0],
        c2[1] + t1*R2_p1[1] + t2*R2_p2[1],
        c2[2] + t1*R2_p1[2] + t2*R2_p2[2],
    };
    double dB[3] = {
        h2[j] * R2_j[0],
        h2[j] * R2_j[1],
        h2[j] * R2_j[2],
    };

    /* Solve for closest pair: minimize |w + s·dA - t·dB|² with w = cA - cB. */
    double w[3];  sub3(w, cA, cB);
    double a     = dot3(dA, dA);
    double b     = dot3(dA, dB);
    double c_qf  = dot3(dB, dB);
    double d_qf  = dot3(dA, w);
    double e_qf  = dot3(dB, w);
    double denom = a*c_qf - b*b;
    /* SAT skipped near-parallel edge crosses (|L|² < 1e-12); denom is large
     * enough here. Cheap safeguard anyway: fall back to s=t=0 if not. */
    double s_par, t_par;
    if (denom > 1e-20) {
        s_par = (b*e_qf - c_qf*d_qf) / denom;
        t_par = (a*e_qf - b   *d_qf) / denom;
    } else {
        s_par = 0.0; t_par = 0.0;
    }
    /* Clamp parameters to the actual segments. */
    if (s_par >  1.0) s_par =  1.0;
    if (s_par < -1.0) s_par = -1.0;
    if (t_par >  1.0) t_par =  1.0;
    if (t_par < -1.0) t_par = -1.0;

    double qA[3] = { cA[0] + s_par*dA[0], cA[1] + s_par*dA[1], cA[2] + s_par*dA[2] };
    double qB[3] = { cB[0] + t_par*dB[0], cB[1] + t_par*dB[1], cB[2] + t_par*dB[2] };

    out[0] = 0.5 * (qA[0] + qB[0]);
    out[1] = 0.5 * (qA[1] + qB[1]);
    out[2] = 0.5 * (qA[2] + qB[2]);
    out[3] = n[0];
    out[4] = n[1];
    out[5] = n[2];
    out[6] = depth;
    return 1;
}


/* ===== entry point ======================================================= */

/* Debug: when this global is non-zero, emit which best_code SAT picked.
 * Useful for verifying that edge-edge / face-clipping code paths actually
 * fire in synthetic test scenarios. Off by default; tests set via ctypes.
 * Not thread-safe — debug aid only. */
int box_box_debug_best_code = -1;

int tact_box_box_manifold(const double *param1, const double *param2,
                     double *out, int max_pts)
{
    double c1[3] = {param1[0], param1[1], param1[2]};
    double c2[3] = {param2[0], param2[1], param2[2]};
    double R1[9], R2[9];
    euler_to_rotation((double*)(param1 + 3), R1, "xyz");
    euler_to_rotation((double*)(param2 + 3), R2, "xyz");
    const double h1[3] = {param1[6], param1[7], param1[8]};
    const double h2[3] = {param2[6], param2[7], param2[8]};

    double D[3]; sub3(D, c2, c1);

    /* SAT: track axis with smallest positive penetration. Face axes win ties
     * (bias < 1) so we get 4-point manifolds on coplanar overlaps instead of
     * a single edge-edge point that would jitter under numeric roundoff. */
    const double FACE_BIAS = 1.0;
    const double EDGE_BIAS = 1.05;

    int   best_code = -1;
    double best_pen_biased = 1e30;
    double best_axis[3] = {0, 0, 0};

#define SAT_TEST(L, L2, code, bias) do {                                        \
        double L2_val = (L2);                                                   \
        if (L2_val > 1e-12) {                                                   \
            double pen = sat_penetration((L), D, R1, h1, R2, h2, L2_val);       \
            if (pen < 0.0) return -1;                                           \
            double biased = pen * (bias);                                       \
            if (biased < best_pen_biased) {                                     \
                best_pen_biased = biased;                                       \
                best_code = (code);                                             \
                double inv = 1.0 / sqrt(L2_val);                                \
                best_axis[0] = (L)[0]*inv;                                      \
                best_axis[1] = (L)[1]*inv;                                      \
                best_axis[2] = (L)[2]*inv;                                      \
            }                                                                   \
        }                                                                       \
    } while (0)

    double Ax[3], Ay[3], Az[3], Bx[3], By[3], Bz[3];
    Rcol(R1, 0, Ax); Rcol(R1, 1, Ay); Rcol(R1, 2, Az);
    Rcol(R2, 0, Bx); Rcol(R2, 1, By); Rcol(R2, 2, Bz);
    SAT_TEST(Ax, 1.0, 0, FACE_BIAS);
    SAT_TEST(Ay, 1.0, 1, FACE_BIAS);
    SAT_TEST(Az, 1.0, 2, FACE_BIAS);
    SAT_TEST(Bx, 1.0, 3, FACE_BIAS);
    SAT_TEST(By, 1.0, 4, FACE_BIAS);
    SAT_TEST(Bz, 1.0, 5, FACE_BIAS);
    double E[3];
    cross3v(E, Ax, Bx); SAT_TEST(E, dot3(E, E), 6,  EDGE_BIAS);
    cross3v(E, Ax, By); SAT_TEST(E, dot3(E, E), 7,  EDGE_BIAS);
    cross3v(E, Ax, Bz); SAT_TEST(E, dot3(E, E), 8,  EDGE_BIAS);
    cross3v(E, Ay, Bx); SAT_TEST(E, dot3(E, E), 9,  EDGE_BIAS);
    cross3v(E, Ay, By); SAT_TEST(E, dot3(E, E), 10, EDGE_BIAS);
    cross3v(E, Ay, Bz); SAT_TEST(E, dot3(E, E), 11, EDGE_BIAS);
    cross3v(E, Az, Bx); SAT_TEST(E, dot3(E, E), 12, EDGE_BIAS);
    cross3v(E, Az, By); SAT_TEST(E, dot3(E, E), 13, EDGE_BIAS);
    cross3v(E, Az, Bz); SAT_TEST(E, dot3(E, E), 14, EDGE_BIAS);

#undef SAT_TEST

    if (best_code < 0) return -1;
    box_box_debug_best_code = best_code;     /* expose for debug */

    /* Orient normal: param1 → param2. */
    double n[3] = {best_axis[0], best_axis[1], best_axis[2]};
    if (dot3(n, D) < 0.0) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }

    /* Unbiased depth along the chosen unit axis (sat_penetration on the unit
     * vector returns the actual signed penetration). */
    double depth = sat_penetration(n, D, R1, h1, R2, h2, 1.0);
    if (depth < 0.0) depth = 0.0;

    if (max_pts <= 0) return 0;

    if (best_code <= 5) {
        /* Face winner → face-face / face-vertex manifold via clipping. */
        return face_face_manifold(best_code, n, c1, R1, h1, c2, R2, h2, out, max_pts);
    } else {
        /* Edge-cross winner → closest pair of edges (single contact point). */
        return edge_edge_contact(best_code, n, depth, c1, R1, h1, c2, R2, h2, out, max_pts);
    }
}
