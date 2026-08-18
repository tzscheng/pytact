/* mpr.c — generic convex narrowphase: MPR (Minkowski Portal Refinement) + EPA
 * portal, the fallback used by tact_collision_check for shape pairs without a
 * dedicated detector. Holds the libccd-derived ccd_* core, the per-shape support
 * function (reads mesh vertices from shape.c), size_of_param, and tact_collision_check_mpr.
 * (Formerly the bulk of ccd.c.) */
/* Portions of this file are derived from libccd.
 * Copyright (c) 2010-2012 Daniel Fiser and the libccd contributors.
 * Licensed under the BSD 3-Clause License. See docs/THIRD_PARTY_NOTICES for
 * the complete upstream copyright and license notice. */
#include <float.h>
#include "core.h"
#include "shape.h"

#define CCD_OBJ_MESH    TACT_MESH
#define CCD_OBJ_BOX     TACT_BOX
#define CCD_OBJ_SPHERE  TACT_SPHERE
#define CCD_OBJ_CYL     TACT_CYL
#define CCD_OBJ_CAPSULE TACT_CAPSULE
#define CCD_OBJ_HFIELD  TACT_HFIELD

typedef struct {
    double v[3];
} ccd_vec3_t;

typedef struct {
    double q[4]; //!< x, y, z, w
} ccd_quat_t;

typedef struct{
    int type;
    ccd_vec3_t pos;
    ccd_quat_t quat;
    double param[16]; //box: x, y, z  sphere: radius  cylinder: radius, height
} ccd_obj_t;

typedef struct{
    ccd_vec3_t v;  //!< Support point in minkowski sum
    ccd_vec3_t v1; //!< Support point in obj1
    ccd_vec3_t v2; //!< Support point in obj2
} ccd_support_t;

typedef struct{
    ccd_support_t ps[4];
    int last; //!< index of last added point
} ccd_simplex_t;

typedef void (*ccd_support_fn)(const void *obj, const ccd_vec3_t *dir, ccd_vec3_t *vec);
typedef void (*ccd_first_dir_fn)(const void *obj1, const void *obj2, ccd_vec3_t *dir);
typedef void (*ccd_center_fn)(const void *obj1, ccd_vec3_t *center);

typedef struct{
    ccd_support_fn support1; //!< Function that returns support point of first object
    ccd_support_fn support2; //!< Function that returns support point of second object
    ccd_center_fn center1; //!< Function that returns geometric center of first object
    ccd_center_fn center2; //!< Function that returns geometric center of second object
    unsigned long max_iterations; //!< Maximal number of iterations
    double mpr_tolerance; //!< Boundary tolerance for MPR algorithm
} ccd_t;


ccd_vec3_t _ccd_vec3_origin = {{0, 0, 0}};
const ccd_vec3_t* ccd_vec3_origin = &_ccd_vec3_origin;

int ccdIsZero(double val){
    return fabs(val) < DBL_EPSILON;
}

void ccdVec3Sub2(ccd_vec3_t *d, const ccd_vec3_t *v, const ccd_vec3_t *w){
    d->v[0] = v->v[0] - w->v[0];
    d->v[1] = v->v[1] - w->v[1];
    d->v[2] = v->v[2] - w->v[2];
}

void ccdVec3Set(ccd_vec3_t *v, double x, double y, double z){
    v->v[0] = x;
    v->v[1] = y;
    v->v[2] = z;
}

void ccdVec3Add(ccd_vec3_t *v, const ccd_vec3_t *w){
    v->v[0] += w->v[0];
    v->v[1] += w->v[1];
    v->v[2] += w->v[2];
}

void ccdVec3Scale(ccd_vec3_t *d, double k){
    d->v[0] *= k;
    d->v[1] *= k;
    d->v[2] *= k;
}

double ccdVec3Dot(const ccd_vec3_t *a, const ccd_vec3_t *b){
    double dot;

    dot  = a->v[0] * b->v[0];
    dot += a->v[1] * b->v[1];
    dot += a->v[2] * b->v[2];
    return dot;
}

void ccdVec3Cross(ccd_vec3_t *d, const ccd_vec3_t *a, const ccd_vec3_t *b){
    d->v[0] = (a->v[1] * b->v[2]) - (a->v[2] * b->v[1]);
    d->v[1] = (a->v[2] * b->v[0]) - (a->v[0] * b->v[2]);
    d->v[2] = (a->v[0] * b->v[1]) - (a->v[1] * b->v[0]);
}

double ccdVec3Dist2(const ccd_vec3_t *a, const ccd_vec3_t *b){
    ccd_vec3_t ab;
    ccdVec3Sub2(&ab, a, b);
    return ccdVec3Dot(&ab, &ab); // ccdVec3Len2(&ab);
}

void ccdVec3Normalize(ccd_vec3_t *d){
    double k = 1.0 / sqrt(ccdVec3Dot(d, d)); //sqrt(ccdVec3Len2(d));
    ccdVec3Scale(d, k);
}

void ccdVec3Copy(ccd_vec3_t *v, const ccd_vec3_t *w){
    *v = *w;
}

int ccdSign(double val){
    if (ccdIsZero(val)) return 0;
    else if (val < 0.0) return -1;
    return 1;
}

int ccdEq(double _a, double _b){
    double ab;
    double a, b;

    ab = fabs(_a - _b);
    if (fabs(ab) < DBL_EPSILON) return 1;

    a = fabs(_a);
    b = fabs(_b);
    if (b > a) return ab < DBL_EPSILON * b;
    else return ab < DBL_EPSILON * a;
}

int ccdVec3Eq(const ccd_vec3_t *a, const ccd_vec3_t *b){
    return ccdEq(a->v[0], b->v[0]) && ccdEq(a->v[1], b->v[1]) && ccdEq(a->v[2], b->v[2]);
}

double __ccdVec3PointSegmentDist2(const ccd_vec3_t *P, const ccd_vec3_t *x0, const ccd_vec3_t *b, ccd_vec3_t *witness){
    // The computation comes from solving equation of segment:
    //      S(t) = x0 + t.d
    //          where - x0 is initial point of segment
    //                - d is direction of segment from x0 (|d| > 0)
    //                - t belongs to <0, 1> interval
    // 
    // Than, distance from a segment to some point P can be expressed:
    //      D(t) = |x0 + t.d - P|^2
    //          which is distance from any point on segment. Minimization
    //          of this function brings distance from P to segment.
    // Minimization of D(t) leads to simple quadratic equation that's
    // solving is straightforward.
    //
    // Bonus of this method is witness point for free.

    double dist, t;
    ccd_vec3_t d, a;

    // direction of segment
    ccdVec3Sub2(&d, b, x0);

    // precompute vector from P to x0
    ccdVec3Sub2(&a, x0, P);

    t  = -1.0 * ccdVec3Dot(&a, &d);
    t /= ccdVec3Dot(&d, &d); // ccdVec3Len2(&d);

    if (t < 0.0 || ccdIsZero(t)){
        dist = ccdVec3Dist2(x0, P);
	if (witness) ccdVec3Copy(witness, x0);
    }

    else if (t > 1.0 || ccdEq(t, 1.0)){
        dist = ccdVec3Dist2(b, P);
        if (witness) ccdVec3Copy(witness, b);
    }

    else{
        if (witness) {
            ccdVec3Copy(witness, &d);
            ccdVec3Scale(witness, t);
            ccdVec3Add(witness, x0);
            dist = ccdVec3Dist2(witness, P);
        }

	else {
            // recycling variables
            ccdVec3Scale(&d, t);
            ccdVec3Add(&d, &a);
            dist = ccdVec3Dot(&d, &d); // ccdVec3Len2(&d);
        }
    }

    return dist;
}

double ccdVec3PointTriDist2(const ccd_vec3_t *P, const ccd_vec3_t *x0, const ccd_vec3_t *B, const ccd_vec3_t *C, ccd_vec3_t *witness){
    // Computation comes from analytic expression for triangle (x0, B, C)
    //      T(s, t) = x0 + s.d1 + t.d2, where d1 = B - x0 and d2 = C - x0 and
    // Then equation for distance is:
    //      D(s, t) = | T(s, t) - P |^2
    // This leads to minimization of quadratic function of two variables.
    // The solution from is taken only if s is between 0 and 1, t is
    // between 0 and 1 and t + s < 1, otherwise distance from segment is
    // computed.

    ccd_vec3_t d1, d2, a;
    double u, v, w, p, q, r, d;
    double s, t, dist, dist2;
    ccd_vec3_t witness2;

    ccdVec3Sub2(&d1, B, x0);
    ccdVec3Sub2(&d2, C, x0);
    ccdVec3Sub2(&a, x0, P);

    u = ccdVec3Dot(&a, &a);
    v = ccdVec3Dot(&d1, &d1);
    w = ccdVec3Dot(&d2, &d2);
    p = ccdVec3Dot(&a, &d1);
    q = ccdVec3Dot(&a, &d2);
    r = ccdVec3Dot(&d1, &d2);

    d = w * v - r * r;
    if (ccdIsZero(d)){
        // To avoid division by zero for zero (or near zero) area triangles
        s = t = -1.;
    }

    else {
        s = (q * r - w * p) / d;
        t = (-s * r - q) / w;
    }

    if ((ccdIsZero(s) || s > 0.0) && (ccdEq(s, 1.0) || s < 1.0) && (ccdIsZero(t) || t > 0.0) && (ccdEq(t, 1.0) || t < 1.0) && (ccdEq(t + s, 1.0) || t + s < 1.0)){
	
        if (witness) {
            ccdVec3Scale(&d1, s);
            ccdVec3Scale(&d2, t);
            ccdVec3Copy(witness, x0);
            ccdVec3Add(witness, &d1);
            ccdVec3Add(witness, &d2);
            dist = ccdVec3Dist2(witness, P);
        }

	else {
            dist  = s * s * v;
            dist += t * t * w;
            dist += 2.0 * s * t * r;
            dist += 2.0 * s * p;
            dist += 2.0 * t * q;
            dist += u;
        }
    }

    else{
        dist = __ccdVec3PointSegmentDist2(P, x0, B, witness);
        dist2 = __ccdVec3PointSegmentDist2(P, x0, C, &witness2);
	
        if (dist2 < dist){
            dist = dist2;
            if (witness) ccdVec3Copy(witness, &witness2);
        }

        dist2 = __ccdVec3PointSegmentDist2(P, B, C, &witness2);
	if (dist2 < dist){
            dist = dist2;
            if (witness) ccdVec3Copy(witness, &witness2);
        }
    }

    return dist;
}

double ccdQuatLen2(const ccd_quat_t *q){
    double len;
    len  = q->q[0] * q->q[0];
    len += q->q[1] * q->q[1];
    len += q->q[2] * q->q[2];
    len += q->q[3] * q->q[3];

    return len;
}

int ccdQuatInvert(ccd_quat_t *q){
    double len2 = ccdQuatLen2(q);
    if (len2 < DBL_EPSILON) return -1;
    len2 = 1.0 / len2;

    q->q[0] = -q->q[0] * len2;
    q->q[1] = -q->q[1] * len2;
    q->q[2] = -q->q[2] * len2;
    q->q[3] = q->q[3] * len2;

    return 0;
}

int ccdQuatInvert2(ccd_quat_t *dest, const ccd_quat_t *src){
    *dest = *src;  //ccdQuatCopy(dest, src);
    return ccdQuatInvert(dest);
}

void ccdQuatRotVec(ccd_vec3_t *v, const ccd_quat_t *q){
    // original version: 31 mul + 21 add
    // optimized version: 18 mul + 12 add
    // formula: v = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    double cross1_x, cross1_y, cross1_z, cross2_x, cross2_y, cross2_z;
    double x, y, z, w;
    double vx, vy, vz;

    vx = v->v[0]; //ccdVec3X(v);
    vy = v->v[1]; //ccdVec3Y(v);
    vz = v->v[2]; //ccdVec3Z(v);

    w = q->q[3];
    x = q->q[0];
    y = q->q[1];
    z = q->q[2];

    cross1_x = y * vz - z * vy + w * vx;
    cross1_y = z * vx - x * vz + w * vy;
    cross1_z = x * vy - y * vx + w * vz;
    cross2_x = y * cross1_z - z * cross1_y;
    cross2_y = z * cross1_x - x * cross1_z;
    cross2_z = x * cross1_y - y * cross1_x;
    ccdVec3Set(v, vx + 2 * cross2_x, vy + 2 * cross2_y, vz + 2 * cross2_z);
}

const ccd_support_t *ccdSimplexPoint(const ccd_simplex_t *s, int idx){
    // here is no check on boundaries
    return &s->ps[idx];
}

void ccdSupportCopy(ccd_support_t *d, const ccd_support_t *s){
    *d = *s;
}

void __ccdSupport(const void *obj1, const void *obj2, const ccd_vec3_t *_dir, const ccd_t *ccd, ccd_support_t *supp){
    ccd_vec3_t dir;
    
    ccdVec3Copy(&dir, _dir);
    ccd->support1(obj1, &dir, &supp->v1);
    ccdVec3Scale(&dir, -1.0);
    ccd->support2(obj2, &dir, &supp->v2);
    ccdVec3Sub2(&supp->v, &supp->v1, &supp->v2);
}

int ccdSimplexSize(const ccd_simplex_t *s){
    return s->last + 1;
}

ccd_support_t *ccdSimplexPointW(ccd_simplex_t *s, int idx){
    return &s->ps[idx];
}

void ccdSimplexSetSize(ccd_simplex_t *s, int size){
    s->last = size - 1;
}

void ccdSimplexSet(ccd_simplex_t *s, size_t pos, const ccd_support_t *a){
    ccdSupportCopy(s->ps + pos, a);
}

void ccdSimplexSwap(ccd_simplex_t *s, size_t pos1, size_t pos2){
    ccd_support_t supp;
    ccdSupportCopy(&supp, &s->ps[pos1]);
    ccdSupportCopy(&s->ps[pos1], &s->ps[pos2]);
    ccdSupportCopy(&s->ps[pos2], &supp);
}

void findOrigin(const void *obj1, const void *obj2, const ccd_t *ccd, ccd_support_t *center){
    ccd->center1(obj1, &center->v1);
    ccd->center2(obj2, &center->v2);
    ccdVec3Sub2(&center->v, &center->v1, &center->v2);
}


void portalDir(const ccd_simplex_t *portal, ccd_vec3_t *dir){
    ccd_vec3_t v2v1, v3v1;
    ccdVec3Sub2(&v2v1, &ccdSimplexPoint(portal, 2)->v, &ccdSimplexPoint(portal, 1)->v);
    ccdVec3Sub2(&v3v1, &ccdSimplexPoint(portal, 3)->v, &ccdSimplexPoint(portal, 1)->v);
    ccdVec3Cross(dir, &v2v1, &v3v1);
    ccdVec3Normalize(dir);
}

int portalEncapsulesOrigin(const ccd_simplex_t *portal, const ccd_vec3_t *dir){
    double dot;
    dot = ccdVec3Dot(dir, &ccdSimplexPoint(portal, 1)->v);
    return ccdIsZero(dot) || dot > 0.0;
}

//int portalCanEncapsuleOrigin(const ccd_simplex_t *portal, const ccd_support_t *v4, const ccd_vec3_t *dir){
int portalCanEncapsuleOrigin(const ccd_support_t *v4, const ccd_vec3_t *dir){
    
    double dot;
    dot = ccdVec3Dot(&v4->v, dir);
    return ccdIsZero(dot) || dot > 0.0;
}

void expandPortal(ccd_simplex_t *portal, const ccd_support_t *v4){
    double dot;
    ccd_vec3_t v4v0;

    ccdVec3Cross(&v4v0, &v4->v, &ccdSimplexPoint(portal, 0)->v);
    dot = ccdVec3Dot(&ccdSimplexPoint(portal, 1)->v, &v4v0);
    if (dot > 0.0){
        dot = ccdVec3Dot(&ccdSimplexPoint(portal, 2)->v, &v4v0);
        if (dot > 0.0) ccdSimplexSet(portal, 1, v4);
	else ccdSimplexSet(portal, 3, v4);
    }

    else {
        dot = ccdVec3Dot(&ccdSimplexPoint(portal, 3)->v, &v4v0);
        if (dot > 0.0) ccdSimplexSet(portal, 2, v4);
	else ccdSimplexSet(portal, 1, v4);
    }
}

int portalReachTolerance(const ccd_simplex_t *portal, const ccd_support_t *v4, const ccd_vec3_t *dir, const ccd_t *ccd){
    double dv1, dv2, dv3, dv4;
    double dot1, dot2, dot3;

    // find the smallest dot product of dir and {v1-v4, v2-v4, v3-v4}

    dv1 = ccdVec3Dot(&ccdSimplexPoint(portal, 1)->v, dir);
    dv2 = ccdVec3Dot(&ccdSimplexPoint(portal, 2)->v, dir);
    dv3 = ccdVec3Dot(&ccdSimplexPoint(portal, 3)->v, dir);
    dv4 = ccdVec3Dot(&v4->v, dir);

    dot1 = dv4 - dv1;
    dot2 = dv4 - dv2;
    dot3 = dv4 - dv3;

    dot1 = fmin(dot1, dot2);
    dot1 = fmin(dot1, dot3);

    return ccdEq(dot1, ccd->mpr_tolerance) || dot1 < ccd->mpr_tolerance;
}

int discoverPortal(const void *obj1, const void *obj2, const ccd_t *ccd, ccd_simplex_t *portal){
    ccd_vec3_t dir, va, vb;
    double dot;
    int cont;

    // vertex 0 is center of portal
    findOrigin(obj1, obj2, ccd, ccdSimplexPointW(portal, 0));
    ccdSimplexSetSize(portal, 1);

    if (ccdVec3Eq(&ccdSimplexPoint(portal, 0)->v, ccd_vec3_origin)){
        // Portal's center lies on origin (0,0,0) => we know that objects
        // intersect but we would need to know penetration info.
        // So move center little bit...
        ccdVec3Set(&va, DBL_EPSILON * 10.0, 0.0, 0.0);
        ccdVec3Add(&ccdSimplexPointW(portal, 0)->v, &va);
    }


    // vertex 1 = support in direction of origin
    ccdVec3Copy(&dir, &ccdSimplexPoint(portal, 0)->v);
    ccdVec3Scale(&dir, -1.0);
    ccdVec3Normalize(&dir);
    __ccdSupport(obj1, obj2, &dir, ccd, ccdSimplexPointW(portal, 1));
    ccdSimplexSetSize(portal, 2);

    // test if origin isn't outside of v1
    dot = ccdVec3Dot(&ccdSimplexPoint(portal, 1)->v, &dir);
    if (ccdIsZero(dot) || dot < 0.0) return -1;


    // vertex 2
    ccdVec3Cross(&dir, &ccdSimplexPoint(portal, 0)->v, &ccdSimplexPoint(portal, 1)->v);
    if (ccdIsZero(ccdVec3Dot(&dir, &dir))){
        if (ccdVec3Eq(&ccdSimplexPoint(portal, 1)->v, ccd_vec3_origin)) return 1;
        else return 2;
    }

    ccdVec3Normalize(&dir);
    __ccdSupport(obj1, obj2, &dir, ccd, ccdSimplexPointW(portal, 2));
    dot = ccdVec3Dot(&ccdSimplexPoint(portal, 2)->v, &dir);
    if (ccdIsZero(dot) || dot < 0.0) return -1;

    ccdSimplexSetSize(portal, 3);

    // vertex 3 direction
    ccdVec3Sub2(&va, &ccdSimplexPoint(portal, 1)->v, &ccdSimplexPoint(portal, 0)->v);
    ccdVec3Sub2(&vb, &ccdSimplexPoint(portal, 2)->v, &ccdSimplexPoint(portal, 0)->v);
    ccdVec3Cross(&dir, &va, &vb);
    ccdVec3Normalize(&dir);

    // it is better to form portal faces to be oriented "outside" origin
    dot = ccdVec3Dot(&dir, &ccdSimplexPoint(portal, 0)->v);
    if (dot > 0.0){
        ccdSimplexSwap(portal, 1, 2);
        ccdVec3Scale(&dir, -1.0);
    }

    while (ccdSimplexSize(portal) < 4){
        __ccdSupport(obj1, obj2, &dir, ccd, ccdSimplexPointW(portal, 3));
        dot = ccdVec3Dot(&ccdSimplexPoint(portal, 3)->v, &dir);
        if (ccdIsZero(dot) || dot < 0.0) return -1;
        cont = 0;

        // test if origin is outside (v1, v0, v3) - set v2 as v3 and
        // continue
        ccdVec3Cross(&va, &ccdSimplexPoint(portal, 1)->v, &ccdSimplexPoint(portal, 3)->v);
        dot = ccdVec3Dot(&va, &ccdSimplexPoint(portal, 0)->v);
	
        if (dot < 0.0 && !ccdIsZero(dot)){
            ccdSimplexSet(portal, 2, ccdSimplexPoint(portal, 3));
            cont = 1;
        }

        if (!cont){
            // test if origin is outside (v3, v0, v2) - set v1 as v3 and
            // continue
            ccdVec3Cross(&va, &ccdSimplexPoint(portal, 3)->v, &ccdSimplexPoint(portal, 2)->v);
            dot = ccdVec3Dot(&va, &ccdSimplexPoint(portal, 0)->v);
	    
            if (dot < 0.0 && !ccdIsZero(dot)){
                ccdSimplexSet(portal, 1, ccdSimplexPoint(portal, 3));
                cont = 1;
            }
        }

        if (cont){
            ccdVec3Sub2(&va, &ccdSimplexPoint(portal, 1)->v, &ccdSimplexPoint(portal, 0)->v);
            ccdVec3Sub2(&vb, &ccdSimplexPoint(portal, 2)->v, &ccdSimplexPoint(portal, 0)->v);
            ccdVec3Cross(&dir, &va, &vb);
            ccdVec3Normalize(&dir);
        }

	else ccdSimplexSetSize(portal, 4);
        
    }

    return 0;
}

//void findPenetrTouch(const void *obj1, const void *obj2, const ccd_t *ccd, ccd_simplex_t *portal, double *depth, ccd_vec3_t *dir, ccd_vec3_t *pos){
void findPenetrTouch(ccd_simplex_t *portal, double *depth, ccd_vec3_t *dir, ccd_vec3_t *pos){
    
    // Touching contact on portal's v1 - so depth is zero and direction
    // is unimportant and pos can be guessed
    *depth = 0.0;
    ccdVec3Copy(dir, ccd_vec3_origin);

    ccdVec3Copy(pos, &ccdSimplexPoint(portal, 1)->v1);
    ccdVec3Add(pos, &ccdSimplexPoint(portal, 1)->v2);
    ccdVec3Scale(pos, 0.5);
}

//void findPenetrSegment(const void *obj1, const void *obj2, const ccd_t *ccd, ccd_simplex_t *portal, double *depth, ccd_vec3_t *dir, ccd_vec3_t *pos){
void findPenetrSegment(ccd_simplex_t *portal, double *depth, ccd_vec3_t *dir, ccd_vec3_t *pos){
    
    ccdVec3Copy(pos, &ccdSimplexPoint(portal, 1)->v1);
    ccdVec3Add(pos, &ccdSimplexPoint(portal, 1)->v2);
    ccdVec3Scale(pos, 0.5);

    ccdVec3Copy(dir, &ccdSimplexPoint(portal, 1)->v);
    *depth = sqrt(ccdVec3Dot(dir, dir)); //sqrt(ccdVec3Len2(dir));
    ccdVec3Normalize(dir);
}


//void findPos(const void *obj1, const void *obj2, const ccd_t *ccd, const ccd_simplex_t *portal, ccd_vec3_t *pos){
void findPos(const ccd_simplex_t *portal, ccd_vec3_t *pos){
    
    ccd_vec3_t dir;
    size_t i;
    double b[4], sum, inv;
    ccd_vec3_t vec, p1, p2;

    portalDir(portal, &dir);

    // use barycentric coordinates of tetrahedron to find origin
    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 1)->v, &ccdSimplexPoint(portal, 2)->v);
    b[0] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 3)->v);

    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 3)->v, &ccdSimplexPoint(portal, 2)->v);
    b[1] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 0)->v);

    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 0)->v, &ccdSimplexPoint(portal, 1)->v);
    b[2] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 3)->v);

    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 2)->v, &ccdSimplexPoint(portal, 1)->v);
    b[3] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 0)->v);

    sum = b[0] + b[1] + b[2] + b[3];

    if (ccdIsZero(sum) || sum < 0.0){
	b[0] = 0.0;

        ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 2)->v, &ccdSimplexPoint(portal, 3)->v);
        b[1] = ccdVec3Dot(&vec, &dir);
        ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 3)->v, &ccdSimplexPoint(portal, 1)->v);
        b[2] = ccdVec3Dot(&vec, &dir);
        ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 1)->v, &ccdSimplexPoint(portal, 2)->v);
        b[3] = ccdVec3Dot(&vec, &dir);
	sum = b[1] + b[2] + b[3];
    }

    inv = 1.0 / sum;
    ccdVec3Copy(&p1, ccd_vec3_origin);
    ccdVec3Copy(&p2, ccd_vec3_origin);

    for (i = 0; i < 4; i++){
        ccdVec3Copy(&vec, &ccdSimplexPoint(portal, i)->v1);
        ccdVec3Scale(&vec, b[i]);
        ccdVec3Add(&p1, &vec);

        ccdVec3Copy(&vec, &ccdSimplexPoint(portal, i)->v2);
        ccdVec3Scale(&vec, b[i]);
        ccdVec3Add(&p2, &vec);
    }

    ccdVec3Scale(&p1, inv);
    ccdVec3Scale(&p2, inv);

    ccdVec3Copy(pos, &p1);
    ccdVec3Add(pos, &p2);
    ccdVec3Scale(pos, 0.5);
}

void findPenetr(const void *obj1, const void *obj2, const ccd_t *ccd, ccd_simplex_t *portal, double *depth, ccd_vec3_t *pdir, ccd_vec3_t *pos){
    ccd_vec3_t dir;
    ccd_support_t v4;
    unsigned long iterations;

    iterations = 0UL;
    while (1){
        // compute portal direction and obtain next support point
        portalDir(portal, &dir);
        __ccdSupport(obj1, obj2, &dir, ccd, &v4);

        // reached tolerance -> find penetration info
        if (portalReachTolerance(portal, &v4, &dir, ccd) || iterations > ccd->max_iterations){
            *depth = ccdVec3PointTriDist2(ccd_vec3_origin, &ccdSimplexPoint(portal, 1)->v, &ccdSimplexPoint(portal, 2)->v, &ccdSimplexPoint(portal, 3)->v, pdir);
            *depth = sqrt(*depth);
	    
            if (ccdIsZero(*depth)){
                // If depth is zero, then we have a touching contact.
                // So following findPenetrTouch(), we assign zero to
                // the direction vector (it can actually be anything
                // according to the decription of ccdMPRPenetration
                // function).
                ccdVec3Copy(pdir, ccd_vec3_origin);
            }

	    else ccdVec3Normalize(pdir);

            // barycentric coordinates:
            //findPos(obj1, obj2, ccd, portal, pos);
	    findPos(portal, pos);
            return;
        }

        expandPortal(portal, &v4);
        iterations++;
    }
}

int refinePortal(const void *obj1, const void *obj2, const ccd_t *ccd, ccd_simplex_t *portal){
    ccd_vec3_t dir;
    ccd_support_t v4;

    while (1){
        // compute direction outside the portal (from v0 throught v1,v2,v3 face)
        portalDir(portal, &dir);

        // test if origin is inside the portal
        if (portalEncapsulesOrigin(portal, &dir)) return 0;

        // get next support point
        __ccdSupport(obj1, obj2, &dir, ccd, &v4);

        // test if v4 can expand portal to contain origin and if portal expanding doesn't reach given tolerance
        //if (!portalCanEncapsuleOrigin(portal, &v4, &dir) || portalReachTolerance(portal, &v4, &dir, ccd)) {return -1;}
	if (!portalCanEncapsuleOrigin(&v4, &dir) || portalReachTolerance(portal, &v4, &dir, ccd)) {return -1;}

        // v1-v2-v3 triangle must be rearranged to face outside Minkowski difference (direction from v0).
        expandPortal(portal, &v4);
    }

    return -1;
}

int ccdMPRPenetration(const void *obj1, const void *obj2, const ccd_t *ccd, double *depth, ccd_vec3_t *dir, ccd_vec3_t *pos){
    ccd_simplex_t portal;
    int res;

    // Phase 1: Portal discovery
    res = discoverPortal(obj1, obj2, ccd, &portal);
    if (res < 0){
        // Origin isn't inside portal - no collision.
        return -1;
    }

    else if (res == 1){
        // Touching contact on portal's v1.
        //findPenetrTouch(obj1, obj2, ccd, &portal, depth, dir, pos);
	findPenetrTouch(&portal, depth, dir, pos);
    }

    else if (res == 2){
        // Origin lies on v0-v1 segment.
        //findPenetrSegment(obj1, obj2, ccd, &portal, depth, dir, pos);
	findPenetrSegment(&portal, depth, dir, pos);
    }

    else if (res == 0){
        // Phase 2: Portal refinement
        res = refinePortal(obj1, obj2, ccd, &portal);
        if (res < 0) return -1;

        // Phase 3. Penetration info
        findPenetr(obj1, obj2, ccd, &portal, depth, dir, pos);
    }

    return 0;
}

void ccd_center(const void *_obj, ccd_vec3_t *center){
    ccd_obj_t *obj = (ccd_obj_t *)_obj;
    ccdVec3Set(center, 0.0, 0.0, 0.0);
    // rotation is not needed
    ccdVec3Add(center, &obj->pos);
}

//Support functions from Gino van den Bergen's paper : A Fast and Robust CCD Implementation for Collision Detection of Convex Objects
void ccd_support(const void *_obj, const ccd_vec3_t *_dir, ccd_vec3_t *v) {
    ccd_obj_t *obj = (ccd_obj_t *)_obj;
    ccd_vec3_t dir;
    ccd_quat_t qinv;

    ccdVec3Copy(&dir, _dir);
    ccdQuatInvert2(&qinv, &obj->quat);
    ccdQuatRotVec(&dir, &qinv);

    if (obj->type == CCD_OBJ_MESH){
	double max_dot = -FLT_MAX;
	int mesh_idx = (int)obj->param[0];
	
	if (num_vertex[mesh_idx] == 0) {
	    num_vertex[mesh_idx] = load_mesh(mesh_idx);
	}
	
	for (int i = 0; i < num_vertex[mesh_idx]; i++) {
	    double d = vertex[mesh_idx][i][0]*dir.v[0] + vertex[mesh_idx][i][1]*dir.v[1] + vertex[mesh_idx][i][2]*dir.v[2];
	
	    if (d > max_dot) {
		max_dot = d;
		ccdVec3Set(v, vertex[mesh_idx][i][0], vertex[mesh_idx][i][1], vertex[mesh_idx][i][2]);
	    }
	}
    }
    
    else if (obj->type == CCD_OBJ_BOX){
	double hx = obj->param[0];
	double hy = obj->param[1];
	double hz = obj->param[2];
	ccdVec3Set(v, ccdSign(dir.v[0])*hx, ccdSign(dir.v[1])*hy, ccdSign(dir.v[2])*hz);
    }

    else if (obj->type == CCD_OBJ_SPHERE){
        double len = ccdVec3Dot(&dir, &dir); //ccdVec3Len2(&dir);
	double radius = obj->param[0];
	
        if (len - DBL_EPSILON > 0.0){
            ccdVec3Copy(v, &dir);
            ccdVec3Scale(v, radius / sqrt(len));
        }

	else ccdVec3Set(v, 0.0, 0.0, 0.0);
    }

    else if (obj->type == CCD_OBJ_CYL){
        double zdist = sqrt(dir.v[0] * dir.v[0] + dir.v[1] * dir.v[1]);
	double radius = obj->param[0];
	double hh     = obj->param[1];   /* half-height (MuJoCo convention) */

	if (ccdIsZero(zdist)){
	    ccdVec3Set(v, 0.0, 0.0, ccdSign(dir.v[2])*hh);
        }

	else {
            double rad = radius / zdist;
	    ccdVec3Set(v, rad*dir.v[0], rad*dir.v[1], ccdSign(dir.v[2])*hh);
        }
    }

    else if (obj->type == CCD_OBJ_CAPSULE){
	double len = sqrtf(dir.v[0]*dir.v[0] + dir.v[1]*dir.v[1] + dir.v[2]*dir.v[2]);
	double radius = obj->param[0];
	double hh     = obj->param[1];   /* half cylindrical-section length */

	if(len < 1e-6f){
	    ccdVec3Set(v, 0.0, 0.0, 0.0);
	}

	else {
	    double dx = dir.v[0]/len;
	    double dy = dir.v[1]/len;
	    double dz = dir.v[2]/len;
	    double cap_z = (dz > 0.0 ? hh : -hh);
	    ccdVec3Set(v, dx*radius, dy*radius, cap_z + dz*radius);
	}
    }
		       
    // transform support vertex
    ccdQuatRotVec(v, &obj->quat);
    ccdVec3Add(v, &obj->pos);
}

int size_of_param(int type){
    int ret = -1;
    switch(type){
    case CCD_OBJ_MESH: ret = sizeof(double); break;
    case CCD_OBJ_BOX: ret = 3*sizeof(double); break;
    case CCD_OBJ_SPHERE: ret = sizeof(double); break;
    case CCD_OBJ_CYL: ret = 2*sizeof(double); break;
    case CCD_OBJ_CAPSULE: ret = 2*sizeof(double); break;
    case CCD_OBJ_HFIELD: ret = sizeof(double); break;   /* slot id (like mesh) */
    default: printf("wrong type = %d\n", type);
    }

    return ret;
}

/* Single-point MPR-based collision check — the original implementation. Used as
 * the fallback path inside tact_collision_check() for all shape combinations that
 * don't have a dedicated multi-point detector. Exposed (non-static) for sign-
 * convention regression testing of new dedicated detectors. */
int tact_collision_check_mpr(int type1, double* param1, int type2, double *param2, double* out){
      ccd_t ccd;

      ccd.support1 = ccd_support;
      ccd.support2 = ccd_support;
      ccd.center1 = ccd_center;
      ccd.center2 = ccd_center;
      ccd.max_iterations = (unsigned long)-1;
      ccd.mpr_tolerance = 0.000001;

      ccd_obj_t obj1, obj2;
      double q1[4], q2[4];

      obj1.type = type1;
      memcpy(obj1.pos.v, param1, 3*sizeof(double));
      euler_to_quat(param1+3, q1, "xyz");
      memcpy(obj1.quat.q, q1, 4*sizeof(double));
      memcpy(obj1.param, param1+6, size_of_param(type1));

      obj2.type = type2;
      memcpy(obj2.pos.v, param2, 3*sizeof(double));
      euler_to_quat(param2+3, q2, "xyz");
      memcpy(obj2.quat.q, q2, 4*sizeof(double));
      memcpy(obj2.param, param2+6, size_of_param(type2));

      double depth;
      ccd_vec3_t dir, pos;

      int intersect = ccdMPRPenetration(&obj1, &obj2, &ccd, &depth, &dir, &pos);

      out[0] = pos.v[0];
      out[1] = pos.v[1];
      out[2] = pos.v[2];
      out[3] = dir.v[0];
      out[4] = dir.v[1];
      out[5] = dir.v[2];
      out[6] = depth;

      return intersect;
}
