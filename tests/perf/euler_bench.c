/* Bench: explicit fast path vs general (3 elementary rots + 2 matmul) vs smart switch.
 * Build: gcc -O3 -march=native -ffast-math -funroll-loops -o euler_bench euler_bench.c -lm
 *
 * - fast_explicit:  9 element formulas inlined for one sequence (mirrors tact.py fast path)
 * - general_naive:  build R0,R1,R2 then matmul twice (mirrors tact.py general path)
 * - smart_switch:   24-way switch with explicit formula inlined per case
 *
 * To prevent dead-code elimination, each iteration nudges the input and accumulates a checksum.
 */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef double M3[3][3];

/* ---------- elementary rotations (used by general path) ---------- */
static inline void rot_x(double t, M3 R) {
    double c = cos(t), s = sin(t);
    R[0][0]=1; R[0][1]=0; R[0][2]=0;
    R[1][0]=0; R[1][1]=c; R[1][2]=-s;
    R[2][0]=0; R[2][1]=s; R[2][2]=c;
}
static inline void rot_y(double t, M3 R) {
    double c = cos(t), s = sin(t);
    R[0][0]=c;  R[0][1]=0; R[0][2]=s;
    R[1][0]=0;  R[1][1]=1; R[1][2]=0;
    R[2][0]=-s; R[2][1]=0; R[2][2]=c;
}
static inline void rot_z(double t, M3 R) {
    double c = cos(t), s = sin(t);
    R[0][0]=c; R[0][1]=-s; R[0][2]=0;
    R[1][0]=s; R[1][1]=c;  R[1][2]=0;
    R[2][0]=0; R[2][1]=0;  R[2][2]=1;
}
static inline void mm3(const M3 A, const M3 B, M3 C) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += A[i][k]*B[k][j];
            C[i][j] = s;
        }
}

/* ---------- 1) fast_explicit: tact.py's fast-path body for XYZ intrinsic ---------- */
static inline void fast_explicit_XYZ(const double x[3], M3 R) {
    double c0=cos(x[0]), s0=sin(x[0]);
    double c1=cos(x[1]), s1=sin(x[1]);
    double c2=cos(x[2]), s2=sin(x[2]);
    R[0][0] =  c1*c2;
    R[0][1] = -c1*s2;
    R[0][2] =  s1;
    R[1][0] =  s0*s1*c2 + c0*s2;
    R[1][1] = -s0*s1*s2 + c0*c2;
    R[1][2] = -s0*c1;
    R[2][0] = -c0*s1*c2 + s0*s2;
    R[2][1] =  c0*s1*s2 + s0*c2;
    R[2][2] =  c0*c1;
}

/* ---------- 2) general_naive: 3 elementary + 2 matmul, with runtime axis dispatch ---------- */
/* axes encoded as 'x'/'y'/'z' chars; intrinsic flag picks composition order. */
static inline void general_naive(const double x[3], const char axes[3], int intrinsic, M3 R) {
    M3 R0, R1, R2, T;
    switch (axes[0]) { case 'x': rot_x(x[0], R0); break;
                       case 'y': rot_y(x[0], R0); break;
                       default:  rot_z(x[0], R0); break; }
    switch (axes[1]) { case 'x': rot_x(x[1], R1); break;
                       case 'y': rot_y(x[1], R1); break;
                       default:  rot_z(x[1], R1); break; }
    switch (axes[2]) { case 'x': rot_x(x[2], R2); break;
                       case 'y': rot_y(x[2], R2); break;
                       default:  rot_z(x[2], R2); break; }
    if (intrinsic) { mm3(R0, R1, T); mm3(T, R2, R); }
    else           { mm3(R2, R1, T); mm3(T, R0, R); }
}

/* ---------- 3) smart_switch: switch on a 3-char code, each case runs the explicit
 * formula for that sequence. We only fill in 4 cases to demonstrate; the rest jump
 * to a default that does the same work as XYZ (for a fair switch-overhead measurement).
 * In a real port you'd generate all 24 cases via macros. */
static inline int seq_code(const char *s) {
    return ((unsigned char)s[0] << 16) | ((unsigned char)s[1] << 8) | (unsigned char)s[2];
}
#define CODE3(a,b,c) (((a)<<16)|((b)<<8)|(c))
static inline void smart_switch(const double x[3], const char *seq, M3 R) {
    int code = seq_code(seq);
    double c0=cos(x[0]), s0=sin(x[0]);
    double c1=cos(x[1]), s1=sin(x[1]);
    double c2=cos(x[2]), s2=sin(x[2]);
    switch (code) {
        case CODE3('X','Y','Z'):
            R[0][0] =  c1*c2; R[0][1] = -c1*s2; R[0][2] =  s1;
            R[1][0] =  s0*s1*c2 + c0*s2;
            R[1][1] = -s0*s1*s2 + c0*c2;
            R[1][2] = -s0*c1;
            R[2][0] = -c0*s1*c2 + s0*s2;
            R[2][1] =  c0*s1*s2 + s0*c2;
            R[2][2] =  c0*c1;
            break;
        case CODE3('Z','Y','X'):
            R[0][0] = c0*c1; R[0][1] = c0*s1*s2 - s0*c2; R[0][2] = c0*s1*c2 + s0*s2;
            R[1][0] = s0*c1; R[1][1] = s0*s1*s2 + c0*c2; R[1][2] = s0*s1*c2 - c0*s2;
            R[2][0] = -s1;   R[2][1] = c1*s2;            R[2][2] = c1*c2;
            break;
        case CODE3('x','y','z'):
            R[0][0] = c2*c1; R[0][1] = c2*s1*s0 - s2*c0; R[0][2] = c2*s1*c0 + s2*s0;
            R[1][0] = s2*c1; R[1][1] = s2*s1*s0 + c2*c0; R[1][2] = s2*s1*c0 - c2*s0;
            R[2][0] = -s1;   R[2][1] = c1*s0;            R[2][2] = c1*c0;
            break;
        case CODE3('z','y','x'):
            R[0][0] =  c1*c0; R[0][1] = -c1*s0; R[0][2] =  s1;
            R[1][0] =  s2*s1*c0 + c2*s0; R[1][1] = -s2*s1*s0 + c2*c0; R[1][2] = -s2*c1;
            R[2][0] = -c2*s1*c0 + s2*s0; R[2][1] =  c2*s1*s0 + s2*c0; R[2][2] =  c2*c1;
            break;
        default: /* placeholder — same shape of work as a Tait-Bryan case */
            R[0][0] = c1*c2; R[0][1] = -c1*s2; R[0][2] = s1;
            R[1][0] = s0*s1*c2 + c0*s2; R[1][1] = -s0*s1*s2 + c0*c2; R[1][2] = -s0*c1;
            R[2][0] = -c0*s1*c2 + s0*s2; R[2][1] = c0*s1*s2 + s0*c2; R[2][2] = c0*c1;
            break;
    }
}

/* ---------- timing ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9*ts.tv_nsec;
}

#define N_ITERS 50000000ULL

int main(void) {
    /* prevent the compiler from hoisting the work out of the loop */
    volatile double sink = 0.0;
    double dx[3] = {0.1, 0.2, 0.3};
    M3 R;

    /* warmup all three */
    for (int i = 0; i < 100000; i++) {
        fast_explicit_XYZ(dx, R); sink += R[0][0];
        general_naive(dx, "xyz", 0, R); sink += R[0][0];
        smart_switch(dx, "XYZ", R); sink += R[0][0];
    }

    double t0, t1, ck;

    /* 1) fast_explicit XYZ */
    ck = 0.0;
    t0 = now_sec();
    for (unsigned long long i = 0; i < N_ITERS; i++) {
        double x[3] = {0.1 + i*1e-12, 0.2 - i*1e-12, 0.3 + i*1e-12};
        fast_explicit_XYZ(x, R);
        ck += R[0][0] + R[1][1] + R[2][2];
    }
    t1 = now_sec();
    sink += ck;
    double ns_fast = (t1 - t0) * 1e9 / (double)N_ITERS;
    printf("fast_explicit  XYZ   : %7.2f ns/call   (checksum=%.6f)\n", ns_fast, ck/N_ITERS);

    /* 2) general_naive XYZ (intrinsic) */
    ck = 0.0;
    t0 = now_sec();
    for (unsigned long long i = 0; i < N_ITERS; i++) {
        double x[3] = {0.1 + i*1e-12, 0.2 - i*1e-12, 0.3 + i*1e-12};
        general_naive(x, "xyz", 1, R);  /* axes 'xyz', intrinsic=1 == XYZ */
        ck += R[0][0] + R[1][1] + R[2][2];
    }
    t1 = now_sec();
    sink += ck;
    double ns_gen = (t1 - t0) * 1e9 / (double)N_ITERS;
    printf("general_naive  XYZ   : %7.2f ns/call   (checksum=%.6f)\n", ns_gen, ck/N_ITERS);

    /* 3) smart_switch XYZ */
    ck = 0.0;
    t0 = now_sec();
    for (unsigned long long i = 0; i < N_ITERS; i++) {
        double x[3] = {0.1 + i*1e-12, 0.2 - i*1e-12, 0.3 + i*1e-12};
        smart_switch(x, "XYZ", R);
        ck += R[0][0] + R[1][1] + R[2][2];
    }
    t1 = now_sec();
    sink += ck;
    double ns_smart = (t1 - t0) * 1e9 / (double)N_ITERS;
    printf("smart_switch   XYZ   : %7.2f ns/call   (checksum=%.6f)\n", ns_smart, ck/N_ITERS);

    /* 4) smart_switch with a sequence not in the fast-path 4 (default branch) */
    ck = 0.0;
    t0 = now_sec();
    for (unsigned long long i = 0; i < N_ITERS; i++) {
        double x[3] = {0.1 + i*1e-12, 0.2 - i*1e-12, 0.3 + i*1e-12};
        smart_switch(x, "YXZ", R);  /* not in switch — hits default */
        ck += R[0][0] + R[1][1] + R[2][2];
    }
    t1 = now_sec();
    sink += ck;
    double ns_smart_def = (t1 - t0) * 1e9 / (double)N_ITERS;
    printf("smart_switch   YXZ(d): %7.2f ns/call   (checksum=%.6f)\n", ns_smart_def, ck/N_ITERS);

    printf("\n");
    printf("ratio general / fast       : %.2fx\n", ns_gen / ns_fast);
    printf("ratio smart_switch / fast  : %.2fx\n", ns_smart / ns_fast);
    printf("ratio smart(default)/ fast : %.2fx\n", ns_smart_def / ns_fast);
    printf("(volatile sink: %g)\n", sink);
    return 0;
}
