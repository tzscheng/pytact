/* rbd.c — rigid-body dynamics + the linear-algebra / euler-quaternion / spatial
 * primitives the dynamics layer is built on. Public surface declared in tact.h. */
#include "core.h"

//--------------------basic linear algebra--------------------------------

//vector cross product
void cross3(double a1, double a2, double a3, double b1, double b2, double b3, double *v){
    v[0] = a2*b3 - a3*b2;
    v[1] = a3*b1 - a1*b3;
    v[2] = a1*b2 - a2*b1;
}

//make an unit vector (n) from a vector (v)
void normalize(double *n, double* v, int row){
    double sqsum = 0, norm;

    for(int i = 0; i < row; i++) sqsum += v[i]*v[i];
    norm = sqrt(sqsum);
    
    if (fabs(norm) < TACT_EPS) {
	fprintf(stderr, "vector norm is too small\n");
	exit(0);
    }
    
    for(int i = 0; i < row; i++) n[i] = v[i]/norm;
}

//matrix scalar multiplication: m2 = a * m1
void matscalar(double *m2, double a, double *m1, int row, int col){
    for (int i = 0; i < row; i++) {
	for (int j = 0; j < col; j++) {
	    m2[col*i+j] = a * m1[col*i+j];
	}
    }
}

//matrix multiplication: m3 = m1 * m2
void matmul(double *m3, double *m1, double *m2, int r1, int c1r2, int c2){
    for (int i = 0; i < r1; i++) {
	for (int j = 0; j < c2; j++) {
	    m3[c2*i+j] = 0;
	    for (int k = 0; k < c1r2; k++) m3[c2*i+j] += m1[c1r2*i+k] * m2[c2*k+j];
	}
    }
}

//---- size-specialized 6x6 inline kernels (hot path of aba_featherstone / rne_featherstone).
//generic matmul above can't be unrolled/SIMD-ized at call sites because r1/c1r2/c2 are
//runtime args; these inlines have fixed bounds so gcc -O3 -march=native -funroll-loops
//emits unrolled vectorized code. mTm66/mTv66 also avoid materializing A^T.
static inline void mm66(double *C, const double *A, const double *B){
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            double s = 0.0;
            for (int k = 0; k < 6; k++) s += A[6*i+k] * B[6*k+j];
            C[6*i+j] = s;
        }
}

static inline void mv66(double *out, const double *A, const double *v){
    for (int i = 0; i < 6; i++) {
        double s = 0.0;
        for (int k = 0; k < 6; k++) s += A[6*i+k] * v[k];
        out[i] = s;
    }
}

static inline void mTm66(double *C, const double *A, const double *B){     //C = A^T·B
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            double s = 0.0;
            for (int k = 0; k < 6; k++) s += A[6*k+i] * B[6*k+j];
            C[6*i+j] = s;
        }
}

static inline void mTv66(double *out, const double *A, const double *v){   //out = A^T·v
    for (int i = 0; i < 6; i++) {
        double s = 0.0;
        for (int k = 0; k < 6; k++) s += A[6*k+i] * v[k];
        out[i] = s;
    }
}

//matrix summation: m3 = m1 + m2
void matsum(double *m3, double *m1, double *m2, int row, int col){
    for (int i = 0; i < row; i++) {
	for (int j = 0; j < col; j++) {
	    m3[col*i+j] = m1[col*i+j] + m2[col*i+j];
	}
    }
}

//matrix subtract: m3 = m1 - m2
void matsub(double *m3, double *m1, double *m2, int row, int col){
    for (int i = 0; i < row; i++) {
	for (int j = 0; j < col; j++) {
	    m3[col*i+j] = m1[col*i+j] - m2[col*i+j];
	}
    }
}

//3x3 matrix inverse
void matinv3(double *Minv, double* M){
    double det = M[0]*M[4]*M[8] + M[1]*M[5]*M[6] + M[2]*M[3]*M[7] - M[2]*M[4]*M[6] - M[0]*M[5]*M[7] - M[1]*M[3]*M[8];
    if (fabs(det) < TACT_EPS) {
	fprintf(stderr, "Matrix determinent is too small\n");
	exit(0);
    }

    double invdet = 1/det;
    Minv[0] = invdet*(M[4]*M[8] - M[5]*M[7]); Minv[1] = invdet*(M[2]*M[7] - M[1]*M[8]); Minv[2] = invdet*(M[1]*M[5] - M[2]*M[4]);
    Minv[3] = invdet*(M[5]*M[6] - M[3]*M[8]); Minv[4] = invdet*(M[0]*M[8] - M[2]*M[6]); Minv[5] = invdet*(M[2]*M[3] - M[0]*M[5]);
    Minv[6] = invdet*(M[3]*M[7] - M[4]*M[6]); Minv[7] = invdet*(M[1]*M[6] - M[0]*M[7]); Minv[8] = invdet*(M[0]*M[4] - M[1]*M[3]);
}
    
//affine 4x4 matrix inverse
void matinv4_affine(double *Tinv, const double *T){
    Tinv[0] = T[0]; Tinv[1] = T[4]; Tinv[2]  = T[8];  Tinv[3]  = -(T[0]*T[3] + T[4]*T[7] + T[8]*T[11]);
    Tinv[4] = T[1]; Tinv[5] = T[5]; Tinv[6]  = T[9];  Tinv[7]  = -(T[1]*T[3] + T[5]*T[7] + T[9]*T[11]);
    Tinv[8] = T[2]; Tinv[9] = T[6]; Tinv[10] = T[10]; Tinv[11] = -(T[2]*T[3] + T[6]*T[7] + T[10]*T[11]);
    Tinv[12] = 0.0; Tinv[13] = 0.0; Tinv[14] = 0.0;   Tinv[15] = 1.0;
}

//overwrite a matrix(m1) onto a bigger matrix (m2) starting at row a column b
void matover(double *m2, int r2, int c2, int a, int b, double* m1, int r1, int c1){
    if (a + r1 > r2) { printf("error1\n"); exit(0);}
    if (b + c1 > c2) { printf("error2\n"); exit(0);}
    
    for (int i = 0; i < r1; i++) {
	for (int j = 0; j < c1; j++){
	    m2[c2*(i+a)+j+b] = m1[c1*i+j];
	}
    }
}

//make an identity matrix
void identity(double *m, int size){
    memset(m, '\0', size*size*sizeof(double));
    for (int i = 0; i < size; i++) m[size*i+i] = 1;
}

//make a diagonal matrix from a vector
void diagonal(double* m, double *v, int size){
    memset(m, '\0', size*size*sizeof(double));
    for (int i = 0; i < size; i++) m[size*i+i] = v[i];
}    

//get a transpose matrix m2 from m1
void transpose(double* m2, double *m1, int r1, int c1){
    for (int i = 0; i < r1; i++) {
	for (int j = 0; j < c1; j++) {
	    m2[r1*j+i] = m1[c1*i+j];
	}
    }
}

//matrix print
void matprint(double *m, int row, int col){
    int off=0;
    char buf[4096];
    
    off = sprintf(buf, "\n");
    for(int i =0; i<row; i++){
	for(int j =0; j<col; j++) off += sprintf(buf+off,"%10.3f ", m[col*i+j]);
	off += sprintf(buf+off,"\n");
    }
    printf("%s\n", buf);
}

//make random real number between lb to ub (file-local helper for randmat)
static double randf(double lb, double ub){
    return lb + (ub - lb) * ((double)rand()/RAND_MAX);
}

//make random matrix
void randmat(double *m, int row, int col){
    for(int i =0; i<row; i++){
	for(int j =0; j<col; j++){
	    m[col*i+j] = randf(-1.0, 1.0);
	}
    }
}

//------------------------------------------------------------

//set R[9] (3x3 row-major) to elementary rotation about axis (0=x, 1=y, 2=z) by angle t.
static void _rot_axis(int axis, double t, double R[9]){
    double c = cos(t), s = sin(t);
    if (axis == 0) {                    //rot_x
        R[0]=1; R[1]=0; R[2]=0;
        R[3]=0; R[4]=c; R[5]=-s;
        R[6]=0; R[7]=s; R[8]=c;
    } else if (axis == 1) {             //rot_y
        R[0]=c;  R[1]=0; R[2]=s;
        R[3]=0;  R[4]=1; R[5]=0;
        R[6]=-s; R[7]=0; R[8]=c;
    } else {                            //rot_z
        R[0]=c; R[1]=-s; R[2]=0;
        R[3]=s; R[4]=c;  R[5]=0;
        R[6]=0; R[7]=0;  R[8]=1;
    }
}

//parse a 3-character eulerseq string. lowercase = extrinsic, uppercase = intrinsic.
//writes axes[0..2] in {0,1,2} for {x,y,z}. returns 1 if intrinsic, 0 if extrinsic.
//exits on invalid input (length != 3, mixed case, axis not x/y/z, adjacent axes equal).
static int _euler_axes(const char *seq, int axes[3]){
    int intr = -1;
    for (int i = 0; i < 3; i++) {
        char c = seq[i];
        if (c == 0) {
            fprintf(stderr, "eulerseq must be 3 chars, got \"%s\"\n", seq); exit(1);
        }
        int isup = (c >= 'A' && c <= 'Z');
        int islo = (c >= 'a' && c <= 'z');
        if (!isup && !islo) {
            fprintf(stderr, "invalid char in eulerseq: '%c' in \"%s\"\n", c, seq); exit(1);
        }
        if      (intr == -1) intr = isup;
        else if (intr != isup) {
            fprintf(stderr, "eulerseq must be all-lower or all-upper: \"%s\"\n", seq); exit(1);
        }
        char lo = isup ? (c - 'A' + 'a') : c;
        if      (lo == 'x') axes[i] = 0;
        else if (lo == 'y') axes[i] = 1;
        else if (lo == 'z') axes[i] = 2;
        else { fprintf(stderr, "axis must be x/y/z, got '%c'\n", c); exit(1); }
    }
    if (seq[3] != 0) {
        fprintf(stderr, "eulerseq must be 3 chars, got \"%s\"\n", seq); exit(1);
    }
    if (axes[0] == axes[1] || axes[1] == axes[2]) {
        fprintf(stderr, "adjacent axes in eulerseq must differ: \"%s\"\n", seq); exit(1);
    }
    return intr;
}

//mirrors rbd.py:euler_to_rotation. composes elementary axis rotations for any of
//the 24 standard 3-axis conventions. ~1.5x slower than the explicit-formula form
//at -O2 in C, but at -O3 -march=native gcc dead-code-eliminates the elementary-
//rotation zeros and the two forms produce identical assembly (measured 21->21 ns).
//Makefile release builds use -O3; debug builds keep the slower generic path
//visible for easier inspection.
void euler_to_rotation(double *e, double *R, const char *eulerseq){
    int axes[3];
    int intrinsic = _euler_axes(eulerseq, axes);
    double R0[9], R1[9], R2[9], T[9];
    _rot_axis(axes[0], e[0], R0);
    _rot_axis(axes[1], e[1], R1);
    _rot_axis(axes[2], e[2], R2);
    if (intrinsic) {                    //intrinsic ABC: R = R_A R_B R_C
        matmul(T, R0, R1, 3, 3, 3);
        matmul(R, T,  R2, 3, 3, 3);
    } else {                            //extrinsic abc: R = R_c R_b R_a
        matmul(T, R2, R1, 3, 3, 3);
        matmul(R, T,  R0, 3, 3, 3);
    }
}

//extrinsic abc with angles (α,β,γ) ≡ intrinsic CBA with angles (γ,β,α);
//we extract for the equivalent intrinsic sequence and reverse if extrinsic was requested.
//tait-bryan parity σ=+1 if (i,j,k) is an even permutation of (0,1,2), else -1;
//proper-euler uses the third axis l=3-i-j with the same parity.
void rotation_to_euler(double *R, double *e, const char *eulerseq){
    int axes[3];
    int intrinsic = _euler_axes(eulerseq, axes);
    int i, j, k;
    if (intrinsic) { i = axes[0]; j = axes[1]; k = axes[2]; }
    else           { i = axes[2]; j = axes[1]; k = axes[0]; }       //extr abc <-> intr CBA
    int sigma = ((i + 1) % 3 == j) ? 1 : -1;
    double alpha, beta, gamma;
    const double GIMBAL_TOL = 1e-9;

    if (i != k) {                       //Tait-Bryan
        double s_b = sigma * R[3*i+k];
        if      (s_b >  1.0) s_b =  1.0;
        else if (s_b < -1.0) s_b = -1.0;
        beta = asin(s_b);
        if (fabs(s_b) > 1.0 - GIMBAL_TOL) {     //gimbal lock at β=±π/2: γ undetermined, set to 0
            double sgn = (s_b > 0) ? 1.0 : -1.0;
            alpha = atan2(sgn * R[3*j+i], R[3*j+j]);
            gamma = 0.0;
        } else {
            alpha = atan2(-sigma * R[3*j+k], R[3*k+k]);
            gamma = atan2(-sigma * R[3*i+j], R[3*i+i]);
        }
    } else {                            //proper Euler (i==k)
        int l = 3 - i - j;
        double c_b = R[3*i+i];
        if      (c_b >  1.0) c_b =  1.0;
        else if (c_b < -1.0) c_b = -1.0;
        beta = acos(c_b);
        if (fabs(c_b) > 1.0 - GIMBAL_TOL) {     //β≈0 or π: γ undetermined, set to 0
            if (c_b > 0) alpha = atan2(-sigma * R[3*j+l], R[3*l+l]);
            else         alpha = atan2( sigma * R[3*j+l], -R[3*l+l]);
            gamma = 0.0;
        } else {
            alpha = atan2(R[3*j+i], -sigma * R[3*l+i]);
            gamma = atan2(R[3*i+j],  sigma * R[3*i+l]);
        }
    }

    if (intrinsic) { e[0] = alpha; e[1] = beta; e[2] = gamma; }
    else           { e[0] = gamma; e[1] = beta; e[2] = alpha; }     //reverse for extrinsic
}

void quat_to_rotation(double *q, double *R){
    R[3*0+0] = 2.0*(q[0]*q[0] + q[1]*q[1]) - 1.0;
    R[3*0+1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    R[3*0+2] = 2.0*(q[1]*q[3] + q[0]*q[2]);
    R[3*1+0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    R[3*1+1] = 2.0*(q[0]*q[0] + q[2]*q[2]) - 1.0;
    R[3*1+2] = 2.0*(q[2]*q[3] - q[0]*q[1]);
    R[3*2+0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    R[3*2+1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    R[3*2+2] = 2.0*(q[0]*q[0] + q[3]*q[3]) - 1.0;
}

void rotation_to_quat(double* R, double* q){
    double trace = R[0] + R[4] + R[8];
    double s, w, x, y, z;
    
    if(trace > 0){
        s = 0.5/sqrt(trace + 1.0);
        w = 0.25/s;
        x = (R[7] - R[5])*s;
        y = (R[2] - R[6])*s;
        z = (R[3] - R[1])*s;
    }
    
    else if(R[0] > R[4] && R[0] > R[8]){
        s = 2.0*sqrt(1.0 + R[0] - R[4] - R[8]);
        w = (R[7] - R[5])/s;
        x = 0.25*s;
        y = (R[1] + R[3])/s;
        z = (R[2] + R[6])/s;
    }
    
    else if(R[4] > R[8]){
        s = 2.0*sqrt(1.0 + R[4] - R[0] - R[8]);
        w = (R[2] - R[6])/s;
        x = (R[1] + R[3])/s;
        y = 0.25*s;
        z = (R[5] + R[7])/s;
    }
    
    else{
        s = 2.0*sqrt(1.0 + R[8] - R[0] - R[4]);
        w = (R[3] - R[1])/s;
        x = (R[2] + R[6])/s;
        y = (R[5] + R[7])/s;
        z = 0.25*s;
    }

    q[0] = x;
    q[1] = y;
    q[2] = z;
    q[3] = w;
}   

void euler_to_quat(double* e, double* q, const char *eulerseq){
    double R[9];
    euler_to_rotation(e, R, eulerseq);
    rotation_to_quat(R, q);
}

void quat_to_euler(double* q, double *e, const char *eulerseq){
    double R[9];
    quat_to_rotation(q, R);
    rotation_to_euler(R, e, eulerseq);
}

void xyzeuler_to_xyzquat(double *x6, double *x7, const char *eulerseq){
    double R[9];
    x7[0] = x6[0];
    x7[1] = x6[1];
    x7[2] = x6[2];
    euler_to_rotation(x6+3, R, eulerseq);
    rotation_to_quat(R, x7+3);
}

void xyzquat_to_xyzeuler(double *x7, double *x6, const char *eulerseq){
    double R[9];
    x6[0] = x7[0];
    x6[1] = x7[1];
    x6[2] = x7[2];
    quat_to_rotation(x7+3, R);
    rotation_to_euler(R, x6+3, eulerseq);
}

void homogeneous_to_xyzeuler(double *T, double *x, const char *eulerseq){
    double R[9];
    x[0] = T[3]; x[1] = T[7]; x[2] = T[11];
    R[0] = T[0]; R[1] = T[1]; R[2] = T[2];
    R[3] = T[4]; R[4] = T[5]; R[5] = T[6];
    R[6] = T[8]; R[7] = T[9]; R[8] = T[10];
    rotation_to_euler(R, x+3, eulerseq);
}

void xyzeuler_to_homogeneous(double *x, double *T, const char *eulerseq){
    double R[9];    
    euler_to_rotation(x+3, R, eulerseq);
    T[0] = R[0]; T[1] = R[1]; T[2]  = R[2]; T[3]  = x[0];
    T[4] = R[3]; T[5] = R[4]; T[6]  = R[5]; T[7]  = x[1];
    T[8] = R[6]; T[9] = R[7]; T[10] = R[8]; T[11] = x[2];
    T[12] = 0.0; T[13] = 0.0; T[14] = 0.0;  T[15] = 1.0;
}

void xyzquat_to_homogeneous(double *x, double *T){
    double R[9];
    quat_to_rotation(x+3, R);
    T[0] = R[0]; T[1] = R[1]; T[2]  = R[2]; T[3]  = x[0];
    T[4] = R[3]; T[5] = R[4]; T[6]  = R[5]; T[7]  = x[1];
    T[8] = R[6]; T[9] = R[7]; T[10] = R[8]; T[11] = x[2];
    T[12] = 0.0; T[13] = 0.0; T[14] = 0.0;  T[15] = 1.0;
}

void homogeneous_to_xyzquat(double *T, double *x){
    double R[9];
    x[0] = T[3]; x[1] = T[7]; x[2] = T[11];
    R[0] = T[0]; R[1] = T[1]; R[2] = T[2];
    R[3] = T[4]; R[4] = T[5]; R[5] = T[6];
    R[6] = T[8]; R[7] = T[9]; R[8] = T[10];
    rotation_to_quat(R, x+3);
}

void xyheading_to_homogeneous(double x, double y, double heading, double *T){
    T[3*0+0] = cos(heading);
    T[3*0+1] = -sin(heading);
    T[3*0+2] = x;
    T[3*1+0] = -T[0];
    T[3*1+1] = T[0];
    T[3*1+2] = y;
    T[3*2+2] = 1.0;
}

void rotxyz_to_homogeneous(double* R, double* x, double* T){
    T[0] = R[0]; T[1] = R[1]; T[2]  = R[2]; T[3]  = x[0];
    T[4] = R[3]; T[5] = R[4]; T[6]  = R[5]; T[7]  = x[1];
    T[8] = R[6]; T[9] = R[7]; T[10] = R[8]; T[11] = x[2];
    T[12] = 0.0; T[13] = 0.0; T[14] = 0.0;  T[15] = 1.0;
}

//R1: desired  R2: now   →  e ∈ ℝ³ such that R1 = expmap_so3(e) · R2.
//Mirrors Python rotation_error; world-frame rotation vector, ‖e‖ = θ.
void rotation_error(double* R1, double* R2, double* e){
    /* M = R1 · R2ᵀ */
    double M[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M[3*i + j] = R1[3*i+0]*R2[3*j+0] + R1[3*i+1]*R2[3*j+1] + R1[3*i+2]*R2[3*j+2];
    logmap_so3(M, e);
}

//R1: desired  R2: now   — antisymmetric-vee form of the old sin-shape error.
//Kept available for controllers that intentionally use the pre-logmap behavior.
void rotation_error2(double* R1, double* R2, double* e){
    /* M = R1 · R2ᵀ */
    double M[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M[3*i + j] = R1[3*i+0]*R2[3*j+0] + R1[3*i+1]*R2[3*j+1] + R1[3*i+2]*R2[3*j+2];
    e[0] = 0.5 * (M[7] - M[5]);
    e[1] = 0.5 * (M[2] - M[6]);
    e[2] = 0.5 * (M[3] - M[1]);
}

//T1: desired, T2: now    (orientation in e[0..3], translation in e[3..6])
void homogeneous_error(double* T1, double* T2, double* e){
    double R1[9] = {T1[0],T1[1],T1[2], T1[4],T1[5],T1[6], T1[8],T1[9],T1[10]};
    double R2[9] = {T2[0],T2[1],T2[2], T2[4],T2[5],T2[6], T2[8],T2[9],T2[10]};
    rotation_error(R1, R2, e);
    e[3] = T1[3]  - T2[3];
    e[4] = T1[7]  - T2[7];
    e[5] = T1[11] - T2[11];
}

/* ---- SO(3) exp/log — mirrors rbd.py:skew/expmap_so3/logmap_so3/integrate_so3.
 * Used by free (jtype=3) free joint with axis-angle rotation vector. */

/* K = skew(v) row-major 3×3. */
void skew3(double *K, const double *v){
    K[0]=0;     K[1]=-v[2];  K[2]= v[1];
    K[3]=v[2];  K[4]=0;      K[5]=-v[0];
    K[6]=-v[1]; K[7]= v[0];  K[8]=0;
}

/* Rodrigues: R = I + sin(θ)·K + (1-cos(θ))·K²  where K = skew(w/θ).
 * Stable at θ=0 via 2nd-order Taylor. */
void expmap_so3(const double *w, double *R){
    double theta2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    double theta  = sqrt(theta2);
    double K[9], KK[9];
    if (theta < 1e-9) {
        skew3(K, w);
        matmul(KK, K, K, 3, 3, 3);
        identity(R, 3);
        for (int i = 0; i < 9; ++i) R[i] += K[i] + 0.5*KK[i];
        return;
    }
    double n[3] = {w[0]/theta, w[1]/theta, w[2]/theta};
    skew3(K, n);
    matmul(KK, K, K, 3, 3, 3);
    double s = sin(theta), c1 = 1.0 - cos(theta);
    identity(R, 3);
    for (int i = 0; i < 9; ++i) R[i] += s*K[i] + c1*KK[i];
}

/* log: R → rotation vector. Stable at θ=0 and θ=π. */
void logmap_so3(const double *R, double *w){
    double trace = R[0] + R[4] + R[8];
    double cos_t = 0.5 * (trace - 1.0);
    if (cos_t >  1.0) cos_t =  1.0;
    if (cos_t < -1.0) cos_t = -1.0;
    double theta = acos(cos_t);
    if (theta < 1e-9) {
        /* near identity: w ≈ 0.5·vee(R - Rᵀ) */
        w[0] = 0.5*(R[7] - R[5]);
        w[1] = 0.5*(R[2] - R[6]);
        w[2] = 0.5*(R[3] - R[1]);
        return;
    }
    if (3.14159265358979323846 - theta < 1e-6) {
        /* near antipodal: M = (R+I)/2 ≈ n·nᵀ; recover axis from largest diagonal */
        double M[9];
        for (int i = 0; i < 9; ++i) M[i] = 0.5*R[i];
        M[0] += 0.5; M[4] += 0.5; M[8] += 0.5;
        int k = 0;
        if (M[4] > M[k*4]) k = 1;
        if (M[8] > M[k*4]) k = 2;
        double n[3] = {0, 0, 0};
        n[k] = sqrt(M[k*4] > 0 ? M[k*4] : 0);
        if (n[k] > 1e-12) {
            for (int j = 0; j < 3; ++j) {
                if (j != k) n[j] = M[k*3 + j] / n[k];
            }
        }
        w[0] = theta*n[0]; w[1] = theta*n[1]; w[2] = theta*n[2];
        return;
    }
    double s = 2.0 * sin(theta);
    w[0] = theta * (R[7] - R[5]) / s;
    w[1] = theta * (R[2] - R[6]) / s;
    w[2] = theta * (R[3] - R[1]) / s;
}

/* integrate_so3: w_next = log(exp(w) · exp(ω·dt)). w and w_next may alias. */
void integrate_so3(const double *w, const double *omega, double dt, double *w_next){
    double R1[9], R2[9], R3[9];
    expmap_so3(w, R1);
    double omega_dt[3] = {omega[0]*dt, omega[1]*dt, omega[2]*dt};
    expmap_so3(omega_dt, R2);
    matmul(R3, R1, R2, 3, 3, 3);
    logmap_so3(R3, w_next);
}

//---- latent memset-size fix (rot_x/y/z, T_rot_x/y/z, T_trans below). originals
//      all used memset(X, 0, N*M) which zeros bytes-not-doubles; only the
//      explicitly-set elements were correct, the rest was uninit stack. X_rot_z
//      (jcalc/aba_featherstone path) was not affected because it uses sizeof-aware
//      memset already. T_rot_z and T_trans fire when C _fk runs — masked while
//      _fk stayed in Python. fixed at sizeof(double)*N*M.
void rot_x(double* R, double th){
    memset(R, 0, sizeof(double)*9);
    R[0] = 1.0; R[4] = cos(th); R[5] = -sin(th); R[7] = -R[5]; R[8] = R[4];
}

void rot_y(double *R, double th){
    memset(R, 0, sizeof(double)*9);
    R[0] = cos(th); R[2] = sin(th); R[4] = 1.0; R[6] = -R[2]; R[8] = R[0];
}

void rot_z(double *R, double th){
    memset(R, 0, sizeof(double)*9);
    R[0] = cos(th); R[1] = -sin(th); R[3] = -R[1]; R[4] = R[0]; R[8] = 1.0;
}

void T_rot_x(double *T, double th){
    memset(T, 0, sizeof(double)*16);
    T[0] = 1.0; T[5] = cos(th); T[6] = -sin(th); T[9] = -T[6]; T[10] = T[5]; T[15] = 1.0;
}

void T_rot_y(double *T, double th){
    memset(T, 0, sizeof(double)*16);
    T[0] = cos(th); T[2] = sin(th); T[5] = 1.0; T[8] = -T[2]; T[10] = T[0]; T[15] = 1.0;
}

void T_rot_z(double *T, double th){
    memset(T, 0, sizeof(double)*16);
    T[0] = cos(th); T[1] = -sin(th); T[4] = -T[1]; T[5] = T[0]; T[10] = 1.0; T[15] = 1.0;
}

void T_trans(double *T, double x, double y, double z){
    memset(T, 0, sizeof(double)*16);
    T[0] = 1.0; T[3] = x; T[5] = 1.0; T[7] = y; T[10] = 1.0; T[11] = z; T[15] = 1.0;
}


void X_rot_x(double *X, double th){
    double c = cos(th);
    double s = sin(th);
    memset(X, 0, sizeof(double)*36);
    X[6*0+0] = 1.0; X[6*1+1] = c; X[6*1+2] = s; X[6*2+1] = -s; X[6*2+2] = c;
    X[6*3+3] = 1.0; X[6*4+4] = c; X[6*4+5] = s; X[6*5+4] = -s; X[6*5+5] = c;
}

void X_rot_y(double *X, double th){
    double c = cos(th);
    double s = sin(th);
    memset(X, 0, sizeof(double)*36);
    X[6*0+0] = c; X[6*0+2] = -s; X[6*1+1] = 1.0; X[6*2+0] = s; X[6*2+2] = c;
    X[6*3+3] = c; X[6*3+4] = -s; X[6*4+4] = 1.0; X[6*5+3] = s; X[6*5+5] = c;
}

void X_rot_z(double *X, double th){
    double c = cos(th);
    double s = sin(th);
    memset(X, 0, sizeof(double)*36);
    X[6*0+0] = c; X[6*0+1] = s; X[6*1+0] = -s; X[6*1+1] = c; X[6*2+2] = 1.0;
    X[6*3+3] = c; X[6*3+4] = s; X[6*4+3] = -s; X[6*4+4] = c; X[6*5+5] = 1.0;
}

//---- original X_trans had two latent bugs (never called from C before Stage A):
//       X[6*3+2] = -x  (should be -y),  and lower diag X[3][3]/[4][4]/[5][5]
//       were missing. revealed when aba_featherstone calls jcalc(jtype=2).
//void X_trans(double *X, double x, double y, double z){
//    memset(X, 0, sizeof(double)*36);
//    X[6*0+0] = 1.0; X[6*1+1] = 1.0; X[6*2+2] = 1.0;
//    X[6*3+1] =  z; X[6*3+2] = -x;
//    X[6*4+0] = -z; X[6*4+2] =  x;
//    X[6*5+0] =  y; X[6*5+1] = -x;
//}
void X_trans(double *X, double x, double y, double z){
    memset(X, 0, sizeof(double)*36);
    X[6*0+0] = 1.0; X[6*1+1] = 1.0; X[6*2+2] = 1.0;     //upper-left I
    X[6*3+1] =  z;  X[6*3+2] = -y;                       //lower-left = -r×
    X[6*4+0] = -z;  X[6*4+2] =  x;
    X[6*5+0] =  y;  X[6*5+1] = -x;
    X[6*3+3] = 1.0; X[6*4+4] = 1.0; X[6*5+5] = 1.0;     //lower-right I
}

void crm(double* M, double *v){
    memset(M, 0, sizeof(double)*36);
    M[0*6+1] = -v[2];
    M[0*6+2] =  v[1];
    M[1*6+0] =  v[2];
    M[1*6+2] = -v[0];
    M[2*6+0] = -v[1];
    M[2*6+1] =  v[0];
    M[3*6+1] = -v[5];
    M[3*6+2] =  v[4];
    M[3*6+4] = -v[2];
    M[3*6+5] =  v[1];
    M[4*6+0] =  v[5];
    M[4*6+2] = -v[3];
    M[4*6+3] =  v[2];
    M[4*6+5] = -v[0];
    M[5*6+0] = -v[4];
    M[5*6+1] =  v[3];
    M[5*6+3] = -v[1];
    M[5*6+4] =  v[0];
}

void crf(double* M, double *v){
    memset(M, 0, sizeof(double)*36);
    M[0*6+1] = -v[2];
    M[0*6+2] =  v[1];
    M[0*6+4] = -v[5];
    M[0*6+5] =  v[4];
    M[1*6+0] =  v[2];
    M[1*6+2] = -v[0];
    M[1*6+3] =  v[5];
    M[1*6+5] = -v[3];
    M[2*6+0] = -v[1];
    M[2*6+1] =  v[0];
    M[2*6+3] = -v[4];
    M[2*6+4] =  v[3];
    M[3*6+4] = -v[2];
    M[3*6+5] =  v[1];
    M[4*6+3] =  v[2];
    M[4*6+5] = -v[0];
    M[5*6+3] = -v[1];
    M[5*6+4] =  v[0];
}

void jcalc(double *XJ, double* S, int jtype, double q){
    memset(S, 0, sizeof(double)*6);
    switch(jtype){
    case 0: //fixed joint
	identity(XJ, 6);
	break;

    case 1: //revolute joint along Z-axis
	X_rot_z(XJ, q);
	S[2] = 1.0;
	break;

    case 2: //prismatic joint along Z-axis
	X_trans(XJ, 0, 0, q);
	S[5] = 1.0;
	break;

    default:
	printf("wrong joint type\n");
	exit(0);
    }
}

/* jcalc for the 6-DoF axis-angle free joint (jtype=3).
 * q6 layout: [px, py, pz, wx, wy, wz].
 *   XJ : 6×6 spatial transform (row-major) — homogeneous_to_pluker(T(p,R))
 *   S  : 6×6 motion subspace (row-major) — block-anti-diag([0,I;I,0])
 * Mirrors rbd.py:jcalc_free. qd convention [v_body; ω_body] → spatial [ω; v]. */
void jcalc_free(double *XJ, double *S, const double *q6){
    /* R = expmap(w) */
    double R[9];
    expmap_so3(q6 + 3, R);
    /* E = R^T, r = -E · p */
    double E[9]; transpose(E, R, 3, 3);
    double r[3];
    r[0] = -(E[0]*q6[0] + E[1]*q6[1] + E[2]*q6[2]);
    r[1] = -(E[3]*q6[0] + E[4]*q6[1] + E[5]*q6[2]);
    r[2] = -(E[6]*q6[0] + E[7]*q6[1] + E[8]*q6[2]);
    /* SkE = skew(r) · E */
    double Sk[9], SkE[9];
    skew3(Sk, r);
    matmul(SkE, Sk, E, 3, 3, 3);

    memset(XJ, 0, sizeof(double)*36);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            XJ[i*6     + j    ] = E[i*3 + j];      /* top-left = E */
            XJ[(i+3)*6 + j    ] = SkE[i*3 + j];    /* bot-left = SkE */
            XJ[(i+3)*6 + (j+3)] = E[i*3 + j];      /* bot-right = E */
        }
    }

    /* S = [[0, I3], [I3, 0]] — permutes qd=[v_body; ω_body] → spatial [ω; v]. */
    memset(S, 0, sizeof(double)*36);
    S[0*6 + 3] = 1.0;  S[1*6 + 4] = 1.0;  S[2*6 + 5] = 1.0;
    S[3*6 + 0] = 1.0;  S[4*6 + 1] = 1.0;  S[5*6 + 2] = 1.0;
}

/* jcalc for the 3-DoF ball (spherical) joint (jtype=4) — the rotation half of
 * the free joint. q3 = rotation vector w (child-frame axis-angle, R = expmap(w)).
 * qd = ω (child-frame angular velocity) — the dof velocity IS the body ω, no
 * euler-rate map. Mirrors rbd.py:jcalc_ball.
 *   XJ : 6×6 spatial transform, rotation-only: [[E, 0], [0, E]], E = Rᵀ
 *   S  : 6×3 motion subspace col-major — column c maps ω_c to spatial row c
 *        (spatial convention [ω; v], so the top 3 rows are I, bottom 3 zero). */
void jcalc_ball(double *XJ, double *S, const double *q3){
    double R[9];
    expmap_so3(q3, R);
    double E[9]; transpose(E, R, 3, 3);

    memset(XJ, 0, sizeof(double)*36);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            XJ[i*6     + j    ] = E[i*3 + j];      /* top-left = E */
            XJ[(i+3)*6 + (j+3)] = E[i*3 + j];      /* bot-right = E */
        }
    }

    /* S col-major (6 doubles per column): S[:,c] = e_c on the angular rows. */
    memset(S, 0, sizeof(double)*18);
    S[0*6 + 0] = 1.0;  S[1*6 + 1] = 1.0;  S[2*6 + 2] = 1.0;
}

//forward kinematics. mirrors rbd.py:_fk. T is (nb,4,4) row-major flat = 16*nb.
//---- memcpy size bug fix: originals copied 16 bytes (= 2 doubles) instead of
//      16*sizeof(double); active elements were corrupted on the root-pass.
//      latent until C _fk became reachable from model.step (Stage A+).
void _fk(double *T, int nb, double *Ti, int *parent, int *jtype, double *q){
    double T_tmp[16];
    double Tb[16*TACT_MAX_NB];

    /* q-index walks per-body. Each body consumes nv[i] slots: 0 for jtype 0 (fixed),
     * 1 for jtype 1/2, 6 for jtype 3 (free), 3 for jtype 4 (ball). */
    int q_idx = 0;
    for (int i = 0; i < nb; i++) {
        if (jtype[i] == 0) {
            memcpy(Tb+16*i, Ti+16*i, 16*sizeof(double));
        } else if (jtype[i] == 1) {
            T_rot_z(T_tmp, q[q_idx]);
            matmul(Tb+16*i, Ti+16*i, T_tmp, 4, 4, 4);
            q_idx += 1;
        } else if (jtype[i] == 2) {
            T_trans(T_tmp, 0, 0, q[q_idx]);
            matmul(Tb+16*i, Ti+16*i, T_tmp, 4, 4, 4);
            q_idx += 1;
        } else if (jtype[i] == 4) {
            /* ball: q[q_idx:q_idx+3] = rotation vector. T_local = [R(w), 0; 0, 1]. */
            double R[9];
            expmap_so3(q + q_idx, R);
            T_tmp[ 0]=R[0]; T_tmp[ 1]=R[1]; T_tmp[ 2]=R[2]; T_tmp[ 3]=0.0;
            T_tmp[ 4]=R[3]; T_tmp[ 5]=R[4]; T_tmp[ 6]=R[5]; T_tmp[ 7]=0.0;
            T_tmp[ 8]=R[6]; T_tmp[ 9]=R[7]; T_tmp[10]=R[8]; T_tmp[11]=0.0;
            T_tmp[12]=0.0;  T_tmp[13]=0.0;  T_tmp[14]=0.0;  T_tmp[15]=1.0;
            matmul(Tb+16*i, Ti+16*i, T_tmp, 4, 4, 4);
            q_idx += 3;
        } else if (jtype[i] == 3) {
            /* free: q[q_idx:q_idx+6] = (p, w). Compose T_local = [R(w), p; 0, 1]. */
            double R[9];
            expmap_so3(q + q_idx + 3, R);
            T_tmp[ 0]=R[0]; T_tmp[ 1]=R[1]; T_tmp[ 2]=R[2]; T_tmp[ 3]=q[q_idx];
            T_tmp[ 4]=R[3]; T_tmp[ 5]=R[4]; T_tmp[ 6]=R[5]; T_tmp[ 7]=q[q_idx+1];
            T_tmp[ 8]=R[6]; T_tmp[ 9]=R[7]; T_tmp[10]=R[8]; T_tmp[11]=q[q_idx+2];
            T_tmp[12]=0.0;  T_tmp[13]=0.0;  T_tmp[14]=0.0;  T_tmp[15]=1.0;
            matmul(Tb+16*i, Ti+16*i, T_tmp, 4, 4, 4);
            q_idx += 6;
        }
    }

    for (int i = 0; i < nb; i++) {
        if (parent[i] < 0) memcpy(T+16*i, Tb+16*i, 16*sizeof(double));
        else               matmul(T+16*i, T+16*parent[i], Tb+16*i, 4, 4, 4);
    }
}

//---- original jacob_whitney had multiple latent bugs (never called from C
//      before Stage C): memset size was bytes-but-passed-as-doubles count;
//      Z-axis indexing used T[16*i+3] (position) instead of T[16*i+2] (z-col);
//      _T was indexed as if per-body though Python passes a single 4×4.
//      original below for reference; corrected version follows.
//void jacob_whitney(double *J, int nb, double* T, double* _T, int* parent, int* jtype, int idx){
//    double tmp[3];
//    int i = idx;
//    memset(J, 0, 6*nb);
//    while(1){
//        if (jtype[i] == 1) {
//            cross3(T[16*i+2], T[16*i+6], T[16*i+10], _T[16*i+3]-T[16*i+3], _T[16*i+7]-T[16*i+7], _T[16*i+11]-T[16*i+11], tmp);
//            J[0*nb+i] = tmp[0];    J[1*nb+i] = tmp[1];    J[2*nb+i] = tmp[2];
//            J[3*nb+i] = T[16*i+3]; J[4*nb+i] = T[16*i+7]; J[5*nb+i] = T[16*i+7];
//        }
//        else if (jtype[i] == 2) {
//            J[0*nb+i] = T[16*i+3]; J[1*nb+i] = T[16*i+7]; J[2*nb+i] = T[16*i+7];
//        }
//        i = parent[i];
//        if(i < 0) break;
//    }
//}

//Jacobian matrix. T is (nb,4,4) flat = 16*nb. _T is a single 4×4 = 16 doubles
//(the contact-point frame). idx is the leaf body to walk up from. J is 6×nb,
//cleared at entry and filled only on the parent chain from idx to root.
void jacob_whitney(double *J, int nb, double *T, double *_T, int *parent, int *jtype, int idx){
    /* Compute q_base/nq from jtype. J is 6×nq row-major (stride nq).
     * Fixed joints contribute 0 columns (no DoF). */
    int q_base[TACT_MAX_NB];
    int nq = 0;
    for (int j = 0; j < nb; j++) {
        q_base[j] = nq;
        nq += jt_nv(jtype[j]);
    }
    memset(J, 0, 6*nq*sizeof(double));
    int i = idx;
    while (1) {
        int qb = q_base[i];
        if (jtype[i] == 1) {                    //revolute about body z-axis
            double zx = T[16*i+2],  zy = T[16*i+6],  zz = T[16*i+10];
            double dx = _T[3]-T[16*i+3], dy = _T[7]-T[16*i+7], dz = _T[11]-T[16*i+11];
            double tmp[3];
            cross3(zx, zy, zz, dx, dy, dz, tmp);
            J[0*nq+qb] = tmp[0]; J[1*nq+qb] = tmp[1]; J[2*nq+qb] = tmp[2];
            J[3*nq+qb] = zx;     J[4*nq+qb] = zy;     J[5*nq+qb] = zz;
        } else if (jtype[i] == 2) {             //prismatic along body z-axis
            J[0*nq+qb] = T[16*i+2]; J[1*nq+qb] = T[16*i+6]; J[2*nq+qb] = T[16*i+10];
        } else if (jtype[i] == 4) {             //ball — 3 ω_body columns at qb..qb+2
            /* Same as the free joint's ω_body columns: linear = -skew(r)·R[:,c],
             * angular = R[:,c] (R = body axes in world, r = point offset). */
            double R[9];
            R[0]=T[16*i+0]; R[1]=T[16*i+1]; R[2]=T[16*i+2];
            R[3]=T[16*i+4]; R[4]=T[16*i+5]; R[5]=T[16*i+6];
            R[6]=T[16*i+8]; R[7]=T[16*i+9]; R[8]=T[16*i+10];
            double rx = _T[3]  - T[16*i+3];
            double ry = _T[7]  - T[16*i+7];
            double rz = _T[11] - T[16*i+11];
            for (int c = 0; c < 3; c++) {
                double rc = R[c], rc3 = R[3+c], rc6 = R[6+c];
                J[0*nq + qb + c] =  rz*rc3 - ry*rc6;
                J[1*nq + qb + c] = -rz*rc  + rx*rc6;
                J[2*nq + qb + c] =  ry*rc  - rx*rc3;
                J[3*nq + qb + c] = rc;
                J[4*nq + qb + c] = rc3;
                J[5*nq + qb + c] = rc6;
            }
        } else if (jtype[i] == 3) {             //free — fill 6 columns at qb..qb+5
            /* R = T[i, :3, :3] (body axes in world), p = T[i, :3, 3]
             * r = _T - p. J cols 0..2 (v_body): linear = R, angular = 0.
             * J cols 3..5 (ω_body): linear = -skew(r)·R, angular = R. */
            double R[9];
            R[0]=T[16*i+0]; R[1]=T[16*i+1]; R[2]=T[16*i+2];
            R[3]=T[16*i+4]; R[4]=T[16*i+5]; R[5]=T[16*i+6];
            R[6]=T[16*i+8]; R[7]=T[16*i+9]; R[8]=T[16*i+10];
            double rx = _T[3]  - T[16*i+3];
            double ry = _T[7]  - T[16*i+7];
            double rz = _T[11] - T[16*i+11];
            /* For c=0..2 (v_body cols): top 3 rows = R[:,c], bottom 3 = 0 */
            for (int c = 0; c < 3; c++) {
                J[0*nq + qb + c] = R[c];      /* R[0,c] */
                J[1*nq + qb + c] = R[3+c];    /* R[1,c] */
                J[2*nq + qb + c] = R[6+c];    /* R[2,c] */
            }
            /* For c=3..5 (ω_body cols): top 3 = -skew(r)·R[:,c-3], bottom 3 = R[:,c-3]
             * -skew(r)·v = (r × v) but with sign — same as cross(r, v): output[i] = (r × v)[i]
             * Wait, -skew(r)·v = -(r × v) = v × r. Let me recompute.
             * skew(r) @ v = r × v.  -skew(r) @ v = -r × v = v × r.
             * Actually for the Jacobian column: ω·(r × ω̂) gives velocity from rotation
             * at offset r — that's r × ω̂ = skew(r) @ ω̂.
             * Our formula: J[0:3, c+3] = -skew(r) @ R[:,c]. Let me match Python:
             *   J[0:3, qb+3:qb+6] = -skew_r @ R.   Yes. */
            for (int c = 0; c < 3; c++) {
                double rc = R[c], rc3 = R[3+c], rc6 = R[6+c];
                /* -skew(r) @ [rc; rc3; rc6]:
                 *   row 0:  0·rc - (-rz)·rc3 -  ry·rc6  = rz·rc3 - ry·rc6
                 *   row 1: rz·rc - 0·rc3 + (-rx)·rc6 ... wait let me just use skew formula
                 * skew(r) = [[0,-rz,ry],[rz,0,-rx],[-ry,rx,0]], so -skew(r) negates these.
                 * -skew(r)·v has rows: (rz*v1 - ry*v2), (-rz*v0 + rx*v2), (ry*v0 - rx*v1) */
                J[0*nq + qb + 3 + c] =  rz*rc3 - ry*rc6;
                J[1*nq + qb + 3 + c] = -rz*rc  + rx*rc6;
                J[2*nq + qb + 3 + c] =  ry*rc  - rx*rc3;
                J[3*nq + qb + 3 + c] = rc;
                J[4*nq + qb + 3 + c] = rc3;
                J[5*nq + qb + 3 + c] = rc6;
            }
        }
        i = parent[i];
        if (i < 0) break;
    }
}

//-----------------------------------------------------------------------------
// dynamics: forward dynamics (articulated-body) and integrators.
// mirrors rbd.py:aba_featherstone / euler_step. all input/output/workspace
// buffers are caller-allocated. layout matches Python's flat numpy storage:
//   X[nb*36], I6[nb*36]    : row-major 6x6 per body
//   v,c,S,U,pA,a,f,f_ext   : nb*6 (6-vector per body)
//   q,qd,tau,qdd,d,u       : nb
//   parent[nb]             : int, -1 for root
//   jtype[nb]              : 0=fixed, 1=rev(z), 2=lin(z)
//   workspace              : 104*nb doubles (Xup 36 + S 6 + v 6 + c 6
//                            + IA 36 + pA 6 + U 6 + d 1 + u 1)
//-----------------------------------------------------------------------------
//if full=0, skips the post-loop that writes f_out/v_out and adds the v×v cross-term
//to a_out's last 3 components. used by rk4_step for the 3 intermediate stages where
//only qdd is needed. when full=0, f_out and v_out are not touched (caller may pass
//any valid pointer including a scratch buffer).
//
//ff[nb], sk[nb], dt_imp: optional semi-implicit damping/spring. When ff!=NULL
//or sk!=NULL (with dt_imp>0), per active joint we apply
//    d[i] += ff[i]*dt_imp + sk[i]*dt_imp*dt_imp
//    u[i] -= ff[i]*qd[i]  + sk[i]*q[i]
//Kp_j/Kd_j/q_ref/qd_ref: optional joint-space implicit PD (capability+activation).
//PD activated when q_ref or qd_ref is non-NULL. When active:
//    d[i] += (Kd[i])·dt_imp + (Kp[i])·dt_imp²   (Kp term only if both Kp_j and q_ref non-NULL)
//    u[i] -= Kp[i]·(q[i] - qref[i]) + Kp[i]·dt_imp·qd[i] + Kd[i]·(qd[i] - qdref[i])
//qdref defaults to 0 (regulate to rest) when qd_ref is NULL.
//so qdd = (tau - bias - ff·qd - sk·q - Kp·(q-qref) - Kp·dt·qd - Kd·(qd-qdref))
//        / (M + (ff+Kd)·dt + (sk+Kp)·dt²). Mirrors rbd.py aba_featherstone.
//Pass NULL/0 for everything to get legacy explicit behavior.
void aba_featherstone(int nb, double *X, double *I6, int *parent, int *jtype, double *q, double *qd, double *tau, double *f_ext, double *g, double *qdd, double *f_out, double *a_out, double *v_out, double *workspace, double *ff, double *sk, double *armature, double dt_imp, double *Kp_j, double *Kd_j, double *q_ref, double *qd_ref, int full)
{
    /* Per-body DoF count and q-base. nv[i]: 0 fixed, 6 free, 3 ball, else 1. */
    int q_base[TACT_MAX_NB], nv[TACT_MAX_NB], d_offset[TACT_MAX_NB];
    int nq = 0, d_total = 0;
    for (int i = 0; i < nb; i++) {
        q_base[i] = nq;
        nv[i]     = jt_nv(jtype[i]);
        d_offset[i] = d_total;
        nq      += nv[i];
        d_total += nv[i] * nv[i];
    }

    /* Flat workspace slicing — see comment in tact.c for total size budget.
     * S/U store 6×nv per body col-major; for 1-DoF a body's block is 6 doubles
     * (a single column), preserving the original S[6*i+k] indexing when q_base[i]=i. */
    double *Xup = workspace;            //36*nb
    double *S   = Xup + 36*nb;          // 6*nq (was 6*nb)
    double *vv  = S   +  6*nq;          // 6*nb
    double *cc  = vv  +  6*nb;          // 6*nb
    double *IA  = cc  +  6*nb;          //36*nb
    double *pA  = IA  + 36*nb;          // 6*nb
    double *U   = pA  +  6*nb;          // 6*nq (was 6*nb)
    double *d   = U   +  6*nq;          // d_total ≤ 36*nb (was nb)
    double *u   = d   +   d_total;      //   nq (was nb)

    double a_grav[6] = {0.0, 0.0, 0.0, g[0], g[1], g[2]};
    double Xj[36], Si1[6], vJ[6], crmv[36], crfv[36];
    double tmp36a[36], tmp36b[36], tmp6[6];
    double Ia[36], pa[6];
    double S6[36], d6[36];   /* free scratch: full 6×6 S and 6×6 d (LDLᵀ factored) */

    /* ----- pass 1 (root → leaves): velocities, biases, IA, pA ----- */
    for (int i = 0; i < nb; i++) {
        int qbi = q_base[i];
        double *Sb = S + 6 * qbi;       /* body i's S block: 6 × nv columns col-major */

        if (jtype[i] == 3) {
            jcalc_free(Xj, S6, q + qbi);
            for (int k = 0; k < 36; k++) Sb[k] = S6[k];          /* copy 6×6 S */
            /* vJ = S6 @ qd[qbi:qbi+6] */
            for (int r = 0; r < 6; r++) {
                double s = 0.0;
                for (int c = 0; c < 6; c++) s += S6[6*c + r] * qd[qbi + c];
                vJ[r] = s;
            }
        } else if (jtype[i] == 4) {
            jcalc_ball(Xj, S6, q + qbi);
            for (int k = 0; k < 18; k++) Sb[k] = S6[k];          /* copy 6×3 S */
            /* vJ = S @ qd[qbi:qbi+3] = [ω; 0] (S columns are e_c on angular rows) */
            for (int r = 0; r < 6; r++) {
                double s = 0.0;
                for (int c = 0; c < 3; c++) s += S6[6*c + r] * qd[qbi + c];
                vJ[r] = s;
            }
        } else if (jtype[i] == 0) {
            /* fixed joint: Xj=I, no S columns (nv=0), no qd contribution. Avoid
             * indexing q/qd at qbi — fixed shares qbi with the next body. */
            identity(Xj, 6);
            for (int k = 0; k < 6; k++) vJ[k] = 0.0;
        } else {
            jcalc(Xj, Si1, jtype[i], q[qbi]);
            for (int k = 0; k < 6; k++) Sb[k] = Si1[k];
            for (int k = 0; k < 6; k++) vJ[k] = Si1[k] * qd[qbi];
        }

        mm66(Xup + 36*i, Xj, X + 36*i);                          /* Xup[i] = Xj·X[i] */

        if (parent[i] < 0) {
            for (int k = 0; k < 6; k++) vv[6*i+k] = vJ[k];
            for (int k = 0; k < 6; k++) cc[6*i+k] = 0.0;
        } else {
            mv66(tmp6, Xup + 36*i, vv + 6*parent[i]);
            for (int k = 0; k < 6; k++) vv[6*i+k] = tmp6[k] + vJ[k];
            crm(crmv, vv + 6*i);
            mv66(cc + 6*i, crmv, vJ);
        }

        memcpy(IA + 36*i, I6 + 36*i, 36*sizeof(double));

        crf(crfv, vv + 6*i);
        mm66(tmp36a, crfv, I6 + 36*i);
        mv66(pA + 6*i, tmp36a, vv + 6*i);
        for (int k = 0; k < 6; k++) pA[6*i+k] -= f_ext[6*i+k];
    }

    /* ----- pass 2 (leaves → root): U, d, u, propagate IA/pA up ----- */
    int pd_on    = (q_ref || qd_ref);
    int implicit = (((ff || sk) || pd_on) && dt_imp > 0.0);
    for (int i = nb - 1; i >= 0; i--) {
        int qbi = q_base[i], nvi = nv[i];
        double *Sb = S + 6 * qbi;
        double *Ub = U + 6 * qbi;
        double *db = d + d_offset[i];
        double *ub = u + qbi;

        /* U[i] = IA[i] · S[i]   (6×nvi) */
        for (int c = 0; c < nvi; c++) {
            mv66(Ub + 6*c, IA + 36*i, Sb + 6*c);
        }
        /* d[i] = Sᵀ · U  (nvi×nvi)  and  u[i] = tau − Sᵀ·pA (nvi) */
        for (int r = 0; r < nvi; r++) {
            for (int c = 0; c < nvi; c++) {
                double s = 0.0;
                for (int k = 0; k < 6; k++) s += Sb[6*r + k] * Ub[6*c + k];
                db[r*nvi + c] = s;
            }
            double sp = 0.0;
            for (int k = 0; k < 6; k++) sp += Sb[6*r + k] * pA[6*i + k];
            ub[r] = tau[qbi + r] - sp;
        }

        /* armature (rotor/reflected inertia, MuJoCo-style): add to the d diagonal —
           a true inertia (no dt factor), all DoF types. NULL = 0 → bit-identical. */
        if (armature)
            for (int r = 0; r < nvi; r++) db[r*nvi + r] += armature[qbi + r];

        /* Semi-implicit damping/spring/PD — 1-DoF joints only (free PD deferred). */
        if (implicit && (jtype[i] == 1 || jtype[i] == 2)) {
            double ff_i = ff ? ff[qbi] : 0.0;
            double sk_i = sk ? sk[qbi] : 0.0;
            double Kp_i = (pd_on && Kp_j && q_ref) ? Kp_j[qbi] : 0.0;
            double Kd_i = (pd_on && Kd_j)          ? Kd_j[qbi] : 0.0;
            double qr_i  = q_ref  ? q_ref[qbi]  : 0.0;
            double qdr_i = qd_ref ? qd_ref[qbi] : 0.0;
            db[0] += (ff_i + Kd_i)*dt_imp + (sk_i + Kp_i)*dt_imp*dt_imp;
            ub[0] -= ff_i*qd[qbi] + sk_i*q[qbi]
                  +  Kp_i*(q[qbi] - qr_i) + Kp_i*dt_imp*qd[qbi]
                  +  Kd_i*(qd[qbi] - qdr_i);
        }

        if (parent[i] < 0) continue;

        if (jtype[i] == 0) {
            /* fixed joint: pass IA through */
            memcpy(Ia, IA + 36*i, 36*sizeof(double));
            mv66(pa, Ia, cc + 6*i);
            for (int k = 0; k < 6; k++) pa[k] += pA[6*i+k];
        } else if (nvi == 3) {
            /* ball: Ia = IA - U · d⁻¹ · Uᵀ via LDLᵀ on the 3×3 d block. Same
             * shape as the free branch below with nvi=3 (kept separate so the
             * free path's literal-6 loops stay bit-identical). */
            memcpy(d6, db, 9*sizeof(double));
            int ld_ok = ldlt_factor(d6, 3);
            if (ld_ok != 0) {
                memcpy(Ia, IA + 36*i, 36*sizeof(double));
                mv66(pa, Ia, cc + 6*i);
                for (int k = 0; k < 6; k++) pa[k] += pA[6*i+k];
            } else {
                /* Y = d⁻¹ · Uᵀ (3×6), solved row-of-U by row-of-U. */
                double Y[18];                       /* Y[3*c + k] = (d⁻¹ Uᵀ)[k, c] */
                for (int c = 0; c < 6; c++) {
                    double rhs[3];
                    for (int k = 0; k < 3; k++) rhs[k] = Ub[6*k + c];   /* Uᵀ[:, c] = U[c, :] */
                    ldlt_solve(d6, 3, rhs);
                    for (int k = 0; k < 3; k++) Y[3*c + k] = rhs[k];
                }
                /* Ia = IA - U · Y  →  Ia[r,c] = IA[r,c] - Σ_k U[r,k]·Y[k,c]
                 * U col-major 6×3: U[r,k] = Ub[6*k + r]. Y[k,c] = Y[3*c + k]. */
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++) {
                        double s = 0.0;
                        for (int k = 0; k < 3; k++) s += Ub[6*k + r] * Y[3*c + k];
                        Ia[6*r + c] = IA[36*i + 6*r + c] - s;
                    }
                /* pa = pA + Ia·cc + U · (d⁻¹ · u) */
                mv66(pa, Ia, cc + 6*i);
                double dinv_u[3];
                for (int k = 0; k < 3; k++) dinv_u[k] = ub[k];
                ldlt_solve(d6, 3, dinv_u);
                for (int r = 0; r < 6; r++) {
                    double s = 0.0;
                    for (int c = 0; c < 3; c++) s += Ub[6*c + r] * dinv_u[c];
                    pa[r] += pA[6*i + r] + s;
                }
            }
        } else if (nvi == 1) {
            /* 1-DoF: rank-1 update (bit-identical to original code when q_base[i]==i) */
            double inv_d = 1.0 / db[0];
            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 6; c++)
                    Ia[6*r + c] = IA[36*i + 6*r + c] - Ub[r] * Ub[c] * inv_d;
            double scale = ub[0] * inv_d;
            mv66(pa, Ia, cc + 6*i);
            for (int k = 0; k < 6; k++) pa[k] += pA[6*i+k] + Ub[k] * scale;
        } else {
            /* free: Ia = IA - U · d⁻¹ · Uᵀ  via LDLᵀ on the 6×6 d block */
            memcpy(d6, db, 36*sizeof(double));
            int ld_ok = ldlt_factor(d6, 6);
            if (ld_ok != 0) {
                /* singular d — should not happen for SPD inertia, but degrade gracefully */
                memcpy(Ia, IA + 36*i, 36*sizeof(double));
                mv66(pa, Ia, cc + 6*i);
                for (int k = 0; k < 6; k++) pa[k] += pA[6*i+k];
            } else {
                /* Y[:, c] = d⁻¹ · Uᵀ[:, c] = d⁻¹ · (row c of U). Mirrors Python
                 * `Y = solve(d, Uᵀ); Ia = IA − U·Y`. (The original code here
                 * solved against U's COLUMNS, i.e. computed U·Uᵀ·d⁻¹ instead of
                 * U·d⁻¹·Uᵀ — latent, since free joints are always tree roots in
                 * the Python model builder and roots skip this propagation.) */
                double Y[36];                       /* Y[6*c + k] = (d⁻¹Uᵀ)[k, c] */
                for (int c = 0; c < 6; c++) {
                    double rhs[6];
                    for (int k = 0; k < 6; k++) rhs[k] = Ub[6*k + c];   /* U[c, :] */
                    ldlt_solve(d6, 6, rhs);     /* rhs ← d6⁻¹ · rhs  (6-vec)        */
                    for (int k = 0; k < 6; k++) Y[6*c + k] = rhs[k];
                }
                /* Ia = IA - U · Y  →  Ia[r,c] = IA[r,c] - Σ_k U[r,k]·(d⁻¹Uᵀ)[k,c]
                 * U col-major (6×6): U[r,k] = Ub[6*k+r]. */
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++) {
                        double s = 0.0;
                        for (int k = 0; k < 6; k++) s += Ub[6*k + r] * Y[6*c + k];
                        Ia[6*r + c] = IA[36*i + 6*r + c] - s;
                    }
                /* pa = pA + Ia·cc + U · (d⁻¹ · u) */
                mv66(pa, Ia, cc + 6*i);
                double dinv_u[6];
                for (int k = 0; k < 6; k++) dinv_u[k] = ub[k];
                ldlt_solve(d6, 6, dinv_u);     /* dinv_u ← d⁻¹·u */
                /* U·dinv_u: U col-major (6×6) times 6-vec → 6-vec */
                for (int r = 0; r < 6; r++) {
                    double s = 0.0;
                    for (int c = 0; c < 6; c++) s += Ub[6*c + r] * dinv_u[c];
                    pa[r] += pA[6*i + r] + s;
                }
            }
        }

        /* IA[parent] += Xupᵀ · Ia · Xup;  pA[parent] += Xupᵀ · pa */
        mTm66(tmp36b, Xup + 36*i, Ia);
        mm66(tmp36a, tmp36b, Xup + 36*i);
        for (int k = 0; k < 36; k++) IA[36*parent[i] + k] += tmp36a[k];
        mTv66(tmp6, Xup + 36*i, pa);
        for (int k = 0; k < 6; k++) pA[6*parent[i] + k] += tmp6[k];
    }

    /* ----- pass 3 (root → leaves): accelerations ----- */
    for (int i = 0; i < nb; i++) {
        int qbi = q_base[i], nvi = nv[i];
        double *Sb = S + 6 * qbi;
        double *Ub = U + 6 * qbi;
        double *db = d + d_offset[i];
        double *ub = u + qbi;
        double *qddb = qdd + qbi;

        if (parent[i] < 0) {
            double neg_g[6] = {-a_grav[0],-a_grav[1],-a_grav[2],-a_grav[3],-a_grav[4],-a_grav[5]};
            mv66(a_out + 6*i, Xup + 36*i, neg_g);
        } else {
            mv66(a_out + 6*i, Xup + 36*i, a_out + 6*parent[i]);
        }
        for (int k = 0; k < 6; k++) a_out[6*i+k] += cc[6*i+k];

        if (jtype[i] == 0) {
            /* fixed: no qdd slot to write (nv=0); qddb shares offset with the
             * next body, so explicit zeroing would corrupt it. */
        } else if (nvi == 1) {
            double Ua = 0.0;
            for (int k = 0; k < 6; k++) Ua += Ub[k] * a_out[6*i + k];
            qddb[0] = (ub[0] - Ua) / db[0];
        } else if (nvi == 3) {
            /* ball: qdd_slice = d⁻¹ · (u - Uᵀ·a), 3×3 LDLᵀ */
            double rhs[3];
            for (int c = 0; c < 3; c++) {
                double Ua = 0.0;
                for (int k = 0; k < 6; k++) Ua += Ub[6*c + k] * a_out[6*i + k];
                rhs[c] = ub[c] - Ua;
            }
            memcpy(d6, db, 9*sizeof(double));
            ldlt_factor(d6, 3);
            ldlt_solve(d6, 3, rhs);
            for (int k = 0; k < 3; k++) qddb[k] = rhs[k];
        } else {
            /* free: qdd_slice = d⁻¹ · (u - Uᵀ·a) */
            double rhs[6];
            for (int c = 0; c < 6; c++) {
                double Ua = 0.0;
                for (int k = 0; k < 6; k++) Ua += Ub[6*c + k] * a_out[6*i + k];
                rhs[c] = ub[c] - Ua;
            }
            memcpy(d6, db, 36*sizeof(double));
            ldlt_factor(d6, 6);
            ldlt_solve(d6, 6, rhs);
            for (int k = 0; k < 6; k++) qddb[k] = rhs[k];
        }

        /* a[i] += S · qdd_slice */
        for (int c = 0; c < nvi; c++) {
            double qc = qddb[c];
            for (int k = 0; k < 6; k++) a_out[6*i + k] += Sb[6*c + k] * qc;
        }
    }

    if (!full) return;

    /* ----- "full" outputs: f, _a (a with cross-term), v ----- */
    for (int i = 0; i < nb; i++) {
        mv66(f_out + 6*i, IA + 36*i, a_out + 6*i);
        for (int k = 0; k < 6; k++) f_out[6*i + k] += pA[6*i + k];
    }
    memcpy(v_out, vv, 6*nb*sizeof(double));
    for (int i = 0; i < nb; i++) {
        double cx[3];
        cross3(vv[6*i+0], vv[6*i+1], vv[6*i+2], vv[6*i+3], vv[6*i+4], vv[6*i+5], cx);
        a_out[6*i+3] += cx[0];
        a_out[6*i+4] += cx[1];
        a_out[6*i+5] += cx[2];
    }
}

//mirrors rbd.py:rne_featherstone (recursive Newton-Euler inverse dynamics, full=True path).
//given (q, qd, qdd) computes joint torques tau plus the "full" outputs f, _a, v.
//workspace must be at least 42*nb doubles (Xup 36 + S 6); a_out, f_out, v_out are
//caller-allocated outputs (also used internally for ancestor lookups).
void rne_featherstone(int nb, double *X, double *I6, int *parent, int *jtype, double *q, double *qd, double *qdd, double *f_ext, double *g, double *tau, double *f_out, double *a_out, double *v_out, double *workspace)
{
    /* Per-body q-base and nv (free=6, ball=3, fixed=0, else 1). */
    int q_base[TACT_MAX_NB];
    int nq = 0;
    for (int i = 0; i < nb; i++) {
        q_base[i] = nq;
        nq += jt_nv(jtype[i]);
    }

    double *Xup = workspace;            //36*nb
    double *S   = Xup + 36*nb;          // 6*nq  (was 6*nb)

    double a_grav[6] = {0.0, 0.0, 0.0, g[0], g[1], g[2]};
    double Xj[36], Si1[6], vJ[6], crmv[36], crfv[36];
    double tmp36a[36], tmp6a[6], tmp6b[6], S6[36];

    /* ----- pass 1 (root → leaves): velocities, accelerations, body forces ----- */
    for (int i = 0; i < nb; i++) {
        int qbi = q_base[i];
        double *Sb = S + 6 * qbi;
        double aS[6] = {0,0,0,0,0,0};   /* S · qdd_slice (6-vec) */

        if (jtype[i] == 3) {
            jcalc_free(Xj, S6, q + qbi);
            for (int k = 0; k < 36; k++) Sb[k] = S6[k];
            /* vJ = S6 · qd[qbi:], aS = S6 · qdd[qbi:] */
            for (int r = 0; r < 6; r++) {
                double sv = 0.0, sa = 0.0;
                for (int c = 0; c < 6; c++) {
                    sv += S6[6*c + r] * qd[qbi + c];
                    sa += S6[6*c + r] * qdd[qbi + c];
                }
                vJ[r] = sv; aS[r] = sa;
            }
        } else if (jtype[i] == 4) {
            jcalc_ball(Xj, S6, q + qbi);
            for (int k = 0; k < 18; k++) Sb[k] = S6[k];
            /* vJ = S · qd[qbi:qbi+3], aS = S · qdd[qbi:qbi+3] */
            for (int r = 0; r < 6; r++) {
                double sv = 0.0, sa = 0.0;
                for (int c = 0; c < 3; c++) {
                    sv += S6[6*c + r] * qd[qbi + c];
                    sa += S6[6*c + r] * qdd[qbi + c];
                }
                vJ[r] = sv; aS[r] = sa;
            }
        } else if (jtype[i] == 0) {
            /* fixed: Xj=I, no S/qd/qdd contribution (nv=0, qbi shared with next) */
            identity(Xj, 6);
            for (int k = 0; k < 6; k++) vJ[k] = 0.0;
        } else {
            jcalc(Xj, Si1, jtype[i], q[qbi]);
            for (int k = 0; k < 6; k++) Sb[k]  = Si1[k];
            for (int k = 0; k < 6; k++) vJ[k]  = Si1[k] * qd[qbi];
            for (int k = 0; k < 6; k++) aS[k]  = Si1[k] * qdd[qbi];
        }

        mm66(Xup + 36*i, Xj, X + 36*i);

        if (parent[i] < 0) {
            for (int k = 0; k < 6; k++) v_out[6*i+k] = vJ[k];
            double neg_g[6] = {-a_grav[0],-a_grav[1],-a_grav[2],-a_grav[3],-a_grav[4],-a_grav[5]};
            mv66(a_out + 6*i, Xup + 36*i, neg_g);
            for (int k = 0; k < 6; k++) a_out[6*i+k] += aS[k];
        } else {
            mv66(tmp6a, Xup + 36*i, v_out + 6*parent[i]);
            for (int k = 0; k < 6; k++) v_out[6*i+k] = tmp6a[k] + vJ[k];
            mv66(tmp6a, Xup + 36*i, a_out + 6*parent[i]);
            crm(crmv, v_out + 6*i);
            mv66(tmp6b, crmv, vJ);
            for (int k = 0; k < 6; k++) a_out[6*i+k] = tmp6a[k] + tmp6b[k] + aS[k];
        }

        mv66(f_out + 6*i, I6 + 36*i, a_out + 6*i);
        crf(crfv, v_out + 6*i);
        mm66(tmp36a, crfv, I6 + 36*i);
        mv66(tmp6a, tmp36a, v_out + 6*i);
        for (int k = 0; k < 6; k++) f_out[6*i+k] += tmp6a[k] - f_ext[6*i+k];
    }

    /* ----- pass 2 (leaves → root): tau[qb:qb+nv] = Sᵀ·f, propagate f to parent ----- */
    for (int i = nb - 1; i >= 0; i--) {
        int qbi = q_base[i];
        double *Sb = S + 6 * qbi;
        int nvi = jt_nv(jtype[i]);
        for (int r = 0; r < nvi; r++) {
            double sf = 0.0;
            for (int k = 0; k < 6; k++) sf += Sb[6*r + k] * f_out[6*i + k];
            tau[qbi + r] = sf;
        }
        if (parent[i] >= 0) {
            mTv66(tmp6a, Xup + 36*i, f_out + 6*i);
            for (int k = 0; k < 6; k++) f_out[6*parent[i] + k] += tmp6a[k];
        }
    }

    /* ----- full=True post-pass: _a[i,3:] = a[i,3:] + v[i,:3] × v[i,3:] ----- */
    for (int i = 0; i < nb; i++) {
        double cx[3];
        cross3(v_out[6*i+0], v_out[6*i+1], v_out[6*i+2], v_out[6*i+3], v_out[6*i+4], v_out[6*i+5], cx);
        a_out[6*i+3] += cx[0];
        a_out[6*i+4] += cx[1];
        a_out[6*i+5] += cx[2];
    }
}

//mirrors rbd.py:crb_featherstone. Composite Rigid-Body Algorithm: builds the
//joint-space mass matrix H (row-major, nb*nb). Workspace layout: 78*nb doubles
//(Xup 36 + S 6 + Ic 36 per body).
//
//Pass 1: jcalc → Xup[i]=Xj·X[i]; seed Ic[i]=I[i].
//Pass 2 (leaves→root): Ic[parent] += Xup[i]^T · Ic[i] · Xup[i].
//Pass 3: for each i, fh=Ic[i]·S[i]; H[i,i]=S[i]·fh; walk ancestor chain,
//        fh ← Xup[j]^T·fh; H[i,parent_chain]=S[j]·fh (symmetric fill).
//
//Fixed joints (jtype==0) contribute no q/v slots — their rows/cols simply
//don't appear in H (no zero padding).
void crb_featherstone(int nb, double *X, double *I6, int *parent, int *jtype, double *q, double *H, double *workspace)
{
    /* H is (nq, nq) row-major. Fixed contributes 0 slots; free=6; ball=3; else 1. */
    int q_base[TACT_MAX_NB];
    int nq = 0;
    for (int i = 0; i < nb; i++) {
        q_base[i] = nq;
        nq += jt_nv(jtype[i]);
    }

    double *Xup = workspace;            //36*nb
    double *S   = Xup + 36*nb;          // 6*nq (was 6*nb) — body i col block at S+6*q_base[i]
    double *Ic  = S   +  6*nq;          //36*nb

    double Xj[36], Si1[6], S6[36];
    double tmp36a[36], tmp36b[36];

    /* ----- pass 1 (root → leaves): kinematics + seed composite inertia ----- */
    for (int i = 0; i < nb; i++) {
        int qbi = q_base[i];
        double *Sb = S + 6 * qbi;
        if (jtype[i] == 3) {
            jcalc_free(Xj, S6, q + qbi);
            for (int k = 0; k < 36; k++) Sb[k] = S6[k];
        } else if (jtype[i] == 4) {
            jcalc_ball(Xj, S6, q + qbi);
            for (int k = 0; k < 18; k++) Sb[k] = S6[k];
        } else if (jtype[i] == 0) {
            /* fixed: Xj=I, no S columns (nv=0; qbi shared with next body, so
             * don't write Sb). */
            identity(Xj, 6);
        } else {
            jcalc(Xj, Si1, jtype[i], q[qbi]);
            for (int k = 0; k < 6; k++) Sb[k] = Si1[k];
        }
        mm66(Xup + 36*i, Xj, X + 36*i);
        memcpy(Ic + 36*i, I6 + 36*i, 36*sizeof(double));
    }

    /* ----- pass 2 (leaves → root): propagate composite inertia up the tree ----- */
    for (int i = nb - 1; i >= 0; i--) {
        if (parent[i] < 0) continue;
        mm66 (tmp36a, Ic + 36*i, Xup + 36*i);
        mTm66(tmp36b, Xup + 36*i, tmp36a);
        for (int k = 0; k < 36; k++) Ic[36*parent[i]+k] += tmp36b[k];
    }

    /* ----- pass 3: build H (nq × nq). Diagonal block + ancestor blocks only. ----- */
    memset(H, 0, (size_t)nq * nq * sizeof(double));
    double fh_buf[36];                 /* up to 6 columns of fh (6×nv col-major) */
    double tmp6[6];
    for (int i = 0; i < nb; i++) {
        int qbi = q_base[i];
        int nvi = jt_nv(jtype[i]);
        double *Sb = S + 6 * qbi;

        /* fh = Ic[i] · S[i]   (6 × nvi) col-major */
        for (int c = 0; c < nvi; c++) {
            mv66(fh_buf + 6*c, Ic + 36*i, Sb + 6*c);
        }
        /* H[qb_i:qb_i+nvi, qb_i:qb_i+nvi] = Sᵀ · fh  (nvi × nvi) */
        for (int r = 0; r < nvi; r++) {
            for (int c = 0; c < nvi; c++) {
                double s = 0.0;
                for (int k = 0; k < 6; k++) s += Sb[6*r + k] * fh_buf[6*c + k];
                H[(size_t)nq*(qbi + r) + (qbi + c)] = s;
            }
        }

        int j = i;
        while (parent[j] >= 0) {
            /* fh ← Xup[j]^T · fh   (apply column-wise; element-wise assign to
             * avoid GCC stringop-overflow false positive on memcpy into fh_buf). */
            for (int c = 0; c < nvi; c++) {
                mTv66(tmp6, Xup + 36*j, fh_buf + 6*c);
                for (int kk = 0; kk < 6; kk++) fh_buf[6*c + kk] = tmp6[kk];
            }
            j = parent[j];
            int qbj = q_base[j];
            int nvj = jt_nv(jtype[j]);
            double *Sjb = S + 6 * qbj;
            /* H[qb_j..qb_j+nvj, qb_i..qb_i+nvi] = Sjᵀ · fh   (nvj × nvi) */
            for (int r = 0; r < nvj; r++) {
                for (int c = 0; c < nvi; c++) {
                    double s = 0.0;
                    for (int k = 0; k < 6; k++) s += Sjb[6*r + k] * fh_buf[6*c + k];
                    H[(size_t)nq*(qbj + r) + (qbi + c)] = s;
                    H[(size_t)nq*(qbi + c) + (qbj + r)] = s;   /* symmetric */
                }
            }
        }
    }
}

//---- general dense SPD LDL^T --------------------------------------------------
//In-place LDL^T factor of n×n row-major SPD matrix A. Lower triangle holds L
//(unit diagonal implicit), diagonal holds D, upper triangle is untouched.
//Returns 0 on success; -(k+1) when pivot D[k] ≤ TACT_EPS (matrix not numerically SPD).
//
//Outer-product variant: D[j] = A[j,j] - Σ_{k<j} L[j,k]² · D[k];
//                       L[i,j] = (A[i,j] - Σ_{k<j} L[i,k] · L[j,k] · D[k]) / D[j].
//
//Use with crb_featherstone (mass matrix M): caller compresses M_full to the
//free-joint submatrix (jtype>0 rows/cols) before calling — keeps the kernel
//generic and reusable for IK / future SPD problems.
int ldlt_factor(double *A, int n)
{
    for (int j = 0; j < n; j++) {
        double Dj = A[j*n + j];
        for (int k = 0; k < j; k++) {
            double Ljk = A[j*n + k];
            Dj -= Ljk * Ljk * A[k*n + k];
        }
        if (Dj <= TACT_EPS) return -(j + 1);
        A[j*n + j] = Dj;
        double inv_Dj = 1.0 / Dj;
        for (int i = j + 1; i < n; i++) {
            double s = A[i*n + j];
            for (int k = 0; k < j; k++) s -= A[i*n + k] * A[j*n + k] * A[k*n + k];
            A[i*n + j] = s * inv_Dj;
        }
    }
    return 0;
}

//Solve A·x = b in place (b overwritten with x) given LDL^T factor from ldlt_factor.
//Forward L·y = b (unit-diag), diagonal D·z = y, backward L^T·x = z.
void ldlt_solve(const double *A, int n, double *b)
{
    for (int i = 0; i < n; i++) {                       //forward: L y = b
        double s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*n + k] * b[k];
        b[i] = s;
    }
    for (int i = 0; i < n; i++) b[i] /= A[i*n + i];     //diagonal: z = y / D
    for (int i = n - 1; i >= 0; i--) {                  //backward: L^T x = z
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= A[k*n + i] * b[k];
        b[i] = s;
    }
}

/* q_step: advance q by qd over dt with per-jtype manifold semantics. 1-DoF
 * joints use q_next = q + qd·dt. free (jtype=3) integrates translation via
 * the body-frame v (p_next = p + R(w)·v·dt) and rotation via SO(3) exp map.
 * Mirrors rbd.py:_q_step. q_next may alias q. */
void q_step(int nb, int *jtype, const double *q, const double *qd, double dt, double *q_next)
{
    int q_idx = 0;
    for (int i = 0; i < nb; i++) {
        if (jtype[i] == 3) {
            double R[9];
            expmap_so3(q + q_idx + 3, R);
            double vw[3];
            vw[0] = R[0]*qd[q_idx+0] + R[1]*qd[q_idx+1] + R[2]*qd[q_idx+2];
            vw[1] = R[3]*qd[q_idx+0] + R[4]*qd[q_idx+1] + R[5]*qd[q_idx+2];
            vw[2] = R[6]*qd[q_idx+0] + R[7]*qd[q_idx+1] + R[8]*qd[q_idx+2];
            double pnew[3];
            pnew[0] = q[q_idx+0] + vw[0]*dt;
            pnew[1] = q[q_idx+1] + vw[1]*dt;
            pnew[2] = q[q_idx+2] + vw[2]*dt;
            double wnew[3];
            integrate_so3(q + q_idx + 3, qd + q_idx + 3, dt, wnew);
            q_next[q_idx+0] = pnew[0];
            q_next[q_idx+1] = pnew[1];
            q_next[q_idx+2] = pnew[2];
            q_next[q_idx+3] = wnew[0];
            q_next[q_idx+4] = wnew[1];
            q_next[q_idx+5] = wnew[2];
            q_idx += 6;
        } else if (jtype[i] == 4) {
            /* ball: rotation vector via SO(3) exp-map composition */
            double wnew[3];
            integrate_so3(q + q_idx, qd + q_idx, dt, wnew);
            q_next[q_idx+0] = wnew[0];
            q_next[q_idx+1] = wnew[1];
            q_next[q_idx+2] = wnew[2];
            q_idx += 3;
        } else if (jtype[i] == 0) {
            /* fixed: no q slot to advance */
        } else {
            q_next[q_idx] = q[q_idx] + qd[q_idx] * dt;
            q_idx += 1;
        }
    }
}

//mirrors rbd.py:euler_step. forward dynamics + 1st-order semi-implicit integration.
void euler_step(int nb, double *X, double *I6, int *parent, int *jtype, double *q, double *qd, double *tau, double *f_ext, double *g, double dt, double *q_next, double *qd_next, double *qdd, double *f_out, double *a_out, double *v_out, double *workspace)
{
    aba_featherstone(nb, X, I6, parent, jtype, q, qd, tau, f_ext, g, qdd, f_out, a_out, v_out, workspace, /*ff=*/NULL, /*sk=*/NULL, /*armature=*/NULL, /*dt_imp=*/0.0, /*Kp_j=*/NULL, /*Kd_j=*/NULL, /*q_ref=*/NULL, /*qd_ref=*/NULL, /*full=*/1);
    /* per-DoF: nq elements */
    int nq = 0;
    for (int i = 0; i < nb; i++) nq += jt_nv(jtype[i]);
    for (int i = 0; i < nq; i++) qd_next[i] = qd[i] + qdd[i] * dt;
    q_step(nb, jtype, q, qd_next, dt, q_next);
}

//mirrors rbd.py:rk4_step. 4× aba_featherstone(full=0) + final rne_featherstone.
//workspace layout: [aba slice ≈ 90*nb+13*nq+d_total] + [10*nq intermediates] + [nq tau_dummy].
void rk4_step(int nb, double *X, double *I6, int *parent, int *jtype, double *q, double *qd, double *tau, double *f_ext, double *g, double dt, double *q_next, double *qd_next, double *qdd, double *f_out, double *a_out, double *v_out, double *workspace)
{
    /* Compute nq and aba's actual workspace footprint so the rk4 intermediates
     * slice cleanly off the tail. d_total ≤ 36*nb (worst case all free). */
    int nq = 0, d_total = 0;
    for (int i = 0; i < nb; i++) {
        int nvi = jt_nv(jtype[i]);
        nq      += nvi;
        d_total += nvi * nvi;
    }
    size_t aba_off = (size_t)90*nb + (size_t)13*nq + (size_t)d_total + 64;

    double *ab_ws = workspace;
    double *qdd1  = ab_ws  + aba_off;                 // nq
    double *qdd2  = qdd1   +     nq;
    double *qdd3  = qdd2   +     nq;
    double *qdd4  = qdd3   +     nq;
    double *qk1   = qdd4   +     nq;
    double *qk2   = qk1    +     nq;
    double *qk3   = qk2    +     nq;
    double *qdk1  = qk3    +     nq;
    double *qdk2  = qdk1   +     nq;
    double *qdk3  = qdk2   +     nq;
    double *tau_dummy = qdk3 + nq;                    // nq (replaces stack array, free-safe)

    /* stage 1: at (q, qd) */
    aba_featherstone(nb, X, I6, parent, jtype, q, qd, tau, f_ext, g, qdd1, f_out, a_out, v_out, ab_ws, NULL, NULL, NULL, 0.0, NULL, NULL, NULL, NULL, /*full=*/0);
    for (int i = 0; i < nq; i++) qdk1[i] = qd[i] + qdd1[i] * dt;
    q_step(nb, jtype, q, qdk1, dt, qk1);

    /* stage 2: at (qk1, qdk1) */
    aba_featherstone(nb, X, I6, parent, jtype, qk1, qdk1, tau, f_ext, g, qdd2, f_out, a_out, v_out, ab_ws, NULL, NULL, NULL, 0.0, NULL, NULL, NULL, NULL, /*full=*/0);
    for (int i = 0; i < nq; i++) qdk2[i] = qd[i] + qdd2[i] * 0.5 * dt;
    q_step(nb, jtype, q, qdk2, 0.5*dt, qk2);

    /* stage 3: at (qk2, qdk2) */
    aba_featherstone(nb, X, I6, parent, jtype, qk2, qdk2, tau, f_ext, g, qdd3, f_out, a_out, v_out, ab_ws, NULL, NULL, NULL, 0.0, NULL, NULL, NULL, NULL, /*full=*/0);
    for (int i = 0; i < nq; i++) qdk3[i] = qd[i] + qdd3[i] * 0.5 * dt;
    q_step(nb, jtype, q, qdk3, 0.5*dt, qk3);

    /* stage 4: at (qk3, qdk3) */
    aba_featherstone(nb, X, I6, parent, jtype, qk3, qdk3, tau, f_ext, g, qdd4, f_out, a_out, v_out, ab_ws, NULL, NULL, NULL, 0.0, NULL, NULL, NULL, NULL, /*full=*/0);
    /* qk4 / qdk4 (stage-4 endpoint) computed on the fly inside the average below */

    /* averaged outputs: qdd and qd_next are per-DoF linear; q_next uses linear
     * average for 1-DoF, then q_step overwrites for free (SO(3) not linear). */
    double inv6 = 1.0 / 6.0;
    for (int i = 0; i < nq; i++) {
        qdd[i]      = (qdd1[i] + 2.0*qdd2[i] + 2.0*qdd3[i] + qdd4[i]) * inv6;
        double qdk4_i = qd[i] + qdd4[i] * dt;
        qd_next[i]  = (qdk1[i] + 2.0*qdk2[i] + 2.0*qdk3[i] + qdk4_i) * inv6;
    }
    /* q_next: linear average for 1-DoF DoFs */
    {
        int q_idx = 0;
        for (int i = 0; i < nb; i++) {
            if (jtype[i] == 3) { q_idx += 6; continue; }
            if (jtype[i] == 4) { q_idx += 3; continue; }
            if (jtype[i] == 0) continue;             /* fixed: no q slot */
            /* qk4_i = q + qdk4_i*dt where qdk4_i = qd + qdd4*dt */
            double qdk4_i = qd[q_idx] + qdd4[q_idx] * dt;
            double qk4_i  = q[q_idx]  + qdk4_i * dt;
            q_next[q_idx] = (qk1[q_idx] + 2.0*qk2[q_idx] + 2.0*qk3[q_idx] + qk4_i) * inv6;
            q_idx += 1;
        }
    }
    /* free slots: overwrite with single-shot q_step from averaged qd_next */
    {
        int q_idx = 0;
        for (int i = 0; i < nb; i++) {
            if (jtype[i] == 0) continue;             /* fixed: no q slot */
            if (jtype[i] == 3) {
                /* compute SO(3)-aware update from q (start) + qd_next over dt */
                double R[9];
                expmap_so3(q + q_idx + 3, R);
                double vw0 = R[0]*qd_next[q_idx+0] + R[1]*qd_next[q_idx+1] + R[2]*qd_next[q_idx+2];
                double vw1 = R[3]*qd_next[q_idx+0] + R[4]*qd_next[q_idx+1] + R[5]*qd_next[q_idx+2];
                double vw2 = R[6]*qd_next[q_idx+0] + R[7]*qd_next[q_idx+1] + R[8]*qd_next[q_idx+2];
                q_next[q_idx+0] = q[q_idx+0] + vw0*dt;
                q_next[q_idx+1] = q[q_idx+1] + vw1*dt;
                q_next[q_idx+2] = q[q_idx+2] + vw2*dt;
                integrate_so3(q + q_idx + 3, qd_next + q_idx + 3, dt, q_next + q_idx + 3);
                q_idx += 6;
            } else if (jtype[i] == 4) {
                /* ball: SO(3)-aware update from q (start) + averaged qd_next over dt */
                integrate_so3(q + q_idx, qd_next + q_idx, dt, q_next + q_idx);
                q_idx += 3;
            } else {
                q_idx += 1;
            }
        }
    }

    /* final inverse-dynamics call at the averaged qdd (workspace head reused by rne). */
    rne_featherstone(nb, X, I6, parent, jtype, q, qd, qdd, f_ext, g, tau_dummy, f_out, a_out, v_out, ab_ws);
}

//---- moved from ccd.c: orthonormal frame whose z-column = normalized z_in (used by contact_lcp) ----
//builds a 3×3 orthonormal frame whose z-column equals the (normalized) input z.
//columns: [x|y|z]. R is row-major, R[3*r+c]. used by contact_lcp.
void choose_rotation(double *z_in, double *R){
    double n = sqrt(z_in[0]*z_in[0] + z_in[1]*z_in[1] + z_in[2]*z_in[2]);
    double z[3] = {z_in[0]/n, z_in[1]/n, z_in[2]/n};
    double perp[3];
    if (z[0] != 0.0 || z[1] != 0.0) { perp[0] = -z[1]; perp[1] =  z[0]; perp[2] = 0.0; }
    else                            { perp[0] =  0.0;  perp[1] =  z[2]; perp[2] = -z[1]; }
    double pn = sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2]);
    double y[3] = {perp[0]/pn, perp[1]/pn, perp[2]/pn};
    double x[3];
    cross3(y[0], y[1], y[2], z[0], z[1], z[2], x);
    R[0]=x[0]; R[1]=y[0]; R[2]=z[0];
    R[3]=x[1]; R[4]=y[1]; R[5]=z[1];
    R[6]=x[2]; R[7]=y[2]; R[8]=z[2];
}
