"""Rigid-body dynamics primitives — pure functions split out from sim.py.

Contents:
  - rotation/quaternion/homogeneous helpers (eulerseq convention,
    lowercase=extrinsic, uppercase=intrinsic)
  - spatial-vector building blocks: crm, crf, jcalc, get_spatial_inertia,
    get_spatial_transform, _fk
  - dynamics algorithms: crb_featherstone, rne_featherstone,
    aba_featherstone, rne_lwp, inertia/cc/gravity/jacob_whitney, fbik
  - integrators: euler_step, euler_step2, rk4_step
  - contact: contact_ground_sphere, contact_lcp
    (call into libtact.so via _clib for narrow-phase + LCP solvers)
  - ray casts: ray_intersects_{triangle,mesh,box,sphere,cylinder,capsule}

Model / Env / CEnv (the stateful simulator classes) live in sim.py and
re-export this module via `from .rbd import *` so callers can stay flat:
`tact.crm(...)`, `tact.aba_featherstone(...)`."""
import ctypes, math, copy
import numpy as np
from ._clib import clib
#------------------------basic frame calculation functions---------------------

#convention: lowercase eulerseq = extrinsic, uppercase = intrinsic.
#supported: any 3-char sequence over {x,y,z} with adjacent axes differing,
#i.e. 12 Tait-Bryan (xyz/xzy/yxz/yzx/zxy/zyx and their uppercase) and
#6 proper-Euler (xyx/xzx/yxy/yzy/zxz/zyz and their uppercase) conventions.
_GIMBAL_TOL = 1e-9
_AXIS_IDX = {'x': 0, 'y': 1, 'z': 2}

# Contact manifold capacity per cpair (kept in sync with tact.h #define).
# Currently the narrowphase still returns 1 point per pair (sub_id always 0), so
# slots 1..3 are unused. Phase 2+: box-box SAT/clipping narrowphase will fill
# 0..MAX_PTS_PER_PAIR-1. Indexing: warm-start slot = cpair_idx*MAX_PTS_PER_PAIR + sub_id.
MAX_PTS_PER_PAIR = 4

def _parse_eulerseq(eulerseq):
    if not isinstance(eulerseq, str) or len(eulerseq) != 3:
        raise ValueError("eulerseq must be a 3-character string, got %r" % (eulerseq,))
    if   eulerseq.islower(): intrinsic = False
    elif eulerseq.isupper(): intrinsic = True
    else: raise ValueError("eulerseq must be all-lower (extrinsic) or all-upper (intrinsic), got %r" % eulerseq)
    axes = eulerseq.lower()
    if any(a not in 'xyz' for a in axes):
        raise ValueError("eulerseq axes must be drawn from x/y/z, got %r" % eulerseq)
    if axes[0] == axes[1] or axes[1] == axes[2]:
        raise ValueError("adjacent axes in eulerseq must differ, got %r" % eulerseq)
    return axes, intrinsic

def _rot_axis(c, t):
    if c == 'x': return rot_x(t)
    if c == 'y': return rot_y(t)
    return rot_z(t)

#composes elementary axis rotations for any of the 24 standard conventions.
#~1.5x slower than the original 4-case explicit-formula version in pure Python,
#but a C port at -O3 -march=native dead-code-eliminates the elementary-rotation
#zeros and produces identical assembly to the explicit form (measured: 21->21 ns).
def euler_to_rotation(x, eulerseq='xyz', deg=False):
    axes, intrinsic = _parse_eulerseq(eulerseq)
    x = np.asarray(x, dtype=float)
    if deg: x = np.deg2rad(x)
    R0 = _rot_axis(axes[0], x[0])
    R1 = _rot_axis(axes[1], x[1])
    R2 = _rot_axis(axes[2], x[2])
    return R0 @ R1 @ R2 if intrinsic else R2 @ R1 @ R0      #intr ABC vs extr abc

#extrinsic abc with angles (α,β,γ) ≡ intrinsic CBA with angles (γ,β,α);
#we extract for the equivalent intrinsic sequence and reverse if extrinsic was requested.
#tait-bryan parity σ=+1 if (i,j,k) is an even permutation of (0,1,2), else -1;
#proper-euler uses the third axis l=3-i-j with the same parity.
def rotation_to_euler(R, eulerseq='xyz', deg=False):
    axes, intrinsic = _parse_eulerseq(eulerseq)
    if intrinsic: i, j, k = (_AXIS_IDX[a] for a in axes)
    else:         i, j, k = (_AXIS_IDX[a] for a in axes[::-1])
    sigma = 1 if (i + 1) % 3 == j else -1

    if i != k:                                 #Tait-Bryan
        s_b = max(-1.0, min(1.0, sigma * R[i, k]))
        beta = math.asin(s_b)
        if abs(s_b) > 1.0 - _GIMBAL_TOL:       #gimbal lock at β=±π/2: γ undetermined, set to 0
            sgn = 1.0 if s_b > 0 else -1.0
            alpha = math.atan2(sgn * R[j, i], R[j, j])
            gamma = 0.0
        else:
            alpha = math.atan2(-sigma * R[j, k], R[k, k])
            gamma = math.atan2(-sigma * R[i, j], R[i, i])
    else:                                      #proper Euler (i==k)
        l = 3 - i - j
        c_b = max(-1.0, min(1.0, R[i, i]))
        beta = math.acos(c_b)
        if abs(c_b) > 1.0 - _GIMBAL_TOL:       #β≈0 or π: γ undetermined, set to 0
            if c_b > 0: alpha = math.atan2(-sigma * R[j, l], R[l, l])
            else:       alpha = math.atan2( sigma * R[j, l], -R[l, l])
            gamma = 0.0
        else:
            alpha = math.atan2(R[j, i], -sigma * R[l, i])
            gamma = math.atan2(R[i, j],  sigma * R[i, l])

    x = np.array([alpha, beta, gamma])
    if not intrinsic: x = x[::-1].copy()       #extrinsic: reverse the intrinsic-CBA result
    if deg: x = np.rad2deg(x)
    return x

#https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
def quat_to_rotation(q):
    #q: [w, x, y, z]
    R = np.zeros((3, 3))
    R[0][0] = 2*(q[0]*q[0] + q[1]*q[1]) - 1
    R[0][1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0][2] = 2*(q[1]*q[3] + q[0]*q[2])
    R[1][0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[1][1] = 2*(q[0]*q[0] + q[2]*q[2]) - 1
    R[1][2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2][0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2][1] = 2*(q[2]*q[3] + q[0]*q[1])
    R[2][2] = 2*(q[0]*q[0] + q[3]*q[3]) - 1
    return R

def rotation_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5/math.sqrt(trace + 1.0)
        w = 0.25/s
        x = (R[2, 1] - R[1, 2])*s
        y = (R[0, 2] - R[2, 0])*s
        z = (R[1, 0] - R[0, 1])*s
        
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0*math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2])/s
        x = 0.25*s
        y = (R[0, 1] + R[1, 0])/s
        z = (R[0, 2] + R[2, 0])/s
        
    elif R[1, 1] > R[2, 2]:
        s = 2.0*math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0])/s
        x = (R[0, 1] + R[1, 0])/s
        y = 0.25*s
        z = (R[1, 2] + R[2, 1])/s
        
    else:
        s = 2.0*math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1])/s
        x = (R[0, 2] + R[2, 0])/s
        y = (R[1, 2] + R[2, 1])/s
        z = 0.25*s
    q =  np.array([w, x, y, z])
    #if q[3] < 0: q = -q
    return q

def euler_to_quat(e, eulerseq='xyz', deg=False):
    R = euler_to_rotation(e, eulerseq, deg)
    q = rotation_to_quat(R)
    return q 
    
def quat_to_euler(q, eulerseq='xyz', deg=False):
    R = quat_to_rotation(q)
    e = rotation_to_euler(R, eulerseq, deg)
    return e 

def xyzeuler_to_xyzquat(x6, eulerseq='xyz', deg=False):
    x7 = np.zeros(7)
    x7[0:3] = x6[0:3]
    q = euler_to_quat(x6[3:], eulerseq, deg)
    x7[3:] = q[0:]
    return x7

def xyzquat_to_xyzeuler(x7, eulerseq='xyz', deg=False):
    x6 = np.zeros(6)
    x6[0:3] = x7[0:3]
    x6[3:6] = quat_to_euler(x7[3:7], eulerseq, deg)
    return x6

def homogeneous_to_xyzeuler(T, eulerseq='xyz', deg=False):
    x = np.zeros(6)
    x[0] = T[0][3]
    x[1] = T[1][3]
    x[2] = T[2][3]
    x[3:6] = rotation_to_euler(T[:3, :3], eulerseq, deg);
    return x
    
def xyzeuler_to_homogeneous(x, eulerseq='xyz', deg=False):
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation(x[3:6], eulerseq, deg)
    T[0][3] = x[0]
    T[1][3] = x[1]
    T[2][3] = x[2]
    return T

def xyzquat_to_homogeneous(v7):
    T = np.eye(4)
    R = quat_to_rotation(v7[3:7])
    T[:3, :3] = R
    T[:3, 3] = v7[:3]
    return T

def homogeneous_to_xyzquat(T):
    q = rotation_to_quat(T[:3, :3])
    v7 = np.array([T[0][3], T[1][3], T[2][3], q[0], q[1], q[2], q[3]])
    return v7

def xyheading_to_homogeneous(x, y, heading):
    T = np.eye(3)
    T[0][0] = math.cos(heading)
    T[0][1] = -math.sin(heading)
    T[0][2] = x
    T[1][0] = math.sin(heading)
    T[1][1] = math.cos(heading)
    T[1][2] = y
    return T

def rotxyz_to_homogeneous(R, p):
    p = np.reshape(p, (3, 1))
    tmp1 = np.hstack((R, p))
    tmp2 = np.array([0, 0, 0, 1])
    T = np.vstack((tmp1, tmp2))
    return T

#this function is used in contact_lcp()
def choose_rotation(z):
    z = z/np.linalg.norm(z)
    if z[0] != 0 or z[1] != 0: perpendicular = np.array([-z[1], z[0], 0])
    else: perpendicular = np.array([0, z[2], -z[1]])
    y = perpendicular/np.linalg.norm(perpendicular)
    x = np.cross(y, z)
    R = np.array([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]])
    return R

#R1: desired  R2: now    →  e ∈ ℝ³ such that R1 = expmap_so3(e) @ R2.
#World-frame rotation vector; ‖e‖ = θ (monotonic 0..π), direction = rotation
#axis. Matches the bottom 3 rows of jacob_whitney (world-frame ω). Old impl
#returned ~sin(θ)·n, which was identical to first order at small angles but
#decayed past 90° and vanished at 180° (unstable equilibrium in JTC).
def rotation_error(R1, R2):
    return logmap_so3(R1 @ R2.T)

#R1: desired  R2: now    — antisymmetric-vee form of the old sin-shape error.
#Mathematically equivalent to the pre-logmap rotation_error; kept around as a
#reference implementation but never wired in. Commented out since rotation_error
#was switched to logmap-based; uncomment if you need the sin-shape behavior back.
#def rotation_error2(R1, R2):
#    R_e = R1 @ R2.T
#    e = 0.5*np.array([R_e[2, 1] - R_e[1, 2], R_e[0, 2] - R_e[2, 0], R_e[1, 0] - R_e[0, 1]])
#    return e

#T1: desired  T2: now
def homogeneous_error(T1, T2):
    e_t = np.array(T1[0:3, 3] - T2[0:3, 3])
    e_o = rotation_error(T1[:3, :3], T2[:3, :3])
    e = np.append(e_t, e_o)
    return e

#----- SO(3) exp/log -----
# Used by axis-angle free joint (jtype=3): q[3:6] is the rotation vector
# (axis * angle, radians), qd[3:6] is body-frame angular velocity ω.
# Integration is q_next = log(exp(q) · exp(ω·dt)) which keeps SO(3) closed
# under finite rotations without the Euler-rate singularity at pitch=±π/2.

def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

def expmap_so3(w):
    """Rotation vector w ∈ ℝ³ → R ∈ SO(3) via Rodrigues. Smooth at θ=0."""
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        K = skew(w)
        return np.eye(3) + K + 0.5 * (K @ K)   # 2nd-order Taylor
    n = w / theta
    K = skew(n)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def logmap_so3(R):
    """R ∈ SO(3) → rotation vector w ∈ ℝ³. Stable at θ=0 and θ=π."""
    cos_t = 0.5 * (R[0,0] + R[1,1] + R[2,2] - 1.0)
    cos_t = min(1.0, max(-1.0, cos_t))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        # near identity: log ≈ 0.5 * (R - R^T) vee
        return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    if math.pi - theta < 1e-6:
        # near antipodal: sin(θ) ≈ 0. Recover axis from diag of (R + I)/2 = n n^T.
        M = 0.5 * (R + np.eye(3))
        # pick the largest diagonal for numerical safety
        k = int(np.argmax(np.array([M[0,0], M[1,1], M[2,2]])))
        n = np.zeros(3)
        n[k] = math.sqrt(max(0.0, M[k,k]))
        if n[k] > 1e-12:
            for j in range(3):
                if j != k: n[j] = M[k,j] / n[k]
        return theta * n
    return (theta / (2.0 * math.sin(theta))) * np.array(
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def integrate_so3(w, omega_body, dt):
    """Compose rotation w (vector) with body-frame ω over dt → new rotation vector."""
    return logmap_so3(expmap_so3(w) @ expmap_so3(omega_body * dt))

def rot_x(t):
    R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return R

def rot_y(t):
    R = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    return R

def rot_z(t):
    R = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    return R

def T_rot_x(t):
    T = np.array([[1, 0, 0, 0], [0, math.cos(t), -math.sin(t), 0], [0, math.sin(t), math.cos(t), 0], [0, 0, 0, 1]])
    return T

def T_rot_y(t):
    T = np.array([[math.cos(t), 0, math.sin(t), 0], [0, 1, 0, 0], [-math.sin(t), 0, math.cos(t), 0], [0, 0, 0, 1]])
    return T

def T_rot_z(t):
    T = np.array([[math.cos(t), -math.sin(t), 0, 0], [math.sin(t), math.cos(t), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return T

def T_trans(x):
    T = np.array([[1, 0, 0, x[0]], [0, 1, 0, x[1]], [0, 0, 1, x[2]], [0, 0, 0, 1]])
    return T

#-----------------------rigid body dynamic simulation functions (mostly featherstone's spatial_v1 matlab library) -----------------

#this function is not used in this library but do not remove
def X_rot_x(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    X = np.array([[1, 0, 0, 0, 0, 0], [0, c, s, 0, 0, 0], [0, -s, c, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, c, s], [0, 0, 0, 0, -s, c]])
    return X

#this function is not used in this library but do not remove
def X_rot_y(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    X = np.array([[c, 0, -s, 0, 0, 0], [0, 1, 0, 0, 0, 0], [s, 0, c, 0, 0, 0], [0, 0, 0, c, 0, -s], [0, 0, 0, 0, 1, 0], [0, 0, 0, s, 0, c]])
    return X

def X_rot_z(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    X = np.array([[c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]])
    return X

def X_trans(r):
    X = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, r[2], -r[1], 1, 0, 0], [-r[2], 0, r[0], 0, 1, 0], [r[1], -r[0], 0, 0, 0, 1]])
    return X

'''
def mcI2rbi(m, c, I):
    C = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]]) #skew
    rbi = np.zeros((6, 6))
    rbi[0:3, 0:3] = I + m * C @ np.transpose(C)
    rbi[0:3, 3:6] = m * C
    rbi[3:6, 0:3] = m * np.transpose(C)
    rbi[3:6, 3:6] = m * np.eye(3)
    return rbi

def rbi2mcI(rbi):
    m = rbi[5, 5]
    mC = rbi[0:3, 3:6]
    c = 0.5*np.array([rbi[2][1] - rbi[1][2], rbi[0][2] - rbi[2][0], rbi[1][0] - rbi[0][1]]) #skew
    I = rbi[0:3, 0:3] - (1/m)*mC*np.transpose(mC)
    return m, c, I
'''

def homogeneous_to_pluker(T):
    X = np.zeros((6, 6))
    T = np.linalg.inv(T)
    E = T[0:3, 0:3]
    r = T[0:3, 3]

    X[0:3, 0:3] = E
    X[3:6, 0:3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]]) @ E 
    X[3:6, 3:6] = E
    return X

def pluker_to_homogeneous(X):
    T = np.eye(4)
    E = X[0:3, 0:3]
    mErx = X[3:6, 0:3]
    
    T[0:3, 0:3] = E
    T[0:3, 3] = 0.5 * np.array([mErx[2, 1] - mErx[1, 2], mErx[0, 2] - mErx[2, 0], mErx[1, 0] - mErx[0, 1]])
    return np.linalg.inv(T)


def crm(v):
    vcross = np.array([[    0, -v[2],  v[1],     0,     0,     0],
                       [ v[2],     0, -v[0],     0,     0,     0],
                       [-v[1],  v[0],     0,     0,     0,     0],
                       [    0, -v[5],  v[4],     0, -v[2],  v[1]],
                       [ v[5],    0,  -v[3],  v[2],     0, -v[0]],
                       [-v[4],  v[3],     0, -v[1],  v[0],     0]])
    return vcross

#original crf from featherstone's matlab code
#def crf(v):
#    vcross = -np.transpose(crm(v))
#    return vcross

#modeifed crf for performance
def crf(v):
    vcross = np.array([[    0, -v[2],  v[1],     0, -v[5],  v[4]],
                       [ v[2],     0, -v[0],  v[5],     0, -v[3]],
                       [-v[1],  v[0],     0, -v[4],  v[3],     0],
                       [    0,     0,     0,     0, -v[2],  v[1]],
                       [    0,     0,     0,  v[2],     0, -v[0]],
                       [    0,     0,     0, -v[1],  v[0],    0]])    
    return vcross

def jcalc(jtype, q):
    if jtype == 0: #fixed joint
        Xj = np.eye(6)
        S = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

    elif jtype == 1: #revolute joint - Z
        Xj = X_rot_z(q)
        S = np.array([0, 0, 1, 0, 0, 0], dtype=np.float64)

    elif jtype == 2: #prismatic joint - Z
        Xj = X_trans([0, 0, q])
        S = np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)

    return Xj, S

#----- free (jtype=3) primitives -----
# Single-body 6-DoF free joint with axis-angle rotation.
#   q[0:3]  = p_world  (world-frame position of body origin)
#   q[3:6]  = w        (rotation vector; R = expmap_so3(w) maps body→world)
#   qd[0:3] = v_body   (body-frame linear velocity)
#   qd[3:6] = ω_body   (body-frame angular velocity)
# Featherstone spatial-velocity convention is [ω; v], so the motion subspace
# S permutes qd's two halves into spatial layout: S @ qd = [ω; v].
_S_FREE6 = np.zeros((6, 6))
_S_FREE6[:3, 3:] = np.eye(3)
_S_FREE6[3:, :3] = np.eye(3)

def jcalc6(q6):
    """jcalc for the 6-DoF axis-angle free joint (jtype=3).
    q6 layout: [px, py, pz, wx, wy, wz]. Returns (XJ 6×6, S 6×6)."""
    T = np.eye(4)
    T[:3, :3] = expmap_so3(q6[3:6])
    T[:3, 3]  = q6[0:3]
    XJ = homogeneous_to_pluker(T)
    return XJ, _S_FREE6.copy()

def _build_qidx(jtype):
    """Compute state-vector indexing for both the q (position) and v (velocity)
    sides. Returns (q_base, v_base, nq_per_body, nv_per_body, nq, nv).

      q_base[i]      — first index in q vector for body i
      v_base[i]      — first index in qd/qdd/tau/etc. vector for body i
      nq_per_body[i] — q-slot count for body i (per-body position state)
      nv_per_body[i] — DoF count for body i (per-body velocity state)
      nq             — total q vector length (sum of nq_per_body)
      nv             — total v vector length (sum of nv_per_body)

    Current axis-angle convention: nq_per_body[i] == nv_per_body[i] for every
    body, so q_base == v_base and nq == nv numerically. The q/v distinction is
    kept in naming to make hot-path indexing self-documenting and to make a
    potential future swap to quaternion (where jtype=3 would have
    nq_per_body=7 ≠ nv_per_body=6) mostly mechanical.
    Joint mapping:
      jtype 0 (fixed)     : nq_per_body=0, nv_per_body=0 (no state)
      jtype 1 (revolute Z): nq_per_body=1, nv_per_body=1
      jtype 2 (prismatic Z): nq_per_body=1, nv_per_body=1
      jtype 3 (free axis-angle): nq_per_body=6, nv_per_body=6"""
    nb = len(jtype)
    nv_per_body = np.array([6 if jt == 3 else (0 if jt == 0 else 1) for jt in jtype], dtype=np.int32)
    nq_per_body = nv_per_body                       # equal under axis-angle
    q_base = np.zeros(nb, dtype=np.int32)
    v_base = np.zeros(nb, dtype=np.int32)
    for i in range(1, nb):
        q_base[i] = q_base[i-1] + nq_per_body[i-1]
        v_base[i] = v_base[i-1] + nv_per_body[i-1]
    nq = int(q_base[-1] + nq_per_body[-1]) if nb > 0 else 0
    nv = int(v_base[-1] + nv_per_body[-1]) if nb > 0 else 0
    return q_base, v_base, nq_per_body, nv_per_body, nq, nv

#convert 3x3 inertia tensor -> 6x6 spatial inertia
def get_spatial_inertia(m, c, I):
    nb = len(c)
    I6 = np.zeros((nb, 6, 6))
    
    for i in range(nb):
        C = np.array([[0, -c[i][2], c[i][1]], [c[i][2], 0, -c[i][0]], [-c[i][1], c[i][0], 0]])
        I6[i][0:3, 0:3] = I[i] + m[i] * C @ np.transpose(C)
        I6[i][0:3, 3:6] = m[i] * C
        I6[i][3:6, 0:3] = m[i] * np.transpose(C)
        I6[i][3:6, 3:6] = m[i] * np.eye(3)
    return I6

#convert 4x4 homogeneous transform -> 6x6 spatial transform
def get_spatial_transform(Ti):
    nb = len(Ti)
    X = np.zeros((nb, 6, 6))
    
    for i in range(nb):
        tmp = np.linalg.inv(Ti[i])
        E = tmp[0:3, 0:3]
        r = tmp[0:3, 3]

        X[i][0:3, 0:3] = E
        X[i][3:6, 0:3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]]) @ E 
        X[i][3:6, 3:6] = E                
    return X

#--------------------------------------------

def _fk(Ti, parent, jtype, q):
    nb = len(Ti)
    q_base, _, _, _, _, _ = _build_qidx(jtype)
    T = np.zeros((nb, 4, 4))
    Tb = np.zeros((nb, 4, 4))

    for i in range(nb):
        if jtype[i] == 0:
            Tb[i] = Ti[i].copy()
        elif jtype[i] == 1:
            qi0 = q[q_base[i]]
            Tb[i] = Ti[i] if qi0 == 0 else Ti[i] @ T_rot_z(qi0)
        elif jtype[i] == 2:
            qi0 = q[q_base[i]]
            Tb[i] = Ti[i] if qi0 == 0 else Ti[i] @ T_trans([0, 0, qi0])
        elif jtype[i] == 3:
            qb = q_base[i]
            T_local = np.eye(4)
            T_local[:3, :3] = expmap_so3(q[qb+3:qb+6])
            T_local[:3, 3]  = q[qb:qb+3]
            Tb[i] = Ti[i] @ T_local
        else:
            raise ValueError(f"unknown jtype {jtype[i]}")

    for i in range(nb):
        if parent[i] is None: T[i] = Tb[i]
        else:                  T[i] = T[parent[i]] @ Tb[i]
    return T

def jacob_whitney(T, _T, parent, jtype, idx):
    """Spatial Jacobian (6 × nv) at the world-frame point _T on body idx.
    Top 3 rows = world linear velocity columns; bottom 3 = world angular.
    Columns are v-indexed (Jacobian maps qd → spatial velocity). For jtype=3
    (free) body, fills a 6×6 block at v_base[idx]:v_base[idx]+6 mapping
    qd=[v_body; ω_body] to the contact point's spatial velocity."""
    _, v_base, _, _, _, nv = _build_qidx(jtype)
    J = np.zeros((6, nv))
    i = idx
    while True:
        vb = v_base[i]
        if jtype[i] == 1:
            J[0:3, vb] = np.cross(T[i, 0:3, 2], _T[0:3, 3] - T[i, 0:3, 3])
            J[3:6, vb] = T[i, 0:3, 2]
        elif jtype[i] == 2:
            J[0:3, vb] = T[i, 0:3, 2]
        elif jtype[i] == 3:
            R = T[i, 0:3, 0:3]
            r = _T[0:3, 3] - T[i, 0:3, 3]
            skew_r = np.array([[0, -r[2], r[1]],
                               [r[2], 0, -r[0]],
                               [-r[1], r[0], 0]])
            # v_body cols → world linear = R, world angular = 0
            J[0:3, vb:vb+3] = R
            # ω_body cols → world linear = -skew(r)·R (= r × (R·êk) per col), angular = R
            J[0:3, vb+3:vb+6] = -skew_r @ R
            J[3:6, vb+3:vb+6] = R
        i = parent[i]
        if i is None: break
    return J

def gravity_lagrange(T, m, c, parent, jtype, gv=[0, 0, -9.81]):
    nb = len(T)
    _, _, _, _, _, nv = _build_qidx(jtype)   # bias/gravity is v-indexed
    gv = np.array(gv)
    g = np.zeros(nv)

    B = np.zeros((nb, 4, 4))
    for i in range(nb): B[i] = T[i] @ T_trans(c[i])

    Jc = np.zeros((nb, 6, nv))
    for i in range(nb):
        Jc[i] = jacob_whitney(T, B[i], parent, jtype, i)

    for i in range(nv):
        for j in range(nb):
            g[i] += m[j] * gv @ Jc[j, 0:3, i]
    return g

def inertia_lagrange(T, m, c, I, parent, jtype):
    nb = len(T)
    _, _, _, _, _, nv = _build_qidx(jtype)   # mass matrix is nv × nv
    B = np.zeros((nb, 4, 4))
    for i in range(nb): B[i] = T[i] @ T_trans(c[i])

    Jc = np.zeros((nb, 6, nv))
    for i in range(nb):
        Jc[i] = jacob_whitney(T, B[i], parent, jtype, i)

    M = np.zeros((nv, nv))
    for i in range(nb):
        M += m[i] * Jc[i, 0:3, :].T @ Jc[i, 0:3, :] + Jc[i, 3:6, :].T @ B[i, 0:3, 0:3].T @ I[i] @ B[i, 0:3, 0:3] @ Jc[i, 3:6, :]
    return M

def com_lagrange(T, m, c):
    """Total CoM position in world frame. Σ m_i · (T_i · c_i) / Σ m_i."""
    nb = len(T)
    mtot = float(sum(m))
    r = np.zeros(3)
    for i in range(nb):
        r += m[i] * (T[i] @ np.array([c[i][0], c[i][1], c[i][2], 1.0]))[:3]
    return r / mtot

def com_jacob_lagrange(T, m, c, parent, jtype):
    """CoM linear Jacobian (3 × nv): v_com_world = J_com · qd.
    Mass-weighted average of per-body CoM linear Jacobians (top 3 rows of
    Whitney at B_i = T_i · T_trans(c_i))."""
    nb = len(T)
    _, _, _, _, _, nv = _build_qidx(jtype)
    mtot = float(sum(m))
    Jc = np.zeros((3, nv))
    for i in range(nb):
        Bi = T[i] @ T_trans(c[i])
        Ji = jacob_whitney(T, Bi, parent, jtype, i)
        Jc += m[i] * Ji[:3, :]
    return Jc / mtot

def com_inertia(T, m, c, I):
    """3×3 rotational inertia about the total CoM, expressed in world frame
    (rotational block of CCRBI, a.k.a. centroidal rotational inertia I_G).
    Each body's inertia is rotated to world, then parallel-axis-translated to CoM."""
    nb = len(T)
    r_com = com_lagrange(T, m, c)
    Ig = np.zeros((3, 3))
    for i in range(nb):
        R_i = T[i, 0:3, 0:3]
        r_i_world = (T[i] @ np.array([c[i][0], c[i][1], c[i][2], 1.0]))[:3]
        I_world = R_i @ I[i] @ R_i.T
        d = r_i_world - r_com
        Ig += I_world + m[i] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return Ig

'''
# Coriolis/centripetal matrix C(q, qd) via finite-difference of M(q).
# Disabled: only caller was Model.bias4 (also disabled). For jtype=3 (free)
# the q+ε·eᵢ perturbation is invalid on SO(3) — would need SO(3)-aware
# finite-diff (~20 lines) to revive.
def cc_finitediff(Ti, m, c, I, parent, jtype, q, q_dot):
    if any(jt == 3 for jt in jtype):
        raise NotImplementedError("cc_finitediff: jtype=3 (free) not supported")
    nb = len(Ti)
    T = _fk(Ti, parent, jtype, q)
    M0 = inertia_lagrange(T, m, c, I, parent, jtype)
    D = np.zeros((nb, nb, nb))
    C = np.zeros((nb, nb))
    dd = 0.00001

    for i in range(nb):
        dq = np.zeros(nb)
        dq[i] = dd
        T = _fk(Ti, parent, jtype, q + dq)
        D[i] = (inertia_lagrange(T, m, c, I, parent, jtype) - M0)/dd

    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                C[i][j] += 0.5*(D[k][i][j] + D[j][i][k] - D[i][j][k])*q_dot[k]
    return C
'''

def crb_featherstone(X, I, parent, jtype, q):
    """Composite-rigid-body mass matrix. Returns H of shape (nv, nv) — the
    joint-space mass matrix lives in velocity space. Supports mixed 1-DoF and
    free (jtype=3) joints."""
    nb = len(X)
    q_base, v_base, nq_pb, nv_pb, _, nv = _build_qidx(jtype)
    Ic = np.zeros((nb, 6, 6))
    Xup = np.zeros((nb, 6, 6))
    S = [None] * nb
    H = np.zeros((nv, nv))

    for i in range(nb):
        if jtype[i] == 3:
            qi = q[q_base[i]:q_base[i]+nq_pb[i]]
            Xj, Si = jcalc6(qi)
        elif jtype[i] == 0:
            Xj = np.eye(6)
            Si = np.zeros((6, 0))                  # no DoF columns
        else:
            qi0 = q[q_base[i]]
            Xj, Sv = jcalc(jtype[i], qi0)
            Si = Sv.reshape(6, 1)
        S[i] = Si
        Xup[i] = Xj @ X[i]
        Ic[i] = I[i]

    for i in range(nb-1, -1, -1):
        if parent[i] is not None:
            Ic[parent[i]] += Xup[i].T @ Ic[i] @ Xup[i]

    for i in range(nb):
        vb_i = v_base[i]; nvi = nv_pb[i]
        fh = Ic[i] @ S[i]                                  # 6 × nvi
        H[vb_i:vb_i+nvi, vb_i:vb_i+nvi] = S[i].T @ fh      # nvi × nvi
        j = i
        while parent[j] is not None:
            fh = Xup[j].T @ fh
            j = parent[j]
            vb_j = v_base[j]; nvj = nv_pb[j]
            block = S[j].T @ fh                             # nvj × nvi
            H[vb_j:vb_j+nvj, vb_i:vb_i+nvi] = block
            H[vb_i:vb_i+nvi, vb_j:vb_j+nvj] = block.T
    return H

#inverse-dynamics algorithm using recursive Newton-Euler method. If qdd = 0, results in bias force
def rne_featherstone(X, I, parent, jtype, q, qd, qdd, f_ext, g, full=False):
    """Inverse dynamics. Returns tau of shape (nv,) — joint forces live in
    velocity space. q is nq-length, qd/qdd are nv-length."""
    nb = len(X)
    q_base, v_base, nq_pb, nv_pb, _, nv = _build_qidx(jtype)
    Xup = np.zeros((nb, 6, 6))
    S = [None] * nb
    v = np.zeros((nb, 6))
    a = np.zeros((nb, 6))
    f = np.zeros((nb, 6))
    tau = np.zeros(nv)
    a_grav = np.array([0, 0, 0, g[0], g[1], g[2]])

    for i in range(nb):
        if jtype[i] == 3:
            qi   = q [q_base[i] : q_base[i] + nq_pb[i]]
            qdi  = qd[v_base[i] : v_base[i] + nv_pb[i]]
            qddi = qdd[v_base[i] : v_base[i] + nv_pb[i]]
            Xj, Si = jcalc6(qi)
            vJ = Si @ qdi
            aS = Si @ qddi
        elif jtype[i] == 0:
            Xj = np.eye(6)
            Si = np.zeros((6, 0))                  # no DoF columns
            vJ = np.zeros(6)
            aS = np.zeros(6)
        else:
            qi0  = q [q_base[i]]
            qdi0 = qd[v_base[i]]
            qddi0 = qdd[v_base[i]]
            Xj, Sv = jcalc(jtype[i], qi0)
            Si = Sv.reshape(6, 1)
            vJ = Sv * qdi0
            aS = Sv * qddi0
        S[i] = Si
        Xup[i] = Xj @ X[i]

        if parent[i] is None:
            v[i] = vJ
            a[i] = Xup[i] @ -a_grav + aS
        else:
            v[i] = Xup[i] @ v[parent[i]] + vJ
            a[i] = Xup[i] @ a[parent[i]] + crm(v[i]) @ vJ + aS

        f[i] = I[i] @ a[i] + crf(v[i]) @ I[i] @ v[i]
        if f_ext is not None: f[i] -= f_ext[i]

    for i in range(nb-1, -1, -1):
        vb = v_base[i]; nvi = nv_pb[i]
        tau[vb:vb+nvi] = S[i].T @ f[i]
        if parent[i] is not None:
            f[parent[i]] += Xup[i].T @ f[i]

    if full == False: return tau
    _a = a.copy()
    for i in range(nb): _a[i, 3:] = a[i, 3:] + np.cross(v[i, :3], v[i, 3:])
    return tau, f, _a, v

def aba_featherstone(X, I, parent, jtype, q, qd, tau, f_ext, g, ff=None, sk=None, armature=None, dt=None, Kp_j=None, Kd_j=None, q_ref=None, qd_ref=None, full=False):
    """Articulated-body forward dynamics. Returns qdd of length nv — joint
    accelerations live in velocity space (same as qd, tau, ff, sk, armature, Kp_j, Kd_j).
    `armature` (per-DoF rotor/reflected inertia, MuJoCo-style) is added to the diagonal
    of each joint's articulated inertia `d` — a true inertia (no dt factor), unlike the
    semi-implicit damping/spring/PD terms. None = 0 (bit-identical to pre-armature).
    q is nq-length. Supports mixed 1-DoF and free (jtype=3) joints; per-body S
    storage handles variable nv_per_body uniformly via the matmul/solve path.
    Implicit PD currently 1-DoF only (free joint PD deferred)."""
    nb = len(X)
    q_base, v_base, nq_pb, nv_pb, _, nv = _build_qidx(jtype)

    Xup = np.zeros((nb, 6, 6))
    S = [None] * nb     # 6 × nv_pb[i]
    v = np.zeros((nb, 6))
    c = np.zeros((nb, 6))
    IA = np.zeros((nb, 6, 6))
    pA = np.zeros((nb, 6))
    U = [None] * nb     # 6 × nv_pb[i]
    d = [None] * nb     # nv_pb[i] × nv_pb[i]
    u = [None] * nb     # nv_pb[i]
    a = np.zeros((nb, 6))
    qdd = np.zeros(nv)
    a_grav = np.array([0, 0, 0, g[0], g[1], g[2]])

    # ---- outward pass: spatial velocities and bias forces ----
    for i in range(nb):
        if jtype[i] == 3:
            qi  = q [q_base[i] : q_base[i] + nq_pb[i]]
            qdi = qd[v_base[i] : v_base[i] + nv_pb[i]]
            Xj, Si = jcalc6(qi)              # Si 6×6
            vJ = Si @ qdi
        elif jtype[i] == 0:
            Xj = np.eye(6)
            Si = np.zeros((6, 0))            # no DoF columns
            vJ = np.zeros(6)
        else:
            qi0  = q [q_base[i]]
            qdi0 = qd[v_base[i]]
            Xj, Sv = jcalc(jtype[i], qi0)
            Si = Sv.reshape(6, 1)            # promote 6-vec → 6×1
            vJ = Sv * qdi0
        S[i] = Si
        Xup[i] = Xj @ X[i]

        if parent[i] is None:
            v[i] = vJ
            c[i] = np.zeros(6)
        else:
            v[i] = Xup[i] @ v[parent[i]] + vJ
            c[i] = crm(v[i]) @ vJ

        IA[i] = I[i]
        pA[i] = crf(v[i]) @ I[i] @ v[i] - f_ext[i]

    # ---- inward pass: articulated body inertia and bias ----
    for i in range(nb-1, -1, -1):
        Si = S[i]
        vb = v_base[i]; nvi = nv_pb[i]
        Ui = IA[i] @ Si                       # 6 × nvi
        di = Si.T @ Ui                        # nvi × nvi
        if armature is not None and nvi > 0:  # rotor/reflected inertia on the diagonal
            for k in range(nvi):
                di[k, k] += armature[vb+k]
        ui = tau[vb:vb+nvi] - Si.T @ pA[i]    # nvi (tau is v-indexed)
        U[i] = Ui; d[i] = di; u[i] = ui

        # semi-implicit damping + spring + joint-space PD (1-DoF joints only —
        # free joint PD is intentionally deferred; ff/sk/Kp/Kd arrays are
        # v-indexed but here scalar since nvi=1).
        pd_on = (q_ref is not None) or (qd_ref is not None)
        any_imp = (ff is not None or sk is not None or pd_on)
        if any_imp and dt is not None and jtype[i] in (1, 2):
            qb = q_base[i]   # nv_pb=nq_pb=1 here; q and v indices coincide
            ff_i  = 0.0 if ff is None else ff[vb]
            sk_i  = 0.0 if sk is None else sk[vb]
            if pd_on and Kp_j is not None and q_ref is not None:
                Kp_i = Kp_j[vb]; qr_i = q_ref[qb]
            else:
                Kp_i = 0.0;     qr_i = 0.0
            if pd_on and Kd_j is not None:
                Kd_i = Kd_j[vb]; qdr_i = 0.0 if qd_ref is None else qd_ref[vb]
            else:
                Kd_i = 0.0;     qdr_i = 0.0
            di[0, 0] += (ff_i + Kd_i) * dt + (sk_i + Kp_i) * dt * dt
            ui[0] += (- ff_i * qd[vb] - sk_i * q[qb]
                      - Kp_i * (q[qb] - qr_i) - Kp_i * dt * qd[vb]
                      - Kd_i * (qd[vb] - qdr_i))

        if parent[i] is not None:
            if jtype[i] > 0:
                Y = np.linalg.solve(di, Ui.T)        # nvi × 6
                Ia = IA[i] - Ui @ Y
                pa = pA[i] + Ia @ c[i] + Ui @ np.linalg.solve(di, ui)
            else:
                Ia = IA[i]
                pa = pA[i] + Ia @ c[i]

            IA[parent[i]] = IA[parent[i]] + Xup[i].T @ Ia @ Xup[i]
            pA[parent[i]] = pA[parent[i]] + Xup[i].T @ pa

    # ---- outward pass: accelerations ----
    for i in range(nb):
        if parent[i] is None: a[i] = Xup[i] @ -a_grav + c[i]
        else: a[i] = Xup[i] @ a[parent[i]] + c[i]

        vb = v_base[i]; nvi = nv_pb[i]
        if jtype[i] > 0:
            qdd_i = np.linalg.solve(d[i], u[i] - U[i].T @ a[i])
            qdd[vb:vb+nvi] = qdd_i
            a[i] += S[i] @ qdd_i

    if full == False: return qdd
    f = np.zeros((nb, 6))
    _a = a.copy()
    for i in range(nb):
        f[i] = IA[i] @ a[i] + pA[i]
        _a[i, 3:] = a[i, 3:] + np.cross(v[i, :3], v[i, 3:])
    return qdd, f, _a, v

#inverse dynamics algorithm using recursive Newton-Euler method (Craig form).
#Supports 1-DoF joints (jtype 0/1/2) and free axis-angle joint (jtype 3).
#For jtype=3: q[qb:qb+6] = [p_world, rotation_vec], qd/qdd[vb:vb+6] = [v_body, ω_body].
def rne_lwp(Ti, m, c, I, parent, jtype, q, q_dot, q_ddot, f_ext, g, full=False):
    nb = len(Ti)
    q_base, v_base, _, _, _, nv = _build_qidx(jtype)
    Tb = np.zeros((nb, 4, 4))
    Z = np.array([0, 0, 1.])

    w = np.zeros((nb, 3))
    w_dot = np.zeros((nb, 3))
    v_dot = np.zeros((nb, 3))
    vc_dot = np.zeros((nb, 3))
    if full: v = np.zeros((nb, 3))

    F = np.zeros((nb, 3))
    N = np.zeros((nb, 3))

    f = np.zeros((nb, 3))
    n = np.zeros((nb, 3))
    tau = np.zeros(nv)

    #update Ti according to q
    for i in range(nb):
        qbi = q_base[i]
        if   jtype[i] == 0: Tb[i] = Ti[i].copy()
        elif jtype[i] == 1: Tb[i] = Ti[i] @ T_rot_z(q[qbi])
        elif jtype[i] == 2: Tb[i] = Ti[i] @ T_trans([0, 0, q[qbi]])
        elif jtype[i] == 3:
            T_local = np.eye(4)
            T_local[:3, :3] = expmap_so3(q[qbi+3:qbi+6])
            T_local[:3, 3]  = q[qbi:qbi+3]
            Tb[i] = Ti[i] @ T_local

    #outward iteration
    for i in range(nb):
        RiT = Tb[i, :3, :3].T
        pid = parent[i]
        p = Tb[i, :3, 3]
        vbi = v_base[i]

        if pid == None:  # base (parent=world)
            if jtype[i] == 0:
                v_dot[i] = RiT @ -g
            elif jtype[i] == 1:
                w[i] = q_dot[vbi] * Z
                w_dot[i] = q_ddot[vbi] * Z
                v_dot[i] = RiT @ -g
            elif jtype[i] == 2:
                v_dot[i] = -RiT @ g + q_ddot[vbi]*Z
                if full: v[i] = q_dot[vbi]*Z
            elif jtype[i] == 3:
                # free joint at world root: ω_body and v_body straight from qd[vb+3:vb+6] / qd[vb:vb+3].
                v_body = q_dot[vbi:vbi+3]
                w[i] = q_dot[vbi+3:vbi+6].copy()
                w_dot[i] = q_ddot[vbi+3:vbi+6].copy()
                # body-frame inertial accel of origin = a_body + ω×v_body, minus R^T·g
                v_dot[i] = -RiT @ g + q_ddot[vbi:vbi+3] + np.cross(w[i], v_body)
                if full: v[i] = v_body.copy()

        elif jtype[i] == 0:
            w[i] = RiT @ w[pid]
            w_dot[i] = RiT @ w_dot[pid]
            v_dot[i] = RiT @ (np.cross(w_dot[pid], p) + np.cross(w[pid], np.cross(w[pid], p)) + v_dot[pid])
            if full: v[i] = RiT @ (np.cross(w[pid], p) + v[pid])

        elif jtype[i] == 1:
            w[i] = RiT @ w[pid] + q_dot[vbi]*Z
            w_dot[i] = RiT @ w_dot[pid] + np.cross(RiT @ w[pid], q_dot[vbi]*Z) + q_ddot[vbi]*Z
            v_dot[i] = RiT @ (np.cross(w_dot[pid], p) + np.cross(w[pid], np.cross(w[pid], p)) + v_dot[pid])
            if full: v[i] = RiT @ (np.cross(w[pid], p) + v[pid])

        elif jtype[i] == 2:
            w[i] = RiT @ w[pid]
            w_dot[i] = RiT @ w_dot[pid]
            v_dot[i] = RiT @ (np.cross(w_dot[pid], p) + np.cross(w[pid], np.cross(w[pid], p)) + v_dot[pid]) + q_ddot[vbi]*Z + 2.0*np.cross(w[i], q_dot[vbi]*Z)
            if full: v[i] = RiT @ (np.cross(w[pid], p) + v[pid]) + q_dot[vbi]*Z

        elif jtype[i] == 3:
            # free joint with non-world parent (rare). Add 6-DoF joint motion to propagated parent terms.
            v_body = q_dot[vbi:vbi+3]
            a_body = q_ddot[vbi:vbi+3]
            ωb = q_dot[vbi+3:vbi+6]
            αb = q_ddot[vbi+3:vbi+6]
            w[i]     = RiT @ w[pid] + ωb
            w_dot[i] = RiT @ w_dot[pid] + np.cross(RiT @ w[pid], ωb) + αb
            v_dot[i] = (RiT @ (np.cross(w_dot[pid], p) + np.cross(w[pid], np.cross(w[pid], p)) + v_dot[pid])
                        + a_body + np.cross(w[i], v_body))
            if full: v[i] = RiT @ (np.cross(w[pid], p) + v[pid]) + v_body

        vc_dot[i] = np.cross(w_dot[i], c[i]) + np.cross(w[i], np.cross(w[i], c[i])) + v_dot[i]
        F[i] = m[i] * vc_dot[i]
        N[i] = I[i] @ w_dot[i] + np.cross(w[i], I[i] @ w[i])

        if f_ext is not None:
            F[i] += -f_ext[i][3:]
            N[i] -= f_ext[i][:3] + np.cross(-c[i], f_ext[i][3:])

    #inward iteration
    for i in range(nb-1, -1, -1):
        f[i] = F[i].copy()
        n[i] = N[i] + np.cross(c[i], F[i])

        cv = [j for j in range(nb) if parent[j] == i]
        for cid in cv:
            f[i] += Tb[cid, :3, :3] @ f[cid]
            n[i] += Tb[cid, :3, :3] @ n[cid] + np.cross(Tb[cid, :3, 3], Tb[cid, :3, :3] @ f[cid])

        vbi = v_base[i]
        if   jtype[i] == 1: tau[vbi] = n[i].dot(Z)
        elif jtype[i] == 2: tau[vbi] = f[i].dot(Z)
        elif jtype[i] == 3:
            # free joint: qd[vb:vb+3]=v_body → joint force = f[i] (body-frame); qd[vb+3:vb+6]=ω_body → moment = n[i]
            tau[vbi:vbi+3]   = f[i]
            tau[vbi+3:vbi+6] = n[i]

    if full==False: return tau
    _f = np.zeros((nb, 6))
    _a = np.zeros((nb, 6))
    _v = np.zeros((nb, 6))
    for i in range(nb):
        _f[i] = np.concatenate((n[i], f[i]))
        _a[i] = np.concatenate((w_dot[i], v_dot[i]))
        _v[i] = np.concatenate((w[i], v[i]))
    return tau, _f, _a, _v

#floating body inverse kinematics
def fbik(X, v, a=None):
    q = np.zeros(6)
    qd = np.zeros(6)
    qdd = np.zeros(6)
    
    E = X[0:3, 0:3]
    rx = -E.T @ X[3:6, 0:3]
    r = np.array([rx[2, 1], rx[0, 2], rx[1, 0]])

    q[0:3] = r
    q[4] = np.arctan2(E[2, 0], np.sqrt(E[0, 0]*E[0, 0] + E[1, 0]*E[1, 0]))
    q[5] = np.arctan2(-E[1, 0], E[0, 0])

    if E[2, 0] > 0: q[3] = np.arctan2(E[1, 2] + E[0, 1], E[1, 1] - E[0, 2]) - q[5]
    else: q[3] = np.arctan2(E[1, 2] - E[0, 1], E[1, 1] + E[0, 2]) + q[5]

    if q[3] > np.pi: q[3] = q[3] - 2.0*np.pi
    elif q[3] < -np.pi: q[3] + 2.0*np.pi

    c4 = np.cos(q[3]); s4 = np.sin(q[3])
    c5 = np.cos(q[4]); s5 = np.sin(q[4])
    S = np.array([[1, 0, s5], [0, c4, -s4*c5], [0, s4, c4*c5]])

    omega = v[0:3]
    rd = v[3:6] - np.cross(r, omega)

    qd[0:3] = rd
    qd[3:6] = np.linalg.solve(S, omega)
    if a == None: return q, qd
    
    c4d = -s4*qd[3]; s4d = c4*qd[3]
    c5d = -s5*qd[4]; s5d = c5*qd[4]
    Sd = [[0, 0, s5d], [0, c4d, -s4d*c5 - s4*c5d], [0, s4d, c4d*c5 + c4*c5d]]

    omegad = a[0:3]
    rdd = a[3:6] - np.cross(rd, omega) - np.cross(r, omegad)

    qdd[0:3] = rdd
    qdd[3:6] = np.linalg.solve(S, omegad - Sd*qd[3:6])
    return q, qd, qdd
    
def _q_step(q, qd, dt, jtype, q_base):
    """Advance q by qd over dt, respecting per-jtype manifold structure.
    1-DoF joints get the plain linear update q_i += qd_i·dt. free joints
    (jtype=3) integrate translation in world frame via the body-frame v
    (p_next = p + R(w)·v·dt) and rotation via SO(3) exp-map composition
    (w_next = log(exp(w)·exp(ω·dt)))."""
    q_next = q + qd * dt
    for i in range(len(jtype)):
        if jtype[i] == 3:
            qb = q_base[i]
            w_old = q[qb+3:qb+6]
            R = expmap_so3(w_old)
            q_next[qb:qb+3]   = q[qb:qb+3] + R @ qd[qb:qb+3] * dt
            q_next[qb+3:qb+6] = integrate_so3(w_old, qd[qb+3:qb+6], dt)
    return q_next

def _q_blend_free(q, q_next_linavg, qd_next, dt, jtype, q_base):
    """For RK4: linear weighted average is undefined on SO(3). Overwrite the
    free slots of an already-computed linear-average q_next with a single-shot
    SO(3) integration using the averaged qd_next. Non-free slots stay as-is."""
    if not any(jt == 3 for jt in jtype):
        return q_next_linavg
    q_next = q_next_linavg.copy()
    for i in range(len(jtype)):
        if jtype[i] == 3:
            qb = q_base[i]
            w_old = q[qb+3:qb+6]
            R = expmap_so3(w_old)
            q_next[qb:qb+3]   = q[qb:qb+3] + R @ qd_next[qb:qb+3] * dt
            q_next[qb+3:qb+6] = integrate_so3(w_old, qd_next[qb+3:qb+6], dt)
    return q_next

def euler_step(X, I6, parent, jtype, q, qd, tau, f_ext, g, dt):
    qdd, f, a, v = aba_featherstone(X, I6, parent, jtype, q, qd, tau, f_ext, g, full=True)
    qd_next = qd + qdd * dt
    q_base, _, _, _, _, _ = _build_qidx(jtype)
    q_next = _q_step(q, qd_next, dt, jtype, q_base)
    return q_next, qd_next, qdd, f, a, v

def euler_step2(Ti, m, c, I, parent, jtype, q, qd, tau, f_ext, g, dt):
    T = _fk(Ti, parent, jtype, q)
    M = inertia_lagrange(T, m, c, I, parent, jtype)
    b = rne_lwp(Ti, m, c, I, parent, jtype, q, qd, np.zeros(len(q)), f_ext, g)
    qdd = np.linalg.pinv(M) @ (tau - b)
    _, f, a, v = rne_lwp(Ti, m, c, I, parent, jtype, q, qd, qdd, f_ext, g, full=True)
    qd_next = qd + qdd * dt
    q_base, _, _, _, _, _ = _build_qidx(jtype)
    q_next = _q_step(q, qd_next, dt, jtype, q_base)
    return q_next, qd_next, qdd, f, a, v

def rk4_step(X, I6, parent, jtype, q, qd, tau, f_ext, g, dt):
    q_base, _, _, _, _, _ = _build_qidx(jtype)

    qdd1 = aba_featherstone(X, I6, parent, jtype, q, qd, tau, f_ext, g)
    qd_next1 = qd + qdd1 * dt
    q_next1 = _q_step(q, qd_next1, dt, jtype, q_base)

    qdd2 = aba_featherstone(X, I6, parent, jtype, q_next1, qd_next1, tau, f_ext, g)
    qd_next2 = qd + qdd2 * 0.5*dt
    q_next2 = _q_step(q, qd_next2, 0.5*dt, jtype, q_base)

    qdd3 = aba_featherstone(X, I6, parent, jtype, q_next2, qd_next2, tau, f_ext, g)
    qd_next3 = qd + qdd3 * 0.5*dt
    q_next3 = _q_step(q, qd_next3, 0.5*dt, jtype, q_base)

    qdd4 = aba_featherstone(X, I6, parent, jtype, q_next3, qd_next3, tau, f_ext, g)
    qd_next4 = qd + qdd4 * dt
    q_next4 = _q_step(q, qd_next4, dt, jtype, q_base)

    q_next_linavg = (q_next1 + 2.0*q_next2 + 2.0*q_next3 + q_next4)/6.0
    qd_next = (qd_next1 + 2.0*qd_next2 + 2.0*qd_next3 + qd_next4)/6.0
    # SO(3) cannot accept linear-average pose; overwrite free slots with a
    # single-shot integration using qd_next over full dt. Non-free slots stay
    # bit-identical to the historical linear average.
    q_next = _q_blend_free(q, q_next_linavg, qd_next, dt, jtype, q_base)

    qdd = (qdd1 + 2.0*qdd2 + 2.0*qdd3 + qdd4)/6.0
    _, f, a, v = rne_featherstone(X, I6, parent, jtype, q, qd, qdd, f_ext, g, full=True)
    return q_next, qd_next, qdd, f, a, v

#simplest sphere-ground contact
def contact_ground_sphere(T, parent, jtype, ctype, cbody, ctran, cshape, cparam, qd):
    """Minimal spring-damper ground contact for SPHERE shapes only — a test/reference
    solver (selected by `solver: minimal` in YAML). The ground is the z=0 plane
    (the cpair list and floor geometry are ignored); each sphere whose center dips
    below its radius gets a penalty normal force (k_n·overlap − d_n·v_n) plus simple
    viscous tangential damping. Returns the per-body spatial wrench f_ext
    ([moment, force], body frame) to feed straight into aba_featherstone. Not for
    production — no Coulomb cone, no box/mesh, explicit (needs a small dt)."""
    f_ext = np.zeros((len(T), 6))

    for i in range(len(cbody)):
        if cbody[i] < 0: continue #<------exclude root
        if ctype[i] != 102: continue # sphere only
        
        Tb = T[cbody[i]] #body reference frame
        Tc = T[cbody[i]] @ ctran[i] #contact sphere frame
        z = Tc[2][3]
        rad = cshape[i][0]

        if z - rad > 0: continue        
        overlap = rad - z
        
        #contact point frame in global
        Tp = copy.copy(Tc)
        #Tp[2][2] = 0
        Tp[2][3] = 0
        
        J = jacob_whitney(T, Tp, parent, jtype, cbody[i])
        cv = J @ qd
        
        #contact force in global frame (cparam stride=12: [pair_id, k_n, d_n, k_t, d_t, mu, ...])
        fz =  cparam[i][1] * overlap - cparam[i][2] * cv[2]
        fx = -cparam[i][4] * cv[0]
        fy = -cparam[i][4] * cv[1]

        #fz =  50000 * overlap - 100*cv[2]
        #fx = -50*cv[0]
        #fy = -50*cv[1]

        #fx = 0.05*fz*np.tanh(-100*cv[0])
        #fy = 0.05*fz*np.tanh(-100*cv[1])

        cf = np.array([fx, fy, fz]) #contact force w.r.t global frame

        #calculate contact force and moment in the body frame
        ef = np.transpose(Tb[:3, :3]) @ cf  #contact force in body frame
        em = np.cross((np.linalg.inv(Tb) @ Tp[:, 3])[:3], ef) #contact moment wrt body frame
        f_ext[cbody[i]] += np.concatenate((em, ef))

    return f_ext

def contact_lcp(T, parent, jtype, cpair, ctype, cbody, ctran, cshape, cparam,
                qd_free, M, dt,
                erp=0.2, slop=1e-4, cfm_scale=1e-6, v_rest_thresh=3e-2, iters=20, tol=1e-6, cff=False,
                lam_prev=None, floss=None, lam_fric_prev=None,
                q=None, jnt_lo=None, jnt_hi=None, lam_limit_prev=None):
    """
    Soft-constrained LCP contact solver with Coulomb friction (Stewart-Trinkle /
    Anitescu time-stepping with disk projection for the friction cone).

    Per contact k we stack 6 rows in J (3 linear + 3 angular), solve
        qd_next = qd_free + M^{-1} J^T λ
    with PGS and four independent cone projections sharing λ_n as budget:
        λ_n,k ≥ 0                                      (normal, Baumgarte stabilized)
        ‖(λ_t1,k, λ_t2,k)‖   ≤ μ_k      · λ_n,k        (linear friction disk)
        |λ_spin,k|           ≤ μ_spin,k · λ_n,k        (torsional friction scalar)
        ‖(λ_r1,k,  λ_r2,k )‖ ≤ μ_roll,k · λ_n,k        (rolling friction disk)
    A = J M^{-1} J^T + diag(R) is symmetric PSD (CFM regularized).

    Tangent frame at each contact comes from choose_rotation(n_hat), which fixes
    the tangent directions (and hence the spin/roll axes) deterministically from
    the contact normal.

    Inputs
      T, parent, jtype           -- kinematics (T from _fk at current q)
      cpair, ctype, cbody, ctran, cshape, cparam
                                 -- per-shape collision arrays (from the Model)
      qd_free   (nq,)            -- free predictor velocity (no contact)
      M         (nq, nq)         -- joint-space mass matrix at q (from crb_featherstone)
      dt                         -- timestep
      erp                        -- Baumgarte coefficient on penetration (0..1)
      slop                       -- penetration deadband (m); avoid jitter for tiny depths
      cfm_scale                  -- regularization scale; maps k_n/k_t,d_n/d_t → R diagonal
      iters, tol                 -- PGS budget and convergence threshold on max |Δλ|
      cff                        -- if True, also return per-contact force list (for viz/log)
      lam_prev (npair*6,)        -- λ from the previous step indexed by cpair index
                                    (zeros for inactive pairs). If provided, PGS
                                    initializes from these values (warm-start), which
                                    typically cuts PGS iterations needed and stabilizes
                                    switching-contact configurations. Default None = cold.

    Returns
      dqd       (nq,)      -- velocity correction to add to qd_free
      lam       (6*nc,)    -- impulse per *active* contact, packed densely. Layout
                              [n, t1, t2, spin, r1, r2,  n, t1, t2, spin, r1, r2, ...]
      info      dict       -- {nc, iters, residual, points, normals, depths, R_tan,
                              lam_n, lam_t1, lam_t2, lam_spin, lam_r1, lam_r2,
                              lam_full, cpair_idx}
                              lam_full is npair*6, indexed by cpair position — pass back
                              as lam_prev next step to warm-start.
      f_ext     (nb, 6)    -- equivalent body-frame spatial wrench per body
                              (format [moment, force])
      (cfs)                -- if cff: list of [px, py, pz, fx, fy, fz, i, j] (world)
    """
    nq = M.shape[0]
    npair = cpair.shape[0]

    # ---- Pass 1: collision detection → active contact set ----------------
    # Per contact we cache: bodies, contact point, tangent frame R_tan (columns
    # = [t1, t2, n_hat]), depth, and material params:
    #   normal CFM:    k_n, d_n        (cparam idx 1,2)
    #   linear tangent CFM + cone:  k_t, d_t, mu       (idx 3,4,5)
    #   spin CFM + cone:            k_spin, d_spin, mu_spin   (idx 6,7,8)
    #   roll CFM + cone:            k_roll, d_roll, mu_roll   (idx 9,10,11)
    cdata = []
    for n in range(npair):
        i = cpair[n, 0]; j = cpair[n, 1]
        T1 = ctran[i] if cbody[i] < 0 else T[cbody[i]] @ ctran[i]
        T2 = ctran[j] if cbody[j] < 0 else T[cbody[j]] @ ctran[j]
        param1 = np.concatenate((homogeneous_to_xyzeuler(T1), cshape[i]))
        param2 = np.concatenate((homogeneous_to_xyzeuler(T2), cshape[j]))
        _p1  = (ctypes.c_double*len(param1))(*param1)
        _p2  = (ctypes.c_double*len(param2))(*param2)
        # Multi-point narrowphase: up to MAX_PTS_PER_PAIR points per cpair (box-box
        # face manifold returns up to 4; other type combos return 1). 7 doubles
        # per point: [pos_x, pos_y, pos_z, n_x, n_y, n_z, depth].
        _out_buf = (ctypes.c_double * (7 * MAX_PTS_PER_PAIR))(*np.zeros(7 * MAX_PTS_PER_PAIR))
        npts = clib.collision_check(ctype[i], _p1, ctype[j], _p2, _out_buf, MAX_PTS_PER_PAIR)
        if npts <= 0:
            continue
        out_all = np.array(_out_buf).reshape(MAX_PTS_PER_PAIR, 7)
        for s in range(npts):
            out = out_all[s]
            depth = out[6]
            dvec  = out[3:6]
            dnorm = np.linalg.norm(dvec)
            if dnorm < 1e-4:
                continue
            R_tan = choose_rotation(dvec)           # columns: [t1, t2, n_hat]
            cdata.append((
                i, j,
                out[0:3].copy(),                    # 2: contact point (world)
                R_tan,                              # 3: tangent frame
                float(depth),                       # 4: depth
                0.5 * (cparam[i][1]  + cparam[j][1]),   # 5:  k_n
                0.5 * (cparam[i][2]  + cparam[j][2]),   # 6:  d_n
                0.5 * (cparam[i][3]  + cparam[j][3]),   # 7:  k_t
                0.5 * (cparam[i][4]  + cparam[j][4]),   # 8:  d_t
                0.5 * (cparam[i][5]  + cparam[j][5]),   # 9:  mu
                0.5 * (cparam[i][6]  + cparam[j][6]),   # 10: k_spin
                0.5 * (cparam[i][7]  + cparam[j][7]),   # 11: d_spin
                0.5 * (cparam[i][8]  + cparam[j][8]),   # 12: mu_spin
                0.5 * (cparam[i][9]  + cparam[j][9]),   # 13: k_roll
                0.5 * (cparam[i][10] + cparam[j][10]),  # 14: d_roll
                0.5 * (cparam[i][11] + cparam[j][11]),  # 15: mu_roll
                n,                                       # 16: cpair index (for warm-start)
                s,                                       # 17: sub_id within cpair
                min(cparam[i][12], cparam[j][12]),       # 18: restitution e (min blend)
            ))

    nc = len(cdata)
    nb = len(T)

    # Free subspace (jtype>0 DoFs) and joint Coulomb-friction rows. Mirrors lcp.c:
    # each 1-DoF rev/lin joint with floss>0 adds ONE constraint row whose Jacobian
    # is e_{fpos} (selects that DoF's velocity in the free subspace), target 0, with
    # a CONSTANT box bound ±floss·dt. Computed before the early-out because friction
    # rows exist independent of contacts (a joint resists motion with no contact).
    _, _vb, _, _nv_pb, _, _ = _build_qidx(jtype)
    free_idx = np.array(
        [_vb[i] + k for i in range(len(jtype)) if jtype[i] > 0 for k in range(_nv_pb[i])],
        dtype=int)
    fpos_of = {int(v): p for p, v in enumerate(free_idx)}
    fric = []   # (fpos in free subspace, dof v-index, bound = floss·dt)
    if floss is not None:
        for i in range(len(jtype)):
            if jtype[i] in (1, 2):                       # 1-DoF rev/lin only (v1)
                vidx = int(_vb[i])
                if floss[vidx] > 0.0:
                    fric.append((fpos_of[vidx], vidx, float(floss[vidx]) * dt))
    n_fric = len(fric)

    # Joint limit rows (mirrors lcp.c): one-sided position constraint, posed at the
    # velocity level like a contact normal. Active when q is at/past a bound (lo<hi).
    # (fpos, vidx, sign, baumgarte_b). sign +1 lower / −1 upper.
    lim = []
    if q is not None and jnt_lo is not None and jnt_hi is not None:
        for i in range(len(jtype)):
            if jtype[i] not in (1, 2):
                continue
            vidx = int(_vb[i])
            lo, hi = jnt_lo[vidx], jnt_hi[vidx]
            if not (lo < hi):
                continue
            if q[vidx] <= lo:
                depth = lo - q[vidx]
                lim.append((fpos_of[vidx], vidx, +1.0, (erp/dt) * max(0.0, depth - slop)))
            elif q[vidx] >= hi:
                depth = q[vidx] - hi
                lim.append((fpos_of[vidx], vidx, -1.0, (erp/dt) * max(0.0, depth - slop)))
    n_limit = len(lim)

    if nc == 0 and n_fric == 0 and n_limit == 0:
        empty_info = {'nc': 0, 'iters': 0, 'residual': 0.0,
                      'points': [], 'normals': [], 'depths': [], 'R_tan': [],
                      'lam_n':    np.zeros(0), 'lam_t1':   np.zeros(0), 'lam_t2': np.zeros(0),
                      'lam_spin': np.zeros(0), 'lam_r1':   np.zeros(0), 'lam_r2': np.zeros(0),
                      'lam_full': np.zeros(6 * MAX_PTS_PER_PAIR * npair),
                      'lam_fric_full': np.zeros(nq),
                      'lam_limit_full': np.zeros(nq),
                      'cpair_idx': np.zeros(0, dtype=int)}
        empty = (np.zeros(nq), np.zeros(0), empty_info, np.zeros((nb, 6)))
        return empty + ([],) if cff else empty

    # ---- Pass 2: stack J ∈ R^{6nc × nq} — 3 linear + 3 angular rows ------
    # Linear part (top 3 rows of jacob_whitney) → normal + 2 tangent.
    # Angular part (bottom 3 rows) → spin + 2 roll.
    # Tangent frame R_tan columns are [t1, t2, n_hat]; same direction set is
    # used for both linear (force) and angular (moment) projections, so the
    # spin axis = normal and roll axes = tangent axes.
    J = np.zeros((6 * nc, nq))
    for k, (ci, cj, p_world, R_tan, *_rest) in enumerate(cdata):
        if cbody[ci] >= 0:
            Tp = T[cbody[ci]].copy(); Tp[0:3, 3] = p_world
            J6A = jacob_whitney(T, Tp, parent, jtype, cbody[ci])     # (6, nq)
            JvA, JwA = J6A[0:3, :], J6A[3:6, :]
        else:
            JvA = np.zeros((3, nq)); JwA = np.zeros((3, nq))
        if cbody[cj] >= 0:
            Tp = T[cbody[cj]].copy(); Tp[0:3, 3] = p_world
            J6B = jacob_whitney(T, Tp, parent, jtype, cbody[cj])
            JvB, JwB = J6B[0:3, :], J6B[3:6, :]
        else:
            JvB = np.zeros((3, nq)); JwB = np.zeros((3, nq))
        Jv_rel = JvB - JvA                          # linear relative velocity rows
        Jw_rel = JwB - JwA                          # angular relative velocity rows
        J[6*k+0, :] = R_tan[:, 2] @ Jv_rel          # normal     (force along n)
        J[6*k+1, :] = R_tan[:, 0] @ Jv_rel          # tangent 1  (force along t1)
        J[6*k+2, :] = R_tan[:, 1] @ Jv_rel          # tangent 2  (force along t2)
        J[6*k+3, :] = R_tan[:, 2] @ Jw_rel          # spin       (moment about n)
        J[6*k+4, :] = R_tan[:, 0] @ Jw_rel          # roll 1     (moment about t1)
        J[6*k+5, :] = R_tan[:, 1] @ Jw_rel          # roll 2     (moment about t2)

    # friction rows: J[6nc+r] = e_{vidx} (single 1.0 at the DoF column). After the
    # free-subspace slice below this becomes e_{fpos}; the generic A/c build then
    # yields A_diag=(M⁻¹)_{fpos,fpos}, contact cross-terms, and c=qd_free_{fpos}.
    if n_fric:
        J_fric = np.zeros((n_fric, nq))
        for r, (fpos, vidx, bound) in enumerate(fric):
            J_fric[r, vidx] = 1.0
        J = np.vstack([J, J_fric])

    # limit rows: J = sign·e_{vidx} (one-sided; Baumgarte bias added to b below).
    if n_limit:
        J_lim = np.zeros((n_limit, nq))
        for r, (fpos, vidx, sign, b_lim) in enumerate(lim):
            J_lim[r, vidx] = sign
        J = np.vstack([J, J_lim])

    # ---- Pass 3: Delassus A and stabilization c --------------------------
    # M from crb_featherstone has zero rows/cols where jtype==0 (fixed joints
    # have S=0), so it is singular. Work in the jtype>0 subspace (free_idx, computed
    # above). J_f now has 6nc contact rows + n_fric friction rows; A is built generic
    # over all of them (friction rows couple to contacts through M_inv_f, exactly the
    # point of solving them together).
    M_f      = M[np.ix_(free_idx, free_idx)]
    J_f      = J[:, free_idx]
    qd_f     = qd_free[free_idx]
    M_inv_f  = np.linalg.inv(M_f)        # TODO: Cholesky/ABA-column for larger nq
    A = J_f @ M_inv_f @ J_f.T
    # Newton restitution. post-impact v_n^+ ≥ -e·v_n^-, applied only when approach
    # speed exceeds v_rest_thresh — otherwise resting contacts get a tiny rebound each
    # step and never settle. e is per-contact (min-blended material restitution, cdata
    # idx 18); v_rest_thresh is a global numerical gate, now a function argument.
    R = np.empty(6 * nc)
    b = np.zeros(6 * nc)                                  # only normal entry nonzero
    for k, d in enumerate(cdata):
        depth  = d[4]
        k_n, d_n        = d[5], d[6]
        k_t, d_t        = d[7], d[8]
        k_sp, d_sp      = d[10], d[11]
        k_rl, d_rl      = d[13], d[14]
        R[6*k+0] = cfm_scale / (k_n  * dt * dt + d_n  * dt + 1e-12)
        R[6*k+1] = cfm_scale / (k_t  * dt * dt + d_t  * dt + 1e-12)
        R[6*k+2] = R[6*k+1]
        R[6*k+3] = cfm_scale / (k_sp * dt * dt + d_sp * dt + 1e-12)
        R[6*k+4] = cfm_scale / (k_rl * dt * dt + d_rl * dt + 1e-12)
        R[6*k+5] = R[6*k+4]
        v_n_pre  = float(J_f[6*k+0] @ qd_f)               # predictor normal vel (closing < 0)
        e_rest   = d[18]                                   # min-blended restitution for this contact
        b_rest   = -e_rest * v_n_pre if v_n_pre < -v_rest_thresh else 0.0
        b_baum   = (erp / dt) * max(0.0, depth - slop)
        b[6*k+0] = max(b_baum, b_rest)                    # normal: Baumgarte ∪ restitution
        # tangent / spin / roll: no bias (stick-at-zero is the target)
    # friction rows: no CFM, no bias. limit rows: no CFM, Baumgarte push-out bias.
    # pad R with zeros for both; b gets 0 (friction) then the limit bias.
    if n_fric:
        R = np.concatenate([R, np.zeros(n_fric)])
        b = np.concatenate([b, np.zeros(n_fric)])
    if n_limit:
        R = np.concatenate([R, np.zeros(n_limit)])
        b = np.concatenate([b, np.array([b_lim for (_, _, _, b_lim) in lim])])
    A_diag = np.diag(A) + R
    np.fill_diagonal(A, A_diag)
    c = J_f @ qd_f - b                                    # w = A λ + c ≥ 0

    # ---- Pass 4: PGS with 4 cones (normal + tangent disk + spin + roll) --
    # All three friction cones share λ_n as their budget — projected
    # independently each sweep. Inner loop order: normal first (so the bound
    # is fresh), then tangent, spin, roll.
    # Warm-start: if lam_prev (length 6*MAX_PTS_PER_PAIR*npair, indexed by
    # slot = cpair_idx * MAX_PTS_PER_PAIR + sub_id) is given, seed lam from it.
    lam = np.zeros(6 * nc + n_fric + n_limit)
    if lam_prev is not None and len(lam_prev) >= 6 * MAX_PTS_PER_PAIR * npair:
        for k in range(nc):
            slot = cdata[k][16] * MAX_PTS_PER_PAIR + cdata[k][17]
            lam[6*k:6*k+6] = lam_prev[6*slot:6*slot+6]
    if n_fric and lam_fric_prev is not None:
        for r, (fpos, vidx, bound) in enumerate(fric):
            lam[6*nc + r] = lam_fric_prev[vidx]
    if n_limit and lam_limit_prev is not None:
        for r, (fpos, vidx, sign, b_lim) in enumerate(lim):
            lam[6*nc + n_fric + r] = lam_limit_prev[vidx]
    residual = 0.0
    it = 0
    for it in range(iters):
        residual = 0.0
        for k in range(nc):
            mu      = cdata[k][9]
            mu_spin = cdata[k][12]
            mu_roll = cdata[k][15]
            i_n  = 6*k + 0
            i_t1 = 6*k + 1; i_t2 = 6*k + 2
            i_sp = 6*k + 3
            i_r1 = 6*k + 4; i_r2 = 6*k + 5

            # --- (1) normal: project to [0, ∞) ---
            row = float(A[i_n, :] @ lam) - A[i_n, i_n] * lam[i_n] + c[i_n]
            new_n = max(0.0, -row / A[i_n, i_n])
            residual = max(residual, abs(new_n - lam[i_n]))
            lam[i_n] = new_n

            # --- (2) linear tangent disk: ‖(λ_t1, λ_t2)‖ ≤ μ λ_n ---
            row1 = float(A[i_t1, :] @ lam) - A[i_t1, i_t1] * lam[i_t1] + c[i_t1]
            row2 = float(A[i_t2, :] @ lam) - A[i_t2, i_t2] * lam[i_t2] + c[i_t2]
            v1 = -row1 / A[i_t1, i_t1]
            v2 = -row2 / A[i_t2, i_t2]
            bound = mu * new_n
            mag = math.sqrt(v1*v1 + v2*v2)
            if mag > bound and mag > 1e-12:
                s = bound / mag; v1 *= s; v2 *= s
            residual = max(residual, abs(v1 - lam[i_t1]), abs(v2 - lam[i_t2]))
            lam[i_t1] = v1; lam[i_t2] = v2

            # --- (3) spin (1D): |λ_spin| ≤ μ_spin λ_n ---
            row = float(A[i_sp, :] @ lam) - A[i_sp, i_sp] * lam[i_sp] + c[i_sp]
            v = -row / A[i_sp, i_sp]
            bound = mu_spin * new_n
            if v >  bound: v =  bound
            elif v < -bound: v = -bound
            residual = max(residual, abs(v - lam[i_sp]))
            lam[i_sp] = v

            # --- (4) roll disk: ‖(λ_r1, λ_r2)‖ ≤ μ_roll λ_n ---
            row1 = float(A[i_r1, :] @ lam) - A[i_r1, i_r1] * lam[i_r1] + c[i_r1]
            row2 = float(A[i_r2, :] @ lam) - A[i_r2, i_r2] * lam[i_r2] + c[i_r2]
            v1 = -row1 / A[i_r1, i_r1]
            v2 = -row2 / A[i_r2, i_r2]
            bound = mu_roll * new_n
            mag = math.sqrt(v1*v1 + v2*v2)
            if mag > bound and mag > 1e-12:
                s = bound / mag; v1 *= s; v2 *= s
            residual = max(residual, abs(v1 - lam[i_r1]), abs(v2 - lam[i_r2]))
            lam[i_r1] = v1; lam[i_r2] = v2

        # --- (5) joint-friction rows: 1D box clamp to ±floss·dt (constant bound) ---
        for r, (fpos, vidx, bound) in enumerate(fric):
            row = 6*nc + r
            w_row = float(A[row, :] @ lam) - A[row, row] * lam[row] + c[row]
            v = -w_row / A[row, row]
            if   v >  bound: v =  bound
            elif v < -bound: v = -bound
            residual = max(residual, abs(v - lam[row]))
            lam[row] = v

        # --- (6) joint-limit rows: one-sided clamp λ ≥ 0 (like the contact normal) ---
        for r in range(n_limit):
            row = 6*nc + n_fric + r
            w_row = float(A[row, :] @ lam) - A[row, row] * lam[row] + c[row]
            v = max(0.0, -w_row / A[row, row])
            residual = max(residual, abs(v - lam[row]))
            lam[row] = v

        if residual < tol:
            break

    # ---- Pass 5: velocity correction -------------------------------------
    dqd_f = M_inv_f @ (J_f.T @ lam)        # includes friction rows (Jᵀλ routes them to their DoF)
    dqd = np.zeros(nq)
    dqd[free_idx] = dqd_f

    # split λ: contacts (6nc, packed) drive f_ext + warm-start; friction/limit (per-DoF) carry separately.
    lam_c = lam[:6*nc]
    lam_fric_full = lam_fric_prev.copy() if lam_fric_prev is not None else np.zeros(nq)
    for r, (fpos, vidx, bound) in enumerate(fric):
        lam_fric_full[vidx] = lam[6*nc + r]
    lam_limit_full = lam_limit_prev.copy() if lam_limit_prev is not None else np.zeros(nq)
    for r, (fpos, vidx, sign, b_lim) in enumerate(lim):
        lam_limit_full[vidx] = lam[6*nc + n_fric + r]

    lam_n    = lam_c[0::6]
    lam_t1   = lam_c[1::6]
    lam_t2   = lam_c[2::6]
    lam_spin = lam_c[3::6]
    lam_r1   = lam_c[4::6]
    lam_r2   = lam_c[5::6]
    # Scatter active-contact λ back to per-(cpair_idx, sub_id) slot storage for
    # warm-start. Inactive slots keep their previous value (if any) so transient
    # separations don't immediately lose their warm-start budget. lam_full is
    # sized 6 * MAX_PTS_PER_PAIR * npair (one 6-vec per slot).
    lam_full_size = 6 * MAX_PTS_PER_PAIR * npair
    lam_full = lam_prev.copy() if lam_prev is not None and len(lam_prev) >= lam_full_size else np.zeros(lam_full_size)
    cpair_idx = np.empty(nc, dtype=int)
    for k in range(nc):
        cp_idx = cdata[k][16]
        sub_id = cdata[k][17]
        slot   = cp_idx * MAX_PTS_PER_PAIR + sub_id
        cpair_idx[k] = cp_idx
        lam_full[6*slot:6*slot+6] = lam[6*k:6*k+6]
    info = {
        'nc': nc, 'iters': it + 1, 'residual': residual,
        'points':  [d[2] for d in cdata],
        'normals': [d[3][:, 2] for d in cdata],
        'depths':  [d[4] for d in cdata],
        'R_tan':   [d[3] for d in cdata],
        'lam_n':    lam_n,    'lam_t1': lam_t1, 'lam_t2': lam_t2,
        'lam_spin': lam_spin, 'lam_r1': lam_r1, 'lam_r2': lam_r2,
        'lam_full': lam_full, 'lam_fric_full': lam_fric_full,
        'lam_limit_full': lam_limit_full, 'cpair_idx': cpair_idx,
    }

    # ---- Pass 6: synthesize per-body spatial f_ext (for feedback compat) -
    # World-frame quantities at the contact point:
    #   force  cf0     = R_tan @ (λ_t1, λ_t2, λ_n)    / dt
    #   couple m0      = R_tan @ (λ_r1, λ_r2, λ_spin) / dt        ← new
    # Body i gets (-cf0, -m0); body j gets (+cf0, +m0). The body-frame wrench
    # also picks up the arm × force moment from offsetting the application point.
    f_ext_out = np.zeros((nb, 6))
    cfs = [] if cff else None
    for k, d in enumerate(cdata):
        ci, cj, p_world, R_tan = d[0], d[1], d[2], d[3]
        f_local = np.array([lam_t1[k], lam_t2[k], lam_n[k]])   / dt   # contact-frame
        m_local = np.array([lam_r1[k], lam_r2[k], lam_spin[k]]) / dt
        cf0 = R_tan @ f_local                                          # world force
        m0  = R_tan @ m_local                                          # world couple
        p4 = np.array([p_world[0], p_world[1], p_world[2], 1.0])
        if cbody[ci] >= 0:
            Tb = T[cbody[ci]]
            ef = -Tb[:3, :3].T @ cf0
            rp = (np.linalg.inv(Tb) @ p4)[:3]
            em = np.cross(rp, ef) + Tb[:3, :3].T @ (-m0)
            f_ext_out[cbody[ci]] += np.concatenate((em, ef))
        if cbody[cj] >= 0:
            Tb = T[cbody[cj]]
            ef = Tb[:3, :3].T @ cf0
            rp = (np.linalg.inv(Tb) @ p4)[:3]
            em = np.cross(rp, ef) + Tb[:3, :3].T @ m0
            f_ext_out[cbody[cj]] += np.concatenate((em, ef))
        if cff:
            cfs.append([p_world[0], p_world[1], p_world[2],
                        -cf0[0], -cf0[1], -cf0[2], ci, cj])

    if cff:
        return dqd, lam_c, info, f_ext_out, cfs
    return dqd, lam_c, info, f_ext_out

#Ray-intersection primitives — ported to C (ray.c:ray_intersects_*) and
#dispatched via tact_raycast_query / tact_raymap_query. Env.raycast / raymap
#no longer call these. Preserved as the reference implementation (verified
#against 27 analytical + 82 cross-validation tests) and a Python fallback
#if anyone needs it without going through the C handle.
"""
def ray_intersects_triangle(ray_origin, ray_vector, vertex0, vertex1, vertex2):
    EPSILON = 1e-6
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON: return -1 # Ray is parallel to triangle

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0: return -1

    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    if v < 0.0 or u + v > 1.0: return -1

    t = f * np.dot(edge2, q)
    if t > EPSILON: return t # intersection_point
    else: return -1

def ray_intersects_mesh(ray_origin, ray_dir, vertices, faces):
    dist = np.inf

    for face in faces:
        v0, v1, v2 = vertices[face]
        t = ray_intersects_triangle(ray_origin, ray_dir, v0, v1, v2)
        if 0 < t < dist: dist = t

    if dist == np.inf: dist = -1
    return dist
    
def ray_intersects_box(ray_origin, ray_dir, box_center, box_rotation, half_sizes):
    EPSILON = 1e-6
    t_min = -np.inf
    t_max = np.inf
    p = box_center - ray_origin

    for i in range(3):
        axis = box_rotation[:, i]
        e = np.dot(axis, p)
        f = np.dot(ray_dir, axis)

        if abs(f) > EPSILON:
            t1 = (e + half_sizes[i]) / f
            t2 = (e - half_sizes[i]) / f
            if t1 > t2: t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max: return -1 # No hit
            
        else: # Ray is parallel to the slab
            if -e - half_sizes[i] > 0 or -e + half_sizes[i] < 0: return -1 # No hit

    if t_min < 0: return -1 # Box is behind the ray
    return t_min

def ray_intersects_sphere(R0, Rd, C, r):
    L = C - R0
    t_ca = np.dot(L, Rd)
    d2 = np.dot(L, L) - t_ca**2
    r2 = r * r
    if d2 > r2: return -1  # Miss

    thc = np.sqrt(r2 - d2)
    t0 = t_ca - thc
    t1 = t_ca + thc

    # 교차점 중 앞에 있는 것 선택
    t = None
    if t0 >= 0: t = t0
    elif t1 >= 0: t = t1

    if t is None: return -1 # Ray가 뒤에서 시작
    return t

def ray_intersects_cylinder(R0, Rd, P1, P2, r):
    EPSILON = 1e-8
    v = (P2 - P1)/np.linalg.norm(P2 - P1)
    d = Rd
    m = R0 - P1
    md = np.dot(m, v)
    nd = np.dot(d, v)

    a = np.dot(d, d) - nd**2
    b = np.dot(d, m) - nd * md
    c = np.dot(m, m) - md**2 - r**2
    dist = np.inf
    
    if abs(a) > EPSILON:
        disc = b * b - a * c
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / a
            t2 = (-b + sqrt_disc) / a

            for t in [t1, t2]:
                if t < 0: continue
                hit = R0 + t * d
                h = np.dot(hit - P1, v)
                if 0 <= h <= np.linalg.norm(P2 - P1) and t < dist: dist = t
                    
    # Check caps
    for cap_center in [P1, P2]:
        denom = np.dot(d, v)
        if abs(denom) > EPSILON:
            t = np.dot(cap_center - R0, v)/denom            
            if t >= 0:
                hit = R0 + t * d
                radial = hit - cap_center - np.dot(hit - cap_center, v) * v
                if np.linalg.norm(radial) <= r and t < dist: dist = t
                    
    if dist == np.inf: dist = -1
    return dist
    
def ray_intersects_capsule(R0, Rd, P1, P2, r):
    EPSILON = 1e-8
    v = (P2 - P1)/np.linalg.norm(P2 - P1)
    d = Rd
    m = R0 - P1
    md = np.dot(m, v)
    nd = np.dot(d, v)

    a = np.dot(d, d) - nd**2
    b = np.dot(d, m) - nd * md
    c = np.dot(m, m) - md**2 - r**2
    dist = np.inf

    if abs(a) > EPSILON:
        disc = b * b - a * c
        
        if disc >= 0.0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / a
            t2 = (-b + sqrt_disc) / a
            
            for t in [t1, t2]:
                if t < 0: continue
                hit = R0 + t * d
                h = np.dot(hit - P1, v)
                if 0 <= h <= np.linalg.norm(P2 - P1) and t < dist: dist = t

    for cap in [P1, P2]:
        t = ray_intersects_sphere(R0, Rd, cap, r)
        if 0 <= t < dist: dist = t

    if dist == np.inf: dist = -1
    return dist
"""
