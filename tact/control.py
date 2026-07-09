"""Control & planning: PIDController, JacobianTranspose*, ComputedTorqueController,
HybridForcePositionController, MovingAverageWaypointSmoother, pinch primitives, Envelope,
StepGenerator2/4, body-velocity estimators (ComplementaryEstimator, ContactAidedEKF,
InvariantEKF), and perception (MiniElevationMap — split out to a perception module if
this section grows). Controllers receive Model via duck typing (no Model import needed);
StepGenerator{2,4} reach into rbd for a few transform helpers; the estimators use
expmap_so3/skew from rbd."""
import numpy as np
from collections import deque
from .rbd import (
    rotxyz_to_homogeneous, xyzquat_to_homogeneous, xyzeuler_to_homogeneous,
    xyheading_to_homogeneous, homogeneous_to_xyzeuler, xyzeuler_to_xyzquat,
    euler_to_rotation, rotation_to_euler,
    expmap_so3, logmap_so3, skew,
)

    
#---------------controller functions-----------------------------------------------------------------
#way point smoother
class MovingAverageWaypointSmoother:
    def __init__(self, ws):
        self.L = deque() #linear path
        self.Z = deque(maxlen=100)
        self.ws_max = ws #window size
        self.ws = 1
        self.t = -1
        self.avg = []
        
    def is_done(self):
        if len(self.L) < self.ws_max: return True
        return False
        
    def target(self, P, T, p0=None, t=None):
        if t != None and self.t != t:
            self.L.clear()
            self.Z.clear()
            self.L.append(p0)
            self.t = t
            
        while len(self.L) > self.ws: self.L.pop()
        self.ws = len(self.L)
            
        for i in range(len(P)):
            dff = (P[i] - self.L[-1])/T[i]
            tar = self.L[-1] + dff
            for j in range(T[i]): self.L.append(tar+j*dff)
            
    def generate(self, depth=0):
        sum = np.zeros_like(self.L[0])
        avgs = []
        
        for i in range(self.ws + depth):
            idx1 = min(i, len(self.L)-1)
            sum += self.L[idx1]
            
            if i >= self.ws:
                idx2 = min(i-self.ws, len(self.L)-1)
                sum -= self.L[idx2]

            if i >= self.ws - 1:
                avgs.append(sum/self.ws)

        self.avg = avgs[0]
        if len(self.L) > 1: self.L.popleft()
        self.Z.appendleft(self.avg)
        
        if self.ws < self.ws_max: self.ws += 1
        elif self.ws > self.ws_max: self.ws -= 1
        if self.ws > len(self.L): self.ws = len(self.L)
        
        self.t += 1
        #print(len(self.L), self.ws)
        
        if depth > 0: return avgs
        else: return self.avg

    def generate2(self, dt):
        # Backward finite differences over Z. Warmup gate: until ws == ws_max
        # we'd be differencing across moving averages of *different* window
        # sizes → spurious large qd/qdd. Returns zeros until window stabilizes
        # and ≥3 samples accumulate, so callers can use the output unguarded.
        self.generate()
        avg = self.Z[0]
        warmed = (self.ws == self.ws_max) and (len(self.Z) >= 3)
        if warmed:
            avgd  = (self.Z[0] - self.Z[1]) / dt
            avgdd = (self.Z[0] - 2.0*self.Z[1] + self.Z[2]) / (dt*dt)
        else:
            avgd  = np.zeros_like(avg)
            avgdd = np.zeros_like(avg)
        return avg, avgd, avgdd
            
#SE(3) pose smoother — drop-in for MovingAverageWaypointSmoother on 6-D xyz-euler
#pose vectors. Orientation is interpolated on SO(3) (axis-angle ramp ≡ SLERP for
#the two-endpoint case) by anchoring each frame's rotation to its motion-start R0
#and storing/averaging deltas as rotation vectors in that tangent space. The inner
#linear smoother handles the Euclidean math; this wrapper only does the encode/
#decode at the boundary, so I/O shape is identical to the linear smoother (same
#target/generate signature, same flat (n_frames*6,) xyz-euler vectors).
#
#num_frames=1 → single arm (tcp); num_frames=2 → dual arm (tcp1, tcp2); per-frame
#R0 list keeps the two arms' tangent spaces independent. Assumes per-motion
#relative rotation < π (true in kida since workspace clip bounds desired Euler to
#~±π/2 per axis); for larger sweeps split into smaller waypoints.
class SE3WaypointSmoother:
    def __init__(self, ws, num_frames=1, eulerseq='xyz'):
        self.ws = ws
        self.nf = num_frames
        self.eulerseq = eulerseq
        self.inner = MovingAverageWaypointSmoother(ws)
        self.R0 = [None]*num_frames
        self.t = -1

    def is_done(self):
        return self.inner.is_done()

    def _to_local(self, x):
        out = np.zeros_like(x, dtype=float)
        for k in range(self.nf):
            b = 6*k
            R = euler_to_rotation(x[b+3:b+6], self.eulerseq)
            if self.R0[k] is None: self.R0[k] = R
            out[b:b+3]   = x[b:b+3]
            out[b+3:b+6] = logmap_so3(self.R0[k].T @ R)
        return out

    def _from_local(self, l):
        out = np.zeros_like(l, dtype=float)
        for k in range(self.nf):
            b = 6*k
            R = self.R0[k] @ expmap_so3(l[b+3:b+6])
            out[b:b+3]   = l[b:b+3]
            out[b+3:b+6] = rotation_to_euler(R, self.eulerseq)
        return out

    def target(self, P, T, p0=None, t=None):
        if t is not None and self.t != t:
            self.R0 = [None]*self.nf            #re-anchor at every new motion
            self.t = t
            p0L = self._to_local(np.asarray(p0, dtype=float))
        else:
            p0L = None
        PL = np.array([self._to_local(np.asarray(p, dtype=float)) for p in P])
        self.inner.target(PL, T, p0L, t)

    def generate(self):
        return self._from_local(self.inner.generate())


#joint PID controller
class PIDController:
    def __init__(self, k_p, k_d, k_i, dt):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.dt = dt
        self.cnt = 0
        
    def update(self, q_d, q, d):
        e = q_d - q

        if self.cnt == 0:
            self.e_sum = np.zeros(len(q))
            #e_dot = np.zeros(len(q))

        #else: e_dot = (e - self.e_old)/self.dt
        
        #u = self.k_p*e + self.k_i*self.e_sum + self.k_d*e_dot
        u = self.k_p*e - self.k_d*d + self.k_i*self.e_sum

        self.e_sum += e*self.dt
        #self.e_old = e
        self.cnt += 1
        return u
    
#task space position controller
class JacobianTransposeController:
    def __init__(self, model, frame, K_p, K_d):
        self.m = model
        self.frame = frame
        self.K_p = K_p #np.diag(K_p)
        self.K_d = K_d #np.diag(K_d)

    #Optional J skips the second m.jacob call when the caller already needs J
    #for something else (null-space projection, redundancy resolution, etc.) —
    #legacy callers pass nothing and J is computed internally. Return value is
    #always just tau regardless of J path, so existing call sites are unaffected.
    def update(self, x_d, q, d, J=None):
        if J is None: J = self.m.jacob(self.frame, q)
        e = self.m.error(self.frame, q, x_d)
        x_dot = J @ d
        return J.T @ (self.K_p*e - self.K_d*x_dot)
    
#task-space force controller (feed-forward)
class JacobianTransposeForce:
    def __init__(self, model, frame):
        self.m = model
        self.frame = frame
        
    def update(self, f, q):
        J = self.m.jacob(self.frame, q)
        u = np.matmul(np.transpose(J), f)
        return u

#computed torque controller
class ComputedTorqueController:
    def __init__(self, model, kp, kd):
        self.m = model
        self.kp = kp
        self.kd = kd
        
    def update(self, q_d, qd_d, qdd_d, q, qd):
        if len(qd_d) == 0: qd_d = qd
        if len(qdd_d) == 0: qdd_d = np.zeros(len(q_d))
        
        M = self.m.inertia(q)
        b = self.m.bias(q, qd)        
        u = M @ (qdd_d + self.kd @ (qd_d - qd) + self.kp @ (q_d - q)) + b
        return u
        
#hybrid force-position controller
class HybridForcePositionController:
    def __init__(self, model, frame, K_p, K_d):
        self.m = model
        self.frame = frame
        self.K_p = K_p 
        self.K_d = K_d
        self.h = int(len(K_p)/3)

    def hselect(self, f_, num):
        r = np.array([0, 0, 1], dtype=float)
        sigma = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Se = np.identity(3*num)
        f = f_.reshape(num, 3)
    
        for n in range(num):
            norm = np.linalg.norm(f[n])
            if norm < 0.00001: continue
        
            i = f[n]/norm
            j = np.cross(i, r)
            k = np.cross(i, j)
        
            Rt = np.array([i, j, k])
            R = np.transpose(Rt)

            tmp = np.matmul(sigma, Rt)
            omega = np.matmul(R, tmp)
        
            Se[3*n:3*n+3, 3*n:3*n+3] = omega

        #print(R)
        return Se
        
    def update(self, x_d, f, q, d):
        e = self.m.error(self.frame, q, x_d)
        e = np.matmul(self.hselect(f, self.h), e)
        J = self.m.jacob(self.frame, q)
        
        x_dot = np.matmul(J, d)
        u = np.matmul(np.transpose(J), self.K_p*e - self.K_d*x_dot + f)
        return u

    def setgain(self, K_p, K_d):
        self.K_p = K_p
        self.K_d = K_d

#2-finger pinching
def pinch2(x, kg, fi):
    f = np.zeros((int(len(x)/3), 3))
    x = x.reshape(int(len(x)/3), 3)

    #original method using cg
    #cg = np.zeros(3)
    #for i in fi: cg += x[i]
    #cg = cg/2
    ##f[fi[0]] = 2*kg*(cg - x[fi[0]])/np.linalg.norm(cg - x[fi[0]])  #<---------TEMP    
    #f[fi[0]] = kg*(cg - x[fi[0]])/np.linalg.norm(cg - x[fi[0]])
    #f[fi[1]] = kg*(cg - x[fi[1]])/np.linalg.norm(cg - x[fi[1]])

    #simplier method
    f[fi[0]] = kg*(x[fi[1]] - x[fi[0]])/np.linalg.norm(x[fi[1]] - x[fi[0]])
    f[fi[1]] = kg*(x[fi[0]] - x[fi[1]])/np.linalg.norm(x[fi[0]] - x[fi[1]])
    
    return f.reshape(-1)
    
#3-finger parallel pinching
def pinch3p(x, kg, fi):
    f = np.zeros((int(len(x)/3), 3))
    x = x.reshape(int(len(x)/3), 3)

    cg = np.zeros(3)
    for i in fi: cg += x[i]
    cg = cg/3

    f[fi[0]] = kg*(cg - x[fi[0]])/np.linalg.norm(cg - x[fi[0]])
    f[fi[1]] = -0.5*f[fi[0]]
    f[fi[2]] = -0.5*f[fi[0]]
    return f.reshape(-1)

#3-finger pinching based on ICRA2012 paper; don't use this function
def pinch3_icra2012(x, kg, fi):
    f = np.zeros((int(len(x)/3), 3))
    x = x.reshape(int(len(x)/3), 3)

    cg = np.zeros(3)
    for i in fi: cg += x[i]
    cg = cg/3

    for i in fi:
        f[i] = (cg - x[i])/np.linalg.norm(cg - x[i])
    
    alpha = np.zeros(len(f))
    alpha[fi[0]] = kg #<------
    alpha[fi[1]] = kg*np.linalg.norm(cg - x[fi[0]])/np.linalg.norm(cg - x[fi[1]])
    alpha[fi[2]] = np.linalg.norm(kg*f[fi[0]] + alpha[fi[1]]*f[fi[1]])

    for i in fi: f[i] *= alpha[i]
    return f.reshape(-1)

#piony's solution for 3-finger pinching
def pinch3_piony(x, kg, i):
    f = np.zeros((int(len(x)/3), 3))
    x = x.reshape(int(len(x)/3), 3)

    f[i[1]] = kg*(x[i[0]] - x[i[1]])/np.linalg.norm(x[i[0]] - x[i[1]])
    f[i[2]] = kg*(x[i[0]] - x[i[2]])/np.linalg.norm(x[i[0]] - x[i[2]])
    f[i[0]] = -1*(f[i[1]] + f[i[2]])
    return f.reshape(-1)

class Envelope:
    def __init__(self, q, k_p, ti):
        self.k_p = k_p
        self.ti = np.array(ti) #torque controlled joint index
        self.q_d = q #pose desired
                
    def update(self, q, tt):
        u = np.zeros(len(self.q_d))
        for i in range(len(self.q_d)):
            if i in self.ti: u[i] = tt[np.where(self.ti==i)[0][0]]
            else: u[i] = self.k_p*(self.q_d[i] - q[i])
        return u
    
#--------------------trajectry -----------------

class StepGenerator2: #step generator for biped
    def __init__(self, env, FO, LO, LL, HS):
        self.env = env
        self.wp = []

        self.FO = FO #frontal offset
        self.LO = LO #lateral offset
        self.LL = LL #leg length
        self.HS = HS #half stride length
        self.RA = np.pi/12 # 15 deg
        
        self.op = [0, 0]
        self.scan_step = 0.005
        self.n_scan = 30

    def get_global_ref(self, x, R, p, form='v7'):
        T_BL = xyzeuler_to_homogeneous(x[:6])
        T_BR = xyzeuler_to_homogeneous(x[6:])
        T_GB = rotxyz_to_homogeneous(R, p)

        B6 = homogeneous_to_xyzeuler(T_GB)
        L6 = homogeneous_to_xyzeuler(T_GB @ T_BL)
        R6 = homogeneous_to_xyzeuler(T_GB @ T_BR)

        # Stance feet ARE the height reference: their FK z (joint encoders through
        # the estimated base pose) is knowable on a real robot, so it stays as-is.
        # (The old env.get_z snap-to-terrain overwrite here was an absolute-world-z
        # oracle — a sim trick, removed 2026-06-06.) Roll/pitch still flattened so
        # the ref frame stays gravity-aligned.
        L6[3:5] = [0, 0]
        R6[3:5] = [0, 0]

        cp = (L6[:3] + R6[:3])/2
        #print('cp:', cp)

        #calculated mid-yaw right/left foot yaw
        x = np.cos(L6[5]) + np.cos(R6[5])
        y = np.sin(L6[5]) + np.sin(R6[5])
        yaw = np.arctan2(y, x)
        #yaw = vec2heading(heading2vec(L6[5]) + heading2vec(R6[5]))
        
        B6 = np.array([cp[0]+self.FO*np.cos(yaw), cp[1]+self.FO*np.sin(yaw), cp[2]+self.LL, 0, 0, yaw])    
        if   form == 'v6': return np.array([B6, L6, R6])
        elif form == 'v7': return np.array([xyzeuler_to_xyzquat(B6), xyzeuler_to_xyzquat(L6), xyzeuler_to_xyzquat(R6)])
        
    def line_adjust(self, ref):
        # Terrain-edge detect along the candidate foothold's heading axis, on
        # RELATIVE heights: only Δh between scan points is consumed, so the
        # scan's base reference cancels — this reads exactly what a real
        # elevation map provides (env.height_scan = the sim-GT provider of that
        # same contract; was env.get_z absolute-z per point, a sim trick,
        # removed 2026-06-06).
        adjust = 0

        xy = (ref[0][2], ref[1][2])
        yaw = np.arctan2(-ref[0][1], ref[1][1])
        d = self.scan_step * np.arange(self.n_scan)
        offs = np.zeros((2*self.n_scan, 2))
        offs[:self.n_scan, 0] = d       #forward scan
        offs[self.n_scan:, 0] = -d      #backward scan (higher priority)
        h = self.env.height_scan(xy, yaw, offs)
        fz, rz = h[:self.n_scan], h[self.n_scan:]

        for i in range(self.n_scan-1):
            if   fz[i+1] - fz[i] >  0.01: adjust = self.scan_step*i - 0.10; break  #rising edge detect
            elif fz[i+1] - fz[i] < -0.01: adjust = self.scan_step*i + 0.10; break  #falling edge detect

        for i in range(self.n_scan-1):
            if   rz[i+1] - rz[i] >  0.01: adjust = -self.scan_step*i - 0.10; break  #rising edge detect
            elif rz[i+1] - rz[i] < -0.01: adjust = -self.scan_step*i + 0.10; break  #falling edge detect

        out = ref @ np.array([[1, 0, adjust], [0, 1, 0], [0, 0, 1]])
        return out #: 3x3 matrix
        
    def swing_foot_target(self, pv3, phase):
        if self.op[0] == 0 and self.op[1] == 0: return None
        HS = self.op[0] * self.HS
        RA = self.op[1] * self.RA
        
        pv33 = xyheading_to_homogeneous(pv3[0], pv3[1], pv3[2])
        if   phase == 'left_swing' : tar33 = pv33 @ xyheading_to_homogeneous(HS,  2*self.LO, RA)
        elif phase == 'right_swing': tar33 = pv33 @ xyheading_to_homogeneous(HS, -2*self.LO, RA)

        tar33 = self.line_adjust(tar33)
        x, y, heading = tar33[0][2], tar33[1][2], np.arctan2(-tar33[0][1], tar33[1][1])
        return [x, y, heading]

#simple step generator for quadroped
class StepGenerator4:
    def __init__(self, env, offset, body_length, foot_radius):
        self.env = env
        self.wp = []        
        self.stride = [0, 0, 0] #Forward, Lateral, Turn

        self.x_off = offset[0] #0.06
        self.y_off = offset[1] #0.13
        self.z_off = offset[2] #0.38
        
        self.body_length = body_length #0.56
        self.foot_radius = foot_radius #0.025
        
    def body_ref_pose(self, p1, p2, p3, p4):
        #calculate x, y ,z -axis
        v31 = (p1 - p3)/np.linalg.norm(p1 - p3)
        v42 = (p2 - p4)/np.linalg.norm(p2 - p4)
        x = (v31 + v42)/np.linalg.norm(v31 + v42)
        y = np.cross(np.array([0, 0, 1]), x) #calculate y-axis
        z = np.cross(x, y)  #calculate z-axis

        #calculate position
        tmp = (p1 + p2 + p3 + p4)/4 + np.array([0, 0, self.z_off])
        pos = tmp + self.x_off*x

        T = np.array([[x[0], y[0], z[0], pos[0]], [x[1], y[1], z[1], pos[1]], [x[2], y[2], z[2], pos[2]], [0, 0, 0, 1.0]])
        return T

    def get_global_ref(self, x, R, p):
        T = rotxyz_to_homogeneous(R, p)        
        FL = np.concatenate((T @ np.array([x[0], x[1], x[2], 1]), [0, 0, 0]))
        FR = np.concatenate((T @ np.array([x[3], x[4], x[5], 1]), [0, 0, 0]))
        RL = np.concatenate((T @ np.array([x[6], x[7], x[8], 1]), [0, 0, 0]))
        RR = np.concatenate((T @ np.array([x[9], x[10], x[11], 1]), [0, 0, 0]))

        B = self.body_ref_pose(FL[:3], FR[:3], RL[:3], RR[:3])
        B7 = xyzeuler_to_xyzquat(homogeneous_to_xyzeuler(B))
        ref = np.array([B7, FL, FR, RL, RR])
        return ref
    
    def line_adjust(self, P0, P1):
        unit = 0.005
        norm = np.linalg.norm(P1 - P0)
        # Degenerate target (P1 == P0): nothing to scan. (The old code fed
        # nan through goal_dir into the per-point queries and fell through to
        # `return P1` anyway — same result, now explicit and warning-free.)
        if norm < 1e-9: return P1
        goal_dir = (P1 - P0)/norm
        coverage = 1.3*norm
        n_check = int(coverage/unit)
        min_z_diff = 0.01

        # Relative-height scan along P0→P1 (n_check+1 points, `unit` apart): only
        # consecutive Δh is consumed, so the scan's base reference cancels — same
        # edge logic as the old per-point env.get_z but trick-free (no absolute-z
        # oracle; height_scan is the contract a real elevation map provides).
        yaw = np.arctan2(goal_dir[1], goal_dir[0])
        offs = np.zeros((n_check + 1, 2))
        offs[:, 0] = unit * np.arange(n_check + 1)
        hz = self.env.height_scan(P0, yaw, offs)

        for i in range(n_check):
            off1 = unit*i
            off2 = unit*(i+1)

            z_diff = hz[i+1] - hz[i]
            off_eff = 0.5*(off1 + off2)
            #print(z_diff, n_check, coverage, norm)
            
            #upstair
            if z_diff > min_z_diff:
                if off_eff < norm:
                    if off_eff + 0.06 > norm: P1 = P0 + (off_eff + 0.06)*goal_dir
                else: P1 = P0 + (off_eff - 0.06)*goal_dir
                print('! n_check: %d  z_diff: %f   off_eff: %f  i: %d  off1: %f  off2: %f  coverage: %f' %(n_check, z_diff, off_eff, i, off1, off2, coverage))
                break;
            
            #downstair
            elif z_diff < -min_z_diff:
                if off_eff < norm:
                    if off_eff + 0.10 > norm: P1 = P0 + (off_eff + 0.10)*goal_dir
                else: P1 = P0 + (off_eff - 0.04)*goal_dir
                print('!! n_check: %d  z_diff: %f   off_eff: %f  i: %d  off1: %f  off2: %f  coverage: %f' %(n_check, z_diff, off_eff, i, off1, off2, coverage))
                break
                
        return P1    
        
    def swing_foot_target(self, B, p1, p2, p3, p4, phase):
        T0b = xyzquat_to_homogeneous(B)
        FF0p = p1 if phase == 0 else p2   # current stance-pair foot poses (world FK)
        RF0p = p3 if phase == 0 else p4
        FF0, RF0 = FF0p[:2], RF0p[:2]

        if phase == 0: y_off = self.y_off
        else: y_off = -self.y_off

        #front foot
        FF1 = np.array([[1, 0, 0,  0.5*self.body_length - self.x_off + self.stride[0]], [0, 1, 0, y_off + self.stride[1] + self.stride[2]], [0, 0, 1, 0], [0, 0, 0, 1]])
        FF1 = (T0b @ FF1)[:2, 3]
        FF = self.line_adjust(FF0, FF1)

        #rear foot
        RF1 = np.array([[1, 0, 0, -0.5*self.body_length - self.x_off + self.stride[0]], [0, 1, 0, y_off + self.stride[1] - self.stride[2]], [0, 0, 1, 0], [0, 0, 0, 1]])

        RF1 = (T0b @ RF1)[:2, 3]
        RF = self.line_adjust(RF0, RF1)

        # New foot-center z = CURRENT foot's FK z + relative terrain delta
        # old-xy → new-xy (the scan's base ref cancels in the single-offset
        # query). On terrain this equals the old env.get_z(new_xy)+foot_radius
        # whenever the current foot rests on the ground, but every term is
        # knowable on a real robot (kinematics + elevation map) — no
        # absolute-world-z oracle (get_z removed 2026-06-06).
        dz_f = self.env.height_scan(FF0, 0.0, [FF - FF0])[0]
        dz_r = self.env.height_scan(RF0, 0.0, [RF - RF0])[0]
        out1 = np.array([FF[0], FF[1], FF0p[2] + dz_f])
        out2 = np.array([RF[0], RF[1], RF0p[2] + dz_r])

        out1 = np.concatenate((out1, [1, 0, 0, 0])) #new front
        out2 = np.concatenate((out2, [1, 0, 0, 0])) #new rear
        return out1, out2


class GaitScheduler:
    """Time-based contact / swing phase scheduler for legged robots.

    Each foot i has its own (period, duty, offset). The local phase
        p_i(t) = ((t / period) + offset_i)  mod  1
    classifies the foot as
        stance  if  p_i <  duty_i
        swing   if  p_i >= duty_i,  with swing_phase = (p_i - duty_i) / (1 - duty_i)

    Foot index order is whatever the caller adopts (this project uses [FL, FR, RL, RR]
    matching dog.yaml's foot1..foot4). The class is robot-agnostic — works for any
    n-leg robot (biped, quadruped, hexapod).

    Convention: the cycle starts at stance (p=0 ⇒ touchdown). Standard parameters
    (4-leg, period=0.5 s, duty=0.5 s/cycle, stance-first):
        Trot    offsets=[0.0, 0.5, 0.5, 0.0]      # diagonal pairs
        Pace    offsets=[0.0, 0.5, 0.0, 0.5]      # ipsilateral pairs
        Bound   offsets=[0.0, 0.0, 0.5, 0.5]      # fore / hind pairs
        Crawl   period=2.0, duty=0.75, offsets=[0.0, 0.5, 0.75, 0.25]   # one foot at a time

    All scalar args (period, duty, offsets) accept per-foot arrays for asymmetric gaits."""

    def __init__(self, period, duty, offsets):
        self.offsets = np.asarray(offsets, dtype=float)
        self.n       = len(self.offsets)
        self.period  = np.broadcast_to(np.asarray(period, dtype=float), (self.n,)).copy()
        self.duty    = np.broadcast_to(np.asarray(duty,   dtype=float), (self.n,)).copy()
        assert np.all(self.period > 0),                       'period must be > 0'
        assert np.all((self.duty > 0) & (self.duty <= 1.0)),  'duty must be in (0, 1]'

    def local_phase(self, t):
        """Per-foot normalized phase ∈ [0, 1)."""
        return ((t / self.period) + self.offsets) % 1.0

    def update(self, t):
        """(contact[n] bool, swing_phase[n] float).
        swing_phase is 0.0 on stance feet — use the contact mask to distinguish."""
        p = self.local_phase(t)
        contact = p < self.duty
        # safe denom when duty==1 (always-stance leg) — value is masked out anyway
        denom = np.where(self.duty < 1.0, 1.0 - self.duty, 1.0)
        swing_phase = np.where(contact, 0.0, (p - self.duty) / denom)
        return contact, swing_phase

    def time_in_phase(self, t):
        """Elapsed time inside the foot's current stance- or swing-phase, per foot."""
        p = self.local_phase(t)
        contact = p < self.duty
        elapsed_frac = np.where(contact, p, p - self.duty)
        return elapsed_frac * self.period

    def time_to_next_event(self, t):
        """Time until the foot's next stance↔swing transition.
        Stance → time to lift-off;  Swing → time to touch-down."""
        p = self.local_phase(t)
        contact = p < self.duty
        boundary = np.where(contact, self.duty, 1.0)
        return (boundary - p) * self.period

    def stance_duration(self):
        return self.duty * self.period

    def swing_duration(self):
        return (1.0 - self.duty) * self.period


class FootstepPlanner:
    """Raibert-style next-touchdown predictor for an n-foot robot.

    For foot i, the world-frame touchdown position is
        p_step = R · hip_body_i + p_base
                 + 0.5 · v_world · T_stance_i
                 + k · (v_world − v_des_world)

    The first term anchors the step under the hip; the second is the symmetric
    midpoint of the next stance (mean foot-under-body drift); the third is a
    capture-point-like velocity-error correction (Raibert).

    The z coordinate carries through from the hip projection; terrain-aware z is
    the caller's job — anchor to a stance foot's FK z plus a relative elevation
    scan delta (see StepGenerator4.swing_foot_target for the recipe). The old
    optional env.get_z(x, y)+foot_radius snap was an absolute-world-z oracle (sim
    trick) and was removed 2026-06-06."""

    def __init__(self, hips_body, k=0.03):
        self.hips_body   = np.asarray(hips_body, dtype=float)   # (n, 3)
        self.k           = float(k)
        self.n           = self.hips_body.shape[0]

    def target(self, foot_id, R, p_base, v_world, v_des_world, T_stance):
        hip_world = R @ self.hips_body[foot_id] + p_base
        return hip_world + 0.5 * v_world * float(T_stance) + self.k * (v_world - v_des_world)


class BezierSwing:
    """Quadratic Bezier swing-foot trajectory: start → apex → end.

    apex = midpoint(start, end) + (0, 0, height)
    B(s)   = (1−s)²·start + 2(1−s)s·apex + s²·end          s ∈ [0, 1]
    B'(s)  = 2(1−s)·(apex − start) + 2s·(end − apex)
    v(s)   = B'(s) / T_swing"""

    def __init__(self, height=0.05):
        self.height = float(height)

    def _apex(self, p_start, p_end):
        return 0.5 * (np.asarray(p_start) + np.asarray(p_end)) + np.array([0.0, 0.0, self.height])

    def eval(self, s, p_start, p_end):
        s = float(np.clip(s, 0.0, 1.0))
        a = self._apex(p_start, p_end)
        return (1 - s)**2 * np.asarray(p_start) + 2 * (1 - s) * s * a + s**2 * np.asarray(p_end)

    def eval_vel(self, s, T_swing, p_start, p_end):
        s = float(np.clip(s, 0.0, 1.0))
        a = self._apex(p_start, p_end)
        dB = 2 * (1 - s) * (a - np.asarray(p_start)) + 2 * s * (np.asarray(p_end) - a)
        return dB / float(T_swing)


#---------------body-velocity estimators (floating-base robots)------------------------------------
# T1  ComplementaryEstimator — IMU strapdown ⊕ contact-weighted leg ZUPT, fixed α
# T2  ContactAidedEKF        — Bloesch-2013 reduced EKF on [v, b_a]
# T3  InvariantEKF           — Hartley-2020 contact-aided right-invariant EKF on SE_{K+2}(3) × R^6
#
# Common signature:  step(R, w, a, q, qd, f, foot_pos_body, foot_jac_body) → v_world
#
# Conventions (match the tact feed shapes):
#   R: 3×3 body→world rotation (e.g. from quat_to_rotation(y_quat))
#   w: body-frame angular velocity (gyro)
#   a: body-frame specific force (accelerometer; +g_body when stationary upright)
#   q, qd: joint state
#   f: per-foot 6-wrench in foot frame; |f[6i:6i+3]| used as contact magnitude
#   foot_pos_body: (3*nfeet,) body-frame foot positions, concatenated
#   foot_jac_body: (3*nfeet, nq) body-frame foot linear Jacobian (rows per foot)
#
# Single-reference-point contract (load-bearing):
#   All three estimators assume a single body-fixed reference point P. Inputs and outputs
#   must refer to the same P:
#     - R/w/a describe the kinematic state at P (not at the physical IMU mount site)
#     - foot_pos_body / foot_jac_body are expressed relative to P
#     - the returned v is the world velocity of P; GT for evaluation must be at P
#   By convention P = the floating-joint kinematic root (base origin in
#   fixed_base=True models). If the physical IMU is offset from P, the driver layer
#   is expected to transform raw readings to P before they reach the controller (R and w
#   are unchanged for a rigid body; a needs centripetal/tangential correction). The
#   estimators take no frame-offset parameter — they have no way to know about one. See
#   fg/dog/docs/imu-frame.md for the derivation and porting checklist.


class ComplementaryEstimator:
    def __init__(self, dt, alpha=0.05, contact_threshold=15.0,
                 g_world=(0.0, 0.0, -9.81), nfeet=4):
        self.dt = dt
        self.alpha = alpha
        self.contact_threshold = contact_threshold
        self.g_world = np.array(g_world, dtype=np.float64)
        self.nfeet = nfeet
        self.v = np.zeros(3)
        self.last = {}

    def reset(self, v0=None):
        self.v = np.zeros(3) if v0 is None else np.array(v0, dtype=np.float64).copy()

    def step(self, R, w, a, q, qd, f, foot_pos_body, foot_jac_body):
        v_imu = self.v + self.dt * (R @ a + self.g_world)

        ws = np.zeros(self.nfeet)
        v_legs = np.zeros((self.nfeet, 3))
        for i in range(self.nfeet):
            fmag = float(np.linalg.norm(f[6*i:6*i+3]))
            if fmag > self.contact_threshold:
                r  = foot_pos_body[3*i:3*i+3]
                Ji = foot_jac_body[3*i:3*i+3]
                v_legs[i] = -R @ (np.cross(w, r) + Ji @ qd)
                ws[i] = fmag

        tot = ws.sum()
        if tot > 0:
            v_leg = (ws[:, None] * v_legs).sum(axis=0) / tot
            self.v = (1.0 - self.alpha) * v_imu + self.alpha * v_leg
            self.last = dict(v_imu=v_imu, v_leg=v_leg, ws=ws, n_contact=int((ws > 0).sum()))
        else:
            self.v = v_imu
            self.last = dict(v_imu=v_imu, v_leg=None, ws=ws, n_contact=0)

        return self.v


class ContactAidedEKF:
    """Bloesch-2013-style EKF (R taken as known input — "imperfect EKF" form).

    State (6D):
        x[0:3] = v          body linear velocity in world frame
        x[3:6] = b_a        accelerometer bias in body frame (random walk)

    Predict (continuous-time → Euler):
        v̇      = R·(a - b_a) + g_world
        ḃ_a    = 0                                       (Brownian)
        F  = [[I,           -dt·R],
              [0,           I    ]]
        Q  = block_diag(σ_a²·dt² · I_3,  σ_ba²·dt · I_3)

    Measurement (per contact foot, ZUPT):
        v_meas_i = -R·(ω × r_i + J_i·qd)                  body in world frame
        h(x)     = v                                      → H = [I_3, 0_3]
        R_meas   = σ_kin² · I_3
        Sequential update across feet (each one Kalman-folded in turn).

    Tunable σ_a / σ_ba / σ_kin pick the IMU-vs-leg trust ratio. Initial P_b_a
    starts large so bias is allowed to grow from data.
    """
    def __init__(self, dt, contact_threshold=15.0, g_world=(0.0, 0.0, -9.81), nfeet=4,
                 sigma_a=0.2, sigma_ba=0.002, sigma_kin=0.05,
                 P0_v=1e-4, P0_ba=0.01, t_warmup=2000):
        self.dt = dt
        self.contact_threshold = contact_threshold
        self.g_world = np.array(g_world, dtype=np.float64)
        self.nfeet = nfeet
        self.sigma_a = sigma_a
        self.sigma_ba = sigma_ba
        self.sigma_kin = sigma_kin
        self.t_warmup = t_warmup            # steps: freeze bias updates until this
        self._step = 0
        self.x = np.zeros(6)
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P0_v * np.eye(3)
        # P_ba starts at 0; injected at t_warmup transition so learning is gated to
        # the post-settling phase, never to the touchdown-impact transient.
        self._P0 = (P0_v, P0_ba)
        self.last = {}

    def reset(self, v0=None):
        self.x[:] = 0.0
        if v0 is not None: self.x[0:3] = np.asarray(v0)
        P0_v, _ = self._P0
        self.P[:] = 0.0
        self.P[0:3, 0:3] = P0_v * np.eye(3)
        self._step = 0

    @property
    def v(self):       return self.x[0:3].copy()
    @property
    def b_a(self):     return self.x[3:6].copy()

    def step(self, R, w, a, q, qd, f, foot_pos_body, foot_jac_body):
        dt = self.dt
        self._step += 1
        bias_active = (self._step > self.t_warmup)

        v_pre = self.x[0:3]
        b_pre = self.x[3:6]

        # ---- Predict ----
        accel_world = R @ (a - b_pre) + self.g_world
        v_pred = v_pre + dt * accel_world
        x_pred = np.concatenate([v_pred, b_pre])

        F = np.eye(6)
        F[0:3, 3:6] = -dt * R

        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (self.sigma_a * dt)**2 * np.eye(3)
        if bias_active:
            Q[3:6, 3:6] = (self.sigma_ba)**2 * dt * np.eye(3)

        P = F @ self.P @ F.T + Q

        # Inject bias uncertainty at warmup boundary (one-shot) so post-warmup learning can start.
        if self._step == self.t_warmup + 1:
            P[3:6, 3:6] += self._P0[1] * np.eye(3)

        # ---- Update: per-foot sequential ZUPT ----
        R_meas = (self.sigma_kin**2) * np.eye(3)
        x = x_pred.copy()
        n_contact = 0
        for i in range(self.nfeet):
            fmag = float(np.linalg.norm(f[6*i:6*i+3]))
            if fmag <= self.contact_threshold:
                continue
            r  = foot_pos_body[3*i:3*i+3]
            Ji = foot_jac_body[3*i:3*i+3]
            v_meas = -R @ (np.cross(w, r) + Ji @ qd)

            y_inn = v_meas - x[0:3]
            S = P[0:3, 0:3] + R_meas
            K = P[:, 0:3] @ np.linalg.inv(S)          # (6, 3)
            if not bias_active:
                K[3:6, :] = 0.0                       # freeze b_a updates during warmup
            x = x + K @ y_inn
            I_KH = np.eye(6)
            I_KH[:, 0:3] -= K
            P = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
            n_contact += 1

        self.x = x
        self.P = P
        self.last = dict(n_contact=n_contact, b_a=self.x[3:6].copy(),
                         P_v_trace=float(np.trace(P[0:3, 0:3])),
                         P_ba_trace=float(np.trace(P[3:6, 3:6])),
                         bias_active=bias_active)
        return self.v


class InvariantEKF:
    """Contact-aided Right-Invariant EKF (Hartley et al., IJRR 2020).

    State on SE_{K+2}(3) × R^6:
      X = (R, v, p, d_1, ..., d_K)            pose, world-frame velocity, K foot positions
      θ = (b_g, b_a)                          IMU biases (body frame)

    Right-invariant error  ξ ∈ R^{9+3K+6}:
      [ξ_R, ξ_v, ξ_p, ξ_d_1, ..., ξ_d_K, ξ_bg, ξ_ba]   (world-frame perturbations)
      Group correction X = exp(ξ)·X̂  ⇒  R_new = exp([ξ_R]_×)·R̂,
                                        v_new = v̂ + [ξ_R]_× v̂ + ξ_v   (first-order),
                                        likewise p, d_i.   Biases in R^3 add directly.

    Process model (IMU-driven, continuous → Euler):
      Ṙ = R [w - b_g]_×,    v̇ = R(a - b_a) + g_world,    ṗ = v,
      ḋ_i = 0,              ḃ_g, ḃ_a random walk
    Error dynamics A_c (right-invariant, bias-coupled — "imperfect" InEKF):
      ξ̇_R   = -R̂ ξ_bg
      ξ̇_v   = [g]_× ξ_R - [v̂]_× R̂ ξ_bg - R̂ ξ_ba
      ξ̇_p   = ξ_v - [p̂]_× R̂ ξ_bg
      ξ̇_d_i = -[d̂_i]_× R̂ ξ_bg
    Discrete:  Φ = I + A_c·dt,  Q = block_diag(σ²·dt) mapped to right-invariant error.

    Measurement (per contact foot, body-frame FK z_i = foot_pos_body[3i:3i+3]):
      Right-invariant innovation (world frame):  ỹ_i = R̂ z_i - (d̂_i - p̂)
      Jacobian:  H_i = [0, 0, -I_3, 0, ..., +I_3 (foot-i slot), ..., 0, 0]    (state-independent)
      Noise:    N_i = σ_kin² I_3                  (isotropic — R̂ rotation cancels)

    Foot bookkeeping:
      On touchdown (was-off → now-on), reset d_i := p̂ + R̂·z_i and zero its P rows/cols
      (initial-P_d on diagonal). During swing the d_i state is dormant — no Q, no update.

    R is propagated internally from the gyro; the input R argument is used only on the
    first call to bootstrap orientation. Bias-channel Kalman corrections are gated by a
    warmup counter (mirrors T2), and bias-covariance is injected one-shot at the boundary.
    """
    def __init__(self, dt, contact_threshold=15.0, g_world=(0.0, 0.0, -9.81), nfeet=4,
                 sigma_g=0.01, sigma_a=0.2,
                 sigma_bg=1e-4, sigma_ba=2e-3,
                 sigma_slip=1e-3, sigma_kin=0.05,
                 P0_R=1e-4, P0_v=1e-4, P0_p=1e-4, P0_d=1e-2,
                 P0_bg=1e-4, P0_ba=1e-2,
                 t_warmup=2000):
        self.dt = dt
        self.ct = contact_threshold
        self.g  = np.array(g_world, dtype=np.float64)
        self.nfeet = nfeet
        self.sigma_g,  self.sigma_a  = sigma_g,  sigma_a
        self.sigma_bg, self.sigma_ba = sigma_bg, sigma_ba
        self.sigma_slip, self.sigma_kin = sigma_slip, sigma_kin
        self._P0 = (P0_R, P0_v, P0_p, P0_d, P0_bg, P0_ba)
        self.t_warmup = t_warmup
        self._step = 0
        # Error-vector layout
        self.dim = 9 + 3*nfeet + 6
        self.iR  = slice(0, 3)
        self.iv  = slice(3, 6)
        self.ip  = slice(6, 9)
        self.id  = [slice(9 + 3*i, 9 + 3*(i+1)) for i in range(nfeet)]
        self.ibg = slice(9 + 3*nfeet,     9 + 3*nfeet + 3)
        self.iba = slice(9 + 3*nfeet + 3, 9 + 3*nfeet + 6)
        # Mean state (R kept as matrix; rest as vectors)
        self.R  = np.eye(3)
        self.v_ = np.zeros(3)
        self.p_ = np.zeros(3)
        self.d_ = np.zeros((nfeet, 3))
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P  = np.zeros((self.dim, self.dim))
        self._set_P0()
        self.was_contact = np.zeros(nfeet, dtype=bool)
        self._initialized = False
        self.last = {}

    def _set_P0(self):
        P0_R, P0_v, P0_p, P0_d, _, _ = self._P0
        P = np.zeros((self.dim, self.dim))
        I3 = np.eye(3)
        P[self.iR, self.iR] = P0_R * I3
        P[self.iv, self.iv] = P0_v * I3
        P[self.ip, self.ip] = P0_p * I3
        for i in range(self.nfeet):
            P[self.id[i], self.id[i]] = P0_d * I3
        # Bias starts at 0; injected at the warmup boundary (mirrors T2).
        self.P = P

    def reset(self, v0=None, R0=None, p0=None):
        self.R  = np.eye(3) if R0 is None else np.asarray(R0, dtype=np.float64).copy()
        self.v_ = np.zeros(3) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.p_ = np.zeros(3) if p0 is None else np.asarray(p0, dtype=np.float64).copy()
        self.d_[:] = 0.0
        self.bg[:] = 0.0
        self.ba[:] = 0.0
        self._set_P0()
        self.was_contact[:] = False
        self._initialized = False
        self._step = 0

    @property
    def v(self):    return self.v_.copy()
    @property
    def b_a(self):  return self.ba.copy()
    @property
    def b_g(self):  return self.bg.copy()

    def step(self, R_input, w, a, q, qd, f, foot_pos_body, foot_jac_body):
        dt = self.dt
        self._step += 1
        bias_active = (self._step > self.t_warmup)
        I3 = np.eye(3)

        # ---- Bootstrap orientation and foot anchors from the first input ----
        if not self._initialized:
            self.R = np.asarray(R_input, dtype=np.float64).copy()
            for i in range(self.nfeet):
                self.d_[i] = self.p_ + self.R @ foot_pos_body[3*i:3*i+3]
                self.was_contact[i] = float(np.linalg.norm(f[6*i:6*i+3])) > self.ct
            self._initialized = True
            self.last = dict(n_contact=int(self.was_contact.sum()),
                             b_a=self.ba.copy(), b_g=self.bg.copy(),
                             bias_active=bias_active)
            return self.v

        # ---- Predict (mean) ----
        R_pre = self.R.copy()
        v_pre = self.v_.copy()
        p_pre = self.p_.copy()
        d_pre = self.d_.copy()

        omega = w - self.bg
        spec  = a - self.ba
        accel = R_pre @ spec + self.g

        self.R  = R_pre @ expmap_so3(omega * dt)
        self.v_ = v_pre + accel * dt
        self.p_ = p_pre + v_pre * dt + 0.5 * accel * dt**2
        # d_, bg, ba constant in predict mean

        # ---- A_c, Φ, Q (linearized at the *pre*-predict estimate) ----
        A  = np.zeros((self.dim, self.dim))
        gx = skew(self.g)
        vx = skew(v_pre)
        px = skew(p_pre)
        A[self.iR, self.ibg] = -R_pre
        A[self.iv, self.iR ] =  gx
        A[self.iv, self.ibg] = -vx @ R_pre
        A[self.iv, self.iba] = -R_pre
        A[self.ip, self.iv ] =  I3
        A[self.ip, self.ibg] = -px @ R_pre
        for i in range(self.nfeet):
            A[self.id[i], self.ibg] = -skew(d_pre[i]) @ R_pre

        Phi = np.eye(self.dim) + A * dt

        Q = np.zeros((self.dim, self.dim))
        Q[self.iR, self.iR] = (self.sigma_g**2) * dt * I3
        Q[self.iv, self.iv] = (self.sigma_a**2) * dt * I3
        for i in range(self.nfeet):
            if float(np.linalg.norm(f[6*i:6*i+3])) > self.ct:
                Q[self.id[i], self.id[i]] = (self.sigma_slip**2) * dt * I3
            # else: 0 — foot d_i is dormant during swing; reset at touchdown.
        if bias_active:
            Q[self.ibg, self.ibg] = (self.sigma_bg**2) * dt * I3
            Q[self.iba, self.iba] = (self.sigma_ba**2) * dt * I3

        P = Phi @ self.P @ Phi.T + Q

        # One-shot bias-covariance injection so post-warmup learning has room to move.
        if self._step == self.t_warmup + 1:
            _, _, _, _, P0_bg, P0_ba = self._P0
            P[self.ibg, self.ibg] += P0_bg * I3
            P[self.iba, self.iba] += P0_ba * I3

        # ---- Touchdown reset (after predict, before measurement) ----
        _, _, _, P0_d, _, _ = self._P0
        contact_now = np.array([float(np.linalg.norm(f[6*i:6*i+3])) > self.ct
                                for i in range(self.nfeet)])
        for i in range(self.nfeet):
            if contact_now[i] and not self.was_contact[i]:
                self.d_[i] = self.p_ + self.R @ foot_pos_body[3*i:3*i+3]
                P[self.id[i], :] = 0.0
                P[:, self.id[i]] = 0.0
                P[self.id[i], self.id[i]] = P0_d * I3

        # ---- Sequential per-foot Kalman update ----
        n_contact = 0
        for i in range(self.nfeet):
            if not contact_now[i]:
                self.was_contact[i] = False
                continue
            self.was_contact[i] = True

            z_meas = foot_pos_body[3*i:3*i+3]
            y_inn  = self.R @ z_meas - (self.d_[i] - self.p_)

            H = np.zeros((3, self.dim))
            H[:, self.ip   ] = -I3
            H[:, self.id[i]] =  I3

            N = (self.sigma_kin**2) * I3
            S = H @ P @ H.T + N
            K = P @ H.T @ np.linalg.inv(S)
            if not bias_active:
                K[self.ibg, :] = 0.0
                K[self.iba, :] = 0.0

            xi = K @ y_inn

            xi_R   = xi[self.iR]
            xi_R_x = skew(xi_R)
            self.v_ = self.v_ + xi_R_x @ self.v_ + xi[self.iv]
            self.p_ = self.p_ + xi_R_x @ self.p_ + xi[self.ip]
            for j in range(self.nfeet):
                self.d_[j] = self.d_[j] + xi_R_x @ self.d_[j] + xi[self.id[j]]
            self.bg = self.bg + xi[self.ibg]
            self.ba = self.ba + xi[self.iba]
            self.R  = expmap_so3(xi_R) @ self.R

            I_KH = np.eye(self.dim) - K @ H
            P = I_KH @ P @ I_KH.T + K @ N @ K.T
            n_contact += 1

        self.P = P
        self.last = dict(n_contact=n_contact, b_a=self.ba.copy(), b_g=self.bg.copy(),
                         P_R_trace =float(np.trace(P[self.iR,  self.iR])),
                         P_v_trace =float(np.trace(P[self.iv,  self.iv])),
                         P_bg_trace=float(np.trace(P[self.ibg, self.ibg])),
                         P_ba_trace=float(np.trace(P[self.iba, self.iba])),
                         bias_active=bias_active)
        return self.v


#---------------perception-----------------------------------------------------------------
# MiniElevationMap — robot-centric 2.5D elevation map (mapping only, no SLAM).
#
#   Minimal numpy sibling of ETH's elevation_mapping: a square grid of per-cell
#   (height, variance) 1-D Kalman filters that follows the robot, fuses world-frame
#   point clouds, inflates uncertainty as the robot moves (odometry drift), and is
#   sampled as a gravity-aligned, yaw-rotated height scan relative to the terrain
#   under the base — the live counterpart of an RL env's analytic height scan
#   (e.g. dog/rl/dog_stairs_env._height_scan).
#
#   Frame contract (caller-enforced, like the estimators above):
#     - insert() points are in a gravity-aligned world/odom frame (z up). The caller
#       applies sensor extrinsics and the pose estimate; the map never sees the robot
#       or the sensor, and takes no frame-offset parameter.
#     - the SAME pose source must feed insert() and sample(); the relative output
#       then cancels common-mode (z/yaw) drift of that pose source.
#     - self-filtering (removing the robot's own legs from the cloud) is the
#       caller's job — it needs the robot's FK, which the map deliberately lacks.


class MiniElevationMap:
    """Robot-centric 2.5D elevation map with per-cell 1-D Kalman fusion.

    Update loop:
        m.move_to(base_xy)                      # follow the robot (shift + drift inflation)
        m.insert(points_world)                  # per sensor frame (~10-30 Hz)
        h = m.sample(base_xy, yaw, offsets)     # per control tick (~50 Hz)

    sample() returns heights relative to the terrain under base_xy (world-z up),
    so absolute z drift of the pose source drops out. Cells never observed (or
    invalidated by the map shifting past them) fall back to `default` and are
    flagged False in self.last['valid'].
    """

    def __init__(self, cell_size=0.04, map_size=4.0, center=(0.0, 0.0),
                 sigma_meas=0.02, drift_rate=1e-3, gate=3.0):
        self.cell = float(cell_size)
        self.n = int(round(map_size / cell_size))
        self.sigma_meas = sigma_meas
        self.drift_rate = drift_rate        # height variance added per meter traveled
        self.gate = gate                    # innovation gate (in sigmas) -> cell re-init
        self.center = np.array(center, dtype=np.float64)   # world xy of grid center
        self._trail = self.center.copy()    # last move_to position (drift accounting)
        self.h = np.full((self.n, self.n), np.nan)          # cell height (world z)
        self.P = np.full((self.n, self.n), np.inf)          # cell height variance
        self.last = {}

    def reset(self, center=None):
        if center is not None:
            self.center = np.array(center, dtype=np.float64)
        self._trail = self.center.copy()
        self.h[:] = np.nan
        self.P[:] = np.inf

    @property
    def valid(self):
        return ~np.isnan(self.h)

    def move_to(self, base_xy):
        """Re-center the grid on the robot. Shifts by whole cells (cells scrolling
        off the far edge are invalidated) and inflates all variances by
        drift_rate * distance traveled since the last call."""
        c = np.asarray(base_xy, dtype=np.float64)[:2]
        dist = float(np.linalg.norm(c - self._trail))
        if dist > 0.0:
            self.P += self.drift_rate * dist
            self._trail = c.copy()
        shift = np.round((c - self.center) / self.cell).astype(int)
        if shift[0] == 0 and shift[1] == 0:
            return
        if abs(shift[0]) >= self.n or abs(shift[1]) >= self.n:
            self.center += shift * self.cell
            self.h[:] = np.nan
            self.P[:] = np.inf
            return
        for axis, s in enumerate(shift):
            if s == 0:
                continue
            self.h = np.roll(self.h, -s, axis=axis)
            self.P = np.roll(self.P, -s, axis=axis)
            sl = [slice(None)] * 2
            sl[axis] = slice(self.n - s, None) if s > 0 else slice(None, -s)
            self.h[tuple(sl)] = np.nan
            self.P[tuple(sl)] = np.inf
        self.center += shift * self.cell

    def insert(self, points_world, sigma=None):
        """Fuse a world-frame point cloud (N,3). Points sharing a cell are averaged
        into one measurement (noise shrinks with the count); each hit cell gets a
        scalar Kalman update. Innovations beyond `gate` sigmas re-initialize the
        cell (terrain changed / stale memory) instead of being averaged in."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return
        sigma = self.sigma_meas if sigma is None else sigma
        n = self.n
        gi = np.floor((pts[:, 0] - self.center[0]) / self.cell + 0.5 * n).astype(int)
        gj = np.floor((pts[:, 1] - self.center[1]) / self.cell + 0.5 * n).astype(int)
        m = (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)
        if not m.any():
            return
        flat = gi[m] * n + gj[m]
        cnt = np.bincount(flat, minlength=n * n)
        zsum = np.bincount(flat, weights=pts[m, 2], minlength=n * n)
        idx = np.flatnonzero(cnt)
        z = zsum[idx] / cnt[idx]                    # per-cell measurement
        # Averaging N same-frame points would shrink Rm as sigma^2/N, but same-frame
        # noise is partly correlated (pose error, calibration bias), so cap the
        # effective count — otherwise one dense frame (raycloud: tens of points per
        # cell near the sensor) drives K -> 1 and overwrites fused history with a
        # bias the gate can't see.
        Rm = sigma**2 / np.minimum(cnt[idx], 25)    # averaged-measurement variance

        h0 = self.h.flat[idx]
        P0 = self.P.flat[idx]
        fresh = np.isnan(h0)                        # never observed
        h0s = np.where(fresh, 0.0, h0)              # finite stand-ins keep the
        P0s = np.where(fresh, 1.0, P0)              # masked branches NaN/inf-free
        outlier = ~fresh & ((z - h0s)**2 > self.gate**2 * (P0s + Rm))
        reinit = fresh | outlier
        P0s = np.where(reinit, 1.0, P0s)
        K = P0s / (P0s + Rm)
        self.h.flat[idx] = np.where(reinit, z, h0s + K * (z - h0s))
        self.P.flat[idx] = np.where(reinit, Rm, (1.0 - K) * P0s)
        self.last = dict(n_points=int(m.sum()), n_cells=len(idx),
                         n_reinit=int(reinit.sum()))

    def _interp(self, wx, wy, max_variance=None):
        """Validity-weighted bilinear interpolation at world (wx, wy) arrays.
        Returns (heights, ok); a query is ok if any of its 4 corners is valid."""
        n = self.n
        gx = (wx - self.center[0]) / self.cell + 0.5 * n - 0.5
        gy = (wy - self.center[1]) / self.cell + 0.5 * n - 0.5
        i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - i0, gy - j0
        ok_cell = self.valid if max_variance is None else \
            (self.valid & (self.P < max_variance))
        hsum = np.zeros_like(gx)
        wsum = np.zeros_like(gx)
        for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy),       (1, 1, fx * fy)):
            i, j = i0 + di, j0 + dj
            inb = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            ic, jc = np.clip(i, 0, n - 1), np.clip(j, 0, n - 1)
            wv = w * inb * ok_cell[ic, jc]
            hsum += wv * np.where(np.isnan(self.h[ic, jc]), 0.0, self.h[ic, jc])
            wsum += wv
        ok = wsum > 1e-9
        return np.where(ok, hsum / np.where(ok, wsum, 1.0), np.nan), ok

    def sample(self, base_xy, yaw, offsets, default=0.0, max_variance=None):
        """Gravity-aligned height scan: rotate (G,2) yaw-frame offsets to world,
        interpolate the map, subtract the height under base_xy. Mirrors the RL
        env's _height_scan(). Invalid points return `default` ("same height as
        under the base"); per-point validity lands in self.last['valid']. If the
        under-base cell itself is invalid, the reference falls back to 0 absolute
        (self.last['base_valid'] False) — heights are then absolute, not relative."""
        off = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        bx, by = float(base_xy[0]), float(base_xy[1])
        c, s = np.cos(yaw), np.sin(yaw)
        wx = bx + c * off[:, 0] - s * off[:, 1]
        wy = by + s * off[:, 0] + c * off[:, 1]
        h, ok = self._interp(wx, wy, max_variance)
        h0, ok0 = self._interp(np.array([bx]), np.array([by]), max_variance)
        base_valid = bool(ok0[0])
        ref = float(h0[0]) if base_valid else 0.0
        out = np.where(ok, h - ref, default)
        self.last = dict(valid=ok, base_valid=base_valid, ref=ref,
                         n_valid=int(ok.sum()))
        return out
