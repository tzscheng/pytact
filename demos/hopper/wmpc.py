"""WBC+MPC hopping orchestrator — state '2'.

Byte-faithful to the backup at `_/mk1.py` (the controller stack from before the
OSQP/WbcQp work landed). Kept side-by-side with `wmpc2.py` while the user
studies the new pipeline. Uses `Mpc` (scipy L-BFGS-B) + the legacy `Wbc`
class (J^T·K·e foot PD + hip-attitude PD), both defined locally here. No
AHRS-offset fix on qd, no WbcQp.

Controller routes `zmqmsg 2` → `Hop.update(...)` here.
"""

import numpy as np, tact
from scipy.optimize import minimize


class Wbc:
    """Floating-base whole-body controller — original J^T·K·e formulation.

    Built on the proper full-DoF model (`tact.Model(..., fixed_base=False)`),
    so kinematics and dynamics treat the head as a free body — no fixed-base
    approximation. The actuated joints sit at indices 6..8 of the 9-DoF state.

    Control law: Cartesian PD on the foot task, expressed in WORLD frame:

        e_world  = x_d_world - fk_world(q_full)        # 3-vec
        J_full   = jacob(q_full)                       # 3 x 9 (world frame)
        tau_full = J_full^T @ (K_p · e_world)          # 9-vec
        tau_act  = tau_full[6:9]                       # apply to the 3 actuators

    Why world-frame and full-DoF:
      - A body-frame target tied to a tilting body rotates with the floating
        base; foot pinned during stance -> WBC chases a moving target -> body
        torques amplify tilt. World-frame target stays inertial: when the body
        tilts, the foot-position error grows in a direction whose J^T mapping
        gives an *uprighting* hip torque (passive attitude regulator).
      - The actuated columns J_full[:, 6:9] equal R @ J_body_frame, so JT in
        world is numerically identical to JT in body frame on a (R^T·e_world)
        — but using the full model is the cleaner formulation for any future
        extension to full inverse dynamics (M·qdd + h with proper floating-
        base inertia), body-attitude tasks, or contact-constrained QPs.

    Stance fire phase overrides the knee torque (last actuator) with the
    MPC-prescribed push-off force.
    """

    def __init__(self, m_fb, ee):
        self.m = m_fb
        self.ee = ee
        self.K_fly = np.array([50.0, 50.0, 50.0])
        self.K_stn = np.array([50.0, 50.0, 50.0])
        # Body-attitude PD during stance (the standard Raibert attitude loop).
        # Hip-yaw (q_act[0]) rotates the leg about body-x  -> regulates body ROLL.
        # Hip-pitch (q_act[1]) rotates the leg about body-y -> regulates body PITCH.
        # Signs assume: positive joint torque on link applies NEGATIVE reaction on
        # head about the joint axis (Featherstone convention). If the body diverges
        # instead of righting, flip the signs of K_att.
        self.K_att = np.array([20.0, 20.0])        # roll-/pitch-angle gains
        self.K_dmp = np.array([ 8.0,  8.0])        # roll-/pitch-rate damping (raised
                                                    # to suppress residual body oscillation
                                                    # that couples MPC overshoot into x-y axis)
        self.tau_max = 60.0

    def _jt(self, q_full, x_d_world, K):
        x_foot = self.m.fk(self.ee, q_full)          # 3-vec, world
        J      = self.m.jacob(self.ee, q_full)       # 3 x 9
        e      = x_d_world - x_foot
        tau    = J.T @ (K * e)                       # 9-vec
        return np.clip(tau[6:9], -self.tau_max, self.tau_max)

    def flight(self, q_full, x_d_world):
        return self._jt(q_full, x_d_world, self.K_fly)

    def stance(self, q_full, x_d_world, R, omega_body, fire_force=0.0):
        tau = self._jt(q_full, x_d_world, self.K_stn)
        # Body roll  ≈  R[2,1]  (positive when body-z leans toward +y world).
        # Body pitch ≈ -R[2,0]  (positive when body-z leans toward +x world, i.e. forward).
        roll_err  =  R[2, 1]
        pitch_err = -R[2, 0]
        tau[0] += self.K_att[0]*roll_err  + self.K_dmp[0]*omega_body[0]
        tau[1] += self.K_att[1]*pitch_err + self.K_dmp[1]*omega_body[1]
        if fire_force != 0.0:
            tau[2] = fire_force
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau


class Mpc:
    """Step-to-step SLIP predictor — original scipy.optimize.minimize version.

    Reduced state per hop apex k: (vx_k, vy_k, h_k) in body-heading frame.
    Discrete dynamics (linear-in-decision):
        vx_{k+1} = vx_k - alpha*(fx_k - vx_k*T_s/2)
        vy_{k+1} = vy_k - alpha*(fy_k - vy_k*T_s/2)
        h_{k+1}  = (1-loss)*h_k + (Iz_k * v_ref) / (mass * g)
    The vertical map comes from energy: m*g*Δh = work_thrust = Iz * v_avg, with
    v_avg linearized at the desired take-off speed v_ref = sqrt(2*g*h_des).
    Decision: u_k = (fx_k, fy_k, Iz_k), k=0..N-1. Only u_0 is applied; the rest
    is horizon padding so the optimizer accounts for the next few hops.
    """

    def __init__(self, N=3):
        self.N = N
        self.u_prev = np.zeros(3*N)
        self.w_v = 10.0     # velocity-tracking weight
        self.w_h = 200.0    # apex-height weight
        self.w_u = 0.30     # control regularizer (high -> prefers small foot moves)
        self.fmax = 0.05    # foot placement bound [m]
        self.izmax = 1.5    # knee impulse bound [N*s]
        self.hop_loss = 0.05

    def solve(self, v_body, h_apex, v_des, h_des, T_s, T_fire, mass, g, alpha):
        N = self.N
        v_ref = np.sqrt(2*g*max(h_des, 0.05))
        lift = v_ref / (mass * g)
        loss = self.hop_loss

        def cost(u_flat):
            u = u_flat.reshape(N, 3)
            vx, vy, h = float(v_body[0]), float(v_body[1]), float(h_apex)
            J = 0.0
            for k in range(N):
                fx, fy, Iz = u[k]
                vx_n = vx - alpha*(fx - vx*T_s*0.5)
                vy_n = vy - alpha*(fy - vy*T_s*0.5)
                h_n  = (1.0 - loss)*h + Iz*lift
                J += self.w_v*((vx_n - v_des[0])**2 + (vy_n - v_des[1])**2)
                J += self.w_h*(h_n - h_des)**2
                J += self.w_u*(fx*fx + fy*fy + 0.5*Iz*Iz)
                vx, vy, h = vx_n, vy_n, h_n
            return J

        bounds = []
        for _ in range(N):
            bounds += [(-self.fmax, self.fmax), (-self.fmax, self.fmax), (0.0, self.izmax)]

        # Warm-start: shift previous solution by one hop, repeat last
        u0 = np.roll(self.u_prev.reshape(N, 3), -1, axis=0)
        u0[-1] = u0[-2]

        res = minimize(cost, u0.flatten(), bounds=bounds,
                       method='L-BFGS-B', options={'maxiter': 20, 'ftol': 1e-6})
        self.u_prev = res.x
        u_opt = res.x.reshape(N, 3)

        foot = u_opt[0, :2].copy()
        fire_force = u_opt[0, 2] / T_fire
        return foot, fire_force


class Hop:
    """Original WBC+MPC orchestrator. Mirrors HopRaibert's interface.

    Identical to backup `_/mk1.py:Hop_wmpc` — uses Mpc + Wbc directly, no
    WbcQp toggle, no AHRS-offset velocity correction.
    """

    # Plant tunables (matched to hopper.yaml: spring 2000, knee damping ~5)
    T_STANCE = 0.060   # s (60 ticks at dt=1ms)
    T_FIRE   = 0.030   # s (30 ticks)
    LEG_LEN  = 0.50    # nominal leg extension (stance + descent) [m]
    LEG_FLIGHT = 0.40  # retracted leg target during ascent (foot ground clearance) [m]
    M_BODY   = 1.4     # head+leg total mass [kg]
    G        = 9.81
    ALPHA    = 4.0     # Raibert step-to-step momentum-exchange gain

    # ahrs frame offset on the head body (used for q_full position only here)
    AHRS_OFFSET_BODY = np.array([0.0, 0.0, 0.10])

    def __init__(self, m_fb, env, ee):
        self.m, self.ee = m_fb, ee
        self.mpc = Mpc(N=3)
        self.wbc = Wbc(m_fb, ee)

        self.v_des = np.zeros(2)     # body-frame target horiz velocity [m/s]
        self.h_des = 0.20            # target apex height [m]

        # Per-cycle plan (refreshed when entering fly)
        self.plan_foot = np.zeros(2)
        self.plan_fire_force = 20.0
        self.hop_count = 0

        self.flag = None
        self.shift(2)                # start in fire so the first cycle bootstraps cleanly

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1

    V_DES_MAX = 0.30    # cap body-frame velocity command [m/s]

    def msgproc(self, w):
        if   w[0] == 'a': self.v_des[0] += 0.10
        elif w[0] == 'd': self.v_des[0] -= 0.10
        elif w[0] == 'w': self.v_des[1] -= 0.10
        elif w[0] == 's': self.v_des[1] += 0.10
        elif w[0] == 'e': self.v_des[:] = 0.0
        elif w[0] == '1': self.h_des = max(0.05, self.h_des - 0.03)
        elif w[0] == '2': self.h_des += 0.03
        self.v_des = np.clip(self.v_des, -self.V_DES_MAX, self.V_DES_MAX)
        print('wmpc_v2: v_des=%s  h_des=%.3f' %(self.v_des, self.h_des))

    @staticmethod
    def _world_xy_to_body(R, v_xy):
        bx = np.array([R[0, 0], R[1, 0]])
        by = np.array([R[0, 1], R[1, 1]])
        return np.array([bx @ v_xy, by @ v_xy])

    def _apex_h_estimate(self, vz):
        return vz*vz/(2.0*self.G)

    def _build_q_full(self, q_act, R, p_ahrs):
        p_head  = p_ahrs - R @ self.AHRS_OFFSET_BODY
        rot_vec = tact.logmap_so3(R)
        return np.concatenate([p_head, rot_vec, q_act])

    def update(self, q_act, qd_act, f, R, v_world, omega_body, p_ahrs):
        q_full = self._build_q_full(q_act, R, p_ahrs)
        p_head = q_full[:3]

        if self.s == 0:                                         # ---- fly
            if self.t == 0:
                v_body = self._world_xy_to_body(R, v_world[:2])
                h0 = self._apex_h_estimate(v_world[2])
                self.plan_foot, self.plan_fire_force = self.mpc.solve(
                    v_body, h0, self.v_des, self.h_des,
                    self.T_STANCE, self.T_FIRE, self.M_BODY, self.G, self.ALPHA)
                self.hop_count += 1
                #print('hop %3d  v_des=(%+0.3f,%+0.3f)  v=(%+0.3f,%+0.3f)  vz=%+0.3f  foot=(%+0.3f,%+0.3f)  fire=%5.2f  h0=%0.3f  R22=%.3f'
                #      % (self.hop_count,
                #         self.v_des[0], self.v_des[1],
                #         v_body[0],     v_body[1],
                #         v_world[2],
                #         self.plan_foot[0], self.plan_foot[1],
                #         self.plan_fire_force, h0, R[2, 2]))
            # Retract the leg during ascent so the foot clears the ground, then
            # re-extend on descent to prep for landing. (Whether the foot-PD can
            # actually win against the passive knee spring is verified in the trace.)
            leg = self.LEG_FLIGHT if v_world[2] > 0.0 else self.LEG_LEN
            x_d_world = p_head + R[:, 0]*self.plan_foot[0] \
                               + R[:, 1]*self.plan_foot[1] \
                               + np.array([0.0, 0.0, -leg])
            tau = self.wbc.flight(q_full, x_d_world)
            # Only accept a landing while descending — a foot still skimming the
            # ground right after push-off (body rising, vz>0) was falsely tripping
            # f[2]>10 and re-entering 'land', corrupting the hop cycle.
            if f[2] > 10 and v_world[2] < 0.0: self.next_s = 1

        elif self.s == 1:                                       # ---- land/compress
            x_d_world = p_head + np.array([0.0, 0.0, -self.LEG_LEN])
            tau = self.wbc.stance(q_full, x_d_world, R, omega_body)
            if self.t == int(self.T_STANCE*1000): self.next_s = 2

        elif self.s == 2:                                       # ---- fire/push-off
            x_d_world = p_head + np.array([0.0, 0.0, -self.LEG_LEN])
            ff = self.plan_fire_force if self.t < int(self.T_FIRE*1000) else 0.0
            tau = self.wbc.stance(q_full, x_d_world, R, omega_body, fire_force=ff)
            if f[2] < 1: self.next_s = 0

        self.one_step_forward()
        return tau
