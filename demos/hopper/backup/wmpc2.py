"""WBC+MPC hopping orchestrator — state '3'.

Current active work surface. OSQP-based step-to-step MPC and a hybrid WBC
(`WbcQp` does the dyn/contact/friction/torque-box QP, then the actuated tau is
finalized as legacy J^T·K·e + attitude PD feed-forward). Detailed lessons in
MEMORY.md (`project-wbc-mpc-state2`). Self-contained: the pure-legacy J^T·K·e
fallback used to live here behind a `use_qp` flag, but it's now in wmpc.py
under state '2', so this file just runs the hybrid path.

Controller routes `zmqmsg 3` → `Hop.update(...)` here.
"""

import numpy as np, tact
import osqp
import scipy.sparse as sp

# (Legacy `Wbc` is no longer used in this file — if you want the pure
# J^T·K·e path, run state '2' which lives in wmpc.py.)


# ---------------------------------------------------------------------------
# MPC: convex QP step-to-step predictor (was scipy L-BFGS-B in wmpc.Mpc).
# ---------------------------------------------------------------------------

class Mpc:
    """Step-to-step SLIP predictor — convex QP solved with OSQP.

    Reduced state per hop apex k: (vx_k, vy_k, h_k) in body-heading frame.
    Discrete dynamics (linear-in-decision):
        vx_{k+1} = a_v*vx_k + b_v*fx_k       a_v = 1 + alpha*T_s/2,  b_v = -alpha
        vy_{k+1} = a_v*vy_k + b_v*fy_k
        h_{k+1}  = a_h*h_k  + b_h*Iz_k       a_h = 1 - loss,         b_h = lift
    The vertical map comes from energy: m*g*Δh = Iz * v_avg, linearized at the
    desired take-off speed v_ref = sqrt(2*g*h_des), so lift = v_ref / (m*g).

    Decision: u_k = (fx_k, fy_k, Iz_k), k=0..N-1, interleaved as
        x = [fx_0, fy_0, Iz_0, ..., fx_{N-1}, fy_{N-1}, Iz_{N-1}]  (length 3N).
    Only u_0 is applied; remaining hops are horizon padding.

    QP form  (min 0.5 x^T P x + q^T x  s.t.  l ≤ x ≤ u):
        - vx/vy/h dynamics are decoupled, so P is block-sparse on the
          interleaved layout: fx-fx, fy-fy, Iz-Iz only.
        - Box bounds via A = I.
        - Sparsity pattern is invariant across solve() calls (depends only on
          N), so the OSQP workspace is set up once and refreshed each cycle
          with .update(Px=..., q=...).
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

        # OSQP workspace (lazy-init on first solve, since dimensions are fixed
        # by N but the numeric P depends on runtime params).
        self._osqp = None

    def _build_P_q(self, v_body, h_apex, v_des, h_des, T_s, mass, g, alpha):
        """Assemble dense P (3N x 3N) and q (3N) for the current step."""
        N = self.N
        n = 3*N
        v_ref = np.sqrt(2*g*max(h_des, 0.05))
        lift = v_ref / (mass * g)
        loss = self.hop_loss

        a_v = 1.0 + alpha*T_s*0.5
        b_v = -alpha
        a_h = 1.0 - loss
        b_h = lift

        # Per-axis propagation: v_{1:N} = M @ u_axis + c_axis  (column-vector).
        Mv = np.zeros((N, N)); Mh = np.zeros((N, N))
        cv_x = np.zeros(N); cv_y = np.zeros(N); ch = np.zeros(N)
        for k in range(N):
            cv_x[k] = (a_v**(k+1)) * float(v_body[0])
            cv_y[k] = (a_v**(k+1)) * float(v_body[1])
            ch[k]   = (a_h**(k+1)) * float(h_apex)
            for j in range(k+1):
                Mv[k, j] = (a_v**(k-j)) * b_v
                Mh[k, j] = (a_h**(k-j)) * b_h

        Hv  = 2.0*self.w_v * (Mv.T @ Mv)
        gvx = 2.0*self.w_v * (Mv.T @ (cv_x - v_des[0]))
        gvy = 2.0*self.w_v * (Mv.T @ (cv_y - v_des[1]))
        Hh  = 2.0*self.w_h * (Mh.T @ Mh)
        gh  = 2.0*self.w_h * (Mh.T @ (ch - h_des))

        P = np.zeros((n, n))
        q = np.zeros(n)
        for k in range(N):
            for j in range(N):
                P[3*k+0, 3*j+0] += Hv[k, j]   # fx-fx block (vx tracking)
                P[3*k+1, 3*j+1] += Hv[k, j]   # fy-fy block (vy tracking, same dyn)
                P[3*k+2, 3*j+2] += Hh[k, j]   # Iz-Iz block (h  tracking)
            q[3*k+0] = gvx[k]
            q[3*k+1] = gvy[k]
            q[3*k+2] = gh[k]
            # Control regularizer w_u*(fx^2 + fy^2 + 0.5*Iz^2)
            P[3*k+0, 3*k+0] += 2.0*self.w_u
            P[3*k+1, 3*k+1] += 2.0*self.w_u
            P[3*k+2, 3*k+2] += self.w_u
        return P, q

    def _bounds(self):
        N = self.N
        l = np.empty(3*N); u = np.empty(3*N)
        for k in range(N):
            l[3*k+0], u[3*k+0] = -self.fmax, self.fmax
            l[3*k+1], u[3*k+1] = -self.fmax, self.fmax
            l[3*k+2], u[3*k+2] = 0.0, self.izmax
        return l, u

    def solve(self, v_body, h_apex, v_des, h_des, T_s, T_fire, mass, g, alpha):
        N = self.N
        P, q = self._build_P_q(v_body, h_apex, v_des, h_des, T_s, mass, g, alpha)
        # OSQP expects upper-triangular CSC for P; storing the full symmetric
        # matrix gives an unstable sparsity pattern across calls (mirror
        # entries flip between numerically-zero and not).
        P_csc = sp.triu(sp.csc_matrix(P), format='csc')

        if self._osqp is None:
            l, u = self._bounds()
            A_csc = sp.csc_matrix(np.eye(3*N))
            self._osqp = osqp.OSQP()
            self._osqp.setup(P_csc, q, A_csc, l, u,
                             warm_starting=True, verbose=False,
                             eps_abs=1e-7, eps_rel=1e-7, polishing=False,
                             max_iter=4000)
        else:
            # Sparsity is stable; update values + linear cost in place.
            self._osqp.update(Px=P_csc.data, q=q)

        # Warm-start: shift previous solution by one hop, repeat last
        u0 = np.roll(self.u_prev.reshape(N, 3), -1, axis=0)
        u0[-1] = u0[-2]
        self._osqp.warm_start(x=u0.flatten())

        res = self._osqp.solve()
        if res.info.status_val == 1:  # OSQP_SOLVED
            self.u_prev = res.x
            u_opt = res.x.reshape(N, 3)
        else:
            # Solver failed or hit max_iter — reuse the previous plan rather
            # than feeding the controller garbage.
            u_opt = self.u_prev.reshape(N, 3)

        foot = u_opt[0, :2].copy()                  # body-frame xy
        fire_force = u_opt[0, 2] / T_fire           # N (force during fire window)
        return foot, fire_force


# ---------------------------------------------------------------------------
# Whole-body controller: hybrid QP + legacy feed-forward.
# ---------------------------------------------------------------------------

class WbcQp:
    """Weighted single-QP whole-body controller for hopper (OSQP backend).

    A proper inverse-dynamics WBC: instead of computing torque as J^T*K*e and
    bolting attitude PD on the hip torques (as the legacy `Wbc` does), we solve
    one QP per tick whose decision variable is the *physical* generalized state
    consistent with the floating-base dynamics, contact, and actuation limits.

    Decision: x = [qdd (9), tau (3), lambda (3)]                   (length 15)
        qdd     — generalized accelerations of the 9-DoF model
        tau     — actuated joint torques (hip-yaw, hip-pitch, knee)
        lambda  — foot contact wrench at the point contact, world frame

    Equalities (12):
        M(q)*qdd + h(q,qd) = S^T*tau + J^T*lambda            (9, S^T = e_{6:9})
        J(q)*qdd = 0                                         (3, stance only;
                                                              slacked in flight)
        Jdot*qd is dropped — second-order effect, small near upright at <1m/s.

    Inequalities (10):
        -tau_max ≤ tau ≤ tau_max                             (3 box)
        Friction cone (linearized): |λx|≤μλz, |λy|≤μλz, λz≥0  (4+1)
        Lambda is forced to zero in flight via its box.

    Cost (weighted single-priority):
        w_foot * ||J*qdd - acc_foot_des||^2
          + w_att  * ((qdd[3]-α_roll_des)^2 + (qdd[4]-α_pitch_des)^2)
          + (fire phase) w_fire * (tau[2] - fire_force)^2
          + tiny qdd/tau/lambda regularizers (conditioning).
    acc_foot_des = Kp_foot*(x_d - fk) - Kd_foot*(J*qd)        [Cartesian PD]
    α_*_des      = -K_att*R[2,·] - K_dmp*omega_body            [attitude PD]

    Notes:
        - tact's free joint convention: qd[0:3] = body-frame linvel,
          qd[3:6] = body-frame omega. The caller must convert v_world via
          R^T*v_world for qd_full[0:3].
        - The QP is rebuilt and re-setup each tick; OSQP setup() on this size
          (15 vars, 22 rows) is ~100µs which is fine for 1kHz.
    """

    NQ, NU, NL = 9, 3, 3
    NX = NQ + NU + NL          # 15
    NA = 9 + 3 + 3 + 3 + 4     # dyn(9) + contact(3) + tau-box(3) + lam-box(3) + friction(4)

    def __init__(self, m_fb, ee):
        self.m, self.ee = m_fb, ee

        # Cartesian foot PD (desired-acceleration generator). Flight Kp is
        # kept moderate so the leg swing doesn't kick the body around via
        # reaction torque — high foot Kp turns into a body-pitch disturbance
        # that the attitude task then has to fight.
        self.Kp_foot_fly = 100.0
        self.Kd_foot_fly = 20.0
        self.Kp_foot_stn = 400.0
        self.Kd_foot_stn = 40.0

        # Raibert braking gain (legacy-style J^T·K·e feed-forward, applied on
        # top of the QP solution during stance only). Matches legacy K_stn
        # exactly — including the z component, which provides per-tick
        # damping that's essential for lateral stability at non-zero v_des.
        # Without z-axis braking the hop height equilibrates higher (close
        # to h_des), but the longer flight time lets the body translate
        # further per cycle than MPC can correct, leading to divergence.
        self.K_brake = np.array([50.0, 50.0, 50.0])

        # Body-attitude PD (desired body-frame angular acceleration).
        # Drives R[2,1] (roll-like) and R[2,0] (pitch-like) toward zero.
        # Kept at legacy levels to preserve the *natural body lean* that the
        # passive Raibert mechanism needs during stance. Strong attitude
        # correction (we tried K_att=500) suppresses lean, which kills the
        # gravity-redirected braking and turns the hop unstable.
        self.K_att = np.array([20.0, 20.0])
        self.K_dmp = np.array([ 8.0,  8.0])

        # Limits
        self.tau_max     = 60.0
        self.mu_friction = 0.7

        # Cost weights. Stance and flight use different posture profiles
        # because stance is dominated by the *passive* pogo-spring dynamics
        # (the controller's job is mostly attitude, not body translation),
        # whereas flight has no contact and needs active foot placement.
        self.w_foot_fly  =  50.0     # flight: foot must track MPC target
        self.w_foot_stn  =   0.0     # stance: foot is pinned by contact equality → redundant
        self.w_att       = 200.0     # attitude tracking on qdd[3], qdd[4]
        self.w_fire      = 200.0     # knee fire-force tracking (fire window)
        # Asymmetric regularizers: tau is "expensive", lambda is "free". This
        # tells the QP to prefer letting the passive ground reaction support
        # the body (lambda absorbs the load), only spending active torque
        # when an attitude or fire task demands it. Symmetric weights would
        # split the load between tau and lambda and so cause the active
        # controller to fight the passive spring force in stance.
        self.w_reg_tau   = 1.0
        self.w_reg_lam   = 1e-4

        # Posture cost on qdd. In STANCE body-translation entries are zero so
        # the body is free to follow the passive spring (otherwise the QP
        # fights the spring's restoring force, gets clipped at tau_max, and
        # pumps energy into the hop). Joint posture in stance is lightly
        # anchored so the hip joints don't spin out when the attitude task
        # commands aggressive angular accel. In FLIGHT the leg joints are
        # anchored more firmly while body translates ballistically.
        self.w_post_stance = np.array([0.0, 0.0, 0.0,
                                       0.0, 0.0, 1.0,
                                       0.5, 0.5, 0.0])
        self.w_post_flight = np.array([0.0, 0.0, 0.0,
                                       0.0, 0.0, 1.0,
                                       1.0, 1.0, 1.0])

    def _build(self, q_full, qd_full, x_d_world,
               phase, fire_force, Kp_foot, Kd_foot,
               roll_des_accel, pitch_des_accel,
               w_foot, w_post, w_att):
        M = self.m.inertia(q_full)
        h = self.m.bias(q_full, qd_full).copy()
        # tact's bias() returns gravity + Coriolis only. The simulator's
        # aba_featherstone() additionally applies joint damping (ff*qd) and
        # joint spring (sk*q) implicitly inside its integrator. For the QP
        # dynamics to match the plant, fold them into the effective bias:
        #   true_applied = tau_active - ff*qd - sk*q  →  move to bias side:
        #   M*qdd + (h + ff*qd + sk*q) = S^T*tau_active + J^T*lambda
        # hopper: ff[6:9]=[0.8, 0.8, 5.0], sk[8]=2000 (knee pogo spring).
        h += self.m.ff * qd_full + self.m.sk * q_full
        J = self.m.jacob(self.ee, q_full)
        x_foot = self.m.fk(self.ee, q_full)
        v_foot = J @ qd_full

        n_x = self.NX

        # ----- constraint matrix A and bounds (l, u) -----
        A = np.zeros((self.NA, n_x))
        l = np.full(self.NA, -np.inf)
        u = np.full(self.NA, +np.inf)

        # Dynamics (rows 0..8):  M*qdd - S^T*tau - J^T*lambda = -h
        A[0:9, 0:9]    = M
        A[6:9, 9:12]   = -np.eye(3)        # -S^T (actuated rows only)
        A[0:9, 12:15]  = -J.T
        l[0:9] = -h; u[0:9] = -h

        # Contact (rows 9..11): J*qdd = 0 (stance), free in flight
        A[9:12, 0:9] = J
        if phase == 'stance':
            l[9:12] = 0.0; u[9:12] = 0.0

        # Tau box (rows 12..14)
        A[12:15, 9:12] = np.eye(3)
        l[12:15] = -self.tau_max
        u[12:15] = +self.tau_max

        # Lambda box (rows 15..17)
        A[15:18, 12:15] = np.eye(3)
        if phase == 'stance':
            l[15] = -np.inf; u[15] = +np.inf
            l[16] = -np.inf; u[16] = +np.inf
            l[17] = 0.0;     u[17] = +np.inf
        else:                              # flight: zero contact wrench
            l[15:18] = 0.0; u[15:18] = 0.0

        # Friction cone (rows 18..21), linearized: ±λx - μλz ≤ 0, ±λy - μλz ≤ 0
        mu = self.mu_friction
        A[18, 12:15] = [+1.0,  0.0, -mu]
        A[19, 12:15] = [-1.0,  0.0, -mu]
        A[20, 12:15] = [ 0.0, +1.0, -mu]
        A[21, 12:15] = [ 0.0, -1.0, -mu]
        if phase == 'stance':
            l[18:22] = -np.inf; u[18:22] = 0.0
        # flight: leave at ±inf (trivially satisfied since lambda=0)

        # ----- objective P, q -----
        P = np.zeros((n_x, n_x))
        q = np.zeros(n_x)

        # Foot task (active in flight; redundant in stance because the contact
        # equality already pins J*qdd, so we skip the cost there to keep the
        # Hessian conditioning clean).
        acc_des = Kp_foot*(x_d_world - x_foot) - Kd_foot*v_foot
        if w_foot > 0.0:
            P[0:9, 0:9] += 2.0*w_foot * (J.T @ J)
            q[0:9]      += -2.0*w_foot * (J.T @ acc_des)

        # Attitude task (penalty on qdd[3], qdd[4]). Caller may pass w_att=0
        # to disable this entirely — e.g. stance, where the attitude PD is
        # applied as a feed-forward tau instead so the body retains the
        # natural lean needed by the Raibert step-to-step mechanism.
        if w_att > 0.0:
            P[3, 3] += 2.0*w_att
            P[4, 4] += 2.0*w_att
            q[3]    += -2.0*w_att * roll_des_accel
            q[4]    += -2.0*w_att * pitch_des_accel

        # Fire-force task (knee torque target during the fire window)
        if fire_force != 0.0:
            P[11, 11] += 2.0*self.w_fire
            q[11]     += -2.0*self.w_fire * fire_force

        # Posture cost on qdd (per-DoF). In stance this is all zeros so the
        # body is free to follow the passive spring; in flight the leg joints
        # are lightly anchored so the leg posture stays well-defined.
        idx_qdd = np.arange(0, 9)
        idx_tau = np.arange(9, 12)
        idx_lam = np.arange(12, 15)
        if np.any(w_post > 0.0):
            P[idx_qdd, idx_qdd] += 2.0*w_post
        # Small regularizers on tau and lambda (numerical conditioning only)
        P[idx_tau, idx_tau] += 2.0*self.w_reg_tau
        P[idx_lam, idx_lam] += 2.0*self.w_reg_lam

        return P, q, A, l, u

    def _solve(self, P, q, A, l, u):
        P_csc = sp.triu(sp.csc_matrix(P), format='csc')
        A_csc = sp.csc_matrix(A)
        solver = osqp.OSQP()
        solver.setup(P_csc, q, A_csc, l, u,
                     warm_starting=False, verbose=False,
                     eps_abs=1e-6, eps_rel=1e-6, polishing=False,
                     max_iter=2000)
        res = solver.solve()
        if res.info.status_val != 1:
            return None
        return res.x

    def flight(self, q_full, qd_full, x_d_world):
        # Flight uses pure legacy J^T·K·e foot tracking (matches Wbc.flight
        # exactly). The inverse-dynamics QP in flight was driving non-zero
        # knee torque to keep the leg at the posture-task target, which
        # compresses the spring mid-flight; by the time we land the energy
        # balance is off and the Raibert cycle no longer closes. We keep the
        # QP machinery in stance where the contact equality and friction
        # cone constraints are the actual value-add.
        x_foot = self.m.fk(self.ee, q_full)
        J = self.m.jacob(self.ee, q_full)
        e = x_d_world - x_foot
        K_fly = np.array([50.0, 50.0, 50.0])
        tau = (J.T @ (K_fly * e))[6:9]
        return np.clip(tau, -self.tau_max, self.tau_max)

    def stance(self, q_full, qd_full, x_d_world, R, omega_body, fire_force=0.0):
        # The QP no longer carries the attitude task — the min-tau solution
        # makes the QP-computed attitude correction roughly 50× slower than
        # the legacy direct-tau version, which lets the body lean enough
        # during stance for the spring fire to fling it sideways. We instead
        # feed the legacy-style attitude PD directly into the actuated tau
        # below; the QP keeps responsibility for dynamics consistency,
        # contact equality, friction cone, torque box, and fire tracking.
        P, q, A, l, u = self._build(
            q_full, qd_full, x_d_world,
            phase='stance', fire_force=fire_force,
            Kp_foot=self.Kp_foot_stn, Kd_foot=self.Kd_foot_stn,
            roll_des_accel=0.0, pitch_des_accel=0.0,
            w_foot=self.w_foot_stn, w_post=self.w_post_stance,
            w_att=0.0)
        x = self._solve(P, q, A, l, u)
        tau_qp = x[9:12] if x is not None else np.zeros(self.NU)

        # Raibert / cartesian foot PD (legacy J^T·K·e). Contact equality
        # makes the foot task inside the QP a constant, so the leg/body
        # coupling needed for Raibert step-to-step braking must come from
        # this feed-forward channel.
        x_foot = self.m.fk(self.ee, q_full)
        J = self.m.jacob(self.ee, q_full)
        e = x_d_world - x_foot
        tau_braking = (J.T @ (self.K_brake * e))[6:9]

        # Attitude PD on hip torques (legacy convention exactly: hip-yaw uses
        # +R[2,1] and ω_x, hip-pitch uses -R[2,0] and ω_y; K_att=20, K_dmp=8).
        tau_att = np.zeros(3)
        tau_att[0] =  self.K_att[0]*R[2, 1] + self.K_dmp[0]*omega_body[0]
        tau_att[1] = -self.K_att[1]*R[2, 0] + self.K_dmp[1]*omega_body[1]

        tau = tau_qp + tau_braking + tau_att
        # Match legacy exactly: during the fire window the knee torque is an
        # *override*, not an additive contribution. Adding K_brake[2]·e_z on
        # top of fire_force is ~10% extra energy per push and pumps the hop
        # height past its stable equilibrium (h≈0.20 instead of legacy 0.113),
        # which doubles the per-cycle lateral travel and leaves the MPC
        # under-actuated on x/y.
        if fire_force != 0.0:
            tau[2] = fire_force
        return np.clip(tau, -self.tau_max, self.tau_max)


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------

class Hop:
    """WBC+MPC monoped hopping orchestrator (mirrors HopRaibert's interface)."""

    # Plant tunables (matched to hopper.yaml: spring 2000, knee damping ~5)
    T_STANCE = 0.060   # s (60 ticks at dt=1ms)
    T_FIRE   = 0.030   # s (30 ticks)
    LEG_LEN  = 0.50    # nominal leg extension (stance + descent) [m] (matches HopRaibert)
    LEG_FLIGHT = 0.40  # retracted leg target during ascent (foot ground clearance) [m]
    M_BODY   = 1.4     # head+leg total mass [kg] (1.0+0.1+0.2+0.1)
    G        = 9.81
    ALPHA    = 4.0     # Raibert step-to-step momentum-exchange gain. Calibrated
                       # from observed plant: foot=-0.024 from v=0 actually
                       # produces Δv≈+0.08 (alpha≈3.3) — using 4.0 makes the
                       # MPC plan slightly conservative offsets to avoid
                       # per-cycle overshoot.

    # ahrs frame offset on the head body (from hopper.yaml frames spec). Needed to
    # back out head-origin world position from ahrs world position in y[25:28].
    AHRS_OFFSET_BODY = np.array([0.0, 0.0, 0.10])

    def __init__(self, m_fb, env, ee):
        self.m, self.ee = m_fb, ee
        self.mpc = Mpc(N=3)
        self.wbc = WbcQp(m_fb, ee)   # inverse-dynamics QP controller

        self.v_des = np.zeros(2)     # body-frame target horiz velocity [m/s]
        self.h_des = 0.20            # target apex height [m]

        # Per-cycle plan (refreshed when entering fly)
        self.plan_foot = np.zeros(2)
        self.plan_fire_force = 20.0  # mirrors HopRaibert's default
        self.hop_count = 0           # incremented on each flight-phase entry (one log line per hop)

        self.flag = None
        self.shift(2)                # start in fire so the first cycle bootstraps cleanly

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1

    V_DES_MAX = 0.30    # cap body-frame velocity command [m/s] (open-loop stable region)

    def msgproc(self, w):
        # Same key map as HopRaibert (sign flipped on w/s to keep parity with the existing UI).
        if   w[0] == 'a': self.v_des[0] += 0.10
        elif w[0] == 'd': self.v_des[0] -= 0.10
        elif w[0] == 'w': self.v_des[1] -= 0.10
        elif w[0] == 's': self.v_des[1] += 0.10
        elif w[0] == 'e': self.v_des[:] = 0.0
        elif w[0] == '1': self.h_des = max(0.05, self.h_des - 0.03)
        elif w[0] == '2': self.h_des += 0.03
        self.v_des = np.clip(self.v_des, -self.V_DES_MAX, self.V_DES_MAX)
        print('wmpc_v3: v_des=%s  h_des=%.3f' %(self.v_des, self.h_des))

    @staticmethod
    def _world_xy_to_body(R, v_xy):
        # Project a world-xy vector onto the body-heading axes (R columns 0,1 give
        # body-x, body-y expressed in world; their xy parts are the heading axes).
        bx = np.array([R[0, 0], R[1, 0]])
        by = np.array([R[0, 1], R[1, 1]])
        return np.array([bx @ v_xy, by @ v_xy])

    def _apex_h_estimate(self, vz):
        # Ballistic apex height above current altitude: vz^2/(2g).
        return vz*vz/(2.0*self.G)

    def _build_q_full(self, q_act, R, p_ahrs):
        # Floating-base q layout (tact rbd.py:422):
        #   q[0:3] = head world position (body origin, NOT ahrs)
        #   q[3:6] = rotation vector (axis-angle, R = expmap_so3(q[3:6]))
        #   q[6:9] = actuated joint positions
        p_head  = p_ahrs - R @ self.AHRS_OFFSET_BODY
        rot_vec = tact.logmap_so3(R)
        return np.concatenate([p_head, rot_vec, q_act])

    def update(self, q_act, qd_act, f, R, v_world, omega_body, p_ahrs):
        q_full = self._build_q_full(q_act, R, p_ahrs)
        p_head = q_full[:3]
        # qd_full layout (tact free-joint convention):
        #   qd[0:3] = body-frame linear velocity *of the body origin*
        #   qd[3:6] = body-frame angular velocity
        #   qd[6:9] = actuated joint velocities
        # framelinvel(ahrs) reports the world-frame velocity of the AHRS *point*,
        # which is offset from the body origin by AHRS_OFFSET_BODY in body frame.
        # Rigid-body translation: v_origin_world = v_ahrs_world − R·(ω_body × r_ahrs).
        # In body frame:           v_origin_body  = R^T·v_ahrs_world − ω_body × r_ahrs.
        # Skipping this correction (as v1 did) made the QP's qd diverge from the
        # plant whenever ω was non-trivial — e.g. during impact rotation — which
        # broke the Coriolis term in bias(q, qd) and turned the controller unstable.
        v_body_3 = R.T @ v_world - np.cross(omega_body, self.AHRS_OFFSET_BODY)
        qd_full = np.concatenate([v_body_3, omega_body, qd_act])

        # Foot world targets:
        #   flight  -> p_head + (R[:,0]·plan_x + R[:,1]·plan_y) + (0,0,-L)
        #              i.e. body-heading xy offset from MPC + world-vertical leg
        #   stance  -> p_head + (0,0,-L)  (leg straight down in world)
        # World-frame target gives passive attitude regulation: if body tilts,
        # head shifts laterally relative to the pinned foot, so the foot-task
        # error -> hip torque -> body rights itself.
        if self.s == 0:                                         # ---- fly
            if self.t == 0:
                v_body = self._world_xy_to_body(R, v_world[:2])
                h0 = self._apex_h_estimate(v_world[2])
                self.plan_foot, self.plan_fire_force = self.mpc.solve(
                    v_body, h0, self.v_des, self.h_des,
                    self.T_STANCE, self.T_FIRE, self.M_BODY, self.G, self.ALPHA)
                self.hop_count += 1
                #print('hop %3d  v_des=(%+0.3f,%+0.3f)  v=(%+0.3f,%+0.3f)  foot=(%+0.3f,%+0.3f)  fire=%5.2f  h0=%0.3f'
                #      % (self.hop_count,
                #         self.v_des[0], self.v_des[1],
                #         v_body[0],     v_body[1],
                #         self.plan_foot[0], self.plan_foot[1],
                #         self.plan_fire_force, h0))
            # Retract the leg during ascent so the foot clears the ground (the
            # passive knee spring limits how much the foot-PD wins, but it helps),
            # then re-extend on descent to prep for landing.
            leg = self.LEG_FLIGHT if v_world[2] > 0.0 else self.LEG_LEN
            x_d_world = p_head + R[:, 0]*self.plan_foot[0] \
                               + R[:, 1]*self.plan_foot[1] \
                               + np.array([0.0, 0.0, -leg])
            tau = self.wbc.flight(q_full, qd_full, x_d_world)
            # Only accept a landing while descending — a foot still skimming the
            # ground right after push-off (body rising, vz>0) was falsely tripping
            # f[2]>10 and re-entering 'land', corrupting the hop cycle.
            if f[2] > 10 and v_world[2] < 0.0: self.next_s = 1

        elif self.s == 1:                                       # ---- land/compress
            x_d_world = p_head + np.array([0.0, 0.0, -self.LEG_LEN])
            tau = self.wbc.stance(q_full, qd_full, x_d_world, R, omega_body)
            if self.t == int(self.T_STANCE*1000): self.next_s = 2

        elif self.s == 2:                                       # ---- fire/push-off
            x_d_world = p_head + np.array([0.0, 0.0, -self.LEG_LEN])
            ff = self.plan_fire_force if self.t < int(self.T_FIRE*1000) else 0.0
            tau = self.wbc.stance(q_full, qd_full, x_d_world, R, omega_body, fire_force=ff)
            if f[2] < 1: self.next_s = 0

        self.one_step_forward()
        return tau
