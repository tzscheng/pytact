"""Whole-Body Controller (Phase 2): single dense weighted QP, osqp-solved.

Decision vector
    z = [ q̈ (nv),  λ_per_foot (3 × n_foot),  τ (n_act) ]
        with nv = 6 (base) + n_act, foot order = order of `foot_frames` at init.

Constraints
    Floating-base dynamics (equality, nv rows):
        M(q_fb) q̈ + b(q_fb, qd_fb) − Sᵀ τ − Σ J_iᵀ λ_i = 0
        Sᵀ = [0_{6×n_act}; I_{n_act}] — base 6 rows unactuated.

    Per-foot mode switches (equality, 3 rows each — exactly one active per foot):
        Stance:   J_i q̈ + J̇_i qd = 0          (zero foot acceleration in world)
        Swing:    λ_i = 0                       (no contact force)

    Linearized friction cone (inequality, 5 rows per foot, always active —
    vacuously satisfied on swing legs because λ_i = 0 there):
        λ_z ≥ 0,   |λ_x| ≤ μ λ_z,   |λ_y| ≤ μ λ_z

    Torque limits (inequality, 2 rows per joint):
        −τ_max ≤ τ_j ≤ τ_max

Cost (sum of weighted quadratic tasks)
    w_body_lin ‖q̈_base_lin − a_lin_des_body‖²
    w_body_ang ‖q̈_base_ang − ω̇_des_body‖²
    w_swing Σ_(i∈swing) ‖J_i q̈ + J̇_i qd − a_swing_des_world_i‖²
    w_post   ‖q̈_joint − a_post_des‖²
    w_tau    ‖τ‖²

PD references for each task are supplied via BodyTask / SwingTask / PostureTask
on every solve() call. Gains are properties of the task objects so the caller
can tune them per gait without reaching into WBC internals.

Conventions match the rest of tact:
    qd[:3] = v_body (body-frame linear velocity of the base origin)
    qd[3:6] = ω_body (body-frame angular velocity of the base)
    qd[6:] = qd_joint
    Model.jacob returns the world-frame spatial Jacobian; for '3d' frames the
    rows are world-linear, so swing-task references are expressed in world frame.
"""

import numpy as np
import scipy.sparse as sp
import osqp

from .rbd import expmap_so3, logmap_so3


class BodyTask:
    """Base-body PD reference.
       p_ref / v_ref:  world-frame position / linear velocity of base origin.
       R_ref / w_ref:  world-frame rotation (3×3) / body-frame angular velocity desired."""
    def __init__(self, Kp_lin=400.0, Kd_lin=20.0, Kp_ang=100.0, Kd_ang=10.0):
        self.Kp_lin = float(Kp_lin); self.Kd_lin = float(Kd_lin)
        self.Kp_ang = float(Kp_ang); self.Kd_ang = float(Kd_ang)
        self.p_ref = np.zeros(3); self.v_ref = np.zeros(3)
        self.R_ref = np.eye(3);   self.w_ref = np.zeros(3)


class SwingTask:
    """Per-foot world-frame PD reference for the swing trajectory."""
    def __init__(self, Kp=1000.0, Kd=30.0):
        self.Kp = float(Kp); self.Kd = float(Kd)
        self.p_ref    = np.zeros(3)
        self.v_ref    = np.zeros(3)
        self.a_ref_ff = np.zeros(3)


class PostureTask:
    """Joint-space PD regularizer."""
    def __init__(self, q_ref, Kp=10.0, Kd=2.0):
        self.q_ref = np.asarray(q_ref, dtype=float).copy()
        self.Kp = float(Kp); self.Kd = float(Kd)


class CoMTask:
    """World-frame PD reference for the system CoM (not the base origin).

    For models with a non-zero CoM-vs-base offset (most real robots, including
    dog), tracking the base origin instead of the CoM puts a free moment-arm
    on every body acceleration command — small base errors translate to bigger
    CoM errors, and forward acceleration creates a pitch torque that the
    instantaneous BodyTask must fight reactively. Tracking the CoM directly
    cancels that lever-arm by construction. Pair with BodyTask whose Kp_lin/
    Kd_lin are zeroed out so the two don't compete on linear DoF."""
    def __init__(self, Kp=400.0, Kd=20.0):
        self.Kp = float(Kp); self.Kd = float(Kd)
        self.p_ref = np.zeros(3)
        self.v_ref = np.zeros(3)


class WBC:
    """Whole-body QP controller — fixed decision dimension, mode switches via
    constraint-bound activation. See module docstring for the math."""

    def __init__(self, model, foot_frames,
                 mu=0.8, tau_max=40.0,
                 w_body_lin=100.0, w_body_ang=100.0,
                 w_swing=200.0, w_post=1.0, w_tau=1e-3,
                 w_com=100.0, w_grf=1.0,
                 verbose=False):
        assert model.jtype[0] == 3, 'WBC requires floating-base Model (jtype[0]==3)'
        self.m           = model
        self.foot_frames = list(foot_frames)
        self.foot_keys   = {f: '3d' for f in self.foot_frames}
        self.nf          = len(self.foot_frames)
        self.nv          = len(self.m.qd0)
        self.n_act       = self.nv - 6
        self.mu          = float(mu)
        self.tau_max     = np.broadcast_to(np.asarray(tau_max, dtype=float), (self.n_act,)).copy()
        self.w_body_lin  = float(w_body_lin); self.w_body_ang = float(w_body_ang)
        self.w_swing     = float(w_swing)
        self.w_post      = float(w_post)
        self.w_tau       = float(w_tau)
        self.w_com       = float(w_com)
        self.w_grf       = float(w_grf)
        self.verbose     = verbose

        # Decision-vector layout
        self.n_qdd   = self.nv
        self.n_lam   = 3 * self.nf
        self.n_tau   = self.n_act
        self.n_z     = self.n_qdd + self.n_lam + self.n_tau
        self.qdd0    = 0
        self.lam0    = self.n_qdd
        self.tau0    = self.n_qdd + self.n_lam

        # Constraint-row layout (fixed dims; rows toggle active/inactive each solve)
        n_eq_dyn      = self.nv
        n_eq_contact  = 3 * self.nf
        n_eq_lamzero  = 3 * self.nf
        n_ineq_fric   = 5 * self.nf
        n_ineq_tau    = 2 * self.n_act
        cum = 0
        self.row_dyn0     = cum; cum += n_eq_dyn
        self.row_contact0 = cum; cum += n_eq_contact
        self.row_lamzero0 = cum; cum += n_eq_lamzero
        self.row_fric0    = cum; cum += n_ineq_fric
        self.row_tau0     = cum; cum += n_ineq_tau
        self.n_c          = cum

        # Sᵀ (nv × n_act): map τ to non-base rows.
        self.S_T = np.zeros((self.nv, self.n_act))
        self.S_T[6:, :] = np.eye(self.n_act)

        # 5×3 friction block: rows are [λx − μλz, −λx − μλz, λy − μλz, −λy − μλz, −λz] ≤ 0.
        mu = self.mu
        self.fric5 = np.array([
            [ 1.0, 0.0, -mu],
            [-1.0, 0.0, -mu],
            [ 0.0, 1.0, -mu],
            [ 0.0,-1.0, -mu],
            [ 0.0, 0.0, -1.0],
        ])

        self.prob = None
        self._warm = None

    def solve(self, q_fb, qd_fb,
              contact_mask,
              body_task: BodyTask,
              swing_tasks,                   # list[SwingTask] of length nf — only swing-foot entries used
              posture_task: PostureTask,
              com_task: CoMTask = None,      # optional — when given, tracks system CoM in addition to BodyTask
              grf_ref=None,                  # optional (nf, 3) world-frame GRF reference from an MPC; soft-tracked via ‖λ_i − f_ref,i‖²
              f_ext=None):
        """Solve the WBC QP. Returns τ ∈ R^{n_act}. contact_mask[i] True ⇒ stance."""
        nv, nf, n_act = self.nv, self.nf, self.n_act

        # ---- model quantities ----
        M     = self.m.inertia(q_fb)                             # (nv, nv)
        b     = self.m.bias(q_fb, qd_fb, f_ext)                  # (nv,)
        J_all = self.m.jacob(self.foot_keys, q_fb)               # (3 nf, nv)
        Jd_qd = self.m.jacob_dot_qd(self.foot_keys, q_fb, qd_fb)      # (3 nf,)
        x_w   = self.m.fk(self.foot_keys, q_fb).reshape(nf, 3)   # (nf, 3) world

        # ---- assemble dense P, q, A, l, u ----
        P = np.zeros((self.n_z, self.n_z))
        q = np.zeros(self.n_z)
        A = np.zeros((self.n_c, self.n_z))
        l = np.full(self.n_c, -np.inf)
        u = np.full(self.n_c,  np.inf)

        # Floating-base dynamics  (M q̈ − Sᵀ τ − Σ J_iᵀ λ = −b)
        A[self.row_dyn0:self.row_dyn0 + nv, self.qdd0:self.qdd0 + nv] = M
        A[self.row_dyn0:self.row_dyn0 + nv, self.tau0:self.tau0 + n_act] = -self.S_T
        for i in range(nf):
            J_i = J_all[3*i:3*(i+1), :]
            A[self.row_dyn0:self.row_dyn0 + nv,
              self.lam0 + 3*i : self.lam0 + 3*(i+1)] = -J_i.T
        l[self.row_dyn0:self.row_dyn0 + nv] = -b
        u[self.row_dyn0:self.row_dyn0 + nv] = -b

        # Per-foot mode constraints
        for i in range(nf):
            rc = self.row_contact0 + 3*i
            rl = self.row_lamzero0 + 3*i
            J_i = J_all[3*i:3*(i+1), :]
            Jd_i = Jd_qd[3*i:3*(i+1)]
            if contact_mask[i]:
                # stance: J_i q̈ = −J̇_i qd
                A[rc:rc+3, self.qdd0:self.qdd0 + nv] = J_i
                l[rc:rc+3] = -Jd_i
                u[rc:rc+3] = -Jd_i
                # row_lamzero stays inactive (l=-inf, u=+inf, A row all zero)
            else:
                # swing: λ_i = 0
                A[rl:rl+3, self.lam0 + 3*i : self.lam0 + 3*(i+1)] = np.eye(3)
                l[rl:rl+3] = 0.0
                u[rl:rl+3] = 0.0

        # Friction cone (always wired — vacuous on swing because λ=0)
        for i in range(nf):
            rf = self.row_fric0 + 5*i
            A[rf:rf+5, self.lam0 + 3*i : self.lam0 + 3*(i+1)] = self.fric5
            u[rf:rf+5] = 0.0    # one-sided ≤ 0

        # Torque limits  −τ_max ≤ τ ≤ τ_max  (split into ± rows)
        for j in range(n_act):
            r = self.row_tau0 + 2*j
            A[r,   self.tau0 + j] =  1.0; u[r]   =  self.tau_max[j]
            A[r+1, self.tau0 + j] = -1.0; u[r+1] =  self.tau_max[j]

        # ---- cost: body PD ----
        R_base = expmap_so3(q_fb[3:6])
        p_base = q_fb[:3]
        v_world = R_base @ qd_fb[:3]

        a_lin_des_world = (body_task.Kp_lin * (body_task.p_ref - p_base)
                           + body_task.Kd_lin * (body_task.v_ref - v_world))
        a_lin_des_body  = R_base.T @ a_lin_des_world

        err_R = body_task.R_ref @ R_base.T
        err_ang_world = logmap_so3(err_R)
        omega_dot_des_body = (R_base.T @ (body_task.Kp_ang * err_ang_world)
                              + body_task.Kd_ang * (body_task.w_ref - qd_fb[3:6]))

        # ‖q̈[:3] − a_lin_des_body‖² with weight w_body_lin
        for k in range(3):
            P[self.qdd0 + k, self.qdd0 + k] += self.w_body_lin
            q[self.qdd0 + k]                -= self.w_body_lin * a_lin_des_body[k]
        for k in range(3):
            P[self.qdd0 + 3 + k, self.qdd0 + 3 + k] += self.w_body_ang
            q[self.qdd0 + 3 + k]                    -= self.w_body_ang * omega_dot_des_body[k]

        # ---- cost: CoM task (optional) ----
        # ‖J_com q̈ + (J̇_com qd) − a_com_des‖²    (world frame)
        # J̇_com qd via finite-diff in q at small dt (same trick as jacob_dot_qd; SO(3)
        # branch error is O(dt²) and negligible at dt=1e-4 against the controller's
        # 1 kHz). com_jacob is now C-accelerated (~0.01 ms) so two calls is cheap.
        if com_task is not None:
            J_com = self.m.com_jacob(q_fb)
            dt_fd = 1e-4
            Jcom_2 = self.m.com_jacob(q_fb + dt_fd * qd_fb)
            Jdot_com_qd = ((Jcom_2 - J_com) @ qd_fb) / dt_fd
            r_com_now = self.m.com(q_fb)
            v_com_now = J_com @ qd_fb
            a_com_des = (com_task.Kp * (com_task.p_ref - r_com_now)
                         + com_task.Kd * (com_task.v_ref - v_com_now))
            r_lin = a_com_des - Jdot_com_qd
            P[self.qdd0:self.qdd0 + nv, self.qdd0:self.qdd0 + nv] += self.w_com * (J_com.T @ J_com)
            q[self.qdd0:self.qdd0 + nv]                           -= self.w_com * (J_com.T @ r_lin)

        # ---- cost: swing task per foot in swing ----
        for i in range(nf):
            if contact_mask[i]:
                continue
            J_i  = J_all[3*i:3*(i+1), :]
            Jd_i = Jd_qd[3*i:3*(i+1)]
            v_foot_world = J_i @ qd_fb
            st = swing_tasks[i]
            a_des = (st.a_ref_ff
                     + st.Kp * (st.p_ref - x_w[i])
                     + st.Kd * (st.v_ref - v_foot_world))
            # We want   J_i q̈  ≈  a_des − J̇_i qd  =: r
            r = a_des - Jd_i
            P[self.qdd0:self.qdd0 + nv, self.qdd0:self.qdd0 + nv] += self.w_swing * (J_i.T @ J_i)
            q[self.qdd0:self.qdd0 + nv]                          -= self.w_swing * (J_i.T @ r)

        # ---- cost: posture (joint-space PD on q̈_joint) ----
        q_joint  = q_fb[6:]
        qd_joint = qd_fb[6:]
        a_post_des = posture_task.Kp * (posture_task.q_ref - q_joint) - posture_task.Kd * qd_joint
        for k in range(n_act):
            P[self.qdd0 + 6 + k, self.qdd0 + 6 + k] += self.w_post
            q[self.qdd0 + 6 + k]                    -= self.w_post * a_post_des[k]

        # ---- cost: GRF reference tracking (optional, Phase 3 MPC handoff) ----
        # ‖λ_i − f_ref,i‖²  per stance foot. Swing entries are skipped because
        # their λ_i is already constrained to 0 by an equality row above —
        # tracking 0 against an arbitrary f_ref there would just fight the
        # equality. Soft (cost) not hard (equality): MPC and WBC may disagree on
        # GRF when contact constraints, friction, or torque limits force a
        # different split — WBC's instantaneous constraints win in that conflict.
        if grf_ref is not None:
            grf_ref = np.asarray(grf_ref).reshape(nf, 3)
            for i in range(nf):
                if not contact_mask[i]:
                    continue
                base = self.lam0 + 3*i
                for k in range(3):
                    P[base + k, base + k] += self.w_grf
                    q[base + k]           -= self.w_grf * grf_ref[i, k]

        # ---- cost: τ regularizer ----
        for k in range(n_act):
            P[self.tau0 + k, self.tau0 + k] += self.w_tau

        # tiny diagonal for numerical PSD-ness
        P += 1e-9 * np.eye(self.n_z)

        # ---- osqp solve. setup each tick (hot-update via prob.update tried and
        # rejected — OSQP scaling drifted across ticks → primal-infeasible cascades).
        # Speed knobs that don't compromise stability: polishing off, eps loose,
        # iteration cap modest. If still too slow, run start with -f >1 (control
        # rate down-samples; physics stays at 1 kHz). ----
        P_csc = sp.csc_matrix(P)
        A_csc = sp.csc_matrix(A)
        self.prob = osqp.OSQP()
        self.prob.setup(P_csc, q, A_csc, l, u,
                        verbose=self.verbose, warm_start=True,
                        eps_abs=1e-4, eps_rel=1e-4, max_iter=200,
                        polishing=False)
        if self._warm is not None:
            self.prob.warm_start(self._warm)
        res = self.prob.solve()
        if res.info.status not in ('solved', 'solved inaccurate'):
            raise RuntimeError(f'WBC QP did not solve: {res.info.status}')
        self._warm = res.x.copy()

        return res.x[self.tau0:self.tau0 + n_act], res     # τ and full OSQP result for debug
