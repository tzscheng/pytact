"""Convex MPC (Phase 3) for floating-base quadruped/biped — SRBM (single rigid
body model), condensed-QP form, osqp solver.

Lineage: MIT Cheetah 3 (Di Carlo et al., IROS 2018), simplified to yaw=0 (no
linearization around current heading; suitable for straight-line trot). The
linearization point is the body's current pose; foot positions enter through
the per-step input matrix B_k.

State    x = [Θ (3), p (3), ω (3), v (3)] ∈ R^12   (world-frame for p, v, Θ; body-frame for ω)
Input    u = [f_1, f_2, f_3, f_4] ∈ R^{3 nf}        (per-foot world-frame GRF)
Dynamics (linearized at yaw=0):
    Θ_{k+1} = Θ_k + ω_k · dt
    p_{k+1} = p_k + v_k · dt
    ω_{k+1} = ω_k + I_world^{-1} · Σ_i (r_i × f_i) · dt
    v_{k+1} = v_k + (Σ f_i)/m · dt + g · dt
where r_i = foot_i_world − p_com (CoM lever).

Condensed form (state propagated, decision = U over the horizon):
    X = M_x · x_0 + M_u · U + M_d              (X = [x_1 .. x_N])
    J = (X − X_ref)ᵀ Q̃ (X − X_ref) + Uᵀ R̃ U
  →  H = 2 (M_uᵀ Q̃ M_u + R̃),   g = 2 M_uᵀ Q̃ (M_x x_0 + M_d − X_ref)

Constraints (per foot i per step k):
    Friction cone (linearized 4-pyramid):
        ± f_x,i,k ≤ μ f_z,i,k,    ± f_y,i,k ≤ μ f_z,i,k
    Normal force:
        0 ≤ f_z,i,k ≤ f_max
    Swing (contact_schedule[k, i] == False):
        f_i,k = 0  (3-D equality)

Output: the first input u_0 ∈ R^{3 nf}. WBC consumes it as soft GRF reference.
"""

import numpy as np
import scipy.sparse as sp
import osqp


class ConvexMPC:
    def __init__(self, mass, Ig_body, foot_count=4,
                 horizon=10, dt=0.03,
                 mu=0.8, f_max=400.0,
                 Q_diag=None, R_diag=None,
                 g_world=(0.0, 0.0, -9.81),
                 verbose=False):
        self.m_total = float(mass)
        self.Ig      = np.asarray(Ig_body, dtype=float).reshape(3, 3)
        self.Ig_inv  = np.linalg.inv(self.Ig)
        self.nf      = int(foot_count)
        self.N       = int(horizon)
        self.dt      = float(dt)
        self.mu      = float(mu)
        self.f_max   = float(f_max)
        self.g_world = np.asarray(g_world, dtype=float)
        self.verbose = verbose

        # Default cost weights — tuned for dog scale; override per project.
        if Q_diag is None:
            # [Θ_xyz (3), p_xyz (3), ω_xyz (3), v_xyz (3)]
            Q_diag = [50, 50, 50,   50, 50, 100,   1, 1, 1,   1, 1, 10]
        if R_diag is None:
            R_diag = [1e-4] * (3 * self.nf)
        self.Q = np.diag(np.asarray(Q_diag, dtype=float))
        self.R = np.diag(np.asarray(R_diag, dtype=float))

        # Sizes
        self.nx = 12
        self.nu = 3 * self.nf
        self.N_dec   = self.N * self.nu
        self.N_state = self.N * self.nx

        # Constant time-step state-transition A (12 × 12). Foot-independent.
        A = np.eye(self.nx)
        A[0:3, 6:9]  = self.dt * np.eye(3)        # Θ ← ω
        A[3:6, 9:12] = self.dt * np.eye(3)        # p ← v
        self.A = A

        # Gravity additive d (12,)
        self.d = np.zeros(self.nx)
        self.d[9:12] = self.dt * self.g_world      # v ← g

        # Static parts of Q̃ block-diag and R̃ block-diag (sparse)
        self.Q_tilde = sp.block_diag([self.Q] * self.N, format='csc')
        self.R_tilde = sp.block_diag([self.R] * self.N, format='csc')

        self.prob = None
        self._warm = None

    # --------------------------------------------------------------
    # B_k matrix construction. Depends on foot positions relative to CoM at step k.
    # foot_pos_world: (nf, 3) world positions of each foot at this horizon step.
    # p_com_world:    (3,) CoM world position at this step.
    # Output: B (12, 3·nf)
    def _Bk(self, foot_pos_world, p_com_world):
        nf = self.nf
        B = np.zeros((self.nx, 3 * nf))
        m = self.m_total
        for i in range(nf):
            r = foot_pos_world[i] - p_com_world          # foot lever from CoM
            # ω̇ += I_world^{-1} · (r × f_i) · dt  →  block at rows 6:9, cols 3i:3(i+1)
            skew_r = np.array([[0, -r[2], r[1]],
                               [r[2], 0, -r[0]],
                               [-r[1], r[0], 0]])
            B[6:9, 3*i:3*(i+1)] = self.dt * (self.Ig_inv @ skew_r)
            # v̇ += (1/m) · f_i · dt → block at rows 9:12
            B[9:12, 3*i:3*(i+1)] = (self.dt / m) * np.eye(3)
        return B

    # --------------------------------------------------------------
    # Condensed-form matrices for a given trajectory.
    # foot_traj:    (N, nf, 3) — foot world positions at horizon step k = 0..N-1
    # com_traj:     (N, 3) — CoM world positions at horizon step k = 0..N-1
    # Returns M_x (Nnx × nx), M_u (Nnx × Nnu), M_d (Nnx,)
    def _condense(self, foot_traj, com_traj):
        N, nx, nu = self.N, self.nx, self.nu
        M_x = np.zeros((N * nx, nx))
        M_u = np.zeros((N * nx, N * nu))
        M_d = np.zeros(N * nx)

        # Cached A^k powers; we only need up to A^N
        A_pow = [np.eye(nx)]
        for _ in range(N):
            A_pow.append(self.A @ A_pow[-1])

        # B_k per step
        Bs = [self._Bk(foot_traj[k], com_traj[k]) for k in range(N)]

        # x_{k+1} = A^{k+1} x_0 + Σ_{j=0..k} A^{k-j} (B_j u_j + d)
        for k in range(N):
            row = k * nx
            M_x[row:row + nx, :] = A_pow[k + 1]
            # accumulate gravity term
            d_sum = np.zeros(nx)
            for j in range(k + 1):
                d_sum += A_pow[k - j] @ self.d
            M_d[row:row + nx] = d_sum
            for j in range(k + 1):
                col = j * nu
                M_u[row:row + nx, col:col + nu] = A_pow[k - j] @ Bs[j]
        return M_x, M_u, M_d

    # --------------------------------------------------------------
    # Build per-foot per-step inequality block. 6 rows per foot (4 friction + f_z lo/hi).
    def _ineq_block(self):
        mu = self.mu
        # 6 × 3 block: λx − μ λz, −λx − μ λz, λy − μ λz, −λy − μ λz, −λz, +λz
        A = np.array([
            [ 1, 0, -mu],
            [-1, 0, -mu],
            [ 0,  1, -mu],
            [ 0, -1, -mu],
            [ 0,  0, -1.0],
            [ 0,  0,  1.0],
        ])
        # corresponding upper-bound right side; lower stays -inf
        u_rhs = np.array([0, 0, 0, 0, 0, self.f_max])
        return A, u_rhs

    # --------------------------------------------------------------
    def solve(self,
              x0,                       # (12,) current state [Θ, p, ω, v]
              foot_traj,                # (N, nf, 3) world foot positions over horizon
              com_traj,                 # (N, 3) world CoM positions over horizon
              contact_schedule,         # (N, nf) bool — True = stance at step k
              x_ref_traj=None):         # (N, 12) reference states; default = x0 repeated
        N, nx, nu = self.N, self.nx, self.nu

        if x_ref_traj is None:
            x_ref_traj = np.tile(x0, (N, 1))
        x_ref_flat = x_ref_traj.reshape(-1)

        # Condense dynamics
        M_x, M_u, M_d = self._condense(foot_traj, com_traj)
        e0 = M_x @ x0 + M_d - x_ref_flat                       # (Nnx,)

        # QP cost:  H = 2 (M_u^T Q̃ M_u + R̃),  g = 2 M_u^T Q̃ e0
        M_u_csc = sp.csc_matrix(M_u)
        H = 2.0 * (M_u_csc.T @ self.Q_tilde @ M_u_csc + self.R_tilde)
        g = 2.0 * (M_u_csc.T @ (self.Q_tilde @ e0))

        # Inequality: 6 × nu_per_foot per foot per step, total 6·nf·N rows.
        ineqA_block, ineqU_block = self._ineq_block()
        rows_per_step = 6 * self.nf
        n_ineq        = rows_per_step * N
        A_ineq = np.zeros((n_ineq, N * nu))
        l_ineq = np.full(n_ineq, -np.inf)
        u_ineq = np.full(n_ineq,  np.inf)
        for k in range(N):
            for i in range(self.nf):
                r = k * rows_per_step + 6 * i
                c = k * nu + 3 * i
                A_ineq[r:r + 6, c:c + 3] = ineqA_block
                # default upper bounds (no equality)
                u_ineq[r:r + 6]     = ineqU_block      # [0, 0, 0, 0, 0, f_max]
                # but the f_z+ row needs upper = f_max only when in contact;
                # swing-foot zero is handled below via equality

        # Swing equality: f_i,k = 0 when contact_schedule[k, i] is False
        # We add 3 equality rows per swing foot per step.
        eq_rows_per_step = 3 * self.nf  # max if all swing; we'll mask inactive with [-inf, +inf]
        n_eq = eq_rows_per_step * N
        A_eq = np.zeros((n_eq, N * nu))
        l_eq = np.full(n_eq, -np.inf)
        u_eq = np.full(n_eq,  np.inf)
        for k in range(N):
            for i in range(self.nf):
                r = k * eq_rows_per_step + 3 * i
                c = k * nu + 3 * i
                if not contact_schedule[k, i]:
                    A_eq[r:r + 3, c:c + 3] = np.eye(3)
                    l_eq[r:r + 3] = 0.0
                    u_eq[r:r + 3] = 0.0

        # Stack
        A_all = sp.vstack([sp.csc_matrix(A_ineq), sp.csc_matrix(A_eq)], format='csc')
        l_all = np.concatenate([l_ineq, l_eq])
        u_all = np.concatenate([u_ineq, u_eq])

        # OSQP (rebuild each tick — MPC runs at ~30 Hz so setup overhead is acceptable)
        self.prob = osqp.OSQP()
        self.prob.setup(H, np.asarray(g).reshape(-1),
                        A_all, l_all, u_all,
                        verbose=self.verbose, warm_start=True,
                        eps_abs=1e-4, eps_rel=1e-4, max_iter=200,
                        polishing=False)
        if self._warm is not None:
            self.prob.warm_start(self._warm)
        res = self.prob.solve()
        if res.info.status not in ('solved', 'solved inaccurate'):
            raise RuntimeError(f'MPC QP did not solve: {res.info.status}')
        self._warm = res.x.copy()

        # Return first-step GRF and the full solution for debugging
        u0 = res.x[:nu].reshape(self.nf, 3)
        return u0, res
