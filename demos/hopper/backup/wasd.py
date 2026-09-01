#!/usr/bin/env -S uv run python
"""hopper wasd: 2-D world-frame target-position SAC env + entry point in one file.

Artifacts (written next to this file):
  - best.zip   policy with the highest evaluation return so far
  - last.zip   final policy at the end of training

Train:
  uv run python wasd.py --steps 1500000 --range 0.5
  uv run python wasd.py --steps 1500000 --init-from last.zip          # warm-start

Play (interactive WASD; backend defaults to tact):
  uv run python wasd.py --load best.zip
  uv run python wasd.py --load last.zip --backend mujoco

Keys (in play mode):
  w/s = ±target_x, d/a = ±target_y, g = stop here, r = reset env, q = quit
"""
import os, sys, time, ctypes, select, tty, termios, argparse, numpy as np

import gymnasium as gym
from gymnasium import spaces
import tact

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# common/ (mjenv/mjenv.so, env scenes) is vendored into ongoing (ongoing/common), not the tact pkg
COMMON_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common')


class HopperPosEnv(gym.Env):
    """hopper hopping with 2-D world-frame *position* command (target_x, target_y)."""

    metadata = {'render_modes': ['human']}

    U_SCALE = np.array([10.0, 20.0, 100.0], dtype=np.float64)
    OBS_DIM = 28
    FRAME_SKIP = 10
    MAX_STEPS = 2000

    READY_Q  = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    READY_KP = np.array([20.0, 50.0, 50.0], dtype=np.float64)
    READY_KD = np.array([1.0,  2.0,  2.0],  dtype=np.float64)

    H_STATIC          = 0.633
    CONTACT_THR       = 1.0
    PERTURB_STEPS     = 10
    PERTURB_AMP       = 0.1
    APEX_CAP          = 0.30
    UPRIGHT_THR       = 0.6
    MIN_FLIGHT_STEPS  = 5
    CYCLE_EXCESS_THR  = 0.02
    INFLIGHT_CAP      = 200.0
    LAND_VZ_THR       = 1.3
    LAND_PEN_W        = 50.0

    P_BOUND           = 5.0
    D_OBS             = 2.0
    DIST_W            = 2.0
    V_APPROACH_W      = 3.0
    V_APPROACH_CAP    = 1.5
    CMD_RESAMPLE_PROB = 0.005

    def __init__(self, render=False, seed=None,
                 reset_radius=0.5, max_radius=2.0, cmd_resample=True,
                 backend='tact'):
        self.render_flag = render
        self.reset_radius = float(reset_radius)
        self.max_radius   = float(max_radius)
        self.cmd_resample = cmd_resample
        self.backend = backend

        if backend == 'tact':
            self.sim = tact.Env('hopper', render=render, redraw=20)  # hopper.yaml embeds its own floor
        elif backend == 'mujoco':
            cdll = ctypes.CDLL(os.path.join(COMMON_ROOT, 'mjenv', 'mjenv.so'))
            cdll.init(b'1.xml', None, 20 if render else 0)
            self.sim = tact.CEnv(cdll, n_y=28, n_u=3, backend=backend, has_pd=True)
            if render: self.sim.step(np.zeros(3))   # force initial mjr_render so the window isn't blank during SB3 wrapping
        else:
            raise ValueError(f"unknown backend {backend!r}; expected 'tact' or 'mujoco'")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.OBS_DIM,), dtype=np.float32)

        self.t = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self._external_cmd = False
        self._was_contact = True
        self._h_apex = self.H_STATIC
        self._flight_start_t = 0
        self._inflight_paid = 0.0
        self._prev_action = np.zeros(3, dtype=np.float64)
        self._np_random = np.random.default_rng(seed)

    def set_target(self, x, y):
        self.target_x = float(np.clip(x, -self.max_radius, self.max_radius))
        self.target_y = float(np.clip(y, -self.max_radius, self.max_radius))
        self._external_cmd = True

    def nudge_target(self, dx, dy):
        self.set_target(self.target_x + dx, self.target_y + dy)

    def get_state(self):
        return dict(target=(self.target_x, self.target_y),
                    pos=(self.cur_x, self.cur_y),
                    dist=float(np.hypot(self.cur_x - self.target_x,
                                        self.cur_y - self.target_y)))

    def _sample_target(self):
        r  = self._np_random.uniform(0.0, self.reset_radius)
        th = self._np_random.uniform(0.0, 2 * np.pi)
        self.target_x = float(r * np.cos(th))
        self.target_y = float(r * np.sin(th))

    def _resample_near_current(self):
        r  = self._np_random.uniform(0.0, self.reset_radius)
        th = self._np_random.uniform(0.0, 2 * np.pi)
        new_x = self.cur_x + r * np.cos(th)
        new_y = self.cur_y + r * np.sin(th)
        self.target_x = float(np.clip(new_x, -self.max_radius, self.max_radius))
        self.target_y = float(np.clip(new_y, -self.max_radius, self.max_radius))

    def _decode(self, y):
        q  = y[0:3];  qd = y[3:6];  f = y[6:9]
        R  = tact.quat_to_rotation(y[12:16])
        a  = y[16:19]; w = y[19:22]
        v  = y[22:25]; p = y[25:28]
        return q, qd, f, R, a, w, v, p

    def _obs(self, y):
        q, qd, f, R, a, w, v, p = self._decode(y)
        dw = np.array([self.target_x - p[0], self.target_y - p[1], 0.0])
        db = R.T @ dw
        return np.concatenate([
            q,
            np.tanh(0.5 * qd),
            R.flatten(),
            np.tanh(0.5  * w),
            np.tanh(0.05 * a),
            np.tanh(v),
            [np.tanh(2.0 * (p[2] - 0.5))],
            [np.tanh(0.02 * np.linalg.norm(f))],
            [np.tanh(db[0] / self.D_OBS), np.tanh(db[1] / self.D_OBS)],
        ]).astype(np.float32)

    def _hold_pid_step(self, y):
        q, qd = y[0:3], y[3:6]
        tau = self.READY_KP * (self.READY_Q - q) - self.READY_KD * qd
        return self.sim.step(np.clip(tau, -self.U_SCALE, self.U_SCALE))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._np_random = np.random.default_rng(seed)

        if not self._external_cmd:
            self._sample_target()

        self.sim.reset()
        y = self.sim.step(np.zeros(3))

        for _ in range(400):
            y = self._hold_pid_step(y)

        for _ in range(self.PERTURB_STEPS):
            tau = self.U_SCALE * self._np_random.uniform(
                -self.PERTURB_AMP, self.PERTURB_AMP, size=3)
            y = self.sim.step(tau)

        self.t = 0
        self._was_contact = bool(np.linalg.norm(y[6:9]) > self.CONTACT_THR)
        self._h_apex = float(y[27])
        self._flight_start_t = 0
        self._inflight_paid = 0.0
        self._prev_action = np.zeros(3, dtype=np.float64)
        self.cur_x = float(y[25]); self.cur_y = float(y[26])
        return self._obs(y), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        tau = self.U_SCALE * action

        for _ in range(self.FRAME_SKIP):
            y = self.sim.step(tau)

        self.t += 1
        if self.cmd_resample and not self._external_cmd \
                and self._np_random.random() < self.CMD_RESAMPLE_PROB:
            self._resample_near_current()

        q, qd, f, R, a, w, v, p = self._decode(y)
        self.cur_x = float(p[0]); self.cur_y = float(p[1])
        upright = float(R[2, 2])
        height  = float(p[2])
        contact = bool(np.linalg.norm(f) > self.CONTACT_THR)

        liftoff = self._was_contact and (not contact)
        landing = (not self._was_contact) and contact
        self._was_contact = contact

        if liftoff:
            self._flight_start_t = self.t
            self._h_apex = height
            self._inflight_paid = 0.0
        if not contact:
            self._h_apex = max(self._h_apex, height)

        cycle_bonus = 0.0
        landing_pen = 0.0
        if landing:
            flight_dur = self.t - self._flight_start_t
            cycle_excess_raw = max(0.0, self._h_apex - self.H_STATIC)
            if flight_dur >= self.MIN_FLIGHT_STEPS and cycle_excess_raw >= self.CYCLE_EXCESS_THR:
                cycle_bonus = 30.0
            land_overspeed = max(0.0, abs(v[2]) - self.LAND_VZ_THR)
            landing_pen = -self.LAND_PEN_W * land_overspeed ** 2
            self._h_apex = height
        elif contact:
            self._h_apex = height

        flight_dur = (self.t - self._flight_start_t) if not contact else 0
        inflight_reward = 0.0
        if not contact and flight_dur >= self.MIN_FLIGHT_STEPS:
            raw_signal = min(self.APEX_CAP, max(0.0, height - self.H_STATIC))
            raw_step_reward = 30.0 * raw_signal
            cap_left = max(0.0, self.INFLIGHT_CAP - self._inflight_paid)
            inflight_reward = min(raw_step_reward, cap_left)
            self._inflight_paid += inflight_reward

        action_rate = float(np.sum((action - self._prev_action) ** 2))
        gyro_norm   = float(np.linalg.norm(w))
        qd_knee     = float(qd[2] ** 2)

        inflight_knee_pen = -0.3 * (action[2] ** 2) * (0.0 if contact else 1.0)

        dx_t = self.target_x - p[0]
        dy_t = self.target_y - p[1]
        dist = float(np.hypot(dx_t, dy_t))
        if dist > 0.05:
            v_along = float((dx_t * v[0] + dy_t * v[1]) / dist)
        else:
            v_along = 0.0
        v_along = float(np.clip(v_along, -self.V_APPROACH_CAP, self.V_APPROACH_CAP))

        reward = (
            cycle_bonus
          + inflight_reward
          + landing_pen
          + inflight_knee_pen
          + self.V_APPROACH_W * v_along
          - self.DIST_W * dist
          - 5.0   * (1.0 - upright) ** 2
          - 1.0   * gyro_norm
          - 0.10  * action_rate
          - 0.03  * qd_knee
          - 0.0005 * float(np.sum(action ** 2)) * (1.0 if contact else 0.0)
        )

        self._prev_action = action

        off_ground = bool(abs(p[0]) > self.P_BOUND or abs(p[1]) > self.P_BOUND)
        terminated = bool(upright < self.UPRIGHT_THR or height < 0.30 or off_ground)
        truncated  = bool(self.t >= self.MAX_STEPS)
        if terminated: reward -= 10.0

        return self._obs(y), float(reward), terminated, truncated, {}

    def close(self):
        try: self.sim.finish()
        except Exception: pass


def _make_env(rank, seed, reset_radius, max_radius, backend='tact'):
    def _init():
        return Monitor(HopperPosEnv(render=False, seed=seed + rank,
                                    reset_radius=reset_radius, max_radius=max_radius,
                                    backend=backend))
    return _init


class _BestSaver(BaseCallback):
    """Evaluate every `eval_freq` calls; save to best.zip when mean return improves."""
    def __init__(self, eval_env, save_path, eval_freq, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0: return True
        mean_r, _ = evaluate_policy(
            self.model, self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        if self.verbose:
            print(f'[eval] step={self.num_timesteps} mean_reward={mean_r:.2f} best={self.best_mean_reward:.2f}', flush=True)
        if mean_r > self.best_mean_reward:
            self.best_mean_reward = mean_r
            self.model.save(self.save_path)
            if self.verbose:
                print(f'[eval] new best, saved to {self.save_path}', flush=True)
        return True


def train(args):
    if args.n_envs > 1:
        venv = SubprocVecEnv([_make_env(i, args.seed, args.range, args.max_radius, args.backend)
                              for i in range(args.n_envs)])
    else:
        venv = DummyVecEnv([_make_env(0, args.seed, args.range, args.max_radius, args.backend)])

    eval_env = DummyVecEnv([_make_env(10_000, args.seed, args.range, args.max_radius, args.backend)])

    if args.init_from:
        print(f'[warm-start] loading {args.init_from}')
        model = SAC.load(args.init_from, env=venv, device=args.device)
    else:
        model = SAC(
            'MlpPolicy', venv,
            learning_rate = 3e-4,
            buffer_size   = 1_000_000,
            learning_starts = 5_000,
            batch_size    = 256,
            tau           = 0.005,
            gamma         = 0.99,
            train_freq    = (1, 'step'),
            gradient_steps = 1,
            ent_coef      = 'auto',
            policy_kwargs = dict(net_arch=[256, 256]),
            verbose       = 1,
            seed          = args.seed,
            device        = args.device,
        )

    best_cb = _BestSaver(
        eval_env,
        save_path='best',
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=best_cb,
                progress_bar=False, log_interval=10,
                reset_num_timesteps=not bool(args.init_from))
    model.save('last')
    print(f'[done] {time.time()-t0:.1f}s — best.zip / last.zip '
          f'(reset_radius={args.range}, max_radius={args.max_radius})')


def _read_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def play(args):
    print(f'[play] backend={args.backend} load={args.load}')
    env = HopperPosEnv(render=True, cmd_resample=False,
                       reset_radius=0.0, max_radius=args.max_radius,
                       backend=args.backend)
    env.set_target(0.0, 0.0)
    model = SAC.load(args.load, env=env)
    obs, _ = env.reset()
    print('controls: w/s = ±target_x, d/a = ±target_y, g = stop here, '
          'r = reset env, q = quit')

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        last_print = 0.0
        while True:
            k = _read_key()
            if k is not None:
                if   k == 'w': env.nudge_target(+args.step, 0.0)
                elif k == 's': env.nudge_target(-args.step, 0.0)
                elif k == 'd': env.nudge_target(0.0, +args.step)
                elif k == 'a': env.nudge_target(0.0, -args.step)
                elif k == 'g': env.set_target(env.cur_x, env.cur_y)
                elif k == 'r':
                    env.set_target(0.0, 0.0)
                    obs, _ = env.reset()
                    continue
                elif k == 'q':
                    break

            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)

            now = time.time()
            if now - last_print > 0.2:
                s = env.get_state()
                tx, ty = s['target']; px, py = s['pos']
                sys.stdout.write(
                    f'\rtgt=({tx:+.2f}, {ty:+.2f})  '
                    f'pos=({px:+.2f}, {py:+.2f})  '
                    f'd={s["dist"]:.2f}  t={env.t:4d}     ')
                sys.stdout.flush()
                last_print = now

            if term or trunc:
                print(f'\n[episode end] term={term} trunc={trunc} — resetting (target → origin)')
                env.set_target(0.0, 0.0)
                obs, _ = env.reset()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        env.close()
        print('\nbye')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',         type=int, default=3_000_000)
    ap.add_argument('--n-envs',        type=int, default=8)
    ap.add_argument('--seed',          type=int, default=0)
    ap.add_argument('--device',        default='auto')
    ap.add_argument('--range',         type=float, default=0.5, help='target sample radius at reset (m)')
    ap.add_argument('--max-radius',    type=float, default=2.0, help='clip for set_target / nudge_target (m)')
    ap.add_argument('--backend',       default='tact', choices=['tact', 'mujoco'], help="physics backend (default: tact)")
    ap.add_argument('--eval-freq',     type=int, default=50_000, help='env-steps between best.zip evaluations')
    ap.add_argument('--eval-episodes', type=int, default=3,      help='episodes per evaluation')
    ap.add_argument('--init-from',     default=None, help='checkpoint to warm-start training from')
    ap.add_argument('--load',          default=None, help='play a saved policy instead of training')
    ap.add_argument('--step',          type=float, default=0.3,  help='target position increment per key tap (m)')
    args = ap.parse_args()

    if args.load: play(args)
    else: train(args)
