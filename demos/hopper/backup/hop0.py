#!/usr/bin/env -S uv run python
"""hopper hop0: SAC training env + entry point in one file.

Artifacts (written next to this file):
  - best.zip   policy with the highest evaluation return so far
  - last.zip   final policy at the end of training

Train:
  uv run python hop0.py --steps 5000000
  uv run python hop0.py --backend mujoco --steps 5000000

Play (backend defaults to tact):
  uv run python hop0.py --load best.zip
  uv run python hop0.py --load last.zip --backend mujoco
"""
import os, time, ctypes, argparse, numpy as np

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


class HopperEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    U_SCALE = np.array([5.0, 20.0, 100.0], dtype=np.float64)
    OBS_DIM = 26
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
    INFLIGHT_CAP      = 200.0
    LAND_VZ_THR       = 1.3

    def __init__(self, render=False, seed=None, backend='tact'):
        self.render_flag = render
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32)

        self.t = 0
        self._was_contact = True
        self._h_apex = self.H_STATIC
        self._flight_start_t = 0
        self._inflight_paid = 0.0
        self._prev_action = np.zeros(3, dtype=np.float64)
        self._np_random = np.random.default_rng(seed)

    def _decode(self, y):
        q  = y[0:3];  qd = y[3:6];  f = y[6:9]
        R  = tact.quat_to_rotation(y[12:16])
        a  = y[16:19]; w = y[19:22]
        v  = y[22:25]; p = y[25:28]
        return q, qd, f, R, a, w, v, p

    def _obs(self, y):
        q, qd, f, R, a, w, v, p = self._decode(y)
        return np.concatenate([
            q,
            np.tanh(0.5 * qd),
            R.flatten(),
            np.tanh(0.5  * w),
            np.tanh(0.05 * a),
            np.tanh(v),
            [np.tanh(2.0 * (p[2] - 0.5))],
            [np.tanh(0.02 * np.linalg.norm(f))],
        ]).astype(np.float32)

    def _hold_pid_step(self, y):
        q, qd = y[0:3], y[3:6]
        tau = self.READY_KP * (self.READY_Q - q) - self.READY_KD * qd
        return self.sim.step(np.clip(tau, -self.U_SCALE, self.U_SCALE))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._np_random = np.random.default_rng(seed)

        self.sim.reset()
        y = self.sim.step(np.zeros(3))

        for _ in range(400):
            y = self._hold_pid_step(y)

        for _ in range(self.PERTURB_STEPS):
            tau = self.U_SCALE * self._np_random.uniform(-self.PERTURB_AMP, self.PERTURB_AMP, size=3)
            y = self.sim.step(tau)

        self.t = 0
        self._was_contact = bool(np.linalg.norm(y[6:9]) > self.CONTACT_THR)
        self._h_apex = float(y[27])
        self._flight_start_t = 0
        self._inflight_paid = 0.0
        self._prev_action = np.zeros(3, dtype=np.float64)
        return self._obs(y), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        tau = self.U_SCALE * action

        for _ in range(self.FRAME_SKIP):
            y = self.sim.step(tau)

        self.t += 1
        q, qd, f, R, a, w, v, p = self._decode(y)
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
            if flight_dur >= self.MIN_FLIGHT_STEPS and cycle_excess_raw >= 0.02:
                cycle_bonus = 30.0
            land_overspeed = max(0.0, abs(v[2]) - self.LAND_VZ_THR)
            landing_pen = -50.0 * land_overspeed ** 2
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

        reward = (
            cycle_bonus
          + inflight_reward
          + landing_pen
          + inflight_knee_pen
          - 2.0   * abs(v[0])
          - 2.0   * abs(v[1])
          - 5.0   * (1.0 - upright) ** 2
          - 2.0   * abs(p[0])
          - 2.0   * abs(p[1])
          - 1.0   * gyro_norm
          - 0.10  * action_rate
          - 0.03  * qd_knee
          - 0.0005 * float(np.sum(action ** 2)) * (1.0 if contact else 0.0)
        )

        self._prev_action = action

        terminated = bool(upright < self.UPRIGHT_THR or height < 0.30)
        truncated  = bool(self.t >= self.MAX_STEPS)
        if terminated: reward -= 10.0

        return self._obs(y), float(reward), terminated, truncated, {}

    def close(self):
        try: self.sim.finish()
        except Exception: pass


def _make_env(rank, seed=0, backend='tact'):
    def _init():
        return Monitor(HopperEnv(render=False, seed=seed + rank, backend=backend))
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
        venv = SubprocVecEnv([_make_env(i, args.seed, backend=args.backend) for i in range(args.n_envs)])
    else:
        venv = DummyVecEnv([_make_env(0, args.seed, backend=args.backend)])

    eval_env = DummyVecEnv([_make_env(10_000, args.seed, backend=args.backend)])

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
    model.learn(total_timesteps=args.steps, callback=best_cb, progress_bar=False, log_interval=10)
    model.save('last')
    print(f'[done] {time.time()-t0:.1f}s — best.zip / last.zip')


def play(args):
    print(f'[play] backend={args.backend} load={args.load}')
    env = HopperEnv(render=True, backend=args.backend)
    model = SAC.load(args.load, env=env)
    obs, _ = env.reset()
    ep_r, ep_t, ep_i = 0.0, 0, 0
    n_eps = args.episodes
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ep_r += r; ep_t += 1
        if term or trunc:
            ep_i += 1
            print(f'ep {ep_i}: return={ep_r:.2f}, len={ep_t}', flush=True)
            if n_eps and ep_i >= n_eps: break
            obs, _ = env.reset()
            ep_r, ep_t = 0.0, 0
    env.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',         type=int, default=1_000_000)
    ap.add_argument('--n-envs',        type=int, default=8)
    ap.add_argument('--seed',          type=int, default=0)
    ap.add_argument('--device',        default='auto')
    ap.add_argument('--backend',       default='tact', choices=['tact', 'mujoco'], help="physics backend (default: tact)")
    ap.add_argument('--eval-freq',     type=int, default=50_000, help='env-steps between best.zip evaluations')
    ap.add_argument('--eval-episodes', type=int, default=3,      help='episodes per evaluation')
    ap.add_argument('--load',          default=None, help='play a saved policy instead of training')
    ap.add_argument('--episodes',      type=int, default=5,      help='episodes to roll out in play mode')
    args = ap.parse_args()

    if args.load: play(args)
    else: train(args)
