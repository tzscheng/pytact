#!/usr/bin/env -S uv run python
"""Cartpole RL demo (Stable-Baselines3 PPO) over a Gymnasium wrapper of tact.Env.

    uv run python cartpole-sb3.py            # train 10k steps -> saves out.zip
    uv run python cartpole-sb3.py play       # load out.zip, render a rollout

The cartpole model (`cartpole.yml`) is cart (lin) + pole (rev), both active →
dof=2; only the cart is actuated (action drives u[0], pole gets 0 = underactuated).
proprio y (feeds) = [cart_pos, pole_pos, cart_vel, pole_vel].
"""
import os, sys, time, random
import numpy as np
# Run from anywhere: put the checkout root on sys.path so `import tact` works, and chdir into
# this dir so tact.Env('cartpole') resolves cartpole.yml next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.chdir(HERE)
import tact
import stable_baselines3 as sb3, gymnasium as gym

class env_model(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.env = tact.Env('cartpole', render=env_config['render'],
                            redraw=env_config.get('redraw', 20))
        self.sleep = env_config.get('sleep', 0)   # per-step wall-clock pacing (play only)

        self.step_cnt = 0
        self.epi_cnt = 0
        self.r_sum = 0

    def reward_func(self, q, qd):
        reward = np.max((1.0 - np.abs(q[1])/np.pi, 0))
        #reward = np.cos(q[1]) # - 0.01*qd[1]*qd[1] - 0.01*q[0]*q[0]
        #reward = np.cos(q[1]) - np.abs(q[0])

        self.r_sum += reward
        done = False

        if self.step_cnt == 1000: done = True
        elif np.abs(q[1]) > 0.5: done = True

        return reward, done, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset()
        # reset() aliases self.q to the model's persistent q0; copy before mutating
        # so the random start perturbation doesn't corrupt q0 across episodes.
        self.env.q = self.env.q.copy()
        self.env.q[1] = 0.2*random.random() - 0.1

        obs = np.zeros(4)
        obs[0:2] = np.tanh(self.env.q)
        obs[2:4] = np.tanh(self.env.qd)
        print(self.epi_cnt, self.step_cnt, self.r_sum)

        self.epi_cnt += 1
        self.step_cnt = 0
        self.r_sum = 0
        return obs, {}

    def step(self, action):
        u = np.zeros(2)
        u[0] = 100 * action[0]            # force on the cart; pole (u[1]) stays passive

        y = self.env.step(u)              # active-only torque, length dof=2 → proprio y
        q, qd = y[0:2], y[2:4]

        obs = np.zeros(4)
        obs[0:2] = np.tanh(q)
        obs[2:4] = np.tanh(qd)

        reward, done, info = self.reward_func(q, qd)
        self.step_cnt += 1
        if self.sleep: time.sleep(self.sleep)
        return obs, reward, done, False, info

def play():
    env = env_model({'render': True, 'sleep': 0.01, 'redraw': 1})
    #model = sb3.DDPG.load('out')
    model = sb3.PPO.load('out')
    obs, _ = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _, info = env.step(action)

        print(env.step_cnt, env.env.q)
        if done: env.env.finish(); exit(0)

if len(sys.argv) > 1: play()
env = env_model({'render': False, 'sleep': 0})

#model = sb3.DDPG('MlpPolicy', env=env, learning_rate=0.0001)
model = sb3.PPO('MlpPolicy', env=env, learning_rate=0.0001)
#policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
#model = sb3.PPO('MlpPolicy', env=env, learning_rate=0.0001, n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, policy_kwargs=policy_kwargs)

model.learn(total_timesteps = 10000)
model.save('out')
