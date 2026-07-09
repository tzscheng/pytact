import numpy as np
import tact


env = tact.Env("minimal", render=True, redraw=8)
tau = np.zeros(env.dof)
i = 0

while True:
    y = env.step(tau)
    if i % 100 == 0:
        q = y[: env.dof]
        qd = y[env.dof :]
        print(f"{i:04d} q={q.round(4)} qd={qd.round(4)} y={y.round(4)}")
    i += 1
