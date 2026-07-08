#!/usr/bin/env -S uv run python
"""box_wall + interactive ball launcher.

Loads the brick-wall scene, runs the sim in a window, and launches a ball at the
wall every time you press SPACE.

    uv run python demos/box-wall/ball_throw.py

Controls (focus the render window):
  SPACE  — launch a ball at the wall
  ESC    — quit
  mouse  — orbit (drag left), pan (drag right), zoom (scroll)

How it works: the ball is a free-flying sphere (ball.yml) added to the
live scene with env.add(), placed in front of the wall and given an initial
velocity toward it. At most MAX_BALLS balls are kept — older ones are removed
with env.delete() so the scene (and the renderer's mesh slots) don't fill up.
SPACE is captured by the GLFW key callback in render.c, which makes win_render
return 1 once per press (ESC returns -1).
"""
import sys, os, random

HERE = os.path.dirname(os.path.abspath(__file__))    # .../pytact/demos/box-wall
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # .../pytact
sys.path.insert(0, PROJECT_ROOT)                     # so `import tact` works

import numpy as np
import tact

REDRAW    = 8        # render every N physics steps (~real-time-ish playback)
MAX_BALLS = 8        # keep at most this many balls live; delete the oldest beyond
SPEED     = 5.5      # launch speed (m/s) toward the wall

# render=False: we call _win_render() ourselves so we can read its return code
# (1 = SPACE pressed, -1 = ESC) — Env.step()'s built-in render swallows it.
env = tact.Env(os.path.join(HERE, 'box_wall'), render=False)  # dt=0.001 from box_wall.yml

balls = []           # live ball group names, oldest first
n_thrown = 0


def throw():
    """Add a ball in front of the wall and hurl it at the wall (-y)."""
    global n_thrown
    n_thrown += 1
    name = f'ball{n_thrown}'
    env.add(os.path.join(HERE, 'ball'), name=name)  # appended → its 6 DoF are q[-6:]
    balls.append(name)

    x = random.uniform(-0.25, 0.25)              # vary the aim across the wall
    z = random.uniform(0.28, 0.40)               # upper-ish (it drops during flight)
    q  = env.q.copy()
    qd = env.qd.copy()
    q[-6:]  = [x, 0.75, z, 0.0, 0.0, 0.0]        # spawn just in front of the wall (+y)
    qd[-6:] = [0.0, -SPEED, 0.0, 0.0, 0.0, 0.0]  # velocity straight at the wall (-y)
    env.q, env.qd = q, qd

    while len(balls) > MAX_BALLS:                # cap: remove the oldest ball
        old = balls.pop(0)
        try:
            env.delete(old)
        except Exception as e:                   # never let cleanup kill the demo
            print(f'  (delete {old} skipped: {e})')
    print(f'throw #{n_thrown}:  x={x:+.2f}  z={z:.2f}   balls live: {len(balls)}')


def main():
    print(__doc__)
    env._win_render()                            # open the window at the initial pose
    step = 0
    while True:
        env.step()                               # passive: bricks + balls under gravity/contact
        step += 1
        if step % REDRAW == 0:
            ret = env._win_render()
            if ret < 0:
                print('ESC — bye.')
                break
            if ret == 1:
                throw()


if __name__ == '__main__':
    main()
