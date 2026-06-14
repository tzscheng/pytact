"""Point-taxel tactile sensor smoke tests.

Run: UV_CACHE_DIR=/tmp/uv-cache uv run --offline python tests/test_tactile_sensor.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tact

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
fails = 0


def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond:
        fails += 1


base = os.path.join(tempfile.gettempdir(), 'tactile_sensor_test')
csv_path = base + '_taxels.csv'
with open(csv_path, 'w') as f:
    f.write('x,y,z,nx,ny,nz,ux,uy,uz,area\n')
    f.write('0,0,0,0,0,1,1,0,0,0.0001\n')
    f.write('0.2,0,0,0,0,1,1,0,0,0.0001\n')

YML = f"""sim: {{solver: lcp, dt: 0.001, iters: 40}}
materials:
  mat1: {{normal: [20000, 200], tangent: [20000, 200, 1.0], spin: [100, 1, 0.02], roll: [100, 1, 0.005], restitution: 0.0}}
tactiles:
  - name: pad
    body: root
    fps: 1000
    radius: 0.08
    kernel: gaussian
    channels: [normal, shear_u, shear_v, pressure]
    samples_file: {os.path.basename(csv_path)}
bodies:
  - name: root
    shapes:
      - {{type: box, pos: [0, 0, -0.01], param: [1, 1, 0.01], contact: [1, mat1], rgba: [0.6, 0.6, 0.6, 1]}}
  - name: ball
    joint: {{type: free, parent: root, q0: [0, 0, 0.09, 0, 0, 0], eulerseq: xyz}}
    inertial: {{mass: 1.0, tensor: [sphere, 0.1]}}
    shapes:
      - {{type: sphere, param: [0.1], contact: [1, mat1], rgba: [0.3, 0.6, 0.9, 1]}}
"""

with open(base + '.yml', 'w') as f:
    f.write(YML)

env = tact.Env(base, render=False)
check(len(env.tactiles) == 1 and env.tactiles[0]['n'] == 2, 'YAML + CSV tactile samples parsed')

env.reset()
env.step()
idx, data = env.m.contact_reports()
check(len(idx) > 0 and data.shape[1] == 10, 'native contact reports exposed', f'n={len(idx)}')

frames = list(env.tactile_frames())
check(len(frames) == 1 and frames[0][0] == 'pad', 'tactile frame yielded')
arr = np.frombuffer(frames[0][1], '<f4').reshape(2, 4)
check(arr[0, 0] > 0.0, 'center taxel normal force is positive', f'normal={arr[:,0]}')
check(abs(arr[1, 0]) < arr[0, 0] * 0.2, 'far taxel receives little force', f'normal={arr[:,0]}')
check(arr[0, 3] > arr[0, 0], 'pressure channel divides by area', f'force={arr[0,0]:.3g} pressure={arr[0,3]:.3g}')

if fails:
    raise SystemExit(fails)
print('\nOK')
