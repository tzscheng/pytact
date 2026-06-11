"""Packaging-oriented smoke checks for tact's native library.

Run:
  uv run python tact/tests/test_packaging_smoke.py

This is intentionally self-contained and does not require sibling robot
projects. It checks the properties that matter for a future pip package:
`libtact.so` loads without render runtime hard deps, headless physics works, and
offscreen camera rendering works when the host provides EGL/OpenGL/encoders.
"""
import os
import subprocess
import sys
import tempfile

TACT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TACT)
sys.path.insert(0, ROOT)

import tact

PASS, FAIL, SKIP = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m', '\033[33mSKIP\033[0m'
fails = 0


def check(cond, label, detail=''):
    global fails
    print(f'  [{PASS if cond else FAIL}] {label}' + (f'   {detail}' if detail else ''))
    if not cond:
        fails += 1


def skip(label, detail=''):
    print(f'  [{SKIP}] {label}' + (f'   {detail}' if detail else ''))


def cmd(args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


lib = os.path.join(tact.pkg_dir, 'bin', 'libtact.so')
print('\n[native deps]')
check(os.path.exists(lib), 'tact/bin/libtact.so exists', lib)

ldd = cmd(['ldd', lib])
hard_deps = ['libGL', 'libOpenGL', 'libEGL', 'libglfw', 'libGLEW',
             'libturbojpeg', 'libzstd']
bad_ldd = [x for x in hard_deps if x in ldd.stdout]
check(ldd.returncode == 0 and not bad_ldd,
      'libtact.so has no render/encoder hard deps',
      ', '.join(bad_ldd) if bad_ldd else '')

nm = cmd(['nm', '-D', lib])
render_prefixes = ('gl', 'egl', 'glfw', 'tj', 'ZSTD')
bad_nm = []
for line in nm.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[0] == 'U':
        sym = parts[1]
    elif len(parts) >= 3 and parts[-2] == 'U':
        sym = parts[-1]
    else:
        continue
    if sym.startswith(render_prefixes):
        bad_nm.append(sym)
check(nm.returncode == 0 and not bad_nm,
      'libtact.so has no unresolved render/encoder symbols',
      ', '.join(bad_nm[:5]) if bad_nm else '')

print('\n[headless physics]')
env = tact.Env(os.path.join(TACT, 'tests/scenes/box_wall'), render=False)
y0 = env.reset()
y1 = env.step()
check(y0.shape == y1.shape, 'Env(render=False) reset/step works', f'y shape={y1.shape}')

print('\n[offscreen camera]')
scene = os.path.join(tempfile.gettempdir(), 'tact_packaging_camera_smoke')
with open(scene + '.yml', 'w') as f:
    f.write("""sim: {dt: 0.001}
bodies:
  - name: root
    shapes:
      - {type: box, pos: [0, 0, -0.05], param: [1.0, 1.0, 0.05], rgba: [0.8, 0.8, 0.8, 1.0]}
cameras:
  - {name: cam_rgb, type: rgb, body: root, pos: [0, -1.5, 0.8], euler: [0, -60, 0], res: [64, 48], vfov: 45, fps: 30}
  - {name: cam_depth, type: depth, body: root, pos: [0, -1.5, 0.8], euler: [0, -60, 0], res: [32, 24], vfov: 45, fps: 30}
""")

cam_env = tact.Env(scene, render=False)
cam_env.reset()
try:
    frames = list(cam_env.camera_frames())
except RuntimeError as e:
    skip('offscreen camera render unavailable on this host', str(e))
else:
    names = {name for name, _ in frames}
    sizes = {name: len(buf) for name, buf in frames}
    check(names == {'cam_rgb', 'cam_depth'} and all(n > 0 for n in sizes.values()),
          'camera_frames() returns rgb/depth payloads', str(sizes))

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
