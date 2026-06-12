"""Packaging-oriented smoke checks for tact's native library.

Run:
  uv run python tests/test_packaging_smoke.py

This is intentionally self-contained and does not require sibling robot
projects. It checks the properties that matter for a future pip package:
`libtact.so` loads without render runtime hard deps, headless physics works, and
offscreen camera rendering works when the host provides EGL/OpenGL/encoders.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

TACT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TACT)
INSTALLED_MODE = os.environ.get('TACT_PACKAGING_INSTALLED') == '1'
if not INSTALLED_MODE:
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

required_symbols = [
    'tact_load_model',
    'tact_destroy_model',
    'tact_model_info',
    'tact_create_state',
    'tact_destroy_state',
    'tact_step',
    'tact_q',
    'tact_qd',
]
missing_symbols = []
for sym in required_symbols:
    if nm.returncode != 0 or f' {sym}' not in nm.stdout:
        missing_symbols.append(sym)
check(not missing_symbols, 'libtact.so exports standalone tactbin C API',
      ', '.join(missing_symbols))

print('\n[headless physics]')
headless_scene = (os.path.join(tact.pkg_dir, 'demos', 'box_wall')
                  if INSTALLED_MODE else
                  os.path.join(TACT, 'tests/scenes/box_wall'))
env = tact.Env(headless_scene, render=False)
y0 = env.reset()
y1 = env.step()
check(y0.shape == y1.shape, 'Env(render=False) reset/step works', f'y shape={y1.shape}')

print('\n[tactbin]')
tactbin = os.path.join(tempfile.gettempdir(), 'packaging_arm2.tactbin')
src = os.path.join(tact.pkg_dir, 'demos', 'arm2.yml')
compile_cmd = cmd([sys.executable, '-m', 'tact.compile_model', src, '-o', tactbin])
check(compile_cmd.returncode == 0 and os.path.exists(tactbin),
      'python -m tact.compile_model writes tactbin',
      compile_cmd.stdout.strip())


class ModelInfo(ctypes.Structure):
    _fields_ = [
        ('nb', ctypes.c_int),
        ('nq', ctypes.c_int),
        ('n_shape', ctypes.c_int),
        ('n_pair', ctypes.c_int),
        ('n_frame', ctypes.c_int),
        ('n_feed', ctypes.c_int),
        ('y_size', ctypes.c_int),
        ('lam_size', ctypes.c_int),
        ('dt', ctypes.c_double),
    ]


try:
    clib = ctypes.CDLL(lib)
    clib.tact_load_model.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    clib.tact_load_model.restype = ctypes.c_int
    clib.tact_destroy_model.argtypes = [ctypes.c_void_p]
    clib.tact_model_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModelInfo)]
    clib.tact_model_info.restype = ctypes.c_int
    clib.tact_create_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    clib.tact_create_state.restype = ctypes.c_int
    clib.tact_destroy_state.argtypes = [ctypes.c_void_p]
    clib.tact_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    clib.tact_step.restype = ctypes.c_int
    clib.tact_q.argtypes = [ctypes.c_void_p]
    clib.tact_q.restype = ctypes.POINTER(ctypes.c_double)
    clib.tact_qd.argtypes = [ctypes.c_void_p]
    clib.tact_qd.restype = ctypes.POINTER(ctypes.c_double)

    m = ctypes.c_void_p()
    s = ctypes.c_void_p()
    rc_load = clib.tact_load_model(tactbin.encode(), ctypes.byref(m))
    info = ModelInfo()
    rc_info = clib.tact_model_info(m, ctypes.byref(info)) if rc_load == 0 else -1
    rc_state = clib.tact_create_state(m, ctypes.byref(s)) if rc_info == 0 else -1
    rc_step = clib.tact_step(m, s, None) if rc_state == 0 else -1
    q = np.ctypeslib.as_array(clib.tact_q(s), shape=(info.nq,)).copy() if rc_step == 0 else np.array([])
    qd = np.ctypeslib.as_array(clib.tact_qd(s), shape=(info.nq,)).copy() if rc_step == 0 else np.array([])
    if s:
        clib.tact_destroy_state(s)
    if m:
        clib.tact_destroy_model(m)

    py_model = tact.Model(os.path.splitext(src)[0])
    q_py, qd_py, _y, _ctx = py_model.step(py_model.q0, py_model.qd0,
                                          np.zeros_like(py_model.q0))
    check(rc_load == 0 and rc_info == 0 and rc_state == 0 and rc_step == 0 and
          info.nb == 2 and info.nq == 2 and
          np.allclose(q, q_py, rtol=0.0, atol=1e-12) and
          np.allclose(qd, qd_py, rtol=0.0, atol=1e-12),
          'ctypes-loaded tactbin C API steps with Python parity',
          f'rc=({rc_load},{rc_info},{rc_state},{rc_step}) nb={info.nb} nq={info.nq}')
except Exception as e:
    check(False, 'ctypes-loaded tactbin C API steps with Python parity', repr(e))

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
