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
    'tact_load',
    'tact_destroy',
    'tact_info',
    'tact_q0',
    'tact_qd0',
    'tact_create_ctx',
    'tact_destroy_ctx',
    'tact_step',
    'tact_render',
]
missing_symbols = []
for sym in required_symbols:
    if nm.returncode != 0 or f' {sym}' not in nm.stdout:
        missing_symbols.append(sym)
check(not missing_symbols, 'libtact.so exports standalone bin C API',
      ', '.join(missing_symbols))

print('\n[headless physics]')
headless_scene = (os.path.join(tact.pkg_dir, 'demos', 'box_wall')
                  if INSTALLED_MODE else
                  os.path.join(TACT, 'tests/scenes/box_wall'))
env = tact.Env(headless_scene, render=False)
y0 = env.reset()
y1 = env.step()
check(y0.shape == y1.shape, 'Env(render=False) reset/step works', f'y shape={y1.shape}')

print('\n[bin]')
bin_path = os.path.join(tempfile.gettempdir(), 'packaging_arm2.bin')
src = os.path.join(tact.pkg_dir, 'demos', 'arm2.yml')
compile_cmd = cmd([sys.executable, '-m', 'tact.compile', src, '-o', bin_path])
check(compile_cmd.returncode == 0 and os.path.exists(bin_path),
      'python -m tact.compile writes bin',
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
    clib.tact_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    clib.tact_load.restype = ctypes.c_int
    clib.tact_destroy.argtypes = [ctypes.c_void_p]
    clib.tact_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModelInfo)]
    clib.tact_info.restype = ctypes.c_int
    clib.tact_q0.argtypes = [ctypes.c_void_p]
    clib.tact_q0.restype = ctypes.POINTER(ctypes.c_double)
    clib.tact_qd0.argtypes = [ctypes.c_void_p]
    clib.tact_qd0.restype = ctypes.POINTER(ctypes.c_double)
    clib.tact_create_ctx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    clib.tact_create_ctx.restype = ctypes.c_int
    clib.tact_destroy_ctx.argtypes = [ctypes.c_void_p]
    clib.tact_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    clib.tact_step.restype = ctypes.c_int

    m = ctypes.c_void_p()
    ctx0 = ctypes.c_void_p()
    ctx1 = ctypes.c_void_p()
    rc_load = clib.tact_load(bin_path.encode(), ctypes.byref(m))
    info = ModelInfo()
    rc_info = clib.tact_info(m, ctypes.byref(info)) if rc_load == 0 else -1
    rc_ctx0 = clib.tact_create_ctx(m, ctypes.byref(ctx0)) if rc_info == 0 else -1
    rc_ctx1 = clib.tact_create_ctx(m, ctypes.byref(ctx1)) if rc_ctx0 == 0 else -1
    q = (np.ctypeslib.as_array(clib.tact_q0(m), shape=(info.nq,)).copy()
         if rc_ctx1 == 0 else np.array([]))
    qd = (np.ctypeslib.as_array(clib.tact_qd0(m), shape=(info.nq,)).copy()
          if rc_ctx1 == 0 else np.array([]))
    q_next = np.zeros(info.nq, dtype=np.float64)
    qd_next = np.zeros(info.nq, dtype=np.float64)
    y = np.zeros(max(info.y_size, 1), dtype=np.float64)
    tau = np.zeros(info.nq, dtype=np.float64)
    rc_step = (clib.tact_step(m,
                              q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              qd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              ctx0,
                              q_next.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              qd_next.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              ctx1)
               if rc_ctx1 == 0 else -1)
    if ctx0:
        clib.tact_destroy_ctx(ctx0)
    if ctx1:
        clib.tact_destroy_ctx(ctx1)
    if m:
        clib.tact_destroy(m)

    py_model = tact.Model(os.path.splitext(src)[0])
    q_py, qd_py, _y, _ctx = py_model.step(py_model.q0, py_model.qd0,
                                          np.zeros_like(py_model.q0))
    check(rc_load == 0 and rc_info == 0 and rc_ctx0 == 0 and rc_ctx1 == 0 and rc_step == 0 and
          info.nb == 2 and info.nq == 2 and
          np.allclose(q_next, q_py, rtol=0.0, atol=1e-12) and
          np.allclose(qd_next, qd_py, rtol=0.0, atol=1e-12),
          'ctypes-loaded bin C API steps with Python parity',
          f'rc=({rc_load},{rc_info},{rc_ctx0},{rc_ctx1},{rc_step}) nb={info.nb} nq={info.nq}')
except Exception as e:
    check(False, 'ctypes-loaded bin C API steps with Python parity', repr(e))

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
