import os, ctypes, ctypes.util

# libtact.so resolution order:
#   1. <package_dir>/bin/libtact.so — co-located with this _clib.py in the
#      inner `tact` import package. This is the normal editable-install and
#      wheel-install path for the `pytact` distribution.
#   2. ./bin/libtact.so      — legacy/ad-hoc development build in CWD.
#   3. ctypes.util.find_library('tact')  — OS dynamic loader (ldconfig / LD_LIBRARY_PATH /
#      /usr/lib, /usr/local/lib on Linux; DYLD_LIBRARY_PATH on macOS; PATH on Windows).
#      Use this when tact is installed system-wide outside the monorepo layout.
_pkg = os.path.dirname(os.path.abspath(__file__))
_mode = getattr(os, 'RTLD_LOCAL', 0)
if hasattr(os, 'RTLD_LAZY'):
    _mode |= os.RTLD_LAZY
if   os.path.exists(f'{_pkg}/bin/libtact.so'): clib = ctypes.CDLL(f'{_pkg}/bin/libtact.so', mode=_mode)
elif os.path.exists('./bin/libtact.so'):       clib = ctypes.CDLL('./bin/libtact.so', mode=_mode)
elif (_p := ctypes.util.find_library('tact')) is not None: clib = ctypes.CDLL(_p, mode=_mode)
else: raise ImportError('cannot find libtact.so (package / CWD / system loader)')

#ctypes signatures for the dynamics functions ported from Python (Stage A).
#all pointers are non-owning — caller (Python) keeps the underlying numpy arrays alive.
_DBL = ctypes.POINTER(ctypes.c_double)
_INT = ctypes.POINTER(ctypes.c_int)
_FLT = ctypes.POINTER(ctypes.c_float)

#--- Phase 1: handle-based step gateway (local/docs/design-c-state.md §3) ---
#nb, parent, jtype, X, I6, Ti, ff, sk, g, dt, integrator, n_shape, n_pair, ctype, cbody, cshape, ctran, cparam, craycast, cpair,
#  erp, slop, cfm_scale, v_rest_thresh, iters, tol   (last 6 = global LCP solver knobs from YAML sim:)
#(mu is now per-material in cparam, no longer a global; ff/sk are per-joint damping/spring -- C side subtracts ff*qd + sk*q from tau before the integrator;
# floss = per-DoF joint Coulomb friction (constraint row); armature = per-DoF rotor/reflected inertia; taulim = per-DoF actuator torque bound (box-bounded LCP actuator row, 0=unlimited); jnt_lo/jnt_hi = per-DoF joint limit range (constraint row, limited iff lo<hi); craycast is per-shape int flag: 1=visible to ray, 0=skipped)
# 11 _DBL before dt: X, I6, Ti, ff, sk, floss, armature, taulim, jnt_lo, jnt_hi, g
clib.tact_create_from_arrays.argtypes = [ctypes.c_int, _INT, _INT, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, _INT, _INT, _DBL, _DBL, _DBL, _INT, _INT,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_void_p)]
clib.tact_create_from_arrays.restype  = ctypes.c_int

clib.tact_destroy.argtypes = [ctypes.c_void_p]
clib.tact_destroy.restype = None

#h, q, qd, tau (raw actuation -- C subtracts ff*qd + sk*q internally before the integrator),
#Kp_j, Kd_j, q_ref, qd_ref (NULL ok), ctx (solver context, the unified PGS
#warm-start vector [contact | fric | limit]; NULL = cold start). The next
#warm-start lands in the engine-owned h->ctx_next (TactModel.ctx_next) —
#copy it out and pass your copy (pure threading, the sim.py path), or pass
#ctx_next itself to let the engine keep warm-start state internally.
clib.tact_step_lcp.argtypes     = [ctypes.c_void_p, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL]
clib.tact_step_lcp.restype      = ctypes.c_int

#h, n_feeds, kinds, offsets, idx, n_frames, fbody, ftran, ftran_inv, y_size
clib.tact_set_feedback.argtypes = [ctypes.c_void_p, ctypes.c_int, _INT, _INT, _INT, ctypes.c_int, _INT, _DBL, _DBL, ctypes.c_int]
clib.tact_set_feedback.restype  = None

#in-place setters — preserve arena → tact_* views stay valid
clib.tact_edit_model.argtypes   = [ctypes.c_void_p, _DBL, _DBL, _DBL]
clib.tact_edit_model.restype    = None

#Phase 4: query functions — frame loop in C. mode[k]: 0=3d, 1=6d.
clib.tact_fk.argtypes    = [ctypes.c_void_p, _DBL, ctypes.c_int, _INT, _INT, ctypes.c_char_p, _DBL]
clib.tact_fk.restype     = None

clib.tact_error.argtypes = [ctypes.c_void_p, _DBL, _DBL, ctypes.c_int, _INT, _INT, ctypes.c_char_p, _DBL]
clib.tact_error.restype  = None

clib.tact_jacob.argtypes = [ctypes.c_void_p, _DBL, ctypes.c_int, _INT, _INT, _DBL]
clib.tact_jacob.restype  = None

#h, q, g_override (NULL → use h->g), b_out (caller-allocated, length nb)
clib.tact_gravity.argtypes = [ctypes.c_void_p, _DBL, _DBL, _DBL]
clib.tact_gravity.restype  = None

#h, q, H_out (caller-allocated, nq*nq row-major)
clib.tact_inertia.argtypes = [ctypes.c_void_p, _DBL, _DBL]
clib.tact_inertia.restype  = None

#h, q, m (nb), c (3*nb row-major), J_out (3*nq row-major)
clib.tact_com_jacob.argtypes = [ctypes.c_void_p, _DBL, _DBL, _DBL, _DBL]
clib.tact_com_jacob.restype  = None

#h, q, m (nb), c (3*nb row-major), r_out (3,)
clib.tact_com.argtypes = [ctypes.c_void_p, _DBL, _DBL, _DBL, _DBL]
clib.tact_com.restype  = None

#h, q, qd, f_ext (NULL → treated as zero; else 6*nb row-major), b_out (length nq)
clib.tact_bias.argtypes = [ctypes.c_void_p, _DBL, _DBL, _DBL, _DBL]
clib.tact_bias.restype  = None

#h, q_in, x_d, n, frame_idx, mode, eulerseq, advance, tolerance, damping, max_iter, q_out
#returns iter count (≥0) on convergence, -iter count on max_iter without convergence
clib.tact_ik2.argtypes = [ctypes.c_void_p, _DBL, _DBL, ctypes.c_int, _INT, _INT, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, _DBL]
clib.tact_ik2.restype  = ctypes.c_int

#h, q, R0s (n*3, world), Rds (n*3, unit, world), n, t_out (n; -1 = no hit).
#n world rays with per-ray origins — the general primitive (height_scan,
#single-shot); no cone cull. Replaced tact_raycast_query 2026-06-06.
clib.tact_raycast_world.argtypes = [ctypes.c_void_p, _DBL, _DBL, _DBL, ctypes.c_int, _DBL]
clib.tact_raycast_world.restype  = None

#h, q, frame_idx, dirs (n*3, unit, registered-frame), n, t_out (n; -1 = no hit).
#Batched raycast from a sensor frame (shared origin → cone cull) — ray
#generation lives in Python (sim.py Env._ray_grid); replaced tact_raymap_query
#2026-06-06.
clib.tact_raycast_frame.argtypes = [ctypes.c_void_p, _DBL, ctypes.c_int, _DBL, ctypes.c_int, _DBL]
clib.tact_raycast_frame.restype  = None

#Mirror of the public tact_t head (native/tact.h). Field order/types must match
#the C struct exactly — a drift would read garbage, so sim.py cross-validates
#the mirrored dims (nb/nq/n_shape/n_pair/dt) against its own build values right
#after every handle creation and raises on mismatch. The private tail behind
#`core` is opaque; only the head is mirrored.
class TactModel(ctypes.Structure):
    _fields_ = [
        # model dimensions / parameters — fixed after create
        ('nb',            ctypes.c_int),
        ('nq',            ctypes.c_int),
        ('n_shape',       ctypes.c_int),
        ('n_pair',        ctypes.c_int),
        ('n_frames',      ctypes.c_int),
        ('y_size',        ctypes.c_int),
        ('ctx_size',      ctypes.c_int),
        ('contact_count', ctypes.c_int),   # step output: active contacts
        ('dt',            ctypes.c_double),
        ('q0',            _DBL),
        ('qd0',           _DBL),
        # engine-owned step outputs — read after tact_step_lcp
        ('f',             _DBL),
        ('a',             _DBL),
        ('v',             _DBL),
        ('q_next',        _DBL),
        ('qd_next',       _DBL),
        ('y',             _DBL),
        ('ctx_next',      _DBL),   # next warm-start ctx (copy out, or pass back as ctx for engine-kept warm-start)
        # private engine state — do not access
        ('core',          ctypes.c_void_p),
    ]


def model_view(h):
    """Live read-only view of a tact_t handle's public head.

    Attribute reads go straight to the C struct memory, so pointer fields
    updated after creation (e.g. `y` set by tact_set_feedback) read current."""
    return ctypes.cast(h, ctypes.POINTER(TactModel)).contents


clib.tact_contact_reports.argtypes = [ctypes.c_void_p, _INT, _DBL]
clib.tact_contact_reports.restype  = None

clib.tact_frame_count.argtypes = [ctypes.c_void_p]
clib.tact_frame_count.restype  = ctypes.c_int
clib.tact_frame_name.argtypes  = [ctypes.c_void_p, ctypes.c_int]
clib.tact_frame_name.restype   = ctypes.c_char_p
clib.tact_frame_id.argtypes    = [ctypes.c_void_p, ctypes.c_char_p]
clib.tact_frame_id.restype     = ctypes.c_int

# Register filesystem path for a mesh slot. Python resolves relative paths
# against the YAML file's directory before calling this.
clib.tact_set_mesh_path.argtypes = [ctypes.c_int, ctypes.c_char_p]
clib.tact_set_mesh_path.restype  = None

# Push a height-field grid into a slot. Python loads/scales the grid (nrow*ncol
# heights in meters, row-major) and passes it directly; C copies it.
clib.tact_set_hfield_data.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_double, ctypes.c_double, _DBL]
clib.tact_set_hfield_data.restype  = None

# Push light params (pos, target, ortho-frustum half-extent, shadow on/off) into render.c
# module statics. Called before each render; cheap (no GL state touched).
clib.tact_render_set_light.argtypes = [_FLT, _FLT, ctypes.c_float, ctypes.c_int]
clib.tact_render_set_light.restype  = None

# Narrow-phase collision detection (narrow.c). tact_collision_check is the multi-point
# dispatcher; tact_collision_check_mpr is the MPR fallback exposed for benchmarking and
# sign-convention regression tests.
# type ∈ {MESH=100, BOX=101, SPHERE=102, CYL=103, CAPSULE=104, HFIELD=105}.
# param layout: [pos(3), euler_xyz(3), shape_param(≤3)].
# out layout: 7 doubles per contact point (px, py, pz, nx, ny, nz, depth).
clib.tact_collision_check.argtypes     = [ctypes.c_int, _DBL, ctypes.c_int, _DBL, _DBL, ctypes.c_int]
clib.tact_collision_check.restype      = ctypes.c_int
clib.tact_collision_check_mpr.argtypes = [ctypes.c_int, _DBL, ctypes.c_int, _DBL, _DBL]
clib.tact_collision_check_mpr.restype  = ctypes.c_int
# Box-box specific detector (SAT + face clipping). Same out layout as tact_collision_check.
# param: 9 doubles each (pos, euler_xyz, half-extents). max_pts ≤ MAX_PTS_PER_PAIR.
clib.tact_box_box_manifold.argtypes    = [_DBL, _DBL, _DBL, ctypes.c_int]
clib.tact_box_box_manifold.restype     = ctypes.c_int
