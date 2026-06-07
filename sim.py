"""Simulation classes: Model (YAML → kinematic tree + step), Env (window +
EGL image buffer + add()), CEnv (ctypes-CDLL adapter for mujoco/chrono/real
backends). Pure math/dynamics primitives live in rbd.py and are re-exported
here via `from .rbd import *` so internal references stay flat."""
import sys, os, ctypes, math, copy
from typing import NamedTuple
import numpy as np
import yaml
from ._clib import clib, _DBL, _INT
from .rbd import *
from .rbd import _fk, _q_step, _build_qidx   # underscored names are not pulled in by `import *`


class SolverState(NamedTuple):
    """Persistent solver state threaded across Model.step() calls (the `ctx` arg).
    Externalizing it makes Model.step referentially transparent: same
    (q, qd, tau, ..., ctx) → same (q_next, qd_next, y, ctx_next). Env keeps one
    internally so Env users see no change. NamedTuple → JAX-pytree friendly for
    later batched/diff use. `ctx=None` = cold start (zero λ).

    `lam` is ONE vector holding every PGS warm-start λ, blocks in row-table
    order (the layout is the C↔Python ABI; tact_step_lcp slices it by the same
    arithmetic):

        [contact (6·MAX_PTS_PER_PAIR·max(n_pair,1), slot-indexed)
         | joint-friction (nq, per-DoF) | joint-limit (nq, per-DoF)]

    A future constraint-row type appends a block here (and bumps zero_state)
    instead of growing tact_step_lcp's signature / this tuple's fields. `nq` is
    layout metadata for the read-only block views below — introspection/debug
    only; step() consumes `lam` whole."""
    lam: np.ndarray          # unified PGS warm-start λ: [contact | fric | limit]
    nq: int                  # per-DoF block length (layout metadata)

    @property
    def lam_contact(self):   # contact-cone block, slot = cpair_idx*MAX_PTS_PER_PAIR + sub_id
        return self.lam[:len(self.lam) - 2 * self.nq]
    @property
    def lam_fric(self):      # joint Coulomb friction block, per-DoF
        c = len(self.lam) - 2 * self.nq
        return self.lam[c:c + self.nq]
    @property
    def lam_limit(self):     # joint limit block, per-DoF
        return self.lam[len(self.lam) - self.nq:]

# NOTE: the free-joint locking mechanism (YAML `lock: true` → magic 6-DoF PD
# wrench pinning the base at q0 until env.unlock()) was removed 2026-06-06,
# sim-trick reduction. Its real-world twin is a gantry — if staging is ever
# needed again, model it explicitly (a YAML gantry body) or start at a stable
# q0 held by implicit joint-PD (the zen pattern; gains pass per step via
# step(kp=, kd=) since YAML `k:` was removed 2026-06-07). The private
# _PIDController that powered it went with it.

# Sensor (camera/lidar) parse helpers. A sensor in YAML is "a frame plus publish
# metadata": _register_sensors injects each spec into its target body's `frames:`
# list (default body = root), so it flows through the normal frame machinery —
# default-fill, root-offset, fdict/fbody/ftran registration, and add/delete group
# handling — with no parallel code path. The publish metadata (everything but the
# frame-only pose keys) goes into a per-kind registry (Model.cameras / Model.lidars),
# normalized/validated by the `normalize` callback. Cameras and lidars share this so
# the two stay in lockstep; only `normalize` differs.
def _register_sensors(config, specs, prefix, registry, *, kind, normalize):
    if not specs: return
    body_by_name = {b['name']: b for b in config['bodies']}
    for s in specs:
        if 'name' not in s:
            raise ValueError(f"{kind} entry missing required `name`")
        bname = s.get('body', 'root')
        if bname not in body_by_name:
            raise ValueError(f"{kind} {s['name']!r} references unknown body {bname!r} "
                             f"(have: {sorted(body_by_name)})")
        # Frame-only keys position the sensor; everything else is publish metadata.
        frame = {'name': s['name'], 'pos': s.get('pos', [0, 0, 0]),
                 'euler': s.get('euler', [0, 0, 0])}
        if 'eulerseq' in s: frame['eulerseq'] = s['eulerseq']
        body_by_name[bname].setdefault('frames', []).append(frame)
        spec = {k: v for k, v in s.items()
                if k not in ('body', 'pos', 'euler', 'eulerseq')}
        # build() prefixes frame names; mirror it so spec['name'] matches the fdict key.
        spec['name'] = (prefix + s['name']) if prefix else s['name']
        normalize(spec)
        registry.append(spec)


def _normalize_camera(spec):
    # rgb → JPEG; depth → zstd float32 (encoders in Env.camera_frames). res [w,h]
    # sizes the EGL render (grow-only); vfov = vertical FOV (deg, MuJoCo fovy convention).
    spec.setdefault('type', 'rgb')
    if spec['type'] not in ('rgb', 'depth'):
        raise ValueError(f"camera {spec['name']!r}: unsupported type {spec['type']!r} "
                         f"(expected 'rgb' or 'depth')")
    spec['res'] = list(spec.get('res', [640, 480]))
    spec.setdefault('fps', 30)
    spec.setdefault('vfov', 45)  # vertical FOV (deg), like MuJoCo fovy


def _normalize_lidar(spec):
    # 2d → depth map, RAW float32 (range-along-ray in meters with -1 for no-hit;
    # zstd removed 2026-06-06 — depth CAMERAS still compress C-side). dth = degrees
    # per pixel (horizontal FoV = res[0]*dth). pinhole/perpendicular default to the
    # LiDAR convention (angular projection, range along ray).
    # 3d → sensor-frame points, RAW float32 (N, 3) — N varies per frame (no-hit
    # rays dropped; `max_range` drops far hits). perpendicular doesn't apply
    # (ranges are taken along the ray by construction).
    # Both encode inline in Env.lidar_frames over _ray_grid + tact_raycast_frame.
    spec.setdefault('type', '2d')
    if spec['type'] not in ('2d', '3d'):
        raise ValueError(f"lidar {spec['name']!r}: unsupported type {spec['type']!r} "
                         f"(must be '2d' or '3d')")
    spec['res'] = list(spec.get('res', [120, 80]))
    spec.setdefault('fps', 30)
    spec.setdefault('dth', 1.0)            # degrees per pixel (horizontal)
    spec.setdefault('pinhole', False)      # False = angular (LiDAR-like) projection
    spec.setdefault('perpendicular', False)  # False = range along ray (LiDAR convention)
    spec.setdefault('max_range', None)     # 3d only: drop hits beyond this (m)


class Model:
    def __init__(self, modelname, prefix=None, base='root', offset=[0, 0, 0, 0, 0, 0], q0=None, fixed_base=False, name=None):
        self.dt = 0.001
        # contact solver: 'lcp' (contact_lcp + semi-implicit Euler). The legacy
        # 'penalty' (spring-damper + brush friction) solver was removed 2026-05-24
        # (its brush could not hold a planted foot — see git/_ archive). 'lcp' only.
        self.solver = 'lcp'
        # LCP warm-start λ is no longer Model state — it is threaded through
        # Model.step(ctx) as a SolverState (Env holds one in self._ctx).
        self.g = [0, 0, 0] #[0, 0, -9.81]
        # global LCP solver knobs (overridable flat under YAML sim:); defaults match
        # the historical hardcoded values. erp/slop/cfm_scale = Baumgarte / penetration
        # deadband / CFM regularization; v_rest_thresh = restitution velocity gate;
        # iters/tol = PGS budget. lcp path only (ignored by solver: minimal).
        self.erp = 0.2
        self.slop = 1e-4
        self.cfm_scale = 1e-6
        self.v_rest_thresh = 3e-2
        self.iters = 20
        self.tol = 1e-6
        self.use_c = True
        self.view = [0, 0, 0, 3, 45, 20] #[target(3), distance(1), yaw(deg, 1), pitch(deg, 1)]

        # Render lights. YAML `lights: [...]` overrides on first build (partial-merge).
        # Today only lights[0] is used by render.c (single shadow caster); list format
        # left intentionally for future multi-light. Per-light keys:
        #   pos:    [x, y, z]      world position
        #   target: [x, y, z]      shadow camera look-at (default origin)
        #   ortho:  float          shadow frustum half-extent (smaller = sharper)
        #   shadow: bool           whether this light casts a shadow
        self.lights = [{'pos': [7.0, 7.0, 7.0], 'target': [0.0, 0.0, 0.0], 'ortho': 5.0, 'shadow': True}]

        self.jtype = []
        self.parent = []
        self.active = []
        self.fixed = []
        
        self.Ti = [] # relative transform to joint frame
        self.m = []  # mass array
        self.c = []  # relative translaition to body com
        self.I = []  # inertia matrix array

        self.q0 = np.array([], dtype=float)
        self.qd0 = np.array([], dtype=float)
        self.ff = np.array([], dtype=float)  # joint viscous damping coefficient
        self.sk = np.array([], dtype=float)  # joint spring stiffness
        self.floss = np.array([], dtype=float)  # joint Coulomb friction bound (frictionloss); solved as an LCP constraint row
        self.armature = np.array([], dtype=float)  # joint rotor/reflected inertia (MuJoCo armature); added to M diagonal + ABA d
        self.jnt_lo = np.array([], dtype=float)  # joint lower limit (rev: rad, lin: m); limited iff lo < hi
        self.jnt_hi = np.array([], dtype=float)  # joint upper limit; both solved as one-sided LCP constraint rows

        # NOTE: implicit joint-PD gains are NOT model state — `k:` was removed from
        # the YAML schema 2026-06-07 (gains are control-policy inputs, not plant
        # parameters; the YAML default had no claim to represent mode-dependent
        # gains). They pass per step: Model.step(kp=, kd=) / Env.step(kp=, kd=),
        # start reads controller.kp/.kd attrs. Joint-PD only — task-PD was
        # prototyped (Phase 3) but removed because it can't be implemented on
        # MuJoCo / real-hardware backends, breaking tact's "swap backend, same
        # agent" contract. Activation requires q_ref (or qd_ref) in step().

        self.ctype = []  # contact convex info
        self.cbody = []  # attached body index
        self.cshape = [] #contact support function shape parameter
        self.cparam = []  #contact parameters
        self.ctran = []  #transform
        self.crgba = []
        self.craycast = []  # per-shape int flag: 1=visible to raycast, 0=skipped (still renders + collides)
        
        self.f_idx = 1
        self.fdict = {'root': 0}
        self.fbody = [-1]
        self.ftran = [np.eye(4)]
        self.ftran_inv = [np.eye(4)]

        self.feeds = []
        # Sensor publish registries, declared via the YAML top-level `cameras:` /
        # `lidars:` blocks. Each entry is the publish metadata ({name, type, res, fps,
        # ...}); the sensor's pose is registered as a frame (in fdict) attached to its
        # `body`, so the frame-pose resolution inside Env.camera_frames/lidar_frames
        # looks it up by name. `start` iterates env.cameras / env.lidars to bind one
        # ZMQ PUB per sensor and publish at each sensor's own rate.
        self.cameras = []
        self.lidars = []

        # Per-add() group ledger for delete(name). Each entry records the
        # half-open [lo, hi) ranges this add inserted into each parallel array,
        # plus the fdict keys it registered. delete() uses these to splice
        # arrays and reindex remaining bodies/frames.
        self.groups = []

        # Mesh slot allocation: YAML `file:` path → C-side mesh slot index.
        # Slots are global (one C-side table per process), deduplicated by
        # resolved absolute path. add()/delete() do not free slots (~64 slot
        # budget; rare to exhaust in practice — see Q2 design note in
        # CLAUDE.md's "Dynamic add/delete" section).
        self.mesh_path_to_idx = {}    # abs_path → idx
        self._mesh_max_slots = 64     # matches MAX_MESH in shape.h
        self.hf_next_slot = 0         # next free height-field slot (monotonic; slots not freed)
        self._hf_max_slots = 16       # matches MAX_HFIELD in shape.h
        self.add(modelname, prefix, base, offset, q0, fixed_base, name=name)

    def _snapshot_sizes(self):
        return {
            'nb':     len(self.jtype),
            'nq':     len(self.q0),
            'nshape': len(self.ctype),
            'nframe': len(self.fbody),
            'nfeeds': len(self.feeds),
            'ncameras': len(self.cameras),
            'nlidars': len(self.lidars),
            'nfixed': len(self.fixed),
        }

    def add(self, modelname, prefix=None, base='root', offset=[0, 0, 0, 0, 0, 0], q0=None, fixed_base=False, name=None):
        # Resolve group name. Default to prefix or modelname; auto-suffix on collision
        # so duplicate calls (e.g., `add('box')` ×N) each get a unique handle.
        if name is None: base_name = prefix if prefix is not None else modelname
        else: base_name = name
        gname = base_name
        suffix = 1
        existing = {g['name'] for g in self.groups}
        while gname in existing:
            gname = f'{base_name}_{suffix}'
            suffix += 1
        if name is not None and gname != name:
            raise ValueError(f"group name {name!r} already exists (have: {sorted(existing)})")

        # Snapshot sizes before mutation; delete() needs both lo and hi.
        before = self._snapshot_sizes()
        fdict_before = set(self.fdict.keys())

        filename = modelname + '.yml'
        if not os.path.exists(filename): filename = modelname + '.yaml'
        # Directory the YAML lives in — relative `file:` paths for mesh shapes
        # resolve against this so a model can ship its own objs/ alongside.
        yml_dir = os.path.dirname(os.path.abspath(filename))

        with open(filename, 'r') as f:
            text = f.read()
            #text = replace_rand(text)
            config = yaml.safe_load(text)

        # Sensors (cameras + lidars): each is a frame plus publish metadata. The shared
        # _register_sensors helper injects each spec into its target body's `frames:` list
        # (default body = root) so it reuses the full frame machinery below — default-fill,
        # root-offset application, fdict/fbody/ftran registration, and add/delete group
        # handling — without a parallel code path. Env.camera_frames/lidar_frames then
        # look the sensor up by frame name (cameras: egl_render;
        # lidars: raymap/raycloud). Only the normalized publish metadata is kept, in
        # self.cameras / self.lidars. Extra keys (e.g. a transport `port` for a runner that
        # also binds a LAN socket) pass through untouched.
        _register_sensors(config, config.get('cameras', []) or [], prefix,
                           self.cameras, kind='camera', normalize=_normalize_camera)
        _register_sensors(config, config.get('lidars', []) or [], prefix,
                           self.lidars, kind='lidar', normalize=_normalize_lidar)

        offset0 = np.array(offset)
        T0 = xyzeuler_to_homogeneous(offset0, eulerseq='XYZ', deg=True)
        q0_idx = 0

        #materials library: contact: [pair_id, mat_name] expanded into a 13-tuple
        #  [pair_id, k_n, d_n, k_t, d_t, mu, k_spin, d_spin, mu_spin, k_roll, d_roll, mu_roll, restitution]
        #material spec is grouped by physical concept:
        #  {normal: [k_n, d_n], tangent: [k_t, d_t, mu], spin: [k_spin, d_spin, mu_spin], roll: [k_roll, d_roll, mu_roll]}
        #plus an optional scalar `restitution: e` (coefficient of restitution; default 0.0 =
        #fully inelastic, matching the pre-2026-05-25 hardcoded behavior). When two materials
        #meet, e is combined by min(e_i, e_j) — the more dissipative surface caps the rebound
        #(a superball on clay does not bounce); min also reproduces e for identical materials.
        materials = config.get('materials', {}) or {}
        _MAT_GROUPS = (('normal', 2), ('tangent', 3), ('spin', 3), ('roll', 3))

        #restitution is physically [0, 1]. We don't clamp (out-of-range can be useful
        #for experiments) but warn once per material: e<0 is silently treated as 0
        #(fully inelastic — max(b_baum, b_rest) discards the negative bias); e>1 injects
        #energy each bounce and may diverge.
        for _mn, _m in materials.items():
            if 'restitution' in _m:
                _e = float(_m['restitution'])
                if not (0.0 <= _e <= 1.0):
                    print(f"[tact] warning: material '{_mn}' restitution={_e} outside physical range [0, 1] — "
                          + ("e<0 behaves as 0 (fully inelastic)" if _e < 0 else "e>1 injects energy and may diverge"))

        #fill mandatory items if empty
        for body in config['bodies']:
            if 'joint' in body:
                if 'pos' not in body['joint']: body['joint']['pos'] = [0, 0, 0]
                if 'euler' not in body['joint']: body['joint']['euler'] = [0, 0, 0]
                if 'eulerseq' not in body['joint']: body['joint']['eulerseq'] = 'XYZ'

            if 'inertial' in body:
                if 'pos' not in body['inertial']: body['inertial']['pos'] = [0, 0, 0]
                if 'euler' not in body['inertial']: body['inertial']['euler'] = [0, 0, 0]
                if 'eulerseq' not in body['inertial']: body['inertial']['eulerseq'] = 'XYZ'

            if 'shapes' in body:
                for i in range(len(body['shapes'])):
                    sh = body['shapes'][i]
                    if 'pos' not in sh: sh['pos'] = [0, 0, 0]
                    if 'euler' not in sh: sh['euler'] = [0, 0, 0]
                    if 'eulerseq' not in sh: sh['eulerseq'] = 'XYZ'
                    # Color: rgba [r,g,b,a]; a = opacity in [0,1] (a < 1 renders
                    # translucent via back-to-front alpha blending). a is optional
                    # and defaults to 1.0 (opaque). Shapes with no rgba get the
                    # render-skip sentinel [-1,0,0,1.0] (rgba[0] < 0 = invisible +
                    # non-raycast). The legacy `rgb` key is no longer read (silently
                    # ignored — migrate to rgba).
                    if 'rgba' in sh:
                        c = list(sh['rgba'])
                        if len(c) == 3: c = c + [1.0]
                        if len(c) != 4:
                            raise ValueError(f"body '{body['name']}' shape #{i}: rgba must have 3 or 4 values [r,g,b(,a)]")
                        sh['rgba'] = c
                    else:
                        sh['rgba'] = [-1, 0, 0, 1.0]
                    if 'contact' not in sh:
                        sh['contact'] = [-1] + [0.0]*12
                    else:
                        contact = sh['contact']
                        pair_id = int(contact[0])
                        if pair_id < 0:
                            sh['contact'] = [pair_id] + [0.0]*12
                        else:
                            if len(contact) < 2:
                                raise ValueError(f"body '{body['name']}' shape #{i}: contact: [{pair_id}] is missing material name")
                            mat_name = contact[1]
                            if mat_name not in materials:
                                raise ValueError(f"body '{body['name']}' shape #{i}: unknown material '{mat_name}' (available: {sorted(materials)})")
                            m = materials[mat_name]
                            flat = []
                            for grp, n in _MAT_GROUPS:
                                if grp not in m: raise ValueError(f"material '{mat_name}' missing group '{grp}'")
                                vals = m[grp]
                                if len(vals) != n: raise ValueError(f"material '{mat_name}' group '{grp}' needs {n} values, got {len(vals)}")
                                flat.extend(float(x) for x in vals)
                            #optional restitution (scalar); default 0.0 = fully inelastic
                            flat.append(float(m.get('restitution', 0.0)))
                            sh['contact'] = [pair_id] + flat

                    # Resolve mesh `file:` → C-side slot index. Path is resolved
                    # against the YAML's directory (yml_dir) when relative.
                    # Dedup by absolute path so the same mesh asset reused across
                    # bodies/shapes shares one slot.
                    if sh['type'] == 'mesh':
                        if 'file' not in sh:
                            raise ValueError(f"body '{body['name']}' shape #{i}: mesh shape requires `file: <path>` (legacy `param: [N]` no longer supported)")
                        rel = sh['file']
                        abs_path = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(yml_dir, rel))
                        if not os.path.exists(abs_path):
                            raise ValueError(f"body '{body['name']}' shape #{i}: mesh file not found: {abs_path}")
                        if abs_path in self.mesh_path_to_idx:
                            slot = self.mesh_path_to_idx[abs_path]
                        else:
                            slot = len(self.mesh_path_to_idx)
                            if slot >= self._mesh_max_slots:
                                raise ValueError(f"mesh slot table exhausted (MAX_MESH={self._mesh_max_slots}); reduce unique mesh paths")
                            self.mesh_path_to_idx[abs_path] = slot
                            # Push the path to C immediately so subsequent
                            # load_obj / render path lookups succeed.
                            clib.set_mesh_path(slot, abs_path.encode())
                        # Normalize the YAML so downstream build() sees the
                        # same shape struct it did before (cshape[0] = idx).
                        sh['param'] = [float(slot)]

                    # Resolve heightfield grid → C-side slot. The grid is loaded
                    # here (numpy reads .npy directly, or an inline `data:` list)
                    # and pushed to C via set_hfield_data; cshape[0]=slot mirrors
                    # the mesh slot scheme. size: [sx, sy, sz] — sx,sy are XY
                    # half-extents (m); sz multiplies grid values into meters
                    # (use sz=1 for a grid already in meters, or sz=elevation for
                    # a normalized 0..1 heightmap).
                    elif sh['type'] == 'hfield':
                        if 'file' in sh:
                            rel = sh['file']
                            abs_path = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(yml_dir, rel))
                            if not os.path.exists(abs_path):
                                raise ValueError(f"body '{body['name']}' shape #{i}: hfield file not found: {abs_path}")
                            grid = np.load(abs_path)
                        elif 'data' in sh:
                            grid = np.asarray(sh['data'])
                        else:
                            raise ValueError(f"body '{body['name']}' shape #{i}: hfield requires `file:` (.npy) or inline `data:`")
                        grid = np.ascontiguousarray(grid, dtype=np.float64)
                        if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
                            raise ValueError(f"body '{body['name']}' shape #{i}: hfield grid must be 2-D and >=2x2, got {grid.shape}")
                        sz_spec = sh.get('size', None)
                        if sz_spec is None or len(sz_spec) != 3:
                            raise ValueError(f"body '{body['name']}' shape #{i}: hfield requires size: [sx, sy, sz]")
                        nrow, ncol = int(grid.shape[0]), int(grid.shape[1])
                        sx, sy, sz = float(sz_spec[0]), float(sz_spec[1]), float(sz_spec[2])
                        heights = np.ascontiguousarray(grid * sz, dtype=np.float64).ravel()  # row-major data[i*ncol+j]
                        if self.hf_next_slot >= self._hf_max_slots:
                            raise ValueError(f"hfield slot table exhausted (MAX_HFIELD={self._hf_max_slots})")
                        slot = self.hf_next_slot
                        self.hf_next_slot += 1
                        clib.set_hfield_data(slot, nrow, ncol, sx, sy, heights.ctypes.data_as(_DBL))
                        sh['param'] = [float(slot)]

            if 'frames' in body:
                for i in range(len(body['frames'])):
                    if 'pos' not in body['frames'][i]: body['frames'][i]['pos'] = [0, 0, 0]
                    if 'euler' not in body['frames'][i]: body['frames'][i]['euler'] = [0, 0, 0]
                    if 'eulerseq' not in body['frames'][i]: body['frames'][i]['eulerseq'] = 'XYZ'
                    
        for body in config['bodies']:
            #root offset adjust
            if body['name'] == 'root':
                
                #if 'inertial' in body:
                #    for i in range(len(body['inertial'])):
                #        offset1 = body['inertial']['pos'] + body['inertial']['euler']
                #        T1 = xyzeuler_to_homogeneous(offset1, eulerseq=body['inertial']['eulerseq'], deg=True)
                #        T2 = T0 @ T1
                #        offset2 = homogeneous_to_xyzeuler(T2, eulerseq=body['inertial']['eulerseq'], deg=True)
                #        body['inertial']['pos'] = [float(offset2[0]), float(offset2[1]), float(offset2[2])]
                #        body['inertial']['euler'] = [float(offset2[3]), float(offset2[4]), float(offset2[5])]

                if 'shapes' in body:
                    for i in range(len(body['shapes'])):
                        offset1 = body['shapes'][i]['pos'] + body['shapes'][i]['euler']
                        T1 = xyzeuler_to_homogeneous(offset1, eulerseq=body['shapes'][i]['eulerseq'], deg=True)
                        T2 = T0 @ T1
                        offset2 = homogeneous_to_xyzeuler(T2, eulerseq=body['shapes'][i]['eulerseq'], deg=True)
                        body['shapes'][i]['pos'] = [float(offset2[0]), float(offset2[1]), float(offset2[2])]
                        body['shapes'][i]['euler'] = [float(offset2[3]), float(offset2[4]), float(offset2[5])]

                if 'frames' in body:
                    for i in range(len(body['frames'])):
                        offset1 = body['frames'][i]['pos'] + body['frames'][i]['euler']
                        T1 = xyzeuler_to_homogeneous(offset1, eulerseq=body['frames'][i]['eulerseq'], deg=True)
                        T2 = T0 @ T1
                        offset2 = homogeneous_to_xyzeuler(T2, eulerseq=body['frames'][i]['eulerseq'], deg=True)
                        body['frames'][i]['pos'] = [float(offset2[0]), float(offset2[1]), float(offset2[2])]
                        body['frames'][i]['euler'] = [float(offset2[3]), float(offset2[4]), float(offset2[5])]
                            
                if base != 'root': body['name'] = '*' + base #<-------------- Add '*' to root of base changed model
                continue
            
            #floating body offset adjust (axis-angle free joint, jtype=3)
            if body['joint']['type'] == 'free':
                if 'q0' not in body['joint']: body['joint']['q0'] = [0, 0, 0, 0, 0, 0]
                T2 = xyzeuler_to_homogeneous(body['joint']['q0'], eulerseq=body['joint']['eulerseq'], deg=True)
                free_q0 = homogeneous_to_xyzeuler(T0 @ T2, eulerseq=body['joint']['eulerseq'], deg=True)
                body['joint']['q0'] = free_q0.tolist()
                continue

            #none-floating body base & offset adjust
            if body['joint']['parent'] == 'root':
                #change base 
                body['joint']['parent'] = base
                
                #change base position
                offset1 = body['joint']['pos'] + body['joint']['euler']
                T1 = xyzeuler_to_homogeneous(offset1, eulerseq=body['joint']['eulerseq'], deg=True)
                T2 = T0 @ T1
                offset2 = homogeneous_to_xyzeuler(T2, eulerseq=body['joint']['eulerseq'], deg=True)
                body['joint']['pos'] = [float(offset2[0]), float(offset2[1]), float(offset2[2])]
                body['joint']['euler'] = [float(offset2[3]), float(offset2[4]), float(offset2[5])]
                
            #none-free joint q0 override
            if q0 != None:
                if body['joint']['type'] != 'fixed': 
                    if 'q0' not in body['joint']: body['joint']['q0'] = 0
                    body['joint']['q0'] = q0[q0_idx]
                    q0_idx += 1

        if fixed_base:
            i = 0
            base_frames = []
            while True:
                if config['bodies'][i]['name'] == 'root':
                    del config['bodies'][i]
                    continue

                if 'joint' in config['bodies'][i]:
                    if config['bodies'][i]['joint']['type'] == 'free':
                        head = config['bodies'][i]['name']
                        # keep the free body's frames (incl. sensor frames injected
                        # by _register_sensors): in a fixed-base model the world IS
                        # the base origin, so re-anchoring them on root preserves
                        # their pos/euler exactly. (The original root's frames are
                        # still dropped — their anchor, the floating world, doesn't
                        # exist in this model.)
                        base_frames += config['bodies'][i].get('frames', [])
                        del config['bodies'][i]
                        continue

                    if config['bodies'][i]['joint']['parent'] == head:
                        config['bodies'][i]['joint']['parent'] = 'root'

                    i += 1
                if i == len(config['bodies']): break;
            if base_frames:
                config['bodies'].insert(0, {'name': 'root', 'frames': base_frames})

        self.build(config, prefix, modelname)

        after = self._snapshot_sizes()
        self.groups.append({
            'name':       gname,
            'nb':         (before['nb'],     after['nb']),
            'nq':         (before['nq'],     after['nq']),
            'nshape':     (before['nshape'], after['nshape']),
            'nframe':     (before['nframe'], after['nframe']),
            'nfeeds':     (before['nfeeds'], after['nfeeds']),
            'ncameras':   (before['ncameras'], after['ncameras']),
            'nlidars':    (before['nlidars'], after['nlidars']),
            'nfixed':     (before['nfixed'], after['nfixed']),
            'fdict_keys': list(set(self.fdict.keys()) - fdict_before),
        })

    def reset(self):
        # State restoration lives in Env.reset (q/qd/ctx); Model itself is
        # stateless across steps. (Used to re-arm the free-joint lock —
        # mechanism removed 2026-06-06.)
        pass

    def get_inertia_matrix(self, body):
        if   body['inertial']['tensor'][0] == 'zero': I = np.zeros((3, 3))
        elif body['inertial']['tensor'][0] == 'diag': I = np.diag(body['inertial']['tensor'][1:])        
        elif body['inertial']['tensor'][0] == 'sphere':
            m = body['inertial']['mass']
            r = body['inertial']['tensor'][1]
            I = np.array([[0.4*m*r*r, 0, 0], [0, 0.4*m*r*r, 0], [0, 0, 0.4*m*r*r]])
            
        elif body['inertial']['tensor'][0] == 'z-cyl':
            m = body['inertial']['mass']
            r = body['inertial']['tensor'][1]
            h = body['inertial']['tensor'][2]
            I = np.array([[m*(3*r*r+h*h)/12, 0, 0], [0, m*(3*r*r+h*h)/12, 0], [0, 0, m*r*r/2]])

        elif body['inertial']['tensor'][0] == 'box':
            m = body['inertial']['mass']
            x = body['inertial']['tensor'][1]
            y = body['inertial']['tensor'][2]
            z = body['inertial']['tensor'][3]
            I = np.array([[m*(y*y+z*z)/12, 0, 0], [0, m*(x*x+z*z)/12, 0], [0, 0, m*(x*x+y*y)/12]])

        R = euler_to_rotation(body['inertial']['euler'], eulerseq=body['inertial']['eulerseq'], deg=True)
        return R @ I @ R.T
    
    def build(self, config, prefix, modelname=None):
        # Globals (simulation/view) are applied only on the initial load.
        # Subsequent add() calls that carry these keys get a warning and ignore them
        # — globals belong on the root model's YAML `sim:`/`view:` block.
        is_first = (len(self.parent) == 0)

        # YAML format:
        #   sim:    {solver: lcp, dt: 0.001, g: [...]}
        #   view:   {target: [x,y,z], distance: d, yaw: deg, pitch: deg}
        #   lights: [{pos, target, ortho, shadow}, ...]   # only [0] used today
        SIM_KEYS    = ('sim',)
        VIEW_KEYS   = ('view',)
        LIGHTS_KEYS = ('lights',)

        if is_first:
            if 'sim' in config:
                s = config['sim']
                if 'solver' in s:
                    if s['solver'] == 'penalty':
                        raise ValueError("the 'penalty' solver was removed (2026-05-24); "
                                         "change `solver: penalty` to `solver: lcp` in the YAML "
                                         "(contact params may need retuning).")
                    if s['solver'] not in ('lcp', 'minimal'):
                        raise ValueError(f"unknown solver in YAML: {s['solver']!r} "
                                         "(supported: 'lcp', or 'minimal' = test-only "
                                         "spring-damper ground contact for spheres)")
                    self.solver = s['solver']
                if 'dt' in s: self.dt = s['dt']
                if 'g'  in s: self.g  = np.array(s['g'])
                # global LCP solver knobs, flat under sim: (lcp path only). Each falls
                # back to the default set in __init__ when absent.
                if 'erp'           in s: self.erp           = float(s['erp'])
                if 'slop'          in s: self.slop          = float(s['slop'])
                if 'cfm_scale'     in s: self.cfm_scale     = float(s['cfm_scale'])
                if 'v_rest_thresh' in s: self.v_rest_thresh = float(s['v_rest_thresh'])
                if 'iters'         in s: self.iters         = int(s['iters'])
                if 'tol'           in s: self.tol           = float(s['tol'])

            if 'view' in config:
                v = config['view']
                if not isinstance(v, dict):
                    raise ValueError(f"view must be a dict (target/distance/yaw/pitch), got {type(v).__name__}")
                # partial-merge against current defaults
                t = v.get('target', self.view[0:3])
                self.view = [t[0], t[1], t[2],
                             v.get('distance', self.view[3]),
                             v.get('yaw',      self.view[4]),
                             v.get('pitch',    self.view[5])]

            if 'lights' in config:
                # Partial-merge each light dict against current defaults so YAML can
                # omit fields it doesn't care about.
                merged = []
                for i, spec in enumerate(config['lights']):
                    base = self.lights[i] if i < len(self.lights) else self.lights[0]
                    merged.append({**base, **(spec or {})})
                self.lights = merged
        else:
            present = [k for k in SIM_KEYS + VIEW_KEYS + LIGHTS_KEYS if k in config]
            if present:
                src = modelname or prefix or '?'
                #print(f"[tact] warn: {src}.yml has top-level {present}; "
                #      f"globals only apply on the initial Model load — "
                #      f"ignored on add().")
        
        for body in config['bodies']:
            if prefix == None: name = body['name']
            elif body['name'] == 'root': name = body['name']
            elif body['name'][0] == '*': body['name'] = body['name'][1:]; name = body['name']  #<----remove '*' and restore base link name
            else: name = prefix + body['name']
            
            if 'joint' in body and body['joint']['type'] == 'free':
                # Single body, jtype=3, 6 DoFs (axis-angle free joint).
                # YAML q0 = [px, py, pz, ex, ey, ez] with Euler angles in degrees;
                # internally q[3:6] becomes the rotation vector (radians) via
                # Euler → R → log. qd[0:3] is body-frame v, qd[3:6] is body-frame ω.
                num = len(self.jtype)
                self.parent.append(None)
                self.jtype.append(3)

                # per-DoF arrays (6 entries) — match the q layout
                for k in range(6):
                    self.ff       = np.append(self.ff, 0)
                    self.sk       = np.append(self.sk, 0)
                    self.floss    = np.append(self.floss, 0)   # free-joint DoFs: no Coulomb friction (v1)
                    self.armature = np.append(self.armature, 0)
                    self.jnt_lo   = np.append(self.jnt_lo, 0)   # free-joint DoFs: no limits (v1)
                    self.jnt_hi   = np.append(self.jnt_hi, 0)
                    self.active.append(0)

                self.m.append(body['inertial']['mass'])
                self.c.append(np.asarray(body['inertial']['pos'], dtype=np.float64))
                self.I.append(self.get_inertia_matrix(body))

                # jcalc6 computes the joint transform directly from q[0:3] (position)
                # and q[3:6] (rotation vector) — no Euler-aware Ti chain needed.
                self.Ti.append(np.eye(4))

                if 'q0' in body['joint']:
                    q0_y = body['joint']['q0']
                    p0   = np.array(q0_y[:3], dtype=float)
                    R0   = euler_to_rotation(q0_y[3:6],
                                             eulerseq=body['joint']['eulerseq'], deg=True)
                    w0   = logmap_so3(R0)
                    self.q0 = np.append(self.q0, np.concatenate([p0, w0]))
                else:
                    self.q0 = np.append(self.q0, [0, 0, 0, 0, 0, 0])

                # qd0: body-frame [v; ω] in m/s and rad/s — no conversion.
                if 'qd0' in body['joint']:
                    self.qd0 = np.append(self.qd0, body['joint']['qd0'])
                else:
                    self.qd0 = np.append(self.qd0, [0, 0, 0, 0, 0, 0])

                self.fdict[name] = self.f_idx
                self.fbody.append(len(self.jtype) - 1)
                self.ftran.append(np.eye(4))
                self.ftran_inv.append(np.eye(4))
                self.f_idx += 1

            elif 'joint' in body:
                if   body['joint']['type'] == 'fixed': self.fixed.append(len(self.jtype)); self.jtype.append(0)
                elif body['joint']['type'] == 'rev': self.jtype.append(1); self.active.append(1)
                elif body['joint']['type'] == 'lin': self.jtype.append(2); self.active.append(1)

                self.Ti.append(xyzeuler_to_homogeneous(body['joint']['pos'] + body['joint']['euler'], eulerseq=body['joint']['eulerseq'], deg=True))
                #print(len(self.Ti), body['joint']['pos'] + body['joint']['euler'])

                # Fixed joints carry no per-DoF state (q0/qd0/ff/sk/Kp_j/Kd_j slots).
                if body['joint']['type'] != 'fixed':
                    if 'q0' in body['joint']:
                        if   body['joint']['type'] == 'rev': self.q0 = np.append(self.q0, np.deg2rad(body['joint']['q0']))
                        elif body['joint']['type'] == 'lin': self.q0 = np.append(self.q0, body['joint']['q0'])
                    else: self.q0 = np.append(self.q0, 0)

                    if 'qd0' in body['joint']:
                        if   body['joint']['type'] == 'rev': self.qd0 = np.append(self.qd0, np.deg2rad(body['joint']['qd0']))
                        elif body['joint']['type'] == 'lin': self.qd0 = np.append(self.qd0, body['joint']['qd0'])
                    else: self.qd0 = np.append(self.qd0, 0)

                    if 'damping' in body['joint']: self.ff = np.append(self.ff, body['joint']['damping'])
                    else: self.ff = np.append(self.ff, 0)

                    if 'spring' in body['joint']: self.sk = np.append(self.sk, body['joint']['spring'])
                    else: self.sk = np.append(self.sk, 0)

                    # joint Coulomb friction (MuJoCo frictionloss): per-DoF force/torque
                    # bound, solved as an LCP constraint row (rev/lin only). 0 = off.
                    if 'frictionloss' in body['joint']: self.floss = np.append(self.floss, body['joint']['frictionloss'])
                    else: self.floss = np.append(self.floss, 0)

                    # joint armature (MuJoCo): rotor/reflected inertia added to the M
                    # diagonal + ABA d. kg·m² (rev) / kg (lin). 0 = off.
                    if 'armature' in body['joint']: self.armature = np.append(self.armature, body['joint']['armature'])
                    else: self.armature = np.append(self.armature, 0)

                    # joint range limit: `limit: [lo, hi]` — DEGREES for rev / m for lin
                    # (same convention as q0), stored internally in rad/m. Solved as
                    # one-sided LCP constraint rows. Limited iff lo < hi; absent → [0,0]
                    # = unlimited.
                    if 'limit' in body['joint']:
                        lim = body['joint']['limit']
                        if body['joint']['type'] == 'rev':
                            self.jnt_lo = np.append(self.jnt_lo, np.deg2rad(lim[0]))
                            self.jnt_hi = np.append(self.jnt_hi, np.deg2rad(lim[1]))
                        else:  # lin: meters
                            self.jnt_lo = np.append(self.jnt_lo, lim[0])
                            self.jnt_hi = np.append(self.jnt_hi, lim[1])
                    else:
                        self.jnt_lo = np.append(self.jnt_lo, 0)
                        self.jnt_hi = np.append(self.jnt_hi, 0)

                    # `k:` (implicit joint-PD gains) was removed from the YAML schema
                    # (2026-06-07): gains are control-policy inputs, not plant
                    # parameters — they pass per step (env.step(kp=, kd=); start
                    # reads controller.kp/.kd attrs). YAML keeps only plant params.
                    if 'k' in body['joint']:
                        raise ValueError(
                            f"joint `k:` was removed from the YAML schema (2026-06-07) — "
                            f"gains are per-step control inputs now: delete `k:` from "
                            f"body '{body['name']}' and pass kp/kd to step() (start "
                            f"reads controller.kp/.kd attrs)")

                if body['joint']['parent'] == 'root': self.parent.append(None)
                elif prefix == None: self.parent.append(self.fbody[self.fdict[body['joint']['parent']]])
                elif body['joint']['parent'] in self.fdict.keys(): self.parent.append(self.fbody[self.fdict[body['joint']['parent']]]) #<---- base change case
                #else: self.parent.append(self.fbody[self.fdict[prefix + '.' + body['joint']['parent']]])
                else: self.parent.append(self.fbody[self.fdict[prefix + body['joint']['parent']]])

                self.fdict[name] = self.f_idx
                self.fbody.append(len(self.jtype)-1)
                self.ftran.append(np.eye(4))
                self.ftran_inv.append(np.eye(4))
                self.f_idx += 1
                
                self.m.append(body['inertial']['mass'])
                self.c.append(np.asarray(body['inertial']['pos'], dtype=np.float64))
                self.I.append(self.get_inertia_matrix(body))

            if 'frames' in body:
                for v in body['frames']:
                    if prefix == None: framename = v['name']
                    #else: framename = prefix + '.' + v['name']
                    else: framename = prefix + v['name']

                    self.fdict[framename] = self.f_idx

                    if name == 'root': self.fbody.append(-1)
                    else: self.fbody.append(len(self.jtype)-1)
                    
                    self.ftran.append(xyzeuler_to_homogeneous(v['pos'] + v['euler'], eulerseq=v['eulerseq'], deg=True))
                    self.ftran_inv.append(np.linalg.inv(self.ftran[-1]))
                    self.f_idx += 1
                    
            if 'shapes' in body:
                for v in body['shapes']:
                    if   v['type'] == 'mesh': num = 100
                    elif v['type'] == 'box':  num = 101
                    elif v['type'] == 'sphere':   num = 102
                    elif v['type'] == 'cylinder': num = 103
                    elif v['type'] == 'capsule':  num = 104
                    elif v['type'] == 'hfield':   num = 105

                    if name == 'root': self.cbody.append(-1)
                    else: self.cbody.append(self.fbody[self.fdict[name]])

                    self.ctype.append(num)
                    self.ctran.append(xyzeuler_to_homogeneous(v['pos'] + v['euler'], eulerseq=v['eulerseq'], deg=True))
                    self.cshape.append(v['param'])
                    self.cparam.append(v['contact'])
                    # raycast default = visible (rgba[0] >= 0). rgba[0]<0 is the render-skip sentinel,
                    # so an invisible shape is also invisible to ray sensors. Override with `raycast: true/false`.
                    visible = (v['rgba'][0] >= 0)
                    self.craycast.append(1 if v.get('raycast', visible) else 0)
                    self.crgba += v['rgba']
                    
        if 'feeds' in config:
            for v in config['feeds']:
                if   'jointpos' in v: tmp = [1]; target = 'jointpos'
                elif 'jointvel' in v: tmp = [2]; target = 'jointvel'
                elif 'jointact' in v: tmp = [3]; target = 'jointact' #joint activation torque
                elif 'framepos' in v: tmp = [4]; target = 'framepos'
                elif 'framequat' in v: tmp = [5]; target = 'framequat'
                elif 'framelinvel' in v: tmp = [6]; target = 'framelinvel'
                elif 'frameangvel' in v: tmp = [7]; target = 'frameangvel'
                elif 'framelinacc' in v: tmp = [8]; target = 'framelinacc'
                elif 'frameangacc' in v: tmp = [9]; target = 'frameangacc'
                elif 'velocimeter' in v: tmp = [10]; target = 'velocimeter'
                elif 'gyro' in v: tmp = [11]; target = 'gyro'
                elif 'accelerometer' in v: tmp = [12]; target = 'accelerometer'
                elif 'ftsensor' in v: tmp = [13]; target = 'ftsensor'
                elif 'fcontact' in v: tmp = [14]; target = 'fcontact'

                for f in v[target]:
                    if prefix == None: feedname = f
                    #else: feedname = prefix + '.' + f
                    else: feedname = prefix + f
                    if feedname in self.fdict: #<----------------------------------
                        tmp.append(self.fdict[feedname])
                self.feeds.append(tmp)

        self.X = get_spatial_transform(self.Ti)
        self.I6 = get_spatial_inertia(self.m, self.c, self.I)

        self._rebuild_cpair()

        if self.use_c: self._create_c_handle()

    def _rebuild_cpair(self):
        """Recompute self.cpair (collidable shape-shape pairs) and reset per-pair
        brush displacement state. Called from build() and from delete() — the latter
        because shape indices shift after array splicing, invalidating the old cpair."""
        nshape = len(self.ctype)
        self.cpair = np.zeros((nshape*nshape, 2), dtype=int)
        pair_idx = 0

        for i in range(nshape):
            for j in range(nshape):
                #exclude self & double check
                if i >= j: continue

                #exclude of both objects belong to the same link
                if self.cbody[i] == self.cbody[j]: continue

                #exclude if contact flag is minus
                if self.cparam[i][0] < 0 or self.cparam[j][0] < 0: continue

                #exclude if both contact flags are 0
                if self.cparam[i][0] == 0 and self.cparam[j][0] == 0: continue

                #otherwise set collidable
                self.cpair[pair_idx, 0] = i
                self.cpair[pair_idx, 1] = j
                pair_idx += 1

        self.cpair = self.cpair[:pair_idx]

    def delete(self, name):
        """Remove a previously add()-ed group by name. Arbitrary deletion order is
        supported as long as no surviving body still references a deleted body as
        its parent — cross-group parent links arise only from `base=` composition
        and not from the typical "robot + free objects on root" pattern. State of
        surviving bodies (q/qd values at their slots) is preserved via Env.delete()."""
        gi = next((i for i, g in enumerate(self.groups) if g['name'] == name), None)
        if gi is None:
            have = [g['name'] for g in self.groups]
            raise KeyError(f"no group named {name!r} (have: {have})")
        g = self.groups[gi]
        nb_lo,  nb_hi  = g['nb']
        nq_lo,  nq_hi  = g['nq']
        nsh_lo, nsh_hi = g['nshape']
        nf_lo,  nf_hi  = g['nframe']
        db = nb_hi  - nb_lo
        dq = nq_hi  - nq_lo
        ds = nsh_hi - nsh_lo
        df = nf_hi  - nf_lo

        # Dependency check: refuse if any surviving body has a parent in the
        # range we're about to remove. Always clean for the "items on conveyor"
        # / "free objects on root" usage pattern.
        for i, p in enumerate(self.parent):
            if nb_lo <= i < nb_hi: continue
            if p is not None and nb_lo <= p < nb_hi:
                raise RuntimeError(
                    f"cannot delete {name!r}: body {i} has parent inside it "
                    f"(use LIFO order when groups depend via base=)")

        # 1. Splice per-body arrays (plain lists)
        del self.jtype[nb_lo:nb_hi]
        del self.parent[nb_lo:nb_hi]
        del self.Ti[nb_lo:nb_hi]
        del self.m[nb_lo:nb_hi]
        del self.c[nb_lo:nb_hi]
        del self.I[nb_lo:nb_hi]

        # 2. Splice per-DoF arrays (numpy + active list)
        self.q0    = np.concatenate([self.q0[:nq_lo],    self.q0[nq_hi:]])
        self.qd0   = np.concatenate([self.qd0[:nq_lo],   self.qd0[nq_hi:]])
        self.ff    = np.concatenate([self.ff[:nq_lo],    self.ff[nq_hi:]])
        self.sk    = np.concatenate([self.sk[:nq_lo],    self.sk[nq_hi:]])
        self.floss = np.concatenate([self.floss[:nq_lo], self.floss[nq_hi:]])
        self.armature = np.concatenate([self.armature[:nq_lo], self.armature[nq_hi:]])
        self.jnt_lo = np.concatenate([self.jnt_lo[:nq_lo], self.jnt_lo[nq_hi:]])
        self.jnt_hi = np.concatenate([self.jnt_hi[:nq_lo], self.jnt_hi[nq_hi:]])
        self.active = self.active[:nq_lo] + self.active[nq_hi:]

        # 3. Splice per-shape arrays. crgba is flat (4 floats per shape: rgba).
        del self.ctype[nsh_lo:nsh_hi]
        del self.cbody[nsh_lo:nsh_hi]
        del self.cshape[nsh_lo:nsh_hi]
        del self.cparam[nsh_lo:nsh_hi]
        del self.ctran[nsh_lo:nsh_hi]
        del self.craycast[nsh_lo:nsh_hi]
        del self.crgba[nsh_lo*4:nsh_hi*4]

        # 4. Splice per-frame arrays + drop fdict keys
        del self.fbody[nf_lo:nf_hi]
        del self.ftran[nf_lo:nf_hi]
        del self.ftran_inv[nf_lo:nf_hi]
        for k in g['fdict_keys']:
            if k in self.fdict: del self.fdict[k]
        self.f_idx -= df

        # 5. Shift surviving body indices in parent/cbody/fbody/fixed.
        #    None / -1 (root sentinel) stay untouched because they're < nb_lo.
        self.parent = [p if (p is None or p < nb_lo) else p - db for p in self.parent]
        self.cbody  = [c if c < nb_lo else c - db for c in self.cbody]
        self.fbody  = [b if b < nb_lo else b - db for b in self.fbody]
        self.fixed = [(x - db) if x >= nb_hi else x
                      for x in self.fixed if not (nb_lo <= x < nb_hi)]

        # 6. Splice feeds (drop deleted group's own entries) and shift frame indices.
        feed_lo, feed_hi = g['nfeeds']
        new_feeds = []
        for fi, feed in enumerate(self.feeds):
            if feed_lo <= fi < feed_hi: continue   # group's own feed — drop
            shifted = [feed[0]]
            for k in range(1, len(feed)):
                idx = feed[k]
                if nf_lo <= idx < nf_hi:
                    raise RuntimeError(
                        f"cannot delete {name!r}: feed in surviving group "
                        f"references its frame {idx}")
                shifted.append(idx - df if idx >= nf_hi else idx)
            new_feeds.append(shifted)
        self.feeds = new_feeds

        # 6b. Drop the deleted group's sensors (name-based — no frame index to shift;
        #     each sensor's frame is already removed from fdict above, so any stale
        #     reference would no-op, but we splice for a clean registry).
        cam_lo, cam_hi = g['ncameras']
        self.cameras = self.cameras[:cam_lo] + self.cameras[cam_hi:]
        lid_lo, lid_hi = g['nlidars']
        self.lidars = self.lidars[:lid_lo] + self.lidars[lid_hi:]

        # 7. Shift fdict values (frame indices) past the deleted frame range
        for k, v in list(self.fdict.items()):
            if v >= nf_hi: self.fdict[k] = v - df

        # 8. Shift group metadata for groups after this one
        nfeeds_d = feed_hi - feed_lo
        ncam_d   = cam_hi - cam_lo
        nlid_d   = lid_hi - lid_lo
        nfixed_d = g['nfixed'][1] - g['nfixed'][0]
        for g2 in self.groups[gi+1:]:
            g2['nb']     = (g2['nb'][0]     - db, g2['nb'][1]     - db)
            g2['nq']     = (g2['nq'][0]     - dq, g2['nq'][1]     - dq)
            g2['nshape'] = (g2['nshape'][0] - ds, g2['nshape'][1] - ds)
            g2['nframe'] = (g2['nframe'][0] - df, g2['nframe'][1] - df)
            g2['nfeeds'] = (g2['nfeeds'][0] - nfeeds_d, g2['nfeeds'][1] - nfeeds_d)
            g2['ncameras'] = (g2['ncameras'][0] - ncam_d, g2['ncameras'][1] - ncam_d)
            g2['nlidars'] = (g2['nlidars'][0] - nlid_d, g2['nlidars'][1] - nlid_d)
            g2['nfixed'] = (g2['nfixed'][0] - nfixed_d, g2['nfixed'][1] - nfixed_d)
        self.groups.pop(gi)

        # 9. Rebuild derived data + C handle. Any warm-start λ carry (Env._ctx) is
        #    invalid because cpair size changes — Env resets it on delete.
        self.X  = get_spatial_transform(self.Ti)
        self.I6 = get_spatial_inertia(self.m, self.c, self.I)
        self._rebuild_cpair()
        if self.use_c: self._create_c_handle()

    def edit(self, index, m=None, c=None, I=None, Ti=None):
        # `index` may be an int (body index) or a str (body/frame name → resolved
        # via fdict + fbody). Passing a frame name resolves to the body that owns
        # the frame. 'root' is rejected (it's the world, body_idx=-1).
        if isinstance(index, str):
            if index not in self.fdict:
                raise KeyError(f"unknown body/frame name: {index!r}")
            body_idx = self.fbody[self.fdict[index]]
            if body_idx < 0:
                raise ValueError(f"name {index!r} resolves to world (not a body)")
            index = body_idx
        if m  is not None: self.m[index]  = float(m)
        if c  is not None: self.c[index]  = np.array(c,  dtype=np.float64)
        if I  is not None: self.I[index]  = np.array(I,  dtype=np.float64)
        if Ti is not None: self.Ti[index] = np.array(Ti, dtype=np.float64)

        self.X = get_spatial_transform(self.Ti)
        self.I6 = get_spatial_inertia(self.m, self.c, self.I)

        #topology unchanged — push X/I6/Ti in place so existing tact_get_* views survive
        if self.use_c and getattr(self, '_h', None):
            X  = np.ascontiguousarray(np.asarray(self.X),  dtype=np.float64)
            I6 = np.ascontiguousarray(np.asarray(self.I6), dtype=np.float64)
            Ti = np.ascontiguousarray(np.asarray(self.Ti), dtype=np.float64)
            clib.tact_edit_model(self._h, X.ctypes.data_as(_DBL), I6.ctypes.data_as(_DBL), Ti.ctypes.data_as(_DBL))

    #Phase 1+2+3: build a C-side tact_t handle. step()/gravity()/fk() all route through it.
    #On rebuild (add()/edit()) the previous handle is destroyed first — any view
    #obtained before is invalidated (see docs/design-c-state.md §3.5).
    def _create_c_handle(self):
        nb     = len(self.jtype)
        nshape = len(self.ctype)
        npair  = self.cpair.shape[0]

        if getattr(self, '_h', None):
            clib.tact_destroy(self._h)
            self._h = None

        #tact_create copies these into its arena, so the build arrays are only needed
        #for the duration of the call. We still keep refs on self until the next call
        #just to be safe against any pre-collection.
        self._build_X      = np.ascontiguousarray(np.asarray(self.X),  dtype=np.float64)
        self._build_I6     = np.ascontiguousarray(np.asarray(self.I6), dtype=np.float64)
        self._build_Ti     = np.ascontiguousarray(np.asarray(self.Ti), dtype=np.float64)
        self._build_ff     = np.ascontiguousarray(np.asarray(self.ff), dtype=np.float64)
        self._build_sk     = np.ascontiguousarray(np.asarray(self.sk), dtype=np.float64)
        self._build_floss  = np.ascontiguousarray(np.asarray(self.floss), dtype=np.float64)
        self._build_armature = np.ascontiguousarray(np.asarray(self.armature), dtype=np.float64)
        self._build_jnt_lo = np.ascontiguousarray(np.asarray(self.jnt_lo), dtype=np.float64)
        self._build_jnt_hi = np.ascontiguousarray(np.asarray(self.jnt_hi), dtype=np.float64)
        self._build_g      = np.ascontiguousarray(np.asarray(self.g),  dtype=np.float64)
        self._build_parent = np.array([p if p is not None else -1 for p in self.parent], dtype=np.int32)
        self._build_jtype  = np.array(self.jtype, dtype=np.int32)
        self._build_ctype    = np.array(self.ctype, dtype=np.int32) if nshape else np.zeros(0, dtype=np.int32)
        self._build_cbody    = np.array(self.cbody, dtype=np.int32) if nshape else np.zeros(0, dtype=np.int32)
        self._build_craycast = np.array(self.craycast, dtype=np.int32) if nshape else np.zeros(0, dtype=np.int32)
        self._build_cpair    = np.ascontiguousarray(self.cpair, dtype=np.int32) if npair else np.zeros((0, 2), dtype=np.int32)
        cshape_arr = np.zeros((nshape, 3), dtype=np.float64)
        for i in range(nshape):
            sh = np.asarray(self.cshape[i], dtype=np.float64).ravel()
            cshape_arr[i, :len(sh)] = sh
        self._build_cshape = cshape_arr
        self._build_ctran  = np.ascontiguousarray(np.asarray(self.ctran).reshape(nshape, 16) if nshape else np.zeros((0, 16)), dtype=np.float64)
        self._build_cparam = np.ascontiguousarray(np.asarray(self.cparam).reshape(nshape, 13) if nshape else np.zeros((0, 13)), dtype=np.float64)

        self._h = clib.tact_create(
            nb,
            self._build_parent.ctypes.data_as(_INT),
            self._build_jtype.ctypes.data_as(_INT),
            self._build_X.ctypes.data_as(_DBL),
            self._build_I6.ctypes.data_as(_DBL),
            self._build_Ti.ctypes.data_as(_DBL),
            self._build_ff.ctypes.data_as(_DBL),
            self._build_sk.ctypes.data_as(_DBL),
            self._build_floss.ctypes.data_as(_DBL),
            self._build_armature.ctypes.data_as(_DBL),
            self._build_jnt_lo.ctypes.data_as(_DBL),
            self._build_jnt_hi.ctypes.data_as(_DBL),
            self._build_g.ctypes.data_as(_DBL),
            ctypes.c_double(self.dt),
            ctypes.c_int(2),                    # integrator arg is vestigial (lcp uses its own semi-implicit Euler)
            nshape, npair,
            self._build_ctype.ctypes.data_as(_INT),
            self._build_cbody.ctypes.data_as(_INT),
            self._build_cshape.ctypes.data_as(_DBL),
            self._build_ctran.ctypes.data_as(_DBL),
            self._build_cparam.ctypes.data_as(_DBL),
            self._build_craycast.ctypes.data_as(_INT),
            self._build_cpair.ctypes.data_as(_INT),
            # global LCP solver knobs (from YAML sim:, else __init__ defaults)
            ctypes.c_double(self.erp), ctypes.c_double(self.slop), ctypes.c_double(self.cfm_scale),
            ctypes.c_double(self.v_rest_thresh), ctypes.c_int(self.iters), ctypes.c_double(self.tol),
        )

        #wrap arena's dynamic buffers as numpy views — re-acquired on every recreate.
        # Buffers are per-DoF (length nq) — equals nb when no free6 joints, larger
        # by 5 per free6 body otherwise. self.q is already nq-long after build.
        nq = len(self.q0)
        self._h_q_next  = np.ctypeslib.as_array(clib.tact_get_q_next(self._h),  shape=(nq,))
        self._h_qd_next = np.ctypeslib.as_array(clib.tact_get_qd_next(self._h), shape=(nq,))

        #---- Phase 2: marshal feedback descriptors → handle ----
        # output size per feed kind (matches feedback() in this file)
        _Y_PER = {1:1, 2:1, 3:1, 4:3, 5:4, 6:3, 7:3, 8:3, 9:3, 10:3, 11:3, 12:3, 13:6, 14:6}
        kinds, offsets, idx = [], [0], []
        y_size = 0
        for feed in self.feeds:
            kinds.append(int(feed[0]))
            for i in range(1, len(feed)): idx.append(int(feed[i]))
            offsets.append(len(idx))
            y_size += _Y_PER[feed[0]] * (len(feed) - 1)
        n_feeds  = len(kinds)
        n_frames = len(self.ftran)

        self._build_feed_kinds   = np.array(kinds   if kinds   else [0], dtype=np.int32)
        self._build_feed_offsets = np.array(offsets,                     dtype=np.int32)
        self._build_feed_idx     = np.array(idx     if idx     else [0], dtype=np.int32)
        self._build_fbody = np.array(self.fbody, dtype=np.int32) if n_frames else np.zeros(1, dtype=np.int32)
        if n_frames:
            self._build_ftran     = np.ascontiguousarray(np.stack(self.ftran),     dtype=np.float64)
            self._build_ftran_inv = np.ascontiguousarray(np.stack(self.ftran_inv), dtype=np.float64)
        else:
            self._build_ftran = self._build_ftran_inv = np.zeros((1, 4, 4), dtype=np.float64)

        clib.tact_set_feedback(
            self._h, n_feeds,
            self._build_feed_kinds.ctypes.data_as(_INT),
            self._build_feed_offsets.ctypes.data_as(_INT),
            self._build_feed_idx.ctypes.data_as(_INT),
            n_frames,
            self._build_fbody.ctypes.data_as(_INT),
            self._build_ftran.ctypes.data_as(_DBL),
            self._build_ftran_inv.ctypes.data_as(_DBL),
            y_size)
        self._y_size = y_size
        self._h_y = np.ctypeslib.as_array(clib.tact_get_y(self._h), shape=(max(y_size, 1),))

    def __del__(self):
        h = getattr(self, '_h', None)
        if h:
            try: clib.tact_destroy(h)
            except Exception: pass
            self._h = None

    def feedback(self, q, qd, tau, T, f, a, v, f_ext):
        y = []
        # Cases 1/2/3 (jointpos/jointvel/jointforce) are only meaningful for
        # 1-DoF joints; q uses q_base (position state), qd/tau use v_base
        # (velocity state). For 1-DoF joints q_base[i] == v_base[i].
        q_base, v_base, _, _, _, _ = _build_qidx(self.jtype)

        for feed in self.feeds:
            if feed[0] == 1:
                for i in range(1, len(feed)):
                    y.append(q[q_base[self.fbody[feed[i]]]])

            elif feed[0] == 2:
                for i in range(1, len(feed)):
                    y.append(qd[v_base[self.fbody[feed[i]]]])

            elif feed[0] == 3:
                for i in range(1, len(feed)):
                    y.append(tau[v_base[self.fbody[feed[i]]]])
                    
            elif feed[0] == 4: #framepos
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    Tb = T[body_idx] @ self.ftran[feed[i]]
                    y += Tb[:3, 3].tolist()
                
            elif feed[0] == 5: #framequat
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    # BUG FIX: previously returned body's quaternion, ignoring the
                    # frame's own rotation offset (ftran's R block). Mujoco's
                    # `<framequat objtype="site">` returns the SITE/FRAME world
                    # quaternion (= R_body_world @ R_frame_body). For frames with
                    # identity ftran rotation (most "site" cases) both are equal.
                    # OLD: Q = rotation_to_quat(T[body_idx][:3, :3])
                    Tw = T[body_idx] @ self.ftran[feed[i]]
                    Q = rotation_to_quat(Tw[:3, :3])
                    y += Q.tolist()

            elif feed[0] == 6:  #framelinvel
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    v0 = v[body_idx][3:6] #linear velocity at body origin
                    w = v[body_idx][0:3] #angluar velocity of the body
                    r = self.ftran[feed[i]][:3, 3]
                    v1 = v0 + np.cross(w, r)
                    R = T[body_idx][:3, :3]
                    v2 = R @ v1
                    y += v2.tolist()

            elif feed[0] == 7: #frameangvel
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    w0  = v[body_idx][0:3]
                    R = T[body_idx][:3, :3]
                    w1 = R @ w0
                    y += w1.tolist()
                    
            elif feed[0] == 8: #framelinacc
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    a0 = a[body_idx][3:6] #proper accel of body origin (body frame)
                    w = v[body_idx][0:3]
                    w_dot = a[body_idx][0:3]

                    r = self.ftran[feed[i]][:3, 3]
                    R = T[body_idx][:3, :3]

                    # BUG FIX: framelinacc = world-frame KINEMATIC linear acceleration
                    # at frame point (mujoco semantic). rne already returns
                    # _a[3:] = a^F + ω×v0 = proper accel of body origin. So:
                    #   proper(r)_body = a0 + ω̇×r + ω×(ω×r)
                    #   kinematic(r)_world = R @ proper(r)_body + g_world
                    # The old code double-counted the ω×v0 correction AND missed
                    # the +g term (commented out as `#+ self.g`).
                    # OLD: a1 = R @ (a0 + np.cross(w, v0) + np.cross(w_dot, r) + np.cross(w, np.cross(w, r))) #+ self.g
                    a1 = R @ (a0 + np.cross(w_dot, r) + np.cross(w, np.cross(w, r))) + self.g
                    y += a1.tolist()

            elif feed[0] == 9: #frameangacc:
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    w_dot0  = a[body_idx][0:3]
                    R = T[body_idx][:3, :3]
                    w_dot1 = R @ w_dot0
                    y += w_dot1.tolist()

            elif feed[0] == 10: #velocimeter
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    v0 = v[body_idx][3:6] #linear velocity at body origin
                    y += v0.tolist()

            elif feed[0] == 11: #gyro (angular velocimeter)
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    w = v[body_idx][0:3]
                    y += w.tolist()

            elif feed[0] == 12: #accelerometer
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    a0 = a[body_idx][3:6] #proper accel of body origin (body frame)
                    w = v[body_idx][0:3]
                    w_dot = a[body_idx][0:3]

                    r = self.ftran[feed[i]][:3, 3]

                    # BUG FIX: rne already returns _a[3:] = a^F + ω×v0 = proper
                    # acceleration of body origin (= what an accelerometer at the
                    # body origin reads, in body-frame coords). For an accelerometer
                    # at point r on the same rigid body:
                    #   proper(r)_body = proper(O)_body + ω̇×r + ω×(ω×r)
                    # The old code added an extra `+ ω×v0` term (double-counting
                    # the rne correction). At rest ω×v0=0 so bug hidden, but during
                    # dynamic motion (walking) the error is ~|ω||v0| (~2% of g).
                    # OLD: a1 = a0 + np.cross(w, v0) + np.cross(w_dot, r) + np.cross(w, np.cross(w, r))
                    a1 = a0 + np.cross(w_dot, r) + np.cross(w, np.cross(w, r))
                    y += a1.tolist()
                    
            elif feed[0] == 13: #ft-sensor
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    tmp = f[body_idx]
                    y += [tmp[3], tmp[4], tmp[5], tmp[0], tmp[1], tmp[2]]

            elif feed[0] == 14: #contact force (frame-transformed FT sensor)
                for i in range(1, len(feed)):
                    body_idx = self.fbody[feed[i]]
                    T1 = self.ftran_inv[feed[i]]
                    # Use ABA/RNE-propagated f (spatial wrench transmitted from
                    # parent through joint to body) — matches mujoco/Drake/Bullet
                    # constraint Lagrange-multiplier semantics. Includes inertial
                    # terms (I·a + crf·I·v) automatically, so the reading is
                    # correct for any mass distribution, not just massless leaf
                    # bodies. For mass = 0 with no velocity this reduces to
                    # f = -f_ext, so the previous f_ext-based formula gave the
                    # same numerical output in that special case.
                    #
                    # OLDER (contact force only, exact only for massless body):
                    #  f1 = -T1[:3, :3] @ f_ext[body_idx][3:6]
                    #  m1 = -T1[:3, :3] @ (f_ext[body_idx][0:3] + np.cross(T1[:3, 3], f_ext[body_idx][3:6]))
                    #
                    # OLD (f-based but wrong moment-transfer when frame rotates):
                    #  f1 = T1[:3, :3] @ f[body_idx][3:6]
                    #  m1 = T1[:3, :3] @ (f[body_idx][0:3] + np.cross(T1[:3, 3], f[body_idx][3:6]))
                    # The cross above mixed frames: T1[:3, 3] is in FRAME coords
                    # but f[body_idx][3:6] is in BODY coords, so the moment-arm
                    # shift was wrong when ftran has non-identity rotation.
                    #
                    # BUG FIX: do the moment-arm transfer in a single coord frame.
                    # Two equivalent forms:
                    #   (a) shift in body frame, then rotate:
                    #       m1 = T1[:3,:3] @ (f[0:3] - cross(ftran[:3,3], f[3:6]))
                    #   (b) rotate first, then shift in frame frame:
                    #       m1 = T1[:3,:3] @ f[0:3] + cross(T1[:3,3], f1)
                    # We use (b) — fewer transforms, and f1 is already computed.
                    f1 = T1[:3, :3] @ f[body_idx][3:6]
                    m1 = T1[:3, :3] @ f[body_idx][0:3] + np.cross(T1[:3, 3], f1)
                    y = y + f1.tolist() + m1.tolist()
                    
        return np.array(y)

    def zero_state(self):
        """A cold-start SolverState sized for the current topology (all λ = 0).
        Use as the initial `ctx` for a pure Model.step rollout, or after a
        topology change (add/delete) since the λ layout is sized by
        (n_pair, nq): 6*MAX_PTS_PER_PAIR*max(n_pair,1) contact slots + 2*nq."""
        npair, nq = len(self.cpair), len(self.floss)
        return SolverState(lam=np.zeros(6 * MAX_PTS_PER_PAIR * max(npair, 1) + 2 * nq),
                           nq=nq)

    def step(self, q, qd, tau=None, q_ref=None, qd_ref=None, kp=None, kd=None, ctx=None):
        # Referentially transparent: same (q, qd, tau, q_ref, qd_ref, kp, kd, ctx) →
        # same (q_next, qd_next, y, ctx_next). `ctx` (SolverState) carries LCP warm-start λ;
        # ctx=None → cold start (zero λ). ctx is not mutated — ctx_next is fresh.
        # All three input channels are equal-priority and independently optional.
        # tau=None → treated as zero feedforward (passive step under gravity/contact).
        # q_ref/qd_ref activate internal joint-space PD on the LCP path; when both None,
        # behavior is bit-identical to pre-PD step.
        # kp/kd: per-DoF implicit joint-PD gains for THIS step (length nq, like tau).
        # Gains are control-policy inputs, not plant parameters (YAML `k:` removed
        # 2026-06-07) — controllers switching modes pass per-mode gains here.
        # A reference without its gain is an error (the old silent zero-gain
        # fallback masked exactly the bugs this migration removes); gains without
        # references are inert by design (controller keeps kp/kd set while a mode
        # switch turns q_ref off). q_ref+kp without kd = P-only, legitimate.
        # ff damping and sk spring are applied implicitly inside aba_featherstone.
        # tau/q_ref/qd_ref/kp/kd are per-DoF (length nq), not per-body (length nb) —
        # these differ only when free6 (jtype=3) joints are present.
        if tau is None: tau = np.zeros(len(q))
        if q_ref  is not None and kp is None:
            raise ValueError("q_ref requires kp — gains are per-step inputs (YAML k: was removed)")
        if qd_ref is not None and kd is None:
            raise ValueError("qd_ref requires kd — gains are per-step inputs (YAML k: was removed)")
        Kp = None if kp is None else np.ascontiguousarray(kp, dtype=np.float64)
        Kd = None if kd is None else np.ascontiguousarray(kd, dtype=np.float64)

        if self.use_c and self.solver == 'lcp':
            #single ctypes round-trip into tact_step_lcp:
            #   _fk → aba(no-contact) → crb → contact_lcp → semi-implicit → feedback
            #C side reads raw tau; ff/sk damping + implicit joint-PD are applied internally.
            #Gated on solver=='lcp': the C handle ONLY implements lcp, so any other
            #solver must not silently run here (an unknown one falls to the else → raise).
            q_in   = np.ascontiguousarray(q,   dtype=np.float64)
            qd_in  = np.ascontiguousarray(qd,  dtype=np.float64)
            tau_in = np.ascontiguousarray(tau, dtype=np.float64)

            # implicit joint-PD pointers — NULL when activation missing (no q_ref/qd_ref)
            # or capability cleared (kp/kd resolved to None). Default arrays are
            # zero-initialized from YAML, so PD is inert until YAML `k:`, a controller
            # assignment, or the per-step kp/kd kwargs provide non-zero gains.
            Kp_ptr  = Kp.ctypes.data_as(_DBL)  if (Kp is not None and q_ref is not None) else None
            Kd_ptr  = Kd.ctypes.data_as(_DBL)  if (Kd is not None and (q_ref is not None or qd_ref is not None)) else None
            qr_ptr  = q_ref.ctypes.data_as(_DBL)      if q_ref       is not None else None
            qdr_ptr = qd_ref.ctypes.data_as(_DBL)     if qd_ref      is not None else None

            # warm-start carry: ctx.lam in, fresh lam_out (= ctx_next) out. ctx=None →
            # cold (zeros). Distinct in/out buffers keep ctx immutable (C seeds out
            # from in). ONE vector for all PGS row types — [contact | fric | limit],
            # the SolverState layout; tact_step_lcp slices it by the same arithmetic.
            nq_dof  = len(self.floss)
            lam_len = 6 * MAX_PTS_PER_PAIR * max(len(self.cpair), 1) + 2 * nq_dof
            lam_in  = np.zeros(lam_len) if ctx is None else np.ascontiguousarray(ctx.lam, dtype=np.float64)
            lam_out = np.zeros(lam_len, dtype=np.float64)
            clib.tact_step_lcp(self._h, q_in.ctypes.data_as(_DBL), qd_in.ctypes.data_as(_DBL), tau_in.ctypes.data_as(_DBL), Kp_ptr, Kd_ptr, qr_ptr, qdr_ptr,
                               lam_in.ctypes.data_as(_DBL), lam_out.ctypes.data_as(_DBL))

            #copy outputs out of arena (next step would overwrite views)
            q_next  = self._h_q_next.copy()
            qd_next = self._h_qd_next.copy()
            y       = self._h_y[:self._y_size].copy()
            ctx_next = SolverState(lam=lam_out, nq=nq_dof)

        elif not self.use_c and self.solver == 'lcp':
            #LCP path: ABA-with-joint-PD(f_ext=0) → qd_free, CRB → M, contact_lcp solves for
            #impulse λ → semi-implicit Euler. A second ABA call with the contact wrench feeds
            #the feedback layer's f/a/v so accelerometer-like outputs reflect post-contact
            #dynamics. Without that second call, IMU feed shows a bias of order g·support_frac
            #whenever joint torques cancel out contact (e.g. quadruped stance).
            #Joint damping `ff`, spring `sk`, and joint-space implicit PD (Kp_j/Kd_j/q_ref/qd_ref)
            #are all folded into ABA's articulated inertia.
            T = _fk(self.Ti, self.parent, self.jtype, q)
            f_ext_zero = np.zeros((len(self.X), 6))
            qdd_free, f, a, v = aba_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext_zero, self.g, full=True, ff=self.ff, sk=self.sk, armature=self.armature, dt=self.dt, Kp_j=Kp, Kd_j=Kd, q_ref=q_ref, qd_ref=qd_ref)
            qd_free = qd + qdd_free * self.dt
            M = crb_featherstone(self.X, self.I6, self.parent, self.jtype, q)
            M = M + np.diag(self.armature)   # armature (rotor inertia) on the M diagonal — matches the C path + ABA predictor above
            # slice the unified ctx.lam into contact_lcp's per-type warm-starts
            # (views — contact_lcp reads them without mutating, so ctx stays
            # immutable), then pack its per-type outputs back into one vector.
            lam_contact_prev = None if ctx is None else ctx.lam_contact
            lam_fric_prev    = None if ctx is None else ctx.lam_fric
            lam_limit_prev   = None if ctx is None else ctx.lam_limit
            dqd, lam, lcp_info, f_ext = contact_lcp(T, self.parent, self.jtype, self.cpair, self.ctype, self.cbody, self.ctran, self.cshape, self.cparam, qd_free, M, self.dt,
                                                    erp=self.erp, slop=self.slop, cfm_scale=self.cfm_scale, v_rest_thresh=self.v_rest_thresh, iters=self.iters, tol=self.tol,
                                                    lam_contact_prev=lam_contact_prev, floss=self.floss, lam_fric_prev=lam_fric_prev,
                                                    q=q, jnt_lo=self.jnt_lo, jnt_hi=self.jnt_hi, lam_limit_prev=lam_limit_prev)
            cblk = lcp_info['lam_contact_full']
            clen = 6 * MAX_PTS_PER_PAIR * max(len(self.cpair), 1)
            if len(cblk) != clen:   # npair==0: rbd sizes by npair (0), layout keeps the 1-pair slot
                cblk = np.zeros(clen)
            ctx_next = SolverState(lam=np.concatenate([cblk, lcp_info['lam_fric_full'], lcp_info['lam_limit_full']]),
                                   nq=len(self.floss))   #carry for next step's warm-start
            qd_next = qd_free + dqd
            q_base, _, _, _, _, _ = _build_qidx(self.jtype)
            q_next  = _q_step(q, qd_next, self.dt, self.jtype, q_base)
            qdd = (qd_next - qd) / self.dt
            # Kinematic forward pass (RNE): given realized qdd, propagate spatial accels
            # so feedback (a, v, f) reflects post-contact body dynamics.
            _, f, a, v = rne_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, qdd, f_ext, self.g, full=True)
            y = self.feedback(q, qd, tau, T, f, a, v, f_ext)

        elif self.solver == 'minimal':
            # Test-only solver (rbd.contact_ground_sphere): explicit spring-damper
            # ground (z=0) contact for SPHERE shapes, fed as f_ext into ABA forward
            # dynamics + semi-implicit Euler. Runs on the Python path regardless of
            # use_c; the C handle (if created) stays valid for fk/jacob/raycast.
            # Not for production — spheres only, no Coulomb cone, needs a small dt.
            T = _fk(self.Ti, self.parent, self.jtype, q)
            f_ext = contact_ground_sphere(T, self.parent, self.jtype, self.ctype, self.cbody, self.ctran, self.cshape, self.cparam, qd)
            qdd, f, a, v = aba_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext, self.g, full=True, ff=self.ff, sk=self.sk, armature=self.armature, dt=self.dt, Kp_j=Kp, Kd_j=Kd, q_ref=q_ref, qd_ref=qd_ref)
            qd_next = qd + qdd * self.dt
            q_base, _, _, _, _, _ = _build_qidx(self.jtype)
            q_next  = _q_step(q, qd_next, self.dt, self.jtype, q_base)
            y = self.feedback(q, qd, tau, T, f, a, v, f_ext)
            ctx_next = ctx   # minimal solver has no LCP warm-start state — passthrough

        else: raise ValueError(f'unknown solver: {self.solver}')

        return q_next, qd_next, y, ctx_next

    def fk(self, frames, q, eulerseq='xyz'):
        if self.use_c:
            #Phase 4: single C call — frame loop, Te composition, format dispatch all in C
            keys     = list(frames.keys())
            idx_arr  = np.array([self.fdict[f] for f in keys], dtype=np.int32)
            mode_arr = np.array([0 if frames[f]=='3d' else 1 for f in keys], dtype=np.int32)
            out_size = sum(3 if frames[f]=='3d' else 6 for f in keys)
            out      = np.empty(out_size, dtype=np.float64)
            q_in     = np.ascontiguousarray(q, dtype=np.float64)
            clib.tact_fk_query(self._h, q_in.ctypes.data_as(_DBL), len(keys), idx_arr.ctypes.data_as(_INT), mode_arr.ctypes.data_as(_INT),  eulerseq.encode('ascii'), out.ctypes.data_as(_DBL))
            return out

        T = _fk(self.Ti, self.parent, self.jtype, q)
        out = []
        for f in frames.keys():
            frame_idx = self.fdict[f]
            body_idx = self.fbody[frame_idx]
            if body_idx < 0: Te = self.ftran[frame_idx]
            else: Te = T[body_idx] @ self.ftran[frame_idx]
            if   frames[f] == '3d': out = np.concatenate((out, Te[:3, 3]))
            elif frames[f] == '6d': out = np.concatenate((out, homogeneous_to_xyzeuler(Te, eulerseq)))
        return out

    def fkh(self, frames, q, rotation_only=False):
        T = _fk(self.Ti, self.parent, self.jtype, q)
        out = []
        
        for i in range(len(frames)):
        #for f in frames.keys():
            frame_idx = self.fdict[frames[i]]
            #frame_idx = self.fdict[f]
            body_idx = self.fbody[frame_idx]

            #Te = T[body_idx] @ self.ftran[frame_idx]
            if body_idx < 0: Te = self.ftran[frame_idx]
            else: Te = T[body_idx] @ self.ftran[frame_idx]

            if rotation_only: out.append(Te[:3, :3])
            else: out.append(Te)
        return out

    #Jacobian pseudo-inverse based IK algorithm
    def ik(self, frames, q, x_d, eulerseq='xyz', advance=0.5, tolerance=0.001):
        q_now = q.copy()
        for cnt in range(1000):
            e_x = self.error(frames, q_now, x_d, eulerseq)
            if np.linalg.norm(e_x) < tolerance: return q_now
            J = self.jacob(frames, q_now)               #was: self.jacob(frames, q) — frozen-J bug, only safe for tiny ‖q_now-q‖
            q_now += advance * np.linalg.pinv(J) @ e_x
            #q_now += advance * np.linalg.solve(J, e_x)
        raise RuntimeError('ik did not converge: |e|=%g, x_d=%s' %(np.linalg.norm(e_x), x_d))

    #Damped least squares based IK algorithm
    def ik2(self, frames, q, x_d, eulerseq='xyz', advance=0.5, tolerance=0.001, damping=0.01, max_iter=1000):
        if self.use_c:
            #single C call — full Newton loop (error + jacob + JJᵀ+λ²I + Cholesky-solve + Jᵀ update) in C.
            keys      = list(frames.keys())
            idx_arr   = np.array([self.fdict[f] for f in keys], dtype=np.int32)
            mode_arr  = np.array([0 if frames[f]=='3d' else 1 for f in keys], dtype=np.int32)
            q_in      = np.ascontiguousarray(q, dtype=np.float64)
            x_in      = np.ascontiguousarray(x_d, dtype=np.float64)
            q_out     = np.empty(len(q_in), dtype=np.float64)   # nq (per-DoF)
            iters = clib.tact_ik2_query(self._h, q_in.ctypes.data_as(_DBL), x_in.ctypes.data_as(_DBL), len(keys), idx_arr.ctypes.data_as(_INT), mode_arr.ctypes.data_as(_INT), eulerseq.encode('ascii'), advance, tolerance, damping, max_iter, q_out.ctypes.data_as(_DBL))
            if iters < 0: raise RuntimeError('ik2 did not converge in %d iters, x_d=%s' %(max_iter, x_d))
            return q_out

        q_now = q.copy()
        for cnt in range(max_iter):
            e_x = self.error(frames, q_now, x_d, eulerseq)
            if np.linalg.norm(e_x) < tolerance: return q_now
            J = self.jacob(frames, q_now)               #was: self.jacob(frames, q) — same frozen-J bug as ik()
            A = J @ J.T + damping*damping*np.eye(J.shape[0])
            q_now += advance * J.T @ np.linalg.solve(A, e_x)
        raise RuntimeError('ik2 did not converge: |e|=%g, x_d=%s' %(np.linalg.norm(e_x), x_d))
    
    def jacob(self, frames, q):
        # J columns are v-indexed (J maps qd → spatial vel). For axis-angle
        # nv == nq, so we can size from len(q); see Model.jacob for quat note.
        _, _, _, _, _, nv = _build_qidx(self.jtype)

        if self.use_c:
            #Phase 4: single C call — fk + per-frame jacob_whitney + row stacking all in C
            keys     = list(frames.keys())
            idx_arr  = np.array([self.fdict[f] for f in keys], dtype=np.int32)
            mode_arr = np.array([0 if frames[f]=='3d' else 1 for f in keys], dtype=np.int32)
            total_rows = sum(3 if frames[f]=='3d' else 6 for f in keys)
            J_out    = np.empty((total_rows, nv), dtype=np.float64)
            q_in     = np.ascontiguousarray(q, dtype=np.float64)
            clib.tact_jacob_query(self._h, q_in.ctypes.data_as(_DBL), len(keys), idx_arr.ctypes.data_as(_INT), mode_arr.ctypes.data_as(_INT), J_out.ctypes.data_as(_DBL))
            return J_out

        T = _fk(self.Ti, self.parent, self.jtype, q)
        J_out = []
        for f in frames.keys():
            frame_idx = self.fdict[f]
            body_idx = self.fbody[frame_idx]
            Te = T[body_idx] @ self.ftran[frame_idx]
            J = jacob_whitney(T, Te, self.parent, self.jtype, body_idx)
            if frames[f] == '3d': J = J[:3, :]
            elif frames[f] == '6d': pass
            if len(J_out) == 0: J_out = J
            else: J_out = np.vstack((J_out, J))
        return J_out

    def error(self, frames, q, x_d, eulerseq='xyz'):
        if self.use_c:
            #Phase 4: single C call — fk + per-frame Te + 3d/6d error formula all in C
            keys     = list(frames.keys())
            idx_arr  = np.array([self.fdict[f] for f in keys], dtype=np.int32)
            mode_arr = np.array([0 if frames[f]=='3d' else 1 for f in keys], dtype=np.int32)
            out_size = sum(3 if frames[f]=='3d' else 6 for f in keys)
            out      = np.empty(out_size, dtype=np.float64)
            q_in     = np.ascontiguousarray(q,   dtype=np.float64)
            xd_in    = np.ascontiguousarray(x_d, dtype=np.float64)
            clib.tact_error_query(self._h, q_in.ctypes.data_as(_DBL), xd_in.ctypes.data_as(_DBL), len(keys), idx_arr.ctypes.data_as(_INT), mode_arr.ctypes.data_as(_INT), eulerseq.encode('ascii'), out.ctypes.data_as(_DBL))
            return out

        T = _fk(self.Ti, self.parent, self.jtype, q)
        index, out = 0, []
        for f in frames.keys():
            frame_idx = self.fdict[f]
            body_idx = self.fbody[frame_idx]
            Te = T[body_idx] @ self.ftran[frame_idx]
            if frames[f] == '3d':
                e = x_d[index:index+3] - Te[:3, 3]
                out = np.concatenate((out, e))
                index += 3
            elif frames[f] == '6d':
                Td = xyzeuler_to_homogeneous(x_d[index:index+6], eulerseq)
                e = homogeneous_error(Td, Te)
                out = np.concatenate((out, e))
                index += 6
        return out

    def gravity(self, q, g=None):
        if g is None: g = self.g

        if self.use_c:
            nq   = len(q)              # per-DoF, not body count
            q_in = np.ascontiguousarray(q, dtype=np.float64)
            b    = np.empty(nq, dtype=np.float64)
            #pass NULL g_override → C uses h->g; only wrap when caller overrode self.g
            if g is self.g: g_ptr = None
            else:           g_ptr = np.ascontiguousarray(g, dtype=np.float64).ctypes.data_as(_DBL)
            clib.tact_gravity_query(self._h, q_in.ctypes.data_as(_DBL), g_ptr, b.ctypes.data_as(_DBL))
        else: b = rne_featherstone(self.X, self.I6, self.parent, self.jtype, q, np.zeros(len(q)), np.zeros(len(q)), None, g)
        return b

    def gravity2(self, q, g=None):
        if g is None: g = self.g
        T = _fk(self.Ti, self.parent, self.jtype, q)
        u = gravity_lagrange(T, self.m, self.c, self.parent, self.jtype, g)
        return -u

    def inertia(self, q):
        if self.use_c:
            nq   = len(q)
            q_in = np.ascontiguousarray(q, dtype=np.float64)
            H    = np.empty((nq, nq), dtype=np.float64)
            clib.tact_inertia_query(self._h, q_in.ctypes.data_as(_DBL), H.ctypes.data_as(_DBL))
        else: H = crb_featherstone(self.X, self.I6, self.parent, self.jtype, q)
        return H

    def inertia2(self, q):
        T = _fk(self.Ti, self.parent, self.jtype, q)
        M = inertia_lagrange(T, self.m, self.c, self.I, self.parent, self.jtype)
        return M

    def bias(self, q, qd, f_ext=None):
        if self.use_c:
            nq    = len(q)
            q_in  = np.ascontiguousarray(q,  dtype=np.float64)
            qd_in = np.ascontiguousarray(qd, dtype=np.float64)
            b     = np.empty(nq, dtype=np.float64)
            if f_ext is not None:
                fe_in  = np.ascontiguousarray(np.asarray(f_ext).reshape(-1, 6), dtype=np.float64)
                fe_ptr = fe_in.ctypes.data_as(_DBL)
            else: fe_ptr = None
            clib.tact_bias_query(self._h, q_in.ctypes.data_as(_DBL), qd_in.ctypes.data_as(_DBL), fe_ptr, b.ctypes.data_as(_DBL))
        else:
            qdd = np.zeros(len(q))
            b = rne_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, qdd, f_ext, self.g)
        return b

    def bias2(self, q, qd, f_ext=None):
        qdd = np.zeros(len(q))
        b = rne_lwp(self.Ti, self.m, self.c, self.I, self.parent, self.jtype, q, qd, qdd, f_ext, self.g)
        return b

    def com(self, q):
        if self.use_c:
            q_in = np.ascontiguousarray(q, dtype=np.float64)
            if not hasattr(self, '_m_arr_c'):
                self._m_arr_c = np.ascontiguousarray(self.m, dtype=np.float64)
                self._c_arr_c = np.ascontiguousarray(np.asarray(self.c).reshape(-1), dtype=np.float64)
            r = np.empty(3, dtype=np.float64)
            clib.tact_com_query(self._h, q_in.ctypes.data_as(_DBL),
                                self._m_arr_c.ctypes.data_as(_DBL),
                                self._c_arr_c.ctypes.data_as(_DBL),
                                r.ctypes.data_as(_DBL))
            return r
        T = _fk(self.Ti, self.parent, self.jtype, q)
        return com_lagrange(T, self.m, self.c)

    def com_jacob(self, q):
        if self.use_c:
            nq   = len(q)
            q_in = np.ascontiguousarray(q, dtype=np.float64)
            # m, c arrays: prepared once on first call, cached. m is per-body (nb,);
            # c is row-major (nb, 3) flattened to (3*nb,). Both are written by
            # Model.add at build time and edited only by Model.edit, so caching is safe.
            if not hasattr(self, '_m_arr_c'):
                self._m_arr_c = np.ascontiguousarray(self.m, dtype=np.float64)
                self._c_arr_c = np.ascontiguousarray(np.asarray(self.c).reshape(-1), dtype=np.float64)
            J = np.empty((3, nq), dtype=np.float64)
            clib.tact_com_jacob_query(self._h, q_in.ctypes.data_as(_DBL),
                                      self._m_arr_c.ctypes.data_as(_DBL),
                                      self._c_arr_c.ctypes.data_as(_DBL),
                                      J.ctypes.data_as(_DBL))
            return J
        T = _fk(self.Ti, self.parent, self.jtype, q)
        return com_jacob_lagrange(T, self.m, self.c, self.parent, self.jtype)

    def com_inertia(self, q):
        T = _fk(self.Ti, self.parent, self.jtype, q)
        return com_inertia(T, self.m, self.c, self.I)

    def jacob_dot_qd(self, frames, q, qd, dt=1e-4):
        """Time-derivative of jacob times qd: (dJ/dt) · qd. Finite-diff over dt.
        Caveat: for jtype=3 (free), q+dt·qd treats axis-angle q[3:6] as a vector
        update — fine at dt~1e-4 since the SO(3) drift is O(dt²)."""
        J1 = self.jacob(frames, q)
        J2 = self.jacob(frames, q + dt * qd)
        return ((J2 - J1) / dt) @ qd

    def pack_q_fb(self, q_joint, R, p):
        """(q_joint, R, p) → q_fb. Layout: [p(3), axis-angle(3), q_joint(n_act)].
        Requires this Model to be floating-base (jtype[0]==3)."""
        assert self.jtype[0] == 3, 'pack_q_fb requires floating-base Model (jtype[0]==3)'
        return np.concatenate([p, logmap_so3(R), q_joint])

    def pack_qd_fb(self, qd_joint, R, w_body, v_world):
        """(qd_joint, R, w_body, v_world) → qd_fb. Layout: [v_body(3), w_body(3), qd_joint(n_act)].
        v_body = Rᵀ · v_world (per jtype=3 jacob_whitney convention).
        Requires this Model to be floating-base (jtype[0]==3)."""
        assert self.jtype[0] == 3, 'pack_qd_fb requires floating-base Model (jtype[0]==3)'
        return np.concatenate([R.T @ v_world, w_body, qd_joint])



class Env:
    backend = 'tact'

    def __init__(self, src, prefix=None, base='root', offset=[0, 0, 0, 0, 0, 0], q0=None, fixed_base=False, render=False, redraw=20, name=None): #, index=0):
        self.src = src
        self.m = Model(src, prefix, base, offset, q0, fixed_base, name=name)
        self.dof = sum(self.m.active)
        self.q = self.m.q0.copy()
        self.qd = self.m.qd0.copy()
        self._ctx = None    # LCP warm-start carry (SolverState); None = cold next step

        self.cnt = 0
        self.render = render
        self.redraw = redraw

        max_width = 1024
        max_height = 768
        self._imgbuf = (ctypes.c_ubyte*max_width*max_height*4)()
        
    def add(self, src, prefix=None, base='root', offset=[0, 0, 0, 0, 0, 0], q0=None, fixed_base=False, name=None):
        # Extend state in place so existing bodies keep their current q/qd values
        # (was: reset to q0 on every add — broke mid-sim composition like conveyor scenes).
        nq_old = len(self.q)
        self.m.add(src, prefix, base, offset, q0, fixed_base, name=name)
        nq_new = len(self.m.q0)
        if nq_new > nq_old:
            self.q  = np.concatenate([self.q,  self.m.q0[nq_old:]])
            self.qd = np.concatenate([self.qd, self.m.qd0[nq_old:]])
        self.dof = sum(self.m.active)
        self._ctx = None    # cpair size changed → warm-start carry invalid (cold restart)

    def delete(self, name):
        """Remove a previously add()-ed group and preserve current state of the
        surviving bodies. q/qd values at the deleted DoF slots are dropped; the
        rest stays as-is, so a robot mid-motion keeps moving when a free object
        is removed from the scene."""
        g = next((gx for gx in self.m.groups if gx['name'] == name), None)
        if g is None:
            raise KeyError(f"no group named {name!r} (have: {[gx['name'] for gx in self.m.groups]})")
        nq_lo, nq_hi = g['nq']
        self.m.delete(name)
        self.q  = np.concatenate([self.q[:nq_lo],  self.q[nq_hi:]])
        self.qd = np.concatenate([self.qd[:nq_lo], self.qd[nq_hi:]])
        self.dof = sum(self.m.active)
        self._ctx = None    # cpair size changed → warm-start carry invalid (cold restart)

    @property
    def groups(self):
        """List of currently-active group names, in insertion order."""
        return [g['name'] for g in self.m.groups]

    @property
    def cameras(self):
        """Camera publish specs from the YAML `cameras:` block, in declaration
        order. Each is a dict {name, type, res, fps}; `name` is the
        registered frame name (the camera_frames render key) and the ZMQ endpoint.
        `start` iterates this to set up per-camera PUB sockets."""
        return self.m.cameras

    @property
    def lidars(self):
        """LiDAR publish specs from the YAML `lidars:` block, in declaration order.
        Each is a dict {name, type, res, dth, fps, ...}; `name` is the registered
        frame name (the lidar_frames raymap/raycloud key) and the ZMQ endpoint.
        `start` iterates this to set up per-lidar PUB sockets, mirroring cameras."""
        return self.m.lidars

    def edit(self, index, **kw):
        # Thin passthrough to Model.edit — body-property editor (m, c, I, Ti).
        self.m.edit(index, **kw)

    @property
    def has_pd(self):
        # True iff this env handles joint-space PD internally (controller can emit
        # q_ref/qd_ref instead of computing tau). PD is folded into ABA on the LCP
        # path. (The only solver is 'lcp'; kept as a property for the CEnv override.)
        return self.m.solver == 'lcp'

    @property
    def dt(self):
        # Physics integration step (sec). Lives on Model; exposed here for the
        # benefit of runners that derive control-loop rate (e.g. start computes
        # bt = (1/dt)/frameskip). CEnv has no Python-side dt — its caller is
        # responsible for handling that case.
        return self.m.dt
        
    def step(self, tau=None, q_ref=None, qd_ref=None, kp=None, kd=None):
        # All input channels are equal-priority and independently optional.
        # tau=None → zero feedforward (passive step); q_ref/qd_ref=None → backend's
        # internal PD inactive (caller is responsible for torque via tau).
        # kp/kd: per-step implicit joint-PD gains, ACTIVE-only (length dof, same
        # convention as tau/q_ref) — controllers switching control modes pass
        # per-mode gains here. This is the ONLY gain channel (YAML `k:` removed
        # 2026-06-07); a reference without its gain raises in Model.step.
        # Internal arrays are per-DoF (length nq), iterating self.m.active which is
        # the per-DoF active mask. For non-free6 models nq == nb so behavior is
        # unchanged; for free6 models the 6 DoFs per free6 body are all active=0.
        nq = len(self.q)
        tau_full = np.zeros(nq)
        qr_full  = None if q_ref  is None else np.zeros(nq)
        qdr_full = None if qd_ref is None else np.zeros(nq)
        kp_full  = None if kp     is None else np.zeros(nq)
        kd_full  = None if kd     is None else np.zeros(nq)
        idx = 0
        for k in range(nq):
            if self.m.active[k] > 0:
                if tau    is not None: tau_full[k] = tau[idx]
                if qr_full  is not None: qr_full[k]  = q_ref[idx]
                if qdr_full is not None: qdr_full[k] = qd_ref[idx]
                if kp_full  is not None: kp_full[k]  = kp[idx]
                if kd_full  is not None: kd_full[k]  = kd[idx]
                idx += 1

        self.q, self.qd, y, self._ctx = self.m.step(self.q, self.qd, tau=tau_full, q_ref=qr_full, qd_ref=qdr_full, kp=kp_full, kd=kd_full, ctx=self._ctx)
        
        if self.render and self.cnt % self.redraw == 0:
            ret = self._win_render()
            if ret < 0: print('ESC pressed. exit...'); sys.exit()

        self.cnt += 1
        return y

    def finish(self):
        pass

    def reset(self):
        """Reset state to initial pose and return the initial observation y. Gym-style
        bootstrap — caller can do `y = env.reset()` then enter the update→step loop.
        feedback() is called directly on the reset state (q0/qd0, zero f/a/v/f_ext)
        rather than via step(zeros), so the integrator is not advanced by one dt."""
        self.m.reset()
        self.q = self.m.q0
        self.qd = self.m.qd0
        self._ctx = None    # cold restart on reset
        self.cnt = 0

        nj = len(self.m.jtype)
        T = _fk(self.m.Ti, self.m.parent, self.m.jtype, self.q)
        tau = np.zeros(nj)
        z6 = np.zeros((nj, 6))   # zero spatial f/a/v/f_ext, one row per body
        y = self.m.feedback(self.q, self.qd, tau, T, z6, z6, z6, z6)

        if self.render:
            ret = self._win_render()
            if ret < 0: print('ESC pressed. exit...'); sys.exit()
        return y

    # NOTE: is_locked()/unlock() (free-joint locking mechanism) were removed
    # 2026-06-06 with the YAML `lock:` key — see the note above _register_sensors
    # in this file. Last users (mk2/mk3) moved to fgx; they return lock-free.

    # NOTE: get_z(x, y) (absolute world-z terrain query) was removed 2026-06-06
    # as part of sim-trick reduction: no real robot can observe absolute terrain
    # height in a global frame. height_scan below — base-relative, the contract a
    # real elevation map provides — is the only terrain query. Foothold code
    # anchors to a stance foot's FK z and adds relative scan deltas instead
    # (see StepGenerator2/4 in control.py).

    def height_scan(self, base_xy, yaw, offsets, z_top=100.0, default=0.0):
        """Ground-truth terrain height scan — the sim-only twin of
        tact.MiniElevationMap.sample(), with the SAME contract so the two are
        drop-in providers for one consumer: `offsets` is a (G, 2) grid in the
        gravity-aligned base-yaw frame; returns (G,) terrain-top heights
        relative to the terrain under base_xy; points with no terrain hit
        return `default`; per-point validity / base_valid / ref land in
        self.last (dict, same keys as MiniElevationMap.sample).

        One vertical raycast per point from z_top down, so it reads the true
        scene with no sensor, map, latency, or drift. Use it where a terrain
        scan must live in a single thread (tact-native RL, sim2sim, quick
        prototyping) or as the GT baseline; it has no real-hardware
        counterpart, so anything trained on it must be re-validated against
        the perception path (lidar type 3d + mapper), which adds the holes/
        staleness/odometry error this oracle doesn't have.

        Assumes the robot's own shapes opt out of rays (`raycast: false`, the
        convention for robot YAMLs) — otherwise the scan reads the robot's
        back instead of the ground. Overhangs: the FIRST surface from z_top
        down wins (terrain-top convention, like the lidar 2d path)."""
        off = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        c, s = np.cos(yaw), np.sin(yaw)
        wx = float(base_xy[0]) + c * off[:, 0] - s * off[:, 1]
        wy = float(base_xy[1]) + s * off[:, 0] + c * off[:, 1]
        # G+1 vertical rays (last = under-base reference) in ONE tact_raycast_world
        # call — one _fk + shape cache for the whole scan. The per-point
        # query loop this replaced re-ran that fixed cost G+1 times and was
        # 22.8% of a StairsPolicy oracle tick (tests/_prof_height_scan.py);
        # same rays, so results are bit-identical.
        G = len(off)
        R0s = np.empty((G + 1, 3))
        R0s[:G, 0], R0s[:G, 1] = wx, wy
        R0s[G, 0], R0s[G, 1] = base_xy[0], base_xy[1]
        R0s[:, 2] = z_top
        Rds = np.zeros((G + 1, 3)); Rds[:, 2] = -1.0
        t = np.empty(G + 1)
        q = np.ascontiguousarray(self.q, dtype=np.float64)
        clib.tact_raycast_world(self.m._h, q.ctypes.data_as(_DBL),
                                R0s.ctypes.data_as(_DBL), Rds.ctypes.data_as(_DBL),
                                ctypes.c_int(G + 1), t.ctypes.data_as(_DBL))
        h = np.where(t >= 0.0, z_top - t, np.nan)
        ok = ~np.isnan(h[:-1])
        base_valid = not np.isnan(h[-1])
        ref = float(h[-1]) if base_valid else 0.0
        out = np.where(ok, h[:-1] - ref, default)
        self.last = dict(valid=ok, base_valid=base_valid, ref=ref,
                         n_valid=int(ok.sum()))
        return out

    # NOTE: Env.raycast() (n arbitrary world rays) was inlined into height_scan
    # 2026-06-06 (principle (5)): height_scan was its only consumer. Ad-hoc/
    # debug access goes straight to the C primitive — origins/dirs (n, 3)
    # contiguous float64, dirs unit-norm, t (n,) forward ranges, -1 = miss:
    #     t = np.empty(len(R0s))
    #     clib.tact_raycast_world(env.m._h,
    #         np.ascontiguousarray(env.q).ctypes.data_as(_DBL),
    #         R0s.ctypes.data_as(_DBL), Rds.ctypes.data_as(_DBL),
    #         ctypes.c_int(len(R0s)), t.ctypes.data_as(_DBL))

    # NOTE: raymap()/raycloud() (2026-06-06) and then Env._raycast_frame itself
    # were inlined into lidar_frames (principle (5)): each was a thin layer with
    # lidar_frames as the only production consumer. Ad-hoc/debug/bench access
    # composes _ray_grid with the C primitive directly — dirs (N, 3) unit,
    # registered-frame; t (N,) forward ranges, -1 = miss:
    #     dirs = env._ray_grid(w, h, dth, pinhole)          # cached
    #     t = np.empty(len(dirs))
    #     clib.tact_raycast_frame(env.m._h,
    #         np.ascontiguousarray(env.q).ctypes.data_as(_DBL),
    #         ctypes.c_int(env.m.fdict[frame]), dirs.ctypes.data_as(_DBL),
    #         ctypes.c_int(len(dirs)), t.ctypes.data_as(_DBL))
    #     D    = t.reshape(h, w)                            # depth map
    #     pts  = t[t >= 0, None] * dirs[t >= 0]             # sensor-frame cloud

    def _ray_grid(self, width, height, dth, pinhole):
        """Unit ray directions (H*W, 3) in SENSOR-FRAME coordinates, cached per
        (w, h, dth, pinhole). This is THE single source of per-pixel ray
        generation — C is a pure intersector (tact_raycast_frame takes these
        directions verbatim; the old in-C duplicate in tact_raymap_query was
        removed 2026-06-06). Directions live in the frame as registered (YAML
        pos/euler) — the -90° optical roll of the camera/render convention is
        folded into the rays, so a camera and a lidar at the same frame
        pos/euler produce the identically-oriented image, and world points =
        Te[:3,:3] @ p + Te[:3,3] with the PLAIN m.fkh pose. The sensor looks
        along the frame's -Z; an upright, forward-looking sensor pitched α°
        down is a pure Y rotation: euler [0, α-90, 0] (xyz, deg).

        pinhole=False (default): angular projection (LiDAR-like) — pixels
        uniform in (pitch, tilt) angle, dth = degrees per pixel (horizontal
        FoV = W·dth); straight world lines curve in image. pinhole=True:
        rectilinear (standard RGB-D camera) — pixels uniform on the image
        plane, lines stay straight. row-major: index i*W+j ↔ pixel (i=row
        from top, j=col from left)."""
        key = (width, height, dth, bool(pinhole))
        cache = getattr(self, '_ray_grid_cache', None)
        if cache is None:
            cache = self._ray_grid_cache = {}
        if key in cache:
            return cache[key]
        i = np.arange(height, dtype=np.float64)         # row    (0 = top)
        j = np.arange(width, dtype=np.float64)          # column (0 = left)
        if pinhole:
            f = (width / 2.0) / np.tan(np.radians(width * dth) / 2.0)
            u = (j + 0.5 - width / 2.0) / f
            v = (height / 2.0 - i - 0.5) / f
            uu, vv = np.meshgrid(u, v, indexing='xy')   # (H, W) row-major like D
            n = 1.0 / np.sqrt(uu**2 + vv**2 + 1.0)
            cam = np.stack([uu * n, vv * n, -n], axis=-1)
        else:                                           # angular (LiDAR-like)
            # tact.c's even/odd cases both reduce to ((N-1)/2 - idx)*dth in real
            # arithmetic: even N/2 - idx - 0.5 == odd (N-1)/2 - idx.
            pitch = np.radians(((height - 1) / 2.0 - i) * dth)
            tilt = np.radians(((width - 1) / 2.0 - j) * dth)
            tt, pp = np.meshgrid(tilt, pitch, indexing='xy')
            cam = np.stack([-np.sin(tt) * np.cos(pp), np.sin(pp),
                            -np.cos(tt) * np.cos(pp)], axis=-1)
        cam = cam.reshape(-1, 3)
        # optical -> registered frame: p_f = Rz(-90°) @ p_opt  (rows [0,1,0],[-1,0,0],[0,0,1])
        rays = np.stack([cam[:, 1], -cam[:, 0], cam[:, 2]], axis=-1)
        cache[key] = rays
        return rays

    def _push_light(self):
        # Push lights[0] into render.c module statics. Cheap (no GL); called every
        # render so YAML-driven changes via env.m.lights[0][...] = ... apply immediately.
        L = self.m.lights[0]
        pos = (ctypes.c_float * 3)(*L['pos'])
        tgt = (ctypes.c_float * 3)(*L['target'])
        clib.render_set_light(pos, tgt, ctypes.c_float(L['ortho']), ctypes.c_int(1 if L['shadow'] else 0))

    def _geom_arrays(self):
        # Camera-invariant inputs to win_render/egl_render (object poses/shapes/types/
        # colors). These depend only on q + the model, not the camera, so when several
        # renders happen at the same tick (window redraw + per-camera get_rgb_image, or
        # kida.run's 3 cameras per cycle) they share one FK + pose-assembly pass.
        # Rebuilt only when q changes (next sim step) or the object count changes
        # (env.add/delete); for a runtime env.edit() while paused (q frozen), set
        # self._imgcache = None to force a rebuild.
        q = self.q
        n_obj = len(self.m.ctype)
        c = getattr(self, '_imgcache', None)
        if c is None or c[0] != n_obj or not np.array_equal(c[1], q):
            n_padding = 8
            shape = [x for row in self.m.cshape for x in (row + [0]*n_padding)[:n_padding]]
            _shape = (ctypes.c_float*len(shape))(*shape)
            _type = (ctypes.c_int*n_obj)(*self.m.ctype)
            _objcolor = (ctypes.c_float*len(self.m.crgba))(*self.m.crgba)
            T = _fk(self.m.Ti, self.m.parent, self.m.jtype, q)
            objpose = np.array([])
            for i in range(n_obj):
                if self.m.cbody[i] < 0: tmp = self.m.ctran[i]
                else: tmp = T[self.m.cbody[i]] @ self.m.ctran[i]
                objpose = np.concatenate((objpose, tmp.T.flatten()))
            _objpose = (ctypes.c_float*len(objpose))(*objpose)
            c = self._imgcache = (n_obj, q.copy(), _type, _shape, _objcolor, _objpose)
        return c[2], c[3], c[4], c[5]

    def _win_render(self):
        _type, _shape, _objcolor, _objpose = self._geom_arrays()
        campose = self.m.view
        _campose = (ctypes.c_float*len(campose))(*campose)
        self._push_light()
        ret = clib.win_render(len(_type), _type, _shape, _objcolor,_objpose, _campose)
        return ret
    
    # NOTE: the per-type getters (get_rgb_image/get_depth_image/get_lidar_image/
    # get_lidar_points) and the intermediate wrappers (raymap/raycloud,
    # _render_frame) were inlined into camera_frames/lidar_frames 2026-06-06
    # (principle (5)): consumers only ever read the frames() generators, and the
    # middle layers were 1-line delegations or re-looked up specs the caller
    # already held (plus dead undeclared-frame fallbacks). Ad-hoc/debug access:
    # lidar → compose _ray_grid + clib.tact_raycast_frame (recipe below); camera → iterate
    # camera_frames() on a due tick (cnt=0 publishes everything).

    def camera_frames(self):
        """Yield (name, payload_bytes) for each camera due to publish at the current
        step. Rate-gating (camera `fps` vs the internal step counter self.cnt, the
        same one that drives window redraw) and type→encoder dispatch both live
        here; the per-camera publish cycle is computed from the sim dt and cached.
        Sockets/transport stay with the caller — this only renders + gates, so the
        sim core has no IPC dependency. self.cnt is the post-step value (the caller
        runs this after step()), so the publish phase trails window redraw by one
        tick — cadence is identical.

        Wire formats (per camera `type`):
          rgb   → JPEG bytes (decode with PIL/cv2/turbojpeg), encoded C-side in
                  egl_render.
          depth → zstd-compressed little-endian float32, row-major top-to-bottom,
                  linear eye-space distance in meters (no-geometry pixels read the
                  far plane, 200 m). Decode:
                  `np.frombuffer(zstandard.decompress(buf), '<f4').reshape(h, w)`
                  with (w, h) = the camera `res`."""
        for c in self.m.cameras:
            cyc = c.get('_cycle')
            if cyc is None:
                cyc = c['_cycle'] = max(1, round((1.0/self.m.dt)/c['fps'])) if c.get('fps') else 1
            if self.cnt % cyc: continue
            opt = 1 if c['type'] == 'rgb' else 2
            w, h = int(c['res'][0]), int(c['res'][1])
            # Grow the receive buffer to the worst-case payload. JPEG (opt=1) is
            # bounded well under w*h*4. zstd of a w*h float32 depth map (opt=2) can
            # slightly exceed its raw size (ZSTD_compressBound), so add slack
            # (> raw/255) on the depth path.
            need = w * h * 4
            if opt == 2: need += w * h // 16 + 4096
            if ctypes.sizeof(self._imgbuf) < need:
                self._imgbuf = (ctypes.c_ubyte * need)()

            _type, _shape, _objcolor, _objpose = self._geom_arrays()

            # Camera pose: registered frame + the -90° optical roll about -Z — the
            # render-path twin of the roll _ray_grid folds into the lidar rays.
            tmp = self.m.fkh([c['name']], self.q)[0]
            tmp = tmp @ xyzeuler_to_homogeneous([0, 0, 0, 0, 0, -np.pi/2])
            campose = np.linalg.inv(tmp).T.flatten()
            _campose = (ctypes.c_float*len(campose))(*campose)

            self._push_light()
            # vfov passed as c_float: egl_render has no declared argtypes, so a bare
            # Python float would be marshalled as c_double and misread by the C `float`.
            imglen = clib.egl_render(len(_type), _type, _shape, _objcolor, _objpose,
                                     _campose, self._imgbuf, opt, w, h,
                                     ctypes.c_float(c['vfov']))
            if imglen > 0: yield c['name'], ctypes.string_at(self._imgbuf, imglen)

    # NOTE: the lidar wire is RAW float32 since 2026-06-06 (Python-side zstd
    # removed): float coordinates compress only ~x1.5 (high-entropy mantissas)
    # at ~0.5 ms/frame on the single-threaded sim loop, while /dev/shm IPC has
    # ~1000x bandwidth headroom (3d cloud raw ≈ 5.6 MB/s @ 30 fps) — and real
    # lidar drivers publish raw frames too. Depth CAMERAS still arrive
    # zstd-compressed (C-side in egl_render, a separate path); rgb stays JPEG.

    def lidar_frames(self):
        """Yield (name, payload_bytes) for each lidar due to publish at the current step.
        Mirrors camera_frames: fps rate-gating against self.cnt and type→encoder
        dispatch both live here, so the sim core stays IPC-free — the caller owns
        sockets/transport. Cadence matches cameras (publish trails window redraw by
        one tick). A real-hardware lidar driver publishing the same wire format is
        indistinguishable to the consumer.

        Wire formats (per lidar `type`) — RAW little-endian float32, no compression:
          2d → row-major top-to-bottom depth map. Decode:
               `np.frombuffer(buf, '<f4').reshape(h, w)` with (w, h) = the lidar
               `res`. Values are range-along-ray in meters (LiDAR convention),
               with -1 for pixels that hit nothing. (A depth CAMERA's wire is the
               same float32 image but zstd-compressed C-side and far-plane 200 m
               instead of -1.)
          3d → (N, 3) points in SENSOR-FRAME coordinates — the frame as registered
               in the YAML, so the consumer composes the frame extrinsic with its
               own pose estimate (pts_w = pts @ Te[:3,:3].T + Te[:3,3], Te = plain
               fkh). Decode: `np.frombuffer(buf, '<f4').reshape(-1, 3)`.
               N varies per frame: no-hit rays are dropped, and hits beyond the
               spec's `max_range` are dropped too. Rays hit the robot's own
               raycast-on shapes as well — self-filtering is the consumer's job."""
        for l in self.m.lidars:
            cyc = l.get('_cycle')
            if cyc is None:
                cyc = l['_cycle'] = max(1, round((1.0/self.m.dt)/l['fps'])) if l.get('fps') else 1
            if self.cnt % cyc: continue
            dirs = self._ray_grid(int(l['res'][0]), int(l['res'][1]), l['dth'], l['pinhole'])
            # one tact_raycast_frame call = one _fk + shape cache + frustum cull
            # for the whole bundle (frame is guaranteed: _register_sensors
            # registered every lidar as a named frame at parse time)
            t = np.empty(len(dirs))
            q = np.ascontiguousarray(self.q, dtype=np.float64)
            clib.tact_raycast_frame(self.m._h, q.ctypes.data_as(_DBL),
                                    ctypes.c_int(self.m.fdict[l['name']]),
                                    dirs.ctypes.data_as(_DBL),
                                    ctypes.c_int(len(dirs)),
                                    t.ctypes.data_as(_DBL))
            if l['type'] == '2d':
                if l['perpendicular']:
                    # camera-Z depth = range × |cos to look dir| = -dir_z (the
                    # optical roll is about Z) — same factor for both projections
                    # (angular: cos(pitch)cos(tilt); pinhole: 1/√(u²+v²+1)).
                    t = np.where(t >= 0.0, t * -dirs[:, 2], t)
                yield l['name'], t.astype('<f4').tobytes()
            elif l['type'] == '3d':
                hit = t >= 0.0
                if l.get('max_range') is not None:
                    hit &= t <= l['max_range']
                pts = t[hit, None] * dirs[hit]
                yield l['name'], pts.astype('<f4').tobytes()


class CEnv:
    """Thin adapter that gives a ctypes.CDLL (bin/mjenv.so / chenv.so / eio.so)
    the core backend contract (step/reset/finish/backend/has_pd/dt — see
    docs/backend-interface.md) of tact.Env, so callers (the start script, RL
    envs) can stay backend-agnostic. Capabilities beyond the core are optional
    per backend (capability ledger in the same doc); C-symbol capabilities are
    probed + argtypes-declared in __init__ (the get_dt probe is the model).
    Other per-robot C commands (e.g. eio's set_abf) are forwarded transparently
    via __getattr__ (caller owns the signature — prefer ledger-declared methods).

    Usage: load and init the cdll yourself, then wrap:
        cdll = ctypes.CDLL(f'{tact.pkg_dir}/bin/mjenv.so')
        cdll.init(xmlpath.encode(), redraw_or_0)
        env = tact.CEnv(cdll, n_y=28, n_u=12, backend='mujoco', has_pd=True)

    `backend` is a free-form label ('mujoco', 'chrono', 'real', ...) that
    controllers can read via env.backend to pick gains/timing per backend.
    `has_pd` declares whether the backend handles joint-space PD itself
    (caller picks: mujoco=True for dual-actuator XML, chrono=False, real depends
    on firmware). Controllers may override env.has_pd post-construction
    under their own responsibility."""
    def __init__(self, cdll, n_y, n_u, backend, has_pd=False):
        self.cdll = cdll
        self.n_y = n_y
        self.n_u = n_u
        self.backend = backend
        self.has_pd = has_pd
        self._y = (ctypes.c_double*n_y)()
        # Unified contract — every backend exports:
        #   int  step(double* u, double* q_ref, double* qd_ref, double* y)
        #   void reset(double* y)
        # Backends that don't implement implicit PD (eio, current chrono) accept
        # q_ref/qd_ref and ignore them; the NULL pointer (Python None) is the
        # "PD inactive" signal. argtypes MUST be declared so ctypes converts None → NULL.
        self.cdll.step.restype  = ctypes.c_int
        self.cdll.step.argtypes = [_DBL, _DBL, _DBL, _DBL]
        self.cdll.reset.restype  = None
        self.cdll.reset.argtypes = [_DBL]
        # Terrain probe for the height_scan parity wrapper below — mjenv's
        # `height_scan` export carries the FULL contract (yaw rotation, group-1
        # vertical rays, base-relative subtraction, default fill, validity);
        # the Python wrapper is a thin alloc+dict shim. Absolute world-z is NOT
        # exposed to callers (`get_z` stays blocked in __getattr__ — removed
        # 2026-06-06, sim-trick reduction; its C export was replaced by this
        # 2026-06-07). height_scan is bound as an INSTANCE attribute iff the
        # backend has the export, so hasattr(env, 'height_scan') stays False on
        # chrono/real and scan consumers (e.g. dog.py StairsPolicy) fall back
        # to blind mode.
        try:
            self.cdll.height_scan.restype  = None
            self.cdll.height_scan.argtypes = [
                _DBL, ctypes.c_double, _DBL, ctypes.c_int,        # base_xy, yaw, offsets, G
                ctypes.c_double, ctypes.c_double,                  # z_top, default
                _DBL, _INT, _INT, _DBL]                            # h_out, valid, base_valid, ref
            self.height_scan = self._height_scan
        except AttributeError:
            pass
        # Optional sim-timestep query — exported by mjenv (mujoco). Cached here (constant)
        # so `start` can derive the render cadence + real-time pacing for the mujoco backend
        # the same way as the tact backend. None for backends without it (chrono/real) →
        # those stay un-paced (env.dt is None). Model is already loaded (init ran before us).
        self._dt = None
        try:
            self.cdll.get_dt.restype = ctypes.c_double
            self.cdll.get_dt.argtypes = []
            self._dt = float(self.cdll.get_dt())
        except AttributeError:
            pass
        # Optional render-cadence setter (mjenv): start drives it from -s/-f like tact.
        try:
            self.cdll.set_redraw.restype  = None
            self.cdll.set_redraw.argtypes = [ctypes.c_int]
        except AttributeError:
            pass

    def step(self, tau=None, q_ref=None, qd_ref=None, kp=None, kd=None):
        # All input channels are equal-priority and independently optional.
        # tau=None → zero feedforward buffer (C backends currently dereference tau
        # unconditionally, so we always pass a real pointer of length n_u).
        # q_ref/qd_ref=None → NULL pointer = backend's internal PD inactive.
        # kp/kd: per-step PD gains, forwarded to the backend (mjenv writes them
        # into the position actuators' gainprm each step — the XML kp/kv are
        # structural placeholders). Same error semantics as Model.step: a
        # reference without its gain raises. On real eio the gains live in the
        # driver/firmware — the pointers are extra args its step() ignores until
        # its signature picks them up (capability ledger).
        if q_ref  is not None and kp is None:
            raise ValueError("q_ref requires kp — gains are per-step inputs (YAML/XML model gains were retired)")
        if qd_ref is not None and kd is None:
            raise ValueError("qd_ref requires kd — gains are per-step inputs (YAML/XML model gains were retired)")
        if tau is None:
            _tau = (ctypes.c_double*self.n_u)()
        else:
            _tau = (ctypes.c_double*len(tau))(*tau)
        _qr  = (ctypes.c_double*len(q_ref))(*q_ref)    if q_ref  is not None else None
        _qdr = (ctypes.c_double*len(qd_ref))(*qd_ref)  if qd_ref is not None else None
        _kp  = (ctypes.c_double*len(kp))(*kp)          if kp     is not None else None
        _kd  = (ctypes.c_double*len(kd))(*kd)          if kd     is not None else None
        ret = self.cdll.step(_tau, _qr, _qdr, _kp, _kd, self._y)
        if ret < 0: print('ESC pressed. exit...'); sys.exit()
        return np.frombuffer(self._y, dtype=np.float64).copy()

    # NOTE: get_rgb_image wrapper removed 2026-06-06 (principle (5), zero live
    # callers since the archived kida.ws/replay.py streaming era). mjenv.cpp still
    # exports the C symbol; if mujoco camera parity is ever wanted, re-add a
    # declared wrapper matching Env's signature (frame, res, vfov) — see the
    # capability ledger (docs/backend-interface.md).

    def _height_scan(self, base_xy, yaw, offsets, z_top=None, default=0.0):
        """Base-relative terrain scan — thin shim over the backend's
        full-contract `height_scan` C export (mjenv: yaw rotation, group-1
        vertical rays, base-relative subtraction and default fill all in C).
        SAME contract as Env.height_scan: yaw-frame (G, 2) offsets → (G,)
        heights relative to the terrain under base_xy; miss → default;
        validity/base_valid/ref in self.last. Only the RELATIVE quantity
        crosses the interface, so sim2sim controllers stay trick-free.
        z_top=None keeps the historical mjenv ray origin (z=10)."""
        off = np.ascontiguousarray(np.asarray(offsets, dtype=np.float64).reshape(-1, 2))
        bxy = np.ascontiguousarray([float(base_xy[0]), float(base_xy[1])])
        G = len(off)
        h = np.empty(G); valid = np.zeros(G, dtype=np.int32)
        bv, ref = ctypes.c_int(0), ctypes.c_double(0.0)
        self.cdll.height_scan(bxy.ctypes.data_as(_DBL), ctypes.c_double(float(yaw)),
                              off.ctypes.data_as(_DBL), ctypes.c_int(G),
                              ctypes.c_double(10.0 if z_top is None else float(z_top)),
                              ctypes.c_double(float(default)),
                              h.ctypes.data_as(_DBL), valid.ctypes.data_as(_INT),
                              ctypes.byref(bv), ctypes.byref(ref))
        self.last = dict(valid=valid.astype(bool), base_valid=bool(bv.value),
                         ref=float(ref.value), n_valid=int(valid.sum()))
        return h

    def reset(self):
        """Reset backend state and return the initial observation y. C-side reset()
        fills self._y directly (no extra step), so the returned y is the true
        post-reset reading rather than the result of one zero-input step."""
        self.cdll.reset(self._y)
        return np.frombuffer(self._y, dtype=np.float64).copy()
    def finish(self): self.cdll.finish()

    @property
    def dt(self):
        # Sim timestep if the backend exports get_dt (mujoco), else None. Explicit
        # property (not __getattr__) so `start` can sleep-pace + derive redraw uniformly.
        return self._dt

    # tact-only capabilities (sensor publishing, dynamic topology) are deliberately
    # ABSENT here — no stubs. Per the core contract + capability ledger
    # (docs/backend-interface.md) absence is legitimate: hasattr() probes False and
    # callers guard on env.backend. The __getattr__ forwarding below stays — it is
    # the canonical channel for per-robot eio commands (set_abf, ...) — but
    # the tact-only names are blocked from it: they are mutation APIs with short,
    # generic C names, and dlsym finding a same-named unrelated symbol in some
    # future backend .so would otherwise become a silent wrong call (ctypes hands
    # out argtypes-less _FuncPtrs for whatever dlsym resolves).
    _TACT_ONLY = ('add', 'delete', 'groups',
                  'cameras', 'lidars', 'camera_frames', 'lidar_frames')

    def __getattr__(self, name):
        if name in CEnv._TACT_ONLY:
            raise AttributeError(
                f"{name!r} is a tact-only capability (docs/backend-interface.md); "
                f"backend={self.backend!r} does not provide it.")
        if name == 'get_z':
            # Removed 2026-06-06 (sim-trick reduction): absolute world-z terrain
            # queries have no real-robot counterpart. mjenv.so still exports the
            # C symbol, so without this block a stale caller would get an
            # argtypes-less _FuncPtr and silently garbage-marshal doubles.
            raise AttributeError(
                "get_z was removed — use the base-relative height_scan / "
                "scan-provider contract instead (docs/backend-interface.md).")
        return getattr(self.cdll, name)
