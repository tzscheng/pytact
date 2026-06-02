"""Simulation classes: Model (YAML → kinematic tree + step), Env (window +
EGL image buffer + add()), CEnv (ctypes-CDLL adapter for mujoco/chrono/real
backends). Pure math/dynamics primitives live in rbd.py and are re-exported
here via `from .rbd import *` so internal references stay flat."""
import sys, os, ctypes, math, copy
import numpy as np
import yaml
from ._clib import clib, _DBL, _INT
from .rbd import *
from .rbd import _fk, _q_step, _build_qidx   # underscored names are not pulled in by `import *`

# Private PID used by Model to lock free joints at simulation start (e.g., to
# keep a biped upright until the user's high-level controller takes over).
# Mirrors control.PIDController in interface but kept here to avoid sim → control
# coupling. Underscore prefix excludes it from `tact.*` flat namespace.
class _PIDController:
    def __init__(self, k_p, k_d, k_i, dt):
        self.k_p, self.k_d, self.k_i, self.dt = k_p, k_d, k_i, dt
        self.cnt = 0

    def update(self, q_d, q, d):
        return self.update_from_error(q_d - q, d)

    def update_from_error(self, e, d):
        """Same PID law but accept a pre-computed error vector (e.g., SO(3)-aware
        for free-joint lock where straight q_d - q would mix world/body frames)."""
        if self.cnt == 0: self.e_sum = np.zeros(len(e))
        u = self.k_p*e - self.k_d*d + self.k_i*self.e_sum
        self.e_sum += e*self.dt
        self.cnt += 1
        return u


class Model:
    def __init__(self, modelname, prefix=None, base='root', offset=[0, 0, 0, 0, 0, 0], q0=None, fixed_base=False, name=None):
        self.dt = 0.001
        # contact solver: 'lcp' (contact_lcp + semi-implicit Euler). The legacy
        # 'penalty' (spring-damper + brush friction) solver was removed 2026-05-24
        # (its brush could not hold a planted foot — see git/_ archive). 'lcp' only.
        self.solver = 'lcp'
        self.lam_prev = None    # LCP warm-start state (6 * MAX_PTS_PER_PAIR * npair vec,
                                # indexed by slot = cpair_idx * MAX_PTS_PER_PAIR + sub_id)
        self.g = [0, 0, 0] #[0, 0, -9.81]
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
        self.ff = np.array([], dtype=float)  # joint damping friction
        self.sk = np.array([], dtype=float)  # joint spring stiffness

        # implicit joint-space PD gains, per-jtype length. Populated from YAML `joint.k: [kp, kd]`
        # (default 0 when absent). Joint-PD only — task-PD was prototyped (Phase 3) but removed
        # because it can't be implemented on MuJoCo / real-hardware backends, breaking tact's
        # "swap backend, same agent" contract. Activation requires q_ref (or qd_ref) in step();
        # without that, gains are inert. Controllers may overwrite these arrays directly.
        self.Kp_j = np.array([], dtype=float)
        self.Kd_j = np.array([], dtype=float)
        
        self.ctype = []  # contact convex info
        self.cbody = []  # attached body index
        self.cshape = [] #contact support function shape parameter
        self.cparam = []  #contact parameters
        self.ctran = []  #transform
        self.crgb = []
        self.craycast = []  # per-shape int flag: 1=visible to raycast, 0=skipped (still renders + collides)
        
        self.f_idx = 1
        self.fdict = {'root': 0}
        self.fbody = [-1]
        self.ftran = [np.eye(4)]
        self.ftran_inv = [np.eye(4)]

        self.feeds = []
        self.lock_idx = []
        #self.pid = _PIDController(np.array([5000, 5000, 5000, 50, 50, 50.]), np.array([50, 50, 50, 0.5, 0.5, 0.5]), 0, 0.001)
        self.pid = _PIDController(np.array([40000, 40000, 40000, 400, 400, 400.]), np.array([400, 400, 400, 4, 4, 4.]), 0, 0.001)

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
        self._mesh_max_slots = 64     # matches MAX_MESH in ccd.c
        self.add(modelname, prefix, base, offset, q0, fixed_base, name=name)

    def _snapshot_sizes(self):
        return {
            'nb':     len(self.jtype),
            'nq':     len(self.q0),
            'nshape': len(self.ctype),
            'nframe': len(self.fbody),
            'nfeeds': len(self.feeds),
            'nlock':  len(self.lock_idx),
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

        offset0 = np.array(offset)
        T0 = xyzeuler_to_homogeneous(offset0, eulerseq='XYZ', deg=True)
        q0_idx = 0

        #materials library: contact: [pair_id, mat_name] expanded into a 12-tuple
        #  [pair_id, k_n, d_n, k_t, d_t, mu, k_spin, d_spin, mu_spin, k_roll, d_roll, mu_roll]
        #material spec is grouped by physical concept:
        #  {normal: [k_n, d_n], tangent: [k_t, d_t, mu], spin: [k_spin, d_spin, mu_spin], roll: [k_roll, d_roll, mu_roll]}
        materials = config.get('materials', {}) or {}
        _MAT_GROUPS = (('normal', 2), ('tangent', 3), ('spin', 3), ('roll', 3))

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
                    if 'rgb' not in sh: sh['rgb'] = [-1, 0, 0]
                    if 'contact' not in sh:
                        sh['contact'] = [-1] + [0.0]*11
                    else:
                        contact = sh['contact']
                        pair_id = int(contact[0])
                        if pair_id < 0:
                            sh['contact'] = [pair_id] + [0.0]*11
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
            while True:
                if config['bodies'][i]['name'] == 'root':
                    del config['bodies'][i]
                    continue
                
                if 'joint' in config['bodies'][i]:
                    if config['bodies'][i]['joint']['type'] == 'free':
                        head = config['bodies'][i]['name']
                        del config['bodies'][i]
                        continue
                        
                    if config['bodies'][i]['joint']['parent'] == head:
                        config['bodies'][i]['joint']['parent'] = 'root'

                    i += 1
                if i == len(config['bodies']): break;

        self.build(config, prefix, modelname)

        after = self._snapshot_sizes()
        self.groups.append({
            'name':       gname,
            'nb':         (before['nb'],     after['nb']),
            'nq':         (before['nq'],     after['nq']),
            'nshape':     (before['nshape'], after['nshape']),
            'nframe':     (before['nframe'], after['nframe']),
            'nfeeds':     (before['nfeeds'], after['nfeeds']),
            'nlock':      (before['nlock'],  after['nlock']),
            'nfixed':     (before['nfixed'], after['nfixed']),
            'fdict_keys': list(set(self.fdict.keys()) - fdict_before),
        })

    def reset(self):
        if len(self.lock_idx) > 0: self.is_locked = True
        else: self.is_locked = False
        self.lam_prev = None

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
        # — globals belong on the root model or go through env.set(...).
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
                #      f"ignored on add(). Use env.set(...) to change mid-session.")
        
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
                    self.ff   = np.append(self.ff, 0)
                    self.sk   = np.append(self.sk, 0)
                    self.Kp_j = np.append(self.Kp_j, 0)
                    self.Kd_j = np.append(self.Kd_j, 0)
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

                if 'lock' in body['joint']:
                    if body['joint']['lock']: self.lock_idx.append(num)

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

                    # implicit joint-space PD gains: `k: [kp, kd]`. Both default to 0 when absent
                    # (no PD effect). Controllers can still overwrite model.Kp_j / model.Kd_j later.
                    if 'k' in body['joint']:
                        self.Kp_j = np.append(self.Kp_j, body['joint']['k'][0])
                        self.Kd_j = np.append(self.Kd_j, body['joint']['k'][1])
                    else:
                        self.Kp_j = np.append(self.Kp_j, 0)
                        self.Kd_j = np.append(self.Kd_j, 0)

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

                    if name == 'root': self.cbody.append(-1)
                    else: self.cbody.append(self.fbody[self.fdict[name]])

                    self.ctype.append(num)
                    self.ctran.append(xyzeuler_to_homogeneous(v['pos'] + v['euler'], eulerseq=v['eulerseq'], deg=True))
                    self.cshape.append(v['param'])
                    self.cparam.append(v['contact'])
                    # raycast default = visible (rgb[0] >= 0). rgb[0]<0 is the render-skip sentinel,
                    # so an invisible shape is also invisible to ray sensors. Override with `raycast: true/false`.
                    visible = (v['rgb'][0] >= 0)
                    self.craycast.append(1 if v.get('raycast', visible) else 0)
                    self.crgb += v['rgb']
                    
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
                
        #set initial body locking controller
        if len(self.lock_idx) > 0: self.is_locked = True
        else: self.is_locked = False

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
        self.Kp_j  = np.concatenate([self.Kp_j[:nq_lo],  self.Kp_j[nq_hi:]])
        self.Kd_j  = np.concatenate([self.Kd_j[:nq_lo],  self.Kd_j[nq_hi:]])
        self.active = self.active[:nq_lo] + self.active[nq_hi:]

        # 3. Splice per-shape arrays. crgb is flat (3 floats per shape).
        del self.ctype[nsh_lo:nsh_hi]
        del self.cbody[nsh_lo:nsh_hi]
        del self.cshape[nsh_lo:nsh_hi]
        del self.cparam[nsh_lo:nsh_hi]
        del self.ctran[nsh_lo:nsh_hi]
        del self.craycast[nsh_lo:nsh_hi]
        del self.crgb[nsh_lo*3:nsh_hi*3]

        # 4. Splice per-frame arrays + drop fdict keys
        del self.fbody[nf_lo:nf_hi]
        del self.ftran[nf_lo:nf_hi]
        del self.ftran_inv[nf_lo:nf_hi]
        for k in g['fdict_keys']:
            if k in self.fdict: del self.fdict[k]
        self.f_idx -= df

        # 5. Shift surviving body indices in parent/cbody/fbody/lock_idx/fixed.
        #    None / -1 (root sentinel) stay untouched because they're < nb_lo.
        self.parent = [p if (p is None or p < nb_lo) else p - db for p in self.parent]
        self.cbody  = [c if c < nb_lo else c - db for c in self.cbody]
        self.fbody  = [b if b < nb_lo else b - db for b in self.fbody]
        self.lock_idx = [(x - db) if x >= nb_hi else x
                         for x in self.lock_idx if not (nb_lo <= x < nb_hi)]
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

        # 7. Shift fdict values (frame indices) past the deleted frame range
        for k, v in list(self.fdict.items()):
            if v >= nf_hi: self.fdict[k] = v - df

        # 8. Shift group metadata for groups after this one
        nfeeds_d = feed_hi - feed_lo
        nlock_d  = g['nlock'][1]  - g['nlock'][0]
        nfixed_d = g['nfixed'][1] - g['nfixed'][0]
        for g2 in self.groups[gi+1:]:
            g2['nb']     = (g2['nb'][0]     - db, g2['nb'][1]     - db)
            g2['nq']     = (g2['nq'][0]     - dq, g2['nq'][1]     - dq)
            g2['nshape'] = (g2['nshape'][0] - ds, g2['nshape'][1] - ds)
            g2['nframe'] = (g2['nframe'][0] - df, g2['nframe'][1] - df)
            g2['nfeeds'] = (g2['nfeeds'][0] - nfeeds_d, g2['nfeeds'][1] - nfeeds_d)
            g2['nlock']  = (g2['nlock'][0]  - nlock_d,  g2['nlock'][1]  - nlock_d)
            g2['nfixed'] = (g2['nfixed'][0] - nfixed_d, g2['nfixed'][1] - nfixed_d)
        self.groups.pop(gi)

        # 9. Rebuild derived data + C handle. lam_prev (LCP warm-start) is invalid
        #    because cpair size changes.
        self.lam_prev = None
        self.X  = get_spatial_transform(self.Ti)
        self.I6 = get_spatial_inertia(self.m, self.c, self.I)
        self._rebuild_cpair()
        self.is_locked = len(self.lock_idx) > 0
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

    def set(self, solver=None, dt=None, g=None, view=None):
        # Single channel for changing globals (solver, dt, g, view) mid-session — names
        # match the YAML `sim:` block. All args optional; only provided fields update.
        # Solver other than 'lcp' is silently ignored so callers can forward stray
        # kwargs without harm. Env.set is a thin passthrough; env.has_pd is a property
        # derived from self.solver.
        if solver in ('lcp', 'minimal'): self.solver = solver
        if dt is not None: self.dt = float(dt)
        if g is not None and len(g) > 0: self.g = np.array(g, dtype=np.float64)
        if view is not None: self.view = view

        #topology unchanged — push dt/g in place. The integrator arg is vestigial
        #(the lcp path uses its own semi-implicit Euler); pass a fixed value.
        if self.use_c and getattr(self, '_h', None):
            g_arr = np.ascontiguousarray(np.asarray(self.g), dtype=np.float64)
            clib.tact_set_sim(self._h, ctypes.c_double(self.dt), ctypes.c_int(2), g_arr.ctypes.data_as(_DBL))

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
        self._build_cparam = np.ascontiguousarray(np.asarray(self.cparam).reshape(nshape, 12) if nshape else np.zeros((0, 12)), dtype=np.float64)

        self._h = clib.tact_create(
            nb,
            self._build_parent.ctypes.data_as(_INT),
            self._build_jtype.ctypes.data_as(_INT),
            self._build_X.ctypes.data_as(_DBL),
            self._build_I6.ctypes.data_as(_DBL),
            self._build_Ti.ctypes.data_as(_DBL),
            self._build_ff.ctypes.data_as(_DBL),
            self._build_sk.ctypes.data_as(_DBL),
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

    def step(self, q, qd, tau=None, q_ref=None, qd_ref=None):
        # All three input channels are equal-priority and independently optional.
        # tau=None → treated as zero feedforward (passive step under gravity/contact).
        # q_ref/qd_ref activate internal joint-space PD on the LCP path; when both None,
        # behavior is bit-identical to pre-PD step.
        # ff damping and sk spring are applied implicitly inside aba_featherstone.
        # tau/q_ref/qd_ref are per-DoF (length nq), not per-body (length nb) — these
        # differ only when free6 (jtype=3) joints are present.
        if tau is None: tau = np.zeros(len(q))

        #initial body fixing if fb test — SO(3)-aware error for free joint:
        #  position error in body frame (R^T applied to world displacement)
        #  rotation error via log(R^T · R0) (body-frame angular error vector)
        #  qd is already body-frame [v; ω] under the axis-angle free convention.
        if self.is_locked:
            q_base, v_base, _, _, _, _ = _build_qidx(self.jtype)
            for body_idx in self.lock_idx:
                qb = q_base[body_idx]; vb = v_base[body_idx]
                R  = expmap_so3(q[qb+3:qb+6])
                R0 = expmap_so3(self.q0[qb+3:qb+6])
                e_p = R.T @ (self.q0[qb:qb+3] - q[qb:qb+3])
                e_w = logmap_so3(R.T @ R0)
                e   = np.concatenate([e_p, e_w])
                tau[vb:vb+6] += self.pid.update_from_error(e, qd[vb:vb+6])

        if self.solver == 'minimal':
            # Test-only solver (rbd.contact_ground_sphere): explicit spring-damper
            # ground (z=0) contact for SPHERE shapes, fed as f_ext into ABA forward
            # dynamics + semi-implicit Euler. Runs on the Python path regardless of
            # use_c; the C handle (if created) stays valid for fk/jacob/raycast.
            # Not for production — spheres only, no Coulomb cone, needs a small dt.
            T = _fk(self.Ti, self.parent, self.jtype, q)
            f_ext = contact_ground_sphere(T, self.parent, self.jtype, self.ctype,
                                          self.cbody, self.ctran, self.cshape, self.cparam, qd)
            qdd, f, a, v = aba_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, tau,
                                            f_ext, self.g, full=True, ff=self.ff, sk=self.sk,
                                            dt=self.dt, Kp_j=self.Kp_j, Kd_j=self.Kd_j,
                                            q_ref=q_ref, qd_ref=qd_ref)
            qd_next = qd + qdd * self.dt
            q_base, _, _, _, _, _ = _build_qidx(self.jtype)
            q_next  = _q_step(q, qd_next, self.dt, self.jtype, q_base)
            y = self.feedback(q, qd, tau, T, f, a, v, f_ext)
        elif self.use_c and self.solver == 'lcp':
            #single ctypes round-trip into tact_step_lcp:
            #   _fk → aba(no-contact) → crb → contact_lcp → semi-implicit → feedback
            #C side reads raw tau; ff/sk damping + implicit joint-PD are applied internally.
            #Gated on solver=='lcp': the C handle ONLY implements lcp, so any other
            #solver must not silently run here (an unknown one falls to the else → raise).
            q_in   = np.ascontiguousarray(q,   dtype=np.float64)
            qd_in  = np.ascontiguousarray(qd,  dtype=np.float64)
            tau_in = np.ascontiguousarray(tau, dtype=np.float64)

            # implicit joint-PD pointers — NULL when activation missing (no q_ref/qd_ref) or
            # capability explicitly cleared (controller set Kp_j/Kd_j = None). Default arrays
            # are zero-initialized from YAML, so PD is inert until either YAML `k:` or
            # controller-side assignment provides non-zero gains.
            Kp_ptr  = self.Kp_j.ctypes.data_as(_DBL)  if (self.Kp_j  is not None and q_ref is not None) else None
            Kd_ptr  = self.Kd_j.ctypes.data_as(_DBL)  if (self.Kd_j  is not None and (q_ref is not None or qd_ref is not None)) else None
            qr_ptr  = q_ref.ctypes.data_as(_DBL)      if q_ref       is not None else None
            qdr_ptr = qd_ref.ctypes.data_as(_DBL)     if qd_ref      is not None else None
            clib.tact_step_lcp(self._h, q_in.ctypes.data_as(_DBL), qd_in.ctypes.data_as(_DBL), tau_in.ctypes.data_as(_DBL), Kp_ptr, Kd_ptr, qr_ptr, qdr_ptr)

            #copy outputs out of arena (next step would overwrite views)
            q_next  = self._h_q_next.copy()
            qd_next = self._h_qd_next.copy()
            y       = self._h_y[:self._y_size].copy()

        else:
            T = _fk(self.Ti, self.parent, self.jtype, q)
            if self.solver == 'lcp':
                #LCP path: ABA-with-joint-PD(f_ext=0) → qd_free, CRB → M, contact_lcp solves for
                #impulse λ → semi-implicit Euler. A second ABA call with the contact wrench feeds
                #the feedback layer's f/a/v so accelerometer-like outputs reflect post-contact
                #dynamics. Without that second call, IMU feed shows a bias of order g·support_frac
                #whenever joint torques cancel out contact (e.g. quadruped stance).
                #Joint damping `ff`, spring `sk`, and joint-space implicit PD (Kp_j/Kd_j/q_ref/qd_ref)
                #are all folded into ABA's articulated inertia.
                f_ext_zero = np.zeros((len(self.X), 6))
                qdd_free, f, a, v = aba_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext_zero, self.g, full=True, ff=self.ff, sk=self.sk, dt=self.dt, Kp_j=self.Kp_j, Kd_j=self.Kd_j, q_ref=q_ref, qd_ref=qd_ref)
                qd_free = qd + qdd_free * self.dt
                M = crb_featherstone(self.X, self.I6, self.parent, self.jtype, q)
                dqd, lam, lcp_info, f_ext = contact_lcp(T, self.parent, self.jtype, self.cpair, self.ctype, self.cbody, self.ctran, self.cshape, self.cparam, qd_free, M, self.dt, lam_prev=self.lam_prev)
                self.lam_prev = lcp_info['lam_full']   #persist for next step's warm-start
                qd_next = qd_free + dqd
                q_base, _, _, _, _, _ = _build_qidx(self.jtype)
                q_next  = _q_step(q, qd_next, self.dt, self.jtype, q_base)
                qdd = (qd_next - qd) / self.dt
                # Kinematic forward pass (RNE): given realized qdd, propagate spatial accels
                # so feedback (a, v, f) reflects post-contact body dynamics.
                _, f, a, v = rne_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, qdd, f_ext, self.g, full=True)

            else: raise ValueError(f'unknown solver: {self.solver}')
            y = self.feedback(q, qd, tau, T, f, a, v, f_ext)

        #if cff: return q_next, qd_next, y, cfs
        #else: return q_next, qd_next, y, f, a, v
        return q_next, qd_next, y #, f, a, v

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
    


class Env:
    backend = 'tact'

    def __init__(self, src, prefix=None, base='root', offset=[0, 0, 0, 0, 0, 0], q0=None, fixed_base=False, render=False, redraw=20, name=None): #, index=0):
        self.src = src
        self.m = Model(src, prefix, base, offset, q0, fixed_base, name=name)
        self.dof = sum(self.m.active)
        self.q = self.m.q0.copy()
        self.qd = self.m.qd0.copy()

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

    @property
    def groups(self):
        """List of currently-active group names, in insertion order."""
        return [g['name'] for g in self.m.groups]

    def set(self, **kw):
        # Thin passthrough to Model.set — single channel for globals
        # (solver, dt, g, view). env.has_pd is a @property derived from m.solver,
        # so nothing else to sync here.
        self.m.set(**kw)

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
        
    def step(self, tau=None, q_ref=None, qd_ref=None):
        # All three input channels are equal-priority and independently optional.
        # tau=None → zero feedforward (passive step); q_ref/qd_ref=None → backend's
        # internal PD inactive (caller is responsible for torque via tau).
        # Internal arrays are per-DoF (length nq), iterating self.m.active which is
        # the per-DoF active mask. For non-free6 models nq == nb so behavior is
        # unchanged; for free6 models the 6 DoFs per free6 body are all active=0.
        nq = len(self.q)
        tau_full = np.zeros(nq)
        qr_full  = None if q_ref  is None else np.zeros(nq)
        qdr_full = None if qd_ref is None else np.zeros(nq)
        idx = 0
        for k in range(nq):
            if self.m.active[k] > 0:
                if tau    is not None: tau_full[k] = tau[idx]
                if qr_full  is not None: qr_full[k]  = q_ref[idx]
                if qdr_full is not None: qdr_full[k] = qd_ref[idx]
                idx += 1

        self.q, self.qd, y = self.m.step(self.q, self.qd, tau=tau_full, q_ref=qr_full, qd_ref=qdr_full)
        
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

    def is_locked(self):
        return self.m.is_locked

    def unlock(self):
        self.m.is_locked = False

    def get_z(self, x, y, h=10.0):
        t = self.raycast(x, y, h, 0.0, 0.0, -1.0)
        return h - t if t >= 0 else 0.0

    #def get_camera_name(self):
    #    return [k for k in self.m.fdict.keys() if k.endswith('cam')]

    def raycast(self, pos_x, pos_y, pos_z, dir_x, dir_y, dir_z):
        """Single ray vs all collision shapes. Returns forward distance t (>0)
        along the ray, or -1 on miss. ray_dir should be unit-norm (sphere
        primitive in particular assumes |Rd|=1)."""
        R0 = np.ascontiguousarray([pos_x, pos_y, pos_z], dtype=np.float64)
        Rd = np.ascontiguousarray([dir_x, dir_y, dir_z], dtype=np.float64)
        q  = np.ascontiguousarray(self.q, dtype=np.float64)
        t = clib.tact_raycast_query(self.m._h, q.ctypes.data_as(_DBL),
                                    R0.ctypes.data_as(_DBL), Rd.ctypes.data_as(_DBL))
        return t

    def raymap(self, frame, width, height, dth, pinhole=False, perpendicular=False):
        """Depth image (height, width) from a camera frame. `frame` is a frame
        name registered in the model; the camera looks along the frame's -Z.
        `dth` = degrees per pixel (horizontal); horizontal FoV = width × dth.

        pinhole:
          False (default) — angular projection (LiDAR-like). Pixels are uniform
              in (pitch, tilt) angle; straight world lines curve in image.
          True — pinhole / rectilinear projection (standard RGB-D camera).
              Pixels uniform on the image plane; straight lines stay straight.

        perpendicular:
          False (default) — range along each ray (LiDAR convention). A flat
              plane looks bowl-shaped in angular mode.
          True — multiplies by per-pixel cos factor → camera-Z depth. Flat
              surfaces stay flat. Cos formula auto-selected per projection.
        """
        if frame not in self.m.fdict:
            raise KeyError(f"unknown frame: {frame!r}")
        D = np.empty(height * width, dtype=np.float64)
        q = np.ascontiguousarray(self.q, dtype=np.float64)
        clib.tact_raymap_query(self.m._h, q.ctypes.data_as(_DBL),
                               ctypes.c_int(self.m.fdict[frame]),
                               ctypes.c_int(width), ctypes.c_int(height),
                               ctypes.c_double(dth),
                               ctypes.c_int(1 if pinhole else 0),
                               ctypes.c_int(1 if perpendicular else 0),
                               D.ctypes.data_as(_DBL))
        return D.reshape(height, width)

    def _push_light(self):
        # Push lights[0] into render.c module statics. Cheap (no GL); called every
        # render so YAML-driven changes via env.m.lights[0][...] = ... apply immediately.
        L = self.m.lights[0]
        pos = (ctypes.c_float * 3)(*L['pos'])
        tgt = (ctypes.c_float * 3)(*L['target'])
        clib.render_set_light(pos, tgt, ctypes.c_float(L['ortho']),
                              ctypes.c_int(1 if L['shadow'] else 0))

    def _win_render(self):
        n_padding = 8
        shape = [x for row in self.m.cshape for x in (row + [0]*n_padding)[:n_padding]]
        _shape = (ctypes.c_float*len(shape))(*shape)
        _type = (ctypes.c_int*len(self.m.ctype))(*self.m.ctype)
        _objcolor = (ctypes.c_float*len(self.m.crgb))(*self.m.crgb)

        campose = self.m.view
        _campose = (ctypes.c_float*len(campose))(*campose)

        T = _fk(self.m.Ti, self.m.parent, self.m.jtype, self.q)
        objpose = np.array([])
        for i in range(len(self.m.ctype)):
            if self.m.cbody[i] < 0: tmp = self.m.ctran[i]
            else: tmp = T[self.m.cbody[i]] @ self.m.ctran[i]
            objpose = np.concatenate((objpose, tmp.T.flatten()))
        _objpose = (ctypes.c_float*len(objpose))(*objpose)

        self._push_light()
        ret = clib.win_render(len(_type), _type, _shape, _objcolor,_objpose, _campose)
        return ret
    
    def get_rgb_image(self, frame):
        if frame not in self.m.fdict: return None
        n_padding = 8
        
        shape = [x for row in self.m.cshape for x in (row + [0]*n_padding)[:n_padding]]
        _shape = (ctypes.c_float*len(shape))(*shape)        
        _type = (ctypes.c_int*len(self.m.ctype))(*self.m.ctype)
        _objcolor = (ctypes.c_float*len(self.m.crgb))(*self.m.crgb)

        tmp = self.m.fkh([frame], self.q)[0]
        tmp = tmp @ xyzeuler_to_homogeneous([0, 0, 0, 0, 0, -np.pi/2])
        campose = np.linalg.inv(tmp).T.flatten()            
        _campose = (ctypes.c_float*len(campose))(*campose)
        
        T = _fk(self.m.Ti, self.m.parent, self.m.jtype, self.q)
        objpose = np.array([])
        for i in range(len(self.m.ctype)):
            if self.m.cbody[i] < 0: tmp = self.m.ctran[i]
            else: tmp = T[self.m.cbody[i]] @ self.m.ctran[i]
            objpose = np.concatenate((objpose, tmp.T.flatten()))
        _objpose = (ctypes.c_float*len(objpose))(*objpose)

        self._push_light()
        imglen = clib.egl_render(len(_type), _type, _shape, _objcolor, _objpose, _campose, self._imgbuf, 1)
        out = ctypes.string_at(self._imgbuf, imglen)
        return out

    
class CEnv:
    """Thin adapter that gives a ctypes.CDLL (bin/mjenv.so / chenv.so / eio.so)
    the same .step/.reset/.finish interface as tact.Env, so callers (the start
    script, RL envs) can stay backend-agnostic. Other C functions (e.g. unlock,
    lock) are forwarded transparently via __getattr__.

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
        self._imgbuf = None
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
        # Optional ground-height query — exported by MuJoCo (mjenv.cpp) and Chrono backends
        # that simulate a static terrain. Real-HW backend doesn't. Probed once here so
        # controllers can call `env.get_z(x, y)` directly without per-controller setup.
        try:
            self.cdll.get_z.restype  = ctypes.c_double
            self.cdll.get_z.argtypes = [ctypes.c_double, ctypes.c_double]
        except AttributeError:
            pass

    def step(self, tau=None, q_ref=None, qd_ref=None):
        # All three input channels are equal-priority and independently optional.
        # tau=None → zero feedforward buffer (C backends currently dereference tau
        # unconditionally, so we always pass a real pointer of length n_u).
        # q_ref/qd_ref=None → NULL pointer = backend's internal PD inactive.
        if tau is None:
            _tau = (ctypes.c_double*self.n_u)()
        else:
            _tau = (ctypes.c_double*len(tau))(*tau)
        _qr  = (ctypes.c_double*len(q_ref))(*q_ref)    if q_ref  is not None else None
        _qdr = (ctypes.c_double*len(qd_ref))(*qd_ref)  if qd_ref is not None else None
        ret = self.cdll.step(_tau, _qr, _qdr, self._y)
        if ret < 0: print('ESC pressed. exit...'); sys.exit()
        return np.frombuffer(self._y, dtype=np.float64).copy()

    def get_rgb_image(self, name):
        if self._imgbuf is None:
            self.cdll.get_rgb_image.restype = ctypes.c_int
            self.cdll.get_rgb_image.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte)]
            self._imgbuf = (ctypes.c_ubyte * (1024*768*4))()
        imglen = self.cdll.get_rgb_image(name.encode(), self._imgbuf)
        if imglen <= 0: return None
        return ctypes.string_at(self._imgbuf, imglen)

    def reset(self):
        """Reset backend state and return the initial observation y. C-side reset()
        fills self._y directly (no extra step), so the returned y is the true
        post-reset reading rather than the result of one zero-input step."""
        self.cdll.reset(self._y)
        return np.frombuffer(self._y, dtype=np.float64).copy()
    def finish(self): self.cdll.finish()

    # Block topology-editing API at the CEnv boundary. Without these, __getattr__
    # would forward to self.cdll and either AttributeError (mjenv/chenv/eio don't
    # export `add`/`delete` symbols today) or — worst case — silently call an
    # unrelated C symbol if a future backend happens to export the same name.
    def add(self, *args, **kwargs):
        raise NotImplementedError(
            f"add() is tact-backend only; current backend={self.backend!r} "
            f"does not support dynamic topology changes.")
    def delete(self, *args, **kwargs):
        raise NotImplementedError(
            f"delete() is tact-backend only; current backend={self.backend!r} "
            f"does not support dynamic topology changes.")
    @property
    def groups(self):
        raise NotImplementedError(
            f"groups is tact-backend only; current backend={self.backend!r} "
            f"has no group ledger.")

    def __getattr__(self, name): return getattr(self.cdll, name)
