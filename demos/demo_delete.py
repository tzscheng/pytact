#!/usr/bin/env -S uv run python
"""Live demonstration of Env.add() / Env.delete() — objects appear and
disappear on a floor in time order. Press ESC in the render window to quit.

Highlights arbitrary-order deletion (not just LIFO): we drop several
objects (including a mesh from objs/6.obj), then remove a middle-aged
one while newer objects are still present. Each remaining object's
physical state (pose + velocity) is preserved across the delete — only
the deleted body's DoF slots get spliced out of q/qd.

Run from anywhere — the script chdir's into demos/ so the C-side
mesh loader (which probes CWD-relative `objs/<idx>.obj`) finds the
bundled mesh assets:
    uv run python /home/ubuntu/uv/fg/tact/demos/demo_delete.py
"""
import os, sys, tempfile

# Make `tact` importable without installing — script lives at fg/tact/demos/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # → .../fg
# Set CWD so the C-side mesh probe finds `objs/<idx>.obj`.
os.chdir(HERE)
import numpy as np
import tact


# ── Build a tiny YAML library in a temp dir ────────────────────────────────────
TMP = tempfile.mkdtemp(prefix='tact_delete_demo_')

FLOOR_YML = """
method: lcp
dt: 0.001
g: [0, 0, -9.81]
view: [0, 0, 0, 2.5, 30, 25]
materials:
    floor: {normal: [20000, 200], tangent: [20000, 200, 1.0], spin: [200, 2, 0.05], roll: [200, 2, 0.02]}
bodies:
  - name: root
    shapes: [{type: box, pos: [0, 0, -0.05], param: [0.8, 0.8, 0.05], contact: [1, floor], rgb: [0.85, 0.85, 0.85]}]
"""

def obj_yml(shape, spec, rgb, q0):
    """`spec` is the shape parameter — for primitives it's a list literal like
    `[0.07, 0.07, 0.07]` rendered as YAML `param: ...`; for mesh it's a path
    string rendered as YAML `file: ...`."""
    if shape == 'mesh':
        shape_spec = f"file: {spec}"
    else:
        shape_spec = f"param: {spec}"
    return f"""
materials:
    obj: {{normal: [20000, 100], tangent: [10000, 100, 0.8], spin: [100, 1, 0.02], roll: [100, 1, 0.005]}}
bodies:
  - name: thing
    joint: {{type: free, parent: root, q0: {q0}}}
    inertial: {{mass: 0.3, tensor: [sphere, 0.05]}}
    shapes: [{{type: {shape}, {shape_spec}, contact: [1, obj], rgb: {rgb}}}]
"""

def write_yml(name, content):
    """Write <TMP>/<name>.yml and return the path-without-suffix so it can be
    passed directly to tact.Env() / env.add()."""
    with open(os.path.join(TMP, name + '.yml'), 'w') as f:
        f.write(content)
    return os.path.join(TMP, name)

floor_path = write_yml('floor', FLOOR_YML)

# Object pool: each entry → unique YAML. (shape, param, rgb, drop_xy)
# `mesh_teal` references demos/objs/6.obj — see obj2.yml for the pattern.
POOL = {
    'sphere_red':    ('sphere',   '[0.07]',             '[0.9, 0.3, 0.3]',  ( 0.25,  0.00)),
    'box_blue':      ('box',      '[0.07, 0.07, 0.07]', '[0.3, 0.4, 0.9]',  (-0.25,  0.00)),
    'cyl_yellow':    ('cylinder', '[0.06, 0.08]',       '[0.9, 0.9, 0.3]',  ( 0.00,  0.25)),
    'cap_magenta':   ('capsule',  '[0.05, 0.08]',       '[0.8, 0.3, 0.8]',  ( 0.00, -0.25)),
    'sphere_green':  ('sphere',   '[0.06]',             '[0.3, 0.9, 0.4]',  ( 0.22,  0.22)),
    'box_orange':    ('box',      '[0.06, 0.06, 0.06]', '[0.95, 0.6, 0.2]', (-0.22, -0.22)),
    # mesh: spec is a path. Absolute because the YAML lives in TMP (relative
    # paths resolve against TMP, where there are no objs/).
    'mesh_teal':     ('mesh',     f'{HERE}/objs/6.obj', '[0.4, 0.9, 0.9]',  (-0.20,  0.20)),
}
for key, (shape, param, rgb, (x, y)) in POOL.items():
    write_yml(key, obj_yml(shape, param, rgb, f'[{x}, {y}, 0.6, 0, 0, 0]'))


# ── Event schedule (sim seconds == real seconds at dt=1ms + VSync redraw) ─────
# Note the arbitrary-order deletes: after 4 objects are stacked, we remove
# `box_blue` (2nd-oldest), then later `sphere_red` (oldest), all while newer
# objects keep falling and settling. LIFO would not allow this.
SCHEDULE = [
    (0.5,  'add', 'sphere_red'),
    (1.2,  'add', 'box_blue'),
    (1.9,  'add', 'cyl_yellow'),
    (2.6,  'add', 'cap_magenta'),
    (3.3,  'add', 'mesh_teal'),      # ← mesh shape (objs/6.obj)
    (4.0,  'del', 'box_blue'),       # ← middle-aged removal
    (4.5,  'add', 'sphere_green'),
    (5.5,  'del', 'sphere_red'),     # ← oldest removal
    (6.0,  'add', 'box_orange'),
    (6.8,  'del', 'mesh_teal'),      # ← mesh removal
    (7.0,  'del', 'cyl_yellow'),
    (8.0,  'del', 'cap_magenta'),
    (9.0,  'del', 'sphere_green'),
    (10.0, 'del', 'box_orange'),
    (11.5, 'end', None),
]


# ── Run ────────────────────────────────────────────────────────────────────────
env = tact.Env(floor_path, render=True, redraw=16, name='floor')

print()
print(f"  {'t(s)':>5}  {'event':<24} {'nq':>3}  groups")
print( '  ' + '─'*60)

t = 0.0
for t_evt, action, arg in SCHEDULE:
    # advance sim until the next scheduled event
    while t < t_evt:
        env.step(np.zeros(env.dof))
        t += env.m.dt

    if action == 'end':
        print(f"  {t:5.2f}  {'(end)':<24} {len(env.q):3d}  {env.groups}")
        break

    if action == 'add':
        env.add(os.path.join(TMP, arg), name=arg, prefix=arg + '_')
        marker = '＋'
    elif action == 'del':
        env.delete(arg)
        marker = '－'

    print(f"  {t:5.2f}  {marker} {arg:<22} {len(env.q):3d}  {env.groups}")

# brief tail so the final empty-floor frame is visible
while t < 12.5:
    env.step(np.zeros(env.dof))
    t += env.m.dt
