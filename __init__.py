"""tact: robotics simulation + control toolkit.

Splits into two domains, but exposes a flat API:
    import tact
    tact.Env(...)              # from .sim
    tact.PIDController(...)    # from .control

For domain-explicit access:
    tact.sim.Env(...)
    tact.control.PIDController(...)

Package layout (sibling assets):
    tact.pkg_dir                         → '/.../fg/tact'  (the package itself)
    tact.pkg_dir/bin/libtact.so, bin/mjenv.so → native libraries (output of build.sh)
    tact.pkg_dir/examples/               → scene YAMLs (1.yml–5.yml, box1, desk1),
                                            sample model YAMLs (arm2, arm3, arm4, fv,
                                            obj1, obj2, sphere_test), example projects
                                            (cartpole/, rb5/, rb10/), and mesh assets
                                            (objs/). Loaded by start's -e flag and by
                                            per-project RL envs via `tact.pkg_dir`.
"""
import os as _os
pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
del _os
from .rbd import *
from .sim import *
from .control import *

# wbc/mpc import osqp+scipy at module top; load them lazily (PEP 562) so plain
# `import tact` works in slim envs (e.g. RL venvs without osqp) that only need
# rbd/sim/control. tact.WBC etc. resolve on first attribute access as before.
_LAZY = {'WBC': '.wbc', 'BodyTask': '.wbc', 'SwingTask': '.wbc',
         'PostureTask': '.wbc', 'CoMTask': '.wbc', 'ConvexMPC': '.mpc'}

def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        val = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = val      # cache — later accesses bypass __getattr__
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
