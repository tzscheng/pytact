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
from .wbc import WBC, BodyTask, SwingTask, PostureTask, CoMTask
from .mpc import ConvexMPC
