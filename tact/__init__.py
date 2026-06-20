"""tact: a high contact/tactile fidelity dynamics simulator.

Distributed on PyPI as ``pytact`` and imported in Python as ``tact``.
The package wraps the native ``libtact.so`` runtime.

Splits into two domains, but exposes a flat API:
    import tact
    tact.Env(...)              # from .sim
    tact.PIDController(...)    # from .control

For domain-explicit access:
    tact.sim.Env(...)
    tact.control.PIDController(...)

Package layout (sibling assets):
    tact.pkg_dir                         → '/.../fg/pytact/tact' or '/.../fg/tact/tact'
                                            (the import package itself)
    tact.pkg_dir/bin/libtact.so          → installable native library (output of make package-lib)
    ../extras/mjenv.so                   → internal MuJoCo backend for start -m (not packaged)
    ../extras/mjcf/                      → repo-local MuJoCo environment XMLs
    ../extras/envs/                       → repo-local background scene YAMLs
                                            (1.yml–5.yml, d3, hf1, stairs, box1, desk1).
                                            Loaded by extras/start -e; not packaged.
    tact.pkg_dir/demos/                   → tact feature/physics demos grouped by topic:
                                            basic/, box-wall/, cartpole/, raymap/,
                                            topology/.
    tact.pkg_dir/demos/basic/meshes/      → mesh assets used by basic/ demos only.
                                            Other demo folders carry their own assets.
"""
import os as _os
pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
del _os
from .rbd import *
from .sim import *
from .control import *
