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
    tact.pkg_dir                         → '/.../pytact/tact'
                                            (the import package itself)
    tact.pkg_dir/bin/libtact.so          → installable native library (output of make package-lib)
    (MuJoCo backend mjenv.so + mjcf/ scenes + env YAMLs are NOT part of this repo:
     they live in the consumer repo and are loaded by the consumer launcher.)
    ../demos/                             → checkout-only examples for GitHub users;
                                            not included in the Python package.
"""
import os as _os
pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
del _os
from .rbd import *
from .sim import *
from .control import *
