"""Development shim for the monorepo layout.

The installable Python package lives in ``tact/tact``. When Python is launched
from the repository root, the outer ``fg/tact`` tooling directory would otherwise
be seen first as a namespace package named ``tact``. Point import resolution at
the nested package and execute its real ``__init__``.
"""
import os as _os

_pkg = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'tact')
__path__ = [_pkg]
__file__ = _os.path.join(_pkg, '__init__.py')

with open(__file__, 'rb') as _f:
    exec(compile(_f.read(), __file__, 'exec'), globals())

for _n in ('_f', '_os', '_pkg', '_n'):
    globals().pop(_n, None)
