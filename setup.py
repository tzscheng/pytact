import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


ROOT = Path(__file__).resolve().parent
PKG = ROOT / "tact"
NATIVE = ROOT / "native"
LIB = PKG / "bin" / "libtact.so"

TACT_SRC = [
    "rbd.c",
    "shape.c",
    "mpr.c",
    "narrow.c",
    "box_box.c",
    "ray.c",
    "lcp.c",
    "tact.c",
    "model.c",
    "render.c",
]


def build_libtact():
    """Build the package-local libtact.so.

    This intentionally excludes extras/mjenv.cpp. The MuJoCo backend is an
    internal cross-check/dev path and is not part of the installable package.
    """
    LIB.parent.mkdir(parents=True, exist_ok=True)
    cflags = os.environ.get("TACT_CFLAGS", "-W -Wall -O3 -ffast-math -funroll-loops")
    cmd = ["gcc", *cflags.split(), "-shared", "-fPIC", "-o", str(LIB)]
    cmd += [str(NATIVE / src) for src in TACT_SRC]
    cmd += ["-lm", "-ldl"]
    subprocess.check_call(cmd, cwd=ROOT)


class build_py(_build_py):
    def run(self):
        build_libtact()
        super().run()
        out = Path(self.build_lib) / "tact" / "bin" / "libtact.so"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LIB, out)


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel})
