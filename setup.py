import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


ROOT = Path(__file__).resolve().parent
PKG = ROOT / "tact"
LIB = PKG / "bin" / "libtact.so"


def build_libtact():
    """Build the package-local libtact.so from the Makefile recipe."""
    subprocess.check_call(["make", "package-lib"], cwd=ROOT)


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

    def get_tag(self):
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform


class editable_wheel(_editable_wheel):
    def run(self):
        build_libtact()
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    cmdclass={
        "build_py": build_py,
        "bdist_wheel": bdist_wheel,
        "editable_wheel": editable_wheel,
    },
    distclass=BinaryDistribution,
)
