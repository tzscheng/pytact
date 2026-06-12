#!/bin/bash
# Build artifacts:
#   tact/bin/libtact.so — dynamics + collision + contact solvers + render (loaded by _clib.py)
#   mjenv/mjenv.so    — auxiliary MuJoCo backend (dlopen'd by `start -m`; dev/internal only)
#
# Pass `debug` as the first arg for a no-opt build (asserts useful, gdb-friendly):
#   ./build.sh debug
# Otherwise an -O3 -march=native release build is produced.

set -e

if [ "$1" = "debug" ]; then
    CFLAGS="-W -Wall -O0 -g"
else
    CFLAGS="-W -Wall -O3 -march=native -ffast-math -funroll-loops"
fi

mkdir -p tact/bin mjenv

#---- tact/bin/libtact.so ------------------------------------------------------
# All C/C++ sources + headers live in native/ (headers sit beside the sources, so
# `#include "tact.h"` resolves with no -I). Build artifacts go directly to the
# package-local tact/bin/ directory.
# native/rbd.c    : linear algebra, spatial dynamics (ABA / CRB / RNE / LDL^T) + choose_rotation
# native/shape.c  : shared shape-asset slot storage (mesh .obj load + hfield grids) — see shape.h
# native/mpr.c    : generic convex narrowphase (MPR via libccd) + ccd_support + collision_check_mpr
# native/narrow.c : contact dispatch (collision_check) + analytic detectors (box-sphere,
#                   sphere/box-hfield)
# native/box_box.c: box-box manifold (SAT + Sutherland-Hodgman face clipping)
# native/ray.c    : ray-primitive intersections (triangle/mesh/hfield/box/sphere/cyl/capsule)
# native/lcp.c    : contact_lcp       — Stewart-Trinkle LCP with PGS, 4 cones
# native/tact.c   : tact_t handle + tact_step_lcp + queries
# native/tact_model.c : tactbin loader + public tact_model_t/tact_state_t wrapper
# native/render.c : GLFW window + EGL offscreen renderer; GL/GLFW/jpeg/zstd are
#                   loaded lazily with dlopen so headless physics can import on
#                   systems without render runtime libraries.
TACT_SRC="native/rbd.c native/shape.c native/mpr.c native/narrow.c native/box_box.c native/ray.c native/lcp.c native/tact.c native/tact_model.c native/render.c"
TACT_LIBS="-lm -ldl"

gcc $CFLAGS -shared -fPIC -o tact/bin/libtact.so $TACT_SRC $TACT_LIBS
# (install into a system path is no longer needed — tact/bin/libtact.so is loaded via package-relative resolution; see tact/_clib.py)

#---- mjenv/mjenv.so -----------------------------------------------------------
# Built separately because of the C++ source + MuJoCo include path.
g++ -shared -fPIC -o mjenv/mjenv.so mjenv/mjenv.cpp -I/usr/local/include/mujoco -lmujoco -lGL -lglfw
# (mjenv is internal/dev-only and intentionally lives outside the installable package)

echo "built: tact/bin/libtact.so  mjenv/mjenv.so  ($([ "$1" = "debug" ] && echo debug || echo release))"
