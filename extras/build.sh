#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${CXX:=g++}"
: "${MJENV_CXXFLAGS:=-shared -fPIC -I/usr/local/include/mujoco}"
: "${MJENV_LDLIBS:=-lmujoco -lGL -lglfw}"

"$CXX" $MJENV_CXXFLAGS -o mjenv.so mjenv.cpp $MJENV_LDLIBS
