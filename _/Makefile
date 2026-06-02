# Tact build — incremental, header-dep aware.
#
# Outputs:
#   libtact.so   dynamics + collision + contact solvers + render (used by tact.py)
#   mujoco.io    auxiliary MuJoCo backend (dlopen'd by `start -m`)
#
# Usage:
#   make                  # release (-O3 -march=native -ffast-math)
#   make BUILD=debug      # debug   (-O0 -g, asserts/gdb-friendly)
#   make libtact.so       # only the dynamics shared lib
#   make mujoco.io        # only the MuJoCo backend
#   make clean            # remove build/ and all artifacts
#   make install          # cp artifacts to /opt/fg (uses sudo)
#
# Incremental: each .c → build/<mode>/<file>.o, with auto-generated header deps
# (-MMD -MP) so edits to tact.h trigger rebuilds of all dependent objects.
# Switching BUILD modes uses a separate object dir, so no manual `make clean`
# is needed when toggling release ↔ debug.

CC      ?= gcc
CXX     ?= g++

BUILD   ?= release
ifeq ($(BUILD),debug)
    OPTFLAGS := -O0 -g
else ifeq ($(BUILD),release)
    OPTFLAGS := -O3 -march=native -ffast-math -funroll-loops
else
    $(error BUILD must be 'release' or 'debug', got '$(BUILD)')
endif

OBJDIR  := build/$(BUILD)
CFLAGS  := -W -Wall -fPIC $(OPTFLAGS) -MMD -MP

#---- libtact.so ---------------------------------------------------------------
# rbd.c     linear algebra, spatial dynamics (ABA / CRB / RNE / LDL^T)
# ccd.c     narrow-phase collision (GJK/EPA via libccd) + choose_rotation
# penalty.c contact_penalty — spring-damper + brush friction
# lcp.c     contact_lcp     — Stewart-Trinkle LCP with PGS, 4 cones
# tact.c    tact_t handle + tact_step_penalty / tact_step_lcp + queries
# render.c  GLFW window + EGL offscreen renderer
TACT_SRC  := rbd.c ccd.c penalty.c lcp.c tact.c render.c
TACT_OBJ  := $(TACT_SRC:%.c=$(OBJDIR)/%.o)
TACT_LIBS := -lm -lEGL -lGL -lGLEW -lGLU -lglfw -lturbojpeg -lzstd

# Real artifact lives under the per-mode build dir; the top-level libtact.so is
# refreshed from it on every make so switching BUILD modes (release ↔ debug)
# always leaves the active lib matching the requested mode.
$(OBJDIR)/libtact.so: $(TACT_OBJ)
	$(CC) -shared -o $@ $^ $(TACT_LIBS)

.PHONY: libtact.so
libtact.so: $(OBJDIR)/libtact.so
	@cp $< $@

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR):
	@mkdir -p $@

#---- mujoco.io ----------------------------------------------------------------
# Single C++ source; no incremental win, so built directly.
mujoco.io: mjcore.cpp
	$(CXX) -shared -fPIC -I/usr/local/include/mujoco -o $@ $< \
	    -lmujoco -lGL -lglfw -lturbojpeg

#---- meta ---------------------------------------------------------------------
.PHONY: all clean install
.DEFAULT_GOAL := all

all: libtact.so mujoco.io

clean:
	rm -rf build libtact.so mujoco.io

install: libtact.so mujoco.io
	sudo cp libtact.so mujoco.io /opt/fg

# Auto-generated header dependencies (.d files emitted alongside each .o)
-include $(TACT_OBJ:.o=.d)
