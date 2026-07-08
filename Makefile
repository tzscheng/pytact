PREFIX ?= /usr/local
DESTDIR ?=

CC ?= gcc
CXX ?= g++
AR ?= ar
INSTALL ?= install

LIB_DIR ?= native/lib

# no -D_GNU_SOURCE: on glibc >=2.38 it redirects sscanf/strtol to __isoc23_*,
# raising the wheel's glibc floor to 2.38 (manylinux_2_38); without it the
# floor is 2.35 (libmvec atan2 from -ffast-math) -> manylinux_2_35 wheels
CFLAGS ?= -W -Wall -O3 -fPIC -ffast-math -funroll-loops
LDFLAGS ?=
LDLIBS ?= -lm -ldl

TACT_SRC := \
	native/rbd.c \
	native/shape.c \
	native/mpr.c \
	native/narrow.c \
	native/box_box.c \
	native/ray.c \
	native/lcp.c \
	native/tact.c \
	native/model.c \
	native/render.c

TACT_HEADERS := native/tact.h native/shape.h native/model.h
TACT_LIB := $(LIB_DIR)/libtact.so
BIN_TEST := native/demos/basic/bin-test

.PHONY: all clean install uninstall package-lib demos debug

all: package-lib demos

$(TACT_LIB): $(TACT_SRC) $(TACT_HEADERS) | $(LIB_DIR)
	$(CC) $(CFLAGS) -shared -o $@ $(TACT_SRC) $(LDFLAGS) $(LDLIBS)

package-lib: $(TACT_LIB)
	mkdir -p tact/bin
	cp $(TACT_LIB) tact/bin/libtact.so

demos: $(BIN_TEST)

$(BIN_TEST): native/demos/basic/bin_test.c $(TACT_LIB) native/tact.h
	$(CC) -W -Wall -O2 -Inative -o $@ native/demos/basic/bin_test.c -L$(LIB_DIR) -ltact -Wl,-rpath,'$$ORIGIN/../../lib'

debug:
	$(MAKE) -B CFLAGS="-W -Wall -O0 -g -fPIC" $(TACT_LIB) demos

$(LIB_DIR):
	mkdir -p $@

install: $(TACT_LIB)
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/include
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/lib
	$(INSTALL) -m 0644 native/tact.h $(DESTDIR)$(PREFIX)/include/tact.h
	$(INSTALL) -m 0755 $(TACT_LIB) $(DESTDIR)$(PREFIX)/lib/libtact.so

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/include/tact.h
	rm -f $(DESTDIR)$(PREFIX)/lib/libtact.so

clean:
	rm -rf $(LIB_DIR) $(BIN_TEST)
