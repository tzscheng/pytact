PREFIX ?= /usr/local
DESTDIR ?=

CC ?= gcc
AR ?= ar
INSTALL ?= install

BUILD_DIR ?= build/native
LIB_DIR ?= build/lib
PKG_LIB_DIR ?= tact/bin
EXAMPLE_DIR ?= build/examples

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
	native/tact_model.c \
	native/render.c

TACT_HEADERS := native/tact.h native/shape.h
TACT_LIB := $(LIB_DIR)/libtact.so
PKG_TACT_LIB := $(PKG_LIB_DIR)/libtact.so
MANUAL_STEP := $(EXAMPLE_DIR)/manual_step
LOAD_TACTBIN := $(EXAMPLE_DIR)/load_tactbin

.PHONY: all clean install uninstall package-lib examples

all: $(TACT_LIB) package-lib

$(TACT_LIB): $(TACT_SRC) $(TACT_HEADERS) | $(LIB_DIR)
	$(CC) $(CFLAGS) -shared -o $@ $(TACT_SRC) $(LDFLAGS) $(LDLIBS)

package-lib: $(PKG_TACT_LIB)

$(PKG_TACT_LIB): $(TACT_LIB) | $(PKG_LIB_DIR)
	cp $< $@

examples: $(MANUAL_STEP) $(LOAD_TACTBIN)

$(MANUAL_STEP): examples/c/manual_step.c $(TACT_LIB) native/tact.h | $(EXAMPLE_DIR)
	$(CC) -W -Wall -O2 -Inative -o $@ examples/c/manual_step.c -L$(LIB_DIR) -ltact -Wl,-rpath,'$$ORIGIN/../lib'

$(LOAD_TACTBIN): examples/c/load_tactbin.c $(TACT_LIB) native/tact.h | $(EXAMPLE_DIR)
	$(CC) -W -Wall -O2 -Inative -o $@ examples/c/load_tactbin.c -L$(LIB_DIR) -ltact -Wl,-rpath,'$$ORIGIN/../lib'

$(LIB_DIR) $(PKG_LIB_DIR) $(EXAMPLE_DIR):
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
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(EXAMPLE_DIR)
