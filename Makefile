PREFIX ?= /usr/local
DESTDIR ?=

CC ?= gcc
CXX ?= g++
AR ?= ar
INSTALL ?= install

BUILD_DIR ?= build/native
LIB_DIR ?= build/lib
PKG_LIB_DIR ?= tact/bin

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
PKG_TACT_LIB := $(PKG_LIB_DIR)/libtact.so
BIN_TEST := native/demos/basic/bin-test

.PHONY: all clean install uninstall package-lib demos debug

all: package-lib demos

$(TACT_LIB): $(TACT_SRC) $(TACT_HEADERS) | $(LIB_DIR)
	$(CC) $(CFLAGS) -shared -o $@ $(TACT_SRC) $(LDFLAGS) $(LDLIBS)

package-lib: $(PKG_TACT_LIB)

$(PKG_TACT_LIB): $(TACT_LIB) | $(PKG_LIB_DIR)
	cp $< $@

demos: $(BIN_TEST)

$(BIN_TEST): native/demos/basic/bin_test.c $(TACT_LIB) native/tact.h
	$(CC) -W -Wall -O2 -Inative -o $@ native/demos/basic/bin_test.c -L$(LIB_DIR) -ltact -Wl,-rpath,'$$ORIGIN/../../../build/lib'

debug:
	$(MAKE) -B CFLAGS="-W -Wall -O0 -g -fPIC" package-lib demos

$(LIB_DIR) $(PKG_LIB_DIR):
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
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_TEST)
