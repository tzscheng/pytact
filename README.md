# tact

`tact` is a high contact/tactile fidelity dynamics simulator.

This directory contains two faces of the same simulator:

| Name | Role |
| --- | --- |
| `tact` | Core simulator identity, C library name, and Python import namespace |
| `pytact` | Python distribution package installed with `pip install pytact` |
| `libtact.so` | Native shared library used by both Python and standalone C programs |

## Python API

Install the published package:

```bash
pip install pytact
```

For editable development from this checkout, use the repository `uv`
environment instead of the system `pip`. The package requires Python 3.12 or
newer, and the root workspace at `~/fg` owns that environment:

```bash
cd ~/fg
uv pip install -e ./tact
```

The editable install calls `make package-lib` inside `tact/`. That builds the
canonical native library at `native/lib/libtact.so` and copies the same file to
the package-local `tact/bin/libtact.so`, which is what `import tact` loads.

After the editable install, Python source edits under `tact/tact/` are used on
the next run without reinstalling. After C source edits under `native/`, rebuild
the shared library:

```bash
cd ~/fg/tact
make
```

The default `make` target also refreshes `tact/bin/libtact.so`, so the Python
package and native C output stay in sync.

Verify the install:

```bash
cd ~/fg
uv run python -c "import tact; print(tact.__file__)"
```

Use:

```python
import tact

model = tact.Model("scene")
q, qd, y, ctx = model.step(model.q0, model.qd0, None)
```

The Python package owns YAML scene loading, asset resolution, high-level
simulation helpers, controllers, rendering wrappers, and packaging. It wraps the
package-local native library at `tact/bin/libtact.so`.

## C API

The standalone C path is for programs that do not want to import Python at
runtime. YAML is still the source scene format, but it is compiled once into a
binary model file:

```bash
python -m tact.compile tact/demos/basic/arm2.yml -o /tmp/arm2.bin
```

Then a C program loads the `.bin` through `libtact.so`:

```c
#include "tact.h"

tact_t *m = NULL;
tact_ctx_t *ctx = NULL, *ctx_next = NULL;
tact_load("/tmp/arm2.bin", &m);
tact_create_ctx(m, &ctx);
tact_create_ctx(m, &ctx_next);

tact_info_t info;
tact_info(m, &info);

double q[64] = {0}, qd[64] = {0}, q_next[64] = {0}, qd_next[64] = {0};
double tau[64] = {0}, q_ref[64] = {0}, qd_ref[64] = {0}, kp[64] = {0}, kd[64] = {0};
double y[256] = {0};

memcpy(q, tact_q0(m), info.nq * sizeof(double));
memcpy(qd, tact_qd0(m), info.nq * sizeof(double));

tact_step(m, q, qd, tau, q_ref, qd_ref, kp, kd, ctx, q_next, qd_next, y, ctx_next);

tact_destroy_ctx(ctx);
tact_destroy_ctx(ctx_next);
tact_destroy(m);
```

`tact_step` always accepts the implicit-PD control slots. Set gains to zero, or
pass `NULL`, for torque-only stepping. Use `tact_ctx_t` to thread contact solver
warm-start state between steps.

Build and run the minimal C demo from the repository checkout:

```bash
cd tact
make
python -m tact.compile tact/demos/basic/arm2.yml -o /tmp/arm2.bin
native/demos/basic/bin-test /tmp/arm2.bin --headless
```

For the full binary model format and C API contract, see
`docs/design-bin.md`.

## Source Layout

```text
native/              C engine sources and public header tact.h
native/lib/          native C shared library output from make
native/demos/basic/  minimal standalone C demo using compiled .bin models
tact/                Python package and packaged demos
docs/                design notes and API contracts
extras/              repo-local tools/assets not included in the Python package
tests/               regression and packaging smoke tests
```

From a checkout, `make` builds the native C library at `native/lib/libtact.so`
and copies that same library to the package-local `tact/bin/libtact.so` used by
Python, then builds the native demo. `make package-lib` performs only the native
library build plus package-local copy, without building demos. Python packaging
uses that same `make package-lib` path, so editable installs and normal local
builds share one C build recipe.
