# pytact

Install:

```bash
pip install pytact
```

Use:

```python
import tact
```

`pytact` is the Python distribution package for **tact: a high contact/tactile fidelity dynamics simulator**.

The naming is intentional:

| Name | Role |
| --- | --- |
| `tact` | Core simulator identity and Python import namespace |
| `libtact.so` | Native shared library used by the Python package |
| `pytact` | PyPI distribution name installed with `pip install pytact` |

Today, the Python package wraps and ships the package-local `libtact.so`. The long-term direction is for `tact` to also stand as a native C library with a stable C API, while `pytact` remains the Python wrapper and distribution package.
