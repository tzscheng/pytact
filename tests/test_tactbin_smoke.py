"""Smoke test for the Python YAML -> tactbin -> C loader path.

Run:
  uv run python tests/test_tactbin_smoke.py
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TACT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TACT)
sys.path.insert(0, ROOT)

import tact
from tact.compile_model import compile_model


def run(args, cwd=TACT):
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


def parse_vector(stdout: str, label: str) -> np.ndarray:
    prefix = label + ":"
    for line in stdout.splitlines():
        if line.startswith(prefix):
            text = line[len(prefix):].strip()
            if not text:
                return np.zeros(0, dtype=np.float64)
            return np.fromstring(text, sep=" ", dtype=np.float64)
    raise ValueError(f"missing {prefix!r} in output:\n{stdout}")


def _step_python(model: tact.Model, *, steps: int, pd: bool):
    q = model.q0.copy()
    qd = model.qd0.copy()
    ctx = None
    y = np.zeros(int(getattr(model, "_y_size", 0)), dtype=np.float64)
    tau = np.zeros_like(model.q0)
    if pd:
        q_ref = np.zeros_like(model.q0)
        qd_ref = np.zeros_like(model.qd0)
        kp = np.full_like(model.q0, 10.0)
        kd = np.full_like(model.qd0, 0.1)
    for _ in range(steps):
        if pd:
            q, qd, y, ctx = model.step(q, qd, tau,
                                        q_ref=q_ref, qd_ref=qd_ref,
                                        kp=kp, kd=kd, ctx=ctx)
        else:
            q, qd, y, ctx = model.step(q, qd, tau, ctx=ctx)
    return q, qd, y


def check_parity(exe: str, tactbin: str, model: tact.Model, *, pd: bool,
                 steps: int = 1, expected: str | None = None,
                 atol: float = 1e-12, check_y: bool = False) -> int:
    args = [exe, tactbin, "--steps", str(steps)] + (["--pd"] if pd else [])
    ctest = run(args)
    print(ctest.stdout, end="")
    if ctest.returncode != 0:
        return ctest.returncode
    if expected is not None and expected not in ctest.stdout:
        print("unexpected load_tactbin output")
        return 1

    q_py, qd_py, y_py = _step_python(model, steps=steps, pd=pd)
    q_c = parse_vector(ctest.stdout, "q")
    qd_c = parse_vector(ctest.stdout, "qd")
    y_c = parse_vector(ctest.stdout, "y")
    ok = (np.allclose(q_c, q_py, rtol=0.0, atol=atol) and
          np.allclose(qd_c, qd_py, rtol=0.0, atol=atol))
    if check_y:
        ok = ok and np.allclose(y_c, y_py, rtol=0.0, atol=atol)
    if not ok:
        label = "PD" if pd else "torque"
        print(f"C/Python {label} parity failed ({steps} steps)")
        print("q_c ", q_c)
        print("q_py", q_py)
        print("qd_c ", qd_c)
        print("qd_py", qd_py)
        if check_y:
            print("y_c ", y_c)
            print("y_py", y_py)
        print("max|dq| ", np.max(np.abs(q_c - q_py)) if len(q_c) else 0.0)
        print("max|dqd|", np.max(np.abs(qd_c - qd_py)) if len(qd_c) else 0.0)
        if check_y:
            print("max|dy| ", np.max(np.abs(y_c - y_py)) if len(y_c) else 0.0)
        return 1
    return 0


def compile_case(src: str, stem: str) -> tuple[str, tact.Model]:
    out = os.path.join(tempfile.gettempdir(), f"{stem}.tactbin")
    compile_model(src, out)
    return out, tact.Model(os.path.splitext(src)[0])


def write_scene(stem: str, text: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"{stem}.yml")
    with open(path, "w") as f:
        f.write(text)
    return path


def read_tactbin(path: str):
    chunks = []
    with open(path, "rb") as f:
        header = f.read(16)
        magic, version, n_chunks = struct.unpack("<8sII", header)
        for _ in range(n_chunks):
            ch = f.read(64)
            tag, dtype, ndim, d0, d1, d2, d3, nbytes = struct.unpack("<16sII4QQ", ch)
            payload = f.read(nbytes)
            chunks.append([tag, dtype, ndim, [d0, d1, d2, d3], nbytes, payload])
    return magic, version, chunks


def write_tactbin(path: str, magic: bytes, version: int, chunks) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<8sII", magic, version, len(chunks)))
        for tag, dtype, ndim, shape, nbytes, payload in chunks:
            f.write(struct.pack("<16sII4QQ", tag, dtype, ndim,
                                shape[0], shape[1], shape[2], shape[3], nbytes))
            f.write(payload)


def chunk_name(tag: bytes) -> str:
    return tag.split(b"\0", 1)[0].decode("ascii")


def expect_load_fail(exe: str, path: str, label: str) -> int:
    r = run([exe, path])
    print(r.stdout, end="")
    if r.returncode == 0:
        print(f"malformed tactbin unexpectedly loaded: {label}")
        return 1
    if "tact_load_model failed" not in r.stdout:
        print(f"unexpected malformed tactbin failure mode: {label}")
        return 1
    return 0


def check_malformed_rejected(exe: str, valid_tactbin: str) -> int:
    magic, version, chunks = read_tactbin(valid_tactbin)

    bad_version = os.path.join(tempfile.gettempdir(), "bad_version.tactbin")
    write_tactbin(bad_version, magic, version + 1, chunks)
    rc = expect_load_fail(exe, bad_version, "bad version")
    if rc != 0:
        return rc

    missing_sim = os.path.join(tempfile.gettempdir(), "missing_sim_i32.tactbin")
    write_tactbin(missing_sim, magic, version,
                  [ch for ch in chunks if chunk_name(ch[0]) != "sim_i32"])
    rc = expect_load_fail(exe, missing_sim, "missing sim_i32")
    if rc != 0:
        return rc

    bad_shape_chunks = [[tag, dtype, ndim, shape.copy(), nbytes, payload]
                        for tag, dtype, ndim, shape, nbytes, payload in chunks]
    for ch in bad_shape_chunks:
        if chunk_name(ch[0]) == "parent":
            ch[3][0] += 1
            break
    bad_shape = os.path.join(tempfile.gettempdir(), "bad_parent_shape.tactbin")
    write_tactbin(bad_shape, magic, version, bad_shape_chunks)
    rc = expect_load_fail(exe, bad_shape, "bad parent shape")
    if rc != 0:
        return rc

    duplicate_sim = os.path.join(tempfile.gettempdir(), "duplicate_sim_i32.tactbin")
    dup_chunks = chunks + [next(ch for ch in chunks if chunk_name(ch[0]) == "sim_i32")]
    write_tactbin(duplicate_sim, magic, version, dup_chunks)
    rc = expect_load_fail(exe, duplicate_sim, "duplicate sim_i32")
    if rc != 0:
        return rc

    bad_dims_chunks = [[tag, dtype, ndim, shape.copy(), nbytes, bytearray(payload)]
                       for tag, dtype, ndim, shape, nbytes, payload in chunks]
    for ch in bad_dims_chunks:
        if chunk_name(ch[0]) == "dims_i32":
            vals = list(struct.unpack("<8i", ch[5]))
            vals[7] += 1
            ch[5] = struct.pack("<8i", *vals)
            break
    bad_dims = os.path.join(tempfile.gettempdir(), "bad_lam_size.tactbin")
    write_tactbin(bad_dims, magic, version, bad_dims_chunks)
    rc = expect_load_fail(exe, bad_dims, "bad lam_size")
    if rc != 0:
        return rc

    truncated = os.path.join(tempfile.gettempdir(), "truncated_payload.tactbin")
    with open(valid_tactbin, "rb") as src, open(truncated, "wb") as dst:
        data = src.read()
        dst.write(data[:-7])
    return expect_load_fail(exe, truncated, "truncated payload")


def check_asset_malformed_rejected(exe: str) -> int:
    mesh_src = os.path.join(TACT, "tact/demos/obj1.yml")
    mesh_out, _model = compile_case(mesh_src, "obj1_bad_mesh_base")
    magic, version, chunks = read_tactbin(mesh_out)
    bad_mesh_chunks = [[tag, dtype, ndim, shape.copy(), nbytes, payload]
                       for tag, dtype, ndim, shape, nbytes, payload in chunks
                       if chunk_name(tag) != "mesh_paths"]
    bad_mesh = os.path.join(tempfile.gettempdir(), "missing_mesh_paths.tactbin")
    write_tactbin(bad_mesh, magic, version, bad_mesh_chunks)
    rc = expect_load_fail(exe, bad_mesh, "missing mesh_paths")
    if rc != 0:
        return rc

    hfield_src = os.path.join(TACT, "tact/envs/hf1.yml")
    hfield_out, _model = compile_case(hfield_src, "hf1_bad_hfield_base")
    magic, version, chunks = read_tactbin(hfield_out)
    bad_hfield_chunks = [[tag, dtype, ndim, shape.copy(), nbytes, bytearray(payload)]
                         for tag, dtype, ndim, shape, nbytes, payload in chunks]
    for ch in bad_hfield_chunks:
        if chunk_name(ch[0]) == "hfield_offsets":
            vals = list(struct.unpack(f"<{ch[4] // 4}i", ch[5]))
            vals[-1] += 1
            ch[5] = struct.pack(f"<{len(vals)}i", *vals)
            break
    bad_hfield = os.path.join(tempfile.gettempdir(), "bad_hfield_offsets.tactbin")
    write_tactbin(bad_hfield, magic, version, bad_hfield_chunks)
    return expect_load_fail(exe, bad_hfield, "bad hfield offsets")


def main() -> int:
    src = os.path.join(TACT, "tact/demos/arm2.yml")
    out, model = compile_case(src, "arm2_smoke")

    with open(out, "rb") as f:
        magic, version, n_chunks = struct.unpack("<8sII", f.read(16))
    if magic != b"TACTBIN\0" or version != 1 or n_chunks == 0:
        print(f"bad tactbin header: magic={magic!r} version={version} chunks={n_chunks}")
        return 1

    build = run(["make", "examples"])
    if build.returncode != 0:
        print(build.stdout)
        return build.returncode

    exe = os.path.join(TACT, "build/examples/load_tactbin")
    rc = check_malformed_rejected(exe, out)
    if rc != 0:
        return rc
    rc = check_asset_malformed_rejected(exe)
    if rc != 0:
        return rc

    rc = check_parity(exe, out, model, pd=False, expected="nb=2 nq=2",
                      check_y=True)
    if rc != 0:
        return rc
    rc = check_parity(exe, out, model, pd=True, expected="nb=2 nq=2",
                      check_y=True)
    if rc != 0:
        return rc

    friction_src = write_scene("tactbin_joint_friction", """\
sim: {solver: lcp, dt: 0.001, g: [0, 0, 0]}
materials:
  m: {normal: [20000, 50], tangent: [20000, 50, 0.8], spin: [100,1,0.02], roll: [100,1,0.005], restitution: 0.0}
bodies:
  - name: root
    shapes: [{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [-1,0,0,1]}]
  - name: slider
    joint: {type: lin, parent: root, frictionloss: 0.5, qd0: -1.0}
    inertial: {mass: 1.0, tensor: [diag, 0.01, 0.01, 0.01]}
    shapes: [{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [0.3,0.3,0.8,1]}]
""")
    limit_src = write_scene("tactbin_joint_limit", """\
sim: {solver: lcp, dt: 0.001, g: [0, 0, 0]}
materials:
  m: {normal: [20000, 50], tangent: [20000, 50, 0.8], spin: [100,1,0.02], roll: [100,1,0.005], restitution: 0.0}
bodies:
  - name: root
    shapes: [{type: box, param: [0.1,0.1,0.1], contact: [-1, m], rgba: [-1,0,0,1]}]
  - name: link
    joint: {type: rev, parent: root, limit: [-30, 30], q0: 29, qd0: 100}
    inertial: {mass: 1.0, tensor: [diag, 0.01, 0.01, 0.01], pos: [0,0,0]}
    shapes: [{type: box, param: [0.2,0.05,0.05], contact: [-1, m], rgba: [0.3,0.3,0.8,1]}]
""")
    parity_cases = [
        # Free joint, no contact yet: catches nq=6 state stepping and gravity.
        ("sphere_freefall", "tact/demos/sphere_test.yml", 1, "nb=1 nq=6 n_shape=2 n_pair=1"),
        # Same scene after impact: exercises contact solve + warm-start threading.
        ("sphere_contact", "tact/demos/sphere_test.yml", 400, "nb=1 nq=6 n_shape=2 n_pair=1"),
        # Multi free-body, multi-point box contact manifold.
        ("mini_wall_contact", "tact/demos/mini_wall_box.yml", 25, "nb=3 nq=18 n_shape=4 n_pair=6"),
        # Mesh contact over multiple steps; also exercises mesh path registration.
        ("obj1_mesh_contact", "tact/demos/obj1.yml", 400, "nb=1 nq=6 n_shape=4 n_pair=4"),
    ]
    for stem, rel, steps, expected in parity_cases:
        tactbin, case_model = compile_case(os.path.join(TACT, rel), stem)
        rc = check_parity(exe, tactbin, case_model, pd=False,
                          steps=steps, expected=expected)
        if rc != 0:
            return rc

    explicit_cases = [
        ("joint_friction", friction_src, 100, "nb=1 nq=1 n_shape=2 n_pair=0"),
        ("joint_limit", limit_src, 100, "nb=1 nq=1 n_shape=2 n_pair=0"),
    ]
    for stem, src_path, steps, expected in explicit_cases:
        tactbin, case_model = compile_case(src_path, stem)
        rc = check_parity(exe, tactbin, case_model, pd=False,
                          steps=steps, expected=expected)
        if rc != 0:
            return rc

    mesh_out = os.path.join(tempfile.gettempdir(), "obj1_mesh_smoke.tactbin")
    mesh_src = os.path.join(TACT, "tact/demos/obj1.yml")
    compile_model(mesh_src, mesh_out)
    mesh_run = run([exe, mesh_out])
    print(mesh_run.stdout, end="")
    if mesh_run.returncode != 0:
        return mesh_run.returncode
    if "n_shape=4" not in mesh_run.stdout:
        print("unexpected mesh load_tactbin output")
        return 1

    hfield_out = os.path.join(tempfile.gettempdir(), "hf1_hfield_smoke.tactbin")
    hfield_src = os.path.join(TACT, "tact/envs/hf1.yml")
    compile_model(hfield_src, hfield_out)
    hfield_run = run([exe, hfield_out])
    print(hfield_run.stdout, end="")
    if hfield_run.returncode != 0:
        return hfield_run.returncode
    if "n_shape=1" not in hfield_run.stdout:
        print("unexpected hfield load_tactbin output")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
