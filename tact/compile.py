"""Compile a tact YAML model into a C-readable bin file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from .sim import Model


MAGIC = b"TACTMDL\0"
VERSION = 1

DTYPE_I32 = 1
DTYPE_F64 = 2
DTYPE_UTF8 = 3


def _as_i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def _as_f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _chunk_array(tag: str, dtype: int, arr: np.ndarray) -> tuple[bytes, bytes]:
    if len(tag.encode("ascii")) > 15:
        raise ValueError(f"chunk tag too long: {tag!r}")
    if arr.ndim > 4:
        raise ValueError(f"chunk {tag!r}: ndim {arr.ndim} exceeds bin limit")
    shape = list(arr.shape) + [0] * (4 - arr.ndim)
    payload = arr.tobytes(order="C")
    header = struct.pack(
        "<16sII4QQ",
        tag.encode("ascii"),
        dtype,
        arr.ndim,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        len(payload),
    )
    return header, payload


def _chunk_text(tag: str, text: str) -> tuple[bytes, bytes]:
    payload = text.encode("utf-8")
    header = struct.pack(
        "<16sII4QQ",
        tag.encode("ascii"),
        DTYPE_UTF8,
        1,
        len(payload),
        0,
        0,
        0,
        len(payload),
    )
    return header, payload


def _model_from_yaml(path: Path) -> Model:
    prev = Path.cwd()
    try:
        os.chdir(path.parent)
        return Model(path.stem)
    finally:
        os.chdir(prev)


def compile(src: str | os.PathLike[str], out: str | os.PathLike[str]) -> None:
    src_path = Path(src).expanduser().resolve()
    out_path = Path(out).expanduser().resolve()
    model = _model_from_yaml(src_path)

    nb = len(model.jtype)
    nq = len(model.q0)
    n_shape = len(model.ctype)
    n_pair = int(model.cpair.shape[0])
    n_frame = len(model.fbody)
    n_feed = len(model.feeds)
    y_size = int(getattr(model, "_y_size", 0))
    lam_size = int(6 * 4 * max(n_pair, 1) + 2 * nq)

    chunks: list[tuple[bytes, bytes]] = []

    def add_i32(tag: str, arr) -> None:
        chunks.append(_chunk_array(tag, DTYPE_I32, _as_i32(arr)))

    def add_f64(tag: str, arr) -> None:
        chunks.append(_chunk_array(tag, DTYPE_F64, _as_f64(arr)))

    add_i32("dims_i32", [nb, nq, n_shape, n_pair, n_frame, n_feed, y_size, lam_size])
    add_f64("sim_f64", [model.dt, model.erp, model.slop, model.cfm_scale, model.v_rest_thresh, model.tol])
    add_i32("sim_i32", [2, model.iters])

    add_i32("parent", model._build_parent)
    add_i32("jtype", model._build_jtype)
    add_f64("X", model._build_X.reshape(nb, 6, 6))
    add_f64("I6", model._build_I6.reshape(nb, 6, 6))
    add_f64("Ti", model._build_Ti.reshape(nb, 4, 4))
    add_f64("ff", model._build_ff)
    add_f64("sk", model._build_sk)
    add_f64("floss", model._build_floss)
    add_f64("armature", model._build_armature)
    add_f64("jnt_lo", model._build_jnt_lo)
    add_f64("jnt_hi", model._build_jnt_hi)
    add_f64("g", model._build_g)

    add_i32("ctype", model._build_ctype)
    add_i32("cbody", model._build_cbody)
    add_f64("cshape", model._build_cshape.reshape(n_shape, 3))
    add_f64("ctran", model._build_ctran.reshape(n_shape, 4, 4))
    add_f64("cparam", model._build_cparam.reshape(n_shape, 13))
    add_f64("crgba", np.asarray(model.crgba, dtype=np.float64).reshape(n_shape, 4))
    add_i32("craycast", model._build_craycast)
    add_i32("cpair", model._build_cpair.reshape(n_pair, 2))

    add_f64("q0", model.q0)
    add_f64("qd0", model.qd0)
    add_f64("view", model.view)
    light0 = model.lights[0] if model.lights else {
        "pos": [7.0, 7.0, 7.0],
        "target": [0.0, 0.0, 0.0],
        "ortho": 5.0,
        "shadow": True,
    }
    add_f64("light0", [*light0["pos"], *light0["target"],
                       light0["ortho"], 1.0 if light0.get("shadow", True) else 0.0])

    add_i32("feed_kinds", model._build_feed_kinds[:n_feed])
    add_i32("feed_offsets", model._build_feed_offsets)
    n_feed_idx = int(model._build_feed_offsets[-1]) if len(model._build_feed_offsets) else 0
    add_i32("feed_idx", model._build_feed_idx[:n_feed_idx])
    add_i32("fbody", model._build_fbody[:n_frame])
    add_f64("ftran", model._build_ftran[:n_frame].reshape(n_frame, 4, 4))
    add_f64("ftran_inv", model._build_ftran_inv[:n_frame].reshape(n_frame, 4, 4))
    frame_names = [""] * n_frame
    for name, idx in model.fdict.items():
        if 0 <= idx < n_frame:
            frame_names[idx] = name
    chunks.append(_chunk_text("frame_names", "\0".join(frame_names)))

    meta = {
        "format": "bin",
        "version": VERSION,
        "source": str(src_path),
        "fdict": model.fdict,
        "groups": [g["name"] for g in model.groups],
        "cameras": model.cameras,
        "lidars": model.lidars,
        "note": "meta_json is optional; core C loading uses binary chunks.",
    }
    chunks.append(_chunk_text("meta_json", json.dumps(meta, sort_keys=True, separators=(",", ":"))))

    mesh_items = sorted(model.mesh_path_to_idx.items(), key=lambda kv: kv[1])
    add_i32("mesh_slots_i32", [idx for _path, idx in mesh_items])
    chunks.append(_chunk_text("mesh_paths", "\0".join(path for path, _idx in mesh_items)))

    hfields = sorted(model.hfield_data, key=lambda item: item["slot"])
    hfield_offsets = [0]
    hfield_values = []
    for hf in hfields:
        data = _as_f64(hf["data"]).ravel()
        hfield_values.append(data)
        hfield_offsets.append(hfield_offsets[-1] + int(data.size))
    hfield_meta = np.array([[hf["slot"], hf["nrow"], hf["ncol"]] for hf in hfields],
                           dtype=np.int32).reshape(len(hfields), 3)
    hfield_size = np.array([[hf["sx"], hf["sy"]] for hf in hfields],
                           dtype=np.float64).reshape(len(hfields), 2)
    add_i32("hfield_meta_i32", hfield_meta)
    add_f64("hfield_size_f64", hfield_size)
    add_i32("hfield_offsets", hfield_offsets)
    add_f64("hfield_data_f64", np.concatenate(hfield_values) if hfield_values else [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<8sII", MAGIC, VERSION, len(chunks)))
        for header, payload in chunks:
            f.write(header)
            f.write(payload)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile tact YAML to bin.")
    parser.add_argument("src", help="source YAML model")
    parser.add_argument("-o", "--out", required=True, help="output .bin path")
    args = parser.parse_args(argv)
    compile(args.src, args.out)


if __name__ == "__main__":
    main()
