#!/usr/bin/env python3
"""Fix ONNX If-node rank mismatch in MOSS-Audio-Tokenizer-Nano codec ONNX for TRT 10.3+.

Symptom (trtexec):
    [E] /If_OutputLayer: IIfConditionalOutputLayer inputs must have the same shape.
        Shapes are [1,-1] and [1,1,-1].   (codec_decode_step)
        Shapes are [-1,-1] and [-1,1,-1]. (codec_decode_full)

Root cause: control-flow branches return tensors of different rank (rank-2 vs rank-3).
ORT tolerates this, TRT rejects it. Fix: insert Unsqueeze(axis=1) into the rank-2 branch
so both subgraph outputs have rank-3.

Usage:
    python fix_moss_codec_onnx_if_rank.py \
        --in-dir  /path/to/MOSS-Audio-Tokenizer-Nano-ONNX \
        --out-dir /path/to/MOSS-Audio-Tokenizer-Nano-ONNX-trtfix
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper

TARGETS = [
    "codec_decode_step.onnx",
    "codec_decode_full.onnx",
    "moss_audio_tokenizer_decode_step.onnx",
    "moss_audio_tokenizer_decode_full.onnx",
]


def rank_of_vi(vi) -> int | None:
    t = vi.type.tensor_type
    return len(t.shape.dim) if t.HasField("shape") else None


def shape_of_vi(vi) -> list[int] | None:
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    return [d.dim_value if d.HasField("dim_value") else -1 for d in t.shape.dim]


def set_shape(vi, shape: list[int]) -> None:
    tt = vi.type.tensor_type
    del tt.shape.dim[:]
    for d in shape:
        dim = tt.shape.dim.add()
        if isinstance(d, int) and d >= 0:
            dim.dim_value = d
        else:
            dim.dim_param = "unk"


def patch_graph(g) -> int:
    changed = 0
    for node in g.node:
        if node.op_type != "If":
            continue
        then_g = next(a.g for a in node.attribute if a.name == "then_branch")
        else_g = next(a.g for a in node.attribute if a.name == "else_branch")
        changed += patch_graph(then_g)
        changed += patch_graph(else_g)
        for to, eo in zip(then_g.output, else_g.output):
            rt, re = rank_of_vi(to), rank_of_vi(eo)
            if {rt, re} != {2, 3}:
                continue
            low_g, low_o = (then_g, to) if rt == 2 else (else_g, eo)
            high_o = eo if rt == 2 else to
            high_shape = shape_of_vi(high_o)
            old_name = low_o.name
            new_name = old_name + "_trt_rank3"
            axes_name = new_name + "_axes"
            low_g.initializer.append(
                helper.make_tensor(axes_name, TensorProto.INT64, [1], [1])
            )
            low_g.node.append(
                helper.make_node(
                    "Unsqueeze",
                    [old_name, axes_name],
                    [new_name],
                    name=new_name + "_unsqueeze_axis1",
                )
            )
            low_o.name = new_name
            set_shape(low_o, high_shape or [-1, 1, -1])
            changed += 1
    return changed


def fix_one(src: Path, dst: Path) -> int:
    model = onnx.load(str(src))
    n = patch_graph(model.graph)
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    in_dir, out_dir = args.in_dir.resolve(), args.out_dir.resolve()
    if in_dir == out_dir:
        raise SystemExit("--in-dir and --out-dir must differ")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in TARGETS:
        src = in_dir / name
        if not src.exists():
            print(f"[SKIP] {src} (not found)")
            continue
        dst = out_dir / name
        n = fix_one(src, dst)
        print(f"[OK] {src.name}: patched {n} If output(s) -> {dst}")

    for path in in_dir.iterdir():
        if path.is_file() and path.name not in TARGETS:
            target = out_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
                print(f"[COPY] {path.name}")


if __name__ == "__main__":
    main()
