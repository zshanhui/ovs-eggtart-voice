"""P7(b) — Static QDQ INT8 quantization of tail-rest CPU ONNX.

Uses calibration .npz samples produced by gen_calib_p7b.py.
Quantizes Conv + ConvTranspose + MatMul + Gemm.
Per-channel weights, QInt8 activations.

Output (per bucket):
  m_bucket{N}/kokoro-vocoder-tail-rest-cpu[-bucketN].int8static.onnx

Strategy: if a primary build fails on ConvTranspose, retry excluding ConvTranspose nodes.
"""
from __future__ import annotations
import os, sys, glob, hashlib, traceback, time, types, importlib.machinery
from pathlib import Path
import numpy as np

# Workaround: local torch install is ABI-broken (NCCL symbol). Stub before importing
# onnxruntime.quantization, which lazily imports onnxruntime.tools -> torch.
def _stub_torch():
    if "torch" in sys.modules:
        return
    def mk(n):
        m = types.ModuleType(n)
        m.__spec__ = importlib.machinery.ModuleSpec(n, None)
        return m
    t = mk("torch"); tn = mk("torch.nn")
    t.nn = tn
    tn.Module = object
    sys.modules.update({"torch": t, "torch.nn": tn})
_stub_torch()

from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
    CalibrationMethod,
)

ROOT = Path("/home/harve/kokoro-analysis")
BUCKET = int(os.environ.get("BUCKET", "16"))

if BUCKET == 32:
    SRC = ROOT / "m_bucket32" / "kokoro-vocoder-tail-rest-cpu.onnx"
elif BUCKET == 16:
    SRC = ROOT / "m_bucket16" / "kokoro-vocoder-tail-rest-cpu-bucket16.onnx"
elif BUCKET == 8:
    SRC = ROOT / "m_bucket8" / "kokoro-vocoder-tail-rest-cpu-bucket8.onnx"
else:
    raise SystemExit(BUCKET)

CALIB = ROOT / "calib" / f"bucket{BUCKET}"
DST = SRC.with_name(SRC.stem + ".int8static.onnx")
# Limit calibration samples to keep memory + time bounded; minmax over 200 is plenty
MAX_CALIB = int(os.environ.get("MAX_CALIB", "200"))


class TailRestReader(CalibrationDataReader):
    def __init__(self, calib_dir: Path, limit: int):
        self.files = sorted(calib_dir.glob("*.npz"))[:limit]
        self.iter = iter(self.files)
        print(f"  calibration files: {len(self.files)}", flush=True)

    def get_next(self):
        try:
            f = next(self.iter)
        except StopIteration:
            return None
        z = np.load(f)
        return {
            "/decoder/generator/Add_5_output_0": z["voc_add"].astype(np.float32),
            "/Slice_2_output_0": z["style_slice"].astype(np.float32),
            "/decoder/decode.3/Mul_output_0": z["hidden"].astype(np.float32),
        }

    def rewind(self):
        self.iter = iter(self.files)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def attempt(op_types, exclude_nodes=None, tag=""):
    reader = TailRestReader(CALIB, MAX_CALIB)
    print(f"[BUCKET={BUCKET}] static QDQ attempt {tag} op_types={op_types} exclude_nodes={exclude_nodes}", flush=True)
    t0 = time.time()
    quantize_static(
        str(SRC), str(DST),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude_nodes or [],
    )
    dt = time.time() - t0
    sz = os.path.getsize(DST)
    src_sz = os.path.getsize(SRC)
    m = md5f(DST)
    print(f"[BUCKET={BUCKET}] OK in {dt:.1f}s size={sz/1024/1024:.2f}MB (src {src_sz/1024/1024:.2f}MB ratio {sz/src_sz:.2f}) md5={m}", flush=True)
    return True


# Primary: full coverage (Conv + ConvTranspose + MatMul + Gemm)
try:
    attempt(["Conv", "ConvTranspose", "MatMul", "Gemm"], tag="primary")
except Exception:
    print("PRIMARY FAILED ----", flush=True)
    traceback.print_exc()
    print("\n[fallback A] retry without ConvTranspose", flush=True)
    try:
        attempt(["Conv", "MatMul", "Gemm"], tag="no-convtranspose")
    except Exception:
        print("FALLBACK A FAILED ----", flush=True)
        traceback.print_exc()
        print("\n[fallback B] retry MatMul+Gemm only (matches dynamic baseline scope)", flush=True)
        attempt(["MatMul", "Gemm"], tag="matmul-gemm-only")
