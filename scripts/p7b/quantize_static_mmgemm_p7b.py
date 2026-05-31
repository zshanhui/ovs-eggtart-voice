"""P7(b) fallback — Static QDQ MatMul+Gemm only (no Conv/ConvTranspose).

If primary Conv-inclusive QDQ breaks audio, this gives us a static variant
matching the dynamic scope, allowing a static-vs-dynamic perf comparison.

Output: ...int8static_mmgemm.onnx
"""
from __future__ import annotations
import os, sys, hashlib, time, types, importlib.machinery
from pathlib import Path
import numpy as np

def _stub_torch():
    if "torch" in sys.modules: return
    def mk(n):
        m = types.ModuleType(n); m.__spec__ = importlib.machinery.ModuleSpec(n, None); return m
    t = mk("torch"); tn = mk("torch.nn")
    t.nn = tn; tn.Module = object
    sys.modules.update({"torch": t, "torch.nn": tn})
_stub_torch()

from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod,
)

ROOT = Path("/home/harve/kokoro-analysis")
BUCKET = int(os.environ.get("BUCKET", "8"))
if BUCKET == 32:
    SRC = ROOT / "m_bucket32" / "kokoro-vocoder-tail-rest-cpu.onnx"
elif BUCKET == 16:
    SRC = ROOT / "m_bucket16" / "kokoro-vocoder-tail-rest-cpu-bucket16.onnx"
elif BUCKET == 8:
    SRC = ROOT / "m_bucket8" / "kokoro-vocoder-tail-rest-cpu-bucket8.onnx"

CALIB = ROOT / "calib" / f"bucket{BUCKET}"
DST = SRC.with_name(SRC.stem + ".int8static_mmgemm.onnx")
MAX_CALIB = int(os.environ.get("MAX_CALIB", "200"))


class TailRestReader(CalibrationDataReader):
    def __init__(self, calib_dir, limit):
        self.files = sorted(calib_dir.glob("*.npz"))[:limit]
        self.iter = iter(self.files)
        print(f"  calibration files: {len(self.files)}", flush=True)
    def get_next(self):
        try: f = next(self.iter)
        except StopIteration: return None
        z = np.load(f)
        return {"/decoder/generator/Add_5_output_0": z["voc_add"].astype(np.float32),
                "/Slice_2_output_0": z["style_slice"].astype(np.float32),
                "/decoder/decode.3/Mul_output_0": z["hidden"].astype(np.float32)}


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()


reader = TailRestReader(CALIB, MAX_CALIB)
print(f"[BUCKET={BUCKET}] static QDQ MatMul+Gemm only -> {DST.name}", flush=True)
t0 = time.time()
quantize_static(
    str(SRC), str(DST),
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=["MatMul", "Gemm"],
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    calibrate_method=CalibrationMethod.MinMax,
)
dt = time.time() - t0
print(f"[BUCKET={BUCKET}] OK in {dt:.1f}s size={os.path.getsize(DST)/1024/1024:.2f}MB md5={md5f(DST)}", flush=True)
