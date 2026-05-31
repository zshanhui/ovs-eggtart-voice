"""Wall-time microbench: tail-rest ORT FP32 vs dyn-INT8 vs static-QDQ-MMGemm."""
import os, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

BUCKET = int(os.environ.get("BUCKET", "8"))
ROOT = Path("/home/harve/kokoro-analysis")
if BUCKET == 8:
    BR = ROOT / "m_bucket8"; STEM = "kokoro-vocoder-tail-rest-cpu-bucket8"
elif BUCKET == 16:
    BR = ROOT / "m_bucket16"; STEM = "kokoro-vocoder-tail-rest-cpu-bucket16"
elif BUCKET == 32:
    BR = ROOT / "m_bucket32"; STEM = "kokoro-vocoder-tail-rest-cpu"

CALIB = ROOT / "calib" / f"bucket{BUCKET}"
sample = np.load(sorted(CALIB.glob("*.npz"))[0])
feed = {
    "/decoder/generator/Add_5_output_0": sample["voc_add"].astype(np.float32),
    "/Slice_2_output_0": sample["style_slice"].astype(np.float32),
    "/decoder/decode.3/Mul_output_0": sample["hidden"].astype(np.float32),
}

variants = [
    ("FP32", BR / f"{STEM}.onnx"),
    ("dyn-INT8", BR / f"{STEM}.int8.onnx"),
    ("static-mmgemm", BR / f"{STEM}.int8static_mmgemm.onnx"),
]

print(f"BUCKET={BUCKET}")
for name, p in variants:
    if not p.exists():
        print(f"  {name}: MISSING {p}"); continue
    o = ort.SessionOptions(); o.log_severity_level = 3
    sess = ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])
    # warmup
    for _ in range(3):
        sess.run(None, feed)
    times = []
    for _ in range(15):
        t0 = time.perf_counter(); sess.run(None, feed); times.append((time.perf_counter() - t0) * 1000)
    sz = p.stat().st_size / 1024 / 1024
    print(f"  {name:18s} size={sz:6.2f}MB  median={np.median(times):.2f}ms  p10={np.percentile(times,10):.2f}  p90={np.percentile(times,90):.2f}")
