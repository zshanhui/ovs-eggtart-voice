"""Derive bucket-32 calibration samples from bucket-16 activations.

bucket-16 tail-rest expects voc_add [1,256,2180], hidden [1,512,218], style [1,128]
bucket-32 tail-rest expects voc_add [1,256,4200], hidden [1,512,420], style [1,128]

We tile bucket-16 activations along time axis to reach bucket-32 shape; style is identical.
This preserves the per-channel value distribution (what matters for static QDQ calibration scales)
without requiring a bucket-32 vocoder-front ONNX (which doesn't exist on disk).
"""
from pathlib import Path
import numpy as np
import os

SRC = Path("/home/harve/kokoro-analysis/calib/bucket16")
DST = Path("/home/harve/kokoro-analysis/calib/bucket32")
DST.mkdir(parents=True, exist_ok=True)

TARGET_VOC = 4200
TARGET_HID = 420

files = sorted(SRC.glob("*.npz"))
print(f"src files: {len(files)}", flush=True)
for i, f in enumerate(files):
    z = np.load(f)
    voc = z["voc_add"]  # [1,256,2180]
    hid = z["hidden"]   # [1,512,218]
    sty = z["style_slice"]
    # Tile along last axis to >= target, then crop
    rep_v = (TARGET_VOC + voc.shape[-1] - 1) // voc.shape[-1]
    rep_h = (TARGET_HID + hid.shape[-1] - 1) // hid.shape[-1]
    voc32 = np.tile(voc, (1, 1, rep_v))[:, :, :TARGET_VOC].astype(np.float32)
    hid32 = np.tile(hid, (1, 1, rep_h))[:, :, :TARGET_HID].astype(np.float32)
    np.savez(DST / f.name, voc_add=voc32, style_slice=sty.astype(np.float32), hidden=hid32)
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)
print("DONE", flush=True)
