"""P7(b) — Tail-rest INT8-STATIC (QDQ) vs FP32 audio A/B per bucket.

Mirrors audio_ab_int8.py (which tests dynamic INT8). Here B is the static QDQ build.
Gate: rel_l2 <= 0.05 PASS, <= 0.08 MARGINAL, else FAIL.
"""
from __future__ import annotations
import json, hashlib, os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from misaki import en, zh

BUCKET = int(os.environ.get("BUCKET", "16"))
SR = 24000
VOICE_STYLES = 510
STYLE_DIM = 256
STYLE_BYTES = VOICE_STYLES * STYLE_DIM * 4

ROOT = Path("/home/harve/kokoro-analysis")
if BUCKET == 32:
    BROOT = ROOT / "m_bucket32"
    PREFIX = ROOT / "m_bucket16" / "kokoro-prefix-cpu.onnx"
    FRONT = ROOT / "m_bucket16" / "kokoro-decoder-front.onnx"
    VFRONT = ROOT / "m_bucket16" / "kokoro-vocoder-front-half-bucket16.onnx"
    REST_FP32 = BROOT / "kokoro-vocoder-tail-rest-cpu.onnx"
    REST_INT8 = BROOT / "kokoro-vocoder-tail-rest-cpu.int8static.onnx"
    SEQ_LEN = 32
elif BUCKET == 16:
    BROOT = ROOT / "m_bucket16"
    PREFIX = BROOT / "kokoro-prefix-cpu.onnx"
    FRONT = BROOT / "kokoro-decoder-front.onnx"
    VFRONT = BROOT / "kokoro-vocoder-front-half-bucket16.onnx"
    REST_FP32 = BROOT / "kokoro-vocoder-tail-rest-cpu-bucket16.onnx"
    REST_INT8 = BROOT / "kokoro-vocoder-tail-rest-cpu-bucket16.int8static.onnx"
    SEQ_LEN = 16
elif BUCKET == 8:
    BROOT = ROOT / "m_bucket8"
    PREFIX = BROOT / "kokoro-prefix-cpu.onnx"
    FRONT = BROOT / "kokoro-decoder-front.onnx"
    VFRONT = BROOT / "kokoro-vocoder-front-half-bucket8.onnx"
    REST_FP32 = BROOT / "kokoro-vocoder-tail-rest-cpu-bucket8.onnx"
    REST_INT8 = BROOT / "kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx"
    SEQ_LEN = 8

TOKENS_TXT = ROOT / "kokoro-multi-lang-v1_1" / "tokens.txt"
VOICES_BIN = ROOT / "kokoro-multi-lang-v1_1" / "voices.bin"

CASES_BY_BUCKET = {
    8: [("EN","Hi.",False),("EN","Yes.",False),("EN","Okay.",False),("EN","Hello.",False),
        ("EN","Thanks.",False),("EN","Stop.",False),("ZH","你好。",True),("ZH","好的。",True),
        ("ZH","再见。",True),("ZH","谢谢。",True),("ZH","是的。",True),("ZH","晚安。",True)],
    16: [("EN","Hello world.",False),("EN","Good morning.",False),("EN","Birds sing morning.",False),
         ("EN","I love coding.",False),("EN","Coffee is ready.",False),("EN","The sky is blue.",False),
         ("ZH","今天工作很忙。",True),("ZH","我吃过饭了。",True),("ZH","明天我们去公园。",True),
         ("ZH","你好,今天天气真好。",True),("ZH","请告诉我时间。",True),("ZH","祝你好运。",True)],
    32: [("EN","Today the weather is really nice and sunny.",False),
         ("EN","Please tell me what time it is right now.",False),
         ("EN","I would like to order a cup of coffee please.",False),
         ("EN","The quick brown fox jumps over the lazy dog.",False),
         ("EN","Hello world, this is a longer test sentence.",False),
         ("ZH","今天天气真好,我们出去走走吧。",True),
         ("ZH","明天的会议改到下午三点开始。",True),
         ("ZH","请问最近的地铁站怎么走?",True),
         ("ZH","我想喝一杯热咖啡,谢谢。",True),
         ("ZH","这本书写得非常好,我很喜欢。",True),
         ("ZH","祝你今天工作顺利,生活愉快。",True),
         ("ZH","希望大家都能过上幸福的生活。",True)],
}
CASES = CASES_BY_BUCKET[BUCKET]


def load_vocab(p):
    m = {}
    for line in open(p, "r", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line: continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            tok = parts[0] or " "
            m[tok] = int(parts[1])
    if " " not in m: m[" "] = 16
    return m


VOCAB = load_vocab(TOKENS_TXT)
g_en = en.G2P(trf=False, british=False)
g_zh = zh.ZHG2P()


def phonemize(text, is_zh):
    ph, _ = (g_zh(text) if is_zh else g_en(text))
    ids = [0]
    for ch in str(ph):
        tid = VOCAB.get(ch) or VOCAB.get(ch.lower())
        if tid is None: continue
        ids.append(tid)
    ids.append(0)
    n = min(len(ids), SEQ_LEN)
    arr = np.zeros((1, SEQ_LEN), dtype=np.int64)
    arr[0, :n] = np.asarray(ids[:n], dtype=np.int64)
    return arr, n


def load_style(sid, tc):
    idx = max(0, min(VOICE_STYLES-1, int(tc)))
    off = sid*STYLE_BYTES + idx*STYLE_DIM*4
    with open(VOICES_BIN, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(STYLE_DIM*4), dtype=np.float32).reshape(1, STYLE_DIM).copy()


def make_sess(p):
    o = ort.SessionOptions()
    o.log_severity_level = 3
    return ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])


print(f"[BUCKET={BUCKET}] loading sessions...", flush=True)
sp = make_sess(PREFIX); sf = make_sess(FRONT); sv = make_sess(VFRONT)
sa = make_sess(REST_FP32); sb = make_sess(REST_INT8)
speed = np.array([1.0], dtype=np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float32).ravel(); b = np.asarray(b, dtype=np.float32).ravel()
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]
    return float(np.linalg.norm(a-b)/max(1e-9,np.linalg.norm(a))), float(np.max(np.abs(a-b))), n


results = []
for cat, text, is_zh in CASES:
    tokens, n_tok = phonemize(text, is_zh)
    if n_tok < 2: continue
    style = load_style(0, n_tok)
    p_out = sp.run(None, {"tokens": tokens, "style": style, "speed": speed})
    pn = [o.name for o in sp.get_outputs()]
    dec_in = p_out[pn.index("/MatMul_1_output_0")] if "/MatMul_1_output_0" in pn else p_out[0]
    sty_sl = p_out[pn.index("/Slice_2_output_0")] if "/Slice_2_output_0" in pn else p_out[1]
    fA = sf.run(None, {"/MatMul_1_output_0": dec_in, "/Slice_2_output_0": sty_sl})[0]
    vA = sv.run(None, {"/decoder/decode.3/Mul_output_0": fA, "/Slice_2_output_0": sty_sl})[0]
    rest_in = {
        "/decoder/generator/Add_5_output_0": vA,
        "/Slice_2_output_0": sty_sl,
        "/decoder/decode.3/Mul_output_0": fA,
    }
    aA = sa.run(None, rest_in)[0].astype(np.float32).ravel()
    aB = sb.run(None, rest_in)[0].astype(np.float32).ravel()
    rl2, mad, n = rel_l2(aA, aB)
    results.append(dict(cat=cat, text=text, n_tok=n_tok, len=n, rel_l2=rl2, max_abs_diff=mad,
                        rms_A=float(np.sqrt(np.mean(aA**2))), rms_B=float(np.sqrt(np.mean(aB**2))),
                        md5_A=hashlib.md5(aA.tobytes()).hexdigest()[:12],
                        md5_B=hashlib.md5(aB.tobytes()).hexdigest()[:12]))
    print(f"  [{cat}] {text!r:40s} n_tok={n_tok:2d} rel_l2={rl2:.5f} max_abs={mad:.4f}", flush=True)

worst = max((r["rel_l2"] for r in results), default=float("nan"))
median = float(np.median([r["rel_l2"] for r in results])) if results else float("nan")
verdict = "PASS" if worst <= 0.05 else ("MARGINAL" if worst <= 0.08 else "FAIL")
agg = dict(bucket=BUCKET, mode="static-qdq", n=len(results), worst_rel_l2=worst,
           median_rel_l2=median, gate=0.05, verdict=verdict, cases=results)
out = BROOT / f"audio_ab_int8static_bucket{BUCKET}.json"
json.dump(agg, open(out, "w"), indent=2, ensure_ascii=False)
print(f"\n=== BUCKET {BUCKET} STATIC QDQ VERDICT: {verdict} ===")
print(f"worst rel_l2 = {worst:.5f}  median = {median:.5f}  (gate 0.05)")
print(f"report: {out}")
