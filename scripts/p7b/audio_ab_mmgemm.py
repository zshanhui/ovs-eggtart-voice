"""Audio AB: FP32 vs static-QDQ-MatMul+Gemm-only (P7b fallback)."""
from __future__ import annotations
import json, hashlib, os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from misaki import en, zh

BUCKET = int(os.environ.get("BUCKET", "8"))
ROOT = Path("/home/harve/kokoro-analysis")
if BUCKET == 8:
    BROOT = ROOT / "m_bucket8"; PREFIX=BROOT/"kokoro-prefix-cpu.onnx"; FRONT=BROOT/"kokoro-decoder-front.onnx"
    VFRONT=BROOT/"kokoro-vocoder-front-half-bucket8.onnx"
    REST_FP32=BROOT/"kokoro-vocoder-tail-rest-cpu-bucket8.onnx"
    REST_INT8=BROOT/"kokoro-vocoder-tail-rest-cpu-bucket8.int8static_mmgemm.onnx"
    SEQ_LEN=8
elif BUCKET == 16:
    BROOT=ROOT/"m_bucket16"; PREFIX=BROOT/"kokoro-prefix-cpu.onnx"; FRONT=BROOT/"kokoro-decoder-front.onnx"
    VFRONT=BROOT/"kokoro-vocoder-front-half-bucket16.onnx"
    REST_FP32=BROOT/"kokoro-vocoder-tail-rest-cpu-bucket16.onnx"
    REST_INT8=BROOT/"kokoro-vocoder-tail-rest-cpu-bucket16.int8static_mmgemm.onnx"
    SEQ_LEN=16
elif BUCKET == 32:
    BROOT=ROOT/"m_bucket32"; PREFIX=ROOT/"m_bucket16"/"kokoro-prefix-cpu.onnx"
    FRONT=ROOT/"m_bucket16"/"kokoro-decoder-front.onnx"
    VFRONT=ROOT/"m_bucket16"/"kokoro-vocoder-front-half-bucket16.onnx"
    REST_FP32=BROOT/"kokoro-vocoder-tail-rest-cpu.onnx"
    REST_INT8=BROOT/"kokoro-vocoder-tail-rest-cpu.int8static_mmgemm.onnx"
    SEQ_LEN=32

TOKENS_TXT=ROOT/"kokoro-multi-lang-v1_1"/"tokens.txt"
VOICES_BIN=ROOT/"kokoro-multi-lang-v1_1"/"voices.bin"
VOICE_STYLES=510; STYLE_DIM=256; STYLE_BYTES=VOICE_STYLES*STYLE_DIM*4

CASES_BY_BUCKET = {
    8: [("EN","Hi.",False),("EN","Yes.",False),("EN","Hello.",False),("EN","Thanks.",False),("EN","Stop.",False),
        ("ZH","你好。",True),("ZH","好的。",True),("ZH","再见。",True),("ZH","谢谢。",True),("ZH","晚安。",True)],
    16: [("EN","Hello world.",False),("EN","Birds sing morning.",False),("EN","Coffee is ready.",False),
         ("ZH","今天工作很忙。",True),("ZH","你好,今天天气真好。",True),("ZH","请告诉我时间。",True)],
    32: [("EN","Today the weather is really nice and sunny.",False),
         ("ZH","今天天气真好,我们出去走走吧。",True), ("ZH","明天的会议改到下午三点开始。",True)],
}
CASES = CASES_BY_BUCKET[BUCKET]

def load_vocab(p):
    m={}
    for line in open(p,"r",encoding="utf-8"):
        line=line.rstrip("\n")
        if not line: continue
        parts=line.rsplit(None,1)
        if len(parts)==2 and parts[1].lstrip("-").isdigit():
            m[parts[0] or " "]=int(parts[1])
    if " " not in m: m[" "]=16
    return m
VOCAB=load_vocab(TOKENS_TXT); g_en=en.G2P(trf=False,british=False); g_zh=zh.ZHG2P()

def phonemize(text,is_zh):
    ph,_=(g_zh(text) if is_zh else g_en(text))
    ids=[0]
    for ch in str(ph):
        tid=VOCAB.get(ch) or VOCAB.get(ch.lower())
        if tid is None: continue
        ids.append(tid)
    ids.append(0); n=min(len(ids),SEQ_LEN)
    arr=np.zeros((1,SEQ_LEN),dtype=np.int64); arr[0,:n]=np.asarray(ids[:n],dtype=np.int64)
    return arr,n

def load_style(sid,tc):
    idx=max(0,min(VOICE_STYLES-1,int(tc)))
    off=sid*STYLE_BYTES+idx*STYLE_DIM*4
    with open(VOICES_BIN,"rb") as f:
        f.seek(off); return np.frombuffer(f.read(STYLE_DIM*4),dtype=np.float32).reshape(1,STYLE_DIM).copy()

def make(p):
    o=ort.SessionOptions(); o.log_severity_level=3
    return ort.InferenceSession(str(p),o,providers=["CPUExecutionProvider"])

print(f"[BUCKET={BUCKET}] loading...",flush=True)
sp,sf,sv,sa,sb=make(PREFIX),make(FRONT),make(VFRONT),make(REST_FP32),make(REST_INT8)
speed=np.array([1.0],dtype=np.float32)

results=[]
for cat,text,is_zh in CASES:
    tokens,n_tok=phonemize(text,is_zh)
    if n_tok<2: continue
    style=load_style(0,n_tok)
    p_out=sp.run(None,{"tokens":tokens,"style":style,"speed":speed})
    pn=[o.name for o in sp.get_outputs()]
    dec_in=p_out[pn.index("/MatMul_1_output_0")] if "/MatMul_1_output_0" in pn else p_out[0]
    sty_sl=p_out[pn.index("/Slice_2_output_0")] if "/Slice_2_output_0" in pn else p_out[1]
    fA=sf.run(None,{"/MatMul_1_output_0":dec_in,"/Slice_2_output_0":sty_sl})[0]
    vA=sv.run(None,{"/decoder/decode.3/Mul_output_0":fA,"/Slice_2_output_0":sty_sl})[0]
    ri={"/decoder/generator/Add_5_output_0":vA,"/Slice_2_output_0":sty_sl,"/decoder/decode.3/Mul_output_0":fA}
    aA=sa.run(None,ri)[0].astype(np.float32).ravel()
    aB=sb.run(None,ri)[0].astype(np.float32).ravel()
    n=min(len(aA),len(aB)); aA=aA[:n]; aB=aB[:n]
    rl2=float(np.linalg.norm(aA-aB)/max(1e-9,np.linalg.norm(aA)))
    mad=float(np.max(np.abs(aA-aB)))
    results.append(dict(cat=cat,text=text,rel_l2=rl2,max_abs=mad,
                        md5_A=hashlib.md5(aA.tobytes()).hexdigest()[:12],
                        md5_B=hashlib.md5(aB.tobytes()).hexdigest()[:12]))
    print(f"  [{cat}] {text!r:40s} rel_l2={rl2:.5f} max_abs={mad:.4f}",flush=True)

worst=max((r["rel_l2"] for r in results),default=float("nan"))
median=float(np.median([r["rel_l2"] for r in results])) if results else float("nan")
verdict="PASS" if worst<=0.05 else ("MARGINAL" if worst<=0.08 else "FAIL")
agg=dict(bucket=BUCKET,mode="static-qdq-mmgemm",n=len(results),worst_rel_l2=worst,
         median_rel_l2=median,gate=0.05,verdict=verdict,cases=results)
out=BROOT/f"audio_ab_int8static_mmgemm_bucket{BUCKET}.json"
json.dump(agg,open(out,"w"),indent=2,ensure_ascii=False)
print(f"\n=== BUCKET {BUCKET} STATIC MM+GEMM VERDICT: {verdict} ===  worst={worst:.5f} median={median:.5f}")
print(f"report: {out}")
