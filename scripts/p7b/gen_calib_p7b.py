"""P7(b) — Generate calibration activations for tail-rest static QDQ.

Per-bucket: runs prefix -> decoder-front -> vocoder-front FP32 to capture the
3 tail-rest input tensors (voc_add, style_slice, hidden) over a 200+ corpus.

Output: /home/harve/kokoro-analysis/calib/bucket{8,16,32}/sample_NNNN.npz
Keys: voc_add, style_slice, hidden  (input names match tail-rest ONNX inputs).
"""
from __future__ import annotations
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from misaki import en, zh

BUCKET = int(os.environ.get("BUCKET", "16"))
TARGET = int(os.environ.get("TARGET_SAMPLES", "240"))
SR = 24000
VOICE_STYLES = 510
STYLE_DIM = 256
STYLE_BYTES = VOICE_STYLES * STYLE_DIM * 4

ROOT = Path("/home/harve/kokoro-analysis")
OUT_DIR = ROOT / "calib" / f"bucket{BUCKET}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if BUCKET == 32:
    BROOT = ROOT / "m_bucket32"
    PREFIX = ROOT / "m_bucket16" / "kokoro-prefix-cpu.onnx"
    FRONT = ROOT / "m_bucket16" / "kokoro-decoder-front.onnx"
    VFRONT = ROOT / "m_bucket16" / "kokoro-vocoder-front-half-bucket16.onnx"
    SEQ_LEN = 32
elif BUCKET == 16:
    BROOT = ROOT / "m_bucket16"
    PREFIX = BROOT / "kokoro-prefix-cpu.onnx"
    FRONT = BROOT / "kokoro-decoder-front.onnx"
    VFRONT = BROOT / "kokoro-vocoder-front-half-bucket16.onnx"
    SEQ_LEN = 16
elif BUCKET == 8:
    BROOT = ROOT / "m_bucket8"
    PREFIX = BROOT / "kokoro-prefix-cpu.onnx"
    FRONT = BROOT / "kokoro-decoder-front.onnx"
    VFRONT = BROOT / "kokoro-vocoder-front-half-bucket8.onnx"
    SEQ_LEN = 8
else:
    raise SystemExit(BUCKET)

TOKENS_TXT = ROOT / "kokoro-multi-lang-v1_1" / "tokens.txt"
VOICES_BIN = ROOT / "kokoro-multi-lang-v1_1" / "voices.bin"


# Per-bucket corpus: enough variety to span real activation distribution.
CORPUS_8 = [
    "Hi.", "Yes.", "Okay.", "Hello.", "Thanks.", "Stop.", "Sure.", "Right.",
    "Fine.", "Good.", "Wait.", "Done.", "Bye.", "Cool.", "True.", "Nope.",
    "Maybe.", "Now.", "Here.", "Soon.",
    "你好。", "好的。", "再见。", "谢谢。", "是的。", "晚安。", "请讲。", "请稍等。",
    "明白。", "知道了。", "对的。", "不是。", "没问题。", "可以。", "听到。", "收到。",
    "你呢？", "我来。", "走吧。", "辛苦。",
]
CORPUS_16 = [
    "Hello world.", "Good morning everyone.", "Birds sing morning.",
    "I love coding daily.", "Coffee is ready now.", "The sky is blue today.",
    "Thanks for your help.", "Let me check this please.", "Please wait a moment.",
    "What time is it now?", "Where are you going?", "How was your day?",
    "I would like some water.", "Please open the window.", "Turn on the light.",
    "It is getting late tonight.", "I will see you tomorrow morning.",
    "The meeting starts at three.", "Have a wonderful weekend.",
    "Please close the door quietly.",
    "今天工作很忙。", "我吃过饭了。", "明天我们去公园。", "你好,今天天气真好。",
    "请告诉我时间。", "祝你好运。", "我喜欢这本书。", "请帮我打开窗户。",
    "明天会下雨吗？", "现在几点钟了？", "你叫什么名字？", "我们一起出去吧。",
    "请稍等一下。", "请把灯打开。", "今天的会议很重要。", "我已经吃过晚饭了。",
    "请你帮我个忙。", "明天见,晚安。", "天气越来越冷了。", "我想去看电影。",
]
CORPUS_32 = [
    "Today the weather is really nice and sunny.",
    "Please tell me what time it is right now.",
    "I would like to order a cup of coffee please.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a longer test sentence.",
    "Could you help me find the nearest subway station?",
    "I think we should have lunch together tomorrow afternoon.",
    "The library closes at six in the evening on weekdays.",
    "Remember to bring your umbrella because it might rain later.",
    "She decided to take a walk in the park this morning.",
    "Reading a good book before bed helps me relax completely.",
    "The conference will be held in the main hall downstairs.",
    "My favorite season is autumn because of the colorful leaves.",
    "Please send me the report by the end of this week.",
    "We are planning a trip to the mountains next month.",
    "The new restaurant downtown serves excellent Italian food.",
    "今天天气真好,我们出去走走吧。",
    "明天的会议改到下午三点开始。",
    "请问最近的地铁站怎么走?",
    "我想喝一杯热咖啡,谢谢。",
    "这本书写得非常好,我很喜欢。",
    "祝你今天工作顺利,生活愉快。",
    "希望大家都能过上幸福的生活。",
    "下个月我们打算去山里旅行一次。",
    "新开的那家意大利餐厅味道非常好。",
    "图书馆工作日晚上六点关门。",
    "记得带伞,因为下午可能会下雨。",
    "她今天早上决定去公园散步。",
    "睡前读一本好书能让我完全放松。",
    "请在本周末之前把报告发给我。",
    "我最喜欢的季节是秋天,因为叶子很美。",
    "会议将在楼下的主大厅举行。",
]

CORPUS_MAP = {8: CORPUS_8, 16: CORPUS_16, 32: CORPUS_32}
corpus = CORPUS_MAP[BUCKET]


def load_vocab(p):
    m = {}
    for line in open(p, "r", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            tok = parts[0] or " "
            m[tok] = int(parts[1])
    if " " not in m:
        m[" "] = 16
    return m


VOCAB = load_vocab(TOKENS_TXT)
g_en = en.G2P(trf=False, british=False)
g_zh = zh.ZHG2P()


def phonemize(text, is_zh):
    ph, _ = (g_zh(text) if is_zh else g_en(text))
    ids = [0]
    for ch in str(ph):
        tid = VOCAB.get(ch) or VOCAB.get(ch.lower())
        if tid is None:
            continue
        ids.append(tid)
    ids.append(0)
    n = min(len(ids), SEQ_LEN)
    arr = np.zeros((1, SEQ_LEN), dtype=np.int64)
    arr[0, :n] = np.asarray(ids[:n], dtype=np.int64)
    return arr, n


def load_style(sid, tc):
    idx = max(0, min(VOICE_STYLES - 1, int(tc)))
    off = sid * STYLE_BYTES + idx * STYLE_DIM * 4
    with open(VOICES_BIN, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(STYLE_DIM * 4), dtype=np.float32).reshape(1, STYLE_DIM).copy()


def make_sess(p):
    o = ort.SessionOptions()
    o.log_severity_level = 3
    return ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])


print(f"[BUCKET={BUCKET}] loading sessions...", flush=True)
sp = make_sess(PREFIX)
sf = make_sess(FRONT)
sv = make_sess(VFRONT)

speed = np.array([1.0], dtype=np.float32)
# expand corpus by varying speaker style index — cheap activation variety
speakers = [0]  # single voice is enough; calibrate to production voice

# Per-text variations: 6 speed values to enrich distribution
SPEEDS = [0.9, 0.95, 1.0, 1.0, 1.05, 1.1]

samples_written = 0
t0 = time.time()
text_idx = 0
while samples_written < TARGET:
    text_idx += 1
    cycle_idx = (text_idx - 1) % len(corpus)
    text = corpus[cycle_idx]
    is_zh = any(0x4E00 <= ord(c) <= 0x9FFF for c in text)
    tokens, n_tok = phonemize(text, is_zh)
    if n_tok < 2:
        continue
    speed_val = SPEEDS[(text_idx - 1) % len(SPEEDS)]
    speed = np.array([speed_val], dtype=np.float32)
    for sid in speakers:
        style = load_style(sid, n_tok)
        p_out = sp.run(None, {"tokens": tokens, "style": style, "speed": speed})
        pn = [o.name for o in sp.get_outputs()]
        dec_in = p_out[pn.index("/MatMul_1_output_0")] if "/MatMul_1_output_0" in pn else p_out[0]
        sty_sl = p_out[pn.index("/Slice_2_output_0")] if "/Slice_2_output_0" in pn else p_out[1]
        fA = sf.run(None, {"/MatMul_1_output_0": dec_in, "/Slice_2_output_0": sty_sl})[0]
        vA = sv.run(None, {"/decoder/decode.3/Mul_output_0": fA, "/Slice_2_output_0": sty_sl})[0]
        np.savez(OUT_DIR / f"sample_{samples_written:04d}.npz",
                 voc_add=vA.astype(np.float32),
                 style_slice=sty_sl.astype(np.float32),
                 hidden=fA.astype(np.float32))
        samples_written += 1
        if samples_written >= TARGET:
            break
    if samples_written % 20 == 0:
        dt = time.time() - t0
        print(f"  bucket{BUCKET}: {samples_written}/{TARGET} in {dt:.1f}s", flush=True)

dt = time.time() - t0
print(f"[BUCKET={BUCKET}] DONE {samples_written} samples in {dt:.1f}s -> {OUT_DIR}")
