# Rockchip Hardware Analysis — Full Voice Stack (ASR + LLM + TTS)

## The Two SoCs

| Spec | RK3576 | RK3588 |
|------|--------|--------|
| NPU | 6 TOPS (dual-core) | 6 TOPS (tri-core) |
| CPU | 4×A72 + 4×A53 | 4×A76 + 4×A55 |
| RAM (typical boards) | 8 GB LPDDR4X | 8–16 GB LPDDR4X/5 |
| Memory bandwidth | ~17 GB/s | ~25–30 GB/s |
| NPU cores usable by SDK | 2 | 3 |
| GPU | Mali-G52 | Mali-G610 |

---

## Component-by-Component Breakdown

### 1. ASR — Qwen3-ASR on NPU

Encoder-decoder ASR model split across the NPU:

- **Encoder:** RKNN-compiled (fp16), ~370–387 MB. Fixed-size chunk encoder with 2s / 4s / 15s temporal windows. Runs on RKNN (convolution + attention layers).
- **Decoder:** RKLLM-compiled (autoregressive token generation).
  - **RK3576:** w4a16_g128 quantised → ~795 MB. 2 NPU cores.
  - **RK3588:** fp16 (no quantisation) → ~1.53 GB. 3 NPU cores (`NPU_CORE_AUTO`).
- **Embedding table:** 622 MB (shared across both SoCs), loaded once.
- **Tokenizer:** 11.4 MB.

**NPU core allocation:**
- RK3576 `docker-compose.rk.yml`: `ASR_NPU_CORE_MASK=NPU_CORE_1` — reserves NPU_CORE_2 for LLM.
- RK3588 `docker-compose.radxa.yml`: `ASR_NPU_CORE_MASK=NPU_CORE_AUTO` — all 3 cores available.

**Streaming behaviour:**
- 400 ms chunks + left-context encoder + Silero VAD endpoint pre-fire → near-zero stop→final latency.
- Long-audio guard (≥15s): energy-RMS split into ≤4.5s segments with independent `transcribe()` calls to avoid the 512-token decoder context snowballing garbage from chunk to chunk.

**Memory:** ~1.2 GB (RK3576, w4a16 decoder) / ~2.5 GB (RK3588, fp16 decoder).

### 2. LLM — Qwen3 Chat on NPU via RKLLM

Separate Docker container (`openvoicestream-llm`, port 8001) hosting an OpenAI-compatible `/v1/chat/completions` endpoint. Uses `librkllmrt.so` via a ctypes binding reverse-engineered from Rockchip's public `rkllm_api_demo`.

**Model sizes:**

| Model | Disk (.rkllm) | Runtime memory (W4A16) | Target SoC |
|-------|--------------|------------------------|------------|
| Qwen3-0.6B | ~520 MB | ~350 MB | RK3576 ✓ |
| Qwen3-1.7B | ~1.3 GB | ~1.1 GB | RK3588 ✓ |
| Qwen3-4B | ~2.8 GB (est.) | ~2.5 GB (est.) | RK3588 (W8A8 maybe) |

**NPU core allocation:** `RK_LLM_NPU_CORE_MASK=NPU_CORE_2` on RK3576 (dedicated single core).

**Performance expectations** (from `agent/apps/rk3576-chat/config.yaml`):

| Metric | Cold | Warm |
|--------|------|------|
| First token latency | ~500 ms | ~100 ms |
| Generation speed | ~15–25 tok/s (0.6B) | same |
| Max context | 4096 tokens | |
| Max generation | 512 tokens | |
| Timeout (first token) | 10 s | |
| Timeout (stream idle) | 30 s | |

**Chat template:** Manual ChatML implementation (no `transformers` dependency — saves ~100 MB in the RK container).

**Memory:** ~600 MB (0.6B W4A16 on RK3576) / ~1.5 GB (1.7B W4A16 on RK3588).

### 3. TTS — Matcha + Vocos (NPU/CPU hybrid)

- **Matcha encoder + estimator:** RKNN compiled (~18 MB + 24 MB), runs on NPU.
- **Matcha acoustic ONNX:** ORT on CPU (`model-steps-3.onnx`, ~76 MB), 4 threads.
- **Vocos vocoder:** RKNN compiled (`vocos-16khz-600.rknn`, ~28 MB), runs on NPU.
- **Text frontend:** sherpa-onnx + espeak-ng on CPU for phonemisation.

**Why CPU for the acoustic model?** The Matcha acoustic ONNX has dynamic shape operators that don't convert cleanly to RKNN. So the pipeline is: phonemes → NPU (encoder+estimator, ~50ms) → CPU (ORT acoustic, ~200–400ms) → NPU (Vocos, ~30ms).

**Vocos frame count:** 600 frames on RK3576, 256 frames on RK3588. This controls how many audio samples are generated per NPU invocation — RK3588's higher memory bandwidth lets it process smaller chunks without stalling.

**Memory:** ~300 MB total.

---

## NPU Core Allocation Summary

```
RK3576 (2 NPU cores):
  NPU_CORE_1 → ASR + TTS (shared, serialised execution)
  NPU_CORE_2 → LLM  (dedicated, separate container)

RK3588 (3 NPU cores):
  NPU_CORE_AUTO → ASR (all 3 cores, auto-scheduled)
  LLM: not yet wired in docker-compose.radxa.yml
```

The profile JSONs (`rk3576-default.json`, `rk3588-default.json`) both declare:

```json
"execution_policy": {
    "mode": "serialized",
    "shared_resource": "npu"
}
```

This means ASR and TTS never run NPU ops concurrently — the framework serialises them.

---

## Total Memory Budget

### RK3576 (8 GB typical — cat-remote)

| Component | Approx. Memory |
|-----------|---------------|
| Docker + OS overhead | ~500 MB |
| ASR (encoder + decoder + embeds) | ~1.2 GB |
| TTS (Matcha + Vocos + ONNX + sherpa) | ~300 MB |
| LLM (Qwen3-0.6B W4A16) | ~600 MB |
| Python runtime (×2 containers) + overhead | ~500 MB |
| **Total** | **~3.1 GB** |

Container limits: speech 7500 MB, LLM 2000 MB. Fits in 8 GB with ~4–5 GB headroom for file cache and spikes.

### RK3588 (8–16 GB)

| Component | Approx. Memory |
|-----------|---------------|
| ASR (fp16 decoder = larger) | ~2.5 GB |
| TTS | ~300 MB |
| LLM (Qwen3-1.7B W4A16) | ~1.5 GB |
| Runtime overhead | ~600 MB |
| **Total** | **~4.9 GB** |

With 16 GB boards: comfortable. With 8 GB: tight (~3 GB left for OS + spikes). Choosing the 0.6B model on an 8 GB RK3588 board would bring it down to ~4 GB total.

---

## End-to-End Latency Estimate

Typical voice exchange ("What's the weather today?" → 3-sentence reply):

| Stage | RK3576 | RK3588 |
|-------|--------|--------|
| VAD endpoint detection | ~400 ms | ~400 ms |
| ASR final decode | ~200–400 ms | ~100–200 ms |
| LLM first token | ~500 ms (cold) / ~100 ms (warm) | ~200–400 ms (cold) / ~80 ms (warm) |
| LLM completion (~30 tokens) | ~1.5–2.0 s | ~0.8–1.2 s |
| TTS synthesis | ~300–500 ms | ~200–400 ms |
| **Total (cold)** | **~3.0–4.0 s** | **~2.0–3.0 s** |
| **Total (warm, KV cache primed)** | **~1.5–2.5 s** | **~1.0–1.5 s** |

**Where the time goes on RK3576:**
- LLM generation dominates (1.5–2s). At 15–25 tok/s for a 30-token reply, that's 1.2–2s.
- TTS acoustic ONNX on CPU is the second bottleneck (200–400ms) — 4×A72 cores working on a 76 MB ONNX model.
- ASR final decode (200–400ms) is the third contributor.

---

## Assessment

### RK3576 — "It works, but there's no slack"

The RK3576 is the production target for the cat-remote device and the codebase is actively tuned for it. Key observations:

1. **NPU contention is real.** Two NPU cores total. ASR+TTS share one core (serialised), LLM gets the other. This is a reasonable split for a sequential ASR→LLM→TTS pipeline, but it means no component can burst with extra NPU compute.

2. **LLM quality is constrained.** Qwen3-0.6B at W4A16 is the floor for usable conversation. It handles simple Q&A and chitchat but struggles with multi-turn reasoning, complex instructions, or nuanced responses. The 1.7B model won't fit in RK3576's 8 GB alongside ASR and TTS.

3. **"Smooth" is subjective.** Cold starts feel slow (3–4s). Warm conversations (primed KV cache) are acceptable at ~1.5–2.5s. For a product targeting natural conversation flow, the RK3576 is at the edge — usable but not seamless.

4. **CPU TTS is a drag.** Running the Matcha acoustic ONNX on 4×A72 CPU cores adds 200–400ms per utterance. The NPU simply doesn't have the operator coverage to run it natively.

### RK3588 — "The real target for a smooth full stack"

1. **Three NPU cores** means one per component (ASR / LLM / TTS) with no serialisation tax — or all three on ASR for faster decode.

2. **More RAM** (especially 16 GB boards) comfortably fits the 1.7B LLM with fp16 ASR decoder. 1.7B at W4A16 is a meaningful quality jump over 0.6B — better reasoning, better multi-turn, better instruction following.

3. **Faster CPU cores** (A76 vs A72) help the TTS acoustic ONNX path. ~30-40% IPC uplift over A72.

4. **Higher memory bandwidth** (~25–30 GB/s vs ~17 GB/s) directly translates to faster NPU inference, especially for the large decoder models.

5. **Gap:** The `docker-compose.radxa.yml` currently has no LLM service defined. The speech container is there, but the LLM needs to be added to match the RK3576 setup.

### Recommendation

| Use case | Minimum SoC | LLM model |
|----------|------------|-----------|
| ASR + TTS only (no on-device LLM) | RK3576 | N/A |
| Full stack, acceptable latency | RK3576 | Qwen3-0.6B W4A16 |
| Full stack, smooth/natural feel | RK3588 (16 GB) | Qwen3-1.7B W4A16 |
| Full stack, best quality | RK3588 (16 GB) | Qwen3-4B W8A8 |

The RK3576 is viable today but will feel constrained. The RK3588 with 1.7B is the realistic target for a product-quality experience. The RK3576 is better positioned as an ASR+TTS device with LLM inference offloaded to a remote service.

---

## Potential Optimisations (not yet explored)

1. **Pipelining ASR encoder with LLM prefill.** Start LLM KV-cache prefill as soon as the ASR transcript begins streaming (before the full utterance is finalised). Could shave 200–400ms.

2. **LLM speculative decoding on the _other_ NPU core.** On RK3576, the LLM is stuck on one core. A draft model on the A72 CPU could feed candidates to the NPU for parallel verification.

3. **TTS acoustic model on NPU.** If the Matcha ONNX can be split into NPU-friendly subgraphs (encoder on NPU, attention on CPU, etc.), the 200–400ms CPU acoustic step could shrink significantly.

4. **KV-cache compression for the LLM.** Reduces memory and speeds up prefill for multi-turn conversations.

5. **Streaming TTS with overlap.** Begin TTS synthesis on the first LLM token rather than waiting for the full reply. This is already partially enabled by the `matcha_rknn` backend's streaming capability.
