# P1 Final Measurement on Orin Nano 8GB

Date: 2026-04-28
Config: P0a + P0b + ASR max_seq=200 + P1 trt_native + SKIP_ASR_WARMUP=1 + CP_POOL_SIZE=1
Image: jetson-voice-speech:v3.4-slim
Device: Orin Nano 8GB (total physical RAM 7620 MB per tegrastats)

## Result

- **ExitCode**: 137 (SIGKILL)
- **OOMKilled**: true
- **/health**: no response (container dead before serving)
- **Container outcome**: OOM during TTS engine loading (after CP KV engine pool creation)

```
TRTCPKVEnginePool: creating 1 slots (CP_POOL_SIZE=1)
  ← OOM killer fires here
```

## Per-step memory (from instrumentation)

| Stage | RSS (MB) | HWM (MB) | sys MemAvailable (MB) |
|---|---:|---:|---:|
| asr_start (before any model load) | 71.5 | 71.5 | 5867 |
| after_encoder (TRT native encoder) | 665 | 665 | 5352 |
| after_embed (embed_tokens loaded) | 962 | 962 | 5054 |
| after_decoder (TRT decoder + CUDA Graph) | 2243 | 2243 | 2410 |
| asr_ready (warmup skipped) | 2332 | 2371 | 2321 |
| tts_start | 2332 | 2371 | 2322 |
| tts_after_tokenizer | 2338 | 2371 | 2306 |
| TTS TRT engine load → **OOM** | — | — | — |

## tegrastats (system-wide)

- **Peak RAM**: 7495 / 7620 MB
- **Peak swap**: 2664 / 3810 MB (at RAM peak)
- **Free at peak**: 125 MB physical RAM

## Gap analysis

ASR loads fine with healthy margin: at `asr_ready`, the system still had 2321 MB available. But the TTS model (Qwen3 TRT decoder dual-profile + CP KV engine) requires additional memory that the Nano cannot provide:

| Component | Estimated additional need |
|---|---|
| TTS decoder engine load (dual-profile TRT) | ~300–500 MB (encoder weights, KV cache buffers) |
| CP KV engine load | ~200–400 MB |
| Runtime scratch/workspace | ~200 MB |
| **Total TTS** | **~700–1100 MB** |

At `tts_after_tokenizer` only 2306 MB was available. TTS engine loading needed
significantly more than the 125 MB headroom, triggering the OOM killer.

**The remaining gap cannot be closed within 8GB physical RAM.** Even assuming
aggressive optimization (FP8, reduced KV cache), TTS on this Nano configuration
requires additional physical memory or a reduction in model scope.

## Options forward

1. **TTS-only image on Nano**: strip ASR entirely from the image to free ~2.3 GB
   of RSS, leaving room for TTS. V2V pipeline would need two containers or an
   external ASR service.
2. **ASR-only image on Nano**: strip TTS, keep ASR serving (fits with ~2.3 GB margin).
   TTS handled by a separate device.
3. **Upgrade Nano to 16GB SKU**: 16 GB RAM variant would provide enough headroom
   for both ASR + TTS with current optimization.
4. **v3.4-slim architecture**: the image format itself has room for DLAs/FP8 on
   Orin NX but gains nothing on Nano — Nano lacks DLA and has the same
   architectural limits regardless of image variant.
