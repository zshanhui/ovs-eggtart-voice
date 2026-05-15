# OpenVoiceStream

**Streaming voice for edge dialogue.** Few dependencies (just engines + NPU), one container deploy, ~110ms TTFT on Jetson / RK3588 / RPi.

[![GitHub stars](https://img.shields.io/github/stars/suharvest/seeed-local-voice?style=social)](https://github.com/suharvest/seeed-local-voice)
[![sherpa-onnx](https://img.shields.io/badge/engine-sherpa--onnx-green.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![Qwen3](https://img.shields.io/badge/multilingual-Qwen3-blueviolet.svg)](https://huggingface.co/Qwen)
[![Kokoro TTS](https://img.shields.io/badge/TTS-Kokoro%20v1.0-orange.svg)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](https://www.docker.com/)
[![Devices](https://img.shields.io/badge/devices-Jetson%20%7C%20RK3576%20%7C%20RK3588%20%7C%20RPi4%2F5-76b900.svg)](#supported-devices)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="media/hero.png" alt="OpenVoiceStream — sub-200ms streaming ASR + TTS on edge" width="640" />
</p>

<!-- TODO: Add demo GIF showing voice-in → text → voice-out round-trip -->

OpenVoiceStream is a streaming voice stack — ASR and TTS — built for real-time conversational pipelines on edge devices. It deploys as a single container, runs entirely on-device, and exposes a clean WebSocket / HTTP API that doesn't change when you swap engines or hardware.

## Why OpenVoiceStream

| 🪶 **Few dependencies** | 📦 **Easy to deploy** | ⚡ **Low latency** |
| --- | --- | --- |
| Runs on lean inference engines — sherpa-onnx, TensorRT-EdgeLLM, RKNN — with no heavy ML framework at runtime. No PyTorch, no transformers in the hot path. | One Docker image (~900 MB on Jetson, ~600 MB on Rockchip). Models auto-download on first start. `docker run` and you're talking. | **~110ms** ASR + TTS time-to-first-audio on Jetson Orin NX. ~3–5x faster than cloud APIs, and the latency floor is yours to tune, not someone else's API queue. |

## Quick Start

Pull and run. Models auto-download on first start (~1 min) and cache in a Docker volume:

```bash
# Chinese + English (default)
docker run -d --name openvoicestream \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -v openvoicestream-models:/opt/models \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-nvidia-libs:/host-libs:/host-cuda \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v3.0-slim

# Verify (wait ~40s for warmup)
curl http://localhost:8621/health
# {"asr":false,"tts":true,"streaming_asr":true}
```

**English-only mode** (Kokoro TTS + Zipformer ASR):

```bash
docker run -d --name openvoicestream \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -e LANGUAGE_MODE=en \
  -v openvoicestream-models:/opt/models \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-nvidia-libs:/host-libs:/host-cuda \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v3.0-slim
```

**Deploy with compose** (recommended for production):

```bash
git clone --recurse-submodules https://github.com/suharvest/seeed-local-voice.git
cd seeed-local-voice

# Chinese + English (default)
docker compose -f deploy/docker-compose.yml up -d

# English only
LANGUAGE_MODE=en docker compose -f deploy/docker-compose.yml up -d

# Qwen3 multilingual ASR/TTS (52 langs + voice clone)
SEEED_LOCAL_VOICE_PROFILE=jetson-multilang-highperf \
QWEN3_HF_REPO_ID=<your-org/qwen3-edgellm-jetson-artifacts> \
docker compose -f deploy/docker-compose.yml up -d
```

## Table of Contents

- [Why OpenVoiceStream](#why-openvoicestream)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Qwen3 Multilingual Path](#qwen3-multilingual-path)
- [Performance](#performance)
- [Configuration](#configuration)
- [Models](#models)
- [Supported Devices](#supported-devices)
- [Patched sherpa-onnx](#patched-sherpa-onnx)
- [Project Structure](#project-structure)
- [Changelog](#changelog)
- [Acknowledgements](#acknowledgements)

## Key Features

- **Streaming-first** — WebSocket ASR with partial results, sentence-level streaming TTS. Built around how dialogue actually flows, not request/response.
- **Sub-200ms latency** — ~110ms ASR + TTS time-to-first-audio (Chinese+English), ~180ms (English-only). 3–5x faster than cloud APIs.
- **Engine-light** — sherpa-onnx for the bilingual path, TensorRT-EdgeLLM for Jetson multilingual, RKNN for Rockchip NPU. No PyTorch at runtime.
- **One-command deploy** — `docker run` and the service comes up. Models auto-download. No build chain on the target device.
- **Stable API across backends** — same WebSocket / HTTP contract for sherpa, Qwen3, and the Rockchip path. Swap models without changing client code.
- **Multilingual options** — Chinese+English (Paraformer + Matcha-TTS), English-only (Zipformer + Kokoro v1.0, 53 voices), or 52-language Qwen3-ASR + Qwen3-TTS with voice cloning.
- **Lean footprint** — ~650 MB RAM, ~32% CPU idle on Jetson Orin NX. Leaves room for an LLM and other workloads on the same device.
- **Fully on-device** — no cloud, no API keys, no internet required at runtime.

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  Edge device (Jetson Orin / RK3576 / RK3588 / RPi 4–5)    │
│                                                           │
│  FastAPI service (:8000)                                  │
│  ├── WS /asr/stream    Streaming ASR                      │
│  │     └─ zh_en: Paraformer  │  en: Zipformer  │  multi: Qwen3-ASR │
│  ├── POST /asr          SenseVoice offline ASR (zh+en)    │
│  ├── POST /tts          Batch TTS                         │
│  └── POST /tts/stream   Streaming TTS                     │
│        └─ zh_en: Matcha-TTS  │  en: Kokoro v1.0  │  multi: Qwen3-TTS │
│                                                           │
│  Inference: sherpa-onnx · TRT-EdgeLLM · RKNN              │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   Any client (SBC, laptop, robot, kiosk, ...)
```

Models are selected automatically based on `LANGUAGE_MODE`:

| Service | Endpoint | zh_en (default) | en | multilingual | Protocol |
|---------|----------|-----------------|-----|---------------|----------|
| **Streaming ASR** | `WS /asr/stream` | Paraformer bilingual | Zipformer English | Qwen3-ASR (52 langs) | WebSocket: int16 PCM in, JSON out |
| **Streaming TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, raw PCM stream |
| **Batch TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, WAV out |
| Offline ASR | `POST /asr` | SenseVoice (zh+en+ja+ko+yue) | SenseVoice (same) | Qwen3-ASR (52 langs) | HTTP: WAV upload, JSON out |

**Backend capabilities differ:**

| Backend | Speed control | Pitch shift | Voice clone | Languages | Streaming |
|---------|--------------|-------------|-------------|-----------|-----------|
| Sherpa (zh_en/en) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |
| Qwen3 (multilingual) | ❌ | ❌ | ✅ (x-vector) | 52 | ✅ |
| RKNN (Rockchip) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap engines without changing client code. Unsupported parameters return `501` with `{"required_capability": "..."}`.

## API Reference

### Streaming ASR (WebSocket)

```
WS /asr/stream?sample_rate=16000&language=auto
```

- Client sends: raw **int16 PCM bytes** (audio chunks, e.g. 100ms each)
- Client sends: **empty bytes** `b""` to signal end of audio
- Server sends: JSON `{"text": "...", "is_final": bool, "is_stable": bool}`

```python
import asyncio, websockets

async def transcribe():
    async with websockets.connect("ws://device:8000/asr/stream?sample_rate=16000") as ws:
        for chunk in audio_chunks:  # np.int16 arrays
            await ws.send(chunk.tobytes())
            result = await ws.recv()  # partial results
        await ws.send(b"")  # signal end
        final = await ws.recv()  # {"text": "...", "is_final": true}
```

### Offline ASR (HTTP)

```bash
curl -X POST http://device:8000/asr \
  -F "file=@recording.wav" -F "language=auto"
# {"text": "transcribed text"}
```

### TTS (HTTP)

```bash
curl -X POST http://device:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 52, "speed": 1.0}' \
  --output output.wav
```

Parameters: `text` (required), `sid` (speaker ID, default 52), `speed` (rate, default 1.0)

**Note:** `speed` and `pitch` parameters only work in the Sherpa backend (`zh_en`/`en` mode). Qwen3-TTS (`multilingual` mode) does not support speed/pitch adjustment — these are Sherpa-specific capabilities.

### TTS Streaming (HTTP)

Returns raw PCM: first 4 bytes = sample rate (uint32 LE), then int16 samples.

```
POST /tts/stream
Content-Type: application/json
{"text": "Hello world", "sid": 52}
```

### Health Check

```
GET /health  →  {"asr": bool, "tts": bool, "streaming_asr": bool}
```

## Qwen3 Multilingual Path

`LANGUAGE_MODE=multilingual` enables Qwen3-ASR + Qwen3-TTS — 52 languages plus voice cloning. The integration code lives in this repo; Qwen-specific export, engine builds, and worker glue are maintained in the standalone companion repo [`suharvest/qwen3-edgellm-jetson`](https://github.com/suharvest/qwen3-edgellm-jetson) (pinned here as a submodule at `third_party/qwen3-edgellm-jetson/`). Large model artifacts live in [`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts) on Hugging Face.

**Quickest path on a fresh Orin NX:**

```bash
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh \
  --reference /path/to/24kHz_mono.wav   # optional: gates the voice-clone path
```

The orchestrator builds the runtime, downloads + SHA-256-verifies the HF artifacts, builds the slim docker image, starts the service, and runs the verifier (`scripts/verify_reproduction.sh`). Exit 0 means the slim container on port 18092 is healthy and serving the validated stack.

**Two runtime profiles** under the same API surface:

| Profile | Goal | Default behavior |
|---------|------|------------------|
| `official` | Minimal-diff EdgeLLM example. Close enough to upstream that it can be reviewed or upstreamed as a Qwen3 ASR/TTS example. | Semantic/correctness fixes only — tokenizer layout, sampling, runtime contract, stream callback. Regular exported Talker/CodePredictor/Code2Wav directories. |
| `highperf` (default) | Product low-latency dual-resident path for Orin. | Full vocab, ASR FP8 embedding, TTS W8A16 Talker, CP BF16 I/O + `lm_head` pretranspose, stateful Code2Wav, CP decode CUDA graph, `ACTIVE_CP_GROUPS=13`. |

Use `jetson-multilang-highperf-nx` on Orin NX when consuming the NX-native engine set; the default `jetson-multilang-highperf` profile targets the Nano artifact set. Profiles in [`configs/profiles`](configs/profiles) set env defaults only; explicit env vars still override them.

For detailed branch ownership, engine env vars, frozen-baseline numbers, and artifact handling, see [`docs/plans/qwen3-current-frozen-baseline-2026-05-10.md`](docs/plans/qwen3-current-frozen-baseline-2026-05-10.md).

## Performance

### Cross-device comparison (measured 2026-05-13)

Full measured numbers across Jetson Orin Nano / Radxa ROCK 5T (RK3588) / RK3576 / Raspberry Pi 5, plus a use-case-driven device picker, live in **[`docs/performance-comparison.md`](docs/performance-comparison.md)**.

Headline summary:

| Use case | Best fit | Why |
|---|---|---|
| Best English accuracy + voice clone | **Jetson Orin Nano** | 0 % WER short English, only platform with voice cloning |
| Best Chinese accuracy | **Radxa RK3588** | 2.6 % CER short, 5 % Long-EN WER |
| Lowest cost (zh+en commands) | **RPi5** | Real-time streaming, $80 BOM |
| Multilingual ASR (ja/ko/es/de/fr) | **RK3576** | Qwen3 ASR works; TTS still WIP |

Cross-device Finalize-RTF and CER/WER tables, full TTS results, V2V latency, and concurrency numbers are in the comparison doc above.

### Latency on Jetson Orin NX 16GB (representative baseline)

| | ASR TTFT | TTS TTFT | ASR + TTS | vs Cloud |
|---|---------|---------|-----------|----------|
| **zh_en** (Chinese+English) | ~50ms | ~60ms | **~110ms** | 3–5x faster |
| **en** (English only) | ~50ms | ~130ms | **~180ms** | 2–4x faster |

Full voice-to-voice latency depends on LLM inference time (not included above).

### Resource Usage (Jetson Orin NX 16GB)

| State | CPU (8-core) | RAM | GPU |
|-------|-------------|-----|-----|
| Idle (models loaded) | ~32% | ~650 MB | 0% |
| During TTS inference | ~63% | ~658 MB | burst |
| During ASR streaming | ~32% | ~650 MB | minimal |

The service uses only ~650 MB RAM, leaving plenty of headroom for LLM inference (Ollama), computer vision, or robot control on the same device.

### Benchmarks

**zh_en mode:**

| Metric | Value |
|--------|-------|
| Paraformer TTFT | ~50ms |
| Paraformer finalize | ~45ms |
| Paraformer accuracy | 80.8% (26 synthetic sentences) |
| Matcha TTS TTFT | ~60ms (short text) |
| Matcha TTS latency | ~150ms (typical Chinese sentence) |

**en mode:**

| Metric | Value |
|--------|-------|
| Zipformer TTFT | ~50ms |
| Kokoro TTS TTFT | ~130ms (short text) |
| Kokoro TTS latency | ~300ms (typical sentence) |

### TTS Model Comparison

We evaluated 4 TTS models for TTFT (time-to-first-audio-chunk). Matcha-TTS was selected for zh_en mode (best Chinese quality), Kokoro for en mode (best English quality):

| Model | TTFT (short) | TTFT (long) | Chinese Quality | English Quality | Used in |
|-------|-------------|-------------|-----------------|-----------------|---------|
| **Matcha-TTS + Vocos** | ~60ms | ~150ms | Good | Fair | **zh_en** |
| **Kokoro v1.0** | ~130ms | ~300ms | — | Excellent | **en** |
| CosyVoice3 | ~800ms | ~2s | Excellent | — | — |
| F5-TTS | ~2.5s | ~5s | Excellent | — | — |

Benchmark scripts live in `benchmarks/`. See `benchmarks/archive/` for detailed F5-TTS optimization experiments (CUDA, TensorRT, NFE sweep).

### Performance Tuning

Run once after boot on Jetson to lock clocks to max:

```bash
sudo ./setup-performance.sh
```

This sets MAXN power mode, locks CPU/GPU clocks, and disables dynamic frequency scaling. Critical for consistent inference latency.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGUAGE_MODE` | `zh_en` | `zh_en` (Chinese+English), `en` (English only), or `multilingual` (Qwen3, 52 langs) |
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `52` | Default TTS speaker ID (52=af_cute, 3=af_heart) — Sherpa only |
| `TTS_DEFAULT_SPEED` | `1.0` | TTS playback speed — **Sherpa only** |
| `TTS_NUM_THREADS` | `4` | TTS inference threads |
| `TTS_PITCH_SHIFT` | `0` | Pitch shift in semitones — **Sherpa only** |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice language hint |
| `STREAMING_ASR_PROVIDER` | `cuda` | Streaming ASR execution provider |
| `MODEL_DIR` | `/opt/models` | Model storage directory |

Copy `.env.example` to `.env` to customize.

## Models

Auto-downloaded on first start and cached in a Docker volume:

| Model | Size | Mode | Purpose |
|-------|------|------|---------|
| Paraformer streaming zh-en | ~230 MB | `zh_en` | Streaming ASR (bilingual) |
| Matcha-TTS + Vocos zh-en | ~125 MB | `zh_en` | TTS synthesis |
| Zipformer streaming en | ~65 MB | `en` | Streaming ASR (English only) |
| Kokoro TTS v1.0 | ~719 MB | `en` | TTS synthesis (English, 53 speakers) |
| SenseVoice zh-en-ja-ko-yue | ~500 MB | both | Offline ASR (5 languages) |
| Qwen3-TTS 0.6B + TRT engines | ~2.5 GB | `multilingual` | TTS + voice clone (52 languages) |
| Qwen3-ASR encoder + decoder | ~1.5 GB | `multilingual` | ASR (52 languages, streaming) |

## Supported Devices

OpenVoiceStream is validated on the following hardware. Any device in the same class should work; these are the ones we measure against.

| Device class | Validated on | Notes |
|---|---|---|
| **NVIDIA Jetson Orin** | Jetson Orin Nano 8GB, Orin NX 16GB, AGX Orin | CUDA 12.6 / JetPack 6.2. Full feature set including Qwen3 multilingual + voice clone. |
| **Rockchip NPU** | Radxa ROCK 5T (RK3588), Banana Pi BPI-M5 Pro (RK3576) | RKNN runtime. zh+en path mature; multilingual Qwen3-ASR works, Qwen3-TTS WIP. |
| **Raspberry Pi (CPU)** | Raspberry Pi 5 8GB, Raspberry Pi 4 4GB | CPU inference. Lowest BOM (~$80). Real-time zh+en commands. |

Requirements: Docker + ~5 GB disk for models + ~650 MB free RAM. On Jetson, NVIDIA Container Runtime is required; on Rockchip, the host NPU driver (`rknpu`) must be loaded.

## Patched sherpa-onnx

OpenVoiceStream ships a patched sherpa-onnx that fixes Paraformer streaming tail truncation (the stock version drops the last 1–3 characters). The patch:

1. **IsReady()** — forces decode of remaining frames after `InputFinished()`
2. **DecodeStream()** — zero-pads partial final chunks
3. **CIF force-fire** — emits residual tokens at end-of-stream

Pre-built `.so` files live in `patches/sherpa-onnx-lib/` (aarch64, Python 3.10, CUDA 12.6). See `patches/README.md` for rebuild instructions.

## Project Structure

```text
seeed-local-voice/
├── app/                     # FastAPI service
│   ├── main.py              # Endpoints and startup
│   ├── backends/            # Per-engine backends (sherpa / jetson / rk)
│   ├── core/                # VAD, ASR/TTS contracts, streaming primitives
│   └── model_downloader.py  # On-demand model download + voice patching
├── voices/                  # Custom voice embeddings (auto-patched into model)
├── benchmarks/              # TTS model TTFT comparisons
├── bench/                   # Streaming + V2V latency benchmarks (perf harness)
├── patches/                 # Paraformer EOF truncation fix
├── scripts/                 # Model download, ORT patching, engine build glue
├── deploy/
│   ├── docker-compose.yml   # Production deploy (pre-built image)
│   └── docker/
│       ├── Dockerfile.jetson  # Jetson Orin Nano/NX/AGX (zh_en or multilingual)
│       ├── Dockerfile.rk      # Rockchip RK3576/RK3588 NPU
│       └── Dockerfile.rpi     # Raspberry Pi 4/5 (CPU)
├── third_party/             # Submodules (independently maintained)
│   ├── qwen3-edgellm-jetson # Qwen3 export + engine build for Jetson
│   └── rkvoice-stream       # Rockchip NPU streaming voice runtime
└── setup-performance.sh     # Jetson clock/power tuning
```

Clone with `--recurse-submodules` to pull `third_party/*`, or run `git submodule update --init --recursive` after cloning.

## Changelog

### v3.0-slim

- **95% smaller image** — 898 MB vs 17.7 GB. Multi-stage build extracts only the runtime Python packages (onnxruntime + sherpa-onnx) into an `ubuntu:22.04` base
- **Host GPU library mounts** — CUDA/TensorRT/cuDNN libraries are bind-mounted from the host JetPack installation instead of baked into the image, improving cross-JetPack compatibility
- **Same performance** — identical TTS/ASR latency and CUDA provider support (TRT + CUDA + CPU)

> **Note:** v3.0-slim is derived from v2.2 (the full image). There is no standalone v3.0 fat image — `v3.0-slim` is the current production release. This version requires host GPU lib mounts at runtime (see Quick Start). This is standard practice for Jetson containers and matches the pattern used by vision-trt and other optimized images.

### v2.2

- **Endpoint detection** — server proactively sends `is_final` when the speaker pauses (0.6s trailing silence), reducing response latency
- **Fix WebSocket lifecycle** — properly close connections after finalize to prevent stale connection reuse
- **Production deploy compose** — `deploy/docker-compose.yml` with pre-built image (no build step needed)

### v2.1

- Streaming TTS with sentence-level callback
- Custom voice embedding support via pitch shift

### v2.0

- Initial release: Paraformer + Matcha (zh_en), Zipformer + Kokoro (en)
- Patched sherpa-onnx for Paraformer streaming EOF fix

## Contributing

Issues and PRs are welcome. The most useful contributions:

- New backend integrations (other NPUs, other inference engines)
- Streaming benchmarks on additional hardware
- Bug reports with reproducible audio samples and `LANGUAGE_MODE` / profile info
- Documentation improvements, especially deployment recipes for new devices

If you're working on a larger change, open an issue first to align on the approach. Sub-project changes (Qwen3 export, Rockchip runtime) belong in their own repos: [`qwen3-edgellm-jetson`](https://github.com/suharvest/qwen3-edgellm-jetson), [`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream).

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — speech inference engine powering the bilingual ASR and TTS paths
- [next-gen Kaldi](https://github.com/k2-fsa) — research foundation behind sherpa-onnx
- [Paraformer](https://github.com/modelscope/FunASR) — streaming bilingual ASR model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — fast flow-matching TTS (zh+en mode)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — high-quality English TTS with 53 speakers (en mode)
- [Zipformer](https://github.com/k2-fsa/icefall) — efficient transducer ASR (en mode)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multilingual offline ASR
- [Qwen3](https://huggingface.co/Qwen) — multilingual ASR + TTS foundation model (52-language path)
- [TensorRT-EdgeLLM](https://github.com/NVIDIA/TensorRT-LLM) — Jetson inference runtime for the Qwen3 path
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) — Rockchip NPU runtime for the RK3576/RK3588 path
