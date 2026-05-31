# ASR N=2 Phase B Patch-Level Spec (codex audit 2026-05-23)

Companion to `asr-worker-concurrency.md` (older design doc) and mirror of TTS Phase B methodology.

## §0 Strategy Decision

**Strategy B — multi-runtime per slot**. Each in-flight ASR session owns its own `LLMInferenceSpecDecodeRuntime` instance + CUDA stream + `AsrSessionState`. Generic NVIDIA runtime unmodified.

| Strategy | Verdict |
|---|---|
| A. Patch generic `LLMInferenceSpecDecodeRuntime` to be thread-safe | ❌ ~50+ shared fields, ~2500 LOC upstream, breaks rebases |
| **B. Multi-runtime per slot** | ✅ Generic runtime untouched, fully isolated, mirrors TTS Phase B scheduler shape |
| C. Worker-level mutex (current) | Already in place implicitly; no throughput benefit |
| D. Wait for upstream batching | No timeline |

## §1 ASR Worker Request Flow

Streaming PCM path (current N=1):
1. Python sends cumulative `pcm_b64` with `id`, `audio_sec`, `last` (`trt_edge_llm_asr.py:1116-1125`)
2. Worker `main()` parses `"chunk"` → `handleChunk(...)` (`qwen3_asr_worker.cpp:1384-1387`)
3. Mel extraction via global `gMelExtractor`, padded to encoder chunk size (`qwen3_asr_worker.cpp:895-896`)
4. Session computes prefix text from prior raw decoded (`qwen3_asr_worker.cpp:1016-1018`)
5. `runStreamingHop()` builds `LLMGenerationRequest` (`qwen3_asr_worker.cpp:620-649`) — sampling `temperature=1.0`, `topP=1.0`, `topK=1` (greedy)
6. `runtime.handleRequest(llmReq, llmResponse, stream)` (`qwen3_asr_worker.cpp:661-662`)
7. Output: prefix + generatedText → emit `partial`/`final`/`segment_rotation`

Current single-session contract enforced at `qwen3_asr_worker.cpp:78-79` (refuses second `begin` during active session).

## §2 Shared Mutable State in `LLMInferenceSpecDecodeRuntime::handleRequest`

Not safe for two host threads to call concurrently on same instance. `SpecDecodeInferenceContext` is fresh per call (`llmInferenceSpecDecodeRuntime.cpp:516-518`), but ALL of these members are mutated during inference:

- `mTokenizer` — chat template + decode (`:500-513`, `:829-832`)
- `mAudioRunner` + output embedding (`:902-917`, `:963-972`)
- `mMultimodalIndices` (`:889-896`, `:1053-1063`)
- `mIdsInput`, `mInputsEmbeds`, `mContextLengthsInput`, `mLogitsOutput`, `mHostPackedTokenIds` (`:997-1024`)
- `mSamplingWorkspace`, `mSamplingIndices`, `mSamplingScores`, `mHostSelectedTokenIds` (`:1133-1158`, `:1608-1635`)
- `mBaseEngineRunner` exec ctx + cache manager (`:1122-1127`, `:1600-1601`)
- Metrics (`:791-799`)
- `mSystemPromptKVCacheBase` / `mSystemPromptKVCacheDraft` (`:1937-1968`, `:2492-2503`)

KV cache is per-batch-slot inside ONE runtime, not per-concurrent-request (`kvCacheManager.h:36-39`, `kvCacheManager.cpp:72-79`). HybridCacheManager owns shared device KV length tensor (`hybridCacheManager.h:158-166`, `hybridCacheManager.cpp:72-76`).

Spec decode adds: `mDraftEngineRunner`, base/draft hidden state tensors, draft token tables, accepted-token tensors (`llmInferenceSpecDecodeRuntime.h:443-475`), all reshaped/reused (`:1174-1248`, `:1278-1308`, `:1431-1490`).

`LLMEngineRunner` documents synchronous-prefill/decode assumption with no continuous batching (`llmEngineRunner.h:80-84`).

## §3 VRAM Budget

ASR artifacts (from `qwen3_checksums.json`):
| Artifact | Size |
|---|---|
| `llm.engine` (thinker) | 1.13GB |
| Embedding | 153MB |
| Audio encoder | 360MB |
| **Total per runtime** | **~1.64GB** |

Constructor loads more than mmap:
- `LLMEngineRunner` mmaps file + deserializes CUDA engine + creates exec context (`llmEngineRunner.cpp:140-153`)
- RoPE cache, HybridCacheManager, seqLens, dummy tensors, optional graph/state (`:198-239`, `:258-305`, `:324-361`)
- Parent runtime: embedding safetensors (`llmInferenceSpecDecodeRuntime.cpp:114-117`), runtime tensors + ctx memory (`:236-349`, `:441-464`)

**N=2 ASR VRAM overhead** ≈ 2 × (engine + embedding + audio enc + KV + runtime + ctx mem + audio runner).

Tight on Orin NX 16GB. Must validate with `tegrastats` before enabling.

## §4 Patch Outline (Strategy B)

### Worker side (`qwen3_asr_worker.cpp`, ~450-800 LOC)

| Line | Change |
|---|---|
| `:42-49` | Parse `OVS_ASR_WORKER_CONCURRENCY` env/arg |
| `:78-80` | Replace single-session guard with slot-pool assignment |
| `:124-170` | Define `AsrSlot` struct (runtime, stream, session, thread, queue) |
| `:179-182` | Per-slot `gMelExtractor` OR mutex around shared one |
| `:612-614` | `runStreamingHop()` accept `AsrSlot&` instead of globals |
| `:831-834` | `handleChunk()` accept `AsrSlot&` |
| `:1158-1159` | One-shot routes through slot 0 or rejects during streaming |
| `:1268-1279` | Replace singleton stream/runtime with N-slot construction loop |
| `:1284-1293` | Replace single `session` with slot scheduler (map of active sessions, idle slot queue) |
| `:1313-1411` | Replace direct dispatch with stdin dispatcher that demuxes by `id` to slot thread |
| `:844,1035,1096,1121,1127,1230,1282,1361,1402` | Stdout through mutexed `emitEvent()` |

### Python side (`trt_edge_llm_asr.py`, ~120-220 LOC)

| Line | Change |
|---|---|
| `:143-156` | Add `_worker_io` field, concurrency env, scope lifecycle lock |
| `:458-483` | Replace `_worker_request()` with `_ASRWorkerIO` (session-lifetime queue demux, modeled on `trt_edge_llm_tts.py:486-617`) |
| `:1077-1225` | Streaming class holds session queue from `begin` through `final/end` |

### Server side (`app/main.py`)

| Change |
|---|
| `_get_asr_executor` env-configurable max_workers (mirror TTS pattern) |
| Optional Part D disconnect watcher for `/asr/stream` |

## §5 Accuracy Gate

ASR is greedy (`qwen3_asr_worker.cpp:653-655` topK=1; spec decode forces greedy `:490-496`). Output deterministic given identical mel.

Baseline test:
1. Fix WAV file → N=1 old worker → capture final transcript = **baseline**
2. N=1 new worker → same WAV → CER ≤ 0.1% vs baseline (strip language prefix per `trt_edge_llm_asr.py:536-564`)
3. N=2 interleaved: two fixed WAVs concurrent → each final matches its single-session baseline

## §6 Risks

- VRAM may OOM at N=2 on NX 16GB (validate with `tegrastats` after first runtime load)
- Multi-mel-extractor: `gMelExtractor` carries internal buffers — needs duplication or locking
- Stdout serialization: dispatcher writes 9+ event types from multiple threads — must use `coutMutex` pattern (mirrors TTS worker C5 emitEvent)
- Hot reload: each slot has own runtime, hot-swap is more invasive than TTS

## §7 Effort + Sequencing

- 3-5 days minimum for stable N=2 (multi-runtime + Python demux + tests)
- 7-10 days for full production (+ hot-reload, drain, automated CI tests)

Phase order:
1. **A1**: Baseline capture (fixed WAV → expected transcript)
2. **A2**: Worker `AsrSlot` struct + N-slot construction (no dispatch change yet, keep N=1 semantically)
3. **A3**: Stdout `emitEvent()` mutex
4. **A4**: Stdin dispatcher (demux by id), thread-per-slot
5. **A5**: Python `_ASRWorkerIO` with session-lifetime demux
6. **A6**: Server `_get_asr_executor` env knob
7. **A7**: N=1 byte-equiv gate (transcript match)
8. **A8**: N=2 stress gauntlet (30+ sustained bursts, CER ≤ 0.1% gate)
9. **A9**: VRAM validation via `tegrastats` under load
