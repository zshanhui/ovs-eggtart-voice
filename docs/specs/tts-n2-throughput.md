# Spec: Unlock Real N=2 TTS Throughput on Jetson Orin NX

## §1 Goal & non-goals

Goal: safely lift `/tts/stream` from effective N=1 to N=2 on Orin NX, while preserving N=1 correctness. The physical target is not 2.0x; expected parallel TTS ceiling is ~1.4-1.5x. Gates: single-client audio MD5 remains `f515a4376962cca876f21089130d7253`; dual-client slow-client TTFA <= 765 ms, using 510 ms as the N=1 baseline; 100 dual-client iterations produce zero CUDA illegal-access / `cudaMemsetAsync` failures; 5-minute soak shows no VRAM growth. Non-goals: multi-process workers, AGX N=4 tuning, ASR changes, or changing audio quality/sampling.

## §2 Architecture review

Live request path: FastAPI imports `Request` already (`app/main.py:10`) and `/tts/stream` accepts it (`app/main.py:750-751`). The endpoint splits text into sentences (`app/main.py:776-779`), acquires BackendManager for the manager path (`app/main.py:780-785`), yields the 4-byte sample-rate header (`app/main.py:794-798`), then submits sync `backend.generate_streaming()` work into `_get_tts_stream_executor()` (`app/main.py:804-817`). Legacy path repeats the same executor/queue pattern (`app/main.py:845-871`).

Current HTTP concurrency is pinned: `_get_tts_stream_executor()` constructs `ThreadPoolExecutor(max_workers=1)` and documents the N>1 CUDA crash (`app/main.py:251-275`). The NX profile already sets `OVS_TTS_WORKER_CONCURRENCY=2`, so the env is dormant behind the Python cap (`configs/profiles/jetson-multilang-highperf-nx.json:20-22`).

Python worker IPC is mostly N-ready: `_WorkerIO` has a stdin lock, per-request queues, reader thread, and semaphore (`app/backends/jetson/trt_edge_llm_tts.py:486-517`). It inserts the queue before writing stdin (`trt_edge_llm_tts.py:526-540`), demuxes by `request_id`/`id` (`trt_edge_llm_tts.py:584-600`), treats `done` and `cancelled` as terminal (`trt_edge_llm_tts.py:549-553`), and can write cancel messages (`trt_edge_llm_tts.py:559-575`). `_generate_streaming_single()` builds streaming request JSON (`trt_edge_llm_tts.py:1081-1173`), obtains `_WorkerIO` under lifecycle lock only (`trt_edge_llm_tts.py:1189-1193`), yields chunks (`trt_edge_llm_tts.py:1218-1232`), and sends cancel on `GeneratorExit` (`trt_edge_llm_tts.py:1255-1269`).

C++ worker dispatch is also mostly wired: `readConcurrencyEnv()` reads `OVS_TTS_WORKER_CONCURRENCY` (`qwen3_tts_worker.cpp:555-587`), `Code2WavSlotPool` hands out counted slots (`qwen3_tts_worker.cpp:483-523`), per-slot Code2Wav vectors and streams are sized to concurrency (`qwen3_tts_worker.cpp:633-699`), and the main loop parses cancel before capacity wait (`qwen3_tts_worker.cpp:1391-1439`) before spawning bounded request threads (`qwen3_tts_worker.cpp:1446-1474`). Cancel state is shared via `cancelMap` (`qwen3_tts_worker.cpp:452-467`) and checked before chunk vocoder enqueue (`qwen3_tts_worker.cpp:963-976`).

Engine slot pools exist: `TalkerSlot` and `CodePredictorSlot` own per-request tensors/contexts (`qwen3OmniTTSRuntime.h:257-352`), engine capacity reads the same env (`qwen3OmniTTSRuntime.cpp:591-624`), and talker/code-predictor pools eager-init/free-list slots (`qwen3OmniTTSRuntime.cpp:865-891`, `qwen3OmniTTSRuntime.cpp:2099-2121`). Concurrency safety is not fully proven: `handleAudioGeneration()` still uses shared runtime tensors such as `mTalkerLogits`, `mTalkerHiddenStatesBuffer`, `mSeenCodecTokensBuf`, and `mCodecHiddensBuffer` directly (`qwen3OmniTTSRuntime.cpp:4441-4490`, `qwen3OmniTTSRuntime.cpp:4560-4603`, `qwen3OmniTTSRuntime.cpp:4638-4701`, `qwen3OmniTTSRuntime.cpp:4908-4910`). This is a second audit item before declaring runtime N-safe.

## §3 Part D disconnect watcher — implementation plan

Current gap: Starlette cancellation does not reliably close the inner sync generator, and the manager branch has no `request.is_disconnected()` polling around `stream()` (`app/main.py:794-826`); legacy branch has the same absence (`app/main.py:845-873`). Implement the watcher inside each `StreamingResponse` async generator lifetime, scoped only to `/tts/stream`.

Plan: keep `Request` parameter as-is (`app/main.py:750-751`). For each sentence, create the sync generator object explicitly, run a drain function in `_get_tts_stream_executor()`, and have a `threading.Event`/cancel flag checked between chunks. Add an asyncio task polling `await request.is_disconnected()` every 100 ms. On disconnect, set the flag, call `gen.close()` from the drain thread cleanup path, and let `_generate_streaming_single()` translate `GeneratorExit` into `worker_io.cancel(req_id)` (`trt_edge_llm_tts.py:1255-1269`). Do this in both manager and legacy branches. Add a temporary counter around `_WorkerIO.cancel()` (`trt_edge_llm_tts.py:559-575`); stress should show cancel calls >= early-break requests and zero CUDA poison.

## §4 StatefulCode2WavRunner audit + fix plan

Exact failing call site: `StatefulCode2WavRunner::reset()` zeroes `state.read` and `state.write` with `cudaMemsetAsync(..., stream)` (`statefulCode2WavRunner.cpp:254-263`). Constructor sets TensorRT profile on the constructor stream, synchronizes it, allocates buffers, then calls `reset(stream)` (`statefulCode2WavRunner.cpp:55-97`). State buffers are created as GPU `rt::Tensor`s in `allocateBuffers()` (`statefulCode2WavRunner.cpp:154-223`), tensor addresses are bound once (`statefulCode2WavRunner.cpp:226-241`), and `generateChunk()` later enqueues on the caller stream and swaps state tensor objects/addresses (`statefulCode2WavRunner.cpp:396-428`). Worker slot 0 uses the main stream; slots >0 create `c2wStream` lazily (`qwen3_tts_worker.cpp:844-856`) and pass it to constructor/reset/chunk generation (`qwen3_tts_worker.cpp:990-1008`, `qwen3_tts_worker.cpp:1092-1097`).

Hypotheses: H1 buffer allocation/first binding has stream-affinity; partially unproven because `rt::Tensor` construction takes no stream, but profile/address setup does use constructor stream. H2 `swapStateBuffers()` rebinding while a prior enqueue is incomplete is unsafe if a runner is reused too early; less likely because each slot is guarded. H3 runtime global tensors in `handleAudioGeneration()` race before Code2Wav and surface later at reset; plausible from shared tensor citations in §2.

Fix candidates, ranked: (a) surgical: make StatefulCode2WavRunner own a fixed per-slot stream, use it for profile set, reset, memcpy, enqueue, and synchronize; do not accept arbitrary streams in `reset/generateChunk` except assert-equal. ~50 lines, lowest risk. (b) revert per-slot Code2Wav to singleton plus explicit per-instance state buffer pool; medium risk, walks back `qwen3_tts_worker.cpp:469-478`. (c) full thread-safe Code2Wav rewrite with explicit state objects and context rebinding discipline; highest risk. Recommended: do (a) plus audit/fix `handleAudioGeneration()` shared tensors before lifting max_workers.

## §5 Test plan

Extend `bench/perf/load_2client_tts.py` for true simultaneous `/tts/stream` starts, TTFA per client, chunk counts, and stderr-log scan. Gates: 100 dual-client iterations, zero `cudaMemsetAsync`/illegal-access errors; slow-client TTFA <= 765 ms; 5-minute soak with stable VRAM; single-client MD5 remains `f515a4376962cca876f21089130d7253`. Keep N=1 early-break stress and require cancel counter >= disconnect count.

## §6 Risks & rollback

The watcher could affect legacy long-poll/WebSocket behavior, so scope it only inside `/tts/stream`; do not touch WS endpoints. StatefulCode2Wav changes can regress N=1 byte output, so MD5 gate must pass before changing `max_workers`. Rollback binary remains `/opt/jv-workers/qwen3_tts_worker.phase3bb4p1.bak` per investigation notes.

## §7 Sequencing & estimated effort

Phase A: Part D watcher, Python only, 0.5-1 day. Phase B: Code2Wav plus runtime shared-state audit/fix, C++, 1-2 days. Phase C: lift `_tts_stream_executor` to 2 and validate N=2, 0.5 day. Phase D: add independent `inter_segment_gap_ms` bench metric, 0.5 day.
