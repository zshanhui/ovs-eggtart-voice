# Spec: Qwen3 TTS Worker — Configurable Concurrency

**Status**: Approved (2026-05-21, after two codex review rounds) — open for assignment
**Owner (placeholder)**: TBD
**Target platforms**: Jetson Orin Nano (8GB), Jetson Orin NX (16GB), future Jetson AGX
**Estimated effort**: 3–5 days (familiar dev), 1–2 weeks (unfamiliar)

---

## 1. Background

The voice-agent conversation flow streams long text through TTS in multiple segments. Each segment is currently dispatched to the C++ TTS worker **strictly serially**:

- Python side: `threading.Lock` (`_worker_lock`) wraps each request — see
  `app/backends/jetson/trt_edge_llm_tts.py:480`, used at `:610`, `:781`, `:971`.
- Worker side: `while (std::getline(std::cin, line))` main loop — see
  `examples/omni/qwen3_tts_worker.cpp:557` **in `suharvest/TensorRT-Edge-LLM`**.
  One request is fully processed (talker prefill → code-predictor →
  Code2Wav streaming → terminal `done` event) before the next is read.

> **Source-of-truth note (added after dev-agent investigation 2026-05-22):**
> The TTS worker source lives in the `TensorRT-Edge-LLM` fork
> (`https://github.com/suharvest/TensorRT-Edge-LLM`, `examples/omni/qwen3_tts_worker.cpp`,
> 1000 lines as of commit `6239d5f`), **not** in the `qwen3-edgellm-jetson`
> submodule. The toolkit submodule at
> `third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_tts_worker.cpp`
> contains a stale 687-line snapshot that is not part of the deployment
> build pipeline; treat it as deprecated. The ASR worker is the opposite —
> its source of truth is `qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_asr_worker.cpp`
> (1416 lines). See the sibling ASR spec for the ASR side.
>
> All file:line references below refer to `examples/omni/qwen3_tts_worker.cpp`
> in the TensorRT-Edge-LLM fork at the commit listed above. The
> implementation agent's first step is to confirm the line numbers still
> match the current HEAD of `suharvest/TensorRT-Edge-LLM:main`.

**Consequence**: after segment N's `done` arrives, segment N+1 must run its
own ~150–300 ms talker prefill **before** the first audio chunk emerges. The
playback queue drains during that window, producing an audible mid-sentence
gap.

`_split_tts_text` (`app/backends/jetson/trt_edge_llm_tts.py:249-251`)
currently defaults CJK segments to **48 characters**
(`EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS`, matching the profile setting on Orin
NX). Raising the cap further risks distortion in the stateful Code2Wav
engine. With the cap already at the operational maximum, segment count is
already minimized — the only remaining lever to close the inter-segment
gap is to make the worker **pipeline** segment N+1's prefill against
segment N's tail-streaming.

## 2. Goal

Add a single deployment knob:

```yaml
# in profile or .env
OVS_TTS_WORKER_CONCURRENCY: 1   # default — Orin Nano keeps current behavior
OVS_TTS_WORKER_CONCURRENCY: 2   # Orin NX: pipeline depth 2
OVS_TTS_WORKER_CONCURRENCY: 4   # AGX Orin or future hardware
```

**Concurrency = N** means: up to N requests may be in various stages
(prefill / generate / code2wav streaming) inside the worker at once. The
Python side may have up to N requests in flight without blocking on
`_worker_lock`.

**Non-goals**: this spec does **not** propose multiple worker *processes*.
The TRT engine weights (~3 GB) are read-only and shared across requests;
duplicating processes was already considered and rejected (OOM on Orin Nano,
unnecessary on others).

## 3. Current architecture (read these before designing)

### 3.1 Python side

`app/backends/jetson/trt_edge_llm_tts.py`:

| Line | What |
|------|------|
| `:479` | `self._worker: Optional[subprocess.Popen] = None` |
| `:480` | `self._worker_lock = threading.Lock()` |
| `:687` | `_ensure_worker()` — spawns the C++ worker, reads `{"event": "ready"}` |
| `:712` | `subprocess.Popen([worker_binary, ...], stdin=PIPE, stdout=PIPE, stderr=...)` |
| `:781` | one-shot synth path — sends one request inside `with self._worker_lock` |
| `:971` | streaming synth path — same lock, but reads multiple chunk events before `done` |
| `:850` | `generate_streaming(text)` — entry point used by v2v; recursively calls itself per `_split_tts_text` segment, each call entering `_generate_streaming_single` which takes the lock |

The Python protocol is one JSON-per-line on stdin → many JSON-per-line on
stdout. Each chunk event has `{"event": "chunk", ...}`; the terminal event
is `{"event": "done", ...}`. **There is no `request_id` in the stdout
events today** — responses are inferred to belong to the current in-flight
request because the protocol is serial.

### 3.2 Worker side

`examples/omni/qwen3_tts_worker.cpp` in `suharvest/TensorRT-Edge-LLM`
(commit `6239d5f` baseline, 1000 lines):

| Line | What |
|------|------|
| `:443` | `int main(int argc, char** argv)` |
| `:459` | `cudaStreamCreate(&stream)` — primary CUDA stream (talker + code_predictor + main code2wav) |
| `:463-465` | `std::unique_ptr<Code2WavRunner> code2wavRunner`, `statefulCode2wavRunner`, plus `asyncCode2WavStream` declared at `:465` (lazy-init) |
| `:466` | `std::unique_ptr<Code2WavRunner> asyncCode2wavRunner` — a SECOND Code2Wav runner used only when a request opts in via `async_code2wav=true` (created lazily on first use via `getAsyncCode2WavRunner` at `:473-481`) |
| `:510` | `statefulCode2wavRunner = std::make_unique<StatefulCode2WavRunner>(...)` — when stateful mode enabled at process startup |
| `:520` | `code2wavRunner = std::make_unique<Code2WavRunner>(...)` — when stateless mode is the startup choice |
| `:553` | first `std::cout` — `{"event":"ready", "init_ms": ...}` |
| `:557` | main loop — `while (std::getline(std::cin, line))` |
| `:572` | per-request `async_code2wav` flag read from request JSON (default false) |
| `:593` | branch: `statefulCode2Wav && asyncCode2Wav` — different code path that bypasses the stateful runner for async |
| `:627, :654` | chunk JSON build + stdout emit |
| `:670` | lazy re-construction of `StatefulCode2WavRunner` inside the request loop (when stateful kwargs change between requests) |
| `:711` | same for stateless `Code2WavRunner` |
| `:761-827` | async-Code2Wav request path — spawns work onto `asyncCode2WavStream` via `synthesizeWindow(asyncRunner, ...)` |
| `:916, :946` | done JSON build + stdout emit |
| `:990` | error / end-of-request response emit |
| `:996, :998` | shutdown `cudaStreamDestroy` for async stream + primary stream |

Per-request state that must be **isolated** if we run two requests in
parallel:

- Talker KV cache (lives inside `ttsRuntime` — currently single instance)
- Code-predictor state
- `talkerResponse` / `streamedFrames` (already local variables, fine)
- Stateful Code2Wav internal buffers (when `EDGE_LLM_TTS_STATEFUL_CODE2WAV=1`)
- CUDA graph capture state (talker prefill uses a captured graph — capture
  is per-runtime, runtime is shared)

The TRT engines themselves (weights / plans) are **read-only**; multiple
invocations against the same engine on different CUDA streams are
allowed by TensorRT IFF you use separate execution contexts.

## 4. Proposed design

### 4.1 High-level

Add a worker-internal scheduler that owns a fixed pool of `N` "execution
slots". Each slot has:

- One TRT execution context for talker
- One TRT execution context for code predictor
- **One Code2WavRunner instance, owned exclusively by the slot.** The
  existing worker shares a single `Code2WavRunner` across requests
  (`qwen3_tts_worker.cpp:311-315`, reset inline at `:393-397, :605-608`).
  With N>1 slots running concurrently, sharing this runner would interleave
  stateful Code2Wav LSTM state across requests and corrupt audio. Per-slot
  ownership is **mandatory**, not optional — both stateful and stateless
  modes must allocate one runner per slot. Memory budget in §6 reflects N
  copies of the runner.
- One CUDA stream
- A reusable KV-cache buffer sized for `talker_max_seq_len`
- Stateful Code2Wav state buffer (when enabled)

Slot allocation:

```
on request received:
    slot = scheduler.try_acquire()  # non-blocking
    if slot is None:
        # all N busy — wait or queue
        slot = scheduler.acquire_blocking()
    spawn worker_thread(slot, request)

worker_thread(slot, request):
    run prefill on slot.stream
    stream chunks → stdout (tagged with request_id)
    emit done (tagged with request_id)
    slot.release()
```

Stdout writes from N threads must be serialized — wrap `std::cout` with a
mutex. Each JSON event already carries enough fields; just add
`"request_id": "<id>"`.

### 4.2 Python side changes

Two layers.

**(a) `_worker_lock` removal + per-request reader.**

Today's `with self._worker_lock` block both writes the request AND reads
all subsequent stdout events. With concurrency this must split: writer
side serialized; reader side demultiplexed by `request_id`.

Sketch:

```python
class _WorkerIO:
    def __init__(self, proc, concurrency: int):
        self._proc = proc
        self._stdin_lock = threading.Lock()
        self._inflight: dict[str, queue.Queue] = {}
        self._inflight_lock = threading.Lock()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._sem = threading.Semaphore(concurrency)

    def request(self, payload: dict) -> Iterator[dict]:
        self._sem.acquire()
        req_id = payload["id"]
        q = queue.Queue()
        # CRITICAL ordering: insert into map BEFORE writing to stdin, so
        # the reader thread can never see an event before the queue exists.
        with self._inflight_lock:
            self._inflight[req_id] = q
        try:
            with self._stdin_lock:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
        except Exception:
            # Write failed — clean up map and semaphore, then re-raise.
            with self._inflight_lock:
                self._inflight.pop(req_id, None)
            self._sem.release()
            raise
        try:
            while True:
                event = q.get(timeout=60.0)   # raises Empty on hang
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError("TTS worker died mid-request")
                yield event
                if event.get("event") == "done":
                    return
        finally:
            with self._inflight_lock:
                self._inflight.pop(req_id, None)
            self._sem.release()

    def _reader_loop(self):
        try:
            for line in self._proc.stdout:
                event = json.loads(line)
                rid = event.get("request_id") or event.get("id")
                with self._inflight_lock:
                    q = self._inflight.get(rid)
                if q is not None:
                    q.put(event)
                # else: stale / unsolicited, drop
        finally:
            # Worker exited (clean or crash). Wake every in-flight caller
            # with a sentinel so they raise WorkerExitError instead of
            # hanging on q.get().
            with self._inflight_lock:
                for q in self._inflight.values():
                    q.put({"event": "_worker_exit"})
                self._inflight.clear()
```

**Invariants the reader/writer pair must maintain:**

1. **Map insert happens before stdin write.** If the writer inserts after
   the write, the reader could see and drop the response before the
   queue exists.
2. **Write failure cleans up.** A failed `stdin.write` (broken pipe,
   worker died) must pop the map entry and release the semaphore before
   raising, or the slot leaks.
3. **Reader-exit sentinel.** When the reader loop exits (stdout EOF =
   worker process died), every queue gets a `_worker_exit` event so
   waiting callers raise `WorkerExitError`. Without this, callers hang
   on `q.get(timeout=60.0)` for 60 s per call before noticing the
   worker is dead.
4. **Semaphore is released exactly once per `request()` call.** Either
   the success path's `finally`, or the write-failure cleanup — never
   both, never neither.

**(b) Existing `generate_streaming` / `_generate_streaming_single` use
`_WorkerIO.request(payload)` instead of taking `_worker_lock` directly.**
Concurrency limit comes from the semaphore. Default 1 = exactly today's
behavior (one in flight, others block).

### 4.3 Worker side changes

Phase the work to keep risk bounded:

**Phase 1 (small) — Request-ID plumbing**

Add `"request_id"` (alias `id`) to every stdout event. No concurrency yet.
Python keeps `_worker_lock`. Verifies the protocol change in isolation.

Change points:
- All `std::cout << ... << std::endl` sites — wrap with `emitEvent(id, json)`
  helper that ensures the id is set.
- Add `"request_id"` to the `ready` event (use a literal `"__worker__"`).

**Phase 2 (medium) — Per-request CUDA stream + execution-context pool**

Inside `Qwen3OmniTTSRuntime`, refactor so each invocation can use:
- a passed-in `cudaStream_t` (today the runtime captures one in its
  constructor)
- a passed-in execution context (today single per-engine context)

Keep N=1 default. Verify single-stream behavior unchanged.

**Phase 3 (medium) — Scheduler**

```cpp
class Scheduler {
public:
    Scheduler(int n, ...) {
        for (int i = 0; i < n; ++i) {
            slots_.emplace_back(make_slot(...));
        }
    }
    Slot* acquire();  // blocks until one free
    void release(Slot*);
private:
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<bool> busy_;
};

struct Slot {
    cudaStream_t stream;
    std::unique_ptr<TalkerExecutionContext> talkerCtx;
    std::unique_ptr<CodePredictorExecutionContext> cpCtx;
    std::unique_ptr<Code2WavRunner> c2w;  // may share weights, own state
    std::vector<int32_t> kvCacheBuffer;
    StatefulCode2WavState c2wState;
};
```

Main loop becomes:

```cpp
while (std::getline(std::cin, line)) {
    auto req = parseRequest(line);
    Slot* slot = scheduler.acquire();
    std::thread([slot, req = std::move(req)]() {
        processRequest(slot, std::move(req));
        scheduler.release(slot);
    }).detach();
}
```

Wrap `std::cout` writes with a mutex so concurrent threads don't interleave
JSON lines. **Audit every existing stdout site** — not just chunk events.
Current direct writes (in `examples/omni/qwen3_tts_worker.cpp` of
`suharvest/TensorRT-Edge-LLM`) that must route through the new
`emitEvent(rid, kind, json)` helper:

- `:553` — `ready` event
- `:654` — chunk emit (JSON built at `:627`)
- `:946` — `done` event (JSON built at `:916`)
- `:990` — error / end-of-request emit
- All `std::cerr` sites (`:72, :214, :541`) are info/usage/error logs —
  verify they stay on stderr; they do not need the mutex but the audit
  must explicitly classify each.

A grep gate in CI: `grep -nE '(std::cout|std::printf)' examples/omni/qwen3_tts_worker.cpp`
must return zero results outside the `emitEvent()` helper.

**Phase 4 (small) — Config knob**

Read `OVS_TTS_WORKER_CONCURRENCY` env or `--concurrency` flag at startup.
Validate `1 <= N <= 4`. Print resolved value in `ready` event:

```json
{"event": "ready", "init_ms": 1234, "concurrency": 1}
```

### 4.4 IPC protocol diff

Minimal additions:

```diff
 Request (stdin, one JSON per line):
 {
   "id": "<unique>",   ← already present, just must be unique now
   "text": "...",
   ...
 }

 Response events (stdout, one JSON per line):
 {
+  "request_id": "<id from request>",
   "event": "chunk" | "done" | "error",
   ...
 }
```

Python's `_WorkerIO._reader_loop` uses `request_id` to demux. Keep `id` as
an alias for back-compat with older Python clients during rollout.

## 5. Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OVS_TTS_WORKER_CONCURRENCY` | `1` | Max in-flight requests inside one worker process |

Surface in profiles:

```json
// configs/profiles/jetson-multilang-highperf.json (Orin Nano)
"OVS_TTS_WORKER_CONCURRENCY": "1"

// configs/profiles/jetson-multilang-highperf-nx.json (Orin NX)
"OVS_TTS_WORKER_CONCURRENCY": "2"
```

The Python `_WorkerIO` reads the same env and sizes its semaphore. If
Python's value > worker's actual capacity, the worker blocks-acquires
internally — slight overshoot is fine but document the contract: the env
**must match** on both sides (or operator must set Python's value ≤
worker's).

## 6. Memory budget

Per concurrent slot:

| Component | Approx. Orin NX FP8/FP16 |
|---|---|
| Talker KV cache (max seq 2048, 28 layers, FP16) | ~400 MB |
| Code-predictor state | ~50 MB |
| Stateful Code2Wav state (when enabled) | ~30 MB |
| Code2WavRunner instance (per §4.1 — required per slot) | ~50 MB |
| Talker execution context (TRT) | ~50 MB |
| **Per-slot total** | **~580 MB** |

Free GPU memory after baseline load (measured from container logs):

- Orin Nano 8GB: ~1.4 GB → **N=1 max safely**, N=2 not viable
- Orin NX 16GB: ~6 GB → **N=2 comfortable, N=3 tight, N=4 not viable**

**Validation requirement (blocking before profile bump):** The
implementation author must add `[JV_MEM]` log points at slot construction
in `qwen3_tts_worker.cpp` and capture per-slot deltas under
`tegrastats`. The per-slot estimates above are spec-time hypotheses;
the N=2 profile rollout (Plan §9 step 4) cannot proceed without measured
deltas confirming the budget.

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | CUDA stream sync errors when sharing engine across contexts | Each slot gets its own execution context; cross-stream sync only via `cudaStreamSynchronize` at slot boundaries |
| 2 | CUDA graph capture is per-execution-context — must recapture per slot at startup | Add per-slot warmup pass after slot construction; increases startup time by `N × ~5s` |
| 3 | Stdout interleaving — JSON lines get mixed | Single `cout_mutex` wrapping every event write |
| 4 | TensorRT engine thread-safety | TensorRT permits concurrent `enqueueV3` on different contexts and streams from different threads — verify with current TRT version on Jetson (`nvidia-smi --query`) before relying on it |
| 5 | OOM at N=3+ on smaller boards | Validate at startup: probe free memory, refuse to launch if `N × per_slot > available - safety_margin` |
| 6 | Stateful Code2Wav state leak across slots | Each slot has its own state; never shared |
| 7 | Existing async-Code2Wav path is real, not dead code | The production worker has a runtime-selectable `async_code2wav` kwarg (per-request, default false) at `cpp:572`; when set, work routes to a SECOND `Code2WavRunner` on a SECOND CUDA stream (`cpp:466, :473-481, :761-827`). The new scheduler must (a) own the async runner per-slot (so two slots running with `async_code2wav=true` don't collide on a shared async stream) OR (b) deprecate the async path entirely since per-slot streams subsume it. Recommend (b) but verify no current caller depends on the async path before removing |
| 8 | Hot-reload (`/admin/backend/reload`) drain semantics change with N>1 in-flight | Today `unload()` terminates the worker under `_worker_lock` (`trt_edge_llm_tts.py:630-643`). With the new design, `unload()` must wait for all `_inflight` queues to drain (or hit `drain_timeout_s`), then kill the worker and rely on the reader-exit sentinel to wake any remaining callers with `WorkerExitError`. Document: drain timeout must be ≥ longest expected per-segment synth (~2 s) × concurrency to avoid spurious aborts |
| 9 | Reader thread sees event before queue exists | §4.2 mandates "insert map entry before stdin write" ordering. Without this, the writer could send a request, the reader could process the first event, and find no queue — event dropped, caller hangs on `q.get` forever |
| 10 | Sampling defaults (`temperature`, `top_p`, `top_k`) make output non-deterministic | Plan §7 byte-identical acceptance requires pinned seed + greedy sampling. Existing profile already sets `OVS_TTS_TALKER_TEMPERATURE=0.0`, `OVS_TTS_TALKER_TOP_K=1`, `OVS_TTS_SEED=42` (`jetson-multilang-highperf-nx.json:40-43`); regression tests must assert these are set or the byte-identical claim is moot |

## 8. Testing plan

### Unit (worker)

- `test_worker_concurrency_1.cpp` — N=1 matches current behavior byte-for-byte
  (regression). Same input WAV md5 as today's worker.
- `test_worker_concurrency_2.cpp` — N=2, fire 4 requests rapid-fire, verify
  all 4 audio outputs match the N=1 ground truth (byte-identical, since
  temperature=0 top_k=1 deterministic).
- `test_worker_scheduler_overflow.cpp` — fire 8 requests with N=2, verify
  the 5th-8th block-then-process correctly.

### Integration (Python side)

- `test_trt_edge_llm_tts_concurrency.py` — instantiate backend with
  `OVS_TTS_WORKER_CONCURRENCY=2`, fire two streaming requests on separate
  threads, verify audio outputs identical to serial calls.
- Cancel test: cancel one request mid-stream, verify other completes.
- Restart test: kill worker mid-stream of one request, verify both
  Python callers see `WorkerExitError`.

### End-to-end (deployment)

- Orin NX, `OVS_TTS_WORKER_CONCURRENCY=2`: run dashboard, speak a long
  reply, listen for mid-sentence gaps. Compare to N=1.
- Memory probe: `tegrastats` for 10 minutes under load, peak GPU memory
  must be under 90% of board capacity.
- TTFA / latency: measure first-chunk-ms across 50 turns, N=1 vs N=2.
  N=2 must not regress TTFA (per-request latency); only the inter-segment
  gap should shrink.

## 9. Rollout

1. Land Phase 1 (request_id) behind no flag — protocol additive, safe to
   ship in same image.
2. Land Phase 2-4 (concurrency machinery) gated by `OVS_TTS_WORKER_CONCURRENCY=1`
   default (= no behavior change). Image ships everywhere.
3. Bump Orin NX profile to `2`. Measure for 1 week.
4. Bump AGX profile to `4` after AGX validation.
5. Keep Orin Nano at `1` until/unless explicit validation done.

## 10. Open questions (for assignee)

- Does TensorRT 10.x allow N execution contexts to share weights from the
  same engine across N streams? (Believed yes; verify with TRT docs of the
  exact version in the Jetson image.)
- CUDA-graph capture: are captures **per-execution-context** or
  per-engine? If per-engine, only one slot can capture at a time and the
  others must fall back to eager.
- Stateful Code2Wav: is the internal LSTM hidden-state buffer
  re-initializable, or does it need to be allocated once and never
  released? (Affects whether slots can dynamically resize.)
- Should the Python side fall back to serial mode (semaphore=1) if the
  worker reports `concurrency=1` in its `ready` event, even when the
  operator sets a larger Python-side value? Recommended: yes.

## 11. Reference

- Current worker source: `examples/omni/qwen3_tts_worker.cpp` in
  `suharvest/TensorRT-Edge-LLM` (the production fork). The repo is checked
  out on Mac at `/Users/harvest/project/tensorrt-edge-llm` and on orin-nx
  at `/home/harvest/TensorRT-Edge-LLM`. **NOT** in the
  `qwen3-edgellm-jetson` submodule — that copy is stale (see §1
  source-of-truth note).
- Python IPC: `app/backends/jetson/trt_edge_llm_tts.py` (lines 470-1100)
- TRT runtime classes: `Code2WavRunner`, `StatefulCode2WavRunner` — header
  `tensorrt_edge_llm/multimodal/statefulCode2WavRunner.h` (referenced at
  `cpp:11`). `Qwen3OmniTTSRuntime` may live elsewhere; the implementor
  must locate it under the TensorRT-Edge-LLM tree.
- Build entry: `cmake` against the TensorRT-Edge-LLM tree. There is no
  in-repo build script that consumes this source directly today; the
  deployed binary in `deploy/jetson-workers/qwen3_tts_worker` (60.7 MB,
  baked into the Jetson image via Dockerfile `COPY`) was produced
  out-of-band by `cmake --build` inside the TensorRT-Edge-LLM tree on
  orin-nx. The implementor needs to (a) establish a reproducible build,
  (b) document it in `deploy/jetson-release-highperf.sh` or a new
  `deploy/build-qwen3-tts-worker.sh`, and (c) refresh
  `deploy/jetson-workers/qwen3_tts_worker` from the new build.

---

**Acceptance criteria for the assignee:**

1. With `OVS_TTS_WORKER_CONCURRENCY=1`, all existing tests pass and audio
   output is byte-identical to the pre-change worker for the same input,
   given pinned greedy sampling (Plan §7 criterion 1). "Pre-change
   worker" = the production binary built from the **current HEAD** of
   `suharvest/TensorRT-Edge-LLM:main` BEFORE this spec's edits land —
   NOT the stale 687-line toolkit version. The implementor records the
   sha256 of the pre-change binary before starting Phase 1 and uses it
   as the regression baseline.
2. With `OVS_TTS_WORKER_CONCURRENCY=2` on Orin NX, the dashboard
   conversation has measurably fewer mid-sentence gaps (qualitative A/B
   demo plus a quantitative measurement: instrument the worker to log the
   gap between `done` of segment N and the first `chunk` of segment N+1,
   should drop from current ~200ms to <50ms).
3. Memory under load stays below 90% of board capacity for both N=1 and
   N=2 configurations.
4. Restarting the worker mid-stream surfaces clean errors to all in-flight
   Python callers (no hangs).
