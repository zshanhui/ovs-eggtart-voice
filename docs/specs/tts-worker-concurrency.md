# Spec: Qwen3 TTS Worker — Configurable Concurrency

**Status**: Draft — open for assignment
**Owner (placeholder)**: TBD
**Target platforms**: Jetson Orin Nano (8GB), Jetson Orin NX (16GB), future Jetson AGX
**Estimated effort**: 3–5 days (familiar dev), 1–2 weeks (unfamiliar)

---

## 1. Background

The voice-agent conversation flow streams long text through TTS in multiple segments. Each segment is currently dispatched to the C++ TTS worker **strictly serially**:

- Python side: `threading.Lock` (`_worker_lock`) wraps each request — see
  `app/backends/jetson/trt_edge_llm_tts.py:480`, used at `:610`, `:781`, `:971`.
- Worker side: `while (std::getline(std::cin, line))` main loop — see
  `third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_tts_worker.cpp:326`.
  One request is fully processed (talker prefill → code-predictor →
  Code2Wav streaming → terminal `done` event) before the next is read.

**Consequence**: after segment N's `done` arrives, segment N+1 must run its
own ~150–300 ms talker prefill **before** the first audio chunk emerges. The
playback queue drains during that window, producing an audible mid-sentence
gap.

`_split_tts_text` (`trt_edge_llm_tts.py:228`) hard-caps CJK segments at 16
characters (env `EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS`) because the stateful
Code2Wav engine is sensitive to long no-punctuation runs. Raising the cap
risks distortion. Therefore the only clean fix is to make the worker
**pipeline** segment N+1's prefill against segment N's tail-streaming.

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

`third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_tts_worker.cpp`:

| Line | What |
|------|------|
| `:285` | `int main(...)` |
| `:299` | `cudaStreamCreate(&stream)` — main CUDA stream (talker + code_predictor + code2wav default) |
| `:300` | `cudaStreamCreate(&code2wavStream)` — second stream, currently only used when `asyncCode2Wav` (currently hard-coded false at `:341`) |
| `:308` | `Qwen3OmniTTSRuntime ttsRuntime(...)` — single instance, owns talker + code_predictor engines, KV-cache buffers, CUDA-graph state |
| `:314` | `Code2WavRunner code2wavRunner(...)` — single instance (or lazy) |
| `:326` | main loop — `std::getline(std::cin, line)` then `process_one_request(line)` |
| `:374` | `TalkerGenerationResponse talkerResponse` — created per request, holds the generated code-frame stream |
| `:478` | `chunkThread` — optional helper thread for async Code2Wav (already exists but gated by `asyncCode2Wav=false`) |

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
- One TRT execution context for Code2Wav (or pool 1 shared, depending on
  stateful-vs-stateless mode)
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
        try:
            req_id = payload["id"]
            q = queue.Queue()
            with self._inflight_lock:
                self._inflight[req_id] = q
            with self._stdin_lock:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            while True:
                event = q.get(timeout=60.0)   # raises Empty on hang
                yield event
                if event.get("event") == "done":
                    return
        finally:
            with self._inflight_lock:
                self._inflight.pop(req_id, None)
            self._sem.release()

    def _reader_loop(self):
        for line in self._proc.stdout:
            event = json.loads(line)
            rid = event.get("request_id") or event.get("id")
            with self._inflight_lock:
                q = self._inflight.get(rid)
            if q is not None:
                q.put(event)
            # else: stale / unsolicited, drop
```

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
JSON lines.

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
| Talker execution context (TRT) | ~50 MB |
| **Per-slot total** | **~530 MB** |

Free GPU memory after baseline load (measured from container logs):

- Orin Nano 8GB: ~1.4 GB → **N=1 max safely**, N=2 borderline
- Orin NX 16GB: ~6 GB → **N=2 comfortable, N=3 ok, N=4 tight**

Always validate with `nvidia-smi` (Jetson: `tegrastats`) under load. The
spec author **must** measure actual memory delta per slot before
publishing recommended defaults.

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | CUDA stream sync errors when sharing engine across contexts | Each slot gets its own execution context; cross-stream sync only via `cudaStreamSynchronize` at slot boundaries |
| 2 | CUDA graph capture is per-execution-context — must recapture per slot at startup | Add per-slot warmup pass after slot construction; increases startup time by `N × ~5s` |
| 3 | Stdout interleaving — JSON lines get mixed | Single `cout_mutex` wrapping every event write |
| 4 | TensorRT engine thread-safety | TensorRT permits concurrent `enqueueV3` on different contexts and streams from different threads — verify with current TRT version on Jetson (`nvidia-smi --query`) before relying on it |
| 5 | OOM at N=3+ on smaller boards | Validate at startup: probe free memory, refuse to launch if `N × per_slot > available - safety_margin` |
| 6 | Stateful Code2Wav state leak across slots | Each slot has its own state; never shared |
| 7 | Existing async-Code2Wav (`asyncCode2Wav=false` today) helper thread conflicts with new scheduler thread | Pick one model — either keep async-Code2Wav per-slot, or eliminate the asyncCode2Wav branch entirely. Recommend eliminating since per-slot CUDA streams give us the parallelism asyncCode2Wav was reaching for |

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

- Current worker source: `third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_tts_worker.cpp`
- Python IPC: `app/backends/jetson/trt_edge_llm_tts.py` (lines 470-1100)
- TRT runtime classes: `Qwen3OmniTTSRuntime`, `Code2WavRunner` — defined in
  `third_party/qwen3-edgellm-jetson/native/edgellm_voice/`
- Build script: `deploy/jetson-release-highperf.sh` (rebuilds worker
  binary; landed via image rebuild — confirm with deployment owner)

---

**Acceptance criteria for the assignee:**

1. With `OVS_TTS_WORKER_CONCURRENCY=1`, all existing tests pass and audio
   output is byte-identical to the pre-change worker for the same input.
2. With `OVS_TTS_WORKER_CONCURRENCY=2` on Orin NX, the dashboard
   conversation has measurably fewer mid-sentence gaps (qualitative A/B
   demo plus a quantitative measurement: instrument the worker to log the
   gap between `done` of segment N and the first `chunk` of segment N+1,
   should drop from current ~200ms to <50ms).
3. Memory under load stays below 90% of board capacity for both N=1 and
   N=2 configurations.
4. Restarting the worker mid-stream surfaces clean errors to all in-flight
   Python callers (no hangs).
