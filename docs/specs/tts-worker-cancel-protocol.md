# Spec: TTS Worker Cooperative Cancel Protocol

**Status**: Draft — pending codex review
**Owner (placeholder)**: TBD
**Target**: Jetson Orin NX + future devices
**Scope**: Phase 4 of TTS production-hardening — independent of the
deferred N=2 throughput work
**Estimated effort**: 2–3 days

---

## 1. Background — what bug is this fixing

Empirically observed on Orin NX, 2026-05-22:

1. Client calls `POST /tts/stream`, reads the first PCM chunk, **closes
   the connection early** (canonical triggers: browser-tab close,
   voice-agent barge-in interrupting AI mid-response, wstest4-style
   bench client breaking on first audio).
2. Server-side: the Python `_run` callable inside the ThreadPoolExecutor
   keeps iterating `backend.generate_streaming` because nothing tells
   it the consumer is gone. The C++ worker keeps emitting chunks. The
   chunks queue up in the Python side's `asyncio.Queue` but no async
   coroutine drains them (the `StreamingResponse` iterator was
   abandoned by Starlette on disconnect).
3. The C++ worker eventually finishes the full ~2 s generation,
   emitting ~70 chunks → its stdout pipe fills → worker stdin reader
   thread blocks on stdout write → its CUDA work continues
   asynchronously while the worker is logically wedged.
4. **At some point** during this disconnect cleanup, a subsequent
   request (or even just the next CUDA call from the wedged worker)
   triggers:

```
CUDA runtime error in cudaMemsetAsync(state.read.rawPointer(),
                                       0,
                                       state.read.getMemoryCapacity(),
                                       stream):
   an illegal memory access was encountered
```

5. Once that fires, the CUDA context is poisoned: every subsequent CUDA
   call in that process returns the same illegal-access error. The
   worker process is functionally dead but still emits "error" events
   to Python — it stays alive as a zombie until manually restarted.

The same family of bug bit the LLM SSE endpoint earlier and was fixed
in `TensorRT-Edge-LLM` fork commit `49c94ff fix(server): cancel TRT
stream channel on client disconnect`. That fix is the design template
for this spec — apply the same pattern to the TTS pipeline.

## 2. Goal

**Make `/tts/stream` robust to mid-stream client disconnects at N=1.**

When the HTTP client closes connection mid-stream:
- The C++ TTS worker stops generating within ≤ 1 chunk boundary
  (~30–100 ms) of receiving the cancel signal.
- The worker's CUDA work completes naturally to the next chunk
  boundary, then the slot is released cleanly. No half-completed
  CUDA enqueue, no abandoned scratch buffer, no stdout pipe
  back-pressure.
- The next `POST /tts/stream` request lands on a fully-cleaned
  worker — no CUDA context poisoning, no zombie slot, no stale
  state.

This spec is **N=1 only**. The `_tts_stream_executor` stays at
`max_workers=1` (current). N=2 throughput is a deferred follow-up
that builds on this work.

**Non-goal**: throughput. Cancel correctness only.

## 3. Current architecture (read these before designing)

### 3.1 Python side
`app/backends/jetson/trt_edge_llm_tts.py`:
- `_WorkerIO.request(payload)` (line ~519) — generator that writes one
  JSON per call to worker stdin, then yields stdout events demuxed by
  `request_id`. Cleanup on exit (the `finally` block at ~554) pops the
  inflight queue and releases the semaphore.
- `_generate_streaming_single` (line ~1149) — iterates the worker's
  events, yields PCM chunks. Currently has **no** cancel path: if the
  caller (the executor's `_run` callable) abandons the generator, the
  `finally` runs but doesn't notify the worker to stop.

`app/main.py`:
- `@app.post("/tts/stream")` (line ~741) — the SSE-style endpoint.
- The async `stream()` generator (line ~785) submits `_run` to
  `_get_tts_stream_executor()` (currently max_workers=1) and awaits
  chunks via `asyncio.Queue`.
- **No client-disconnect detection** today. When the client closes,
  Starlette cancels the outer StreamingResponse task; the inner
  `_run` thread keeps running until natural completion.

### 3.2 C++ worker side
`examples/omni/qwen3_tts_worker.cpp`:
- Main loop (line ~1140) reads JSON-per-line from stdin, spawns a
  worker thread per request (Phase 3b-B-2 dispatch).
- `handleRequest` lambda (line ~667) — the per-request body. Includes
  the chunk emit loop and emits one `chunk` JSON per Code2Wav output.
- **No cancel-flag plumbing.** A request runs to natural completion
  regardless of consumer state.

### 3.3 Reference fix (LLM SSE)
`tensorrt-edge-llm/experimental/server/engine.py:780` (commit `49c94ff`):
- `LLM.generate_stream()` finally block calls `channel.cancel()` to
  trigger cooperative cancel in the C++ TRT runtime.
- `cpp/runtime/streaming.cpp:96-219` — `StreamChannel::cancel()` sets
  `mCancelled=true`, wakes waiters; the runtime checks via
  `applyCancellationToFinishStates()` at decode-loop iteration
  boundaries.

`tensorrt-edge-llm/experimental/server/api_server.py` (same commit):
- SSE endpoint refactored from sync generator to async with a
  background drain thread + asyncio task polling
  `request.is_disconnected()` at 100 ms.

We're applying this pattern to the TTS pipeline.

## 4. Proposed design

### 4.1 IPC protocol — new message type

**Python → C++ worker (new):**
```json
{"type": "cancel", "id": "<request_id>"}
```

Written via the existing `_WorkerIO._stdin_lock` (so it can't tear-write
mid-request-JSON).

**C++ worker → Python (new event):**
```json
{
  "event": "cancelled",
  "request_id": "<request_id>",
  "id": "<request_id>",
  "ok": true,
  "reason": "client_disconnect"
}
```

Replaces the natural `done` event for cancelled requests. **Critical
(codex round-1 must-fix)**: `ok` MUST be `true`, NOT `false`. Cancellation
is an expected control-flow event, not an error. The current Python
`_generate_streaming_single` body at `trt_edge_llm_tts.py:1181` raises
`RuntimeError` on any event with `ok:false`. If cancelled events were
`ok:false`, every clean cancel would surface as an exception to the
caller, masking real errors and breaking downstream cleanup.

Counted as a terminal event by `_WorkerIO.request()` — the generator
returns. The Python side must explicitly handle `event=="cancelled"`
as terminal-non-error in BOTH:
- `_WorkerIO.request()` at `trt_edge_llm_tts.py:552` (where today only
  `event=="done"` returns from the loop)
- `_generate_streaming_single` at `trt_edge_llm_tts.py:1181` (where
  the `if not event.get("ok"): raise` check must precede the
  `event=="cancelled"` check, OR — cleaner — the cancelled event
  must be checked BEFORE the ok-flag check)

### 4.2 C++ worker changes — `examples/omni/qwen3_tts_worker.cpp`

1. Add a process-wide map keyed by `request_id` → `std::atomic<bool>*`
   pointing to the cancel flag owned by each in-flight worker thread:
   ```cpp
   std::mutex cancelMapMu;
   std::unordered_map<std::string, std::atomic<bool>*> cancelMap;
   ```
2. In `handleRequest`, declare a local `std::atomic<bool> cancelled{false};`
   register it in `cancelMap` keyed by request_id; unregister on
   scope exit via RAII.
3. The chunk emit loop (`emitChunk` / `synthesizeStatefulChunk` call
   sites) checks `cancelled.load(std::memory_order_acquire)` between
   chunks. If set:
   - Break out of generation
   - Emit `{"event":"cancelled","request_id":id,"ok":true,
     "reason":"client_disconnect"}` via `emitEvent` (Phase 1 helper).
     **`ok:true`** per §4.1 — cancel is a normal terminal event,
     not an error.
   - Skip the natural `done` event
   - RAII cleanup releases the slot, the Code2Wav state buffers stay
     in-place (next `reset()` clears them)
4. Main loop (line ~1140) — **CRITICAL ordering (codex round-1 must-fix)**:
   today the dispatcher waits for `inFlight < concurrency` BEFORE
   parsing the JSON (`qwen3_tts_worker.cpp:1281-1294`). At
   `concurrency=1`, a cancel message arriving while one request is
   in flight blocks at the capacity wait → deadlock (the cancel can
   never reach the worker thread it's supposed to interrupt).
   
   Fix: parse the JSON's `type` field FIRST. If `type=="cancel"`:
   - Look up the request_id in `cancelMap`
   - Set the atomic if found
   - `continue` the main loop without touching the capacity counter
   - Do NOT spawn a worker thread
   
   Only after confirming it's a generation request (no `type` field
   or `type=="generate"`) do we wait on the capacity cv and spawn
   a worker thread.

**Atomicity note**: the cancel-write happens on the main thread (stdin
reader). The cancel-read happens on the worker thread. `std::atomic<bool>`
with default sequential ordering is the safe minimum; explicit
acquire/release is the perf-tuned version.

**Critical**: do NOT try to abort kernels mid-enqueue. CUDA has no such
API. The cancel takes effect at the NEXT chunk emit, after the
currently-enqueued GPU work finishes. This is the "cooperative" part —
worst-case cancel latency = one chunk-emit duration (~30–100 ms).

### 4.3 Python side — `_WorkerIO.cancel(req_id)`

`app/backends/jetson/trt_edge_llm_tts.py`:

```python
def cancel(self, req_id: str) -> None:
    """Best-effort cancel for an in-flight request.

    Writes a cancel JSON to the worker's stdin. The worker will check
    its per-request atomic flag at the next chunk boundary and emit
    a `{"event":"cancelled", ...}` terminal event in lieu of `done`.

    Safe to call from any thread. Safe to call after the request has
    naturally completed (worker will silently drop unknown cancels).
    """
    try:
        with self._stdin_lock:
            self._proc.stdin.write(
                json.dumps({"type": "cancel", "id": req_id}) + "\n"
            )
            self._proc.stdin.flush()
    except Exception:
        # Worker may have already exited; the reader-loop sentinel
        # will surface that via WorkerExitError on the next q.get().
        logger.debug("cancel() write failed; worker may be exiting",
                     exc_info=True)
```

### 4.4 Python side — `_generate_streaming_single` cleanup

In `app/backends/jetson/trt_edge_llm_tts.py` around line 1149:

The streaming generator currently iterates `worker_io.request(req)`.
If the consumer (the executor's `_run` callable in `app/main.py`)
breaks out of the for loop early, Python's generator-cleanup invokes
`__del__` / `.close()` → `GeneratorExit` is raised at the next `yield`
point.

Add a `try / except GeneratorExit` around the chunk-emit `yield` such
that the cleanup calls `worker_io.cancel(req_id)` before re-raising.
This is the **bridge** from "abandoned generator" to "send cancel to
worker."

### 4.5 Python side — `/tts/stream` disconnect detection

Mirror the LLM SSE fix at `tensorrt-edge-llm/experimental/server/api_server.py:_generate_stream_sse`.

**First (codex round-1 must-fix)**: add `Request` import and parameter:
- `app/main.py:10` — add `from fastapi import Request` (Request is
  NOT currently imported in this module)
- `app/main.py:751` — change `async def tts_stream(req: TTSRequest):`
  to `async def tts_stream(req: TTSRequest, request: Request):`. The
  parameter must be named `request` AND must NOT be shadowed inside
  the function body (this was the bug codex round-3 caught in the
  LLM SSE fix; `experimental/server/api_server.py:192` originally
  rebound `request` to a TRT request object, silently breaking the
  watcher).

The endpoint has TWO code paths (manager-acquired backend and legacy
direct-backend at `app/main.py:822-870`). BOTH must get the disconnect
watcher — they have nearly-identical executor+queue patterns and
share the same disconnect-poisoning risk.

Then, in both branches:
1. Convert the existing `async def stream()` generator into a
   drain-thread architecture:
   - Background thread iterates `backend.generate_streaming`
     synchronously, pushes chunks onto a `queue.Queue`
   - Async generator awaits the queue via
     `loop.run_in_executor(None, queue.get)`
2. Add an asyncio task that polls
   `request.is_disconnected()` every 100 ms while the stream is
   active. **Use the existing FastAPI `Request` parameter** — do NOT
   shadow it with any other variable named `request` (this was the
   bug codex caught in the LLM SSE fix; see round-3 review).
3. On disconnect: set a `stop_flag` event; the drain thread breaks
   out of the for loop and calls `gen.close()`, which triggers the
   `GeneratorExit` path in `_generate_streaming_single` (4.4), which
   sends the cancel to the worker.

**Don't swallow exceptions**: the watcher's exception catch must be
narrow (`asyncio.CancelledError`, expected disconnect errors). Catching
all `Exception` masks programming errors — codex flagged this in the
LLM SSE round-3 review.

## 5. Validation plan

### 5.1 N=1 byte-equivalence regression (gate)
The exhibit-A baseline:
- Audio MD5 `f515a4376962cca876f21089130d7253`, 157440 bytes, sr=24000,
  on input `"我们都非常震惊。这位母亲表示。"`, greedy sampling
  (OVS_TTS_SEED=42, talker_temperature=0.0, talker_top_k=1).

Run via `python3 /tmp/capture_tts_pcm.py` (reads full body, never
early-breaks → no cancel path exercised). MUST match exactly.

### 5.2 Early-break stress (the bug we're fixing)
Run 100 sequential `wstest4`-style early-break requests. Acceptance:
- Each first request after cold restart returns first PCM chunk in
  ~510 ms (current single-client TTFA).
- **Each subsequent request also returns first PCM chunk in ~510 ms.**
  Today this fails — request 2+ returns the 4-byte SR header only.
- `docker logs deploy-speech-1 | grep -iE 'cudamemset|illegal access'`
  returns 0 hits over the 100-iter run.

### 5.3 Multi-turn lifecycle (unchanged)
`bench/perf/smoke_tts_multiturn.py --host 100.82.225.102:8621` must
still PASS 3 rounds.

### 5.4 Cancel latency
With the bench probe modified to capture the time between client-close
and the worker's `cancelled` event (or its slot-release log line):
- Median ≤ 100 ms
- p99 ≤ 200 ms

(One chunk boundary is ~30 ms; one stdin-roundtrip is ~1 ms; one
asyncio-task poll-tick is ≤ 100 ms.)

### 5.5 Long soak
Run the early-break stress for 10 minutes (continuous). VRAM steady
(no leak), `tegrastats` peak under 90 % of board capacity, no errors
in container logs.

### 5.6 Cancel at specific timing points (codex round-1 must-fix)
Three additional targeted gates:
- **Cancel before first PCM chunk**: client connects, reads ONLY the
  4-byte SR header, closes immediately. Worker should receive cancel
  before its first chunk emit. Next request after this must still
  produce baseline audio.
- **Cancel after SR header but before backend iteration starts**:
  `app/main.py:797` yields the SR header BEFORE the backend's
  generator starts. The disconnect watcher must fire even if the
  client breaks at this exact moment.
- **Cancel racing the final chunk**: client breaks just as the worker
  emits its `done` event. The reader-loop must accept either
  `done` or `cancelled` as terminal (whichever arrives first) and
  silently drop the other.

### 5.7 Worker-protocol unit test
Add a unit test in `app/tests/` that asserts:
- `_WorkerIO.request()` treats `"cancelled"` as terminal non-error
  (no raise)
- `_generate_streaming_single` returns cleanly on a `"cancelled"`
  event (no `RuntimeError`)

This test runs without a live worker by using a fake stdin/stdout
pair.

## 6. Rollout

1. Land the C++ worker change first (cancel-map, atomic flag, emit
   `cancelled` event). At this point the binary handles cancel but
   no Python caller sends it yet → behavior unchanged.
2. Land the Python `_WorkerIO.cancel()` helper + `_generate_streaming_single`
   GeneratorExit bridge. Now generator-abandonment sends cancel, but
   nothing detects HTTP disconnect → behavior unchanged for normal
   completion, partial benefit for "abandoned generator" cases.
3. Land the `/tts/stream` disconnect watcher. End-to-end cancel
   working.
4. Validate.
5. Update `bench/perf/smoke_tts_multiturn.py` and `wstest4.py` to be
   the regression suite.

## 7. Risks (residual after design)

1. **Cancel flag check adds branch in hot path** — chunk emit loop
   runs ~70× per request. Branch is highly predictable (always
   false until cancel), so likely cost is single-digit ns total.
   No measurable TTFA impact expected.

2. **Cancel-mid-Code2Wav** — if cancel fires DURING an in-flight
   Code2Wav vocoder enqueue, the chunk completes but we discard the
   output. Wasted GPU but no correctness issue.

3. **Cancel race with natural done** — worker might emit `done` and
   `cancelled` in close succession if cancel arrives at the very
   last chunk. `_WorkerIO.request()` should treat the first
   terminal event (whichever arrives first) as authoritative;
   reader-loop demux drops the second silently.

4. **TRT plugin / TRT context state after early-exit** — if we break
   out of the chunk emit loop with a slot's IExecutionContext
   in mid-enqueue state, the slot's Code2Wav state buffer might
   contain partial-chunk write. Next request on this slot calls
   `reset()` which clears state buffers — safe.

5. **FastAPI / asyncio cancellation cascade** — same risk as LLM SSE
   fix; mitigation is also the same (narrow exception catches,
   no `request` variable shadowing).

## 8. References

- LLM SSE fix design: `TensorRT-Edge-LLM` fork commit `49c94ff`
- Codex round-3 review of that fix:
  `memory/trt_edge_llm_sse_disconnect_pr.md`
- Phase 3 TTS concurrency context:
  `memory/tts_n2_throughput_investigation.md`
- `StreamChannel::cancel` reference impl:
  `tensorrt-edge-llm/cpp/runtime/streaming.cpp:96-219`

---

**Acceptance criteria for the assignee:**

1. After cold container restart, run 100 sequential wstest4-style
   requests with `--runs 5` each (so 500 total early-break events).
   Audio MD5 of a follow-up `capture_tts_pcm.py` run still equals
   `f515a4376962cca876f21089130d7253`. Container log clean of
   `cudaMemset.*illegal` errors.
2. Single-client TTFA p50 within ±10 % of pre-cancel baseline
   (1216 ms via wstest4 on Orin NX with current locked clocks).
3. `bench/perf/smoke_tts_multiturn.py` 3-round PASS.
4. New cancel path covered by at least one integration smoke
   under `bench/perf/`.
