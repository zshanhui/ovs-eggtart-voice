# Spec: Qwen3 ASR Worker — Configurable Concurrency

**Status**: Approved (2026-05-21, after two codex review rounds) — open for assignment
**Owner (placeholder)**: TBD
**Target platforms**: Jetson Orin Nano (8GB), Jetson Orin NX (16GB), future Jetson AGX
**Estimated effort**: 10–15 days (familiar dev), 4–5 weeks (unfamiliar).
Breakdown: Phase 1 ~1d, Phase 2 ~2d, Phase 3 ~6-9d (six sub-steps in
§4.4), Phase 4 ~0.5d, Python-side reload semantics §8.1 ~1d, integration
tests ~1d, on-device validation ~1-2d. The ASR worker is session-oriented
with non-trivial process-global state (see §3.3) that must be slotified
before concurrency is safe — strictly larger scope than the TTS spec.
Do not let the symmetric §4 structure suggest otherwise.
**Sibling spec**: [`tts-worker-concurrency.md`](./tts-worker-concurrency.md) — share the same scheduler pattern but ASR is session-oriented and more invasive

---

## 1. Background

Every `/v2v/stream` WebSocket client opens a long-lived ASR streaming session
that accumulates audio chunks until `asr_eos` or VAD endpoint. The qwen3 ASR
worker today rejects a second session with a hard error:

```cpp
// qwen3_asr_worker.cpp:78-79
// Single-session worker: a `begin` arriving while another session is active
// is refused with {"event":"error","error":"session_already_active"}.
```

Python serializes on `threading.Lock`:

- `app/backends/jetson/trt_edge_llm_asr.py:147` — `self._worker_lock`
- `:149-152` — comment notes the lock spans `stdout.readline()`, which means
  the lock is held for **the entire ASR inference**, not just the stdin write
- Usage sites: `:460`, `:593` (and inside `_worker_request`)

**Consequence**: with two concurrent v2v sessions A and B, B's `begin` is
rejected at the worker level. Python's `_worker_lock` masks this by making
B's `transcribe()` wait until A's session fully closes (`end` event ack'd).
B's first ASR audio chunk therefore stalls for the entire duration of A's
utterance.

This is the **single bottleneck preventing multi-user voice agents on one
device**. Whereas the TTS spec primarily addresses the same-session
inter-segment gap, this ASR spec primarily unblocks multi-session.

## 2. Goal

Add one deployment knob, symmetric to the TTS spec:

```yaml
# in profile or .env
OVS_ASR_WORKER_CONCURRENCY: 1   # default — Orin Nano keeps current behavior
OVS_ASR_WORKER_CONCURRENCY: 2   # Orin NX: 2 simultaneous ASR sessions
OVS_ASR_WORKER_CONCURRENCY: 4   # AGX Orin or future hardware
```

**Concurrency = N** means: up to N independent ASR sessions may be in
progress at once. Each session retains its own KV cache, mel accumulator,
decoded-token state, and prefix cursor. Sessions are entirely independent —
a session_id collision between concurrent v2v clients is the client's
problem, not the worker's.

**Non-goals**:
- Multiple worker processes (same reasoning as TTS spec: ~3GB engine weights
  shared, duplicating processes blows memory)
- Cross-session beam/batch fusion (worth exploring later — would need ASR
  text variance modeling — but out of scope here)
- Changing the single-session streaming semantics (chunk events, segment
  rotation, prefix continuation). Per-session behavior must remain
  byte-identical to N=1 today.

## 3. Current architecture

### 3.1 Python side

`app/backends/jetson/trt_edge_llm_asr.py`:

| Line | What |
|------|------|
| `:147` | `self._worker_lock = threading.Lock()` |
| `:153` | `self._restart_lock = threading.Lock()` — separate so `restart_worker()` can preempt a request thread blocked on `stdout.readline()` |
| `:418` | `_ensure_worker()` — spawns C++ worker, waits for `ready` |
| `:458` | `_worker_request(input_data)` — write one JSON, read one JSON, both inside `_worker_lock` |
| `:485` | `restart_worker()` — snapshot `self._worker` WITHOUT lock, then `proc.kill()` so blocked readline returns |
| `:829` | `create_stream(language)` returns a `_TRTEdgeLLMStreamingASRStream` |
| `:1087` | per-stream `_session_id = uuid.uuid4().hex` — the session_id surface already exists |
| `:1103, :1121, :1173, :1193` | requests carry `"id": self._session_id` |

So Python already plumbs a per-session `session_id` field; the worker
treats it as `id` but ignores it when matching responses (because there's
only one in-flight session by design).

### 3.2 Worker side

`third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_asr_worker.cpp`:

| Line | What |
|------|------|
| `:78-79` | hard-coded single-session rejection |
| `:99` | `kIdleTimeoutMs = 30000` — force-close session after 30s inactivity |
| `:126` | `AsrSessionState::sessionId` field exists but worker tracks **one** AsrSessionState global |
| `:172-176` | `freeSession()` resets all session state |
| `:559` | `computePrefix(session, ...)` — per-session prefix tracking via `chunkId` and `unfixedChunkNum` |

Per-session state that must be **isolated per slot** (heavier than TTS):

- `AsrSessionState` — sessionId, mel accumulator (`pcmAccum` / mel buffers),
  raw decoded tokens (`rawDecoded`), chunkId counter, unfixedChunkNum,
  fullText accumulator (across segment rotations), VAD state
- ASR talker KV cache (per layer × per session, ~150-300 MB on Orin NX with
  current sequence-length budget)
- TRT execution context for the thinker engine
- TRT execution context for the audio encoder
- Tokenizer state (if any per-session decode buffers)
- Idle-timeout timer

The thinker engine weights and audio-encoder weights are **read-only**;
multiple invocations on different streams with separate contexts are
permitted by TensorRT.

### 3.3 Process-global state inventory (BLOCKING for concurrency)

The codex review of the first draft flagged that several process-global
variables in `qwen3_asr_worker.cpp` are not classified for thread safety.
This section enumerates every global and prescribes the disposition under
N>1.

| Symbol | File:line | Today's role | Disposition under N>1 |
|---|---|---|---|
| `gMelExtractor` | `qwen3_asr_worker.cpp:179-182` | Single instance, mel feature extraction | **Read-only after init.** Verify no internal mutable buffers; if any, lift into the slot. If unverifiable, allocate one extractor per slot. |
| `gTimer` (cumulative profiling counters) | `qwen3_asr_worker.cpp:447-479` | Process-wide timer accumulators, read by `ready`/`done` events | **Must become slot-local.** Concurrent hops on N slots would double-count and corrupt every event's `init_ms`/`hop_ms`. Replace with a `Slot::timer` field; emit per-slot values in per-session events; keep one cumulative counter only if explicitly aggregated under a mutex. |
| Tokenizer singleton (`tokenizer::Tokenizer*`) | `:559` (consumer) | Decode + prefix tracking | **Read-only after init.** Confirm tokenizer encode/decode is reentrant; if it caches internally, fall back to per-slot instance. |
| `kIdleTimeoutMs` constant | `:99` | Idle threshold | **No change** — it's a `constexpr`. The timer that *applies* it moves to per-slot. |
| Plugin shim globals (engine/encoder loaders) | (verify locations) | One-time `dlopen` + factory | **Read-only after init.** Same contract as TRT engine weights. |

**Acceptance rule:** before Phase 3 begins, the implementor must produce a
written audit of every `static` and global in `qwen3_asr_worker.cpp` (use
`grep -nE '^(static |extern |[A-Za-z_]+ +[A-Za-z_]+ *= )' qwen3_asr_worker.cpp`
plus a manual review of `:1-300` for namespace-level definitions). The
audit goes in a follow-up PR comment and gates Phase 3 merge.

## 4. Proposed design

### 4.1 High-level

Same scheduler pattern as TTS spec, but slots hold richer per-session
state and a session can occupy a slot for many seconds (whole utterance)
rather than one prefill-then-stream cycle.

```cpp
struct Slot {
    cudaStream_t stream;
    std::unique_ptr<ThinkerExecutionContext> thinkerCtx;
    std::unique_ptr<AudioEncoderExecutionContext> encoderCtx;
    AsrSessionState state;          // entire per-session state
    std::chrono::steady_clock::time_point lastActivity;
};
```

A slot is allocated on `begin`, retained across all `chunk` events for that
session, and released on `end` (or idle timeout). Each slot's worker thread
is event-driven: it sleeps on a session-local queue until the next chunk
arrives via stdin, processes it on `stream`, emits events tagged with the
session_id.

### 4.2 Event routing inside the worker

This is the major design difference from TTS. The worker has multiple
sessions simultaneously sleeping in their slots, and stdin is a single
stream. The dispatcher must demux incoming events by session_id:

```cpp
struct WorkerScheduler {
    std::unordered_map<std::string, Slot*> sessionToSlot;
    std::mutex mapMutex;
    std::vector<std::unique_ptr<Slot>> slotPool;
    std::mutex coutMutex;   // serialize stdout writes

    Slot* acquireFor(std::string const& sessionId);  // begin
    Slot* lookup(std::string const& sessionId);       // chunk / end
    void release(std::string const& sessionId);       // end / timeout
};
```

Main loop:

```cpp
while (std::getline(std::cin, line)) {
    auto req = parseRequest(line);
    if (req.event == "begin") {
        Slot* s = scheduler.acquireFor(req.sessionId);
        if (s == nullptr) {
            emitError(req.sessionId, "slot_exhausted");
            continue;
        }
        // launch worker thread bound to this slot
        threads_.emplace_back([s, req] { handleSession(s, req); });
    } else {
        Slot* s = scheduler.lookup(req.sessionId);
        if (s == nullptr) {
            emitError(req.sessionId, "no_active_session");
            continue;
        }
        s->incomingQueue.push(req);   // worker thread picks up
    }
}
```

Each session worker thread:

```cpp
void handleSession(Slot* s, BeginRequest beginReq) {
    initSession(s, beginReq);
    while (true) {
        auto req = s->incomingQueue.popWithTimeout(kIdleTimeoutMs);
        if (req == nullptr) {           // idle timeout
            emitEvent(s->sessionId, "error", {{"error", "idle_timeout"}});
            break;
        }
        if (req->event == "chunk") {
            processChunk(s, *req);      // emit asr_partial events
        } else if (req->event == "end") {
            finalizeSession(s);          // emit asr_final
            break;
        }
    }
    scheduler.release(s->sessionId);
}
```

Stdout writes from N session threads must be serialized — wrap every event
emit with `coutMutex`. All events already carry session_id (the existing
protocol uses `"id"`), so demux on the Python side is straightforward.

### 4.3 Python side changes

Two layers, mirroring the TTS spec:

**(a) `_worker_lock` removal + per-session reader.**

```python
class _ASRWorkerIO:
    def __init__(self, proc, concurrency: int):
        self._proc = proc
        self._stdin_lock = threading.Lock()
        self._sessions: dict[str, queue.Queue] = {}
        self._sessions_lock = threading.Lock()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True
        )
        self._reader_thread.start()
        self._slot_sem = threading.Semaphore(concurrency)

    def begin_session(self, session_id: str, payload: dict) -> None:
        self._slot_sem.acquire()
        q = queue.Queue()
        # CRITICAL ordering: insert map entry BEFORE writing stdin, so the
        # reader thread cannot see the ack before the queue exists.
        with self._sessions_lock:
            if session_id in self._sessions:
                self._slot_sem.release()
                raise WorkerProtocolError(
                    f"duplicate session_id {session_id} (caller bug)"
                )
            self._sessions[session_id] = q
        try:
            self._write_event(payload)
        except Exception:
            # Write failed — must remove the just-inserted map entry, or
            # the slot/queue leaks until process restart.
            with self._sessions_lock:
                self._sessions.pop(session_id, None)
            self._slot_sem.release()
            raise

    def send_event(self, payload: dict) -> None:
        self._write_event(payload)

    def recv_event(self, session_id: str, timeout: float) -> dict:
        with self._sessions_lock:
            q = self._sessions.get(session_id)
        if q is None:
            raise WorkerProtocolError(f"unknown session {session_id}")
        return q.get(timeout=timeout)

    def end_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
        self._slot_sem.release()

    def _write_event(self, payload: dict) -> None:
        line = json.dumps(payload) + "\n"
        with self._stdin_lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def _reader_loop(self) -> None:
        for line in self._proc.stdout:
            event = json.loads(line)
            sid = event.get("id") or event.get("session_id")
            with self._sessions_lock:
                q = self._sessions.get(sid)
            if q is not None:
                q.put(event)
            # unsolicited / late events: drop with debug log
```

**(b) `_TRTEdgeLLMStreamingASRStream` (`:1087`) calls `_ASRWorkerIO.begin_session`
at stream open and `end_session` at close instead of taking
`_worker_lock`.** Concurrency limit comes from `_slot_sem`. Default 1 =
exactly today's behavior.

**(c) `restart_worker()` semantics**: when the worker process dies, every
in-flight session must see a `WorkerExitError`. Implement by closing all
queues with a sentinel on reader-thread exit:

```python
def _reader_loop(self) -> None:
    try:
        for line in self._proc.stdout:
            ...
    finally:
        with self._sessions_lock:
            for q in self._sessions.values():
                q.put({"event": "_worker_exit"})
            self._sessions.clear()
```

Each session translates `_worker_exit` to `WorkerExitError` on next
`recv_event`. The `_slot_sem` count must also be restored — easiest is to
recreate the `_ASRWorkerIO` on restart instead of trying to surgically heal
it.

### 4.4 Worker side phases

Phase the work to bound risk:

**Phase 1 (small) — Session-ID demux on stdout**

Today's events already carry `"id"` (the session_id). Verify every event
emission site uses it consistently. Add `"id"` to the `ready` event using
the literal `"__worker__"`. No concurrency yet — Python keeps `_worker_lock`.

Change points: every `std::cout << ... << std::endl` site in
`qwen3_asr_worker.cpp` — wrap with `emitEvent(session_id, kind, payload)`.

**Phase 2 (medium) — Per-session CUDA stream + execution-context pool**

Refactor `ThinkerEngine` / `AudioEncoderEngine` so each invocation accepts
a passed-in `cudaStream_t` and execution context. Keep N=1 default; verify
single-session behavior unchanged via golden-output regression test.

**Phase 3 (large) — Multi-session scheduler**

The single global `AsrSessionState` (referenced at
`qwen3_asr_worker.cpp:1284-1293`) is the central piece that must be
slotified. Concrete refactor steps in dependency order:

1. **Globals audit (gate, ~1 day).** Complete the §3.3 audit. Produce a
   PR comment listing every `static`/global, with disposition (read-only,
   move-to-slot, or wrap-in-mutex). No code changes yet; the audit gates
   step 2.
2. **Extract slot type (~1-2 days).** Create `struct Slot` carrying the
   fields enumerated in §4.1 plus a `Timer slotTimer` (replacing `gTimer`
   reads inside the request path). Construct one `Slot` in `main()`,
   keep N=1, route the existing single-session path through it.
   Acceptance: byte-identical to today's output on the golden WAV.
3. **Replace main loop with scheduler (~2-3 days).** Add the
   `WorkerScheduler` type from §4.2. The dispatcher in `main()` becomes:
   read stdin → look up slot by session_id (for chunk/end) or acquire
   new slot (for begin) → enqueue event into slot's queue. Per-slot
   worker thread blocks on its own queue. Remove the
   "session_already_active" rejection; add "slot_exhausted" /
   "duplicate_session_id" / "no_active_session" / "idle_timeout".
4. **Per-slot CUDA stream wiring (~1-2 days).** Pass each slot's
   `cudaStream_t` and execution contexts (created in step 2 via Phase 2)
   through to every TRT/CUDA call inside the request path. No call may
   reach to a process-global stream.
5. **Stdout serialization (~0.5 day).** Wrap every stdout write with
   `coutMutex` via a single `emitEvent(session_id, kind, payload)`
   helper. CI grep gate: no raw `std::cout` or `std::printf` outside the
   helper.
6. **Idle-timeout move (~0.5 day).** Replace the existing global idle
   timer with per-slot `popWithTimeout(slot.idleMs)` inside the slot's
   worker thread. Trigger session cleanup + slot release on timeout.

Total Phase 3: ~6-9 days, dominated by step 3-4 (main loop rewrite +
CUDA plumbing). Steps must merge in the order listed; do not parallelize
within Phase 3.

**Phase 4 (small) — Config knob**

Read `OVS_ASR_WORKER_CONCURRENCY` env at startup. Validate `1 ≤ N ≤ 4`.
Print resolved value in `ready` event:

```json
{"event": "ready", "init_ms": 1234, "concurrency": 2, "id": "__worker__"}
```

### 4.5 IPC protocol diff

```diff
 Request (stdin):
 {
   "id": "<session_id>",   ← already present; now must be globally unique
   "event": "begin" | "chunk" | "end",
   ...
 }

 Response events (stdout):
 {
   "id": "<session_id>",   ← already emitted; tighten contract to "always set"
   "event": "ready" | "partial" | "final" | "endpoint" | "error" | "done",
   ...
 }

 New error variants:
+  {"event":"error","id":"<sid>","error":"slot_exhausted"}
+  {"event":"error","id":"<sid>","error":"no_active_session"}
+  {"event":"error","id":"<sid>","error":"idle_timeout"}
-  {"event":"error","id":"<sid>","error":"session_already_active"}  ← removed
```

The `session_already_active` error is **removed** — with the scheduler, a
duplicate `begin` for an already-active session_id is a client bug; the
worker emits `"error": "duplicate_session_id"` and refuses the duplicate
without disturbing the existing session.

## 5. Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OVS_ASR_WORKER_CONCURRENCY` | `1` | Max concurrent ASR sessions in one worker process |
| `OVS_ASR_SESSION_IDLE_MS` | `30000` | Idle timeout per session (existing `kIdleTimeoutMs`, surfacing for ops) |

Surface in profiles:

```json
// configs/profiles/jetson-multilang-highperf.json (Orin Nano)
"OVS_ASR_WORKER_CONCURRENCY": "1"

// configs/profiles/jetson-multilang-highperf-nx.json (Orin NX)
"OVS_ASR_WORKER_CONCURRENCY": "2"
```

Contract: Python's value should equal the worker's resolved value (`ready`
event). If Python's is larger, the worker's `slot_exhausted` errors surface
to callers — operator misconfiguration.

## 6. Memory budget

Per concurrent session slot on Orin NX (approximate, must measure):

| Component | Approx. |
|---|---|
| Thinker KV cache (max seq 2048, FP16) | ~250-350 MB |
| Audio encoder activation buffers | ~80 MB |
| Mel accumulator + decoded tokens | ~10 MB |
| Thinker / encoder TRT execution contexts | ~80 MB |
| **Per-slot total** | **~420-520 MB** |

Free GPU memory after baseline load (dual-resident with TTS at N_tts=1):

- Orin Nano 8GB: ~1.0 GB → **N=1 only**, N=2 impossible alongside TTS
- Orin NX 16GB: ~5.5 GB → **N=2 comfortable, N=3 tight**
- AGX Orin 32GB: → N=4 fine

**Joint budget with TTS spec**: with TTS at N_tts=2 (~1 GB) and ASR at
N_asr=2 (~1 GB), Orin NX needs ~2 GB above today's baseline. Should fit
with the current ~5.5 GB headroom; spec author must verify with
`tegrastats` under load.

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Sessions live for seconds, not milliseconds; slot starvation more likely than in TTS | Surface `slot_exhausted` to the WS layer with a 503-style "server busy" frame; client retries |
| 2 | Mid-session worker crash leaves Python sessions hanging | Restart path closes all session queues with sentinel; `_slot_sem` reset by recreating `_ASRWorkerIO` |
| 3 | CUDA graph capture per execution context — startup cost multiplies | Per-slot warmup at scheduler init; bounded by `N × ~5s`, runs once |
| 4 | Stdin parsing must not block any slot — one slow JSON parse stalls all sessions | Stdin reader thread does only parse + enqueue; never touches CUDA |
| 5 | Stdout interleaving between session threads | Single `coutMutex` wrapping every event write |
| 6 | Duplicate `session_id` from two clients | Worker rejects with `duplicate_session_id`; Python uses uuid4 already (`:1087`) |
| 7 | KV cache buffer reuse across sessions causes content bleed | Each slot owns its own buffer; reset on session release; never shared |
| 8 | Idle-timeout firing during a legitimate long pause inside a multi-utterance v2v session | Push the idle timeout up to the WS layer's session_complete logic instead of relying on worker timer alone; tune via `OVS_ASR_SESSION_IDLE_MS` |
| 9 | `restart_worker()` (`:485`) was designed around a single blocking `readline()` — semantics change with N reader queues | New restart path: kill process → reader loop exits → all queues get sentinel; rebuild `_ASRWorkerIO` from scratch |
| 10 | TRT engine thread-safety on Jetson's exact TRT version | Verify with TRT docs + smoke test before relying; same open question as TTS spec |
| 11 | `gTimer` (`qwen3_asr_worker.cpp:447-479`) accumulates timings across all requests | Slotify per §4.4 Phase 3 step 2 — every request-path timer read goes through `slot.timer`. Any remaining `gTimer` usage must be protected by mutex or removed |
| 12 | `gMelExtractor` (`qwen3_asr_worker.cpp:179-182`) thread-safety unverified | Audit per §3.3. If mutable internal state exists, allocate one extractor per slot (cheap — extractor itself is small, just FFT plan + tables) |
| 13 | Stdout writes from non-scheduler paths (parse errors, `ready`, legacy one-shot transcribe) bypass `coutMutex` | §4.4 Phase 3 step 5 mandates an `emitEvent()` helper + CI grep gate. Every existing site at `:736, :1035, :1230, :1361` (and any others) must route through the helper |

## 8. Testing plan

### Unit (worker)

- `test_asr_worker_concurrency_1.cpp` — N=1, byte-identical output to today's
  worker for a known WAV (regression).
- `test_asr_worker_concurrency_2.cpp` — N=2, two sessions concurrent, each
  with its own WAV; verify both finals match their respective
  ground-truth single-session decode.
- `test_asr_worker_scheduler_overflow.cpp` — N=2, fire 3rd `begin` before
  releasing either of the first two; verify `slot_exhausted` and no
  collateral damage.
- `test_asr_worker_duplicate_session.cpp` — fire `begin` with already-active
  session_id; verify `duplicate_session_id` and the original session
  continues.
- `test_asr_worker_idle_timeout.cpp` — `begin` then idle longer than
  `kIdleTimeoutMs`; verify timeout error and slot release.

### Integration (Python side)

- `test_trt_edge_llm_asr_concurrency.py` — instantiate backend with
  `OVS_ASR_WORKER_CONCURRENCY=2`; open two streams in two threads; finalize
  in interleaved order; verify each gets correct transcript.
- `test_asr_worker_restart_during_concurrency.py` — start 2 sessions; kill
  worker mid-stream; verify both Python callers see `WorkerExitError`;
  verify a 3rd session can begin after restart.
- `test_asr_slot_exhaustion.py` — `OVS_ASR_WORKER_CONCURRENCY=1`; second
  `create_stream().finalize()` blocks (semaphore) but does not error;
  verify ordering.

### End-to-end (deployment)

- Orin NX with `OVS_ASR_WORKER_CONCURRENCY=2`: two clients connect
  `/v2v/stream` and speak simultaneously; measure `stop_to_final_ms` on
  both — neither should exceed 1.5× the single-client p50.
- Memory probe: `tegrastats` for 10 min with two long conversations; peak
  GPU memory must stay under 90% of board capacity.
- Joint with TTS N=2: full duplex two-user dashboard test; both sessions
  must hold TTFA within 2× single-user baseline.
- Hot-reload during concurrent sessions — see §8.1 for the binding contract.

### 8.1 Hot-reload contract (binding — referenced from acceptance criterion 5)

`BackendManager.reload()` today drains by waiting for `inflight_http == 0`
and closing WS handles (`app/core/backend_manager.py`). With N>1
concurrent ASR sessions, the contract is an explicit state machine on the
Python `_ASRWorkerIO`:

```python
class DrainState(Enum):
    NORMAL    = 0   # default
    DRAINING  = 1   # reload requested; reject new begin, let in-flight finish
    RELOADING = 2   # drain window expired or worker being killed; all calls reject
```

**State + locks** (added to `_ASRWorkerIO`):
- `self._drain_lock = threading.RLock()` — covers state transitions AND
  session-map membership during drain
- `self._drain_state: DrainState = DrainState.NORMAL`
- `self._drain_complete = threading.Event()` — signaled when `len(_sessions) == 0`

**Algorithm**:

1. **`begin_session(session_id, payload)`** acquires `_drain_lock` BEFORE
   the semaphore. If state ≠ NORMAL, raise `BackendDrainingError` (no
   semaphore acquired, no map insert, no stdin write). If state = NORMAL,
   insert into `_sessions` under the same lock, then release and proceed
   to stdin write per existing logic. **The drain check and map insert
   share one critical section** so a `begin` cannot slip past a drain
   that started a microsecond earlier.

2. **`end_session(session_id)`** under `_drain_lock`: pop the entry,
   release semaphore. If state ≠ NORMAL and `len(_sessions) == 0`, signal
   `_drain_complete`. (This lets `drain()` wake immediately when the last
   session ends naturally, rather than polling.)

3. **`drain(timeout: float)`** entry point called by `BackendManager`:
   - Acquire `_drain_lock`, set `_drain_state = DRAINING`, release lock
   - `_drain_complete.wait(timeout)` — block on the Event
   - On wake:
     - If `len(_sessions) == 0`: drain succeeded, return cleanly
     - Else (timeout): under `_drain_lock`, set `_drain_state = RELOADING`,
       snapshot live session_ids, then for each send
       `{"event":"reload_abort","id":sid}` via `_write_event`. Release
       lock. Return drained_cleanly=False.
   - The reader thread continues forwarding events. Clients receiving
     `reload_abort` are expected to close their WS within their own
     timeout (default 2 s). Stranded sessions get cleaned up by the next
     step.

4. **`force_close()`** called after `drain()` returns (regardless of
   outcome): kill the worker process. Reader-loop hits EOF, runs its
   `finally`, pushes `_worker_exit` sentinel into every still-live queue.
   Any in-flight `recv_event` raises `WorkerExitError`. Semaphore permits
   leak — but the entire `_ASRWorkerIO` is discarded and replaced.

5. **WS layer mapping** (`app/main.py` v2v handler):
   - `BackendDrainingError` → emit `{"type":"error","error":"backend_draining"}`, close WS with code 1013
   - `reload_abort` event arriving on the recv queue → emit `{"type":"error","error":"reload_abort"}`, close WS with code 1013
   - `WorkerExitError` → emit `{"type":"error","error":"backend_swapped"}`, close WS with code 1013

**Drain timeout default**: `drain_timeout_s = max(15.0, N × kIdleTimeoutMs/1000)`
— with N=2 and idle timeout 30 s, that's 60 s. Operator can override per
call via the existing `drain_timeout_s` field in `BackendReloadRequest`.

**Ordering invariants** the implementation must verify:
- `begin_session`'s drain check and map insert are in the SAME
  `_drain_lock` critical section (otherwise a `begin` can race past a
  state transition)
- `end_session`'s map pop and the drain-complete signal are in the SAME
  critical section (otherwise `drain()` can deadlock waiting for an
  already-empty map)
- `drain()` releases `_drain_lock` BEFORE calling `_drain_complete.wait()`
  (otherwise `end_session` cannot signal completion)
- `force_close()` is sequenced after `drain()` returns (the reader
  sentinel must not fire while `drain()` is still polling map state)

The state machine + invariants above are what makes acceptance criterion
5 (Plan §7 #5) testable: a unit test can drive `_ASRWorkerIO` through
each transition and assert the correct error surfaces to the caller.

## 9. Rollout

1. Land Phase 1 (session_id audit) — additive protocol tightening, safe to
   ship in same image as today's worker.
2. Land Phase 2-4 gated by `OVS_ASR_WORKER_CONCURRENCY=1` default
   (= no behavior change). Image ships everywhere.
3. Run validation suite on Orin NX with N=2 + TTS N=2 (joint) for 1 week
   under synthetic two-user load.
4. Bump Orin NX profile to `OVS_ASR_WORKER_CONCURRENCY=2`. Update memory
   with verified behavior.
5. Bump AGX profile to `4` after AGX validation.
6. Keep Orin Nano at `1` permanently — joint memory budget rules out
   higher.

## 10. Open questions (for assignee)

- TRT 10.x multi-context guarantees on Jetson's image — same question as
  TTS spec; can answer both with one smoke test.
- CUDA graph capture per-execution-context vs per-engine — same open
  question. If per-engine, slot 0 owns the graph and slots 1..N-1 fall
  back to eager mode; quantify the gap.
- Audio encoder graph capture: does the encoder also use a captured graph
  today, or is it eager? Check `qwen3_asr_worker.cpp` for `cudaGraph`
  usage in the encoder path.
- Should `slot_exhausted` translate to a WS-layer "server busy" frame
  that the client can use to back off, or should Python block-acquire and
  hide it? Recommendation: surface to WS layer so clients aren't left
  guessing why the first audio chunk has 8s latency.
- Worker-side idle-timeout currently fires globally. With N slots, each
  slot's timer should be independent. Verify the timer source is per-slot
  in the new design.

## 11. Reference

- Current worker source:
  `third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_asr_worker.cpp`
- Python IPC: `app/backends/jetson/trt_edge_llm_asr.py` (`:147-:540`,
  `:1087-:1210`)
- TRT runtime classes: thinker / encoder engines — defined in
  `third_party/qwen3-edgellm-jetson/native/edgellm_voice/`
- Build script: `deploy/jetson-release-highperf.sh`
- Sibling spec: [`tts-worker-concurrency.md`](./tts-worker-concurrency.md)

---

**Acceptance criteria for the assignee:**

1. With `OVS_ASR_WORKER_CONCURRENCY=1`, all existing tests pass and ASR
   output is byte-identical to pre-change worker for the same input.
   Byte-identical here means: with `ASR_TEMPERATURE=1.0`, `ASR_TOP_K=1`,
   `ASR_TOP_P=1.0` (current profile defaults that yield greedy decode),
   the emitted token sequence and final text match exactly. Tests must
   assert these env values are set before claiming byte-identical.
2. With `OVS_ASR_WORKER_CONCURRENCY=2` on Orin NX, two simultaneous v2v
   sessions both finalize within 1.5× the single-session p50 — measured
   on `bench/perf/corpus/short/zh_short_*.wav`.
3. Memory under joint TTS N=2 + ASR N=2 load stays below 90% of board
   capacity over a 10-minute soak.
4. Restarting the worker mid-stream surfaces clean `WorkerExitError` to
   all in-flight Python callers; subsequent sessions begin normally.
5. Hot-reload during concurrent sessions either drains cleanly or returns
   a documented error — never silently corrupts an in-flight transcript.
