# Voice Pipeline Concurrency — Overall Plan

**Status**: Approved (2026-05-21, after two codex review rounds)
**Scope**: Jetson Orin Nano / NX / AGX qwen3 ASR + TTS backends
**Sibling specs**:
- [`tts-worker-concurrency.md`](./tts-worker-concurrency.md) — TTS worker slot pool
- [`asr-worker-concurrency.md`](./asr-worker-concurrency.md) — ASR worker slot pool

---

## 1. Problem statement

Two user-facing pain points share one root cause: both the qwen3 ASR worker
and the qwen3 TTS worker are strictly single-request-at-a-time. The Python
backends serialize on `_worker_lock`; the C++ workers either reject a
second session (ASR `session_already_active`) or do not even read stdin
until the current request's `done` is emitted (TTS main loop).

| Symptom | Root cause | Today's workaround |
|---|---|---|
| **Mid-sentence audio gap** between TTS segments (~450 ms on Orin NX) | Segment N+1's talker prefill cannot start until segment N's `done` | None — `_split_tts_text` already minimizes segment count by maxing CJK chars at 48 |
| **Second concurrent v2v session stalls** for ~entire first user's utterance | ASR worker rejects second `begin`; Python serializes around it | None — multi-user voice deployment effectively impossible |

Empirical baseline (Orin NX, qwen3asr+qwen3tts, short zh ~3.2 s audio,
profile `jetson-multilang-highperf-nx`, single client):

| Metric | p50 |
|---|---|
| ASR `stop → asr_final` | 3 ms |
| TTS `final → first PCM` (first_chunk_frames=7, default) | 506 ms |
| TTS `final → first PCM` (first_chunk_frames=4) | 369 ms |
| TTS `final → first PCM` (first_chunk_frames=1) | 230 ms |
| TTS inter-segment gap (next sentence first PCM minus prev `tts_sentence_done`) | ~440-490 ms |
| Cold start, first TTS call after worker spawn | ~51 s |

Tuning `EDGE_LLM_TTS_FIRST_CHUNK_FRAMES` from 7 → 4 cuts initial TTFA by
27% but makes the inter-segment gap *more* noticeable (the first chunk
plays faster, so the silence between segments stands out earlier in the
playback timeline). Tuning alone cannot close the gap — only worker-side
concurrency can.

## 2. Goal

Ship two complementary changes, gated by independent env knobs, that
together enable:

- **Single-user, multi-segment streaming** — segment N+1 prefill overlaps
  segment N's tail streaming. Target: TTS inter-segment gap ≤ 50 ms p50
  on Orin NX.
- **Multi-user voice agent** — up to N concurrent v2v sessions on one
  Jetson device with each session's TTFA staying within 1.5× the
  single-user baseline.

Both with `N=1` default = byte-identical to today's behavior.

## 3. Two-spec architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  v2v WS layer (app/main.py)                                      │
│  ── unchanged ──                                                  │
└──────┬───────────────────────────────────────┬──────────────────┘
       │                                       │
       ▼                                       ▼
┌──────────────────────────────┐  ┌────────────────────────────────┐
│  ASR backend                 │  │  TTS backend                   │
│  trt_edge_llm_asr.py         │  │  trt_edge_llm_tts.py           │
│                              │  │                                │
│  _ASRWorkerIO (new)          │  │  _WorkerIO (new)               │
│   ├─ _stdin_lock             │  │   ├─ _stdin_lock               │
│   ├─ session_id demux        │  │   ├─ request_id demux          │
│   └─ Semaphore(N_asr)        │  │   └─ Semaphore(N_tts)          │
└──────┬───────────────────────┘  └────────────┬───────────────────┘
       │ stdin JSON / stdout JSON              │ stdin JSON / stdout JSON
       ▼                                       ▼
┌──────────────────────────────┐  ┌────────────────────────────────┐
│  qwen3_asr_worker.cpp        │  │  qwen3_tts_worker.cpp          │
│                              │  │                                │
│  Scheduler (N_asr slots)     │  │  Scheduler (N_tts slots)       │
│   ├─ Slot{stream, ctxs,      │  │   ├─ Slot{stream, ctxs,        │
│   │       AsrSessionState}   │  │   │       KV, c2w state}       │
│   ├─ session→slot map        │  │   ├─ request→slot map          │
│   └─ coutMutex               │  │   └─ coutMutex                 │
└──────────────────────────────┘  └────────────────────────────────┘
```

Both workers share the same scheduler **pattern** but each owns its slot
pool independently. They do not share processes, slots, or threads. They
both run inside the same speech service container on the same Jetson GPU,
and contend only at the GPU/memory level — exactly as they do today.

**Why two separate scheduler implementations and not a shared library:**
- The two workers live in **two different upstream repos**:
  - **TTS** worker source = `examples/omni/qwen3_tts_worker.cpp` in
    `suharvest/TensorRT-Edge-LLM` (the production fork — `/Users/harvest/project/tensorrt-edge-llm`
    on Mac, `/home/harvest/TensorRT-Edge-LLM` on orin-nx). The
    `qwen3-edgellm-jetson` submodule's TTS worker is stale and not part
    of the build pipeline.
  - **ASR** worker source = `native/edgellm_voice_worker/qwen3_asr_worker.cpp`
    in `suharvest/qwen3-edgellm-jetson` (the toolkit — vendored as a
    submodule in seeed-local-voice).
- Adding cross-worker dependencies would mean cross-repo dependencies,
  inviting upstream merge pain.
- The slot state types differ (ASR session vs TTS request).
- A single env knob (`OVS_VOICE_CONCURRENCY`) would couple ASR and TTS
  rollouts; we want to be able to ship TTS concurrency first.

**Side cleanup (separate work, not blocking):** the `qwen3-edgellm-jetson`
submodule still ships a 687-line stale `qwen3_tts_worker.cpp` that no
build consumes. Either delete it (preferred) or port the current
TensorRT-Edge-LLM TTS worker back into the toolkit and update the
toolkit's CMakeLists to consume it. Tracked as a follow-up; do not block
either concurrency spec on this.

## 4. Sequencing & ownership

Both specs are upper-bounded by their respective effort estimates: TTS
3-5d, ASR 10-15d. The ASR work dominates total wall-clock; the joint
deployment can only happen after both ship at N=1.

```
            Week 1            Week 2            Week 3-4          Week 5            Week 6
            ──────            ──────            ─────────────     ──────            ──────
TTS spec    Phase 1-2         Phase 3-4         soak / measure    profile bump      —
            (request_id +     (scheduler +
             stream wiring)    config knob)

ASR spec    Phase 1           Phase 2 + 3 step 1-2 (audit,        Phase 3 step 3-6  Phase 4 +
            (session_id        slot extract)                       (scheduler +      §8.1 +
             audit)                                                CUDA wiring +     tests +
                                                                   stdout +          soak
                                                                   timeout)

Joint                                                                                E2E joint
                                                                                     test +
                                                                                     deployment
                                                                                     (N_tts=2,
                                                                                      N_asr=2)
```

Rationale for sequence:

1. **TTS-first** because the symptom users hear most (mid-sentence gap) is
   visible even to a single user. Multi-user ASR concurrency is invisible
   until you actually deploy multi-user.
2. **TTS Phase 1-2** are protocol/runtime refactors with no behavior
   change at N=1. Shipping them in image-rev N gates the rest behind a
   feature flag.
3. ASR Phase 1 (session_id audit) can start in parallel with TTS Phase 3
   — they touch independent files. Shipping ASR Phase 1 early lets us
   close the existing memory note about `session_already_active`
   semantics.
4. **Joint E2E test only after both are merged at N=1** to prove no
   regression, then **profile bump together** (Orin NX N_asr=2, N_tts=2)
   to validate joint memory budget at once.

## 5. Joint memory budget (Orin NX 16GB)

Measured today, qwen3asr + qwen3tts dual-resident, single slot each:

| Component | Approx |
|---|---|
| OS / docker / Python | ~2.0 GB |
| qwen3 talker engine (weights) | ~3.0 GB |
| qwen3 thinker (ASR) engine (weights) | ~2.5 GB |
| audio encoder + code_predictor + code2wav | ~1.5 GB |
| TTS slot-0 KV + buffers | ~0.5 GB |
| ASR slot-0 KV + buffers | ~0.5 GB |
| **Baseline N=1 + N=1** | **~10.0 GB** |
| Headroom on 16GB NX | ~6.0 GB |

Adding one more slot each (target: TTS N=2 + ASR N=2):

| Addition | Approx |
|---|---|
| TTS slot-1 (per [`tts-worker-concurrency.md`](./tts-worker-concurrency.md) §6) | ~0.5 GB |
| ASR slot-1 (per [`asr-worker-concurrency.md`](./asr-worker-concurrency.md) §6) | ~0.5 GB |
| **Total over baseline** | **~1.0 GB** |

Comfortable on 16GB NX (~5 GB headroom remaining). **Orin Nano (8GB)
cannot afford either spec at N=2** — both stay at 1 permanently on Nano
unless a future engine slimming changes the budget.

## 6. Joint configuration

| Env var | Orin Nano | Orin NX | AGX Orin | Notes |
|---|---|---|---|---|
| `OVS_TTS_WORKER_CONCURRENCY` | 1 | 2 | 4 | TTS slots |
| `OVS_ASR_WORKER_CONCURRENCY` | 1 | 2 | 4 | ASR slots |
| `OVS_ASR_SESSION_IDLE_MS` | 30000 | 30000 | 30000 | unchanged from current `kIdleTimeoutMs` |
| `EDGE_LLM_TTS_FIRST_CHUNK_FRAMES` | 7 | 4 | 4 | already verified -27% TTFA on NX, no quality loss observed |

The first three are **set per-profile** in
`configs/profiles/jetson-multilang-highperf-{nx,nano,agx}.json`. Operators
do not need to touch them; the profile rollout (step 6 below) is the only
deployment-time change required.

## 7. Acceptance criteria (rollup)

| # | Criterion | Source spec |
|---|---|---|
Citations below point at the trailing **Acceptance criteria for the assignee**
block at the bottom of each spec (those criteria are not numbered as §11 — they
follow §11 Reference). When citing them I use "TTS-AC #n" / "ASR-AC #n".

| # | Criterion | Source spec |
|---|---|---|
| 1 | N=1 byte-identical output for both ASR and TTS, under **pinned greedy sampling**: TTS `OVS_TTS_TALKER_TEMPERATURE=0.0`, `OVS_TTS_TALKER_TOP_K=1`, `OVS_TTS_SEED=42` (already set in NX profile, `jetson-multilang-highperf-nx.json:40-43`); ASR `ASR_TEMPERATURE=1.0`, `ASR_TOP_K=1`, `ASR_TOP_P=1.0` (current defaults at `trt_edge_llm_asr.py:234-236`). Tests must assert these env values are present before running the byte-identical check, otherwise the criterion is moot. | TTS-AC #1, ASR-AC #1 |
| 2 | TTS inter-segment gap ≤ 50 ms p50 on NX with N_tts=2, measured using the new bench metric (criterion 7 is a prerequisite for this) | TTS-AC #2 |
| 3 | Two simultaneous v2v sessions on NX, both within 1.5× single-user TTFA p50 | ASR-AC #2 |
| 4 | Joint memory under 90% board capacity over 10-min soak with N_tts=2, N_asr=2 | TTS §6 + ASR §6 + Plan §5 |
| 5 | Hot-reload during concurrent sessions follows the state machine in ASR §8.1 (NORMAL → DRAINING → RELOADING with the four ordering invariants); WS surfaces `backend_draining` / `reload_abort` / `backend_swapped` — never silent corruption | ASR §8.1 |
| 6 | Worker crash mid-stream surfaces `WorkerExitError` to all in-flight callers | ASR-AC #4, TTS-AC #4 |
| 7 | Bench script `measure_v2v_unified.py` reports inter-segment gap as a first-class metric — **promoted into spec scope** because criterion 2 depends on it. Implement during TTS Phase 4 (config knob): add a `gap_ms` field per segment to the bench output, computed as `tts_started_next - tts_sentence_done_prev`. Effort: ~0.5 day, separate PR from the worker work. | (new — owned by TTS Phase 4) |

## 8. Out of scope (deferred)

- **Per-session quota inside the scheduler.** With N=2 and one session
  using both slots for its segments, a second user briefly waits. Fair
  scheduling is a v2 problem; ship FCFS first.
- **Cross-session batch fusion** for ASR or TTS. Would amortize prefill
  cost across sessions but adds beam-management complexity and breaks the
  deterministic single-session output contract. Revisit if multi-user
  becomes the dominant deployment.
- **Speculative prefill from `asr_partial`.** Discussed in codex review of
  TTFA work — would feed last-stable partial into TTS prefill before the
  final, eliminating most of the remaining TTFA. Requires non-trivial
  v2v.py changes (TTS currently only sees `CLIENT_TEXT` events, not
  `asr_partial`). Out of these specs.
- **Bench harness `_drain_nonblocking` bug fix.**
  `bench/perf/measure_v2v_unified.py:263` drops `asr_final` events
  received inside the send-time drain. Tracked separately as a small
  Python fix, not blocking either spec. Note: this is distinct from the
  in-scope `gap_ms` metric work (criterion 7).

## 9. Workflow gates

This plan + the two underlying specs are **drafts**. To advance:

1. **codex review** of both specs (parallel) — surface design holes,
   protocol races, missing risks
2. Address codex feedback; mark specs `Status: Approved`
3. **Implementation agent** kicks off (TTS first, then ASR per §4)
   following spec phases verbatim
4. **codex review** of merged implementation against spec acceptance
   criteria
5. **On-device E2E validation** on Orin NX:
   - Single-user A/B (N_tts=1 vs N_tts=2) — measure inter-segment gap
   - Multi-user A/B (N_asr=1 vs N_asr=2) — measure second-session TTFA
   - Joint soak test
6. Memory note update with verified behavior, then profile bump

Failure to meet any acceptance criterion sends the work back to step 3.
