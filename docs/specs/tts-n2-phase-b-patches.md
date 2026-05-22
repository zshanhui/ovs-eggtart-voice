# Phase B Patch-Level Spec (codex)

Date: 2026-05-22. Companion to `tts-n2-throughput.md` + `tts-n2-shared-tensor-audit.md`. Scoped to native pool path (production profile `jetson-multilang-highperf-nx`).

## §1 Struct field additions

`TalkerSlot` (`qwen3OmniTTSRuntime.h:278-309`). Allocate in `Qwen3TTSTalkerEngine::allocateSlot(...)` after existing `deviceAttentionMask` alloc (`qwen3OmniTTSRuntime.cpp:2192-2215`), formulas from `allocateBuffer()` (`qwen3OmniTTSRuntime.cpp:3544-3547`):

| New field | Type | Replaces | Alloc cite |
|---|---|---|---|
| `thinkerEmbedBuffer` | `rt::Tensor [maxSeqLen,thinkerHiddenSize]` | `mThinkerEmbedBuffer` | `3551-3553` |
| `gpuTokenIdsBuffer` | `rt::Tensor [1,maxSeqLen]` | `mGpuTokenIdsBuffer` | `3551-3553` |
| `mlpWorkspace` | `rt::Tensor [maxSeqLen,thinkerHiddenSize]` | `mMLPWorkspace` | `3555-3557` |
| `projectedBuffer` | `rt::Tensor [maxSeqLen,talkerHiddenSize]` | `mProjectedBuffer` | `3558-3559` |
| `talkerInputEmbeds` | `rt::Tensor [maxSeqLen,talkerHiddenSize]` dtype `mTalkerInputEmbedsDataType` | `mTalkerInputEmbeds` | `3561-3564` |
| `speakerEmbedding` | `rt::Tensor [talkerHiddenSize]` | `mSpeakerEmbedding` | `3564` |
| `talkerLogits` | `rt::Tensor [1,talkerVocabSize]` | `mTalkerLogits` | `3566-3569` |
| `talkerSelectedIndices` | `rt::Tensor [1,1]` | `mTalkerSelectedIndices` | `3569` |
| `seenCodecTokensBuf` | `rt::Tensor [maxKVCacheCapacity]` | `mSeenCodecTokensBuf` | `3624-3625` |
| `talkerHiddenStatesBuffer` | `rt::Tensor [1,maxSeqLen,talkerHiddenSize]` | `mTalkerHiddenStatesBuffer` | `3627-3629` |
| `talkerLastHidden` | `rt::Tensor [1,talkerHiddenSize]` | `mTalkerLastHidden` | `3634-3636` |
| `residualEmbedBuffer` | `rt::Tensor [1,1,talkerHiddenSize]` | `mResidualEmbedBuffer` | `3598-3600` |
| `codecHiddensBuffer` | `rt::Tensor [1,16,talkerHiddenSize]` | `mCodecHiddensBuffer` | `3638-3640` |
| `trailingTextLen` | `int32_t{0}` | `mTrailingTextLen` (h:497) | scalar |
| `hostProjectedBuffer` | `std::vector<float>` | `mHostProjectedBuffer` (h:555) | host companion |

`CodePredictorSlot` (`qwen3OmniTTSRuntime.h:311-342`). Allocate in `Qwen3TTSCodePredictorEngine::allocateSlot(...)` (existing alloc at `qwen3OmniTTSRuntime.cpp:934-970`):

| New field | Type | Replaces |
|---|---|---|
| `codePredictorPrefillInput` | `rt::Tensor [1,2,codePredictorHiddenSize]` | `mCodePredictorPrefillInput` (`3586-3588`) |
| `codePredictorCodecIds` | `rt::Tensor [1,1]` | `mCodePredictorCodecIds` (`3588`) |
| `codePredictorCodecEmbed` | `rt::Tensor [1,1,codePredictorHiddenSize]` | `mCodePredictorCodecEmbed` (`3590-3591`) |
| `rawCodecEmbed` | `rt::Tensor [1,1,talkerHiddenSize]` | `mRawCodecEmbed` (`3592-3594`) |
| `smallToMtpProjectedHidden` | `rt::Tensor [1,codePredictorHiddenSize]` | `mSmallToMtpProjectedHidden` (`3595-3597`) |

## §2 Function signature changes

Add `TalkerSlot& talkerSlot` and/or `CodePredictorSlot& cpSlot` as first params before `cudaStream_t stream`:

- `handleAudioGeneration` (h:197-198, cpp:4441-4442) → +talkerSlot+cpSlot. Callers: `qwen3_tts_worker.cpp:1224, 1246-1247` (already holds slot indices at 837-856; deref from `mTalkerSlots`/`mCPSlots`).
- `prepareTalkerInput` (h:585-586, cpp:4375-4376) → +talkerSlot. Caller: cpp:4554.
- `projectToTalkerInput` + `projectToTalkerInputHost` (h:577-580, cpp:3968-3970, host 4056-4061) → +talkerSlot. Caller: cpp:4427-4428.
- `executeTalkerPrefillStep` (h:362-363, cpp:4116-4118) → +talkerSlot. Caller: cpp:4561.
- `executeTalkerDecodingStep` (h:365-366, cpp:4186-4188) → +talkerSlot. Member refs: cpp:4641, 4654-4655, 4684, 4700, 4735-4739. Caller: cpp:4700.
- `extractTalkerLastHidden` (h:376-377, cpp:5175-5177) — pass `talkerSlot.talkerHiddenStatesBuffer` + `talkerSlot.talkerLastHidden` as args, no new param needed.
- `runCodePredictorGenerationForFrame` (h:368-369, cpp:4788-4790) → +talkerSlot+cpSlot. Redirect cpp:4816-4853 + 4908-4924.
- `computeResidualConnection` + `computeResidualConnectionHost` (h:371-374, cpp:5028-5030, host 5121-5128) → +talkerSlot.

## §3 mCodecHiddensBuffer redirect (Commit C1)

- Current alloc: cpp:3638-3640. Member decl: h:547-548.
- Move target: `TalkerSlot::codecHiddensBuffer` allocated in `allocateSlot` (cpp:2192-2215).
- Sites to redirect:
  - CP decode stores 1-14: cpp:4253-4260
  - Native CP zero/materialize: cpp:4908-4924
  - Legacy CP zero: cpp:4972-4974
  - Active-code materialize: cpp:5010-5017
  - Residual kernel read: cpp:5067-5068
  - Contract comments: cpp:4793-4795, 4961-4965

## §4 resetSampling scope (Commit C4)

- Current: `resetSampling()` cpp:974-994 reseeds global `mRng` + iterates ALL CP slots under `mSlotPoolMutex`. Called at cpp:4547-4550.
- Fix: add overload `void resetSampling(CodePredictorSlot& slot)` — seeds only `slot.rng`, conditionally resets `slot.gpuSamplingOffset` per existing policy (cpp:981-982).
- Replace call at cpp:4547-4550 with slot-scoped overload using acquired cpSlot.

## §5 Code2Wav: mutex vs per-slot (Commit C5)

**Recommended: worker-level mutex** (staged). Worker already vectorizes per-slot runners at `qwen3_tts_worker.cpp:633-650, 680-695` and removed old mutex at 469-478. Acceptance: Code2Wav runs after token generation; slow-client TTFA driven by first audio chunk emitted before mutex contention.

- Place `std::mutex mCode2WavMutex` in worker class
- Lock around `runners[c2wSlot]->reset(c2wStream)` at `qwen3_tts_worker.cpp:1092-1097`
- Lock around `synthesizeStatefulChunk(...)` at `qwen3_tts_worker.cpp:1001-1008`
- `StatefulCode2WavRunner` itself unchanged

Per-slot full isolation alternative: duplicate `statefulCode2WavRunner.h:74-92` mutable state, all mutation sites at cpp:254-262, 287-301, 305-343, 348-373, 381-388. +80-150 LOC + 1× VRAM per slot. Promote to this if mutex contention dominates.

## §6 N=1 byte-equivalence gates

| Commit | Risk | Hazard |
|---|---|---|
| C1 codecHiddensBuffer | Low | 8 sites; missed redirect → residual at cpp:5067-5068 diverges |
| C2 Talker scratch | Med | New alloc addresses; preserve `talkerRng` per-request seeding cpp:4546 |
| C3 CP scratch | Med-high | Native CP RNG already slot-owned (h:330-341); risk = `gpuSamplingOffset` policy at cpp:981-982 |
| C4 resetSampling scope | **High** | Must preserve `gpuSamplingOffset` semantics exactly |
| C5 Code2Wav mutex | Low | Audio downstream of tokens; mutex serializes but doesn't reorder |

## §7 Commit splits

| # | Subject | LOC | Gate |
|---|---|---|---|
| C1 | `codecHiddensBuffer` per-slot | 35-55 | N=1 MD5 |
| C2 | Talker scratch per-slot + plumb signatures | 120-170 | N=1 MD5 |
| C3 | CP scratch per-slot + plumb | 80-130 | N=1 MD5 |
| C4 | `resetSampling` slot-scoped overload | 10-25 | N=1 MD5 + N=2 no-cross-reseed |
| C5 | Code2Wav worker mutex | 10-20 | N=1 MD5 + N=2 TTFA ≤ 765ms + 100-iter stability |
