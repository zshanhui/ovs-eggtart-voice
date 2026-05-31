# TTS N=2 — Shared Runtime Tensor Audit (codex)

Date: 2026-05-22. Source: codex deep audit of `tensorrt-edge-llm/cpp/runtime/qwen3OmniTTSRuntime.{h,cpp}` and `statefulCode2WavRunner.cpp`.

Companion to `tts-n2-throughput.md`. Read this BEFORE starting Phase B — the scope is materially larger than the original spec §4 estimate (~50 LOC). Real scope ≈ **180-260 LOC runtime slotting + 120-180 LOC Code2Wav + 10-20 LOC sampling fix + optional legacy-path guards.**

## §1 Inventory table

| Member | Cite | Ownership now | Concurrent? | Fix |
|---|---|---|---|---|
| `mGpuTokenIdsBuffer`, `mThinkerEmbedBuffer` | `qwen3OmniTTSRuntime.cpp:4384-4393` | runtime-global scratch | Unsafe: N=2 reshape/copy collides before projection | Move to `TalkerSlot`, +2 tensors / +20 LOC |
| `mProjectedBuffer`, `mMLPWorkspace`, `mSpeakerEmbedding`, `mTalkerInputEmbeds`, `mTrailingTextLen`, `mHostProjectedBuffer` | `4001-4026`, `4056-4112`, `5057-5068`, `5121-5129` | runtime-global request state | Unsafe: trailing-text addend cross-contamination; projected buffers overwritten | Move to `TalkerSlot`, +6 fields / +60 LOC |
| `mTalkerLogits`, `mTalkerHiddenStatesBuffer`, `mTalkerSelectedIndices`, `mSeenCodecTokensBuf` | `4561-4577`, `4597-4603`, `4641-4700`, `4715-4739`, `4475-4483` | runtime-global generation buffers | Unsafe: logits/hidden/selected/seen-token cross-contamination | Move to `TalkerSlot`, +4 tensors / +40 LOC |
| `mResidualEmbedBuffer`, `mTalkerLastHidden` | `4641-4655`, `4684-4700`, `5175-5222` | runtime-global scratch | Unsafe: request A may decode with request B's last hidden | Move to `TalkerSlot`, +2 tensors / +20 LOC |
| `mCodePredictorCodecIds`, `mRawCodecEmbed`, `mCodePredictorCodecEmbed`, `mSmallToMtpProjectedHidden`, `mCodePredictorPrefillInput` | `4816-4853`, `4939-4943`, `5010-5017`, `4244-4279` | runtime-global CP scratch | Unsafe: shape reshape collisions mid-frame | Move to `CodePredictorSlot`, +5 tensors / +50 LOC |
| `mCodePredictorLogits`, `mCodePredictorLogitsPerHead`, `mCodePredictorHiddenStatesBuffer`, `mCodePredictorSelectedIndices` | `4950-5003`, `4281-4303`, `3574-3588` | runtime-global CP outputs | Unsafe in legacy CP path | Move to `CodePredictorSlot` OR lock legacy path |
| **`mCodecHiddensBuffer`** | `4908-4924`, `4972-5017`, `5067-5068` | runtime-global frame buffer | **Unsafe + likely the cudaMemsetAsync illegal-access SITE** | Move to per-request / `TalkerSlot`, +1 tensor |
| `mHostReuseKVCacheLengths`, `mHostTalkerContextLength`, `mHostCodePredictorContextLength` | `4147-4183`, `4205-4230` | runtime-global host tensors | Unsafe in legacy LLMRunner path | Per-slot host tensors OR mutex legacy runner |
| Sampling: `resetSampling()` reseeds ALL CP slots | `974-994`, called `4547-4550` | runtime-global RNG fan-out | Unsafe: concurrent reseed of sibling's slot | Reset only acquired slot RNG, +10-20 LOC |
| Engine slot pools internal | `2267-2289`, `2460-2475`, `997-1016`, `1044-1062`, `1895-1937`, `2935-2975` | per-slot pool internally | Mostly safe internally [inferred] | No change |
| Read-only weights/tables | `4007-4026`, `4384-4402`, `4864-4875`, `5067-5068`, `5076-5102` | runtime-global immutable | Safe | Leave shared |
| `mMultimodalMetrics` | `4782` | runtime-global metrics | Host data race [inferred] | Mutex or atomic, +5 LOC |

## §2 Call-graph trace (key)

`handleAudioGeneration()` (`4441-4786`) → `prepareTalkerInput()` (`4384-4437`) → `projectToTalkerInput()` (`3968-4029`) OR host projection (`4031-4113`) → `executeTalkerPrefillStep()` (`4116-4184`, native pool `2267-2289` OR legacy LLMRunner with shared KV at `4147-4157`) → frame loop: `extractTalkerLastHidden()` (`5175-5224`) + `runCodePredictorGenerationForFrame()` (`4788-5026`) + residual (`5028-5172`) + `executeTalkerDecodingStep()` (`4186-4198`, pool `2460-2571`).

**Native CP branch** (`4897-4902`) acquires per-engine slot at `997-1016`/`1044-1062`; slot buffers at `1098-1297` are mostly safe.
**Legacy CP branch** (`4200-4310`) uses shared CP tensors — racy.

Code2Wav decoupled: `handleAudioGeneration()` returns RVQ codes at `4779-4785`. Code2Wav reads codes downstream. `StatefulCode2WavRunner::reset()` zeroes state at `statefulCode2WavRunner.cpp:254-262`. `generateChunk()` (`396-427`) swaps state object addresses — race with shared `mStates`, `mInputCodes*`, `mOutputWaveform*`, `mPositionOffset*`, single `mContext`.

## §3 Must-fix-for-N=2 prioritized list

1. **`mCodecHiddensBuffer` per-slot** — only `{1,16,talkerHiddenSize}` (`3639-3640`), cheap to move. **Highest impact / lowest cost.** Likely the actual `cudaMemsetAsync` crash site.
2. **Runtime scratch tensor slotting** — bundle items 1-5 of table into `TalkerSlot` / `CodePredictorSlot` extensions. +180-260 LOC.
3. **`resetSampling()` scope fix** — reset only acquired slot RNG, not global. +10-20 LOC.
4. **Code2Wav per-slot OR serialized** — per-slot ≈ +120-180 LOC; mutex ≈ +10 LOC (accepts loss of Code2Wav parallelism, still useful for talker/CP parallelism).
5. **Legacy runner KV cache** — if production profile uses native pools (must verify), can skip. Otherwise +10-150 LOC.
6. Nice: thread-safe metrics, guard static dump flags (`4856-4860`).

Combined with Code2Wav fix (spec §4 path a), this resolves illegal-access **only if the runtime scratch race is fixed too**. Code2Wav alone is insufficient — `mCodecHiddensBuffer` memset at `4908-4909`, `4973-4974` is its own crash vector.

## §4 VRAM risk

Talker slot duplication is the biggest cost: `deviceLogits = maxSeqLen * vocabSize * logitsElementSize`, `deviceHidden = maxSeqLen * hiddenSize * hiddenElementSize`, KV `numKVHeads * maxKVSeqLen * headDim * kvElementSize * 2 * numDecoderLayers` (× 2 for kvA/kvB). Plus prompt worst-case preallocation (`2242-2254`). At N=2 these double. CP slots smaller. `mCodecHiddensBuffer` is `{1,16,talkerHiddenSize}` — negligible.

## §5 Verdict on scope

The original spec §4 estimate (~50 LOC, "lowest risk") was **for Code2Wav alone**. Real N=2 enablement requires runtime slotting too — **conservatively 300-450 LOC C++ + 1-2 days of careful audit/test/build cycles**. Risk: high — touches the inner hot path of talker/CP/codec frame generation. N=1 audio MD5 byte-equivalence gate is mandatory after each commit.

**Alternative**: Phase B-lite = wrap `handleAudioGeneration()` in a `std::mutex` so it serializes. Combined with already-parallel `_WorkerIO` HTTP→worker submission, gains: pipelined HTTP IO + worker stdin parsing + early cancel propagation. Loses: actual GPU-side parallelism. Quick to ship (≈30 LOC), throughput improvement modest (likely 10-15% on slow-client TTFA, not 1.4-1.5×).
