# MOSS-TTS-Nano KV buffer dtype ABI mismatch — root cause + fix

**Status:** PRODUCTION FIXED 2026-05-24
**Affected:** `moss_tts_nano_worker` (C++ TRT path) on Orin NX/Nano
**Impact before fix:** Frame-0 audio sane, frame ≥1 garbled → ASR returned trailing-token hallucination. The ORT subprocess path was the only working production option.
**Impact after fix:** C++ TRT path TTFA **157 ms** (19× faster than ORT CPU EP **3000 ms**), 3 Chinese prompts ASR CER=0, byte-identical output across 5 sequential runs.

---

## 1. Symptom

After the v16 ONNX rebuild switched MOSS-TTS-Nano decode-engine KV (`past_key_*`, `past_value_*`, `present_key_*`, `present_value_*`) IO from FP16 to FP32:

| Frame | C++ TRT path | ORT path |
|---|---|---|
| Frame 0 token | Byte-identical to ORT (seed=42 RNG) | Reference |
| Frame ≥1 | Diverges → trailing token hallucination → ASR transcription drift | Sane |

The frame-0 output matching ORT byte-for-byte with the same seed misled diagnostics for several rounds — the bug was hidden by the fact that frame 0's audio sample depends on the prefill `globalHidden` device tensor (a separate FP32 binding allocated independently of KV scratch), so it produces the correct token even when KV is broken.

The corruption manifested starting at frame 1 because that is the first frame whose `past_key_*` input comes from a `present_key_*` that was written to a half-sized scratch — overlapping subsequent layer writes.

## 2. Root cause

`mossTtsNanoRuntime.cpp` had `sizeof(half)` (== 2 bytes) hardcoded in 8 KV-buffer sizing call-sites:

- Per-layer global KV stride (`mGlobalLayerKvBytes`)
- Per-layer local KV stride (`mLocalLayerKvBytes`)
- Max present scratch per layer (`mMaxPresentLayerBytes`)
- Per-step KV bytes during decode loop (multiple places: prefill copy-back, decode bindings, post-decode copy-back, slot KV layer offset)

When TRT engines were rebuilt with FP32 KV IO (v16 rebuild done to chase a separate divergence hypothesis), TRT now wrote `4 × N × heads × dim` bytes per layer into a `2 × N × heads × dim`-allocated scratch region. The K-write for layer `L` overwrote the K-region of layer `L+1`, the V-write of layer `L` then overwrote the still-corrupted layer `L+1` region — by the time `cudaMemcpy`'d back into `slot.globalKvDevice`, every layer past 0 contained mixed payloads.

The corruption was clean enough that decode produced *plausible-looking* tokens that drifted toward the model's high-prior trailing-token distribution rather than crashing or producing NaNs — making it look like a "model quality issue" or "EOS hallucination", which masked it as a model bug for ~1 week (see `[[moss_tts_nano_model_quality_findings]]`).

## 3. Diagnostic trail (6+ rounds, kept here so others don't repeat)

The path to root cause (so the next person porting a TTS model knows what to check):

1. **v1-v3 (rng / sampling parity)** — verified Python vs C++ `np.random.default_rng(42)` byte-identical token-0 output. PASSED → rules out RNG. Frame ≥1 still diverges.
2. **v4-v8 (attention mask / rope cache / paged-vs-flat KV)** — tried 4 paged KV layouts; all matched ORT on frame 0 but drifted thereafter. PASSED layouts → rules out paged-KV mapping.
3. **v9-v11 (voice-prompt length / EOS heuristics / model output sampling)** — hypothesized model EOS at unusual frames. ORT was tested without this issue at same length. RULED OUT.
4. **v13-v15 (ONNX rebuilds + IO format experiments)** — rebuilt decode engine with explicit FP32 KV IO (v16). C++ TRT still diverges; **frame-0 RNG byte-equal trick re-confirmed C++ path is consistent with ORT internally up to KV write-back**.
5. **v16 (byte-level dump of `past_key_0` at frame 1 entry)** — diffed C++ `past_key_0` device contents at the first decode step vs ORT-equivalent. **Found**: only first ~12 KB matched; from byte 12288 onward, content shifted by a half-element pattern. Ratio of misalignment was exactly 2:1 → element-size mismatch.
6. **Final diagnosis** — grep `sizeof(half)` in `mossTtsNanoRuntime.cpp` → 8 hits. Cross-referenced with `nvinfer1::DataType const dtype = mDecodeEngine->getTensorDataType("past_key_0");` returning `kFLOAT` (== 0). **Hardcode was the bug.**

## 4. Fix (committed to fork repo `qwen3-tts-highperf-runtime-w8a16` branch)

`cpp/runtime/mossTtsNanoRuntime.{cpp,h}`:

- Added member `size_t mKvElementSize{2};` (defaults to FP16 for back-compat if probe fails)
- Added helper `size_t tensorElementSize(nvinfer1::DataType) const` that maps `kFLOAT→4`, `kHALF→2`, `kINT32→4`, etc.
- In `loadEngines()`, after `mDecodeEngine` load, probe `getTensorDataType("past_key_0")` and assign `mKvElementSize = tensorElementSize(kvDtype)`. Log:

  ```
  [moss] KV element dtype=0 size=4 bytes (FP32=0 FP16=1)
  ```

- Replaced all 8 `sizeof(half)` hardcodes in KV-buffer sizing arithmetic with `mKvElementSize`.
- `mGlobalHiddenDtype` was already probed; ensured it stays separate from KV dtype (they may differ — global hidden was always FP32 in v16, KV could be either).

Diff stat: `cpp/runtime/mossTtsNanoRuntime.{cpp,h}` were new in the fork (untracked) — being added under one commit.

## 5. Verification (2026-05-24 burn-in on orin-nx)

Worker: `/opt/jv-workers/moss_tts_nano_worker` md5 `3017b3f34bb9c4cbc8391f65ecd84541` (post-fix).
Previous binary backed up to `/opt/jv-workers/moss_tts_nano_worker.before_kvdtype_fix` md5 `7be68fe0c2d83042227d14d314e7d4e2`.

```
SUMMARY (9 scenarios, 0 errors, 0 worker crashes):
  short ("你好")              exit=0 TTFA=152ms dur=0.56s chunks=2  md5=3989341fccd21950508dccc27bb22300
  medium ("...天气真不错")    exit=0 TTFA=154ms dur=4.24s chunks=8  md5=fbe5a9a2e07c63fa88a3bbe1866bfdc1
  longer (41-char zh)         exit=0 TTFA=156ms dur=9.60s chunks=16 md5=c2f9217bb1c0af0c90f5a94bb513b69e
  clone (ref-audio zh_1_ref)  exit=0 TTFA=1063ms dur=1.36s chunks=3  md5=b89bb62595e8a92716ec7c0281d446c2
  rep0..rep4 (same prompt)    exit=0 TTFA=152-153ms dur=4.24s chunks=8 md5=fbe5a9a2... (all 5 byte-identical)
```

Key findings:
- **5 sequential runs of the same prompt are byte-identical (md5 match)** → no KV state accumulation across worker invocations.
- **Longer prompt (41-char Chinese) produces 9.6s audio** — no premature truncation, no EOS hallucination.
- **TTFA stable at 152-156ms** across short/medium/longer prompts (the prior round measured 157ms; consistent).
- **Voice clone path** delivers 1.36s audio with higher TTFA (1063ms) due to reference-audio codec encoding — this is expected, also consistent with prior baseline.

ASR validation (3 prompts CER=0) was performed in the prior diagnostic round (see memory `[[moss_tts_nano_trt_production_ready]]`) using radxa qwen3_asr_rk; current burn-in confirms the binary is deterministic and stable, so the audio quality is preserved.

## 6. Lesson for future TTS porting

Add to `docs/playbooks/tts-model-edge-port-playbook.md` (separate commit):

> Never hardcode `sizeof(half)` (or any element size constant) in KV-buffer sizing in a TRT runtime when the engine IO format can change across rebuilds. Always probe `getTensorDataType(<binding name>)` and compute element size dynamically.

A symptom that hints at this class of bug is when frame-0 matches reference byte-for-byte but frame-1 drifts — frame 0 is computed from prefill outputs, which often have *independent* dtype/binding from the decode-loop KV scratch.

## 7. Files touched

- `TensorRT-Edge-LLM/cpp/runtime/mossTtsNanoRuntime.cpp` (new + modified)
- `TensorRT-Edge-LLM/cpp/runtime/mossTtsNanoRuntime.h` (new + modified)
- `TensorRT-Edge-LLM/cpp/workers/build_moss_worker.sh` (auto-rebuild of runtime obj)
- `seeed-local-voice/configs/profiles/jetson-moss-tts-nano-trt.json` (new profile pointing to C++ binary)
- `seeed-local-voice/docs/runbooks/moss-tts-nano-deployment.md` (TRT path documentation)
- `seeed-local-voice/docs/specs/moss-tts-nano-kv-dtype-abi-fix.md` (this doc)

## 8. Reproduction

Build the post-fix worker on Orin NX (host) — see runbook section "Building the C++ TRT worker".

Smoke test:

```bash
ssh orin-nx 'python3 /tmp/moss_burnin.py 2>&1 | tail -20'
```

Expected: 9/9 scenarios PASS, TTFA 150-160 ms range, 5 repeat runs byte-identical.

Engine log on worker start should contain:

```
[moss] KV element dtype=0 size=4 bytes (FP32=0 FP16=1)
```

If `dtype=1 size=2`, engines are FP16 KV (older rebuild) — the dynamic probe also works for that, no recompile needed.
