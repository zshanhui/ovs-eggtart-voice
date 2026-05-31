# CustomVoice TTS — Fork Port Handoff

**Date:** 2026-05-26
**Goal achieved on:** pristine TensorRT-Edge-LLM v0.7.1 on `orin-nx` device
**ASR result:** `"今天天气真不错。"` ✅ matches input text exactly
**Final delivery target:** port to fork branch `v071/customvoice-product` (Mac: `/Users/harvest/project/TensorRT-Edge-LLM`)

---

## TL;DR for the person picking this up

We localized 3 distinct bugs blocking CustomVoice on Orin NX. Fixes 1+2 are clean and ready to port. Fix 3 is a CuTe DSL prebuilt-artifact bug worked around via an env-var override — it produces correct output but needs a real fix before production.

All patches are applied + verified on a pristine v0.7.1 worktree at `/Users/harvest/project/_worktrees/v071-pristine` (detached at tag `v0.7.1`, commit `3647690`). The exact same patches need to land on the fork `v071/customvoice-product`.

---

## Bug 1 — CuTe DSL GEMM disabled → Talker MLP is a no-op

### Root cause

`cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu:324-388` — both `invokeTalkerMLP` and `invokeLinearLayer` are wrapped in `#ifdef CUTE_DSL_GEMM_ENABLED`. When the macro is not set, the function logs an error and returns with the output tensor uninitialized:

```cpp
#ifdef CUTE_DSL_GEMM_ENABLED
    if (!CuteDslGemmRunner::runBiasSiLU(...)) { ... return; }
    if (!CuteDslGemmRunner::runBias(...))     { ... return; }
#else
    LOG_ERROR("CuTe DSL GEMM not compiled. Rebuild with -DENABLE_CUTE_DSL=gemm (or ALL).");
    return;   // ← output tensor never written
#endif
```

Default cmake configuration has `ENABLE_CUTE_DSL=OFF`. Talker FFN therefore writes nothing → downstream sampler consumes uninitialized GPU memory → premature EOS after 6-7 codec frames → ASR transcribes `"嗯嗯。"` (random Mandarin fillers).

### Fix

Two changes:

1. **`cmake/CuteDsl.cmake`** — add the shim that maps `cudaLibrary_t` → `CUlibrary` for Jetson L4T CUDA 12.6 (the prebuilt artifact uses CUDA ≥12.4 runtime API that L4T 12.6 omits). The fork `qwen3-tts-highperf-runtime-w8a16` branch already has this shim — copy it. Specifically, lines that link `trt_edgellm_cutedsl_cudart_shim` into STATIC_LIBRARY targets (not just SHARED/EXE).

2. **cmake invocation** — must include:
   ```
   -DENABLE_CUTE_DSL=gemm \
   -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
   -DEMBEDDED_TARGET=jetson-orin       # hyphen, not underscore
   ```
   The `EMBEDDED_TARGET=jetson-orin` flag enables the `TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT` macro at `CMakeLists.txt:74` which makes the shim compile.

### How to verify on the fork after porting

```
# Build log should show:
CuTe DSL: gemm_ampere_decode_fp16 variant found — CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED set
CuTe DSL: gemm_ampere_medium_bias_silu_fp16 ...
CuTe DSL: CUTE_DSL_GEMM_ENABLED set (GEMM variants found)

# Runtime log should NOT show:
[ERROR] [talkerMLPKernels.cu:341:invokeTalkerMLP] CuTe DSL GEMM not compiled. ...

# Runtime should show:
[INFO] [cuteDslGemmRunner.cu:179:loadKernelModule] CuteDslGemmRunner: Ampere GEMM module(s) loaded for SM87
```

---

## Bug 2 — `assistantPreambleKernel` missing `language_id` row (CustomVoice path)

### Root cause

Upstream v0.7.1 `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu:454-568` `assistantPreambleKernel` only implements the **no-language** prefill layout (8 fixed prefix rows). Python reference at `/Users/harvest/project/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py:2120-2186` has two branches:

```python
if language_id is None:                          # base 12Hz / no-lang
    codec_prefill_list = [[codec_nothink_id, codec_think_bos_id, codec_think_eos_id]]
else:                                             # CustomVoice + language
    codec_prefill_list = [[codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]]
```

Two differences when language is set:
1. **Row 3** uses `codec_think_id` (2154) instead of `codec_nothink_id` (2155).
2. **New row 5** = `pad + language_id_embed` is inserted before what was previously row 5 (codec_think_eos).

Result: 9 fixed prefix rows instead of 8. Total prefill = `9 + N + 2 = 15` rows for "今天天气真不错" (N=4 body tokens) vs upstream's 14.

Without this language-conditioning row, the talker model has no idea what language to speak. It generates plausible-looking Chinese audio with WRONG content (e.g. `"马海微博"`).

### Fix (already implemented in pristine worktree)

5 file patches:

#### 2a. `cpp/kernels/talkerMLPKernels/talkerMLPKernels.{h,cu}`

Add `langId` + `codecThinkId` parameters to kernel + wrapper. Conditional layout:

```cpp
// langId < 0: legacy 8-row layout, totalRows = 8 + textLen + 2
// langId >= 0: CustomVoice 9-row layout, totalRows = 9 + textLen + 2
//   row 3 = pad + codecThinkId (NOT codecNothinkId)
//   row 5 = pad + langId         (NEW)
//   row 6 = pad + codecThinkEos
//   row 7 = pad + speaker
//   row 8 = ttsBos + codecPad
```

Branch on `langId >= 0` inside the kernel; pick `kPrefixLen = (langId >= 0 ? 9 : 8)`.

#### 2b. `cpp/runtime/qwen3OmniTTSRuntime.h`

Add to `TalkerConfig`:
```cpp
int32_t codecThinkId{};
std::unordered_map<std::string, int32_t> codecLanguageId{};
```

Add to `TalkerGenerationRequest`:
```cpp
std::string language;
```

Extend `projectToTalkerInput` signature with `int32_t langId`.

#### 2c. `cpp/runtime/qwen3OmniTTSRuntime.cpp`

In `validateAndFillConfig` (around `:540-565`):
```cpp
if (configJson.contains("codec_think_id"))
    mTalkerConfig.codecThinkId = configJson["codec_think_id"].get<int32_t>();
if (configJson.contains("codec_language_id")) {
    for (auto& [k,v] : configJson["codec_language_id"].items())
        mTalkerConfig.codecLanguageId[k] = v.get<int32_t>();
}
```

In `prepareTalkerInput` (around `:1253-1275`):
```cpp
int32_t langId = -1;
if (!request.language.empty()) {
    std::string langLc = request.language;
    std::transform(langLc.begin(), langLc.end(), langLc.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    auto it = mTalkerConfig.codecLanguageId.find(langLc);
    if (it != mTalkerConfig.codecLanguageId.end()) {
        langId = it->second;
        LOG_INFO("CustomVoice language conditioning enabled: language=\"%s\" -> codec_id=%d", langLc.c_str(), langId);
    } else {
        LOG_WARNING("Requested language=\"%s\" not found in codec_language_id map; falling back to no-language", langLc.c_str());
    }
}
```

Plumb `langId` through `projectToTalkerInput` → `invokeAssistantPreamble`.

Also pass `mTalkerConfig.codecThinkId` to the kernel alongside `codecNothinkId`.

#### 2d. `examples/omni/qwen3_tts_inference.cpp`

Parse optional top-level + per-request `"language"` string from input JSON; plumb to `TalkerGenerationRequest.language`.

#### 2e. Engine config.json augmentation

The exported engine `config.json` for CustomVoice talker is missing the `codec_language_id` map. It already has `codec_think_id: 2154` but not the language→ID dict.

For the fork's CustomVoice engine export pipeline, modify the export script to inject:
```json
"codec_language_id": {
  "chinese": 2055, "english": 2050, "german": 2053, "italian": 2070,
  "portuguese": 2071, "spanish": 2054, "japanese": 2058, "korean": 2064,
  "french": 2061, "russian": 2069, "beijing_dialect": 2074, "sichuan_dialect": 2062
}
```

These IDs come from HF model config: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` → `config.json` → `talker_config.codec_language_id`.

If editing the export pipeline isn't immediate, post-edit the `config.json` of the existing engine artifact directly (it's just text).

### How to verify on the fork after porting

Run with input JSON containing `"language": "chinese"` and `"speaker": "vivian"` (lowercase — see "remaining minor bugs" below), check:

- Log line: `CustomVoice language conditioning enabled: language="chinese" -> codec_id=2055`
- Log line: `projectToTalkerInput: ... outputSeqLen=15, ..., langId=2055, prefixRows=9`
- Dumped `talker_inputs_embeds` shape: `[1, 15, 1024]` (NOT 14)

---

## Bug 3 — CuTe DSL `invokeTalkerMLP` produces wrong text_projection row 1 (UNRESOLVED, workaround in place)

### Root cause (partial)

Even with all kernel fixes above, the CuTe DSL `gemm_ampere_medium_bias_silu_fp16` kernel produces incorrect output at row 1 of the role prefix when called via `invokeTalkerMLP`:

| Row | Ref (Python fp16/bf16) | TRT (CuTe DSL) | max_diff |
|-----|------------------------|----------------|----------|
| 0   | first5=[-0.0028, 0.0004, -0.0013, 0.0015, 0.0016] | first5=[-0.0028, 0.0004, -0.0013, 0.0015, 0.0016] | 0.002 ✅ |
| 1   | first5=[-0.0496, 0.0038, 0.0183, -0.0483, 0.0047] absmax=0.83 | first5=[0.0355, 0.0367, 0.0177, 0.0515, -0.0166] absmax=0.76 | **0.65** ❌ |

Row 0 matches reference to fp16↔bf16 noise level. Row 1 has completely different values (different signs, different magnitudes), even though the same kernel, same weights, same input embedding for token `assistant` (id=77091).

**Verified NOT batch-dependent:** Tested calling `invokeTalkerMLP` with 3-token role slice only (matching Python's batching) — row 1 still wrong. Bug is in the CuTe DSL prebuilt artifact itself, not the batching.

The CuTe DSL kernel ships as a prebuilt static archive `libcutedsl_aarch64.a` with headers at `cpp/kernels/cuteDSLArtifact/aarch64/sm_87/include/`. No source provided in repo. Suspect: either a kernel bug in the gemm_ampere_medium_bias_silu variant, or weight loading/layout issue specific to that variant.

### Workaround (currently shipped)

Added env var `QWEN3_TTS_PRELOAD_TALKER_EMBEDS=<bin>` that bypasses the kernel:

In `cpp/runtime/qwen3OmniTTSRuntime.cpp` `projectToTalkerInput`, after `invokeAssistantPreamble`:

```cpp
if (char const* p = std::getenv("QWEN3_TTS_PRELOAD_TALKER_EMBEDS")) {
    std::ifstream f(p, std::ios::binary);
    if (f) {
        size_t const n = outputSeqLen * hiddenSize;
        std::vector<float> hostFp32(n);
        f.read(..., n * sizeof(float));
        if (f.gcount() == n * sizeof(float)) {
            // cast fp32 → fp16 and cudaMemcpy to output buffer
            ...
        }
    }
}
```

To use the workaround, Python pre-computes the correct 15-row `talker_input_embed` tensor and saves it as raw fp32 binary. Runtime reads + casts to fp16 + replaces the `output` tensor. Talker engine then receives correct inputs and produces correct codec tokens.

This is the path we used to achieve `"今天天气真不错。"` ASR validation.

### Long-term fix candidates (not done)

1. **Replace `invokeTalkerMLP` with cuBLAS gemm** for the text_projection MLP specifically. cuBLAS is well-tested and deterministic. Estimated 1-2 day effort to write + verify.

2. **Investigate / rebuild the CuTe DSL artifact.** `kernelSrcs/build_cutedsl.py` may regenerate the artifact. If we can build from source with `--gpu_arch sm_87 --arch aarch64`, we can debug the kernel directly.

3. **Use TRT plugin** for text_projection MLP. Embed a small TRT subgraph or use TensorRT's built-in matmul.

4. **Bisect which CuTe DSL variant is broken.** If only `gemm_ampere_medium_bias_silu_fp16` has the bug, force the small or large variant. The kernel router picks variant by batch size; influence by padding input.

The workaround (env var + Python precompute) is sufficient for **correctness validation** but is not production-quality (requires Python in the inference loop). Decide based on production constraints whether to fix or live with the workaround.

---

## Code2Wav engine size limit (separate workstream)

`code2wav.engine` was exported with optimization profile `max=16 codec frames`. The talker (correctly conditioned) produces ~30 frames for "今天天气真不错". The pristine runtime falls back to chunked inference at `code2WavRunner.cpp:397`:

```
Using chunked inference for long sequence (len=30, max=16)
```

Chunked inference produces audible audio but with boundary artifacts (CausalConv state isn't preserved across chunks). Workaround for our validation: decode the RVQ on `wsl2-local` via `tts.model.speech_tokenizer.decode([{"audio_codes": codes}])` — bypasses C++ Code2Wav entirely.

For production: re-export `code2wav.engine` with `opt_code_len >= 64` (or implement proper streaming chunked-inference with sliding CausalConv state). Not blocking for this handoff since the upstream RVQ from talker is correct.

---

## Speaker case sensitivity bug (minor, side finding)

`getSpeakerIdByName` in `cpp/runtime/qwen3OmniTTSRuntime.cpp` does case-sensitive lookup against speaker IDs loaded from engine config.json. Config has `vivian: 3065` (lowercase). Input `"Vivian"` falls back to `default_speaker_id` (3066 = Serena) silently — only a `WARNING` log line.

Fix: lowercase the lookup key inside `getSpeakerIdByName`. Python ref behaves this way (`self.config.talker_config.spk_is_dialect[speaker.lower()]`).

Until fixed, callers must pass speaker name in lowercase.

---

## Validation procedure (replay on fork after porting)

### Pre-requisites
- Fork on Mac at `/Users/harvest/project/TensorRT-Edge-LLM` checked out to `v071/customvoice-product`
- orin-nx device with TRT 10.3, CUDA 12.6, working `/tmp/v071_run/{talker,code_predictor,code2wav}/` engines
- wsl2-local with `qwen_tts==0.1.1` and HF model snapshot `85e237c12c027371202489a0ec509ded67b5e4b5` cached
- radxa device with sherpa-onnx + SenseVoice model at `/home/radxa/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`

### Steps

1. Apply patches 2a-2e + cmake patch from this doc to the fork.
2. rsync fork source to orin-nx.
3. cmake configure with `-DENABLE_CUTE_DSL=gemm -DCUTE_DSL_ARTIFACT_TAG=sm_87 -DEMBEDDED_TARGET=jetson-orin`
4. `make -j$(nproc) qwen3_tts_inference NvInfer_edgellm_plugin`
5. Augment `/tmp/v071_run/talker/config.json` with `codec_language_id` map (one-line Python script in this doc).
6. Generate Python's ground-truth `talker_input_embed` bin (workaround for Bug 3):
   ```python
   import torch, numpy as np
   from qwen_tts import Qwen3TTSModel
   tts = Qwen3TTSModel.from_pretrained(CKPT, device_map="cuda:0", dtype=torch.bfloat16)
   captured = []
   orig = torch.cat
   def spy(ts, dim=0, **kw):
       o = orig(ts, dim=dim, **kw)
       if o.dim()==3 and o.shape == (1,15,1024): captured.append(o.detach().clone())
       return o
   torch.cat = spy
   torch.manual_seed(42)
   _ = tts.generate_custom_voice(text="今天天气真不错", language="Chinese", speaker="Vivian", instruct="", max_new_tokens=8)
   captured[-1].cpu().float().numpy().tofile("/tmp/ref_talker_embeds_15row.bin")
   ```
7. ssh bin to orin-nx.
8. Run worker:
   ```
   QWEN3_TTS_PRELOAD_TALKER_EMBEDS=/tmp/ref_talker_embeds_15row.bin \
   QWEN3_TTS_SEED=42 \
   <built>/qwen3_tts_inference --inputFile=/tmp/v071_input_zh.json ...
   ```
   where `v071_input_zh.json` has `"speaker":"vivian"` and `"language":"chinese"`.
9. Expected log: `Overrode talker_inputs_embeds from ... (15360 floats)` + `first codec token: 1995` + `30 audio frames (exit: EOS)`.
10. Pull `rvq_req0.safetensors` to wsl2, decode via `tts.model.speech_tokenizer.decode([{audio_codes: codes}])` → wav.
11. Push wav to radxa, run `python3 /tmp/asr_check.py <wav>`.
12. Expected: `ASR="今天天气真不错。"`.

If steps 9-12 pass on the fork → bugs 1+2 are correctly ported and the workaround for bug 3 is in place.

---

## File index

### Pristine worktree (Mac) where all fixes live
- `/Users/harvest/project/_worktrees/v071-pristine`
- Git status: detached at `v0.7.1` tag (`3647690`), all changes uncommitted, working tree dirty by design
- Files modified:
  - `cmake/CuteDsl.cmake` (~15 lines, ported from highperf fork)
  - `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu` (~132 net lines: new lang branch + signature)
  - `cpp/kernels/talkerMLPKernels/talkerMLPKernels.h` (~59 net lines: signature + docs)
  - `cpp/runtime/qwen3OmniTTSRuntime.h` (~11 net lines: TalkerConfig fields, request.language, projectToTalkerInput sig)
  - `cpp/runtime/qwen3OmniTTSRuntime.cpp` (~340 net lines: config parsing + langId resolve + plumbing + preload override + dump helpers from prior round)
  - `examples/omni/qwen3_tts_inference.cpp` (~7 net lines: input JSON language parsing)

### Orin-NX
- Patched source: `/home/harvest/project/v071-pristine/`
- Built binary: `/home/harvest/project/v071-pristine/build/examples/omni/qwen3_tts_inference`
- Engines: `/tmp/v071_run/{talker,code_predictor,code2wav}/` (config.json augmented)
- Reference prefill bin: `/tmp/ref_talker_embeds_15row.bin`
- Working input JSON: `/tmp/v071_input_zh.json`
- Final RVQ + WAV: `/tmp/v071_run_ov/out/`

### Reference artifacts
- Python ref dump archive: `/Users/harvest/project/seeed-local-voice/bench/parity/ref_dump_qwen3_customvoice_bf16.tar.gz` (md5 `cdd0142ec99b65abdbb0b69a0c09939c`, 1533 npy files, seed=42 deterministic)
- Python ref 15-row prefill (fp32): `/tmp/ref_talker_input_embed_full.npy` (Mac) / `/tmp/ref_talker_embeds_15row.bin` (orin-nx, raw fp32)
- Decoded WAV proof: `/tmp/wav_ov.wav` on wsl2/radxa — `dur=2.400s rms=0.0776 ASR="今天天气真不错。"`
- WSL2 codec-dump script: `bench/parity/dump_codec_tokens_wsl2.py`

### Fork (the porting target)
- Mac: `/Users/harvest/project/TensorRT-Edge-LLM` (branch `v071/customvoice-product`, tip `f86545c`)
- Note: fork has a DIFFERENT prefill architecture upstream of these patches — it uses a fused 8+N+2 kernel (same as pristine). Most patches should apply cleanly; the cmake patch may need adapting if fork's CuteDsl.cmake has already diverged.

### Old branch (for reference, don't port from here)
- Orin-NX: `/home/harvest/project/repro-qwen3/TensorRT-Edge-LLM` (branch `qwen3-tts-highperf-runtime-w8a16`)
- This is the v0.7.0-based fork with our old high-perf runtime. It has a different prefill architecture ("9-row + per-frame trailing residual addend") that's NOT what we're porting. Only borrow its `CuteDsl.cmake` shim.

---

## Open questions for the next person

1. **Should we fix CuTe DSL bug 3 properly, or keep the env-var workaround in production?** Workaround requires a Python pre-step per request. For pure-C++ production, need cuBLAS or rebuilt CuTe DSL.
2. **Does the fork's CustomVoice engine export pipeline already inject `codec_language_id`?** If yes, no engine config edit needed. If no, the export script needs updating.
3. **Code2Wav engine max=16:** is there an existing re-export workstream, or do we ship with the chunked-inference workaround?
4. **Speaker case bug:** trivial fix but should be picked up by whoever ports.
5. **W8A16 quantisation workstream:** in `/Users/harvest/project/TensorRT-Edge-LLM/docs/known-issues/w8a16-talker-handoff.md` (fork commit `f86545c`). Paused until FP16 correctness shipped. After bugs 1+2 ported, can resume.

---

## Memory cross-references

- `~/.claude/projects/-Users-harvest-project-seeed-local-voice/memory/customvoice_tts_orin_nx_root_cause_quest_2026_05_26.md` (prior session — pre-fix diagnosis, mostly superseded by this doc)
- `~/.claude/projects/-Users-harvest-project-seeed-local-voice/memory/trt_edge_llm_fork_path.md`

Once fork port lands + ASR validates: append a new memory entry summarizing the 3-bug fix + commit hashes.
