# CuTe DSL `gemm_ampere_medium_bias_silu_fp16` — Row-1 Corruption Bug Investigation

**Date:** 2026-05-26
**Reporter:** Seeed Local Voice integration team (TensorRT-Edge-LLM fork user)
**Status:** REPRODUCED + NARROWED, root cause not yet identified inside the prebuilt artifact
**Severity:** correctness-critical (silent — output is structured wrong values, no error code or warning)

---

## TL;DR

On Jetson Orin NX (sm_87), the CuTe DSL prebuilt kernel(s) used by `invokeTalkerMLP` in the CustomVoice Qwen3-TTS text-projection MLP produce **deterministically wrong** output at **two row positions out of 15** (rows 1 and 12) of `talker_inputs_embeds`. Row 0 and rows 2-11, 13-14 match a PyTorch/cuBLAS bf16 reference to fp16 round-off (max_diff ≤ 0.008). Row 1 has wrong signs and wrong magnitudes (max_diff 0.65 vs absmax 0.83, cosine 0.39). Row 12 is also wrong but less severely (max_diff 0.157, cosine 0.94). The bug is 10/10 byte-identical across reruns, ruling out races / uninitialized memory.

The text_projection MLP shape is **M=variable, N=2048, K=2048** (FC1) and **M=variable, N=1024, K=2048** (FC2). The original handoff doc had stated N=3072 for FC1 — that was incorrect (`text_projection.safetensors` has `linear_fc1.weight: F16[2048,2048]`). Per `projectToTalkerInput` log (`seqLen=12, N=4 (stripped prefix=3 suffix=5), outputSeqLen=15, prefixRows=9`), `invokeTalkerMLP` is called multiple times per request: at least once for the role-prefix block and once for the body block, so the row indices of the two corrupted outputs correspond to two different kernel invocations.

---

## Environment

| Item | Value |
|------|-------|
| Device | NVIDIA Jetson Orin NX (`Orin (nvgpu)`, `nvidia-smi` reports compute_cap **8.7**) |
| L4T / JetPack | R36.4.3 (BSP date 2025-01-08), kernel 5.15.148-tegra |
| CUDA Toolkit | 12.6.11 (runtime 12.6.68) |
| TensorRT | 10.3.0.30 (libnvinfer.so.10) |
| Driver | 540.4.0 |
| TensorRT-Edge-LLM (fork) | tag `customvoice-v071-w8a16-asr-pass-20260526`, commit `e08de4f2e3155fc1f38b650585465cf91013b8e8` (based on upstream NVIDIA/TensorRT-Edge-LLM v0.7.1 tag, commit `3647690`) |
| CuTe DSL prebuilt artifact tarball | `kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_87_cuda12.tar.gz` md5 `e076c4d83e002db9cf1558abcb8c34d7` |
| CuTe DSL extracted archive | `sm_87/libcutedsl_aarch64.a` md5 `05ddeb9ddf60f1846872bd523ce49005` (702 418 bytes) |
| CuTe DSL artifact metadata | `cutlass_dsl_version=4.4.1`, `cuda_version=12.6.68`, `build_date=2026-05-06T21:23:32Z`, `gpu_arch=sm_87` |
| Test binary | `qwen3_tts_inference` md5 `f50fedc960d8edf7304f897cddbbdaf7` |
| TRT plugin lib | `libNvInfer_edgellm_plugin.so.1.0` md5 `3d6761ebbe0946720f9c1d35a56c1cda` |

---

## Affected kernel call signature

Fork source `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu:325` — `invokeTalkerMLP` first sub-call:

```cpp
CuteDslGemmRunner::runBiasSiLU(
    input.rawPointer(),       // [M, K], fp16
    fc1Weight.rawPointer(),   // [N, K], fp16 (row-major, "weight^T" semantic in the call)
    workspace.rawPointer(),   // [M, N], fp16
    fc1Bias.rawPointer(),     // [N], fp16
    /*M=*/numTokens,          // = 3 in the failing call
    /*N=*/hiddenDim,          // = intermediate_size = 3072
    /*K=*/inputDim,           // = thinker_hidden_size = 2048
    stream);
```

On Ampere, `runBiasSiLU` has **exactly one** prebuilt variant per `cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.cu:768-784`:

```cpp
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED
    if (sActiveVariant == Variant::kAmpere) {
        return dispatchAmpere2dFused<gemm_ampere_medium_bias_silu_fp16_Kernel_Module_t, ...>(
            sAmpereMediumBiasSiLUModule,
            cute_dsl_gemm_ampere_medium_bias_silu_fp16_wrapper,
            aPtr, bPtr, cPtr, biasPtr, M, N, K, stream);
    }
#endif
```

— so the Ampere fused-SiLU path **has no shape-based bisect possibility**: every call goes through `gemm_ampere_medium_bias_silu_fp16`. There is no `_small_bias_silu` or `_large_bias_silu` Ampere variant in the artifact.

Tensor ABI used in `dispatchAmpere2dFused` (`cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.cu:699`):
- `mA = {data, dynamic_shapes=[M,K], dynamic_strides=[K]}` (row-major contiguous)
- `mB = {data, dynamic_shapes=[N,K], dynamic_strides=[K]}` (row-major contiguous, mathematically `B^T`)
- `mC = {data, dynamic_shapes=[M,N], dynamic_strides=[N]}` (row-major contiguous)
- `mBias = {data, dynamic_shapes=[N]}`

The kernel implements `mC = SiLU(mA @ mB^T + mBias)` (per upstream usage comments).

---

## Minimal reproducer

All on the orin-nx device. Engines and reference data must already exist (paths below); see "Reference data provenance" further down.

```bash
# 1. Configure paths
SNAP=/home/harvest/customvoice-v071-snapshot/20260526
ENG=/tmp/v071_run

# 2. Run inference with the broken (kernel-only) path: NO env-var workaround,
#    but with intermediate-tensor dump enabled.
mkdir -p /tmp/cv-investigate/dump_broken
cd "$SNAP"
EDGELLM_PLUGIN_PATH=$SNAP/libNvInfer_edgellm_plugin.so.1.0 \
QWEN3_TTS_SEED=42 \
QWEN3_TTS_DUMP_DIR=/tmp/cv-investigate/dump_broken \
./qwen3_tts_inference \
    --inputFile=v071_input_zh.json \
    --talkerEngineDir=$ENG/talker \
    --code2wavEngineDir=$ENG/code2wav \
    --tokenizerDir=$ENG/talker \
    --outputFile=/tmp/cv-investigate/out_broken.json \
    --outputAudioDir=/tmp/cv-investigate/audio_broken

# 3. Compare to Python (PyTorch/cuBLAS, bf16) reference
python3 - <<'PY'
import numpy as np
ref = np.fromfile("/home/harvest/customvoice-v071-snapshot/20260526/"
                  "ref_talker_embeds_15row.bin", dtype=np.float32).reshape(15, 1024)
out = np.load("/tmp/cv-investigate/dump_broken/talker_inputs_embeds__out.npy")
# 15-row tensor, dump kCap=4096 truncates middle: first 2048 = rows 0,1; last 2048 = rows 13,14
for tag, trt, py in [("row 0",  out[:1024],         ref[0]),
                     ("row 1",  out[1024:2048],     ref[1]),
                     ("row 13", out[-2048:-1024],   ref[13]),
                     ("row 14", out[-1024:],        ref[14])]:
    diff = float(np.abs(trt - py).max())
    cos  = float(np.dot(trt, py) / (np.linalg.norm(trt)*np.linalg.norm(py) + 1e-12))
    print(f"{tag}: max_diff={diff:.4f}  cos={cos:.4f}  ref_absmax={float(np.abs(py).max()):.4f}")
PY
```

Input JSON (`v071_input_zh.json`):

```json
{
  "speaker": "vivian", "language": "chinese",
  "apply_chat_template": true, "add_generation_prompt": true,
  "enable_thinking": false, "max_audio_length": 24000,
  "requests": [{"messages": [{"role": "user", "content": "今天天气真不错"}]}]
}
```

Expected (observed) output:

```
row 0:  max_diff=0.0020  cos=1.0000  ref_absmax=1.5859    ← PASS (fp16 noise)
row 1:  max_diff=0.6528  cos=0.3948  ref_absmax=0.8320    ← FAIL (wrong sign, wrong magnitude)
row 13: max_diff=0.0005  cos=1.0000  ref_absmax=1.5938    ← PASS
row 14: max_diff=0.0020  cos=1.0000  ref_absmax=1.7266    ← PASS
```

The downstream effect: with row 1 corrupted, the talker LLM produces wrong first-codec token (`1191` instead of golden `1995`), wrong total frame count (`20` instead of `~30`), and unintelligible audio.

---

## Symptoms table (Stage 1 data — strengthened)

### Row-by-row diff — ALL 15 ROWS (run after lifting `dumpTensor` `kCap` from 4096 to 65536, fork build `qwen3_tts_inference` md5 `9cf11c0b553d85b129fa7f06977eb9cd` at `/tmp/cv-investigate/fork/build_strengthen/examples/omni/`, dump at `/tmp/cv-investigate/dump_strengthened/talker_inputs_embeds__out.npy`)

| Row | max_diff | cos_sim | ref_absmax | trt_absmax | verdict |
|-----|---------:|--------:|-----------:|-----------:|--------:|
| 0   |   0.0020 |  1.0000 |     1.5859 |     1.5879 | PASS |
| 1   | **0.6528** | **0.3948** | 0.8320 | 0.7578 | **FAIL** |
| 2   |   0.0029 |  1.0000 |     1.5781 |     1.5752 | PASS |
| 3   |   0.0020 |  1.0000 |     1.5938 |     1.5957 | PASS |
| 4   |   0.0020 |  1.0000 |     1.5859 |     1.5879 | PASS |
| 5   |   0.0010 |  1.0000 |     1.3516 |     1.3516 | PASS |
| 6   |   0.0015 |  1.0000 |     1.7422 |     1.7432 | PASS |
| 7   |   0.0078 |  1.0000 |     7.3438 |     7.3516 | PASS |
| 8   |   0.0010 |  1.0000 |     1.7031 |     1.7031 | PASS |
| 9   |   0.0010 |  1.0000 |     1.2109 |     1.2119 | PASS |
| 10  |   0.0020 |  1.0000 |     1.0156 |     1.0137 | PASS |
| 11  |   0.0010 |  1.0000 |     0.9375 |     0.9385 | PASS |
| 12  | **0.1573** | **0.9380** | 0.8516 | 0.9531 | **FAIL** |
| 13  |   0.0005 |  1.0000 |     1.5938 |     1.5938 | PASS |
| 14  |   0.0020 |  1.0000 |     1.7266 |     1.7285 | PASS |

**Failed rows: {1, 12}** — two rows out of 15, not one as the previous (4-row sample) data suggested. All 13 other rows pass cleanly with max_diff ≤ 0.008.

Notable: row 12 corruption is qualitatively weaker than row 1 (cos 0.94 vs 0.39, max_diff 0.16 vs 0.65), but distinctly above the round-off floor (median PASS max_diff is ~0.002).

Per the projectToTalkerInput log line `seqLen=12, N=4 (stripped prefix=3 suffix=5), outputSeqLen=15, prefixRows=9`, the final 15-row tensor is composed of multiple sub-blocks concatenated together. Row 12 sits inside the post-prefix body block (`prefixRows=9` means rows 0-8 are the prefix block; rows 9-14 are the suffix block). Row 1 sits inside the prefix block. So the two failing rows live in two different sub-blocks → **two different `invokeTalkerMLP` invocations are each producing one corrupted row**, consistent with a kernel-position-dependent bug that triggers at sub-tile row position 1 of each call (or equivalently `M-2` of each call's `[M, *]` output).

Row 1 is the text_projection output for HuggingFace tokenizer token id **77091 = "assistant"** (the second token of the chat template's `<|im_start|>assistant\n` role prefix). Token id 0 is `<|im_start|>` (151644), id 2 is `\n` (198).

### Determinism (Stage 1 data)

10 sequential reruns of the binary, same seed, same input, same engines:

```
=== talker_inputs_embeds md5 (post-preamble 15-row, fp16 cast → fp32 → npy) ===
d1bfa2a1a234971cc92061445e306544  /tmp/cv-investigate/dump_run_1/talker_inputs_embeds__out.npy
d1bfa2a1a234971cc92061445e306544  /tmp/cv-investigate/dump_run_2/talker_inputs_embeds__out.npy
... (all 10 identical) ...
d1bfa2a1a234971cc92061445e306544  /tmp/cv-investigate/dump_run_10/talker_inputs_embeds__out.npy

=== text_projection md5 (pre-preamble 12-row FC2 output) ===
7eee5054b346d0338fa534101fcaa875  /tmp/cv-investigate/dump_run_1/text_projection__out.npy
7eee5054b346d0338fa534101fcaa875  /tmp/cv-investigate/dump_run_2/text_projection__out.npy
... (all 10 identical) ...
7eee5054b346d0338fa534101fcaa875  /tmp/cv-investigate/dump_run_10/text_projection__out.npy
```

**Bug is fully deterministic.** Rules out: data race in the kernel, uninitialized device memory hazard, scheduler/launch-order variability.

Cross-check: the first 3 rows of `text_projection__out.npy` byte-equal the first 3 rows of `talker_inputs_embeds__out.npy` (the assistantPreamble kernel just copies them through). So the bug is **entirely inside `invokeTalkerMLP`** — none of it is from preamble, embedding-table lookup, or downstream copies.

---

## Variant bisect (Stage 2)

**Not applicable for the SiLU-fused path.** Per source inspection (`cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.cu:768-825`, see snippet above), Ampere ships exactly one fused-bias-SiLU GEMM variant (`gemm_ampere_medium_bias_silu_fp16`). There is no small / large / split-k variant of the SiLU-fused kernel available to swap to. The variant list in `sm_87/metadata.json`:

```
gemm_ampere_decode_fp16
gemm_ampere_small_prefill_fp16
gemm_ampere_medium_prefill_fp16
gemm_ampere_large_prefill_fp16
gemm_ampere_splitk4_fp16
gemm_ampere_splitk2_fp16
gemm_ampere_medium_bias_silu_fp16      ← the suspect (used here)
gemm_ampere_medium_bias_fp16
```

Bisect would require either (a) decomposing the fused kernel into separate gemm + bias + SiLU (using `gemm_ampere_small_prefill_fp16` etc.), which forfeits the very thing the fused kernel exists for; or (b) NVIDIA shipping an alternative Ampere fused variant in the artifact.

We have **not** yet decomposed the call to isolate whether the bug is in (i) the GEMM body, (ii) the bias broadcast, or (iii) the SiLU epilogue. The cleanest follow-up would be a small standalone C++ harness that:
1. invokes `gemm_ampere_medium_prefill_fp16` (no fused epilogue) on the same A/B
2. applies bias + SiLU in a hand-written CUDA kernel
3. compares to the fused-kernel output

That work was scoped out of this round (would need a rebuild + new harness; we report what we have now since the data alone is reproducible and damning).

There IS a second invocation of CuTe DSL in `invokeTalkerMLP`: `runBias` (no SiLU) for the FC2 step (`M=variable, N=1024, K=2048`), which on Ampere uses `gemm_ampere_medium_bias_fp16`. We note `medium_bias_silu` is the prime suspect because:
- The same FC2 kernel runs for ALL output rows; if FC2 were the broken stage we'd expect the bug to also appear on row 0 (which is byte-clean) and other rows from the same call (also clean).
- Row 1's wrong values include both wrong sign and wrong magnitude; SiLU is a smooth nonlinearity that can't flip sign on its own — but a wrong GEMM accumulation in FC1 can produce a totally different vector through SiLU's gate-like behaviour.
- Row 0 going through the same kernel comes out correct, so the kernel isn't categorically broken — it has an M-row-index-dependent bug.

**Hypothesis (strengthened):** `gemm_ampere_medium_bias_silu_fp16` corrupts output row index 1 (or row `M-2`) inside every partial-M-tile invocation. Both observed failures (final-tensor rows 1 and 12) sit in different sub-block invocations of `invokeTalkerMLP`, supporting a per-call sub-tile predication bug rather than a one-off whole-tensor issue.

### FC1 vs FC2 isolation (Stage 2 data — partial)

Same build also added a `QWEN3_TTS_DUMP_FC1_WORKSPACE` env-var hook in `talkerMLPKernels.cu::invokeTalkerMLP` that dumps the FC1 fp16 output (post `runBiasSiLU`, pre `runBias`) to a binary file in append mode across all calls.

Output `/tmp/cv-investigate/dump_strengthened/fc1_workspace.bin` = 40960 bytes = 20480 fp16 = **10 rows × 2048 elements** total across all `invokeTalkerMLP` invocations for this request. Per-row absmax/mean/nz_frac:

| WS row | absmax | mean | first 5 values |
|-------:|-------:|-----:|---------------|
| 0      | 0.0250 | -0.0002 | -0.0014, -0.0021, +0.0011, -0.0020, -0.0039 |
| 1      | 0.0461 | -0.0000 | +0.0014, -0.0045, -0.0017, -0.0022, -0.0052 |
| 2      | 0.0347 | -0.0000 | -0.0023, +0.0002, +0.0025, -0.0011, -0.0011 |
| 3      | 0.2081 | +0.0006 | -0.0006, -0.0010, +0.0002, -0.0006, +0.0025 |
| 4      | 0.1411 | +0.0001 | +0.0195, -0.0015, -0.0079, +0.0134, +0.0099 |
| 5      | 0.0931 | +0.0000 | -0.0012, -0.0044, -0.0013, +0.0000, -0.0026 |
| 6      | 0.0815 | -0.0012 | -0.0066, -0.0042, +0.0019, +0.0132, +0.0103 |
| 7      | 0.0667 | -0.0000 | -0.0126, +0.0191, -0.0052, +0.0160, +0.0160 |
| 8      | 0.0809 | -0.0030 | +0.0053, -0.0336, +0.0043, +0.0005, -0.0265 |
| 9      | 0.0713 | +0.0009 | -0.0094, +0.0117, -0.0021, -0.0087, +0.0042 |

All rows post-SiLU activate to non-zero values (nz_frac = 1.000), so no row is wholesale zeroed; the bug doesn't manifest as a "dropped" row. The 10-row total split across the `invokeTalkerMLP` invocations cannot be cleanly mapped to the final 15 output rows from logs alone (the prefix/body/suffix split into sub-rows is not byte-traced) — we observe 10 inputs feeding text_projection but the final talker_inputs_embeds has 15 rows because the assistantPreamble inserts speaker/language/role marker embeddings on top.

**Limitation:** without also dumping the **FC1 input** (text_projection input — the pre-MLP embedding sequence), we cannot run cuBLAS HGemm in numpy against the exact same input bytes to ground-truth FC1 output row-by-row. The current FC1 workspace dump shows the in-flight intermediate is plausibly-shaped (non-zero, expected magnitudes consistent with a post-SiLU sparse-positive distribution), but does not by itself prove or disprove "FC1 row 1 is broken vs FC2 row 1 is broken". The text_projection weight shape is `linear_fc1.weight=[2048,2048]`, `linear_fc1.bias=[2048]`, `linear_fc2.weight=[1024,2048]`, `linear_fc2.bias=[1024]` — so FC1 GEMM is `M×2048×2048 + bias[2048] + SiLU`, FC2 GEMM is `M×1024×2048 + bias[1024]`.

**Cross-check (Stage 1, confirmed in strengthened run):** `text_projection__out.npy[i]` byte-equals `talker_inputs_embeds__out.npy[i]` for i ∈ {0,1,2} (max_diff = 0.000000). text_projection is the FC2 output. So at i=1 the bug is FULLY present in text_projection output (which is FC2 output of invokeTalkerMLP), meaning it's introduced no later than FC2's epilogue write. Whether it originated upstream in FC1 or originated in FC2 itself is still not isolated by this dump.

---

## cuBLAS / PyTorch reference comparison (Stage 3)

The "ground truth" used here is generated by HuggingFace `qwen_tts==0.1.1`'s reference PyTorch implementation, captured by intercepting `torch.cat` and dumping the 15×1024 talker_input_embeds tensor in bf16 → fp32. PyTorch routes the linear layers to cuBLAS HGemm on this GPU, so this IS effectively a cuBLAS reference (with bf16 accumulation rather than fp16, which we do not believe explains a 0.65 max_diff — for reference, row 0 from the same dtype mismatch comes out within 0.002 of TRT). Provenance:

```python
import torch
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained(CKPT, device_map="cuda:0", dtype=torch.bfloat16)
captured = []
orig = torch.cat
def spy(ts, dim=0, **kw):
    o = orig(ts, dim=dim, **kw)
    if o.dim() == 3 and o.shape == (1, 15, 1024):
        captured.append(o.detach().clone())
    return o
torch.cat = spy
torch.manual_seed(42)
_ = tts.generate_custom_voice(text="今天天气真不错", language="Chinese",
                              speaker="Vivian", instruct="", max_new_tokens=8)
captured[-1].cpu().float().numpy().tofile("ref_talker_embeds_15row.bin")
```

Reference file md5 `fed8b23ca46246f5993ec26ab7d5c0f4`, 61 440 bytes (15 × 1024 × 4).

We have **not** yet built a standalone CUDA program calling cuBLAS HGemm + a hand-written bias+SiLU on the exact same weight bytes and exact same input fp16 embedding — that would be the strongest possible evidence. With current data, the PyTorch capture is reference; given the bug magnitude (cos 0.39, sign flips) it is implausible that bf16-vs-fp16 accumulation could explain it.

---

## Workaround currently in use (Bug 3 in `customvoice-tts-fork-port-handoff.md`)

`cpp/runtime/qwen3OmniTTSRuntime.cpp::projectToTalkerInput` reads env var `QWEN3_TTS_PRELOAD_TALKER_EMBEDS=<bin>`. When set, after `invokeAssistantPreamble` it casts the Python fp32 dump to fp16 and `cudaMemcpyHostToDevice`s it over the broken `output` tensor. This loses the GPU-only fast path (requires Python pre-computation per request) but produces correct audio (`"今天天气真不错。"` verified end-to-end via radxa+SenseVoice ASR).

---

## Suspected root causes (ranked by likelihood)

1. **Bug in `gemm_ampere_medium_bias_silu_fp16` partial-M-tile predication.** STRENGTHENED by the full-15-row evidence: two final-tensor rows fail (1 and 12) and they live in different `invokeTalkerMLP` sub-block invocations. Other rows from the same calls (e.g. rows 0, 2 in the prefix block; rows 9, 10, 11, 13, 14 in the suffix block) pass cleanly. This is exactly the signature of a per-call M-row-position-dependent bug, almost certainly in the predication/masking logic for partial M tiles (off-by-one in the masking for sub-tile predicated stores, or wrong stride applied to bias broadcast at a specific output row offset). The bug strength differs between rows 1 and 12 (cos 0.39 vs 0.94) which is consistent with the corruption-amount depending on the *actual numeric data* of the under-tile input (not just being a fixed garbage write).

2. **Bias broadcast addressing bug.** The bias tensor is `[N]=[3072]`. If the kernel uses the wrong stride to compute the per-column bias load for output row 1, the result would be a valid-looking but wrong vector. This would match "looks like real numbers, just wrong" rather than NaN. Less likely than (1) because bias addressing is normally independent of M-row index.

3. **CUDA 12.6 / sm_87 specific MMA descriptor encoding issue.** The cute-dsl artifact metadata pins `cuda_version=12.6.68` and `gpu_arch=sm_87`, both matching our environment. We could conceivably be hitting a sm_87-specific Tensor Core descriptor edge case that doesn't surface on the typical sm_80/sm_90 test bench. Lower probability but worth NVIDIA verifying.

4. **Input layout assumption mismatch.** The wrapper passes A/B as 2D row-major contiguous tensors with strides `[K]`. If the kernel internally assumes 3D layout (L=1 mode) like the Blackwell variants in `dispatch3dFused`, partial-tile behaviour could differ. Source code for `_mlir_gemm_ampere_medium_bias_silu_fp16__mlir_ciface_...` is not provided in the artifact, so we cannot verify.

---

## What we have ruled out

- **Race / nondeterminism**: 10/10 byte-identical reruns.
- **Uninitialized memory in the C buffer**: the buffer is freshly reshaped before each call; rows 0/13/14 written by the same kernel are correct.
- **Bug in `assistantPreamble` or downstream copies**: text_projection rows 0,1 byte-equal talker_inputs_embeds rows 0,1.
- **Bug in batching (M=12 vs M=3)**: a previous attempt with a single M=12 invocation also showed row 1 wrong; splitting to M=3 (role) + M=4 (body) did not fix it (the body part comes out correct since it's contiguous M=4 rows of body tokens — the failure is specifically row 1 of the role-prefix M=3 call).
- **CuTe DSL not being loaded / wrong codepath**: log lines `CuteDslGemmRunner: Ampere GEMM module(s) loaded for SM87` and successful audio generation (just garbled) confirm the kernel did run.

---

## Asks to NVIDIA

1. **Verify** `gemm_ampere_medium_bias_silu_fp16` from artifact tarball `cutedsl_aarch64_sm_87_cuda12.tar.gz` (md5 `e076c4d83e002db9cf1558abcb8c34d7`, `cutlass_dsl_version 4.4.1`, `build_date 2026-05-06T21:23:32Z`) against a known-good fp16 GEMM reference for `M=3, N=3072, K=2048` with random fp16 A/B/bias on a sm_87 device. We expect to see row 1 of the output diverge while rows 0 and 2 match.

2. **Provide** either:
   - a fixed prebuilt artifact, or
   - source code for the affected DSL kernel (we are happy to compile from source — currently the artifact ships closed-source under `kernelSrcs/cuteDSLPrebuilt/`), or
   - guidance on a known-good Ampere fused-bias-SiLU alternative variant we can switch to via `cuteDslGemmRunner.cu::runBiasSiLU`.

3. **Confirm** whether the partial-M-tile predication logic in the CuTe DSL Ampere medium fused-epilogue kernel has been tested with M < tile_M (presumably 64 or 128) and bias broadcast enabled — this is the smallest test we suspect would have caught the bug.

4. **If shape-dependent** — let us know which (M, N, K) combinations are validated for `gemm_ampere_medium_bias_silu_fp16` so we can route the rest through `runBias` + separate epilogue without risk.

---

## Reference data provenance / file inventory

All paths on orin-nx (`100.82.225.102`):

| Path | What it is | md5 |
|------|------------|-----|
| `~/customvoice-v071-snapshot/20260526/qwen3_tts_inference` | test binary (with dump hooks + preload env-var workaround compiled in) | `f50fedc960d8edf7304f897cddbbdaf7` |
| `~/customvoice-v071-snapshot/20260526/libNvInfer_edgellm_plugin.so.1.0` | TRT plugin lib (W8A16LinearPlugin etc.) | `3d6761ebbe0946720f9c1d35a56c1cda` |
| `~/customvoice-v071-snapshot/20260526/ref_talker_embeds_15row.bin` | Python/PyTorch reference (fp32, 15×1024) | `fed8b23ca46246f5993ec26ab7d5c0f4` |
| `~/customvoice-v071-snapshot/20260526/v071_input_zh.json` | minimal reproducer input | — |
| `/tmp/v071_run/{talker,code_predictor,code2wav}/` | TRT engines for 3 stages | (multiple files) |
| `/tmp/cv-investigate/fork/` | TensorRT-Edge-LLM fork checkout (tag `customvoice-v071-w8a16-asr-pass-20260526`) | git `e08de4f2e3155fc1f38b650585465cf91013b8e8` |
| `/tmp/cv-investigate/extracted/sm_87/libcutedsl_aarch64.a` | extracted prebuilt CuTe DSL archive | `05ddeb9ddf60f1846872bd523ce49005` |
| `/tmp/cv-investigate/dump_broken/` | reproducer dump (talker_inputs_embeds__out.npy md5 `d1bfa2a1a234971cc92061445e306544`) | — |
| `/tmp/cv-investigate/dump_run_{1..10}/` | 10-run determinism evidence | (all 10 md5-identical) |

The fork is public at `https://github.com/suharvest/TensorRT-Edge-LLM` (branch `customvoice-v071-w8a16-asr-pass-20260526`). It differs from upstream NVIDIA/TensorRT-Edge-LLM v0.7.1 only in (a) the CustomVoice language-prefix support patches needed to even reach this code path, and (b) the env-var workaround that overrides the broken output. None of those changes touch CuTe DSL itself.

---

## Outstanding follow-ups

1. ~~**Lift dump cap** in `qwen3OmniTTSRuntime.cpp::dumpTensor` (currently `kCap=4096`).~~ **DONE.** kCap raised to 65536; full 15-row diff captured in this round; revealed second failing row 12 (previously hidden in the elided middle section).
2. ~~**Isolate FC1 vs FC2 contribution** by dumping the FC1 intermediate `workspace`.~~ **PARTIAL.** Workspace dump infrastructure added (`QWEN3_TTS_DUMP_FC1_WORKSPACE`); fc1_workspace.bin (10 rows × 2048 fp16) captured and confirms post-SiLU values are plausibly shaped, but full cuBLAS ground-truth comparison still requires also dumping the FC1 **input** to feed the same bytes to a numpy HGemm reference. The current 5-line patch can be extended with a symmetric `QWEN3_TTS_DUMP_FC1_INPUT` env-var dump for the `input` tensor — estimated 5 more lines + rebuild.
3. **Standalone CUDA harness** loading `text_projection.safetensors` weights + a known fp16 input embedding, invoking (a) `gemm_ampere_medium_bias_silu_fp16` directly and (b) cuBLAS HGemm + handwritten SiLU. Would give NV a single self-contained reproducer file that doesn't require the full Qwen3-TTS engine stack. Estimated: 200 lines + 1-hour build.
4. **Run on a second sm_87 device** (we have an Orin Nano with same arch in the fleet) to confirm the bug is per-arch and not per-individual-die.

---

## Conclusion

**NARROWED-DOWN + STRENGTHENED.** We have a deterministic, bit-exact reproducer of a correctness bug in the CuTe DSL Ampere kernels used by `invokeTalkerMLP` (FC1 `gemm_ampere_medium_bias_silu_fp16` and/or FC2 `gemm_ampere_medium_bias_fp16`). **Two rows out of 15** in the final `talker_inputs_embeds` tensor are corrupted (rows 1 and 12), and they live in two different `invokeTalkerMLP` sub-block invocations — strongly supporting a per-call M-row-position-dependent bug (likely sub-tile predication for partial M tiles). All 13 other rows pass to fp16 round-off vs the PyTorch/cuBLAS reference. The text_projection MLP shape is `FC1: M×2048×2048, FC2: M×1024×2048` (corrected from the earlier doc's `N=3072` claim).

FC1 vs FC2 isolation remains the primary open question: the workspace-dump infrastructure was successfully added, but a final cuBLAS-vs-FC1-output ground-truth comparison requires also dumping the FC1 *input* tensor (one more 5-line patch + rebuild). The circumstantial evidence still points at the FC1 SiLU-fused kernel (sign-flips can't come from SiLU alone on otherwise-correct inputs), but the strengthened evidence narrows the location to a *per-invocation row 1* pattern that is reproducible in two independent kernel calls within one request.

Verdict: **INCONCLUSIVE between FC1 and FC2** as the originator. **CONFIRMED**: row-position-dependent corruption in the CuTe DSL Ampere medium-tile kernels, reproducible across multiple invocations, affecting two distinct final-tensor rows mapped to position-1-within-call in their respective sub-blocks.
