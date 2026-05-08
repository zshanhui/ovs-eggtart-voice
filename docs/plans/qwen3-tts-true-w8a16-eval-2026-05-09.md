# Qwen3-TTS Talker true W8A16 feasibility

Date: 2026-05-09

Scope: Qwen3-TTS Talker only. Qwen3 ASR/TTS dual residency and streaming latency remain the gating constraints.

## Baseline

- Existing `talker_decode_bf16.engine`: 876 MiB.
- Existing TensorRT INT8/PTQ engines: about 445 MiB, but quality failed ASR round-trip and cannot be used as product candidates.
- Decode ONNX has 197 constant `MatMul` weights:
  - total weights: 443,547,648 elements
  - FP16/BF16 weight bytes: about 887 MB
  - INT8 weight bytes: about 444 MB
  - per-output FP16 scales: about 0.7 MB
  - group128 FP16 scales: about 6.9 MB

Conclusion: true weight-only W8A16 has real memory upside. The hard part is the execution path, not the quantized weight files.

## Option 1: build a targeted dense W8A16 plugin

Feasibility: highest.

Why:

- Talker streaming decode is dominated by `M=1` Linear calls.
- A minimal GEMV path can target `A16 activation x INT8 weight -> A16/FP32 accumulate/output` first.
- This avoids TensorRT INT8 activation calibration, which is what broke quality in the current PTQ engines.
- Per-output or groupwise scales are small enough that memory benefit remains close to the 445 MB INT8/PTQ engine size.

Risks:

- A naive dequant-on-read GEMV can be slower than BF16 cuBLAS, so it must be microbenchmarked on Orin SM87 before replacing all 197 Linear layers.
- Full graph integration still requires ONNX/plugin replacement for constant-weight MatMuls.
- Prefill/batched `M>1` support can be deferred only if the runtime keeps iterative prefill for the prompt path.

Recommended first proof:

1. Quantize a few representative Talker weights per-output INT8.
2. Build one standalone CUDA GEMV microbench for shapes `(1024,1024)`, `(1024,2048)`, `(2048,1024)`, `(1024,3072)`, `(3072,1024)`.
3. Compare latency against BF16/FP16 baseline on Orin Nano.
4. Only then wire a TensorRT plugin and replace a small subset of MatMuls for correctness/quality tracing.

## Option 2: reuse official / existing EdgeLLM Marlin paths

Feasibility: partial reuse only.

What can be reused:

- `Int4GroupwiseGemmPlugin` already has a dense plugin shape, constant quantized weight input, scale input, and small-M dispatch. This is useful as a plugin skeleton.
- The existing INT4 kernels show the expected memory layout and Orin-compatible CUDA style.

What does not directly transfer:

- `Int4GroupwiseGemmPlugin` is W4A16, not W8A16. It accepts HALF activation, packed INT4 weights shaped `[N/2, K]`, HALF scales, and emits HALF output.
- MoE Marlin is wired for expert routing and W4A16 AWQ: `aType=kFloat16`, `bType=kU4`, `cType=kFloat16`, `sType=kFloat16`.
- The repo's compiled Marlin instantiations are W4-oriented. Marlin itself is not Thor-only for this path, but the current integration is not a dense Talker W8 Linear.
- Porting Marlin into dense W8A16 would require weight repacking, new kernel instantiations, a dense wrapper/plugin, and TensorRT graph replacement. That is more work than a minimal GEMV proof.

Decision: do not start with full Marlin migration. Reuse the dense plugin scaffolding and small-M scheduling ideas, but implement/prototype W8 GEMV directly.

## Current Ranking

1. Targeted dense W8A16 GEMV plugin proof on Orin Nano.
2. Reuse `Int4GroupwiseGemmPlugin` structure where it reduces boilerplate.
3. Full Marlin dense W8 migration only if the targeted plugin proves too slow and an existing W8 Marlin dense kernel can be introduced cleanly.

