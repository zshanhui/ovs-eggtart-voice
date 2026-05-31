# MOSS-TTS-Nano KV Cache Adapter Spec

**Status:** Draft (2026-05-23)
**Decision:** Option B — patch ONNX export to paged FP16
**Author:** codex recon + main-thread review

## Context

Porting OpenMOSS/MOSS-TTS-Nano 0.1B (12L / hidden=768 / 12 head / head_dim=64 / n_vq=16 RVQ) to our TensorRT-Edge-LLM fork. The official ONNX export uses a KV cache interface incompatible with our paged FP16 KV manager (used in production by qwen3_tts / qwen3_trt).

### MOSS KV interface (confirmed)
- Flat per-layer `(key, value)` tensor list, each step returns full `present_*` and consumes prior `past_*`
- Per-layer shape `[B, T, 12, 64]`
- dtype FP32
- `past_valid_lengths` int32 tensor separate from KV tensors
- Local transformer has its own independent KV (16-step serial window)
- Refs: `export_moss_tts_browser_onnx.py:2528-2535`, `:3140`, `:3320`, `ort_cpu_runtime.py:3046-3078`

### Our trt_edge_llm interface
- Paged KV + block table managed by C++ cacheManager
- FP16
- TRT engine owns paged GPU pool; cross-request slot lifecycle proven by N=2 stability work (`tts_n2_phase_b_stability_landed`)

## Option A — thin adapter around flat ONNX

Keep MOSS ONNX as-is, wrap with:
1. `gather_kv_from_paged_pool` — per-layer gather from paged pool → contiguous flat `[B, T, 12, 64]`
2. `fp16_to_fp32_cast`
3. `onnxruntime_run` (or TRT engine if we still build engines from unmodified ONNX)
4. `fp32_to_fp16_cast` on `present_*` output
5. `scatter_kv_to_paged_pool`

**Per-step memory traffic (200-token sentence, B=1, 12 layers):**
- Per layer K or V: 200 × 12 × 64 = 153,600 elements
- 12 layers × 2 (K+V) = 3,686,400 elements
- ~51.6 MB adapter-side traffic per step, ~66 MB including ONNX present-output allocation pressure

Grows linearly with sequence length, fires every generated token (MOSS returns full-sequence present each step).

## Option B — patch ONNX export to paged FP16

Modify `export_moss_tts_browser_onnx.py`:
- `:2528-2535` — flat per-layer KV declaration: dtype FP32 → FP16, shape `[B, T, 12, 64]` → paged-block layout matching cacheManager convention
- `:3140` — add block-table input node
- `:3320` — replace full-sequence present outputs with block-write / append-only outputs
- Keep `past_valid_lengths` as separate int32
- Local KV (16-step window): stays non-paged, just per-slot ring buffer FP16

## Trade-off analysis

| Dimension | Option A | Option B |
|---|---|---|
| Per-step overhead | 51-66 MB gather + 2× cast + scatter | append + paged read via block table |
| Bandwidth pressure (Orin NX) | likely > model compute itself | minimal |
| N=2 slot lifecycle | needs new per-slot FP32 staging — risks Code2Wav-class slot contention bugs | reuses proven cacheManager + per-slot tensor pool |
| Upstream MOSS upgrades | painless (no fork) | rebase risk on every upstream change |
| Code touched | trt_edge_llm wrapper only | ONNX export script + IR + TRT path |
| Local KV strategy | flat FP32, per-slot | per-slot ring buffer FP16 (consistency) |

## Decision: Option B

**Rationale:** Patching ONNX export is more invasive upfront, but eliminates per-step full-sequence gather/cast/scatter overhead and composes cleanly with the already-stable cacheManager N=2 slot lifecycle. Avoiding a new slot-contention surface (the Code2Wav lesson from `tts_n2_phase_b_stability_landed`) is worth the rebase risk.

## Effort

- **Option A:** 8-12 person-days (wrapper kernels, FP32 staging, slot wiring, tests, qwen3_tts coexistence)
- **Option B:** 10-15 person-days (export patch 3-4d, IR validation 2d, TRT engine rebuild 1-2d, integration 2-3d, N=2 validation 1-2d, rebase buffer 1-2d) ← **chosen**

## Implementation Spec (codex 2026-05-23 refinement)

### Paged KV layout (assumptions to verify against our cacheManager)

- `tokens_per_block = 16`（短流式 utterance 合适，可调）
- Per-layer paged pool: `[num_blocks, 16, 12, 64] float16`
- `block_table_i32`: `[B, max_blocks_per_seq]`
- `past_valid_lengths_i32`: `[B]` (保留，给位置 mask + ring offset)

### 6 个 hunk（带行号 + diff，应用在 `onnx/export_moss_tts_browser_onnx.py`）

**Hunk 1** `:2312-2318` flatten helper FP32 → FP16，函数改名 `_flatten_past_key_values` → `_flatten_paged_kv_deltas`，输出 `key_delta/value_delta` 而非全 sequence。

**Hunk 2** `:2528-2535` attention KV gather/present：
- 删 `torch.cat([past_key, key], dim=1)` 全量拼接
- 改成 `gather_paged_kv(paged_kv_cache, block_table, past_valid_lengths, key, value, layer_index)`
- present 只输出当前 step `key[:, -1:] / value[:, -1:]` FP16

**Hunk 3** `:3106-3120` prefill forward 签名加 `block_table_i32`, `paged_kv_cache_f16` 两个输入，调 `_run_transformer_paged`

**Hunk 4** `:3133-3158` decode_step forward 同 prefill 签名调整

**Hunk 5** `:3313-3372` local cached_step 走 ring：`_run_decode_step_ring(local_kv_ring_f16, ring_pos_i32)`

**Hunk 6** `:3476-3492` 零初始化：`[B, num_layers, 2, 16, num_heads, head_dim] FP16`

### **关键设计：ONNX 只产 delta，paged write 留给 C++**

ONNX `ScatterND` 不能表达 in-place paged append。所以 ONNX graph 不做 paged write，每层输出 `k_delta/v_delta [B,1,H,D] FP16`，C++ runtime 用：
```
offset = past_len % 16
block_idx = block_table[b, past_len // 16]
```
+ 自写 append kernel + `IExecutionContext::setTensorAddress` 把 paged pool 绑给 ONNX 的 gather 输入。
这就把"paged op 进 ONNX graph 失败"的风险绕过去了。

### Local transformer KV

独立非 paged，FP16 ring `[B, num_local_layers, 2, 16, num_heads, head_dim]` + `local_ring_pos_i32 [B]`。匹配 16-step 串行窗口，不走 cacheManager。

### FP16 转换策略

**用 export-time `model.half()` + FP16 dummy tensors**，不要用 `onnxconverter_common.float16` 后处理（exporter 直接控 `_flatten` dtype 更安全）。`moss_tts_global_shared.data` 体积砍半，`merge_shared_external_data` 重跑即可。

### 验证三阶段

1. **Stage 1 — ORT schema check**：`tools/check_moss_paged_onnx.py` 跑 `onnx.checker` + 断言 `block_table:int32` 输入 + KV tensor FP16。**不验语义**（ORT 不能跑 paged）。
2. **Stage 2 — TRT FP16 build**：`trtexec --fp16 --minShapes/--optShapes/--maxShapes` 含 `block_table_i32:1x128/1x128/2x128`。
3. **Stage 3 — Numerical diff**：`tools/diff_moss_paged_vs_flat.py` 与未改的 FP32 flat baseline 比对，max diff < 1e-2，max-steps 256。

### 工作量分解（PD = person-day）

| Phase | PD |
|---|---|
| Design + 与 cacheManager 接口对齐 | 2.0 |
| Export script 实施（hunks 1-6 + ring path）| 4.0 |
| TRT C++ append kernel + setTensorAddress wiring | 3.0 |
| Stage 1-3 验证 + N=2 slot stress | 3.0 |
| Rebase + checker buffer | 1.5 |
| **总计** | **13.5 PD** |

### Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| onnx.checker 拒绝 in-place KV graph | HIGH | ONNX 只产 delta，paged write 全在 TRT C++ |
| ORT 无法验 paged runtime 语义 | MED | Stage 1 只验 I/O schema + delta numerics；语义靠 Stage 3 TRT diff |
| TRT builder 对 dynamic block_table 失败 | HIGH | 不在 ONNX 里用 ScatterND，全靠 C++ append |
| FP16 quantization drift > 1e-2 | MED | 与 FP32 baseline 比 logits + 生成 token IDs，超阈值 fail |
| Local ring off-by-one 错位声道 | MED | 单独单测 16-step wraparound + 每通道 parity |

## Open items (blocked until T7 trtexec results)

- Confirm Option B end-to-end TTFA stays below ≤ 2× qwen3_tts baseline (~740 ms target) on orin-nx
- Decide whether local KV ring buffer is FP16 or stays FP32 (depends on whether local_cached_step trtexec FP16 path passes byte-equiv tests)
- Engine plan upgrade strategy: ship pre-built engines vs build-on-device (see `kokoro_trt_hot_reload_verified` for MIN_T rebuild precedent)

## Related memory
- [[moss_tts_nano_port_recon]] — source-code recon (IO schema, codec ONNX location)
- [[tts_n2_phase_b_stability_landed]] — N=2 cacheManager + per-slot tensor pool that Option B reuses
- [[tts_worker_phase1_landed]] — TRT-Edge-LLM fork location + iterate loop
- [[trt_edge_llm_tts_env_staleness]] — module-level env staleness rule for new backends
