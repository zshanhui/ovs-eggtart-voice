# MOSS-TTS-Nano C++ runtime spec（线性 KV）

**Status:** Design locked 2026-05-23 — Linear KV (NOT paged), follow qwen3_tts dual-buffer pattern
**Author:** codex recon + 主线程评审
**Decision rationale:** [docs/specs/moss-tts-nano-kv-adapter.md](./moss-tts-nano-kv-adapter.md)

## 决策摘要

经 codex 读真实 fork 代码确认：**`/Users/harvest/project/TensorRT-Edge-LLM` 没有 paged KV allocator**。`KVCacheManager` 是线性 `[maxBatchSize, 2, numKVHeads, maxSeqLen, headDim]`；`KvBlockArray` 只是 attention kernel 参数结构，没对应的 runtime block manager。

qwen3_tts 在 production 跑的是**线性 KV 双缓冲**（per-slot `kvA/kvB`），见 `cpp/runtime/qwen3OmniTTSRuntime.cpp:2283:allocateSlot` 和 `:2713:allocateBuffers`。

**结论**：MOSS-TTS-Nano 也走线性 KV，复用 qwen3_tts 模板。0.1B 模型 + N≤4 边缘并发场景下 paged 几乎无收益（线性 144MB/slot × 4 slot = 576MB，Orin NX 16GB 余量充足）。Paged 的 6.5× 显存节省只有在 N≥8 或大模型 serving 场景才有意义，我们目标场景不在此处。

**工作量**：5-6 PD（codex 原 10.5 PD 里 ~4-5 PD 是搭 paged block allocator，砍掉）。

---

## §1 KV 布局（线性，每 slot 独立）

每个 slot 持有：

**Global KV**（12 层 transformer）：
- 形状：`[num_layers=12, 2 (K+V), max_seq_len=1024, num_heads=12, head_dim=64]` FP16
- 单 slot 占用：12 × 2 × 1024 × 12 × 64 × 2 = **36 MB**
- max_seq_len=1024 对 12.5Hz × 80秒音频上限够用（实际平均 200-300 token）

**Local KV**（local_transformer_layers 层，待 codex 从 metadata 读）：
- 形状：`[local_layers, 2 (K+V), local_window=16, local_heads, local_head_dim]` FP16
- ring buffer，每 16 步 wraparound
- 单 slot 占用：估算 <1 MB

每 slot 总 KV ~ 37 MB；N=2 总 KV 74 MB；N=4 总 KV 148 MB。加 codec/engine plan 共享 ~700 MB engine 后总 < 1 GB，Orin NX 16GB 充分。

---

## §2 MOSS append kernel（线性 KV，不需要 block table）

签名：

```cpp
// cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.h
namespace trt_edgellm::kernel {

// 把 ONNX present_* 输出（delta 或 prefill full）追加到线性 KV buffer 的 past_len 位置
void appendMossLinearKV(
    half*          kvBuffer,       // [2, max_seq_len, H, D] FP16 — single layer K+V combined
    half const*    delta,          // [B, S, H, D] FP16  (S=1 for decode, S>1 for prefill)
    int32_t        batchSize,      // 必须 == 1（每 slot 独立 buffer）
    int32_t        pastLen,        // 当前已写入的 token 数
    int32_t        seqLen,         // 本次写入的 token 数
    int32_t        maxSeqLen,
    int32_t        numHeads,
    int32_t        headDim,
    int32_t        kvIdx,          // 0=K, 1=V
    cudaStream_t   stream);

} // namespace trt_edgellm::kernel
```

设备算法（`(s, h, d)` 三元组对应单元素）：
```
logical_pos = pastLen + s
dst_offset  = ((kvIdx * maxSeqLen + logical_pos) * H + h) * D + d
src_offset  = ((s * H + h) * D + d)
kvBuffer[dst_offset] = delta[src_offset]
```

**关键**：单 slot batchSize=1，无 block table 查找，纯连续 memcpy 模式。可直接 `cudaMemcpyAsync` 替代——只是 maxSeqLen 维度不连续，所以需要按 token 切片 copy 或写 kernel。

**Gather 不需要单独 kernel**：past input 直接 `setTensorAddress` 把 kvBuffer 指针 + offset 0 给 ONNX，shape `[1, pastLen, H, D]`——TRT 自动按 shape 读连续地址。

---

## §3 setTensorAddress wiring 流程

每次 MOSS global inference call 的 6 步：

**步骤 1**: `beginRequest()` 取 slot（复用 qwen3_tts EnginesRequestGuard 模式，`cpp/runtime/qwen3OmniTTSRuntime.cpp:4675:EnginesRequestGuard`）

**步骤 2**: 为每层绑 past 输入：
- shape `Dims4{1, pastLen, numHeads, headDim}` （MOSS dim order `[B, T, H, D]`，**注意与 qwen3_tts `[1, H, T, D]` 不同**）
- 地址 = slot.kvBuffer[layer].rawPtr + kvIdx_offset

**步骤 3**: 为每层绑 present 输出（写入 staging buffer）：
- prefill: shape `[1, S, H, D]`
- decode/local_cached: shape `[1, 1, H, D]`

**步骤 4**: `enqueueV3(stream)` 跑 engine

**步骤 5**: `appendMossLinearKV` 把 present 写回 slot.kvBuffer

**步骤 6**: 更新 `slot.pastLen += S`

---

## §4 Slot 生命周期（复用 qwen3_tts）

直接复用：
- `acquirePoolSlot` / `releasePoolSlot`（`cpp/runtime/qwen3OmniTTSRuntime.cpp:3042 / :3071`）
- `beginRequest()` / `endRequest()`
- `EnginesRequestGuard` RAII（`cpp/runtime/qwen3OmniTTSRuntime.cpp:4675`）
- per-slot TRT execution context（解决 [[tts_n2_phase_b_stability_landed]] 里 C6 那条 path）
- per-slot 预分配 scratch tensor（解决 Code2Wav 那套 race fixes 已经验证的 [[tts_n2_phase_b_stability_landed]] Option B 模式）

MOSS 特有的 per-slot state：
- `slot.kvBufferGlobal[num_layers]` — 12 个线性 KV buffer
- `slot.kvBufferLocal[local_layers]` — local ring buffer
- `slot.pastLen` int32
- `slot.localRingPos` int32
- `slot.presentScratch` — 给 setTensorAddress 用的 staging
- `slot.codecState` — codec streaming state（参考 [[tts_n2_phase_b_stability_landed]] Code2Wav slot 处理）

---

## §5 Local KV ring buffer

Local_cached_step 跑 16 步串行，每步生成 1 个 RVQ codebook token。第 17 步开始 wraparound 到位置 0。

布局：`[local_layers, 2, 16, H, D]` FP16

每步流程：
1. 从 ring buffer gather `[1, min(localPastLen, 16), H, D]` flat past（gather 顺序按逻辑时间，不是物理 ring 索引）→ 但**当 localPastLen < 16 时简化为直接连续读 [0:localPastLen]**
2. enqueue local engine
3. 把 `[1, 1, H, D]` local_present 写到 ring buffer 位置 `localPastLen % 16`
4. `localPastLen += 1`

当 localPastLen >= 16 时需要 logical reorder gather kernel，但 MOSS 16-step 串行是**每个 audio frame 内**的；每帧开始 localPastLen 重置为 0。所以**实际不会 wraparound**——简化为 fixed-size buffer，最多写 16 步。

---

## §6 文件分布

| 路径 | 内容 |
|---|---|
| `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.h` | append kernel host launcher 声明 |
| `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.cu` | append kernel + headDim=64 dispatch |
| `cpp/runtime/mossTtsNanoRuntime.h` | MossRuntime / MossSlot 定义（模仿 Qwen3TTSTalkerEngine） |
| `cpp/runtime/mossTtsNanoRuntime.cpp` | 6 步 inference loop + slot 管理 |
| `cpp/runtime/CMakeLists.txt` | 加入新文件构建 |
| `third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/moss_tts_worker.cpp` | 进程外壳，模仿 `qwen3_tts_worker.cpp` JSON-line 协议 |

---

## §7 工作量分解（线性 KV 版）

| 任务 | 估算 |
|------|------|
| `appendMossLinearKV` CUDA kernel + headDim=64 dispatch + unit test | 1.0 d |
| `mossTtsNanoRuntime` C++ runner（slot, engine binding, 6 步 loop） | 2.0 d |
| Local KV ring buffer（fixed-size 16 step） | 0.5 d |
| Codec streaming 接入（复用 [[tts_n2_phase_b_stability_landed]] Code2Wav 模式） | 1.0 d |
| `moss_tts_worker.cpp` 进程外壳 + JSON-line | 0.5 d |
| Numerical diff harness + N=1 byte-equivalent + N=2 smoke | 1.0 d |
| **总计** | **6.0 PD** |

---

## §8 风险

1. **ABI dim order mismatch**：MOSS past shape `[B, T, H, D]` vs qwen3_tts `[1, H, T, D]`。不能照抄 setTensorAddress 调用，必须按 MOSS 维度顺序绑定。错误会 silent misread。
2. **prefill batchSize=1 假设**：当前 spec 假设 B=1 单 slot，append kernel 拒绝 B>1。如未来要 batch prefill 需要新写 kernel。
3. **Local 16 步串行内每步都要重新 setTensorAddress + enqueue**：16 次 launch overhead 可能成为 host enqueue bound 瓶颈（如 trtexec 实测 local_fixed_sampled_frame 已经显示 host enqueue bound 风险）。CUDA Graph 化是优化候选。
4. **48kHz stereo 下游 chunk size**：现有 voice agent pipeline 是 24kHz mono，下游 WS chunk pacing 要按 4× 字节数调，否则缓冲水位错。
5. **Local KV layer count 从 metadata 读**：codex 提示导出脚本写了 `local_layers` 字段，runtime 要在 startup 读 meta.json 而不是硬编码。
6. **module-level env 陷阱**：写 Python backend wrapper 时遵守 [[trt_edge_llm_tts_env_staleness]] 规则——禁止 module scope `os.environ.get(...)`。

---

## 相关 memory / spec
- [[moss_tts_nano_port_recon]] — IO schema
- [[moss_tts_nano_trtexec_orin_nx]] — ONNX paged FP16 实测延迟
- [[tts_n2_phase_b_stability_landed]] — qwen3_tts N=2 slot 隔离已验证模式
- [[trt_edge_llm_fork_path]] — fork 位置
- [docs/specs/moss-tts-nano-kv-adapter.md](./moss-tts-nano-kv-adapter.md) — ONNX patch 设计
