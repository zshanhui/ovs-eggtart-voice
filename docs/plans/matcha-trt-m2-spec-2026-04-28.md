# Matcha TRT M2 实施 spec — 2026-04-28（codex 设计）

> 上游：`matcha-paraformer-trt-2026-04-27.md` §2/§6 + `matcha-paraformer-trt-m1-manifest-2026-04-28.md`
> 下游：M2 实施由 claude-rescue (GLM-5) 按 §6 prompt 模板分阶段执行

## §0 ODE Step Ablation（P0 决策点）

M2 第一件事不是 build engine，而是冻结 estimator 的 `ODE step count`。rkvoice split runtime 用 `MATCHA_ODE_STEPS` 控制：`n_steps = int(os.environ.get('MATCHA_ODE_STEPS', '1'))`，`dt = 1.0 / n_steps`，循环 `z = z + dt * v`（`rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:381-392`）。注释明确 1-step Euler 在 RK3576 FP16 更快更准（同文件 381-384），但 TRT + Orin NX 必须复验。

固定同一批 `tokens/noise_scale/length_scale`，ORT-CPU 原始 ONNX 输出作 `golden mel`。比较 `N=1/3/10`：
- `mel_relative_L2 = ||mel_trt - mel_ort|| / ||mel_ort||`
- audio RMS、non-silent ratio
- TTFT、RTF
- 主观听感

`noise` 固定 seed 或外部输入，否则 L2 不可比。

**验收**：选满足 `RTF ≤ 0.15` + `golden L2 < 5%` + 无爆音/静音/时长漂移的最小 N。
- N=1 过线 → estimator 固化 1-step
- 只有 N=3 过线 → 3-step
- N=10 仅作 fallback（除非 CUDA Graph 后仍满足 TTFT/RTF）

## §1 ONNX Surgery

M1 manifest 确认 acoustic ONNX `/opt/models/matcha-icefall-zh-en/model-steps-3.onnx`，输入 `x [N,L] int64`、`x_length [N] int64`、`noise_scale [1] f32`、`length_scale [1] f32`，输出 `mel [N,80,L] f32`（manifest 26-50）。含 1 个 `RandomNormalLike`（54、257-260）。

### RandomNormalLike externalization
- onnx-graphsurgeon 找唯一 `op == "RandomNormalLike"` node
- 原 node name 为空 → 命名 `matcha_noise_random_normal_like`
- 新增 graph input `noise_like: float32[1,80,T_mel_bucket]`，原 RNL output 的 consumers 接到 `noise_like`
- rkvoice probe-first 流程：probe 原模型 Range/RNL shape → onnxsim 固定 shape → 替换问题 op（`fix_matcha_rknn.py:13-25, 64-115`）
- RK 版替成 seed=42 constant（261-297）；TRT 版改 external input，便于 golden 对齐 + 多请求随机性

### 黑名单 op
M1 acoustic 实际命中 `Range`、`Slice`、`Where`，无 `Scan/Loop/If`（manifest 34, 59-60）：
- `Range` → probe constant tensor 替换（参考 `fix_range_nodes()` `fix_matcha_rknn.py:136-159`）
- 动态 `Slice` index/ends → 识别非 initializer/Constant 的 index 输入，probe 转 initializer（参考 `fix_dynamic_slice_ends()` 189-258）
- `Where` 条件来自 shape mask → constant-fold；数据相关 → 保留并 `trtexec --verbose` 验证 parser 支持

### int64 → int32 cast
`x` 与 `x_length` 在 manifest 为 int64（manifest 40-41），TRT 覆盖有限。surgery 时改 graph input dtype 为 int32，需要 int64 的 shape 子图前插局部 Cast。runtime tokens 也直接生成 int32。

### 输出命名
- `matcha_encoder_s{32|64|128}_trt.onnx`
- `matcha_estimator_n{N}_m{mel_bucket}_trt.onnx`
- `vocos_m{72|256|600}_trt.onnx`

## §2 3 TRT Engine Build

精度策略：encoder/estimator 用 BF16（防 QK^T 溢出），attention/Softmax/LayerNorm/InstanceNorm 强制 FP32 obey；vocoder 用 FP16，保留 final mag/x/y FP32。

```bash
# Encoder（BF16 + FP32 attention obey）
trtexec --onnx=matcha_encoder_s64_trt.onnx \
  --saveEngine=matcha_encoder_s64_bf16.engine \
  --bf16 --stronglyTyped --precisionConstraints=obey \
  --minShapes=x:1x1,x_length:1,noise_scale:1,length_scale:1 \
  --optShapes=x:1x64,x_length:1,noise_scale:1,length_scale:1 \
  --maxShapes=x:1x64,x_length:1,noise_scale:1,length_scale:1 \
  --memPoolSize=workspace:2048 --verbose 2>&1 | tee encoder_build.log

# Estimator（BF16，N 由 §0 ablation 冻结，示例 N=1）
trtexec --onnx=matcha_estimator_n1_m600_trt.onnx \
  --saveEngine=matcha_estimator_n1_m600_bf16.engine \
  --bf16 --stronglyTyped --precisionConstraints=obey \
  --minShapes=z:1x80x72,mu:1x80x72,mask:1x1x72,noise_like:1x80x72 \
  --optShapes=z:1x80x256,mu:1x80x256,mask:1x1x256,noise_like:1x80x256 \
  --maxShapes=z:1x80x600,mu:1x80x600,mask:1x1x600,noise_like:1x80x600 \
  --memPoolSize=workspace:3072 --verbose 2>&1 | tee estimator_build.log

# Vocos（FP16）
trtexec --onnx=vocos_m600_trt.onnx \
  --saveEngine=vocos_m600_fp16.engine \
  --fp16 --stronglyTyped \
  --minShapes=mels:1x80x72 \
  --optShapes=mels:1x80x256 \
  --maxShapes=mels:1x80x600 \
  --memPoolSize=workspace:2048 --verbose 2>&1 | tee vocos_build.log
```

Vocos 输入 `mels [batch,80,time]`，输出 `mag/x/y`（manifest 72-84），FFT bin 约 513（manifest 89-90）。CPU ISTFT 参数：`N_FFT=1024`、`HOP_LENGTH=256`（rkvoice matcha.py:31-34）。

## §3 MatchaTRTBackend Skeleton

```python
class MatchaTRTBackend(TTSBackend):
    """
    Dataflow:
    text -> _text_to_tokens (espeak/lexicon) -> int32 tokens
    -> _select_bucket (mel_frames ≈ 11.9*num_tokens+51, matcha.py:423-428)
    -> _run_encoder (TRT) -> mu [1,80,T_mel], mask [1,1,T_mel], z noise
    -> _run_estimator (TRT, n_steps frozen) -> mel [1,80,T_mel]
    -> _maybe_smooth_mel (anomaly hook, ref matcha.py:546-598)
    -> _run_vocos (TRT, mel padded to bucket) -> mag,x,y
    -> _cpu_istft (torch.istft) -> float32 audio -> crop to mel_frames*HOP_LENGTH
    """
    name: str = "matcha_trt"
    capabilities: set = {"basic_tts"}
    sample_rate: int = 16000

    def __init__(self, model_dir: Path, engine_dir: Path, max_tokens: int = 128): ...
    def preload(self) -> None:
        """load lexicon, load 3 TRT engines via tensorrt.Runtime, warmup each bucket"""
    def synthesize(self, text: str, speaker_id: int = 0, speed: float = 1.0, **kwargs) -> tuple[bytes, dict]: ...
    def _text_to_tokens(self, text: str) -> np.ndarray: ...
    def _select_bucket(self, num_tokens: int, length_scale: float) -> tuple[int, int]:
        """Returns (text_bucket, mel_bucket). mel_frames ≈ 11.9*num_tokens+51 * length_scale * 1.2"""
    def _run_encoder(self, tokens, x_length, noise_scale, length_scale) -> tuple: ...
    def _run_estimator(self, mu, mask, noise, n_steps: int) -> np.ndarray: ...
    def _maybe_smooth_mel(self, mel: np.ndarray) -> np.ndarray:
        """Window-5 local median; ratio <0.5 or >1.8 frames blended with neighbors. Log n_anomaly."""
    def _run_vocos(self, mel_padded: np.ndarray) -> tuple: ...
    def _cpu_istft(self, mag, x, y, mel_frames: int) -> np.ndarray:
        """torch.istft N_FFT=1024 HOP_LENGTH=256; crop to mel_frames*HOP_LENGTH samples"""
```

## §4 Golden Round-trip Test

文件：`tests/tts/test_matcha_trt_golden.py`

测试文本集（≥20 条）：
- 中文短：`今天天气不错。`、`重庆银行行长正在开会。`（多音字）
- 中文长：`这个项目需要重构和回归测试，请先读文档再动手。`、`二零二六年四月二十八日，温度是二十三点五度。`
- 数字/标点：`价格是999元，折扣是8.5折。`
- 英文：`Hello, this is a TensorRT test.`、`The quick brown fox jumps over the lazy dog.`
- 中英混合：`请打开 Wi-Fi，然后运行 Python script。`

Reference 生成：固定 `noise_like = np.random.default_rng(42).standard_normal(...).astype(np.float32)`，externalized ONNX + ORT-CPU 跑完整 pipeline，保存 `mel_ref.npy`/`mag_ref.npy`/`wav_ref.wav`/`metadata_ref.json`。

验收：
- `mel_relative_L2 < 0.05`
- audio duration error < 2%，RMS ratio ∈ [0.8, 1.25]，无全静音
- `RTF ≤ 0.15`，`TTFT ≤ 200ms`（P50, 20 samples）

## §5 Risk Points

RK NPU 陷阱在 TRT path 不会出现：
- RKNN 不支持 `Ceil` CPU op → TRT 直接解析
- RKNN 固定 output shape 导致 ODE FP16 累积误差 → TRT dynamic profile + BF16 layer precision 解决
- RKNN split 模式 CPU FP32 ODE loop → TRT 可 BF16 + CUDA Graph
- RKNN/RKLLM domain isolation 冲突 → TRT 无

TRT 特有坑：
- multi-step ODE 每步 `enqueueV3` dispatch ~12ms（Paraformer 经验），N=10 必须 CUDA Graph
- same-shape Myelin crash → engine pool N=2（参考 Qwen3 CP pool）
- Orin NX 8GB 同时跑 Qwen3 ASR + Paraformer + Matcha，workspace 竞争需启动时显存校验
- FP16 attention/norm 溢出 → obey-precision FP32 必选

M2 out of scope：
- streaming `/tts/stream`（M3）
- INT8 量化
- 抽 `TRTAutoregressiveEngine` base class
- Kokoro TRT
- Nano 8GB 多语言内存救援

## §6 GLM-5 Dispatch Prompt Templates

统一 anti-recursion header：
```
ANTI-RECURSION GUARD: You are the worker agent. Do NOT delegate, do NOT spawn subagents, do NOT write plan docs. Execute exactly this bounded task in <=25 bash steps. Stop immediately before any destructive op and report.
FORBIDDEN OPS: rm -rf (outside build/), docker-compose down/up, git reset --hard, mv/rename docker-compose*.yml or .env, sudo rm any path, container recreate.
```

### Task A: ODE step ablation
- 目标：固定 noise seed，N=1/3/10 mel L2 + RTF 决策表
- 可写范围：`/tmp/matcha_ablation/`
- 禁区：不修改模型文件，不修改容器配置
- EVIDENCE：ONNX md5、固定 noise.npy md5、raw benchmark stdout、N×{L2,RTF,TTFT,verdict} 表、WAV md5

### Task B: ONNX surgery
- 目标：externalize RNL、处理 Range/Slice/Where、int64→int32 cast，输出 3 个 surgery ONNX
- 可写范围：`/opt/models/matcha-icefall-zh-en/surgery/`
- 禁区：不修改原始 `model-steps-3.onnx`，不修改其他目录
- EVIDENCE：source/target ONNX md5、raw `polygraphy inspect` 前后对比、ORT parity L2 < 5%

### Task C: TRT engine build
- 目标：`trtexec` build 3 engines（profile 同 §2）
- 可写范围：`/opt/engines/matcha/`
- 禁区：不跑 bare cmake/make，只 `trtexec`
- EVIDENCE：engine md5、完整 `trtexec --verbose` log（含 TRT/CUDA 版本）、profile 表、peak memory

### Task D: Backend integration（offline /tts only）
- 目标：实现 `MatchaTRTBackend`，接入 factory，验证 `/tts`
- 可写范围：`app/backends/`、`app/tts_service.py`（factory branch）
- 禁区：不改 docker-compose，不改 `/asr` 路由，不跑 `docker compose down/up`（只 restart）
- EVIDENCE：touched file 列表、raw service 启动日志、`/tts/capabilities` JSON、生成 WAV md5、timing metadata JSON

### Task E: Golden pytest
- 目标：新建 `tests/tts/test_matcha_trt_golden.py`，20 样本 ORT-CPU reference vs TRT
- 可写范围：`tests/tts/`
- 禁区：不修改现有 ASR/Paraformer 测试，不改 conftest.py
- EVIDENCE：test file md5、raw `uv run pytest -v` 完整输出、失败样本 dump（mel L2/RMS/RTF）、汇总表
