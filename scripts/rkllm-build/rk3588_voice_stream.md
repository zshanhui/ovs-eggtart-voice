# RK3588 Voice Stream — Full Stack Setup Guide

Target: **Radxa ROCK 5T** (RK3588, 16 GB RAM, 6 TOPS NPU, 3 NPU cores).
Also works on other RK3588 boards (Orange Pi 5, Firefly ITX-3588J) with
≥8 GB RAM.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  RK3588 Device                                       │
│                                                      │
│  ┌────────────────────┐  ┌─────────────────────────┐ │
│  │ speech (:8621)     │  │ llm (:8001)             │ │
│  │                    │  │                         │ │
│  │ ASR: Qwen3-ASR     │  │ LLM: Qwen3 via RKLLM    │ │
│  │   encoder: RKNN    │  │   librkllmrt.so         │ │
│  │   decoder: RKLLM   │  │   NPU_CORE_AUTO         │ │
│  │ TTS: Matcha+Vocos  │  │                         │ │
│  │   NPU + CPU hybrid │  │ model: 0.6B / 1.7B     │ │
│  │ NPU_CORE_AUTO      │  │                         │ │
│  └────────────────────┘  └─────────────────────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │ ovs-agent (apps/rk3588-chat)                    │ │
│  │   mic → VAD → SLV(:8621) → LLM(:8001) → TTS →  │ │
│  │   speaker                                        │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## 1. Hardware & OS

### Required

| Item | Specification |
|------|--------------|
| Board | Radxa ROCK 5T, Orange Pi 5 Plus, or any RK3588 SBC |
| RAM | 16 GB recommended (8 GB minimum for 0.6B model) |
| Storage | ≥ 32 GB free (model files + Docker images) |
| OS | Debian 12 (Bookworm) or Ubuntu 22.04+, aarch64 |
| Kernel | Linux 5.10+ with Rockchip NPU drivers |
| Docker | 24+ with `--privileged` support |
| Audio | USB microphone + speaker, or headset |

### Verify NPU is visible

```bash
ls -la /dev/rknpu*         # should show NPU device nodes
dmesg | grep -i rknpu      # kernel loaded the NPU driver
```

### Verify Docker

```bash
docker --version            # ≥ 24.0
docker run --rm hello-world # must work without sudo
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/suharvest/openvoicestream.git
cd openvoicestream
git checkout eggtart
```

---

## 3. Model Files

### Option A — Auto-download (recommended)

The speech container auto-downloads ASR and TTS models from HuggingFace
(`harvestsu/seeed-local-voice-rk-artifacts`) at first startup. Set the
env var to enable it:

```bash
export RK_ARTIFACT_AUTO_DOWNLOAD=1
```

The LLM chat model must be placed manually (see below).

### Option B — Convert on a CUDA PC

For a custom model, follow the conversion workflow in `scripts/`:

```bash
# On a CUDA GPU instance (RTX 3060+, ≥ 12 GB VRAM)
scp scripts/setup_rkllm.sh scripts/convert_model.sh scripts/build_rkllm_model.py user@gpu-host:/tmp/
ssh user@gpu-host

cd /tmp
chmod +x setup_rkllm.sh convert_model.sh
./setup_rkllm.sh

# Convert 0.6B (fastest, fits 8 GB board)
TARGET="RK3588" MODEL_ID="Qwen/Qwen3-0.6B" QUANT="FP16" ./convert_model.sh

# Or 1.7B (better quality, needs 16 GB board)
TARGET="RK3588" MODEL_ID="Qwen/Qwen3-1.7B" QUANT="W8A8" ./convert_model.sh
```

Then copy the `.rkllm` file back to your local machine and upload to the device.

### LLM Model Placement

```bash
# On the RK3588 device
sudo mkdir -p /opt/models/rkllm

# Copy from your local machine
scp models/qwen3-0.6b_FP16_RK3588.rkllm user@rk-device:/opt/models/rkllm/
```

If using a different model, set the path env var before starting:

```bash
export RKLLM_MODEL_PATH=/opt/models/rkllm/qwen3-1.7b_W8A8_RK3588.rkllm
```

---

## 4. Pull Docker Image

```bash
docker pull sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-v1.4-closedloop
```

The image is ~950 MB, shared by both the `speech` and `llm` services.
It contains the RKNN/RKLLM userspace runtimes (`librknnrt.so`,
`librkllmrt.so`), Python 3.12, sherpa-onnx, rknn-toolkit-lite2, and the
application code.

---

## 5. Start the Stack

```bash
cd openvoicestream

# Pull the image if not already done
docker compose -f deploy/docker-compose.radxa.yml pull

# Start both services
RKLLM_ENGINE=real \
RKLLM_MODEL_PATH=/opt/models/rkllm/qwen3-0.6b_FP16_RK3588.rkllm \
docker compose -f deploy/docker-compose.radxa.yml up -d
```

Or set defaults in a `.env` file:

```bash
# .env (in repo root)
OVS_PROFILE_DEFAULT=rk3588-multilang
RKLLM_ENGINE=real
RKLLM_MODEL_PATH=/opt/models/rkllm/qwen3-0.6b_FP16_RK3588.rkllm
RKLLM_SYSTEM_PROMPT="你是一个运行在瑞芯微芯片上的语音助手。请用简洁自然的中文回答，控制在三句话以内。"
```

Then just:

```bash
docker compose -f deploy/docker-compose.radxa.yml up -d
```

---

## 6. Verify the Services

### Speech service (ASR + TTS)

```bash
curl http://localhost:8621/health
# → {"status": "ok"}

curl http://localhost:8621/v1/models
# → list of loaded ASR/TTS backends
```

### LLM service

```bash
curl http://localhost:8001/health
# → {"status": "ok"}

curl http://localhost:8001/v1/models
# → {"data": [{"id": "qwen3-0.6b", ...}]}

# Test a chat completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 64,
    "stream": false
  }'
```

### Check logs

```bash
docker logs openvoicestream      # speech container
docker logs openvoicestream-llm  # LLM container
```

First startup takes 1–2 minutes — the speech container downloads
ASR/TTS model artifacts from HuggingFace.

---

## 7. Install and Run the Agent

### Install

```bash
cd openvoicestream/agent
pip install -e .
```

### Verify config loads

```bash
python3 -c "
from openvoicestream_agent.config import load_config
cfg = load_config('apps/rk3588-chat')
print(f'llm_model={cfg.llm_model}, backend={cfg.llm_backend}')
"
# → llm_model=qwen3-0.6b, backend=openai_compat
```

### Run the voice assistant

```bash
ovs-agent run apps/rk3588-chat
```

The agent opens the microphone, streams audio to the speech service for
ASR, sends the transcript to the LLM, and plays the TTS response through
the speaker. Speak to start a conversation. Say "停" or "stop" to abort
TTS playback mid-stream.

### Audio device selection

If you have multiple audio devices, list them:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Then set in the config or env:

```bash
export OVS_AUDIO_INPUT_DEVICE=2   # your mic device index
export OVS_AUDIO_OUTPUT_DEVICE=3  # your speaker device index
ovs-agent run apps/rk3588-chat
```

---

## 8. Using a Different LLM Model

### Switching between 0.6B and 1.7B

```bash
# Stop the stack
docker compose -f deploy/docker-compose.radxa.yml down

# Set the new model path
export RKLLM_MODEL_PATH=/opt/models/rkllm/qwen3-1.7b_W8A8_RK3588.rkllm

# Restart
docker compose -f deploy/docker-compose.radxa.yml up -d
```

The agent picks up the model name from `llm_model` in its config — if the
LLM server reports a different model ID, update it:

```bash
export OVS_LLM_MODEL=qwen3-1.7b
ovs-agent run apps/rk3588-chat
```

### Dummy engine (test without NPU)

Set `RKLLM_ENGINE=dummy` to test the pipeline without loading a real
model on the NPU. The dummy engine returns canned responses. Useful for
validating the full audio chain (mic → ASR → LLM → TTS → speaker) before
wiring the real model.

---

## 9. Resource Tuning

### NPU core allocation

By default, the speech container uses `NPU_CORE_AUTO` (all 3 cores).
The LLM container uses the same — the kernel schedules NPU work across
both containers. For heavy concurrent load, you can pin cores:

```yaml
# docker-compose.radxa.yml overrides
speech:
  environment:
    - ASR_NPU_CORE_MASK=NPU_CORE_0_1   # first 2 cores for ASR+TTS
llm:
  environment:
    - RK_NPU_CORE_MASK=NPU_CORE_2       # last core for LLM
```

### Memory limits

| Service | Default | For 1.7B model |
|---------|---------|---------------|
| speech | 7500m | 7500m (unchanged) |
| llm | 4000m | 4000m (adequate) |

If you see OOM kills in `docker logs`, increase the LLM limit:

```bash
# In docker-compose.radxa.yml, change:
mem_limit: 6000m
memswap_limit: 6000m
```

---

## 10. Troubleshooting

### "RKLLM engine failed to load"

```
RuntimeError: Failed to load RKLLM model: librkllmrt.so: cannot open shared object file
```

The NPU runtime libraries aren't mounted. Check:

```bash
docker exec openvoicestream-llm ls -la /opt/asr/lib/librkllmrt.so
docker exec openvoicestream-llm ls -la /dev/rknpu*
```

### "Connection refused" from agent

The services aren't running:

```bash
docker ps | grep openvoicestream
```

If only `openvoicestream` is running but `openvoicestream-llm` is not,
check LLM container logs:

```bash
docker logs openvoicestream-llm
```

### "Model not found" in LLM logs

The `.rkllm` file isn't at the expected path inside the container:

```bash
docker exec openvoicestream-llm ls -la /opt/models/rkllm/
```

Make sure the file is in the `rk-llm-models` Docker volume:

```bash
docker volume inspect rk-llm-models
# Mountpoint shows the host path — place the .rkllm file there
```

### ASR artifacts not downloading

If the HuggingFace download fails (common on CN networks):

```bash
export HF_ENDPOINT=https://hf-mirror.com
docker compose -f deploy/docker-compose.radxa.yml up -d
```

### Agent can't find audio devices

```bash
# Install portaudio (required by sounddevice)
sudo apt-get install -y portaudio19-dev
pip install sounddevice
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

### Slow first token (> 2s)

Cold start is normal. Subsequent turns are much faster (KV cache
primed). If every turn is slow, check:

```bash
# NPU is under load
docker stats openvoicestream-llm

# Model is quantized correctly (FP16 or W8A8)
docker exec openvoicestream-llm ls -lh /opt/models/rkllm/
```

Expected cold first token: 200–500ms for 0.6B, 500–800ms for 1.7B.
Warm first token: ~80ms for 0.6B, ~150ms for 1.7B.

---

## Quick Reference

```bash
# Start everything
docker compose -f deploy/docker-compose.radxa.yml up -d

# View logs
docker compose -f deploy/docker-compose.radxa.yml logs -f

# Stop everything
docker compose -f deploy/docker-compose.radxa.yml down

# Run voice assistant
ovs-agent run apps/rk3588-chat

# Test LLM directly
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Check which models are loaded
curl -s http://localhost:8001/v1/models | python3 -m json.tool
curl -s http://localhost:8621/v1/models | python3 -m json.tool
```
