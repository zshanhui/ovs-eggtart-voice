#!/bin/bash
# Setup TRT-Edge-LLM ONNX export environment on WSL2 (x86_64 + NVIDIA GPU).
#
# Creates an isolated uv venv with all dependencies and applies transformers
# 5.x compatibility patches for qwen_tts/qwen_asr.
#
# Usage:
#   bash scripts/setup_trt_export_env.sh
#
# After setup, activate with:
#   cd /tmp/trt-export && uv run <export-command>
#
# Export commands:
#   uv run tensorrt-edgellm-export-llm --model_dir <HF_SNAP> --output_dir <OUT> [--export_models talker] --device cuda
#   uv run tensorrt-edgellm-export-audio --model_dir <HF_SNAP> --output_dir <OUT> [--export_models audio_encoder|tokenizer_decoder|speaker_encoder] --device cuda

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export http_proxy="" https_proxy="" no_proxy="*"

MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TRT_SRC="${TRT_SRC:-$HOME/project/tensorrt-edge-llm}"
PROJECT_DIR="/tmp/trt-export"
PYTHON="${PYTHON:-3.12}"

echo "=== Step 1: Install uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=== Step 2: Create uv project ==="
rm -rf "$PROJECT_DIR" 2>/dev/null || true
uv init --no-readme "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo "$PYTHON" > .python-version

echo "=== Step 3: Install base dependencies ==="
uv add --index-url "$MIRROR" \
    torch transformers==5.3.0 onnx nvidia-modelopt einops numpy \
    onnxruntime onnx-graphsurgeon requests sox nagisa tiktoken \
    librosa soundfile 2>&1 | tail -5

echo "=== Step 4: Add TRT-Edge-LLM as editable ==="
uv add --editable "$TRT_SRC" --index-url "$MIRROR" 2>&1 | tail -3

echo "=== Step 5: Copy qwen packages from system ==="
SYS_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
VENV_SITE="$PROJECT_DIR/.venv/lib/python$PYTHON/site-packages"

for pkg in qwen_asr qwen_tts qwen_omni_utils; do
    if [ -d "$SYS_SITE/$pkg" ]; then
        cp -r "$SYS_SITE/$pkg" "$VENV_SITE/"
        echo "  Copied $pkg"
    else
        echo "  WARNING: $pkg not found in $SYS_SITE"
    fi
done

echo "=== Step 6: Apply transformers 5.x compatibility patches ==="

# 6a. Flash-attn mock
mkdir -p "$VENV_SITE/flash_attn"
cat > "$VENV_SITE/flash_attn/__init__.py" << 'PYEOF'
__version__ = "2.0.0"
def flash_attn_func(*a, **k): return None
def flash_attn_varlen_func(*a, **k): return None
def flash_attn_qkvpacked_func(*a, **k): return None
PYEOF

# 6b. RoPE: replace all hardcoded "default" with "linear"
for f in \
    "$VENV_SITE/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py" \
    "$VENV_SITE/qwen_tts/core/models/modeling_qwen3_tts.py" \
    "$VENV_SITE/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py"
do
    sed -i 's/rope_type = "default"/rope_type = "linear"/g' "$f" 2>/dev/null || true
done
# Also fix the fallback in qwen_asr
sed -i 's/config.rope_scaling.get("rope_type", "default")/config.rope_scaling.get("rope_type", "linear")/g' \
    "$VENV_SITE/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py" 2>/dev/null || true

# 6c. Remove check_model_inputs (removed in transformers 5.x)
for f in \
    "$VENV_SITE/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py" \
    "$VENV_SITE/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
do
    sed -i 's/, check_model_inputs//g' "$f" 2>/dev/null || true
    sed -i 's/@check_model_inputs()//g' "$f" 2>/dev/null || true
    sed -i 's/from transformers.utils.generic import check_model_inputs//g' "$f" 2>/dev/null || true
done

# 6d. Guard torchaudio import in qwen_tts
QTT_VQ="$VENV_SITE/qwen_tts/core/tokenizer_25hz/vq/speech_vq.py"
if [ -f "$QTT_VQ" ]; then
    sed -i 's/import torchaudio.compliance.kaldi as kaldi/try:\n    import torchaudio.compliance.kaldi as kaldi\nexcept Exception:\n    kaldi = None/' "$QTT_VQ"
fi

# 6e. Fix transformers tokenizer duplicate kwarg
TOK_UTILS="$VENV_SITE/transformers/tokenization_utils_tokenizers.py"
if [ -f "$TOK_UTILS" ]; then
    sed -i 's|fix_mistral_regex=kwargs.get("fix_mistral_regex"),|# fix_mistral_regex handled via **kwargs|' "$TOK_UTILS"
fi

echo "=== Step 7: Fix HF model configs (rope_type: default → linear) ==="
# Fix cached HF model configs
for snap in $(find ~/.cache/huggingface/hub -name "config.json" -path "*Qwen3-ASR*" -o -name "config.json" -path "*Qwen3-TTS*" 2>/dev/null); do
    python3 -c "
import json
with open('$snap') as f:
    cfg = json.load(f)

def fix_config(d, path=''):
    if isinstance(d, dict):
        rs = d.get('rope_scaling', {})
        if rs:
            t = rs.get('rope_type', rs.get('type', ''))
            if t in ('default', 'linear'):
                rs['rope_type'] = 'linear'
                rs['type'] = 'linear'
                rs.setdefault('factor', 1.0)
                rs.setdefault('rope_theta', float(d.get('rope_theta', 1000000)))
        for k, v in d.items():
            fix_config(v, f'{path}.{k}')
    elif isinstance(d, list):
        for i, v in enumerate(d):
            fix_config(v, f'{path}[{i}]')

fix_config(cfg)
with open('$snap', 'w') as f:
    json.dump(cfg, f, indent=2)
" 2>/dev/null
done

echo "=== Done! ==="
echo "Environment ready at: $PROJECT_DIR"
echo ""
echo "Usage examples:"
echo "  cd $PROJECT_DIR"
echo "  uv run tensorrt-edgellm-export-llm --model_dir <HF_SNAP> --output_dir <OUT> --device cuda"
echo "  uv run tensorrt-edgellm-export-audio --model_dir <HF_SNAP> --output_dir <OUT> --export_models tokenizer_decoder --device cuda"
