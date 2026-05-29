#!/bin/bash
# -*- shell-script -*-
#
# setup_rkllm.sh — install system deps + RKLLM-Toolkit on a CUDA instance.
#
# Run once after provisioning a GPU instance (Vast.ai, Lambda, etc.).
# After this, run convert_model.sh as many times as you like.
#
# Usage:
#   chmod +x setup_rkllm.sh
#   ./setup_rkllm.sh
#
# Env vars:
#   VENV_DIR       Path to Python venv   (default: /tmp/rkllm-env)
#   RKLLM_VERSION  Toolkit version       (default: 1.2.3)
#   RKLLM_TOOLKIT_URL  Override wheel URL (default: fetch from GitHub)

set -euo pipefail

VENV_DIR="${VENV_DIR:-/tmp/rkllm-env}"
RKLLM_VERSION="${RKLLM_VERSION:-1.2.3}"

echo "============================================================"
echo " RKLLM Environment Setup"
echo "============================================================"
echo " Venv:        ${VENV_DIR}"
echo " RKLLM ver:   ${RKLLM_VERSION}"
echo "============================================================"
echo ""

# ── System deps ────────────────────────────────────────────────────────
echo "==> Installing system packages ..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev git curl build-essential 2>&1 | tail -1

# ── Virtual env + uv ────────────────────────────────────────────────────
echo "==> Creating venv at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Installing uv ..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ── Install packages explicitly (no auto dep resolution) ──────────
# The RKLLM wheel lists auto-gptq as a hard dependency, but auto-gptq
# fails to build from source on most systems.  We skip it entirely
# (it's only needed for GPTQ quantization, which we never use).
# Strategy: install the wheel with --no-deps, then install all its
# real dependencies except auto-gptq.
echo "==> Installing RKLLM-Toolkit dependencies ..."

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

case "${PY_VER}" in
    39) WHL_TAG="cp39-cp39" ;;
    310) WHL_TAG="cp310-cp310" ;;
    311) WHL_TAG="cp311-cp311" ;;
    312) WHL_TAG="cp312-cp312" ;;
    *)
        echo "ERROR: Unsupported Python version 3.${PY_VER:1}."
        echo "RKLLM-Toolkit requires Python 3.9–3.12."
        exit 1
        ;;
esac

WHEEL="rkllm_toolkit-${RKLLM_VERSION}-${WHL_TAG}-linux_x86_64.whl"
GITHUB_BASE="https://raw.githubusercontent.com/airockchip/rknn-llm/main/rkllm-toolkit/packages"
WHEEL_URL="${RKLLM_TOOLKIT_URL:-${GITHUB_BASE}/${WHEEL}}"

echo "  Python:  3.${PY_VER:1}  →  ${WHL_TAG}"

# 1. Install the wheel without its dependency list
echo "  Installing RKLLM wheel (--no-deps) ..."
uv pip install -q --no-deps "${WHEEL_URL}"

# 2. Install all real dependencies except auto-gptq
echo "  Installing dependencies ..."
uv pip install -q \
  "setuptools<70" \
  numpy \
  transformers \
  torch==2.6.0 \
  datasets \
  pyarrow \
  tqdm \
  sentencepiece \
  accelerate \
  protobuf \
  transformers_stream_generator \
  einops \
  scipy \
  tiktoken \
  tabulate \
  Jinja2 \
  safetensors \
  colorlog \
  datamodel_code_generator \
  jsonschema \
  flatbuffers \
  torchvision \
  pillow \
  optimum \
  jsonlines \
  timm \
  easydict \
  addict \
  huggingface_hub

python3 -c "from rkllm.api import RKLLM; print('  OK — rkllm.api.RKLLM imported')" || {
    echo ""
    echo "  RKLLM-Toolkit import failed."
    echo "  The GitHub wheel may be incompatible. Try overriding the URL:"
    echo "    RKLLM_TOOLKIT_URL=https://... ./setup_rkllm.sh"
    exit 1
}

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  To activate the environment later:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  To run a conversion:"
echo "    ./convert_model.sh"
echo ""
