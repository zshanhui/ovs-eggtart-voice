#!/bin/bash
# -*- shell-script -*-
#
# convert_model.sh — run RKLLM model conversion (requires setup_rkllm.sh first).
#
# Usage:
#   ./convert_model.sh
#
# Override defaults via env vars:
#   MODEL_ID      HuggingFace model  (default: Qwen/Qwen3-0.6B)
#   TARGET        RK3576 or RK3588   (default: RK3576)
#   QUANT         W4A16 / W8A8 / FP16 (default: W4A16)
#   MAX_CONTEXT   Context length     (default: 4096)
#   GPU_DEVICE    CUDA device index  (default: 0)
#   DTYPE         float16/bfloat16/float32 (default: float16)
#   VENV_DIR      Path to Python venv (default: /tmp/rkllm-env)
#   SCRIPT_DIR    Path to build_rkllm_model.py (auto-detected)
#
# Examples:
#   # 0.6B model for RK3576 (default)
#   ./convert_model.sh
#
#   # 1.7B model for RK3588
#   MODEL_ID="Qwen/Qwen3-1.7B" TARGET="RK3588" ./convert_model.sh
#
#   # 4B model for RK3588 with better quantization
#   MODEL_ID="Qwen/Qwen3-4B" TARGET="RK3588" QUANT="W8A8" ./convert_model.sh

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
TARGET="${TARGET:-RK3576}"
QUANT="${QUANT:-W4A16}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
GPU_DEVICE="${GPU_DEVICE:-0}"
DTYPE="${DTYPE:-float16}"
VENV_DIR="${VENV_DIR:-/tmp/rkllm-env}"

echo "============================================================"
echo " RKLLM Model Conversion"
echo "============================================================"
echo " Model:       ${MODEL_ID}"
echo " Target:      ${TARGET}"
echo " Quant:       ${QUANT}"
echo " Max context: ${MAX_CONTEXT}"
echo " Dtype:       ${DTYPE}"
echo " GPU:         ${GPU_DEVICE}"
echo "============================================================"
echo ""

# ── Activate venv ──────────────────────────────────────────────────────
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV_DIR}"
    echo "Run setup_rkllm.sh first."
    exit 1
fi

# ── Locate build script ────────────────────────────────────────────────
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BUILD_SCRIPT="${SCRIPT_DIR}/build_rkllm_model.py"

if [ ! -f "${BUILD_SCRIPT}" ]; then
    echo "ERROR: build_rkllm_model.py not found at ${BUILD_SCRIPT}"
    echo "Set SCRIPT_DIR to the directory containing build_rkllm_model.py"
    exit 1
fi

# ── Run conversion ─────────────────────────────────────────────────────
echo "==> Running conversion ..."
python3 "${BUILD_SCRIPT}" \
    --model "${MODEL_ID}" \
    --target "${TARGET}" \
    --quant "${QUANT}" \
    --max-context "${MAX_CONTEXT}" \
    --gpu "${GPU_DEVICE}" \
    --dtype "${DTYPE}"

# ── Show result ────────────────────────────────────────────────────────
MODEL_SLUG=$(basename "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')
OUTFILE="${MODEL_SLUG}_${QUANT^^}_${TARGET}.rkllm"

if [ ! -f "${OUTFILE}" ]; then
    echo "ERROR: Output file ${OUTFILE} not found — check the logs above."
    exit 1
fi

SIZE=$(du -h "${OUTFILE}" | cut -f1)
SHA=$(sha256sum "${OUTFILE}" | cut -d' ' -f1)
BYTES=$(stat -c%s "${OUTFILE}" 2>/dev/null || stat -f%z "${OUTFILE}")

echo ""
echo "============================================================"
echo "  Conversion complete!"
echo "============================================================"
echo "  File:     ${OUTFILE}"
echo "  Size:     ${SIZE}  (${BYTES} bytes)"
echo "  SHA256:   ${SHA}"
echo "============================================================"
echo ""
echo "  Download to your local machine:"
echo "    scp user@host:${PWD}/${OUTFILE} ./"
echo ""
echo "  Upload to RK device:"
echo "    scp ${OUTFILE} cat@<rk-device>:/opt/models/rkllm/"
echo ""
echo "  Then on the device:"
echo "    export RKLLM_MODEL_PATH=/opt/models/rkllm/${OUTFILE}"
echo "    export RKLLM_ENGINE=real"
echo "    docker compose -f deploy/docker-compose.rk.yml up -d llm"
echo ""
echo "  Manifest entry for deploy/artifacts/rk_manifest.json:"
echo "    {"
echo "      \"path\": \"opt/models/rkllm/${OUTFILE}\","
echo "      \"size_bytes\": ${BYTES},"
echo "      \"sha256\": \"${SHA}\""
echo "    }"
