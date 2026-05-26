#!/bin/bash
# -*- shell-script -*-
#
# setup_and_convert.sh — one-shot RKLLM model conversion on a CUDA instance.
#
# Run this on a fresh Ubuntu 22.04+ instance with an NVIDIA GPU (Vast.ai,
# Lambda, local workstation, etc.).  The script installs every dependency,
# builds a venv, downloads the RKLLM-Toolkit, and runs the conversion.
#
# Usage (after SSH'ing in):
#
#   chmod +x setup_and_convert.sh
#   MODEL_ID="Qwen/Qwen3-0.6B-Instruct" TARGET="RK3576" QUANT="W4A16" ./setup_and_convert.sh
#
# Or for the 1.8B model on RK3588:
#
#   MODEL_ID="Qwen/Qwen3-1.8B-Instruct" TARGET="RK3588" QUANT="W4A16" ./setup_and_convert.sh
#
# Env vars:
#   MODEL_ID     HuggingFace model  (default: Qwen/Qwen3-0.6B-Instruct)
#   TARGET       RK3576 or RK3588   (default: RK3576)
#   QUANT        W4A16 / W8A8 / FP16 (default: W4A16)
#   MAX_CONTEXT  Context length     (default: 4096)
#   GPU_DEVICE   CUDA device index  (default: 0)
#   RKLLM_SDK_URL  Override SDK download URL (default: Rockchip CDN)
#   SCRIPT_DIR   Path to build_rkllm_model.py (auto-detected)

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B-Instruct}"
TARGET="${TARGET:-RK3576}"
QUANT="${QUANT:-W4A16}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
GPU_DEVICE="${GPU_DEVICE:-0}"
VENV_DIR="${VENV_DIR:-/tmp/rkllm-env}"
WORK_DIR="${WORK_DIR:-/tmp/rkllm-work}"

# Rockchip SDK download (public link — may need updating for newer SDK versions).
# Download the zip from https://console.zbox.filez.com/l/RJJDmB (code: rkllm)
# and host it somewhere accessible, or set RKLLM_SDK_ZIP to a local path.
RKLLM_SDK_ZIP="${RKLLM_SDK_ZIP:-}"

echo "============================================================"
echo " RKLLM Model Conversion"
echo "============================================================"
echo " Model:       ${MODEL_ID}"
echo " Target:      ${TARGET}"
echo " Quant:       ${QUANT}"
echo " Max context: ${MAX_CONTEXT}"
echo " GPU:         ${GPU_DEVICE}"
echo "============================================================"
echo ""

# ── System deps ────────────────────────────────────────────────────────
echo "==> Installing system packages ..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv python3-dev git curl 2>&1 | tail -1

# ── Virtual env ────────────────────────────────────────────────────────
echo "==> Creating venv at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install -q --upgrade pip setuptools wheel

# ── HuggingFace deps ───────────────────────────────────────────────────
echo "==> Installing HuggingFace packages ..."
pip install -q transformers huggingface_hub accelerate

# ── RKLLM-Toolkit ──────────────────────────────────────────────────────
echo "==> Installing RKLLM-Toolkit ..."
if [ -n "${RKLLM_SDK_ZIP}" ] && [ -f "${RKLLM_SDK_ZIP}" ]; then
    # Local zip file provided.
    unzip -qo "${RKLLM_SDK_ZIP}" -d "${WORK_DIR}/rkllm-sdk"
elif [ -n "${RKLLM_SDK_ZIP}" ] && [[ "${RKLLM_SDK_ZIP}" == http* ]]; then
    # Remote URL.
    mkdir -p "${WORK_DIR}"
    curl -L -o "${WORK_DIR}/rkllm-sdk.zip" "${RKLLM_SDK_ZIP}"
    unzip -qo "${WORK_DIR}/rkllm-sdk.zip" -d "${WORK_DIR}/rkllm-sdk"
else
    # Attempt to download from the default Rockchip CDN.
    # The SDK is distributed as a zip containing .whl files for rkllm-toolkit.
    # This URL is the public Lenovo Box link — you may need to re-download
    # and re-host if it expires.
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  RKLLM SDK not provided.                                    ║"
    echo "  ║                                                              ║"
    echo "  ║  Download manually from:                                     ║"
    echo "  ║  https://console.zbox.filez.com/l/RJJDmB  (code: rkllm)     ║"
    echo "  ║                                                              ║"
    echo "  ║  Then re-run with:                                           ║"
    echo "  ║  RKLLM_SDK_ZIP=/path/to/rkllm-sdk.zip ./setup_and_convert.sh ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
fi

# Install rkllm-toolkit wheel(s) from the SDK.
if [ -d "${WORK_DIR}/rkllm-sdk" ]; then
    # Find and install the rkllm-toolkit wheel.
    WHL=$(find "${WORK_DIR}/rkllm-sdk" -name "rkllm_toolkit*.whl" -o -name "rkllm*.whl" 2>/dev/null | head -1)
    if [ -n "${WHL}" ]; then
        pip install -q "${WHL}"
        echo "  Installed: ${WHL}"
    else
        # Try installing from the extracted directory.
        if [ -f "${WORK_DIR}/rkllm-sdk/setup.py" ]; then
            pip install -q "${WORK_DIR}/rkllm-sdk/"
        elif [ -f "${WORK_DIR}/rkllm-sdk/rkllm-toolkit/setup.py" ]; then
            pip install -q "${WORK_DIR}/rkllm-sdk/rkllm-toolkit/"
        else
            echo "  WARNING: Could not find rkllm-toolkit wheel or setup.py in SDK."
            echo "  Listing SDK contents:"
            find "${WORK_DIR}/rkllm-sdk" -maxdepth 3 -type f | head -20
        fi
    fi
fi

# ── Verify toolkit installed ───────────────────────────────────────────
echo "==> Verifying RKLLM-Toolkit ..."
python3 -c "from rkllm.api import RKLLM; print('  OK — rkllm.api.RKLLM imported')" || {
    echo ""
    echo "  RKLLM-Toolkit import failed."
    echo "  This is normal if the SDK zip hasn't been downloaded yet."
    echo "  After downloading it, re-run this script with RKLLM_SDK_ZIP set."
    echo ""
    exit 1
}

# ── Run conversion ─────────────────────────────────────────────────────
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BUILD_SCRIPT="${SCRIPT_DIR}/build_rkllm_model.py"

if [ ! -f "${BUILD_SCRIPT}" ]; then
    echo "ERROR: build_rkllm_model.py not found at ${BUILD_SCRIPT}"
    echo "Set SCRIPT_DIR to the directory containing build_rkllm_model.py"
    exit 1
fi

echo "==> Running conversion ..."
python3 "${BUILD_SCRIPT}" \
    --model "${MODEL_ID}" \
    --target "${TARGET}" \
    --quant "${QUANT}" \
    --max-context "${MAX_CONTEXT}" \
    --gpu "${GPU_DEVICE}"

# ── Show result ────────────────────────────────────────────────────────
MODEL_SLUG=$(basename "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')
OUTFILE="${MODEL_SLUG}_${QUANT^^}_${TARGET}.rkllm"

if [ -f "${OUTFILE}" ]; then
    SIZE=$(du -h "${OUTFILE}" | cut -f1)
    SHA=$(sha256sum "${OUTFILE}" | cut -d' ' -f1)
    echo ""
    echo "============================================================"
    echo "  Conversion complete!"
    echo "============================================================"
    echo "  File:     ${OUTFILE}"
    echo "  Size:     ${SIZE}"
    echo "  SHA256:   ${SHA}"
    echo ""
    echo "  Download to your local machine:"
    echo "    scp user@host:${PWD}/${OUTFILE} ./"
    echo ""
    echo "  Upload to RK device:"
    echo "    scp ${OUTFILE} cat@<rk-device>:/opt/models/rkllm/"
    echo ""
    echo "  Then on the device, update docker-compose:"
    echo "    RKLLM_MODEL_PATH=/opt/models/rkllm/${OUTFILE}"
    echo "    RKLLM_ENGINE=real"
    echo "============================================================"
    echo ""
    echo "  Add this to deploy/artifacts/rk_manifest.json:"
    echo "    {"
    echo "      \"path\": \"opt/models/rkllm/${OUTFILE}\","
    echo "      \"size_bytes\": $(stat -c%s "${OUTFILE}" 2>/dev/null || stat -f%z "${OUTFILE}"),"
    echo "      \"sha256\": \"${SHA}\""
    echo "    }"
else
    echo "ERROR: Output file not found — conversion may have failed."
    exit 1
fi
