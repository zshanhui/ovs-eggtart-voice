#!/usr/bin/env bash
# Build CustomVoice v0.7.1 patched qwen3_tts_inference binary + generate
# ref_talker_embeds_15row.bin from source on a Jetson Orin (sm_87, CUDA 12.6).
#
# Source: TensorRT-Edge-LLM fork branch v071/customvoice-product, tag
#   customvoice-v071-w8a16-asr-pass-20260526
# All 3 root-bug fixes are already in the tag — no extra patches needed.
#
# NOTE: this script intentionally does NOT build libNvInfer_edgellm_plugin.so.
# The patched fork's plugin build fails at ~97% in fp4SupportKernels/buildLayout.h
# (header missing — known limitation). The production plugin .so is reused
# unchanged across customvoice and high-perf paths (md5 3d6761ebbe0946720f9c1d35a56c1cda).
# Pull it from a snapshot or Docker image; see fetch_customvoice_jetson_artifacts.sh.
#
# Spec: docs/specs/customvoice-tts-fork-port-handoff.md

set -euo pipefail

DEFAULT_TAG="customvoice-v071-w8a16-asr-pass-20260526"
DEFAULT_FORK="https://github.com/suharvest/TensorRT-Edge-LLM.git"
DEFAULT_CUDA="/usr/local/cuda-12.6"
DEFAULT_ARCH="87"
DEFAULT_OUT="deploy/jetson-workers/customvoice-v071"
DEFAULT_HF_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

TAG="$DEFAULT_TAG"
FORK_REPO="$DEFAULT_FORK"
WORKDIR=""
CUDA_ROOT="$DEFAULT_CUDA"
CUDA_ARCH="$DEFAULT_ARCH"
OUTPUT_DIR="$DEFAULT_OUT"
SKIP_EMBEDS=0
HF_MODEL="$DEFAULT_HF_MODEL"

EXPECTED_EMBEDS_MD5="fed8b23ca46246f5993ec26ab7d5c0f4"

usage() {
    cat <<EOF
Usage: $0 [options]

Build CustomVoice v0.7.1 patched qwen3_tts_inference + generate ref embeds.

Options:
  --tag <git-tag>         Fork tag to build (default: $DEFAULT_TAG)
  --fork-repo <url>       Fork git URL (default: $DEFAULT_FORK)
  --workdir <dir>         Build workdir (default: /tmp/cv-build-<epoch>)
  --cuda-root <path>      CUDA toolkit root (default: $DEFAULT_CUDA)
  --cuda-arch <num>       CUDA arch (default: $DEFAULT_ARCH, orin-nx/orin-nano)
  --output-dir <dir>      Where to copy artifacts (default: $DEFAULT_OUT)
  --hf-model <id>         HF model id for embeds dump (default: $DEFAULT_HF_MODEL)
  --skip-embeds           Skip ref_talker_embeds_15row.bin generation
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --fork-repo) FORK_REPO="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --hf-model) HF_MODEL="$2"; shift 2 ;;
        --skip-embeds) SKIP_EMBEDS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$WORKDIR" ]]; then
    WORKDIR="/tmp/cv-build-$(date +%s)"
fi

echo "==> CustomVoice v0.7.1 patched binary build"
echo "    tag       : $TAG"
echo "    fork repo : $FORK_REPO"
echo "    workdir   : $WORKDIR"
echo "    cuda root : $CUDA_ROOT"
echo "    cuda arch : $CUDA_ARCH"
echo "    output    : $OUTPUT_DIR"
echo "    skip embeds: $SKIP_EMBEDS"
echo

# ---- prereq checks ----
echo "==> Checking prerequisites"
for tool in git cmake make python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found in PATH" >&2
        exit 1
    fi
done
if [[ ! -d "$CUDA_ROOT" ]]; then
    echo "ERROR: CUDA root not found: $CUDA_ROOT" >&2
    exit 1
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
    echo "ERROR: nvcc not found at $CUDA_ROOT/bin/nvcc" >&2
    exit 1
fi
echo "    nvcc: $("$CUDA_ROOT"/bin/nvcc --version | tail -1)"

# md5 helper (md5sum on Linux, md5 -r on macOS)
if command -v md5sum >/dev/null 2>&1; then
    md5_cmd() { md5sum "$1" | awk '{print $1}'; }
else
    md5_cmd() { md5 -r "$1" | awk '{print $1}'; }
fi

# ---- clone fork @ tag ----
echo
echo "==> Fetching fork @ $TAG"
mkdir -p "$WORKDIR"
SRC_DIR="$WORKDIR/TensorRT-Edge-LLM"
if [[ -d "$SRC_DIR/.git" ]]; then
    echo "    workdir already has clone, fetching tag"
    git -C "$SRC_DIR" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
    git -C "$SRC_DIR" checkout "tags/$TAG"
else
    git clone --depth 1 --branch "$TAG" "$FORK_REPO" "$SRC_DIR"
fi
echo "    head: $(git -C "$SRC_DIR" rev-parse --short HEAD)"

# ---- configure + build ----
BUILD_DIR="$SRC_DIR/build_container"
echo
echo "==> Configuring cmake (Release, sm_$CUDA_ARCH, jetson-orin)"
mkdir -p "$BUILD_DIR"
( cd "$BUILD_DIR" && cmake .. \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DENABLE_CUTE_DSL=gemm \
    -DCUTE_DSL_ARTIFACT_TAG="sm_$CUDA_ARCH" \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_DIR="$CUDA_ROOT" \
    -DCUDA_CTK_VERSION=12.6 )

echo
echo "==> Building qwen3_tts_inference (Plugin .so NOT built — see header comment)"
NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
( cd "$BUILD_DIR" && make -j"$NPROC" qwen3_tts_inference )

BIN_SRC="$BUILD_DIR/examples/omni/qwen3_tts_inference"
if [[ ! -x "$BIN_SRC" ]]; then
    echo "ERROR: build did not produce $BIN_SRC" >&2
    exit 1
fi
echo "    built: $BIN_SRC ($(md5_cmd "$BIN_SRC"))"

# ---- ref embeds dump ----
EMBEDS_PATH="$WORKDIR/ref_talker_embeds_15row.bin"
if [[ "$SKIP_EMBEDS" -eq 1 ]]; then
    echo
    echo "==> Skipping ref_talker_embeds_15row.bin generation (--skip-embeds)"
else
    echo
    echo "==> Dumping ref_talker_embeds_15row.bin (deterministic seed=42)"
    echo "    model: $HF_MODEL"
    QWEN3_TTS_SEED=42 \
    EMBEDS_OUT="$EMBEDS_PATH" \
    HF_MODEL_ID="$HF_MODEL" \
    python3 - <<'PYEOF'
import os
import torch
from qwen_tts import Qwen3TTSModel  # noqa: F401

ckpt = os.environ["HF_MODEL_ID"]
out = os.environ["EMBEDS_OUT"]

torch.manual_seed(int(os.environ.get("QWEN3_TTS_SEED", "42")))
tts = Qwen3TTSModel.from_pretrained(ckpt, device_map="cuda:0", dtype=torch.bfloat16)

captured = []
orig_cat = torch.cat

def spy(ts, dim=0, **kw):
    o = orig_cat(ts, dim=dim, **kw)
    if o.dim() == 3 and tuple(o.shape) == (1, 15, 1024):
        captured.append(o.detach().clone())
    return o

torch.cat = spy
try:
    _ = tts.generate_custom_voice(
        text="今天天气真不错",
        language="Chinese",
        speaker="Vivian",
        instruct="",
        max_new_tokens=8,
    )
finally:
    torch.cat = orig_cat

if not captured:
    raise SystemExit("ERROR: no (1,15,1024) tensor captured during generate_custom_voice")
captured[-1].cpu().float().numpy().tofile(out)
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
PYEOF
    if [[ ! -f "$EMBEDS_PATH" ]]; then
        echo "ERROR: embeds dump did not produce $EMBEDS_PATH" >&2
        exit 1
    fi
    got_md5="$(md5_cmd "$EMBEDS_PATH")"
    if [[ "$got_md5" == "$EXPECTED_EMBEDS_MD5" ]]; then
        echo "    md5 OK: $got_md5"
    else
        echo "    WARNING: md5 mismatch — got=$got_md5 want=$EXPECTED_EMBEDS_MD5" >&2
        echo "    (may be benign if HF checkpoint was updated; verify ASR end-to-end)" >&2
    fi
fi

# ---- copy artifacts ----
echo
echo "==> Copying artifacts to $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
cp -v "$BIN_SRC" "$OUTPUT_DIR/qwen3_tts_inference"
echo "    OK qwen3_tts_inference  $(md5_cmd "$OUTPUT_DIR/qwen3_tts_inference")"
if [[ "$SKIP_EMBEDS" -ne 1 ]]; then
    cp -v "$EMBEDS_PATH" "$OUTPUT_DIR/ref_talker_embeds_15row.bin"
    echo "    OK ref_talker_embeds_15row.bin  $(md5_cmd "$OUTPUT_DIR/ref_talker_embeds_15row.bin")"
fi

# ---- next-step hints ----
echo
echo "==> Done. Next steps:"
echo "  * libNvInfer_edgellm_plugin.so.1.0 (~44 MB) is NOT built here — patched"
echo "    fork's plugin build fails at fp4SupportKernels/buildLayout.h (known)."
echo "    Reuse the production plugin .so:"
echo "      - extract from a running Docker image, or"
echo "      - pull from the orin-nx snapshot:"
echo "          scripts/fetch_customvoice_jetson_artifacts.sh"
echo "        (expected md5 3d6761ebbe0946720f9c1d35a56c1cda)"
echo "  * Build the Jetson image:"
echo "      docker build -f deploy/docker/Dockerfile.jetson ..."
echo "  * See docs/specs/customvoice-tts-fork-port-handoff.md for verification."
