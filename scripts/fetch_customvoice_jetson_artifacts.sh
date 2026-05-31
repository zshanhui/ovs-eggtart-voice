#!/usr/bin/env bash
# Populate deploy/jetson-workers/customvoice-v071/ with the patched binary
# + plugin .so + reference talker embeds required to build the Jetson image
# (Dockerfile.jetson's CustomVoice COPY stage).
#
# These files are intentionally gitignored (deploy/jetson-workers/ in .gitignore)
# because they are large pre-built artifacts (~62 MB total) tied to a specific
# Orin NX target (sm_87, CUDA 12.6, TRT 10.3, JetPack 6.x).
#
# Source of truth:
#   - qwen3_tts_inference binary: TensorRT-Edge-LLM fork tag
#       customvoice-v071-w8a16-asr-pass-20260526 (branch v071/customvoice-product)
#   - libNvInfer_edgellm_plugin.so.1.0: production plugin reused unchanged
#       (orin-nx ~/spike-v071-nx/build/, md5 3d6761ebbe0946720f9c1d35a56c1cda)
#   - ref_talker_embeds_15row.bin: CuTe DSL init-order workaround embedding
#
# Snapshot location (orin-nx): ~/customvoice-v071-snapshot/20260526/
#
# This script copies from a local snapshot directory or pulls from the orin-nx
# snapshot via fleet. It does not currently download from HF because the
# binary is deliberately not redistributed there (it's baked into the Docker
# image instead).

set -euo pipefail

DEST="${DEST:-deploy/jetson-workers/customvoice-v071}"
SNAPSHOT="${SNAPSHOT:-}"
FLEET_DEVICE="${FLEET_DEVICE:-orin-nx}"
REMOTE_SNAPSHOT="${REMOTE_SNAPSHOT:-/home/harvest/customvoice-v071-snapshot/20260526}"
FLEET_BIN="${FLEET_BIN:-uv run --project $HOME/project/_hub python $HOME/project/_hub/fleet.py}"

# --build flag: build qwen3_tts_inference + ref embeds from source on this host
# (must be a Jetson Orin with CUDA 12.6 toolkit installed). Falls back to
# snapshot/fleet for the plugin .so since that one is not buildable from the
# patched fork — see build_customvoice_jetson_binary.sh header.
BUILD_FROM_SOURCE=0
BUILD_ARGS=()
parsed_args=()
for arg in "$@"; do
    case "$arg" in
        --build) BUILD_FROM_SOURCE=1 ;;
        *) parsed_args+=("$arg") ;;
    esac
done
if [[ ${#parsed_args[@]} -gt 0 ]]; then
    set -- "${parsed_args[@]}"
else
    set --
fi

if [[ "$BUILD_FROM_SOURCE" -eq 1 ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "==> --build: invoking build_customvoice_jetson_binary.sh"
    "$script_dir/build_customvoice_jetson_binary.sh" \
        --output-dir "$DEST" "${BUILD_ARGS[@]}"
    echo
    echo "==> Plugin .so (libNvInfer_edgellm_plugin.so.1.0) still needs to be"
    echo "    fetched separately — continuing with snapshot/fleet for that file."
fi

declare -A EXPECTED_MD5=(
    [qwen3_tts_inference]="f50fedc960d8edf7304f897cddbbdaf7"
    [libNvInfer_edgellm_plugin.so.1.0]="3d6761ebbe0946720f9c1d35a56c1cda"
    [ref_talker_embeds_15row.bin]="fed8b23ca46246f5993ec26ab7d5c0f4"
)

# When --build was passed, binary + embeds were already produced from source;
# only the plugin .so still needs to be fetched.
if [[ "$BUILD_FROM_SOURCE" -eq 1 ]]; then
    FETCH_FILES=("libNvInfer_edgellm_plugin.so.1.0")
else
    FETCH_FILES=("${!EXPECTED_MD5[@]}")
fi

mkdir -p "$DEST"

if [[ -n "$SNAPSHOT" && -d "$SNAPSHOT" ]]; then
    echo "Copying from local snapshot: $SNAPSHOT"
    for f in "${FETCH_FILES[@]}"; do
        cp -v "$SNAPSHOT/$f" "$DEST/$f"
    done
else
    echo "Pulling from $FLEET_DEVICE:$REMOTE_SNAPSHOT via fleet"
    for f in "${FETCH_FILES[@]}"; do
        $FLEET_BIN pull "$FLEET_DEVICE" "$REMOTE_SNAPSHOT/$f" "$DEST/$f"
    done
fi

echo
echo "Verifying md5:"
md5_cmd=$(command -v md5sum >/dev/null && echo "md5sum" || echo "md5 -r")
ok=true
for f in "${!EXPECTED_MD5[@]}"; do
    got=$($md5_cmd "$DEST/$f" | awk '{print $1}')
    want="${EXPECTED_MD5[$f]}"
    if [[ "$got" == "$want" ]]; then
        echo "  OK  $f  $got"
    else
        echo "  BAD $f  got=$got  want=$want" >&2
        ok=false
    fi
done

$ok || { echo "md5 mismatch — aborting"; exit 1; }
echo "All artifacts staged in $DEST"
