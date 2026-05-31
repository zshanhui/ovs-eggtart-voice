#!/usr/bin/env bash
# Build MOSS-TTS-Nano TensorRT engines from ONNX on a Jetson device.
#
# Inputs (env vars or defaults):
#   ONNX_DIR        — dir containing moss_tts_*.onnx + .data + tts_browser_onnx_meta.json
#                     and browser_poc_manifest.json + tokenizer.model
#   CODEC_ONNX_DIR  — dir containing moss_audio_tokenizer_*.onnx + codec_browser_onnx_meta.json
#   OUT_DIR         — root of /opt/models/moss-tts-nano (engines/ + codec_onnx/ subdirs)
#   TRTEXEC         — path to trtexec (default /usr/src/tensorrt/bin/trtexec)
#
# Usage:
#   ONNX_DIR=$HOME/moss-onnx \
#   CODEC_ONNX_DIR=$HOME/moss-codec-onnx \
#   OUT_DIR=/opt/models/moss-tts-nano \
#   bash scripts/build_moss_tts_engines.sh

set -euo pipefail

ONNX_DIR=${ONNX_DIR:-/opt/models/moss-tts-nano/onnx}
CODEC_ONNX_DIR=${CODEC_ONNX_DIR:-/opt/models/moss-tts-nano/codec_onnx}
OUT_DIR=${OUT_DIR:-/opt/models/moss-tts-nano}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

ENGINES_DIR="${OUT_DIR}/engines"
CODEC_DIR="${OUT_DIR}/codec_onnx"

mkdir -p "${ENGINES_DIR}" "${CODEC_DIR}"

echo "[build] ONNX_DIR=${ONNX_DIR}"
echo "[build] CODEC_ONNX_DIR=${CODEC_ONNX_DIR}"
echo "[build] OUT_DIR=${OUT_DIR}"
echo "[build] trtexec: $(${TRTEXEC} --version 2>&1 | head -1)"

# ---- 1) Prefill engine ------------------------------------------------------
echo "[build] (1/6) moss_tts_prefill.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_prefill.onnx" --fp16 \
  --minShapes=input_ids:1x1x17,attention_mask:1x1,past_valid_lengths:1 \
  --optShapes=input_ids:1x32x17,attention_mask:1x32,past_valid_lengths:1 \
  --maxShapes=input_ids:1x256x17,attention_mask:1x256,past_valid_lengths:1 \
  --saveEngine="${ENGINES_DIR}/moss_tts_prefill.plan" 2>&1 | tail -8

# ---- 2) Decode step (12 layers KV cache; all past_key_*/past_value_* dynamic) ---
echo "[build] (2/6) moss_tts_decode_step.plan ..."
PAST_MIN="" PAST_OPT="" PAST_MAX=""
for i in $(seq 0 11); do
  PAST_MIN+="past_key_${i}:1x1x12x64,past_value_${i}:1x1x12x64,"
  PAST_OPT+="past_key_${i}:1x64x12x64,past_value_${i}:1x64x12x64,"
  PAST_MAX+="past_key_${i}:1x512x12x64,past_value_${i}:1x512x12x64,"
done
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_decode_step.onnx" --fp16 \
  --minShapes="input_ids:1x1x17,attention_mask:1x1,past_valid_lengths:1,${PAST_MIN%,}" \
  --optShapes="input_ids:1x1x17,attention_mask:1x64,past_valid_lengths:1,${PAST_OPT%,}" \
  --maxShapes="input_ids:1x1x17,attention_mask:1x512,past_valid_lengths:1,${PAST_MAX%,}" \
  --saveEngine="${ENGINES_DIR}/moss_tts_decode_step.plan" 2>&1 | tail -8

# ---- 3) Local decoder (one-shot, static shapes) ----------------------------
echo "[build] (3/6) moss_tts_local_decoder.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_local_decoder.onnx" --fp16 \
  --saveEngine="${ENGINES_DIR}/moss_tts_local_decoder.plan" 2>&1 | tail -5

# ---- 4) Local cached step (optional, MVP unused but loaded) -----------------
echo "[build] (4/6) moss_tts_local_cached_step.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_local_cached_step.onnx" --fp16 \
  --saveEngine="${ENGINES_DIR}/moss_tts_local_cached_step.plan" 2>&1 | tail -5 || \
  echo "[build] WARN: local_cached_step build failed (non-fatal; engine is optional)"

# ---- 5) Local fixed sampled frame (production sampler, static) -------------
echo "[build] (5/6) moss_tts_local_fixed_sampled_frame.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_local_fixed_sampled_frame.onnx" --fp16 \
  --saveEngine="${ENGINES_DIR}/moss_tts_local_fixed_sampled_frame.plan" 2>&1 | tail -5

# ---- 6) Codec decode_step (audio_codes dynamic on frame count) -------------
echo "[build] (6/6) codec_decode_step.plan ..."
${TRTEXEC} --onnx="${CODEC_ONNX_DIR}/moss_audio_tokenizer_decode_step.onnx" --fp16 \
  --minShapes=audio_codes:1x1x16,audio_code_lengths:1 \
  --optShapes=audio_codes:1x4x16,audio_code_lengths:1 \
  --maxShapes=audio_codes:1x8x16,audio_code_lengths:1 \
  --saveEngine="${CODEC_DIR}/codec_decode_step.plan" 2>&1 | tail -8

# ---- Stage sidecar files (metadata, tokenizer, data files) -----------------
echo "[build] staging metadata + tokenizer + .data files ..."
for f in tts_browser_onnx_meta.json browser_poc_manifest.json tokenizer.model \
         moss_tts_global_shared.data moss_tts_local_shared.data; do
  [[ -f "${ONNX_DIR}/${f}" ]] && cp "${ONNX_DIR}/${f}" "${ENGINES_DIR}/${f}"
done
for f in codec_browser_onnx_meta.json moss_audio_tokenizer_encode.onnx \
         moss_audio_tokenizer_encode.data moss_audio_tokenizer_decode_shared.data; do
  [[ -f "${CODEC_ONNX_DIR}/${f}" ]] && cp "${CODEC_ONNX_DIR}/${f}" "${CODEC_DIR}/${f}"
done

# Symlink codec assets into engines/ (worker hardcodes codec_*.{plan,json} lookup under engineDir).
ln -sf "${CODEC_DIR}/codec_decode_step.plan" "${ENGINES_DIR}/codec_decode_step.plan"
ln -sf "${CODEC_DIR}/codec_browser_onnx_meta.json" "${ENGINES_DIR}/codec_browser_onnx_meta.json"

echo "[build] DONE -> ${OUT_DIR}"
ls -lh "${ENGINES_DIR}" "${CODEC_DIR}"
