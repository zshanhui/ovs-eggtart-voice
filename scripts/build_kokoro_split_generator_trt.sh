#!/usr/bin/env bash
set -euo pipefail

# Build the validated Kokoro split-generator TensorRT artifacts on Jetson.
#
# Product path:
#   TRT encoder prefix -> CPU length regulator -> TRT decoder backbone FP16
#   -> TRT source BF16 -> TRT generator rest FP16 -> CPU post/ISTFT
#
# This script intentionally avoids ONNX Runtime GPU/TensorRT EP. It prepares
# the ONNX subgraphs used by the split runtime and builds one requested engine
# selected by ENGINE_NAME. engine_resolver calls this script once per engine.

MODEL_DIR="${MODEL_DIR:-/opt/models/kokoro-multi-lang-v1_0}"
OUT_DIR="${OUT_DIR:-${MODEL_DIR}/engines}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PYTHON="${PYTHON:-python3}"
ENGINE_NAME="${ENGINE_NAME:-kokoro_prefix_encoder_dyn4_128_fp16.engine}"
WS="${WS:-512}"

MIN_T="${MIN_T:-4}"
OPT_T="${OPT_T:-150}"
MAX_T="${MAX_T:-256}"
MIN_SEQ="${MIN_SEQ:-4}"
OPT_SEQ="${OPT_SEQ:-64}"
MAX_SEQ="${MAX_SEQ:-128}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${TRTEXEC}" ]]; then
  echo "trtexec not found or not executable: ${TRTEXEC}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.onnx" ]]; then
  echo "Kokoro model.onnx not found: ${MODEL_DIR}/model.onnx" >&2
  exit 1
fi

MIN_2T=$((MIN_T * 2))
OPT_2T=$((OPT_T * 2))
MAX_2T=$((MAX_T * 2))
MIN_SRC=$((MIN_T * 120 + 1))
OPT_SRC=$((OPT_T * 120 + 1))
MAX_SRC=$((MAX_T * 120 + 1))

FULL_STFT_CONV="${OUT_DIR}/model_stft_conv.onnx"
PREFIX_ONNX="${OUT_DIR}/kokoro_prefix_encoder.onnx"
LENGTH_ONNX="${OUT_DIR}/cpu_length_regulator.onnx"
DECODER_ONNX="${OUT_DIR}/decoder_backbone_prefix.onnx"
SOURCE_ONNX="${OUT_DIR}/cpu_generator_source.onnx"
GENERATOR_ONNX="${OUT_DIR}/generator_rest_preexp.onnx"
ISTFT_ONNX="${OUT_DIR}/cpu_postspec_istft.onnx"

if [[ ! -f "${FULL_STFT_CONV}" ]]; then
  "${PYTHON}" scripts/rewrite_onnx_stft_to_conv.py \
    --input "${MODEL_DIR}/model.onnx" \
    --output "${FULL_STFT_CONV}" \
    --check
fi

if [[ ! -f "${PREFIX_ONNX}" ]]; then
  "${PYTHON}" scripts/split_kokoro_hybrid.py \
    --model "${MODEL_DIR}/model.onnx" \
    --out-dir "${OUT_DIR}" \
    --cut-output "/encoder/Cast_2_output_0"
fi

if [[ ! -f "${LENGTH_ONNX}" ]]; then
  "${PYTHON}" scripts/extract_onnx_subgraph.py \
    --model "${FULL_STFT_CONV}" \
    --output-model "${LENGTH_ONNX}" \
    --input "/encoder/Cast_2_output_0" \
    --input "/encoder/CumSum_output_0" \
    --input "/encoder/Slice_output_0" \
    --input "/encoder/predictor/text_encoder/Concat_4_output_0" \
    --input "tokens" \
    --output "/encoder/MatMul_1_output_0" \
    --output "/decoder/decoder/F0_conv/Conv_output_0" \
    --output "/decoder/decoder/N_conv/Conv_output_0" \
    --output "/decoder/decoder/Unsqueeze_output_0"
fi

if [[ ! -f "${DECODER_ONNX}" ]]; then
  "${PYTHON}" scripts/extract_onnx_subgraph.py \
    --model "${FULL_STFT_CONV}" \
    --output-model "${DECODER_ONNX}" \
    --input "/encoder/MatMul_1_output_0" \
    --input "/decoder/decoder/F0_conv/Conv_output_0" \
    --input "/decoder/decoder/N_conv/Conv_output_0" \
    --input "/decoder/decoder/Unsqueeze_output_0" \
    --input "style" \
    --output "/decoder/decoder/decode.3/Div_4_output_0"
fi

if [[ ! -f "${SOURCE_ONNX}" ]]; then
  "${PYTHON}" scripts/extract_onnx_subgraph.py \
    --model "${FULL_STFT_CONV}" \
    --output-model "${SOURCE_ONNX}" \
    --input "/decoder/decoder/Unsqueeze_output_0" \
    --output "/decoder/decoder/generator/Concat_3_output_0"
fi

if [[ ! -f "${GENERATOR_ONNX}" ]]; then
  "${PYTHON}" scripts/extract_onnx_subgraph.py \
    --model "${FULL_STFT_CONV}" \
    --output-model "${GENERATOR_ONNX}" \
    --input "/decoder/decoder/decode.3/Div_4_output_0" \
    --input "/decoder/decoder/generator/Concat_3_output_0" \
    --input "style" \
    --output "/decoder/decoder/generator/Slice_1_output_0" \
    --output "/decoder/decoder/generator/Slice_2_output_0"
fi

if [[ ! -f "${ISTFT_ONNX}" ]]; then
  "${PYTHON}" scripts/extract_onnx_subgraph.py \
    --model "${FULL_STFT_CONV}" \
    --output-model "${ISTFT_ONNX}" \
    --input "/decoder/decoder/generator/Slice_1_output_0" \
    --input "/decoder/decoder/generator/Slice_2_output_0" \
    --output "audio"
fi

case "${ENGINE_NAME}" in
  kokoro_prefix_encoder_dyn4_128_fp16.engine)
    "${TRTEXEC}" \
      --onnx="${PREFIX_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --fp16 \
      --minShapes=tokens:1x"${MIN_SEQ}",style:1x256,speed:1 \
      --optShapes=tokens:1x"${OPT_SEQ}",style:1x256,speed:1 \
      --maxShapes=tokens:1x"${MAX_SEQ}",style:1x256,speed:1 \
      --memPoolSize=workspace:"${WS}"
    ;;
  kokoro_decoder_backbone_dyn64_256_fp16.engine)
    "${TRTEXEC}" \
      --onnx="${DECODER_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --fp16 \
      --builderOptimizationLevel=0 \
      --memPoolSize=workspace:"${WS}" \
      --minShapes=/encoder/MatMul_1_output_0:1x512x"${MIN_T}",/decoder/decoder/F0_conv/Conv_output_0:1x1x"${MIN_T}",/decoder/decoder/N_conv/Conv_output_0:1x1x"${MIN_T}",/decoder/decoder/Unsqueeze_output_0:1x1x"${MIN_2T}",style:1x256 \
      --optShapes=/encoder/MatMul_1_output_0:1x512x"${OPT_T}",/decoder/decoder/F0_conv/Conv_output_0:1x1x"${OPT_T}",/decoder/decoder/N_conv/Conv_output_0:1x1x"${OPT_T}",/decoder/decoder/Unsqueeze_output_0:1x1x"${OPT_2T}",style:1x256 \
      --maxShapes=/encoder/MatMul_1_output_0:1x512x"${MAX_T}",/decoder/decoder/F0_conv/Conv_output_0:1x1x"${MAX_T}",/decoder/decoder/N_conv/Conv_output_0:1x1x"${MAX_T}",/decoder/decoder/Unsqueeze_output_0:1x1x"${MAX_2T}",style:1x256
    ;;
  kokoro_decoder_backbone_dyn256_512_fp16.engine)
    MIN_T="${LONG_MIN_T:-256}" OPT_T="${LONG_OPT_T:-384}" MAX_T="${LONG_MAX_T:-512}"
    MIN_2T=$((MIN_T * 2)); OPT_2T=$((OPT_T * 2)); MAX_2T=$((MAX_T * 2))
    "${TRTEXEC}" \
      --onnx="${DECODER_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --fp16 \
      --builderOptimizationLevel=0 \
      --memPoolSize=workspace:"${WS}" \
      --minShapes=/encoder/MatMul_1_output_0:1x512x"${MIN_T}",/decoder/decoder/F0_conv/Conv_output_0:1x1x"${MIN_T}",/decoder/decoder/N_conv/Conv_output_0:1x1x"${MIN_T}",/decoder/decoder/Unsqueeze_output_0:1x1x"${MIN_2T}",style:1x256 \
      --optShapes=/encoder/MatMul_1_output_0:1x512x"${OPT_T}",/decoder/decoder/F0_conv/Conv_output_0:1x1x"${OPT_T}",/decoder/decoder/N_conv/Conv_output_0:1x1x"${OPT_T}",/decoder/decoder/Unsqueeze_output_0:1x1x"${OPT_2T}",style:1x256 \
      --maxShapes=/encoder/MatMul_1_output_0:1x512x"${MAX_T}",/decoder/decoder/F0_conv/Conv_output_0:1x1x"${MAX_T}",/decoder/decoder/N_conv/Conv_output_0:1x1x"${MAX_T}",/decoder/decoder/Unsqueeze_output_0:1x1x"${MAX_2T}",style:1x256
    ;;
  kokoro_generator_source_dyn128_512_bf16.engine)
    "${TRTEXEC}" \
      --onnx="${SOURCE_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --bf16 \
      --builderOptimizationLevel=0 \
      --memPoolSize=workspace:"${WS}" \
      --minShapes=/decoder/decoder/Unsqueeze_output_0:1x1x"${MIN_2T}" \
      --optShapes=/decoder/decoder/Unsqueeze_output_0:1x1x"${OPT_2T}" \
      --maxShapes=/decoder/decoder/Unsqueeze_output_0:1x1x"${MAX_2T}"
    ;;
  kokoro_generator_source_dyn512_1024_bf16.engine)
    MIN_T="${LONG_MIN_T:-256}" OPT_T="${LONG_OPT_T:-384}" MAX_T="${LONG_MAX_T:-512}"
    MIN_2T=$((MIN_T * 2)); OPT_2T=$((OPT_T * 2)); MAX_2T=$((MAX_T * 2))
    "${TRTEXEC}" \
      --onnx="${SOURCE_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --bf16 \
      --builderOptimizationLevel=0 \
      --memPoolSize=workspace:"${WS}" \
      --minShapes=/decoder/decoder/Unsqueeze_output_0:1x1x"${MIN_2T}" \
      --optShapes=/decoder/decoder/Unsqueeze_output_0:1x1x"${OPT_2T}" \
      --maxShapes=/decoder/decoder/Unsqueeze_output_0:1x1x"${MAX_2T}"
    ;;
  kokoro_generator_rest_preexp_dyn64_256_fp16.engine)
    "${TRTEXEC}" \
      --onnx="${GENERATOR_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --fp16 \
      --builderOptimizationLevel=0 \
      --memPoolSize=workspace:"${WS}" \
      --minShapes=/decoder/decoder/decode.3/Div_4_output_0:1x512x"${MIN_2T}",/decoder/decoder/generator/Concat_3_output_0:1x22x"${MIN_SRC}",style:1x256 \
      --optShapes=/decoder/decoder/decode.3/Div_4_output_0:1x512x"${OPT_2T}",/decoder/decoder/generator/Concat_3_output_0:1x22x"${OPT_SRC}",style:1x256 \
      --maxShapes=/decoder/decoder/decode.3/Div_4_output_0:1x512x"${MAX_2T}",/decoder/decoder/generator/Concat_3_output_0:1x22x"${MAX_SRC}",style:1x256
    ;;
  kokoro_generator_rest_preexp_dyn256_512_fp16.engine)
    MIN_T="${LONG_MIN_T:-256}" OPT_T="${LONG_OPT_T:-384}" MAX_T="${LONG_MAX_T:-512}"
    MIN_2T=$((MIN_T * 2)); OPT_2T=$((OPT_T * 2)); MAX_2T=$((MAX_T * 2))
    MIN_SRC=$((MIN_T * 120 + 1)); OPT_SRC=$((OPT_T * 120 + 1)); MAX_SRC=$((MAX_T * 120 + 1))
    "${TRTEXEC}" \
      --onnx="${GENERATOR_ONNX}" \
      --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
      --fp16 \
      --builderOptimizationLevel=0 \
      --memPoolSize=workspace:"${WS}" \
      --minShapes=/decoder/decoder/decode.3/Div_4_output_0:1x512x"${MIN_2T}",/decoder/decoder/generator/Concat_3_output_0:1x22x"${MIN_SRC}",style:1x256 \
      --optShapes=/decoder/decoder/decode.3/Div_4_output_0:1x512x"${OPT_2T}",/decoder/decoder/generator/Concat_3_output_0:1x22x"${OPT_SRC}",style:1x256 \
      --maxShapes=/decoder/decoder/decode.3/Div_4_output_0:1x512x"${MAX_2T}",/decoder/decoder/generator/Concat_3_output_0:1x22x"${MAX_SRC}",style:1x256
    ;;
  *)
    echo "Unsupported Kokoro split engine name: ${ENGINE_NAME}" >&2
    exit 1
    ;;
esac

ls -lh "${OUT_DIR}/${ENGINE_NAME}"
