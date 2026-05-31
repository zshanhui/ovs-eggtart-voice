#!/usr/bin/env bash
# verify_customvoice_tts_radxa_asr.sh
# One-shot CustomVoice TTS (v0.7.1) smoke + radxa SenseVoice ASR round-trip.
# Background: see memory `customvoice_tts_w8a16_PASS_2026_05_26.md`
#             and `customvoice_tts_fork_port_PASS_2026_05_26.md`.
# This script DOES NOT touch the production worker / Python wrapper path —
# it drives the C++ inference binary against the snapshot (or image) layout.

set -euo pipefail

# --- defaults -----------------------------------------------------------------
VARIANT="fp16"
TEXT="今天天气真不错"
SPEAKER="vivian"
LANGUAGE="chinese"
SOURCE="snapshot"
ORIN_NX_HOST="orin-nx"
RADXA_HOST="radxa"
SEED="42"
OUTPUT_DIR="/tmp"
EXPECTED_TEXT=""
SIMILARITY_THRESHOLD="0.9"  # accepted but only used to gate substring check
IMAGE_TAG="openvoicestream:jetson-customvoice-v071"

FLEET="${FLEET:-uv run --project ${HOME}/project/_hub python ${HOME}/project/_hub/fleet.py}"

# --- arg parse ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)              VARIANT="$2"; shift 2 ;;
    --text)                 TEXT="$2"; shift 2 ;;
    --speaker)              SPEAKER="$2"; shift 2 ;;
    --language)             LANGUAGE="$2"; shift 2 ;;
    --source)               SOURCE="$2"; shift 2 ;;
    --orin-nx-host)         ORIN_NX_HOST="$2"; shift 2 ;;
    --radxa-host)           RADXA_HOST="$2"; shift 2 ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
    --expected-text)        EXPECTED_TEXT="$2"; shift 2 ;;
    --similarity-threshold) SIMILARITY_THRESHOLD="$2"; shift 2 ;;
    --image-tag)            IMAGE_TAG="$2"; shift 2 ;;
    -h|--help)              sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ "$VARIANT" == "fp16" || "$VARIANT" == "w8a16" ]] || { echo "FAIL: --variant must be fp16|w8a16" >&2; exit 2; }
[[ "$SOURCE" == "snapshot" || "$SOURCE" == "image" ]] || { echo "FAIL: --source must be snapshot|image" >&2; exit 2; }
[[ -z "$EXPECTED_TEXT" ]] && EXPECTED_TEXT="$TEXT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_JSON="${OUTPUT_DIR}/verify_result_${VARIANT}_${TIMESTAMP}.json"
START_TS=$(date +%s)

emit_result() {
  local status="$1" wav_md5="${2:-}" wav_dur="${3:-0}" asr_text="${4:-}" passed="${5:-false}"
  local elapsed=$(( $(date +%s) - START_TS ))
  cat > "$RESULT_JSON" <<EOF
{
  "status": "$status",
  "variant": "$VARIANT",
  "source": "$SOURCE",
  "text": "$TEXT",
  "expected_text": "$EXPECTED_TEXT",
  "speaker": "$SPEAKER",
  "language": "$LANGUAGE",
  "seed": $SEED,
  "wav_md5": "$wav_md5",
  "wav_duration_s": $wav_dur,
  "asr_text": "$asr_text",
  "similarity_passed": $passed,
  "similarity_threshold": $SIMILARITY_THRESHOLD,
  "elapsed_s": $elapsed,
  "timestamp": "$TIMESTAMP"
}
EOF
  echo "--- result JSON: $RESULT_JSON ---"
  cat "$RESULT_JSON"
}

die_infra() { echo "[INFRA_ERROR] $*" >&2; emit_result INFRA_ERROR; exit 2; }

# --- 1. fleet connectivity ----------------------------------------------------
echo "[1/7] Checking fleet connectivity..."
$FLEET status "$ORIN_NX_HOST" --json 2>/dev/null | grep -q '"online": true' || die_infra "$ORIN_NX_HOST offline"
$FLEET status "$RADXA_HOST"   --json 2>/dev/null | grep -q '"online": true' || die_infra "$RADXA_HOST offline"

# --- 2. setup symlinks + input json on orin-nx --------------------------------
echo "[2/7] Preparing /tmp/v071_run on $ORIN_NX_HOST (variant=$VARIANT)..."
ENG_BASE="\$HOME/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx"
TALKER_DIR="$ENG_BASE/$([[ "$VARIANT" == "fp16" ]] && echo talker || echo talker-w8a16)"
INPUT_JSON_PATH="/tmp/v071_run/input_${VARIANT}.json"
OUT_DIR_REMOTE="/tmp/v071_run/out_${VARIANT}_run"

$FLEET exec "$ORIN_NX_HOST" -- "
set -e
mkdir -p /tmp/v071_run $OUT_DIR_REMOTE
ln -sfn $TALKER_DIR /tmp/v071_run/talker
ln -sfn $ENG_BASE/code_predictor /tmp/v071_run/code_predictor
ln -sfn $ENG_BASE/code2wav       /tmp/v071_run/code2wav
cat > $INPUT_JSON_PATH <<'JSON'
{
  \"speaker\": \"$SPEAKER\",
  \"language\": \"$LANGUAGE\",
  \"apply_chat_template\": true,
  \"add_generation_prompt\": true,
  \"enable_thinking\": false,
  \"max_audio_length\": 24000,
  \"requests\": [
    {\"messages\": [{\"role\": \"user\", \"content\": \"$TEXT\"}]}
  ]
}
JSON
" >/dev/null || die_infra "failed to stage input on $ORIN_NX_HOST"

# --- 3. run TTS ---------------------------------------------------------------
echo "[3/7] Running CustomVoice TTS binary (source=$SOURCE)..."
WAV_REMOTE="$OUT_DIR_REMOTE/audio_req0.wav"

if [[ "$SOURCE" == "snapshot" ]]; then
  RUN_CMD="cd /tmp/v071_run && \
    EDGELLM_PLUGIN_PATH=\$HOME/customvoice-v071-snapshot/20260526/libNvInfer_edgellm_plugin.so.1.0 \
    QWEN3_TTS_PRELOAD_TALKER_EMBEDS=\$HOME/customvoice-v071-snapshot/20260526/ref_talker_embeds_15row.bin \
    QWEN3_TTS_SEED=$SEED \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:\${LD_LIBRARY_PATH:-} \
    \$HOME/customvoice-v071-snapshot/20260526/qwen3_tts_inference \
      --inputFile=$INPUT_JSON_PATH \
      --talkerEngineDir=/tmp/v071_run/talker \
      --code2wavEngineDir=/tmp/v071_run/code2wav \
      --tokenizerDir=/tmp/v071_run/talker \
      --outputAudioDir=$OUT_DIR_REMOTE \
      --outputFile=$OUT_DIR_REMOTE/result.json"
else
  RUN_CMD="docker run --rm --runtime=nvidia \
      -v /tmp/v071_run:/tmp/v071_run \
      -e QWEN3_TTS_SEED=$SEED \
      $IMAGE_TAG \
      /opt/customvoice/qwen3_tts_inference \
        --inputFile=$INPUT_JSON_PATH \
        --talkerEngineDir=/tmp/v071_run/talker \
        --code2wavEngineDir=/tmp/v071_run/code2wav \
        --tokenizerDir=/tmp/v071_run/talker \
        --outputAudioDir=$OUT_DIR_REMOTE \
        --outputFile=$OUT_DIR_REMOTE/result.json"
fi

$FLEET exec --timeout 300 "$ORIN_NX_HOST" -- "$RUN_CMD" >/tmp/cv_tts_run_"${TIMESTAMP}".log 2>&1 \
  || { tail -40 /tmp/cv_tts_run_"${TIMESTAMP}".log >&2; die_infra "TTS binary failed (log: /tmp/cv_tts_run_${TIMESTAMP}.log)"; }

# --- 4. verify WAV ------------------------------------------------------------
echo "[4/7] Verifying WAV on $ORIN_NX_HOST..."
WAV_INFO="$($FLEET exec "$ORIN_NX_HOST" -- "stat -c %s $WAV_REMOTE 2>/dev/null && md5sum $WAV_REMOTE 2>/dev/null" 2>/dev/null || true)"
WAV_SIZE="$(echo "$WAV_INFO" | sed -n '1p')"
WAV_MD5="$(echo  "$WAV_INFO" | sed -n '2p' | awk '{print $1}')"
[[ -n "$WAV_SIZE" && "$WAV_SIZE" -gt 1024 ]] || { emit_result FAIL "$WAV_MD5"; echo "WAV too small ($WAV_SIZE B)" >&2; exit 1; }
WAV_DUR=$(awk -v b="$WAV_SIZE" 'BEGIN{printf "%.3f", (b-44)/(24000*2)}')
echo "  wav_size=$WAV_SIZE  wav_md5=$WAV_MD5  duration~${WAV_DUR}s"

# --- 5. transfer to radxa -----------------------------------------------------
echo "[5/7] Transferring WAV $ORIN_NX_HOST -> $RADXA_HOST..."
RADXA_WAV="/tmp/cv_${VARIANT}_${TIMESTAMP}.wav"
$FLEET transfer "${ORIN_NX_HOST}:${WAV_REMOTE/\$HOME/\/home\/harvest}" "${RADXA_HOST}:${RADXA_WAV}" >/dev/null 2>&1 \
  || $FLEET transfer "${ORIN_NX_HOST}:${WAV_REMOTE}" "${RADXA_HOST}:${RADXA_WAV}" >/dev/null 2>&1 \
  || die_infra "fleet transfer failed"

# --- 6. radxa ASR -------------------------------------------------------------
echo "[6/7] Running radxa SenseVoice ASR..."
ASR_RAW="$($FLEET exec --timeout 120 "$RADXA_HOST" -- "python3 - <<'PY'
import sherpa_onnx, wave, json, os
home = os.path.expanduser('~')
mdir = os.path.join(home, 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17')
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=os.path.join(mdir, 'model.int8.onnx'),
    tokens=os.path.join(mdir, 'tokens.txt'),
    language='zh', use_itn=True, num_threads=2)
s = rec.create_stream()
with wave.open('$RADXA_WAV','rb') as wf:
    sr = wf.getframerate()
    import numpy as np
    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype('float32')/32768.0
s.accept_waveform(sr, pcm)
rec.decode_stream(s)
print(json.dumps({'text': s.result.text}, ensure_ascii=False))
PY
" 2>&1 | tail -1)"
ASR_TEXT="$(printf '%s' "$ASR_RAW" | python3 -c "import sys,json
try: print(json.loads(sys.stdin.read())['text'])
except Exception: pass" 2>/dev/null || true)"
[[ -z "$ASR_TEXT" ]] && { echo "ASR raw: $ASR_RAW" >&2; emit_result FAIL "$WAV_MD5" "$WAV_DUR" ""; exit 1; }
echo "  asr_text=\"$ASR_TEXT\""

# --- 7. similarity check ------------------------------------------------------
echo "[7/7] Comparing ASR vs expected..."
norm() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:][:space:]。，！？、'; }
ASR_NORM="$(norm "$ASR_TEXT")"
EXP_NORM="$(norm "$EXPECTED_TEXT")"
PASSED="false"
if [[ -n "$EXP_NORM" && "$ASR_NORM" == *"$EXP_NORM"* ]]; then PASSED="true"; fi

if [[ "$PASSED" == "true" ]]; then
  echo "[PASS] variant=$VARIANT  wav_md5=$WAV_MD5  asr=\"$ASR_TEXT\""
  emit_result PASS "$WAV_MD5" "$WAV_DUR" "$ASR_TEXT" true
  cp -f "$RESULT_JSON" "$RESULT_JSON" # noop
  exit 0
else
  echo "[FAIL] expected substring \"$EXPECTED_TEXT\" not in ASR \"$ASR_TEXT\""
  emit_result FAIL "$WAV_MD5" "$WAV_DUR" "$ASR_TEXT" false
  exit 1
fi
