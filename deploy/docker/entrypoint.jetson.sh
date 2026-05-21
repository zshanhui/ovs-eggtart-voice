#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/openvoicestream.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/openvoicestream.env
  set +a
elif [[ -f /etc/seeed-local-voice.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/seeed-local-voice.env
  set +a
fi

if [[ -z "${OVS_PROFILE:-}" && -z "${OVS_PROFILE_JSON:-}" ]]; then
  if [[ -n "${OVS_PROFILE_DEFAULT:-}" ]]; then
    export OVS_PROFILE="${OVS_PROFILE_DEFAULT}"
  else
    case "${LANGUAGE_MODE:-zh_en}" in
      zh_en)
        export OVS_PROFILE="jetson-zh-en"
        ;;
      multilanguage)
        export OVS_PROFILE="jetson-qwen3asr-matcha-nx"
        ;;
    esac
  fi
fi

exec "$@"
