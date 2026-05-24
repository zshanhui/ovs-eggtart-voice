#!/bin/sh
# Multi-device parity gate one-shot entry point.
#
# Thin wrapper around bench/parity/run_parity.py. POSIX sh compatible
# (works under macOS zsh + Linux bash 3.2+ without bash-4 features).
#
# Usage:
#   scripts/parity_gate.sh --mode mock
#   scripts/parity_gate.sh --mode remote --devices jetson --fail-on-budget
#   scripts/parity_gate.sh --mode remote --devices all \
#       --fail-on-budget --fail-on-cer-regression
#
# Manual-trigger only. Not wired into per-commit CI.

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/bench/parity/run_parity.py"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found" >&2
    exit 3
fi

PYTHON="${PYTHON:-python3}"

# Default mode: mock (safe everywhere). Remote requires explicit --mode remote.
MODE="mock"
ARGS=""
for arg in "$@"; do
    case "$arg" in
        --mode)
            ;;  # handled by next iteration; just pass through
    esac
    ARGS="$ARGS $arg"
done

# If --mode remote, pre-flight check that fleet CLI exists.
case " $* " in
    *" --mode remote "*|*" --mode=remote "*)
        MODE="remote"
        if ! command -v fleet >/dev/null 2>&1; then
            echo "ERROR: --mode remote requires the fleet CLI (see ~/.claude/CLAUDE.md fleet section)" >&2
            exit 3
        fi
        echo "[parity_gate] remote mode: fleet target list:" >&2
        # Print devices but do not fail if fleet doesn't support --json yet.
        fleet list 2>/dev/null || true
        ;;
esac

echo "[parity_gate] mode=$MODE python=$PYTHON" >&2
echo "[parity_gate] args:$ARGS" >&2

# shellcheck disable=SC2086
exec "$PYTHON" "$SCRIPT" $ARGS
