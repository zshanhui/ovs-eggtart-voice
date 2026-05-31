#!/bin/bash
# -*- shell-script -*-
#
# run_tests.sh — run all locally-testable suites for OpenVoiceStream.
#
# Usage:
#   chmod +x run_tests.sh
#   ./run_tests.sh
#
# Requires: Python 3.10+ (3.11–3.13 recommended; 3.14 has known asyncio
# incompatibilities in a handful of tests).
#
# The script creates a throwaway venv at /tmp/ovs-test-env if one doesn't
# already exist, installs dependencies, and runs every test suite that
# does NOT require NPU hardware, audio devices, or external services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${OVS_TEST_VENV:-/tmp/ovs-test-env}"
PASSED=0
FAILED=0
SKIPPED=0

# ── Colour helpers ───────────────────────────────────────────────────────────
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

banner() {
    echo ""
    bold "══════════════════════════════════════════════════════════════"
    bold "  $1"
    bold "══════════════════════════════════════════════════════════════"
}

# ── Create / refresh venv ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "Installing/updating dependencies ..."
pip install -q --upgrade pip 2>/dev/null || true
pip install -q \
  fastapi uvicorn pydantic \
  pytest pytest-asyncio \
  pyyaml numpy soundfile \
  openai aiohttp websockets \
  requests httpx \
  2>/dev/null

# Suppress Starlette deprecation warnings (httpx → httpx2)
export PYTHONWARNINGS="ignore::DeprecationWarning"

# ── Helper ───────────────────────────────────────────────────────────────────
run_suite() {
    local label="$1"
    shift
    echo ""
    yellow "[${label}]"
    local exit_code=0
    python3 "$@" 2>&1 || exit_code=$?
    if [ $exit_code -eq 0 ]; then
        PASSED=$((PASSED + 1))
        green "  [PASS] ${label}"
    else
        FAILED=$((FAILED + 1))
        red "  [FAIL] ${label}"
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
banner "LLM Server & Pipeline Tests"
# ══════════════════════════════════════════════════════════════════════════════

run_suite "LLM server smoke"   services/llm/test_server.py
run_suite "LLM E2E pipeline"   services/llm/test_e2e.py

# ══════════════════════════════════════════════════════════════════════════════
banner "Chat Template Unit Tests"
# ══════════════════════════════════════════════════════════════════════════════

run_suite "Chat template" -c "
from services.llm.chat_template import apply_chat_template, estimate_tokens, IM_START, IM_END

# Basic formatting
r = apply_chat_template([{'role': 'user', 'content': 'Hello'}])
assert f'{IM_START}assistant\n' in r;              print('  [OK] basic formatting')

# Custom system prompt
r = apply_chat_template([{'role': 'user', 'content': 'X'}], system_prompt='Be concise.')
assert 'Be concise.' in r;                          print('  [OK] custom system prompt')

# Multi-turn
r = apply_chat_template([{'role':'user','content':'Q1'},{'role':'assistant','content':'A1'},{'role':'user','content':'Q2'}])
assert r.count(IM_START + 'user') == 2;              print('  [OK] multi-turn')

# No generation prompt
r = apply_chat_template([{'role':'user','content':'X'}], add_generation_prompt=False)
assert not r.endswith(IM_START + 'assistant\n');     print('  [OK] no generation prompt')

# Token estimation
assert estimate_tokens('') == 0;                     print('  [OK] token est: empty')
assert estimate_tokens('hello') == 1;                print('  [OK] token est: en')
assert estimate_tokens('你好世界') == 4;              print('  [OK] token est: zh')
assert estimate_tokens('hello你好') == 3;             print('  [OK] token est: mixed')

# zh system prompt (from rk3588-chat config)
sys = '你是一个运行在瑞芯微芯片上的语音助手。请用简洁自然的中文回答，控制在三句话以内，就像和朋友聊天一样。'
r = apply_chat_template([{'role':'user','content':'推荐一道菜'}], system_prompt=sys)
assert sys in r;                                     print('  [OK] zh system prompt')

# Empty messages → default system inserted
r = apply_chat_template([])
assert IM_START + 'system' in r;                     print('  [OK] empty messages')

# Existing system message not duplicated
r = apply_chat_template([{'role':'system','content':'You are a chef.'},{'role':'user','content':'Recipe?'}])
assert 'You are a chef.' in r
assert 'helpful assistant' not in r;                 print('  [OK] existing system message')

# Unknown role → user
r = apply_chat_template([{'role':'unknown','content':'X'}])
assert IM_START + 'user' in r;                       print('  [OK] unknown role → user')

# Chef KDS system prompt
sys = '你是一套智能厨房显示系统（KDS），帮助厨师和后厨团队管理订单、安排出菜顺序、监控库存和协调备餐流程。'
r = apply_chat_template([{'role':'user','content':'现在有哪些待做的单子？'}], system_prompt=sys)
assert sys in r;                                     print('  [OK] chef KDS system prompt')

# Mixed zh/en input
r = apply_chat_template([{'role':'user','content':'I want kung pao chicken 不要辣'}])
assert 'kung pao chicken' in r and '不要辣' in r;    print('  [OK] mixed zh/en')

print()
print('All 11 chat template tests passed.')
"

# ══════════════════════════════════════════════════════════════════════════════
banner "Agent Tests"
# ══════════════════════════════════════════════════════════════════════════════

# Skip tests that need hardware: e2e (needs audio/network), test_e2e_orin (needs Jetson)
# The Python 3.14 event-loop test is skipped automatically
run_suite "Agent unit tests" -m pytest agent/tests/ \
  --ignore=agent/tests/e2e \
  --ignore=agent/tests/test_e2e_orin.py \
  -q --tb=short 2>&1

# ══════════════════════════════════════════════════════════════════════════════
banner "App Tests"
# ══════════════════════════════════════════════════════════════════════════════

# Skip tests that need NPU hardware or the rkvoice-stream submodule
run_suite "App unit tests" -m pytest app/tests/ \
  --ignore=app/tests/test_v2v_vad_event.py \
  --ignore=app/tests/test_v2v_eos_multi_utterance.py \
  --ignore=app/tests/test_main_hot_swap.py \
  --ignore=app/tests/test_asr_stream_protocol.py \
  --ignore=app/tests/test_session_limiter.py \
  -q --tb=short 2>&1

# ══════════════════════════════════════════════════════════════════════════════
banner "Agent Config Loading"
# ══════════════════════════════════════════════════════════════════════════════

run_suite "RK3576 config loads" -c "
from openvoicestream_agent.config import load_config
cfg = load_config('$(pwd)/agent/apps/rk3576-chat/config.yaml')
assert cfg.llm_backend == 'openai_compat'
assert cfg.llm_model == 'qwen3-0.6b-instruct'
print('  RK3576 config OK  — llm_model={}, backend={}'.format(cfg.llm_model, cfg.llm_backend))
"

run_suite "RK3588 config loads" -c "
from openvoicestream_agent.config import load_config
cfg = load_config('$(pwd)/agent/apps/rk3588-chat/config.yaml')
assert cfg.llm_backend == 'openai_compat'
assert cfg.llm_model == 'qwen3-0.6b'
print('  RK3588 config OK  — llm_model={}, backend={}'.format(cfg.llm_model, cfg.llm_backend))
"

# ══════════════════════════════════════════════════════════════════════════════
banner "Profile Selector (no RPi)"
# ══════════════════════════════════════════════════════════════════════════════

run_suite "Profile selector" -c "
from app.core.profile_selector import PRESET_TABLE
tiers = sorted(set(t for t,_ in PRESET_TABLE))
presets = sorted(set(p for _,p in PRESET_TABLE))
assert 'rpi4' not in tiers, 'RPi4 tier should be removed'
assert 'rpi5' not in tiers, 'RPi5 tier should be removed'
assert 'asr_zh_en' not in presets, 'asr_zh_en preset should be removed'
print(f'  Tiers:  {tiers}')
print(f'  Presets: {presets}')
print('  OK — no RPi tiers or presets')
"

# ══════════════════════════════════════════════════════════════════════════════
banner "Results"
# ══════════════════════════════════════════════════════════════════════════════

TOTAL=$((PASSED + FAILED))
echo ""
if [ $FAILED -eq 0 ]; then
    green "All ${TOTAL} suites passed."
else
    red "${FAILED}/${TOTAL} suites FAILED."
fi
