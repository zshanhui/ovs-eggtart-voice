"""Qwen3 chat template (manual — no ``transformers`` dependency).

The RK container does not ship ``transformers`` (~100 MB).  We implement the
Qwen3 template inline.  Reference:
https://huggingface.co/Qwen/Qwen3-0.6B-Instruct
"""

from __future__ import annotations

# Qwen3 uses ChatML-style tokens (same as Qwen2.5).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM_PROMPT = "You are a helpful assistant."


def apply_chat_template(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    system_prompt: str | None = None,
) -> str:
    """Convert OpenAI-format *messages* to a Qwen3 prompt string.

    Example output::

        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
    """
    parts: list[str] = []

    # System message always comes first.  If the caller didn't provide one
    # we insert the default.
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        parts.append(f"{IM_START}system\n{system_prompt or SYSTEM_PROMPT}{IM_END}\n")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if has_system and parts:
                # Already handled above; skip duplicate.
                continue
            role = "system"
        elif role == "user":
            role = "user"
        elif role == "assistant":
            role = "assistant"
        else:
            # Unknown roles → treat as user.
            role = "user"
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")

    return "".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count (4 chars ≈ 1 token for CJK, 4:1 for English).

    Used for pre-flight rejection when input clearly exceeds the model
    context window.  The RKLLM runtime handles exact tokenisation internally
    via its embedded tokenizer.
    """
    if not text:
        return 0
    # Crude heuristic compatible with Qwen's BPE tokenizer.
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿")
    other = len(text) - cjk
    # CJK characters are ~1 token each; other chars ~0.3 tokens each.
    return int(cjk + other * 0.3)
