"""No-op LLM backend (for translator-only apps)."""
from __future__ import annotations

from typing import Any, AsyncIterator

from .base import LLMBackend


class NoopLLM(LLMBackend):
    """LLM backend that yields nothing (used when LLM is not needed)."""

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        # Yield nothing; caller will see empty iterator.
        if False:  # pragma: no cover
            yield ""
