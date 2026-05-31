"""Abstract LLM backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal


@dataclass
class LLMEvent:
    """One unit of streaming output from an LLM backend.

    A single upstream chunk may produce one or more ``LLMEvent``s:
      * ``kind="text"`` — incremental assistant text (``text`` set).
      * ``kind="tool_call_delta"`` — partial tool_call info; the runner
        accumulates fragments per ``tool_call_index``.
      * ``kind="finish"`` — terminal marker; ``finish_reason`` carries
        OpenAI's value ("stop" / "tool_calls" / "length" / "error").
    """

    kind: Literal["text", "tool_call_delta", "finish"]
    # text fields
    text: str | None = None
    # tool_call_delta fields (per OpenAI index-based accumulation)
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # finish field
    finish_reason: str | None = None


class LLMBackend(ABC):
    """Streaming LLM backend.

    Implementations expose two channels:

    * ``stream_events`` (preferred) — yields :class:`LLMEvent` instances
      covering text, tool-call deltas, and a finish marker. This is the
      channel the tool-calling runner consumes.
    * ``stream`` (back-compat) — yields plain text strings. Default
      implementation filters ``stream_events`` to text deltas only so
      existing callers don't have to change.

    Lifecycle hook contract:

    * ``warmup(**kwargs)`` — optional pre-flight call from app startup.
      Default is a no-op that returns ``{}`` and never raises. Backends
      override this to pay engine cold-start costs (KV prefix cache,
      CUDA-graph capture, etc.) before the first user turn. Always
      treated as fire-and-forget: the dict return value is opportunistic
      metadata (``cache_warmed``, ``graph_warmed``, ``prompt_chars``,
      ``cache_warmup_ms``, ``graph_warmup_ms`` for EdgeLLMBackend; may
      include ``engine_max_seq_len`` when the server exposes ``/v1/info``),
      never required for correctness.
    * ``aclose()`` — optional resource release on shutdown. Default
      no-op. Backends that hold network/transport handles
      (httpx.AsyncClient, websocket) override this.

    Both hooks are safe to call multiple times (idempotent) and on
    backends that don't override them.
    """

    async def stream_events(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        """Yield :class:`LLMEvent`s from the model.

        Default implementation delegates to :meth:`stream` (text-only
        legacy channel) so subclasses that only override ``stream`` keep
        working — their text is wrapped in ``LLMEvent(kind="text")`` plus
        a synthetic ``finish`` event at end-of-stream.

        Backends that produce real tool-call deltas (e.g.
        :class:`OpenAICompatBackend`) override this method directly.
        """
        async for tok in self.stream(messages, **kw):
            if tok:
                yield LLMEvent(kind="text", text=tok)
        yield LLMEvent(kind="finish", finish_reason="stop")

    @abstractmethod
    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        """Yield text deltas (already-decoded strings). Subclasses may
        instead override :meth:`stream_events` for richer output and
        leave ``stream`` as the default text filter."""
        # Concrete implementations must be async generators -- this stub
        # exists so the base class can be ABC-instantiated for typing.
        if False:  # pragma: no cover
            yield ""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any held network/transport resources. Default: no-op."""
        return None

    async def warmup(
        self,
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        timeout_s: float | None = 60.0,
    ) -> dict[str, Any]:
        """Optional pre-flight warmup.

        Default no-op: returns ``{}`` and never raises. Backends with
        engine-specific warmup paths (e.g. :class:`EdgeLLMBackend` warms
        the prefix KV cache *and* runs one real-shape forward pass to
        capture the TRT-LLM CUDA graph) override this to do the work.

        Callers should treat this as fire-and-forget: any non-empty
        return value is opportunistic metadata for the dashboard /
        session, never required for correctness.
        """
        return {}
