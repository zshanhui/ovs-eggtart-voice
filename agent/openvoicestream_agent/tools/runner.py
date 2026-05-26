"""Multi-turn LLM ↔ tool runner.

Drives the dialog: stream LLM events; if the model emits
``finish_reason="tool_calls"`` accumulate the deltas, execute each
tool, append ``role:tool`` results, and re-issue the LLM call. Repeat
until a non-tool finish or the iteration cap.

The runner mutates ``session.history`` AND the caller-supplied
``messages`` list in lock-step. On cancel or iteration-cap it rolls
both back to the pre-call anchor so the next user turn sees clean
state (no orphan ``assistant(tool_calls)`` without matching ``tool``
follow-up).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..session import Session
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCallCtx:
    """Per-turn context passed to tools. Tools that need access to app
    state declare ``ctx: ToolCallCtx`` (or just ``ctx``) in their
    signature; the registry injects this on dispatch."""

    session: Session
    mode_manager: Any = None
    event_bus: Any = None
    config: Any = None


@dataclass
class _ToolCallAcc:
    """Accumulator for one tool_call's streamed deltas (per OpenAI
    index slot)."""

    id: str = ""
    name: str = ""
    arguments: str = ""


# Callback signatures.
AssistantTokenCB = Callable[[str], Awaitable[None]]
ToolStartedCB = Callable[[dict[str, Any]], Awaitable[None]]
ToolCompletedCB = Callable[[dict[str, Any], dict[str, Any], float], Awaitable[None]]
# Fired right after on_tool_started, ONLY if the dispatched tool was
# registered with a non-empty ``preamble_text``. The string is the
# verbatim preamble (e.g. "好的。") — the app wires this to its TTS
# channel. Callers that don't care can leave it None.
ToolPreambleCB = Callable[[str], Awaitable[None]]


def _open_stream(llm: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]):
    """Return an async iterator of LLMEvent regardless of which streaming
    channel the backend exposes.

    Tests + a handful of legacy callers implement only ``stream`` (text
    deltas as ``str``). Wrap those on the fly so the runner only needs
    to know about ``LLMEvent`` shape.
    """
    if hasattr(llm, "stream_events"):
        return llm.stream_events(messages, **kwargs)
    # Lazy import keeps the module loadable when LLMEvent isn't needed.
    from ..llm.base import LLMEvent

    async def _wrap():
        async for tok in llm.stream(messages, **kwargs):
            if tok:
                yield LLMEvent(kind="text", text=tok)
        yield LLMEvent(kind="finish", finish_reason="stop")

    return _wrap()


async def stream_with_tools(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    session: Session,
    registry: ToolRegistry,
    allowed_tools: set[str] | None,
    ctx: ToolCallCtx,
    max_iterations: int = 5,
    on_assistant_token: AssistantTokenCB,
    on_tool_started: ToolStartedCB | None = None,
    on_tool_preamble: ToolPreambleCB | None = None,
    on_tool_completed: ToolCompletedCB | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    first_token_timeout_s: float | None = None,
    idle_timeout_s: float | None = None,
    on_timeout: Callable[[str, float, str], BaseException] | None = None,
) -> str:
    """Run LLM ↔ tool rounds until a text-only final answer.

    Returns the final assistant text (also appended to
    ``session.history``). Mutates both ``session.history`` and the
    caller's ``messages`` list in lock-step.

    On cancel: rolls back any messages added during this call so
    ``session.history`` stays strict-valid.
    """
    tools_schema = registry.list_openai_tools(allowed_tools) or None
    iterations_done = 0
    rollback_anchor = len(session.history)
    # ``messages`` typically looks like ``[system, *session.history]``.
    # When we mirror a rollback we need to truncate ``messages`` to the
    # same logical anchor: ``messages_offset`` is the count of
    # non-history prefix items (1 for system; 0 if absent).
    messages_offset = max(0, len(messages) - rollback_anchor)
    try:
        for iter_idx in range(max_iterations):
            iterations_done += 1
            text_chunks: list[str] = []
            tool_accs: dict[int, _ToolCallAcc] = {}
            finish_reason: str | None = None
            # Track which tool_call slots have already had their preamble
            # fired (early, on first name delta). Prevents the post-stream
            # dispatch-time fallback from re-firing the same preamble.
            preamble_fired: set[int] = set()

            kwargs: dict[str, Any] = dict(llm_kwargs or {})
            kwargs["session"] = session
            kwargs["tools"] = tools_schema
            # Must-fix #1 (codex review): on tool-loop iterations *after*
            # the first the message list shape has changed
            # (assistant_tool_calls + tool results appended). The
            # server-side formatted_request cache was keyed against the
            # original shape; force the no-prefix path for iter >0.
            if iter_idx > 0:
                caller_extra = dict(kwargs.get("extra_body") or {})
                caller_extra["prefix_cache"] = False
                kwargs["extra_body"] = caller_extra

            stream = _open_stream(llm, messages, kwargs)
            # Distinguish first-payload vs idle timeouts only when the
            # caller configured them. A "payload" is any event that
            # carries content (text or tool_call_delta) — finish-only
            # events don't reset the first-token gate (matches the spec
            # answer to Q3: any LLMEvent with payload counts as first).
            received_payload = False
            it = stream.__aiter__()
            while True:
                use_first = (
                    first_token_timeout_s is not None
                    and not received_payload
                )
                use_idle = (
                    idle_timeout_s is not None
                    and received_payload
                )
                try:
                    if use_first:
                        ev = await asyncio.wait_for(
                            it.__anext__(), timeout=first_token_timeout_s
                        )
                    elif use_idle:
                        ev = await asyncio.wait_for(
                            it.__anext__(), timeout=idle_timeout_s
                        )
                    else:
                        ev = await it.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    kind = "first_token" if not received_payload else "stream_idle"
                    t_used = (
                        float(first_token_timeout_s)
                        if not received_payload
                        else float(idle_timeout_s)
                    )
                    # Close the stream so the upstream backend sees a
                    # cancel (matches the legacy behaviour in app_mode).
                    aclose = getattr(stream, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:  # pragma: no cover
                            logger.debug(
                                "stream aclose during timeout failed",
                                exc_info=True,
                            )
                    if on_timeout is not None:
                        raise on_timeout(kind, t_used, "".join(text_chunks))
                    raise asyncio.TimeoutError(
                        f"LLM {kind} timeout after {t_used:.1f}s"
                    )
                if ev.kind == "text" and ev.text:
                    received_payload = True
                    text_chunks.append(ev.text)
                    await on_assistant_token(ev.text)
                elif ev.kind == "tool_call_delta":
                    received_payload = True
                    idx = (
                        ev.tool_call_index
                        if ev.tool_call_index is not None
                        else 0
                    )
                    slot = tool_accs.setdefault(idx, _ToolCallAcc())
                    if ev.tool_call_id:
                        slot.id = ev.tool_call_id
                    if ev.name:
                        slot.name = ev.name
                        # Early-fire on_tool_preamble as soon as we know
                        # the tool name, instead of waiting for the full
                        # arguments JSON + dispatch. This drops voice
                        # preamble latency from ~stream-end to ~one token
                        # after the model commits to a tool call.
                        if (
                            on_tool_preamble is not None
                            and idx not in preamble_fired
                        ):
                            tool_meta = registry._tools.get(ev.name)
                            pre_text = (
                                getattr(tool_meta, "preamble_text", "") or ""
                            )
                            if pre_text:
                                preamble_fired.add(idx)
                                logger.info(
                                    "tool preamble (early): text=%r tool=%s",
                                    pre_text,
                                    ev.name,
                                )
                                try:
                                    await on_tool_preamble(pre_text)
                                except Exception:  # noqa: BLE001
                                    logger.debug(
                                        "on_tool_preamble (early) raised",
                                        exc_info=True,
                                    )
                    if ev.arguments:
                        slot.arguments += ev.arguments
                elif ev.kind == "finish":
                    finish_reason = ev.finish_reason

            if not tool_accs or finish_reason != "tool_calls":
                final_text = "".join(text_chunks)
                if final_text:
                    session.add_assistant(final_text)
                    messages.append(
                        {"role": "assistant", "content": final_text}
                    )
                return final_text

            # Commit assistant(tool_calls) to messages + session.
            preamble = "".join(text_chunks) or None
            tc_payload: list[dict[str, Any]] = [
                {
                    "id": acc.id or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": acc.name,
                        "arguments": acc.arguments or "{}",
                    },
                }
                for idx, acc in sorted(tool_accs.items())
            ]
            messages.append({
                "role": "assistant",
                "content": preamble,
                "tool_calls": tc_payload,
            })
            session.add_assistant_tool_calls(preamble, tc_payload)

            # Execute each tool sequentially. Track index for preamble
            # fallback: tc_payload is built sorted by tool_accs keys, so
            # enumerate aligns with the original tool_call indexes.
            sorted_idxs = sorted(tool_accs.keys())
            for tc_pos, tc in enumerate(tc_payload):
                tc_idx = sorted_idxs[tc_pos] if tc_pos < len(sorted_idxs) else tc_pos
                if on_tool_started is not None:
                    try:
                        await on_tool_started(tc)
                    except Exception:  # noqa: BLE001
                        logger.debug("on_tool_started raised", exc_info=True)
                # Per-tool preamble: fire the metadata-declared verbal
                # acknowledgement BEFORE the (potentially slow) tool
                # dispatches. We pull the registered Tool directly off
                # the registry so callers that bypass the decorator
                # (programmatic ``registry.register(...)``) still
                # benefit. Lookup-miss + empty preamble = no-op.
                if (
                    on_tool_preamble is not None
                    and tc_idx not in preamble_fired
                ):
                    tname = tc.get("function", {}).get("name") or ""
                    tool_meta = registry._tools.get(tname) if tname else None
                    preamble = getattr(tool_meta, "preamble_text", "") or ""
                    if preamble:
                        logger.info(
                            "tool preamble: text=%r tool=%s", preamble, tname
                        )
                        try:
                            await on_tool_preamble(preamble)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "on_tool_preamble raised", exc_info=True
                            )
                t0 = time.monotonic()
                args_raw = tc["function"]["arguments"]
                try:
                    args = json.loads(args_raw or "{}")
                except json.JSONDecodeError:
                    result: dict[str, Any] = {
                        "success": False,
                        "error": f"invalid arguments JSON: {args_raw!r}",
                    }
                else:
                    result = await registry.dispatch(
                        tc["function"]["name"], args, ctx
                    )
                dt_ms = (time.monotonic() - t0) * 1000.0
                content = json.dumps(result, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                })
                session.add_tool_result(tc["id"], content)
                if on_tool_completed is not None:
                    try:
                        await on_tool_completed(tc, result, dt_ms)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "on_tool_completed raised", exc_info=True
                        )
            # loop continues

        # Iteration cap hit. Must-fix #2 (codex review): the partial
        # tool round added in the last failing iteration has no terminal
        # assistant(text), so leaving it in history would haunt every
        # future turn. Roll back to anchor — equivalent to "this turn
        # never happened" for history (user_text added by app_mode
        # *before* we were called survives).
        logger.warning(
            "tool iteration cap reached (%d); rolling back",
            max_iterations,
        )
        dropped = session.rollback_to(rollback_anchor)
        # Must-fix #4: mirror the rollback on the caller's messages list
        # too, otherwise the next turn re-uses a stale list.
        del messages[rollback_anchor + messages_offset:]
        bus = getattr(ctx, "event_bus", None)
        if bus is not None:
            try:
                bus.emit(
                    "on_tool_iteration_limit",
                    {
                        "iterations": iterations_done,
                        "dropped": dropped,
                        "sid": session.sid,
                    },
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("event_bus emit failed", exc_info=True)
        return ""
    except asyncio.CancelledError:
        # Must-fix #4: truncate both session.history AND the caller's
        # local messages list, otherwise the next turn sees mismatched
        # state.
        dropped = session.rollback_to(rollback_anchor)
        del messages[rollback_anchor + messages_offset:]
        logger.info(
            "tool round cancelled, rolled back %d messages", dropped
        )
        raise
    except BaseException:
        # Codex review (HIGH #2): any non-cancel exception escaping the
        # dispatch loop after we've appended assistant(tool_calls) +
        # tool result messages would pin an incomplete tool round in
        # history. Tool timeout, JSON decode error on result, an
        # upstream LLM error mid-continuation — all would leave the
        # session strict-invalid (orphan assistant_tool_calls with no
        # closing assistant text) and turn-aware trim would anchor
        # forever on it. Roll back symmetrically with cancel, then
        # re-raise so the caller's existing error path still fires.
        dropped = session.rollback_to(rollback_anchor)
        del messages[rollback_anchor + messages_offset:]
        if dropped:
            logger.info(
                "tool round aborted by exception, rolled back %d messages",
                dropped,
            )
        raise
