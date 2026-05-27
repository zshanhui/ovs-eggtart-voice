"""BaseApp orchestrator -- wires SLV + LLM + Audio + Plugins.

Lifecycle:
  1. `await slv.connect()` (one persistent WS).
  2. Spawn `_mic_pump_task` (mic -> WS binary) and `_slv_dispatch_task`
     (WS events -> hooks / on_user_utterance routing).
  3. Call each registered plugin's `start()`.
  4. Wait on shutdown event.
  5. `shutdown()` reverses everything.

Plugin hook dispatch is parallel via `asyncio.gather(return_exceptions=True)`
so observers don't block one another or the dispatch loop.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import TYPE_CHECKING


class TypedLLMError(RuntimeError):
    """RuntimeError subclass carrying a structured payload for the dashboard.

    Keeps full backward compatibility with the old on_error contract
    (``isinstance(exc, RuntimeError)`` + ``str(exc)`` continue to work),
    but exposes ``.payload`` so plugins like debug_dashboard can render
    typed/coloured errors instead of opaque strings.
    """

    def __init__(
        self,
        type_: str,
        message: str,
        *,
        exc_class: str = "",
        **extra,
    ) -> None:
        super().__init__(message)
        self.payload: dict = {
            "type": type_,
            "message": message,
            "exc_class": exc_class,
            "timestamp": time.time(),
            **extra,
        }

from .app_mode import LLMTimeoutError
from .audio_io import AudioIO
from .config import Config
from .event_bus import EventBus
from .llm import EdgeLLMBackend, LLMBackend, LLMStreamError, OpenAICompatBackend, NoopLLM
from .plugins.llm_availability import LLMUnavailable
from .session import Session
from .state import ConvState
# Importing .tools also imports .tools.builtin which @-decorates the
# built-in tools onto default_registry as an import side-effect. Cheap
# (no IO, no model load) so we pay it unconditionally — actual use is
# gated by config.tools_enabled per turn in app_mode.
from .tools import default_registry as _default_tool_registry
from .translator import CTranslate2Translator, NoopTranslator, TranslatorBackend
from .vad import create_vad
from .slv_client import (
    ASREndpoint,
    ASRFinal,
    ASRPartial,
    SLVClient,
    SLVError,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
)

if TYPE_CHECKING:
    from .plugin import Plugin

logger = logging.getLogger(__name__)


# ── Low-signal ASR final filter ─────────────────────────────────────
# An open-mic always-on pipeline will, fairly often, emit ASR finals
# that are just one Chinese character or one English letter — these
# are almost always noise / ambient speech / breath the silero VAD
# happened to clip out, NOT real intent. Routing them to the LLM
# triggers a "safe fallback" reply ("我在这里呢…") that, after a few
# repeats, locks a small quantised model into an echo loop where it
# emits the same fallback forever no matter what you say next.
_INTERJECTIONS: frozenset[str] = frozenset(
    {
        # Chinese: noncommittal acknowledgements / filler.
        "嗯", "啊", "哦", "呃", "唉", "诶", "哎", "噢", "唔", "呀", "哈",
        "哇", "呢", "吧", "吗", "呐", "嘛", "诶呀", "啊啊", "嗯嗯",
        # English: same idea — too short to convey intent on a voice mic.
        "uh", "um", "ah", "oh", "ok", "okay", "hmm", "huh", "yeah", "yep",
        "you", "the", "and", "a", "i",
    }
)


def _strip_for_signal(text: str) -> str:
    """Return the input with whitespace + common punctuation removed,
    lowercased, for low-signal comparison against ``_INTERJECTIONS``.
    Keeps Chinese chars and ASCII alphanumerics as-is.
    """
    import unicodedata
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Drop separators / punctuation; keep letters and digits.
        if cat[0] in {"L", "N"}:
            out.append(ch)
    return "".join(out).lower()


def _build_llm(config: Config) -> LLMBackend:
    backend = config.llm_backend.lower()
    if backend == "noop":
        return NoopLLM()
    if backend == "edge_llm":
        return EdgeLLMBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            retry_on_transient=config.llm_retry_on_transient,
            retry_backoff_s=config.llm_retry_backoff_s,
        )
    if backend in ("openai_compat", "openai"):
        return OpenAICompatBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            retry_on_transient=config.llm_retry_on_transient,
            retry_backoff_s=config.llm_retry_backoff_s,
        )
    raise ValueError(f"Unknown llm_backend: {config.llm_backend!r}")


def _build_translator(config: Config) -> TranslatorBackend:
    backend = config.translator_backend.lower()
    if backend == "noop":
        return NoopTranslator()
    if backend == "ctranslate2":
        return CTranslate2Translator(
            base_url=config.translator_url,
            timeout=config.translator_timeout_s,
        )
    raise ValueError(f"Unknown translator_backend: {config.translator_backend!r}")


class BaseApp:
    """Subclass and implement `on_user_utterance` to define an App."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.events = EventBus()
        self.slv = SLVClient(config.slv_url, config.slv_config)
        self.audio = AudioIO(
            input_device=config.audio_input_device,
            output_device=config.audio_output_device,
            input_sr=config.audio_input_sample_rate,
            output_sr=config.audio_output_sample_rate,
        )
        self.llm: LLMBackend = _build_llm(config)
        self.translator: TranslatorBackend = _build_translator(config)
        # Tool registry — single global default. Tests may construct
        # dedicated `ToolRegistry()` instances and inject via mode_ctx.
        self.tool_registry = _default_tool_registry
        self.session = Session(
            locale=str(config.slv_config.get("asr_language", "zh")).lower()[:2],
            max_input_tokens=getattr(config, "session_max_input_tokens", None),
            tokenizer_model=getattr(
                config, "session_tokenizer_model", "Qwen/Qwen3-4B-AWQ"
            ),
            event_bus=self.events,
        )
        self.plugins: list["Plugin"] = []
        # Set by LLMAvailabilityPlugin.start(); read by app_mode to fail-fast
        # when the LLM is DOWN instead of hitting the 15s first-token timeout.
        self.llm_availability = None
        self._shutdown_evt: asyncio.Event | None = None
        self._mic_task: asyncio.Task | None = None
        self._mic_watchdog_task: asyncio.Task | None = None
        self._mic_restart_lock: asyncio.Lock | None = None
        self._last_mic_chunk_ts: float | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._llm_turn_task: asyncio.Task | None = None
        self._first_tts_seen = False
        # Client-side VAD state machine. Drives manual asr_eos to SLV when
        # server-side VAD is disabled (slv_config.vad == "none"), so the
        # ASR model gets a chance to accumulate enough audio before being
        # asked to finalize.
        self._client_vad = None
        if getattr(config, "client_vad_backend", "off") != "off":
            try:
                self._client_vad = create_vad(
                    config.client_vad_backend,
                    sample_rate=config.audio_input_sample_rate,
                    threshold=getattr(config, "client_vad_threshold", None),
                )
                logger.info(
                    "client VAD: %s (threshold=%s)",
                    self._client_vad.name,
                    self._client_vad.threshold,
                )
            except Exception as e:
                logger.warning("client VAD init failed (%s); disabled", e)
                self._client_vad = None
        self._vad_state = "idle"  # "idle" | "speech"
        self._vad_speech_ms = 0
        self._vad_silence_ms = 0
        self._vad_eos_sent = False
        # ── v2: conversation state machine + observability ──
        # Initial state depends on pipeline_mode: always_on boots IDLE
        # (legacy), wake_word / push_to_talk boot SLEEPING.
        if getattr(config, "pipeline_mode", "always_on") == "always_on":
            self._state: ConvState = ConvState.IDLE
        else:
            self._state = ConvState.SLEEPING
        self._slv_reconnect_count: int = 0
        # Auto-sleep timer (only armed when pipeline_mode != always_on).
        self._sleep_task: asyncio.Task | None = None
        # Push-to-talk: when True, the next asr_final is the explicit
        # close of a PTT turn — used to short-circuit empty-final guards
        # that would otherwise drop a clipped PTT utterance.
        self._ptt_explicit_eos_pending: bool = False
        # Per-turn EOS dedupe: VAD silence and PTT/end can both want to
        # send asr_eos. Send at most one per turn. Cleared on every
        # ASRFinal, on PTT/start (next turn), and on reconnect.
        self._eos_sent_this_turn: bool = False
        # Watchdog: SLV in `always_on` pipeline mode does NOT emit
        # asr_final when ASR yields empty text (it filters server-side
        # to avoid noise turns). Without this watchdog the state machine
        # would stay THINKING forever after the very first VAD trigger
        # on mic noise. Started in send_asr_eos_once, cancelled in the
        # ASRFinal / SLVError / reconnect paths.
        self._asr_watchdog_task: asyncio.Task | None = None
        # Watchdog: SLV in jetson-qwen3asr-matcha-nx (and possibly others)
        # can silently fail to produce TTS after a successful asr_final →
        # tool_call → text-stream → tts_flush sequence. State stays
        # THINKING forever waiting for tts_started that never arrives.
        # Armed when we transition into THINKING; cancelled by the first
        # TTSStarted / TTSDone / ASRFinal (real activity) or SLVError.
        # On fire, force state back to IDLE so the next user turn isn't
        # blocked. Configurable via ``thinking_timeout_s`` (default 20s).
        self._thinking_watchdog_task: asyncio.Task | None = None
        # Dashboard mic RMS is a best-effort visualization signal. Never let
        # a slow browser/plugin backpressure the hot mic pump; at most one
        # RMS hook fanout may be in flight and newer samples are dropped.
        self._mic_rms_broadcast_task: asyncio.Task | None = None
        # Rate-limited stop-word matcher cache (compiled per Config update).
        self._stop_words_cache: tuple[list[str], list[str]] | None = None

    # ── startup budget validation ───────────────────────────────────

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Cheap upper-bound token estimate without loading a tokenizer.

        ``len(text) // 3`` is a conservative bound: Chinese chars ≈ 0.4
        token, English chars ≈ 0.25 token; 1/3 sits above both. Used
        only for startup sanity-check logging, not for trim accounting.
        """
        if not text:
            return 0
        return max(1, len(text) // 3)

    @classmethod
    def _approx_tokens_for_tools(cls, tools: list[dict] | None) -> int:
        if not tools:
            return 0
        try:
            import json as _json
            return cls._approx_tokens(_json.dumps(tools, ensure_ascii=False))
        except Exception:
            return 0

    def _validate_session_budget(
        self, system_prompt: str, tools: list[dict] | None
    ) -> None:
        """Sanity-check session_max_input_tokens vs the fixed prefix.

        The trim threshold is ``session_max_input_tokens * 0.75``. The
        fixed prefix (system_prompt + tools schema) is charged against
        the same budget on every turn. If the fixed prefix is already
        at or above the trim threshold, *every* turn trims, which
        clears ``Session.cache_warmed`` and defeats the upstream
        KV-cache hot path permanently. This validator surfaces that
        misconfiguration at startup with an explicit ERROR.
        """
        max_input = getattr(self.config, "session_max_input_tokens", None)
        if not max_input:
            logger.info(
                "Session trim disabled (session_max_input_tokens=None); "
                "skipping budget validation"
            )
            return
        sys_tokens = self._approx_tokens(system_prompt or "")
        tools_tokens = self._approx_tokens_for_tools(tools)
        fixed_tokens = sys_tokens + tools_tokens
        budget = int(max_input * 0.75)
        pct = (fixed_tokens / budget * 100) if budget else 0.0
        if fixed_tokens >= budget:
            recommended = int(fixed_tokens / 0.75) + 2000
            logger.error(
                "FIXED PREFIX (system_prompt %d + tools %d = %d tokens) >= trim "
                "budget %d tokens (%.0f%% of max %d). Every turn will trim, KV "
                "cache will be invalidated, hot path disabled. Raise "
                "session_max_input_tokens to at least %d.",
                sys_tokens, tools_tokens, fixed_tokens, budget, pct,
                max_input, recommended,
            )
        elif fixed_tokens > budget * 0.5:
            logger.warning(
                "Fixed prefix uses %.0f%% of trim budget (%d / %d tokens, "
                "system=%d tools=%d). Few turns of history will fit before "
                "trim. Consider raising session_max_input_tokens.",
                pct, fixed_tokens, budget, sys_tokens, tools_tokens,
            )
        else:
            logger.info(
                "Session budget OK: fixed prefix %d / budget %d tokens "
                "(%.0f%% used, system=%d tools=%d, max_input=%d).",
                fixed_tokens, budget, pct, sys_tokens, tools_tokens, max_input,
            )

    # ── v2: state machine + stop intent ─────────────────────────────

    def _set_state(self, new: ConvState) -> None:
        """Transition the conversation state. Logs + emits hook/event on change.

        Safe to call from any coroutine in the same event loop. Tests build
        BaseApp via __new__ without invoking __init__, so default missing
        attributes to IDLE rather than crashing.
        """
        old = getattr(self, "_state", ConvState.IDLE)
        if new == old:
            return
        self._state = new
        logger.info("ConvState: %s → %s", old.value, new.value)
        bus = getattr(self, "events", None)
        if bus is not None:
            try:
                bus.emit("state_change", {"state": new.value, "prev": old.value})
            except Exception:  # pragma: no cover - defensive
                logger.debug("EventBus state_change emit failed", exc_info=True)
        try:
            asyncio.get_running_loop().create_task(
                self._broadcast(
                    "on_state_change", {"state": new.value, "prev": old.value}
                )
            )
        except RuntimeError:
            pass

    def _normalise_for_stop(self, text: str) -> str:
        """Lowercase + strip whitespace and trailing punctuation."""
        if not text:
            return ""
        s = text.strip()
        # Strip trailing CJK + ASCII sentence punctuation.
        while s and s[-1] in "。，！？.!?,;:":
            s = s[:-1]
        return s.strip().lower()

    def _is_stop_intent(self, text: str) -> bool:
        """Match per spec: Chinese -> exact full-string; English -> case-
        insensitive whole-utterance OR word-boundary prefix (>= 2 chars).
        """
        norm = self._normalise_for_stop(text)
        if not norm:
            return False
        # Partition stop_words by ASCII-ness.
        cfg = getattr(self, "config", None)
        words = (getattr(cfg, "stop_words", []) if cfg is not None else []) or []
        for w in words:
            if not w:
                continue
            wn = w.strip().lower()
            if not wn:
                continue
            is_ascii = wn.isascii()
            if not is_ascii:
                # CJK / unicode: full-string equality only.
                if norm == wn:
                    return True
            else:
                # English: whole-utterance equality OR
                # word-boundary prefix (matched word is at least 2 chars).
                if norm == wn:
                    return True
                if len(wn) >= 2 and (
                    norm.startswith(wn + " ")
                    or norm.startswith(wn + ",")
                    or norm.startswith(wn + "!")
                    or norm.startswith(wn + "?")
                    or norm.startswith(wn + ".")
                ):
                    return True
        return False

    # ── pipeline_mode: wake / sleep / sleep-timer ──────────────────

    async def wake(self, source: str = "external") -> None:
        """Transition SLEEPING → IDLE and (re-)arm the sleep timer.

        No-op if not currently SLEEPING. Broadcasts on_wake with source.
        """
        if getattr(self, "_state", ConvState.IDLE) != ConvState.SLEEPING:
            return
        logger.info("wake from %s", source)
        try:
            await self._broadcast("on_wake", {"source": source})
        except Exception:
            logger.exception("on_wake broadcast failed")
        # Clear the playback discard latch sleep() armed — otherwise the
        # first post-wake TTS (especially typed-text path with no ASRFinal)
        # would be silently dropped.
        try:
            arm = getattr(self.audio, "arm_for_next_turn", None)
            if callable(arm):
                arm()
        except Exception:  # pragma: no cover - defensive
            pass
        self._set_state(ConvState.IDLE)
        self._reset_sleep_timer()

    async def sleep(self) -> None:
        """Forcibly transition to SLEEPING — cancel LLM turn, abort SLV,
        drop playback. Idempotent if already SLEEPING."""
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        logger.info("sleep")
        try:
            await self._broadcast("on_sleep", None)
        except Exception:
            logger.exception("on_sleep broadcast failed")
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.slv.abort()
        except Exception:
            pass
        try:
            await self.audio.stop_playback()
        except Exception:
            pass
        self._first_tts_seen = False
        self._set_state(ConvState.SLEEPING)
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        self._sleep_task = None

    def _reset_sleep_timer(self) -> None:
        """(Re-)start the auto-sleep countdown. No-op for always_on."""
        if getattr(self.config, "pipeline_mode", "always_on") == "always_on":
            return
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        timeout = float(getattr(self.config, "sleep_timeout_s", 30.0))
        try:
            self._sleep_task = asyncio.create_task(
                self._sleep_after(timeout), name="sleep-timer"
            )
        except RuntimeError:
            # No running loop (called from sync context like tests).
            self._sleep_task = None

    async def _sleep_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        # Only sleep if still IDLE — an in-flight turn delays.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.IDLE:
            await self.sleep()

    # ── public API ──────────────────────────────────────────────────

    def register(self, plugin: "Plugin") -> bool:
        if not plugin.setup():
            logger.info("plugin %s setup() returned False -- skipped", plugin.name)
            return False
        self.plugins.append(plugin)
        return True

    async def on_user_utterance(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """Subclasses MUST override. Default raises.

        ``detected_language`` is the ASR-reported language for this turn
        (e.g. ``"Chinese"``) or ``None`` if the backend doesn't do LID.
        Passed per-call so mode lifecycle hooks (enter/exit) never see a
        stale value.
        """
        raise NotImplementedError("Subclass BaseApp and implement on_user_utterance")

    async def run(self) -> None:
        self._shutdown_evt = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_evt.set)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread -- caller is responsible.
                pass

        await self.slv.connect()
        self._mic_restart_lock = asyncio.Lock()
        self._mic_task = asyncio.create_task(self._mic_pump(), name="mic-pump")
        self._mic_watchdog_task = asyncio.create_task(
            self._mic_watchdog(), name="mic-watchdog"
        )
        self._dispatch_task = asyncio.create_task(self._slv_dispatch(), name="slv-dispatch")

        # LLM backend warmup — runs after all plugins have registered
        # (so tool_registry is fully populated) and BEFORE any plugin
        # start() so the very first user turn never pays cold-start
        # cost. EdgeLLMBackend warms both the prefix KV cache and the
        # TRT-LLM CUDA graph; other backends inherit the default no-op.
        try:
            tools_payload = None
            registry = getattr(self, "tool_registry", None)
            if registry is not None and hasattr(registry, "list_openai_tools"):
                try:
                    tools_payload = registry.list_openai_tools(allow=None) or None
                except Exception:
                    logger.debug("warmup: tool_registry lookup failed", exc_info=True)
                    tools_payload = None
            sys_prompt = ""
            try:
                overrides = getattr(self.config, "mode_overrides", {}) or {}
                mode_cfg = overrides.get("chat") if isinstance(overrides, dict) else None
                if isinstance(mode_cfg, dict):
                    sys_prompt = mode_cfg.get("system_prompt") or ""
            except Exception:
                pass
            if not sys_prompt:
                sys_prompt = getattr(self.config, "system_prompt", "") or ""
            # Validate session budget BEFORE warmup. If the fixed prefix
            # (system_prompt + tools schema) is too close to the trim
            # threshold, every turn will trim → cache_warmed cleared →
            # KV-cache hot path defeated. Logs ERROR if misconfigured,
            # WARNING if tight, INFO if healthy.
            self._validate_session_budget(sys_prompt, tools_payload)
            warmup_result = await self.llm.warmup(
                system_prompt=sys_prompt,
                tools=tools_payload,
                enable_thinking=False,
            )
            if warmup_result:
                logger.info("LLM backend warmup result: %s", warmup_result)
                if warmup_result.get("cache_warmed") and self.session is not None:
                    self.session.cache_warmed = True
                    logger.info(
                        "session.cache_warmed=True after backend warmup; "
                        "first turn will use prefix_cache"
                    )
        except Exception:
            logger.warning("LLM warmup failed; first turn may be cold", exc_info=True)

        for p in self.plugins:
            try:
                await p.start()
            except Exception:
                logger.exception("plugin %s start() failed", p.name)

        try:
            await self._shutdown_evt.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        # 0. cancel any in-flight LLM turn
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        # 0a. cancel auto-sleep timer too — otherwise a pending
        # _sleep_after coroutine can fire mid-shutdown, racing with
        # the rest of the cleanup (and emitting on_sleep after plugins
        # have already stopped).
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
            try:
                await self._sleep_task
            except (asyncio.CancelledError, Exception):
                pass
        self._sleep_task = None
        mic_watchdog_task = getattr(self, "_mic_watchdog_task", None)
        if mic_watchdog_task is not None and not mic_watchdog_task.done():
            mic_watchdog_task.cancel()
            try:
                await mic_watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        self._mic_watchdog_task = None
        # 1. stop mic capture
        if self._mic_task is not None:
            self._mic_task.cancel()
        # 2. cancel TTS if any
        if self.audio.is_playing:
            try:
                await self.slv.abort()
            except Exception:  # pragma: no cover
                pass
        # 3. stop plugins in reverse registration order
        for p in reversed(self.plugins):
            try:
                await p.stop()
            except Exception:
                logger.exception("plugin %s stop() failed", p.name)
        # 4. cancel dispatch
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
        for t in (self._mic_task, self._dispatch_task):
            if t is None:
                continue
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # 5. close transport
        try:
            await self.slv.close()
        except Exception:  # pragma: no cover
            pass
        # 6. drain playback
        try:
            await self.audio.stop_playback()
            await self.audio.close()
        except Exception:  # pragma: no cover
            pass
        # 7. release LLM client resources (HTTP connection pool, etc.)
        try:
            await self.llm.aclose()
        except Exception:  # pragma: no cover
            pass
        # 8. release translator client resources
        try:
            await self.translator.aclose()
        except Exception:  # pragma: no cover
            pass

    def request_shutdown(self) -> None:
        if self._shutdown_evt is not None:
            self._shutdown_evt.set()

    # ── internal pumps ──────────────────────────────────────────────

    async def restart_mic_capture(self, reason: str = "manual") -> None:
        """Restart only the local sounddevice input stream + mic pump.

        This is cheaper than restarting the whole agent and is useful after
        CoreAudio device changes / PaMacCore errors leave the input stream
        alive but no longer delivering useful chunks.
        """
        lock = getattr(self, "_mic_restart_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._mic_restart_lock = lock
        async with lock:
            logger.warning("restarting mic capture (%s)", reason)
            task = getattr(self, "_mic_task", None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                stop_input = getattr(self.audio, "_stop_input_stream", None)
                if callable(stop_input):
                    stop_input()
            except Exception:
                logger.debug("stop input stream failed during mic restart", exc_info=True)
            self._vad_state = "idle"
            self._vad_speech_ms = 0
            self._vad_silence_ms = 0
            self._vad_eos_sent = False
            try:
                reset = getattr(self._client_vad, "reset", None)
                if callable(reset):
                    reset()
            except Exception:
                logger.debug("client VAD reset failed during mic restart", exc_info=True)
            self._last_mic_chunk_ts = time.monotonic()
            self._mic_task = asyncio.create_task(self._mic_pump(), name="mic-pump")

    async def _mic_watchdog(self) -> None:
        """Recover from dead CoreAudio/sounddevice capture streams."""
        stale_s = 5.0
        try:
            while True:
                await asyncio.sleep(2.0)
                if getattr(self, "_shutdown_evt", None) is not None and self._shutdown_evt.is_set():
                    return
                task = getattr(self, "_mic_task", None)
                if task is None:
                    await self.restart_mic_capture("watchdog:no-task")
                    continue
                if task.done():
                    exc = None
                    try:
                        exc = task.exception()
                    except (asyncio.CancelledError, Exception):
                        exc = None
                    logger.warning("mic pump stopped; restarting (exc=%r)", exc)
                    await self.restart_mic_capture("watchdog:task-done")
                    continue
                last = getattr(self, "_last_mic_chunk_ts", None)
                if last is not None and (time.monotonic() - last) > stale_s:
                    await self.restart_mic_capture("watchdog:stale")
        except asyncio.CancelledError:
            raise

    async def _send_audio_nonblocking(self, pcm: bytes) -> None:
        """Send a mic chunk to SLV with a short ceiling on how long the
        send may block.

        Why: ``SLVClient._send_lock`` serialises the send half of the WS,
        and the dispatch loop's auto-reconnect (``slv.reconnect()``) holds
        the same lock for the duration of a fresh ``ws_connect`` — which
        can stall for several seconds on a network blip / DNS hiccup.
        Without a ceiling here, every mic chunk during reconnect parks on
        the lock, the mic_pump coroutine stops draining its input queue,
        sounddevice's callback thread floods ``call_soon_threadsafe`` with
        un-consumed PCM, and the log starts hemorrhaging
        ``mic queue full -- dropping chunk`` for the entire outage —
        which is exactly the "agent feels dead" symptom.

        Bounded wait + drop is the right trade for a mic stream: the
        chunks we drop while SLV is briefly unreachable would have been
        useless anyway (the WS that would have carried them is closed),
        and the post-reconnect first ASR utterance starts from fresh
        chunks. Pre-roll is still preserved for the *current* speech
        segment whose onset already won the VAD race.
        """
        try:
            await asyncio.wait_for(self.slv.send_audio(pcm), timeout=0.5)
        except asyncio.TimeoutError:
            # SLV is mid-reconnect / unreachable; don't wedge the mic pump.
            # Logged at debug to avoid floods during normal reconnect blips.
            logger.debug("send_audio timed out (slv reconnecting?); dropping chunk")
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.debug("send_audio failed; dropping chunk", exc_info=True)

    def _schedule_mic_rms_broadcast(self, data: dict) -> bool:
        task = getattr(self, "_mic_rms_broadcast_task", None)
        if task is not None and not task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._mic_rms_broadcast_task = loop.create_task(
            self._broadcast("on_mic_rms", data),
            name="mic-rms-broadcast",
        )
        return True

    async def _mic_pump(self) -> None:
        """Mic capture loop. When client VAD is enabled, only forwards audio
        to SLV during (and just before) actual speech — pre-roll buffer
        ensures the first ~300ms of an utterance isn't lost while the VAD
        is still confirming speech-start. Idle silence is never sent.

        Why: streaming background noise for minutes at a time saturates the
        WS write pipeline and starves websockets' keepalive ping coroutine,
        triggering 1011 keepalive ping timeout. Dropping idle chunks keeps
        the connection mostly quiet between turns.
        """
        from collections import deque

        try:
            chunk_ms = getattr(self.audio, "chunk_ms", 100)
            preroll_max = max(1, 400 // max(chunk_ms, 1))  # ~400ms
            preroll: deque[bytes] = deque(maxlen=preroll_max)
            import numpy as _np
            # Rate-limit on_mic_rms broadcast: 10Hz is overkill for a
            # dashboard sparkline, and awaiting every plugin every 100ms
            # was starving the mic queue during TTS playback — VAD never
            # saw the burst of audio when the user spoke, so barge-in
            # never fired. Broadcast at most every ~200ms (every 2nd
            # chunk at chunk_ms=100, every 4th at chunk_ms=50).
            rms_broadcast_every = max(1, 200 // max(chunk_ms, 1))
            rms_chunk_counter = 0
            async for chunk in self.audio.start_capture():
                self._last_mic_chunk_ts = time.monotonic()
                # pipeline_mode gating: drop audio entirely while SLEEPING.
                # WS stays connected so wake-time reconnect cost is zero.
                if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
                    # Also clear pre-roll so we don't leak pre-sleep audio
                    # into the next wake's first utterance.
                    preroll.clear()
                    continue
                # Per-chunk mic RMS for the dashboard. Rate-limited so a
                # slow WS client doesn't backpressure the mic queue and
                # starve VAD (which kills barge-in detection during TTS).
                rms_chunk_counter = (rms_chunk_counter + 1) % rms_broadcast_every
                if rms_chunk_counter == 0:
                    try:
                        arr = _np.frombuffer(chunk, dtype=_np.int16)
                        if arr.size:
                            rms = float(_np.sqrt(_np.mean((arr.astype(_np.float32) / 32768.0) ** 2)))
                        else:
                            rms = 0.0
                        thr = float(getattr(self.config, "client_vad_threshold", None) or 0.012)
                        self._schedule_mic_rms_broadcast(
                            {"rms": rms, "threshold": thr, "state": self._vad_state}
                        )
                        # TEMP DIAGNOSTIC: log loud chunks so we can see
                        # whether mic is even picking up user speech post-TTS.
                        # 0.03 ≈ ordinary indoor voice at 30cm; <0.01 is silence.
                        if rms > 0.03:
                            logger.info(
                                "mic chunk loud: rms=%.4f state=%s convstate=%s",
                                rms, self._vad_state,
                                getattr(self, "_state", ConvState.IDLE).value,
                            )
                    except Exception:  # pragma: no cover - defensive
                        pass

                if self._client_vad is None:
                    # No VAD configured: stream everything (legacy behaviour).
                    await self._send_audio_nonblocking(chunk)
                    continue

                # Update VAD first; it may transition idle→speech this chunk.
                try:
                    await self._update_vad(chunk, chunk_ms)
                except Exception:
                    logger.exception("client VAD update failed")

                if self._vad_state == "speech":
                    # Drain the pre-roll buffer at speech onset, then stream
                    # this chunk plus subsequent ones in real time.
                    if preroll:
                        for buffered in preroll:
                            await self._send_audio_nonblocking(buffered)
                        preroll.clear()
                    await self._send_audio_nonblocking(chunk)
                else:
                    # Idle: keep a short rolling buffer but don't transmit.
                    preroll.append(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mic pump crashed")

    async def send_asr_eos_once(self) -> bool:
        """Send asr_eos to SLV at most once per turn.

        Returns True if this call actually sent the EOS, False if it
        was a duplicate (already sent this turn). The flag is reset on
        ASRFinal / PTT-start / SLV reconnect.

        Arms an `_asr_final_watchdog` so the state machine self-recovers
        if SLV doesn't echo an ASR final back (always_on pipeline mode
        silently drops empty finals — without this, the FSM stays
        THINKING forever after a noise-triggered turn).
        """
        if getattr(self, "_eos_sent_this_turn", False):
            return False
        self._eos_sent_this_turn = True
        try:
            await self.slv.asr_eos()
        except Exception:
            logger.exception("asr_eos send failed")
            # Don't clear the flag — even on failure we don't want to
            # retry a second time within the same turn and risk the SLV
            # state machine getting into an inconsistent state.
        # Arm watchdog (cancels any stale one from a prior failed turn).
        self._cancel_asr_watchdog()
        self._asr_watchdog_task = asyncio.create_task(
            self._asr_final_watchdog(),
            name="asr-final-watchdog",
        )
        return True

    def _cancel_asr_watchdog(self) -> None:
        """Cancel any pending asr_final watchdog (idempotent)."""
        task = getattr(self, "_asr_watchdog_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._asr_watchdog_task = None

    async def _interrupt_current_turn_for_barge_in(self) -> None:
        """Stop the audible assistant turn before accepting barge-in audio.

        Barge-in semantics are intentionally ordered:
          1. cancel the local LLM streaming task so no more text is sent;
          2. stop local speaker playback immediately;
          3. send SLV's in-band abort control to cancel the already queued /
             in-flight TTS synthesis;
          4. keep the SLV WebSocket alive so the user's current speech keeps
             flowing to ASR without a reconnect gap.

        The current SLV protocol multiplexes ASR input and TTS output on one
        connection. Closing/reconnecting it here also drops exactly the audio
        we need for the barge-in utterance, which turns an immediate interrupt
        into a multi-second delayed response. The right control is the in-band
        `abort` frame: SLV cancels current TTS and drains queued sentences
        without tearing down the WebSocket.
        """
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.audio.stop_playback()
        except Exception:
            logger.exception("stop_playback failed during barge-in")
        try:
            await asyncio.wait_for(self.slv.abort(), timeout=0.5)
            logger.info("SLV abort sent during barge-in")
        except asyncio.TimeoutError:
            logger.warning("SLV abort timed out during barge-in")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("SLV abort failed during barge-in")
        self._eos_sent_this_turn = False
        self._cancel_asr_watchdog()
        self._first_tts_seen = False

    def _arm_thinking_watchdog(self) -> None:
        """Re-arm the THINKING-state watchdog.

        Idempotent: a prior task is cancelled. Called every time we
        transition INTO THINKING (real asr_final, dashboard text inject).
        """
        if self._thinking_watchdog_task is not None and not self._thinking_watchdog_task.done():
            self._thinking_watchdog_task.cancel()
        self._thinking_watchdog_task = asyncio.create_task(
            self._thinking_watchdog(), name="thinking-watchdog",
        )

    def _cancel_thinking_watchdog(self) -> None:
        """Cancel the THINKING watchdog if armed. Idempotent."""
        if self._thinking_watchdog_task is not None and not self._thinking_watchdog_task.done():
            self._thinking_watchdog_task.cancel()
        self._thinking_watchdog_task = None

    async def _thinking_watchdog(self) -> None:
        """Force state back to IDLE if THINKING never resolves into
        SPEAKING (no ``tts_started`` event from SLV).

        Observed failure on jetson-qwen3asr-matcha-nx: after a successful
        ASR → LLM → tool_call → text-stream → flush_tts cycle, the SLV
        server occasionally fails to start TTS playback (no further
        events from the WS even though it stays connected). Without this
        watchdog the FSM stays THINKING forever and every subsequent
        user utterance is ignored. On fire we reconnect SLV (which gives
        a fresh worker) and reset state to IDLE so the next turn can
        proceed.
        """
        timeout = float(getattr(self.config, "thinking_timeout_s", 20.0))
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if getattr(self, "_state", ConvState.IDLE) != ConvState.THINKING:
            return  # state moved on naturally; nothing to do
        logger.warning(
            "thinking watchdog fired (no tts_started in %.1fs after asr_final); "
            "resetting state to IDLE and reconnecting SLV",
            timeout,
        )
        # Drop any LLM turn task that's still believed to be in flight.
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        # Force a fresh WS — the server-side TTS pipeline is likely
        # wedged on the current session. Best-effort.
        try:
            await asyncio.wait_for(self.slv.reconnect(), timeout=3.0)
            self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
            await self._broadcast(
                "on_slv_reconnect", {"count": self._slv_reconnect_count}
            )
        except Exception:
            logger.exception("SLV reconnect failed during thinking-watchdog recovery")
        self._first_tts_seen = False
        # Symmetric latch reset with the SLVError / dispatch-reconnect
        # paths. Without clearing this, the next utterance could
        # short-circuit ``send_asr_eos_once`` and never receive a final.
        self._eos_sent_this_turn = False
        self._set_state(ConvState.IDLE)
        self._reset_sleep_timer()

    async def _asr_final_watchdog(self) -> None:
        """Force state back to IDLE if asr_final never arrives after asr_eos.

        SLV's always_on pipeline filters empty-text finals server-side, so
        an EOS triggered by mic noise produces no client-visible final and
        the FSM would stay in THINKING forever. Real finals cancel this
        task before it fires.
        """
        timeout = float(getattr(self.config, "asr_final_timeout_s", 3.0))
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        # Only act if (a) we still believe an EOS is outstanding and
        # (b) the FSM hasn't moved on (e.g. via SLVError, a late final
        # that arrived just before us, or a barge-in).
        if not getattr(self, "_eos_sent_this_turn", False):
            return
        if getattr(self, "_state", ConvState.IDLE) != ConvState.THINKING:
            logger.info(
                "asr_final watchdog fired after state moved to %s; "
                "clearing stale EOS latch",
                getattr(self, "_state", ConvState.IDLE).value,
            )
            self._eos_sent_this_turn = False
            return
        logger.warning(
            "asr_final not received within %.1fs after asr_eos; "
            "assuming empty/dropped final — resetting to IDLE", timeout,
        )
        self._eos_sent_this_turn = False
        self._set_state(ConvState.IDLE)

    async def _update_vad(self, chunk: bytes, chunk_ms: int) -> None:
        """Client-side speech-end detector. Sends asr_eos to SLV after a
        period of silence following speech, so Paraformer has accumulated
        enough audio to produce a non-empty final."""
        assert self._client_vad is not None
        # Gate while SLEEPING: don't update counters or fire eos.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        # PTT mode with explicit-eos-only: skip VAD silence accumulation
        # entirely so the only EOS path is /api/control/ptt/end.
        cfg = getattr(self, "config", None)
        if (
            cfg is not None
            and getattr(cfg, "pipeline_mode", "always_on") == "push_to_talk"
            and getattr(cfg, "push_to_talk_no_vad_silence", True)
        ):
            return
        is_speech = self._client_vad.is_speech(chunk)
        if self._vad_state == "idle":
            if is_speech:
                self._vad_speech_ms += chunk_ms
                if self._vad_speech_ms >= self.config.client_vad_speech_min_ms:
                    self._vad_state = "speech"
                    self._vad_silence_ms = 0
                    self._vad_eos_sent = False
                    logger.info("client VAD: speech started")
                    if (
                        getattr(self, "_state", ConvState.IDLE) == ConvState.THINKING
                        and getattr(self, "_eos_sent_this_turn", False)
                    ):
                        logger.info(
                            "client VAD: new speech while waiting for asr_final; "
                            "starting a fresh ASR turn"
                        )
                        self._eos_sent_this_turn = False
                        self._cancel_asr_watchdog()
                    # If TTS is currently playing, this is a barge-in.
                    # Transition straight to BARGED_IN so the dispatch
                    # loop's later ASRPartial check (which races SLV's
                    # ~610ms first-decode latency) doesn't miss the
                    # transition. mic_pump fires first because client
                    # VAD detects speech the moment we send chunks.
                    if self.audio.is_playing:
                        logger.info("BARGE-IN fired (VAD-driven, state=%s)", self._state.value)
                        self._set_state(ConvState.BARGED_IN)
                        await self._interrupt_current_turn_for_barge_in()
                    else:
                        self._set_state(ConvState.LISTENING)
            else:
                self._vad_speech_ms = 0
        elif self._vad_state == "speech":
            if not is_speech:
                self._vad_silence_ms += chunk_ms
                if self._vad_silence_ms >= self.config.client_vad_silence_ms:
                    if not self._vad_eos_sent:
                        import time as _t
                        drove_eos = bool(getattr(self.config, "client_vad_drive_eos", False))
                        if drove_eos:
                            logger.info("client VAD: speech ended -> asr_eos")
                            await self._broadcast(
                                "on_user_speech_end_client",
                                {"ts": int(_t.time() * 1000), "drove_eos": True},
                            )
                            # Dedup: PTT/end may also try to send. Only
                            # one asr_eos per turn — race protection.
                            await self.send_asr_eos_once()
                        else:
                            logger.debug(
                                "client VAD: speech ended (paraformer-endpoint mode, no asr_eos)"
                            )
                            await self._broadcast(
                                "on_user_speech_end_client",
                                {"ts": int(_t.time() * 1000), "drove_eos": False},
                            )
                        self._set_state(ConvState.THINKING)
                        # Arm the thinking-watchdog so a wedged SLV TTS
                        # can't strand the FSM. (Symmetric with the
                        # server-VAD path at _dispatch_one.)
                        self._arm_thinking_watchdog()
                        self._vad_eos_sent = True
                    self._vad_state = "idle"
                    self._vad_speech_ms = 0
                    self._vad_silence_ms = 0
                    self._client_vad.reset()
            else:
                self._vad_silence_ms = 0

    async def _slv_dispatch(self) -> None:
        """Drive SLV events into the FSM. Auto-reconnects whenever the
        events() iterator returns naturally — SLV closes the WS after
        every asr_eos round (even in multi_utterance mode), and an empty
        / dropped final means no ASRFinal session_complete=True ever
        fires the in-band reconnect at line ~768. Without this outer
        loop the dispatch task silently dies after one bad turn and
        every subsequent asr_eos hits a closed WS ("send_json: WS closed
        mid-send" floods the log) — which also kills barge-in because
        TTS never reaches the speaker again.
        """
        backoff = 0.5
        while not getattr(self.slv, "_closed", False):
            try:
                async for evt in self.slv.events():
                    try:
                        await self._dispatch_one(evt)
                    except Exception:
                        logger.exception("dispatch error on %r", evt)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("slv events iterator crashed")
            # events() returned: reader died (SLV closed the WS or net
            # blip). Reconnect and resume — unless we're shutting down.
            if getattr(self.slv, "_closed", False):
                return
            try:
                logger.info("slv dispatch: reader exited, reconnecting...")
                await asyncio.wait_for(self.slv.reconnect(), timeout=5.0)
                self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                self._first_tts_seen = False
                self._eos_sent_this_turn = False
                self._cancel_asr_watchdog()
                if getattr(self, "_state", ConvState.IDLE) in {
                    ConvState.THINKING,
                    ConvState.BARGED_IN,
                }:
                    self._set_state(ConvState.IDLE)
                try:
                    await self._broadcast(
                        "on_slv_reconnect", {"count": self._slv_reconnect_count}
                    )
                except Exception:
                    pass
                backoff = 0.5
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("auto-reconnect failed, sleeping %.1fs", backoff)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 5.0)

    async def _dispatch_one(self, evt) -> None:  # noqa: ANN001
        # ── pipeline_mode SLEEPING gate ─────────────────────────────
        # SLEEPING means the user explicitly silenced the agent (or it
        # auto-slept). The mic pump already drops audio, but events
        # already queued by SLV (partial / endpoint / final) may still
        # arrive after the sleep call. Honour the gate at the dispatch
        # boundary so a late asr_final can't wake the agent and trigger
        # a new LLM turn.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            if isinstance(evt, (ASRPartial, ASREndpoint)):
                return
            if isinstance(evt, ASRFinal):
                # SLV closed its WS on its side after asr_eos — must
                # still reconnect or the next user turn fails silently
                # with "WS closed mid-send". But DON'T broadcast the
                # utterance or spawn an LLM turn.
                if evt.session_complete:
                    try:
                        await self.slv.reconnect()
                        self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                        self._first_tts_seen = False
                        await self._broadcast(
                            "on_slv_reconnect", {"count": self._slv_reconnect_count}
                        )
                    except Exception:
                        logger.exception("SLV reconnect failed (sleeping)")
                return
            # TTS frames during SLEEPING shouldn't normally arrive (we
            # aborted on sleep), but if they do, ignore — playback is
            # stopped anyway. Fall through for SLVError so we still log.

        if isinstance(evt, ASRPartial):
            # SLV's silero VAD can fire spurious empty endpoints from breath/
            # ambient noise; ignore empty partials so we don't trigger
            # bogus barge-ins or noise the dashboard.
            partial_text = (evt.text or "").strip()
            if not partial_text:
                return
            # Barge-in: user spoke (real text) while we were SPEAKING.
            #
            # We gate on FSM state, NOT audio.is_playing alone. Reason: a
            # TTSDone event flips state SPEAKING→IDLE the moment SLV says
            # "no more frames", but audio_io is still draining buffered
            # PCM through the speaker for another 100-500ms. During that
            # window, a fresh ASRPartial from a NEW user utterance (a
            # legitimate new turn, NOT a barge-in over a still-playing
            # reply) would fire barge-in, abort SLV, and cancel a brand
            # new LLM turn that hadn't even started — wedging the agent
            # into a loop of fake barge-ins on garbled partials. By
            # requiring SPEAKING/BARGED_IN here we treat a partial during
            # IDLE/THINKING as the start of a normal utterance.
            #
            # ALSO: require a MINIMUM partial length AND a minimum delay
            # since TTS began. The reSpeaker XVF3800's hardware AEC isn't
            # perfect — the speaker's first 200-500ms of output bleeds
            # into the open mic, server silero triggers, and the FIRST
            # partial we see is the agent's own "好的。" coming back. If
            # we honour that as barge-in we cancel our own turn before
            # the user could possibly speak. Min length (2 chars) +
            # min delay since TTS started (500ms) suppresses the echo
            # blip; a real barge-in of "Hey Jarvis ..." easily meets both.
            now_ts = time.monotonic()
            barge_min_chars = int(getattr(self.config, "barge_in_min_chars", 2))
            barge_min_speaking_ms = int(getattr(self.config, "barge_in_min_speaking_ms", 500))
            speaking_since = getattr(self, "_speaking_since_ts", 0.0)
            elapsed_ms = (now_ts - speaking_since) * 1000 if speaking_since else 99999
            if len(partial_text) < barge_min_chars:
                logger.debug(
                    "barge-in skipped: partial too short (len=%d < %d): %r",
                    len(partial_text), barge_min_chars, partial_text,
                )
                return
            if elapsed_ms < barge_min_speaking_ms:
                logger.debug(
                    "barge-in skipped: elapsed only %.0fms since TTS start (need >=%dms) — "
                    "treating partial %r as echo",
                    elapsed_ms, barge_min_speaking_ms, partial_text,
                )
                return
            if self._state in (ConvState.SPEAKING, ConvState.BARGED_IN) and self.audio.is_playing:
                logger.info(
                    "BARGE-IN fired (state=%s, partial=%r)",
                    self._state.value, evt.text[:40]
                )
                if self._state == ConvState.SPEAKING:
                    self._set_state(ConvState.BARGED_IN)
                # Cancel any in-flight LLM turn FIRST: otherwise it keeps
                # streaming tokens to SLV which immediately restarts TTS
                # and undoes our barge-in stop_playback below.
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await self._interrupt_current_turn_for_barge_in()
            await self._broadcast("on_user_partial", evt.text)
            return

        if isinstance(evt, ASREndpoint):
            await self._broadcast("on_user_speech_start")
            return

        if isinstance(evt, ASRFinal):
            # A real final arrived — disarm the watchdog so it doesn't
            # later reset state out from under whatever dispatch we're
            # about to run.
            self._cancel_asr_watchdog()
            # Clear the per-turn EOS dedupe flag for ALL final paths
            # (duplicate-of-streamed, empty, and real). Previously it was
            # only reset in the non-empty branch below, so a duplicate or
            # empty final would leave the flag set and the next turn's
            # send_asr_eos_once would early-return → SLV never receives
            # EOS → no final → state stuck THINKING forever (worse than
            # the empty-final bug the watchdog was designed to catch,
            # because the watchdog never even arms).
            self._eos_sent_this_turn = False
            if evt.duplicate_of_streamed:
                # A duplicate final means there is no new utterance to route.
                # If the duplicate is the only final after client-driven EOS,
                # cancelling the watchdog and returning here would strand the
                # FSM in THINKING forever.
                if getattr(self, "_state", ConvState.IDLE) == ConvState.THINKING:
                    logger.info(
                        "duplicate asr_final ignored while THINKING; resetting to IDLE"
                    )
                    self._set_state(ConvState.IDLE)
                    self._reset_sleep_timer()
                return
            # SLV closes the WS after every asr_eos-triggered final
            # (session_complete=True), regardless of whether the final
            # text is empty. Reconnect FIRST, then decide whether the
            # text was worth an LLM turn. If we skipped reconnect on
            # empty finals, the next user utterance would silently fail
            # with "send_json: WS closed mid-send, dropping asr_eos".
            if evt.session_complete:
                try:
                    await self.slv.reconnect()
                    logger.debug("SLV reconnected after session_complete final")
                    self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                    # Reset first-TTS-frame flag so the next turn re-emits the
                    # THINKING→SPEAKING transition cleanly after reconnect.
                    self._first_tts_seen = False
                    await self._broadcast(
                        "on_slv_reconnect", {"count": self._slv_reconnect_count}
                    )
                except Exception:
                    logger.exception("SLV reconnect failed")
            # Drop empty finals — clawd's proven pattern. SLV's server-side
            # VAD or a too-short utterance produces empty text. Treating
            # those as real utterances would call the LLM with no input.
            # Also drop *low-signal* finals (1 visible char or pure
            # interjection / filler): they're almost always ASR noise on
            # an open mic, and feeding them to the LLM is the canonical
            # trigger for an in-context echo loop — the model emits a
            # short "safe" fallback, that fallback enters history, and
            # after 3-4 such turns the small model latches onto the
            # pattern and replies with the same canned line forever.
            stripped_for_signal = _strip_for_signal(evt.text or "")
            if (
                not (evt.text or "").strip()
                or len(stripped_for_signal) <= 1
                or stripped_for_signal in _INTERJECTIONS
            ):
                logger.info(
                    "low-signal asr_final ignored (text=%r, signal=%r, state=%s)",
                    (evt.text or "")[:30], stripped_for_signal,
                    getattr(self, "_state", ConvState.IDLE).name,
                )
                # Clear the discard latch so a later legitimate turn
                # (including dashboard-typed text with no ASRFinal) gets
                # audible TTS — a prior barge-in / abort may have armed it.
                try:
                    arm = getattr(self.audio, "arm_for_next_turn", None)
                    if callable(arm):
                        arm()
                except Exception:  # pragma: no cover - defensive
                    pass
                # State transitions — be conservative. A noise / silence
                # asr_final must NOT cancel an in-flight LLM/TTS turn
                # belonging to a PREVIOUS real utterance. Symptom: with
                # always-on mic, server VAD frequently emits empty
                # asr_finals while the model is mid-tool-call; forcing
                # state back to IDLE here was cancelling the runner, which
                # then re-fired text-only completions, looped tool
                # invocations, and ultimately stranded the agent in a
                # broken speaking↔idle ping-pong that needed a restart.
                #
                # State transitions on low-signal final:
                #   LISTENING — agent was waiting for THIS final and it
                #     turned out to be noise; recover to IDLE.
                #   BARGED_IN — barge-in fired (cancelled TTS+LLM)
                #     expecting the user's actual command; got noise
                #     instead. Treat as cancelled barge-in: clear back
                #     to IDLE so the next real utterance flows cleanly.
                #     Without this the agent stays in BARGED_IN forever
                #     when the barge-in audio produces only short noise
                #     finals ('Yeah.', '头', etc.).
                #   THINKING / SPEAKING — a noise final mid-turn must
                #     NOT cancel the in-flight LLM/TTS belonging to a
                #     PREVIOUS real utterance.
                #   IDLE / SLEEPING — already terminal; no transition.
                cur_state = getattr(self, "_state", ConvState.IDLE)
                if cur_state in (ConvState.LISTENING, ConvState.BARGED_IN):
                    self._set_state(ConvState.IDLE)
                    self._reset_sleep_timer()
                elif cur_state in (ConvState.THINKING, ConvState.SPEAKING):
                    logger.debug(
                        "low-signal final arrived during %s; FSM left alone "
                        "(in-flight turn keeps running)",
                        cur_state.name,
                    )
                return
            logger.info(
                "asr_final received: %r (language=%r)", evt.text, evt.language
            )
            # Re-enable speaker playback for the next turn. stop_playback
            # latched discard=True on the prior barge-in / sleep so SLV's
            # tail-end TTS didn't keep playing; clear that now so the new
            # turn's TTS is actually audible.
            try:
                arm = getattr(self.audio, "arm_for_next_turn", None)
                if callable(arm):
                    arm()
            except Exception:  # pragma: no cover - defensive
                pass
            # New utterance round about to begin — clear client VAD state so
            # the next speech_start fires fresh. (getattr-guarded so tests
            # that build BaseApp via __new__ don't have to set every field.)
            self._vad_state = "idle"
            self._vad_speech_ms = 0
            self._vad_silence_ms = 0
            self._vad_eos_sent = False
            # Allow the NEXT turn to send asr_eos again.
            self._eos_sent_this_turn = False
            _cv = getattr(self, "_client_vad", None)
            if _cv is not None:
                try:
                    _cv.reset()
                except Exception:  # pragma: no cover
                    pass
            await self._broadcast("on_user_utterance", evt.text)
            # (reconnect already happened above, before the empty-text guard)
            # Stop-intent: user said "停下" / "stop" — cancel everything,
            # do NOT route to LLM and do NOT extend session.history (the
            # user asked for quiet, not for more conversation).
            if self._is_stop_intent(evt.text):
                logger.info("stop intent matched: %r", evt.text)
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover - best effort
                    pass
                try:
                    await self.audio.stop_playback()
                except Exception:  # pragma: no cover - best effort
                    pass
                self._set_state(ConvState.IDLE)
                self._reset_sleep_timer()
                await self._broadcast("on_user_stop_intent", evt.text)
                return
            # Spawn the LLM turn as a tracked task so the dispatch loop
            # stays free to handle queued TTSAudio (playback) and
            # ASRPartial (barge-in) while the model streams.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Ensure THINKING fires on the server-VAD path (where
            # _update_vad never runs and no client-side transition has
            # set it). Idempotent for client-VAD path which already set
            # THINKING in _update_vad.
            self._set_state(ConvState.THINKING)
            # Arm the thinking watchdog so a wedged SLV TTS pipeline
            # can't strand the FSM here forever (see _thinking_watchdog
            # docstring). Cancelled by the first tts_started/tts_done
            # event or by SLVError, whichever comes first.
            self._arm_thinking_watchdog()
            self._llm_turn_task = asyncio.create_task(
                self._run_user_utterance(evt.text, evt.language),
                name="llm-turn",
            )
            return

        if isinstance(evt, TTSStarted):
            # Real TTS started — thinking watchdog can stand down.
            self._cancel_thinking_watchdog()
            await self._broadcast("on_assistant_sentence_start", evt.sentence)
            return

        if isinstance(evt, TTSAudio):
            # If we're in BARGED_IN, the tail of SLV's prior-turn TTS is
            # still draining over the WS. Don't reset state to SPEAKING or
            # play the audio (audio.play() also drops it via the discard
            # latch, but skip the state flip here too).
            if self._state == ConvState.BARGED_IN:
                return
            first_frame = False
            if not self._first_tts_seen:
                self._first_tts_seen = True
                first_frame = True
                self.audio.set_output_sample_rate(evt.sample_rate)
                self._set_state(ConvState.SPEAKING)
                # Stamp the moment TTS playback actually began so the
                # barge-in gate (see ASRPartial handler) can suppress the
                # echo blip from the speaker's first 200-500ms output
                # leaking back through the open mic.
                self._speaking_since_ts = time.monotonic()
            await self._broadcast(
                "on_tts_audio_frame",
                {
                    "sample_rate": evt.sample_rate,
                    "frame_len": len(evt.pcm),
                    "first": first_frame,
                },
            )
            await self.audio.play(evt.pcm)
            return

        if isinstance(evt, TTSSentenceDone):
            await self._broadcast("on_assistant_sentence", evt.sentence)
            return

        if isinstance(evt, TTSDone):
            # Reset first-frame flag so the NEXT turn re-emits SPEAKING.
            self._first_tts_seen = False
            self._cancel_thinking_watchdog()
            # Authoritative is_playing reset (audio_io stopped doing this on
            # transient empty queue to keep barge-in checks reliable).
            mark = getattr(self.audio, "mark_playback_done", None)
            if callable(mark):
                mark()
            # Don't override BARGED_IN: the user is mid-utterance and the
            # VAD silence-end / ASRFinal path will drive state forward.
            # Forcing IDLE here would also kick the auto-sleep timer in
            # push_to_talk mode while the user is still speaking.
            if self._state != ConvState.BARGED_IN:
                self._set_state(ConvState.IDLE)
                self._reset_sleep_timer()
            await self._broadcast("on_assistant_done")
            # Proactive reconnect after TTS done. On serialized profiles
            # (e.g. jetson-qwen3asr-matcha-nx) the ASR worker shares a
            # coord lock with TTS; observed in voice-arm validation that
            # silero VAD on the server stops firing speech_start after
            # the first tts_done — loud mic audio (RMS 0.12+) goes
            # nowhere, no SLV event arrives, agent goes to sleep on the
            # next sleep_timeout. A fresh WS forces a clean ASR session.
            #
            # The earlier removal of this reconnect (commit 1acf7f3) was
            # because the SLV WS session-limiter would reject reconnects
            # that landed inside the previous session's 3-40 ms teardown.
            # That race is now handled by SLVClient.reconnect()'s
            # 50 ms grace + backoff retry (commit 7e1b8c8), so reconnect
            # here is safe again.
            try:
                await asyncio.wait_for(self.slv.reconnect(), timeout=4.0)
                self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                self._first_tts_seen = False
                self._eos_sent_this_turn = False
                await self._broadcast(
                    "on_slv_reconnect", {"count": self._slv_reconnect_count}
                )
            except asyncio.TimeoutError:
                logger.warning("SLV reconnect timed out after tts_done")
            except Exception:
                logger.exception("SLV reconnect failed after tts_done")
            return

        if isinstance(evt, SLVError):
            # Transport died — any pending asr_final / thinking watchdog
            # is moot; SLVError handling below already drives state
            # back to IDLE.
            self._cancel_asr_watchdog()
            self._cancel_thinking_watchdog()
            old_state = getattr(self, "_state", ConvState.IDLE)
            await self._broadcast(
                "on_error",
                TypedLLMError(
                    "slv_error",
                    evt.message,
                    exc_class="SLVError",
                ),
            )
            # Don't leave the FSM stuck in THINKING/SPEAKING after a transport
            # error — cancel any in-flight LLM turn.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            # If we were SLEEPING when the transport error fired, stay
            # SLEEPING — a transport hiccup must never wake the agent
            # (would hot-mic in wake_word mode).
            if old_state != ConvState.SLEEPING:
                self._set_state(ConvState.IDLE)
            else:
                logger.info("SLVError while SLEEPING; staying SLEEPING")
            # No proactive reconnect here — same race as TTSDone (the SLV
            # server immediately closes a fresh connection that arrives
            # while it's still tearing the previous session down).
            # ``SLVClient.send_audio`` / ``_send_json`` already auto-call
            # ``connect()`` when ``_ws is None``, and the dead-WS detection
            # in those paths nulls _ws on ``ConnectionClosed``. The next
            # mic chunk (or text send) naturally reopens the transport
            # ~100ms later — enough headroom for SLV to finalize the prior
            # session cleanly.
            return

    async def _run_user_utterance(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """Wrap on_user_utterance so a crashing LLM turn doesn't kill the task silently."""
        try:
            await self.on_user_utterance(text, detected_language=detected_language)
            # Success path: tell the availability plugin so a transient
            # failure that earlier flipped us to DEGRADED gets cleared.
            # Skip if LLM is disabled (noop backend).
            if isinstance(self.llm, NoopLLM):
                return
            avail = getattr(self, "llm_availability", None)
            if avail is not None:
                try:
                    avail.report_request_success()
                except Exception:  # pragma: no cover - defensive
                    pass
        except NotImplementedError:
            logger.error("BaseApp.on_user_utterance not overridden -- text dropped")
            self._set_state(ConvState.IDLE)
        except asyncio.CancelledError:
            # Cancellation happens on barge-in / shutdown / stop-intent;
            # caller already drove the appropriate state transition.
            raise
        except LLMUnavailable as e:
            # Fail-fast path: the availability state machine already
            # decided the LLM is DOWN. Don't bother A3 retry — surface
            # to the dashboard and return to IDLE immediately.
            logger.warning("LLM unavailable, fail-fast: %s", e)
            try:
                await self._broadcast(
                    "on_error",
                    TypedLLMError(
                        "llm_unavailable",
                        f"LLM 不可用：{e}",
                        exc_class=type(e).__name__,
                    ),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass
            return
        except LLMTimeoutError as e:
            logger.warning(
                "LLM %s timeout after %.1fs (partial=%r)",
                e.kind, e.timeout_s, e.partial_text[:80],
            )
            # Real-world failure — push the state machine forward without
            # waiting for the next probe.
            avail = getattr(self, "llm_availability", None)
            if avail is not None:
                try:
                    avail.report_request_failure()
                except Exception:  # pragma: no cover - defensive
                    pass
            msg = (
                f"LLM 响应超时（{e.kind}, >{e.timeout_s:.0f}s）。"
                "可能 edge-llm 服务挂了或输入太长。"
            )
            try:
                await self._broadcast(
                    "on_error",
                    TypedLLMError(
                        "llm_timeout",
                        msg,
                        exc_class="LLMTimeoutError",
                        kind=e.kind,
                        timeout_s=e.timeout_s,
                    ),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass
        except Exception as e:
            logger.exception("on_user_utterance failed")
            # Real-world failure — feed back to the availability machine
            # (only for LLM-class errors; other exceptions might be local
            # bugs and shouldn't poison the breaker).
            try:
                from openai import APIError as _APIError
                _is_llm_err = isinstance(e, (_APIError, LLMStreamError))
            except Exception:  # pragma: no cover - defensive
                _is_llm_err = isinstance(e, LLMStreamError)
            if _is_llm_err:
                avail = getattr(self, "llm_availability", None)
                if avail is not None:
                    try:
                        avail.report_request_failure()
                    except Exception:  # pragma: no cover - defensive
                        pass
            # A3: surface non-timeout LLM failures to the dashboard so
            # operators see *something* when edge-llm crashes or returns
            # a 4xx. Wrap the original exception's repr into a clean
            # RuntimeError (the on_error contract already accepts a
            # BaseException and prefers str()).
            try:
                exc_class = type(e).__name__
                msg = f"LLM 调用失败（{exc_class}）：{e}"
                err_type = (
                    "llm_stream_error"
                    if isinstance(e, LLMStreamError)
                    else "llm_failure"
                )
                await self._broadcast(
                    "on_error",
                    TypedLLMError(err_type, msg, exc_class=exc_class),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass

    async def broadcast(self, hook_name: str, *args) -> None:
        """Public hook broadcaster -- call from subclasses to fan out events.

        Used by DialogueApp.on_user_utterance to fan out per-token deltas
        (`on_assistant_token`) since the dispatch loop has no access to
        the LLM's token stream.
        """
        plugins = getattr(self, "plugins", None)
        if not plugins:
            return
        coros = []
        for p in plugins:
            fn = getattr(p, hook_name, None)
            if fn is None:
                continue
            coros.append(_safe_call(p.name, hook_name, fn, *args))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    # Backwards-compatible alias used internally by the dispatch loop.
    _broadcast = broadcast


async def _safe_call(plugin_name: str, hook: str, fn, *args) -> None:  # noqa: ANN001
    try:
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("plugin %s.%s failed", plugin_name, hook)


__all__ = ["BaseApp"]
