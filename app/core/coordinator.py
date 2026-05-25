"""Backend execution coordinator.

execution_policy in profile JSON drives this:
- concurrent  : no lock, ASR and TTS run in parallel
- serialized  : single asyncio.Lock shared by both slots; mutually exclusive
- exclusive   : same lock + slot tracking; switching slot calls dormant
                backend.unload() before yielding. Best-effort: backends not
                overriding unload() stay resident.

Spec docs/specs/concurrency-capability-framework.md §4: backend capability
is the ceiling, profile.execution_policy is the floor. ``concurrent`` is
permitted only when BOTH active backends declare
``supports_parallel=True`` with ``max_concurrent > 1``. If either is
``supports_parallel=False``, the mode is downgraded to ``serialized``.
Profile ``exclusive`` is always honored as-is.
"""
from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Dict, Literal, Optional

logger = logging.getLogger(__name__)

Slot = Literal["asr", "tts"]


def _resolve_mode(policy: dict, profile: Optional[dict]) -> str:
    """Resolve the effective coordinator mode per spec §4.

    ``exclusive`` is unconditional; otherwise we start from
    ``policy.mode`` (default ``concurrent``) and downgrade to
    ``serialized`` if either active backend declares
    ``supports_parallel=False`` or ``max_concurrent <= 1``.

    ``profile`` is optional; without it (legacy callers / tests) we
    return the raw ``policy.mode`` to preserve behavior.
    """
    requested = (policy or {}).get("mode", "concurrent")
    if requested == "exclusive":
        return "exclusive"
    if not isinstance(profile, dict):
        return requested

    try:
        from app.core.asr_backend import _ASR_REGISTRY
        from app.core.tts_backend import _TTS_REGISTRY
        from app.core.concurrency_capability import ConcurrencyCapability
    except Exception:
        return requested

    def _cap(key: str, registry: dict) -> ConcurrencyCapability:
        spec = profile.get(key)
        if not spec or spec not in registry:
            return ConcurrencyCapability.default()
        module_path, cls_name = registry[spec]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            return cls.concurrency_capability(profile)
        except Exception as exc:
            logger.debug(
                "coordinator: capability lookup failed for %s=%s: %s",
                key, spec, exc,
            )
            return ConcurrencyCapability.default()

    asr_cap = _cap("asr_backend", _ASR_REGISTRY)
    tts_cap = _cap("tts_backend", _TTS_REGISTRY)

    def _parallel_ok(cap: ConcurrencyCapability) -> bool:
        if not cap.supports_parallel:
            return False
        # None means "no hard cap" (e.g. per-stream paraformer) -> parallel.
        if cap.max_concurrent is None:
            return True
        return cap.max_concurrent > 1

    if requested == "concurrent" and not (_parallel_ok(asr_cap) and _parallel_ok(tts_cap)):
        logger.info(
            "coordinator: downgrading concurrent -> serialized "
            "(asr.supports_parallel=%s/max=%s, tts.supports_parallel=%s/max=%s)",
            asr_cap.supports_parallel, asr_cap.max_concurrent,
            tts_cap.supports_parallel, tts_cap.max_concurrent,
        )
        return "serialized"
    return requested


class BackendCoordinator:
    def __init__(self, policy: dict, profile: Optional[dict] = None):
        self._mode = _resolve_mode(policy, profile)
        self._lock: Optional[asyncio.Lock] = None
        if self._mode in ("serialized", "exclusive"):
            self._lock = asyncio.Lock()
        self._active_slot: Optional[Slot] = None
        # store callables to fetch backends lazily (set after services start)
        self._backend_getters: Dict[Slot, Callable] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def register_backend(self, slot: Slot, getter: Callable):
        """Register a callable returning the currently-loaded backend for the slot."""
        self._backend_getters[slot] = getter

    @asynccontextmanager
    async def acquire(self, slot: Slot) -> AsyncIterator[None]:
        if self._mode == "concurrent" or self._lock is None:
            yield
            return
        async with self._lock:
            if self._mode == "exclusive" and self._active_slot not in (None, slot):
                # unload the previously active slot's backend if available
                other = self._active_slot
                getter = self._backend_getters.get(other)
                if getter is not None:
                    backend = getter()
                    if backend is not None and hasattr(backend, "unload"):
                        backend.unload()
            self._active_slot = slot
            yield


_coordinator: Optional[BackendCoordinator] = None


def init_coordinator(
    policy: dict, profile: Optional[dict] = None
) -> BackendCoordinator:
    global _coordinator
    _coordinator = BackendCoordinator(policy, profile=profile)
    return _coordinator


def get_coordinator() -> BackendCoordinator:
    if _coordinator is None:
        raise RuntimeError("coordinator not initialized; call init_coordinator() at startup")
    return _coordinator
