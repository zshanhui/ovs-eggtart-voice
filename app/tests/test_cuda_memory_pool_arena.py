"""Unit tests for the arena-backed CudaMemoryPool.

The arena turns ``allocate()`` into a bump-pointer sub-allocation against a
single pre-allocated chunk of device memory. We exercise this entirely with a
mocked ``cuda.cudart`` module so the test runs on any host (Mac dev, CI,
Jetson).
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.backends.jetson.matcha_trt import CudaMemoryPool, _read_arena_size_bytes


@pytest.fixture
def fake_cudart(monkeypatch):
    """Install a stub ``cuda.cudart`` that tracks malloc/free + stream calls."""

    class _Enum:
        cudaSuccess = 0

    class _Kind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    state = {
        "next_ptr": 0x10000,
        "mallocs": [],  # list[(ptr, size)]
        "frees": [],
        "stream_creates": 0,
        "stream_destroys": [],
    }

    def cudaMalloc(size):
        ptr = state["next_ptr"]
        state["next_ptr"] += max(size, 1)
        state["mallocs"].append((ptr, int(size)))
        return _Enum.cudaSuccess, ptr

    def cudaFree(ptr):
        state["frees"].append(int(ptr))
        return _Enum.cudaSuccess

    def cudaStreamCreate():
        state["stream_creates"] += 1
        return _Enum.cudaSuccess, 0xABCD0000

    def cudaStreamDestroy(handle):
        state["stream_destroys"].append(int(handle))
        return _Enum.cudaSuccess

    def cudaStreamSynchronize(_handle):
        return _Enum.cudaSuccess

    fake = types.SimpleNamespace(
        cudaMalloc=cudaMalloc,
        cudaFree=cudaFree,
        cudaStreamCreate=cudaStreamCreate,
        cudaStreamDestroy=cudaStreamDestroy,
        cudaStreamSynchronize=cudaStreamSynchronize,
        cudaError_t=_Enum,
        cudaMemcpyKind=_Kind,
    )
    mod = types.ModuleType("cuda")
    mod.cudart = fake  # type: ignore[attr-defined]
    cudart_mod = types.ModuleType("cuda.cudart")
    for attr in (
        "cudaMalloc",
        "cudaFree",
        "cudaStreamCreate",
        "cudaStreamDestroy",
        "cudaStreamSynchronize",
        "cudaError_t",
        "cudaMemcpyKind",
    ):
        setattr(cudart_mod, attr, getattr(fake, attr))
    monkeypatch.setitem(sys.modules, "cuda", mod)
    monkeypatch.setitem(sys.modules, "cuda.cudart", cudart_mod)
    return state


def test_read_arena_size_specific_env_wins(monkeypatch):
    monkeypatch.setenv("OVS_TEST_ARENA_MB", "8")
    monkeypatch.setenv("OVS_CUDA_ARENA_SIZE_MB", "32")
    assert _read_arena_size_bytes("OVS_TEST_ARENA_MB") == 8 * 1024 * 1024


def test_read_arena_size_falls_back_to_generic(monkeypatch):
    monkeypatch.delenv("OVS_TEST_ARENA_MB", raising=False)
    monkeypatch.setenv("OVS_CUDA_ARENA_SIZE_MB", "24")
    assert _read_arena_size_bytes("OVS_TEST_ARENA_MB") == 24 * 1024 * 1024


def test_read_arena_size_default(monkeypatch):
    monkeypatch.delenv("OVS_TEST_ARENA_MB", raising=False)
    monkeypatch.delenv("OVS_CUDA_ARENA_SIZE_MB", raising=False)
    assert _read_arena_size_bytes("OVS_TEST_ARENA_MB", default_mb=16) == 16 * 1024 * 1024


def test_arena_single_cudaMalloc(fake_cudart):
    """Arena should call cudaMalloc once (for the arena) regardless of how many
    sub-allocations the caller does."""
    pool = CudaMemoryPool(arena_size_bytes=64 * 1024)
    p1 = pool.allocate(1000)
    p2 = pool.allocate(2000)
    p3 = pool.allocate(500)
    # One cudaMalloc for the arena itself; sub-allocations don't hit cudart.
    assert len(fake_cudart["mallocs"]) == 1
    assert fake_cudart["mallocs"][0][1] == 64 * 1024
    # Pointers must be distinct + aligned + within arena.
    assert p1 != p2 != p3
    arena_base = fake_cudart["mallocs"][0][0]
    for p in (p1, p2, p3):
        assert (p - arena_base) % 256 == 0


def test_arena_reuse_across_free_all_cycles(fake_cudart):
    """free_all() resets the bump offset but does NOT free the arena."""
    pool = CudaMemoryPool(arena_size_bytes=64 * 1024)
    p1 = pool.allocate(4096)
    pool.free_all()
    p2 = pool.allocate(4096)
    # cudaMalloc still called only once for the arena; cudaFree NOT called.
    assert len(fake_cudart["mallocs"]) == 1
    assert fake_cudart["frees"] == []
    # Same offset → same pointer.
    assert p1 == p2


def test_arena_overflow_falls_back_to_cuda_malloc(fake_cudart):
    """Allocations that don't fit in the arena route to per-call cudaMalloc and
    are released by free_all()."""
    pool = CudaMemoryPool(arena_size_bytes=4 * 1024)
    p_small = pool.allocate(1024)
    p_big = pool.allocate(16 * 1024)  # > arena → overflow
    # Two cudaMallocs: arena + overflow.
    assert len(fake_cudart["mallocs"]) == 2
    assert fake_cudart["mallocs"][1][1] == 16 * 1024
    assert pool._overflow_count == 1
    assert pool._overflow_bytes == 16 * 1024
    assert p_small != p_big
    pool.free_all()
    # Overflow chunk freed, arena NOT freed.
    assert fake_cudart["frees"] == [p_big]
    assert pool._overflow_allocs == []


def test_destroy_frees_arena_plus_overflow_plus_stream(fake_cudart):
    pool = CudaMemoryPool(arena_size_bytes=4 * 1024)
    pool.allocate(1024)              # arena
    p_overflow = pool.allocate(8192)  # overflow
    arena_base = fake_cudart["mallocs"][0][0]
    pool.destroy()
    # cudaFree was called for both the overflow chunk and the arena ptr.
    assert p_overflow in fake_cudart["frees"]
    assert arena_base in fake_cudart["frees"]
    # Stream destroyed.
    assert fake_cudart["stream_destroys"] == [0xABCD0000]
    # State reset.
    assert pool._arena_ptr is None
    assert pool._stream is None
    assert pool._initialized is False
    # Idempotent.
    pool.destroy()


def test_peak_telemetry_updates(fake_cudart):
    pool = CudaMemoryPool(arena_size_bytes=1024 * 1024)
    pool.allocate(1000)
    pool.allocate(2000)
    peak_after_2 = pool._peak_offset
    pool.free_all()
    # Peak is the high-water mark, NOT reset by free_all.
    assert pool._peak_offset == peak_after_2
    pool.allocate(500)
    assert pool._peak_offset == peak_after_2  # smaller alloc doesn't lower peak


def test_legacy_mode_without_arena(fake_cudart):
    """arena_size_bytes=None preserves the original cudaMalloc-per-call path
    (back-compat for tests / consumers that drive the pool directly)."""
    pool = CudaMemoryPool()  # no arena
    p1 = pool.allocate(1024)
    p2 = pool.allocate(2048)
    assert len(fake_cudart["mallocs"]) == 2
    assert p1 in pool._allocations and p2 in pool._allocations
    pool.free_all()
    assert sorted(fake_cudart["frees"]) == sorted([p1, p2])
    assert pool._allocations == []
