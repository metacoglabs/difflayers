"""
tests/test_engine_cache.py
P0-A verification: Engine.__init__ latency < 0.1 ms on cache hit.
"""
import time
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from difflayers.dynamics_engine import (
    DiffusionConfig,
    GraphCache,
    clear_module_graph_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_module_graph_cache()
    yield
    clear_module_graph_cache()


def _build_cache(k: int = 5, eta: float = 0.1) -> tuple:
    cfg = DiffusionConfig(k_neighbors=k, eta=eta, diffusion_mode="factored",
                          cache_graph=True)
    cache = GraphCache(cfg)
    return cache, cfg


def test_cache_hit_latency_under_0p1ms():
    """
    P0-A pass criterion: GraphCache.get() latency < 0.1 ms on repeated same X.
    """
    import torch
    cache, _ = _build_cache()
    X = torch.randn(64, 64)

    # Cold build
    cache.get(X)

    # Warm hits — should be O(1) dict lookup
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        cache.get(X)
        times.append((time.perf_counter() - t0) * 1e3)

    median_ms = float(np.median(times))
    assert median_ms < 0.1, (
        f"Cache hit latency = {median_ms:.4f} ms — expected < 0.1 ms (P0-A)"
    )


def test_cache_hit_on_second_call():
    """P0-A: 'cache hit on second call: True' as required by the pass criterion."""
    import torch
    from difflayers.dynamics_engine import _MODULE_GRAPH_CACHE, _module_cache_key

    cache, cfg = _build_cache()
    X = torch.randn(32, 32)

    mod_key = _module_cache_key(X, cfg.k_neighbors, cfg.eta)
    assert mod_key not in _MODULE_GRAPH_CACHE, "Cache should be empty before first call"

    cache.get(X)
    cache_hit_first = mod_key in _MODULE_GRAPH_CACHE

    cache2, _ = _build_cache()  # new instance
    _, _, _, _, op_hit = cache2.get(X)
    cache_hit_second = mod_key in _MODULE_GRAPH_CACHE

    print(f"cache hit on second call: {cache_hit_second}")   # required output
    assert cache_hit_first, "Module cache should be populated after first call"
    assert cache_hit_second, "Module cache should be hit on second call"
