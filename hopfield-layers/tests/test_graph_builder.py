"""
tests/test_graph_builder.py
P2-C: GraphCache and GraphBuilder correctness tests.
"""
import pytest
import torch

from difflayers.dynamics_engine import (
    DiffusionConfig,
    GraphCache,
    _MODULE_GRAPH_CACHE,
    clear_module_graph_cache,
)


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Ensure the module-level graph cache starts empty for each test."""
    clear_module_graph_cache()
    yield
    clear_module_graph_cache()


def _make_cache(k: int = 5, eta: float = 0.1, mode: str = "factored") -> GraphCache:
    cfg = DiffusionConfig(k_neighbors=k, eta=eta, diffusion_mode=mode, cache_graph=True)
    return GraphCache(cfg)


def test_adjacency_symmetry():
    """W must be symmetric (W = Wᵀ) — required for a valid Laplacian."""
    cache = _make_cache()
    X = torch.randn(20, 16)
    W, deg, adj, L, op = cache.get(X)
    W_dense = W.to_dense() if W.is_sparse else W
    assert torch.allclose(W_dense, W_dense.t(), atol=1e-5), (
        "Adjacency W is not symmetric"
    )


def test_adjacency_non_negative():
    """W_ij ≥ 0 — required for a valid Laplacian (PRD §9.1)."""
    cache = _make_cache()
    X = torch.randn(20, 16)
    W, *_ = cache.get(X)
    W_dense = W.to_dense() if W.is_sparse else W
    assert (W_dense >= -1e-7).all(), "Adjacency W contains negative entries"


def test_cache_hit_on_same_data_ptr():
    """GraphCache must return the same operator object for identical X (P0-A)."""
    cache = _make_cache()
    X = torch.randn(20, 16)
    _, _, _, _, op1 = cache.get(X)
    _, _, _, _, op2 = cache.get(X)
    assert op1 is op2, (
        "GraphCache must return the cached operator on repeated calls with same X"
    )


def test_module_cache_populated_after_first_call():
    """P0-A: module-level cache is populated after the first slow-path build."""
    assert len(_MODULE_GRAPH_CACHE) == 0
    cache = _make_cache()
    X = torch.randn(20, 16)
    cache.get(X)
    assert len(_MODULE_GRAPH_CACHE) == 1, (
        "Module-level graph cache should have 1 entry after first build"
    )


def test_module_cache_cross_instance_hit():
    """P0-A: second DiffusedHopfield instance shares cached graph objects."""
    cache1 = _make_cache()
    cache2 = _make_cache()
    X = torch.randn(20, 16)

    _, _, _, _, op1 = cache1.get(X)
    _, _, _, _, op2 = cache2.get(X)

    # Both should have pulled from the module cache — operators identical
    assert op1 is op2, (
        "Two GraphCache instances with same X must return the same cached operator"
    )
    # Module cache should still have only 1 entry (no double-build)
    assert len(_MODULE_GRAPH_CACHE) == 1


def test_clear_module_cache():
    """clear_module_graph_cache() evicts all entries."""
    cache = _make_cache()
    X = torch.randn(20, 16)
    cache.get(X)
    assert len(_MODULE_GRAPH_CACHE) > 0
    clear_module_graph_cache()
    assert len(_MODULE_GRAPH_CACHE) == 0
