"""
tests/test_dynamics_wiring.py
P0-B verification: DiffusedHopfield._associate calls run_dynamics, not a shortcut.
"""
import pytest
import torch

from difflayers.dynamics_engine import (
    EnergyTracker,
    DynamicsEngine,
    DiffusionConfig,
    GraphCache,
    clear_module_graph_cache,
)
from difflayers.attention_operator import AttentionOperator


@pytest.fixture(autouse=True)
def _clear():
    clear_module_graph_cache()
    yield
    clear_module_graph_cache()


def test_run_dynamics_not_shortcut():
    """
    DiffusedHopfield._associate must call run_dynamics with the full T-step loop.

    Verified by: EnergyTracker accumulates T entries, not 0 or 1.
    """
    T = 3
    N, d = 16, 8

    cfg   = DiffusionConfig(k_neighbors=4, eta=0.1, diffusion_mode="simple",
                            steps=T, cache_graph=True)
    cache = GraphCache(cfg)
    X     = torch.randn(N, d)
    W, deg, adj, L, op = cache.get(X)

    tracker = EnergyTracker(beta=1.0, eta=0.1, tol=0.0)
    engine  = DynamicsEngine(
        diffusion_op=op,
        attention_op=AttentionOperator(beta=1.0, mode="dense"),
        steps=T,
        energy_tracker=tracker,
    )

    Q = torch.randn(N, d)
    K = X.clone()
    V = torch.randn(N, d)
    engine.run_dynamics(Q, K, V, adj_indices=adj, L=L, W=W, deg=deg,
                        diffuse_query=False, diffuse_key=True)

    assert len(tracker.records) == T, (
        f"run_dynamics must call the full T={T} step loop; "
        f"got {len(tracker.records)} energy records"
    )


def test_energy_tracker_records_non_empty_after_forward():
    """energy_tracker.records is non-empty after a single forward pass (P0-B)."""
    N, d, T = 12, 8, 4
    cfg   = DiffusionConfig(k_neighbors=4, eta=0.1, diffusion_mode="factored",
                            steps=T, cache_graph=True)
    cache = GraphCache(cfg)
    X     = torch.randn(N, d)
    W, deg, adj, L, op = cache.get(X)

    tracker = EnergyTracker(beta=1.0, eta=0.1, tol=0.0)
    engine  = DynamicsEngine(
        diffusion_op=op,
        attention_op=AttentionOperator(beta=1.0, mode="dense"),
        steps=T,
        energy_tracker=tracker,
    )

    Q = torch.randn(N, d)
    engine.run_dynamics(Q, X.clone(), torch.randn(N, d),
                        adj_indices=adj, L=None, W=W, deg=deg,
                        diffuse_query=False, diffuse_key=True)

    assert len(tracker.records) > 0, (
        "energy_tracker.records must be non-empty after forward pass"
    )
