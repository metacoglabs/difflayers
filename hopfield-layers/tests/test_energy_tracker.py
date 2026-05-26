"""
tests/test_energy_tracker.py
P0-B / P1-A verification: EnergyTracker with dense (L) and factored (W,deg) modes.
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


def _run_engine(N=16, d=8, T=4, mode="simple", track_energy=True):
    cfg   = DiffusionConfig(k_neighbors=4, eta=0.1, diffusion_mode=mode,
                            steps=T, cache_graph=True)
    cache = GraphCache(cfg)
    X     = torch.randn(N, d)
    W, deg, adj, L, op = cache.get(X)

    tracker = EnergyTracker(beta=1.0, eta=0.1, tol=0.0) if track_energy else None
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
    return tracker


def test_energy_tracker_non_empty_simple():
    """EnergyTracker.records must be non-empty after simple-mode run."""
    tracker = _run_engine(mode="simple")
    assert len(tracker.records) > 0, (
        "EnergyTracker.records should be non-empty after run_dynamics (simple mode)"
    )


def test_energy_tracker_non_empty_factored():
    """EnergyTracker.records must be non-empty in factored mode (BUG-03 fix)."""
    tracker = _run_engine(mode="factored")
    assert len(tracker.records) > 0, (
        "EnergyTracker.records should be non-empty after run_dynamics (factored mode)"
    )


def test_energy_tracker_has_T_entries():
    """Number of energy records must equal T (no premature early stop at tol=0)."""
    T = 5
    tracker = _run_engine(T=T, mode="simple")
    assert len(tracker.records) == T, (
        f"Expected {T} records, got {len(tracker.records)}"
    )


def test_energy_records_alias():
    """EnergyTracker.records and .history must return the same values."""
    tracker = EnergyTracker(beta=1.0, eta=0.1)
    tracker._history.extend([1.0, 0.8, 0.7])
    assert tracker.records == tracker.history
    assert tracker.records == [1.0, 0.8, 0.7]


def test_energy_monotone_factored_multiple_seeds():
    """E(t) non-increasing on ≥ 90% of 10 seeds in factored mode (Theorem 2)."""
    N, d, T = 12, 8, 5
    monotone = 0
    for seed in range(10):
        torch.manual_seed(seed)
        tracker = _run_engine(N=N, d=d, T=T, mode="factored")
        E = tracker.history
        if all(E[i] >= E[i + 1] - 0.05 for i in range(len(E) - 1)):
            monotone += 1
    assert monotone >= 9, (
        f"Energy monotone on {monotone}/10 seeds (factored); expected ≥ 9"
    )
