"""
tests/test_attention_operator.py
P2-C: AttentionOperator correctness, graph mode reachability, accuracy gap.
"""
import pytest
import warnings
import torch
import torch.nn.functional as F

from difflayers.attention_operator import AttentionOperator
from difflayers.dynamics_engine import DynamicsEngine, DiffusionConfig, GraphCache


# ---------------------------------------------------------------------------
# P1-C: graph mode is reachable (BUG-02 fix)
# ---------------------------------------------------------------------------

def test_graph_mode_reachable():
    """attend(mode='graph') must not raise — graph path not dead code (BUG-02)."""
    N, d, k = 20, 8, 4
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    adj = torch.randint(0, N, (N, k))

    op = AttentionOperator(beta=1.0, mode="dense")
    # Explicitly route to graph via attend()
    out = op.attend(Q, K, V, adj_indices=adj, mode="graph")
    assert out.shape == (N, d), f"Expected ({N},{d}), got {out.shape}"


def test_graph_force_bypasses_fallback():
    """mode='graph_force' must route to graph even at N < 512."""
    N, d, k = 30, 8, 4
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    adj = torch.randint(0, N, (N, k))

    op = AttentionOperator(beta=1.0, mode="graph_force")
    out = op.attend(Q, K, V, adj_indices=adj, mode="graph_force")
    assert out.shape == (N, d)


def test_attend_dense_matches_manual():
    """attend(mode='dense') must match hand-computed softmax attention."""
    N, d = 10, 8
    beta = 2.0
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    op  = AttentionOperator(beta=beta, mode="dense")
    out = op.attend(Q, K, V, mode="dense")

    logits  = beta * (Q @ K.t())
    weights = F.softmax(logits, dim=-1)
    ref     = weights @ V

    assert torch.allclose(out, ref, atol=1e-5), (
        f"Dense attention output differs from manual: max diff "
        f"{(out - ref).abs().max():.2e}"
    )


def test_n_lt_512_fallback_warning():
    """
    DynamicsEngine.run_dynamics must emit a RuntimeWarning and fall back to dense
    when mode='graph' is requested at N < 512 (P1-C).
    """
    N, d = 32, 16
    cfg   = DiffusionConfig(k_neighbors=4, eta=0.1, diffusion_mode="factored",
                            attention_mode="graph", cache_graph=True)
    cache = GraphCache(cfg)
    X     = torch.randn(N, d)
    W, deg, adj, L, op = cache.get(X)

    attn_op = AttentionOperator(beta=1.0, mode="graph")
    engine  = DynamicsEngine(diffusion_op=op, attention_op=attn_op, steps=1)

    Q = torch.randn(N, d)
    K = X.clone()
    V = torch.randn(N, d)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.run_dynamics(Q, K, V, adj_indices=adj, L=None, W=W, deg=deg,
                            diffuse_query=False, diffuse_key=True)

    fallback_warnings = [
        w for w in caught
        if "512" in str(w.message) or "break-even" in str(w.message).lower()
        or "falling back" in str(w.message).lower()
    ]
    assert len(fallback_warnings) >= 1, (
        "Expected a RuntimeWarning about graph→dense fallback at N < 512"
    )


# ---------------------------------------------------------------------------
# P1-C: dense vs graph accuracy gap < 5% at N ≥ 512
# ---------------------------------------------------------------------------

def test_dense_graph_accuracy_gap_large_n():
    """
    Dense vs. graph accuracy gap must be < 5% at N=512, k=16 (PRD §9.4).

    Accuracy here: cosine similarity of outputs averaged over patterns.
    """
    N, d, k = 512, 32, 16
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    # Build kNN adjacency using top-k cosine similarity
    sim    = Q @ K.t()                     # (N, N)
    _, adj = sim.topk(k, dim=-1)           # (N, k)

    op   = AttentionOperator(beta=1.0, mode="dense")
    out_dense = op.attend(Q, K, V, mode="dense")
    out_graph = op.attend(Q, K, V, adj_indices=adj, mode="graph_force")

    # Cosine similarity between output rows
    cos_sim = F.cosine_similarity(out_dense, out_graph, dim=-1).mean().item()
    gap     = 1.0 - cos_sim
    assert gap < 0.05, (
        f"Dense vs graph accuracy gap = {gap:.3f} (> 0.05 threshold) at N={N}, k={k}"
    )
