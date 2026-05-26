"""
tests/test_diffusion_ops.py
P2-C: DiffusionOperator correctness, spectral cache, over-smoothing guard.
"""
import pytest
import warnings
import torch

from difflayers.diffusion import (
    SimpleDiffusion,
    IterativeDiffusion,
    SpectralDiffusion,
    FactoredDiffusion,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_laplacian(N: int = 16) -> torch.Tensor:
    """Build a simple cycle-graph Laplacian for reproducible tests."""
    L = torch.zeros(N, N)
    for i in range(N):
        j = (i + 1) % N
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0
    return L


def _make_adj_deg(N: int = 16, k: int = 4) -> tuple:
    """Return (W, deg) for a random k-regular graph."""
    W = torch.zeros(N, N)
    for i in range(N):
        neighbors = torch.randperm(N)[:k]
        for j in neighbors:
            if j != i:
                W[i, j] = 1.0
                W[j, i] = 1.0
    deg = W.sum(dim=1)
    return W, deg


# ---------------------------------------------------------------------------
# Operator spectral norm ≤ 1  (stability condition, PRD §9.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["simple", "iterative"])
def test_diffusion_operator_norm(mode):
    """‖D‖₂ ≤ 1 + ε for simple and iterative (spectral stability condition)."""
    N = 16
    L = _make_laplacian(N)
    op = (SimpleDiffusion(eta=0.1) if mode == "simple"
          else IterativeDiffusion(eta=0.1, steps=1))
    op.precompute(L)
    D = op._op
    # spectral norm via largest singular value
    sigma_max = torch.linalg.norm(D, ord=2).item()
    assert sigma_max <= 1.05, (
        f"{mode}: ‖D‖₂ = {sigma_max:.4f} > 1.05 — operator is not contractive"
    )


# ---------------------------------------------------------------------------
# Simple vs Factored agreement
# ---------------------------------------------------------------------------

def test_simple_vs_factored_agreement():
    """simple and factored must agree on D·K to tolerance 1e-4 (PRD §9.2)."""
    N, d = 20, 8
    L = _make_laplacian(N)
    eta = 0.1

    # Simple: D = I - eta*L, one step
    simple = SimpleDiffusion(eta=eta, steps=1).precompute(L)

    # Factored: (1-eta*deg)⊙K + eta*W@K
    diag_vals = L.diagonal()
    W = torch.diag(diag_vals) - L
    deg = diag_vals.clone()
    factored = FactoredDiffusion(eta=eta, steps=1).precompute_from_graph(W, deg)

    X = torch.randn(N, d)
    out_simple  = simple(X)
    out_factored = factored(X)

    assert torch.allclose(out_simple, out_factored, atol=1e-4), (
        f"Simple vs Factored max diff = {(out_simple - out_factored).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# IterativeDiffusion over-smoothing guard  (P2-A BUG-05)
# ---------------------------------------------------------------------------

def test_iterative_oversmoothing_guard_convergence():
    """Warning must fire when steps trigger early convergence (P2-A)."""
    N, d = 10, 4
    L = _make_laplacian(N)
    # Use very large steps with tiny eta so convergence fires quickly
    op = IterativeDiffusion(eta=0.0, steps=20, early_stop_tol=1e-3)
    op.precompute(L)
    X = torch.randn(N, d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op(X)
    # With eta=0, X_new == X every step → convergence delta=0 < tol → warning
    convergence_warnings = [
        w for w in caught if "early stop" in str(w.message).lower()
        or "convergence" in str(w.message).lower()
    ]
    assert len(convergence_warnings) >= 1, (
        "IterativeDiffusion: expected a convergence/over-smoothing warning"
    )


def test_iterative_signal_energy_guard():
    """Signal energy collapse guard fires when energy drops below 10% (P2-A).

    Setup: complete-graph Laplacian K_N with eta=1/N annihilates all non-DC
    components in a single step.  Centering X removes the DC component so
    ||D·X|| → 0 after step 1, triggering signal_energy < 0.1.
    """
    N, d = 10, 4
    # Complete-graph Laplacian: L = N·I - J (eigenvalues: 0 for DC, N for rest)
    J = torch.ones(N, N)
    L_complete = N * torch.eye(N) - J
    # eta = 1/N  ⟹  D·non_DC_component = (1 - eta*N)·x = 0
    op = IterativeDiffusion(eta=1.0 / N, steps=5, early_stop_tol=0.0)
    op.precompute(L_complete)
    # Center X so DC component (mean) is zero → D·X = 0 in one step
    torch.manual_seed(0)
    X = torch.randn(N, d)
    X = X - X.mean(dim=0, keepdim=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = op(X)
    energy_warnings = [
        w for w in caught if "signal energy" in str(w.message).lower()
        or "over-smoothing" in str(w.message).lower()
    ]
    assert len(energy_warnings) >= 1, (
        "IterativeDiffusion: expected a signal-energy collapse warning"
    )
    # Guard must return the last good state (X before collapse), not zeros
    assert result.abs().max().item() > 1e-6, (
        "Signal energy guard should return last good state, not zeros"
    )


# ---------------------------------------------------------------------------
# SpectralDiffusion eigendecomp cache  (P1-B)
# ---------------------------------------------------------------------------

def test_spectral_precompute_cached_across_instances():
    """P1-B: eigendecomp must be computed once and shared across instances."""
    SpectralDiffusion.clear_cache()
    N = 12
    L = _make_laplacian(N)

    sd1 = SpectralDiffusion(eta=0.1)
    sd2 = SpectralDiffusion(eta=0.1)

    sd1.precompute(L)
    H1 = sd1._op
    sd2.precompute(L)
    H2 = sd2._op

    assert H1 is H2, (
        "SpectralDiffusion: eigendecomp should be cached — H1 and H2 should be "
        "the same object in memory (P1-B)"
    )
    assert len(SpectralDiffusion._EIGENDECOMP_CACHE) == 1, (
        "Cache should contain exactly one entry after two identical precompute calls"
    )
    SpectralDiffusion.clear_cache()


def test_spectral_forward_correct():
    """SpectralDiffusion forward should match reference heat kernel exactly."""
    SpectralDiffusion.clear_cache()
    N, d = 10, 6
    L = _make_laplacian(N)
    eta = 0.2

    eigenvalues, U = torch.linalg.eigh(L.float())
    H_ref = U @ torch.diag(torch.exp(-eta * eigenvalues)) @ U.t()

    sd = SpectralDiffusion(eta=eta).precompute(L)
    X  = torch.randn(N, d)
    out = sd(X)
    ref = H_ref @ X

    assert torch.allclose(out, ref, atol=1e-5), (
        f"SpectralDiffusion forward max diff = {(out - ref).abs().max():.2e}"
    )
    SpectralDiffusion.clear_cache()


# ---------------------------------------------------------------------------
# FactoredDiffusion apply_with_laplacian_trace  (P1-A)
# ---------------------------------------------------------------------------

def test_factored_laplacian_trace_matches_direct():
    """apply_with_laplacian_trace must match tr(KᵀLK) computed via dense L."""
    N, d = 12, 6
    eta = 0.1
    W, deg = _make_adj_deg(N, k=4)
    L_dense = torch.diag(deg) - W            # unnormalised Laplacian

    fd = FactoredDiffusion(eta=eta).precompute_from_graph(W, deg)

    K  = torch.randn(N, d)
    DK, lap_trace_factored = fd.apply_with_laplacian_trace(K)

    lap_trace_direct = float(torch.trace(K.t() @ L_dense @ K).item())

    assert abs(lap_trace_factored - lap_trace_direct) < 1e-3, (
        f"apply_with_laplacian_trace: factored={lap_trace_factored:.4f} "
        f"direct={lap_trace_direct:.4f}"
    )
