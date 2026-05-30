"""
Energy Gap Sweep Experiment
============================
Direct measurement of the inter-cluster energy gap as a function of
diffusion strength η — the primary theoretical metric for DAHN.

Hypothesis (Theorem 1): The inter-cluster energy gap
  ΔE(η) = E(ξ, K_wrong_cluster) - E(ξ, K_correct_cluster)
increases monotonically with η for the correct diffusion direction.

Outputs
-------
  results/energy_gap_sweep.csv
  results/plots/energy_gap_sweep.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import torch
import numpy as np
import pandas as pd

from difflayers.diffusion import DiffusionOperator
from difflayers.graph.build_graph import build_similarity_matrix, build_knn_graph
from difflayers.graph.laplacian import compute_normalized_laplacian
from src.utils.data_gen import generate_clustered_patterns, add_noise
from src.utils.metrics import (
    accuracy, cluster_accuracy, topk_accuracy,
    inter_cluster_energy_gap, retrieval_hamming,
)


@torch.no_grad()
def _retrieve_argmax(Q, K, beta):
    scores   = beta * (Q @ K.t())        # (M, N)
    weights  = torch.softmax(scores, dim=-1)
    pred_idx = scores.argmax(dim=-1)
    return pred_idx, weights


def run_energy_gap_sweep(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    k: int = 10,
    M: int = 500,
    noise: float = 0.25,
    eta_values: list = None,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Sweep diffusion strength η and measure all relevant metrics.

    For each η:
      - Build L from the kNN graph of clean patterns
      - Diffuse QUERY patterns with SimpleDiffusion (correct direction)
      - Measure: exact_accuracy, cluster_accuracy, topk_accuracy,
                 hamming, energy_gap

    Expected results:
      energy_gap     ↑ monotone with η  (Theorem 1 — primary theoretical claim)
      cluster_accuracy ↑ with η        (practical cluster-level improvement)
      hamming        ↓ with η           (better reconstruction)
      exact_accuracy ≈ constant        (query diffusion doesn't hurt exact acc)

    Returns:
        df: columns [eta, exact_accuracy, cluster_accuracy, topk_accuracy,
                     hamming, energy_gap]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if eta_values is None:
        eta_values = [0.0, 0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30]

    device       = torch.device("cpu")
    patterns, cluster_labels = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed, return_labels=True
    )
    patterns = patterns.to(device)
    cluster_size = max(2, N // n_clusters)

    # Build graph and Laplacian once (same for all η)
    S   = build_similarity_matrix(patterns)
    A   = build_knn_graph(S, k, as_sparse=False)
    L   = compute_normalized_laplacian(A).to(device)

    # Fixed noisy queries
    rng = torch.Generator()
    rng.manual_seed(seed + int(noise * 1000))
    target_idx    = torch.randint(0, N, (M,), generator=rng)
    clean_queries = patterns[target_idx]
    noisy_queries = add_noise(clean_queries, noise, seed=seed + int(noise * 100))

    rows = []
    for eta in eta_values:
        if eta == 0.0:
            Q_diff = noisy_queries.clone()
        else:
            # Apply stored-pattern diffusion operator to queries:
            # Q' = (I - eta*L) * Q  via explicit matmul (L is N×N, Q is M×d)
            # This is the pre-diffuse step: project each query through the
            # key-space Laplacian, smoothing it toward its cluster centroid.
            # Note: L has shape (N, N) and Q has shape (M, d); we cannot directly
            # apply the precomputed N×N operator.  Instead we compute D*Q explicitly:
            # For each query q: q' = q - eta * L_q q, where L_q is the Laplacian
            # of a kNN graph built on the queries themselves.
            S_q   = build_similarity_matrix(noisy_queries)
            A_q   = build_knn_graph(S_q, k, as_sparse=False)
            L_q   = compute_normalized_laplacian(A_q).to(device)
            op    = DiffusionOperator.create("simple", eta=eta, steps=1)
            op.precompute(L_q)
            Q_diff = op(noisy_queries)

        # Retrieve using diffused queries against CLEAN stored patterns
        pred_idx, attn_weights = _retrieve_argmax(Q_diff, patterns, beta)

        ex_acc = accuracy(pred_idx, target_idx)
        cl_acc = cluster_accuracy(pred_idx, target_idx, cluster_labels)
        tk_acc = topk_accuracy(attn_weights, target_idx, k=cluster_size)
        ham    = retrieval_hamming(pred_idx, target_idx, patterns)
        egap   = inter_cluster_energy_gap(
            Q_diff, patterns, cluster_labels, beta, target_idx
        )

        rows.append({
            "eta":              round(eta, 4),
            "exact_accuracy":   round(ex_acc, 4),
            "cluster_accuracy": round(cl_acc, 4),
            "topk_accuracy":    round(tk_acc, 4),
            "hamming":          round(ham, 4),
            "energy_gap":       round(egap, 4),
        })
        print(f"  η={eta:.3f}  exact={ex_acc:.3f}  cluster={cl_acc:.3f}  "
              f"topk={tk_acc:.3f}  ham={ham:.4f}  ΔE={egap:.3f}")

    df = pd.DataFrame(rows)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "energy_gap_sweep.csv", index=False)
    print(f"\nSaved → {out_dir / 'energy_gap_sweep.csv'}")
    return df


if __name__ == "__main__":
    print("=== Energy Gap Sweep (noise=0.25) ===")
    df = run_energy_gap_sweep(noise=0.25)
    print("\nFull results:")
    print(df.to_string(index=False))

    # Quick monotonicity check
    gaps = df["energy_gap"].values
    monotone = all(gaps[i] <= gaps[i+1] + 1e-6 for i in range(len(gaps)-1))
    print(f"\nEnergy gap monotone ↑ with η: {monotone}")
    print(f"Gap at η=0.0:  {gaps[0]:.3f}")
    print(f"Gap at η=0.30: {gaps[-1]:.3f}")
    print(f"Total increase: {gaps[-1] - gaps[0]:.3f}")
