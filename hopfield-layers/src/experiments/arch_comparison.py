"""
Architecture Comparison Experiment
====================================
Compares five models across the full noise spectrum:

  1. baseline    — standard Modern Hopfield Network (MHN)
  2. dahn        — DiffusedHopfield (fixed: pre-diffuse only, query diffusion)
  3. gpah        — Graph-Prior Attention Hopfield
  4. chgr        — Contrastive Hopfield with Graph Repulsion
  5. tshh        — Two-Stage Hierarchical Hopfield

Metrics (the full set required for honest evaluation):
  * exact_accuracy    — argmax == target_index  (same as all previous experiments)
  * cluster_accuracy  — retrieved pattern is in the correct cluster
  * topk_accuracy     — target is in top-K retrieved (K = cluster size)
  * hamming           — Hamming distance to clean target pattern
  * energy_gap        — inter-cluster energy gap (primary DAHN theoretical metric)

Outputs
-------
  results/arch_comparison.csv
  results/plots/arch_comparison.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import torch
import numpy as np
import pandas as pd

from difflayers import Hopfield, DiffusedHopfield
from difflayers.gpah import GPAHopfield
from difflayers.chgr import ContrastiveHopfield
from difflayers.tshh import TwoStageHopfield
from src.utils.data_gen import generate_clustered_patterns, add_noise
from src.utils.metrics import (
    accuracy, cluster_accuracy, topk_accuracy,
    inter_cluster_energy_gap, retrieval_hamming,
)

_STATIC = dict(
    input_size=None,
    stored_pattern_as_static=True,
    state_pattern_as_static=True,
    pattern_projection_as_static=True,
    disable_out_projection=True,
    normalize_stored_pattern=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection=False,
    normalize_pattern_projection_affine=False,
    batch_first=True,
)


def _build_models(beta, eta, k, diffusion_steps, n_clusters, N, device):
    cluster_size = max(2, N // n_clusters)

    models = {}

    # 1. Baseline MHN
    models["baseline"] = Hopfield(**_STATIC, scaling=beta).to(device).eval()

    # 2. DAHN — fixed: query diffusion only, pre-diffuse architecture
    models["dahn"] = DiffusedHopfield(
        **_STATIC, scaling=beta,
        eta=eta, k_neighbors=k,
        use_normalized_laplacian=True,
        diffuse_query=True,
        diffuse_key=False,
        diffusion_mode="simple",
        diffusion_steps=diffusion_steps,
    ).to(device).eval()

    # 3. GPAH — graph prior on logits (binary ±γ, k_prior = cluster_size)
    models["gpah"] = GPAHopfield(
        **_STATIC, scaling=beta,
        gamma=1.0,
        k_prior=cluster_size,
        soft_prior=False,
    ).to(device).eval()

    # 4. CHGR — contrastive repulsion (k_near = cluster_size)
    models["chgr"] = ContrastiveHopfield(
        **_STATIC, scaling=beta,
        k_near=cluster_size,
        lam=0.10,
        beta_neg=None,   # auto β/2
    ).to(device).eval()

    # 5. TSHH — two-stage hierarchical (β₁=2, k_coarse=cluster_size)
    models["tshh"] = TwoStageHopfield(
        **_STATIC, scaling=beta,
        beta1=2.0,
        k_coarse=cluster_size,
    ).to(device).eval()

    return models


@torch.no_grad()
def _retrieve(model, patterns, noisy_queries):
    M, d = noisy_queries.shape
    N    = patterns.shape[0]
    stored = patterns.unsqueeze(0).expand(M, N, d)
    query  = noisy_queries.unsqueeze(1)
    attn = model.get_association_matrix(input=(stored, query, stored))  # (M,1,1,N)
    attn = attn[:, 0, 0, :]                                             # (M, N)
    return attn.argmax(dim=-1), attn                                    # pred_idx, weights


def run_arch_comparison(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    eta: float = 0.10,
    k: int = 10,
    diffusion_steps: int = 3,
    M: int = 500,
    noise_levels: list = None,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run the architecture comparison and return a results DataFrame.

    Returns:
        df: columns [noise_level, model, exact_accuracy, cluster_accuracy,
                     topk_accuracy, hamming, energy_gap]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    device   = torch.device("cpu")
    patterns, cluster_labels = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed, return_labels=True
    )
    patterns = patterns.to(device)
    cluster_size = max(2, N // n_clusters)

    models = _build_models(beta, eta, k, diffusion_steps, n_clusters, N, device)

    rows = []
    for p in noise_levels:
        rng = torch.Generator()
        rng.manual_seed(seed + int(p * 1000))
        target_idx    = torch.randint(0, N, (M,), generator=rng)
        clean_queries = patterns[target_idx]
        noisy_queries = add_noise(clean_queries, p, seed=seed + int(p * 100))

        for name, model in models.items():
            pred_idx, attn_weights = _retrieve(model, patterns, noisy_queries)

            ex_acc  = accuracy(pred_idx, target_idx)
            cl_acc  = cluster_accuracy(pred_idx, target_idx, cluster_labels)
            tk_acc  = topk_accuracy(attn_weights, target_idx, k=cluster_size)
            ham     = retrieval_hamming(pred_idx, target_idx, patterns)
            egap    = inter_cluster_energy_gap(
                noisy_queries, patterns, cluster_labels, beta, target_idx
            )

            rows.append({
                "noise_level":      round(p, 4),
                "model":            name,
                "exact_accuracy":   round(ex_acc, 4),
                "cluster_accuracy": round(cl_acc, 4),
                "topk_accuracy":    round(tk_acc, 4),
                "hamming":          round(ham, 4),
                "energy_gap":       round(egap, 4),
            })

        print(f"  noise={p:.2f}  done")

    df = pd.DataFrame(rows)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "arch_comparison.csv", index=False)
    print(f"\nSaved → {out_dir / 'arch_comparison.csv'}")
    return df


if __name__ == "__main__":
    df = run_arch_comparison()
    print("\n--- Summary at noise=0.25 ---")
    sub = df[df["noise_level"] == 0.25][
        ["model", "exact_accuracy", "cluster_accuracy", "topk_accuracy", "hamming", "energy_gap"]
    ]
    print(sub.to_string(index=False))
    print("\n--- Summary at noise=0.35 ---")
    sub = df[df["noise_level"] == 0.35][
        ["model", "exact_accuracy", "cluster_accuracy", "topk_accuracy", "hamming", "energy_gap"]
    ]
    print(sub.to_string(index=False))
