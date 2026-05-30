"""
Evaluation metrics for graph-regularized Hopfield attention experiments.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Optional


def accuracy(pred: Tensor, target: Tensor) -> float:
    """
    Fraction of examples where pred == target (exact match).

    Args:
        pred:   (N,) integer class predictions.
        target: (N,) integer ground-truth labels.

    Returns:
        acc: scalar float in [0, 1].
    """
    return (pred == target).float().mean().item()


def hamming_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Normalised per-element Hamming distance between two binary {-1, +1} tensors.

    Args:
        x: (..., d) tensor.
        y: (..., d) tensor, same shape as x.

    Returns:
        dist: (...) tensor of per-sample normalised Hamming distances in [0, 1].
    """
    d = x.shape[-1]
    return (x.sign() != y.sign()).float().sum(dim=-1) / d


def attention_entropy(weights: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Shannon entropy of each attention distribution.

    Args:
        weights: (..., S) attention weight matrix (each row sums to ~1).
        eps:     Small constant for numerical stability inside log.

    Returns:
        H: (...) per-query entropy values (in nats).
    """
    return -(weights * (weights + eps).log()).sum(dim=-1)


def attention_sparsity(weights: Tensor, threshold: float = 0.01) -> Tensor:
    """
    Fraction of attention weights that are effectively zero (< threshold).

    Args:
        weights:   (..., S) attention weights.
        threshold: Values below this are considered near-zero.

    Returns:
        sparsity: (...) per-query sparsity in [0, 1].
    """
    return (weights < threshold).float().mean(dim=-1)


def hopfield_energy(Q: Tensor, K: Tensor, L: Tensor,
                    eta: float, beta: float = 1.0) -> float:
    """
    Compute the Hopfield energy with graph-regularisation penalty.

    E = -(β * Q @ K^T).mean() + η * trace(K^T L K) / N

    The first term is the (negative) mean scaled dot-product affinity — lower
    is better (stronger pattern alignment).  The second term is the graph
    smoothness penalty (Dirichlet energy) measuring feature variation across
    graph edges.

    Args:
        Q: (N, d) query patterns.
        K: (N, d) key patterns.
        L: (N, N) graph Laplacian.
        eta: Diffusion / regularisation strength.
        beta: Hopfield scaling (temperature).

    Returns:
        energy: Scalar energy value.
    """
    affinity = -(beta * Q @ K.t()).mean()
    smoothness = eta * torch.trace(K.t() @ L @ K) / K.shape[0]
    return (affinity + smoothness).item()

# ---------------------------------------------------------------------------
# Cluster-level metrics (needed for DAHN v2 evaluation)
# ---------------------------------------------------------------------------

def cluster_accuracy(pred_idx: Tensor, target_idx: Tensor,
                     cluster_labels: Tensor) -> float:
    """
    Fraction of queries where the retrieved pattern belongs to the correct cluster.

    Unlike exact accuracy (argmax == target_idx), this only requires that the
    retrieved pattern is in the same cluster as the target — the correct metric
    for evaluating graph diffusion which collapses intra-cluster discriminability.

    Args:
        pred_idx:       (M,) predicted pattern indices.
        target_idx:     (M,) ground-truth pattern indices.
        cluster_labels: (N,) cluster membership for each of the N stored patterns.

    Returns:
        acc: Scalar float in [0, 1].
    """
    pred_clusters   = cluster_labels[pred_idx]
    target_clusters = cluster_labels[target_idx]
    return (pred_clusters == target_clusters).float().mean().item()


def topk_accuracy(attn_weights: Tensor, target_idx: Tensor, k: int) -> float:
    """
    Fraction of queries where the target pattern is in the top-K retrieved patterns.

    When K = cluster_size, this directly measures whether diffusion keeps the
    correct pattern in the attention peak despite intra-cluster smoothing.

    Args:
        attn_weights: (M, N) attention weight matrix (each row sums to ~1).
        target_idx:   (M,) ground-truth pattern indices.
        k:            Number of top patterns to consider.

    Returns:
        acc: Scalar float in [0, 1].
    """
    topk_idx = attn_weights.topk(k, dim=-1).indices          # (M, k)
    hit = (topk_idx == target_idx.unsqueeze(1)).any(dim=-1)   # (M,)
    return hit.float().mean().item()


def inter_cluster_energy_gap(
    Q: Tensor,
    K: Tensor,
    cluster_labels: Tensor,
    beta: float,
    target_idx: Tensor,
) -> float:
    """
    Mean energy gap between the correct cluster and the nearest spurious cluster.

    This is the primary theoretical metric for DAHN: graph diffusion provably
    increases this gap, making it easier to reject wrong-cluster attractors.

    Gap_m = max_{i in correct_cluster} score(q_m, K_i)
            - max_{j not in correct_cluster} score(q_m, K_j)

    Args:
        Q:              (M, d) query patterns (possibly noisy).
        K:              (N, d) stored key patterns (possibly diffused).
        cluster_labels: (N,) cluster membership for each stored pattern.
        beta:           Hopfield scaling / inverse temperature.
        target_idx:     (M,) ground-truth pattern indices.

    Returns:
        mean_gap: Scalar float. Positive = correct cluster scores higher.
    """
    scores = beta * (Q @ K.t())   # (M, N)
    gaps = []
    for m in range(Q.shape[0]):
        tc         = cluster_labels[target_idx[m]].item()
        same_mask  = (cluster_labels == tc)
        wrong_mask = ~same_mask
        score_correct = scores[m, same_mask].max().item()
        score_wrong   = scores[m, wrong_mask].max().item()
        gaps.append(score_correct - score_wrong)
    return float(np.mean(gaps))


def retrieval_hamming(
    pred_idx: Tensor,
    target_idx: Tensor,
    patterns: Tensor,
) -> float:
    """
    Mean normalised Hamming distance between retrieved and target patterns.

    Measures reconstruction quality — lower is better.
    Complements exact accuracy: even when argmax misses, a lower Hamming
    distance means the retrieved pattern is close to the correct one.

    Args:
        pred_idx:   (M,) predicted pattern indices.
        target_idx: (M,) ground-truth pattern indices.
        patterns:   (N, d) stored pattern matrix.

    Returns:
        mean_hamming: Scalar float in [0, 1].
    """
    retr  = patterns[pred_idx]    # (M, d)
    truth = patterns[target_idx]  # (M, d)
    d = patterns.shape[-1]
    return ((retr.sign() != truth.sign()).float().sum(-1) / d).mean().item()