"""
Synthetic data generation utilities for graph-regularized Hopfield experiments.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple


def generate_patterns(N: int, d: int, seed: int = 42) -> Tensor:
    """
    Generate N random binary patterns of dimension d with values in {-1, +1}.

    Patterns are drawn i.i.d. from a Rademacher distribution and L2-normalised
    to unit vectors (makes cosine-similarity well defined).

    Args:
        N: Number of patterns.
        d: Pattern dimension.
        seed: Random seed for reproducibility.

    Returns:
        patterns: (N, d) float32 tensor with values approximately ±1/√d after
                  normalization.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    bits = torch.randint(0, 2, (N, d), generator=rng).float() * 2 - 1
    patterns = bits / (d ** 0.5)
    return patterns


def generate_clustered_patterns(
    N: int,
    d: int,
    n_clusters: int = 10,
    intra_noise: float = 0.15,
    seed: int = 42,
    return_labels: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Generate N patterns organised into clusters.

    Each cluster has a random centroid (Rademacher ±1) and member patterns
    are noisy copies of the centroid.  This creates patterns with real graph
    structure — nearby patterns in the kNN graph genuinely share features,
    making graph diffusion more effective than on fully random patterns.

    Args:
        N:             Total number of patterns.
        d:             Pattern dimension.
        n_clusters:    Number of clusters.
        intra_noise:   Per-bit flip probability within a cluster
                       (lower = tighter clusters, larger diffusion benefit).
        seed:          Random seed for reproducibility.
        return_labels: If True, also return a (N,) cluster-label tensor.
                       Required for cluster_accuracy and energy gap metrics.

    Returns:
        patterns:       (N, d) float32 tensor, L2-normalised.
        cluster_labels: (N,) LongTensor of cluster indices (0..n_clusters-1),
                        or None when return_labels=False.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    per_cluster = N // n_clusters
    remainder   = N - per_cluster * n_clusters

    centroids = torch.randint(0, 2, (n_clusters, d), generator=rng).float() * 2 - 1

    patterns_list = []
    labels_list   = []
    for i in range(n_clusters):
        n_i      = per_cluster + (1 if i < remainder else 0)
        centroid = centroids[i].unsqueeze(0).expand(n_i, d)
        flip     = torch.bernoulli(
            torch.full((n_i, d), intra_noise), generator=rng
        ).bool()
        noisy    = torch.where(flip, -centroid, centroid)
        patterns_list.append(noisy)
        labels_list.append(torch.full((n_i,), i, dtype=torch.long))

    patterns = torch.cat(patterns_list, dim=0)
    patterns = patterns / (d ** 0.5)

    if return_labels:
        return patterns, torch.cat(labels_list, dim=0)
    return patterns, None


def add_noise(x: Tensor, p: float, seed: int = 0) -> Tensor:
    """
    Corrupt a pattern by independently flipping each dimension's sign with
    probability p (bit-flip noise on the binary ±1 encoding).

    Args:
        x: (..., d) input pattern(s).
        p: Flip probability in [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        x_noisy: Same shape as x with approximately p*d bits flipped.
    """
    if p == 0.0:
        return x.clone()
    rng = torch.Generator()
    rng.manual_seed(seed)
    flip_mask = torch.bernoulli(
        torch.full(x.shape, p, dtype=x.dtype, device=x.device),
        generator=rng,
    ).bool()
    return torch.where(flip_mask, -x, x)


def load_mnist_pca(
    N_per_class: int = 20,
    d: int = 64,
    seed: int = 42,
    data_root: str = "./data",
) -> Tuple[Tensor, Tensor]:
    """
    Load MNIST, project to *d*-dimensional PCA space, and return L2-normalised
    float32 patterns together with integer class labels.

    Uses the training split.  Patterns are L2-normalised so cosine similarity
    equals dot product — consistent with the rest of the codebase.

    Args:
        N_per_class: Number of stored patterns per class (0-9).  Total N = 10 * N_per_class.
        d:           PCA output dimension.  Must be ≤ 784.
        seed:        Random seed for selecting which examples to keep.
        data_root:   Directory where torchvision will cache MNIST.

    Returns:
        patterns: (N, d) float32 tensor, L2-normalised.
        labels:   (N,) int64 tensor, class indices in [0, 9].

    Raises:
        ImportError: If torchvision or scikit-learn are not installed.
    """
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("torchvision is required: pip install torchvision") from exc
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

    rng = np.random.RandomState(seed)

    ds = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True,
        transform=T.ToTensor(),
    )

    # Collect N_per_class examples for each of the 10 digit classes
    all_images = []
    all_labels = []
    all_targets = np.array(ds.targets)
    for cls in range(10):
        idxs = np.where(all_targets == cls)[0]
        chosen = rng.choice(idxs, size=N_per_class, replace=False)
        for i in chosen:
            img, lbl = ds[int(i)]
            all_images.append(img.view(-1).numpy())  # (784,)
            all_labels.append(lbl)

    X = np.stack(all_images, axis=0).astype(np.float32)  # (N, 784)
    labels = np.array(all_labels, dtype=np.int64)

    # PCA projection
    pca = PCA(n_components=d, random_state=seed)
    X_pca = pca.fit_transform(X).astype(np.float32)     # (N, d)

    # L2 normalise
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_pca /= norms

    patterns = torch.from_numpy(X_pca)
    labels_t = torch.from_numpy(labels)
    return patterns, labels_t
