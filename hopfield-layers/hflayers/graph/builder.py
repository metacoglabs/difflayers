"""
Graph builder for Graph-Regularized Hopfield attention.

Responsibility: Build the kNN similarity graph (adjacency W and degree vector)
from raw pattern embeddings.  This is the *only* class that calls
``build_similarity_matrix`` and ``build_knn_graph``.

Separating graph construction from Laplacian computation and diffusion
satisfies Single-Responsibility (SOLID) and avoids duplicated graph logic (DRY).

Usage::

    builder = GraphBuilder(k=5, use_sparse=True)
    W, deg, adj_idx = builder.build(X)    # X: (N, d)

Complexity:
    build_similarity_matrix : O(N²d)
    build_knn_graph (dense) : O(N²)  — topk + symmetrize
    build_knn_graph (sparse): O(kN) storage
    degree computation      : O(N)
    adj_indices extraction  : O(kN)

Memory:
    dense  : O(N²)
    sparse : O(kN) for W; O(N) for deg; O(kN) for adj_indices
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .build_graph import build_knn_graph, build_similarity_matrix


class GraphBuilder:
    """
    Builds kNN adjacency matrix W, degree vector, and neighbor-index table
    from pattern embeddings.

    The neighbor-index table (adj_indices) is the (N, k) integer tensor
    required by graph-constrained attention (``AttentionOperator`` in graph
    mode) to avoid forming the full N×N weight matrix.

    Args:
        k:          Number of nearest neighbors per node.
        use_sparse: Return W as ``torch.sparse_coo_tensor`` (O(kN) storage).
                    Enables O(kN) sparse matmuls in ``FactoredDiffusion``.
    """

    def __init__(self, k: int = 5, use_sparse: bool = False) -> None:
        self.k = k
        self.use_sparse = use_sparse

    def build(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build adjacency W, degree deg, and neighbor index table from X.

        Args:
            X: (N, d) float32 pattern embeddings.

        Returns:
            W:          (N, N) adjacency matrix — dense float or sparse_coo.
            deg:        (N,)   degree vector, float32 (row sums of dense W).
            adj_indices:(N, k) LongTensor of kNN neighbor indices (dense W).
                        Required by ``AttentionOperator(mode='graph')``.

        Complexity: O(N²d) + O(N²) + O(kN).
        """
        S = build_similarity_matrix(X)                          # O(N²d)
        N = S.shape[0]
        k_actual = min(self.k, N - 1)

        if self.use_sparse:
            # BUG-7 fix: build sparse W directly from topk of S, avoiding
            # the O(N²) full N×N W_dense intermediate that the dense path
            # would otherwise materialise before conversion to sparse.
            topk_vals, topk_idx = torch.topk(S, k=k_actual, dim=-1)  # (N, k)

            # adj_indices: kNN neighbor table from pre-symmetrisation topk.
            adj_indices = topk_idx                                     # (N, k)

            # Symmetrize: include both directed halves of each edge.
            row_idx = (
                torch.arange(N, device=X.device)
                .unsqueeze(1)
                .expand(-1, k_actual)
                .reshape(-1)
            )                                                          # (N*k,)
            col_idx = topk_idx.reshape(-1)                            # (N*k,)
            vals    = topk_vals.reshape(-1)                           # (N*k,)

            all_rows = torch.cat([row_idx, col_idx])                  # (2*N*k,)
            all_cols = torch.cat([col_idx, row_idx])                  # (2*N*k,)
            all_vals = torch.cat([vals,    vals])                     # (2*N*k,)

            # Coalesce sums duplicate (i,j) entries; the resulting W is
            # symmetric and non-negative — sufficient for diffusion/Laplacian.
            W: Tensor = torch.sparse_coo_tensor(
                torch.stack([all_rows, all_cols]),
                all_vals,
                (N, N),
                dtype=S.dtype,
                device=S.device,
            ).coalesce()

            # Degree from sparse row sums — O(kN), no N×N buffer.
            deg = torch.zeros(N, dtype=S.dtype, device=S.device)
            deg.scatter_add_(0, all_rows, all_vals)                   # (N,)
        else:
            W_dense = build_knn_graph(S, k=k_actual, as_sparse=False)
            deg = W_dense.sum(dim=1)                                  # (N,)
            adj_indices = torch.topk(
                W_dense, k=k_actual, dim=1
            ).indices                                                  # (N, k)
            W = W_dense

        return W, deg, adj_indices
