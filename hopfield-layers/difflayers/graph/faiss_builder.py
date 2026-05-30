"""
FAISS-backed kNN Graph Builder
================================

Drop-in replacement for GraphBuilder at large N (> ~5 000 patterns).

Why FAISS instead of the current O(N²d) similarity matrix:

    current:  build_similarity_matrix — O(N²d) time, O(N²) memory
    FAISS Flat (exact):   O(N·d·log N) time, O(N·d) memory
    FAISS HNSW (approx):  O(N·log N) build, O(log N) per query

The output (W, deg, adj_indices) is identical to GraphBuilder — every
downstream consumer (LaplacianBuilder, DiffusionOperator, AttentionOperator)
is unaffected.

Index types
-----------
  "flat"     — Exact inner-product search.  Use for N < ~100K. Correct kNN.
  "hnsw"     — Hierarchical NSW graph index.  Use for N ≥ 100K.  ~99% recall.
               Build is fast; does not support GPU without faiss-gpu.
  "ivf_flat" — IVF index with flat quantizer.  Fastest for 50K < N < 10M.
               Requires training step (`n_probe` controls recall/speed tradeoff).

Graceful fallback
-----------------
If faiss is not installed, FAISSGraphBuilder falls back to the pure-PyTorch
GraphBuilder automatically (with a one-time UserWarning).

Usage
-----
    from difflayers.graph.faiss_builder import FAISSGraphBuilder

    builder = FAISSGraphBuilder(k=10, index_type="flat")   # exact
    W, deg, adj_indices = builder.build(patterns)          # patterns: (N, d)

    # Persistent index — add patterns incrementally
    builder.train(patterns)        # one-time index build
    W, deg, adj = builder.build_from_index(patterns)   # reuse index
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# faiss-cpu and PyTorch both ship OpenMP on macOS (Apple Silicon).
# Setting this env var before the first faiss import prevents the segfault
# caused by two incompatible libomp.dylib being loaded simultaneously.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from .build_graph import build_similarity_matrix, build_knn_graph


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FAISSGraphBuilder:
    """
    kNN graph builder backed by FAISS for O(N log N) or better complexity.

    Falls back to the pure-PyTorch O(N²d) builder when faiss is unavailable.

    Args:
        k:          Number of nearest neighbours per node.
        index_type: ``'flat'`` (exact, default), ``'hnsw'`` (approx, fast
                    at N ≥ 100K), or ``'ivf_flat'`` (IVF, needs train step).
        hnsw_m:     HNSW connectivity parameter M (default 32).
                    Higher = better recall, more memory.
        nlist:      IVF number of clusters (default 100, rule: sqrt(N)).
        nprobe:     IVF number of clusters to search at query time (default 10).
        use_sparse: Whether to return W as a sparse_coo tensor.
    """

    def __init__(
        self,
        k:          int  = 10,
        index_type: str  = "flat",
        hnsw_m:     int  = 32,
        nlist:      int  = 100,
        nprobe:     int  = 10,
        use_sparse: bool = False,
    ) -> None:
        if index_type not in ("flat", "hnsw", "ivf_flat"):
            raise ValueError(
                f"FAISSGraphBuilder: index_type must be 'flat', 'hnsw', or "
                f"'ivf_flat'; got '{index_type}'."
            )
        self.k          = k
        self.index_type = index_type
        self.hnsw_m     = hnsw_m
        self.nlist      = nlist
        self.nprobe     = nprobe
        self.use_sparse = use_sparse
        self._index     = None   # persistent FAISS index (built lazily)

    # ------------------------------------------------------------------
    # Primary API — same signature as GraphBuilder.build()
    # ------------------------------------------------------------------

    def build(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build (W, deg, adj_indices) from pattern embeddings.

        Args:
            X: (N, d) float32 pattern embeddings.

        Returns:
            W:           (N, N) dense or sparse_coo adjacency.
            deg:         (N,)   degree vector.
            adj_indices: (N, k) LongTensor of kNN neighbour indices.
        """
        if not _FAISS_AVAILABLE:
            warnings.warn(
                "faiss not installed — falling back to O(N²d) PyTorch kNN. "
                "Install with: pip install faiss-cpu",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback_build(X)

        N, d = X.shape
        k_eff = min(self.k, N - 1)

        # L2-normalise so inner product == cosine similarity
        X_norm = F.normalize(X.float(), dim=-1).contiguous()
        X_np   = X_norm.detach().cpu().numpy().astype(np.float32)

        # Build or rebuild index
        index = self._build_index(d)
        index.add(X_np)

        # kNN search — k+1 because FAISS returns the query itself at rank 0
        k_search         = min(k_eff + 1, N)
        distances, I_np  = index.search(X_np, k_search)   # (N, k+1)

        # Remove self-hits (rank 0 is always the query itself in Flat/HNSW)
        # In practice: drop the column where I == row_index
        row_idx      = np.arange(N)[:, None]
        self_mask    = (I_np == row_idx)
        # Keep first k non-self results per row
        adj_np = np.zeros((N, k_eff), dtype=np.int64)
        sim_np = np.zeros((N, k_eff), dtype=np.float32)

        for i in range(N):
            cols = [(I_np[i, j], distances[i, j])
                    for j in range(k_search) if I_np[i, j] != i]
            cols = cols[:k_eff]
            # Pad with 0 if fewer than k_eff non-self neighbours (rare at N≥k)
            while len(cols) < k_eff:
                cols.append((0, 0.0))
            for rank, (idx, sim) in enumerate(cols):
                adj_np[i, rank] = idx
                sim_np[i, rank] = max(0.0, float(sim))   # clamp negatives

        adj_indices = torch.from_numpy(adj_np).long()     # (N, k)
        sim_vals    = torch.from_numpy(sim_np).float()    # (N, k)

        # Build symmetric adjacency W from (adj_indices, sim_vals)
        W, deg = self._build_adjacency(N, adj_indices, sim_vals)

        return W, deg, adj_indices

    # ------------------------------------------------------------------
    # Incremental API — build index once, query many times
    # ------------------------------------------------------------------

    def train(self, X: Tensor) -> None:
        """
        Build FAISS index from patterns.  Required once before
        ``build_from_index``.  Reuse avoids re-building for repeated queries.

        Args:
            X: (N, d) float32 patterns.
        """
        if not _FAISS_AVAILABLE:
            return   # no-op; build() will fall back

        N, d = X.shape
        X_norm = F.normalize(X.float(), dim=-1).contiguous()
        X_np   = X_norm.detach().cpu().numpy().astype(np.float32)

        self._index = self._build_index(d)
        if self.index_type == "ivf_flat":
            self._index.train(X_np)
        self._index.add(X_np)
        self._trained_N = N

    def build_from_index(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Query the pre-built FAISS index (from ``train()``) against X.

        Faster than ``build()`` when the index is already populated and
        X is a query-only set (different from the training patterns).

        Args:
            X: (M, d) query patterns.

        Returns:
            W, deg, adj_indices — same format as ``build()``, but W/deg
            are computed only for the M×k subgraph (no symmetric fill).
        """
        if self._index is None or not _FAISS_AVAILABLE:
            return self.build(X)

        M, d    = X.shape
        k_eff   = min(self.k, self._trained_N - 1)
        X_norm  = F.normalize(X.float(), dim=-1).contiguous()
        X_np    = X_norm.detach().cpu().numpy().astype(np.float32)

        if self.index_type == "hnsw":
            self._index.hnsw.efSearch = max(64, k_eff * 4)

        distances, I_np = self._index.search(X_np, k_eff)   # (M, k)
        adj_indices     = torch.from_numpy(I_np.astype(np.int64)).long()
        sim_vals        = torch.from_numpy(
            np.clip(distances.astype(np.float32), 0, None)
        )

        W, deg = self._build_adjacency(M, adj_indices, sim_vals)
        return W, deg, adj_indices

    def reset(self) -> None:
        """Discard the current FAISS index, freeing memory."""
        self._index   = None
        self._trained_N = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, d: int):
        """Construct a fresh FAISS index for dimension d."""
        if self.index_type == "flat":
            # Exact inner-product search on L2-normalised vectors == cosine
            return faiss.IndexFlatIP(d)

        if self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 200   # build quality; trade-off with speed
            return idx

        # ivf_flat
        quantizer = faiss.IndexFlatIP(d)
        idx       = faiss.IndexIVFFlat(
            quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT
        )
        idx.nprobe = self.nprobe
        return idx

    def _build_adjacency(
        self, N: int, adj_indices: Tensor, sim_vals: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Build symmetric (N, N) adjacency W and degree vector from
        (N, k) adj_indices + sim_vals.

        Dense build — O(N·k) iterations, O(N²) memory.
        For large N use use_sparse=True (returns sparse_coo).
        """
        k_eff = adj_indices.shape[1]

        if not self.use_sparse:
            W = torch.zeros(N, N, dtype=sim_vals.dtype)
            rows = torch.arange(N).unsqueeze(1).expand(-1, k_eff)  # (N, k)
            W.scatter_(1, adj_indices, sim_vals)
            # Symmetrize
            W = torch.max(W, W.t())
            deg = W.sum(dim=1)
            return W, deg

        # Sparse path — O(k·N) memory
        rows    = torch.arange(N).unsqueeze(1).expand(-1, k_eff).reshape(-1)
        cols    = adj_indices.reshape(-1)
        vals    = sim_vals.reshape(-1)

        # Symmetric: add both (i→j) and (j→i) halves
        all_r   = torch.cat([rows, cols])
        all_c   = torch.cat([cols, rows])
        all_v   = torch.cat([vals, vals])

        W_sp = torch.sparse_coo_tensor(
            torch.stack([all_r, all_c]), all_v, (N, N)
        ).coalesce()

        deg = torch.zeros(N, dtype=sim_vals.dtype)
        deg.scatter_add_(0, all_r, all_v)
        return W_sp, deg

    def _fallback_build(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Pure-PyTorch O(N²d) fallback used when faiss is unavailable."""
        from .build_graph import build_similarity_matrix, build_knn_graph
        S   = build_similarity_matrix(X)
        N   = S.shape[0]
        k_eff = min(self.k, N - 1)
        W     = build_knn_graph(S, k=k_eff, as_sparse=self.use_sparse)
        if self.use_sparse:
            deg = torch.zeros(N)
            W_dense = W.to_dense()
            deg = W_dense.sum(dim=1)
        else:
            deg = W.sum(dim=1)
        adj_indices = torch.topk(
            S, k=k_eff, dim=1
        ).indices
        return W, deg, adj_indices
