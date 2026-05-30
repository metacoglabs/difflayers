"""
Graph-Prior Attention Hopfield (GPAH)
======================================

Core idea
---------
Instead of diffusing stored patterns (which collapses intra-cluster
discriminability and *hurts* exact retrieval), GPAH adds a graph-structured
**prior directly to the attention logits** at inference time.

For each query q and stored pattern K_i:

    s_i = β · q^T K_i          (standard Hopfield score)
        + γ · g_i(q)           (graph prior)

where g_i(q) = +1 if K_i is a k-nearest-neighbour of q, -1 otherwise
(optionally soft: g_i(q) = sim(q, K_i) normalised by query rank).

This is the correct use of graph structure for retrieval:
  * Stored patterns K are NEVER modified → intra-cluster discriminability preserved
  * Attention logits are biased toward patterns structurally close to the query
  * The graph is built from query → keys (cross-graph), not keys → keys (self-graph)

Energy function
---------------
    E_GPAH(ξ) = -lse(β, Kξ + γ·G_ξ) + ½‖ξ‖²

where G_ξ ∈ ℝ^N, G_i = +1 if K_i ∈ kNN(ξ) else -1.

This provably increases the energy gap between the correct pattern and
spurious patterns not in the query's neighbourhood, while leaving the
softmax denominator structure unchanged.

Architecture
------------
  * Inherits from Hopfield — drop-in replacement.
  * Only overrides _associate to inject the graph prior into logits.
  * No diffusion of patterns → no EnergyTracker, no DynamicsEngine needed.
  * Graph prior is computed on-the-fly per forward pass (O(Nd) similarity).

Hyperparameters
---------------
  gamma:       Prior strength. γ=0 → reduces exactly to standard Hopfield.
               Recommend γ ∈ [0.5, 2.0] (same order as β at small d).
  k_prior:     Number of nearest neighbours to favour. Default: 5.
  soft_prior:  If True, use soft (cosine-similarity) prior instead of binary
               ±1.  Smoother gradient but weaker signal.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from . import Hopfield
from .graph.build_graph import build_similarity_matrix


class GPAHopfield(Hopfield):
    """
    Graph-Prior Attention Hopfield — adds a query-to-key graph prior to logits.

    Stored patterns are NEVER modified, preserving exact intra-cluster
    discriminability. The kNN prior biases attention toward structurally
    similar patterns without reducing the softmax peak.

    New parameters (beyond standard Hopfield signature)
    ---------------------------------------------------
    gamma:      Prior strength (additive logit bias). Default: 1.0.
                γ=0 degenerates to standard Hopfield exactly.
    k_prior:    kNN degree for the prior graph. Default: 5.
    soft_prior: Use soft (cosine-similarity) prior instead of binary ±1.
                Default: False.

    All remaining kwargs are forwarded to Hopfield unchanged.
    """

    def __init__(
        self,
        *args,
        gamma: float = 1.0,
        k_prior: int = 5,
        soft_prior: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma      = gamma
        self.k_prior    = k_prior
        self.soft_prior = soft_prior

    # ------------------------------------------------------------------
    # Override _associate — inject graph prior into logits
    # ------------------------------------------------------------------

    def _associate(
        self,
        data: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
        return_raw_associations: bool = False,
        return_projected_patterns: bool = False,
        stored_pattern_padding_mask: Optional[Tensor] = None,
        association_mask: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], ...]:
        """
        Hopfield association with graph-prior logit bias.

        Injects an attention-mask prior from a query→key kNN graph before
        the softmax inside HopfieldCore.  Stored patterns are unchanged.
        """
        assert (type(data) == Tensor) or (
            (type(data) == tuple) and (len(data) == 3)
        ), "data must be a tensor or a 3-tuple (stored, state, projection)."

        if type(data) == Tensor:
            stored_pattern = state_pattern = pattern_projection = data
        else:
            stored_pattern, state_pattern, pattern_projection = data

        stored_pattern, state_pattern, pattern_projection = self._maybe_transpose(
            stored_pattern, state_pattern, pattern_projection
        )

        # Optional LayerNorm (mirrors Hopfield._associate)
        if self.norm_stored_pattern is not None:
            stored_pattern = self.norm_stored_pattern(
                input=stored_pattern.reshape(-1, stored_pattern.shape[2])
            ).reshape(stored_pattern.shape)
        if self.norm_state_pattern is not None:
            state_pattern = self.norm_state_pattern(
                input=state_pattern.reshape(-1, state_pattern.shape[2])
            ).reshape(state_pattern.shape)
        if self.norm_pattern_projection is not None:
            pattern_projection = self.norm_pattern_projection(
                input=pattern_projection.reshape(-1, pattern_projection.shape[2])
            ).reshape(pattern_projection.shape)

        # Build graph-prior mask and fold into association_mask
        prior_mask = self._build_prior_mask(state_pattern, stored_pattern)
        # prior_mask: (L_query, S_stored) or (B, L, S) — add to existing mask
        if association_mask is not None:
            association_mask = association_mask + prior_mask
        else:
            association_mask = prior_mask

        return self.association_core(
            query=state_pattern,
            key=stored_pattern,
            value=pattern_projection,
            key_padding_mask=stored_pattern_padding_mask,
            need_weights=False,
            attn_mask=association_mask,          # ← graph prior injected here
            scaling=self.scaling,
            update_steps_max=self.update_steps_max,
            update_steps_eps=self.update_steps_eps,
            return_raw_associations=return_raw_associations,
            return_pattern_projections=return_projected_patterns,
        )

    # ------------------------------------------------------------------
    # Graph prior construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_prior_mask(
        self, Q: Tensor, K: Tensor
    ) -> Tensor:
        """
        Build additive logit mask from query→key kNN graph.

        For each query i:
            prior[i, j] = +γ  if K_j ∈ kNN(Q_i)
            prior[i, j] = -γ  otherwise

        Args:
            Q: (S_q, B, d) or (L, d) query patterns (after optional LayerNorm).
            K: (S_k, B, d) or (S, d) stored patterns.

        Returns:
            mask: Same spatial shape as the logit matrix in association_core.
                  Broadcast-compatible with (B, num_heads, L, S).
        """
        # Flatten to 2D for similarity computation
        if Q.dim() == 3:
            # (S, B, d) → use mean over B as representative (or reshape)
            Q_2d = Q.mean(dim=1)          # (L, d)
            K_2d = K.mean(dim=1)          # (S, d)
        else:
            Q_2d, K_2d = Q, K             # (L, d), (S, d)

        L_q, d = Q_2d.shape
        S_k    = K_2d.shape[0]
        k_eff  = min(self.k_prior, S_k - 1)

        # Cosine similarity between every query and every key: (L, S)
        Q_norm = F.normalize(Q_2d, dim=-1)
        K_norm = F.normalize(K_2d, dim=-1)
        sim    = Q_norm @ K_norm.t()               # (L, S)

        if self.soft_prior:
            # Soft prior: normalised rank-weighted similarity
            # Shift so mean is 0 → same as ±γ binary but smoother
            sim_mean = sim.mean(dim=-1, keepdim=True)
            mask = self.gamma * (sim - sim_mean)   # (L, S)
        else:
            # Binary prior: +γ for top-k, -γ for the rest
            topk_thresh = sim.topk(k_eff, dim=-1).values[:, -1:]  # (L, 1)
            in_nbhd     = (sim >= topk_thresh).float()             # (L, S)
            mask        = self.gamma * (2.0 * in_nbhd - 1.0)      # +γ / -γ

        # Return (L, S) — hopfield_core_forward broadcasts 2D mask over B and H
        return mask    # (L, S)
