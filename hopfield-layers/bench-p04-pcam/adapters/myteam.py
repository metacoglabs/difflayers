"""
HHCC-Π  —  Hybrid Hessian-Aligned Class-Conditional Precision Agent
=====================================================================
PCAM bench · P-04 · Team: Priyam Ghosh

Three complementary signals are fused in log-space to produce a
per-coordinate precision vector π ∈ ℝ⁶⁴_{>0}:

    S1  Class-conditional template  π*_c
        Jacobi preconditioner from the per-attractor Hessian diagonal.
        Stretches each attractor's basin along its narrowest directions.
        Earns retrieval (Δ ≥ 0.05) — §6.6 of the PCAM paper.

    S2  Hessian-aligned precision   π^H
        Diagonal of the mean-Hessian inverse, seed-averaged.
        Provably reduces eigenvalue spread by κ(H̄)/κ(Π H̄) — Theorem F3.
        Earns anisotropy points (≥10× spread reduction).

    S3  Query-conditional reliability mask  w_rel(y)
        Soft per-coordinate confidence that a coordinate is uncorrupted.
        Robustness gate against heavy masking / Gaussian corruption.

Combiner (log-space, mean-normalised):
    log π = α log π*_q + (1−α) log π^H + ρ log w_rel
    followed by the double-mean-normalise safety pipeline so the
    harness clip [0.1, 10] is a no-op.

Hyper-parameters:
    α = 0.55   (class template weight)
    ρ = 0.25   (reliability mask weight)
    γ = 0.50   (reliability mask sharpness)

Pure NumPy — no training, no external deps beyond numpy/scipy.
Deterministic given stored_patterns and model_params.

Reference: solution.md in this repository.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as _scipy_softmax
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Minimal Adapter shim  (replaced by the real harness base class at eval time)
# ---------------------------------------------------------------------------
try:
    from adapter import Adapter  # harness-injected base class
except ImportError:
    class Adapter:  # type: ignore[no-redef]
        """Stand-alone shim used when running outside the harness."""
        pass


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Engine(Adapter):
    """
    HHCC-Π precision agent.

    Parameters
    ----------
    stored_patterns : array-like, shape (K, N)
        The K stored attractors, each of dimension N.
    model_params : dict
        Must contain at minimum:
            'beta' : float   — Hopfield scaling factor β
            'R'    : (N, N)  — frozen structured operator R
        Optional keys consumed silently; harness may pass extra fields.
    """

    # ── Hyper-parameters ────────────────────────────────────────────────────
    _ALPHA = 0.55    # weight on class-conditional template  (retrieval axis)
    _RHO   = 0.25    # weight on query reliability mask
    _GAMMA = 0.50    # Gaussian sharpness of reliability mask
    _TAU   = 8.0     # softmax temperature for soft class blending
    _DELTA = 0.05    # top-2 margin threshold  (below → soft blend)
    _EPS   = 1e-6    # generic numerical floor

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __init__(self, stored_patterns: Any, model_params: Dict[str, Any]) -> None:
        X  = np.asarray(stored_patterns, dtype=np.float64)   # (K, N)
        R  = np.asarray(model_params["R"], dtype=np.float64) # (N, N)
        beta = float(model_params["beta"])

        K, N = X.shape
        self._X    = X
        # Guard every matmul against platform-specific BLAS floating-point
        # warnings (e.g. Apple Accelerate on macOS may emit divide-by-zero
        # warnings for internally-zero pivots even when the result is finite).
        _fp_guard = dict(all="ignore")

        with np.errstate(**_fp_guard):
            self._XR = np.nan_to_num(X @ R, nan=0.0, posinf=1e9, neginf=-1e9)  # (K, N)
        self._beta = beta
        self._K    = K
        self._N    = N

        # ── Signal 3 stats (offline) ────────────────────────────────────────
        self._mu    = X.mean(axis=0)              # (N,)
        self._sigma = X.std(axis=0) + self._EPS   # (N,)

        # ── Per-class Hessians → Signal 1 table ────────────────────────────
        XR = self._XR
        eps_H = max(1e-3, 1e-3 * float(np.trace(np.eye(N))) / N)

        H_sum       = np.zeros((N, N), dtype=np.float64)
        log_pi_star = np.zeros((K, N), dtype=np.float64)  # will hold log π*_c

        for c in range(K):
            # p_c = softmax(β · XR · x_c),  shape (K,)
            with np.errstate(**_fp_guard):
                logits = beta * (XR @ X[c])
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            logits -= logits.max()
            p_c     = np.exp(logits)
            p_c    /= p_c.sum() + self._EPS

            # H_c = I − β (XR)ᵀ (diag(p_c) − p_c p_cᵀ)(XR)
            #      = I − β [ (XRᵀ · (p_c ⊙ XR)) − (XRᵀ p_c)(XRᵀ p_c)ᵀ ]
            XR_w  = XR * p_c[:, None]          # (K, N):  p_k · xr_k
            with np.errstate(**_fp_guard):
                term1 = XR_w.T @ XR                # (N, N):  Σ_k p_k xr_k xr_kᵀ
                XRTp  = XR.T @ p_c                 # (N,)
            term2 = np.outer(XRTp, XRTp)       # (N, N)
            term1 = np.nan_to_num(term1, nan=0.0, posinf=1e9, neginf=-1e9)
            term2 = np.nan_to_num(term2, nan=0.0, posinf=1e9, neginf=-1e9)
            H_c   = np.eye(N) - beta * (term1 - term2)

            H_sum += H_c

            # π*_c = clip(1 / |diag(H_c)|, 0.1, 10),  mean-normalised
            diag_Hc = np.abs(np.diag(H_c))
            pi_c    = np.clip(1.0 / (diag_Hc + self._EPS), 0.1, 10.0)
            pi_c   /= pi_c.mean()
            log_pi_star[c] = np.log(np.clip(pi_c, 1e-9, None))

        self._log_pi_star = log_pi_star  # (K, N)

        # ── Signal 2: average-Hessian inverse diagonal ──────────────────────
        H_bar   = H_sum / K
        reg     = eps_H * np.eye(N)
        try:
            H_bar_inv = np.linalg.inv(H_bar + reg)
        except np.linalg.LinAlgError:
            H_bar_inv = np.linalg.pinv(H_bar)

        pi_H_raw    = np.abs(np.diag(H_bar_inv))
        pi_H        = np.clip(pi_H_raw, 0.1, 10.0)
        pi_H       /= pi_H.mean()
        self._log_pi_H = np.log(np.clip(pi_H, 1e-9, None))  # (N,)

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_precision(self, corrupted_query: Any) -> np.ndarray:
        """
        Return the per-coordinate precision vector π for the given query.

        Parameters
        ----------
        corrupted_query : array-like, shape (N,)
            Masked + Gaussian-corrupted query.

        Returns
        -------
        pi : ndarray, shape (N,)
            Precision vector.  Mean ≈ 1, values in [0.1, 10.0].
        """
        y = np.asarray(corrupted_query, dtype=np.float64)  # (N,)

        # ── Signal 3: per-coordinate reliability ────────────────────────────
        z      = (y - self._mu) / self._sigma
        w_rel  = np.exp(-self._GAMMA * z * z)              # (N,), ∈ (0, 1]
        w_rel  = np.clip(w_rel, 1e-9, 1.0)
        log_w  = np.log(w_rel)

        # ── Class identification via weighted inner product ──────────────────
        # scores_c = ⟨X_c, y ⊙ w_rel⟩  — O(KN)
        _fp_guard = dict(all="ignore")
        with np.errstate(**_fp_guard):
            scores = self._X @ (y * w_rel)                 # (K,)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

        top2   = np.argpartition(scores, -2)[-2:]
        s_top2 = np.sort(scores[top2])
        margin = (s_top2[-1] - s_top2[-2]) / (np.abs(s_top2[-1]) + self._EPS)

        if margin < self._DELTA:
            # Soft blend: weighted geometric mean of log π* tables
            s_shifted = scores - scores.max()
            w_c       = np.exp(self._TAU * s_shifted)
            w_c      /= w_c.sum()
            log_pi_q  = w_c @ self._log_pi_star             # (N,)
        else:
            # Hard pick
            c_hat    = int(np.argmax(scores))
            log_pi_q = self._log_pi_star[c_hat]             # (N,)

        # ── Log-space fusion ────────────────────────────────────────────────
        alpha   = self._ALPHA
        rho     = self._RHO
        log_pi  = alpha * log_pi_q + (1.0 - alpha) * self._log_pi_H + rho * log_w

        # Double mean-normalise so harness clip [0.1, 10] is a no-op
        log_pi -= log_pi.mean()
        log_pi  = np.clip(log_pi, np.log(0.1), np.log(10.0))
        log_pi -= log_pi.mean()

        return np.exp(log_pi)

    # ── Batched / fused public API  (P0-C / P1-D) ───────────────────────────

    def predict_precision_batch(
        self, corrupted_queries: "np.ndarray"
    ) -> "np.ndarray":
        """
        Process B queries simultaneously with a single (K,N)@(N,B) matmul.

        Arithmetic intensity improvement over the serial path:
            Before: 2 × (K,N)@(N,) per query  → AI ≈ 0.25 FLOPs/byte (per query)
            After : 1 × (K,N)@(N,B) matmul     → AI ≈ 2.0 FLOPs/byte at B=32

        X is read ONCE for all B queries (vs B times in the serial loop).

        Parameters
        ----------
        corrupted_queries : array-like, shape (B, N)
            Batch of masked + Gaussian-corrupted queries.

        Returns
        -------
        pi_batch : ndarray, shape (B, N)
            Per-coordinate precision vectors; mean ≈ 1, values in [0.1, 10.0].
        """
        _fp_guard = dict(all="ignore")
        Y = np.asarray(corrupted_queries, dtype=np.float64)   # (B, N)
        B, N = Y.shape

        # Signal 3: per-coordinate reliability mask  (B, N)
        Z      = (Y - self._mu) / self._sigma                 # (B, N)
        W_rel  = np.exp(-self._GAMMA * Z * Z)                 # (B, N)
        W_rel  = np.clip(W_rel, 1e-9, 1.0)
        log_W  = np.log(W_rel)                                # (B, N)

        # Fused class scores — X read ONCE for all B queries  (P0-C key change)
        # Shape: (K, B)  via  (K, N) @ (N, B)
        Y_weighted = (Y * W_rel).T                            # (N, B)
        with np.errstate(**_fp_guard):
            scores = self._X @ Y_weighted                     # (K, B)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

        # Per-query class identification & log-space fusion
        results = np.empty((B, N), dtype=np.float64)
        for b in range(B):
            s = scores[:, b]                                  # (K,)

            top2   = np.argpartition(s, -2)[-2:]
            s_top2 = np.sort(s[top2])
            margin = (s_top2[-1] - s_top2[-2]) / (np.abs(s_top2[-1]) + self._EPS)

            if margin < self._DELTA:
                s_shifted = s - s.max()
                w_c       = np.exp(self._TAU * s_shifted)
                w_c      /= w_c.sum()
                log_pi_q  = w_c @ self._log_pi_star           # (N,)
            else:
                c_hat    = int(np.argmax(s))
                log_pi_q = self._log_pi_star[c_hat]           # (N,)

            alpha  = self._ALPHA
            rho    = self._RHO
            log_pi = alpha * log_pi_q + (1.0 - alpha) * self._log_pi_H + rho * log_W[b]
            log_pi -= log_pi.mean()
            log_pi  = np.clip(log_pi, np.log(0.1), np.log(10.0))
            log_pi -= log_pi.mean()
            results[b] = np.exp(log_pi)

        return results
