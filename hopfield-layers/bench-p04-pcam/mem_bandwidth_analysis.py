"""
Memory Bandwidth Analysis — HHCC-Π + DiffusedHopfield Stack
=============================================================

Covers every hot kernel in the two-layer system:

    Layer A  bench-p04-pcam/adapters/myteam.py  (NumPy float64)
             ├── Engine.__init__   — offline Hessian table
             └── Engine.predict_precision — per-query inference

    Layer B  difflayers/  (PyTorch float32)
             ├── AttentionOperator (dense  O(N²d), graph O(kNd))
             └── DiffusionOperator × 4 modes (factored / simple /
                                               iterative / spectral)

For each kernel this script reports:
    • Theoretical bytes read   (R)
    • Theoretical bytes written(W)
    • Total theoretical bandwidth (R + W)
    • Floating-point operations  (FLOPs)
    • Arithmetic intensity  AI = FLOPs / (R + W)  [FLOPs/byte]
    • Empirical wall-clock time  [ms]
    • Empirical throughput  [GB/s]
    • Roofline bound  (memory-bandwidth-limited vs compute-limited)

Roofline uses an empirically measured peak DRAM bandwidth
(a calibration kernel is run at startup).

Usage
-----
    python bench-p04-pcam/mem_bandwidth_analysis.py

No CLI arguments needed; edit PARAMS below to vary problem size.
"""

from __future__ import annotations

import sys
import time
import textwrap
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Problem-size parameters — edit these to explore different regimes
# ---------------------------------------------------------------------------
@dataclass
class Params:
    K: int   = 64     # number of stored patterns   (attractor count)
    N: int   = 64     # pattern dimension            (feature length)
    d: int   = 64     # attention head dimension
    k: int   = 8      # kNN graph degree             (graph mode)
    T: int   = 3      # diffusion steps              (iterative mode)
    Q: int   = 200    # number of queries for timing the predict loop
    DTYPE_NP = np.float64
    DTYPE_PT = torch.float32
    BYTES_NP: int = field(default=8, init=False)  # float64 -> 8 bytes
    BYTES_PT: int = field(default=4, init=False)  # float32 -> 4 bytes

P = Params()

# ---------------------------------------------------------------------------
# Hardware calibration: measure peak DRAM bandwidth with a large copy kernel
# ---------------------------------------------------------------------------

def _measure_peak_bandwidth_gb_s(size_mb: int = 256) -> float:
    """Stream a large tensor through cache to measure DRAM bandwidth [GB/s]."""
    n_floats = (size_mb * 1024 * 1024) // 4
    a = torch.rand(n_floats, dtype=torch.float32)
    b = torch.empty_like(a)
    WARMUP = 3
    REPS   = 10
    for _ in range(WARMUP):
        b.copy_(a)
    t0 = time.perf_counter()
    for _ in range(REPS):
        b.copy_(a)
    elapsed = (time.perf_counter() - t0) / REPS
    # Read a + write b
    bw = (2 * n_floats * 4) / elapsed / 1e9
    return bw


PEAK_BW_GBS = _measure_peak_bandwidth_gb_s()

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KernelResult:
    name:      str
    bytes_r:   int       # bytes read
    bytes_w:   int       # bytes written
    flops:     int       # arithmetic ops
    time_ms:   float     # wall-clock
    reps:      int = 1   # repetitions used for timing

    @property
    def total_bytes(self) -> int:
        return self.bytes_r + self.bytes_w

    @property
    def ai(self) -> float:
        return self.flops / self.total_bytes if self.total_bytes else 0.0

    @property
    def empirical_bw_gbs(self) -> float:
        return self.total_bytes / (self.time_ms * 1e-3) / 1e9

    @property
    def peak_fraction(self) -> float:
        return self.empirical_bw_gbs / PEAK_BW_GBS

    @property
    def roofline_bound(self) -> str:
        # Ridge point: FLOP_peak / BW_peak  (rough CPU estimates)
        # ~200 GFLOP/s single-core peak, peak BW measured above
        ridge_ai = 200.0 / PEAK_BW_GBS
        return "compute" if self.ai > ridge_ai else "memory-BW"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time(fn, reps: int = 20) -> float:
    """Return median wall-clock time [ms] over `reps` calls."""
    times = []
    for _ in range(reps + 5):          # 5 warmup
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times[5:]))


def _fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.2f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.2f} KiB"
    return f"{b} B"


def _fmt_flops(f: int) -> str:
    if f >= 1e12:
        return f"{f/1e12:.2f} TF"
    if f >= 1e9:
        return f"{f/1e9:.2f} GF"
    if f >= 1e6:
        return f"{f/1e6:.2f} MF"
    return f"{f} F"


# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════
#  LAYER A — myteam.py  (NumPy float64)
# ═══════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

def analyse_engine_init(K: int, N: int) -> KernelResult:
    """
    Engine.__init__: offline Hessian table construction.

    Dominant sub-kernels (all float64 = 8 bytes):

    1. X @ R  → (K,N)@(N,N)→(K,N)
       R: (K·N + N²)·8   W: K·N·8

    2. Per-class loop  ×K:
       a) XR @ X[c]     (K,N)@(N,)→(K,)
          R: (K·N + N)·8   W: K·8   ≈ K·N·8
       b) XR * p_c      element-wise broadcast
          R: 2·K·N·8       W: K·N·8
       c) XR_w.T @ XR   (N,K)@(K,N)→(N,N)
          R: 2·K·N·8       W: N²·8
       d) XR.T @ p_c    (N,K)@(K,)→(N,)
          R: (K·N + K)·8   W: N·8  ≈ K·N·8
       e) outer product  (N,)→(N,N)
          R: 2·N·8         W: N²·8  ≈ N²·8
       f) H_c = I - β·(term1-term2) + H_sum accumulate
          R: 3·N²·8        W: N²·8  = 4·N²·8 (including accumulate)
       Per iteration: ≈ (5·K·N + 6·N²)·8

    3. linalg.inv of (N,N)  (LU: O(N³) flops, O(N²) memory traffic)
       R+W: 2·N²·8

    Total traffic (approx):
       (2·K·N + N²)·8   [X@R]
       + K·(5·K·N + 6·N²)·8   [loop]
       + 2·N²·8   [inv]
    """
    B = P.BYTES_NP
    bytes_xr     = (2 * K * N + N * N) * B
    bytes_loop   = K * (5 * K * N + 6 * N * N) * B
    bytes_inv    = 2 * N * N * B
    bytes_r = bytes_xr + bytes_loop + bytes_inv
    bytes_w = 0  # already included in R+W symmetric accounting
    total   = bytes_r  # we use read+write combined in bytes_r for simplicity

    # FLOPs:
    # X@R: 2·K·N²
    # loop: K × [2·K·N (XR@X[c]) + 2·K·N² (term1) + 2·K·N (XRTp) + N² (outer)]
    #     ≈ K × (4·K·N + 2·K·N² + N²)
    # inv: 2·N³/3  (Gaussian elimination)
    flops_xr   = 2 * K * N * N
    flops_loop = K * (4 * K * N + 2 * K * N * N + N * N)
    flops_inv  = (2 * N ** 3) // 3
    flops = flops_xr + flops_loop + flops_inv

    # Timing: build the actual Engine
    sys.path.insert(0, "bench-p04-pcam/adapters")
    try:
        from myteam import Engine
    except ImportError:
        from importlib.util import spec_from_file_location, module_from_spec
        import importlib
        spec = spec_from_file_location(
            "myteam",
            "bench-p04-pcam/adapters/myteam.py",
        )
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        Engine = mod.Engine

    rng = np.random.default_rng(0)
    X   = rng.standard_normal((K, N))
    R   = rng.standard_normal((N, N))
    R  /= np.linalg.norm(R)
    mp  = {"beta": 2.0, "R": R}

    def _fn():
        Engine(X, mp)

    t = _time(_fn, reps=5)
    return KernelResult(
        name="Engine.__init__ (Hessian table)",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t, reps=5,
    )


def analyse_engine_predict(K: int, N: int, Q_queries: int) -> KernelResult:
    """
    Engine.predict_precision: per-query inference (Q_queries queries).

    Dominant sub-kernels (float64):

    1. X @ (y * w_rel)   (K,N)@(N,)→(K,)
       R: (K·N + N)·8   W: K·8   ≈ K·N·8

    2. w_c @ log_pi_star  (K,)@(K,N)→(N,)   [soft-blend path]
       R: (K + K·N)·8   W: N·8   ≈ K·N·8

    3. Log-space fusion + normalise: O(N) element-wise
       R+W: 5·N·8

    Per-query total ≈ 2·K·N·8 + 5·N·8
    """
    B = P.BYTES_NP
    per_query  = (2 * K * N + 5 * N) * B
    total      = per_query * Q_queries

    flops_per  = 2 * K * N + 2 * K * N + 5 * N  # two matvecs + fusion
    flops      = flops_per * Q_queries

    sys.path.insert(0, "bench-p04-pcam/adapters")
    try:
        from myteam import Engine
    except ImportError:
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location(
            "myteam", "bench-p04-pcam/adapters/myteam.py",
        )
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        Engine = mod.Engine

    rng = np.random.default_rng(0)
    X   = rng.standard_normal((K, N))
    R   = rng.standard_normal((N, N))
    R  /= np.linalg.norm(R)
    mp  = {"beta": 2.0, "R": R}
    eng = Engine(X, mp)
    ys  = rng.standard_normal((Q_queries, N))

    def _fn():
        for y in ys:
            eng.predict_precision(y)

    t = _time(_fn, reps=20)
    return KernelResult(
        name=f"Engine.predict_precision (×{Q_queries} queries)",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t, reps=20,
    )


# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════
#  LAYER B — AttentionOperator  (PyTorch float32)
# ═══════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

def analyse_attention_dense(N: int, d: int) -> KernelResult:
    """
    Dense attention: logits=β·Q@Kᵀ → softmax → weights@V.

    Bytes (float32 = 4):
      Q@Kᵀ   : R=(2Nd)·4  W=N²·4        → (2Nd+N²)·4
      softmax : R=N²·4    W=N²·4        → 2N²·4
      w@V     : R=(N²+Nd)·4  W=Nd·4    → (N²+2Nd)·4
    Total: (3N²+4Nd)·4

    FLOPs:
      Q@Kᵀ  : 2·N²·d
      softmax: ≈4·N²  (exp + sum + div per row)
      w@V   : 2·N²·d
    Total: 4·N²·d + 4·N²
    """
    B = P.BYTES_PT
    bytes_qk  = (2 * N * d + N * N) * B
    bytes_sm  = 2 * N * N * B
    bytes_wv  = (N * N + 2 * N * d) * B
    total = bytes_qk + bytes_sm + bytes_wv

    flops = 4 * N * N * d + 4 * N * N

    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    beta = 1.0

    def _fn():
        logits  = beta * (Q @ K.t())
        weights = F.softmax(logits, dim=-1)
        _out    = weights @ V

    t = _time(_fn)
    return KernelResult(
        name="AttentionOperator dense (2D)",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t,
    )


def analyse_attention_graph(N: int, d: int, k: int) -> KernelResult:
    """
    Graph attention: attend only to k neighbors per query.

    Bytes (float32 = 4):
      gather K[adj]   : R=N·k·d·4   (indexed read, worst-case random)
      gather V[adj]   : R=N·k·d·4
      dot products    : R=(Nd+Nkd)·4  W=Nk·4  → (Nd+2Nkd+Nk)·4
        (Q.unsqueeze × K_nbrs).sum
      softmax         : R=Nk·4  W=Nk·4
      weighted sum    : R=(Nkd+Nk)·4  W=Nd·4  → (Nkd+Nk+Nd)·4

    Total ≈ (4Nkd + 3Nk + 2Nd)·4
    Ratio to dense: 4kd / (3N+4d) ≈ 4k/3N  for large N.
    """
    B = P.BYTES_PT
    total = (4 * N * k * d + 3 * N * k + 2 * N * d) * B
    flops = 4 * N * k * d + 4 * N * k   # dot + weighted sum

    Q   = torch.randn(N, d)
    K   = torch.randn(N, d)
    V   = torch.randn(N, d)
    adj = torch.randint(0, N, (N, k))
    beta = 1.0

    def _fn():
        K_nbrs   = K[adj]
        V_nbrs   = V[adj]
        logits   = beta * (Q.unsqueeze(1) * K_nbrs).sum(dim=-1)
        weights  = F.softmax(logits, dim=-1)
        _out     = (weights.unsqueeze(-1) * V_nbrs).sum(dim=1)

    t = _time(_fn)
    return KernelResult(
        name=f"AttentionOperator graph (k={k})",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t,
    )


# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════
#  LAYER B — DiffusionOperator  (PyTorch float32)
# ═══════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

def analyse_diffusion_factored(N: int, d: int, k: int, T: int) -> KernelResult:
    """
    FactoredDiffusion per forward call (T steps).

    x' = (1 - η·deg)⊙x + η·W@x    [applied T times]

    Bytes per step (sparse W with k·N nnz, float32):
      deg broadcast: R=(N+Nd)·4  W=Nd·4  → (2Nd+N)·4
      W @ x (sparse): R=(kN+Nd)·4  W=Nd·4  → (kN+2Nd)·4
      accumulate:    R=2Nd·4  W=Nd·4  → 3Nd·4
    Per step total: ≈ (kN + 7Nd + N)·4

    T steps total:  T·(kN + 7Nd + N)·4
    """
    B = P.BYTES_PT
    per_step = (k * N + 7 * N * d + N) * B
    total    = T * per_step
    flops    = T * (2 * k * N * d + 2 * N * d)  # W@x + elementwise

    # Build sparse W and deg
    row_idx = torch.arange(N).repeat_interleave(k)
    col_idx = torch.randint(0, N, (N * k,))
    vals    = torch.ones(N * k)
    W_sp    = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]), vals, (N, N)
    ).coalesce()
    deg     = torch.full((N,), float(k))
    eta     = 0.1
    X       = torch.randn(N, d)

    def _fn():
        x = X
        for _ in range(T):
            x = (1.0 - eta * deg).unsqueeze(1) * x + eta * torch.sparse.mm(W_sp, x)

    t = _time(_fn)
    return KernelResult(
        name=f"FactoredDiffusion (T={T}, k={k})",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t,
    )


def analyse_diffusion_simple(N: int, d: int) -> KernelResult:
    """
    SimpleDiffusion: X' = D @ X, D = I - η·L  (dense, precomputed once).

    Bytes per forward call:
      D @ X   : R=(N²+Nd)·4  W=Nd·4  → (N²+2Nd)·4

    Precompute D (once):
      R=(N²)·4  W=N²·4  → 2N²·4  (not amortised here — charged separately)
    """
    B = P.BYTES_PT
    total_precompute = 2 * N * N * B
    total_forward    = (N * N + 2 * N * d) * B
    flops_precompute = N * N          # subtraction I - η*L
    flops_forward    = 2 * N * N * d  # matmul

    L = torch.rand(N, N)
    L = (L + L.t()) / 2  # symmetric
    eta = 0.1
    D   = torch.eye(N) - eta * L
    X   = torch.randn(N, d)

    def _fn():
        return D @ X

    t = _time(_fn)
    # Report forward cost only (precompute is one-time)
    return KernelResult(
        name="SimpleDiffusion forward (dense D)",
        bytes_r=total_forward + total_precompute,
        bytes_w=0, flops=flops_forward + flops_precompute,
        time_ms=t,
    )


def analyse_diffusion_iterative(N: int, d: int, T: int) -> KernelResult:
    """
    IterativeDiffusion: X' = D^T @ X  (T matmuls with same D).

    Bytes per call: T × (N²+2Nd)·4
    """
    B = P.BYTES_PT
    total = T * (N * N + 2 * N * d) * B
    flops = T * 2 * N * N * d

    L   = torch.rand(N, N); L = (L + L.t()) / 2
    eta = 0.1
    D   = torch.eye(N) - eta * L
    X   = torch.randn(N, d)

    def _fn():
        x = X
        for _ in range(T):
            x = D @ x
        return x

    t = _time(_fn)
    return KernelResult(
        name=f"IterativeDiffusion (T={T} steps)",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t,
    )


def analyse_diffusion_spectral_precompute(N: int) -> KernelResult:
    """
    SpectralDiffusion.precompute: builds H = U exp(-η Λ) Uᵀ.

    Bytes:
      eigendecomp (eigh): R=N²·4  W=(N²+N)·4  → (2N²+N)·4
      U @ diag(exp(-ηΛ)): R=(N²+N)·4  W=N²·4  → (2N²+N)·4
      Uᵀ multiply:        R=(N²+N²)·4  W=N²·4 → 3N²·4
    Total precompute: ~(7N²+2N)·4

    FLOPs: O(N³) for eigendecomposition
    """
    B = P.BYTES_PT
    total = (7 * N * N + 2 * N) * B
    flops = (9 * N ** 3) // 2  # symmetric eigendecomp ≈ 4.5 N³

    L = torch.rand(N, N); L = (L + L.t()) / 2
    eta = 0.1

    def _fn():
        eigenvalues, U = torch.linalg.eigh(L)
        H = U @ torch.diag(torch.exp(-eta * eigenvalues)) @ U.t()
        return H

    t = _time(_fn, reps=5)
    return KernelResult(
        name="SpectralDiffusion.precompute (eigendecomp)",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t, reps=5,
    )


def analyse_diffusion_spectral_forward(N: int, d: int) -> KernelResult:
    """
    SpectralDiffusion forward: X' = H @ X  (single dense matmul).

    Bytes: (N²+2Nd)·4   (same as SimpleDiffusion forward)
    """
    B = P.BYTES_PT
    total = (N * N + 2 * N * d) * B
    flops = 2 * N * N * d

    L = torch.rand(N, N); L = (L + L.t()) / 2
    eta = 0.1
    eigenvalues, U = torch.linalg.eigh(L)
    H = U @ torch.diag(torch.exp(-eta * eigenvalues)) @ U.t()
    X = torch.randn(N, d)

    def _fn():
        return H @ X

    t = _time(_fn)
    return KernelResult(
        name="SpectralDiffusion forward (single matmul)",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t,
    )


# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════
#  Full forward pass: DiffusedHopfield (factored + dense attention, T steps)
# ═══════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

def analyse_full_forward(N: int, d: int, k: int, T: int) -> KernelResult:
    """
    One complete forward pass of DiffusedHopfield:

        for t in range(T):
            K' = FactoredDiffusion(K)        — O(kNd)
            Q' = FactoredDiffusion(Q)        — O(kNd)  [if diffuse_query=True]
            Q  = DenseAttention(Q', K', V)   — O(N²d)

    Bytes (float32):
      2 × T × FactoredDiffusion: 2·T·(kN + 7Nd + N)·4
      T × DenseAttention:        T·(3N² + 4Nd)·4
    """
    B = P.BYTES_PT
    bytes_diffuse = 2 * T * (k * N + 7 * N * d + N) * B
    bytes_attn    = T * (3 * N * N + 4 * N * d) * B
    total         = bytes_diffuse + bytes_attn

    flops_diffuse = 2 * T * (2 * k * N * d + 2 * N * d)
    flops_attn    = T * (4 * N * N * d + 4 * N * N)
    flops         = flops_diffuse + flops_attn

    row_idx = torch.arange(N).repeat_interleave(k)
    col_idx = torch.randint(0, N, (N * k,))
    vals    = torch.ones(N * k)
    W_sp    = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]), vals, (N, N)
    ).coalesce()
    deg  = torch.full((N,), float(k))
    eta  = 0.1
    beta = 1.0

    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    def _factored(x):
        for _ in range(1):  # single step per outer iteration
            x = (1.0 - eta * deg).unsqueeze(1) * x + eta * torch.sparse.mm(W_sp, x)
        return x

    def _fn():
        q, k_pat = Q.clone(), K.clone()
        for _ in range(T):
            k_pat = _factored(k_pat)
            q     = _factored(q)
            logits  = beta * (q @ k_pat.t())
            weights = F.softmax(logits, dim=-1)
            q       = weights @ V

    t = _time(_fn)
    return KernelResult(
        name=f"Full DiffusedHopfield forward (T={T}, k={k})",
        bytes_r=total, bytes_w=0, flops=flops,
        time_ms=t,
    )


# ---------------------------------------------------------------------------
# Asymptotic scaling table
# ---------------------------------------------------------------------------

def print_scaling_table(K: int, N: int, d: int, k: int, T: int) -> None:
    B8 = 8  # float64
    B4 = 4  # float32

    rows = [
        # (label, order_expr, formula_str)
        ("Engine.__init__",
         (4 * K * K * N + 10 * K * N * N) * B8,
         f"(4K²N + 10KN²)×8  [K={K},N={N}]"),
        ("Engine.predict_precision (per query)",
         (2 * K * N + 5 * N) * B8,
         f"(2KN + 5N)×8"),
        ("AttentionOperator dense",
         (3 * N * N + 4 * N * d) * B4,
         f"(3N² + 4Nd)×4  [N={N},d={d}]"),
        ("AttentionOperator graph",
         (4 * N * k * d + 3 * N * k + 2 * N * d) * B4,
         f"(4Nkd + 3Nk + 2Nd)×4  [k={k}]"),
        ("FactoredDiffusion (per step)",
         (k * N + 7 * N * d + N) * B4,
         f"(kN + 7Nd + N)×4  [k={k}]"),
        ("SimpleDiffusion (per step)",
         (N * N + 2 * N * d) * B4,
         f"(N² + 2Nd)×4"),
        ("SpectralDiffusion precompute",
         (7 * N * N + 2 * N) * B4,
         f"(7N² + 2N)×4  [O(N³) FLOPs]"),
        ("Full forward (factored+dense, T steps)",
         (2 * T * (k * N + 7 * N * d + N) + T * (3 * N * N + 4 * N * d)) * B4,
         f"T·[(2(kN+7Nd+N) + (3N²+4Nd))]×4  [T={T}]"),
    ]

    COL = 52
    print()
    print("─" * 100)
    print(f"{'Kernel':<{COL}}  {'Theoretical BW':>16}  {'Formula'}")
    print("─" * 100)
    for label, bw_bytes, formula in rows:
        print(f"{label:<{COL}}  {_fmt_bytes(bw_bytes):>16}  {formula}")
    print("─" * 100)


# ---------------------------------------------------------------------------
# Main output
# ---------------------------------------------------------------------------

def print_results(results: List[KernelResult]) -> None:
    header = (
        f"{'Kernel':<52} {'BW(R+W)':>10} {'FLOPs':>10} "
        f"{'AI':>8} {'t[ms]':>8} {'Emp.BW':>10} {'%Peak':>7} {'Bound':>11}"
    )
    sep = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        bnd   = r.roofline_bound
        emoji = "⚡" if bnd == "compute" else "💾"
        print(
            f"{r.name:<52} "
            f"{_fmt_bytes(r.total_bytes):>10} "
            f"{_fmt_flops(r.flops):>10} "
            f"{r.ai:>8.2f} "
            f"{r.time_ms:>8.3f} "
            f"{r.empirical_bw_gbs:>8.2f} GB/s "
            f"{r.peak_fraction*100:>5.1f}% "
            f"{emoji} {bnd}"
        )
    print(sep)


def main() -> None:
    K, N, d, k, T, Q_q = P.K, P.N, P.d, P.k, P.T, P.Q

    print()
    print("=" * 80)
    print("  Memory Bandwidth Analysis  —  HHCC-Π + DiffusedHopfield")
    print("=" * 80)
    print(f"  Parameters: K={K}, N={N}, d={d}, k={k}, T={T}, Q={Q_q}")
    print(f"  float64 (NumPy)  = 8 B/element")
    print(f"  float32 (PyTorch)= 4 B/element")
    print(f"  Peak DRAM BW     = {PEAK_BW_GBS:.2f} GB/s  (calibrated)")
    print()

    # ── Theoretical scaling table ─────────────────────────────────────────
    print("\n[ THEORETICAL BANDWIDTH SCALING TABLE ]")
    print_scaling_table(K, N, d, k, T)

    # ── Empirical kernel benchmarks ──────────────────────────────────────
    print("\n[ EMPIRICAL KERNEL BENCHMARKS ]")
    results = []

    print("  Benchmarking Engine.__init__ …", end=" ", flush=True)
    results.append(analyse_engine_init(K, N))
    print("done")

    print("  Benchmarking Engine.predict_precision …", end=" ", flush=True)
    results.append(analyse_engine_predict(K, N, Q_q))
    print("done")

    print("  Benchmarking AttentionOperator dense …", end=" ", flush=True)
    results.append(analyse_attention_dense(N, d))
    print("done")

    print("  Benchmarking AttentionOperator graph …", end=" ", flush=True)
    results.append(analyse_attention_graph(N, d, k))
    print("done")

    print("  Benchmarking FactoredDiffusion …", end=" ", flush=True)
    results.append(analyse_diffusion_factored(N, d, k, T))
    print("done")

    print("  Benchmarking SimpleDiffusion …", end=" ", flush=True)
    results.append(analyse_diffusion_simple(N, d))
    print("done")

    print("  Benchmarking IterativeDiffusion …", end=" ", flush=True)
    results.append(analyse_diffusion_iterative(N, d, T))
    print("done")

    print("  Benchmarking SpectralDiffusion precompute …", end=" ", flush=True)
    results.append(analyse_diffusion_spectral_precompute(N))
    print("done")

    print("  Benchmarking SpectralDiffusion forward …", end=" ", flush=True)
    results.append(analyse_diffusion_spectral_forward(N, d))
    print("done")

    print("  Benchmarking full DiffusedHopfield forward …", end=" ", flush=True)
    results.append(analyse_full_forward(N, d, k, T))
    print("done")

    print_results(results)

    # ── Key findings ──────────────────────────────────────────────────────
    worst_ai  = min(results, key=lambda r: r.ai)
    most_bw   = max(results, key=lambda r: r.empirical_bw_gbs)
    bottlenck = max(results, key=lambda r: r.time_ms)

    print()
    print("[ KEY FINDINGS ]")
    print(textwrap.dedent(f"""
    Lowest arithmetic intensity (most memory-bound):
      → {worst_ai.name}
         AI = {worst_ai.ai:.2f} FLOPs/byte   (lower = more memory-bound)

    Highest empirical bandwidth achieved:
      → {most_bw.name}
         {most_bw.empirical_bw_gbs:.2f} GB/s  ({most_bw.peak_fraction*100:.1f}% of peak)

    Slowest kernel (wall-clock):
      → {bottlenck.name}
         {bottlenck.time_ms:.3f} ms

    Attention mode comparison (dense vs graph):
      Dense  total BW = {_fmt_bytes((3*N*N + 4*N*d)*P.BYTES_PT)}  →  O(N²)
      Graph  total BW = {_fmt_bytes((4*N*k*d + 3*N*k + 2*N*d)*P.BYTES_PT)}  →  O(kNd),  speedup ratio ≈ {(3*N*N + 4*N*d) / (4*N*k*d + 3*N*k + 2*N*d):.1f}×

    Diffusion mode bandwidth comparison (per step):
      Factored  (k={k}):  {_fmt_bytes((k*N + 7*N*d + N)*P.BYTES_PT)}  [O(kNd)]
      Simple        :  {_fmt_bytes((N*N + 2*N*d)*P.BYTES_PT)}  [O(N²d)]
      Iterative ×T={T}: {_fmt_bytes(T*(N*N + 2*N*d)*P.BYTES_PT)}  [O(T·N²d)]
      Spectral (pre) :  {_fmt_bytes((7*N*N + 2*N)*P.BYTES_PT)}  [O(N³) FLOPs]

    Engine init dominates at K={K}, N={N}:
      Hessian loop  ≈ {_fmt_bytes((4*K*K*N + 10*K*N*N)*P.BYTES_NP)}  (float64, O(K²N + KN²))
      Per-query run ≈ {_fmt_bytes((2*K*N + 5*N)*P.BYTES_NP)} per query  (fast path)
    """))


if __name__ == "__main__":
    main()
