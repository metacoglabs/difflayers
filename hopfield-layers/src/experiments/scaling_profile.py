"""
§11.3 Scaling Profile Sweep — Phase 0 Deliverable
===================================================

Sweeps the full DiffusedHopfield forward pass across
n ∈ {200, 500, 1000, 2000, 5000} at fixed d=64, k=8, T=3 and
records the §11.3 profiling table:

    n             — number of stored patterns
    X_mb          — pattern matrix size in MB  (n × d × 4 bytes)
    theor_bytes   — theoretical R+W bandwidth per forward pass
    ms_per_step   — wall-clock time per diffusion-attention step [ms]
    total_ms      — total forward pass wall-clock [ms]
    empirical_gb_s — empirical memory bandwidth [GB/s]
    ai            — arithmetic intensity [FLOPs/byte]
    peak_bw_pct   — empirical bandwidth as % of calibrated peak DRAM BW
    qps           — queries per second (single-query serial mode)

The "full forward pass" is the default production config:
    FactoredDiffusion (K) + FactoredDiffusion (Q) + DenseAttention  ×  T steps

Bandwidth formulas are derived from bench-p04-pcam/mem_bandwidth_analysis.py
(analyse_full_forward) — kept inline so this file has no dependency on that
bench-specific module.

Output
------
    results/scaling_profile.csv   — §11.3 table as CSV
    stdout                        — formatted table + scaling observations

Usage
-----
    python main.py --exp profile
    python main.py --exp profile --d 64 --k 8 --steps 3
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Scale points  (PRD §11.3)
# ---------------------------------------------------------------------------

SCALE_POINTS: List[int] = [200, 500, 1000, 2000, 5000]

_REPS   = 20   # timing repetitions per n
_WARMUP = 5    # warmup iterations before timing


# ---------------------------------------------------------------------------
# Peak DRAM bandwidth calibration
# ---------------------------------------------------------------------------

def _calibrate_peak_bw(size_mb: int = 256) -> float:
    """
    Measure peak DRAM bandwidth [GB/s] via a stream-copy kernel.

    Reads ``size_mb`` MB of float32 and writes ``size_mb`` MB.
    Reports total bytes transferred / elapsed time.
    """
    n_floats = (size_mb * 1024 * 1024) // 4
    a = torch.rand(n_floats, dtype=torch.float32)
    b = torch.empty_like(a)
    for _ in range(3):       # warmup
        b.copy_(a)
    t0 = time.perf_counter()
    for _ in range(10):
        b.copy_(a)
    elapsed = (time.perf_counter() - t0) / 10
    return (2 * n_floats * 4) / elapsed / 1e9  # read + write


# ---------------------------------------------------------------------------
# Theoretical bandwidth and FLOPs
# (derived from bench-p04-pcam/mem_bandwidth_analysis.py, analyse_full_forward)
# ---------------------------------------------------------------------------

def _theor_bytes(N: int, d: int, k: int, T: int) -> int:
    """
    Total theoretical R+W bytes for one full DiffusedHopfield forward pass.

    FactoredDiffusion (K + Q):  2 × T × (k·N + 7·N·d + N) × 4 bytes
    DenseAttention:                 T × (3·N² + 4·N·d)    × 4 bytes
    """
    B = 4  # float32
    diffuse = 2 * T * (k * N + 7 * N * d + N) * B
    attn    =     T * (3 * N * N + 4 * N * d) * B
    return diffuse + attn


def _theor_flops(N: int, d: int, k: int, T: int) -> int:
    """Total arithmetic operations for one full forward pass."""
    diffuse = 2 * T * (2 * k * N * d + 2 * N * d)
    attn    =     T * (4 * N * N * d + 4 * N * N)
    return diffuse + attn


# ---------------------------------------------------------------------------
# Wall-clock timer
# ---------------------------------------------------------------------------

def _time_forward(N: int, d: int, k: int, T: int) -> float:
    """
    Time one complete DiffusedHopfield forward pass [ms] (median over _REPS runs).

    Replicates the full production loop (DynamicsEngine.run_dynamics default config):

        for t in range(T):
            K = FactoredDiffusion(K)         — O(kNd) sparse
            Q = FactoredDiffusion(Q)         — O(kNd) sparse
            Q = softmax(β·Q·Kᵀ) · V          — O(N²d) dense attention

    Uses a k-regular random graph as a stand-in for the real kNN graph.
    This is conservative: real kNN graphs have the same sparsity pattern
    and produce the same FLOPs.
    """
    # Build k-regular sparse adjacency (same structure as real FactoredDiffusion)
    row_idx = torch.arange(N).repeat_interleave(k)
    col_idx = torch.randint(0, N, (N * k,))
    vals    = torch.ones(N * k)
    W_sp = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]), vals, (N, N)
    ).coalesce()
    deg  = torch.full((N,), float(k))
    eta  = 0.1
    beta = 1.0

    Q0 = torch.randn(N, d)
    K0 = torch.randn(N, d)
    V  = torch.randn(N, d)

    def _forward():
        Q = Q0.clone()
        K = K0.clone()
        for _ in range(T):
            # FactoredDiffusion: x' = (1 - η·deg)⊙x + η·W@x
            K = (1.0 - eta * deg).unsqueeze(1) * K + eta * torch.sparse.mm(W_sp, K)
            Q = (1.0 - eta * deg).unsqueeze(1) * Q + eta * torch.sparse.mm(W_sp, Q)
            # Dense scaled-dot-product attention
            logits  = beta * (Q @ K.t())       # (N, N)
            weights = F.softmax(logits, dim=-1) # (N, N)
            Q       = weights @ V               # (N, d)

    for _ in range(_WARMUP):
        _forward()

    times = []
    for _ in range(_REPS):
        t0 = time.perf_counter()
        _forward()
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.2f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.2f} KiB"
    return f"{b} B"


def _print_table(df: pd.DataFrame, d: int, k: int, T: int, peak_bw: float) -> None:
    """Print the §11.3 formatted table to stdout."""
    SEP = "─" * 90
    HDR = (
        f"{'n':>6}  {'X (MB)':>8}  {'Theor BW':>10}  "
        f"{'ms/step':>9}  {'total ms':>9}  {'GB/s':>8}  {'%Peak':>7}  {'QPS':>8}"
    )
    print()
    print(f"  §11.3 Scaling Profile  —  d={d}, k={k}, T={T} steps  "
          f"(peak DRAM BW = {peak_bw:.2f} GB/s)")
    print("  " + SEP)
    print("  " + HDR)
    print("  " + SEP)
    for _, row in df.iterrows():
        tag = "  ← Phase 1 target" if int(row["n"]) == 5000 else ""
        print(
            f"  {int(row['n']):>6}  "
            f"{row['X_mb']:>8.4f}  "
            f"{_fmt_bytes(int(row['theor_bytes'])):>10}  "
            f"{row['ms_per_step']:>9.3f}  "
            f"{row['total_ms']:>9.3f}  "
            f"{row['empirical_gb_s']:>8.2f}  "
            f"{row['peak_bw_pct']:>6.1f}%  "
            f"{row['qps']:>8.1f}"
            f"{tag}"
        )
    print("  " + SEP)

    # Scaling observations
    if len(df) >= 2:
        first   = df.iloc[0]
        last    = df.iloc[-1]
        n_ratio = last["n"] / first["n"]
        t_ratio = (last["total_ms"] / first["total_ms"]
                   if first["total_ms"] > 0 else float("nan"))
        expected_quadratic = n_ratio ** 2
        print()
        print(f"  Scaling observations  (n: {int(first['n'])} → {int(last['n'])}, {n_ratio:.0f}×)")
        print(f"    time scale-up : {t_ratio:.1f}×  "
              f"(O(N²) would be {expected_quadratic:.0f}×, O(N) would be {n_ratio:.0f}×)")
        regime = (
            "dense attention O(N²) dominates"
            if t_ratio > n_ratio * 1.5
            else "FactoredDiffusion O(kN) dominates — attention not yet bottleneck"
        )
        print(f"    regime        : {regime}")
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_scaling_sweep(
    d: int = 64,
    k: int = 8,
    T: int = 3,
    n_points: Optional[List[int]] = None,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run the §11.3 scaling profile sweep and save ``results/scaling_profile.csv``.

    Args:
        d:           Pattern dimensionality (default 64).
        k:           kNN graph degree (default 8).
        T:           Diffusion steps per forward pass (default 3).
        n_points:    n values to sweep. Defaults to PRD §11.3 SCALE_POINTS.
        results_dir: Output directory.

    Returns:
        DataFrame with one row per n value and columns:
        n, X_mb, theor_bytes, ms_per_step, total_ms, empirical_gb_s, ai,
        peak_bw_pct, qps.
    """
    if n_points is None:
        n_points = SCALE_POINTS

    print("  Calibrating peak DRAM bandwidth …", end=" ", flush=True)
    peak_bw = _calibrate_peak_bw()
    print(f"{peak_bw:.2f} GB/s")
    print()

    rows = []
    for n in n_points:
        print(f"  Profiling n={n:>5}  d={d}  k={k}  T={T} …", end=" ", flush=True)
        X_mb         = n * d * 4 / 1e6
        theor_bytes  = _theor_bytes(n, d, k, T)
        theor_flops  = _theor_flops(n, d, k, T)
        total_ms     = _time_forward(n, d, k, T)
        ms_per_step  = total_ms / T
        emp_gb_s     = theor_bytes / (total_ms * 1e-3) / 1e9
        ai           = theor_flops / theor_bytes if theor_bytes else 0.0
        peak_pct     = emp_gb_s / peak_bw * 100.0
        qps          = 1000.0 / total_ms  # single-query serial throughput

        rows.append({
            "n":               n,
            "X_mb":            round(X_mb, 4),
            "theor_bytes":     theor_bytes,
            "ms_per_step":     round(ms_per_step, 4),
            "total_ms":        round(total_ms, 4),
            "empirical_gb_s":  round(emp_gb_s, 4),
            "ai":              round(ai, 4),
            "peak_bw_pct":     round(peak_pct, 2),
            "qps":             round(qps, 2),
        })

        print(
            f"{total_ms:7.3f} ms  "
            f"({ms_per_step:.3f} ms/step  "
            f"{emp_gb_s:.2f} GB/s  "
            f"{peak_pct:.1f}% peak)"
        )

    df = pd.DataFrame(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(results_dir, "scaling_profile.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    _print_table(df, d, k, T, peak_bw)
    return df
