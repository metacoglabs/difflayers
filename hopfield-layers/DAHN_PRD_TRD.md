# DAHN — Diffusion-Augmented Hopfield Networks
## Product & Technical Requirements Document

**Version:** 1.0 — June 2026  
**Status:** Active Research  
**Authors:** Ghosh, P. · Jaiswal, K. · Gupta, S.  
**Companion to:** NeurIPS 2026 Submission & IEEE ICACI 2026

---

# PART I — PRODUCT REQUIREMENTS DOCUMENT (PRD)

## §1 Executive Summary

Diffusion-Augmented Hopfield Networks (DAHN) is a graph-regularized associative memory framework that extends Modern Hopfield Networks (MHN) through two orthogonal augmentations:

1. **Graph Laplacian diffusion** as a pre-retrieval manifold-smoothing operator on stored patterns and queries.
2. A **time-varying precision operator** that continuously deforms in-basin attractor geometry at inference time without modifying stored representations.

Together, these mechanisms form a mathematically rigorous, experimentally benchmarkable associative retrieval system — positioned between the one-shot fixed-landscape retrieval of standard MHN and the full thermodynamic dynamics of Extropic-style hardware.

### Positioning Statement

DAHN is not a reproduction of the full PCAM research stack. It is a clean, composable extension of the `hflayers` / Hopfield attention library with:
- (a) provable convergence guarantees
- (b) graph-structured memory topology
- (c) inference-time precision control

— building a system that can be benchmarked against classical Hopfield, MHN, and PCAM baselines on standard vision and multi-modal tasks.

---

## §2 Problem Statement

### 2.1 Gaps in Modern Hopfield Networks

Modern Hopfield Networks (Ramsauer et al., 2021) achieve exponential storage capacity but exhibit four fundamental limitations:

- **One-shot retrieval:** a single softmax pass over stored patterns `Q → softmax(QKᵀ)V` commits to a fixed retrieval without iterative refinement.
- **Independent memory states:** stored patterns are treated as an unstructured set with no relational topology between them.
- **Static energy landscape:** the location, depth, and basin geometry of each attractor are determined entirely at design time by `X` and `β`.
- **No inference-time control:** the only knobs available are the query itself and (optionally) the softmax temperature `β`.

### 2.2 The DAHN Hypothesis

> **Core Hypothesis**  
> Associative retrieval improves when memory states interact over a structured semantic manifold before and during retrieval. Specifically: (a) graph Laplacian diffusion of keys and queries smooths the energy landscape in the phase-transition noise regime, and (b) a precision operator Π on the gradient vector field gives provably-Lipschitz inference-time control of attractor locations.

### 2.3 What Is Being Built

| Component | Description |
|-----------|-------------|
| `DiffusedHopfield` | Drop-in Hopfield replacement with pre-attention Laplacian smoothing of K and Q via `D = I − ηL` |
| `DynamicsEngine` | Full T-step interleaved diffuse→attend loop (currently dead code — wiring it in is a primary deliverable) |
| `AttentionOperator` | Dense O(N²) and graph-sparse O(kN) attention backends; graph mode unreachable (another primary deliverable) |
| PCAM Precision Module | Inference-time precision operator Π(t) as per NeurIPS 2026 paper; to be integrated as an optional layer on top of `DiffusedHopfield` |
| Benchmark Suite | Unified evaluation harness: mask, salt-and-pepper, Gaussian noise on MNIST-PCA and raw MNIST; retrieval accuracy, energy dissipation, Lipschitz constant measurement |

---

## §3 Goals & Non-Goals

### 3.1 Goals

1. Wire `DynamicsEngine.run_dynamics()` into `DiffusedHopfield._associate()` as the primary forward path, replacing the current single-step diffusion shortcut.
2. Make `AttentionOperator`'s graph-sparse O(kN) backend reachable and benchmarked against the dense O(N²) backend.
3. Resolve the `EnergyTracker` / factored-diffusion incompatibility so energy is trackable in the default operating mode.
4. Integrate the PCAM precision operator (Theorem 5, NeurIPS 2026) as an optional composable module.
5. Establish a reproducible benchmark suite with statistical reporting (mean ± SD across K=10 seeds) that matches the NeurIPS 2026 paper protocol.
6. Demonstrate measurable retrieval improvement over MHN baseline in the noise phase-transition regime (p = 0.20–0.35 mask corruption).
7. Provide memory-bandwidth profiling at varying n (200 → 5000) to characterise the scaling ceiling on RTX 3090 hardware.

### 3.2 Non-Goals

- Does **not** aim to reproduce the full biologically-faithful PCAM or metacognitive substrate research stack.
- No new training procedures or learned pattern encoders — patterns are stored directly (zero-shot associative memory).
- No quantum, neuromorphic, or Extropic hardware targets in this phase.
- No deployment API, inference server, or productionised model weights.
- No direct image-space Hopfield retrieval (d > 64 via PCA is the current scope boundary).

---

## §4 Success Metrics & Acceptance Criteria

| Metric | Baseline (MHN) | DAHN Target |
|--------|---------------|-------------|
| Retrieval accuracy @ p=0.30 mask | ~59% | ≥ 62% (+3pp over MHN) |
| Retrieval accuracy @ p=0.20 mask | ~94% | ≥ 95% (preserve) |
| Energy dissipation | N/A (one-shot) | ≥ 99% reduction, ≤ 0.05% upward excursions |
| Lipschitz bound empirical vs. theory | N/A | Empirical C̃ ≤ Theoretical C (Theorem 5) |
| Multi-well coverage @ β=16, n≤200 | N/A | ≥ 95% of patterns have corresponding attractor |
| `run_dynamics` wired in | Not wired | All forward passes go through T-step loop |
| Graph O(kN) backend reachable | Unreachable | Benchmarked vs. dense O(N²), ≤ 5% accuracy gap |
| Scaling table produced | None | n ∈ {200,500,1000,2000,5000}: MB, GB/s, ms/step |

---

## §5 Stakeholders & Roles

| Role | Responsibility |
|------|---------------|
| Research Lead (Ghosh, P.) | Mathematical framework, theorem validation, NeurIPS / IEEE paper ownership |
| Engineer (Jaiswal, K.) | Core implementation, DynamicsEngine wiring, FAISS integration, profiling |
| Co-author (Gupta, S.) | Experimental design, benchmark protocol, statistical reporting |
| External Reviewers (NeurIPS PAT) | Formal correctness auditing — resolved issues documented in Appendix I of NeurIPS paper |

---

## §6 High-Level Timeline

| Phase / Dates | Focus | Key Deliverable |
|---------------|-------|-----------------|
| Phase 0 — Weeks 1–4 (Jun 2026) | Profiling + dead-code audit | Baseline profiling report; DynamicsEngine wired in |
| Phase 1 — Months 1–3 (Jul–Sep 2026) | Algorithmic scaling + validation | FAISS integration; n=5000 validation of Theorems 4 & 5 |
| Phase 2 — Months 3–8 (Oct 2026–Jan 2027) | New benchmark + precision utility | Non-MNIST domain; Π⋆ > MHN by ≥5%; learned Π prototype |
| Phase 3 — Months 8–18 (Feb–Dec 2027) | Systems paper + open-source library | pcam-torch on GitHub; JMLR submission |

---

# PART II — TECHNICAL REQUIREMENTS DOCUMENT (TRD)

## §7 System Architecture Overview

DAHN is structured as three composable layers that sit on top of the unmodified JKU `hflayers` library. No modifications are made to the upstream library source; all extensions are additive imports.

### 7.1 Layer Structure

| Layer | Components |
|-------|-----------|
| Layer 0 — Upstream (read-only) | JKU hflayers: `Hopfield`, `HopfieldCore`, `HopfieldPooling`, `HopfieldLayer` — zero modifications |
| Layer 1 — Graph Infrastructure | `GraphBuilder` (kNN cosine similarity → adjacency), `LaplacianBuilder` (normalised L), `GraphCache` (O(1) cache by `data_ptr()`), 4 `DiffusionOperator` modes (factored O(kN), simple, iterative, spectral) |
| Layer 2 — Retrieval Dynamics | `DiffusedHopfield` (`_associate` override), `AttentionOperator` (dense + graph-sparse), `DynamicsEngine` (T-step interleaved diffuse→attend loop), `EnergyTracker` (Hopfield energy per step) |
| Layer 3 — Precision Control (new) | `PrecisionOperator` (diagonal / full SPD Π(t)), `PCAMDynamics` (PCAM gradient flow integrator), `PrecisionSchedule` (uniform / random diagonal / per-class structured) |

### 7.2 Forward Pass — Current vs. Target

**Current (Broken) Forward Path:**
```
K' = D·K (once) → Q' = D·Q (once) → HopfieldCore._associate(Q', K', V)
Result: single-step diffusion; DynamicsEngine.run_dynamics() is dead code; AttentionOperator graph-mode is unreachable.
```

**Target (Correct) Forward Path:**
```
For t = 0 … T−1:
  Kₜ = D · Kₜ    [DiffusionOperator, chosen mode]
  Qₜ = D · Qₜ    [DiffusionOperator]
  Aₜ = AttentionOperator(Qₜ, Kₜ, V)    [dense or graph-sparse]
  EnergyTracker.record(Qₜ, Kₜ, Aₜ)
Return Aₜ at convergence or t = T.
```

---

## §8 Mathematical Specification

### 8.1 Graph Laplacian Diffusion

The diffusion operator is derived from a kNN cosine-similarity graph over stored patterns $X \in \mathbb{R}^{d \times n}$:

$$W_{ij} = \cos(x_i, x_j) \quad \text{if } j \in \text{kNN}(i), \text{ else } 0 \quad [\text{symmetric, non-negative}]$$

$$L = D_{\text{deg}} - W \quad [\text{unnormalised graph Laplacian}, D_{\text{deg}} = \text{diag}(W \cdot \mathbf{1})]$$

$$L_{\text{norm}} = I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}} \quad [\text{normalised Laplacian, eigenvalues} \in [0,2]]$$

$$D = I - \eta \cdot L \quad [\text{diffusion operator}, \eta \in (0, 1/\lambda_{\max}(L)) \text{ for stability}]$$

The graph-regularised retrieval step at iteration t becomes:

$$Q_{t+1} = \text{softmax}(\beta \cdot (D \cdot Q_t)(D \cdot K_t)^\top) \cdot V$$

This introduces a semantic manifold constraint: patterns that are geometrically close in the kNN graph pull each other's attention weights toward a shared consensus, smoothing the energy landscape in the retrieval basin.

### 8.2 Hopfield Energy with Graph Regularisation

The tracked Hopfield energy at step t is:

$$E_t = -\frac{\beta}{N} \cdot \text{tr}(Q_t \cdot K_t^\top) + \frac{\eta}{N} \cdot \text{tr}(K_t^\top \cdot L \cdot K_t)$$

The first term is the standard MHN log-sum-exp energy (maximised at convergence); the second term is the graph Dirichlet energy (minimised when K is smooth over the graph). `EnergyTracker` records both components per step.

### 8.3 PCAM Precision-Modulated Dynamics (Layer 3)

The precision-modulated gradient flow (NeurIPS 2026, Eq. 4) on the log-sum-exp energy landscape:

$$\dot{a}(t) = -\Pi(t) \nabla E(a(t)) + J \cdot u(t)$$

$$E(a) = -\frac{1}{\beta} \log \sum_i \exp(\beta \cdot x_i^\top \cdot a) + \frac{\lambda}{2} \|a\|^2$$

$$\nabla E(a) = -X \cdot \sigma(\beta X^\top a) + \lambda a \quad [\sigma = \text{softmax}]$$

Discrete Euler integration at step Δt:

$$a_{t+1} = a_t + \Delta t \cdot (-\Pi_t \cdot \nabla E(a_t) + J \cdot u_t)$$

Precision operator Π(t) is symmetric positive-definite and admissible: $\pi_{\min} \cdot I \preceq \Pi(t) \preceq \pi_{\max} \cdot I$. The Implicit Function Theorem guarantees that each isolated stable equilibrium $a^*$ shifts Lipschitz-continuously under admissible perturbations of Π, with explicit constant $C = \|Ju_0\| / (\pi_{\min}^2 \cdot \mu)$ where $\mu = \lambda_{\min}(\nabla^2 E(a^*))$.

> **Critical Design Constraint**  
> The PCAM precision operator must **NOT** be formulated as a global Banach contraction (requiring C₃ < 1 globally). A global contraction has a unique fixed point, destroying multi-attractor associative memory. The correct formulation is the local Implicit Function Theorem applied at each isolated stable equilibrium — as proven in Theorem 5 of the NeurIPS 2026 paper and implemented in Appendix F.

---

## §9 Module Specifications

### 9.1 GraphBuilder

| Field | Detail |
|-------|--------|
| **Input** | `X: Tensor (N, d)` — stored patterns (keys) |
| **Output** | `W: Tensor (N, N)` or SparseCOO — symmetric kNN adjacency matrix |
| **Parameters** | `k: int` — number of nearest neighbours; `similarity: {cosine, dot, euclidean}`; `symmetric: bool` (default True) |
| **Complexity** | O(N²·d) dense; O(k·N·d) with FAISS approximate kNN |
| **Cache** | `GraphCache` caches by `X.data_ptr()`; invalidated on pattern update |
| **Requirement** | Must produce W_ij ≥ 0 and W = Wᵀ for valid Laplacian |

### 9.2 DiffusionOperator — 4 Modes

| Mode | Description |
|------|-------------|
| `factored` **[DEFAULT]** | `D·K = K − η·L·K` computed as `K − η·(D_deg·K − W·K)`. O(k·N·d) with sparse W. Does **NOT** build dense L — incompatible with `EnergyTracker`'s Laplacian term. **FIX:** compute L·K without materialising L. |
| `simple` | `D = I − η·L` built densely; `D·K = D @ K`. O(N²) memory. Step-invariant (same D applied each step). |
| `iterative` | `K_{s+1} = K_s − η·L·K_s` repeated S times. Over-smooths at S ≥ 5; guard required (check signal energy). |
| `spectral` | `D = V·diag(1 − η·Λ)·Vᵀ` via eigendecomposition of L. Step-invariant. O(N³) precompute, O(N·d) per application. |

### 9.3 DynamicsEngine — PRIMARY DELIVERABLE (currently dead code)

| Field | Detail |
|-------|--------|
| **Responsibility** | Run the T-step interleaved diffuse→attend loop and wire it into `DiffusedHopfield._associate()` |
| **Signature** | `run_dynamics(Q, K, V, T, beta, eta, attn_mode) → Q_T (converged query)` |
| **Convergence** | Stop at first t with `‖Q_{t+1} − Q_t‖ < ε` or `t = T`. ε default 1e-6. |
| **Energy tracking** | At each step, call `EnergyTracker.record(Q_t, K_t)`. Requires L to be available — use `simple` or `spectral` mode, not `factored`. |
| **Threading** | Batched: all B queries in a batch processed in parallel via batch matmul. Must not loop over queries serially. |
| **Wiring fix** | `DiffusedHopfield._associate()` must call `self.dynamics_engine.run_dynamics(...)` and return the result, replacing the current single-step shortcut. |

### 9.4 AttentionOperator — GRAPH MODE UNREACHABLE (fix required)

| Mode | Description |
|------|-------------|
| **Dense mode** | Standard scaled dot-product: `softmax(β·Q·Kᵀ)·V`. O(N²·d). Used in all current experiments. |
| **Graph mode (target)** | Compute attention only within k-hop neighbourhood of each query token. O(k·N·d). Requires sparse matmul or FAISS lookup. |
| **Wiring fix** | `DynamicsEngine` must expose `attn_mode: {dense, graph}` parameter and route to `AttentionOperator.attend_dense()` or `AttentionOperator.attend_graph()` accordingly. |
| **Accuracy requirement** | Graph mode must achieve ≥ 95% of dense mode retrieval accuracy at k ≥ 10 for n ≤ 1000 patterns. |

### 9.5 PrecisionOperator — NEW (Layer 3)

| Field | Detail |
|-------|--------|
| **Class** | `PrecisionOperator(d, mode: {scalar, diagonal, full_spd})` |
| **Scalar mode** | Π = π·I. Single float parameter π > 0. Cheapest: removes one matrix multiply per step. |
| **Diagonal mode** | Π = diag(π₁,...,πd). Element-wise multiply. Consistent with Free Energy Principle (feature-wise gain). |
| **Full SPD mode** | Π = SPD matrix. Theorem 5 holds verbatim. Enable via Cholesky parameterisation Π = LLᵀ + ε·I. |
| **Schedule** | `PrecisionSchedule(mode: {uniform, random_diag, per_class})` — three schedules as in NeurIPS paper Appendix H.3 |
| **Admissibility** | Must enforce `π_min·I ≼ Π ≼ π_max·I` at all times. Clamp eigenvalues after each update. |

---

## §10 Data, Protocols & Benchmarking

### 10.1 Datasets

| Dataset | Protocol |
|---------|----------|
| MNIST (LeCun et al., 1998) — CC BY-SA 3.0 | 28×28 → flatten 784 → PCA to d=64. Corruption applied in raw 784-d pixel space before projection. |
| Synthetic Gaussian (Exp 1–3) | X ∈ ℝ⁶⁴ˣ³², rows i.i.d. N(0,1), columns ℓ²-normalised. K=10 random seeds. |
| Phase 1 scale-up | Full MNIST 60k train, n=5000 patterns (500/class), d=128 PCA. |
| Phase 2 domain | TBD — requirement: heterogeneous feature importance across classes/contexts for precision advantage. |

### 10.2 Corruption Protocols

| Corruption | Specification |
|-----------|---------------|
| Mask noise | Each pixel zeroed independently with probability p ∈ {0.10, 0.20, 0.30, 0.40} |
| Salt-and-pepper | Each pixel set to {0,1} uniformly at random with probability p ∈ {0.10, 0.20, 0.30, 0.40} |
| Gaussian additive | i.i.d. N(0, σ²) added, clipped to [0,1]. σ ∈ {0.10, 0.20, 0.30, 0.40} |
| **Application order** | **CRITICAL:** corruption applied in raw 784-d pixel space, then projected via PCA basis to ℝ⁶⁴ |

### 10.3 Integration Protocol

| Parameter | Value |
|-----------|-------|
| Integrator | Explicit Euler, Δt = 0.01, cap T = 3000 steps |
| Convergence tolerance | `‖a_{t+1} − a_t‖ < 1e-6`; terminate at first crossing |
| Query initialisation | MNIST: `a(0) = corrupted PCA query q`; Synthetic: `a(0) ~ N(0, 0.1·I)` |
| Input protocol | `J = I_64`, `u(t) = q·1{t ≤ T_in}`, T_in = 100 steps; precision Π = I (default) or Π⋆ |
| Seeds | K = 10 random seeds per experiment; report mean ± SD |

### 10.4 Accuracy Metric

**Retrieval Accuracy Definition:** Nearest-neighbour classification in d-dimensional PCA space: the retrieved equilibrium a* is mapped to the stored pattern with smallest Euclidean distance. Accuracy = fraction of queries for which the predicted pattern's class matches the query's ground-truth class. This is identical to the metric in NeurIPS 2026 §5.3.

### 10.5 Baselines

| Baseline | Protocol |
|----------|----------|
| Classical Hopfield (Hopfield 1982) | Hebbian `W = (1/n)·Σ sgn(x_k)sgn(x_k)ᵀ` on bipolarised PCA features. 100 synchronous update steps. Fixed point projected back via 10 nearest stored continuous PCA patterns. |
| Modern Hopfield (Ramsauer 2021) | One-shot: `a* = X·σ(β_MHN·Xᵀ·q)`. β_MHN tuned from {4,8,16} on held-out validation split per noise level. |
| DAHN-Π=I | Proposed dynamics (4) with Π = I (uniform precision). Comparable to 'no inference-time control' variant. |
| DAHN-Π⋆ | Proposed dynamics (4) with Π⋆ = I + 0.2·diag(m), m = top-32 PCA component indicator. Per-class structured precision. |

---

## §11 Infrastructure, Profiling & Scaling Requirements

### 11.1 Memory Bandwidth Analysis

DAHN's core bottleneck is memory-bandwidth, not compute. The pattern matrix X is read at every integration step. At T = 3000 steps and n = 1000, d = 512, X is accessed approximately 3 million times per query.

| Scale Point | Requirement |
|-------------|-------------|
| n=200, d=64 (current baseline) | Must profile with `torch.profiler`: record GPU memory (MB), memory bandwidth (GB/s), compute utilisation (%), ms/step |
| n=5000, d=128 (Phase 1 target) | Must fit X in RTX 3090 HBM (24GB). At float32: 5000×128×4B = 2.56MB — fits easily. |
| n=100k, d=512 (Phase 3 horizon) | X = 100k×512×4B = 204MB. Competes with activations. Quantise to float16 (102MB) as first mitigation. |
| Scaling ceiling identification | Identify n at which X causes HBM eviction to DRAM. This is the hardware ceiling for RTX 3090. |

### 11.2 Algorithmic Scaling Mitigations

| Mitigation | Target Phase |
|-----------|-------------|
| Batched inference: B = 32/64/128 queries in parallel (batch matmul). Near-zero code change, expected ~4–8× throughput. | Phase 0 |
| FAISS approximate softmax: IVF-Flat index replaces O(n) full softmax with O(log n) ANN lookup. Reduces memory pressure at large n. | Phase 1 |
| Pattern matrix quantisation: X in float16 or int8. Theorem 5's explicit C gives tolerance budget for quantisation error. | Phase 2/3 |
| Fused CUDA kernel: fuse Xᵀa (batch matmul) + softmax + X·σ + precision scale into 3 operations. Write or adapt FlashAttention-style kernel. | Phase 3 |

### 11.3 Profiling Report Schema (Phase 0 Output)

| n | Memory (MB) | Bandwidth / ms/step / QPS |
|---|-------------|--------------------------|
| 200 | TBD (baseline) | TBD |
| 500 | TBD | TBD |
| 1,000 | TBD | TBD |
| 2,000 | TBD | TBD |
| 5,000 | TBD — Phase 1 target | TBD |

---

## §12 Known Defects & Required Fixes

The following defects are identified in the `SPEC_ALIGNMENT_AUDIT.md` and must be resolved as primary deliverables:

| Defect | Required Fix |
|--------|-------------|
| **BUG-01:** `DynamicsEngine.run_dynamics()` never called from `DiffusedHopfield._associate()`. Dead code. | Wire `run_dynamics()` as the primary forward path. All reported results currently come from single-step diffusion feeding into original `HopfieldCore` — which is a valid design but not the specified architecture. |
| **BUG-02:** `AttentionOperator` graph-mode O(kN) backend constructed but never invoked. | `DynamicsEngine` must expose `attn_mode` parameter and route to `AttentionOperator.attend_graph()`. Benchmark graph vs. dense at n ∈ {200, 1000, 5000}. |
| **BUG-03:** `EnergyTracker` incompatible with `factored` diffusion mode. `factored` never builds L; energy computation requires L. | Two options: (A) switch default mode to `simple` for experiments requiring energy tracking, or (B) compute L·K in `factored` mode without materialising dense L using sparse matmul. **Option B is preferred for scaling.** |
| **BUG-04:** No unit test suite. | Add pytest suite covering: `GraphBuilder` output validity (W ≥ 0, W = Wᵀ); `DiffusionOperator` stability (‖D‖ ≤ 1); `DynamicsEngine` energy dissipation on synthetic data; `AttentionOperator` dense vs. graph agreement at small n. |
| **BUG-05:** Over-smoothing in `iterative` mode not guarded (accuracy degrades at steps ≥ 5). | Add early-stop guard: if `‖K_{s+1} − K_s‖ < δ` or signal energy drops below threshold, stop iterating regardless of S. Log a warning. |

---

## §13 Theorem-to-Test Validation Mapping

Each theorem from the NeurIPS 2026 paper must be empirically validated:

| Theorem | Empirical Validation Test |
|---------|--------------------------|
| **Theorem 1** (Well-posedness) | At any (n, d, Π, u) in benchmark range: ODE integrator never diverges or produces NaN/Inf. Track: zero NaN events across 10k random seeds. Passes if Δt ≤ Δt_max (derived from Theorem 2). |
| **Theorem 2** (Energy dissipation + UUB) | Plot E(a(t)) for all runs. Verify: (a) E non-increasing except bounded excursions ≤ J²_M U²_M / (4π_min), (b) all trajectories enter ball B(R∞) within bounded T. At n=5k measure empirical R∞ vs. theoretical formula. |
| **Theorem 3** (Asymptotic invariance) | Run Π(t) = I + 0.5·exp(−t/100)·diag(s) → I. Trajectory must converge to same equilibrium as constant Π∞ = I run from same initialisation. Tolerance: `‖a*_varying − a*_constant‖ < 0.01`. |
| **Theorem 4** (Multi-well structure) | At β ≥ β₀: for each x_i, verify ≥ 1 trajectory from B(x_i/λ, r) converges to critical point within ε(β) of x_i/λ. Coverage ≥ 95% at β=16, n ≤ 32. Extend to n = 5000. |
| **Theorem 5** (Lipschitz precision modulation) | Measure empirical C̃ = max ‖a*(Π₁) − a*(Π₂)‖ / ‖Π₁ − Π₂‖ across 1000 random Π pairs. Compare to theoretical C = ‖Ju₀‖/(π²_min · μ). Requirement: C̃ ≤ C. If C̃ > C, investigate proof error or numerical artefact. |
| **Capacity bound** (future) | Find empirical max n before retrieval accuracy drops below 50% under 20% mask noise at fixed d and λ. Compare to theoretical O(exp(d/2)) once derived. |

---

## §14 Dependency Stack & Environment

| Dependency | Version / Requirement |
|-----------|----------------------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 (for `torch.compile` and FlashAttention-2 compatibility) |
| hflayers (JKU) | Upstream library — zero modifications. Append-only imports in `__init__.py`. |
| FAISS | `faiss-gpu` ≥ 1.7.4. IVF-Flat index for Phase 1 approximate kNN. |
| NumPy / SciPy | Standard. SciPy for eigendecomposition in spectral diffusion mode. |
| scikit-learn | PCA, kNN graph construction (cross-validation with `GraphBuilder` output). |
| torch.profiler | Phase 0 profiling. Export Chrome trace format for memory bandwidth analysis. |
| pytest | Unit test suite (BUG-04 fix). Coverage requirement ≥ 80% of custom code. |
| Hardware (current) | Single NVIDIA RTX 3090 24GB, Intel i9-12900K, 64GB DDR5 RAM. |
| Hardware (Phase 3 target) | Multi-GPU or CXL memory server for n ≥ 100k scaling. |

---

## §15 Open Research Questions & Future Extensions

### 15.1 Theoretical Extensions (from NeurIPS Limitations §8)

| Question | Current Gap |
|----------|------------|
| Capacity bound | No explicit n = f(d, β, λ, Δ) derived. Analogous to Ramsauer 2021 Theorem 3–4 for the coercive variant. Publishable as standalone result. |
| Non-diagonal Π | Theorem 5 holds verbatim for full SPD Π but not empirically verified. Close Limitation (i) from §8. |
| Learned Π(context) | Precision as a function of context rather than fixed schedule. Close Limitation (ii) from §8. Next paper. |
| Large-scale retrieval | Direct image-space (d > 64) or convolutional feature retrieval. Close Limitation (iii) from §8. |
| Langevin variant | Add β^{−½} dWₜ noise term to dynamics. Compare basin coverage: deterministic flow vs. stochastic sampling. |

### 15.2 Hardware Collaboration Readiness

The mathematical framework is ready for hardware collaboration. The pitch to potential partners (Logical Intelligence, Extropic, neuromorphic startups):

- **For ML Researchers:** DAHN provides inference-time precision control of attractor locations via a local IFT argument — avoiding the global contraction contradiction that would destroy multi-attractor structure.
- **For Systems Engineers:** the core bottleneck per step is Xᵀa (d×n matmul) + softmax + X·σ + Π·gradient. Predictable access pattern (X read every step) is ideal for prefetching and CXL memory pooling.
- **For Hardware Partners:** the ODE ȧ = −Π(t)∇E(a) maps directly to programmable gain on physical noise (thermodynamic) or conductance (memristor) layers. We provide convergence guarantees; hardware provides throughput.

---

## §16 Immediate Action Items — Next 7 Days

| Milestone / Deliverable | Owner | Status |
|------------------------|-------|--------|
| **Day 1:** Install `torch.profiler`; run MNIST baseline; export trace; record top-3 memory consumers in `baseline_profile.txt` | Engineer | Not Started |
| **Day 2:** Compute arithmetic intensity (FLOPs/byte) at d=64 n=200 and d=512 n=10k; compare to RTX 3090 balance point (~25 FLOP/byte at fp32) | Engineer | Not Started |
| **Day 3:** Clone ENSO/Kona repo; run Sudoku demo; profile Langevin loop; produce 3-bullet comparison with DAHN memory access pattern | Engineer | Not Started |
| **Day 4–5:** Wire `DynamicsEngine.run_dynamics()` into `DiffusedHopfield._associate()` — primary BUG-01 fix; verify existing tests still pass | Engineer | Not Started |
| **Day 5:** Implement batched inference (B=32/64/128); measure GPU utilisation saturation point; record throughput improvement | Engineer | Not Started |
| **Day 6:** Read arXiv:2605.07223 (Hardware-aware Hopfield); note capacity scaling K~0.3N^1.2; write TODO for analogous DAHN capacity bound | Research Lead | Not Started |
| **Day 7:** Write 'where I am' half-page document: current n ceiling, memory bottleneck, ENSO comparison, batching speedup. This is Introduction of systems paper. | All | Not Started |

---

## §17 Risk Register

| Risk | Mitigation | Likelihood |
|------|-----------|------------|
| Diffusion gain ≤ 1% over MHN in Phase 2 benchmark | Re-examine whether benchmark task has heterogeneous feature importance. Precision modulation is correct but practically neutral on tasks without feature-selective retrieval. | Medium |
| `EnergyTracker` + `factored` mode incompatibility blocks energy validation | Switch to `simple` mode for energy-tracked experiments; `factored` mode for scaling experiments. Document clearly in benchmark protocol. | Low |
| FAISS approximate kNN degrades retrieval below 95% of exact at n=5000 | Tune `nprobe` parameter. If insufficient, use product quantization (PQ). Increase pattern separation Δ at storage time. | Medium |
| Learned Π(context) overfits on small benchmark | Constrain Π to diagonal; use validation set for early stopping; report on held-out test set only. | Medium |
| Hardware collaboration requires custom CUDA; exceeds team capacity | Use PyTorch custom ops or Triton for fused kernel. FlashAttention-2 source is open — adapt the fused matmul pattern. | Low |

---

*DAHN PRD + TRD · Version 1.0 · June 2026 · Confidential Working Document*  
*Companion to NeurIPS 2026 Submission (Precision-Modulated EBM) and IEEE ICACI 2026 (Ghosh, Jaiswal, Gupta)*
