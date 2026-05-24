# difflayers — Diffusion-Augmented Hopfield Networks

<p align="center">
  <a href="https://pypi.org/project/difflayers/"><img src="https://img.shields.io/pypi/v/difflayers?color=blue&label=PyPI&cacheSeconds=0" alt="PyPI"></a>
  <a href="https://pypi.org/project/difflayers/"><img src="https://img.shields.io/pypi/dm/difflayers?color=blue&label=downloads&cacheSeconds=0" alt="Downloads"></a>
  <a href="https://pypi.org/project/difflayers/"><img src="https://img.shields.io/pypi/pyversions/difflayers?cacheSeconds=0" alt="Python Versions"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A51.9-orange" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD-green" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.1.2-brightgreen" alt="Version 0.1.2">
</p>

**difflayers** is a PyTorch library that extends modern continuous Hopfield networks with graph-based Laplacian diffusion. At its core is the **Diffusion-Augmented Hopfield Network (DAHN)** — a drop-in upgrade to standard Hopfield attention that pre-smooths stored patterns over a learned kNN graph before each association step, suppressing spurious retrievals and sharpening energy minima.

The library ships the complete Hopfield layer suite (`Hopfield`, `HopfieldPooling`, `HopfieldLayer`) together with the DAHN extensions (`DiffusedHopfield`, four diffusion operators, a graph-construction pipeline, and a dynamical-memory engine) under a single, consistent API.

---

## Table of Contents

1. [Background](#background)
2. [What DAHN Adds](#what-dahn-adds)
3. [Architecture Overview](#architecture-overview)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Modules](#core-modules)
   - [Hopfield](#hopfield)
   - [HopfieldPooling](#hopfieldpooling)
   - [HopfieldLayer](#hopfieldlayer)
   - [DiffusedHopfield](#diffusedhopfield)
7. [Diffusion Modes](#diffusion-modes)
8. [DiffusionConfig Reference](#diffusionconfig-reference)
9. [Graph Pipeline](#graph-pipeline)
10. [Advanced Usage](#advanced-usage)
11. [Transformer Integration](#transformer-integration)
12. [Example Notebooks](#example-notebooks)
13. [Running Experiments](#running-experiments)
14. [API Reference](#api-reference)
15. [Complexity Guide](#complexity-guide)
16. [Background Paper](#background-paper)
17. [Releases](#releases)
18. [Disclaimer](#disclaimer)
19. [License](#license)

---

## Background

Modern Hopfield networks with continuous states were introduced in [Ramsauer et al. (2020)](https://arxiv.org/abs/2008.02217), where it was shown that the transformer **attention mechanism is exactly the update rule of a continuous Hopfield network**. This re-framing unlocks exponential storage capacity, single-step convergence, and a clean energy-based interpretation of deep attention.

The energy function of a continuous Hopfield network is:

$$E = -\text{lse}(\beta, X \xi) + \frac{1}{2}\xi^T \xi + \frac{1}{\beta}\log N + \frac{1}{2}M^2$$

where $\text{lse}(\beta, z) = \frac{1}{\beta}\log\sum_i e^{\beta z_i}$ is the log-sum-exp, $\xi$ is the state pattern (query), $X$ are the stored patterns (keys), $\beta$ is the inverse temperature, and $N$, $M$ are dimensional constants.

Energy minimisation via one synchronous update yields the familiar softmax attention:

$$\xi^{\text{new}} = X^\top \text{softmax}(\beta X \xi)$$

The network can store **exponentially many patterns** (in the dimension $d$), converges in **one update step**, and has exponentially small retrieval errors — properties not shared by classical binary Hopfield networks.

Three classes of fixed points arise naturally:

| Fixed-point type | Regime | Behaviour |
|---|---|---|
| **Global averaging** | Low $\beta$ | Weighted average over all stored patterns |
| **Metastable states** | Medium $\beta$ | Subset retrieval — analogous to multi-head attention |
| **Single-pattern storage** | High $\beta$ | Sharp retrieval of one stored pattern |

---

## What DAHN Adds

Standard Hopfield attention treats every stored pattern as equally reachable from any query. In high-noise or high-density memory regimes, the attention distribution spreads over spurious neighbours, degrading retrieval accuracy.

**DAHN** addresses this by building a $k$-nearest-neighbour graph over the pattern set and pre-smoothing patterns with the graph Laplacian before every association step:

$$\text{for } t = 1, \ldots, T:$$
$$K' = \underbrace{(I - \eta L)}_{\text{diffusion}} K, \quad Q' = (I - \eta L) Q \quad \text{(optional)}$$
$$\text{output} = \text{softmax}(\beta \, Q' {K'}^\top) \, V$$

where $L$ is the (optionally symmetric-normalised) graph Laplacian of the kNN similarity graph over $K$, and $\eta$ is the diffusion strength. This smoothing:

- **Clusters** related patterns before retrieval, reducing inter-cluster interference
- **Sharpens** metastable energy minima, improving single-pattern retrieval accuracy under noise
- **Preserves** the Hopfield energy landscape — diffusion only decreases energy, never introduces new spurious minima
- **Scales** gracefully: with `FactoredDiffusion` and sparse adjacency the full loop costs $O(kNd)$ per step

---

## Architecture Overview

```
difflayers/
│
├── __init__.py              # Public API — 18 exported names
│
├── activation.py            # HopfieldCore  (multi-head Hopfield attention kernel)
├── functional.py            # hopfield_core_forward  (low-level functional API)
├── transformer.py           # HopfieldEncoderLayer, HopfieldDecoderLayer
│
├── diffused_attention.py    # DiffusedHopfield  ← DAHN entry point
├── diffusion.py             # DiffusionOperator ABC + 4 concrete strategies
│                            #   SimpleDiffusion, IterativeDiffusion,
│                            #   SpectralDiffusion, FactoredDiffusion
├── dynamics_engine.py       # DiffusionConfig, GraphCache, DynamicsEngine,
│                            #   EnergyTracker
├── attention_operator.py    # AttentionOperator (dense / graph-constrained)
│
├── graph/
│   ├── build_graph.py       # build_similarity_matrix, build_knn_graph
│   ├── laplacian.py         # compute_laplacian, compute_normalized_laplacian
│   ├── builder.py           # GraphBuilder  (fluent graph-construction API)
│   └── laplacian_builder.py # LaplacianBuilder
│
└── auxiliary/
    └── data.py              # LookupTableDataset
```

---

## Installation

### From PyPI

```bash
pip install difflayers
pip install "difflayers==0.1.2"  # pin to a specific version
```

### From source

```bash
git clone https://github.com/Prigoistic/difflayers.git
cd difflayers
pip install -e .
```

### Dependencies

| Package | Minimum version |
|---|---|
| Python | 3.8 |
| PyTorch | 1.9.0 |
| NumPy | 1.20.0 |
| SciPy | 1.7.0 |

For the example notebooks:

```bash
pip install -r examples/requirements.txt
```

---

## Quick Start

```python
import torch
from difflayers import Hopfield, HopfieldPooling, HopfieldLayer, DiffusedHopfield

# Standard Hopfield attention
hopfield    = Hopfield(input_size=64, num_heads=4, batch_first=True)
stored      = torch.randn(8, 50, 64)
queries     = torch.randn(8, 10, 64)
projections = torch.randn(8, 50, 64)
output = hopfield((stored, queries, projections))  # (8, 10, 64)

# Hopfield pooling — fixed-size embedding from a variable-length sequence
pooling = HopfieldPooling(input_size=64, num_heads=1, batch_first=True)
pooled  = pooling(torch.randn(8, 100, 64))  # (8, 1, 64)

# Hopfield lookup — trainable static memory slots
lookup = HopfieldLayer(input_size=64, num_pattern_repetitions=32)
result = lookup(torch.randn(8, 10, 64))  # (8, 10, 64)

# DiffusedHopfield — graph-diffusion augmented retrieval
dh = DiffusedHopfield(
    input_size=64,
    num_heads=4,
    batch_first=True,
    eta=0.1,
    k_neighbors=8,
    diffusion_mode="factored",
    diffusion_steps=3,
    diffuse_key=True,
)
output = dh((stored, queries, projections))  # (8, 10, 64)
```

---

## Core Modules

### Hopfield

A multi-head attention layer whose update rule and scaling are derived from the Hopfield energy function, rather than from learned linear projections.

```python
from difflayers import Hopfield

hopfield = Hopfield(
    input_size=128,           # feature depth of state (query) patterns
    hidden_size=64,           # Hopfield association space depth
    output_size=128,          # output projection depth
    num_heads=8,
    scaling=None,             # inverse temperature beta; None => 1/sqrt(d_head)
    update_steps_max=0,       # 0 = one synchronous update (recommended)
    update_steps_eps=1e-4,
    normalize_stored_pattern=True,
    normalize_state_pattern=True,
    batch_first=True,
    dropout=0.1,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_size` | `int` | `None` | Feature depth of state (query) patterns |
| `hidden_size` | `int` | `None` | Hopfield association space depth; defaults to `input_size` |
| `output_size` | `int` | `None` | Output projection depth; defaults to `input_size` |
| `num_heads` | `int` | `1` | Parallel association heads |
| `scaling` | `float` | `None` | Inverse temperature beta; `None` => 1/sqrt(d_head) |
| `update_steps_max` | `int` | `0` | Max synchronous update iterations (`None` = run to convergence) |
| `batch_first` | `bool` | `True` | Input layout `(batch, seq, d)` when `True` |
| `stored_pattern_as_static` | `bool` | `False` | Freeze stored patterns (no gradient through keys) |
| `disable_out_projection` | `bool` | `False` | Skip the final linear projection |

---

### HopfieldPooling

Replaces traditional pooling with a Hopfield-energy-based alternative. A single trainable state pattern acts as the query, computing softmax weights over the input sequence to produce a fixed-size summary regardless of input length.

```python
from difflayers import HopfieldPooling

pooling  = HopfieldPooling(input_size=128, num_heads=4, batch_first=True)
sequence = torch.randn(batch, seq_len, 128)
pooled   = pooling(sequence)  # (batch, 1, 128)
```

Suitable for permutation-invariant sequence summarisation: MIL classification, set encoding, immune repertoire profiling.

---

### HopfieldLayer

A trainable, input-independent lookup table. Stored patterns and their projections are learned parameters; the layer retrieves the energy-aligned stored vector for each input query — functioning as a content-addressable memory with learned slots.

```python
from difflayers import HopfieldLayer

lookup = HopfieldLayer(input_size=128, num_pattern_repetitions=64, batch_first=True)
result = lookup(torch.randn(batch, seq_len, 128))  # (batch, seq_len, 128)
```

Distinct from `Hopfield` in that the memory contents are **learned parameters**, not runtime inputs. Suitable for slot-attention and prototype networks.

---

### DiffusedHopfield

The DAHN module — a drop-in replacement for `Hopfield` that augments association with a graph-diffusion pre-processing step. Internally it builds a kNN cosine-similarity graph over stored patterns, constructs the graph Laplacian, and runs a configurable diffuse→attend loop.

```python
from difflayers import DiffusedHopfield

dh = DiffusedHopfield(
    input_size=128,
    num_heads=4,
    batch_first=True,
    scaling=1.0,
    eta=0.1,
    k_neighbors=8,
    diffusion_mode="factored",     # "factored" | "simple" | "iterative" | "spectral"
    diffusion_steps=3,
    use_normalized_laplacian=True,
    diffuse_key=True,
    diffuse_query=False,
    use_sparse=False,
    use_logit_diffusion=False,
    logit_eta=None,
    adaptive_eta=False,
    cache_graph=True,
    energy_stop_tol=0.0,
)
```

Forward signature is identical to `Hopfield`:

```python
output = dh((stored_patterns, state_patterns, pattern_projections))
output = dh((stored_patterns, state_patterns, pattern_projections),
            stored_pattern_padding_mask=mask)
```

---

## Diffusion Modes

Four diffusion strategies are available, trading off speed, memory, and smoothing quality:

### `"factored"` *(default)*

```
x' = (1 - eta * deg) * x  +  eta * W @ x
```

Never materialises the full Laplacian. Stores only the sparse adjacency `W` and degree vector `deg`. Each step costs `O(kNd)` in time and `O(kN)` in memory. Recommended for large N.

### `"simple"`

```
x' = (I - eta * L) @ x
```

One explicit Euler step of heat diffusion. Forms `D = I - eta*L` once and applies it. Cost: `O(N^2 d)` per step.

### `"iterative"`

```
x' = (I - eta * L)^T @ x
```

Applies operator `D` repeatedly for `T` steps (`diffusion_steps`). Provides deeper smoothing at `T * O(N^2 d)` per call. Includes a numerical guard against divergence.

### `"spectral"`

```
x' = U @ diag(exp(-eta * lambda)) @ U.T @ x
```

Exact heat-kernel diffusion via eigendecomposition of `L`. Precomputes `U` and `lambda` once at `O(N^3)`, then applies the diagonal filter in `O(N^2)` per call. Most accurate; not suitable for large N.

| Mode | Precompute | Per-step | Memory | Best for |
|---|---|---|---|---|
| `factored` | O(N^2) build kNN | O(kNd) | O(kN) | Large N, production |
| `simple` | O(N^2) build D | O(N^2 d) | O(N^2) | Moderate N, one-shot |
| `iterative` | O(N^2) build D | O(T N^2 d) | O(N^2) | Deep smoothing |
| `spectral` | O(N^3) eigen | O(N^2) | O(N^2) | Small N, exact kernel |

---

## DiffusionConfig Reference

`DiffusionConfig` is a frozen dataclass that bundles all diffusion hyperparameters. Pass one explicitly to `DiffusedHopfield`, or let the constructor build it from keyword arguments.

```python
from difflayers import DiffusionConfig

cfg = DiffusionConfig(
    eta=0.1,
    beta=1.0,
    steps=3,
    diffusion_mode="factored",
    attention_mode="dense",
    k_neighbors=5,
    use_normalized_laplacian=True,
    use_sparse=False,
    diffuse_key=True,
    diffuse_query=False,
    use_logit_diffusion=False,
    logit_eta=None,
    adaptive_eta=False,
    adaptive_temperature=5.0,
    adaptive_threshold=1.0,
    cache_graph=True,
    energy_stop_tol=0.0,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `eta` | `float` | `0.1` | Diffusion strength; use eta < 0.5 with normalised L |
| `beta` | `float` | `1.0` | Hopfield inverse temperature |
| `steps` | `int` | `3` | Number of diffuse→attend iterations |
| `diffusion_mode` | `str` | `"factored"` | `"factored"`, `"simple"`, `"iterative"`, or `"spectral"` |
| `attention_mode` | `str` | `"dense"` | `"dense"` (full O(N^2)) or `"graph"` (kNN-constrained O(kN)) |
| `k_neighbors` | `int` | `5` | kNN graph degree |
| `use_normalized_laplacian` | `bool` | `True` | Symmetric-normalised L; eigenvalues in [0, 2] |
| `use_sparse` | `bool` | `False` | Store adjacency as `sparse_coo` for O(kN) memory |
| `diffuse_key` | `bool` | `True` | Smooth stored patterns (keys) before attention |
| `diffuse_query` | `bool` | `False` | Smooth state patterns (queries) before attention |
| `use_logit_diffusion` | `bool` | `False` | Smooth post-softmax attention weights over the key graph |
| `logit_eta` | `float\|None` | `None` | Separate eta for logit diffusion; falls back to `eta` |
| `adaptive_eta` | `bool` | `False` | Scale eta by attention entropy at runtime |
| `cache_graph` | `bool` | `True` | Reuse the built graph across forward passes |
| `energy_stop_tol` | `float` | `0.0` | Early-stop when `abs(ΔE) < tol`; 0 disables |

---

## Graph Pipeline

The graph subpackage can be used standalone to build Laplacians for any downstream task:

```python
import torch
from difflayers.graph.build_graph import build_similarity_matrix, build_knn_graph
from difflayers.graph.laplacian import compute_laplacian, compute_normalized_laplacian
from difflayers.graph.builder import GraphBuilder

# Manual pipeline
X = torch.randn(100, 64)
S = build_similarity_matrix(X)             # (100, 100) cosine similarity
A = build_knn_graph(S, k=8, as_sparse=False)
L = compute_normalized_laplacian(A)        # symmetric-normalised Laplacian

# Fluent builder API
graph = (
    GraphBuilder(X)
    .cosine_similarity()
    .knn(k=8, sparse=True)
    .normalized_laplacian()
    .build()
)
# graph.L — Laplacian, graph.W — adjacency, graph.deg — degree vector
```

**`build_similarity_matrix(X)`** — Pairwise cosine similarity, negatives clamped to zero, diagonal zeroed. O(N^2 d).

**`build_knn_graph(S, k, as_sparse)`** — Keeps top-k neighbours per node, then symmetrises. Returns `torch.sparse_coo_tensor` when `as_sparse=True`.

**`compute_laplacian(A)`** — Unnormalised L = D − A. Eigenvalues in [0, d_max].

**`compute_normalized_laplacian(A)`** — Symmetric normalised L_sym = D^{−1/2}(D − A)D^{−1/2}. Eigenvalues in [0, 2]. Recommended for diffusion because the eigenvalue bound makes a stable eta independent of the input.

---

## Advanced Usage

### Static retrieval

Content-addressable memory without learned projections:

```python
model = DiffusedHopfield(
    input_size=None,
    stored_pattern_as_static=True,
    state_pattern_as_static=True,
    pattern_projection_as_static=True,
    disable_out_projection=True,
    normalize_stored_pattern=False,
    normalize_state_pattern=False,
    normalize_pattern_projection=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection_affine=False,
    batch_first=True,
    scaling=4.0,
    eta=0.15,
    k_neighbors=10,
    diffusion_mode="iterative",
    diffusion_steps=5,
)
```

### Ablation: key-only, query-only, or both

```python
dh_k    = DiffusedHopfield(input_size=64, diffuse_key=True,  diffuse_query=False, eta=0.1)
dh_q    = DiffusedHopfield(input_size=64, diffuse_key=False, diffuse_query=True,  eta=0.1)
dh_both = DiffusedHopfield(input_size=64, diffuse_key=True,  diffuse_query=True,  eta=0.1)
```

### Logit-level diffusion

```python
dh = DiffusedHopfield(
    input_size=64,
    diffuse_key=True,
    use_logit_diffusion=True,
    logit_eta=0.05,
)
```

### Adaptive diffusion strength

```python
dh = DiffusedHopfield(
    input_size=64,
    adaptive_eta=True,
    eta=0.2,
    adaptive_temperature=5.0,
    adaptive_threshold=1.0,
)
```

### DynamicsEngine and EnergyTracker

```python
from difflayers import DiffusionConfig, DynamicsEngine, EnergyTracker, GraphCache
from difflayers.diffusion import FactoredDiffusion
from difflayers.attention_operator import AttentionOperator

cfg          = DiffusionConfig(eta=0.1, steps=5, k_neighbors=8)
cache        = GraphCache(cfg)
graph        = cache.get(patterns)

diffusion_op = FactoredDiffusion(graph.W, graph.deg, cfg.eta)
attn_op      = AttentionOperator(beta=cfg.beta, mode=cfg.attention_mode)

engine  = DynamicsEngine(diffusion_op, attn_op, cfg)
tracker = EnergyTracker(enabled=True)
engine.run(Q, K, V, tracker=tracker)

print(tracker.energies)  # Hopfield energy per step
```

---

## Transformer Integration

`HopfieldEncoderLayer` and `HopfieldDecoderLayer` are drop-in replacements for PyTorch's built-in transformer layers with the attention kernel replaced by the Hopfield update rule:

```python
from difflayers import HopfieldEncoderLayer, HopfieldDecoderLayer
import torch.nn as nn

encoder = nn.TransformerEncoder(
    encoder_layer=HopfieldEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True,
    ),
    num_layers=6,
)

decoder = nn.TransformerDecoder(
    decoder_layer=HopfieldDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True,
    ),
    num_layers=6,
)
```

---

## Example Notebooks

Three fully worked demonstrations are in [examples/](examples/). Install dependencies first:

```bash
pip install -r examples/requirements.txt
```

### [Bit Pattern Set](examples/bit_pattern/bit_pattern_demo.ipynb)

Binary MIL classification. Positive bags contain class-defining bit patterns absent in negative bags. Demonstrates that `Hopfield`, `HopfieldPooling`, and `HopfieldLayer` learn to identify discriminative patterns with high accuracy even as bag size and noise increase.

### [Latch Sequence Set](examples/latch_sequence/latch_sequence_demo.ipynb)

Long-term dependency task. A sequence starts with symbol **A** or **B**; after a variable delay, the model must recall it. The Hopfield layer concentrates attention on the first position without positional encoding.

### [Attention-based Deep MIL (MNIST Bags)](examples/mnist_bags/mnist_bags_demo.ipynb)

Canonical MIL benchmark from [Ilse & Tomczak (2018)](https://arxiv.org/abs/1802.04712). Each bag is a set of MNIST images; positive bags contain a target digit. Benchmarks Hopfield pooling against classic attention-MIL.

---

## Running Experiments

All experiment scripts are in [src/experiments/](src/experiments/) and write results to [results/](results/).

```bash
python -m src.experiments.ablation          # key-only / query-only / both vs. none
python -m src.experiments.benchmark         # speed benchmark across diffusion modes
python -m src.experiments.noise_robustness  # accuracy vs. noise level
python -m src.experiments.steps_sweep       # vary diffusion steps T = 1 ... 10
python -m src.experiments.mode_comparison   # standard vs. diffused Hopfield
python -m src.experiments.logit_vs_feature  # logit-level vs. feature-level diffusion
python -m src.experiments.attention_analysis
```

---

## API Reference

| Name | Type | Description |
|---|---|---|
| `Hopfield` | `nn.Module` | Base continuous Hopfield attention layer |
| `HopfieldPooling` | `nn.Module` | Hopfield pooling with a trainable query |
| `HopfieldLayer` | `nn.Module` | Trainable static-memory lookup layer |
| `HopfieldCore` | `nn.Module` | Low-level multi-head Hopfield kernel |
| `DiffusedHopfield` | `nn.Module` | DAHN: graph-diffusion augmented Hopfield |
| `HopfieldEncoderLayer` | `nn.Module` | Transformer encoder layer with Hopfield attention |
| `HopfieldDecoderLayer` | `nn.Module` | Transformer decoder layer with Hopfield attention |
| `DiffusionOperator` | `ABC` | Abstract base for diffusion strategies |
| `SimpleDiffusion` | `DiffusionOperator` | One-step explicit Euler diffusion |
| `IterativeDiffusion` | `DiffusionOperator` | T-step iterative diffusion |
| `SpectralDiffusion` | `DiffusionOperator` | Exact heat-kernel via eigendecomposition |
| `FactoredDiffusion` | `DiffusionOperator` | Laplacian-free O(kNd) factored form |
| `apply_diffusion` | `function` | Functional API for a single diffusion step |
| `DiffusionConfig` | `dataclass` | Serialisable config for all DAHN hyperparameters |
| `GraphCache` | `class` | Builds and caches the kNN graph and Laplacian |
| `DynamicsEngine` | `class` | Orchestrates the diffuse→attend loop |
| `EnergyTracker` | `class` | Per-step Hopfield energy logging and early-stop |
| `GraphBuilder` | `class` | Fluent graph-construction API |

---

## Complexity Guide

| Operation | Time | Memory | Notes |
|---|---|---|---|
| Build similarity matrix | O(N^2 d) | O(N^2) | `build_similarity_matrix` |
| Build kNN graph (dense) | O(N^2) | O(N^2) | `build_knn_graph` |
| Build kNN graph (sparse) | O(N^2) | O(kN) | `as_sparse=True` |
| Laplacian | O(N^2) | O(N^2) | |
| `FactoredDiffusion` step | O(kNd) | O(kN) | Recommended for large N |
| `SimpleDiffusion` step | O(N^2 d) | O(N^2) | |
| `IterativeDiffusion` T steps | O(T N^2 d) | O(N^2) | |
| `SpectralDiffusion` precompute | O(N^3) | O(N^2) | Eigendecomposition |
| `SpectralDiffusion` apply | O(N^2) | O(N^2) | Per forward pass |
| Dense Hopfield attention | O(N^2 d) | O(N^2) | `attention_mode="dense"` |
| Graph-constrained attention | O(kNd) | O(kN) | `attention_mode="graph"` |
| Full DAHN (factored + dense) | O(T kNd + N^2 d) | O(N^2) | Typical configuration |
| Full DAHN (factored + graph) | O(T kNd) | O(kN) | Fully sparse end-to-end |

N = number of patterns, d = feature dimension, k = kNN degree, T = diffusion steps.

---

## Background Paper

The continuous Hopfield network foundation is described in:

> **Hopfield Networks is All You Need**
> Hubert Ramsauer, Bernhard Schaefl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber,
> Markus Holzleitner, Milena Pavlovic, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp,
> Gunter Klambauer, Johannes Brandstetter, Sepp Hochreiter
> *ICLR 2021* — [arxiv.org/abs/2008.02217](https://arxiv.org/abs/2008.02217)

Companion blog post: [ml-jku.github.io/hopfield-layers](https://ml-jku.github.io/hopfield-layers/).

---

## Releases

### [0.1.2](https://pypi.org/project/difflayers/0.1.2/) — 2026-05-25

- README rewritten to production-level documentation.
- Cleaned code examples: removed divider comments, condensed verbose inline annotations.
- No API changes.

### [0.1.1](https://pypi.org/project/difflayers/0.1.1/) — 2026-05-25

- Fixed PyPI package description: rebuilt distributions after README rewrite.
- Version bump only; no API changes.

### [0.1.0](https://pypi.org/project/difflayers/0.1.0/) — 2026-05-25

- Initial public release of `difflayers`.
- Complete `Hopfield`, `HopfieldPooling`, `HopfieldLayer` suite.
- DAHN (`DiffusedHopfield`) with four diffusion modes: `factored`, `simple`, `iterative`, `spectral`.
- Graph pipeline, `DynamicsEngine`, `EnergyTracker`, `GraphCache`, `DiffusionConfig`.
- `HopfieldEncoderLayer` and `HopfieldDecoderLayer` transformer drop-ins.
- Three example notebooks and seven experiment runners.

---

## Disclaimer

Parts of this implementation are based on [PyTorch v1.6.0](https://github.com/pytorch/pytorch/tree/v1.6.0) and extended for the Hopfield/DAHN setting:

| Module | Based on |
|---|---|
| [`difflayers/activation.py` — `HopfieldCore`](difflayers/activation.py) | [`torch.nn.MultiheadAttention`](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/modules/activation.py#L771) |
| [`difflayers/functional.py` — `hopfield_core_forward`](difflayers/functional.py) | [`torch.nn.functional.multi_head_attention_forward`](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/functional.py#L3854) |
| [`difflayers/transformer.py` — `HopfieldEncoderLayer`](difflayers/transformer.py) | [`torch.nn.TransformerEncoderLayer`](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/modules/transformer.py#L241) |
| [`difflayers/transformer.py` — `HopfieldDecoderLayer`](difflayers/transformer.py) | [`torch.nn.TransformerDecoderLayer`](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/modules/transformer.py#L303) |

---

## License

BSD-style license — see [LICENSE](LICENSE).
