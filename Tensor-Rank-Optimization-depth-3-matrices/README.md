# Tensor Rank Optimization for Depth-3 Matrices

**Research Project under Prof. Nitin Saxena**  
Department of Computer Science & Engineering  
Indian Institute of Technology Kanpur (IITK)

---

> **Pushing the boundaries of matrix multiplication complexity by forcing a 22-step solution to the 3x3 tensor rank problem using continuous optimization.**

This repository extends DeepMind's AlphaTensor framework — which discovered matrix multiplication algorithms via deep reinforcement learning — with a novel **continuous relaxation and gradient descent pipeline** that tests the rigidity of the Laderman 23-step limit for 3x3 matrix multiplication.

---

## Table of Contents

- [Overview](#overview)
- [The AlphaTensor Foundation](#the-alphatensor-foundation)
  - [The Paper](#the-paper)
  - [Tensor Rank Decomposition](#tensor-rank-decomposition)
  - [DeepMind's Key Results](#deepminds-key-results)
- [Our Novel Contribution: Matrix Optimizer](#our-novel-contribution-matrix-optimizer)
  - [Motivation](#motivation)
  - [The Pipeline (Phase-by-Phase)](#the-pipeline-phase-by-phase)
  - [Key Implementation Details](#key-implementation-details)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Citations](#citations)
- [License](#license)

---

## Overview

Matrix multiplication is the fundamental bottleneck in modern computing. For decades, discovering faster ways to multiply matrices relied on human intuition. DeepMind's AlphaTensor shifted this paradigm by framing algorithm discovery as a single-player 3D puzzle called **TensorGame**.

**Our Objective:** To test the absolute rigidity of the 23-step Laderman limit by forcing a 22-step solution using continuous optimization on local GPU hardware — a fundamentally different approach from DeepMind's discrete reinforcement learning across massive TPU clusters.

| Feature | DeepMind's AlphaTensor | Our Implementation |
|---|---|---|
| **Core Method** | Deep Reinforcement Learning + MCTS | Continuous Relaxation & Gradient Descent |
| **Search Space** | Discrete (exact integers) | Continuous (fluid decimals) |
| **Hardware** | Google TPU Supercomputers | Local GPU |
| **Target** | Discovering algorithms from scratch | Compressing a known 23-step algorithm to 22 |

---

## The AlphaTensor Foundation

### The Paper

> **Discovering faster matrix multiplication algorithms with reinforcement learning**  
> Fawzi, A. et al. *Nature* **610**, 47–53 (2022)  
> [DOI: 10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4)

**Summary:** The paper introduces AlphaTensor, a deep reinforcement learning agent based on AlphaZero, trained to discover efficient and provably correct matrix multiplication algorithms. AlphaTensor frames algorithm discovery as TensorGame, where the agent decomposes a 3D matrix multiplication tensor into rank-one factors. Key achievements include:

- **First improvement in 50 years** over Strassen's two-level algorithm for 4x4 matrices in modular arithmetic (47 steps vs 49)
- **State-of-the-art complexity** for over 70 matrix multiplication tensor sizes
- **Hardware-tailored algorithms** optimized for actual runtime on GPUs and TPUs
- **Discovery of thousands of non-equivalent algorithms**, revealing a richer space of matrix multiplication algorithms than previously known

A detailed summary document is available in [`paper/paper_summary.md`](paper/paper_summary.md).

### Tensor Rank Decomposition

The rigid rules of matrix multiplication can be represented as a 3D grid of numbers called a target tensor ($T_{\text{target}}$). For $3 \times 3$ matrix multiplication, this tensor is a $9 \times 9 \times 9$ cube where each non-zero entry encodes a bilinear relationship between entries of matrices $A$, $B$, and $C = AB$.

AlphaTensor's goal is to decompose this target tensor into a sum of rank-1 outer products. Each outer product ($u \otimes v \otimes w$) represents a single scalar multiplication step:

$$T_{\text{target}} = \sum_{r=1}^{R} u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$$

Finding the smallest possible $R$ (the **tensor rank**) means discovering the fastest possible algorithm.

### DeepMind's Key Results

| Tensor Size | Best Known Rank (Before) | AlphaTensor Rank | Notes |
|---|---|---|---|
| $T_{3,3,3}$ | 23 (Laderman, 1976) | 23 | Matched state-of-the-art |
| $T_{4,4,4}$ ($\mathbb{Z}_2$) | 49 (Strassen², 1969) | **47** | **First improvement in 50 years** |
| $T_{4,5,5}$ ($\mathbb{R}$) | 80 | **76** | Improved by 4 multiplications |
| $T_{3,4,5}$ ($\mathbb{R}$) | 50 | **47** | Improved by 3 multiplications |

---

## Our Novel Contribution: Matrix Optimizer

While DeepMind used reinforcement learning across massive TPU clusters to search discrete integer spaces, this repository explores a **completely different algorithmic angle** using local GPU hardware.

### Motivation

The 3x3 matrix multiplication tensor has a known rank of 23 (the Laderman limit). This bound has stood since 1976. Our approach asks: **can we beat this limit by starting from DeepMind's 23-step solution and compressing it to 22 steps through continuous optimization?**

### The Pipeline (Phase-by-Phase)

The full implementation lives in [`algorithms/Matrix_Optimizer.ipynb`](algorithms/Matrix_Optimizer.ipynb).

#### Phase 1: Tensor Formulation & Rank Reduction

We extract DeepMind's published 23-step integer solution for the $3 \times 3$ matrix from `factorizations_r.npz`. This provides three matrices ($U, V, W$) shaped $9 \times 23$. We deliberately drop the weakest column (identified by norm magnitude), reducing the rank to **22**. This fundamentally breaks the mathematical formula, creating a structural error gap.

#### Phase 2: Continuous Relaxation & Noise Injection

We convert the remaining rigid integers into continuous real numbers ($\mathbb{R}$) and inject Gaussian noise:

```python
U_22 = U_22 + torch.randn_like(U_22) * noise_scale
```

This physically shakes the parameters out of DeepMind's original mathematical geometry, forcing the optimizer to find a completely new pathway to cover the deleted step.

#### Phase 3: The Forward Pass

The GPU reconstructs the predicted 3D tensor using the current 22 continuous steps via Einstein summation, computing bilinear outer products across all 729 coordinates simultaneously:

```python
T_pred = torch.einsum('ir,jr,kr->ijk', U_22, V_22, W_22)
```

#### Phase 4: Loss Calculation & $L_1$ Regularization

We evaluate against the perfect target tensor using Mean Squared Error (MSE), plus an $L_1$ penalty that acts as a "mathematical freezer":

```python
loss = F.mse_loss(T_pred, T_target)
l1_penalty = 0.01 * (|U_22|.mean() + |V_22|.mean() + |W_22|.mean())
total_loss = loss + l1_penalty
```

As MSE drops, the $L_1$ penalty applies constant pressure on the decimals, attempting to force fractional values toward stable integers ($-1, 0, 1$).

#### Phase 5: Backpropagation & Optimization

The Adam optimizer calculates gradients for all 198 active variables and nudges them down the loss landscape. The loop runs for up to 50,000 epochs with early stopping if MSE drops below $10^{-6}$.

### Key Implementation Details

**File:** `algorithms/Matrix_Optimizer.ipynb`

- **Framework:** PyTorch with CUDA support
- **Target tensor:** $9 \times 9 \times 9$ representing $3 \times 3$ matrix multiplication
- **Starting point:** DeepMind's rank-23 factorization from `factorizations_r.npz`
- **Optimizer:** Adam (lr=0.01)
- **Regularization:** $L_1$ penalty on factor entries
- **Early stopping:** MSE threshold of $10^{-6}$
- **Device:** Auto-detects CUDA (NVIDIA GPU) or falls back to CPU

---

## Repository Structure

```
.
├── algorithms/
│   ├── Matrix_Optimizer.ipynb    # OUR NOVEL CONTRIBUTION: Continuous optimization
│   │                             # pipeline to compress rank-23 to rank-22
│   ├── explore_factorizations.ipynb  # AlphaTensor: Load & explore factorizations
│   ├── factorizations_r.npz      # Precomputed factorizations in standard arithmetic (ℝ)
│   └── factorizations_f2.npz     # Precomputed factorizations in modular arithmetic (ℤ₂)
│
├── benchmarking/
│   ├── README.md                 # Benchmarking instructions
│   ├── factorizations.py         # AlphaTensor's GPU/TPU-optimized factorizations
│   ├── utils.py                  # Factorization-to-algorithm conversion utilities
│   ├── run_gpu_benchmark.py      # GPU benchmarking script (V100)
│   ├── test_correctness.py       # Unit tests for factorization correctness
│   └── requirements.txt          # JAX, NumPy, etc.
│
├── nonequivalence/
│   ├── inspect_factorizations_notebook.ipynb  # Notebook for computing invariants
│   └── alphatensor_14236_factorizations.npz   # 14,236 non-equivalent factorizations
│
├── recombination/
│   ├── recombination.py          # Core recombination algorithm
│   ├── sota.py                   # State-of-the-art rank table
│   ├── recombination_test.py     # Unit tests
│   ├── example.py                # Example: rank-255 for T_{4,9,10}
│   └── requirements.txt          # NumPy, absl-py
│
├── paper/
│   └── paper_summary.md          # Detailed summary of the AlphaTensor paper
│
├── .gitignore
└── README.md                     # This file
```

### File Details

#### `algorithms/` — Algorithms & Factorizations

| File | Description |
|---|---|
| `Matrix_Optimizer.ipynb` | **Our contribution.** Implements the continuous optimization pipeline. Loads DeepMind's rank-23 $T_{3,3,3}$ factorization, drops the weakest column to force rank 22, and uses gradient descent with $L_1$ regularization to attempt reconstruction. |
| `explore_factorizations.ipynb` | AlphaTensor's original Colab notebook. Loads `.npz` factorization files and verifies correctness against the matrix multiplication tensor. |
| `factorizations_r.npz` | Precomputed factorizations found by AlphaTensor in **standard arithmetic** ($\mathbb{R}$) for various matrix sizes. The `'3,3,3'` key contains the rank-23 Laderman-matching algorithm used as our starting point. |
| `factorizations_f2.npz` | Precomputed factorizations in **modular arithmetic** ($\mathbb{Z}_2$). |

#### `benchmarking/` — GPU Benchmarking

| File | Description |
|---|---|
| `factorizations.py` | Provides `get_4x4x4_alphatensor_gpu()`, `get_4x4x4_alphatensor_tpu()`, and `get_4x4x4_strassen_squared()` — the factorizations used for GPU/TPU benchmarking. |
| `utils.py` | Converts tensor factorizations into executable JAX matrix multiplication algorithms. Includes `block_split`, `algorithm_from_factors`, and benchmarking utilities. |
| `run_gpu_benchmark.py` | Benchmarks AlphaTensor's GPU-tailored algorithm against Strassen² and `jnp.dot` on an NVIDIA V100 GPU. Reports ~8.5% speedup on 8192x8192 matrices. |
| `test_correctness.py` | Unit tests verifying that factorizations correctly decompose the matrix multiplication tensor, and that the resulting algorithms produce correct results. |

#### `nonequivalence/` — Algorithm Diversity

| File | Description |
|---|---|
| `inspect_factorizations_notebook.ipynb` | Notebook demonstrating that AlphaTensor discovered **14,236 non-equivalent** rank-49 factorizations of $T_{4,4,4}$ by computing matrix rank invariants $\mathscr{R}$ and $\mathscr{K}$. |
| `alphatensor_14236_factorizations.npz` | The 14,236 factorizations in compressed NumPy format. |

#### `recombination/` — Recursive Decomposition

| File | Description |
|---|---|
| `recombination.py` | Implements recursive decomposition of large matrix multiplication tensors by recombining factorizations of smaller ones (based on Sedoglavic 2017 and Drevet et al. 2011). |
| `sota.py` | Database of best known tensor ranks for $T_{a,b,c}$ (up to $a,b,c \leq 12$), including AlphaTensor's improvements. |
| `example.py` | Demonstrates recombination to find a rank-255 decomposition of $T_{10,4,9}$, improving over the previous best known rank of 259. |

---

## Installation

### Requirements

- Python 3.7+
- PyTorch (for Matrix Optimizer)
- CUDA-capable GPU recommended (tested on GTX 1080 Ti)

### Matrix Optimizer

```bash
pip install torch numpy
```

Launch Jupyter and open `algorithms/Matrix_Optimizer.ipynb`.

### AlphaTensor Original Code

#### `algorithms/` (Factorizations)

No installation required. Open `explore_factorizations.ipynb` in Colab or Jupyter and upload the relevant `.npz` file.

#### `benchmarking/`

Requires an NVIDIA V100 GPU and JAX:

```bash
git clone https://github.com/deepmind/alphatensor.git
cd alphatensor
pip install -r benchmarking/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m alphatensor.benchmarking.test_correctness
```

#### `recombination/`

```bash
pip install numpy absl-py==1.2.0
python -m alphatensor.recombination.example
```

---

## Usage

### Running the Matrix Optimizer

1. **Set up the environment:** Place `factorizations_r.npz` in the same directory as `Matrix_Optimizer.ipynb`.
2. **Launch the notebook:** Open `algorithms/Matrix_Optimizer.ipynb`.
3. **Execute cells sequentially:**
   - **Cell 1:** Loads AlphaTensor's data and extracts the $T_{3,3,3}$ factorization.
   - **Cell 2:** Constructs the $9 \times 9 \times 9$ target tensor and moves it to GPU.
   - **Cell 3:** Compresses to rank 22 by dropping the weakest column, initializes with noise, and sets up the Adam optimizer.
   - **Cell 4:** Runs the training loop — up to 50,000 epochs with live MSE reporting.

### Exploring AlphaTensor Factorizations

Open `algorithms/explore_factorizations.ipynb` and upload `factorizations_r.npz` or `factorizations_f2.npz` to browse all discovered algorithms and verify their correctness.

---

## Results & Insights

Through thousands of GPU iterations, the continuous optimizer successfully navigated the non-convex landscape to achieve highly accurate decimal approximations (**MSE reaching as low as 0.000018**).

However, the pipeline also highlighted the **extreme rigidity of the Laderman limit**:
- The near-perfect decimal scores relied heavily on micro-decimals that could not be resolved to exact integers.
- When $L_1$ regularization applied maximum pressure to force discrete integer convergence ($-1, 0, 1$), the geometric structure resisted, **validating the extreme mathematical difficulty — and potential impossibility — of a 22-step integer solution** for standard arithmetic under standard constraints.

This suggests that while continuous optimization can find highly accurate low-rank approximations (effectively proving that rank 22 is achievable within a small error tolerance in $\mathbb{R}$), the rigidity of the exact integer problem likely requires new mathematical insight rather than pure optimization.

---

## Citations

If you use this code or build upon this work, please cite both the original AlphaTensor paper and this repository:

```bibtex
@Article{AlphaTensor2022,
  author  = {Fawzi, Alhussein and Balog, Matej and Huang, Aja
             and Hubert, Thomas and Romera-Paredes, Bernardino
             and Barekatain, Mohammadamin and Novikov, Alexander
             and Ruiz, Francisco J. R. and Schrittwieser, Julian
             and Swirszcz, Grzegorz and Silver, David
             and Hassabis, Demis and Kohli, Pushmeet},
  journal = {Nature},
  title   = {Discovering faster matrix multiplication algorithms
             with reinforcement learning},
  year    = {2022},
  volume  = {610},
  number  = {7930},
  pages   = {47--53},
  doi     = {10.1038/s41586-022-05172-4}
}
```

---

## License

**Copyright 2022 DeepMind Technologies Limited**

All software is licensed under the **Apache License, Version 2.0**.  
All other materials are licensed under the **Creative Commons Attribution 4.0 International License (CC-BY)**.

This is not an official Google product.
