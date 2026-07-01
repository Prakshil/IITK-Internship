# Paper Summary: Discovering Faster Matrix Multiplication Algorithms with Reinforcement Learning

> **Fawzi, A., Balog, M., Huang, A., Hubert, T., Romera-Paredes, B., Barekatain, M., Novikov, A., Ruiz, F.J.R., Schrittwieser, J., Swirszcz, G., Silver, D., Hassabis, D. & Kohli, P.**  
> *Nature* **610**, 47–53 (2022) — [DOI: 10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4)

---

## 1. Problem Statement

Matrix multiplication is a primitive operation at the core of countless computational tasks — from neural networks to scientific computing. The standard algorithm for multiplying $n \times n$ matrices requires $O(n^3)$ scalar multiplications, but faster algorithms exist (e.g., Strassen's $O(n^{2.807})$). Finding algorithms that minimize the exponent (the so-called **matrix multiplication exponent** $\omega$) or the exact number of multiplications for fixed-size matrices is a fundamental open problem in algebraic complexity theory.

The paper addresses the problem of **automatically discovering provably correct and efficient matrix multiplication algorithms** using machine learning. This is framed as finding low-rank decompositions of a fixed 3D tensor — the matrix multiplication tensor.

---

## 2. Core Idea: Algorithms as Tensor Decompositions

Matrix multiplication is a bilinear operation: $(A, B) \mapsto AB$. A bilinear operation on $n \times n$ matrices can be fully encoded as a 3D tensor $\mathscr{T}_n$ of shape $n^2 \times n^2 \times n^2$ with entries in $\{0,1\}$.

A matrix multiplication algorithm corresponds to decomposing this target tensor into a sum of rank-1 terms:

$$\mathscr{T}_n = \sum_{r=1}^{R} u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$$

where each term $u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$ represents one scalar multiplication. The **rank** $R$ is the number of scalar multiplications used by the algorithm. Minimizing $R$ yields faster algorithms, both for fixed-size multiplication and (through recursion) for asymptotic complexity $O(N^{\log_n(R)})$.

---

## 3. Approach: AlphaTensor

### 3.1 TensorGame

AlphaTensor frames algorithm discovery as a **single-player game called TensorGame**:

- **State:** A 3D tensor $\mathscr{S}_t$, initially set to $\mathscr{T}_n$.
- **Action:** The player selects a triplet $(u^{(t)}, v^{(t)}, w^{(t)})$, subtracting the resulting rank-1 tensor from the state.
- **Goal:** Reach the zero tensor in as few moves as possible.
- **Reward:** $-1$ per step (to encourage short decompositions), plus a terminal penalty tied to the rank of the residual tensor if the move limit is exhausted.

### 3.2 AlphaTensor Agent

Built on **AlphaZero** with extensions for large action spaces (Sampled AlphaZero):

| Component | Description |
|---|---|
| **Neural Network** | Transformer-based architecture with axial attention over three 2D grid projections of the input tensor. |
| **Policy Head** | Autoregressive model that samples actions $(u,v,w)$ from a discrete set $F$ (e.g., $\{-2,-1,0,1,2\}$). |
| **Value Head** | MLP predicting distribution over returns (i.e., the rank of the current state). |
| **Search** | Sample-based Monte Carlo Tree Search (MCTS) guides action selection during gameplay. |

### 3.3 Key Innovations

1. **Synthetic demonstrations:** Since constructing a tensor from random factors is easy while decomposition is hard, a dataset of 5M synthetic $(tensor, factorization)$ pairs is used for supervised pretraining alongside RL.

2. **Change of basis:** At the start of each game, a random change of basis is applied to the target tensor. This injects diversity while preserving rank, allowing the agent to explore different geometric views of the same problem.

3. **Multi-target training:** A single agent is trained on multiple tensor sizes simultaneously ($n,m,p \leq 5$), enabling transfer learning across related problems.

4. **Data augmentation:** Factor order invariance is exploited to create additional training examples from completed games.

---

## 4. Key Results

### 4.1 Algorithm Discovery

| Tensor | Best Known Rank (Before) | AlphaTensor Rank | Significance |
|---|---|---|---|
| $\mathscr{T}_{3,3,3}$ | **23** (Laderman, 1976) | **23** | Matched the 46-year record |
| $\mathscr{T}_{4,4,4}$ ($\mathbb{Z}_2$) | **49** (Strassen², 1969) | **47** | **First improvement in 50 years** |
| $\mathscr{T}_{4,5,5}$ ($\mathbb{R}$) | **80** | **76** | New state of the art |
| $\mathscr{T}_{3,4,5}$ ($\mathbb{R}$) | **50** | **47** | New state of the art |

The 4x4 improvement in $\mathbb{Z}_2$ is especially notable: it surpasses Strassen's two-level algorithm for the first time since its discovery in 1969, yielding asymptotic complexity $O(N^{2.778})$.

### 4.2 Large Scale Improvements

By combining discovered algorithms recursively, AlphaTensor improves over state-of-the-art for **more than 70 matrix multiplication tensor sizes** (up to $12 \times 12 \times 12$).

### 4.3 Algorithm Diversity

AlphaTensor discovered **over 14,000 non-equivalent rank-49 factorizations** of $\mathscr{T}_{4,4,4}$ — meaning the space of matrix multiplication algorithms is far richer than previously known. These are provably distinct under symmetry transformations.

### 4.4 Hardware-Tailored Algorithms

When optimizing for practical runtime (rather than rank), AlphaTensor discovered algorithms achieving:
- **~8.5% speedup** over Strassen² on NVIDIA V100 GPUs for 8192×8192 matrices
- **Comparable gains** on Google TPUv2
- Algorithms optimized for one hardware do **not** transfer to the other, highlighting the importance of hardware-specific optimization

### 4.5 Beyond Standard Matrix Multiplication

AlphaTensor discovered a near-optimal algorithm for **skew-symmetric matrix-vector multiplication** using $(n-1)(n+2)/2 \sim \frac{1}{2}n^2$ multiplications, improving from the previously known $O(n^2)$ and matching the theoretical lower bound asymptotically.

---

## 5. Methodology Details

### 5.1 Reinforcement Learning Setup

- **State representation:** $S \times S \times S$ tensor (padded to $25 \times 25 \times 25$ for multi-target training).
- **Action space:** All triplets $(u,v,w)$ with entries in $F$ — astronomically large ($>10^{12}$ for $\mathscr{T}_4$).
- **Network architecture:** Transformer with axial attention, processing three $S \times S$ cyclic projections of the input tensor.
- **Training:** Distributed RL with actors running MCTS and a central learner updating network parameters via Adam.

### 5.2 Reward Schemes

1. **Rank optimization:** $-1$ per step + terminal penalty = encourages minimal $R$.
2. **Runtime optimization:** Additional terminal reward equal to negative benchmarking time on target hardware.

### 5.3 Search

Sample-based MCTS with:
- Probabilistic UCT action selection
- Transposition tables (exploiting action commutativity)
- Risk-seeking value estimate (focus on best trajectory, not average)
- Adaptive temperature for policy smoothing

---

## 6. Significance & Impact

1. **First ML-discovered improvement** over human-designed matrix multiplication algorithms for a classic open problem.
2. **Demonstrates DRL viability** for tackling NP-hard mathematical problems (tensor decomposition).
3. **Practical impact:** Direct speedups in real hardware (GPU/TPU) without any hardware knowledge priors.
4. **Theoretical impact:** Rich new dataset of algorithms for algebraic complexity research.

---

## 7. Limitations

1. **Discrete coefficient set $F$** must be predefined, potentially missing efficient algorithms requiring larger coefficients.
2. **Computational cost:** Requires massive distributed TPU/GPU compute for training.
3. **Fixed-size focus:** Algorithms discovered for fixed $n$ need recursion for arbitrary sizes, losing optimality.
4. **Proof of optimality:** AlphaTensor finds upper bounds on rank but cannot prove lower bounds — the optimal rank for most tensors remains unknown.

---

## 8. Connections to Our Work

This repository builds directly on the AlphaTensor foundation:

| AlphaTensor Component | Our Usage |
|---|---|
| Published $T_{3,3,3}$ rank-23 factorization | **Input seed** for our optimizer |
| Tensor reconstruction via $\sum u \otimes v \otimes w$ | **Forward pass** in our pipeline |
| Integer-coefficient constraint | **Challenge** our $L_1$ regularization aims to enforce |
| TPU-scale RL search | Replaced with **local GPU gradient descent** on continuous parameters |

Our **Matrix Optimizer** (`algorithms/Matrix_Optimizer.ipynb`) tests a fundamentally different hypothesis: rather than searching the discrete space from scratch (as AlphaTensor does), we attempt to continuously deform AlphaTensor's known rank-23 solution into a rank-22 solution, probing whether the Laderman limit is truly rigid or merely difficult to discover with discrete methods.
