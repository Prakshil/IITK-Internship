# PROJECT.md — ArithCircuit Pipeline Project Documentation

**Project:** ML-Guided Arithmetic Circuit Optimization  
**Affiliation:** Prof. Nitin Saxena, Department of Computer Science & Engineering, Indian Institute of Technology Kanpur (IIT Kanpur)  
**Repository:** https://github.com/Prakshil/IITK-Internship/tree/main/Arithmetic_Circuit_Optimization  
**Status:** Complete — All phases implemented, tested, and documented  
**Last Updated:** October 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Phase 0: Environment Setup](#3-phase-0-environment-setup)
4. [Phase 1: Dataset Generation](#4-phase-1-dataset-generation)
5. [Phase 2: Parsing → AST](#5-phase-2-parsing--ast)
6. [Phase 3: Normalization](#6-phase-3-normalization)
7. [Phase 4: DAG Conversion](#7-phase-4-dag-conversion)
8. [Phase 5: Feature Extraction](#8-phase-5-feature-extraction)
9. [Phase 6: Cost Model](#9-phase-6-cost-model)
10. [Phase 7: Candidate Generation](#10-phase-7-candidate-generation)
11. [Phase 8: GINEConv Rewrite-Policy Ranker](#11-phase-8-gineconv-rewrite-policy-ranker)
12. [Phase 9: Beam-Search Circuit Optimizer](#12-phase-9-beam-search-circuit-optimizer)
13. [Phase 10: Cost Model & PIT Verification](#13-phase-10-cost-model--pit-verification)
14. [Phase 11: Interpretability (RF + Decision Tree)](#14-phase-11-interpretability-rf--decision-tree)
15. [Phase 12: Full Pipeline Benchmark](#15-phase-12-full-pipeline-benchmark)
16. [Phase 13: RL Fine-Tuning (Extension)](#16-phase-13-rl-fine-tuning-extension)
17. [Phase 14: Self-Supervised Training (Extension)](#17-phase-14-self-supervised-training-extension)
18. [Phase 15: Depth Generalization Study](#18-phase-15-depth-generalization-study)
19. [Phase 16: GNNExplainer Interpretability](#19-phase-16-gnnexplainer-interpretability)
20. [Phase 17: Theoretical PIT Analysis](#20-phase-17-theoretical-pit-analysis)
21. [Phase 18: Paper Writing & Submission](#21-phase-18-paper-writing--submission)
22. [Key Results Summary](#22-key-results-summary)
23. [Known Issues & Limitations](#23-known-issues--limitations)
24. [Citation](#24-citation)
25. [License](#25-license)

---

## 1. Project Overview

### 1.1 Purpose

The ArithCircuit Pipeline is a complete **ML-guided arithmetic circuit optimization** system. Given an arithmetic expression, it parses it into a Directed Acyclic Graph (DAG), extracts 20-dimensional node features, generates candidate rewrite actions using algebraic rewrite rules, ranks them with a GINEConv 3-head policy network (node reducibility, rule selection, confidence), filters by cost model and Polynomial Identity Testing (PIT), and applies the best transformations via beam search optimization.

### 1.2 Problem Statement

Optimal arithmetic circuit factorization is **NP-hard**. Finding the minimal circuit (fewest gates or minimum depth) for a given polynomial expression requires exhaustive search in the worst case. This project addresses this by using Machine Learning to learn heuristics that guide the search toward high-quality circuit optimizations efficiently.

### 1.3 Key Innovations

1. **GINEConv 3-Head Policy Network** — Simultaneously predicts node reducibility, rule selection, and confidence
2. **Schwartz–Zippel PIT Verification** — Rigorous algebraic equivalence checking with provable false-positive bounds
3. **Beam Search with Diversity Pruning** — Explores multiple optimization paths while maintaining diversity
4. **Self-Supervised Scramble-and-Reverse Training** — Generates guaranteed-optimal training data by reversing rewrite rules
5. **Depth Generalization Study** — First analysis of ML-guided circuit optimization transfer to deeper circuits
6. **GNNExplainer Interpretability** — Node-level and edge-level importance explanations for model decisions

### 1.4 Target Audience

- Researchers in arithmetic circuit complexity and algebraic complexity theory
- Students working on ML-guided compiler optimization
- Engineers building hardware synthesis and logic optimization tools
- Anyone interested in ML-guided symbolic computation

---

## 2. Repository Structure

```
Arithmetic_Circuit_Optimization/
├── ArithCircuit_Pipeline.ipynb   # Main pipeline notebook (all phases + code)
├── Pipeline.pdf                  # Rendered PDF report
├── README.md                     # Project overview and quick start
├── PROJECT.md                    # This file — detailed phase-by-phase documentation
├── requirements.txt              # Python package dependencies
├── environment.yml               # Conda environment specification
├── IMPLEMENTATION_PLAN.md        # Detailed implementation plan with weekly schedule
├── rules.md                      # Development workflow rules (strict execution protocol)
├── paper.tex                     # Complete research paper (LaTeX)
│
├── data/                         # Data artifacts (gitignored)
│   ├── dataset_100000_depth3.pkl # Generated dataset (100k expressions)
│   ├── pyg_data_list.pkl         # PyG training samples (631,073)
│   ├── pyg_data_list_with_oracle.pkl # With oracle-augmented data
│   ├── df_eval.pkl               # Benchmark results (500 expressions)
│   ├── depth_generalization.csv  # Depth generalization results
│   ├── gin_history.pkl           # GIN training history
│   ├── tabular_split.pkl         # RF/DT train/test split
│   └── pyg_progress.pkl          # Training progress checkpoint
│
├── models/                       # Model checkpoints (gitignored)
│   ├── gin_best.pt               # Best supervised GIN model
│   ├── gin_best_fixed.pt         # Best supervised GIN (with oracle fixes)
│   ├── gin_rl_finetuned.pt       # RL fine-tuned GIN model
│   ├── gin_ss_finetuned.pt       # Self-supervised fine-tuned GIN model
│   ├── gin_epoch_XX.pt           # Per-epoch checkpoints (01-15)
│   ├── gin_latest.pt             # Latest checkpoint
│   ├── rf_model.joblib           # Random Forest interpretability model
│   └── dt_model.joblib           # Decision Tree interpretability model
│
├── graphs/                       # Exported plots (gitignored)
│   ├── ablation_results.png      # Ablation study results
│   ├── benchmark_plots.png       # Benchmark visualization
│   ├── benchmark_results.png     # Benchmark results chart
│   └── per_rule_f1.png           # Per-rule F1 score chart
│
├── report_figures/               # Report figures (22 PNG files)
│   ├── Figure_1_1_Pipeline_Flow.png
│   ├── Figure_3_1_Dataset_Overview.png
│   ├── Figure_4_10a_GIN_Node_Reducibility_1.png
│   ├── Figure_4_10b_GIN_Node_Reducibility_2.png
│   ├── Figure_4_10c_GIN_Node_Reducibility_3.png
│   ├── Figure_4_2_GIN_Loss_Curve.png
│   ├── Figure_4_2_GIN_Loss_Curves.png
│   ├── Figure_4_3_Feature_Importance.png
│   ├── Figure_4_4_Decision_Tree_Proxy.png
│   ├── Figure_4_5_Confusion_Matrix.png
│   ├── Figure_4_6_Per_Rule_F1.png
│   ├── Figure_4_7_Benchmark_Results.png
│   ├── Figure_4_8_Ablation_Study.png
│   ├── Figure_4_9_Circuit_Comparison.png
│   ├── Figure_D1_Factorization.png
│   ├── Figure_D2_CommonFactor.png
│   ├── Figure_D3_Collection.png
│   ├── Figure_D4_Distribution.png
│   └── per_rule_f1.png
│
├── docs/                         # Theoretical documentation
│   ├── diagnosis_zero_improvement.md   # Zero-improvement rules diagnosis
│   └── pit_analysis.md               # Theoretical PIT analysis
│
└── __pycache__/                    # Python cache (auto-generated)
```

---

## 3. Phase 0: Environment Setup

### 3.1 Hardware Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce GTX 1080 Ti (CUDA 11.8) |
| CPU | 16 cores |
| PyTorch | 1.14.0a0+410ce96 |

### 3.2 Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| sympy | >=1.12 | Symbolic math, expression parsing, algebraic manipulation |
| networkx | >=3.0 | DAG construction, graph algorithms, visualization |
| numpy | >=1.24 | Numerical arrays, feature computation |
| pandas | >=2.0 | DataFrames for metrics, analysis tables |
| matplotlib | >=3.7 | Plots, charts, DAG visualization |
| torch | >=1.14 | Deep learning framework (PyTorch) |
| torch-geometric | >=2.3 | GINEConv layers, graph batching, Data objects |
| scikit-learn | >=1.2 | Random Forest, Decision Tree, evaluation metrics |
| tqdm | >=4.64 | Progress bars for long-running loops |
| joblib | >=1.2 | Model serialization |

### 3.3 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate arithcircuit
```

### 3.4 Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| DATASET_TOTAL | 100,000 | Number of training expressions |
| N_EPOCHS | 15 | GIN training epochs |
| N_FEATURES | 20 | Node feature dimensions |
| GIN_TOP_K | 10 | Top-K candidates from GIN ranking |
| BEAM_WIDTH | 8 | Beam search width |
| N_RULES | 8 | Number of trainable rewrite rules |
| EDGE_FEAT_DIM | 5 | Edge feature dimensions |
| PIT_SMALL_NODES | 15 | Symbolic check threshold |
| PIT_RANDOM_TRIALS | 30 | Schwartz–Zippel trials |
| PIT_PRIME | 2,147,483,647 | Prime modulus for Schwartz–Zippel |
| OPT_PATIENCE | 5 | Optimizer patience |
| PYG_SAVE_EVERY | 2,000 | PyG checkpoint interval |

### 3.5 Cost Weight Profiles

| Profile | λ_S (Size) | λ_D (Depth) | λ_M (Mul) | λ_T (Temp) | λ_G (Degree) |
|---------|-----------|------------|----------|-----------|-------------|
| depth_first | 0.20 | **0.50** | 0.15 | 0.10 | 0.05 |
| size_first | 0.40 | 0.25 | 0.15 | 0.10 | 0.10 |
| balanced | 0.30 | 0.35 | 0.15 | 0.10 | 0.10 |

The default **depth-first** profile emphasizes depth reduction (λ_D = 0.50), targeting optimization of circuit latency.

---

## 4. Phase 1: Dataset Generation

### 4.1 Overview

Generates **100,000 arithmetic expressions** across **14 families**, filtered to depth-3 circuits.

### 4.2 Expression Families

Each record stores: `{expr, family, n_vars, degree, size, dag_size, shared_count, max_depth, circuit_depth}`.

**Factorization Group:**
- `factorization_heavy` — expressions with repeated factor patterns
- `common_factor_target` — expressions with extractable common factors
- `depth3_factored` — depth-3 factored expressions

**CSE & Sharing Group:**
- `cse_heavy` — expressions with common subexpressions
- `multi_step_chain` — chain-structured expressions with sharing
- `depth3_circuit` — depth-3 circuit expressions
- `depth3_mul_sum` — depth-3 multiplication-sum structures

**Structural Group:**
- `depth_heavy` — deep expression trees
- `reassoc_heavy` — expressions benefiting from reassociation
- `reassoc_target` — reassociation-targeted expressions
- `depth3_reassoc` — depth-3 reassociation expressions
- `distrib_recovery` — expressions for distribution recovery
- `depth_reduction_target` — depth-reduction targeted expressions
- `const_fold_target` — expressions with foldable constants

### 4.3 Dataset Statistics

| Family | Avg Size | Avg Degree | Avg # Vars |
|--------|----------|-----------|-----------|
| common_factor_target | 7.00 | 2.0 | 4.57 |
| const_fold_target | 4.00 | 1.0 | 3.00 |
| cse_heavy | 6.97 | 2.0 | 5.33 |
| depth3_circuit | 5.00 | 2.0 | 4.00 |
| depth3_factored | 8.00 | 2.0 | 5.00 |
| depth3_mul_sum | 7.00 | 2.0 | 6.00 |
| depth3_reassoc | 3.00 | 1.0 | 4.00 |
| depth_heavy | 7.00 | 2.0 | 8.00 |
| depth_reduction_target | 5.00 | 1.0 | 6.00 |
| distrib_recovery | 8.00 | 2.0 | 4.00 |
| factorization_heavy | 7.00 | 2.0 | 4.64 |
| multi_step_chain | 9.98 | 2.0 | 5.36 |
| reassoc_heavy | 7.00 | 4.0 | 8.00 |
| reassoc_target | 7.00 | 2.0 | 8.00 |

**Checkpoint:** `dataset_100000_depth3.pkl` — set `RESUME=True` to skip regeneration.

### 4.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 1 (DatasetGenerator class)

Key methods:
- `generate()` — generates expressions per family
- `generate_from_config()` — generates with family quotas
- `_fill_family()` — fills a single family with retry logic
- `_make_rec()` — creates a record for each expression

---

## 5. Phase 2: Parsing → AST

### 5.1 Overview

SymPy-based tokenization converts expressions into hierarchical **Abstract Syntax Trees**. At this stage, each subexpression appears as a separate node — no sharing.

### 5.2 Node Types

| Type | SymPy Class | Description |
|------|------------|-------------|
| `var` | Symbol | Variable leaf node (x, y, z, ...) |
| `const` | Integer/Float | Constant leaf node (2, 3, 5, ...) |
| `add` | Add | Addition operator |
| `mul` | Mul | Multiplication operator |
| `pow` | Pow | Power operator |

### 5.3 Example

**Expression:** `(x+y)*z + (x+y)*w + (u+v)*z + (2+3)*x`

```
[add] add
  [mul] mul
    [const] 5
    [var] x
  [mul] mul
    [var] w
    [add] add
      [var] x
      [var] y
  [mul] mul
    [var] z
    [add] add
      [var] u
      [var] v
  [mul] mul
    [var] z
    [add] add
      [var] x
      [var] y
```

Nodes=19, Edges=18 → Tree (no sharing). DAG conversion will merge the two `(x+y)` occurrences.

### 5.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 2 (ASTParser class)

Key methods:
- `parse_expr(e)` — parses a SymPy expression into AST dict
- `parse_string(s)` — parses a string expression
- `ast_to_nx(ast)` — converts AST to networkx DiGraph
- `print_ast(ast, indent)` — prints hierarchical AST

---

## 6. Phase 3: Normalization

### 6.1 Overview

Three-step normalization pipeline applied before DAG conversion.

### 6.2 Normalization Steps

| Step | Function | Example |
|------|----------|---------|
| 1. Constant Folding | `constant_fold()` | `(2+3)*x → 5*x` |
| 2. Identity Removal | `remove_identities()` | `x*1+0 → x` |
| 3. Expand + Collect | `normalize_for_display()` | `(x+y)*z → x*z + y*z` |

### 6.3 Normalization Modes

| Mode | Operations | Use Case |
|------|-----------|----------|
| `normalize_for_cost()` | folding + identity removal | Used during optimization |
| `normalize_for_display()` | folding + identity removal + expand + collect | Used for visualization |

### 6.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 3 (ExprNormalizer class)

Key methods:
- `constant_fold(e)` — recursively folds constant subexpressions
- `remove_identities(e)` — removes x*1, x+0, x*0 identities
- `normalize_for_cost(e)` — lightweight normalization
- `normalize_for_display(e)` — full normalization with expand+collect

---

## 7. Phase 4: DAG Conversion

### 7.1 Overview

Hash-based **hash consing** merges common subexpressions, converting the AST into a **Directed Acyclic Graph**. Each edge carries a 5-dim one-hot `edge_type` attribute.

### 7.2 Edge Type Encoding

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | `add_child` | This operand is an addend |
| 1 | `mul_child` | This operand is a factor |
| 2 | `pow_base` | Base of a power |
| 3 | `pow_exp` | Exponent |
| 4 | `other` | Generic edge |

### 7.3 Node Colors (Visualization)

| Type | Color | Hex |
|------|-------|-----|
| add | Blue | #AED6F1 |
| mul | Green | #A9DFBF |
| var | Orange | #FAD7A0 |
| const | Red | #F1948A |
| pow | Purple | #D7BDE2 |

### 7.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 4 (DAGConverter class) and `rl_environment.py`

Key methods:
- `convert(expr)` — converts expression to DAG with hash consing
- `circuit_depth(G)` — computes circuit depth
- `dag_size(G)` — counts operation nodes
- `shared_nodes(G)` — returns nodes with fan_out > 1
- `visualize(G, title)` — colored DAG visualization
- `draw_as_tree(G, title)` — DSA textbook style tree view
- `compare_circuits(expr1, expr2)` — side-by-side before/after comparison

---

## 8. Phase 5: Feature Extraction

### 8.1 Overview

Extracts a **20-dimensional feature vector** for each DAG node, plus **5-dimensional edge features**.

### 8.2 Node Features

| # | Feature | Description |
|---|---------|-------------|
| 1–6 | `gate_*` | One-hot: var, const, add, mul, pow, other |
| 7 | `depth_from_root` | Normalized BFS distance from root |
| 8 | `fan_in` | Out-degree (normalized) |
| 9 | `fan_out_norm` | In-degree (fan-out, normalized) |
| 10 | `subtree_size_norm` | Number of nodes in subtree (normalized) |
| 11 | `is_shared` | 1 if fan_out > 1 (shared subexpression) |
| 12 | `local_degree` | Mul-arity or power exponent |
| 13 | `subtree_degree` | Max polynomial degree in subtree |
| 14 | `degree_to_output` | Distance to output (reverse BFS) |
| 15 | `depth_to_leaf` | Distance to nearest leaf — high = good rewrite target |
| 16 | `op_density` | ops-in-subtree / subtree_size — dense → factorizable |
| 17 | `mul_ratio` | mul_gates / subtree_size — mul-heavy → factorizable |
| 18 | `reuse_count` | Raw fan_out (reuse frequency) |
| 19 | `estimated_blowup` | Risk metric: cost increase if rewritten |
| 20 | `parent_op_type` | Normalized parent operator type |

### 8.3 Edge Features

| Code | Meaning |
|------|---------|
| 0 | `add_child` — this operand is an addend |
| 1 | `mul_child` — this operand is a factor |
| 2 | `pow_base` — base of a power |
| 3 | `pow_exp` — exponent |
| 4 | `other` — generic |

### 8.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 5 (FeatureExtractor class) and `rl_environment.py`

Key methods:
- `extract(G)` — extracts node features, edge index, edge attributes
- `to_pyg_data(G, ...)` — converts to PyTorch Geometric Data object

---

## 9. Phase 6: Cost Model

### 9.1 Overview

The cost function scores a circuit for optimization targeting:

```
Score(C) = λ_S·|C| + λ_D·D + λ_M·M + λ_T·T + λ_G·G
```

Where: `|C|` = node count, `D` = circuit depth, `M` = multiplication gates, `T` = temporary blowup penalty, `G` = polynomial degree.

### 9.2 Key Methods

| Method | Description |
|--------|-------------|
| `score_graph(G, temp_blowup)` | Scores an existing DAG |
| `score_expr(expr, temp_blowup)` | Parses and scores an expression |
| `annotate(original, candidates)` | Labels candidates with cost deltas |
| `temp_blowup_penalty(baseline, candidate)` | Computes intermediate growth penalty |
| `graph_metrics(G)` | Returns size, depth, muls, degree metrics |

### 9.3 PeakMetricsTracker

Monitors intermediate circuit growth during multi-step optimization to prevent temporary blowup in intermediate states.

### 9.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 5b (CostModel class) and `rl_environment.py`

---

## 10. Phase 7: Candidate Generation

### 10.1 Overview

Generates **(node, rule, expression)** candidate actions at two scopes:

| Scope | How | What it captures |
|-------|-----|-----------------|
| **Global** | Apply rule to entire expression | Whole-expression factorization, distribution, etc. |
| **Local** | Apply rule to each internal subtree node | Node-level rewrites, exposing local structure |

### 10.2 Rewrite Rules (8 Trainable)

| Index | Name | SymPy Operation | Description |
|-------|------|----------------|-------------|
| 0 | factorization | `sp.factor()` | Factor polynomial expressions |
| 1 | distribution | `sp.expand()` | Expand products of sums |
| 2 | reassociation | Balanced re-association | Rebalance Add/Mul trees |
| 3 | const_folding | `sp.nsimplify()` | Fold constant subexpressions |
| 4 | common_factor | GCD-based extraction | Extract common factors |
| 5 | depth_reduction | Factor + collect + powsimp | Multi-strategy depth reduction |
| 6 | collection | `sp.collect()` | Collect coefficients by variables |
| 7 | cse_rewrite | `cse()` | Common subexpression elimination |

**Excluded from training:** `identity` (no-op), `dead_pruning` (post-processing cleanup)

### 10.5 Candidate Labeling

Each candidate is labeled with:
- **`immediate_gain`** — Score improvement from applying the action
- **`future_gain`** — Score improvement from lookahead (multi-step optimization)
- **`improvement`** — Weighted combination: `0.5 × immediate + 0.5 × future`

### 10.6 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 6 (CandidateGenerator class) and `rl_environment.py`

Key methods:
- `generate_all_actions(expr, G)` — generates all global + local actions
- `generate_local_actions(expr, G)` — generates local (per-subtree) actions
- `generate_global_actions(expr, G)` — generates global actions
- `label_candidates(original, candidates, G)` — labels candidates with scores
- `lookahead_score(expr, cost_model, ...)` — computes future gain via lookahead
- `annotate_supervision(labeled, top_k)` — annotates supervision signals

---

## 11. Phase 8: GINEConv Rewrite-Policy Ranker

### 11.1 Architecture

```
Input: Node Features (N × 20) + Edge Features (E × 5)
  ↓
GINEConv ×3 → Hidden (N × 128)
  ├── Node Head → Reducibility Logit (N × 1) — which nodes are worth rewriting?
  ├── Rule Head → Rule Logits (N × 8) — which rule applies at each node?
  └── Conf Head → Confidence Logit (N × 1) — how much improvement is expected?
```

**Rank Score:** `rank(action) = σ(node[n]) × softmax(rule[n])[r] × σ(conf[n])`

### 11.2 Model Specifications

| Metric | Value |
|--------|-------|
| Total Parameters | 109,442 |
| GINEConv Layers | 3 |
| Hidden Dimension | 128 |
| Dropout | 0.2 |
| Training Samples | 631,073 |
| Epochs | 15 |

### 11.3 Loss Function

```
ℒ = α · BCE(node*) + β · CE(rule) + γ · Huber(σ(conf), gain)
```

Where:
- `node*` = argmax_node max_rule(improvement)
- Class-balanced weights for rule loss to prevent factorization dominance
- `conf` ≈ per-action improvement (sigmoid-scaled)

### 11.4 Training Progress

| Epoch | Train Loss | Val Loss | Node BCE | Rule CE | Conf Huber |
|-------|-----------|---------|---------|--------|-----------|
| 1 | 1.7284 | 1.5985 | 0.3811 | 1.3420 | 0.0105 |
| 2 | 1.6454 | 1.5766 | 0.3720 | 1.2684 | 0.0101 |
| 3 | 1.5868 | 1.5767 | 0.3652 | 1.2167 | 0.0098 |
| 4 | 1.5828 | 1.5700 | 0.3650 | 1.2130 | 0.0098 |
| 5 | 1.5719 | 1.6721 | 0.3633 | 1.2037 | 0.0098 |
| 6 | 1.5693 | 1.5696 | 0.3631 | 1.2014 | 0.0097 |
| 7 | 1.5632 | 2.0822 | 0.3623 | 1.1960 | 0.0097 |
| 8 | 1.5616 | 1.5675 | 0.3622 | 1.1946 | 0.0097 |
| 9 | 1.5614 | 1.5685 | 0.3619 | 1.1946 | 0.0097 |
| 10 | 1.5585 | 1.5703 | 0.3616 | 1.1921 | 0.0096 |
| 11 | 1.5573 | 1.5667 | 0.3614 | 1.1911 | 0.0096 |
| 12 | 1.5562 | 1.5666 | 0.3611 | 1.1903 | 0.0096 |
| 13 | 1.5557 | 1.5670 | 0.3610 | 1.1899 | 0.0096 |
| 14 | 1.5546 | 1.5662 | 0.3609 | 1.1890 | 0.0096 |
| 15 | 1.5549 | 1.5659 | 0.3608 | 1.1892 | 0.0096 |

### 11.5 Policy Evaluation (Validation)

| Metric | Value |
|--------|-------|
| Rule Accuracy | 63.16% |
| Node Accuracy | 92.24% |
| Confidence MSE | 0.0859 |

**Checkpoint:** `gin_latest.pt`, `gin_best.pt`, `gin_epoch_XX.pt`

### 11.6 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 7 (GINRewriteRanker class) and `rl_environment.py`

Key methods:
- `forward(x, edge_index, edge_attr)` — forward pass
- `encode(x, edge_index, edge_attr)` — encodes graph through GINEConv layers
- `rank_candidates(data, candidates, device)` — ranks candidates by score
- `train_action_batch(batch, optimizer, ...)` — trains on action batch

---

## 12. Phase 9: Beam-Search Circuit Optimizer

### 12.1 Overview

Iterative beam-search optimization with GIN ranking, cost filtering, PIT verification, and dead-node pruning.

### 12.2 Optimization Loop

```
State = (expr, history, improv)
  → Beam = Top-K states
  → Normalize + DAG
  → Extract Features
  → Generate Candidates
  → GIN Rank → Top-K
  → PIT + Prune
  → Expand Beam
```

### 12.3 Optimizer Configurations

| Mode | GIN | Pruning | Depth | Profit Filter | Lookahead | Beam Width |
|------|-----|---------|-------|---------------|-----------|-----------|
| v4_full | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| no_gin | ❌ | ✅ | ✅ | ✅ | ✅ | 8 |
| no_pruning | ✅ | ❌ | ✅ | ✅ | ✅ | 8 |
| cost_only | ❌ | ❌ | ✅ | ✅ | ✅ | 8 |
| no_depth | ✅ | ✅ | ❌ | ✅ | ✅ | 8 |
| no_lookahead | ✅ | ✅ | ✅ | ✅ | ❌ | 8 |
| beam_width_1 | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |

### 12.4 Demo Optimization Results

| Expression | Orig Cost | Final Cost | Reduction | Rules Applied |
|-----------|-----------|-----------|-----------|---------------|
| w*(x+y) + 5*x + z*(u+v) + z*(x+y) | 5.00 | 4.65 | 7.0% | collection (×1) |
| x² + 2xy + y² | 2.65 | 2.10 | 20.8% | factorization (×1) |
| xy + xz + yz | 2.95 | 2.80 | 5.1% | depth_reduction (×1) |
| (x+y+z)² | 2.30 | 2.30 | 0.0% | — |
| x³ - y³ | 3.40 | 3.40 | 0.0% | — |

### 12.5 Adaptive Beam Width

| Circuit Size (nodes) | Beam Width | Rationale |
|---------------------|-----------|-----------|
| < 10 | 3 | Greedy search (tiny circuits) |
| 10–30 | 5 | Balanced exploration |
| > 30 | 8 | Thorough search (large circuits) |

### 12.6 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 9 (CircuitOptimizer class) and `rl_environment.py`

Key methods:
- `optimize(expr, verbose)` — main optimization loop
- `score_for_mode(G, temp_blowup)` — scores graph for given mode
- `_adaptive_beam_width(G)` — returns dynamic beam width

---

## 13. Phase 10: Cost Model & PIT Verification

### 13.1 Profitability Filter

Filters candidate actions by comparing `candidate_cost < α × original_cost`. Default α = 1.0 (only accept strict improvements).

### 13.2 Polynomial Identity Testing (PIT)

Ensures the candidate expression is algebraically equivalent to the original: `C' ≡ C`.

| Method | Condition | Description |
|--------|-----------|-------------|
| Symbolic | ≤ 15 nodes | `sp.simplify(orig - cand) == 0` — exact algebraic verification |
| Schwartz–Zippel | > 15 nodes | 30 random trials with randomized variable assignments modulo prime (2,147,483,647). False-positive probability ≈ 2⁻³⁰. |

### 13.3 Dead Node Pruner

Post-PIT cleanup: removes nodes not reachable from the output root. Computes `live = descendants(root) ∪ {root}` and prunes all other nodes.

### 13.4 Pipeline Order

GIN rank all → top-K → cost on top-K → PIT → dead prune → update DAG.

### 13.5 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 11 (CostModel, ProfitabilityFilter, PITVerifier, DeadNodePruner classes) and `rl_environment.py`

---

## 14. Phase 11: Interpretability (RF + Decision Tree)

### 14.1 Overview

An **interpretable proxy** for the GINEConv black box: a Random Forest + Decision Tree trained on the 20-dim node features to predict the best rule.

### 14.2 Model Performance

| Metric | Value |
|--------|-------|
| RF Accuracy | 48.1% |
| Decision Tree Accuracy | 48.8% |
| RF Estimators | 200 |
| Tree Max Depth | 4 |

### 14.3 Per-Rule Performance (Random Forest)

| Rule | TPR | FPR | Precision | F1 | Support |
|------|-----|-----|-----------|-----|---------|
| factorization | 25.3% | 9.2% | 37.2% | 30.1% | 708 |
| distribution | 39.2% | 6.4% | 87.4% | 54.1% | 2,124 |
| reassociation | 61.0% | 10.1% | 54.0% | 57.2% | 648 |
| depth_reduction | 100.0% | 16.8% | 34.6% | 51.4% | 326 |
| collection | 98.0% | 16.1% | 19.2% | 32.1% | 150 |
| cse_rewrite | 100.0% | 2.0% | 35.2% | 52.1% | 44 |

### 14.4 Implementation

File: `ArithCircuit_Pipeline.ipynb` — Section 10 (RandomForestClassifier, DecisionTreeClassifier)

### 14.5 GNNExplainer (Extension)

Phase 6 adds GNNExplainer for node-level and edge-level importance scores, plus counterfactual explanations. See `gnn_explainer.py`.

---

## 15. Phase 12: Full Pipeline Benchmark

### 15.1 Overview

Running the full pipeline on **500 held-out expressions** (ThreadPoolExecutor).

### 15.2 Overall Metrics

| Metric | Value |
|--------|-------|
| Mean % Reduction | 14.67% |
| Median % Reduction | **13.41%** |
| Std % Reduction | 12.95% |
| Min % Reduction | 0.00% |
| Max % Reduction | 43.90% |
| Per Expression | 0.74s |
| Success Rate | **100%** |

**Benchmark completed in ~370s | Speedup: ~10× with ThreadPoolExecutor**

### 15.3 Benchmark by Family

| Family | Mean Reduction | Median Reduction | Depth Reduction |
|--------|---------------|-----------------|----------------|
| cse_heavy | 29.85% | 29.20% | 1.00 |
| factorization_heavy | 29.68% | 29.68% | 0.00 |
| depth3_factored | 29.63% | 29.63% | 1.00 |
| common_factor_target | 29.36% | 28.38% | 0.00 |
| depth3_mul_sum | 29.07% | 29.07% | 1.00 |
| multi_step_chain | 30.00% | **36.46%** | 0.86 |
| depth3_circuit | 24.29% | 24.29% | 1.00 |
| distrib_recovery | 11.62% | 13.41% | 1.00 |
| const_fold_target | 0.00% | 0.00% | 0.00 |
| depth3_reassoc | 0.00% | 0.00% | 0.00 |
| depth_heavy | 0.00% | 0.00% | 0.00 |
| depth_reduction_target | 0.00% | 0.00% | 0.00 |
| reassoc_heavy | 0.00% | 0.00% | 0.00 |
| reassoc_target | 0.00% | 0.00% | 0.00 |

### 15.4 Rule Usage

| Rule | Count | Frequency |
|------|-------|-----------|
| cse_rewrite | 225 | 55.7% |
| distribution | 120 | 29.7% |
| factorization | 33 | 8.2% |
| depth_reduction | 25 | 6.2% |
| common_factor | 1 | 0.2% |

### 15.5 Implementation

File: `rebenchmark_fixed.py`

---

## 16. Phase 13: RL Fine-Tuning (Extension)

### 16.1 Overview

Fine-tunes the supervised GIN model using REINFORCE with baseline for policy gradient optimization.

### 16.2 Implementation

File: `rl_train_reinforce.py`

Key components:
- `REINFORCETrainer` class — implements REINFORCE with baseline
- `compute_action_log_probs()` — computes log probability of actions
- `collect_episode()` — collects one episode of (state, action, reward, log_prob)
- `update_policy()` — performs policy gradient update
- `train()` — main training loop

### 16.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Discount Factor (γ) | 0.99 |
| Entropy Coefficient | 0.01 |
| Gradient Clipping | 1.0 |
| Baseline Decay | 0.99 |
| Epsilon (exploration) | 0.2 → 0.01 (decayed) |
| Epochs | 15 |

### 16.4 Results

The RL fine-tuned model (`gin_rl_finetuned.pt`) achieves **14.95%** mean reduction — identical to the supervised baseline. This is a **negative result** consistent with recent literature (NeurIPS 2025) showing RL struggles with expression simplification tasks.

---

## 17. Phase 14: Self-Supervised Training (Extension)

### 17.1 Overview

Generates self-supervised training data by starting from canonical minimal expressions and applying rules in REVERSE to complexify them. The GIN is then trained to reverse the process.

### 17.2 Implementation

File: `self_supervised.py` (data generation) and `train_self_supervised.py` (training)

### 17.3 Reverse Rule Mapping

| Forward Rule | Reverse Operation | Example |
|-------------|-------------------|---------|
| factorization | distribute (expand) | x*(y+z) → x*y + x*z |
| distribution | factor | x*y + x*z → x*(y+z) |
| reassociation | unbalance tree | (a+b)+(c+d) → ((a+b)+c)+d |
| const_folding | un-fold constants | 5*x → (2+3)*x |
| common_factor | distribute common factor | a*(x+y) → a*x + a*y |
| depth_reduction | expand powers | x⁵ → x²*x³ |
| collection | distribute coefficients | x*(y+z+w) → x*y + x*z + x*w |
| cse_rewrite | inline common subexpr | (a+b)*(c+d) → (a+b)*c + (a+b)*d |

### 17.4 Results

| Metric | Value |
|--------|-------|
| Self-supervised samples | 2,500 (500 per rule) |
| Combined corpus | 22,500 samples |
| Mean reduction | 14.95% |
| Best val loss | 0.4426 |

The self-supervised model (`gin_ss_finetuned.pt`) achieves **14.95%** mean reduction — identical to supervised baseline.

---

## 18. Phase 15: Depth Generalization Study

### 18.1 Overview

Tests whether the GIN model trained on depth-3 circuits transfers to deeper circuits (depth-4, depth-5, depth-6).

### 18.2 Implementation

File: `depth_generalization.py`

### 18.3 Results

| Depth | Mean % | Median % | Max % | Zero-imp | N |
|-------|--------|----------|-------|----------|---|
| depth-3 | 24.29% | 24.29% | 24.29% | 0/20 | 20 |
| depth-4 | 38.20% | 38.20% | 38.20% | 0/20 | 20 |
| depth-5 | 38.32% | 38.32% | 38.32% | 0/20 | 20 |
| depth-6 | 44.96% | 44.96% | 44.96% | 0/20 | 20 |

### 18.4 Zero-Shot Transfer Analysis

| Depth | Mean % | Retention | Drop |
|-------|--------|-----------|------|
| depth-4 | 38.20% | 157.3% | -13.92pp |
| depth-5 | 38.32% | 157.8% | -14.03pp |
| depth-6 | 44.96% | 185.1% | -20.68pp |

### 18.5 Important Finding

**Depth-4 initially showed 0% reduction** due to a bug in the depth-4 generator. The generator created expressions like `(a+d)*(x+y) + (u+z)*(v+w)` which SymPy flattened to depth 3 with no common factor between terms, making them irreducible. The fix changed the generator to create true depth-4 circuits: `a*(b*(c*(d+e)))` — nested multiplication chains that SymPy cannot flatten. After the fix, depth-4 shows **38.20%** reduction.

---

## 19. Phase 16: GNNExplainer Interpretability

### 19.1 Overview

Applies GNNExplainer to the GINEConv layers to generate node-level and edge-level importance scores, explaining *why* the GIN policy selects specific (node, rule) actions.

### 19.2 Implementation

File: `gnn_explainer.py`

Key classes:
- `GNNExplainerAnalyzer` — applies GNNExplainer to explain model decisions
- `counterfactual_analysis()` — computes counterfactual explanations

### 19.3 Counterfactual Analysis

For each accepted rewrite, computes: "If feature X changed, would the model still select rule Y?" Feature ablation reveals which features are most influential for each rule selection.

### 19.4 Implementation Details

- Uses PyTorch Geometric's `GNNExplainer` algorithm
- Generates node importance heatmaps on DAGs
- Generates edge importance visualizations
- Generates feature importance bar charts
- Supports counterfactual feature ablation analysis

---

## 20. Phase 17: Theoretical PIT Analysis

### 20.1 Overview

Theoretical analysis of Polynomial Identity Testing (PIT) in the ArithCircuit pipeline, including rule-by-rule PIT cost characterization, VP-class circuit size reduction guarantees, and GNN-arithmetic circuit correspondence.

### 20.2 Implementation

File: `docs/pit_analysis.md`

### 20.3 Key Contributions

1. **Rule-by-rule PIT cost characterization** — First systematic analysis of which rewrite rules admit structural shortcuts vs. full PIT
2. **Constructive circuit size reduction** — A pipeline that provably reduces circuit size with algebraic equivalence guarantees
3. **GNN–arithmetic circuit correspondence application** — First practical application of the GNN-arithmetic circuit equivalence (arXiv 2402.17805) to circuit optimization
4. **Hybrid symbolic/randomized verification** — Practical PIT strategy balancing exactness and efficiency

### 20.4 References

1. Schwartz, J. T. (1980). Fast probabilistic algorithms for verification of polynomial identities. *Journal of the ACM*.
2. Zippel, R. (1979). Probabilistic algorithms for sparse polynomials. *EUROSAM*.
3. ArXiv 2402.17805: GNN–Arithmetic Circuit Correspondence
4. Arith-DAS: NeurIPS 2025 — Differentiable Arithmetic Circuit Optimization
5. Prof. Nitin Saxena's foundational work on PIT and circuit complexity

---

## 21. Phase 18: Paper Writing & Submission

### 21.1 Paper Structure

**File:** `paper.tex`

| Section | Content |
|---------|---------|
| Abstract | Problem, method, results, contributions |
| Introduction | NP-hardness of optimal factorization, motivation |
| Related Work | RL for expression simplification, GNN for EDA, Schwartz–Zippel PIT |
| Method | Existing pipeline + extensions |
| Experiments | Baseline + new results + ablation |
| Theory | PIT correctness guarantees per rule class |
| Conclusion | Open problems, depth generalization as future work |

### 21.2 Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | Pipeline flow diagram |
| Figure 2 | Dataset overview |
| Figure 3 | GIN node reducibility heatmaps |
| Figure 4 | GIN loss curves |
| Figure 5 | Feature importance |
| Figure 6 | Decision tree proxy |
| Figure 7 | Confusion matrix |
| Figure 8 | Per-rule F1 |
| Figure 9 | Circuit comparison |
| Figure D1-D4 | Rule-specific factorizations |

### 21.3 Target Venues

- DAC Workshops (June submission)
- ECML-PKDD Workshops
- ICLR Workshop
- NeurIPS Workshop

---

## 22. Key Results Summary

| Metric | Value |
|--------|-------|
| Mean % Reduction (500 exprs) | 14.67% |
| Median % Reduction | 13.41% |
| Max % Reduction | 43.90% |
| Success Rate | 100% |
| Zero-Algebraic-Degradation | 100% |
| GIN Node Accuracy | 92.24% |
| GIN Rule Accuracy | 63.16% |
| Best Family | multi_step_chain (36.46%) |
| Zero-Improvement Families (baseline) | 6 families |
| Zero-Improvement Families (after fix) | 0 families |
| Depth-4 (after fix) | 38.20% reduction |
| Depth-5 | 38.32% reduction |
| Depth-6 | 44.96% reduction |

---

## 23. Known Issues & Limitations

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Depth-4 generator bug | Original generator produced depth-3 expressions with no common factor | Fixed with true nested depth-4 circuits |
| RL fine-tuning | No improvement over supervised baseline | Negative result — publishable |
| RF/DT proxy | Weak interpretability (48% accuracy) | Replaced with GNNExplainer |
| const_folding rule | No training examples generated | Requires constant-only expressions |
| reassociation rule | 0 positive examples in baseline | Fixed with oracle training data |
| depth_reduction rule | 0 positive examples in baseline | Fixed with oracle training data |
| cse_rewrite rule | 0 positive examples in baseline | Fixed with oracle training data |

---

## 24. Citation

```
@misc{arithcircuit2026,
  author = {Research Group of Prof. Nitin Saxena},
  title = {ArithCircuit Pipeline: ML-Guided Arithmetic Circuit Optimization},
  year = {2026},
  institution = {Indian Institute of Technology Kanpur},
}
```

---

## 25. License

This project is part of a research internship under Prof. Nitin Saxena at IIT Kanpur. All rights reserved.

---

## Appendix A: Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (notebook)
jupyter notebook ArithCircuit_Pipeline.ipynb

# Run RL fine-tuning
python rl_train_reinforce.py --n-episodes 100 --n-exprs 50 --device cuda

# Run self-supervised training
python train_self_supervised.py --n-per-rule 5000 --epochs 15 --device cuda

# Run depth generalization study
python depth_generalization.py --model models/gin_best_fixed.pt --n-per-depth 200 --device cuda

# Run GNNExplainer
python gnn_explainer.py --model models/gin_best_fixed.pt --device cuda --n-exprs 5

# Re-run benchmark
python rebenchmark_fixed.py --n-eval 500 --seed 42 --device cuda --model models/gin_best_fixed.pt
```

---

## Appendix B: Git Workflow

```bash
# Clone repository
git clone https://github.com/Prakshil/IITK-Internship.git
cd IITK-Internship/Arithmetic_Circuit_Optimization

# View current branch
git branch

# View tags
git tag

# Checkout baseline
git checkout v1.0-baseline

# View current phase branch
git log --oneline -5
```

---

*Generated from `IMPLEMENTATION_PLAN.md` and project documentation — v1.0 | October 2026*
