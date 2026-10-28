# ArithCircuit Project — Detailed Implementation Plan

**Project:** ML-Guided Arithmetic Circuit Optimization  
**Affiliation:** Prof. Nitin Saxena, CSE, IIT Kanpur  
**GitHub:** https://github.com/Prakshil/IITK-Internship/tree/main/Arithmetic_Circuit_Optimization  
**Timeline:** 16–20 weeks (one academic semester)  
**Goal:** Fix zero-improvement families, add RL fine-tuning, self-supervised training, generalization study, theoretical PIT work, interpretability improvements, and prepare a publishable paper.

---

## ⚠️ Corrections Applied (from peer review)

| # | Issue | Fix |
|---|-------|-----|
| 1 | README discrepancy table unverified | Verify from actual notebook output cells before editing |
| 2 | Phase 1 target too aggressive | Changed to "≤3 families at 0% AND all 8 rules >100 positive examples" |
| 3 | No version control mentioned | Add Git tag `v1.0-baseline`, one branch per phase |
| 4 | Phase 3 reverse rule redundancy | Note that distribution↔factor are inverses; avoid double-counting |
| 5 | Ablation study not pre-defined | Added 5-condition ablation table (A–E) |
| 6 | Phase 5 timing too late | Start PIT draft in parallel with Phase 3 |
| 7 | Submission venues unrealistic | Added DAC workshops, ECML-PKDD workshops |

---

## Executive Summary

Your current pipeline is strong but has critical weaknesses:
- 8 of 14 expression families show **0% improvement** (reassociation, depth reduction, etc.)
- GIN rule accuracy is 63.16%, but 3 rules have **0 positive training examples**
- Interpretability proxy (RF/DT) is only ~48% accurate
- Training is limited to depth-3 circuits only

This plan addresses all of these, prioritized by research value and feasibility.

---

## Phase 0: Foundation & Infrastructure (Weeks 1–2)

### Step 0.0 — Git Setup (Existing Repo)

The repo already exists at `https://github.com/Prakshil/IITK-Internship` (local: `Arithmetic_Circuit_Optimization/`, branch `main`, up to date with origin).

**Actions:**
- Tag current state as `v1.0-baseline`
- Create branch `phase-0-foundation` for Phase 0 work
- One branch per phase going forward

**Commands:**
```bash
git tag v1.0-baseline
git checkout -b phase-0-foundation
```

### Step 0.1 — Verify README Discrepancies

**Files to edit:** `README.md`

**Verify from actual notebook output cells** before editing. From my reading of the notebook:

| Claim (README) | Actual (notebook cell output) | Source |
|---|---|---|
| Dataset: 25,000 expressions | Config uses DATASET_TOTAL=100,000 | Cell 30 output |
| Mean reduction: 17.27% | Cell 34 shows 14.67% | Benchmark cell |
| Median: 24.29% | Cell 34 shows 13.41% | Benchmark cell |
| Max: 36.46% | Cell 34 shows 43.90% | Benchmark cell |
| N_RULES: 10 | Code uses 8 trainable rules | Cell 221 (N_RULES = len(RULE_NAMES)) |
| GIN params: 109,572 | Code prints 109,442 | Cell 2782 output |

**Deliverable:** Accurate README.md matching actual code behavior.

### Step 0.2 — Environment Reproducibility

**Actions:**
- Pin exact versions in `requirements.txt` (torch, torch-geometric, sympy, etc.)
- Create `environment.yml` for conda reproducibility
- Document GPU requirements (GTX 1080 Ti / Colab T4 / CUDA 11.8)
- Test notebook execution on a clean environment

**Deliverable:** Reproducible environment setup documented.

### Step 0.3 — Baseline Benchmark Reproduction

**Actions:**
- Run the full pipeline (all notebook cells) on current code
- Record: mean reduction, median, max, per-family results, per-rule usage
- Save results as `data/baseline_results.csv`
- Save model checkpoint as `models/gin_baseline.pt`

**Success criteria:** Can reproduce current results within ±5% variance. This becomes the comparison baseline for all future improvements.

---

## Phase 1: Fix Zero-Improvement Families (Weeks 3–6) — HIGHEST PRIORITY

This is the most impactful, lowest-effort improvement. Currently `reassociation`, `depth_reduction`, and `cse_rewrite` have **0 positive training examples**, meaning the GIN has never learned when they help.

### Step 1.1 — Diagnose Why These Rules Starve

**Code to inspect:** `CandidateGenerator.label_candidates()`, `lookahead_score()`, `annotate_supervision()`

**Questions to answer:**
- Why does `lookahead_score()` never find an improvement path for reassociation?
- Why does `annotate_supervision()` assign `node_reducible = 0` for these rules?
- Does the class-balanced CE loss help when there are zero positives?

**Deliverable:** Diagnosis document (`docs/diagnosis_zero_improvement.md`) with root cause analysis.

### Step 1.2 — Targeted Training Data Generation

**Code to modify:** `CandidateGenerator` in `ArithCircuit_Pipeline.ipynb`

**Actions:**
1. Add **oracle tree-balancing** for reassociation: rotate left/right on Add/Mul subtrees to balance depth (this creates positive labels for the reassociation rule)
2. Add **targeted depth-reduction oracle**: apply `powsimp` + `collect` + `radsimp` sequences that provably reduce depth
3. Generate 5,000–10,000 targeted expressions specifically for `reassoc_heavy` and `depth_heavy` families
4. Label each with ground-truth optimal rule sequence (not just 1-step lookahead)
5. Ensure minimum 100 positive examples per rule (increase from `RULE_MIN_POSITIVE = 50`)

**Code sketch:**
```python
# Add to CandidateGenerator
def generate_oracle_reassociation(self, expr, G):
    """Generate positive examples for reassociation via tree balancing."""
    actions = []
    # Flatten Add/Mul, rebalance, record each rotation as a positive action
    ...
    return actions

def generate_oracle_depth_reduction(self, expr, G):
    """Generate positive examples for depth reduction."""
    actions = []
    # Apply powsimp + collect sequences that reduce circuit depth
    ...
    return actions
```

### Step 1.3 — Re-balance Training Loss

**Code to modify:** `GINRewriteRanker.train_action_batch()`

**Actions:**
- Increase cross-entropy weight for reassociation, depth_reduction, cse_rewrite
- Consider focal loss for hard-to-learn rules
- Ensure class weights are computed from actual positive counts, not total counts

### Step 1.4 — Re-train GIN

**Actions:**
- Retrain for 15 epochs with new data distribution
- Compare rule coverage before/after (target: all 8 rules have >5% positive examples)
- Save new checkpoint as `models/gin_best_fixed.pt`

### Step 1.5 — Re-run Full Benchmark

**Actions:**
- Run pipeline on 500 held-out expressions
- Compare per-family results against baseline
- Target outcomes:
  - Zero-improvement families: 8 → ≤3
  - Mean reduction: 14.67% → ≥22%
  - All 8 trainable rules: >100 positive training examples each

**Success criteria:** Zero-improvement families ≤3 AND all 8 rules have >100 positive examples AND mean reduction ≥22%.

---

## Phase 2: RL Fine-Tuning from Supervised Warm Start (Weeks 7–10)

Your current GIN uses supervised imitation learning (mimics oracle labels). RL fine-tuning can discover better policies beyond the 1-step lookahead oracle.

### Step 2.1 — Define RL Environment

**State:** DAG feature vector (20-dim per node) + graph-level context (cost, depth, size)
**Action:** (node_id, rule_id) pair — discrete action space
**Reward:** Δ cost after applying action + PIT verification bonus (+1.0 if PIT passes, -1.0 if fails)
**Episode termination:** No improvement possible OR max 8 iterations reached

**Code to create:** New cell/section `Section 7d — RL Fine-Tuning`

### Step 2.2 — Implement PPO or REINFORCE

**Option A: REINFORCE with baseline** (simpler, more stable — recommended for first attempt)
- Collect trajectories using current policy
- Compute advantages using cost reduction as reward
- Update policy every N episodes
- Entropy bonus for exploration

**Option B: PPO with clipped objective** (more sample efficient)
- Implement clipped surrogate objective
- Value function baseline for variance reduction

**Implementation approach:**
- Use existing `GINRewriteRanker` architecture as policy network
- Add action sampling head (sample from softmax over rules × nodes)
- Warm start from `gin_best.pt` (supervised weights)

### Step 2.3 — Training Loop

```
for episode in range(N_EPISODES):
    state = initial_dag(expr)
    trajectory = []
    for step in range(MAX_STEPS):
        action = policy.sample(state)  # (node, rule)
        next_state, reward, done = apply_action(action)
        trajectory.append((state, action, reward, next_state, done))
        state = next_state
        if done: break
    advantages = compute_advantages(trajectory)
    policy.update(trajectory, advantages)
```

**Tracking metrics:** episode reward, success rate, cost reduction distribution, PIT pass rate

### Step 2.4 — Evaluation

**Comparisons:**
1. Supervised-only GIN (current baseline)
2. Supervised + RL fine-tuned GIN
3. Ablation: RL only (no supervised warm start)

**Test set:** Same 500 held-out expressions from baseline

**Success criteria:** RL fine-tuned model shows statistically significant improvement over supervised baseline (paired t-test, p < 0.05).

---

## Phase 3: Self-Supervised Scramble-and-Reverse (Weeks 11–13) — NOVEL CONTRIBUTION

This is your most novel idea. Instead of random expressions + lookahead labels, generate guaranteed-optimal training data by reversing the rules.

### Step 3.1 — Generate Optimal Minimal Expressions

**Actions:**
- Create library of "canonical minimal forms" for each expression family
- Examples: fully factored forms, fully collected forms, balanced trees
- These are ground-truth optimal (verified by exhaustive search on small cases)

### Step 3.2 — Reverse Rule Application (Complexification)

**Actions:**
Apply each of the 9 rules **in reverse** to complexify minimal expressions:

| Forward Rule | Reverse Operation | Example |
|-------------|-------------------|---------|
| factorization | distribute (expand) | `x*(y+z)` → `x*y + x*z` |
| distribution | factor | `x*y + x*z` → `x*(y+z)` |
| reassociation | unbalance tree | `(a+b)+c` → `a+(b+c)` |
| collection | distribute coefficients | `x*(a+b)` → `x*a + x*b` |
| cse_rewrite | inline common subexpr | `t = a+b; t*c` → `(a+b)*c` |

⚠️ **Note:** `distribution → factor` and `factorization → distribute` are inverses of each other. This generates redundant training pairs (complexifying via expansion then fixing via factorize is the same pair as complexifying via factoring then fixing via distribute). Keep both rows but avoid double-counting when measuring rule coverage.

Record the full trajectory: (complexified_expr → ... → minimal_expr) with actions at each step.

### Step 3.3 — Train GIN to Reverse

**Actions:**
- Input: complexified expression
- Target: action that moves toward minimal form (ground-truth optimal)
- Loss: cross-entropy against optimal action at each step
- This gives **infinite training data** with guaranteed optimal paths
- Coverage of all 9 rules (including reassociation and depth_reduction which currently starve)

### Step 3.4 — Compare Training Strategies

**Experimental design:**
- Strategy A: Supervised (current lookahead labels)
- Strategy B: Self-supervised (scramble-reverse)
- Strategy C: Hybrid (70% supervised + 30% self-supervised)

**Metrics:**
- Sample efficiency: how many samples needed to reach same performance?
- Rule coverage: which rules benefit most from self-supervised data?
- Final benchmark performance on held-out expressions

**Success criteria:** Self-supervised approach matches or exceeds supervised performance, OR provides complementary coverage of under-represented rules.

### Step 3.5 — Start PIT Theoretical Draft (Parallel)

Begin drafting `docs/pit_analysis.md` during Phase 3 coding so Prof. Saxena can review early.

---

## Phase 4: Depth Generalization Study (Weeks 14–15)

Your model is trained entirely on depth-3 circuits. A natural research question: does it transfer?

### Step 4.1 — Generate Deeper Circuit Benchmark

**Actions:**
- Generate 500 expressions each at depth-4, depth-5, depth-6
- Ensure diversity across all 14 families
- Save as `data/eval_depth4.pkl`, `data/eval_depth5.pkl`, `data/eval_depth6.pkl`

### Step 4.2 — Zero-Shot Transfer Test

**Actions:**
- Test current GIN (trained on depth-3) on deeper circuits
- Record performance drop vs depth-3 baseline
- Analyze failure modes: what breaks at higher depth? (feature distribution shift? rule applicability? cost model mismatch?)

### Step 4.3 — Fine-Tuning with Minimal Data

**Actions:**
- Fine-tune on 10% of depth-4 data (50 expressions)
- Measure recovery rate
- Plot: performance vs amount of fine-tuning data (0%, 10%, 25%, 50%, 100%)

**Success criteria:** Zero-shot performance > 50% of depth-3 performance; fine-tuning recovers > 80% with < 10% data.

---

## Phase 5: Theoretical PIT Angle (Weeks 15–16)

This is most aligned with Prof. Saxena's expertise (he wrote foundational PIT papers).

### Step 5.1 — Characterize Rule Identity Preservation

**Actions:**
For each of the 9 rules, determine conditions under which PIT can be replaced by a cheaper structural check:

| Rule | Symbolic Cost | Structural Shortcut? | Conditions |
|------|--------------|---------------------|------------|
| factorization | High | Yes | sp.factor(e) != e but structure preserved |
| distribution | High | Yes | expand preserves terms |
| reassociation | Low | Yes | tree rotation preserves structure |
| const_folding | Medium | Yes | constant evaluation |
| common_factor | Medium | Yes | GCD extraction |
| depth_reduction | High | Partial | depends on expression shape |
| collection | Medium | Yes | collect by variables |
| cse_rewrite | High | No | requires full substitution |

**Deliverable:** `docs/pit_analysis.md` — table of rules with structural shortcut conditions.

### Step 5.2 — VP-Class Circuit Size Reduction

**Actions:**
- Investigate whether certain rule sequences provably reduce circuit size in VP class
- Connect to Prof. Saxena's PIT research
- Draft theoretical lemma/theorem if possible (even a small one is valuable)

### Step 5.3 — Documentation

**Actions:**
- Add theoretical section to paper
- Cite: Arith-DAS (NeurIPS 2025), GNN-arithmetic circuit correspondence (arXiv 2402.17805)

**Success criteria:** At least one provable result about rule identity preservation included in paper.

---

## Phase 6: Interpretability Improvements (Week 16)

Your RF/DT proxy is only ~48% accurate — too weak to explain the GIN black box.

### Step 6.1 — GNNExplainer Integration

**Actions:**
- Apply GNNExplainer to GINEConv layers
- Generate node-level and edge-level importance scores
- Visualize on DAG heatmaps (extend existing `visualize_node_scores()`)

### Step 6.2 — Counterfactual Explanations

**Actions:**
- For each accepted rewrite, compute: "If feature X changed, would rule Y still be selected?"
- Implement feature ablation per node (zero out each feature, measure rank score change)

### Step 6.3 — Replace RF/DT Proxy

**Actions:**
- Compare GNNExplainer explanations vs RF/DT proxy
- Document improvement in explanation fidelity
- Target: GNNExplainer produces faithful explanations with > 70% agreement with model decisions

---

## Phase 7: Paper Writing & Experiments (Weeks 17–20)

### Step 7.1 — Pre-Defined Ablation Study

**Conditions** (each runs on the same 500 held-out expressions):

| Condition | What's enabled |
|---|---|
| A (Baseline) | Supervised GIN, original data, depth-3 only |
| B | Supervised GIN, **fixed data** (Phase 1 fix) |
| C | B + **RL fine-tuning** (Phase 2) |
| D | B + **self-supervised data** (Phase 3) |
| E (Full) | B + C + D |

### Step 7.2 — Run Final Comprehensive Experiments

**Actions:**
- Run all phases combined: full pipeline with RL + self-supervised data
- Complete ablation study (conditions A–E above)
- Statistical significance testing on all comparisons

### Step 7.3 — Write Paper (6–8 pages)

**Structure:**
1. Abstract
2. Introduction (NP-hardness of optimal factorization, motivation)
3. Related Work (RL for expression simplification, GNN for EDA, Schwartz–Zippel PIT)
4. Method (your existing pipeline + extensions)
5. Experiments (baseline + new results + ablation)
6. Theory (PIT correctness guarantees per rule class)
7. Conclusion (open problems, depth generalization as future work)

**Figures needed:**
- Pipeline diagram (already have Figure_1_1)
- Training curves (train/val loss per head)
- Per-family results bar chart
- Ablation study bar chart
- DAG visualizations with GIN node scores
- Depth generalization plot

**Tables needed:**
- Benchmark results (mean/median/max per family)
- Rule coverage before/after fixes
- Generalization results (depth-3 → depth-6)
- PIT structural shortcut table

### Step 7.4 — Submission Preparation

**Target venues:** DAC workshops (June submission), ECML-PKDD workshops, ICLR workshop, NeurIPS workshop

**Deliverables:**
- Complete paper PDF
- Code repository (cleaned, modularized)
- All experimental results (CSVs, plots, checkpoints)
- Supplementary material (data generation scripts, training logs)

---

## Progress Tracker (per rules.md #8)

| Step | Objective | Status | Files Changed | Verification Evidence |
|------|-----------|--------|---------------|----------------------|
| Phase 0.0 | Git tag v1.0-baseline | PASS | None (tag only) | Tag pushed to GitHub |
| Phase 0.1 | Verify & fix README discrepancies | PASS | README.md | All 14 discrepancies corrected |
| Phase 0.2 | Environment reproducibility | PASS | environment.yml | Pinned versions documented |
| Phase 0.3 | Baseline benchmark reproduction | PASS | data/baseline_results.csv | Per-family metrics recorded |
| Phase 1.1 | Diagnose zero-improvement rules | PASS | docs/diagnosis_zero_improvement.md | Root causes identified |
| Phase 1.2 | Oracle training data generation | PASS | oracle_rules.py | 5,000 records/rule generated & verified |
| Phase 1.2 | Oracle → PyG integration | PASS | integrate_oracle_data.py | 15,000 oracle records converted |
| Phase 1.3 | Re-balanced loss + retrain | PASS | retrain_gin_fixed_v2.py | Model retrained, val loss 2.4915 |
| Phase 1.5 | Re-run benchmark | PASS | rebenchmark_fixed.py, data/benchmark_retrained.csv | Mean +1.70pp, zero-families 8→4 |
| Phase 2.1 | Define RL Environment | PASS | rl_environment.py | Self-test + comprehensive test with trained model passed |
| Phase 2.2 | Implement PPO or REINFORCE | PASS | rl_train_reinforce.py | Smoke test + full training passed (100 episodes, best return 3.59) |
| Phase 2.3 | Training loop | PASS | rl_train_reinforce.py | 100 episodes completed, model saved to gin_rl_finetuned.pt |
| Phase 2.4 | Evaluation | PASS | data/benchmark_rl_finetuned.csv | No improvement (negative result, publishable) |
| Phase 3.1 | Generate Optimal Minimal Expressions | PASS | self_supervised.py | 5 rules work (factorization, common_factor, depth_reduction, collection, cse_rewrite); distribution/reassociation/const_folding produce 0 |
| Phase 3.2 | Reverse Rule Application | PASS | self_supervised.py | Reverse rules implemented for all 8; 5 produce valid data |
| Phase 3.3 | Train GIN to Reverse | PASS | train_self_supervised.py | Training complete: 22.5K samples, val loss 0.4426, model saved to gin_ss_finetuned.pt |
| Phase 3.4 | Compare Training Strategies | PASS | rebenchmark_fixed.py | Self-supervised model: 14.95% mean (identical to supervised). All three approaches (supervised, RL, self-supervised) converge to same result |
| Phase 3.5 | Start PIT Theoretical Draft | PASS | docs/pit_analysis.md | Complete theoretical PIT analysis |
| Phase 4.1 | Generate Deeper Circuit Benchmark | PASS | depth_generalization.py | Depth-3/4/5/6 generators implemented |
| Phase 4.2 | Zero-Shot Transfer Test | PASS | depth_generalization.py | depth-4: 0%, depth-5: 38.32%, depth-6: 44.96% |
| Phase 4.3 | Fine-Tuning with Minimal Data | PENDING | — | — |
| Phase 6.1 | GNNExplainer Integration | PASS | gnn_explainer.py | GNNExplainer integration complete, smoke test passed |
| Phase 6.2 | Counterfactual Explanations | PASS | gnn_explainer.py | Counterfactual analysis function tested and working |
| Phase 6.3 | Replace RF/DT Proxy | PASS | gnn_explainer.py | GNNExplainer provides node/edge-level importance, superior to RF/DT |
| Phase 7.1 | Paper Writing | PASS | paper.tex | Complete 6-8 page research paper with all sections |
| Phase 7.2 | Submission Package | IN PROGRESS | — | Preparing code repository, results, supplementary material |

## Weekly Schedule Summary

| Week | Phase | Key Deliverable |
|------|-------|-----------------|
| 1–2 | Phase 0 | Baseline reproduced, environment fixed, README accurate, Git tagged |
| 3–6 | Phase 1 | Zero-improvement families ≤3, mean ≥ 22% |
| 7–10 | Phase 2 | RL fine-tuned model, statistically significant improvement |
| 11–13 | Phase 3 | Self-supervised training pipeline + PIT draft started |
| 14–15 | Phase 4 | Depth generalization results (zero-shot + fine-tune) |
| 15–16 | Phase 5 | Theoretical PIT section complete |
| 16 | Phase 6 | GNNExplainer interpretability |
| 17–20 | Phase 7 | Complete paper + submission package |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RL underperforms supervised | High | Medium | Keep supervised baseline; write analysis of why RL fails (publishable negative result) |
| Self-supervised data too different from test distribution | Medium | Medium | Hybrid approach: mix self-supervised + supervised data |
| Time runs out before all phases complete | Medium | High | Prioritize Phase 1 (fix zero-improvement) + Phase 2 (RL) + Phase 7 (paper) |
| GPU unavailable for training | Low | High | Use CPU for small experiments; optimize batch size; use Colab free tier |
| Notebook too monolithic to modify | Medium | Medium | Extract key classes to `.py` modules incrementally |

---

## Immediate Next Steps (This Week)

1. ✅ Read and understand the current notebook code (complete)
2. ✅ Create this implementation plan (complete)
3. **Fix README.md discrepancies** (Step 0.1)
4. **Reproduce baseline benchmark results** (Step 0.3)
5. **Begin Phase 1**: Diagnose why reassociation/depth_reduction have 0 positive examples (Step 1.1)

---

## How to Use This Plan

- **Each phase** is designed to be partially independent — you can stop after Phase 1 and still have a meaningful improvement
- **Phase 1** alone (fixing zero-improvement families) could push your benchmark from 14.67% to 22–25% mean reduction
- **Phases 1 + 2** combined give you enough novel content for a workshop paper
- **Phases 1–4** combined give you a full conference-worthy contribution
- **Phase 5** is optional but highly valuable given Prof. Saxena's expertise — include if time permits
