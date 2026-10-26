# Diagnosis: Zero-Improvement Rules (reassociation, depth_reduction, cse_rewrite)

## Problem

Three of eight trainable rules have **0 positive training examples** in the generated PyG corpus:

| Rule | Total Examples | Positive Examples | % Positive | Status |
|------|---------------|-------------------|------------|--------|
| reassociation | 106,620 | 0 | 0.0% | SPARSE |
| depth_reduction | 69,968 | 0 | 0.0% | SPARSE |
| cse_rewrite | 69,968 | 0 | 0.0% | SPARSE |

This means the GINEConv rule head has **never learned** when to apply these rules — they are effectively dead during inference.

## Root Cause

Two compounding issues in the supervision pipeline:

### Issue 1: `label_candidates()` — Lookahead Only for Top-5 (Line 2386)

```python
if immediate_norm > 0:
    if is_opt or i < TOP_LOOKAHEAD:  # TOP_LOOKAHEAD = 5
        _, best_future_cost, _ = self.lookahead_score(...)
        future_norm = delta_norm(old_score, best_future_cost)
    else:
        future_norm = immediate_norm
else:
    future_norm = 0.0
```

- Reassociation and depth_reduction typically produce `immediate_gain = 0` (no immediate cost reduction).
- These candidates rank low after sorting by `immediate_gain`.
- Only top-5 candidates get `lookahead_score()`; the rest get `future_norm = immediate_norm = 0`.
- Result: `imp_norm = 0.5 * 0 + 0.5 * 0 = 0` for reassociation/depth_reduction candidates.

### Issue 2: `annotate_supervision()` — Only Best Node Gets Positive Label (Line 2434)

```python
if nid == best_nid and imp >= best_gain - 1e-6 and best_gain > 0:
    node_red = 1.0
elif imp >= best_gain * 0.9 and best_gain > 0:
    node_red = imp / best_gain
else:
    node_red = 0.0
```

- `best_gain` comes from a rule that *does* produce immediate gain (typically factorization, distribution, or common_factor).
- Reassociation/depth_reduction have `imp = 0`, so they fall into the `else` branch → `node_red = 0.0`.
- The BCE loss on `node_reducible` therefore never sees a positive example for these rules.

### Issue 3: Rule Implementation May Not Fire (Secondary)

- `_reassociate()` only triggers when `len(args) >= 3` AND the balanced result differs structurally.
- `_depth_reduce()` applies `powsimp`/`collect`/`radsimp` and requires `count_ops(result) < count_ops(e)`.
- Many depth-3 expressions don't meet these thresholds, so no candidate is even generated.

## Impact

- GIN rule head accuracy for these rules is undefined (no positive signal).
- During beam search optimization, GIN never ranks reassociation/depth_reduction candidates highly.
- Expression families that rely on these rules (`reassoc_heavy`, `depth_heavy`, `depth_reduction_target`, etc.) show **0% improvement**.

## Fix (Step 1.2)

Add **oracle training data** that explicitly generates expressions where reassociation and depth_reduction provably reduce cost, independent of the random-expression + lookahead pipeline:

1. **Oracle reassociation**: Generate left-leaning Add/Mul trees (e.g., `((a+b)+c)+d`) and label the balanced re-association as a positive action with `node_reducible = 1.0`.
2. **Oracle depth_reduction**: Generate expressions where `powsimp`/`collect` provably reduces circuit depth, label as positive.
3. Ensure minimum **100 positive examples per rule** (increase from `RULE_MIN_POSITIVE = 50`).

This guarantees these rules have positive training signal regardless of whether they emerge from random data generation.
