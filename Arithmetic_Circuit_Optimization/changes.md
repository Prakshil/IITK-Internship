Now I have a complete read of every cell. Here's the full status — going from critical to small.

---

## 🔴 Will Crash at Runtime

### 1. `generate_from_config` and `_get_family_builders` don't exist in `DatasetGenerator`

Section 1 does:
```python
_family_names = list(gen._get_family_builders().keys()) if hasattr(gen, '_get_family_builders') else []
# ...
dataset = gen.generate_from_config(config, DEPTH_FILTER=DEPTH_FILTER)
```

Neither method exists. `hasattr` protects the first line so `_family_names = []`, meaning `config.resolve([])` runs and `config.split_quotas = {}`. Then `generate_from_config` throws `AttributeError`. The entire Section 1 dataset path is broken. Add both to `DatasetGenerator`:

```python
def _get_family_builders(self):
    """Return the FAMILY_BUILDERS dict. Defined here so DatasetConfig can resolve quotas before generate() is called."""
    import sympy as sp
    x, y, z, w, u, v, a, b, c, d, e, f = sp.symbols("x y z w u v a b c d e f")
    # Paste all gen_* inner functions here, identical to generate(),
    # then return the FAMILY_BUILDERS dict
    return {
        "factorization_heavy": ...,
        # ... all 14 families
    }

def generate_from_config(self, config, DEPTH_FILTER=None):
    """Generate dataset respecting DatasetConfig per-family quotas."""
    FAMILY_BUILDERS = self._get_family_builders()
    # Resolve quotas if not yet done
    if not config.split_quotas:
        config.resolve(list(FAMILY_BUILDERS.keys()))

    dataset = []
    for fam, builder in FAMILY_BUILDERS.items():
        n_target = config.split_quotas.get(fam, config.total_samples // len(FAMILY_BUILDERS))
        n_overshoot = n_target * (4 if DEPTH_FILTER is not None else 1)
        count, attempts = 0, 0
        while count < n_target and attempts < n_overshoot * 5:
            attempts += 1
            try:
                rec = self._make_rec(builder(self.rng), fam)
                if DEPTH_FILTER is None or rec.get("circuit_depth") == DEPTH_FILTER:
                    dataset.append(rec)
                    count += 1
            except:
                pass
        print(f"  {fam:<28} target={n_target:>5,} got={count:>5,}")

    self.rng.shuffle(dataset)
    return dataset[:config.total_samples]
```

---

### 2. `_make_rec` calls `DAGConverter` which isn't defined until Section 4

`DatasetGenerator._make_rec()` does:
```python
G_tmp = DAGConverter().convert(ExprNormalizer().normalize_for_cost(expr))
```

When Section 1 runs, `DAGConverter` doesn't exist yet. The bare `except: pass` swallows the `NameError` and falls back to `circ_depth = true_depth`, so it doesn't crash, but **all depth filtering silently breaks** — `circuit_depth` always equals `true_depth` (SymPy tree depth), not the actual DAG circuit depth. The `DEPTH_FILTER = 3` filter operates on the wrong metric and produces wrong results.

Fix: run Section 4 before Section 1, OR reorder so DAGConverter is defined early, OR compute circuit depth lazily after Section 4 is defined:

```python
# At the bottom of Section 4, after DAGConverter is defined, add:
if 'dataset' in dir() and dataset:
    print("Backfilling circuit_depth now that DAGConverter is available...")
    _dc = DAGConverter(); _norm = ExprNormalizer()
    for r in tqdm(dataset, desc="circuit_depth"):
        if r.get("circuit_depth") == r.get("true_depth"):   # was a fallback value
            try:
                G = _dc.convert(_norm.normalize_for_cost(r["expr"]))
                r["circuit_depth"] = DAGConverter.circuit_depth(G)
            except:
                pass
    print("Done.")
```

---

### 3. `save_gin_ckpt` silently drops epoch-specific checkpoints

The guard condition:
```python
def save_gin_ckpt(model, optimizer, scheduler, epoch, history, fname="gin_latest.pt"):
    if "epoch" in fname or not fname.endswith(".pt") or not fname.startswith("gin_"):
        return
```

`"epoch" in "gin_epoch_05.pt"` → `True`, so the entire epoch checkpoint is silently dropped despite `GIN_CKPT_EVERY_EPOCH = True`. `gin_latest.pt` saves correctly, `gin_best.pt` saves correctly, but `gin_epoch_05.pt` etc. never save. Fix:

```python
def save_gin_ckpt(model, optimizer, scheduler, epoch, history, fname="gin_latest.pt"):
    # Block only unknown filenames, not epoch-named ones
    valid = (fname == "gin_latest.pt" or
             fname == "gin_best.pt" or
             fname.startswith("gin_epoch_"))
    if not valid:
        return
    path = _ckpt(fname)
    torch.save({...}, path)
    print(f"  GIN checkpoint: {path} (epoch {epoch})")
```

---

### 4. `identity` (rule 0) and `dead_pruning` (rule 6) are in `RULE_NAMES` but generate zero training samples

`RULE_NAMES = ["identity", "factorization", ..., "dead_pruning", ...]`

- `identity` is explicitly `continue`'d in `generate_local_actions` and returns `None` in global, so 0 candidates ever.
- `dead_pruning` is not in `_rule_fns` at all.

`verify_rule_training_coverage()` will flag both as ⚠️ MISSING. More critically, in the training loop:

```python
train_rule_counts = torch.bincount(torch.tensor([int(d.target_rule) for d in train_data]), minlength=N_RULES).float().clamp(min=1)
global_rule_weights = (1.0 / train_rule_counts)
```

Rules 0 and 6 get `count = 1` (from the clamp), giving them a non-zero weight in CE loss. The model is then penalized for predicting rules that literally never appear in data. This corrupts the rule head training.

Fix — remove these from the trainable set:

```python
RULE_NAMES = [
    "factorization", "distribution", "reassociation",
    "const_folding", "common_factor", "depth_reduction",
    "collection", "cse_rewrite",
]
N_RULES = len(RULE_NAMES)  # 8, not 10

# Update _rule_fns indices to be contiguous 0-7:
self._rule_fns = [
    (self._factorize,     0, "factorization"),
    (self._distribute,    1, "distribution"),
    (self._reassociate,   2, "reassociation"),
    (self._const_fold,    3, "const_folding"),
    (self._common_factor, 4, "common_factor"),
    (self._depth_reduce,  5, "depth_reduction"),
    (self._collection,    6, "collection"),
    (self._cse_rewrite,   7, "cse_rewrite"),
]
```

If you want to keep the semantic labels for documentation, use a separate `ALL_RULE_NAMES` list and `TRAINABLE_RULE_NAMES` for training.

---

## 🟠 Logic Errors That Silently Corrupt Results

### 5. `_gin_lock` in Section 11 wraps the entire optimizer, not just GPU calls

```python
# Section 11:
with _gin_lock:
    res = circuit_optimizer.optimize(rec["expr"], verbose=False)  # entire call locked
```

This serializes ALL work across threads. Every SymPy rewrite, every networkx traversal, every DAG conversion — all serialized. The `ThreadPoolExecutor(max_workers=10)` effectively runs at 1 worker. The speedup is ~1x, not 6x as claimed.

Move the lock inside `CircuitOptimizer.optimize()` around only the GPU forward pass:

```python
# In CircuitOptimizer.optimize(), replace the GIN ranking block with:
if self.cfg["use_gin"] and self.model is not None:
    data = self._feat.to_pyg_data(G)
    ranked_cands = []
    for i in range(0, len(cands), BATCH):
        chunk = cands[i : i + BATCH]
        with _gin_lock:                           # ← only GPU call locked
            res = self.model.rank_candidates(data, chunk, device=DEVICE)["ranked"]
        ranked_cands.extend(res)
    gin_ranked = ranked_cands
```

The same issue applies in Section 12 ablation — `eval_optimizer_variant` doesn't use `_gin_lock` at all, so multiple threads call `model.rank_candidates()` simultaneously on the GPU without protection.

---

### 6. `visualize()` uses `width=1.0` for `_hierarchical_pos` on the DAG directly

```python
# In visualize():
pos = DAGConverter._hierarchical_pos(G, G.graph.get("root", ...), width=1.0)  # ← always 1.0
```

For a depth-3 circuit with 7 nodes, all leaf nodes get x-coords in [0, 1] compressed into a 1-unit range. The graph will look like all leaves are stacked on top of each other. Also, shared nodes in a DAG will be visited twice by `_recurse` and their position overwritten, causing edge crossings. Fix:

```python
pos = DAGConverter._hierarchical_pos(G, root, width=max(G.number_of_nodes(), 4))
```

And add a visited-set guard in `_hierarchical_pos` for DAG mode (shared nodes):

```python
@staticmethod
def _hierarchical_pos(G, root, width=1.0, vert_gap=1.0):
    def _recurse(n, left, right, depth, pos, visited):
        if n in visited:
            return (left + right) / 2.0   # shared node: return cached mid without re-placing
        visited.add(n)
        children = list(G.successors(n))
        if not children:
            mid = (left + right) / 2.0
            pos[n] = (mid, -depth)
            return mid
        step = (right - left) / len(children)
        mids = []
        for i, child in enumerate(children):
            mid = _recurse(child, left + i*step, left+(i+1)*step, depth+1, pos, visited)
            mids.append(mid)
        my_x = (mids[0] + mids[-1]) / 2.0
        pos[n] = (my_x, -depth)
        return my_x

    pos = {}
    _recurse(root, 0, width, 0, pos, set())
    return pos
```

---

### 7. `edge_color="#55"` in `visualize()` — invalid hex color

```python
nx.draw(G, pos, ..., edge_color="#55", ...)   # ← "#55" is only 2 hex digits, not 3 or 6
```

matplotlib will either ignore it silently or raise. Should be `"#555"` or `"#555555"`.

---

### 8. Section 7b `from distutils.command import build_ext` — unused and deprecated import

```python
from distutils.command import build_ext    # ← never used, distutils removed in Python 3.12
```

This will cause an `ImportError` on Python 3.12+. Remove it.

---

### 9. `_apply_rule_at_node` uses `full_expr.subs(sub, result)` which can substitute at multiple sites

When `sub` appears more than once in `full_expr` (which is common in CSE-heavy depth-3 circuits), `expr.subs(sub, result)` replaces ALL occurrences, not just the targeted node. This means a "local" action at `node_id=3` actually rewrites the whole expression globally. The node_id tracking becomes incorrect — the training labels say "node 3 is the action site" but the candidate was generated by a global substitution.

The fundamental fix requires substituting only the specific instance:

```python
def _apply_rule_at_node(self, full_expr, G, node_id, fn, ridx, rname, seen):
    sub = G.nodes[node_id].get("expr")
    if sub is None:
        return None
    result = fn(sub)
    if result is None or result == sub:
        return None
    
    # Only substitute once (first occurrence), not all
    # sp.Expr.subs replaces all; use a count-limited approach:
    n_occurrences = full_expr.count(sub)
    if n_occurrences == 0:
        return None
    if n_occurrences == 1:
        Cp_expr = full_expr.subs(sub, result)
    else:
        # For shared nodes: substitute globally (semantically equivalent for pure rewrites)
        # but mark scope as "global" so training label reflects this correctly
        Cp_expr = full_expr.subs(sub, result)
        rname = rname   # keep rule, but node_id should be GLOBAL_NODE_ID for shared nodes
        if G.nodes[node_id].get("fan_out", 0) > 1:
            node_id = GLOBAL_NODE_ID   # shared node rewrite is effectively global
    
    if Cp_expr == full_expr:
        return None
    return self._build_action(full_expr, G, node_id, ridx, rname, Cp_expr, seen)
```

---

## 🟡 Medium — Research Quality and Correctness

### 10. `generate()` still has dead `n_per_family * 6 / 14` scaling

```python
def generate(self, n_per_family=4167, DEPTH_FILTER=None):
    # ...
    n_per_family = int((n_per_family * 6) / len(FAMILY_BUILDERS))  # ← * 6 is from old 6-family code
```

With 14 families this becomes `n_per_family * 0.43`. The primary path now uses `generate_from_config`, but `generate()` can still be called directly (e.g. for top-ups). Clean it up:

```python
n_per_family = n_per_family  # just use it directly, no scaling
```

---

### 11. Ablation Section 12 creates 1400 `CircuitOptimizer` instances simultaneously

With 7 variants × 200 expressions, all tasks are submitted before any are collected. Each `CircuitOptimizer.__init__` creates `DAGConverter()`, `CandidateGenerator()`, `FeatureExtractor()`, `ExprNormalizer()`, `CostModel()`. 1400 such instances in memory at once is ~200-400MB of Python object overhead depending on cache sizes. With 6 workers actually running at once, only 6 instances are ever needed. Run variants sequentially to cap memory:

```python
# Replace the single ThreadPoolExecutor block with:
for variant_name, variant_config in variants.items():
    print(f"\n  Running variant: {variant_name}")
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(eval_optimizer_variant, rec["expr"], variant_name, variant_config)
            for rec in ablation_exprs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=variant_name):
            try:
                results.append(fut.result(timeout=120))
            except:
                results.append(0.0)
    ablation_results[variant_name] = np.array(results)
```

---

### 12. `_schwartz_zippel` uses a fixed seed `rng = random.Random(42)`, generating the same substitution points every call

```python
rng = random.Random(42)
for _ in range(self.trials):
    subs = {s: rng.randint(1, self.prime - 1) for s in syms}
```

With 30 trials and a fixed seed, the same 30 point evaluations are used for every PIT call. If a family of expressions happens to be equal at these specific points but not in general (a polynomial identity that holds at these particular integers), PIT returns a false positive. This undermines the semantic correctness guarantee. The `random.Random(42)` was added "for deterministic caching" but the result is already cached by `pipeline_caches.pit_cache`, so determinism across calls is already handled by the cache key. The random seed should be truly random:

```python
def _schwartz_zippel(self, orig, cand, syms):
    if not syms:
        return str(orig) == str(cand)
    rng = random.Random()  # fresh unpredictable seed each time (cached result makes this safe)
    for _ in range(self.trials):
        subs = {s: rng.randint(1, self.prime - 1) for s in syms}
        try:
            val_orig = int(orig.subs(subs)) % self.prime
            val_cand = int(cand.subs(subs)) % self.prime
            if val_orig != val_cand:
                return False
        except Exception:
            return False
    return True
```

---

### 13. `label_candidates` only runs lookahead on top-`TOP_LOOKAHEAD=2` candidates

```python
for i, c in enumerate(ranked):
    if immediate_norm > 0:
        if is_opt or i < TOP_LOOKAHEAD:   # ← only i=0,1 get lookahead during labeling
            _, best_future_cost, _ = self.lookahead_score(...)
        else:
            future_norm = immediate_norm   # ← rest get immediate gain as proxy for future
```

Candidates 2-N get `future_norm = immediate_norm`, so the `improvement = 0.5*immediate + 0.5*future` collapses to `improvement = immediate_norm` for 80%+ of training examples. This defeats the purpose of the lookahead and makes the "combined gain" label identical to the immediate gain for most data. Either increase `TOP_LOOKAHEAD` to something reasonable (like 5) or accept that only top-2 get real lookahead labels and document it clearly.

---

### 14. `n_iters` in results means "path length" not "beam iterations"

```python
return {
    ...
    "n_iters": len(path),   # ← length of accepted rewrites, not optimizer iterations
```

But the variable is named `n_iters` and the benchmark analysis groups by `n_iters` calling it "# Rewrites Accepted" in one plot and "# Optimization Steps" in another. These are different things — if beam iteration 3 produces 0 accepted rewrites (beam collapses), `n_iters = 0` but 3 beam iterations ran. Rename to `n_rewrites_accepted` and add a separate `n_beam_iters` counter:

```python
return {
    ...
    "n_rewrites_accepted": len(path),
    "n_beam_iters": it + 1,   # it is the loop variable from `for it in range(self.max_iters)`
    ...
}
```

---

## 🔵 Minor

### 15. Section 10 RF feature importance title says "17 node features" — there are 20

```python
plt.title("RF Feature Importance — 17 node features (v3)", ...)
```

Change to `f"RF Feature Importance — {N_FEATURES} node features"`.

---

### 16. `X_tab` uses `d.x.mean(dim=0)` — averaging node features loses graph structure

The Random Forest is trained on the mean node feature vector per graph, but the GIN is trained on per-node vectors. The RF interpretability baseline is therefore modeling a fundamentally different signal — it's predicting the rule from the average circuit properties, not from individual node properties. This is fine as a baseline but should be documented. Also, since the rule label is per-action (per node×rule pair), using the mean over all nodes is a lossy aggregation for nodes that aren't the target. Consider using only the target node's features:

```python
for d in tqdm(pyg_data_list[:TABULAR_N], desc="Tabularising"):
    # Use target node features instead of mean
    node_idx = int(d.target_node.item()) if hasattr(d, "target_node") else 0
    x_node = d.x[node_idx].numpy()   # (20,) — features of the actual action node
    X_tab.append(x_node)
```

---

### 17. Section 14 summary table (markdown) has outdated numbers

The table says `Dataset size: 100 000 expressions` but the code uses 25,000. It says `Node feature dim: 17` (should be 20). `Beam width: 5` (should be 8). These are copy-paste leftovers from earlier versions. The markdown table should match the actual constants.

---

### 18. `from distutils.command import build_ext` at top of Section 7b

As mentioned above — `distutils` is removed in Python 3.12 and this import serves no purpose. Single-line removal.

---

### 19. `visualize_node_scores` in Section 13 starts recursion counter at `len(T)` but `T` is a fresh DiGraph

```python
T = nx.DiGraph()
def _unroll(n_dag, parent_tree=None):
    n_tree = len(T)          # ← correct: starts at 0 for fresh graph
    T.add_node(n_tree, ...)
```

This is actually correct. But it becomes wrong if `_unroll` is called twice on the same `T` (e.g. if someone loops the visualization). Not an immediate bug, but wrapping `T` inside the function body (instead of as a closure) would be safer:

```python
def visualize_node_scores(expr, model, title=None):
    T = nx.DiGraph()   # ← already here — fine as is, just noting it's a closure issue if refactored
```

---

## Priority Summary

| # | Issue | Section | Severity |
|---|-------|---------|----------|
| 1 | `generate_from_config` / `_get_family_builders` missing | 1 | 🔴 Crash |
| 2 | `DAGConverter` not yet defined when `_make_rec` runs | 1 | 🔴 Silent wrong depths |
| 3 | `save_gin_ckpt` drops all `gin_epoch_*.pt` files | 7c | 🔴 No epoch checkpoints |
| 4 | Rules 0 (identity) & 6 (dead_pruning) in `RULE_NAMES` with 0 training data | 7b/7c | 🔴 Corrupts loss weights |
| 5 | `_gin_lock` wraps entire optimizer — ThreadPool is serialized | 11/12 | 🟠 ~1x speedup, not 6x |
| 6 | `_hierarchical_pos` called with `width=1.0` in `visualize()` + no visited-set for DAG | 4 | 🟠 Bad visualization |
| 7 | `edge_color="#55"` invalid hex | 4 | 🟠 Matplotlib error |
| 8 | `from distutils.command import build_ext` removed in Py 3.12 | 7b | 🟠 ImportError |
| 9 | `subs(sub, result)` rewrites all occurrences, not just target node | 6 | 🟠 Wrong node_id labels |
| 10 | Fixed Schwartz-Zippel seed gives same 30 points every call | 8 | 🟡 False positives |
| 11 | `n_per_family * 6` scaling dead code in `generate()` | 1 | 🟡 Wrong counts if called directly |
| 12 | 1400 simultaneous `CircuitOptimizer` instances in ablation | 12 | 🟡 Memory |
| 13 | `TOP_LOOKAHEAD=2` means 80%+ of training labels ignore lookahead | 6 | 🟡 Weaker labels |
| 14 | `n_iters` means rewrites accepted, not beam iterations | 9/11 | 🟡 Misleading metric |
| 15 | RF uses mean node features instead of target-node features | 10 | 🔵 Interpretability |
| 16 | Section 14 markdown has wrong dataset size, feature dim, beam width | 14 | 🔵 Docs |
| 17 | RF importance title says "17 node features" | 10 | 🔵 Docs |

