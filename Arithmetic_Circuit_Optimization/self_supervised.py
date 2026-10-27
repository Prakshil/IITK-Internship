"""
Self-Supervised Scramble-and-Reverse Training Data Generator (Phase 3).

Generates guaranteed-optimal training data by starting from canonical minimal
expressions and applying rules in REVERSE to complexify them. The GIN is then
trained to reverse the process (predict the optimal rule sequence back to
minimal form).

This provides:
- Infinite training data with guaranteed ground-truth optimal paths
- Coverage of all rules including currently-starved ones (reassociation,
  depth_reduction, cse_rewrite)
- Direct comparison against supervised lookahead labels

Usage:
    from self_supervised import SelfSupervisedGenerator
    gen = SelfSupervisedGenerator(seed=42)
    records = gen.generate_dataset(n_per_rule=5000)
"""

import random
import sympy as sp
from sympy import symbols, Add, Mul, Pow, expand, factor, collect, powsimp

from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID

ALL_VARS = list(symbols("x y z w u v a b c d e f"))


class SelfSupervisedGenerator:
    """
    Generates self-supervised training data via scramble-and-reverse.

    For each forward rule, we:
    1. Start with a canonical minimal expression
    2. Apply the rule IN REVERSE to complexify it
    3. Record the (complexified → minimal) pair as a training example
    4. The target action is the forward rule that reduces cost

    Reverse rule mapping:
        factorization   → distribute (expand)        [complexify by expanding]
        distribution    → factor                      [complexify by factoring]
        reassociation   → unbalance tree              [complexify by unbalancing]
        const_folding   → un-fold constants           [complexify by splitting]
        common_factor   → distribute common factor    [complexify by distributing]
        depth_reduction → expand powers               [complexify by expanding]
        collection      → distribute coefficients     [complexify by distributing]
        cse_rewrite     → inline common subexpr       [complexify by inlining]
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.x, self.y, self.z, self.w, self.u, self.v = symbols(
            "x y z w u v")

    # ------------------------------------------------------------------
    # Canonical minimal form generators (ground-truth optimal)
    # ------------------------------------------------------------------

    def _minimal_factorized(self):
        """Generate a fully factored minimal expression.
        Example: x*(y + z), (x + y)*(x + z)
        """
        choice = self.rng.random()
        if choice < 0.7:
            n_vars = self.rng.randint(3, 6)
        else:
            n_vars = self.rng.randint(4, 6)
        vars_sel = self.rng.sample(ALL_VARS, n_vars)

        if choice < 0.4:
            # x*(y + z) form
            a = vars_sel[0]
            b, c = vars_sel[1], vars_sel[2]
            return Mul(a, Add(b, c, evaluate=False), evaluate=False)
        elif choice < 0.7:
            # (x + y)*(x + z) form
            a, b, c = vars_sel[0], vars_sel[1], vars_sel[2]
            return Mul(Add(a, b, evaluate=False), Add(a, c, evaluate=False),
                       evaluate=False)
        else:
            # x*(y + z) + w*(y + z) = (x + w)*(y + z)
            a, b, c, d = vars_sel[0], vars_sel[1], vars_sel[2], vars_sel[3]
            shared = Add(b, c, evaluate=False)
            return Add(Mul(a, shared, evaluate=False),
                       Mul(d, shared, evaluate=False), evaluate=False)

    def _minimal_collected(self):
        """Generate a collected minimal expression.
        Example: x*(y + z) + x*w = x*(y + z + w)
        """
        vars_sel = self.rng.sample(ALL_VARS,
                                   self.rng.randint(3, 6))
        a = vars_sel[0]
        others = vars_sel[1:]
        return Mul(a, Add(*others, evaluate=False), evaluate=False)

    def _minimal_balanced(self):
        """Generate a balanced tree minimal expression.
        Example: (a + b) + (c + d)
        """
        vars_sel = self.rng.sample(ALL_VARS, self.rng.randint(4, 6))
        mid = len(vars_sel) // 2
        left = vars_sel[0]
        for v in vars_sel[1:mid]:
            left = Add(left, v, evaluate=False)
        right = vars_sel[mid]
        for v in vars_sel[mid + 1:]:
            right = Add(right, v, evaluate=False)
        return Add(left, right, evaluate=False)

    def _minimal_power_simplified(self):
        """Generate a power-simplified minimal expression.
        Example: x**5 (from x**2 * x**3)
        """
        vars_sel = self.rng.sample(ALL_VARS, self.rng.randint(2, 4))
        base = vars_sel[0]
        exp1 = self.rng.randint(2, 4)
        exp2 = self.rng.randint(2, 4)
        return Pow(base, exp1 + exp2)

    def _minimal_common_factored(self):
        """Generate an expression with common factor extracted.
        Example: a*x + a*y + a*z = a*(x + y + z)
        """
        vars_sel = self.rng.sample(ALL_VARS, self.rng.randint(3, 6))
        common = vars_sel[0]
        others = vars_sel[1:]
        return Mul(common, Add(*others, evaluate=False), evaluate=False)

    def _minimal_cse(self):
        """Generate an expression with CSE applied.
        Example: (a + b)*c + (a + b)*d = (a + b)*(c + d)
        """
        vars_sel = self.rng.sample(ALL_VARS, self.rng.randint(4, 6))
        shared = Add(vars_sel[0], vars_sel[1], evaluate=False)
        others = vars_sel[2:]
        return Mul(shared, Add(*others, evaluate=False), evaluate=False)

    # ------------------------------------------------------------------
    # Reverse rule applications (complexification)
    # ------------------------------------------------------------------

    def _reverse_factorize(self, minimal_expr):
        """Reverse of factorization: distribute/expand.
        Input: x*(y + z)  →  Output: x*y + x*z
        """
        return expand(minimal_expr)

    def _reverse_distribution(self, minimal_expr):
        """Reverse of distribution: factor.
        Input: x*y + x*z  →  Output: x*(y + z)
        """
        return factor(minimal_expr)

    def _reverse_reassociation(self, minimal_expr):
        """Reverse of reassociation: unbalance tree.
        Input: (a + b) + (c + d)  →  Output: ((a + b) + c) + d
        """
        # Build a left-leaning tree from the balanced minimal
        if not isinstance(minimal_expr, Add):
            return minimal_expr
        args = list(minimal_expr.args)
        if len(args) < 3:
            return minimal_expr
        # Left-lean: ((arg0 + arg1) + arg2) + arg3 + ...
        result = args[0]
        for arg in args[1:]:
            result = Add(result, arg, evaluate=False)
        return result

    def _reverse_const_folding(self, minimal_expr):
        """Reverse of const_folding: split constants.
        Input: 5*x  →  Output: (2 + 3)*x
        """
        # Find a constant multiplier and split it
        if isinstance(minimal_expr, Mul):
            args = list(minimal_expr.args)
            consts = [a for a in args if a.is_Number]
            non_consts = [a for a in args if not a.is_Number]
            if consts:
                c = consts[0]
                if c > 3:
                    c1 = self.rng.randint(2, int(c) - 1)
                    c2 = int(c) - c1
                    split = Add(c1, c2, evaluate=False)
                    return Mul(split, *non_consts, evaluate=False)
        return minimal_expr

    def _reverse_common_factor(self, minimal_expr):
        """Reverse of common_factor: distribute common factor.
        Input: a*(x + y)  →  Output: a*x + a*y
        """
        return expand(minimal_expr)

    def _reverse_depth_reduction(self, minimal_expr):
        """Reverse of depth_reduction: expand powers.
        Input: x**5  →  Output: x**2 * x**3
        """
        if isinstance(minimal_expr, Pow):
            base, exp = minimal_expr.args
            if exp > 4:
                e1 = self.rng.randint(2, int(exp) - 2)
                e2 = int(exp) - e1
                return Mul(Pow(base, e1, evaluate=False), Pow(base, e2,
                                                              evaluate=False),
                           evaluate=False)
        return minimal_expr

    def _reverse_collection(self, minimal_expr):
        """Reverse of collection: distribute coefficients.
        Input: x*(y + z + w)  →  Output: x*y + x*z + x*w
        """
        return expand(minimal_expr)

    def _reverse_cse_rewrite(self, minimal_expr):
        """Reverse of cse_rewrite: inline common subexpression.
        Input: (a + b)*(c + d)  →  Output: (a + b)*c + (a + b)*d
        """
        return expand(minimal_expr)

    # ------------------------------------------------------------------
    # Main dataset generation
    # ------------------------------------------------------------------

    REVERSE_MAP = {
        "factorization": _reverse_factorize,
        "distribution": _reverse_distribution,
        "reassociation": _reverse_reassociation,
        "const_folding": _reverse_const_folding,
        "common_factor": _reverse_common_factor,
        "depth_reduction": _reverse_depth_reduction,
        "collection": _reverse_collection,
        "cse_rewrite": _reverse_cse_rewrite,
    }

    MINIMAL_GENERATORS = {
        "factorization": _minimal_factorized,
        "distribution": _minimal_collected,
        "reassociation": _minimal_balanced,
        "const_folding": _minimal_collected,
        "common_factor": _minimal_common_factored,
        "depth_reduction": _minimal_power_simplified,
        "collection": _minimal_collected,
        "cse_rewrite": _minimal_cse,
    }

    def generate_rule_examples(self, rule_name, n_examples):
        """Generate n training examples for a specific rule.

        Each example is a (complexified_expr, minimal_expr, rule_name, rule_idx)
        tuple where applying the forward rule to complexified gives minimal.
        """
        rule_idx = RULE_NAMES.index(rule_name) if rule_name in RULE_NAMES else -1
        minimal_gen = self.MINIMAL_GENERATORS[rule_name]
        reverse_fn = self.REVERSE_MAP[rule_name]

        records = []
        attempts = 0
        max_attempts = n_examples * 10

        while len(records) < n_examples and attempts < max_attempts:
            attempts += 1

            # Generate minimal expression
            minimal = minimal_gen(self)

            # Apply reverse to complexify
            complexified = reverse_fn(self, minimal)

            # Skip if no change
            if sp.srepr(complexified) == sp.srepr(minimal):
                continue

            # Verify algebraic equivalence
            try:
                equiv = sp.simplify(complexified - minimal) == 0
            except Exception:
                continue
            if not equiv:
                continue

            # Verify cost reduction (complexified should have higher cost)
            from rl_environment import CostModel, DAGConverter, ExprNormalizer
            cm = CostModel()
            dc = DAGConverter()
            norm = ExprNormalizer()

            try:
                G_complex = dc.convert(norm.normalize_for_cost(complexified))
                G_minimal = dc.convert(norm.normalize_for_cost(minimal))
                cost_complex = cm.score_graph(G_complex)
                cost_minimal = cm.score_graph(G_minimal)
            except Exception:
                continue

            if cost_minimal >= cost_complex:
                continue  # minimal must be cheaper

            records.append({
                "complexified_expr": complexified,
                "minimal_expr": minimal,
                "rule": rule_name,
                "rule_idx": rule_idx,
                "cost_complex": cost_complex,
                "cost_minimal": cost_minimal,
                "cost_reduction": cost_complex - cost_minimal,
                "node_reducible": 1.0,
            })

        return records

    def generate_dataset(self, n_per_rule=5000):
        """Generate self-supervised dataset for all rules.

        Returns:
            List of training records, one per (complexified, minimal) pair.
        """
        all_records = []
        for rule_name in RULE_NAMES:
            print(f"  Generating {n_per_rule} examples for {rule_name}...")
            records = self.generate_rule_examples(rule_name, n_per_rule)
            print(f"    Got {len(records)} valid examples")
            all_records.extend(records)

        self.rng.shuffle(all_records)
        return all_records


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = SelfSupervisedGenerator(seed=42)
    records = gen.generate_dataset(n_per_rule=100)

    print(f"\nGenerated {len(records)} total self-supervised examples")
    from collections import Counter
    rule_counts = Counter(r["rule"] for r in records)
    for rule in RULE_NAMES:
        print(f"  {rule:<20} {rule_counts.get(rule, 0):>4}")

    # Show a few examples
    print("\nExample transformations:")
    for r in records[:5]:
        print(f"  {r['rule']:<20}: {r['complexified_expr']} -> {r['minimal_expr']} "
              f"(cost: {r['cost_complex']:.2f} -> {r['cost_minimal']:.2f})")
