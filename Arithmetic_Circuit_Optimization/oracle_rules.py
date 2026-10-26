"""
Oracle rule training data generator for ArithCircuit pipeline.

Generates targeted expressions where reassociation and depth_reduction
provably reduce circuit cost, independent of the random-expression +
lookahead pipeline. These are labeled with node_reducible=1.0 to ensure
these rules have positive training signal.

Usage:
    from oracle_rules import generate_oracle_dataset
    oracle_records = generate_oracle_dataset(n_per_rule=5000, seed=42)
"""

import random
import math
import sympy as sp
from sympy import symbols, Add, Mul, Pow, expand, factor, collect, powsimp, count_ops

ALL_VARS = list(symbols("x y z w u v a b c d e f"))

# Constants shared with the main pipeline
GLOBAL_NODE_ID = -1
N_RULES = 8
RULE_NAMES = [
    "factorization", "distribution", "reassociation",
    "const_folding", "common_factor", "depth_reduction",
    "collection", "cse_rewrite",
]


def _rng_sample(rng, vars_list, n):
    """Sample n distinct variables from vars_list using rng."""
    return rng.sample(vars_list, min(n, len(vars_list)))


def _expr_depth(expr):
    """Compute the AST depth of a SymPy expression."""
    if not hasattr(expr, "args") or not expr.args:
        return 0
    return 1 + max(_expr_depth(a) for a in expr.args)


def generate_oracle_reassociation(rng, n_exprs):
    """
    Generate left-leaning Add/Mul trees that reassociation can balance.

    Example:
        Input:  Add(Add(Add(a, b), c), d)   (left-leaning, depth 3)
        Output: Add(Add(a, b), Add(c, d))   (balanced, depth 2)

    Uses evaluate=False to prevent SymPy from flattening the tree.
    """
    records = []
    for _ in range(n_exprs):
        n_vars = rng.randint(4, 8)
        vars_selected = _rng_sample(rng, ALL_VARS, n_vars)

        # Build left-leaning Add tree with evaluate=False: ((v0+v1)+v2)+v3+...
        expr = vars_selected[0]
        for v in vars_selected[1:]:
            expr = Add(expr, v, evaluate=False)

        # Build balanced target: (v0+v1)+(v2+v3)+(v4+...)
        if len(vars_selected) >= 4:
            mid = len(vars_selected) // 2
            left = vars_selected[0]
            for v in vars_selected[1:mid]:
                left = Add(left, v, evaluate=False)
            right = vars_selected[mid]
            for v in vars_selected[mid + 1:]:
                right = Add(right, v, evaluate=False)
            target = Add(left, right, evaluate=False)
        else:
            target = expr

        orig_depth = _expr_depth(expr)
        target_depth = _expr_depth(target)

        records.append({
            "expr": expr,
            "target_expr": target,
            "rule": "reassociation",
            "rule_idx": 2,
            "orig_depth": orig_depth,
            "target_depth": target_depth,
            "improvement": float(max(0, orig_depth - target_depth)) * 0.50,
            "node_reducible": 1.0,
        })
    return records


def generate_oracle_depth_reduction(rng, n_exprs):
    """
    Generate expressions where depth_reduction provably reduces circuit depth.

    Uses perfect square trinomials:
        Input:  (x+y)**2 + 2*(x+y)*z + z**2
        Output: (x+y+z)**2

    Also uses power simplification:
        Input:  x**2 * x**3   (kept unevaluated)
        Output: x**5
    """
    records = []
    for _ in range(n_exprs):
        vars_selected = _rng_sample(rng, ALL_VARS, 3)
        x, y, z = vars_selected[0], vars_selected[1], vars_selected[2]

        choice = rng.random()

        if choice < 0.5:
            # Perfect square trinomial: (x+y)^2 + 2*(x+y)*z + z^2 = (x+y+z)^2
            expr = Add(
                Pow(Add(x, y, evaluate=False), 2),
                Mul(2, Add(x, y, evaluate=False), z, evaluate=False),
                Pow(z, 2),
                evaluate=False,
            )
            target = Pow(Add(x, y, z), 2)

        elif choice < 0.8:
            # Power simplification: x^2 * x^3 = x^5
            expr = Mul(Pow(x, 2), Pow(x, 3), evaluate=False)
            target = Pow(x, 5)

        else:
            # Common factor extraction via collect: x*y + x*z = x*(y+z)
            expr = Add(Mul(x, y, evaluate=False), Mul(x, z, evaluate=False), evaluate=False)
            target = Mul(x, Add(y, z, evaluate=False), evaluate=False)

        orig_depth = _expr_depth(expr)
        target_depth = _expr_depth(target)

        # Only keep if there's actual depth reduction
        if target_depth < orig_depth:
            records.append({
                "expr": expr,
                "target_expr": target,
                "rule": "depth_reduction",
                "rule_idx": 5,
                "orig_depth": orig_depth,
                "target_depth": target_depth,
                "improvement": float(max(0, orig_depth - target_depth)) * 0.50,
                "node_reducible": 1.0,
            })

    # If we didn't get enough, pad with perfect square trinomials
    while len(records) < n_exprs:
        vars_selected = _rng_sample(rng, ALL_VARS, 3)
        x, y, z = vars_selected[0], vars_selected[1], vars_selected[2]
        expr = Add(
            Pow(Add(x, y, evaluate=False), 2),
            Mul(2, Add(x, y, evaluate=False), z, evaluate=False),
            Pow(z, 2),
            evaluate=False,
        )
        target = Pow(Add(x, y, z), 2)
        orig_depth = _expr_depth(expr)
        target_depth = _expr_depth(target)
        if target_depth < orig_depth:
            records.append({
                "expr": expr,
                "target_expr": target,
                "rule": "depth_reduction",
                "rule_idx": 5,
                "orig_depth": orig_depth,
                "target_depth": target_depth,
                "improvement": float(max(0, orig_depth - target_depth)) * 0.50,
                "node_reducible": 1.0,
            })

    return records[:n_exprs]


def generate_oracle_cse_rewrite(rng, n_exprs):
    """
    Generate expressions where cse_rewrite provably reduces node count.

    Example:
        Input:  (a+b)*c + (a+b)*d + (a+b)*e   (shared subexpr a+b repeated 3x)
        Output: (a+b)*(c+d+e)

    Uses evaluate=False to prevent SymPy from auto-factoring.
    """
    records = []
    for _ in range(n_exprs):
        vars_selected = _rng_sample(rng, ALL_VARS, 5)
        a, b, c, d, e = vars_selected[0], vars_selected[1], vars_selected[2], vars_selected[3], vars_selected[4]

        # Shared subexpression pattern with evaluate=False
        shared = Add(a, b, evaluate=False)
        expr = Add(
            Mul(shared, c, evaluate=False),
            Mul(shared, d, evaluate=False),
            Mul(shared, e, evaluate=False),
            evaluate=False,
        )
        target = Mul(shared, Add(c, d, e, evaluate=False), evaluate=False)

        orig_depth = _expr_depth(expr)
        target_depth = _expr_depth(target)

        records.append({
            "expr": expr,
            "target_expr": target,
            "rule": "cse_rewrite",
            "rule_idx": 7,
            "orig_depth": orig_depth,
            "target_depth": target_depth,
            "improvement": 0.5,
            "node_reducible": 1.0,
        })
    return records


def generate_oracle_dataset(n_per_rule=5000, seed=42):
    """
    Generate oracle-labeled dataset for starved rules.

    Args:
        n_per_rule: Number of expressions per rule (reassociation, depth_reduction, cse_rewrite)
        seed: Random seed for reproducibility

    Returns:
        List of dict records, each with expr, target_expr, rule, rule_idx,
        improvement, node_reducible fields.
    """
    rng = random.Random(seed)
    records = []

    # Reassociation oracle
    reassoc_records = generate_oracle_reassociation(rng, n_per_rule)
    records.extend(reassoc_records)

    # Depth reduction oracle
    depth_records = generate_oracle_depth_reduction(rng, n_per_rule)
    records.extend(depth_records)

    # CSE rewrite oracle
    cse_records = generate_oracle_cse_rewrite(rng, n_per_rule)
    records.extend(cse_records)

    rng.shuffle(records)
    return records


def verify_oracle_records(records):
    """
    Verify that oracle target expressions are algebraically equivalent
    to the originals and provably reduce cost.
    """
    from sympy import simplify

    stats = {"reassociation": 0, "depth_reduction": 0, "cse_rewrite": 0}
    failures = []

    for rec in records:
        orig = rec["expr"]
        target = rec["target_expr"]
        rule = rec["rule"]

        # Verify algebraic equivalence
        try:
            equiv = sp.simplify(orig - target) == 0
        except Exception:
            equiv = False

        if not equiv:
            failures.append({
                "rule": rule,
                "orig": str(orig),
                "target": str(target),
                "reason": "not algebraically equivalent",
            })
            continue

        # Verify cost reduction
        orig_depth = _expr_depth(orig)
        target_depth = _expr_depth(target)

        if target_depth >= orig_depth:
            failures.append({
                "rule": rule,
                "orig": str(orig),
                "target": str(target),
                "reason": f"no depth reduction: {orig_depth} -> {target_depth}",
            })
            continue

        stats[rule] += 1

    return stats, failures


if __name__ == "__main__":
    # Quick self-test
    records = generate_oracle_dataset(n_per_rule=100, seed=42)
    stats, failures = verify_oracle_records(records)

    print(f"Generated {len(records)} oracle records")
    print(f"Per-rule stats (passing verification): {stats}")
    print(f"Failures: {len(failures)}")
    for f in failures[:5]:
        print(f"  {f['rule']}: {f['reason']}")
        print(f"    orig: {f['orig']}")
        print(f"    target: {f['target']}")
