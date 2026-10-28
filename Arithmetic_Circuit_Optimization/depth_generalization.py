"""
Depth Generalization Study (Phase 4).

Tests whether the GIN model trained on depth-3 circuits transfers to
deeper circuits (depth-4, depth-5, depth-6).

Usage:
    python depth_generalization.py --model models/gin_best_fixed.pt --device cuda

Steps:
    4.1 Generate deeper circuit benchmark (depth-4, 5, 6)
    4.2 Zero-shot transfer test
    4.3 Fine-tuning with minimal data (10% of depth-4)
"""

import argparse
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID  # noqa: E402
from rl_environment import (
    GINRewriteRanker, CostModel, DAGConverter, ExprNormalizer,
    FeatureExtractor, CandidateGenerator, CircuitOptimizer, PITVerifier
)  # noqa: E402


# ---------------------------------------------------------------------------
# Depth Circuit Generators
# ---------------------------------------------------------------------------

class DepthCircuitGenerator:
    """Generates arithmetic circuits of specified depth (3, 4, 5, 6)."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.x, self.y, self.z, self.w, self.u, self.v = sp.symbols(
            "x y z w u v")
        self.ALL_VARS = [self.x, self.y, self.z, self.w, self.u, self.v]

    def _sample_vars(self, n):
        return self.rng.sample(self.ALL_VARS, min(n, len(self.ALL_VARS)))

    def generate_depth_3(self, n_exprs):
        """Generate depth-3 expressions using existing families."""
        records = []
        for _ in range(n_exprs):
            vs = self._sample_vars(6)
            # depth3_circuit: shared*(v2) + shared*(v3) where shared = v0+v1
            shared = sp.Add(vs[0], vs[1], evaluate=False)
            expr = sp.Add(
                sp.Mul(shared, vs[2], evaluate=False),
                sp.Mul(shared, vs[3], evaluate=False),
                evaluate=False
            )
            records.append({"expr": expr, "family": "depth3_circuit", "depth": 3})
        return records

    def generate_depth_4(self, n_exprs):
        """Generate depth-4 expressions.
        Structure: (shared1 * shared2) + (shared3 * shared4)
        where shared = sum of 2 vars each → depth 4
        """
        records = []
        for _ in range(n_exprs):
            vs = self._sample_vars(8)
            s1 = sp.Add(vs[0], vs[1], evaluate=False)
            s2 = sp.Add(vs[2], vs[3], evaluate=False)
            s3 = sp.Add(vs[4], vs[5], evaluate=False)
            s4 = sp.Add(vs[6], vs[7], evaluate=False)
            expr = sp.Add(
                sp.Mul(s1, s2, evaluate=False),
                sp.Mul(s3, s4, evaluate=False),
                evaluate=False
            )
            records.append({"expr": expr, "family": "depth4", "depth": 4})
        return records

    def generate_depth_5(self, n_exprs):
        """Generate depth-5 expressions.
        Structure: ((shared1 * shared2) + shared3) * shared4
        Depth: shared(1) * shared(1) = 2, + shared(1) = 3, * shared(1) = 4... need 5
        Let's do: (s1*s2 + s3) * (s4 + s5) where s = sum of 2 vars
        Depth: s1(1), s2(1), s1*s2(2), s3(1), s1*s2+s3(3), s4(1), s5(1), s4+s5(2), final(4)... hmm
        
        Better: (((a+b)*(c+d)) + e) * (f+g)
        a+b:1, c+d:1, (a+b)*(c+d):2, +e:3, f+g:1, final:4... still 4.
        
        Need depth 5: (((a+b)*(c+d) + e*f) * (g+h))
        a+b:1, c+d:1, product:2, e*f:1, sum:3, g+h:1, final:4... still 4.
        
        Let's try: (((a+b)*(c+d) + e)*(f+g))
        a+b:1, c+d:1, product:2, +e:3, f+g:1, final:4... still 4.
        
        Need one more level: (((a+b)*(c+d)*(e+f)) + g)
        a+b:1, c+d:1, e+f:1, (a+b)*(c+d):2, *(e+f):3, +g:4... still 4.
        
        OK let's just use: ((a+b)*(c+d) + (e+f)*(g+h)) * (i+j)
        (a+b):1, (c+d):1, product:2, (e+f):1, (g+h):1, product:2, sum:3, (i+j):1, final:4... still 4.
        
        Hmm, depth counting with SymPy Add/Mul args is flat. Let me use nested structure:
        sp.Add(sp.Mul(a, sp.Mul(b, c, evaluate=False), evaluate=False), d, evaluate=False)
        This gives: a*(b*c) + d, depth: b*c=1, a*(b*c)=2, +d=3. Still not enough.
        
        For depth 5: a*(b*(c*(d+e)))+f
        d+e:1, c*(d+e):2, b*(c*(d+e)):3, a*(...):4, +f:5. Yes!
        """
        records = []
        for _ in range(n_exprs):
            vs = self._sample_vars(6)
            # a*(b*(c*(d+e))) + f → depth 5
            inner = sp.Add(vs[3], vs[4], evaluate=False)  # d+e: depth 1
            level2 = sp.Mul(vs[2], inner, evaluate=False)  # c*(d+e): depth 2
            level3 = sp.Mul(vs[1], level2, evaluate=False)  # b*(c*(d+e)): depth 3
            level4 = sp.Mul(vs[0], level3, evaluate=False)  # a*(...): depth 4
            expr = sp.Add(level4, vs[5], evaluate=False)  # +f: depth 5
            records.append({"expr": expr, "family": "depth5", "depth": 5})
        return records

    def generate_depth_6(self, n_exprs):
        """Generate depth-6 expressions.
        Structure: a*(b*(c*(d*(e+f)))) + g
        e+f:1, d*(e+f):2, c*(d*(e+f)):3, b*(c*(d*(e+f))):4, a*(b*(c*(d*(e+f)))):5, +g:6
        """
        records = []
        for _ in range(n_exprs):
            vs = self._sample_vars(7)
            inner = sp.Add(vs[4], vs[5], evaluate=False)  # e+f: depth 1
            l2 = sp.Mul(vs[3], inner, evaluate=False)  # d*(e+f): depth 2
            l3 = sp.Mul(vs[2], l2, evaluate=False)  # c*(d*(e+f)): depth 3
            l4 = sp.Mul(vs[1], l3, evaluate=False)  # b*(c*(d*(e+f))): depth 4
            l5 = sp.Mul(vs[0], l4, evaluate=False)  # a*(...): depth 5
            expr = sp.Add(l5, vs[6], evaluate=False)  # +g: depth 6
            records.append({"expr": expr, "family": "depth6", "depth": 6})
        return records


# ---------------------------------------------------------------------------
# Depth Generalization Study
# ---------------------------------------------------------------------------

def run_depth_generalization(model_path, n_per_depth=200, device="cuda"):
    """Run complete depth generalization study."""
    
    # Load model
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GINRewriteRanker().to(device_obj)
    checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    # Create optimizer
    env = CircuitOptimizer(model=model, max_iters=8, gin_top_k=10, beam_width=8)

    # Generate datasets
    gen = DepthCircuitGenerator(seed=42)
    
    depth3_exprs = gen.generate_depth_3(n_per_depth)
    depth4_exprs = gen.generate_depth_4(n_per_depth)
    depth5_exprs = gen.generate_depth_5(n_per_depth)
    depth6_exprs = gen.generate_depth_6(n_per_depth)

    print(f"\nGenerated datasets: depth-3={len(depth3_exprs)}, depth-4={len(depth4_exprs)}, "
          f"depth-5={len(depth5_exprs)}, depth-6={len(depth6_exprs)}")

    # Run optimization on each depth
    results = {}
    for depth_name, exprs in [("depth-3", depth3_exprs), ("depth-4", depth4_exprs),
                               ("depth-5", depth5_exprs), ("depth-6", depth6_exprs)]:
        print(f"\n{'='*60}")
        print(f"Evaluating {depth_name} ({len(exprs)} expressions)")
        print(f"{'='*60}")

        depth_results = []
        t0 = time.time()
        for i, rec in enumerate(exprs):
            try:
                res = env.optimize(rec["expr"], verbose=False)
                depth_results.append({
                    "family": rec["family"],
                    "depth": rec["depth"],
                    "orig_cost": res["orig_cost"],
                    "final_cost": res["final_cost"],
                    "pct_reduction": res["pct_reduction"],
                    "delta_score": res["delta_score"],
                    "n_iters": res.get("n_rewrites_accepted", 0),
                    "accepted_rules": res.get("accepted_rules", []),
                })
            except Exception as e:
                print(f"  Failed on {rec['family']}: {e}")
                continue

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(exprs)} completed")

        t_elapsed = time.time() - t0
        print(f"  Completed in {t_elapsed:.1f}s ({t_elapsed/len(depth_results):.2f}s/expr)")

        results[depth_name] = depth_results

        # Print summary
        if depth_results:
            pcts = [r["pct_reduction"] for r in depth_results]
            print(f"  Mean % reduction: {np.mean(pcts):.2f}%")
            print(f"  Median % reduction: {np.median(pcts):.2f}%")
            print(f"  Max % reduction: {np.max(pcts):.2f}%")
            print(f"  Zero-improvement: {sum(1 for p in pcts if p == 0)}/{len(pcts)}")

    # Save results
    all_results = []
    for depth_name, depth_results in results.items():
        all_results.extend(depth_results)

    import pandas as pd
    df = pd.DataFrame(all_results)
    output_path = Path("data/depth_generalization.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"DEPTH GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Depth':<10} {'Mean %':>10} {'Median %':>12} {'Max %':>10} {'Zero-imp':>12} {'N':>6}")
    print(f"{'-'*70}")
    for depth_name in ["depth-3", "depth-4", "depth-5", "depth-6"]:
        if depth_name in results:
            pcts = [r["pct_reduction"] for r in results[depth_name]]
            zero = sum(1 for p in pcts if p == 0)
            print(f"{depth_name:<10} {np.mean(pcts):>10.2f} {np.median(pcts):>12.2f} "
                  f"{np.max(pcts):>10.2f} {zero:>10}/{len(pcts):<4} {len(pcts):>6}")

    # Zero-shot transfer analysis
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT TRANSFER ANALYSIS")
    print(f"{'='*70}")
    d3_mean = np.mean([r["pct_reduction"] for r in results["depth-3"]])
    for depth_name in ["depth-4", "depth-5", "depth-6"]:
        if depth_name in results:
            d_mean = np.mean([r["pct_reduction"] for r in results[depth_name]])
            retention = d_mean / d3_mean * 100 if d3_mean > 0 else 0
            drop = d3_mean - d_mean
            print(f"{depth_name}: mean={d_mean:.2f}% | retention={retention:.1f}% | drop={drop:.2f}pp")

    return results


def main():
    parser = argparse.ArgumentParser(description="Depth generalization study")
    parser.add_argument("--model", default="models/gin_best_fixed.pt",
                        help="Model checkpoint path")
    parser.add_argument("--n-per-depth", type=int, default=200,
                        help="Number of expressions per depth")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    run_depth_generalization(
        model_path=args.model,
        n_per_depth=args.n_per_depth,
        device=args.device,
    )


if __name__ == "__main__":
    main()
