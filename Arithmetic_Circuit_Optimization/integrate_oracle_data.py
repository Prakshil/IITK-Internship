"""
Integrate oracle rule training data into the PyG corpus.

This script:
1. Generates oracle-labeled records for reassociation, depth_reduction, cse_rewrite
2. Converts each to PyG Data format (matching the existing pipeline's interface)
3. Appends to the existing pyg_data_list.pkl
4. Saves the combined corpus

Usage:
    python integrate_oracle_data.py --n-per-rule 5000 --seed 42

Self-contained: replicates DAGConverter, FeatureExtractor, ExprNormalizer
from the notebook so it can run without importing the .ipynb file.
"""

import argparse
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_rules import generate_oracle_dataset

# ---------------------------------------------------------------------------
# Constants (mirrored from notebook Section 0)
# ---------------------------------------------------------------------------

SEED = 42
VARS = list(sp.symbols("x y z w u v a b c d e f"))
x, y, z, w, u, v = sp.symbols("x y z w u v")

EDGE_TYPE = {"add_child": 0, "mul_child": 1, "pow_base": 2, "pow_exp": 3, "other": 4}
EDGE_FEAT_DIM = 5

FEATURE_NAMES = [
    "gate_var", "gate_const", "gate_add", "gate_mul", "gate_pow", "gate_other",
    "depth_from_root", "fan_in", "fan_out_norm", "subtree_size_norm",
    "is_shared",
    "local_degree", "subtree_degree", "degree_to_output",
    "depth_to_leaf", "op_density", "mul_ratio", "reuse_count",
    "estimated_blowup", "parent_op_type",
]
N_FEATURES = len(FEATURE_NAMES)

GLOBAL_NODE_ID = -1
N_RULES = 8
RULE_NAMES = [
    "factorization", "distribution", "reassociation",
    "const_folding", "common_factor", "depth_reduction",
    "collection", "cse_rewrite",
]

LAMBDA_SIZE, LAMBDA_DEPTH, LAMBDA_MUL, LAMBDA_TEMP, LAMBDA_DEG = 0.20, 0.50, 0.15, 0.10, 0.05


# ---------------------------------------------------------------------------
# DAGConverter (mirrored from notebook Section 4)
# ---------------------------------------------------------------------------

class DAGConverter:
    def __init__(self):
        self._reset()

    def _reset(self):
        self._ctr = 0
        self._memo = {}

    @staticmethod
    def _gate_type(e):
        if isinstance(e, sp.Symbol): return "var"
        if isinstance(e, (sp.Integer, sp.Float, sp.Rational, sp.Number)): return "const"
        if isinstance(e, sp.Add): return "add"
        if isinstance(e, sp.Mul): return "mul"
        if isinstance(e, sp.Pow): return "pow"
        return "other"

    @staticmethod
    def _edge_type(parent_gtype, child_index):
        if parent_gtype == "add": return 0
        if parent_gtype == "mul": return 1
        if parent_gtype == "pow": return 2 if child_index == 0 else 3
        return 4

    def _add(self, G, expr):
        key = hash(expr)
        if key in self._memo:
            return self._memo[key]

        nid = self._ctr
        self._ctr += 1
        gt = self._gate_type(expr)
        lbl = str(expr) if gt in ("var", "const") else {"add": "+", "mul": "×", "pow": "^"}.get(gt, gt)

        G.add_node(nid, node_type=gt, label=lbl, expr=expr, fan_out=0)
        self._memo[key] = nid

        for i, arg in enumerate(expr.args or []):
            cid = self._add(G, arg)
            et = self._edge_type(gt, i)
            G.add_edge(nid, cid, edge_type=et)
            G.nodes[cid]["fan_out"] = G.nodes[cid].get("fan_out", 0) + 1

        return nid

    def convert(self, expr):
        self._reset()
        G = nx.DiGraph()
        root = self._add(G, expr)
        G.graph["root"] = root
        return G


# ---------------------------------------------------------------------------
# ExprNormalizer (mirrored from notebook Section 3)
# ---------------------------------------------------------------------------

class ExprNormalizer:
    @staticmethod
    def constant_fold(e):
        if not e.free_symbols:
            try:
                v = float(e)
                return sp.Integer(int(v)) if v == int(v) else sp.Float(v)
            except Exception:
                return e
        return e.func(*[ExprNormalizer.constant_fold(a) for a in e.args]) if e.args else e

    @staticmethod
    def remove_identities(e):
        if isinstance(e, sp.Add):
            args = [ExprNormalizer.remove_identities(a) for a in e.args if a != sp.S.Zero]
            return sp.Add(*args) if args else sp.S.Zero
        if isinstance(e, sp.Mul):
            if sp.S.Zero in e.args:
                return sp.S.Zero
            args = [ExprNormalizer.remove_identities(a) for a in e.args if a != sp.S.One]
            return sp.Mul(*args) if args else sp.S.One
        if isinstance(e, sp.Pow):
            b, ex = e.args
            if ex == 0:
                return sp.S.One
            if ex == 1:
                return ExprNormalizer.remove_identities(b)
            return sp.Pow(ExprNormalizer.remove_identities(b), ex)
        return e

    def normalize_for_cost(self, e):
        s = self.constant_fold(e)
        s = self.remove_identities(s)
        return s


# ---------------------------------------------------------------------------
# FeatureExtractor (mirrored from notebook Section 5)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    GATE_TYPES = {"var": 0, "const": 1, "add": 2, "mul": 3, "pow": 4, "other": 5}
    N_TYPES = 6

    @staticmethod
    def _bfs_depths(G, root):
        return dict(nx.single_source_shortest_path_length(G, root))

    @staticmethod
    def _depth_to_output(G, root):
        rev = G.reverse(copy=False)
        return dict(nx.single_source_shortest_path_length(rev, root))

    @staticmethod
    def _subtree_sizes(G):
        sz = {}
        for n in reversed(list(nx.topological_sort(G))):
            ch = list(G.successors(n))
            sz[n] = 1 + sum(sz.get(c, 1) for c in ch)
        return sz

    @staticmethod
    def _subtree_muls(G):
        muls = {}
        for n in reversed(list(nx.topological_sort(G))):
            ch = list(G.successors(n))
            mine = 1 if G.nodes[n].get("node_type") == "mul" else 0
            muls[n] = mine + sum(muls.get(c, 0) for c in ch)
        return muls

    @staticmethod
    def _subtree_ops(G):
        ops = {}
        for n in reversed(list(nx.topological_sort(G))):
            ch = list(G.successors(n))
            mine = 1 if G.nodes[n].get("node_type") not in ("var", "const") else 0
            ops[n] = mine + sum(ops.get(c, 0) for c in ch)
        return ops

    @staticmethod
    def _depth_to_leaf(G):
        dtl = {}
        for n in nx.topological_sort(G):
            ch = list(G.successors(n))
            dtl[n] = 0 if not ch else 1 + max(dtl.get(c, 0) for c in ch)
        return dtl

    @staticmethod
    def _poly_degree(expr):
        if expr is None:
            return 0.0
        try:
            return float(min(int(sp.total_degree(expr)), 12))
        except Exception:
            return 0.0

    @staticmethod
    def _local_degree(G, n):
        nt = G.nodes[n].get("node_type", "other")
        if nt == "mul":
            return float(G.out_degree(n))
        if nt == "pow":
            for c in G.successors(n):
                if G.nodes[c].get("node_type") == "const":
                    try:
                        return float(abs(G.nodes[c].get("expr", 2.0)))
                    except Exception:
                        return 2.0
            return 2.0
        return 1.0

    @staticmethod
    def _subtree_max_degree(G, n):
        degs = [FeatureExtractor._poly_degree(G.nodes[x].get("expr"))
                for x in nx.descendants(G, n)]
        degs.append(FeatureExtractor._poly_degree(G.nodes[n].get("expr")))
        return max(degs, default=0.0)

    @staticmethod
    def _estimated_blowup(G, n, sizes):
        nt = G.nodes[n].get("node_type", "other")
        sz = sizes.get(n, 1)
        mx = max(sizes.values(), default=1)
        if nt == "mul":
            return float(np.clip(sz / mx * 1.5, 0, 1))
        if nt == "add" and sz > 3:
            return float(np.clip(0.3 + sz / mx, 0, 1))
        return float(np.clip(sz / (2 * mx), 0, 0.5))

    def extract(self, G):
        root = G.graph.get("root", next(iter(nx.topological_sort(G))))
        depths_root = self._bfs_depths(G, root)
        depths_out = self._depth_to_output(G, root)
        sizes = self._subtree_sizes(G)
        muls = self._subtree_muls(G)
        ops = self._subtree_ops(G)
        dtl = self._depth_to_leaf(G)

        mx_d = max(depths_root.values(), default=1) or 1
        mx_do = max(depths_out.values(), default=1) or 1
        mx_sz = max(sizes.values(), default=1) or 1
        mx_fo = max((G.in_degree(n) for n in G.nodes()), default=1) or 1
        mx_fi = max((G.out_degree(n) for n in G.nodes()), default=1) or 1
        mx_dtl = max(dtl.values(), default=1) or 1
        mx_subdeg = 12.0

        node_list = sorted(G.nodes())
        rows = []
        for n in node_list:
            nd = G.nodes[n]
            gt = nd.get("node_type", "other")
            oh = [0] * 6
            oh[self.GATE_TYPES.get(gt, 5)] = 1
            sz_n = sizes.get(n, 1)
            raw_fo = G.in_degree(n)

            parent_op_type = 5.0
            parents = list(G.predecessors(n))
            if parents:
                pt = G.nodes[parents[0]].get("node_type", "other")
                parent_op_type = float(self.GATE_TYPES.get(pt, 5))
            parent_op_norm = parent_op_type / max(self.N_TYPES - 1, 1)

            feat = oh + [
                depths_root.get(n, 0) / mx_d,
                G.out_degree(n) / max(mx_fi, 1),
                raw_fo / max(mx_fo, 1),
                sz_n / mx_sz,
                1.0 if raw_fo > 1 else 0.0,
                min(self._local_degree(G, n) / 6, 1.0),
                self._subtree_max_degree(G, n) / mx_subdeg,
                depths_out.get(n, 0) / mx_do,
                dtl.get(n, 0) / mx_dtl,
                ops.get(n, 0) / max(sz_n, 1),
                muls.get(n, 0) / max(sz_n, 1),
                float(min(raw_fo, 10)) / 10,
                self._estimated_blowup(G, n, sizes),
                parent_op_norm,
            ]
            rows.append(feat)

        X = np.array(rows, dtype=np.float32)
        edges = list(G.edges(data=True))
        if edges:
            id_map = {n: i for i, n in enumerate(node_list)}
            ei_rows, ea_rows = [], []
            for u, v, d in edges:
                ei_rows.append([id_map[u], id_map[v]])
                etype = d.get("edge_type", 4)
                ea = [0] * 5
                ea[min(etype, 4)] = 1
                ea_rows.append(ea)
            edge_index = np.array(ei_rows, dtype=np.int64).T
            edge_attr = np.array(ea_rows, dtype=np.float32)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 5), dtype=np.float32)
        return node_list, X, edge_index, edge_attr


# ---------------------------------------------------------------------------
# CostModel (lightweight, mirrored from notebook Section 5b)
# ---------------------------------------------------------------------------

class CostModel:
    def __init__(self, lam_s=LAMBDA_SIZE, lam_d=LAMBDA_DEPTH, lam_m=LAMBDA_MUL,
                 lam_t=LAMBDA_TEMP, lam_g=LAMBDA_DEG):
        self.lam_s, self.lam_d = lam_s, lam_d
        self.lam_m, self.lam_t, self.lam_g = lam_m, lam_t, lam_g

    @staticmethod
    def graph_metrics(G):
        root = G.graph.get("root", next(iter(nx.topological_sort(G))))
        depths = nx.single_source_shortest_path_length(G, root)
        return {
            "size": G.number_of_nodes(),
            "depth": max(depths.values(), default=0),
            "muls": sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "mul"),
            "degree": max(
                [int(sp.total_degree(G.nodes[n].get("expr", 0)))
                 for n in G.nodes() if G.nodes[n].get("expr") is not None],
                default=0,
            ),
        }

    def score_graph(self, G, temp_blowup=0.0):
        m = self.graph_metrics(G)
        return (self.lam_s * m["size"] + self.lam_d * m["depth"] +
                self.lam_m * m["muls"] + self.lam_t * temp_blowup +
                self.lam_g * m["degree"])


# ---------------------------------------------------------------------------
# Oracle -> PyG Data conversion
# ---------------------------------------------------------------------------

def oracle_record_to_pyg_data(record, dag_conv, feat_ext):
    """
    Convert an oracle record to a PyG Data object.

    Whole-expression rewrite (global scope): target_node = GLOBAL_NODE_ID (-1) -> index 0.
    """
    expr = record["expr"]
    target_expr = record["target_expr"]
    rule_idx = record["rule_idx"]
    improvement = record["improvement"]
    node_reducible = record["node_reducible"]

    norm = ExprNormalizer().normalize_for_cost(expr)
    G = dag_conv.convert(norm)

    if G.number_of_nodes() < 2:
        return None

    node_list, X, edge_index, edge_attr = feat_ext.extract(G)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    data.node_list = node_list

    # GLOBAL_NODE_ID -> index 0 (whole-expression rewrite)
    data.target_node = torch.tensor([0], dtype=torch.long)
    data.target_rule = torch.tensor([int(rule_idx)], dtype=torch.long)
    data.improvement = torch.tensor([float(improvement)], dtype=torch.float)
    data.node_reducible = torch.tensor([float(node_reducible)], dtype=torch.float)
    data.confidence = torch.tensor([float(improvement)], dtype=torch.float)
    data.in_top_k = torch.tensor([1.0], dtype=torch.float)

    return data


def main():
    parser = argparse.ArgumentParser(description="Integrate oracle rule data into PyG corpus")
    parser.add_argument("--n-per-rule", type=int, default=5000,
                        help="Number of oracle expressions per rule")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--pyg-ckpt", default="data/pyg_data_list.pkl",
                        help="Path to existing pyg_data_list.pkl")
    parser.add_argument("--output", default="data/pyg_data_list_with_oracle.pkl",
                        help="Output path for combined corpus")
    args = parser.parse_args()

    # Generate oracle records
    print(f"Generating {args.n_per_rule} oracle records per rule...")
    oracle_records = generate_oracle_dataset(
        n_per_rule=args.n_per_rule, seed=args.seed
    )
    print(f"Generated {len(oracle_records)} total oracle records")

    # Initialize pipeline components
    dag_conv = DAGConverter()
    feat_ext = FeatureExtractor()

    # Convert oracle records to PyG Data
    print("Converting oracle records to PyG Data format...")
    oracle_data_list = []
    skipped = 0
    for rec in oracle_records:
        try:
            data = oracle_record_to_pyg_data(rec, dag_conv, feat_ext)
            if data is not None:
                oracle_data_list.append(data)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"  Skipped record: {e}")

    print(f"Converted {len(oracle_data_list)} oracle records (skipped {skipped})")

    # Load existing PyG corpus
    pyg_path = Path(args.pyg_ckpt)
    if pyg_path.exists():
        print(f"Loading existing PyG corpus from {pyg_path}...")
        with open(pyg_path, "rb") as f:
            existing_data = pickle.load(f)
        print(f"Existing corpus: {len(existing_data)} samples")
    else:
        print(f"No existing corpus found at {pyg_path}, starting fresh")
        existing_data = []

    # Combine
    combined = existing_data + oracle_data_list
    print(f"Combined corpus: {len(combined)} samples")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved combined corpus to {output_path}")

    # Print rule distribution
    rule_counts = Counter()
    for d in combined:
        if hasattr(d, "target_rule"):
            rule_counts[int(d.target_rule.item())] += 1
    print("\nRule distribution in combined corpus:")
    for rule_idx in range(N_RULES):
        name = RULE_NAMES[rule_idx] if rule_idx < len(RULE_NAMES) else f"rule_{rule_idx}"
        count = rule_counts.get(rule_idx, 0)
        print(f"  {name:<20} {count:>6}")


if __name__ == "__main__":
    main()
