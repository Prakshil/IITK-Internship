"""
Re-run benchmark with retrained GIN model (Step 1.5).

Loads gin_best_fixed.pt and evaluates on 500 held-out expressions,
producing per-family results comparable to the baseline.

Usage:
    python rebenchmark_fixed.py --n-eval 500 --seed 42 --device cuda
"""

import argparse
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load all needed classes from the notebook pipeline
from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID

import networkx as nx
import sympy as sp

# Import the CircuitOptimizer and related classes by loading them from notebook
# We'll replicate them here for standalone execution
LAMBDA_SIZE, LAMBDA_DEPTH, LAMBDA_MUL, LAMBDA_TEMP, LAMBDA_DEG = 0.20, 0.50, 0.15, 0.10, 0.05


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
        G.add_node(nid, node_type=gt, label=str(expr), expr=expr, fan_out=0)
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


class ExprNormalizer:
    @staticmethod
    def normalize_for_cost(e):
        if not e.free_symbols:
            try:
                v = float(e)
                return sp.Integer(int(v)) if v == int(v) else sp.Float(v)
            except Exception:
                return e
        return e.func(*[ExprNormalizer.normalize_for_cost(a) for a in e.args]) if e.args else e


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

    def score_expr(self, expr, temp_blowup=0.0, dag_conv=None, normalizer=None):
        dc = dag_conv or DAGConverter()
        norm = (normalizer or ExprNormalizer()).normalize_for_cost(expr)
        return self.score_graph(dc.convert(norm), temp_blowup)


# Copy of GINRewriteRanker for inference
import torch
import torch.nn as nn
import torch.nn.functional as F


class GINRewriteRanker(nn.Module):
    def __init__(self, in_ch=20, hidden=128, n_rules=N_RULES,
                 edge_dim=5, dropout=0.2):
        super().__init__()
        self.dropout_p = dropout
        self.n_rules = n_rules

        def mlp(i, o):
            return nn.Sequential(
                nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o), nn.BatchNorm1d(o))

        try:
            from torch_geometric.nn import GINEConv
            self.conv1 = GINEConv(mlp(in_ch, hidden), edge_dim=edge_dim)
            self.conv2 = GINEConv(mlp(hidden, hidden), edge_dim=edge_dim)
            self.conv3 = GINEConv(mlp(hidden, hidden), edge_dim=edge_dim)
            self._use_edge = True
        except Exception:
            from torch_geometric.nn import GINConv
            self.conv1 = GINConv(mlp(in_ch, hidden))
            self.conv2 = GINConv(mlp(hidden, hidden))
            self.conv3 = GINConv(mlp(hidden, hidden))
            self._use_edge = False

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.node_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rule_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_rules))
        self.conf_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def _conv(self, conv, x, edge_index, edge_attr):
        if self._use_edge and edge_attr is not None and edge_attr.shape[0] > 0:
            return conv(x, edge_index, edge_attr)
        if self._use_edge:
            ea = torch.zeros(edge_index.shape[1], 5, device=x.device)
            return conv(x, edge_index, ea)
        return conv(x, edge_index)

    def encode(self, x, edge_index, edge_attr=None):
        x = F.relu(self.bn1(self._conv(self.conv1, x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn2(self._conv(self.conv2, x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self._conv(self.conv3, x, edge_index, edge_attr)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        embs = self.encode(x, edge_index, edge_attr)
        return (self.node_head(embs),
                self.rule_head(embs),
                self.conf_head(embs))

    @staticmethod
    def _global_target_nodes(batch):
        t = batch.target_node.long().view(-1)
        if hasattr(batch, "ptr") and batch.ptr is not None:
            return t + batch.ptr[:-1].to(t.device)
        return t

    def rank_candidates(self, data, candidates, device=None):
        self.eval()
        dev = device or next(self.parameters()).device
        data = data.to(dev)
        ea = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        node_logits, rule_logits, conf = self(data.x, data.edge_index, edge_attr=ea)

        node_p = torch.sigmoid(node_logits).squeeze(-1).detach().cpu().numpy()
        rule_p = torch.softmax(rule_logits, dim=-1).detach().cpu().numpy()
        conf_p = torch.sigmoid(conf).squeeze(-1).detach().cpu().numpy()

        ranked = []
        for c in candidates:
            nid = c.get("node_id", GLOBAL_NODE_ID)
            ridx = c.get("rule_idx", 0)
            ni = self._node_index(data, nid)
            rp = float(rule_p[ni, ridx]) if ridx < rule_p.shape[1] else 0.0
            np_ = float(node_p[ni]) if ni < len(node_p) else 0.5
            cp = float(conf_p[ni]) if ni < len(conf_p) else 0.5
            score = np_ * rp * cp
            ranked.append({**c, "gin_rank_score": score, "gin_node_prob": np_,
                           "gin_rule_prob": rp, "gin_conf": cp})
        ranked.sort(key=lambda d: d["gin_rank_score"], reverse=True)
        return {"ranked": ranked, "node_probs": node_p, "rule_probs": rule_p, "conf": conf_p}

    def _node_index(self, data, node_id):
        nl = getattr(data, "node_list", None)
        if nl is None:
            return 0 if node_id == GLOBAL_NODE_ID else max(0, int(node_id))
        if node_id == GLOBAL_NODE_ID:
            return 0
        if node_id not in nl:
            raise ValueError(f"node_id {node_id} not found in node_list")
        return nl.index(node_id)


# FeatureExtractor (standalone copy)
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

    def to_pyg_data(self, G, target_node=None, target_rule=None, improvement=None,
                    node_reducible=None, confidence=None, in_top_k=None):
        import torch_geometric.data
        node_list, X, ei, ea = self.extract(G)
        data = torch_geometric.data.Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=torch.tensor(ei, dtype=torch.long),
            edge_attr=torch.tensor(ea, dtype=torch.float),
        )
        data.node_list = node_list
        if target_node is not None:
            idx = 0 if target_node == GLOBAL_NODE_ID else (
                node_list.index(target_node) if target_node in node_list else 0)
            data.target_node = torch.tensor([idx], dtype=torch.long)
        if target_rule is not None:
            data.target_rule = torch.tensor([int(target_rule)], dtype=torch.long)
        if improvement is not None:
            data.improvement = torch.tensor([float(improvement)], dtype=torch.float)
        if node_reducible is not None:
            data.node_reducible = torch.tensor([float(node_reducible)], dtype=torch.float)
        if confidence is not None:
            data.confidence = torch.tensor([float(confidence)], dtype=torch.float)
        if in_top_k is not None:
            data.in_top_k = torch.tensor([float(in_top_k)], dtype=torch.float)
        return data


# CandidateGenerator (standalone copy with oracle rules)
import math


class CandidateGenerator:
    def __init__(self, timeout=1.5, cost_model=None):
        self.timeout = timeout
        self._dc = DAGConverter()
        self._norm = ExprNormalizer()
        self._cm = cost_model

        self._rule_fns = [
            (self._factorize,      0, "factorization"),
            (self._distribute,     1, "distribution"),
            (self._reassociate,    2, "reassociation"),
            (self._const_fold,     3, "const_folding"),
            (self._common_factor,  4, "common_factor"),
            (self._depth_reduce,   5, "depth_reduction"),
            (self._collection,     6, "collection"),
            (self._cse_rewrite,    7, "cse_rewrite"),
        ]

        self._graph_cache = {}
        self._score_cache = {}
        self._lookahead_cache = {}

    def _expr_key(self, expr):
        return hash(expr)

    def _graph_from_expr_cached(self, expr):
        key = hash(expr)
        if key not in self._graph_cache:
            norm = self._norm.normalize_for_cost(expr)
            self._graph_cache[key] = self._dc.convert(norm)
        return self._graph_cache[key]

    def _score_graph_from_expr_cached(self, expr, cm):
        key = hash(expr)
        if key not in self._score_cache:
            G = self._graph_from_expr_cached(expr)
            self._score_cache[key] = cm.score_graph(G)
        return self._score_cache[key]

    def _safe(self, fn, e):
        try:
            r = fn(e)
            return r if (r is not None and r != e) else None
        except Exception:
            return None

    @staticmethod
    def _balance_args(args, op):
        if len(args) <= 1:
            return args[0] if args else None
        if len(args) == 2:
            return op(args[0], args[1], evaluate=False)
        mid = len(args) // 2
        return op(
            CandidateGenerator._balance_args(args[:mid], op),
            CandidateGenerator._balance_args(args[mid:], op),
            evaluate=False,
        )

    def _identity(self, e):
        return None

    def _dead_pruning(self, e):
        return None

    def _collection(self, e):
        try:
            if not isinstance(e, sp.Add):
                return None
            free_symbols = list(e.free_symbols)
            if len(free_symbols) == 0:
                return None
            result = sp.collect(e, free_symbols)
            if result != e:
                return result
            return None
        except Exception:
            return None

    def _reassociate(self, e):
        def _flatten(expr, Op):
            if type(expr) is not Op:
                return [expr]
            leaves = []
            for arg in expr.args:
                leaves.extend(_flatten(arg, Op))
            return leaves
        for OpType in [sp.Add, sp.Mul]:
            if isinstance(e, OpType):
                args = _flatten(e, OpType)
                if len(args) >= 3:
                    r = self._balance_args(args, OpType)
                    if r is not None and sp.srepr(r) != sp.srepr(e):
                        return r
        return None

    def _depth_reduce(self, e):
        try:
            r1 = sp.powsimp(sp.combsimp(e), force=True)
            if r1 != e and sp.count_ops(r1) <= sp.count_ops(e):
                return r1
            r2 = sp.collect(sp.expand(e), sorted(e.free_symbols, key=str))
            if r2 != e and sp.count_ops(r2) < sp.count_ops(e):
                return r2
            r3 = sp.radsimp(e)
            if r3 != e and sp.count_ops(r3) < sp.count_ops(e):
                return r3
            return None
        except Exception:
            return None

    def _factorize(self, e):
        return self._safe(sp.factor, e)

    def _distribute(self, e):
        return self._safe(sp.expand, e)

    def _const_fold(self, e):
        try:
            if not e.free_symbols:
                try:
                    val = complex(e)
                    if val.imag == 0 and float(val.real) == int(float(val.real)):
                        return sp.Integer(int(float(val.real)))
                except Exception:
                    pass
                return sp.nsimplify(e, rational=True)
            ex = sp.expand(e)
            if isinstance(ex, sp.Add):
                terms = list(ex.args)
                numeric_coeffs = []
                for t in terms:
                    if t.is_number:
                        numeric_coeffs.append(abs(t))
                    elif isinstance(t, sp.Mul):
                        nums = [a for a in t.args if a.is_number and a != sp.S.NegativeOne]
                        numeric_coeffs.append(abs(nums[0]) if nums else sp.S.One)
                    else:
                        numeric_coeffs.append(sp.S.One)
                g = reduce(lambda a, b: sp.gcd(a, b), numeric_coeffs)
                if g.is_Integer and int(g) >= 2:
                    factored = g * sp.Add(*[sp.cancel(t / g) for t in terms])
                    factored = sp.simplify(factored)
                    if factored != e and sp.count_ops(factored) <= sp.count_ops(e):
                        return factored
            return None
        except Exception:
            return None

    def _common_factor(self, e):
        try:
            ex = sp.expand(e)
            if not isinstance(ex, sp.Add):
                return None
            terms = list(ex.args)
            common = reduce(lambda a, b: sp.gcd(a, b), terms)
            if common not in (sp.S.One, sp.S.Zero, 1):
                factored = common * sp.Add(*[sp.cancel(t / common) for t in terms])
                r = sp.simplify(factored)
                if r != e:
                    return r
            for sym in ex.free_symbols:
                r = sp.collect(ex, sym)
                if r != ex:
                    return r
            return None
        except Exception:
            return None

    def _cse_rewrite(self, e):
        try:
            from sympy import cse
            reps, red = cse([e])
            if not reps:
                return None
            dominant_sub = reps[0][1]
            if dominant_sub.is_Symbol or dominant_sub.is_Number:
                return None
            result = sp.collect(sp.expand(e), dominant_sub)
            if result != e and sp.count_ops(result) <= sp.count_ops(e):
                return result
            result2 = sp.factor(e)
            if result2 != e and sp.count_ops(result2) < sp.count_ops(e):
                return result2
            return None
        except Exception:
            return None

    def _build_action(self, full_expr, G, node_id, ridx, rname, Cp_expr, seen):
        h = hash(Cp_expr)
        rule_key = (ridx, h)
        if rule_key in seen:
            return None
        seen.add(rule_key)
        if node_id == GLOBAL_NODE_ID:
            nd = {'label': 'GLOBAL'}
        else:
            nd = G.nodes[node_id]
        return {
            "node_id": node_id,
            "rule": rname,
            "rule_idx": ridx,
            "rule_name": rname,
            "candidate_expr": Cp_expr,
            "expr": Cp_expr,
            "candidate_graph": None,
            "scope": "local",
            "node_label": nd.get("label", "?"),
        }

    def _apply_rule_at_node(self, full_expr, G, node_id, fn, ridx, rname, seen):
        sub = G.nodes[node_id].get("expr")
        if sub is None:
            return None
        result = fn(sub)
        if result is None or result == sub:
            return None
        try:
            n_occ = full_expr.count(sub)
        except Exception:
            n_occ = 1
        if n_occ == 0:
            return None
        Cp_expr = full_expr.subs(sub, result)
        if Cp_expr == full_expr:
            return None
        effective_node_id = GLOBAL_NODE_ID if (n_occ > 1 and G.nodes[node_id].get("fan_out", 0) > 1) else node_id
        return self._build_action(full_expr, G, effective_node_id, ridx, rname, Cp_expr, seen)

    def generate_local_actions(self, expr, G):
        actions, seen = [], set()
        reducible = [
            n for n in G.nodes()
            if G.nodes[n].get("node_type") in ("add", "mul", "pow")
            and getattr(G.nodes[n].get("expr"), "args", None)
        ]
        for node_id in reducible:
            sub = G.nodes[node_id].get("expr")
            if sub is None:
                continue
            for fn, ridx, rname in self._rule_fns:
                if rname == "identity":
                    continue
                a = self._apply_rule_at_node(expr, G, node_id, fn, ridx, rname, seen)
                if a:
                    actions.append(a)
        return actions

    def generate_global_actions(self, expr, G):
        actions = []
        seen = set()
        for fn, ridx, rname in self._rule_fns:
            result = fn(expr)
            if result is None or result == expr:
                continue
            Cp_expr = result
            node_id = GLOBAL_NODE_ID
            actions.append(self._build_action(expr, G, node_id, ridx, rname, Cp_expr, seen))
        return [a for a in actions if a is not None]

    def generate_all_actions(self, expr, G):
        ck = hash(expr)
        ck_full = (ck, "v1")
        if ck_full in getattr(self, "_candidate_cache", {}):
            return self._candidate_cache[ck_full]
        out = self.generate_global_actions(expr, G) + self.generate_local_actions(expr, G)
        if not hasattr(self, "_candidate_cache"):
            self._candidate_cache = {}
        self._candidate_cache[ck_full] = out
        return out

    def label_candidates(self, original, candidates, G=None, cost_model=None, lookahead=None, is_opt=False):
        cm = cost_model or self._cm
        if cm is None:
            cm = CostModel()
            self._cm = cm
        if G is None:
            G = self._graph_from_expr_cached(original)

        old_score = cm.score_graph(G)

        def delta_norm(old_s, new_s):
            return max(0.0, (old_s - new_s) / max(abs(old_s), 1e-6))

        for c in candidates:
            ce = c["candidate_expr"]
            new_score = self._score_graph_from_expr_cached(ce, cm)
            c["immediate_gain"] = old_score - new_score

        candidates = sorted(candidates, key=lambda x: x["immediate_gain"], reverse=True)

        ranked = sorted(candidates, key=lambda x: x["immediate_gain"], reverse=True)

        labeled = []
        for i, c in enumerate(ranked):
            immediate_norm = delta_norm(old_score, old_score - c["immediate_gain"])
            if immediate_norm > 0:
                _, best_future_cost, _ = self.lookahead_score(
                    c["candidate_expr"], cm, self, depth=3, beam=5
                )
                future_norm = delta_norm(old_score, best_future_cost)
            else:
                future_norm = 0.0

            imp_norm = 0.5 * immediate_norm + 0.5 * future_norm
            c["future_gain"] = future_norm * max(abs(old_score), 1e-6)
            c["combined_gain"] = imp_norm

            labeled.append({
                **c,
                "old_score": old_score,
                "new_score": old_score - c["immediate_gain"],
                "score_improvement": c["immediate_gain"],
                "future_improvement": c["future_gain"],
                "improvement": imp_norm,
                "delta_vr": imp_norm,
                "is_improvement": imp_norm > 0.0,
            })

        return labeled

    def lookahead_score(self, expr, cost_model, cand_gen, depth=3, beam=5):
        cm = cost_model
        key = (hash(expr), depth, beam)
        if key in self._lookahead_cache:
            return self._lookahead_cache[key]

        current = [(expr, self._score_graph_from_expr_cached(expr, cm), [])]
        best_expr = expr
        best_cost = current[0][1]
        best_path = []

        for _ in range(depth):
            next_states = []
            for e, score, path in current:
                try:
                    G = self._graph_from_expr_cached(e)
                    cands = self.generate_all_actions(e, G)
                except Exception:
                    continue

                labeled = []
                for c in cands:
                    try:
                        ce = c["candidate_expr"]
                        c_score = self._score_graph_from_expr_cached(ce, cm)
                        score_imp = score - c_score
                        c["score_improvement"] = score_imp
                        labeled.append(c)
                    except Exception:
                        pass

                labeled = sorted(labeled, key=lambda x: x.get("score_improvement", 0), reverse=True)

                for c in labeled[:beam]:
                    ne = c["candidate_expr"]
                    try:
                        ncost = self._score_graph_from_expr_cached(ne, cm)
                    except Exception:
                        continue

                    new_path = path + [c]
                    next_states.append((ne, ncost, new_path))

                    if ncost < best_cost:
                        best_cost = ncost
                        best_expr = ne
                        best_path = new_path

            current = sorted(next_states, key=lambda t: t[1])[:beam]

        result = (best_expr, best_cost, best_path)
        self._lookahead_cache[key] = result
        return result


# CircuitOptimizer (simplified for benchmarking)
import threading


class CircuitOptimizer:
    def __init__(self, model, max_iters=8, gin_top_k=10, beam_width=8,
                 patience=5, mode="v4_full"):
        self.model = model
        self.max_iters = max_iters
        self.gin_top_k = gin_top_k
        self.beam_width = beam_width
        self.patience = patience
        self.mode = mode
        self._norm = ExprNormalizer()
        self._dag = DAGConverter()
        self._feat = FeatureExtractor()
        self._cgen = CandidateGenerator()
        self._cost = CostModel()
        self._gin_lock = threading.Lock()

    def score_for_mode(self, G, temp_blowup=0.0):
        return self._cost.score_graph(G, temp_blowup=temp_blowup)

    def _adaptive_beam_width(self, G):
        n = G.number_of_nodes()
        if n < 10:
            w = 3
        elif n < 30:
            w = 5
        else:
            w = 8
        return min(w, self.beam_width)

    def optimize(self, expr, verbose=False):
        best_expr = expr
        try:
            G0 = self._dag.convert(self._norm.normalize_for_cost(expr))
            best_cost = self.score_for_mode(G0)
        except:
            best_cost = float('inf')

        path = []
        states = [(expr, best_cost, best_cost, [])]
        orig_cost = best_cost

        n_beam_iters = 0
        for it in range(self.max_iters):
            n_beam_iters = it + 1
            expanded = []

            for e, cost, est_fut_cost, p in states:
                try:
                    ne = self._norm.normalize_for_cost(e)
                    G = self._dag.convert(ne)
                except:
                    continue

                beam_width = self._adaptive_beam_width(G)

                cands = self._cgen.generate_all_actions(ne, G)
                if not cands:
                    continue

                if self.model is not None:
                    data = self._feat.to_pyg_data(G)
                    ranked_cands = []
                    BATCH = 32
                    for i in range(0, len(cands), BATCH):
                        chunk = cands[i:i + BATCH]
                        with self._gin_lock:
                            res = self.model.rank_candidates(data, chunk, device=None)["ranked"]
                        ranked_cands.extend(res)
                    gin_ranked = ranked_cands
                else:
                    gin_ranked = cands

                # Deduplicate + diverse top-K
                seen_exprs = set()
                deduped = []
                for c in gin_ranked:
                    ce = str(c["candidate_expr"])
                    if ce not in seen_exprs:
                        seen_exprs.add(ce)
                        deduped.append(c)
                gin_topk = deduped[:self.gin_top_k]

                for c in gin_topk:
                    ne_cand = c["candidate_expr"]
                    try:
                        n_dag_nodes = G.number_of_nodes()
                        if not self._pit_verify(ne, ne_cand, n_nodes=n_dag_nodes):
                            continue
                        Gp_cand = self._dag.convert(self._norm.normalize_for_cost(ne_cand))
                        if Gp_cand.number_of_nodes() == 0:
                            continue
                        root_p = Gp_cand.graph.get("root")
                        if root_p is not None and root_p in Gp_cand.nodes():
                            ne_cand = Gp_cand.nodes[root_p].get("expr", ne_cand)
                        c["candidate_graph"] = Gp_cand
                        ncost = self.score_for_mode(Gp_cand)
                        c["score_improvement"] = cost - ncost
                    except:
                        continue

                    predicted_gain = c.get("gin_rank_score", c.get("score_improvement", 0))
                    est_future_cost = ncost - (predicted_gain * max(abs(ncost), 1.0))

                    npth = p + [c]
                    expanded.append((ne_cand, ncost, est_future_cost, npth))

                    if ncost < best_cost:
                        best_cost = ncost
                        best_expr = ne_cand
                        path = npth

            states = sorted(expanded, key=lambda t: t[2])[:beam_width]

            if verbose:
                print(f"  iter {it+1} | best_cost={best_cost:.2f} | path_len={len(path)}")

            if len(expanded) == 0:
                break

        try:
            final_G = self._dag.convert(self._norm.normalize_for_cost(best_expr))
            final_m = self._cost.graph_metrics(final_G)
        except:
            final_m = {"size": 0, "depth": 0}

        try:
            G0 = self._dag.convert(self._norm.normalize_for_cost(expr))
            base_m = self._cost.graph_metrics(G0)
        except:
            base_m = {"size": 0, "depth": 0}

        return {
            "expr": expr,
            "optimized": best_expr,
            "orig_cost": orig_cost,
            "final_cost": best_cost,
            "pct_reduction": 100.0 * (orig_cost - best_cost) / max(orig_cost, 1e-9),
            "delta_score": orig_cost - best_cost,
            "n_rewrites_accepted": len(path),
            "n_iters": len(path),
            "n_beam_iters": n_beam_iters,
            "path": path,
            "rewrite_sequence": [p.get("rule") for p in path],
            "accepted_rules": [p.get("rule") for p in path],
            "accepted_count": len(path),
            "size_reduction_frac": (base_m["size"] - final_m["size"]) / max(base_m["size"], 1),
            "depth_reduction": base_m["depth"] - final_m["depth"],
            "pit_ok": True,
        }

    def _pit_verify(self, orig_expr, cand_expr, n_nodes=None):
        key = (hash(orig_expr), hash(cand_expr), n_nodes, 15, 30, 2147483647, "v1")
        if key in getattr(self, "_pit_cache", {}):
            return self._pit_cache[key]

        if n_nodes is None:
            try:
                G = DAGConverter().convert(orig_expr)
                n_nodes = G.number_of_nodes()
            except Exception:
                n_nodes = 999

        syms = list(orig_expr.free_symbols | cand_expr.free_symbols)

        if n_nodes <= 15:
            try:
                res = sp.simplify(orig_expr - cand_expr) == 0
            except Exception:
                res = False
        else:
            if not syms:
                res = str(orig_expr) == str(cand_expr)
            else:
                rng = random.Random()
                res = True
                for _ in range(30):
                    subs = {s: rng.randint(1, 2147483646) for s in syms}
                    try:
                        val_orig = int(orig_expr.subs(subs)) % 2147483647
                        val_cand = int(cand_expr.subs(subs)) % 2147483647
                        if val_orig != val_cand:
                            res = False
                            break
                    except Exception:
                        res = False
                        break

        if not hasattr(self, "_pit_cache"):
            self._pit_cache = {}
        self._pit_cache[key] = res
        return res


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def load_dataset():
    """Load dataset from pickle (with temp copy for OneDrive fix)."""
    src = Path("data/dataset_100000_depth3.pkl")
    tmp_dir = Path("C:/Users/Prakshil/AppData/Local/Temp/opencode")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "dataset_100000_depth3.pkl"
    if not tmp_file.exists():
        print(f"Copying dataset to temp location...")
        shutil.copy2(src, tmp_file)

    print(f"Loading dataset from temp location...")
    with open(tmp_file, "rb") as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset):,} expressions")
    return dataset


import shutil


def main():
    parser = argparse.ArgumentParser(description="Re-run benchmark with retrained GIN")
    parser.add_argument("--n-eval", type=int, default=500, help="Number of expressions to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--model", default="models/gin_best_fixed.pt", help="Model checkpoint path")
    parser.add_argument("--baseline", default="data/baseline_results.csv", help="Baseline results CSV")
    parser.add_argument("--output", default="data/benchmark_retrained.csv", help="Output CSV path")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load dataset
    dataset = load_dataset()

    # Sample held-out expressions
    eval_records = random.sample(dataset, min(args.n_eval, len(dataset)))
    print(f"Evaluating on {len(eval_records)} expressions")

    # Load retrained model
    model = GINRewriteRanker().to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from {args.model} (epoch {checkpoint.get('epoch', '?')})")

    # Create optimizer with retrained model
    optimizer = CircuitOptimizer(model=model, max_iters=8, gin_top_k=10, beam_width=8)

    # Run benchmark
    results = []
    t_start = time.time()

    for i, rec in enumerate(eval_records):
        try:
            res = optimizer.optimize(rec["expr"], verbose=False)
            results.append({
                "family": rec["family"],
                "orig_cost": res["orig_cost"],
                "final_cost": res["final_cost"],
                "delta_score": res["delta_score"],
                "pct_reduction": res["pct_reduction"],
                "size_reduction_frac": res.get("size_reduction_frac", 0.0),
                "depth_reduction": res.get("depth_reduction", 0.0),
                "n_iters": res.get("n_rewrites_accepted", 0),
                "accepted_count": res.get("accepted_count", 0),
                "accepted_rules": res.get("accepted_rules", []),
            })
        except Exception as e:
            print(f"  Failed on {rec['family']}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(eval_records)} completed")

    t_elapsed = time.time() - t_start
    print(f"\nBenchmark completed in {t_elapsed:.1f}s ({t_elapsed/len(results):.2f}s/expr)")

    # Create DataFrame
    df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS (Retrained GIN)")
    print(f"{'='*60}")
    print(f"Expressions: {len(df):,}")
    print(f"Mean % reduction: {df['pct_reduction'].mean():.2f}%")
    print(f"Median % reduction: {df['pct_reduction'].median():.2f}%")
    print(f"Std % reduction: {df['pct_reduction'].std():.2f}%")
    print(f"Min % reduction: {df['pct_reduction'].min():.2f}%")
    print(f"Max % reduction: {df['pct_reduction'].max():.2f}%")

    print(f"\nBy Family:")
    family_stats = df.groupby("family")[["pct_reduction"]].agg(["mean", "median", "count"]).round(2)
    print(family_stats.to_string())

    # Count zero-improvement families
    zero_families = family_stats[family_stats[("pct_reduction", "mean")] == 0.0].index.tolist()
    print(f"\nZero-improvement families: {len(zero_families)}")
    for f in zero_families:
        print(f"  - {f}")

    # Compare with baseline
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH BASELINE")
    print(f"{'='*60}")
    baseline_df = pd.read_csv(args.baseline)
    baseline_mean = baseline_df["mean_reduction"].mean()
    retrained_mean = df["pct_reduction"].mean()
    print(f"Baseline mean:  {baseline_mean:.2f}%")
    print(f"Retrained mean: {retrained_mean:.2f}%")
    print(f"Improvement:    {retrained_mean - baseline_mean:+.2f} percentage points")

    # Rule usage
    from collections import Counter
    rule_counts = Counter()
    for rules in df["accepted_rules"]:
        for rule in rules:
            rule_counts[rule] += 1
    print(f"\nRule usage:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule:<20} {count:>6}")


if __name__ == "__main__":
    main()
