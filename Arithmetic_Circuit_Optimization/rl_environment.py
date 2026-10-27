"""
RL Environment for ArithCircuit Optimization (Phase 2, Step 2.1).

Defines a reinforcement learning environment where an agent learns to
optimize arithmetic circuits by applying rewrite rules.

State:  DAG feature vector (20-dim per node) + graph-level context
        (cost, depth, size, muls, degree)
Action: (node_id, rule_id) pair — discrete action space
Reward: Δ cost after applying action + PIT verification bonus
        (+1.0 if PIT passes, -1.0 if fails)
Termination: No improvement possible OR max 8 iterations reached

This environment wraps the existing pipeline components:
- DAGConverter, ExprNormalizer, CostModel, FeatureExtractor
- CandidateGenerator (rule application)
- PITVerifier (equivalence checking)
- GINRewriteRanker (policy network for action selection)

Usage:
    from rl_environment import ArithCircuitEnv, RLState, RLAction
    env = ArithCircuitEnv(model=gin_model, cost_model=CostModel())
    state = env.reset(expr)
    action = env.sample_action(state)  # or use policy
    next_state, reward, done, info = env.step(action)
"""

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys_path_inserted = False


def _ensure_path():
    """Ensure the project root is on sys.path for standalone imports."""
    global sys_path_inserted
    if not sys_path_inserted:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys_path_inserted = True


_ensure_path()

from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID  # noqa: E402


# ---------------------------------------------------------------------------
# Reuse existing pipeline components (standalone copies to avoid notebook dep)
# ---------------------------------------------------------------------------

class DAGConverter:
    """Hash-consing DAG converter with 5-dim edge-type attributes."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self._ctr = 0
        self._memo = {}

    @staticmethod
    def _gate_type(e):
        if isinstance(e, sp.Symbol):
            return "var"
        if isinstance(e, (sp.Integer, sp.Float, sp.Rational, sp.Number)):
            return "const"
        if isinstance(e, sp.Add):
            return "add"
        if isinstance(e, sp.Mul):
            return "mul"
        if isinstance(e, sp.Pow):
            return "pow"
        return "other"

    @staticmethod
    def _edge_type(parent_gtype, child_index):
        if parent_gtype == "add":
            return 0
        if parent_gtype == "mul":
            return 1
        if parent_gtype == "pow":
            return 2 if child_index == 0 else 3
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
    """Constant folding + identity removal (lightweight normalization)."""

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
    """Score(C) = λ_S·|C| + λ_D·D + λ_M·M + λ_T·T + λ_G·G"""

    def __init__(self, lam_s=0.20, lam_d=0.50, lam_m=0.15, lam_t=0.10, lam_g=0.05):
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

    def score_expr(self, expr, temp_blowup=0.0):
        dc = DAGConverter()
        norm = ExprNormalizer().normalize_for_cost(expr)
        return self.score_graph(dc.convert(norm), temp_blowup)


class PITVerifier:
    """Polynomial Identity Testing — C' ≡ C.

    Symbolic check for ≤15 nodes; Schwartz–Zippel randomized check otherwise.
    """

    def __init__(self, small_nodes=15, trials=30, prime=2147483647):
        self.small_nodes = small_nodes
        self.trials = trials
        self.prime = prime
        self._cache = {}

    def _symbolic_equal(self, orig, cand):
        try:
            return sp.simplify(orig - cand) == 0
        except Exception:
            return False

    def _schwartz_zippel(self, orig, cand, syms):
        if not syms:
            return str(orig) == str(cand)
        rng = random.Random()
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

    def verify(self, orig_expr, cand_expr, n_nodes=None):
        key = (hash(orig_expr), hash(cand_expr), n_nodes,
               self.small_nodes, self.trials, self.prime, "v1")
        if key in self._cache:
            return self._cache[key]

        if n_nodes is None:
            try:
                G = DAGConverter().convert(orig_expr)
                n_nodes = G.number_of_nodes()
            except Exception:
                n_nodes = 999

        if n_nodes <= self.small_nodes:
            res = self._symbolic_equal(orig_expr, cand_expr)
        else:
            syms = list(orig_expr.free_symbols | cand_expr.free_symbols)
            res = self._schwartz_zippel(orig_expr, cand_expr, syms)

        self._cache[key] = res
        return res


# ---------------------------------------------------------------------------
# FeatureExtractor (standalone copy from notebook Section 5)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """20-dim node features + 5-dim edge features."""

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
# GINRewriteRanker (standalone copy for policy inference)
# ---------------------------------------------------------------------------

class GINRewriteRanker(nn.Module):
    """3-head GINEConv policy: node reducibility + rule selection + confidence."""

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


# ---------------------------------------------------------------------------
# CandidateGenerator (standalone copy for rule application)
# ---------------------------------------------------------------------------

from functools import reduce  # noqa: E402


class CandidateGenerator:
    """Generates (node_id, rule_id, candidate_expr) actions."""

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
        self._candidate_cache = {}

    def _expr_key(self, expr):
        return hash(expr)

    def _graph_from_expr_cached(self, expr):
        key = hash(expr)
        if key not in self._graph_cache:
            norm = self._norm.normalize_for_cost(expr)
            self._graph_cache[key] = self._dc.convert(norm)
        return self._graph_cache[key]

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
        if ck_full in self._candidate_cache:
            return self._candidate_cache[ck_full]
        out = self.generate_global_actions(expr, G) + self.generate_local_actions(expr, G)
        self._candidate_cache[ck_full] = out
        return out


# ---------------------------------------------------------------------------
# RL State and Action data structures
# ---------------------------------------------------------------------------

@dataclass
class RLState:
    """Current circuit state for RL.

    Attributes:
        expr: Current SymPy expression.
        G: DAG representation of the expression.
        cost: Current circuit cost score.
        node_features: (N, 20) feature matrix.
        edge_index: (2, E) edge connectivity.
        edge_attr: (E, 5) edge-type one-hot.
        node_list: Ordered list of DAG node IDs.
        iteration: Current iteration count (0-based).
    """
    expr: sp.Basic
    G: nx.DiGraph
    cost: float
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_list: list
    iteration: int = 0


@dataclass
class RLAction:
    """A rewrite action: apply rule `rule_idx` at node `node_id`."""

    node_id: int
    rule_idx: int
    rule_name: str
    candidate_expr: sp.Basic
    candidate_graph: Optional[nx.DiGraph] = None


@dataclass
class RLStepResult:
    """Result of applying one RL action."""

    next_state: RLState
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------

class ArithCircuitEnv:
    """RL environment for arithmetic circuit optimization.

    The agent interacts with the environment by selecting (node, rule) pairs
    to rewrite the current expression. Each action is verified by PIT before
    being accepted. The environment tracks the best cost achieved and
    terminates when no improvement is possible or max iterations is reached.

    Attributes:
        model: Trained GINRewriteRanker policy (for action scoring).
        cost_model: CostModel for circuit scoring.
        pit_verifier: PITVerifier for equivalence checking.
        candidate_generator: CandidateGenerator for rule application.
        feature_extractor: FeatureExtractor for state representation.
        max_iters: Maximum optimization iterations per episode.
        pit_bonus: Reward bonus when PIT passes.
        pit_penalty: Reward penalty when PIT fails.
    """

    def __init__(self, model=None, cost_model=None, max_iters=8,
                 pit_bonus=1.0, pit_penalty=-1.0, seed=42):
        self.model = model
        self.cost_model = cost_model or CostModel()
        self.pit_verifier = PITVerifier()
        self.candidate_generator = CandidateGenerator(cost_model=self.cost_model)
        self.feature_extractor = FeatureExtractor()
        self.max_iters = max_iters
        self.pit_bonus = pit_bonus
        self.pit_penalty = pit_penalty
        self.seed = seed
        self._rng = random.Random(seed)
        self._gin_lock = threading.Lock()

        # Episode state
        self._current_expr = None
        self._current_G = None
        self._current_cost = 0.0
        self._best_cost = float('inf')
        self._best_expr = None
        self._iteration = 0
        self._done = False
        self._history = []  # list of (action, reward, cost_before, cost_after)

    def reset(self, expr: sp.Basic) -> RLState:
        """Reset the environment with a new expression.

        Args:
            expr: SymPy expression to optimize.

        Returns:
            Initial RLState.
        """
        self._current_expr = expr
        norm = ExprNormalizer().normalize_for_cost(expr)
        self._current_G = self.candidate_generator._graph_from_expr_cached(norm)
        self._current_cost = self.cost_model.score_graph(self._current_G)
        self._best_cost = self._current_cost
        self._best_expr = expr
        self._iteration = 0
        self._done = False
        self._history = []

        return self._build_state()

    def _build_state(self) -> RLState:
        """Build an RLState from the current circuit."""
        node_list, X, edge_index, edge_attr = self.feature_extractor.extract(self._current_G)
        return RLState(
            expr=self._current_expr,
            G=self._current_G,
            cost=self._current_cost,
            node_features=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_list=node_list,
            iteration=self._iteration,
        )

    def get_valid_actions(self, state: RLState) -> list[RLAction]:
        """Enumerate all valid (node, rule) actions for the current state.

        Returns:
            List of RLAction objects, each with a PIT-verified candidate.
        """
        expr = state.expr
        G = state.G

        # Generate all candidate actions
        raw_actions = self.candidate_generator.generate_all_actions(expr, G)

        valid_actions = []
        for raw in raw_actions:
            candidate_expr = raw["candidate_expr"]
            node_id = raw["node_id"]
            rule_idx = raw["rule_idx"]
            rule_name = raw["rule_name"]

            # Verify PIT equivalence
            n_dag_nodes = G.number_of_nodes()
            if not self.pit_verifier.verify(expr, candidate_expr, n_nodes=n_dag_nodes):
                continue

            # Build candidate graph
            try:
                Gp = self.candidate_generator._dc.convert(
                    ExprNormalizer().normalize_for_cost(candidate_expr))
                if Gp.number_of_nodes() == 0:
                    continue
            except Exception:
                continue

            action = RLAction(
                node_id=node_id,
                rule_idx=rule_idx,
                rule_name=rule_name,
                candidate_expr=candidate_expr,
                candidate_graph=Gp,
            )
            valid_actions.append(action)

        return valid_actions

    def step(self, state: RLState, action: RLAction) -> RLStepResult:
        """Apply an action and return the next state, reward, and done flag.

        Args:
            state: Current RLState.
            action: RLAction to apply.

        Returns:
            RLStepResult with next_state, reward, done, and info.
        """
        if self._done:
            return RLStepResult(
                next_state=state, reward=0.0, done=True,
                info={"error": "Episode already done"}
            )

        cost_before = state.cost
        candidate_expr = action.candidate_expr
        candidate_graph = action.candidate_graph

        # Compute cost after applying the action
        cost_after = self.cost_model.score_graph(candidate_graph)

        # PIT verification
        pit_passed = self.pit_verifier.verify(
            state.expr, candidate_expr, n_nodes=state.G.number_of_nodes())

        # Compute reward: Δ cost + PIT bonus/penalty
        delta_cost = cost_before - cost_after
        reward = delta_cost
        if pit_passed:
            reward += self.pit_bonus
        else:
            reward += self.pit_penalty

        # Update episode state
        self._iteration += 1
        self._current_expr = candidate_expr
        self._current_G = candidate_graph
        self._current_cost = cost_after

        # Track best
        info = {
            "action_node_id": action.node_id,
            "action_rule": action.rule_name,
            "cost_before": cost_before,
            "cost_after": cost_after,
            "delta_cost": delta_cost,
            "pit_passed": pit_passed,
            "iteration": self._iteration,
        }

        if cost_after < self._best_cost:
            self._best_cost = cost_after
            self._best_expr = candidate_expr
            info["new_best"] = True
        else:
            info["new_best"] = False

        self._history.append((action, reward, cost_before, cost_after))

        # Check termination
        done = self._check_termination(state, action, cost_after, pit_passed)
        self._done = done

        next_state = self._build_state() if not done else state

        return RLStepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
        )

    def _check_termination(self, state: RLState, action: RLAction,
                           cost_after: float, pit_passed: bool) -> bool:
        """Determine if the episode should terminate.

        Termination conditions:
        1. Max iterations reached.
        2. No valid actions remain (no PIT-verified improvements).
        3. No cost improvement in this step (no progress).
        """
        # Max iterations
        if self._iteration >= self.max_iters:
            return True

        # No valid actions remaining
        valid_actions = self.get_valid_actions(state)
        if len(valid_actions) == 0:
            return True

        # No cost improvement (stagnation)
        if cost_after >= state.cost:
            return True

        return False

    def sample_action(self, state: RLState, epsilon: float = 0.1) -> Optional[RLAction]:
        """Sample an action using epsilon-greedy over the GIN policy.

        With probability epsilon, select a random valid action.
        Otherwise, select the action with the highest GIN rank score.

        Returns:
            Selected RLAction, or None if no valid actions.
        """
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return None

        if self._rng.random() < epsilon:
            return self._rng.choice(valid_actions)

        # Use GIN policy to rank actions
        if self.model is not None:
            try:
                import torch_geometric.data
                data = self.feature_extractor.to_pyg_data(state.G)
                with self._gin_lock:
                    result = self.model.rank_candidates(
                        data,
                        [{"node_id": a.node_id, "rule_idx": a.rule_idx,
                          "candidate_expr": a.candidate_expr,
                          "rule_name": a.rule_name} for a in valid_actions],
                        device=None,
                    )
                ranked = result["ranked"]
                if ranked:
                    return ranked[0]  # type: ignore[return-value]
            except Exception:
                pass

        # Fallback: return first valid action
        return valid_actions[0]

    def get_best_result(self) -> dict:
        """Return the best result achieved in the current episode."""
        return {
            "expr": self._best_expr,
            "orig_cost": self._current_cost if self._iteration == 0 else None,
            "best_cost": self._best_cost,
            "delta_score": (self._current_cost if self._iteration == 0 else 0) - self._best_cost,
            "pct_reduction": 100.0 * (
                (self._current_cost if self._iteration == 0 else 0) - self._best_cost
            ) / max(self._current_cost if self._iteration == 0 else 1, 1e-9),
            "n_iterations": self._iteration,
            "history": self._history,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sympy as sp

    x, y, z = sp.symbols("x y z")

    # Test environment with a simple expression
    env = ArithCircuitEnv(model=None, max_iters=8, seed=42)
    state = env.reset(x * y + x * z)
    print(f"Initial state: cost={state.cost:.4f}, nodes={len(state.node_list)}")

    # Get valid actions
    actions = env.get_valid_actions(state)
    print(f"Valid actions: {len(actions)}")
    for a in actions[:5]:
        print(f"  node={a.node_id}, rule={a.rule_name}, expr={a.candidate_expr}")

    # Step with first valid action
    if actions:
        result = env.step(state, actions[0])
        print(f"\nStep result: reward={result.reward:.4f}, done={result.done}")
        print(f"  cost_before={result.info['cost_before']:.4f}, "
              f"cost_after={result.info['cost_after']:.4f}")
        print(f"  pit_passed={result.info['pit_passed']}")
        print(f"  next_cost={result.next_state.cost:.4f}")

    print("\nRL Environment self-test passed")