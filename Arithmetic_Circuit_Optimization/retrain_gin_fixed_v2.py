"""
Retrain GINRewriteRanker with oracle-augmented corpus (efficient version).

Copies large pickle to temp dir first to avoid OneDrive locking issues,
then loads original corpus subset + generates oracle data on-the-fly.
"""

import argparse
import pickle
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader as PyGLoader
import torch_geometric.data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_rules import (
    RULE_NAMES, N_RULES, GLOBAL_NODE_ID,
    generate_oracle_dataset,
)

import networkx as nx
import sympy as sp

EDGE_TYPE = {"add_child": 0, "mul_child": 1, "pow_base": 2, "pow_exp": 3, "other": 4}
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

    def train_action_batch(self, batch, optimizer, alpha=1.0, beta=1.0, gamma=0.5,
                           rule_weights=None, focal_gamma=2.0, use_focal=False):
        self.train()
        device = next(self.parameters()).device
        batch = batch.to(device)
        ea = batch.edge_attr if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None
        node_logits, rule_logits, conf = self(batch.x, batch.edge_index, batch.batch, ea)

        t_nodes = self._global_target_nodes(batch)
        t_rules = batch.target_rule.to(device).long().view(-1)
        timprov = batch.improvement.to(device).float().view(-1)

        node_sel = node_logits.view(-1)[t_nodes]
        node_lbl = batch.node_reducible.float().view(-1) if hasattr(batch, "node_reducible") else timprov.clamp(0, 1)
        loss_node = F.binary_cross_entropy_with_logits(node_sel, node_lbl)

        rule_rows = rule_logits[t_nodes]
        if rule_weights is None:
            rule_counts = torch.bincount(t_rules, minlength=N_RULES).float().clamp(min=1)
            rule_weights = (1.0 / rule_counts)
            rule_weights = rule_weights / rule_weights.sum() * N_RULES
            rule_weights = rule_weights.to(rule_rows.device)

        if use_focal:
            ce_loss = F.cross_entropy(rule_rows, t_rules, weight=rule_weights, reduction='none')
            pt = torch.exp(-ce_loss)
            loss_rule = ((1 - pt) ** focal_gamma * ce_loss).mean()
        else:
            loss_rule = F.cross_entropy(rule_rows, t_rules, weight=rule_weights)

        conf_sel = conf.view(-1)[t_nodes]
        loss_conf = F.huber_loss(torch.sigmoid(conf_sel), timprov)

        loss = alpha * loss_node + beta * loss_rule + gamma * loss_conf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "loss": loss.item(),
            "loss_node": loss_node.item(),
            "loss_rule": loss_rule.item(),
            "loss_conf": loss_conf.item(),
        }


def oracle_record_to_pyg_data(record, dag_conv, feat_ext):
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

    data = torch_geometric.data.Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    data.node_list = node_list
    data.target_node = torch.tensor([0], dtype=torch.long)
    data.target_rule = torch.tensor([int(rule_idx)], dtype=torch.long)
    data.improvement = torch.tensor([float(improvement)], dtype=torch.float)
    data.node_reducible = torch.tensor([float(node_reducible)], dtype=torch.float)
    data.confidence = torch.tensor([float(improvement)], dtype=torch.float)
    data.in_top_k = torch.tensor([1.0], dtype=torch.float)

    return data


def compute_positive_rule_weights(data_list, n_rules=N_RULES):
    pos_counts = Counter()
    total_counts = Counter()
    for d in data_list:
        ridx = int(d.target_rule.item()) if hasattr(d, "target_rule") else -1
        total_counts[ridx] += 1
        if hasattr(d, "improvement") and float(d.improvement) > 0.01:
            pos_counts[ridx] += 1

    weights = torch.zeros(n_rules)
    for ridx in range(n_rules):
        pc = pos_counts.get(ridx, 0)
        weights[ridx] = 1.0 / max(pc, 1)
    weights = weights / weights.sum() * n_rules
    return weights, pos_counts, total_counts


def run_epoch(model, loader, optimizer, device, train=True,
              alpha=1.0, beta=1.0, gamma=0.5, rule_weights=None,
              focal_gamma=2.0, use_focal=False):
    model.train() if train else model.eval()
    tot = tot_n = tot_r = tot_c = 0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            if train:
                res = model.train_action_batch(
                    batch, optimizer,
                    alpha=alpha, beta=beta, gamma=gamma,
                    rule_weights=rule_weights,
                    focal_gamma=focal_gamma, use_focal=use_focal,
                )
            else:
                model.eval()
                with torch.no_grad():
                    ea = batch.edge_attr if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None
                    node_logits, rule_logits, conf = model(
                        batch.x, batch.edge_index, batch.batch, ea)
                    t_nodes = GINRewriteRanker._global_target_nodes(batch)
                    t_rules = batch.target_rule.to(device).long().view(-1)
                    timprov = batch.improvement.to(device).float().view(-1)
                    node_lbl = batch.node_reducible.float().view(-1) if hasattr(batch, "node_reducible") else timprov.clamp(0, 1)
                    node_sel = node_logits.view(-1)[t_nodes]
                    loss_node = F.binary_cross_entropy_with_logits(node_sel, node_lbl)
                    rule_rows = rule_logits[t_nodes]
                    loss_rule = F.cross_entropy(rule_rows, t_rules, weight=rule_weights)
                    conf_sel = conf.view(-1)[t_nodes]
                    loss_conf = F.huber_loss(torch.sigmoid(conf_sel), timprov)
                    loss = alpha * loss_node + beta * loss_rule + gamma * loss_conf
                    res = {"loss": loss.item(), "loss_node": loss_node.item(),
                           "loss_rule": loss_rule.item(), "loss_conf": loss_conf.item()}

            tot += res["loss"]
            tot_n += res["loss_node"]
            tot_r += res["loss_rule"]
            tot_c += res["loss_conf"]
            n += 1
    return tot / n, tot_n / n, tot_r / n, tot_c / n


def main():
    parser = argparse.ArgumentParser(description="Retrain GIN with oracle-augmented corpus")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--n-original", type=int, default=50000,
                        help="Number of samples from original corpus (subset for speed)")
    parser.add_argument("--n-oracle-per-rule", type=int, default=5000,
                        help="Number of oracle records per starved rule")
    parser.add_argument("--output", default="models/gin_best_fixed.pt",
                        help="Output checkpoint path")
    parser.add_argument("--use-focal", action="store_true",
                        help="Use focal loss for hard-to-learn rules")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Copy corpus to temp to avoid OneDrive locking issues
    src = Path("data/pyg_data_list.pkl")
    tmp_dir = Path("C:/Users/Prakshil/AppData/Local/Temp/opencode")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "pyg_data_list.pkl"
    if not tmp_file.exists():
        print(f"Copying corpus to temp location...")
        shutil.copy2(src, tmp_file)
    else:
        print(f"Using cached corpus at temp location")

    # Load original corpus
    print(f"Loading original corpus (first {args.n_original:,} samples)...")
    t0 = time.time()
    with open(tmp_file, "rb") as f:
        pyg_data_list = pickle.load(f)
    t1 = time.time()
    print(f"Loaded {len(pyg_data_list):,} total samples in {t1-t0:.1f}s")

    # Take a subset
    if args.n_original < len(pyg_data_list):
        pyg_data_list = pyg_data_list[:args.n_original]
        print(f"Using subset: {len(pyg_data_list):,} samples")

    # Generate oracle data on-the-fly
    print(f"\nGenerating {args.n_oracle_per_rule:,} oracle records per rule...")
    oracle_records = generate_oracle_dataset(
        n_per_rule=args.n_oracle_per_rule, seed=args.seed
    )
    print(f"Generated {len(oracle_records)} total oracle records")

    # Convert to PyG Data
    print("Converting oracle records to PyG Data...")
    dag_conv = DAGConverter()
    feat_ext = FeatureExtractor()
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
    print(f"Converted {len(oracle_data_list)} oracle records (skipped {skipped})")

    # Combine
    combined = pyg_data_list + oracle_data_list
    print(f"\nCombined corpus: {len(combined):,} samples")

    # Compute rule weights
    rule_weights, pos_counts, total_counts = compute_positive_rule_weights(combined)
    rule_weights = rule_weights.to(device)
    print("\nRule positive-count weights:")
    for ridx in range(N_RULES):
        name = RULE_NAMES[ridx] if ridx < len(RULE_NAMES) else f"rule_{ridx}"
        pos = pos_counts.get(ridx, 0)
        total = total_counts.get(ridx, 0)
        w = rule_weights[ridx].item()
        print(f"  {name:<20} pos={pos:>6} total={total:>7} weight={w:.3f}")

    # Split train/val
    random.shuffle(combined)
    n_train = int(0.9 * len(combined))
    train_data = combined[:n_train]
    val_data = combined[n_train:]
    print(f"\nTrain: {len(train_data):,}  Val: {len(val_data):,}")

    train_loader = PyGLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = PyGLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = GINRewriteRanker().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 3))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[3])

    # Training loop
    best_val = float('inf')
    history = []
    print(f"\nTraining {args.epochs} epochs...")
    print(f"{'Ep':>3}  {'Train':>8}  {'Val':>8}  {'Node':>8}  {'Rule':>8}  {'Conf':>8}")
    print("-" * 55)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, tn, tr, tc = run_epoch(
            model, train_loader, optimizer, device, train=True,
            rule_weights=rule_weights,
            focal_gamma=2.0, use_focal=args.use_focal,
        )
        vl, vn, vr, vc = run_epoch(
            model, val_loader, None, device, train=False,
            rule_weights=rule_weights,
        )
        scheduler.step()
        history.append({"ep": ep, "train": tl, "val": vl, "node": tn, "rule": tr, "conf": tc})
        print(f"{ep:>3}  {tl:>8.4f}  {vl:>8.4f}  {tn:>8.4f}  {tr:>8.4f}  {tc:>8.4f}  ({time.time()-t0:.1f}s)")

        if vl < best_val:
            best_val = vl
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "rule_weights": rule_weights.cpu(),
                "pos_counts": dict(pos_counts),
                "total_counts": dict(total_counts),
            }, args.output)
            print(f"  * New best val loss {vl:.4f} — saved {args.output}")

    print(f"\nRetraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved: {args.output}")


if __name__ == "__main__":
    main()
