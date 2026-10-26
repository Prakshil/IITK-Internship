"""
Retrain GINRewriteRanker with oracle-augmented corpus and re-balanced loss.

This implements Step 1.3 of the implementation plan:
- Load combined corpus (supervised + oracle data)
- Re-balance CE loss: weight by POSITIVE counts (not total counts)
- Add focal loss option for hard-to-learn rules
- Train for 15 epochs, save best checkpoint

Usage:
    python retrain_gin_fixed.py --epochs 15 --seed 42 --device cuda
"""

import argparse
import math
import pickle
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID


# ---------------------------------------------------------------------------
# GINRewriteRanker (standalone copy, no notebook dependency)
# ---------------------------------------------------------------------------

class GINRewriteRanker(nn.Module):
    """3-head GINEConv policy: node + rule + confidence."""

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
        node_logits = self.node_head(embs)
        rule_logits = self.rule_head(embs)
        conf = self.conf_head(embs)
        return node_logits, rule_logits, conf

    @staticmethod
    def _global_target_nodes(batch):
        t = batch.target_node.long().view(-1)
        if hasattr(batch, "ptr") and batch.ptr is not None:
            return t + batch.ptr[:-1].to(t.device)
        return t

    def train_action_batch(self, batch, optimizer, alpha=1.0, beta=1.0, gamma=0.5,
                           rule_weights=None, focal_gamma=2.0, use_focal=False):
        """Action-level loss with optional focal loss and positive-count weighting."""
        self.train()
        device = next(self.parameters()).device
        batch = batch.to(device)
        ea = batch.edge_attr if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None
        node_logits, rule_logits, conf = self(batch.x, batch.edge_index, batch.batch, ea)

        t_nodes = self._global_target_nodes(batch)
        t_rules = batch.target_rule.to(device).long().view(-1)
        timprov = batch.improvement.to(device).float().view(-1)

        # Node loss: BCE on reducibility
        node_sel = node_logits.view(-1)[t_nodes]
        node_lbl = batch.node_reducible.float().view(-1) if hasattr(batch, "node_reducible") else timprov.clamp(0, 1)
        loss_node = F.binary_cross_entropy_with_logits(node_sel, node_lbl)

        # Rule loss: CE with optional focal
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

        # Confidence loss: Huber on improvement
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


# ---------------------------------------------------------------------------
# Positive-count rule weighting (Step 1.3 core fix)
# ---------------------------------------------------------------------------

def compute_positive_rule_weights(data_list, n_rules=N_RULES):
    """
    Compute class weights based on POSITIVE examples per rule, not total counts.

    This is the key fix for Step 1.3: previously, rules with zero positives
    got no signal. Now we weight by positive count so starved rules get
    appropriate emphasis.
    """
    pos_counts = Counter()
    total_counts = Counter()
    for d in data_list:
        ridx = int(d.target_rule.item()) if hasattr(d, "target_rule") else -1
        total_counts[ridx] += 1
        if hasattr(d, "improvement") and float(d.improvement) > 0.01:
            pos_counts[ridx] += 1

    # Use positive counts with floor at 1 for rules with zero positives
    weights = torch.zeros(n_rules)
    for ridx in range(n_rules):
        pc = pos_counts.get(ridx, 0)
        weights[ridx] = 1.0 / max(pc, 1)  # floor at 1

    # Normalize so sum = n_rules
    weights = weights / weights.sum() * n_rules
    return weights, pos_counts, total_counts


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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
    parser.add_argument("--corpus", default="data/pyg_data_list_with_oracle.pkl",
                        help="Path to combined PyG corpus")
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

    # Load combined corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run integrate_oracle_data.py first to generate it.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "rb") as f:
        pyg_data_list = pickle.load(f)
    print(f"Loaded {len(pyg_data_list):,} samples")

    # Compute positive-count rule weights (Step 1.3 core fix)
    rule_weights, pos_counts, total_counts = compute_positive_rule_weights(pyg_data_list)
    rule_weights = rule_weights.to(device)
    print("\nRule positive-count weights:")
    for ridx in range(N_RULES):
        name = RULE_NAMES[ridx] if ridx < len(RULE_NAMES) else f"rule_{ridx}"
        pos = pos_counts.get(ridx, 0)
        total = total_counts.get(ridx, 0)
        w = rule_weights[ridx].item()
        print(f"  {name:<20} pos={pos:>6} total={total:>7} weight={w:.3f}")

    # Split train/val
    random.shuffle(pyg_data_list)
    n_train = int(0.9 * len(pyg_data_list))
    train_data = pyg_data_list[:n_train]
    val_data = pyg_data_list[n_train:]
    print(f"\nTrain: {len(train_data):,}  Val: {len(val_data):,}")

    train_loader = PyGLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = PyGLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = GINRewriteRanker().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Warmup + cosine scheduler
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
