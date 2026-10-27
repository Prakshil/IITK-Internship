"""
Self-Supervised GIN Training (Phase 3, Steps 3.3-3.4).

Trains GINRewriteRanker on self-supervised scramble-and-reverse data.
Compares against supervised baseline.

Usage:
    python train_self_supervised.py --n-per-rule 5000 --epochs 15 --seed 42 --device cuda
"""

import argparse
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

from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID  # noqa: E402
from self_supervised import SelfSupervisedGenerator  # noqa: E402
from rl_environment import (
    GINRewriteRanker, FeatureExtractor, DAGConverter, ExprNormalizer, CostModel
)  # noqa: E402


# ---------------------------------------------------------------------------
# Convert self-supervised records to PyG Data
# ---------------------------------------------------------------------------

def records_to_pyg_data(records, dag_conv, feat_ext):
    """Convert self-supervised records to PyG Data objects."""
    data_list = []
    skipped = 0

    for rec in records:
        expr = rec["complexified_expr"]
        rule_idx = rec["rule_idx"]
        improvement = rec["cost_reduction"]

        norm = ExprNormalizer().normalize_for_cost(expr)
        G = dag_conv.convert(norm)

        if G.number_of_nodes() < 2:
            skipped += 1
            continue

        node_list, X, edge_index, edge_attr = feat_ext.extract(G)

        import torch_geometric.data
        data = torch_geometric.data.Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )
        data.node_list = node_list
        # Whole-expression rewrite (GLOBAL_NODE_ID -> index 0)
        data.target_node = torch.tensor([0], dtype=torch.long)
        data.target_rule = torch.tensor([int(rule_idx)], dtype=torch.long)
        data.improvement = torch.tensor([float(improvement)], dtype=torch.float)
        data.node_reducible = torch.tensor([1.0], dtype=torch.float)  # all are positive
        data.confidence = torch.tensor([float(improvement)], dtype=torch.float)
        data.in_top_k = torch.tensor([1.0], dtype=torch.float)

        data_list.append(data)

    return data_list, skipped


# ---------------------------------------------------------------------------
# Training (same 3-head loss as supervised)
# ---------------------------------------------------------------------------

def compute_positive_rule_weights(data_list, n_rules=N_RULES):
    """Class weights based on positive examples per rule."""
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
    parser = argparse.ArgumentParser(
        description="Self-supervised GIN training (scramble-and-reverse)")
    parser.add_argument("--n-per-rule", type=int, default=5000,
                        help="Self-supervised examples per rule")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--model-in", default="models/gin_best_fixed.pt",
                        help="Input model checkpoint (supervised warm start)")
    parser.add_argument("--model-out", default="models/gin_ss_finetuned.pt",
                        help="Output model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--mix-supervised", action="store_true",
                        help="Mix self-supervised with supervised data")
    parser.add_argument("--n-supervised", type=int, default=50000,
                        help="Supervised samples to mix (if --mix-supervised)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Generate self-supervised data
    print(f"Generating {args.n_per_rule} self-supervised examples per rule...")
    gen = SelfSupervisedGenerator(seed=args.seed)
    ss_records = gen.generate_dataset(n_per_rule=args.n_per_rule)
    print(f"Generated {len(ss_records)} self-supervised records")

    # Convert to PyG Data
    print("Converting to PyG Data...")
    dag_conv = DAGConverter()
    feat_ext = FeatureExtractor()
    ss_data, skipped = records_to_pyg_data(ss_records, dag_conv, feat_ext)
    print(f"Converted {len(ss_data)} samples (skipped {skipped})")

    # Optionally mix with supervised data
    if args.mix_supervised:
        print(f"Loading {args.n_supervised} supervised samples...")
        tmp_dir = Path("C:/Users/Prakshil/AppData/Local/Temp/opencode")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / "pyg_data_list.pkl"
        if not tmp_file.exists():
            import shutil
            shutil.copy2(Path("data/pyg_data_list.pkl"), tmp_file)
        with open(tmp_file, "rb") as f:
            sup_data = pickle.load(f)
        sup_data = sup_data[:args.n_supervised]
        print(f"Loaded {len(sup_data)} supervised samples")
        combined = sup_data + ss_data
    else:
        combined = ss_data

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
    # Load supervised warm start
    checkpoint = torch.load(args.model_in, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.train()
    print(f"\nLoaded supervised warm start from {args.model_in}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

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
            focal_gamma=2.0, use_focal=True,
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
            Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "rule_weights": rule_weights.cpu(),
                "pos_counts": dict(pos_counts),
                "total_counts": dict(total_counts),
            }, args.model_out)
            print(f"  * New best val loss {vl:.4f} — saved {args.model_out}")

    print(f"\nSelf-supervised training complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved: {args.model_out}")


if __name__ == "__main__":
    main()
