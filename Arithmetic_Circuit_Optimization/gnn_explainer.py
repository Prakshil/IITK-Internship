"""
GNNExplainer Integration for ArithCircuit Pipeline (Phase 6).

Applies GNNExplainer to explain GINEConv layer decisions:
- Node-level importance scores (which nodes matter for rewriting)
- Edge-level importance scores (which edges matter for rule selection)
- Visualizes on DAG heatmaps

Usage:
    python gnn_explainer.py --model models/gin_best_fixed.pt --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID, FEATURE_NAMES  # noqa: E402
from rl_environment import (
    GINRewriteRanker, FeatureExtractor, DAGConverter, ExprNormalizer,
    CostModel, CandidateGenerator
)  # noqa: E402
from torch_geometric.explain import Explainer, GNNExplainer  # noqa: E402


class GNNExplainerAnalyzer:
    """GNNExplainer integration for ArithCircuit GINEConv model.

    Provides node-level and edge-level importance scores explaining
    why the GIN policy selects specific (node, rule) actions.
    """

    def __init__(self, model, device, feature_extractor, dag_converter):
        self.model = model
        self.device = device
        self.feat_ext = feature_extractor
        self.dag_conv = dag_converter
        self.model.eval()

    def explain_expression(self, expr, n_hops=2, epochs=100):
        """Generate GNNExplainer explanations for an expression.

        Args:
            expr: SymPy expression to explain
            n_hops: Number of hops for explanation subgraph
            epochs: Number of explanation optimization epochs

        Returns:
            dict with node_mask, edge_mask, explanation details
        """
        # Normalize and convert to DAG
        norm = ExprNormalizer().normalize_for_cost(expr)
        G = self.dag_conv.convert(norm)

        if G.number_of_nodes() < 2:
            return None

        # Extract features
        node_list, X, edge_index, edge_attr = self.feat_ext.extract(G)

        # Create PyG Data
        import torch_geometric.data
        data = torch_geometric.data.Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )
        data = data.to(self.device)

        # Get candidates for this expression
        cand_gen = CandidateGenerator()
        raw_actions = cand_gen.generate_all_actions(norm, G)

        if not raw_actions:
            return None

        # Use the first valid action as explanation target
        # (in practice, we'd explain the top-ranked action)
        target_action = raw_actions[0]
        target_node = target_action["node_id"]
        target_rule = target_action["rule_idx"]

        # Map target_node to index in node_list
        if target_node == GLOBAL_NODE_ID:
            target_node_idx = 0
        elif target_node in node_list:
            target_node_idx = node_list.index(target_node)
        else:
            target_node_idx = 0

        # Create Explainer
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epochs, lr=0.01),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="regression",
                task_level="node",
                return_type="raw",
            ),
        )

        # Get explanation
        # We need to provide target as node index
        explanation = explainer(
            data.x,
            data.edge_index,
            edge_attr=data.edge_attr,
            index=target_node_idx,
        )

        # Extract masks
        node_mask = explanation.node_mask.detach().cpu().numpy() \
            if explanation.node_mask is not None else None
        edge_mask = explanation.edge_mask.detach().cpu().numpy() \
            if explanation.edge_mask is not None else None

        # Get node features importance (top features per node)
        node_feat_imp = None
        if hasattr(explanation, 'node_feat_mask') and explanation.node_feat_mask is not None:
            node_feat_imp = explanation.node_feat_mask.detach().cpu().numpy()

        return {
            "expression": str(expr),
            "normalized_expression": str(norm),
            "target_action": {
                "node_id": target_node,
                "rule": target_action["rule_name"],
                "rule_idx": target_rule,
                "candidate": str(target_action["candidate_expr"]),
            },
            "target_node_idx": target_node_idx,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "node_feat_imp": node_feat_imp,
            "node_list": node_list,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }

    def explain_multiple(self, expressions, n_hops=2, epochs=50):
        """Explain multiple expressions.

        Returns:
            list of explanation dicts
        """
        results = []
        for expr in expressions:
            try:
                result = self.explain_expression(
                    expr, n_hops=n_hops, epochs=epochs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Failed to explain {expr}: {e}")
                continue
        return results

    def visualize_explanation(self, explanation, save_path=None):
        """Create visualization of GNNExplainer results.

        Shows:
        1. DAG with node colors = importance score
        2. Edge widths = importance score
        3. Feature importance bar chart
        """
        if explanation is None:
            return

        G = self.dag_conv.convert(
            ExprNormalizer().normalize_for_cost(
                sp.sympify(explanation["expression"])))

        node_mask = explanation["node_mask"]
        edge_mask = explanation["edge_mask"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Node importance heatmap
        ax1 = axes[0]
        if node_mask is not None:
            node_scores = node_mask.mean(axis=1) if node_mask.ndim > 1 \
                else node_mask
        else:
            node_scores = np.ones(G.number_of_nodes()) * 0.5

        node_list = sorted(G.nodes())
        colors = plt.cm.YlOrRd(node_scores / max(node_scores.max(), 1e-8))
        labels = {n: G.nodes[n].get("label", "?") for n in G.nodes()}

        pos = self._hierarchical_layout(G)
        nx.draw(G, pos, labels=labels, node_color=colors,
                node_size=800, font_size=8, ax=ax1,
                edge_color="gray", width=1.0)
        ax1.set_title("Node Importance (GNNExplainer)", fontweight="bold")
        sm1 = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd,
            norm=plt.Normalize(0, max(node_scores.max(), 1e-8)))
        sm1.set_array([])
        plt.colorbar(sm1, ax=ax1, label="Node Importance")

        # Plot 2: Edge importance
        ax2 = axes[1]
        if edge_mask is not None:
            edge_widths = 1 + 4 * edge_mask / max(edge_mask.max(), 1e-8)
        else:
            edge_widths = np.ones(G.number_of_edges())
        nx.draw(G, pos, labels=labels, node_color="lightblue",
                node_size=800, font_size=8, ax=ax2,
                edge_color=edge_mask if edge_mask is not None else "gray",
                width=edge_widths, edge_cmap=plt.cm.YlOrRd,
                edge_vmin=0, edge_vmax=max(edge_mask.max(), 1e-8)
                if edge_mask is not None else None)
        ax2.set_title("Edge Importance (GNNExplainer)", fontweight="bold")
        if edge_mask is not None:
            sm2 = plt.cm.ScalarMappable(
                cmap=plt.cm.YlOrRd,
                norm=plt.Normalize(0, max(edge_mask.max(), 1e-8)))
            sm2.set_array([])
            plt.colorbar(sm2, ax=ax2, label="Edge Importance")

        # Plot 3: Feature importance
        ax3 = axes[2]
        from oracle_rules import FEATURE_NAMES
        if explanation["node_feat_imp"] is not None:
            feat_imp = explanation["node_feat_imp"].mean(axis=0)
            feat_imp = feat_imp / max(feat_imp.max(), 1e-8)
            sorted_idx = np.argsort(feat_imp)[::-1][:10]
            ax3.barh(range(len(sorted_idx)), feat_imp[sorted_idx])
            ax3.set_yticks(range(len(sorted_idx)))
            ax3.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx])
            ax3.set_xlabel("Importance")
            ax3.set_title("Top Feature Importance", fontweight="bold")
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, "No feature importance available",
                     ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Feature Importance (N/A)", fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved visualization to {save_path}")
        plt.show()
        plt.close()

    def _hierarchical_layout(self, G):
        """Simple hierarchical layout for DAG visualization."""
        root = G.graph.get("root", next(iter(nx.topological_sort(G))))
        pos = {}
        visited = set()

        def _place(node, x, y, depth=0, dx=1.0):
            if node in visited:
                return
            visited.add(node)
            pos[node] = (x, -depth)
            children = list(G.successors(node))
            if children:
                dx_step = dx / len(children)
                for i, child in enumerate(children):
                    _place(child, x + i * dx_step, -depth - 1, depth + 1, dx_step)

        _place(root, 0, 0)
        return pos


# ---------------------------------------------------------------------------
# Counterfactual Explanations (Phase 6.2)
# ---------------------------------------------------------------------------

def counterfactual_analysis(explainer, expr, feature_names=None):
    """Generate counterfactual explanations for an expression.

    For each node feature, computes: "If this feature changed, would
    the model still select the same rule?"
    """
    norm = ExprNormalizer().normalize_for_cost(expr)
    G = explainer.dag_conv.convert(norm)
    node_list, X, edge_index, edge_attr = explainer.feat_ext.extract(G)

    import torch_geometric.data
    data = torch_geometric.data.Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    ).to(explainer.device)

    # Get current top action
    cand_gen = CandidateGenerator()
    raw_actions = cand_gen.generate_all_actions(norm, G)
    if not raw_actions:
        return None

    top_action = raw_actions[0]
    target_rule = top_action["rule_idx"]

    # Get node importance for current prediction
    node_list_idx = sorted(G.nodes())
    if top_action["node_id"] == GLOBAL_NODE_ID:
        target_node_idx = 0
    elif top_action["node_id"] in node_list_idx:
        target_node_idx = node_list_idx.index(top_action["node_id"])
    else:
        target_node_idx = 0

    # Feature ablation: zero out each feature, check if rule changes
    feature_importance = {}
    from oracle_rules import FEATURE_NAMES

    for feat_idx in range(X.shape[1]):
        X_modified = X.copy()
        X_modified[:, feat_idx] = 0.0

        data_mod = torch_geometric.data.Data(
            x=torch.tensor(X_modified, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        ).to(explainer.device)

        with torch.no_grad():
            node_logits, rule_logits, conf_logits = explainer.model(
                data_mod.x, data_mod.edge_index, edge_attr=data_mod.edge_attr)

        # Check if rule prediction changed at target node
        if target_node_idx < rule_logits.shape[0]:
            orig_rule = rule_logits[target_node_idx].argmax().item()
            # We need original logits too for comparison
            with torch.no_grad():
                node_logits_orig, rule_logits_orig, conf_logits_orig = explainer.model(
                    data.x, data.edge_index, edge_attr=data.edge_attr)
            orig_rule_orig = rule_logits_orig[target_node_idx].argmax().item()

            changed = orig_rule != orig_rule_orig
            feature_importance[FEATURE_NAMES[feat_idx]] = {
                "changed": changed,
                "orig_rule": RULE_NAMES[orig_rule_orig] if orig_rule_orig < len(RULE_NAMES) else f"rule_{orig_rule_orig}",
                "new_rule": RULE_NAMES[orig_rule] if orig_rule < len(RULE_NAMES) else f"rule_{orig_rule}",
            }

    return {
        "expression": str(expr),
        "target_rule": top_action["rule_name"],
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNNExplainer integration")
    parser.add_argument("--model", default="models/gin_best_fixed.pt",
                        help="Model checkpoint path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--n-exprs", type=int, default=5,
                        help="Number of expressions to explain")
    parser.add_argument("--epochs", type=int, default=50,
                        help="GNNExplainer epochs")
    parser.add_argument("--output-dir", default="graphs/gin_explainer",
                        help="Output directory for visualizations")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = GINRewriteRanker().to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from {args.model} (epoch {checkpoint.get('epoch', '?')})")

    # Create analyzer
    feat_ext = FeatureExtractor()
    dag_conv = DAGConverter()
    analyzer = GNNExplainerAnalyzer(model, device, feat_ext, dag_conv)

    # Test expressions
    x, y, z, w = sp.symbols("x y z w")
    test_exprs = [
        x * y + x * z,
        x**2 + 2 * x * y + y**2,
        (x + y) * z + (x + y) * w + 5 * x,
        x * y + x * z + x * w,
        (x + y) * (x + z),
    ][:args.n_exprs]

    print(f"\nExplaining {len(test_exprs)} expressions...")
    results = analyzer.explain_multiple(
        test_exprs, n_hops=2, epochs=args.epochs)

    # Visualize each
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        print(f"\nExpression {i+1}/{len(results)}: {result['expression']}")
        print(f"  Target action: {result['target_action']['rule']} "
              f"(node={result['target_action']['node_id']})")
        print(f"  Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")

        if result["node_mask"] is not None:
            print(f"  Node importance: mean={result['node_mask'].mean():.3f}, "
                  f"max={result['node_mask'].max():.3f}")
        if result["edge_mask"] is not None:
            print(f"  Edge importance: mean={result['edge_mask'].mean():.3f}, "
                  f"max={result['edge_mask'].max():.3f}")

        save_path = output_dir / f"explainer_expr_{i}.png"
        analyzer.visualize_explanation(result, save_path=save_path)

    # Counterfactual analysis
    print("\n" + "=" * 70)
    print("COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    for expr in test_exprs[:3]:
        cf = counterfactual_analysis(analyzer, expr)
        if cf:
            print(f"\nExpression: {cf['expression']}")
            print(f"  Target rule: {cf['target_rule']}")
            changed_features = [(k, v) for k, v in cf["feature_importance"].items()
                                if v["changed"]]
            print(f"  Features that change prediction: {len(changed_features)}")
            for fname, info in changed_features[:5]:
                print(f"    {fname}: {info['orig_rule']} → {info['new_rule']}")

    print("\nGNNExplainer integration complete!")


if __name__ == "__main__":
    main()
