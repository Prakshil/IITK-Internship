"""
REINFORCE with Baseline for RL fine-tuning of GINRewriteRanker (Phase 2, Step 2.2).

Implements the REINFORCE policy gradient algorithm with:
- Running-average baseline for variance reduction
- Entropy bonus for exploration
- Gradient clipping for stability
- Epsilon-greedy action sampling during collection
- Warm start from supervised GIN weights (gin_best_fixed.pt)

Usage:
    python rl_train_reinforce.py --n-episodes 200 --seed 42 --device cuda
"""

import argparse
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_rules import RULE_NAMES, N_RULES, GLOBAL_NODE_ID  # noqa: E402
from rl_environment import ArithCircuitEnv, GINRewriteRanker, CostModel  # noqa: E402


class REINFORCETrainer:
    """REINFORCE with baseline for fine-tuning the GIN policy.

    The policy π(a|s) is defined over valid (node, rule) actions:
        score(n, r) = σ(node_logit[n]) × softmax(rule_logits[n])[r] × σ(conf_logit[n])
        P(a=(n,r) | s) = score(n, r) / Σ_{(n',r') valid} score(n', r')

    The gradient update is:
        ∇J = E[∇ log π(a|s) × (G_t - b)]
    where G_t is the discounted return and b is a running-average baseline.
    """

    def __init__(self, model, device, lr=1e-4, gamma=0.99,
                 baseline_decay=0.99, entropy_coef=0.01, grad_clip=1.0):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Running-average baseline
        self.baseline = 0.0
        self.episode_returns = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Action probability computation
    # ------------------------------------------------------------------

    def compute_action_log_probs(self, node_logits, rule_logits, conf_logits,
                                 valid_actions, node_list):
        """Compute log probability of each valid action under the current policy.

        Args:
            node_logits: (N, 1) tensor of node reducibility logits
            rule_logits: (N, n_rules) tensor of rule logits per node
            conf_logits: (N, 1) tensor of confidence logits
            valid_actions: list of RLAction objects
            node_list: ordered list of DAG node IDs

        Returns:
            log_probs: tensor of shape (len(valid_actions),)
        """
        # Map node_id -> index in node_list
        node_id_to_idx = {nid: i for i, nid in enumerate(node_list)}
        if GLOBAL_NODE_ID not in node_id_to_idx:
            node_id_to_idx[GLOBAL_NODE_ID] = 0  # GLOBAL maps to index 0

        log_scores = []
        for action in valid_actions:
            # Resolve node index
            nid = action.node_id
            if nid == GLOBAL_NODE_ID:
                node_idx = 0
            elif nid in node_id_to_idx:
                node_idx = node_id_to_idx[nid]
            else:
                # Node not present in graph — assign very low score
                log_scores.append(-1e10)
                continue

            # P(node=n) via sigmoid (Bernoulli-like, used as unnormalized score)
            node_logit = node_logits[node_idx, 0]
            log_node_score = -F.softplus(-node_logit)  # log σ(x)

            # P(rule=r | node=n) via softmax over rules at this node
            rule_logit_vec = rule_logits[node_idx]  # (n_rules,)
            log_rule_score = F.log_softmax(rule_logit_vec, dim=0)[action.rule_idx]

            # Confidence via sigmoid
            conf_logit = conf_logits[node_idx, 0]
            log_conf_score = -F.softplus(-conf_logit)  # log σ(x)

            # Combined log score
            log_score = log_node_score + log_rule_score + log_conf_score
            log_scores.append(log_score)

        log_scores = torch.stack(log_scores)  # (num_valid_actions,)

        # Normalize to get proper categorical distribution over valid actions
        log_probs = log_scores - torch.logsumexp(log_scores, dim=0)

        return log_probs

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(self, env, expr, epsilon=0.1):
        """Collect one episode of (state, action, reward, log_prob) tuples.

        Args:
            env: ArithCircuitEnv instance
            expr: SymPy expression to optimize
            epsilon: exploration rate for epsilon-greedy sampling

        Returns:
            states, actions, rewards, log_probs, returns, total_return
        """
        state = env.reset(expr)

        states = []
        actions = []
        rewards = []
        log_probs = []

        done = False
        while not done:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break

            # Build PyG Data for model forward pass
            import torch_geometric.data
            data = env.feature_extractor.to_pyg_data(state.G)

            # Move data to device
            data.x = data.x.to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data.edge_attr = data.edge_attr.to(self.device)

            # Forward pass through GIN (WITH grad tracking for policy gradient)
            node_logits, rule_logits, conf_logits = self.model(
                data.x, data.edge_index, edge_attr=data.edge_attr)

            # Compute action probabilities under current policy
            log_probs_actions = self.compute_action_log_probs(
                node_logits, rule_logits, conf_logits,
                valid_actions, state.node_list)

            # Epsilon-greedy action selection
            probs = torch.exp(log_probs_actions)
            if random.random() < epsilon:
                action_idx = random.randint(0, len(valid_actions) - 1)
            else:
                action_idx = torch.argmax(probs).item()

            action = valid_actions[action_idx]
            log_prob = log_probs_actions[action_idx]

            # Step the environment
            result = env.step(state, action)

            states.append(state)
            actions.append(action)
            rewards.append(result.reward)
            log_probs.append(log_prob)

            state = result.next_state
            done = result.done

        # Compute discounted returns G_t = Σ γ^k r_{t+k}
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        total_return = returns[0] if returns else 0.0

        return states, actions, rewards, log_probs, returns, total_return

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(self, log_probs, returns):
        """Perform one REINFORCE policy update.

        Args:
            log_probs: list of log probability tensors (one per step)
            returns: list of discounted returns (one per step)

        Returns:
            dict of training metrics
        """
        log_probs_tensor = torch.stack(log_probs).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32,
                                      device=self.device)

        # Update running-average baseline
        episode_return = returns_tensor[0].item()
        self.episode_returns.append(episode_return)
        self.baseline = (self.baseline_decay * self.baseline +
                         (1.0 - self.baseline_decay) * episode_return)

        # Compute advantages: G_t - b
        advantages = returns_tensor - self.baseline

        # Policy loss: -E[log π(a|s) × advantage]
        policy_loss = -(log_probs_tensor * advantages).mean()

        # Entropy bonus: encourage exploration
        # H = -Σ p log p = -Σ exp(log_p) × log_p
        probs = torch.exp(log_probs_tensor)
        entropy = -(probs * log_probs_tensor).mean()
        entropy_loss = -self.entropy_coef * entropy

        # Total loss
        loss = policy_loss + entropy_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "baseline": self.baseline,
            "episode_return": episode_return,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, expressions, n_episodes=100, epsilon_start=0.2,
              epsilon_end=0.01, epsilon_decay=0.99, eval_every=10):
        """Train the policy with REINFORCE.

        Args:
            env: ArithCircuitEnv instance
            expressions: list of SymPy expressions to train on
            n_episodes: number of training episodes
            epsilon_start: initial exploration rate
            epsilon_end: final exploration rate
            epsilon_decay: per-episode decay multiplier
            eval_every: print progress every N episodes

        Returns:
            list of episode metric dicts
        """
        history = []
        epsilon = epsilon_start

        print(f"Starting REINFORCE training: {n_episodes} episodes, "
              f"{len(expressions)} expressions, lr={self.optimizer.param_groups[0]['lr']}")

        for episode in range(n_episodes):
            # Cycle through expressions
            expr = expressions[episode % len(expressions)]

            # Collect episode
            (states, actions, rewards, log_probs,
             returns, total_return) = self.collect_episode(
                 env, expr, epsilon=epsilon)

            # Update policy
            metrics = self.update_policy(log_probs, returns)

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Record metrics
            episode_metrics = {
                "episode": episode,
                "expr": str(expr),
                "total_return": total_return,
                "n_steps": len(rewards),
                "baseline": self.baseline,
                **metrics,
            }
            history.append(episode_metrics)

            # Progress report
            if (episode + 1) % eval_every == 0:
                recent = history[-eval_every:]
                avg_return = np.mean([h["total_return"] for h in recent])
                print(f"  Ep {episode+1:>3}/{n_episodes}: "
                      f"ret={total_return:>7.4f}  "
                      f"avg_ret={avg_return:>7.4f}  "
                      f"base={self.baseline:>7.4f}  "
                      f"eps={epsilon:.3f}  "
                      f"steps={len(rewards)}")

        return history


# ---------------------------------------------------------------------------
# Expression sampling utilities
# ---------------------------------------------------------------------------

def load_dataset_expressions(n_exprs=50, seed=42):
    """Load a small set of expressions from the dataset for RL training."""
    import pickle
    from pathlib import Path

    # Copy to temp to avoid OneDrive locking issues
    src = Path("data/dataset_100000_depth3.pkl")
    tmp_dir = Path("C:/Users/Prakshil/AppData/Local/Temp/opencode")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "dataset_rl_train.pkl"
    if not tmp_file.exists():
        import shutil
        shutil.copy2(src, tmp_file)

    with open(tmp_file, "rb") as f:
        dataset = pickle.load(f)

    rng = random.Random(seed)
    records = rng.sample(dataset, min(n_exprs, len(dataset)))
    return [r["expr"] for r in records]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE fine-tuning of GINRewriteRanker")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--n-exprs", type=int, default=50,
                        help="Number of expressions to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--model-in", default="models/gin_best_fixed.pt",
                        help="Input model checkpoint (supervised warm start)")
    parser.add_argument("--model-out", default="models/gin_rl_finetuned.pt",
                        help="Output model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=0.2,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Per-episode epsilon decay")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Print progress every N episodes")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load supervised warm-start model
    model = GINRewriteRanker().to(device)
    checkpoint = torch.load(args.model_in, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.train()  # set to training mode
    print(f"Loaded supervised model from {args.model_in} "
          f"(epoch {checkpoint.get('epoch', '?')})")

    # Create environment
    env = ArithCircuitEnv(model=model, max_iters=8,
                          pit_bonus=1.0, pit_penalty=-1.0, seed=args.seed)

    # Load training expressions
    print(f"Loading {args.n_exprs} expressions for RL training...")
    expressions = load_dataset_expressions(n_exprs=args.n_exprs, seed=args.seed)
    print(f"Loaded {len(expressions)} expressions")

    # Create trainer
    trainer = REINFORCETrainer(
        model=model, device=device,
        lr=args.lr, gamma=args.gamma,
        entropy_coef=args.entropy_coef,
    )

    # Train
    t0 = time.time()
    history = trainer.train(
        env=env,
        expressions=expressions,
        n_episodes=args.n_episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_every=args.eval_every,
    )
    t_elapsed = time.time() - t0

    # Save fine-tuned model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "baseline": trainer.baseline,
        "history": history,
        "args": vars(args),
    }, args.model_out)
    print(f"\nFine-tuned model saved to {args.model_out}")
    print(f"Training time: {t_elapsed:.1f}s")

    # Final summary
    if history:
        final_returns = [h["total_return"] for h in history]
        print(f"\nFinal episode return: {final_returns[-1]:.4f}")
        print(f"Best episode return:  {max(final_returns):.4f}")
        print(f"Mean return (last 20): {np.mean(final_returns[-20:]):.4f}")


if __name__ == "__main__":
    main()
