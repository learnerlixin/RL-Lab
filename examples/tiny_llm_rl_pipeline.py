"""A tiny synthetic language-model RL demo: Reward Model -> DPO -> GRPO.

This is a self-contained educational smoke example. It uses small randomly
initialized PyTorch models and synthetic token preferences; it is not a useful
language model, a real alignment run, or a benchmark.
"""
import argparse
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from algorithms.llm_rl import (
    GRPOTrainer,
    RewardModelHead,
    dpo_train_step,
    reward_model_train_step,
)
from utils import seed_everything


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        hidden = torch.tanh(self.projection(self.embedding(input_ids)))
        return SimpleNamespace(logits=self.lm_head(hidden))


def run_demo(steps=20, seed=7):
    seed_everything(seed)
    device = torch.device("cpu")

    # Synthetic chosen/rejected pairs share the same two-token prompt.
    chosen_ids = torch.tensor([[1, 2, 20, 21], [3, 4, 22, 23]], device=device)
    rejected_ids = torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]], device=device)
    attention = torch.ones_like(chosen_ids)
    completion_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], device=device)

    # Stage 1: fit a scalar reward head with Bradley-Terry preferences.
    encoder = TinyEncoder().to(device)
    reward_head = RewardModelHead(hidden_size=32, dropout=0.0).to(device)
    reward_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(reward_head.parameters()), lr=1e-3)
    reward_metrics = None
    for _ in range(steps):
        reward_metrics = reward_model_train_step(
            encoder, reward_head, reward_optimizer,
            chosen_ids, attention, rejected_ids, attention)
    print(f"reward_model loss={reward_metrics['loss']:.4f} "
          f"preference_accuracy={reward_metrics['preference_accuracy']:.3f}")

    # Stage 2: optimize a policy against a frozen reference model with DPO.
    policy = TinyCausalLM().to(device)
    reference = deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    dpo_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    dpo_metrics = None
    for _ in range(steps):
        dpo_metrics = dpo_train_step(
            policy, reference, dpo_optimizer, beta=0.1,
            chosen_input_ids=chosen_ids,
            chosen_attention_mask=attention,
            chosen_completion_mask=completion_mask,
            rejected_input_ids=rejected_ids,
            rejected_attention_mask=attention,
            rejected_completion_mask=completion_mask,
        )
    print(f"dpo loss={dpo_metrics['loss']:.4f} "
          f"preference_accuracy={dpo_metrics['preference_accuracy']:.3f}")

    # Stage 3: callback-driven GRPO with tiny autoregressive token sampling.
    def generate(model, prompt_ids, group_size):
        completions = []
        for _ in range(group_size):
            tokens = list(prompt_ids)
            for _ in range(2):
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                logits = model(input_ids=ids, attention_mask=mask).logits[0, -1]
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1).item()
                tokens.append(int(next_token))
            completions.append(tokens[len(prompt_ids):])
        return completions

    # A deterministic synthetic score makes the demo's reward pipeline transparent.
    def reward(prompt_ids, completion_ids):
        return float(sum(completion_ids))

    grpo_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    trainer = GRPOTrainer(
        policy, grpo_optimizer, generate, reward,
        pad_token_id=0, reference_model=reference, kl_coef=0.01,
    )
    grpo_metrics = trainer.train_batch([[1, 2], [3, 4]], group_size=4)
    print(f"grpo loss={grpo_metrics['loss']:.4f} "
          f"reward_mean={grpo_metrics['reward_mean']:.2f} "
          f"approx_kl={grpo_metrics['approx_kl']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    run_demo(args.steps, args.seed)
