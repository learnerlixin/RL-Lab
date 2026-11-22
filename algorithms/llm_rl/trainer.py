"""Callback-driven GRPO trainer that connects rollouts, packing and one update."""
import torch

from .grpo_trainer import grpo_train_step
from .rollouts import collect_grouped_rollouts, prepare_grpo_batch


class GRPOTrainer:
    """Orchestrate one GRPO batch without prescribing a tokenizer/generation API.

    generate_fn(policy_model, prompt_token_ids, group_size) must return a list
    of completion token-ID lists. reward_fn(prompt_token_ids, completion_ids)
    returns one scalar reward. Generation is called with the live policy model;
    it remains unchanged until the batch update has been completed.

    The trainer keeps dropout disabled for both old-policy log-prob computation
    and the PPO update, ensuring the importance ratio compares the same policy
    distribution. The model's original train/eval state is restored afterwards.
    """

    def __init__(self, policy, optimizer, generate_fn, reward_fn, pad_token_id,
                 reference_model=None, clip_range=0.2, kl_coef=0.0,
                 max_grad_norm=1.0, device=None):
        if clip_range <= 0 or kl_coef < 0 or max_grad_norm <= 0:
            raise ValueError("invalid GRPO clipping, KL or gradient norm setting")
        self.policy = policy
        self.optimizer = optimizer
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.pad_token_id = int(pad_token_id)
        self.reference_model = reference_model
        self.clip_range = float(clip_range)
        self.kl_coef = float(kl_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

    def train_batch(self, prompt_token_ids, group_size=4):
        """Collect groups, pack tokens, and apply a single GRPO optimizer step."""
        previous_mode = self.policy.training
        self.policy.eval()
        try:
            rollouts = collect_grouped_rollouts(
                prompt_token_ids,
                generate_fn=lambda prompt, count: self.generate_fn(
                    self.policy, prompt, count),
                reward_fn=self.reward_fn,
                group_size=group_size,
            )
            batch = prepare_grpo_batch(
                rollouts, old_policy_model=self.policy,
                pad_token_id=self.pad_token_id,
                reference_model=self.reference_model,
                device=self.device,
            )
            metrics = grpo_train_step(
                self.policy, self.optimizer,
                batch["input_ids"], batch["attention_mask"],
                batch["completion_mask"], batch["old_logps"],
                batch["advantages"],
                reference_logps=batch.get("reference_logps"),
                clip_range=self.clip_range, kl_coef=self.kl_coef,
                max_grad_norm=self.max_grad_norm,
            )
            metrics.update({
                "reward_mean": float(batch["rewards"].mean()),
                "reward_std": float(batch["rewards"].std(unbiased=False)),
                "prompt_count": len(prompt_token_ids),
                "group_size": group_size,
            })
            return metrics
        finally:
            self.policy.train(previous_mode)


__all__ = ["GRPOTrainer"]
