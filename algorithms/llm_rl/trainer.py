"""Callback-driven GRPO trainer connecting rollouts, packing and PPO epochs."""
import torch

from .grpo_trainer import grpo_train_step
from .rollouts import collect_grouped_rollouts, prepare_grpo_batch


class GRPOTrainer:
    """Orchestrate GRPO rollout collection and repeated PPO minibatch updates.

    generate_fn(policy_model, prompt_token_ids, group_size) returns one list of
    completion token-ID lists. reward_fn(prompt_token_ids, completion_ids)
    returns a finite scalar reward.

    The old-policy log-probabilities and advantages are computed once per
    rollout and remain fixed across all PPO epochs/minibatches. Dropout is
    disabled consistently for generation, old/reference scoring, and policy
    updates, then the model's original train/eval mode is restored.
    """

    def __init__(self, policy, optimizer, generate_fn, reward_fn, pad_token_id,
                 reference_model=None, clip_range=0.2, kl_coef=0.0,
                 max_grad_norm=1.0, device=None, ppo_epochs=1,
                 minibatch_size=None):
        if clip_range <= 0 or kl_coef < 0 or max_grad_norm <= 0:
            raise ValueError("invalid GRPO clipping, KL or gradient norm setting")
        if not isinstance(ppo_epochs, int) or ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be a positive integer")
        if minibatch_size is not None and (
                not isinstance(minibatch_size, int) or minibatch_size <= 0):
            raise ValueError("minibatch_size must be a positive integer or None")
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
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def train_batch(self, prompt_token_ids, group_size=4):
        """Collect one rollout then train it for fixed-old-policy PPO epochs."""
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one prompt")
        previous_mode = self.policy.training
        self.policy.eval()
        try:
            def generate(prompt, count):
                with torch.inference_mode():
                    return self.generate_fn(self.policy, prompt, count)

            rollouts = collect_grouped_rollouts(
                prompt_token_ids, generate_fn=generate,
                reward_fn=self.reward_fn, group_size=group_size,
            )
            batch = prepare_grpo_batch(
                rollouts, old_policy_model=self.policy,
                pad_token_id=self.pad_token_id,
                reference_model=self.reference_model,
                device=self.device,
            )
            prompt_count = batch["input_ids"].shape[0]
            minibatch_size = self.minibatch_size or prompt_count
            aggregate = {"loss": 0.0, "approx_kl": 0.0, "grad_norm": 0.0}
            update_count = 0
            for _ in range(self.ppo_epochs):
                permutation = torch.randperm(prompt_count, device=batch["input_ids"].device)
                for start in range(0, prompt_count, minibatch_size):
                    indices = permutation[start:start + minibatch_size]
                    minibatch = {
                        key: value.index_select(0, indices)
                        for key, value in batch.items()
                        if key in {
                            "input_ids", "attention_mask", "completion_mask",
                            "old_logps", "advantages", "reference_logps",
                        }
                    }
                    metrics = grpo_train_step(
                        self.policy, self.optimizer,
                        minibatch["input_ids"], minibatch["attention_mask"],
                        minibatch["completion_mask"], minibatch["old_logps"],
                        minibatch["advantages"],
                        reference_logps=minibatch.get("reference_logps"),
                        clip_range=self.clip_range, kl_coef=self.kl_coef,
                        max_grad_norm=self.max_grad_norm,
                    )
                    for key in aggregate:
                        aggregate[key] += metrics[key]
                    update_count += 1
            metrics = {key: value / update_count for key, value in aggregate.items()}
            metrics.update({
                "reward_mean": float(batch["rewards"].mean()),
                "reward_std": float(batch["rewards"].std(unbiased=False)),
                "prompt_count": prompt_count,
                "group_size": group_size,
                "ppo_epochs": self.ppo_epochs,
                "optimizer_steps": update_count,
            })
            return metrics
        finally:
            self.policy.train(previous_mode)


__all__ = ["GRPOTrainer"]
