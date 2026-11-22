"""Callback-driven token-level LLM-PPO trainer."""
import torch

from .llm_ppo import (
    causal_token_logps,
    kl_shaped_rewards,
    llm_ppo_policy_step,
    llm_ppo_value_step,
    token_level_gae,
)


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _value_predictions(value_model, input_ids, attention_mask):
    output = value_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(output, "values"):
        values = output.values
    elif isinstance(output, (tuple, list)):
        values = output[0]
    else:
        values = output
    if not isinstance(values, torch.Tensor):
        raise TypeError("value model must return a tensor or expose .values")
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.shape != input_ids.shape:
        raise ValueError("value model must return one scalar value per input token")
    return values


class LLMPPOTrainer:
    """Orchestrate generation, reward/KL shaping, token GAE and PPO updates.

    generate_fn(policy_model, prompt_ids, max_new_tokens) returns completion
    token IDs only. reward_fn(prompt_ids, completion_ids) returns one scalar.
    Old/reference log-probs, values, rewards and advantages are calculated once
    per rollout and held fixed across all PPO epochs/minibatches.
    """

    def __init__(self, policy, value_model, policy_optimizer, value_optimizer,
                 generate_fn, reward_fn, pad_token_id, eos_token_id=None,
                 reference_model=None, clip_range=0.2, kl_coef=0.1,
                 gamma=1.0, gae_lambda=0.95, max_grad_norm=1.0,
                 ppo_epochs=1, minibatch_size=None, device=None):
        if clip_range <= 0 or kl_coef < 0 or max_grad_norm <= 0:
            raise ValueError("invalid clipping, KL or gradient norm setting")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0,1]")
        if not isinstance(ppo_epochs, int) or ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be a positive integer")
        if minibatch_size is not None and (
                not isinstance(minibatch_size, int) or minibatch_size <= 0):
            raise ValueError("minibatch_size must be a positive integer or None")
        self.policy = policy
        self.value_model = value_model
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.reference_model = reference_model
        self.clip_range = float(clip_range)
        self.kl_coef = float(kl_coef)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = device

    def _collect(self, prompts, max_new_tokens):
        sequences, completions, rewards = [], [], []
        for prompt in prompts:
            prompt = [int(token) for token in prompt]
            if not prompt:
                raise ValueError("prompt token sequences must be non-empty")
            with torch.inference_mode():
                completion = self.generate_fn(self.policy, prompt, max_new_tokens)
            completion = [int(token) for token in completion]
            if self.eos_token_id is not None and self.eos_token_id in completion:
                completion = completion[:completion.index(self.eos_token_id) + 1]
            if not completion:
                raise ValueError("generation callback must return a non-empty completion")
            reward = float(self.reward_fn(prompt, completion))
            if not torch.isfinite(torch.tensor(reward)):
                raise ValueError("reward_fn must return a finite scalar")
            sequences.append(prompt + completion)
            completions.append((len(prompt), len(completion), completion[-1] == self.eos_token_id
                                if self.eos_token_id is not None else False))
            rewards.append(reward)
        return sequences, completions, torch.tensor(rewards, dtype=torch.float32)

    def train_batch(self, prompt_token_ids, max_new_tokens=128):
        """Run one generated rollout and PPO epochs over its fixed token targets."""
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        previous_policy_mode = self.policy.training
        previous_value_mode = self.value_model.training
        previous_reference_mode = (
            self.reference_model.training if self.reference_model is not None else None
        )
        self.policy.eval()
        self.value_model.eval()
        if self.reference_model is not None:
            self.reference_model.eval()
        try:
            sequences, completion_info, environment_rewards = self._collect(
                prompt_token_ids, max_new_tokens)
            batch_size = len(sequences)
            max_length = max(len(sequence) for sequence in sequences)
            device = torch.device(self.device) if self.device is not None else _model_device(self.policy)
            input_ids = torch.full(
                (batch_size, max_length), self.pad_token_id,
                dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            completion_mask = torch.zeros_like(input_ids, dtype=torch.float32)
            terminals = torch.zeros(
                (batch_size, max_length - 1), dtype=torch.float32, device=device)

            for row, (sequence, (prompt_length, completion_length, ended_with_eos)) in enumerate(
                    zip(sequences, completion_info)):
                length = len(sequence)
                input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
                attention_mask[row, :length] = 1
                completion_mask[row, prompt_length:length] = 1.0
                if ended_with_eos:
                    terminal_position = prompt_length + completion_length - 2
                    terminals[row, terminal_position] = 1.0

            with torch.no_grad():
                policy_logps = causal_token_logps(
                    self.policy, input_ids, attention_mask)
                if self.reference_model is None:
                    reference_logps = policy_logps.detach().clone()
                else:
                    reference_device = _model_device(self.reference_model)
                    reference_logps = causal_token_logps(
                        self.reference_model,
                        input_ids.to(reference_device),
                        attention_mask.to(reference_device),
                    ).to(device)
                values = _value_predictions(
                    self.value_model, input_ids, attention_mask)

            shifted_completion_mask = (
                completion_mask[:, 1:] * attention_mask[:, 1:].to(torch.float32)
            )
            shaped_rewards = kl_shaped_rewards(
                environment_rewards.to(device),
                policy_logps.detach(), reference_logps.detach(),
                shifted_completion_mask, kl_coef=self.kl_coef,
            )
            advantages, value_targets = token_level_gae(
                shaped_rewards, values, terminals, shifted_completion_mask,
                gamma=self.gamma, gae_lambda=self.gae_lambda,
            )

            aggregate = {
                "policy_loss": 0.0, "approx_kl": 0.0,
                "value_loss": 0.0, "value_grad_norm": 0.0,
            }
            update_count = 0
            minibatch_size = self.minibatch_size or batch_size
            for _ in range(self.ppo_epochs):
                permutation = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start:start + minibatch_size]
                    mb_ids = input_ids.index_select(0, indices)
                    mb_attention = attention_mask.index_select(0, indices)
                    mb_completion_mask = completion_mask.index_select(0, indices)
                    mb_old_logps = policy_logps.index_select(0, indices).detach()
                    mb_reference_logps = reference_logps.index_select(0, indices).detach()
                    mb_advantages = advantages.index_select(0, indices).detach()
                    mb_returns = value_targets.index_select(0, indices).detach()
                    mb_value_mask = shifted_completion_mask.index_select(0, indices)

                    value_metrics = llm_ppo_value_step(
                        self.value_model, self.value_optimizer,
                        mb_ids, mb_attention, mb_returns, mb_value_mask,
                        max_grad_norm=self.max_grad_norm,
                    )
                    policy_metrics = llm_ppo_policy_step(
                        self.policy, self.policy_optimizer,
                        mb_ids, mb_attention, mb_completion_mask,
                        mb_old_logps, mb_advantages,
                        clip_range=self.clip_range,
                        reference_token_logps=mb_reference_logps,
                        kl_coef=self.kl_coef,
                    )
                    aggregate["policy_loss"] += policy_metrics["loss"]
                    aggregate["approx_kl"] += policy_metrics["approx_kl"]
                    aggregate["value_loss"] += value_metrics["value_loss"]
                    aggregate["value_grad_norm"] += value_metrics["grad_norm"]
                    update_count += 1

            metrics = {key: value / update_count for key, value in aggregate.items()}
            metrics.update({
                "reward_mean": float(environment_rewards.mean()),
                "reward_std": float(environment_rewards.std(unbiased=False)),
                "prompt_count": batch_size,
                "ppo_epochs": self.ppo_epochs,
                "optimizer_steps": update_count,
                "completion_tokens": int(shifted_completion_mask.sum().item()),
            })
            return metrics
        finally:
            self.policy.train(previous_policy_mode)
            self.value_model.train(previous_value_mode)
            if self.reference_model is not None:
                self.reference_model.train(previous_reference_mode)


__all__ = ["LLMPPOTrainer"]
