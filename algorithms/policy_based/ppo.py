"""PPO clipped updates for categorical policies and vector observations."""
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_range=0.2, value_coef=0.5, entropy_coef=0.01,
               epochs=4, batch_size=64):
    """Optimize one collected rollout; tensors have leading sample dimension."""
    if clip_range <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("clip_range, epochs and batch_size must be positive")
    if value_coef < 0 or entropy_coef < 0 or states.shape[0] == 0:
        raise ValueError("loss coefficients must be nonnegative and rollout non-empty")
    if not (actions.shape[0] == old_log_probs.shape[0] == returns.shape[0]
            == advantages.shape[0] == states.shape[0]):
        raise ValueError("all rollout tensors must share the leading dimension")
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    size = states.shape[0]
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    for _ in range(epochs):
        indices = torch.randperm(size, device=states.device)
        for start in range(0, size, batch_size):
            idx = indices[start:start + batch_size]
            logits, values = model(states[idx])
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[idx])
            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages[idx]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[idx])
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy.detach())
            updates += 1
    return {key: value / max(updates, 1) for key, value in metrics.items()}
