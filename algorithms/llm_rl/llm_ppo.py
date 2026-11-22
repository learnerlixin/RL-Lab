"""Language-model PPO policy update and KL-shaped reward utilities."""
import torch
import torch.nn.functional as F


def causal_token_logps(model, input_ids, attention_mask):
    """Return next-token log-probabilities with shape [B, L-1]."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    labels = input_ids[:, 1:]
    return F.log_softmax(logits[:, :-1, :], dim=-1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)


def kl_shaped_rewards(environment_rewards, policy_logps, reference_logps,
                      completion_mask, kl_coef=0.1):
    """Subtract sampled policy-vs-reference KL per token from reward.

    environment_rewards may be [B] terminal rewards or [B,T] token rewards.
    """
    if kl_coef < 0:
        raise ValueError("kl_coef must be nonnegative")
    log_ratio = policy_logps - reference_logps
    kl_estimate = torch.exp(-log_ratio) - 1.0 + log_ratio
    if environment_rewards.ndim == 1:
        rewards = torch.zeros_like(policy_logps)
        positions = torch.arange(rewards.shape[1], device=rewards.device).expand_as(completion_mask)
        last_positions = torch.where(completion_mask > 0, positions, -1).max(dim=-1).values
        if (last_positions < 0).any():
            raise ValueError("each sequence must contain a completion token")
        rows = torch.arange(rewards.shape[0], device=rewards.device)
        rewards[rows, last_positions] = environment_rewards
    else:
        rewards = environment_rewards
    return (rewards - kl_coef * kl_estimate) * completion_mask.to(policy_logps.dtype)


def llm_ppo_policy_step(model, optimizer, input_ids, attention_mask, completion_mask,
                        old_token_logps, advantages, clip_range=0.2,
                        reference_token_logps=None, kl_coef=0.0):
    """One clipped actor update. Critic/GAE and rollout collection are external."""
    if clip_range <= 0 or kl_coef < 0:
        raise ValueError("invalid clip_range or kl_coef")
    new_logps = causal_token_logps(model, input_ids, attention_mask)
    mask = (completion_mask[:, 1:] * attention_mask[:, 1:]).to(new_logps.dtype)
    if mask.sum() <= 0:
        raise ValueError("batch must contain at least one completion token")
    if old_token_logps.shape != new_logps.shape or mask.shape != new_logps.shape:
        raise ValueError("old log-probs and shifted completion mask must match model output")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(new_logps - old_token_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    per_token_loss = -torch.minimum(unclipped, clipped)
    kl_value = torch.zeros((), device=new_logps.device)
    if reference_token_logps is not None and kl_coef:
        log_ratio_ref = reference_token_logps - new_logps
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        per_token_loss = per_token_loss + kl_coef * kl
        kl_value = (kl * mask).sum() / mask.sum()
    loss = (per_token_loss * mask).sum() / mask.sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach()), "approx_kl": float(kl_value.detach())}
