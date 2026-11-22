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
    if policy_logps.shape != reference_logps.shape:
        raise ValueError("policy_logps and reference_logps must have the same shape")
    if completion_mask.shape != policy_logps.shape:
        raise ValueError("completion_mask must align with token log-probabilities")
    if environment_rewards.ndim == 1 and environment_rewards.shape[0] != policy_logps.shape[0]:
        raise ValueError("environment_rewards batch size must match token log-probabilities")
    if environment_rewards.ndim == 2 and environment_rewards.shape != policy_logps.shape:
        raise ValueError("token-level environment_rewards must match token log-probabilities")
    if environment_rewards.ndim not in (1, 2):
        raise ValueError("environment_rewards must have shape [B] or [B, T]")
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



def token_level_gae(rewards, values, terminals, mask, gamma=0.99, gae_lambda=0.95):
    """Compute masked GAE; values has shape [B,T+1] including bootstrap value.

    terminals is true only for genuine episode termination. A time-limit
    truncation should keep terminal=0 so the final value can bootstrap.
    """
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    if rewards.ndim != 2 or values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("rewards must be [B,T] and values must be [B,T+1]")
    if terminals.shape != rewards.shape or mask.shape != rewards.shape:
        raise ValueError("terminals and mask must match rewards [B,T]")
    rewards = rewards.to(device=values.device, dtype=values.dtype)
    terminals = terminals.to(device=values.device, dtype=values.dtype)
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.sum() <= 0:
        raise ValueError("mask must contain at least one valid token")
    if not torch.isfinite(terminals).all() or not torch.isfinite(mask).all():
        raise ValueError("terminals and mask must be finite")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[0], dtype=values.dtype, device=values.device)
    for t in reversed(range(rewards.shape[1])):
        nonterminal = 1.0 - terminals[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * nonterminal - values[:, t]
        gae = (delta + gamma * gae_lambda * nonterminal * gae) * mask[:, t]
        advantages[:, t] = gae
    returns = (advantages + values[:, :-1]) * mask
    return advantages, returns


def llm_ppo_value_step(value_model, optimizer, input_ids, attention_mask,
                       value_targets, value_mask, max_grad_norm=1.0):
    """Fit a token value head to GAE returns with a masked MSE update.

    The value model must return a [B,L] tensor, [B,L,1] tensor, or an object
    exposing a matching .values tensor (tuple/list output[0] is also accepted).
    """
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    output = value_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(output, "values"):
        predictions = output.values
    elif isinstance(output, (tuple, list)):
        predictions = output[0]
    else:
        predictions = output
    if not isinstance(predictions, torch.Tensor):
        raise TypeError("value model output must be a tensor")
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    if predictions.shape != value_targets.shape or value_mask.shape != value_targets.shape:
        raise ValueError("predictions, value_targets and value_mask must have matching [B,L] shapes")
    mask = value_mask.to(device=predictions.device, dtype=predictions.dtype)
    if mask.sum() <= 0:
        raise ValueError("value_mask must contain at least one valid token")
    per_token_loss = F.mse_loss(predictions, value_targets.to(device=predictions.device, dtype=predictions.dtype).detach(), reduction="none")
    loss = (per_token_loss * mask).sum() / mask.sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
    optimizer.step()
    return {"value_loss": float(loss.detach()), "grad_norm": float(grad_norm.detach())}


def llm_ppo_policy_step(model, optimizer, input_ids, attention_mask, completion_mask,
                        old_token_logps, advantages, clip_range=0.2,
                        reference_token_logps=None, kl_coef=0.0):
    """One clipped actor update; use token_level_gae/value_step for critic targets."""
    if clip_range <= 0 or kl_coef < 0:
        raise ValueError("invalid clip_range or kl_coef")
    new_logps = causal_token_logps(model, input_ids, attention_mask)
    mask = (completion_mask[:, 1:] * attention_mask[:, 1:]).to(new_logps.dtype)
    if mask.sum() <= 0:
        raise ValueError("batch must contain at least one completion token")
    if old_token_logps.shape != new_logps.shape or mask.shape != new_logps.shape:
        raise ValueError("old log-probs and shifted completion mask must match model output")
    if advantages.ndim == 1:
        if advantages.shape[0] != new_logps.shape[0]:
            raise ValueError("advantages batch size must match token log-probabilities")
        advantages = advantages.unsqueeze(-1)
    elif advantages.shape not in ((new_logps.shape[0], 1), new_logps.shape):
        raise ValueError("advantages must have shape [B], [B,1] or [B,T]")
    if reference_token_logps is not None and reference_token_logps.shape != new_logps.shape:
        raise ValueError("reference token log-probabilities must match model output")
    old_token_logps = old_token_logps.detach()
    advantages = advantages.detach()
    if reference_token_logps is not None:
        reference_token_logps = reference_token_logps.detach()
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
