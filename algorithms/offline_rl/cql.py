"""Conservative Q-Learning (CQL) regularizers for discrete and continuous actions."""
import torch
import torch.nn.functional as F


def discrete_cql_penalty(all_action_q_values, dataset_action_q_values, temperature=1.0):
    """Penalize high Q for unseen actions relative to dataset actions.

    Args:
        all_action_q_values: [batch, action_count].
        dataset_action_q_values: [batch] or [batch, 1].
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    data_q = dataset_action_q_values.reshape(-1)
    if all_action_q_values.ndim != 2 or all_action_q_values.shape[0] != data_q.shape[0]:
        raise ValueError("Q values must have shapes [B,A] and [B]")
    conservative_gap = temperature * torch.logsumexp(
        all_action_q_values / temperature, dim=-1) - data_q
    return conservative_gap.mean()


def _candidate_q_values(q_function, states, candidate_actions):
    if candidate_actions.ndim != 3 or candidate_actions.shape[0] != states.shape[0]:
        raise ValueError("candidate actions must have shape [B,N,A]")
    batch, count, action_dim = candidate_actions.shape
    expanded_states = states.unsqueeze(1).expand(-1, count, -1)
    values = q_function(expanded_states.reshape(batch * count, -1),
                        candidate_actions.reshape(batch * count, action_dim))
    return values.reshape(batch, count, -1).squeeze(-1)


def continuous_cql_penalty(q_function, states, dataset_actions, random_actions,
                           policy_actions, random_log_probs=None,
                           policy_log_probs=None, temperature=1.0):
    """CQL(H)-style conservative gap over sampled continuous actions.

    Candidate actions have shape [B,N,A]. Optional log probabilities correct
    for the sampling density (for example, uniform random and current policy).
    The critic is called with flattened [B*N, ...] state/action tensors.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if states.ndim != 2 or dataset_actions.ndim != 2 or states.shape[0] != dataset_actions.shape[0]:
        raise ValueError("states and dataset_actions must have shapes [B,S] and [B,A]")
    random_q = _candidate_q_values(q_function, states, random_actions)
    policy_q = _candidate_q_values(q_function, states, policy_actions)
    expected_shape = random_q.shape
    if policy_q.shape != expected_shape:
        raise ValueError("random and policy candidate action shapes must match")
    if random_log_probs is None:
        random_log_probs = torch.zeros_like(random_q)
    if policy_log_probs is None:
        policy_log_probs = torch.zeros_like(policy_q)
    if random_log_probs.shape != expected_shape or policy_log_probs.shape != expected_shape:
        raise ValueError("candidate log probabilities must have shape [B,N]")
    data_q = q_function(states, dataset_actions).reshape(states.shape[0], -1)
    if data_q.shape[1] != 1:
        raise ValueError("critic must return one Q value per state-action pair")
    candidates = torch.cat([random_q - random_log_probs,
                            policy_q - policy_log_probs], dim=1)
    gap = temperature * torch.logsumexp(candidates / temperature, dim=1) - data_q.squeeze(-1)
    return gap.mean()


def continuous_cql_critic_loss(q_function, states, dataset_actions, td_targets,
                               random_actions, policy_actions,
                               random_log_probs=None, policy_log_probs=None,
                               conservative_weight=1.0, temperature=1.0):
    """Combine TD regression and the continuous-action conservative penalty."""
    if conservative_weight < 0:
        raise ValueError("conservative_weight must be nonnegative")
    data_q = q_function(states, dataset_actions).reshape(-1)
    targets = td_targets.detach().reshape(-1)
    if data_q.shape != targets.shape:
        raise ValueError("td_targets must contain one target per transition")
    td_loss = F.mse_loss(data_q, targets)
    conservative_loss = continuous_cql_penalty(
        q_function, states, dataset_actions, random_actions, policy_actions,
        random_log_probs, policy_log_probs, temperature)
    total = td_loss + conservative_weight * conservative_loss
    return total, {"td_loss": td_loss.detach(),
                   "conservative_loss": conservative_loss.detach(),
                   "q_data": data_q.detach().mean()}


def continuous_cql_critic_step(critic, optimizer, states, dataset_actions, td_targets,
                               random_actions, policy_actions,
                               random_log_probs=None, policy_log_probs=None,
                               conservative_weight=1.0, temperature=1.0,
                               max_grad_norm=1.0):
    """Apply one critic update; action sampling/TD target construction are caller-owned."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    loss, components = continuous_cql_critic_loss(
        critic, states, dataset_actions, td_targets, random_actions, policy_actions,
        random_log_probs, policy_log_probs, conservative_weight, temperature)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm.detach()),
            **{key: float(value) for key, value in components.items()}}


__all__ = ["discrete_cql_penalty", "continuous_cql_penalty",
           "continuous_cql_critic_loss", "continuous_cql_critic_step"]
