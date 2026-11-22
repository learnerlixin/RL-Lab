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



def discrete_cql_critic_loss(q_network, states, dataset_actions, td_targets,
                             conservative_weight=1.0, temperature=1.0):
    """Combine discrete-action TD regression and conservative regularization."""
    if conservative_weight < 0:
        raise ValueError("conservative_weight must be nonnegative")
    all_q = q_network(states)
    if all_q.ndim != 2 or all_q.shape[0] != dataset_actions.shape[0]:
        raise ValueError("Q network must return [B, action_count]")
    data_q = all_q.gather(1, dataset_actions.long().reshape(-1, 1)).squeeze(-1)
    targets = td_targets.detach().reshape(-1)
    if data_q.shape != targets.shape:
        raise ValueError("td_targets must contain one target per transition")
    td_loss = F.mse_loss(data_q, targets)
    conservative_loss = discrete_cql_penalty(all_q, data_q, temperature)
    total = td_loss + conservative_weight * conservative_loss
    return total, {"td_loss": td_loss.detach(),
                   "conservative_loss": conservative_loss.detach(),
                   "q_data": data_q.detach().mean()}


def discrete_cql_critic_step(q_network, optimizer, states, dataset_actions, td_targets,
                             conservative_weight=1.0, temperature=1.0,
                             max_grad_norm=1.0):
    """Apply one discrete CQL Q-network update."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    loss, components = discrete_cql_critic_loss(
        q_network, states, dataset_actions, td_targets,
        conservative_weight, temperature)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm.detach()),
            **{key: float(value) for key, value in components.items()}}

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



class ContinuousCQLAgent:
    """SAC-style continuous CQL agent trained from fixed offline minibatches.

    The dataset sampler is caller-owned. The agent samples uniform and current
    policy actions internally for the conservative penalty and applies twin-Q,
    entropy-regularized actor, and Polyak target updates.
    """
    def __init__(self, observation_dim, action_dim, action_low=-1.0,
                 action_high=1.0, hidden_dim=256, learning_rate=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2,
                 conservative_weight=1.0, cql_temperature=1.0,
                 num_action_samples=10, device="cpu"):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("gamma must be in [0,1] and tau in (0,1]")
        if alpha < 0 or conservative_weight < 0 or cql_temperature <= 0:
            raise ValueError("invalid alpha or CQL hyperparameters")
        if num_action_samples < 1:
            raise ValueError("num_action_samples must be positive")
        from algorithms.policy_optimization.sac import GaussianActor, QNetwork
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("action_high must be greater than action_low")
        self.device = torch.device(device)
        self.actor = GaussianActor(
            observation_dim, action_dim, low, high, hidden_dim).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q2 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)
        self.gamma, self.tau = float(gamma), float(tau)
        self.alpha = float(alpha)
        self.conservative_weight = float(conservative_weight)
        self.cql_temperature = float(cql_temperature)
        self.num_action_samples = int(num_action_samples)

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        import numpy as np
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        return self.actor.sample(state)[2 if deterministic else 0].squeeze(0).cpu().numpy()

    def update(self, batch):
        """Update critics and actor from an offline transition minibatch."""
        states = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32,
                                  device=self.device).reshape(-1, 1)
        next_states = torch.as_tensor(batch["next_observations"], dtype=torch.float32,
                                      device=self.device)
        terminals = torch.as_tensor(batch["terminals"], dtype=torch.float32,
                                    device=self.device).reshape(-1, 1)
        if states.ndim != 2 or actions.ndim != 2 or next_states.shape != states.shape:
            raise ValueError("CQL expects vector observations and vector continuous actions")
        if len(states) == 0 or actions.shape[0] != len(states) or rewards.shape[0] != len(states):
            raise ValueError("all batch fields must have the same nonzero batch dimension")
        batch_size = states.shape[0]
        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_states)
            next_q = torch.minimum(self.target_q1(next_states, next_actions),
                                   self.target_q2(next_states, next_actions))
            td_targets = rewards + self.gamma * (1.0 - terminals) * (
                next_q - self.alpha * next_logp)

        sample_count = self.num_action_samples
        low = self.actor.action_bias - self.actor.action_scale
        high = self.actor.action_bias + self.actor.action_scale
        random_actions = low + torch.rand(
            batch_size, sample_count, actions.shape[-1], device=self.device
        ) * (high - low)
        random_logp_scalar = -torch.log((high - low).abs().clamp_min(1e-6)).sum()
        random_logps = torch.full(
            (batch_size, sample_count), float(random_logp_scalar), device=self.device)
        repeated_states = states.unsqueeze(1).expand(-1, sample_count, -1).reshape(
            batch_size * sample_count, -1)
        with torch.no_grad():
            sampled_policy_actions, sampled_policy_logps, _ = self.actor.sample(repeated_states)
        policy_actions = sampled_policy_actions.reshape(batch_size, sample_count, -1)
        policy_logps = sampled_policy_logps.reshape(batch_size, sample_count)

        q1_data, q2_data = self.q1(states, actions), self.q2(states, actions)
        td_loss = F.mse_loss(q1_data, td_targets) + F.mse_loss(q2_data, td_targets)
        cql1 = continuous_cql_penalty(
            self.q1, states, actions, random_actions, policy_actions,
            random_logps, policy_logps, self.cql_temperature)
        cql2 = continuous_cql_penalty(
            self.q2, states, actions, random_actions, policy_actions,
            random_logps, policy_logps, self.cql_temperature)
        conservative_loss = cql1 + cql2
        critic_loss = td_loss + self.conservative_weight * conservative_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.critic_optimizer.step()

        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(False)
        try:
            sampled_actions, logp, _ = self.actor.sample(states)
            actor_q = torch.minimum(self.q1(states, sampled_actions),
                                    self.q2(states, sampled_actions))
            actor_loss = (self.alpha * logp - actor_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
        finally:
            for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
                parameter.requires_grad_(True)

        with torch.no_grad():
            for target, online in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
                for target_param, param in zip(target.parameters(), online.parameters()):
                    target_param.mul_(1.0 - self.tau).add_(self.tau * param)
        return {
            "critic_loss": float(critic_loss.detach()),
            "td_loss": float(td_loss.detach()),
            "conservative_loss": float(conservative_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "critic_grad_norm": float(critic_grad_norm.detach()),
            "actor_grad_norm": float(actor_grad_norm.detach()),
        }


__all__ = ["discrete_cql_penalty", "discrete_cql_critic_loss", "discrete_cql_critic_step",
           "continuous_cql_penalty",
           "continuous_cql_critic_loss", "continuous_cql_critic_step", "ContinuousCQLAgent"]
