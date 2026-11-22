"""On-policy rollout storage with correct termination/truncation GAE semantics."""
import numpy as np
import torch


class RolloutBuffer:
    """Store transition data for PPO/A2C and produce GAE targets/minibatches.

    terminated masks value bootstrap. truncated stops GAE recursion at an
    episode boundary but still bootstraps from the supplied next_value.
    The caller must pass the value estimate of the actual next observation.
    """

    def __init__(self, capacity, device="cpu"):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.device = torch.device(device)
        self.clear()

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.next_values = []
        self.terminated = []
        self.truncated = []
        self._advantages = None
        self._returns = None
        self._observation_shape = None
        self._action_shape = None

    @staticmethod
    def _scalar(value, name):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def add(self, observation, action, reward, value, log_prob,
            next_value, terminated, truncated=False):
        """Append one transition; action may be scalar or a vector."""
        if len(self) >= self.capacity:
            raise OverflowError("rollout buffer is full; update and clear before adding")
        obs = np.asarray(observation, dtype=np.float32).copy()
        act = np.asarray(action).copy()
        if obs.ndim == 0 or not np.isfinite(obs).all():
            raise ValueError("observation must be a finite array")
        if not np.isfinite(act).all():
            raise ValueError("action must contain finite values")
        if self._observation_shape is None:
            self._observation_shape = obs.shape
            self._action_shape = act.shape
        elif obs.shape != self._observation_shape or act.shape != self._action_shape:
            raise ValueError("observation and action shapes must stay constant within a rollout")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
            raise TypeError("terminated and truncated must be bool values")
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(self._scalar(reward, "reward"))
        self.values.append(self._scalar(value, "value"))
        self.log_probs.append(self._scalar(log_prob, "log_prob"))
        self.next_values.append(self._scalar(next_value, "next_value"))
        self.terminated.append(float(terminated))
        self.truncated.append(float(truncated))
        self._advantages = self._returns = None

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95,
                                       normalize_advantages=False):
        """Compute GAE and value targets for every stored transition."""
        if not len(self):
            raise ValueError("cannot compute GAE for an empty rollout")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0,1]")
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(self))):
            # True terminal: no value bootstrap. Time-limit truncation: bootstrap,
            # but do not propagate advantages into the next episode.
            bootstrap_mask = 1.0 - self.terminated[index]
            continuation_mask = 1.0 - max(self.terminated[index], self.truncated[index])
            delta = (self.rewards[index] + gamma * bootstrap_mask * self.next_values[index]
                     - self.values[index])
            gae = delta + gamma * gae_lambda * continuation_mask * gae
            advantages[index] = gae
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self._advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        self._returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        return self._advantages, self._returns

    def get(self):
        """Return the complete rollout as a tensor dictionary."""
        if self._advantages is None or self._returns is None:
            raise RuntimeError("call compute_returns_and_advantages before get")
        return {
            "observations": torch.as_tensor(np.stack(self.observations),
                                            dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(np.stack(self.actions), device=self.device),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device),
            "values": torch.as_tensor(self.values, dtype=torch.float32, device=self.device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device),
            "next_values": torch.as_tensor(self.next_values, dtype=torch.float32, device=self.device),
            "terminated": torch.as_tensor(self.terminated, dtype=torch.float32, device=self.device),
            "truncated": torch.as_tensor(self.truncated, dtype=torch.float32, device=self.device),
            "advantages": self._advantages,
            "returns": self._returns,
        }

    def minibatches(self, batch_size, shuffle=True):
        """Yield indexed mini-batches after GAE targets have been computed."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        data = self.get()
        count = len(self)
        indices = (torch.randperm(count, device=self.device) if shuffle else
                   torch.arange(count, device=self.device))
        for start in range(0, count, batch_size):
            selected = indices[start:start + batch_size]
            yield {key: value[selected] for key, value in data.items()}


__all__ = ["RolloutBuffer"]
