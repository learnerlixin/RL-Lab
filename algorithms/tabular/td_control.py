"""Tabular TD control: epsilon-greedy, SARSA and Q-learning.

All algorithms assume discrete states/actions and a Gymnasium-like environment:
reset() -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)

The implementation intentionally keeps the update equations visible.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Choose a random action with probability epsilon; otherwise choose a greedy one."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    # Random tie-breaking avoids a permanent preference for action zero at initialization.
    best = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(best))


def _run_control(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int | None,
    on_episode: Callable[[int, float], None] | None,
    use_sarsa: bool,
) -> np.ndarray:
    """Shared training loop; use_sarsa selects on-policy SARSA vs off-policy Q-learning."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        action = epsilon_greedy(q_table[state], epsilon, rng)
        total_reward = 0.0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            if done:
                # A terminal state has no future value. Do not bootstrap from it.
                target = float(reward)
                q_table[state, action] += alpha * (target - q_table[state, action])
                break

            if use_sarsa:
                next_action = epsilon_greedy(q_table[next_state], epsilon, rng)
                target = reward + gamma * q_table[next_state, next_action]
            else:
                next_action = None
                target = reward + gamma * np.max(q_table[next_state])

            # TD update: Q <- Q + alpha * (target - Q)
            q_table[state, action] += alpha * (target - q_table[state, action])
            state = next_state
            action = next_action if use_sarsa else epsilon_greedy(q_table[state], epsilon, rng)

        if on_episode is not None:
            on_episode(episode, total_reward)

    return q_table


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1,
               seed=None, on_episode=None) -> np.ndarray:
    """Off-policy TD control using max_a Q(s', a) as the bootstrap target."""
    return _run_control(env, episodes, alpha, gamma, epsilon, seed, on_episode, False)


def sarsa(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1,
          seed=None, on_episode=None) -> np.ndarray:
    """On-policy TD control using the next action actually selected by the policy."""
    return _run_control(env, episodes, alpha, gamma, epsilon, seed, on_episode, True)
