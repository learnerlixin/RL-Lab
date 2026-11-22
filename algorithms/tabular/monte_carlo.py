"""First-visit Monte Carlo prediction from complete episodes."""
import numpy as np


def first_visit_mc_prediction(episodes, n_states, gamma=0.99):
    """Each episode is [(state, reward_after_state), ...]."""
    samples = [[] for _ in range(n_states)]
    for episode in episodes:
        returns, g = [0.0] * len(episode), 0.0
        for i in reversed(range(len(episode))):
            state, reward = episode[i]
            g = float(reward) + gamma * g
            returns[i] = g
        seen = set()
        for (state, _), value in zip(episode, returns):
            if state not in seen:
                samples[state].append(value)
                seen.add(state)
    values, counts = np.zeros(n_states), np.zeros(n_states, dtype=np.int64)
    for s, values_s in enumerate(samples):
        counts[s] = len(values_s)
        if values_s:
            values[s] = np.mean(values_s)
    return values, counts
