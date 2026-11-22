"""First-visit Monte Carlo prediction from complete episodes."""
import numpy as np


def first_visit_mc_prediction(episodes, n_states, gamma=0.99):
    """episodes contain [(state, reward_after_state), ...] tuples."""
    samples = [[] for _ in range(n_states)]
    for episode in episodes:
        returns = [0.0] * len(episode)
        g = 0.0
        for i in reversed(range(len(episode))):
            state, reward = episode[i]
            g = float(reward) + gamma * g
            returns[i] = g
        seen = set()
        for (state, _), value in zip(episode, returns):
            if state not in seen:
                samples[state].append(value)
                seen.add(state)
    values = np.zeros(n_states)
    counts = np.zeros(n_states, dtype=np.int64)
    for state, values_for_state in enumerate(samples):
        counts[state] = len(values_for_state)
        if values_for_state:
            values[state] = np.mean(values_for_state)
    return values, counts
