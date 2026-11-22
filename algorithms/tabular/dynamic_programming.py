"""Dynamic programming for finite MDPs with a known transition model.

transitions[s][a] is a list of (probability, next_state, reward, terminated).
"""
import numpy as np


def policy_evaluation(transitions, policy, n_states, n_actions, gamma=0.99, theta=1e-8, max_iterations=100_000):
    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            old, new = values[s], 0.0
            for a in range(n_actions):
                for p, ns, r, done in transitions[s][a]:
                    new += policy[s, a] * p * (r + (0.0 if done else gamma * values[ns]))
            values[s] = new
            delta = max(delta, abs(old - new))
        if delta < theta:
            break
    return values


def value_iteration(transitions, n_states, n_actions, gamma=0.99, theta=1e-8, max_iterations=100_000):
    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            old = values[s]
            q = [sum(p * (r + (0.0 if done else gamma * values[ns]))
                     for p, ns, r, done in transitions[s][a]) for a in range(n_actions)]
            values[s] = max(q)
            delta = max(delta, abs(old - values[s]))
        if delta < theta:
            break
    q_values = np.array([[sum(p * (r + (0.0 if done else gamma * values[ns]))
                              for p, ns, r, done in transitions[s][a])
                          for a in range(n_actions)] for s in range(n_states)])
    return values, np.argmax(q_values, axis=1)


def policy_iteration(transitions, n_states, n_actions, gamma=0.99, theta=1e-8, max_iterations=10_000):
    policy = np.full((n_states, n_actions), 1.0 / n_actions)
    for _ in range(max_iterations):
        values = policy_evaluation(transitions, policy, n_states, n_actions, gamma, theta)
        stable = True
        for s in range(n_states):
            q = np.array([sum(p * (r + (0.0 if done else gamma * values[ns]))
                              for p, ns, r, done in transitions[s][a]) for a in range(n_actions)])
            row = np.zeros(n_actions)
            row[int(np.argmax(q))] = 1.0
            if not np.array_equal(row, policy[s]):
                stable = False
            policy[s] = row
        if stable:
            break
    return values, np.argmax(policy, axis=1)
