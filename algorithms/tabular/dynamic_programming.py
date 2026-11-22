"""Dynamic programming for finite MDPs with a known transition model.

transitions[s][a] is a list of (probability, next_state, reward, terminated).
"""
import numpy as np


def policy_evaluation(transitions, policy, n_states, n_actions, gamma=0.99,
                      theta=1e-8, max_iterations=100_000):
    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(n_states):
            old = values[state]
            new = 0.0
            for action in range(n_actions):
                for prob, next_state, reward, terminated in transitions[state][action]:
                    future = 0.0 if terminated else gamma * values[next_state]
                    new += policy[state, action] * prob * (reward + future)
            values[state] = new
            delta = max(delta, abs(old - new))
        if delta < theta:
            break
    return values


def value_iteration(transitions, n_states, n_actions, gamma=0.99,
                    theta=1e-8, max_iterations=100_000):
    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(n_states):
            old = values[state]
            q = []
            for action in range(n_actions):
                q.append(sum(prob * (reward + (0.0 if done else gamma * values[next_state]))
                             for prob, next_state, reward, done in transitions[state][action]))
            values[state] = max(q)
            delta = max(delta, abs(old - values[state]))
        if delta < theta:
            break
    q_values = np.zeros((n_states, n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            q_values[state, action] = sum(
                p * (r + (0.0 if done else gamma * values[ns]))
                for p, ns, r, done in transitions[state][action])
    return values, np.argmax(q_values, axis=1)


def policy_iteration(transitions, n_states, n_actions, gamma=0.99,
                     theta=1e-8, max_iterations=10_000):
    policy = np.full((n_states, n_actions), 1.0 / n_actions)
    for _ in range(max_iterations):
        values = policy_evaluation(transitions, policy, n_states, n_actions,
                                   gamma=gamma, theta=theta)
        stable = True
        for state in range(n_states):
            q = np.array([sum(p * (r + (0.0 if done else gamma * values[ns]))
                              for p, ns, r, done in transitions[state][a])
                          for a in range(n_actions)])
            row = np.zeros(n_actions)
            row[int(np.argmax(q))] = 1.0
            if not np.array_equal(row, policy[state]):
                stable = False
            policy[state] = row
        if stable:
            break
    return values, np.argmax(policy, axis=1)
