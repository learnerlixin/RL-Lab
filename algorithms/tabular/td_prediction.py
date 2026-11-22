"""TD(0) state-value prediction for a tabular Gymnasium environment."""
import numpy as np


def td0_prediction(env, policy, episodes=500, alpha=0.1, gamma=0.99, seed=None):
    if episodes <= 0 or not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("invalid episodes, alpha or gamma")
    values = np.zeros(env.observation_space.n)
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        while True:
            action = int(policy(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            target = reward if terminated else reward + gamma * values[next_state]
            values[state] += alpha * (target - values[state])
            if terminated or truncated:
                break
            state = next_state
    return values
