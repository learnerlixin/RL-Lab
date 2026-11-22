"""Train SAC on Gymnasium Pendulum-v1 with random warm-up."""
import numpy as np
import gymnasium as gym
from algorithms.policy_optimization.sac import SACAgent, ContinuousReplayBuffer


def main(episodes=200, seed=7):
    env = gym.make("Pendulum-v1")
    low, high = env.action_space.low, env.action_space.high
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], low, high)
    replay = ContinuousReplayBuffer(200_000)
    rng = np.random.default_rng(seed)
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total = 0.0
        while True:
            if len(replay) < 2_000:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            replay.add(state, action, reward, next_state, terminated)
            if len(replay) >= 256:
                agent.update(replay, batch_size=256)
            state = next_state
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        if (episode + 1) % 10 == 0:
            print(f"episode={episode + 1:4d} mean_return_10={np.mean(returns[-10:]):.1f}")
    env.close()


if __name__ == "__main__":
    main()
