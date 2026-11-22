"""Train DDPG on Gymnasium Pendulum-v1."""
import argparse

import gymnasium as gym
import numpy as np

from algorithms.policy_optimization.ddpg import DDPGAgent
from algorithms.policy_optimization.sac import ContinuousReplayBuffer


def main(episodes=200, seed=7):
    np.random.seed(seed)
    env = gym.make("Pendulum-v1")
    agent = DDPGAgent(
        env.observation_space.shape[0], env.action_space.shape[0],
        action_low=env.action_space.low, action_high=env.action_space.high,
    )
    replay = ContinuousReplayBuffer(capacity=200_000)
    returns = []
    total_steps = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed if episode == 0 else None)
        episode_return = 0.0
        while True:
            # Initial random exploration improves replay coverage.
            if total_steps < 1_000:
                action = env.action_space.sample()
            else:
                action = agent.act(observation, noise_std=0.1)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            replay.add(observation, action, reward, next_observation, terminated)
            observation = next_observation
            episode_return += float(reward)
            total_steps += 1
            if total_steps >= 1_000:
                agent.update(replay, batch_size=256)
            if terminated or truncated:
                break
        returns.append(episode_return)
        if (episode + 1) % 10 == 0:
            print(f"episode={episode + 1:4d} mean_return_10={np.mean(returns[-10:]):.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.episodes, args.seed)
