"""Train continuous PPO on Gymnasium Pendulum-v1."""
import argparse

import gymnasium as gym
import numpy as np
import torch

from algorithms.policy_based.continuous_ppo import (
    ContinuousActorCritic,
    continuous_ppo_update,
)
from algorithms.policy_based.rollout_buffer import RolloutBuffer


def main(episodes=300, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v1")
    model = ContinuousActorCritic(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    episode_returns = []
    buffer = RolloutBuffer(capacity=10000)

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed if episode == 0 else None)
        total_reward = 0.0
        while True:
            state = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, value = model.act(state.unsqueeze(0))
            next_observation, reward, terminated, truncated, _ = env.step(
                action.squeeze(0).cpu().numpy().astype(np.float32)
            )
            next_state = torch.as_tensor(next_observation, dtype=torch.float32)
            with torch.no_grad():
                next_value = model.value(next_state.unsqueeze(0)).squeeze(0)

            buffer.add(
                state.numpy(), action.squeeze(0).cpu().numpy(), float(reward),
                float(value.squeeze(0).item()), float(log_prob.squeeze(0).item()),
                float(next_value.item()), terminated=bool(terminated),
                truncated=bool(truncated),
            )
            total_reward += float(reward)
            observation = next_observation
            if terminated or truncated:
                break

        buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        rollout = buffer.get()
        continuous_ppo_update(
            model, optimizer, rollout["observations"], rollout["actions"],
            rollout["log_probs"], rollout["returns"], rollout["advantages"],
            epochs=10, batch_size=64, entropy_coef=0.001,
        )
        buffer.clear()
        episode_returns.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(
                f"episode={episode + 1:4d} "
                f"mean_return_10={np.mean(episode_returns[-10:]):.1f}"
            )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.episodes, args.seed)
