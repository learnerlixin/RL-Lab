"""Train continuous PPO on Gymnasium Pendulum-v1."""
import argparse

import gymnasium as gym
import numpy as np
import torch

from algorithms.policy_based.actor_critic import generalized_advantage_estimate
from algorithms.policy_based.continuous_ppo import (
    ContinuousActorCritic,
    continuous_ppo_update,
)


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

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed if episode == 0 else None)
        states, actions, logps = [], [], []
        rewards, values, next_values, terminals = [], [], [], []
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

            states.append(state)
            actions.append(action.squeeze(0))
            logps.append(log_prob.squeeze(0))
            rewards.append(float(reward))
            values.append(value.squeeze(0))
            next_values.append(torch.zeros_like(next_value) if terminated else next_value)
            terminals.append(float(terminated))
            total_reward += float(reward)
            observation = next_observation
            if terminated or truncated:
                break

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        value_tensor = torch.stack(values)
        next_value_tensor = torch.stack(next_values)
        terminal_tensor = torch.tensor(terminals, dtype=torch.float32)
        advantages, returns = generalized_advantage_estimate(
            reward_tensor, value_tensor, next_value_tensor, terminal_tensor
        )
        continuous_ppo_update(
            model, optimizer, torch.stack(states), torch.stack(actions),
            torch.stack(logps).detach(), returns.detach(), advantages.detach(),
            epochs=10, batch_size=64, entropy_coef=0.001,
        )
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
