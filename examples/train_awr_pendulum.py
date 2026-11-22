"""Offline AWR training on a random Pendulum transition dataset.

The random behavior dataset is intentionally low quality; this example
demonstrates return calculation and AWR training rather than benchmark results.
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.offline_rl import AWRAgent, OfflineTransitionDataset, monte_carlo_returns
from utils import evaluate_policy, seed_everything


def collect_random_dataset(env, episodes, seed):
    env.action_space.seed(seed)
    observations, actions, rewards, next_observations, terminals, timeouts = [], [], [], [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        while True:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            observations.append(state.copy()); actions.append(action.copy())
            rewards.append(reward); next_observations.append(next_state.copy())
            terminals.append(float(terminated)); timeouts.append(float(truncated))
            state = next_state
            if terminated or truncated:
                break
    return OfflineTransitionDataset(observations, actions, rewards, next_observations,
                                    terminals, timeouts)


def main(episodes=40, epochs=10, batch_size=256, seed=7, dataset_path=None):
    seed_everything(seed)
    env = gym.make("Pendulum-v1")
    dataset = (OfflineTransitionDataset.from_npz(dataset_path) if dataset_path
               else collect_random_dataset(env, episodes, seed))
    if not dataset_path:
        dataset.save_npz("runs/pendulum_awr_dataset.npz")
    returns = monte_carlo_returns(
        dataset.rewards, dataset.terminals, dataset.timeouts, gamma=0.99)
    training_data = TensorDataset(
        torch.as_tensor(dataset.observations, dtype=torch.float32),
        torch.as_tensor(dataset.actions, dtype=torch.float32),
        returns,
    )
    loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    agent = AWRAgent(
        env.observation_space.shape[0], env.action_space.shape[0],
        action_low=env.action_space.low, action_high=env.action_space.high,
    )
    print(f"offline transitions: {len(dataset)}")
    for epoch in range(epochs):
        metrics = []
        for observations, actions, batch_returns in loader:
            metrics.append(agent.update({
                "observations": observations, "actions": actions, "returns": batch_returns,
            }))
        if metrics and (epoch + 1) % 2 == 0:
            print(f"epoch={epoch + 1:3d} actor_loss={np.mean([m['actor_loss'] for m in metrics]):.4f} "
                  f"value_loss={np.mean([m['value_loss'] for m in metrics]):.4f}")
    result = evaluate_policy(env, agent, episodes=5, deterministic=True, seed=seed + 10_000)
    print(f"evaluation mean_return={result['mean_return']:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    main(args.episodes, args.epochs, args.batch_size, args.seed, args.dataset)
