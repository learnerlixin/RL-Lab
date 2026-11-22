"""Offline IQL training pipeline on a collected Pendulum transition dataset.

The random behavior policy is intentionally a low-quality dataset. This script
demonstrates dataset handling and the IQL update loop; it is not a performance
benchmark or a claim that random data yields a strong controller.
"""
import argparse

import gymnasium as gym
import numpy as np
from torch.utils.data import DataLoader

from algorithms.offline_rl import IQLAgent, OfflineTransitionDataset
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
        dataset.save_npz("runs/pendulum_random_dataset.npz")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    agent = IQLAgent(
        env.observation_space.shape[0], env.action_space.shape[0],
        action_low=env.action_space.low, action_high=env.action_space.high,
    )
    print(f"offline transitions: {len(dataset)}")
    for epoch in range(epochs):
        metrics = []
        for batch in loader:
            metrics.append(agent.update(batch))
        if metrics and (epoch + 1) % 2 == 0:
            print(f"epoch={epoch + 1:3d} value_loss={np.mean([m['value_loss'] for m in metrics]):.4f} "
                  f"critic_loss={np.mean([m['critic_loss'] for m in metrics]):.4f} "
                  f"actor_loss={np.mean([m['actor_loss'] for m in metrics]):.4f}")
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
