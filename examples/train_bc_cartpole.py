"""Collect scripted CartPole demonstrations and train a behavior-cloned policy."""
import argparse

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from algorithms.offline_rl.behavior_cloning import behavior_cloning_train_step
from algorithms.offline_rl.dataset import OfflineTransitionDataset
from utils import evaluate_policy, seed_everything


class DiscretePolicy(nn.Module):
    def __init__(self, observation_dim, action_count):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, action_count))

    def forward(self, observations):
        return self.network(observations)

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return int(self(state).argmax(dim=-1).item())


def collect_demonstrations(env, episodes, seed):
    observations, actions, rewards, next_observations, terminals, timeouts = [], [], [], [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        while True:
            # A small angle/velocity controller supplies reproducible labels.
            action = int(state[2] + 0.2 * state[3] > 0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            observations.append(state.copy()); actions.append(action); rewards.append(reward)
            next_observations.append(next_state.copy()); terminals.append(float(terminated))
            timeouts.append(float(truncated))
            state = next_state
            if terminated or truncated:
                break
    return OfflineTransitionDataset(observations, actions, rewards, next_observations,
                                    terminals, timeouts)


def main(episodes=80, epochs=30, seed=7, dataset_path=None):
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    if dataset_path:
        dataset = OfflineTransitionDataset.from_npz(dataset_path)
    else:
        dataset = collect_demonstrations(env, episodes, seed)
    print(f"offline transitions: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    policy = DiscretePolicy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    for epoch in range(epochs):
        losses, accuracies = [], []
        for batch in loader:
            metrics = behavior_cloning_train_step(
                policy, optimizer, batch["observations"], batch["actions"],
                action_type="discrete")
            losses.append(metrics["loss"]); accuracies.append(metrics["accuracy"])
        if (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1:3d} loss={np.mean(losses):.4f} label_accuracy={np.mean(accuracies):.3f}")
    if dataset_path is None:
        dataset.save_npz("runs/cartpole_demonstrations.npz")
    evaluation = evaluate_policy(env, policy, episodes=10, deterministic=True, seed=seed + 10_000)
    print(f"evaluation mean_return={evaluation['mean_return']:.1f} mean_length={evaluation['mean_length']:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", type=str, default=None, help="optional existing NPZ offline dataset")
    args = parser.parse_args()
    main(args.episodes, args.epochs, args.seed, args.dataset)
