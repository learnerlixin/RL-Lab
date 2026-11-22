"""Train the integrated Rainbow baseline on Gymnasium CartPole."""
import argparse
import random
import numpy as np
import torch
import gymnasium as gym

from algorithms.value_based.advanced_dqn import PrioritizedReplayBuffer, NStepAccumulator
from algorithms.value_based.rainbow_agent import RainbowAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=50_000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state, _ = env.reset(seed=args.seed)
    agent = RainbowAgent(len(state), env.action_space.n, atom_count=51,
                         value_min=-10, value_max=10)
    replay = PrioritizedReplayBuffer(args.capacity)
    n_step = NStepAccumulator(replay, n=3, gamma=agent.gamma)
    total_steps = 0

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Gymnasium truncation is a time limit, not an MDP terminal.
            n_step.add(state, action, reward, next_state, terminated, truncated)
            state = next_state
            episode_return += reward
            total_steps += 1
            beta = min(1.0, 0.4 + 0.6 * total_steps / 200_000)
            agent.update(replay, batch_size=64, beta=beta)
        if (episode + 1) % 10 == 0:
            print(f"episode={episode + 1:4d} return={episode_return:7.1f} "
                  f"replay={len(replay)}")
    env.close()


if __name__ == "__main__":
    main()
