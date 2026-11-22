"""Train an A2C agent on Gymnasium CartPole-v1 with shared utilities."""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from algorithms.policy_based import A2CAgent, RolloutBuffer
from utils import CSVLogger, evaluate_policy, save_checkpoint, seed_everything


def main(episodes=500, seed=7, run_dir="runs/a2c_cartpole", eval_interval=20):
    seed_everything(seed)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 100_000)
    agent = A2CAgent(env.observation_space.shape[0], env.action_space.n)
    buffer = RolloutBuffer(capacity=5000)
    total_steps = 0
    fields = ["episode", "steps", "train_return", "eval_mean_return",
              "loss", "policy_loss", "value_loss", "entropy"]
    with CSVLogger(run_path / "metrics.csv", fields) as logger:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                state = torch.as_tensor(observation, dtype=torch.float32)
                action, value = agent.act(state)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = torch.as_tensor(next_observation, dtype=torch.float32)
                with torch.no_grad():
                    _, next_value = agent.network(next_state.unsqueeze(0))
                buffer.add(
                    observation, action, reward, value, 0.0,
                    float(next_value.squeeze(0).item()),
                    terminated=bool(terminated), truncated=bool(truncated),
                )
                observation = next_observation
                episode_return += float(reward)
                total_steps += 1
                if terminated or truncated:
                    break

            buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=1.0)
            rollout = buffer.get()
            update_metrics = agent.update(
                rollout["observations"], rollout["actions"].long(), rollout["returns"])
            buffer.clear()

            eval_return = float("nan")
            if (episode + 1) % eval_interval == 0:
                evaluation = evaluate_policy(
                    eval_env, agent, episodes=5, deterministic=True,
                    seed=seed + 1_000_000 + episode * 5,
                )
                eval_return = evaluation["mean_return"]
            logger.log({
                "episode": episode + 1,
                "steps": total_steps,
                "train_return": episode_return,
                "eval_mean_return": eval_return,
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
            })
            if (episode + 1) % eval_interval == 0:
                print(
                    f"episode={episode + 1:4d} train_return={episode_return:6.1f} "
                    f"eval_return={eval_return:6.1f}"
                )
                save_checkpoint(
                    run_path / "latest.pt", agent.network, agent.optimizer,
                    step=total_steps,
                    metadata={"algorithm": "A2C", "episode": episode + 1, "seed": seed},
                )

    if episodes % eval_interval:
        save_checkpoint(
            run_path / "latest.pt", agent.network, agent.optimizer,
            step=total_steps,
            metadata={"algorithm": "A2C", "episode": episodes, "seed": seed},
        )
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=str, default="runs/a2c_cartpole")
    parser.add_argument("--eval-interval", type=int, default=20)
    args = parser.parse_args()
    main(args.episodes, args.seed, args.run_dir, args.eval_interval)
