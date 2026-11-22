"""Small runnable example: compare SARSA and Q-learning on CliffWalking."""
import gymnasium as gym
from algorithms.tabular import q_learning, sarsa


def main():
    for name, learner in (("SARSA", sarsa), ("Q-learning", q_learning)):
        env = gym.make("CliffWalking-v0")
        returns = []
        q_table = learner(
            env,
            episodes=500,
            alpha=0.5,
            gamma=1.0,
            epsilon=0.1,
            seed=42,
            on_episode=lambda episode, reward: returns.append(reward),
        )
        print(f"{name}: mean return over last 50 episodes = {sum(returns[-50:]) / 50:.2f}")
        print(f"Learned Q-table shape: {q_table.shape}")
        env.close()


if __name__ == "__main__":
    main()
