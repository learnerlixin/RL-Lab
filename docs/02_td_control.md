# 02. TD 控制：SARSA 与 Q-learning

## 一句话抓住差别

- **SARSA**：用策略实际选出的 \(A_{t+1}\) 更新，学习当前行为策略的价值。
- **Q-learning**：假设下一步采取最优动作，用 \(\max_{a'}Q(S_{t+1},a')\) 更新。

两者共享 TD 误差结构：

\[
\delta_t=\text{target}-Q(S_t,A_t),\qquad
Q\leftarrow Q+\alpha\delta_t.
\]

SARSA 的 target 为 \(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})\)；
Q-learning 的 target 为 \(R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')\)。

## 为什么一个 on-policy，一个 off-policy？

SARSA 的更新目标依赖 epsilon-greedy 行为策略抽出的下一动作，因此探索行为也会进入被评估的回报。
Q-learning 的 target 则直接取当前 Q 表中的最大动作价值，与行为策略下一步是否真的选它无关。

## 运行

安装依赖后，可在安装了 Gymnasium 的环境中使用：

```python
import gymnasium as gym
from algorithms.tabular import q_learning, sarsa

env = gym.make("CliffWalking-v0")
q = q_learning(env, episodes=2_000, seed=7)
env.close()
```

将 `q_learning` 替换为 `sarsa` 即可比较两种更新目标。建议固定多个随机种子，
比较训练回报的均值和方差，而不是只看单次曲线。

## 代码阅读提示

1. `epsilon_greedy` 用随机 tie-breaking，避免初始 Q 值相同时总选第一个动作。
2. `use_sarsa` 分支体现了两种算法唯一关键的 target 差异。
3. 终止转移不 bootstrap。
4. 该教学版本假设离散状态与动作，并直接用整数状态索引 Q 表；连续状态需先离散化或改用函数近似。
