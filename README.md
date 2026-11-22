# RL-Lab：强化学习与大模型强化学习实践

一个以 Python / PyTorch 为主的强化学习学习仓库：从 MDP 和表格型算法，逐步进入深度强化学习与 LLM preference optimization。优先保证公式、实现、测试三者能互相对应。

## 当前已实现

| 模块 | 算法 / 组件 | 状态 |
|---|---|---|
| 基础 | MDP、Bellman 方程 | 文档 |
| 表格型 | Policy Evaluation、Policy Iteration、Value Iteration | 实现 + 测试 |
| 表格型 | First-visit Monte Carlo、TD(0) prediction | 实现 + 测试 |
| 表格型 | epsilon-greedy、SARSA、Q-learning | 实现 + 测试 |
| 深度价值 | DQN、Replay Buffer、Target Network | 实现 + 测试 |
| 深度价值 | Double DQN、Dueling 网络、Prioritized Replay | 实现 + 测试 |
| 深度价值 | Rainbow：C51、Double-DQN target、Dueling、NoisyNet、PER 加权更新、n-step accumulator | 集成 Agent + 测试 + CartPole 示例 |
| 策略梯度 | REINFORCE、Actor-Critic、GAE、同步 A2C | 实现 + 测试 |
| 策略优化 | 离散动作 PPO minibatch update | 实现 + 测试 |\n| 策略优化 | 连续动作 PPO（tanh Gaussian Actor-Critic、动作边界与 log-prob 修正） | 实现 + 测试 |
| 连续控制 | DDPG（确定性 Actor-Critic、target networks、Polyak 更新） | Agent + 测试 + Pendulum 示例 |
| 连续控制 | SAC、TD3（双 Q、target 网络、连续动作策略） | 实现 + 测试 |
| 离线 RL | IQL expectile、Q target、advantage-weighted BC、AWR weighted regression、离散 CQL penalty | 目标函数组件 + 测试 |
| LLM RL | Bradley–Terry loss、DPO objective | 实现 + 测试 |
| LLM RL | token-level PPO update、KL-shaped reward、GRPO clipped objective 与组内优势 | 训练组件 + 测试 |
| LLM RL | Masked pooling Reward Model head 与 Bradley–Terry optimizer step | 模型头 + 训练步骤 + 测试 |
| LLM RL | Causal LM completion log-prob 与 DPO train step | 训练步骤 + 测试 |

## 目录

- `algorithms/tabular/`：动态规划、Monte Carlo、TD prediction/control。
- `algorithms/value_based/`：DQN、Double DQN、Rainbow。
- `algorithms/policy_based/`：REINFORCE、Actor-Critic、A2C、GAE、离散 / 连续动作 PPO。
- `algorithms/policy_optimization/`：DDPG、SAC、TD3。
- `algorithms/offline_rl/`：IQL、AWR、CQL 核心目标函数。
- `algorithms/llm_rl/`：偏好优化、奖励模型头、PPO/GRPO 目标函数与训练步骤。
- `docs/`：数学原理、公式推导、算法差异和实现边界。
- `examples/`：小型可运行示例。
- `tests/`：单元测试。
- `.github/workflows/`：GitHub Actions 自动测试。

## 安装与测试

建议 Python 3.10+。

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

## 示例

```bash
python examples/tabular_control.py
python examples/train_dqn_cartpole.py
python examples/train_ppo_cartpole.py
python examples/train_continuous_ppo_pendulum.py --episodes 300
python examples/train_ddpg_pendulum.py --episodes 200
python examples/train_sac_pendulum.py
python examples/train_rainbow_cartpole.py --episodes 300
```

## 学习顺序

1. 阅读 `docs/01_mdp_and_bellman.md` 和 `docs/02_td_control.md`。
2. 运行 `examples/tabular_control.py`，理解 SARSA 与 Q-learning。
3. 学习 DP、MC、TD，再阅读 DQN、Double DQN 和 Rainbow。
4. 从 REINFORCE 过渡到 Actor-Critic、A2C、GAE 和 PPO。
5. 阅读 LLM 偏好优化文档，理解 Reward Model、DPO、PPO、GRPO 的联系与区别。

## LLM 强化学习的重要边界

- SFT 是监督微调，不等同于强化学习。
- RLHF 是一类训练流程；Reward Model 是奖励建模组件，PPO 是可能采用的策略优化方法。
- DPO 直接从偏好对构造目标，不要求显式在线 PPO 训练循环。
- GRPO 通常使用同一 prompt 下多个 completion 的组内相对优势。
- 这里的 PPO/GRPO 是可复用的目标函数或优化组件，并非开箱即用的大模型训练框架。完整训练还需要 tokenizer、模型生成、rollout storage、奖励管线、分布式训练、checkpoint、评测与安全审查。

## 实现原则

每个算法尽量遵循：问题定义 → 数学目标 → 公式推导 → 核心实现 → 单元测试 → 实验与局限。任何简化都会在文档中说明，不把教学实现包装成论文级或工业级复现。

## 当前限制与 Roadmap

Rainbow Agent 已集成 C51、Double-DQN 动作选择、Dueling、NoisyNet 和 PER 重要性采样更新。n-step accumulator 已接入 Rainbow 的 replay 与 C51 bootstrap discount；PER 使用 O(N) 采样，适合教学而非大规模训练。连续动作 PPO 与 DDPG 均已提供 Pendulum 训练示例。仍需补齐更多离线 RL 算法、完整 GRPO rollout/trainer 适配、实验日志与 checkpoint、基准测试和消融实验。
