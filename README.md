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
| 深度价值 | Double DQN、Dueling 网络、Prioritized Replay primitives | 实现 + 测试 |
| 策略梯度 | REINFORCE、Actor-Critic、GAE | 实现 + 测试 |
| 策略优化 | 离散动作 PPO minibatch update | 实现 + 测试 |
| 连续控制 | SAC、TD3（双 Q、target 网络、连续动作策略） | 实现 + 测试 |
| 离线 RL | IQL expectile、Q target、advantage-weighted BC | 目标函数组件 + 测试 |
| LLM RL | Bradley–Terry loss、DPO objective | 实现 + 测试 |
| LLM RL | PPO clipped objective、GRPO clipped objective、组内优势 | 目标函数组件 + 测试 |
| LLM RL | Masked pooling Reward Model head | 模型头组件 + 测试 |
| LLM RL | Causal LM completion log-prob 与 DPO train step | 训练步骤 + 测试 |

## 目录

- `algorithms/tabular/`：动态规划、Monte Carlo、TD prediction/control。
- `algorithms/value_based/`：DQN 与扩展 DQN 组件。
- `algorithms/policy_based/`：REINFORCE、Actor-Critic、GAE、PPO。
- `algorithms/policy_optimization/`：SAC、TD3。
- `algorithms/offline_rl/`：IQL 核心目标函数。
- `algorithms/llm_rl/`：偏好优化、奖励模型头、PPO/GRPO 目标函数。
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

## 学习顺序

1. 阅读 `docs/01_mdp_and_bellman.md` 和 `docs/02_td_control.md`。
2. 运行 `examples/tabular_control.py`，理解 SARSA 与 Q-learning。
3. 学习 DP、MC、TD，再阅读 DQN 与 Double DQN。
4. 从 REINFORCE 过渡到 Actor-Critic、GAE 和 PPO。
5. 阅读 LLM 偏好优化文档，理解 Reward Model、DPO、PPO、GRPO 的联系与区别。

## LLM 强化学习的重要边界

- SFT 是监督微调，不等同于强化学习。
- RLHF 是一类训练流程；Reward Model 是奖励建模组件，PPO 是可能采用的策略优化方法。
- DPO 直接从偏好对构造目标，不要求显式在线 PPO 训练循环。
- GRPO 通常使用同一 prompt 下多个 completion 的组内相对优势。
- 这里的 PPO/GRPO 是可复用的目标函数或优化组件，并非开箱即用的大模型训练框架。完整训练还需要 tokenizer、模型生成、rollout storage、奖励管线、分布式训练、checkpoint、评测与安全审查。

## 实现原则

每个算法尽量遵循：问题定义 → 数学目标 → 公式推导 → 核心实现 → 单元测试 → 实验与局限。任何简化都会在文档中说明，不把教学实现包装成论文级或工业级复现。

## Roadmap

后续扩展方向包括 Rainbow、更多离线 RL 方法、连续动作 PPO、可复现的 Gymnasium 端到端训练脚本、LLM rollout/GRPO 训练框架适配、算法基准实验和消融分析。
