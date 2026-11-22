# RL-Lab：强化学习与大模型强化学习实践

从 MDP、动态规划到深度强化学习，再到 LLM post-training。项目以可读的最小实现、数学推导和可复现实验为目标。教学简化版会明确标注，不冒充工业级复现。

## 学习路线
1. 基础：MDP、回报、价值函数、Bellman 方程。
2. 表格型：DP、Monte Carlo、TD、SARSA、Q-learning。
3. 深度价值：DQN、Double DQN、Dueling DQN、Rainbow。
4. 策略梯度：REINFORCE、Actor-Critic、GAE。
5. 策略优化：TRPO、PPO、SAC、TD3。
6. 离线/模型式：BC、IQL、CQL、Dyna。
7. LLM RL：Reward Model、RLHF、PPO、DPO、GRPO、RLAIF。

## 目录
- `algorithms/tabular/`：DP、MC、TD prediction/control。
- `algorithms/policy_based/`：策略梯度。
- `algorithms/llm_rl/`：偏好优化目标函数。
- `docs/`：原理、公式、推导、边界与风险。
- `examples/`：小型可运行实验。
- `tests/`：单元测试。

## 安装
建议 Python 3.10+。
```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## 算法边界
SFT 是监督微调，不等于 RL。RLHF 常见流程包含偏好数据、奖励模型和策略优化；PPO 是策略优化算法。DPO 从偏好对直接构造目标，不要求显式在线 PPO loop。GRPO 常见做法是对同一 prompt 的多个 completion 构造组内相对优势。具体系统仍涉及采样、token mask、KL 约束、长度处理、评估和安全问题。

## 教程规范
每种算法按问题定义 → 数学目标 → 公式推导 → 核心代码 → 实验 → 局限性展开。项目会持续补齐 DQN、PPO、SAC、离线 RL 和完整 LLM 训练管线；未实现的内容不会标成已完成。
