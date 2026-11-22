# RL-Lab：强化学习与大模型对齐实践

> 从马尔可夫决策过程到 LLM post-training：用可运行的代码理解强化学习。

本仓库面向希望系统学习强化学习（Reinforcement Learning, RL）及大语言模型强化学习/对齐方法的读者。目标不是封装算法调用，而是尽量从数学定义出发，使用 Python、NumPy 与 PyTorch 实现核心算法，并通过小型实验验证关键结论。

**当前阶段：** 项目骨架与第一批基础教程。算法实现会按路线逐步补齐；未实现的模块会明确标注，不把目录占位误当成已完成实现。

## 学习路线

1. **基础数学与问题建模**：MDP、回报、价值函数、Bellman 方程、动态规划。
2. **无模型学习**：Monte Carlo、TD(0)、SARSA、Q-learning、Expected SARSA。
3. **深度价值学习**：DQN、Double DQN、Dueling DQN、Prioritized Replay、Rainbow。
4. **策略梯度与 Actor-Critic**：REINFORCE、A2C/A3C、GAE。
5. **策略优化与连续控制**：TRPO、PPO、DDPG、TD3、SAC。
6. **离线与模型式 RL**：Behavior Cloning、IQL、CQL、Dyna、Model-Based RL。
7. **LLM 强化学习与偏好优化**：SFT、Reward Model、RLHF、PPO、DPO、IPO、KTO、GRPO、RLAIF、在线/离线偏好优化。

## 目录

```text
docs/                 原理讲解、公式推导与阅读路线
algorithms/
  tabular/            动态规划、MC、TD、SARSA、Q-learning
  value_based/        DQN 系列
  policy_based/       REINFORCE、Actor-Critic
  policy_optimization/ PPO、TRPO、SAC 等
  offline/            离线强化学习
  model_based/        模型式强化学习
  llm_rl/             RLHF、PPO、DPO、GRPO 等
examples/             可直接运行的小实验
configs/              实验配置
tests/                单元测试
```

## 环境安装

建议 Python 3.10–3.12。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

核心概念示例尽量只依赖 NumPy；神经网络算法使用 PyTorch。Gymnasium 用于标准环境交互。LLM 部分会把算法数学原理与大规模训练工程区分开，默认提供小型、可在普通设备上理解/验证的示例，不要求下载大型模型。

## 每个算法的讲解规范

- **问题定义**：状态、动作、奖励、终止条件和假设。
- **数学原理**：目标函数、关键公式及符号说明。
- **推导直觉**：公式从哪里来、为什么这样更新。
- **代码实现**：张量形状、关键函数、逐步注释。
- **实验**：运行命令、预期现象、超参数影响。
- **局限性**：稳定性、样本效率、计算成本和适用边界。

## LLM 强化学习专题说明

LLM 的 post-training 不等于简单地“把 PPO 用在文本上”。本仓库会明确区分：

- **SFT**：监督微调，通常是偏好优化/强化学习之前的策略初始化阶段，不本身等同于 RL。
- **Reward Modeling**：从比较偏好中学习奖励函数，例如 Bradley–Terry 形式。
- **RLHF**：以人类偏好训练奖励模型，再使用 PPO 等方法优化策略；具体系统可能采用不同变体。
- **PPO**：通过概率比率裁剪等机制限制每轮策略更新幅度。
- **DPO / IPO / KTO**：直接或以不同统计目标利用偏好数据优化模型；它们与显式在线 RL 的训练流程不同。
- **GRPO**：对同一 prompt 的多条生成进行组内相对比较，构造优势信号；其具体实现存在不同变体。
- **RLAIF**：使用 AI 反馈辅助构造偏好或奖励信号。
- **评估与安全**：奖励 hacking、分布偏移、KL 约束、长度偏差和可验证奖励等问题。

## 重要约定

- 教程优先采用可复现、可测试的最小实现；与论文/工业系统不同的地方会明确说明。
- 公式中的符号在首次出现时定义，代码变量尽量与公式对应。
- 随机实验会固定 seed，并说明环境/依赖版本可能造成的差异。
- 不把单次训练曲线当成算法优越性的证据；比较时应使用多 seed 与统计汇总。

## 贡献与引用

欢迎提出问题、修正推导、补充实验。论文方法请优先引用原论文或作者公开资料；第三方实现仅作为工程参考。

## License

待项目首批内容稳定后补充许可证。
