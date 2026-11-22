# RL-Lab：强化学习与大模型强化学习实践

一个以 Python / PyTorch 为主的强化学习学习仓库，覆盖表格型 RL、深度价值学习、策略优化、Offline RL 与 LLM preference optimization。项目目标是让公式、实现、单元测试和小型实验相互对应。

## 算法覆盖

| 方向 | 已实现内容 | 交付形式 |
|---|---|---|
| 表格型 RL | MDP/Bellman、Policy Evaluation/Iteration、Value Iteration、Monte Carlo、TD(0)、SARSA、Q-learning | 实现、测试、示例 |
| 深度价值 | DQN、Double DQN、Dueling、PER、Rainbow（C51、NoisyNet、n-step） | Agent、测试、CartPole 示例 |
| Policy Gradient | REINFORCE、Actor-Critic、GAE、A2C、离散 PPO、连续 tanh-Gaussian PPO | 算法组件、测试、CartPole/Pendulum 训练示例 |
| 连续控制 | DDPG、SAC、TD3 | Agent、测试、Pendulum 示例 |
| Offline RL | Behavior Cloning、IQL Agent、AWR Agent、离散 CQL critic step、连续 CQL Agent | Agent、训练步骤、数据集工具、Pendulum/CartPole 示例、测试 |
| LLM RL | Bradley–Terry Reward Model、DPO、token-level PPO + GAE/Value head、GRPO rollout preparation | Objective、训练步骤、批次构造、测试 |
| 实验基础设施 | 随机种子、CSV 日志、Checkpoint、策略评估 | 工具、测试、文档 |

## 多随机种子实验\n\n`run_seed_sweep` 和 `summarize_seed_results` 提供统一多 seed 执行与 mean/std/置信区间汇总，说明见 [`docs/33_multi_seed_benchmark.md`](docs/33_multi_seed_benchmark.md)。\n\n## PPO rollout 缓冲区

`RolloutBuffer` 提供 transition 存储、GAE/return 计算与 minibatch 采样，并区分 terminated 和 truncated。用法见 [`docs/31_on_policy_rollout_buffer.md`](docs/31_on_policy_rollout_buffer.md)。

## 公共 API

常用算法可通过 `algorithms.tabular`、`algorithms.value_based`、`algorithms.policy_based`、`algorithms.policy_optimization`、`algorithms.offline_rl` 和 `algorithms.llm_rl` 导入。示例见 [`docs/30_public_api.md`](docs/30_public_api.md)。

## 项目结构

- `algorithms/tabular/`：动态规划、Monte Carlo、TD prediction/control。
- `algorithms/value_based/`：DQN、Rainbow 及其组件。
- `algorithms/policy_based/`：REINFORCE、Actor-Critic、A2C、PPO。
- `algorithms/policy_optimization/`：DDPG、SAC、TD3。
- `algorithms/offline_rl/`：BC、IQL、AWR、CQL 与离线 transition dataset。
- `algorithms/llm_rl/`：Reward Model、DPO、LLM-PPO、GRPO trainer 与 rollout packing。
- `utils/`：reproducibility、CSV logger、checkpoint、evaluation。
- `examples/`：可运行训练脚本。
- `docs/`：算法推导、公式、实现边界与使用说明。
- `tests/`：单元测试。
- `.github/workflows/`：Python 3.10/3.11/3.12 CI。

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
# Tabular + online RL
python examples/tabular_control.py
python examples/train_dqn_cartpole.py
python examples/train_ppo_cartpole.py
python examples/train_a2c_cartpole.py --episodes 500
python examples/train_rainbow_cartpole.py --episodes 300

# Continuous control
python examples/train_continuous_ppo_pendulum.py --episodes 300
python examples/train_ddpg_pendulum.py --episodes 200
python examples/train_sac_pendulum.py

# Offline RL
python examples/train_bc_cartpole.py --episodes 80 --epochs 30
python examples/train_iql_pendulum.py --episodes 40 --epochs 10
python examples/train_cql_pendulum.py --episodes 40 --epochs 10
python examples/train_awr_pendulum.py --episodes 40 --epochs 10
```

BC 示例收集一个简单控制器生成的 CartPole 演示数据；IQL 示例展示 NPZ 离线 transition dataset 和训练循环，默认数据由随机 Pendulum 策略采集。后者是管线示例，不是性能 benchmark，也不保证随机数据能学出高回报策略。

## Offline RL API

`OfflineTransitionDataset` 将 observations、actions、rewards、next_observations、terminals 和 timeouts 统一存储，并支持 NPZ 加载/保存。真实 MDP 终止与时间限制截断分开记录，避免把 time-limit truncation 错误当作 terminal 阻断 bootstrap。

- BC：离散动作交叉熵、连续动作 MSE、统一 optimizer step。
- IQL：连续动作高斯策略、双 Q、expectile value、target Q、advantage-weighted policy update。
- AWR：带 advantage 权重的 actor 更新和 value regression step。
- CQL：离散 Q penalty/critic step；连续 CQL Agent 集成 SAC-style actor/twin critics、随机/策略候选动作采样、密度校正和 conservative update。独立连续 critic helper 仍允许调用方提供候选动作与 TD target。

完整 API 与简化假设见 `docs/27_offline_rl_training.md`。

## LLM RL 的边界

DPO、LLM-PPO、GRPO 和 Reward Model trainer 提供可复用的目标函数或 optimizer step，不是开箱即用的 LLM 对齐框架。模型生成、tokenizer、rollout storage、奖励打分、参考模型管理、分布式调度、实验跟踪和模型评估仍需按具体模型/任务接入。GRPO 支持 callback 驱动的 group rollout 收集、prompt/completion padding、old/reference log-prob 计算、批次构造与一步式 GRPOTrainer；模型生成策略、tokenizer 与 reward scoring 仍由调用方提供。LLM-PPO 还提供 token-level GAE 和 value head 更新；模型生成和 PPO minibatch/epoch 调度仍由调用方组织。

## 实验复现

`seed_everything`、`CSVLogger`、`save_checkpoint/load_checkpoint` 与 `evaluate_policy` 提供实验基础能力。更严格的复现还需要记录依赖锁定、环境版本、硬件/CUDA、训练配置、数据集版本和代码 commit。

## 实现原则与后续工作

本仓库以教学实现和模块化训练组件为主，不把简化实现称为论文级或工业级复现。新增 A2C 端到端脚本示例会统一使用 rollout buffer、CSV 日志、独立评估环境与 checkpoint。后续仍值得持续加强：标准离线 benchmark 数据接入、完整在线/离线算法对照实验、真实模型生成与 reward pipeline 的端到端编排、分布式训练、超参数 sweep 与消融测试。
