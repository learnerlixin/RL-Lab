# 公共 API 导入指南

仓库各算法子包提供稳定的常用符号导出入口。学习者可以先从子包导入核心对象，而不必记住每个实现文件的内部路径。

## 表格型强化学习

```python
from algorithms.tabular import (
    policy_evaluation, policy_iteration, value_iteration,
    first_visit_mc_prediction, td0_prediction, q_learning, sarsa,
)
```

## 深度价值学习与策略优化

```python
from algorithms.value_based import DQNAgent, DoubleDQNAgent, RainbowAgent
from algorithms.policy_based import A2CAgent, ActorCritic, RolloutBuffer, ppo_update
from algorithms.policy_optimization import DDPGAgent, SACAgent, TD3Agent
```

## Offline RL

```python
from algorithms.offline_rl import (
    OfflineTransitionDataset, IQLAgent, AWRAgent, ContinuousCQLAgent,
    behavior_cloning_train_step,
)
```

## LLM preference optimization

```python
from algorithms.llm_rl import (
    RewardModelHead, GRPOTrainer, dpo_train_step, grpo_train_step,
    collect_grouped_rollouts, prepare_grpo_batch,
    token_level_gae, llm_ppo_policy_step, llm_ppo_value_step,
)
```

实验工具可从 utils 导入：

```python
from utils import run_seed_sweep, summarize_seed_results
```

公开导出是便利入口，不代表所有模块具有相同抽象层级。Objective 函数、训练步骤、完整 Agent 仍需分别阅读各自文档，确认输入形状、终止状态语义和调用方责任。
