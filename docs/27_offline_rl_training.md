# 离线 RL：Behavior Cloning、数据集与连续 CQL

本页补充三个可以直接进入离线训练流程的模块：行为克隆、统一 transition 数据集，以及连续动作 CQL critic 更新。

## 1. Behavior Cloning

Behavior Cloning (BC) 将数据集中的动作视为监督标签。离散动作使用交叉熵；连续动作使用均方误差：

\[
\mathcal L_{BC}^{disc}=-\mathbb E_{(s,a)\sim D}\log \pi_\theta(a|s),\qquad
\mathcal L_{BC}^{cont}=\mathbb E_{(s,a)\sim D}\|\mu_\theta(s)-a\|_2^2.
\]

接口：

- `discrete_behavior_cloning_loss(logits, actions, mask=None)`
- `continuous_behavior_cloning_loss(predicted_actions, dataset_actions, mask=None)`
- `behavior_cloning_train_step(policy, optimizer, observations, actions, action_type=...)`

训练步骤只负责监督损失和 optimizer update；模型结构、数据采样与动作分布选择由训练脚本负责。示例脚本 `examples/train_bc_cartpole.py` 会采集演示、训练策略并评估。连续 MSE 适合确定性 actor 的基线，并不等价于最大化任意概率分布策略的似然。

## 2. 离线 Transition 数据集

`OfflineTransitionDataset` 统一存储 observations、actions、rewards、next_observations、terminals 和 timeouts，支持 NPZ 序列化：

```python
from algorithms.offline_rl import OfflineTransitionDataset

dataset = OfflineTransitionDataset.from_npz("dataset.npz")
print(len(dataset), dataset[0]["observations"])
dataset.save_npz("copy.npz")
```

**terminated 和 truncated 必须区分。** `terminals` 只表示环境的真实 MDP 终止；time-limit 截断写入 `timeouts`。构造 bootstrap target 时，通常只用 terminal mask 阻止 bootstrap，不能把 timeout 无条件当作真正 terminal。

数据加载使用 `allow_pickle=False`。输入数组长度、维度、非有限值和二值终止标记会经过校验。对于大型数据集，此实现仍是内存数据集；更大规模场景应改用 mmap、分片存储或流式 loader。

## 3. 连续动作 CQL

连续 CQL 从随机动作和当前策略动作中采样候选集，对候选动作 Q 值做 log-sum-exp，并减去数据动作 Q：

\[
\mathcal L_{CQL}=\tau\log\sum_{a\in\mathcal A_{sample}}
\exp\left(\frac{Q_\phi(s,a)-\log p(a|s)}{\tau}\right)
-Q_\phi(s,a_D).
\]

代码接口 `continuous_cql_penalty` 接收 caller 提供的候选动作和可选采样 log-density。若使用均匀随机采样、策略采样，应将对应 log-probability 传入；函数不会代替 actor 生成候选动作。

`continuous_cql_critic_step` 将 TD 回归和 conservative penalty 合并：

\[
\mathcal L_Q=\mathcal L_{TD}+\alpha\mathcal L_{CQL}.
\]

目标 Q、target network、policy action sampling、随机动作采样和 entropy/log-density 的准确计算都留给调用方，以便适配 SAC 等不同 actor-critic 结构。离散动作同样提供 `discrete_cql_critic_step`，将数据动作 TD 误差和全动作保守项组合。CQL 接口是 critic 更新组件，不是完整的离线 CQL Agent。

## 4. IQL Agent 与 AWR 更新\n\n`IQLAgent` 提供连续动作 IQL 更新：双 Q target、expectile value 回归、TD critic 回归、tanh-Gaussian policy，以及按优势加权的最大似然策略更新。调用方传入 transition minibatch，字段与 `OfflineTransitionDataset` 相同。\n\n`awr_train_step` 提供 actor/value 两个 optimizer 的更新接口。policy 需暴露 `log_prob(states, actions)`；returns/GAE 由调用方提供。它与 IQL 的策略提取形式相似，但 AWR 的 value target 由调用方准备。\n\nIQL 示例：\n\n```bash\npython examples/train_iql_pendulum.py --episodes 40 --epochs 10
python examples/train_cql_pendulum.py --episodes 40 --epochs 10\n```\n\n## 5. CartPole 端到端示例

```bash
python examples/train_bc_cartpole.py --episodes 80 --epochs 30
```

示例通过简单的角度/角速度控制器收集演示轨迹，训练离散 BC policy，保存为 NPZ 并进行独立 episode 评估。它用于演示离线数据管线，不代表专家策略或 benchmark 最优结果。
