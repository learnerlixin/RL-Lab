# PPO/A2C Rollout Buffer

`RolloutBuffer` 提供定长 on-policy transition 存储、GAE/return 计算和 minibatch 切分。它可以用于离散 PPO、连续 PPO 或 A2C 的采样批次。

## 使用方式

```python
from algorithms.policy_based import RolloutBuffer

buffer = RolloutBuffer(capacity=2048)
buffer.add(
    observation=obs,
    action=action,
    reward=reward,
    value=value,
    log_prob=old_log_prob,
    next_value=next_value,
    terminated=terminated,
    truncated=truncated,
)
advantages, returns = buffer.compute_returns_and_advantages(
    gamma=0.99, gae_lambda=0.95)
rollout = buffer.get()
# Or iterate:
for minibatch in buffer.minibatches(batch_size=128):
    ...
```

## Terminated 与 truncated

每条 transition 都保存真实 MDP 终止和 time-limit 截断两个标记。GAE 的 TD residual 使用：

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\]

其中 (d_t) 是 terminated。GAE 递推则在 terminated 或 truncated 时切断跨 episode 的递推。这样 time-limit 截断仍能 bootstrap 到真实 next observation 的 value，但不会将下一个 episode 的优势传播回来。

因此，采样器必须传入实际 next observation 对应的 next_value。terminated 时该值会被 mask；truncated 时该值会被保留用于 bootstrap。

`RolloutBuffer` 不负责环境交互、策略动作采样或 PPO optimizer update；它输出的 observations/actions/log_probs/advantages/returns 可以直接传给对应算法更新函数。使用完一轮 rollout 后调用 `clear()`。
