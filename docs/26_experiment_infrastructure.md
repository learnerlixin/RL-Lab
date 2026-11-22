# 实验基础设施：随机种子、日志、Checkpoint 与评估

强化学习实验通常包含随机初始化、环境随机性、采样策略和优化器状态。只保存模型参数不足以复现实验；本仓库提供四个轻量工具，帮助训练脚本统一这些常见工作。

## 1. 随机种子

`seed_everything(seed, deterministic=False)` 同时设置 Python、NumPy、PyTorch CPU 与可用 CUDA 随机数生成器。

```python
from utils import seed_everything

seed_everything(42, deterministic=True)
```

确定性算法模式可能降低速度；某些算子没有确定性实现时，PyTorch 会给出警告。环境自身也应在 `env.reset(seed=...)` 时单独设置种子。

## 2. CSV 训练日志

`CSVLogger(path, fieldnames)` 使用固定字段写入 CSV，每次写入后 flush，适合记录 episode return、loss、环境步数等标量。

```python
from utils import CSVLogger

with CSVLogger("runs/metrics.csv", ["step", "episode_return"]) as logger:
    logger.log({"step": 1000, "episode_return": 42.0})
```

字段必须与初始化时的 schema 完全一致，避免不同训练阶段悄悄写入结构不一致的数据。

## 3. 模型与优化器 Checkpoint

`save_checkpoint` 保存模型参数、可选优化器状态、训练 step、metadata，以及可选的 Python/NumPy/PyTorch RNG 状态。

```python
from utils import save_checkpoint, load_checkpoint

save_checkpoint("runs/latest.pt", model, optimizer, step=10_000,
                metadata={"algorithm": "PPO"})
state = load_checkpoint("runs/latest.pt", model, optimizer,
                        map_location="cpu", restore_rng=True)
print(state["step"], state["metadata"])
```

恢复 RNG 需要显式传入 `restore_rng=True`。Checkpoint 文件应只加载可信来源；PyTorch checkpoint 本质上是序列化数据，不应随意加载陌生文件。

## 4. 策略评估

`evaluate_policy(env, policy, episodes=5, deterministic=True, seed=None)` 兼容 Gymnasium 的 terminated/truncated API，也兼容旧版四返回值 API。policy 可以是可调用对象，也可以是具有 `act` 方法的 agent；若其方法声明了 `deterministic` 参数，评估器会传入对应设置。

```python
from utils import evaluate_policy

metrics = evaluate_policy(env, agent, episodes=10, deterministic=True, seed=100)
print(metrics["mean_return"], metrics["mean_length"])
```

评估器不会关闭传入的环境；环境生命周期仍由调用方管理。若使用自定义环境包装器，应确认 reset/step 遵循上述接口。

## 5. 设计边界

这些工具刻意保持轻量，不负责实验配置系统、分布式 checkpoint、视频录制、超参数搜索或多进程环境管理。要实现严格复现，还需固定依赖版本、环境版本、硬件与 CUDA/cuDNN 配置，并记录完整配置和代码版本。
